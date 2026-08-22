"""Turn an uploaded document into the adaptive reading structure.

    upload (.pdf/.docx/.md/.txt)
        -> extract text
        -> segment into concepts
        -> derive three complexity tiers per concept
        -> derive rescue payloads (explain / example / prereq / diagram)

Everything here is extractive and deterministic: it selects, compresses and
rearranges sentences that are already in the document. Nothing is invented,
so the reading always reflects what the student actually uploaded, and the
same file produces the same result every time — which matters when you have
to demo it twice in front of a panel.

An optional LLM path (see llm.py) produces better prose for the plain tier and
the rescue payloads. It is off by default and falls back to this module on any
error, so the app never depends on a network call at demo time.
"""
from __future__ import annotations

import html
import json
import re
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src import assets
from src import config as C

# Parse-output version. Increment on any change to how concepts, tiers,
# rescue payloads or diagrams are built, so previously ingested documents
# are refreshed instead of silently serving output from an older parser.
SCHEMA = 5

# ---------------------------------------------------------------- text utils

STOPWORDS = set("""
a an the and or but if then than that this these those of in on at to for from by
with without within into onto is are was were be been being it its as not no nor
we you they he she i our your their his her them us can could may might must
will would shall should do does did done have has had having more most much many
such some any each every other another which who whom whose what when where why
how there here also however therefore thus hence because since while although
though even so very just only about over under between among during before after
above below up down out off again further once all both few own same too s t
""".split())

# Academic register -> plain register. Applied word-for-word for the plain tier.
PLAIN_SWAPS = {
    "utilise": "use", "utilize": "use", "utilises": "uses", "utilizes": "uses",
    "commence": "start", "commences": "starts", "terminate": "end",
    "demonstrate": "show", "demonstrates": "shows", "demonstrated": "showed",
    "indicate": "show", "indicates": "shows", "indicated": "showed",
    "facilitate": "help", "facilitates": "helps", "facilitated": "helped",
    "endeavour": "try", "endeavor": "try", "ascertain": "find out",
    "subsequently": "later", "previously": "earlier", "approximately": "about",
    "sufficient": "enough", "numerous": "many", "additional": "more",
    "consequently": "so", "furthermore": "also", "moreover": "also",
    "nevertheless": "even so", "notwithstanding": "despite",
    "in order to": "to", "with regard to": "about", "in the event that": "if",
    "prior to": "before", "subsequent to": "after", "in the case of": "for",
    "a number of": "several", "the majority of": "most",
    "is able to": "can", "are able to": "can", "has the ability to": "can",
    "it is possible that": "maybe", "due to the fact that": "because",
    "in spite of the fact that": "although", "for the purpose of": "to",
    "aforementioned": "earlier", "elucidate": "explain", "delineate": "outline",
    "constitutes": "is", "comprises": "contains", "exhibits": "shows",
    "employ": "use", "employed": "used", "employs": "uses",
    "methodology": "method", "functionality": "function",
}

# Openings that carry no information and can be dropped wholesale.
HEDGES = re.compile(
    r"^(it (is|should be) (worth |important |necessary )?(noting|noted|emphasi[sz]ed|"
    r"mentioned|observed|stressed) that|it (is|has been) (widely |generally |commonly )?"
    r"(known|accepted|argued|shown|found|established) that|"
    r"as (has been |is )?(noted|mentioned|discussed) (above|earlier|previously)[,\s]|"
    r"in (this|the present) (section|chapter|paper|study)[,\s]|"
    r"needless to say[,\s]|it goes without saying that)\s*", re.IGNORECASE)

EXAMPLE_CUES = re.compile(
    r"\b(for example|for instance|e\.?g\.?|such as|consider|suppose|imagine|"
    r"to illustrate|as an illustration|say that|take the case)\b", re.IGNORECASE)

DEF_CUES = re.compile(
    r"\b(is defined as|is called|refers to|is known as|means that|is the|are the|"
    r"can be defined|denotes|is a|are a|consists of|is termed)\b", re.IGNORECASE)

ABBREV = re.compile(r"\b(e\.g|i\.e|et al|cf|vs|Fig|Eq|Dr|Prof|Mr|Ms|St|approx|no)\.$",
                    re.IGNORECASE)


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")                 # soft hyphens from PDFs
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)      # de-hyphenate across line breaks
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter that survives common academic abbreviations."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out, buf = [], ""
    for chunk in re.split(r"(?<=[.!?])\s+", text):
        buf = f"{buf} {chunk}".strip() if buf else chunk
        if ABBREV.search(buf) or re.search(r"\b[A-Z]\.$", buf):
            continue                             # abbreviation, not a boundary
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return [s for s in out if len(s.split()) >= 3]


def words(s: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]+", s.lower())


def content_words(s: str) -> list[str]:
    return [w for w in words(s) if w not in STOPWORDS and len(w) > 2]


# ------------------------------------------------------------------ parsing

def read_document(path: Path, assets_dir: Path | None = None) -> list[dict]:
    """Return blocks: {heading, text, tables, images} in document order.

    Headings are recovered from real structure where the format has it
    (markdown '#', Word heading styles) and inferred heuristically otherwise.
    Tables and figures stay attached to the section that introduces them.
    """
    ext = path.suffix.lower()
    if ext == ".docx":
        return _read_docx(path, assets_dir)
    if ext == ".pdf":
        return _read_pdf_blocks(path, assets_dir)
    if ext in {".md", ".markdown"}:
        return _read_markdown(path.read_text(encoding="utf-8", errors="replace"))
    if ext in {".txt", ".text", ""}:
        return _infer_blocks(path.read_text(encoding="utf-8", errors="replace"))
    raise ValueError(f"Unsupported file type '{ext}'. Use PDF, DOCX, MD or TXT.")


CAPTION_LINE = re.compile(
    r"^\s*(fig(?:ure)?|table|chart|plot)\.?\s*([0-9]+(?:\.[0-9]+)?)\s*[.:\-–]\s*(.+)$",
    re.IGNORECASE)


def _read_pdf_blocks(path: Path, assets_dir: Path | None) -> list[dict]:
    """Build blocks page by page so figures land in the right section.

    A PDF has no structural link between an image and the text around it, but
    it does have pages, and a figure is almost always on the page of the
    section that introduces it. Within a page, a caption line ("Figure 3. …")
    is a much stronger signal than position, so images are matched to captions
    first and fall back to the last block on the page.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("pip install pypdf to read PDFs") from e

    reader = PdfReader(str(path))
    all_blocks: list[dict] = []
    total_words = 0

    for page in reader.pages:
        text = clean_text(page.extract_text() or "")
        total_words += len(text.split())

        # Pull caption lines out of the prose: they name the figure, they do
        # not read as body text, and leaving them inline strands them. The
        # line immediately before a caption is recorded too — it identifies
        # which block the figure sits under far more reliably than position.
        captions, kept = [], []
        for line in text.split("\n"):
            m = CAPTION_LINE.match(line.strip())
            if m and len(line.strip()) < 220:
                anchor = next((l for l in reversed(kept) if l.strip()), "")
                captions.append({
                    "text": f"{m.group(1).title()} {m.group(2)}. {m.group(3).strip()}",
                    "anchor": anchor.strip()[:60]})
            else:
                kept.append(line)
        page_blocks = _infer_blocks("\n".join(kept))

        images = []
        if assets_dir is not None:
            try:
                for im in getattr(page, "images", []):
                    data = im.data
                    if len(data) < 8_000:        # rules, bullets, logos
                        continue
                    images.append({"file": assets.save_image(
                        data, Path(im.name).suffix, assets_dir), "caption": ""})
            except Exception:
                images = []

        for i, img in enumerate(images):
            cap = captions[i] if i < len(captions) else None
            if cap:
                img["caption"] = cap["text"]

            target = None
            # 1. The block containing the line the caption directly follows.
            if cap and cap["anchor"]:
                target = next((b for b in page_blocks if cap["anchor"] in b["text"]), None)
            # 2. A block whose prose names the figure by number.
            if target is None and cap:
                num = re.search(r"(\d+(?:\.\d+)?)", cap["text"])
                if num:
                    ref = re.compile(rf"\b(?:fig(?:ure)?)\.?\s*{re.escape(num.group(1))}\b",
                                     re.IGNORECASE)
                    target = next((b for b in reversed(page_blocks)
                                   if ref.search(b["text"])), None)
            # 3. Otherwise the last block on the page — figures follow text.
            if target is None and page_blocks:
                target = page_blocks[-1]
            if target is not None:
                target["images"].append(img)

        # Any caption with no image still belongs to the page's last block.
        for extra in captions[len(images):]:
            if page_blocks:
                page_blocks[-1]["text"] += "\n" + extra["text"]

        all_blocks.extend(page_blocks)

    if total_words < 40:
        raise ValueError(
            "Almost no text came out of this PDF — it's probably a scan. "
            "Run OCR on it first, or upload the DOCX/text version.")

    # Merge a block continuing across a page break into the one before it.
    merged: list[dict] = []
    for b in all_blocks:
        if merged and not b["heading"] and not merged[-1]["images"] and not b["images"]:
            merged[-1]["text"] += " " + b["text"]
        elif merged and not b["heading"] and b["images"] and not b["text"].strip():
            merged[-1]["images"].extend(b["images"])
        else:
            merged.append(b)
    return merged


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("pip install pypdf to read PDFs") from e
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = clean_text("\n\n".join(pages))
    if len(text.split()) < 40:
        raise ValueError(
            "Almost no text came out of this PDF — it's probably a scan. "
            "Run OCR on it first, or upload the DOCX/text version.")
    return text


def _read_docx(path: Path, assets_dir: Path | None = None) -> list[dict]:
    """Walk the document body in order, keeping tables and images in place.

    python-docx exposes paragraphs and tables as separate collections, which
    loses their interleaving. Iterating the underlying body elements keeps a
    table attached to the section that introduces it.
    """
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:
        raise RuntimeError("pip install python-docx to read Word files") from e

    doc = docx.Document(str(path))
    body = doc.element.body
    blocks: list[dict] = []
    cur = {"heading": None, "text": [], "tables": [], "images": []}
    pending_caption = ""

    def flush():
        if cur["text"] or cur["tables"] or cur["images"]:
            blocks.append({"heading": cur["heading"],
                           "text": "\n".join(cur["text"]),
                           "tables": cur["tables"], "images": cur["images"]})

    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]

        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            style = (para.style.name or "").lower()

            # Images live inside runs; pull them before treating the text.
            for blip in child.iter(
                    "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
                rid = blip.get("{http://schemas.openxmlformats.org/officeDocument/"
                               "2006/relationships}embed")
                if not rid or assets_dir is None:
                    continue
                try:
                    part = doc.part.related_parts[rid]
                    if len(part.blob) < 8_000:   # logos, rules, bullets
                        continue
                    name = assets.save_image(part.blob,
                                             Path(part.partname).suffix, assets_dir)
                    cur["images"].append({"file": name, "caption": pending_caption})
                    pending_caption = ""
                except Exception:
                    continue

            if not text:
                continue
            # Word's Caption style is the reliable signal, but authors very
            # often type "Fig. 6.1: ..." as body text. Treat that shape as a
            # caption too, or it stays stranded in the prose while the figure
            # it names sits underneath unlabelled.
            is_caption = (style.startswith("caption")
                          or (len(text) < 220
                              and assets.FIG_REF.match(text)
                              and re.match(r"^\s*\w+\.?\s*[\d.]+\s*[:.\-–]\s*\S", text)))
            if is_caption:
                # A caption may precede or follow its figure.
                if cur["images"] and not cur["images"][-1]["caption"]:
                    cur["images"][-1]["caption"] = text
                elif cur["tables"] and not cur["tables"][-1]["caption"]:
                    cur["tables"][-1]["caption"] = text
                else:
                    pending_caption = text
                continue
            if style.startswith("heading") or style in {"title", "subtitle"}:
                flush()
                cur = {"heading": text, "text": [], "tables": [], "images": []}
            else:
                cur["text"].append(text)

        elif tag == "tbl":
            table = Table(child, doc)
            grid = [[c.text.strip() for c in row.cells] for row in table.rows]
            grid = [r for r in grid if any(x for x in r)]
            if len(grid) >= 2:
                headers, rows = grid[0], grid[1:]
                cur["tables"].append({
                    "headers": headers, "rows": rows,
                    "caption": pending_caption,
                    "summary": assets.describe_table(headers, rows, pending_caption),
                    "chart": assets.build_table_chart(headers, rows),
                })
                pending_caption = ""
    flush()

    if len(blocks) <= 1 and blocks:
        inferred = _infer_blocks(blocks[0]["text"])
        if inferred:
            inferred[0]["tables"] = blocks[0]["tables"]
            inferred[0]["images"] = blocks[0]["images"]
        return inferred
    return blocks


def _md_tables(lines: list[str]) -> tuple[list[str], list[dict]]:
    """Pull GitHub-style pipe tables out of markdown, returning the rest."""
    out, tables, i = [], [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.count("|") >= 2 and i + 1 < len(lines) and re.match(
                r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
            def cells(row):
                return [c.strip() for c in row.strip().strip("|").split("|")]
            headers = cells(ln)
            rows, i = [], i + 2
            while i < len(lines) and lines[i].count("|") >= 2:
                rows.append(cells(lines[i]))
                i += 1
            if rows:
                tables.append({"headers": headers, "rows": rows, "caption": "",
                               "summary": assets.describe_table(headers, rows),
                               "chart": assets.build_table_chart(headers, rows)})
            continue
        out.append(ln)
        i += 1
    return out, tables


def _read_markdown(text: str) -> list[dict]:
    blocks, heading, buf = [], None, []
    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            # Emit even when the body is empty: a top-level "# Title" followed
            # straight by "## Section" would otherwise be lost, and it is
            # usually the best document title available.
            if buf or heading is not None:
                body, tbls = _md_tables(buf)
                blocks.append({"heading": heading, "text": "\n".join(body),
                               "tables": tbls, "images": []})
            buf = []
            heading = m.group(2).strip()
        else:
            # strip the lightweight markup that would otherwise reach the reader,
            # but keep pipe tables intact for the table parser
            if "|" not in line:
                line = re.sub(r"[*_`]{1,3}", "", line)
                line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
            buf.append(line)
    if buf or heading is not None:
        body, tbls = _md_tables(buf)
        blocks.append({"heading": heading, "text": "\n".join(body),
                       "tables": tbls, "images": []})
    return blocks or [{"heading": None, "text": text, "tables": [], "images": []}]


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not (3 <= len(line) <= 90):
        return False
    if line.endswith((".", ",", ";", ":")):
        return False
    if re.match(r"^(\d+(\.\d+)*|[IVXLC]+\.|[A-Z]\.)\s+\S", line):
        return True          # numbered heading
    wds = line.split()
    if len(wds) > 12:
        return False
    caps = sum(1 for w in wds if w[:1].isupper())
    return caps / max(len(wds), 1) >= 0.6


def _group_paragraphs(text: str, target_words: int = 70) -> list[dict]:
    """Chunk unstructured prose into concept-sized blocks.

    Accumulates whole paragraphs until a chunk is substantial enough to carry
    three complexity tiers. Short paragraphs are merged forward rather than
    dropped — discarding them silently loses content the student uploaded.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [p.strip() for p in text.split("\n") if p.strip()]

    chunks, buf, count = [], [], 0
    for p in paras:
        buf.append(p)
        count += len(p.split())
        if count >= target_words:
            chunks.append(" ".join(buf))
            buf, count = [], 0
    if buf:
        # Trailing remainder joins the previous chunk unless it stands alone.
        if chunks and count < target_words * 0.4:
            chunks[-1] += " " + " ".join(buf)
        else:
            chunks.append(" ".join(buf))
    return [{"heading": None, "text": c, "tables": [], "images": []} for c in chunks]


def _infer_blocks(text: str) -> list[dict]:
    """Segment unstructured text: heading-ish lines first, paragraphs otherwise."""
    text = clean_text(text)
    lines = text.split("\n")
    blocks, heading, buf = [], None, []

    for line in lines:
        if _looks_like_heading(line) and (buf or heading is None):
            if buf:
                blocks.append({"heading": heading, "text": "\n".join(buf),
                               "tables": [], "images": []})
                buf = []
            heading = line.strip()
        elif line.strip():
            buf.append(line.strip())
    if buf:
        blocks.append({"heading": heading, "text": "\n".join(buf),
                       "tables": [], "images": []})

    # Fall back to paragraph grouping only when no heading was found at all.
    # Testing `len(blocks) <= 1` also fires on a page that is a single
    # well-formed section, throwing away the heading it just identified.
    if not blocks or (len(blocks) == 1 and not blocks[0]["heading"]):
        return _group_paragraphs(text)
    return blocks


# --------------------------------------------------------------- scoring

def score_sentences(sentences: list[str], title_terms: set[str]) -> list[float]:
    """Frequency-based extractive scoring, normalised for sentence length."""
    freq: dict[str, int] = {}
    for s in sentences:
        for w in content_words(s):
            freq[w] = freq.get(w, 0) + 1
    if not freq:
        return [0.0] * len(sentences)
    top = max(freq.values())

    scores = []
    for i, s in enumerate(sentences):
        cw = content_words(s)
        if not cw:
            scores.append(0.0)
            continue
        base = sum(freq.get(w, 0) / top for w in cw) / len(cw)
        if i == 0:
            base *= 1.35                                  # topic sentence
        if title_terms & set(cw):
            base *= 1.25                                  # echoes the heading
        if DEF_CUES.search(s):
            base *= 1.20                                  # definitional
        n = len(s.split())
        if n < 8 or n > 45:
            base *= 0.75                                  # too short / unwieldy
        scores.append(base)
    return scores


def key_terms(sentences: list[str], k: int = 5) -> list[str]:
    """Most salient repeated terms, preferring multi-word phrases.

    A phrase only counts if it actually recurs. Weighting every adjacent word
    pair would let one-off fragments like "otherwise numerous" outrank the
    genuinely repeated single words the section is about.
    """
    text = " ".join(sentences).lower()

    bigrams: dict[str, int] = {}
    for m in re.finditer(r"\b([a-z][a-z'-]+)\s+([a-z][a-z'-]+)\b", text):
        a, b = m.group(1), m.group(2)
        if a in STOPWORDS or b in STOPWORDS or len(a) < 3 or len(b) < 3:
            continue
        bigrams[f"{a} {b}"] = bigrams.get(f"{a} {b}", 0) + 1

    freq: dict[str, int] = {}
    for phrase, n in bigrams.items():
        if n >= 2:                      # recurring phrase: strong signal
            freq[phrase] = n * 3
    for w in content_words(text):
        freq[w] = freq.get(w, 0) + 1

    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[str] = []
    for term, count in ranked:
        if count < 2:
            continue
        if any(term in seen or seen in term for seen in out):
            continue                    # skip terms contained in one already kept
        out.append(term)
        if len(out) >= k:
            break
    return out


# ------------------------------------------------------- tier construction

def simplify(sentence: str) -> list[str]:
    """One dense sentence -> one or more plainer ones."""
    s = HEDGES.sub("", sentence).strip()
    s = re.sub(r"\s*\([^)]{0,120}\)", "", s)          # drop short parentheticals
    s = re.sub(r"\s*\[[^\]]{0,60}\]", "", s)          # drop bracketed citations
    s = re.sub(r"\s*,\s*(?:e\.g\.|i\.e\.)\s*,?\s*", ", ", s, flags=re.IGNORECASE)

    for long, short in PLAIN_SWAPS.items():
        s = re.sub(rf"\b{re.escape(long)}\b", short, s, flags=re.IGNORECASE)

    # Substitution can leave a connective stranded mid-clause: "consequently"
    # becomes "so", and "which consequently improves" reads as "which so
    # improves". Inside a relative clause the connective carries nothing, so
    # drop it rather than translate it.
    s = re.sub(r"\b(which|that|and|but)\s+(so|also|even so)\s+", r"\1 ", s, flags=re.IGNORECASE)
    s = re.sub(r"^(so|also)\s+,?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s)

    # Break at semicolons and em dashes, which almost always join clauses.
    parts = [p.strip(" ,;—-") for p in re.split(r"\s*[;—]\s*|\s+--\s+", s) if p.strip()]

    out = []
    for part in parts:
        # Split remaining long sentences at a coordinating conjunction.
        if len(part.split()) > 26:
            m = re.search(r",\s+(and|but|which|while|whereas|although|so that)\s+", part)
            if m:
                head, tail = part[:m.start()].strip(), part[m.end():].strip()
                if len(head.split()) >= 5 and len(tail.split()) >= 5:
                    out += [head, tail[:1].upper() + tail[1:]]
                    continue
        out.append(part)

    fixed = []
    for p in out:
        p = p[:1].upper() + p[1:] if p else p
        if p and p[-1] not in ".!?":
            p += "."
        if len(p.split()) >= 3:
            fixed.append(p)
    return fixed or [sentence]


def build_tiers(sentences: list[str], scores: list[float]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (tiers, key_sentence_per_tier).

    The key sentence is the highest-scoring one, returned verbatim as it
    appears in each tier so the reader can mark it inline. Marking one
    sentence is the point: underlining a whole paragraph emphasises nothing.
    """
    full = " ".join(sentences)
    best = max(range(len(sentences)), key=lambda i: scores[i]) if sentences else 0

    plain_parts: list[str] = []
    for s in sentences:
        plain_parts += simplify(s)
    budget = max(2, int(len(sentences) * 0.7) + 1)
    kept = list(range(len(sentences)))
    if len(plain_parts) > budget * 2:
        kept = sorted(sorted(range(len(sentences)), key=lambda i: -scores[i])[:budget])
        if best not in kept:            # never drop the sentence we're marking
            kept = sorted(kept[:-1] + [best])
        plain_parts = []
        for i in kept:
            plain_parts += simplify(sentences[i])
    plain = " ".join(plain_parts)

    key_full = sentences[best] if sentences else ""
    key_plain = simplify(sentences[best])[0] if sentences else ""
    essential = " ".join(simplify(sentences[best])[:2]) if sentences else ""

    tiers = {"full": full, "plain": plain, "essential": essential}
    # The essential tier is the key sentence, so marking it would be redundant.
    keys = {"full": key_full, "plain": key_plain, "essential": ""}
    return tiers, keys


# ----------------------------------------------------- rescue construction

SEQ_CUES = re.compile(
    r"\b(first(?:ly)?|second(?:ly)?|third(?:ly)?|then|next|after that|afterwards|"
    r"subsequently|finally|lastly|initially|begins? by|starts? by|followed by|"
    r"step \d+|stage \d+|once .{3,40}? is (?:done|complete)|before .{3,30}? can)\b",
    re.IGNORECASE)

ARROW = re.compile(r"\s*(?:->|→|=>)\s*")


LABELLED_STEP = re.compile(r"(?:^|(?<=[.;]\s))([A-Z][A-Za-z/ -]{2,28}?):\s+(?=[A-Z(])")


def extract_steps(sentences: list[str], max_steps: int = 7) -> list[str]:
    """Pull an ordered pipeline out of prose, if there is one.

    Three shapes are recognised, in order of confidence: an explicit arrow
    chain, a run of labelled steps ("Filtering: ... Epoching: ..."), and
    sentences carrying sequential cue words. Returns [] when the section isn't
    describing a process, so the caller falls back to a concept map.
    """
    text = " ".join(sentences)

    # 1. Arrow chains are unambiguous — take them verbatim.
    for s in sentences:
        if len(ARROW.findall(s)) >= 2:
            parts = [p.strip(" .,;:") for p in ARROW.split(s) if p.strip()]
            # "Fig 6.1: block diagram summarises X -> Y -> Z" leads with caption
            # text that is not a step. Drop it rather than draw it.
            parts = [p for p in parts if not CAPTION_NOISE.search(p)]
            if len(parts) >= 3:
                return [_short_label(p) for p in parts[:max_steps]]

    # 2. Labelled steps. Methodology sections are written this way constantly,
    #    and the labels are already the step names — no compression needed.
    labels = [m.group(1).strip() for m in LABELLED_STEP.finditer(text)]
    labels = [l for l in labels if len(l.split()) <= 4]
    if len(labels) >= 3:
        seen, ordered = set(), []
        for l in labels:
            if l.lower() not in seen:
                seen.add(l.lower())
                ordered.append(l[:26])
        return ordered[:max_steps]

    # 3. Otherwise, sentences that open with or contain a sequence cue.
    cued = [s for s in sentences if SEQ_CUES.search(s)]
    if len(cued) < 2:
        return []

    steps = []
    for s in cued[:max_steps]:
        m = SEQ_CUES.search(s)
        tail = s[m.end():].strip(" ,;:—-") or s
        steps.append(_short_label(tail))
    return [s for s in steps if s]


CAPTION_NOISE = re.compile(
    r"\b(fig(?:ure)?\.?\s*\d|table\s*\d|block diagram|diagram summaris|"
    r"summaris(?:es|ing)|shown below|as shown|see (?:fig|table))\b", re.IGNORECASE)


def _short_label(text: str, max_words: int = 4) -> str:
    """Compress a clause to a diagram-sized label.

    Truncation happens at word boundaries: a label ending in "meas" reads as a
    rendering bug, which costs more trust than the extra word would have cost
    space.
    """
    text = re.sub(r"^(the|a|an|is|are|we|it|this|that)\s+", "", text.strip(),
                  flags=re.IGNORECASE)
    wds = re.findall(r"[A-Za-z][A-Za-z'-]*|\d+", text)
    keep, seen_stop = [], 0
    for w in wds:
        if w.lower() in STOPWORDS and keep:
            seen_stop += 1
            if seen_stop > 1:
                break
            continue
        keep.append(w)
        if len(keep) >= max_words:
            break

    label = " ".join(keep)
    if len(label) > 26:                      # trim whole words, never mid-word
        out = []
        for w in keep:
            if len(" ".join(out + [w])) > 26:
                break
            out.append(w)
        label = " ".join(out) or keep[0][:26]
    return label


def build_flowchart(title: str, steps: list[str]) -> str:
    """Vertical flow of ordered steps, sized to the reading column."""
    box_h = 28 if len(steps) > 5 else 30
    gap = 16 if len(steps) > 5 else 20
    height = 26 + len(steps) * (box_h + gap)
    rows = []
    for i, s in enumerate(steps):
        y = 26 + i * (box_h + gap)
        label = html.escape(s)
        cls = "dg-box dg-active" if i == len(steps) - 1 else "dg-box"
        rows.append(
            f'<rect x="46" y="{y}" width="228" height="{box_h}" rx="3" class="{cls}"/>'
            f'<text x="160" y="{y + box_h / 2 + 4:.0f}" class="dg-lbl" '
            f'text-anchor="middle">{label}</text>')
        if i < len(steps) - 1:
            rows.append(
                f'<path d="M160 {y + box_h} L160 {y + box_h + gap - 6}" class="dg-arrow"/>'
                f'<path d="M156 {y + box_h + gap - 10} L160 {y + box_h + gap - 4} '
                f'L164 {y + box_h + gap - 10}" class="dg-arrow"/>')

    return (f'<svg viewBox="0 0 320 {height}" role="img" '
            f'aria-label="Process described in {html.escape(title)}">'
            f'<text x="0" y="12" class="dg-cap">SEQUENCE IN THIS SECTION</text>'
            + "".join(rows) + "</svg>")


# --------------------------------------------------------------- structure
# A diagram is only worth drawing when the prose has a shape a diagram can
# hold: an ordered process, a decomposition, a contrast, or a causal chain.
# Anything else gets no diagram at all — a box of keywords is not a diagram,
# and rendering one anyway trains the reader to ignore the figure.

NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "both": 2, "several": 3}

_TAX_NOUNS = (r"kinds?|types?|categories|classes|stages?|phases?|components?|"
              r"parts?|levels?|tiers?|layers?|dimensions?|forms?|varieties")

TAXONOMY_CUE = re.compile(
    # "three kinds", "two stages" — a count makes the noun sufficient on its own
    rf"\b(two|three|four|five|six|\d+)\s+({_TAX_NOUNS})\b"
    # "kinds of X" — no count, so the "of" is what marks it as a decomposition
    rf"|\b({_TAX_NOUNS})\s+of\b"
    r"|\b(?:consists? of|comprises?|is divided into|are divided into|"
    r"is organised into|is organized into|is made up of|falls? into|"
    r"can be grouped into|is composed of|breaks? down into)\b", re.IGNORECASE)

CONTRAST_CUE = re.compile(
    r"\b(whereas|in contrast|by contrast|unlike|on the other hand|"
    r"rather than|instead of|as opposed to|compared (?:to|with)|"
    r"differs? from|while .{5,60}?,)\b", re.IGNORECASE)

CAUSAL_CUE = re.compile(
    r"\b(leads? to|results? in|resulting in|causes?|gives? rise to|"
    r"produces?|drives?|triggers?|so that|therefore|consequently|thereby|"
    r"which (?:reduces?|increases?|improves?|degrades?|lowers?|raises?)|"
    r"because|since|due to)\b", re.IGNORECASE)


def _subject_of(sentence: str, max_words: int = 4) -> str:
    """The noun phrase a sentence opens with, as a diagram label."""
    s = re.sub(r"^(the|a|an|this|that|these|those|its|their)\s+", "",
               sentence.strip(), flags=re.IGNORECASE)
    m = re.match(r"^(.{2,60}?)\s+(?:is|are|was|were|arises?|comes?|refers?|"
                 r"means?|consists?|includes?|involves?|describes?|denotes?|"
                 r"occurs?|happens?|applies|covers?)\b", s, re.IGNORECASE)
    head = m.group(1) if m else s
    return _short_label(head, max_words)


def extract_taxonomy(sentences: list[str]) -> tuple[str, list[str]] | None:
    """A parent concept and the members it decomposes into."""
    for i, s in enumerate(sentences):
        m = TAXONOMY_CUE.search(s)
        if not m:
            continue

        count_tok = (m.group(1) or "").lower()
        want = int(count_tok) if count_tok.isdigit() else NUM_WORDS.get(count_tok, 0)

        parent = _short_label(s[:m.start()], 5) or _subject_of(s, 5)
        tail = s[m.end():]

        # "is organised into two stages" — the cue matched the verb phrase, so
        # the count sits after it rather than before.
        if not want:
            m2 = re.match(r"\s*(two|three|four|five|six|\d+)\b", tail, re.IGNORECASE)
            if m2:
                tok = m2.group(1).lower()
                want = int(tok) if tok.isdigit() else NUM_WORDS.get(tok, 0)

        # A stated count is authoritative: "two stages" really does mean two.
        # Without one, demand at least three members before calling it a
        # decomposition, so an incidental "parts of" doesn't trigger a diagram.
        need = want if want >= 2 else 3

        # (a) members enumerated in the same sentence, after a colon or as a list
        if ":" in tail:
            tail = tail.split(":", 1)[1]
        parts = re.split(r",\s*(?:and\s+)?|\s+and\s+|;\s*", tail.strip(" .:"))
        parts = [_short_label(p, 4) for p in parts if p.strip()]
        parts = [p for p in parts if p and len(p) > 2]
        if len(parts) >= need:
            return parent, parts[:6]

        # (b) members introduced by the sentences that follow
        follow = sentences[i + 1: i + 1 + max(need, 4)]
        subs = [_subject_of(f) for f in follow]
        subs = [x for x in subs if x and len(x) > 2]
        if len(subs) >= need:
            return parent, subs[:need if want else 6]
    return None


def extract_contrast(sentences: list[str]) -> tuple[str, str, list[tuple[str, str]]] | None:
    """Two things being set against each other, with their differing traits."""
    pairs: list[tuple[str, str]] = []
    for s in sentences:
        m = CONTRAST_CUE.search(s)
        if not m:
            continue
        left, right = s[:m.start()].strip(" ,;"), s[m.end():].strip(" ,;")
        if len(left.split()) < 3 or len(right.split()) < 3:
            continue
        pairs.append((_short_label(left, 5), _short_label(right, 5)))
        if len(pairs) == 4:
            break
    if len(pairs) < 2:
        return None
    return "This", "That", pairs


def extract_causal(sentences: list[str]) -> list[str] | None:
    """A cause-and-effect chain: A leads to B leads to C."""
    for s in sentences:
        m = CAUSAL_CUE.search(s)
        if not m:
            continue
        marker = m.group(0).lower()
        head = marker.split()[0]
        cause, effect = s[:m.start()].strip(" ,;"), s[m.end():].strip(" ,;")

        if head in {"because", "since", "due"}:
            if m.start() <= 2:
                # "Because A, B" — nothing precedes the marker, so the two
                # clauses divide at the first comma rather than at the marker.
                rest = s[m.end():].strip(" ,;")
                bits = re.split(r",\s+", rest, maxsplit=1)
                if len(bits) < 2 or len(bits[1].split()) < 3:
                    continue
                cause, effect = bits[0], bits[1]
            else:
                # "B because A" — the marker still separates them, reversed.
                cause, effect = effect, cause

        # Two words is a legitimate node — "Comprehension degrades" says
        # everything a diagram box needs to. Demanding three discards real
        # causal statements for no gain.
        if len(cause.split()) < 2 or len(effect.split()) < 2:
            continue

        # Label from the clause adjacent to the marker. Taking the first words
        # of a long sentence labels the node with its opening subordinate
        # clause — "Subsequent" — rather than the thing actually causing the
        # effect. Trailing "…, which" splits to an empty part, so take the
        # last clause that has content rather than simply the last one.
        clauses = [p.strip() for p in re.split(r",\s+", cause)]
        clauses = [p for p in clauses if len(p.split()) >= 2
                   and p.lower() not in {"which", "that"}]
        if clauses:
            cause = clauses[-1]
        chain = [_short_label(cause, 4), _short_label(effect, 4)]
        # Extend if the effect clause itself contains a further consequence.
        m2 = CAUSAL_CUE.search(effect)
        if m2 and len(effect[m2.end():].split()) >= 3:
            chain[1] = _short_label(effect[:m2.start()], 4)
            chain.append(_short_label(effect[m2.end():], 4))
        chain = [c for c in chain if c and len(c) > 2]
        if len(chain) >= 2:
            return chain[:4]
    return None


# ------------------------------------------------------------- renderers

def _wrap(text: str, width: int = 22) -> list[str]:
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > width and line:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out[:2]


def build_taxonomy(parent: str, children: list[str]) -> str:
    """Parent box with its members branching below."""
    n = len(children)
    col_w, gap = 96, 10
    total = n * col_w + (n - 1) * gap
    scale = min(1.0, 300 / total)
    col_w, gap = col_w * scale, gap * scale
    total = n * col_w + (n - 1) * gap
    x0 = (320 - total) / 2

    parts = [f'<text x="0" y="12" class="dg-cap">BREAKS DOWN INTO</text>',
             f'<rect x="86" y="22" width="148" height="30" rx="3" class="dg-box dg-active"/>',
             f'<text x="160" y="41" class="dg-lbl" text-anchor="middle">'
             f'{html.escape(parent[:24])}</text>',
             f'<path d="M160 52 L160 66" class="dg-arrow"/>']

    for i, c in enumerate(children):
        cx = x0 + i * (col_w + gap) + col_w / 2
        parts.append(f'<path d="M{x0 + col_w/2:.0f} 66 L{x0 + total - col_w/2:.0f} 66" class="dg-arrow"/>')
        parts.append(f'<path d="M{cx:.0f} 66 L{cx:.0f} 78" class="dg-arrow"/>')
        lines = _wrap(c, 14)
        parts.append(f'<rect x="{cx - col_w/2:.0f}" y="78" width="{col_w:.0f}" '
                     f'height="42" rx="3" class="dg-box"/>')
        for k, ln in enumerate(lines):
            parts.append(f'<text x="{cx:.0f}" y="{95 + k * 13}" class="dg-lbl" '
                         f'text-anchor="middle" style="font-size:10px">{html.escape(ln)}</text>')
    return (f'<svg viewBox="0 0 320 132" role="img" aria-label="{html.escape(parent)} '
            f'breaks into {len(children)} parts">' + "".join(parts) + "</svg>")


def build_contrast(pairs: list[tuple[str, str]]) -> str:
    """Two columns, one row per point of difference."""
    row_h = 34
    height = 40 + len(pairs) * row_h
    parts = ['<text x="0" y="12" class="dg-cap">SET AGAINST EACH OTHER</text>',
             '<line x1="160" y1="20" x2="160" y2="%d" class="dg-axis" '
             'stroke-dasharray="3 3"/>' % (height - 4)]
    for i, (l, r) in enumerate(pairs):
        y = 24 + i * row_h
        parts.append(f'<rect x="0" y="{y}" width="150" height="{row_h - 8}" rx="3" class="dg-box"/>')
        parts.append(f'<rect x="170" y="{y}" width="150" height="{row_h - 8}" rx="3" '
                     f'class="dg-box dg-active"/>')
        for x, txt in ((75, l), (245, r)):
            lines = _wrap(txt, 24)
            for k, ln in enumerate(lines):
                parts.append(f'<text x="{x}" y="{y + (14 if len(lines) == 1 else 10) + k * 11}" '
                             f'class="dg-lbl" text-anchor="middle" style="font-size:10px">'
                             f'{html.escape(ln)}</text>')
    return (f'<svg viewBox="0 0 320 {height}" role="img" '
            f'aria-label="Contrast across {len(pairs)} points">' + "".join(parts) + "</svg>")


def build_causal(chain: list[str]) -> str:
    """Cause and effect, left to right."""
    n = len(chain)
    box_h = 44
    height = 30 + box_h
    gap = 14
    box_w = (320 - (n - 1) * gap) / n
    parts = ['<text x="0" y="12" class="dg-cap">CAUSE AND EFFECT</text>']
    for i, node in enumerate(chain):
        x = i * (box_w + gap)
        cls = "dg-box dg-active" if i == n - 1 else "dg-box"
        parts.append(f'<rect x="{x:.0f}" y="22" width="{box_w:.0f}" height="{box_h}" '
                     f'rx="3" class="{cls}"/>')
        lines = _wrap(node, int(box_w / 5))
        for k, ln in enumerate(lines):
            parts.append(f'<text x="{x + box_w/2:.0f}" y="{(42 if len(lines) == 1 else 37) + k * 12}" '
                         f'class="dg-lbl" text-anchor="middle" style="font-size:10px">'
                         f'{html.escape(ln)}</text>')
        if i < n - 1:
            ax = x + box_w
            parts.append(f'<path d="M{ax:.0f} 44 L{ax + gap - 4:.0f} 44" class="dg-arrow"/>')
            parts.append(f'<path d="M{ax + gap - 8:.0f} 40 L{ax + gap - 3:.0f} 44 '
                         f'L{ax + gap - 8:.0f} 48" class="dg-arrow"/>')
    return (f'<svg viewBox="0 0 320 {height}" role="img" '
            f'aria-label="Causal chain of {n} steps">' + "".join(parts) + "</svg>")


def simplify_diagram(kind: str, title: str, steps: list[str],
                     chain: list[str]) -> str:
    """A reduced version of the same diagram, for the top rung.

    Visual complexity is load too. A seven-box pipeline read under overload is
    seven things to hold; the endpoints alone are two. The full version stays
    one tap away, so this withholds detail rather than destroying it.
    """
    if kind == "flow" and len(steps) >= 4:
        return build_flowchart(title, [steps[0], "…", steps[-1]])
    if kind == "causal" and len(chain) >= 3:
        return build_causal([chain[0], chain[-1]])
    return ""


def choose_diagram(title: str, sentences: list[str]) -> tuple[str, str, str]:
    """Pick the diagram the section's own structure calls for.

    Returns (svg, kind, simplified_svg); kind 'none' means the prose has no
    shape worth drawing and the reader is better served without a figure.
    """
    steps = extract_steps(sentences)
    if len(steps) >= 3:
        return (build_flowchart(title, steps), "flow",
                simplify_diagram("flow", title, steps, []))

    tax = extract_taxonomy(sentences)
    if tax:
        return build_taxonomy(*tax), "taxonomy", ""

    con = extract_contrast(sentences)
    if con:
        return build_contrast(con[2]), "contrast", ""

    cause = extract_causal(sentences)
    if cause and len(cause) >= 2:
        return (build_causal(cause), "causal",
                simplify_diagram("causal", title, [], cause))

    return "", "none", ""


def build_rescue(title: str, sentences: list[str], scores: list[float],
                 terms: list[str], earlier_terms: dict[str, str]) -> tuple[dict[str, str], str]:
    order = sorted(range(len(sentences)), key=lambda i: -scores[i])

    # Explain: the best sentence that isn't the one already shown as essential.
    explain = ""
    for i in order[1:]:
        if DEF_CUES.search(sentences[i]) or len(sentences[i].split()) > 10:
            explain = " ".join(simplify(sentences[i]))
            break
    if not explain and len(sentences) > 1:
        explain = " ".join(simplify(sentences[order[-1]]))
    if not explain:
        explain = "This section states a single idea; there is no second framing of it in the text."

    # Example: a sentence carrying an explicit example cue.
    example = next((" ".join(simplify(s)) for s in sentences if EXAMPLE_CUES.search(s)), "")
    if not example:
        example = ("The document doesn't give a worked example here. The terms it leans on are: "
                   + ", ".join(terms[:3]) + "." if terms else
                   "The document doesn't give a worked example for this section.")

    # Prerequisite: terms used here that an earlier section introduced.
    hits = [(t, sec) for t, sec in earlier_terms.items()
            if re.search(rf"\b{re.escape(t)}\b", " ".join(sentences), re.IGNORECASE)]
    if hits:
        prereq = "This builds on " + "; ".join(
            f"“{t}” (introduced in {sec})" for t, sec in hits[:3]) + "."
    else:
        prereq = ("Nothing earlier in the document is assumed here — this is where "
                  f"{terms[0]} first appears." if terms else
                  "This section doesn't depend on anything earlier in the document.")

    # The diagram type comes from the section's own structure: an ordered
    # process, a decomposition, a contrast, or a causal chain. If the prose has
    # none of those shapes, no diagram is drawn.
    diagram, kind, simple = choose_diagram(title, sentences)

    return ({"explain": explain, "example": example, "prereq": prereq,
             "diagram": diagram, "diagram_simple": simple}, kind)


# ------------------------------------------------------------------ models

@dataclass
class Concept:
    id: str
    eyebrow: str
    title: str
    tiers: dict[str, str]
    rescue: dict[str, str]
    terms: list[str] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)   # key sentence per tier
    diagram_kind: str = "map"                            # map | flow | ...
    tables: list = field(default_factory=list)
    images: list = field(default_factory=list)


@dataclass
class Document:
    id: str
    title: str
    source: str
    n_words: int
    concepts: list[Concept]
    generator: str = "extractive"
    # Bumped whenever the parser changes shape. A stored document with an older
    # number is re-ingested from its original file on next open, so a parser
    # change never leaves stale diagrams or tiers on screen.
    schema: int = SCHEMA
    stored: str = ""          # filename under data/uploads, for re-ingestion
    completed: bool = False
    completed_at: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


GENERIC_HEADINGS = re.compile(
    r"^\s*(abstract|introduction|summary|overview|contents?|table of contents|"
    r"background|conclusion|references|bibliography|index|preface|chapter\s*\d*)\s*$",
    re.IGNORECASE)

NUM_PREFIX = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)\s*")


def _title_from(blocks, fallback: str) -> str:
    """Best available document title, in order of reliability."""
    # 1. An explicit declaration on a cover page — the most trustworthy signal.
    for blk in blocks[:2]:
        m = re.search(r"(?:^|\n)\s*Title\s*[:\-–]\s*(.+)", blk["text"])
        if m:
            return m.group(1).strip().rstrip(".")[:90]

    # 2. The first heading that names a topic rather than a document part.
    #    Numbering is stripped first, so "1. Introduction" is still generic.
    for blk in blocks:
        h = blk.get("heading")
        if not h:
            continue
        clean = NUM_PREFIX.sub("", h.strip())
        if 3 <= len(clean) <= 90 and not GENERIC_HEADINGS.match(clean):
            return clean

    # 3. The filename.
    return fallback


def _is_front_matter(heading: str | None, sentences: list[str]) -> bool:
    """Cover pages, author blocks and reference lists aren't readable concepts.

    They are mostly proper nouns and fragments rather than prose, so they
    produce meaningless tiers and pollute the term index used for
    prerequisites. Detected by low prose density rather than by keyword.
    """
    if heading and re.match(r"^\s*(references|bibliography|acknowledge?ments?)\s*$",
                            heading, re.IGNORECASE):
        return True
    body = " ".join(sentences)
    wds = body.split()
    if len(wds) < 25:
        return True
    caps = sum(1 for w in wds if w[:1].isupper())
    stops = sum(1 for w in words(body) if w in STOPWORDS)
    # Real prose is roughly a third stopwords and mostly lowercase.
    return (caps / len(wds) > 0.42) and (stops / max(len(wds), 1) < 0.22)


def ingest(path: Path, title: str | None = None, max_concepts: int = 24,
           source_name: str | None = None, stored: str | None = None,
           assets_root: Path | None = None) -> Document:
    """Full pipeline: file on disk -> Document ready for the reader.

    source_name lets the caller record the name the user actually uploaded,
    rather than the de-duplicated filename it was stored under; stored records
    that de-duplicated name so the document can be re-parsed later.
    """
    doc_id = uuid.uuid4().hex[:10]
    assets_dir = (C.ASSETS / doc_id) if assets_root is None else assets_root / doc_id
    raw_blocks = read_document(path, assets_dir)

    # Title selection sees every heading, including ones with no body of its
    # own; concept building only sees blocks that carry prose or a figure.
    blocks = []
    for b in raw_blocks:
        b = dict(b)
        b["text"] = clean_text(b.get("text") or "")
        if b["text"].strip() or b.get("tables") or b.get("images"):
            blocks.append(b)
    if not blocks:
        raise ValueError("No readable text found in that file.")

    # Merge blocks too short to stand as their own concept, carrying their
    # figures forward rather than dropping them with the text.
    merged: list[dict] = []
    for b in blocks:
        if (merged and len(b["text"].split()) < 30 and not b.get("heading")
                and not b.get("tables") and not b.get("images")):
            merged[-1]["text"] += " " + b["text"]
        else:
            merged.append(b)
    blocks = merged[:max_concepts]

    concepts: list[Concept] = []
    earlier_terms: dict[str, str] = {}
    n = 0

    for blk in blocks:
        heading, body = blk.get("heading"), blk["text"]
        tables, images = blk.get("tables", []), blk.get("images", [])
        sentences = split_sentences(body)
        # A block with a figure but little prose is still worth keeping: the
        # figure may be the whole point of the section.
        if not sentences and not tables and not images:
            continue
        # A cover page carrying a crest is still a cover page. Only a table
        # earns a block an exemption from the front-matter filter, because a
        # table is content in a way a decorative image is not.
        if sentences and not tables and _is_front_matter(heading, sentences):
            continue
        n += 1
        if not sentences:
            sentences = [heading or f"Figure in section {n}."]

        head = heading or " ".join(sentences[0].split()[:7]).rstrip(".,;:")
        # "3." / "3.1" / "IV." prefixes, including the trailing dot
        head = re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)\s*", "", head).strip()
        head = (head[:70] + "…") if len(head) > 70 else head

        title_terms = set(content_words(head))
        scores = score_sentences(sentences, title_terms)
        terms = key_terms(sentences)
        tiers, keys = build_tiers(sentences, scores)
        rescue, kind = build_rescue(head, sentences, scores, terms, dict(earlier_terms))

        concepts.append(Concept(
            id=f"s{n}",
            eyebrow=f"{n}",
            title=head or f"Section {n}",
            tiers=tiers,
            rescue=rescue,
            terms=terms,
            keys=keys,
            diagram_kind=kind,
            tables=tables,
            images=[{**im, "context": assets.figure_context(body, k + 1)}
                    for k, im in enumerate(images)],
        ))
        for t in terms[:2]:
            earlier_terms.setdefault(t, f"§{n}")

    if not concepts:
        raise ValueError("Couldn't segment that document into readable sections.")

    total = sum(len(c.tiers["full"].split()) for c in concepts)
    display = source_name or path.name
    return Document(
        id=doc_id,
        title=title or _title_from(raw_blocks, Path(display).stem.replace("_", " ")),
        source=display,
        n_words=total,
        concepts=concepts,
        stored=stored or path.name,
    )


# ------------------------------------------------------------------ storage

def save(doc: Document, store: Path) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    out = store / f"{doc.id}.json"
    out.write_text(json.dumps(doc.to_dict(), indent=1), encoding="utf-8")
    return out


def load(doc_id: str, store: Path, uploads: Path | None = None) -> dict:
    """Load a stored document, re-parsing it first if the parser has moved on.

    Without this, a parser change leaves every previously uploaded reading
    showing output from the old code — the diagrams and tiers on screen would
    no longer match what the current pipeline produces, with nothing to
    indicate why.
    """
    path = store / f"{doc_id}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))

    if doc.get("schema") == SCHEMA or uploads is None:
        return doc

    original = uploads / doc.get("stored", "")
    if not doc.get("stored") or not original.exists():
        doc["stale"] = True          # source gone; the reader is told to re-upload
        return doc

    fresh = ingest(original, title=doc.get("title"),
                   source_name=doc.get("source"), stored=doc.get("stored"))
    fresh.id = doc["id"]             # keep the id so links and the active doc hold
    fresh.completed = doc.get("completed", False)
    fresh.completed_at = doc.get("completed_at")
    out = fresh.to_dict()
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def catalogue(store: Path) -> list[dict]:
    """Light index for the library screen — no concept bodies."""
    items = []
    for p in sorted(store.glob("*.json"), key=lambda q: -q.stat().st_mtime):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append({"id": d["id"], "title": d["title"], "source": d["source"],
                      "n_words": d["n_words"], "n_concepts": len(d["concepts"]),
                      "generator": d.get("generator", "extractive"),
                      "completed": d.get("completed", False),
                      "stale": d.get("schema") != SCHEMA})
    return items


if __name__ == "__main__":
    import sys
    doc = ingest(Path(sys.argv[1]))
    print(f"{doc.title}  —  {len(doc.concepts)} concepts, {doc.n_words} words\n")
    for c in doc.concepts[:4]:
        print(f"[{c.eyebrow}] {c.title}")
        print(f"   terms    : {', '.join(c.terms)}")
        print(f"   essential: {c.tiers['essential'][:150]}")
        print(f"   plain    : {c.tiers['plain'][:150]}")
        print(f"   prereq   : {c.rescue['prereq'][:150]}\n")