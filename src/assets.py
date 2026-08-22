"""Figures and tables: extraction, storage, and what can honestly be said.

Documents carry meaning in things that are not prose. A methodology table or a
results figure is often the densest object on the page, and dropping it during
ingestion silently deletes the part the reader most needs.

Two different problems live here, and they deserve different treatment:

* A **table** is structured data. Its content can be genuinely analysed —
  shape, which columns are numeric, where the extremes fall, how a column
  ranges. Those statements are computed from the cells and are true.

* A **figure** is pixels. Without a vision model nothing can be said about
  what it depicts, so nothing is. What *can* be recovered is its caption and
  the sentence in the body that refers to it, both of which are the author's
  own words. Inventing a description of an image the system cannot see would
  be the single most damaging thing this pipeline could do, because it would
  be indistinguishable from a real one.
"""
from __future__ import annotations

import hashlib
import html
import re
import statistics
from pathlib import Path

NUM = re.compile(r"^\s*[-+]?[\d,]*\.?\d+\s*%?\s*$")
FIG_REF = re.compile(r"\b(fig(?:ure)?|table|chart|plot|graph)\.?\s*([0-9]+(?:\.[0-9]+)?)",
                     re.IGNORECASE)


# ------------------------------------------------------------------ helpers

def as_number(cell: str) -> float | None:
    """Parse a cell as a number, tolerating thousands separators and percents."""
    if not cell or not NUM.match(cell):
        return None
    try:
        return float(cell.replace(",", "").replace("%", "").strip())
    except ValueError:
        return None


INDEX_HEADER = re.compile(
    r"^\s*(s\.?\s*no\.?|sl\.?\s*no\.?|sr\.?\s*no\.?|no\.?|#|index|serial|"
    r"item|id|rank)\s*$", re.IGNORECASE)


def _is_index_column(header: str, vals: list[float]) -> bool:
    """A row counter, not data.

    Reporting that "S.No is highest for the last row" is arithmetically true
    and completely worthless. Index columns are detected both by their header
    and by their shape — a run of consecutive integers starting at 0 or 1.
    """
    if INDEX_HEADER.match(header or ""):
        return True
    if len(vals) < 3:
        return False
    if not all(float(v).is_integer() for v in vals):
        return False
    ints = [int(v) for v in vals]
    return ints == list(range(ints[0], ints[0] + len(ints))) and ints[0] in (0, 1)


def numeric_columns(headers: list[str], rows: list[list[str]]) -> dict[int, list[float]]:
    """Columns where most cells parse as numbers, with their values.

    Index columns are excluded: they are numeric but carry no information.
    """
    out: dict[int, list[float]] = {}
    for c in range(len(headers)):
        vals = [as_number(r[c]) for r in rows if c < len(r)]
        good = [v for v in vals if v is not None]
        if not good or len(good) < max(2, int(0.6 * len(vals))):
            continue
        if _is_index_column(headers[c] if c < len(headers) else "", good):
            continue
        out[c] = good
    return out


def label_column(headers: list[str], rows: list[list[str]],
                 numeric: dict[int, list[float]]) -> int:
    """The column that names each row — the first non-numeric one."""
    for c in range(len(headers)):
        if c not in numeric:
            return c
    return 0


# ------------------------------------------------------------- table sense

def is_key_value(headers: list[str], rows: list[list[str]]) -> bool:
    """A two-column metadata table: field name on the left, value on the right.

    These are self-describing. Analysing one adds nothing a reader cannot see.
    """
    if len(headers) != 2 or len(rows) < 2:
        return False
    left = [(r[0] or "").strip() for r in rows if len(r) >= 2]
    if len(left) != len(rows) or len(set(l.lower() for l in left)) != len(left):
        return False
    return all(len(l.split()) <= 4 for l in left)


def categorical_columns(headers: list[str], rows: list[list[str]],
                        numeric: dict[int, list[float]]) -> dict[int, dict[str, int]]:
    """Columns drawn from a small repeated vocabulary, with their counts.

    A column of High/Medium/Low is data even though it isn't numeric — the
    distribution is a real finding, and it is the thing a reader scanning
    thirty rows cannot see at a glance.
    """
    out: dict[int, dict[str, int]] = {}
    for c in range(len(headers)):
        if c in numeric:
            continue
        vals = [(r[c] or "").strip() for r in rows if c < len(r) and (r[c] or "").strip()]
        if len(vals) < 4:
            continue
        uniq = {}
        for v in vals:
            if len(v.split()) > 3:          # free text, not a category
                uniq = {}
                break
            uniq[v] = uniq.get(v, 0) + 1
        # A category repeats. All-distinct values are labels, not categories.
        if uniq and 2 <= len(uniq) <= max(2, len(vals) // 2):
            out[c] = dict(sorted(uniq.items(), key=lambda kv: -kv[1]))
    return out


def describe_table(headers: list[str], rows: list[list[str]],
                   caption: str = "") -> str:
    """What the table actually says, computed from its cells.

    Returns an empty string when there is nothing to compute. Restating the
    row count and the column names, which the reader can already see, is worse
    than silence: it teaches them the analysis is never worth reading.
    """
    if not rows or is_key_value(headers, rows):
        return ""

    numeric = numeric_columns(headers, rows)
    categorical = categorical_columns(headers, rows, numeric)
    if not numeric and not categorical:
        return ""

    lab = label_column(headers, rows, numeric)
    bits: list[str] = []

    for c, vals in list(numeric.items())[:3]:
        name = headers[c] or f"column {c + 1}"
        pairs = []
        for r in rows:
            v = as_number(r[c]) if c < len(r) else None
            if v is not None:
                pairs.append(((r[lab] if lab < len(r) else "").strip() or "—", v))
        if not pairs:
            continue
        hi = max(pairs, key=lambda p: p[1])
        lo = min(pairs, key=lambda p: p[1])
        mean = statistics.fmean(v for _, v in pairs)
        if hi[0] == lo[0] or hi[1] == lo[1]:
            bits.append(f"{name} is {hi[1]:g} throughout.")
        else:
            bits.append(
                f"{name} is highest for {hi[0]} ({hi[1]:g}) and lowest for "
                f"{lo[0]} ({lo[1]:g}); mean {mean:.4g}.")

    for c, counts in list(categorical.items())[:3]:
        name = headers[c] or f"column {c + 1}"
        total = sum(counts.values())
        top, n = next(iter(counts.items()))
        spread = ", ".join(f"{v} {k}" for k, v in list(counts.items())[:4])
        bits.append(f"{name}: {spread}" + ("." if len(counts) <= 4 else ", …")
                    + (f" Most rows are {top} ({n} of {total})."
                       if n > total / 2 else ""))

    return " ".join(bits)


def build_table_chart(headers: list[str], rows: list[list[str]]) -> str:
    """A bar chart of the table's first numeric column."""
    numeric = numeric_columns(headers, rows)
    if not numeric or len(rows) < 2:
        return ""
    c = next(iter(numeric))
    lab = label_column(headers, rows, numeric)

    pairs = []
    for r in rows[:8]:
        v = as_number(r[c]) if c < len(r) else None
        if v is not None:
            pairs.append(((r[lab] if lab < len(r) else "").strip()[:16] or "—", v))
    if len(pairs) < 2:
        return ""

    top = max(abs(v) for _, v in pairs) or 1
    row_h, x0 = 22, 108
    height = 26 + len(pairs) * row_h
    parts = [f'<text x="0" y="12" class="dg-cap">'
             f'{html.escape((headers[c] or "VALUES").upper()[:34])}</text>']
    for i, (name, v) in enumerate(pairs):
        y = 22 + i * row_h
        w = max(2, abs(v) / top * (300 - x0))
        parts.append(f'<text x="{x0 - 6}" y="{y + 12}" class="dg-lbl" '
                     f'text-anchor="end" style="font-size:10px">{html.escape(name)}</text>')
        parts.append(f'<rect x="{x0}" y="{y + 2}" width="{w:.0f}" height="14" '
                     f'rx="2" class="dg-fill-a"/>')
        parts.append(f'<text x="{x0 + w + 5:.0f}" y="{y + 13}" class="dg-cap">{v:g}</text>')
    return (f'<svg viewBox="0 0 320 {height}" role="img" '
            f'aria-label="Bar chart of {html.escape(headers[c] or "values")}">'
            + "".join(parts) + "</svg>")


# ------------------------------------------------------------ figure sense

def figure_context(text: str, index: int) -> str:
    """The sentence in the body that refers to this figure, if there is one.

    The author's own pointer to a figure is usually the best available gloss
    on it, and it is genuinely theirs.
    """
    for sent in re.split(r"(?<=[.!?])\s+", text):
        m = FIG_REF.search(sent)
        if m and (m.group(2) == str(index) or len(sent.split()) > 6):
            return sent.strip()
    return ""


def save_image(data: bytes, ext: str, dest_dir: Path) -> str:
    """Write an extracted image, named by content hash so re-ingest is stable."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(data).hexdigest()[:12]
    ext = (ext or "png").lstrip(".").lower()
    if ext not in {"png", "jpg", "jpeg", "gif", "webp", "bmp", "emf", "wmf"}:
        ext = "png"
    name = f"{digest}.{ext}"
    path = dest_dir / name
    if not path.exists():
        path.write_bytes(data)
    return name