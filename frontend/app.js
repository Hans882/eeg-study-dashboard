/* Adaptive Study Dashboard — frontend controller.
 *
 * Two views, switched by html[data-view]:
 *   library  — upload and pick a reading
 *   reading  — the adaptive reader
 *
 * Inside the reader, everything is driven by two attributes:
 *   html[data-state] = low | high              layout density
 *   html[data-tier]  = full | plain | essential  content complexity
 */

const root = document.documentElement;
const $ = id => document.getElementById(id);

/* Null-safe event binding. A missing element means a missing feature, never a
   dead script: an unguarded addEventListener on null throws and every
   statement after it — including the boot sequence — silently stops running. */
function on(id, event, handler) {
  const node = document.getElementById(id);
  if (node) node.addEventListener(event, handler);
  else console.warn(`[dashboard] #${id} not in the DOM — is index.html current?`);
  return node;
}

const el = {
  ribbon: $("ribbon"), playhead: $("playhead"),
  library: $("library"), libList: $("libList"), libError: $("libError"),
  dropzone: $("dropzone"), fileInput: $("fileInput"),
  dzTitle: $("dzTitle"), dzHint: $("dzHint"), dzBar: $("dzBar"),
  article: $("article"), toc: $("toc"), tierChip: $("tierChip"),
  docTitle: $("docTitle"), docMeta: $("docMeta"),
  railTitle: $("railTitle"), railSource: $("railSource"), termList: $("termList"),
  stateLabel: $("stateLabel"), meta: $("meta"), timer: $("timer"),
  statHigh: $("statHigh"), statAdapt: $("statAdapt"), statRescue: $("statRescue"),
  rescue: $("rescue"), rescueTopic: $("rescueTopic"), rescueBody: $("rescueBody"),
  recovery: $("recovery"), recoveryNote: $("recoveryNote"),
  breath: $("breath"), breathText: $("breathText"),
  spark: $("spark"), tlPanel: $("timelinePanel"), tlChart: $("tlChart"),
  whyPanel: $("whyPanel"), whyList: $("whyList"), whySub: $("whySub"),
  whyThresholds: $("whyThresholds"), budgetTable: $("budgetTable"),
  pager: $("pager"), pagerPos: $("pagerPos"), dnd: $("dndBanner"), dndDetail: $("dndDetail"),
  bands: Object.fromEntries([...document.querySelectorAll(".band")]
    .map(b => [b.dataset.band, b.querySelector("i")])),
};

/* A tab's content is live only while that tab is on screen — polling a chart
   nobody is looking at is wasted work. */
function panelTabVisible(name) {
  const panel = $("sidePanel");
  const body = document.querySelector(`.sp-body[data-panel="${name}"]`);
  return panel && !panel.hidden && body && !body.hidden;
}

const TIER_LABEL = { full: "Full text", plain: "Plain language", essential: "Essentials only" };

let doc = null;           // active document from the API
let tier = null;          // currently rendered tier
let state = "low";
let activeConcept = null;
let breakTimer = null;
let level = 0;             // 0 low, 1 elevated, 2 high, 3 overload
let pageIdx = 0;           // which concept is shown in single-concept mode
let paging = false;

const esc = s => String(s).replace(/[&<>"']/g,
  m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
const fmt = s => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const post = (path, body) =>
  fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body || {}) }).catch(() => {});
const logEvent = (kind, detail) => post("/api/event", { kind, detail: detail || {} });

/* ==================== library ==================== */
async function loadLibrary() {
  try {
    const r = await fetch("/api/documents");
    const { documents } = await r.json();
    el.libList.innerHTML = documents.length ? documents.map(d => `
      <article class="lib-item${d.stale ? " stale" : ""}" data-id="${esc(d.id)}">
        <div class="li-main">
          <h3>${d.completed ? '<span class="tick">\u2713</span> ' : ""}${esc(d.title)}</h3>
          <p class="li-meta">${d.n_concepts} sections &middot; ${d.n_words.toLocaleString()} words &middot; ${esc(d.source)}${
            d.stale ? ' &middot; <b class="li-stale">original file missing — re-upload to refresh</b>' : ""}</p>
        </div>
        <div class="li-actions">
          <button class="ghost" data-open="${esc(d.id)}">Read</button>
          <button class="icon-btn" data-del="${esc(d.id)}" aria-label="Delete">&times;</button>
        </div>
      </article>`).join("")
      : `<p class="lib-empty">Nothing here yet. Upload a reading to begin.</p>`;
  } catch {
    el.libList.innerHTML = `<p class="lib-empty">Backend unreachable.</p>`;
  }
}

el.libList.addEventListener("click", async e => {
  const open = e.target.closest("[data-open]");
  const del = e.target.closest("[data-del]");
  if (open) return openDocument(open.dataset.open);
  if (del) {
    if (!confirm("Delete this reading?")) return;
    await fetch(`/api/documents/${del.dataset.del}`, { method: "DELETE" });
    loadLibrary();
  }
});

async function openDocument(id) {
  const r = await fetch(`/api/documents/${id}/select`, { method: "POST" });
  if (!r.ok) return showError("Couldn't open that reading.");
  await loadActive();
  root.dataset.view = "reading";
  window.scrollTo(0, 0);
}

function showError(msg) {
  el.libError.textContent = msg;
  el.libError.hidden = false;
  setTimeout(() => { el.libError.hidden = true; }, 8000);
}

/* ---------- upload ---------- */
function busy(on, label) {
  el.dropzone.classList.toggle("busy", on);
  el.dzBar.hidden = !on;
  el.dzTitle.textContent = on ? label : "Drop a PDF, Word, Markdown or text file";
  el.dzHint.textContent = on ? "parsing and segmenting…" : "or click to choose · up to 25 MB";
}

async function upload(file) {
  if (!file) return;
  el.libError.hidden = true;
  busy(true, `Reading ${file.name}`);
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/api/documents", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Upload failed.");
    busy(false);
    await loadLibrary();
    await loadActive();
    root.dataset.view = "reading";
    window.scrollTo(0, 0);
  } catch (err) {
    busy(false);
    showError(err.message);
  }
}

el.dropzone.addEventListener("click", () => el.fileInput.click());
el.dropzone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
});
el.fileInput.addEventListener("change", e => upload(e.target.files[0]));
["dragenter", "dragover"].forEach(ev =>
  el.dropzone.addEventListener(ev, e => { e.preventDefault(); el.dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach(ev =>
  el.dropzone.addEventListener(ev, e => { e.preventDefault(); el.dropzone.classList.remove("over"); }));
el.dropzone.addEventListener("drop", e => upload(e.dataTransfer.files[0]));

on("backBtn", "click", backToLibrary);
on("backBtn2", "click", backToLibrary);
function backToLibrary() {
  root.dataset.view = "library";
  loadLibrary();
  window.scrollTo(0, 0);
}

/* ==================== reading ==================== */
async function loadActive() {
  try {
    const r = await fetch("/api/active");
    if (!r.ok) { doc = null; return; }
    doc = await r.json();
    tier = null;                       // force a re-render for the new document
    activeConcept = doc.concepts[0] || null;
    el.docTitle.textContent = doc.title;
    el.railTitle.textContent = doc.title;
    el.railSource.textContent = doc.source;
    el.docMeta.textContent = `${doc.concepts.length} sections · ${doc.n_words.toLocaleString()} words`;
    if (activeConcept) el.rescueTopic.textContent = activeConcept.title;
    paintComplete();
  } catch { doc = null; }
}

/* Wrap the key sentence in a <mark> so CSS can surface it only under high
   workload. Everything is escaped first, then the marker is inserted into the
   escaped string — never the reverse, or the document's own text could inject
   markup. */
function markKey(text, key) {
  const safe = esc(text);
  if (!key) return safe;
  const safeKey = esc(key);
  const at = safe.indexOf(safeKey);
  if (at === -1) return safe;
  return safe.slice(0, at) + "<mark>" + safeKey + "</mark>" + safe.slice(at + safeKey.length);
}

const FIG_CAPTION = {
  flow: "Sequence described above",
  taxonomy: "How this breaks down",
  contrast: "The two sides compared",
  causal: "What leads to what",
};

function render(t) {
  if (!doc || t === tier) return;
  tier = t;
  root.dataset.tier = t;
  el.tierChip.textContent = TIER_LABEL[t];

  el.article.innerHTML = doc.concepts.map((c, i) => {
    const body = markKey(c.tiers[t] || c.tiers.full, (c.keys || {})[t]);
    // Sections whose prose has no diagrammable shape get no figure at all.
    // The SVG is generated server-side and carries no document markup.
    const svg = c.rescue && c.rescue.diagram;
    const simple = c.rescue && c.rescue.diagram_simple;
    // Visual complexity is load too: at the top rung the reduced diagram is
    // shown, with the full one one tap away rather than gone.
    const figure = svg ? `
      <figure class="concept-fig" data-kind="${esc(c.diagram_kind || "")}">
        <div class="fig-full">${svg}</div>
        ${simple ? `<div class="fig-simple">${simple}</div>
          <button class="reveal fig-toggle" type="button">Show full diagram \u2192</button>` : ""}
        <figcaption>${esc(FIG_CAPTION[c.diagram_kind] || "")}</figcaption>
      </figure>` : "";
    return `
    <section class="concept" id="c-${esc(c.id)}" data-idx="${i}">
      <div class="concept-head">
        <span class="concept-num">${esc(c.eyebrow)}</span>
        <h2>${esc(c.title)}</h2>
      </div>
      <p>${body}</p>
      ${(c.tables || []).map(renderTable).join("")}
      ${(c.images || []).map(im => renderImage(im, doc.id)).join("")}
      ${t === "essential" ? '<button class="reveal" type="button" data-reveal="plain">Understand more \u2192</button>' : ""}
      ${figure}
    </section>`;
  }).join("");

  el.toc.innerHTML = doc.concepts.map(c =>
    `<li><a href="#c-${esc(c.id)}" data-id="${esc(c.id)}"><i>${esc(c.eyebrow)}</i><span>${esc(c.title)}</span></a></li>`
  ).join("");

  observeConcepts();
}


/* ---------- figures and tables ----------
   A table's analysis is computed from its cells and is true; a figure's gloss
   is the author's own caption and the sentence that refers to it. Nothing is
   said about what an image depicts, because nothing can be. */
const TABLE_PEEK = 8;

/* A two-column metadata table is a list of fields. Rendered as a grid it
   carries a header row, a column border and a scroll container that all say
   nothing; rendered as a definition list it says the same thing in less. */
function isKeyValue(t) {
  if (!t.headers || t.headers.length !== 2 || t.rows.length < 2) return false;
  const left = t.rows.map(r => (r[0] || "").trim());
  if (left.some(l => !l || l.split(/\s+/).length > 4)) return false;
  return new Set(left.map(l => l.toLowerCase())).size === left.length;
}

function renderKeyValue(t) {
  return `
    <figure class="doc-kv">
      <dl>${t.rows.map(r =>
        `<dt>${esc(r[0])}</dt><dd>${esc(r[1] || "")}</dd>`).join("")}</dl>
      ${t.caption ? `<figcaption>${esc(t.caption)}</figcaption>` : ""}
    </figure>`;
}

function renderTable(t, i) {
  if (isKeyValue(t)) return renderKeyValue(t);
  const head = `<tr>${t.headers.map(h => `<th>${esc(h)}</th>`).join("")}</tr>`;
  // A thirty-row table dropped whole into the reading is a wall. Show enough
  // to see its shape, keep the rest one tap away — the same disclosure logic
  // the prose tiers use.
  const body = t.rows.map((r, k) =>
    `<tr${k >= TABLE_PEEK ? ' class="row-extra"' : ""}>${
      r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("");
  const more = t.rows.length > TABLE_PEEK
    ? `<button class="reveal table-toggle" type="button">Show all ${t.rows.length} rows \u2192</button>`
    : "";
  return `
    <figure class="doc-table${t.rows.length > TABLE_PEEK ? " capped" : ""}" data-idx="${i}">
      <div class="table-scroll"><table>${head}${body}</table></div>
      ${more}
      ${t.caption ? `<figcaption>${esc(t.caption)}</figcaption>` : ""}
      ${t.summary ? `<p class="fig-read"><b>What it says</b> ${esc(t.summary)}</p>` : ""}
      ${t.chart ? `<div class="table-chart">${t.chart}</div>` : ""}
    </figure>`;
}

function renderImage(im, docId) {
  const gloss = im.caption || im.context;
  return `
    <figure class="doc-image">
      <img src="/assets/${esc(docId)}/${esc(im.file)}" alt="${esc(im.caption || "Figure from the document")}" loading="lazy">
      ${gloss ? `<figcaption>${esc(gloss)}</figcaption>` : ""}
    </figure>`;
}

/* ---------- track the concept in view ----------
   Exactly one concept is "in view" at a time: the one whose midpoint is
   closest to the middle of the screen. An IntersectionObserver marks every
   element that touches the viewport, which highlights three sections at once
   on short readings. */
let sections = [];
let rafPending = false;

function observeConcepts() {
  sections = [...document.querySelectorAll(".concept")];
  pickConcept();
}

function pickConcept() {
  if (!sections.length || !doc) return;
  const centre = window.innerHeight / 2;

  let best = null, bestDist = Infinity;
  for (const s of sections) {
    const r = s.getBoundingClientRect();
    if (r.bottom < 0 || r.top > window.innerHeight) continue;   // off-screen
    // Distance from the screen centre to the nearest point of this section.
    const dist = r.top > centre ? r.top - centre
              : r.bottom < centre ? centre - r.bottom
              : 0;
    if (dist < bestDist) { bestDist = dist; best = s; }
  }
  if (!best) return;

  sections.forEach(s => s.classList.toggle("in-view", s === best));

  const c = doc.concepts[+best.dataset.idx];
  if (c && (!activeConcept || c.id !== activeConcept.id)) {
    activeConcept = c;
    el.rescueTopic.textContent = c.title;
    el.termList.innerHTML = (c.terms || []).length
      ? c.terms.map(t => `<li>${esc(t)}</li>`).join("")
      : `<li class="muted">no repeated terms</li>`;
    closeRescueBody();
    document.querySelectorAll(".toc a").forEach(a =>
      a.classList.toggle("active", a.dataset.id === c.id));
  }
}

function onScroll() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; pickConcept(); });
}
window.addEventListener("scroll", onScroll, { passive: true });
window.addEventListener("resize", onScroll, { passive: true });

/* ---------- rescue ---------- */
function closeRescueBody() {
  el.rescueBody.hidden = true;
  el.rescueBody.innerHTML = "";
  document.querySelectorAll(".rescue-actions button").forEach(b => b.classList.remove("on"));
}

document.querySelectorAll(".rescue-actions button").forEach(btn => {
  btn.addEventListener("click", () => {
    if (!activeConcept) return;
    const kind = btn.dataset.rescue;
    const already = btn.classList.contains("on");
    closeRescueBody();
    if (already) return;
    btn.classList.add("on");
    const payload = activeConcept.rescue[kind];
    if (!payload) {
      el.rescueBody.innerHTML =
        "<p>This section doesn't describe a process, a breakdown or a "
        + "cause-and-effect chain, so there's no diagram that would say more "
        + "than the text already does.</p>";
      el.rescueBody.hidden = false;
      return;
    }
    // Only the diagram is trusted markup — generated server-side as SVG.
    el.rescueBody.innerHTML = kind === "diagram" ? payload : `<p>${esc(payload)}</p>`;
    el.rescueBody.hidden = false;
    logEvent("rescue", { concept: activeConcept.id, variant: kind, at_state: state });
  });
});

on("rescueClose", "click", () => {
  el.rescue.classList.add("dismissed");
  el.rescue.classList.remove("forced");
  logEvent("rescue_dismissed", { concept: activeConcept ? activeConcept.id : null });
});

/* ---------- recovery ---------- */
function showRecovery(show) {
  if (show === !el.recovery.hidden) return;
  el.recovery.hidden = !show;
  if (show) {
    el.breath.classList.remove("run");
    el.breathText.textContent = "Take a moment";
    el.recoveryNote.textContent = "";
    $("takeBreak").focus();
    logEvent("recovery_shown", {});
  }
}

on("takeBreak", "click", () => {
  post("/api/recovery", { action: "taken" });
  el.breath.classList.add("run");
  let left = 90;
  el.recoveryNote.textContent = `${left}s`;
  $("takeBreak").disabled = true;
  clearInterval(breakTimer);
  breakTimer = setInterval(() => {
    left -= 1;
    el.breathText.textContent = (left % 10 >= 4) ? "Breathe out" : "Breathe in";
    el.recoveryNote.textContent = `${left}s`;
    if (left <= 0) {
      clearInterval(breakTimer);
      $("takeBreak").disabled = false;
      el.breath.classList.remove("run");
      showRecovery(false);
    }
  }, 1000);
});

on("skipBreak", "click", () => {
  clearInterval(breakTimer);
  $("takeBreak").disabled = false;
  post("/api/recovery", { action: "dismissed" });
  showRecovery(false);
});

document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  if (!$("tour").hidden) { setTour(false); return; }
  if (!el.recovery.hidden) $("skipBreak").click();
});

/* ---------- timeline ---------- */
function path(points, w, h) {
  if (points.length < 2) return "";
  return points.map((p, i) =>
    `${i ? "L" : "M"}${(i / (points.length - 1) * w).toFixed(1)} ${((1 - p) * h).toFixed(1)}`
  ).join(" ");
}

function drawSpark(points) {
  const ps = points.slice(-80).map(p => p.p);
  if (ps.length < 2) return;
  el.spark.innerHTML =
    `<line class="mid" x1="0" y1="13" x2="160" y2="13"/>` +
    `<path class="trace" d="${path(ps, 160, 26)}"/>`;
}

function drawTimeline(points, stats) {
  const w = 900, h = 200;
  if (points.length < 2) { el.tlChart.innerHTML = ""; return; }

  let bands = "", runStart = null;
  points.forEach((p, i) => {
    if (p.state === 1 && runStart === null) runStart = i;
    if ((p.state === 0 || i === points.length - 1) && runStart !== null) {
      const x = runStart / (points.length - 1) * w;
      const x2 = i / (points.length - 1) * w;
      bands += `<rect class="hiband" x="${x.toFixed(1)}" y="0" width="${Math.max(1, x2 - x).toFixed(1)}" height="${h}"/>`;
      runStart = null;
    }
  });

  // Adaptation markers turn the chart from an EEG trace into a causal
  // interaction history: what the system did, and at which moment.
  const span = points[points.length - 1].t - points[0].t || 1;
  const marks = (window.__adaptations || []).filter(
      a => a.session_sec >= points[0].t).map(a => {
    const x = Math.max(2, Math.min(w - 2, (a.session_sec - points[0].t) / span * w));
    const up = a.direction === "escalation";
    return `<line class="tl-mark ${up ? "up" : "down"}" x1="${x.toFixed(1)}" y1="0"
              x2="${x.toFixed(1)}" y2="${h}"/>
            <text class="tl-marklbl" x="${(x + 4).toFixed(1)}" y="14">#${a.n} ${esc(a.to_state)}</text>`;
  }).join("");

  el.tlChart.innerHTML = bands +
    `<line class="thresh" x1="0" y1="${h / 2}" x2="${w}" y2="${h / 2}"/>` +
    `<path class="trace" d="${path(points.map(p => p.p), w, h)}"/>` + marks;

  $("tlNow").textContent = fmt(points[points.length - 1].t);
  $("tlHigh").textContent = Math.round(stats.high_ratio * 100) + "%";
  $("tlAdapt").textContent = stats.adaptations;
  $("tlOver").textContent = stats.overrides;
  $("tlRescue").textContent = stats.rescues;
  $("tlBreak").textContent = stats.breaks_taken;
  $("tlAgree").textContent = stats.agreement == null ? "—"
    : Math.round(stats.agreement * 100) + "%";
}

async function pollHistory() {
  try {
    const r = await fetch("/api/history");
    if (!r.ok) return;
    const { points, stats } = await r.json();
    if (panelTabVisible("timeline")) {
      try {
        const a = await (await fetch("/api/adaptations")).json();
        window.__adaptations = a.adaptations;
      } catch { /* markers are optional decoration on the chart */ }
    }
    drawSpark(points);
    if (panelTabVisible("timeline")) drawTimeline(points, stats);
    el.statHigh.textContent = Math.round(stats.high_ratio * 100) + "%";
    el.statAdapt.textContent = stats.adaptations;
    el.statRescue.textContent = stats.rescues;
  } catch { /* /api/state already surfaces backend failure */ }
}



/* ---------- single-concept mode (top rung) ----------
   At overload the reader stops scrolling and advances deliberately. Removing
   the "should I keep scrolling?" decision is itself a reduction in extraneous
   load, and it makes the intervention a clean experimental condition. */
function applyPaging(on_) {
  if (on_ === paging) { if (on_) showPage(); return; }
  paging = on_;
  el.pager.hidden = !on_;
  if (on_) {
    const cur = doc && activeConcept
      ? doc.concepts.findIndex(c => c.id === activeConcept.id) : 0;
    pageIdx = Math.max(0, cur);
    showPage();
  } else {
    sections.forEach(sec => { sec.hidden = false; });
    pickConcept();
  }
}

function showPage() {
  if (!doc || !sections.length) return;
  pageIdx = Math.max(0, Math.min(pageIdx, sections.length - 1));
  sections.forEach((sec, i) => { sec.hidden = i !== pageIdx; });
  sections[pageIdx].classList.add("in-view");
  el.pagerPos.textContent = `Concept ${pageIdx + 1} of ${sections.length}`;
  const prev = $("prevConcept"), next = $("nextConcept");
  if (prev) prev.disabled = pageIdx === 0;
  if (next) next.disabled = pageIdx === sections.length - 1;
  const c = doc.concepts[pageIdx];
  if (c) { activeConcept = c; el.rescueTopic.textContent = c.title; closeRescueBody(); }
}

on("prevConcept", "click", () => { pageIdx -= 1; showPage(); logEvent("page", { dir: "prev" }); });
on("nextConcept", "click", () => { pageIdx += 1; showPage(); logEvent("page", { dir: "next" }); });

/* ---------- progressive disclosure ----------
   The essential tier hides detail rather than destroying it. One tap reveals
   the plain version, a second the full text, so a reader who is coping can
   always get the rest back without leaving the rung. */
el.article.addEventListener("click", e => {
  const tbl = e.target.closest(".table-toggle");
  if (tbl) {
    const f = tbl.closest(".doc-table");
    const shown = f.classList.toggle("expanded");
    const n = f.querySelectorAll("tbody tr, table tr").length - 1;
    tbl.textContent = shown ? "Show fewer rows \u2192" : `Show all ${n} rows \u2192`;
    logEvent("table_rows", { expanded: shown });
    return;
  }
  const fig = e.target.closest(".fig-toggle");
  if (fig) {
    const f = fig.closest(".concept-fig");
    f.classList.toggle("expanded");
    fig.textContent = f.classList.contains("expanded")
      ? "Show simpler diagram \u2192" : "Show full diagram \u2192";
    logEvent("diagram_detail", { expanded: f.classList.contains("expanded") });
    return;
  }
  const btn = e.target.closest("[data-reveal]");
  if (!btn || !doc) return;
  const sec = btn.closest(".concept");
  const c = doc.concepts[+sec.dataset.idx];
  const next = btn.dataset.reveal;
  sec.querySelector("p").innerHTML = markKey(c.tiers[next], (c.keys || {})[next]);
  if (next === "plain") { btn.dataset.reveal = "full"; btn.textContent = "Show full detail \u2192"; }
  else { btn.remove(); }
  logEvent("disclosure", { concept: c.id, revealed: next });
});


/* ---------- explain this sentence ----------
   Section-level help assumes the reader knows which section is the problem.
   Often they know exactly which sentence it is, and until now nothing smaller
   than a section was addressable. Alt-click, or double-click, any sentence. */
function sentenceAt(node, clientX, clientY) {
  const p = node.closest("p");
  if (!p) return null;
  const text = p.textContent;
  const parts = text.split(/(?<=[.!?])\s+/).filter(x => x.trim().length > 12);
  if (!parts.length) return null;
  // Map the click to a sentence by walking the text nodes' rectangles.
  const range = document.createRange();
  let offset = 0;
  for (const part of parts) {
    const idx = text.indexOf(part, offset);
    if (idx === -1) continue;
    offset = idx + part.length;
    try {
      const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
      let seen = 0, startNode = null, startOff = 0, endNode = null, endOff = 0, n;
      while ((n = walker.nextNode())) {
        const len = n.textContent.length;
        if (!startNode && seen + len > idx) { startNode = n; startOff = idx - seen; }
        if (startNode && seen + len >= idx + part.length) {
          endNode = n; endOff = idx + part.length - seen; break;
        }
        seen += len;
      }
      if (!startNode || !endNode) continue;
      range.setStart(startNode, startOff);
      range.setEnd(endNode, endOff);
      for (const r of range.getClientRects()) {
        if (clientX >= r.left && clientX <= r.right && clientY >= r.top && clientY <= r.bottom) {
          return part.trim();
        }
      }
    } catch { /* ranges across re-rendered nodes can fail; try the next */ }
  }
  return null;
}

async function explainSentence(text) {
  if (!text) return;
  el.rescue.classList.remove("dismissed");
  // The dock is normally revealed by the high rung. An explicit request from
  // the reader outranks that: asking a question at low workload should not
  // write the answer into a panel CSS is keeping invisible.
  el.rescue.classList.add("forced");
  closeRescueBody();
  el.rescueBody.hidden = false;
  el.rescueBody.innerHTML = "<p class='dim'>Working on that sentence…</p>";
  try {
    const r = await fetch("/api/explain", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, concept_id: activeConcept ? activeConcept.id : null }),
    });
    if (!r.ok) throw new Error();
    const d = await r.json();
    el.rescueBody.innerHTML = `
      <p class="quoted">${esc(d.original)}</p>
      ${d.changed ? `<p class="lbl">In plainer words</p><p>${esc(d.simple)}</p>`
                  : `<p class="dim">This sentence is already about as plain as it gets.</p>`}
      ${d.definitions.length ? `<p class="lbl">Terms it leans on</p>` +
        d.definitions.map(x => `<p><b>${esc(x.term)}</b> <span class="dim">(§ ${esc(x.section)})</span><br>${esc(x.text)}</p>`).join("") : ""}
      ${d.relationship ? `<p class="lbl">The relationship</p>${d.relationship}` : ""}`;
    el.rescueBody.hidden = false;   // a poll may have closed it mid-fetch
    logEvent("explain_sentence", { concept: activeConcept ? activeConcept.id : null,
                                   chars: text.length });
  } catch {
    el.rescueBody.innerHTML = "<p class='dim'>Couldn't explain that one.</p>";
  }
}

el.article.addEventListener("dblclick", e => {
  const s = sentenceAt(e.target, e.clientX, e.clientY);
  if (s) { e.preventDefault(); explainSentence(s); }
});
el.article.addEventListener("click", e => {
  if (!e.altKey) return;
  const s = sentenceAt(e.target, e.clientX, e.clientY);
  if (s) explainSentence(s);
});

/* ---------- the inspection panel ---------- */
const COST_ORDER = { essential: 0, helpful: 1, moderate: 2, high: 3 };

async function loadWhy() {
  try {
    const [aRes, bRes] = await Promise.all([fetch("/api/adaptations"), fetch("/api/budget")]);
    if (!aRes.ok || !bRes.ok) return;
    const { adaptations, total, thresholds } = await aRes.json();
    const { budget } = await bRes.json();

    el.whySub.textContent = total
      ? `${total} adaptation${total === 1 ? "" : "s"} this session \u00b7 most recent first`
      : "No adaptations yet this session.";

    el.whyThresholds.innerHTML = `
      <span>Enter elevated <b>p \u2265 ${thresholds.enter_elevated}</b></span>
      <span>Enter high <b>p \u2265 ${thresholds.enter_high}</b></span>
      <span>Step down <b>p \u2264 ${thresholds.exit}</b></span>
      <span>Min confidence <b>${thresholds.min_confidence}</b></span>
      <span>Dwell <b>${thresholds.dwell_sec}s</b></span>
      <span>Overload after <b>${thresholds.overload_after_sec}s</b></span>`;

    el.whyList.innerHTML = adaptations.length ? adaptations.map(a => `
      <article class="why-item ${a.direction === "escalation" ? "up" : "down"}">
        <header><b>Adaptation #${a.n}</b><span class="mono2">${esc(a.timestamp)}</span></header>
        <dl>
          <dt>Workload</dt><dd class="cap">${esc(a.to_state)}</dd>
          <dt>p(high)</dt><dd>${a.p_high} <span class="dim">(confidence ${a.confidence})</span></dd>
          <dt>Sustained high</dt><dd>${a.sustained_high_sec}s</dd>
          <dt>Trigger</dt><dd>${esc(a.trigger)}</dd>
          <dt>Previous held</dt><dd>${a.held_previous_for_sec}s</dd>
          <dt>Layout</dt><dd>${esc(a.from_layout)} \u2192 <b>${esc(a.to_layout)}</b></dd>
          <dt>Content tier</dt><dd class="cap">${esc(a.content_tier)}</dd>
          <dt>Elements withdrawn</dt><dd>${a.n_elements_withdrawn}${
            a.elements_withdrawn.length
              ? ` <span class="dim">\u2014 ${esc(a.elements_withdrawn.join(", "))}</span>` : ""}</dd>
        </dl>
      </article>`).join("")
      : `<p class="lib-empty">The interface hasn't changed yet. It will record its reasoning here the moment it does.</p>`;

    el.budgetTable.innerHTML =
      `<tr><th>Element</th><th>Extraneous cost</th><th>Status</th></tr>` +
      budget.slice().sort((x, y) => COST_ORDER[y.cost] - COST_ORDER[x.cost])
        .map(b => `<tr class="${b.withdrawn ? "gone" : ""}">
            <td>${esc(b.element)}</td>
            <td><span class="cost cost-${esc(b.cost)}">${esc(b.cost)}</span></td>
            <td>${b.withdrawn ? "withdrawn" : "shown"}</td></tr>`).join("");
  } catch { /* backend down; /api/state already reports it */ }
}



/* ---------- side panel ----------
   Controls, history and evidence are things you go and look at, not things
   that need to occupy screen space while reading. One opener, three tabs,
   closed by default. */
function setPanel(open, tab) {
  const panel = $("sidePanel"), scrim = $("panelScrim"), btn = $("panelBtn");
  if (!panel) return;
  panel.hidden = !open;
  if (scrim) scrim.hidden = !open;
  if (btn) btn.setAttribute("aria-expanded", String(open));
  if (open) {
    if (tab) showTab(tab);
    const active = document.querySelector(".sp-tabs button.on");
    if (active && active.dataset.tab === "timeline") pollHistory();
    if (active && active.dataset.tab === "why") loadWhy();
  }
}

function showTab(name) {
  document.querySelectorAll(".sp-tabs button").forEach(b =>
    b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".sp-body").forEach(b =>
    b.hidden = b.dataset.panel !== name);
  if (name === "timeline") pollHistory();
  if (name === "why") loadWhy();
  logEvent("panel_tab", { tab: name });
}

document.querySelectorAll(".sp-tabs button").forEach(b =>
  b.addEventListener("click", () => showTab(b.dataset.tab)));

on("panelBtn", "click", () => setPanel($("sidePanel").hidden));
on("panelClose", "click", () => setPanel(false));
on("panelScrim", "click", () => setPanel(false));
// The workload pill is a shortcut straight to the evidence behind it.
on("sbPill", "click", () => setPanel(true, "why"));

document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("sidePanel").hidden) setPanel(false);
});

/* ---------- home ----------
   Reachable from every view and every rung. */
on("homeBtn", "click", () => { setPanel(false); backToLibrary(); });

/* ---------- theme ---------- */
function applyTheme(mode) {
  root.dataset.theme = mode;
  const btn = $("themeBtn");
  if (btn) btn.textContent = mode === "dark" ? "\u2600" : "\u263D";
  try { localStorage.setItem("asd-theme", mode); } catch { /* private mode */ }
}
on("themeBtn", "click", () => {
  applyTheme(root.dataset.theme === "dark" ? "light" : "dark");
  logEvent("theme", { mode: root.dataset.theme });
});
(() => {
  let saved = null;
  try { saved = localStorage.getItem("asd-theme"); } catch { /* private mode */ }
  applyTheme(saved || (window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark" : "light"));
})();

/* ---------- mark as complete ---------- */
on("completeBtn", "click", async () => {
  if (!doc) return;
  const btn = $("completeBtn");
  const next = !doc.completed;
  btn.disabled = true;
  try {
    const r = await fetch(`/api/documents/${doc.id}/complete?done=${next}`, { method: "POST" });
    if (!r.ok) throw new Error();
    const d = await r.json();
    doc.completed = d.completed;
    paintComplete();
  } catch {
    btn.textContent = "Couldn't save \u2014 try again";
  } finally { btn.disabled = false; }
});

function paintComplete() {
  const btn = $("completeBtn");
  if (!btn || !doc) return;
  btn.textContent = doc.completed ? "\u2713 Completed" : "Mark as complete";
  btn.classList.toggle("done", !!doc.completed);
}


/* ---------- help: reopening the dock ----------
   Dismissing the rescue dock used to be a one-way door for the rest of the
   high-workload episode. Hiding something must never be the last time you can
   reach it. */
on("helpBtn", "click", () => {
  const forced = el.rescue.classList.contains("forced")
    && !el.rescue.classList.contains("dismissed");
  el.rescue.classList.remove("dismissed");
  el.rescue.classList.toggle("forced", !forced);
  if (!forced) logEvent("help_opened", { via: "topbar" });
});

/* ---------- break scheduling ----------
   Modes are exclusive by design. Running a fixed clock alongside the adaptive
   one would mean a break could no longer be attributed to either mechanism,
   which is the one thing the evaluation needs it to be. */
const BREAK_HINT = {
  adaptive: "Breaks are offered when sustained high workload is measured.",
  fixed: "Breaks every 25 minutes regardless of workload — the control condition.",
  off: "No breaks offered. Baseline condition.",
};
const breakBtns = document.querySelectorAll(".sp-body [data-break]");
breakBtns.forEach(btn => btn.addEventListener("click", async () => {
  breakBtns.forEach(b => b.classList.remove("on"));
  btn.classList.add("on");
  const hint = $("breakHint");
  if (hint) hint.textContent = BREAK_HINT[btn.dataset.break];
  await post("/api/break-mode", { mode: btn.dataset.break });
  poll();
}));

/* ---------- first-run guide ----------
   An interface that rearranges itself without warning is indistinguishable
   from one that is malfunctioning. Saying so first is cheaper than explaining
   afterwards. */
const TOUR = [
  { el: null, title: "This page reads you back",
    body: "It measures cognitive workload from EEG and changes itself in response. "
        + "Nothing here is decoration — every change is a reaction to a signal, and "
        + "you can always see why." },
  { el: ".ribbon", title: "The strip at the top is your EEG",
    body: "Four bars: delta, theta, alpha and beta power, updating every two seconds. "
        + "Teal at low workload, amber at high. It thickens for a moment whenever the "
        + "layout adapts, so you notice the change even while reading." },
  { el: "#sbPill", title: "Your current workload",
    body: "Four levels — low, elevated, high, overload. Each one removes a little more "
        + "from the screen. Click it to see the evidence behind the current reading." },
  { el: "#article", title: "The reading adapts around you",
    body: "At higher workload the side panels leave, the type grows, the prose is "
        + "rewritten in plainer language, and the key sentence is marked. Double-click "
        + "any sentence you're stuck on to have it explained." },
  { el: "#helpBtn", title: "Help is always one click away",
    body: "Under high workload a panel offers four ways into whatever you're reading: "
        + "a different explanation, an example, a diagram, and what it assumes. "
        + "Closed it by accident? This button brings it back." },
  { el: "#panelBtn", title: "Everything else lives here",
    body: "Controls to override the adaptation, a timeline of your session, and a full "
        + "record of why the interface changed. If a change ever surprises you, the "
        + "answer is in the Why tab." },
];

let tourIdx = 0;

function showTourStep(i) {
  tourIdx = Math.max(0, Math.min(i, TOUR.length - 1));
  const step = TOUR[tourIdx];
  $("tourStep").textContent = `${tourIdx + 1} of ${TOUR.length}`;
  $("tourTitle").textContent = step.title;
  $("tourBody").textContent = step.body;
  $("tourPrev").disabled = tourIdx === 0;
  $("tourNext").textContent = tourIdx === TOUR.length - 1 ? "Start reading" : "Next";

  document.querySelectorAll(".tour-target").forEach(n => n.classList.remove("tour-target"));
  const target = step.el && document.querySelector(step.el);
  const tour = $("tour");
  // offsetParent is null for any fixed-position element, which is most of the
  // chrome this tour points at. Measure the box instead.
  const r = target ? target.getBoundingClientRect() : null;
  if (r && r.width > 0 && r.height > 0) {
    target.classList.add("tour-target");
    // Place the card clear of the element it is pointing at.
    const below = r.bottom + 16;
    const fits = below + 210 < window.innerHeight;
    tour.style.top = (fits ? below : Math.max(16, r.top - 226)) + "px";
    tour.style.left = Math.min(Math.max(16, r.left - 40),
                               window.innerWidth - 400) + "px";
    tour.style.transform = "none";
  } else {
    tour.style.top = "50%";
    tour.style.left = "50%";
    tour.style.transform = "translate(-50%, -50%)";
  }
}

function setTour(open, replay) {
  $("tour").hidden = !open;
  $("tourScrim").hidden = !open;
  if (open) {
    showTourStep(0);
    logEvent("tour_started", { replay: !!replay });
  } else {
    document.querySelectorAll(".tour-target").forEach(n => n.classList.remove("tour-target"));
    try { localStorage.setItem("asd-tour-seen", "1"); } catch { /* private mode */ }
  }
}

on("tourNext", "click", () => {
  if (tourIdx === TOUR.length - 1) { setTour(false); logEvent("tour_completed"); }
  else showTourStep(tourIdx + 1);
});
on("tourPrev", "click", () => showTourStep(tourIdx - 1));
on("tourSkip", "click", () => { setTour(false); logEvent("tour_skipped", { at: tourIdx + 1 }); });
on("tourScrim", "click", () => { setTour(false); logEvent("tour_skipped", { at: tourIdx + 1 }); });
on("tourBtn", "click", () => { setPanel(false); setTour(true, true); });
window.addEventListener("resize", () => { if (!$("tour").hidden) showTourStep(tourIdx); });

/* ---------- layout + band ribbon ---------- */
const seen = {};
function norm(band, v) {
  const s = (seen[band] ??= { lo: v, hi: v });
  s.lo = Math.min(s.lo, v); s.hi = Math.max(s.hi, v);
  const span = s.hi - s.lo;
  return span < 1e-6 ? 0.5 : 0.08 + 0.88 * ((v - s.lo) / span);
}

function paint(s) {
  if (s.effective_state !== state) {
    state = s.effective_state;
    level = s.level;
    root.dataset.state = state;          // low | elevated | high | overload
    root.dataset.level = String(level);
    el.ribbon.classList.add("pulse");
    setTimeout(() => el.ribbon.classList.remove("pulse"), 2000);
    if (level === 0) el.rescue.classList.remove("dismissed");
    if (panelTabVisible("why")) loadWhy();  // keep the explanation live
  }
  level = s.level;

  render(s.tier);
  applyPaging(level >= 3);

  // The break dialog must state which mechanism fired it, or a fixed-schedule
  // break reads as a workload claim the system never made.
  const rt = $("recoveryTitle");
  if (rt) rt.textContent = s.recovery_reason === "fixed schedule"
    ? "Time for a scheduled break"
    : "You've been at high workload for a while";

  // Do-not-disturb is stated, not silent: an interface that quietly withholds
  // things the reader expects is indistinguishable from one that is broken.
  el.dnd.hidden = level < 1;
  if (level >= 1) {
    el.dndDetail.textContent = level >= 3
      ? "One concept at a time \u00b7 everything else paused"
      : level >= 2
        ? `Notifications, navigation and side panels paused \u00b7 ${s.n_withdrawn} elements withdrawn`
        : "Notifications and session statistics paused";
  }
  showRecovery(s.recovery_due && root.dataset.view === "reading");

  el.stateLabel.textContent = state;
  const pillState = $("pillState");
  if (pillState) pillState.textContent = state;
  el.timer.textContent = fmt(s.session_sec);

  const parts = [`p ${s.p_high.toFixed(2)} \u2192 smoothed ${s.p_smooth.toFixed(2)}`,
                 `epoch ${s.epoch_index}/${s.epoch_total}`];
  if (s.sustained_high_sec > 0) parts.push(`sustained ${Math.round(s.sustained_high_sec)}s`);
  if (s.n_adaptations) parts.push(`${s.n_adaptations} adaptations`);
  if (s.override !== "auto") parts.push("layout: manual");
  if (s.tier_override) parts.push("detail: manual");
  el.meta.textContent = parts.join(" · ");

  for (const [band, node] of Object.entries(el.bands)) {
    if (s.bands[band] != null) node.style.height = (norm(band, s.bands[band]) * 100).toFixed(1) + "%";
  }
  const pct = s.epoch_total ? (s.epoch_index % s.epoch_total) / s.epoch_total * 100 : 0;
  el.playhead.style.left = pct + "%";
}

async function poll() {
  try {
    const r = await fetch("/api/state");
    if (r.ok) paint(await r.json());
  } catch {
    el.meta.textContent = "backend unreachable — is uvicorn running on port 8000?";
  }
}

/* ---------- manual controls ----------
   Scoped to .sp-body: <html> itself carries data-tier, so an unscoped
   [data-tier] selector would bind the handler to the whole document and fire
   a tier override on every click anywhere on the page. The scope must track
   the markup — these buttons moved from the old bottom bar into the panel. */
const modeBtns = document.querySelectorAll(".sp-body [data-mode]");
const tierBtns = document.querySelectorAll(".sp-body [data-tier]");

modeBtns.forEach(btn => btn.addEventListener("click", async () => {
  modeBtns.forEach(b => b.classList.remove("on"));
  btn.classList.add("on");
  await post("/api/override", { mode: btn.dataset.mode });
  poll();
}));

tierBtns.forEach(btn => btn.addEventListener("click", async () => {
  tierBtns.forEach(b => b.classList.remove("on"));
  btn.classList.add("on");
  await post("/api/tier", { tier: btn.dataset.tier || null });
  poll();
}));

// Demo only: compresses the replay so the full escalation (adapt -> plain ->
// break prompt -> essential) fits inside a viva slot. Thresholds scale with
// it, so the sequence is unchanged.
const speedBtns = document.querySelectorAll(".sp-body [data-speed]");
speedBtns.forEach(btn => btn.addEventListener("click", async () => {
  speedBtns.forEach(b => b.classList.remove("on"));
  btn.classList.add("on");
  await post("/api/speed", { speed: Number(btn.dataset.speed) });
  poll();
}));

// Jump straight to where the recording switches to high workload. Far better
// than waiting or over-cranking the speed: the timing stays realistic, only
// the boring first half is skipped.
//
// Bound defensively. Every handler below is optional, so a markup file that
// is a version behind degrades to a missing button instead of throwing and
// taking the whole controller — polling, rendering, everything — down with it.
on("seekBtn", "click", async () => {
  await post("/api/seek", { to: "transition" });
  poll();
  pollHistory();
});

/* ---------- boot ---------- */
(async () => {
  await loadLibrary();
  await loadActive();
  // Land straight in the reader if a document is already active.
  root.dataset.view = doc ? "reading" : "library";
  let seen = null;
  try { seen = localStorage.getItem("asd-tour-seen"); } catch { /* private mode */ }
  if (!seen) setTour(true);

  poll();
  pollHistory();
  setInterval(poll, 1000);
  setInterval(pollHistory, 2000);
})();