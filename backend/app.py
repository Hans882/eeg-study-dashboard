"""FastAPI service (methodology 6.2).

Serves the trained classifier over a replayed EEG stream, and exposes the
derived state the adaptive interface needs:

  * smoothed workload label with a dwell gate      -> adaptive layout
  * sustained-high duration                        -> content complexity tier
  * per-tick probability history                   -> workload timeline
  * recovery prompt after prolonged high workload  -> break recommendation
  * an event log of every adaptation and override  -> usability evaluation

Run from the project root:
    uvicorn backend.app:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config as C
from src import ingest

FRONTEND = C.ROOT / "frontend"


class Engine:
    """Replays pre-recorded epochs through the classifier with smoothing."""

    def __init__(self):
        bundle = joblib.load(C.MODELS / "svm_workload.joblib")
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.demo_subject = bundle["demo_subject"]

        demo = np.load(C.MODELS / "demo_stream.npz")
        self.X, self.y_true = demo["X"], demo["y"]

        self.i = 0
        self.window: deque[float] = deque(maxlen=C.SMOOTH_WINDOW)
        self.history: deque[dict] = deque(maxlen=C.HISTORY_MAX)

        self.state = 0                     # rung on the intervention ladder
        self.raw_label = 0
        self.p_high = 0.0
        self.p_smooth = 0.0
        self.confidence = 0.5
        self.adaptations: list[dict] = []
        self.bands: dict[str, float] = {b: 0.0 for b in C.BANDS}

        self.t0 = time.monotonic()
        self.last_change = self.t0
        self.high_since: float | None = None

        self.override = "auto"                  # auto | normal | simplified
        self.tier_override: str | None = None   # full | plain | essential
        self.recovery_snoozed_until = 0.0
        self.recovery_taken = 0
        self.break_mode = C.DEFAULT_BREAK_MODE
        self.pomodoro_start = time.monotonic()
        self.events: list[dict] = []
        self.running = True
        self.speed = 1.0        # replay multiplier; >1 compresses a demo

        # Active reading. Restored on boot so a restart doesn't lose the session.
        self.document: dict | None = None
        cat = ingest.catalogue(C.DOCS)
        if cat:
            try:
                self.document = ingest.load(cat[0]["id"], C.DOCS, C.UPLOADS)
            except Exception:
                self.document = None

    def load_document(self, doc: dict | None):
        """Switch the active reading. Per-document counters reset; the
        workload trace does not, since the EEG session is continuous."""
        self.document = doc
        self.tier_override = None
        self.recovery_snoozed_until = 0.0
        self._log("document", doc_id=(doc or {}).get("id"),
                  title=(doc or {}).get("title"))

    # ------------------------------------------------------------------ utils
    def _log(self, _kind: str, **detail):
        # The parameter is underscore-prefixed so that a caller passing
        # detail={"kind": ...} through **detail cannot collide with it.
        entry = {
            "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_sec": round(time.monotonic() - self.t0, 1),
            "kind": _kind,
            "epoch_index": self.i,
        }
        # Frontend-supplied detail must never clobber the log's own fields,
        # otherwise /api/event 500s on a payload that happens to carry "kind".
        for k, v in detail.items():
            entry[f"detail_{k}" if k in entry else k] = v
        self.events.append(entry)

    def _band_means(self, row: np.ndarray) -> dict[str, float]:
        n = len(C.CHANNELS)
        return {b: float(row[k * n:(k + 1) * n].mean())
                for k, b in enumerate(C.BANDS)}

    # -- timing thresholds, scaled so a sped-up replay stays proportional ----
    # Scaling is floored. Below roughly three seconds the layout flips faster
    # than a reader can re-orient, which is worse than useless as a demo and
    # inflates the adaptation count in the session stats.
    @property
    def dwell_sec(self) -> float:
        return max(3.0, C.DWELL_SEC / self.speed)

    @property
    def recovery_after_sec(self) -> float:
        return max(10.0, C.RECOVERY_AFTER_SEC / self.speed)

    @property
    def essential_after_sec(self) -> float:
        return max(18.0, C.ESSENTIAL_AFTER_SEC / self.speed)

    @property
    def snooze_sec(self) -> float:
        return max(25.0, C.RECOVERY_SNOOZE_SEC / self.speed)

    @property
    def overload_after_sec(self) -> float:
        return max(8.0, C.OVERLOAD_AFTER_SEC / self.speed)

    @property
    def transition_index(self) -> int:
        """First epoch where the ground truth flips low -> high.

        The demo subject's recording is low for the first half and high for the
        second, so this is where anything interesting starts happening.
        """
        if len(self.y_true) < 2:
            return 0
        changes = np.flatnonzero(self.y_true != self.y_true[0])
        return int(changes[0]) if changes.size else 0

    def seek(self, index: int):
        """Jump the replay to an epoch.

        The smoothing window is refilled with the label at the destination
        rather than merely cleared: an empty window would take SMOOTH_WINDOW
        further epochs to reach a majority, so a 'skip to the interesting
        part' button would still leave the demo waiting.
        """
        self.i = max(0, min(int(index), len(self.X) - 1))
        target = int(self.y_true[self.i])
        # Prime with probabilities matching the destination so the ladder is
        # already where the recording is, not ten epochs behind it.
        self.window.clear()
        self.window.extend([0.92 if target else 0.08] * C.SMOOTH_WINDOW)
        self.state = 2 if target else 0
        self.high_since = None
        self.last_change = time.monotonic() - self.dwell_sec  # allow an immediate adapt
        self.recovery_snoozed_until = 0.0
        self.p_smooth = sum(self.window) / len(self.window)
        self._log("seek", to_epoch=self.i, label=C.LABELS[target])

    # ------------------------------------------------------------------- tick
    def _ladder_target(self, p_smooth: float) -> int:
        """Which rung the smoothed estimate calls for, with hysteresis.

        Climbing needs a decisively high estimate; falling needs a decisively
        low one. Between EXIT_THRESHOLD and ENTER_ELEVATED the current rung is
        held, which is what stops an estimate hovering near a boundary from
        oscillating the interface.
        """
        if p_smooth >= C.ENTER_HIGH:
            return max(self.state, 2)
        if p_smooth >= C.ENTER_ELEVATED:
            return max(self.state, 1) if self.state <= 1 else self.state - 1
        if p_smooth <= C.EXIT_THRESHOLD:
            return max(0, self.state - 1)
        return self.state                      # hysteresis band: hold

    def _record_adaptation(self, prev: int, new: int, p_smooth: float, now: float):
        """Everything needed to answer 'why did the interface just change?'"""
        highs = sum(1 for v in self.window if v >= C.ENTER_HIGH)
        hidden = [b["element"] for b in C.DISTRACTION_BUDGET
                  if b["hidden_from"] is not None and new >= b["hidden_from"]]
        direction = "escalation" if new > prev else "de-escalation"
        self.adaptations.append({
            "n": len(self.adaptations) + 1,
            "session_sec": round(now - self.t0, 1),
            "timestamp": self._clock(now - self.t0),
            "direction": direction,
            "from_state": C.STATES[prev],
            "to_state": C.STATES[new],
            "p_high": round(p_smooth, 3),
            "confidence": round(self.confidence, 3),
            "sustained_high_sec": round(self.sustained_high, 1),
            "trigger": f"{highs}/{len(self.window)} predictions above "
                       f"{C.ENTER_HIGH:.2f}",
            "held_previous_for_sec": round(now - self.last_change, 1),
            "from_layout": self._layout_name(prev),
            "to_layout": self._layout_name(new),
            "content_tier": C.TIER_BY_STATE[new],
            "elements_withdrawn": hidden,
            "n_elements_withdrawn": len(hidden),
            "epoch_index": self.i,
        })

    @staticmethod
    def _clock(seconds: float) -> str:
        s = int(seconds)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    @staticmethod
    def _layout_name(state: int) -> str:
        return {0: "Full", 1: "Reduced", 2: "Focus", 3: "Single concept"}[state]

    def tick(self):
        if not self.running or len(self.X) == 0:
            return
        now = time.monotonic()
        row = self.X[self.i % len(self.X)]

        proba = self.model.predict_proba(row.reshape(1, -1))[0]
        self.p_high = float(proba[1])
        self.raw_label = int(np.argmax(proba))
        self.confidence = float(max(self.p_high, 1 - self.p_high))
        # The window holds probabilities, not hard labels: averaging them keeps
        # the information a majority vote throws away, which is what the
        # hysteresis thresholds operate on.
        self.window.append(self.p_high)
        self.p_smooth = sum(self.window) / len(self.window)

        target = self._ladder_target(self.p_smooth)

        # Escalating on a coin-flip estimate is worse than not escalating:
        # a confident 0.91 earns a rung, an uncertain 0.53 does not.
        if target > self.state and self.confidence < C.MIN_CONFIDENCE:
            target = self.state

        if C.STEP_ONE_RUNG and target != self.state:
            target = self.state + (1 if target > self.state else -1)

        # Climbing may skip the dwell wait; falling may not. Stripping the
        # interface back is the response to a reader in difficulty and should
        # not be held behind a timer, whereas restoring density too eagerly is
        # what makes an adaptive interface flicker. The gate is asymmetric on
        # purpose: escalate on half the dwell, de-escalate on the full one.
        gate = self.dwell_sec * (0.5 if target > self.state else 1.0)
        if target != self.state and (now - self.last_change) >= gate:
            self._record_adaptation(self.state, target, self.p_smooth, now)
            self._log("adapt", frm=C.STATES[self.state], to=C.STATES[target],
                      held_sec=round(now - self.last_change, 1))
            self.state = target
            self.last_change = now

        # Sustained-high clock: runs while the rung is 'high' or above.
        if self.state >= 2:
            if self.high_since is None:
                self.high_since = now
        else:
            self.high_since = None

        # The top rung is reached by persistence, not by probability alone.
        if (self.state == 2 and self.sustained_high >= self.overload_after_sec
                and (now - self.last_change) >= self.dwell_sec):
            self._record_adaptation(2, 3, self.p_smooth, now)
            self._log("adapt", frm="high", to="overload",
                      held_sec=round(now - self.last_change, 1))
            self.state = 3
            self.last_change = now

        self.bands = self._band_means(row)
        self.history.append({
            "t": round(now - self.t0, 1),
            "p": round(self.p_high, 3),
            "smooth": round(self.p_smooth, 3),
            "state": self.state,
            "truth": int(self.y_true[self.i % len(self.y_true)]),
        })
        self.i += 1

    # ---------------------------------------------------------------- derived
    @property
    def sustained_high(self) -> float:
        return 0.0 if self.high_since is None else time.monotonic() - self.high_since

    def effective_state(self) -> str:
        """The rung actually in force, after any manual override."""
        if self.override == "auto":
            return C.STATES[self.state]
        return {"normal": "low", "simplified": "high",
                "focus": "overload"}.get(self.override, "low")

    def effective_level(self) -> int:
        name = self.effective_state()
        return next(k for k, v in C.STATES.items() if v == name)

    def budget(self) -> list[dict]:
        """The distraction budget annotated with what is withdrawn right now."""
        lvl = self.effective_level()
        return [{**b,
                 "withdrawn": b["hidden_from"] is not None and lvl >= b["hidden_from"]}
                for b in C.DISTRACTION_BUDGET]

    def tier(self) -> str:
        """Content complexity: full -> plain -> essential."""
        if self.tier_override:
            return self.tier_override
        return C.TIER_BY_STATE[self.effective_level()]

    @property
    def pomodoro_left(self) -> float:
        """Seconds until the fixed-schedule break, ignoring workload entirely."""
        elapsed = (time.monotonic() - self.pomodoro_start) * self.speed
        return max(0.0, C.POMODORO_WORK_SEC - elapsed)

    def recovery_due(self) -> bool:
        """Whether a break is being offered, and on what grounds.

        The grounds differ by mode, and only one mode is ever active — a break
        that could have come from either mechanism is a break the evaluation
        cannot attribute.
        """
        if time.monotonic() < self.recovery_snoozed_until:
            return False
        if self.break_mode == "off":
            return False
        if self.break_mode == "fixed":
            return self.pomodoro_left <= 0
        return (self.effective_level() >= 2
                and self.sustained_high >= self.recovery_after_sec)

    def break_status(self) -> dict:
        """When the next break can happen, and what is currently blocking it.

        Every input to that decision is already tracked; leaving it implicit
        makes a working system look broken whenever it is merely waiting.
        """
        now = time.monotonic()
        snoozed = max(0.0, self.recovery_snoozed_until - now)
        if self.break_mode == "off":
            return {"state": "disabled", "detail": "Breaks are turned off.",
                    "seconds": None}
        if snoozed > 0:
            return {"state": "snoozed", "seconds": round(snoozed, 1),
                    "detail": f"Snoozed for another {int(snoozed)}s after "
                              "'Keep reading'."}
        if self.break_mode == "fixed":
            return {"state": "waiting", "seconds": round(self.pomodoro_left, 1),
                    "detail": f"Next scheduled break in {int(self.pomodoro_left)}s."}
        if self.effective_level() < 2:
            return {"state": "waiting", "seconds": None,
                    "detail": "Waiting for high workload. The clock starts when "
                              "the interface reaches the high rung."}
        remaining = max(0.0, self.recovery_after_sec - self.sustained_high)
        return {"state": "counting", "seconds": round(remaining, 1),
                "detail": f"{int(remaining)}s more of sustained high workload "
                          "before a break is offered."}

    def recovery_reason(self) -> str:
        if self.break_mode == "fixed":
            return "fixed schedule"
        return "sustained high workload"

    def snapshot(self) -> dict:
        return {
            "auto_state": C.STATES[self.state],
            "effective_state": self.effective_state(),
            "override": self.override,
            "tier": self.tier(),
            "tier_override": self.tier_override,
            "raw_prediction": C.LABELS[self.raw_label],
            "level": self.effective_level(),
            "p_smooth": round(self.p_smooth, 3),
            "n_adaptations": len(self.adaptations),
            "last_adaptation": self.adaptations[-1] if self.adaptations else None,
            "n_withdrawn": sum(1 for b in self.budget() if b["withdrawn"]),
            "overload_after_sec": round(self.overload_after_sec, 1),
            "p_high": round(self.p_high, 3),
            "confidence": round(max(self.p_high, 1 - self.p_high), 3),
            "seconds_in_state": round(time.monotonic() - self.last_change, 1),
            "sustained_high_sec": round(self.sustained_high, 1),
            "recovery_due": self.recovery_due(),
            "recovery_reason": self.recovery_reason(),
            "break_status": self.break_status(),
            "break_mode": self.break_mode,
            "pomodoro_left_sec": round(self.pomodoro_left, 1),
            "pomodoro_work_sec": C.POMODORO_WORK_SEC,
            "recovery_after_sec": round(self.recovery_after_sec, 1),
            "speed": self.speed,
            "session_sec": round(time.monotonic() - self.t0, 1),
            "epoch_index": self.i,
            "epoch_total": len(self.X),
            "ground_truth": C.LABELS[int(self.y_true[(self.i - 1) % len(self.y_true)])],
            "thresholds": {"enter_elevated": C.ENTER_ELEVATED,
                           "enter_high": C.ENTER_HIGH,
                           "exit": C.EXIT_THRESHOLD,
                           "min_confidence": C.MIN_CONFIDENCE},
            "demo_subject": self.demo_subject,
            "bands": self.bands,
            "running": self.running,
            "document_id": (self.document or {}).get("id"),
            "document_title": (self.document or {}).get("title"),
        }

    def stats(self) -> dict:
        """Session summary for the timeline panel."""
        h = list(self.history)
        if not h:
            return {"high_ratio": 0.0, "adaptations": 0, "overrides": 0,
                    "rescues": 0, "breaks_taken": 0, "agreement": None}
        kinds = [e["kind"] for e in self.events]
        return {
            "high_ratio": round(sum(1 for r in h if r["state"] >= 2) / len(h), 3),
            "agreement": round(sum(1 for r in h if (r["state"] >= 2) == bool(r["truth"]))
                              / len(h), 3),
            "adaptations": len(self.adaptations),
            "overrides": kinds.count("override") + kinds.count("tier_override"),
            "rescues": kinds.count("rescue"),
            "breaks_taken": self.recovery_taken,
            "break_mode": self.break_mode,
        }


engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = Engine()

    async def loop():
        while True:
            engine.tick()
            await asyncio.sleep(C.HOP_SEC / engine.speed)

    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(title="Adaptive Study Dashboard", lifespan=lifespan)


# ------------------------------------------------------------------ schemas
class OverrideIn(BaseModel):
    mode: str


class TierIn(BaseModel):
    tier: str | None = None


class EventIn(BaseModel):
    kind: str
    detail: dict = {}


class RecoveryIn(BaseModel):
    action: str  # taken | dismissed


class FeaturesIn(BaseModel):
    features: list[float]


# ------------------------------------------------------------------- routes
@app.get("/api/state")
def get_state():
    return engine.snapshot()


@app.get("/api/history")
def get_history():
    """Per-tick probability trace for the workload timeline."""
    return {"points": list(engine.history), "stats": engine.stats()}


@app.post("/api/override")
def set_override(body: OverrideIn):
    if body.mode not in {"auto", "normal", "simplified", "focus"}:
        raise HTTPException(400, "mode must be auto, normal, simplified or focus")

    before = engine.effective_level()
    engine.override = body.mode
    after = engine.effective_level()

    # A manual override is an adaptation too. Leaving it out of the record
    # makes the inspection panel look broken to anyone exploring with the
    # buttons, and hides the most interesting events in a usability study —
    # the moments the reader overruled the system.
    if after != before:
        engine._record_adaptation(before, after, engine.p_smooth, time.monotonic())
        engine.adaptations[-1]["direction"] = "manual override"
        engine.adaptations[-1]["trigger"] = (
            f"reader selected '{body.mode}' — classifier was reading "
            f"{C.STATES[engine.state]}")

    engine._log("override", mode=body.mode, auto_state=C.STATES[engine.state])
    return engine.snapshot()


@app.post("/api/tier")
def set_tier(body: TierIn):
    if body.tier not in {None, "full", "plain", "essential"}:
        raise HTTPException(400, "tier must be full, plain, essential or null")
    engine.tier_override = body.tier
    engine._log("tier_override", tier=body.tier or "auto")
    return engine.snapshot()


@app.post("/api/recovery")
def recovery(body: RecoveryIn):
    if body.action not in {"taken", "dismissed"}:
        raise HTTPException(400, "action must be taken or dismissed")
    engine.recovery_snoozed_until = time.monotonic() + engine.snooze_sec
    engine.pomodoro_start = time.monotonic()      # restart the fixed clock too
    if body.action == "taken":
        engine.recovery_taken += 1
    engine._log("recovery", action=body.action, mode=engine.break_mode,
                reason=engine.recovery_reason(),
                sustained_sec=round(engine.sustained_high, 1))
    return engine.snapshot()


@app.post("/api/event")
def log_event(body: EventIn):
    """Frontend-side interactions worth measuring, e.g. rescue-panel usage."""
    engine._log(body.kind, **body.detail)
    return {"logged": True, "count": len(engine.events)}


@app.get("/api/log")
def get_log():
    """Full event log — the behavioural half of the usability evaluation."""
    return {"events": engine.events, "stats": engine.stats(),
            "session_sec": round(time.monotonic() - engine.t0, 1)}


@app.post("/api/pause")
def toggle_pause():
    engine.running = not engine.running
    return {"running": engine.running}


class BreakModeIn(BaseModel):
    mode: str


@app.post("/api/break-mode")
def set_break_mode(body: BreakModeIn):
    """Choose how breaks are scheduled: adaptive, fixed, or off.

    This is the experimental condition. Recording it per session is what makes
    'do physiologically-timed breaks beat clock-timed ones?' a question the
    logs can answer.
    """
    if body.mode not in C.BREAK_MODES:
        raise HTTPException(400, f"mode must be one of {C.BREAK_MODES}")
    engine.break_mode = body.mode
    engine.pomodoro_start = time.monotonic()
    engine.recovery_snoozed_until = 0.0
    engine._log("break_mode", mode=body.mode)
    return engine.snapshot()


class SpeedIn(BaseModel):
    speed: float


@app.post("/api/speed")
def set_speed(body: SpeedIn):
    """Compress the replay for a live demo.

    Speed 1 is real time: the low->high switch lands around 2.5 minutes in and
    the break prompt 90 s after that. Speed 6 puts the whole escalation inside
    a minute. Thresholds scale with it, so the sequence of adaptations is
    identical — only the wall clock changes. Not for anything you report.
    """
    if not 0.25 <= body.speed <= 20:
        raise HTTPException(400, "speed must be between 0.25 and 20")
    engine.speed = float(body.speed)
    engine._log("speed", speed=engine.speed)
    return engine.snapshot()


class SeekIn(BaseModel):
    to: str | int = "transition"


@app.post("/api/seek")
def seek(body: SeekIn):
    """Jump the replay. 'transition' lands a few epochs before the recording
    switches to high workload, which is where a demo actually wants to start.
    """
    if body.to == "transition":
        target = engine.transition_index
    elif body.to == "start":
        target = 0
    else:
        try:
            target = int(body.to)
        except (TypeError, ValueError):
            raise HTTPException(400, "to must be 'transition', 'start' or an epoch index")
    engine.seek(target)
    return engine.snapshot()


@app.post("/api/reset")
def reset():
    engine.i = 0
    engine.window.clear()
    engine.history.clear()
    engine.state = 0
    engine.adaptations.clear()
    engine.high_since = None
    engine.last_change = time.monotonic()
    engine.recovery_snoozed_until = 0.0
    return engine.snapshot()


@app.post("/api/predict")
def predict(body: FeaturesIn):
    """Stateless endpoint: 59-dim feature vector in, workload label out.

    This is the seam where a live Emotiv stream would plug in instead of
    the dataset replay.
    """
    x = np.asarray(body.features, dtype=float)
    if x.shape != (len(engine.feature_names),):
        raise HTTPException(400, f"expected {len(engine.feature_names)} features")
    proba = engine.model.predict_proba(x.reshape(1, -1))[0]
    return {"label": C.LABELS[int(np.argmax(proba))], "p_high": float(proba[1])}


# --------------------------------------------------------- document library
class ExplainIn(BaseModel):
    text: str
    concept_id: str | None = None


@app.post("/api/explain")
def explain_sentence(body: ExplainIn):
    """Explain one sentence the reader has picked out.

    Section-level help assumes the reader knows which section is the problem.
    Often they know exactly which sentence it is, and nothing smaller than a
    section was addressable — so this takes the sentence itself and returns a
    plainer rewrite, the terms it leans on with definitions drawn from
    elsewhere in the same document, and a relationship chain if the sentence
    carries one.
    """
    text = (body.text or "").strip()
    if not 10 <= len(text) <= 1200:
        raise HTTPException(400, "select a sentence to explain")

    simple = " ".join(ingest.simplify(text))
    terms = ingest.key_terms([text], k=3)

    # Definitions are looked up in the reader's own document rather than
    # invented, so an explanation never introduces a claim the source didn't make.
    definitions = []
    doc = engine.document or {}
    for term in terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for c in doc.get("concepts", []):
            if body.concept_id and c["id"] == body.concept_id:
                continue                       # a definition from elsewhere is the point
            for sent in ingest.split_sentences(c["tiers"]["full"]):
                if pattern.search(sent) and ingest.DEF_CUES.search(sent):
                    definitions.append({"term": term, "section": c["title"],
                                        "text": " ".join(ingest.simplify(sent))})
                    break
            if definitions and definitions[-1]["term"] == term:
                break

    chain = ingest.extract_causal([text]) or []
    relationship = ingest.build_causal(chain) if len(chain) >= 2 else ""

    return {"original": text, "simple": simple, "terms": terms,
            "definitions": definitions[:3], "relationship": relationship,
            "changed": simple.strip() != text.strip()}


@app.get("/api/adaptations")
def adaptations(limit: int = 20):
    """Why the interface changed — one record per adaptation.

    This is the explainability surface. Every layout change is accompanied by
    the evidence that produced it: the smoothed estimate, how confident it
    was, how many predictions in the window crossed the threshold, how long
    the previous rung was held, and exactly which elements were withdrawn.
    An adaptive interface the user cannot interrogate is indistinguishable
    from one that is malfunctioning.
    """
    items = engine.adaptations[-limit:][::-1]
    return {"adaptations": items, "total": len(engine.adaptations),
            "thresholds": {"enter_elevated": C.ENTER_ELEVATED,
                           "enter_high": C.ENTER_HIGH,
                           "exit": C.EXIT_THRESHOLD,
                           "min_confidence": C.MIN_CONFIDENCE,
                           "dwell_sec": round(engine.dwell_sec, 1),
                           "overload_after_sec": round(engine.overload_after_sec, 1)}}


@app.get("/api/budget")
def budget():
    """The distraction budget, annotated with what is currently withdrawn."""
    return {"level": engine.effective_level(),
            "state": engine.effective_state(),
            "budget": engine.budget()}


@app.get("/api/documents")
def list_documents():
    return {"documents": ingest.catalogue(C.DOCS)}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    if not re.fullmatch(r"[a-z0-9]{1,32}", doc_id):
        raise HTTPException(400, "bad document id")
    try:
        return ingest.load(doc_id, C.DOCS, C.UPLOADS)
    except FileNotFoundError:
        raise HTTPException(404, "no such document")


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...),
                          title: str | None = Form(None)):
    """Accept a reading, ingest it, and return the parsed document."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in C.ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext or '?'}'. "
                                 "Upload a PDF, DOCX, Markdown or text file.")

    raw = await file.read()
    if len(raw) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is over {C.MAX_UPLOAD_MB} MB.")
    if not raw:
        raise HTTPException(400, "That file is empty.")

    # Keep the original alongside the parse so a re-ingest never needs re-upload.
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename).name)[:80]
    tmp = C.UPLOADS / f"{uuid.uuid4().hex[:8]}_{safe}"
    tmp.write_bytes(raw)

    try:
        doc = ingest.ingest(tmp, title=title or None,
                            source_name=Path(file.filename).name,
                            stored=tmp.name)
    except (ValueError, RuntimeError) as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(422, str(e))

    ingest.save(doc, C.DOCS)
    engine.load_document(doc.to_dict())
    return doc.to_dict()


@app.post("/api/documents/{doc_id}/select")
def select_document(doc_id: str):
    """Make a document the active reading for this session."""
    if not re.fullmatch(r"[a-z0-9]{1,32}", doc_id):
        raise HTTPException(400, "bad document id")
    try:
        doc = ingest.load(doc_id, C.DOCS, C.UPLOADS)
    except FileNotFoundError:
        raise HTTPException(404, "no such document")
    engine.load_document(doc)
    return {"active": doc["id"], "title": doc["title"]}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if not re.fullmatch(r"[a-z0-9]{1,32}", doc_id):
        raise HTTPException(400, "bad document id")
    path = C.DOCS / f"{doc_id}.json"
    if not path.exists():
        raise HTTPException(404, "no such document")
    path.unlink()
    shutil.rmtree(C.ASSETS / doc_id, ignore_errors=True)
    if engine.document and engine.document["id"] == doc_id:
        engine.load_document(None)
    return {"deleted": doc_id}


@app.post("/api/documents/{doc_id}/complete")
def mark_complete(doc_id: str, done: bool = True):
    """Mark a reading finished. Persisted with the document, so it survives a
    restart and a re-parse."""
    if not re.fullmatch(r"[a-z0-9]{1,32}", doc_id):
        raise HTTPException(400, "bad document id")
    path = C.DOCS / f"{doc_id}.json"
    if not path.exists():
        raise HTTPException(404, "no such document")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["completed"] = bool(done)
    doc["completed_at"] = (datetime.now(timezone.utc).isoformat(timespec="seconds")
                           if done else None)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    if engine.document and engine.document.get("id") == doc_id:
        engine.document = doc
    engine._log("complete", doc_id=doc_id, completed=bool(done))
    return {"id": doc_id, "completed": doc["completed"],
            "completed_at": doc["completed_at"]}


@app.get("/api/active")
def active_document():
    """The reading the session is currently on, concepts included."""
    if engine.document is None:
        raise HTTPException(404, "no active document")
    return engine.document


# Extracted figures. Mounted before the catch-all static mount so the
# document's own images resolve, and read-only by construction.
app.mount("/assets", StaticFiles(directory=C.ASSETS), name="assets")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="static")