"""Central configuration. Everything else imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "stew"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DOCS = ROOT / "data" / "documents"      # ingested readings (JSON)
UPLOADS = ROOT / "data" / "uploads"     # original files as uploaded
ASSETS  = ROOT / "data" / "assets"      # figures extracted from documents

for _p in (DATA_RAW, DATA_PROC, MODELS, DOCS, UPLOADS, ASSETS):
    _p.mkdir(parents=True, exist_ok=True)

# --- Uploads ---
MAX_UPLOAD_MB = 25
ALLOWED_EXT = {".pdf", ".docx", ".md", ".markdown", ".txt", ".text"}

# --- EEG acquisition (fixed by the STEW dataset) ---
# Emotiv EPOC, 14 channels, 128 Hz, 2.5 min per condition per subject.
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
            "P8", "T8", "FC6", "F4", "F8", "AF4"]
SFREQ = 128.0

# Emotiv EPOC least-significant-bit ~0.51 uV/count. STEW ships raw counts.
# Converting to volts keeps MNE's plotting/units honest. Band power is
# log-transformed downstream, so a constant scale error would only shift the
# features by a constant -- but get it right anyway.
COUNTS_TO_VOLTS = 0.51e-6

# --- Preprocessing ---
L_FREQ, H_FREQ = 1.0, 40.0     # band-pass
EPOCH_SEC = 4.0
OVERLAP_SEC = 2.0              # 50% overlap -> hop of 2 s
REJECT_FACTOR = 3.0            # drop epoch if peak-to-peak > 3x median p2p

# --- Features ---
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}
PSD_NFFT = 256                 # 2 s window at 128 Hz -> 0.5 Hz resolution
PSD_NOVERLAP = 128

# --- Real-time behaviour ---
HOP_SEC = 2.0                  # replay advances one epoch hop per tick
SMOOTH_WINDOW = 10             # rolling window of predictions (~20 s)
DWELL_SEC = 25.0               # min time in a state before it may change

# --- The intervention ladder ---
# Four rungs rather than a binary flip. Each rung is a proportionate response
# to how severe and how persistent the detected load is, so the interface
# never rebuilds itself wholesale at a single threshold.
STATES = {0: "low", 1: "elevated", 2: "high", 3: "overload"}

# Hysteresis: the level needed to climb is well above the level needed to fall,
# so an estimate hovering near a boundary cannot oscillate the interface.
ENTER_ELEVATED = 0.55
ENTER_HIGH = 0.70
EXIT_THRESHOLD = 0.40          # below this, step down a rung
MIN_CONFIDENCE = 0.60          # escalation requires a confident estimate;
                               # de-escalation never does
OVERLOAD_AFTER_SEC = 60.0      # unbroken time at 'high' before the top rung
STEP_ONE_RUNG = True           # climb and descend one rung at a time

# Content complexity follows the rung.
TIER_BY_STATE = {0: "full", 1: "full", 2: "plain", 3: "essential"}

# --- Distraction budget ---
# Every element carries an estimated extraneous cost and the rung at which it
# is withdrawn. Adaptation is then "progressively reduce extraneous cost",
# not "hide the sidebar" — and the ordering is inspectable rather than
# buried in a stylesheet.
DISTRACTION_BUDGET = [
    {"element": "Main text",             "cost": "essential", "hidden_from": None},
    {"element": "Section heading",       "cost": "essential", "hidden_from": None},
    {"element": "Workload indicator",    "cost": "essential", "hidden_from": None},
    {"element": "Diagram",               "cost": "helpful",   "hidden_from": None},
    {"element": "Diagram detail",        "cost": "helpful",   "hidden_from": 3},
    {"element": "Session statistics",    "cost": "high",      "hidden_from": 1},
    {"element": "Key terms panel",       "cost": "moderate",  "hidden_from": 1},
    {"element": "Table of contents",     "cost": "moderate",  "hidden_from": 2},
    {"element": "Document navigation",   "cost": "moderate",  "hidden_from": 2},
    {"element": "Footer actions",        "cost": "moderate",  "hidden_from": 2},
    {"element": "Neighbouring sections", "cost": "moderate",  "hidden_from": 3},
    {"element": "Free scrolling",        "cost": "moderate",  "hidden_from": 3},
]

# --- Break scheduling ---
# Three modes, so the project's central claim can actually be tested rather
# than assumed. 'adaptive' is the system under study: breaks are offered when
# sustained high workload is measured. 'fixed' is the control condition — a
# conventional Pomodoro clock that knows nothing about the reader's state.
# 'off' is the no-intervention baseline. Running a fixed timer *alongside*
# the adaptive one would make the evaluation unreadable: a break taken could
# no longer be attributed to either mechanism.
BREAK_MODES = ("adaptive", "fixed", "off")
DEFAULT_BREAK_MODE = "adaptive"
POMODORO_WORK_SEC = 1500.0     # 25 minutes, the conventional interval
POMODORO_BREAK_SEC = 300.0     # 5 minutes

HISTORY_MAX = 900              # ticks retained for the workload timeline (~30 min)
RECOVERY_AFTER_SEC = 90.0      # sustained high-or-above before a break is offered
RECOVERY_SNOOZE_SEC = 240.0    # don't re-offer for this long after dismissal
ESSENTIAL_AFTER_SEC = 150.0    # retained for reference; the ladder now governs tier

LABELS = {0: "low", 1: "high"}