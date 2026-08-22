"""Central configuration. Everything else imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "stew"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

for _p in (DATA_RAW, DATA_PROC, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

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
SMOOTH_WINDOW = 10             # majority vote over last N predictions (~20 s)
DWELL_SEC = 25.0               # min time in a state before it may change

# --- Adaptive features ---
HISTORY_MAX = 900              # ticks retained for the workload timeline (~30 min)
RECOVERY_AFTER_SEC = 90.0      # sustained high workload before a break is offered
RECOVERY_SNOOZE_SEC = 240.0    # don't re-offer for this long after dismissal
ESSENTIAL_AFTER_SEC = 150.0    # sustained high before content drops to essential tier

LABELS = {0: "low", 1: "high"}