# Adaptive Study Dashboard Using EEG Cognitive Workload

Raw STEW EEG → MNE band power → LOSO-validated SVM → FastAPI → a study
interface that simplifies itself when workload is high.

```
data/stew/*.txt ──► preprocess.py ──► features.py ──► train.py ──► models/*.joblib
                    filter, ICA,      Welch band       LOSO SVM         │
                    re-ref, epoch     power (59-d)     + baselines      ▼
                                                                   backend/app.py
                                                                        │  /api/state
                                                                        ▼
                                                                   frontend/ (2 states)
```

---

## 0. Open it in VS Code

```bash
cd eeg-study-dashboard
code .
```

When VS Code prompts about recommended extensions, accept — you need
**Python** and **Pylance**. Then create the environment:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Press `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one inside
`.venv`. `.vscode/settings.json` already points there, so this is usually
automatic.

**Always run from the project root**, and always as a module (`python -m
src.features`, not `python src/features.py`). The `src.` imports break
otherwise.

`.vscode/launch.json` has the four steps preconfigured — hit `F5` and pick
one from the dropdown instead of typing commands.

---

## 1. Get the STEW dataset

Download from IEEE DataPort: *STEW: Simultaneous Task EEG Workload Dataset*
(Lim, Sourina & Wang, 2018). Unzip so the `.txt` recordings sit directly in
`data/stew/`:

```
data/stew/
  sub01_lo.txt      # 2.5 min eyes-open rest      -> label 0
  sub01_hi.txt      # 2.5 min SIMKAP multitasking -> label 1
  sub02_lo.txt
  ...
  ratings.txt       # ignored by the loader
```

Each file is 19200 rows × 14 columns, space-separated, 128 Hz, in Emotiv raw
counts. Nothing else needs to be renamed.

**Don't have it yet?** Generate fake recordings with the same shape so you can
verify the plumbing today:

```bash
python -m src.make_synthetic --subjects 8
```

These will score near 100% because the signal difference is planted. Delete
them before you report a single number.

---

## 2. Build the feature matrix

```bash
python -m src.features            # add --ica for ICA artifact removal (~3x slower)
```

Per recording: band-pass 1–40 Hz → optional ICA (AF3/AF4 as pseudo-EOG for
blinks, `find_bads_muscle` for EMG) → average re-reference → 4 s epochs with
50% overlap → epochs whose peak-to-peak exceeds 3× the median are dropped →
Welch PSD → log₁₀ power in delta/theta/alpha/beta for all 14 channels.

That's 56 features, plus 3 scalar ratios (theta/alpha, theta/beta, and the
engagement index β/(α+θ)) that survive per-subject amplitude differences.
**59 features per 4-second epoch.**

Output: `data/processed/features.npz`. Expect roughly 148 epochs per subject,
so ~7000 rows for the full 48-subject dataset.

---

## 3. Train and validate

```bash
python -m src.train --fast --baselines     # minutes  — no hyperparameter search
python -m src.train --baselines            # ~30–60 min — full grid search
```

Leave-one-subject-out: every subject is held out once, and the inner
hyperparameter search is *also* grouped by subject, so epochs from one person
never appear on both sides of a split. This is the difference between an
honest cross-subject number and the inflated same-subject numbers the
literature review flags.

Written to `models/`:

| File | What it is |
| --- | --- |
| `loso_results.csv` | per-subject accuracy, precision, recall, F1, chosen C/γ |
| `confusion_matrix.png` | pooled across all folds — drop into the report |
| `summary.json` | headline means, including baselines |
| `svm_workload.joblib` | final model, trained on all subjects but one |
| `demo_stream.npz` | that held-out subject's epochs, for the live demo |

Compare against the two baselines the report promises: `DummyClassifier`
(majority class ≈ 0.50, since lo/hi are balanced) and Random Forest.

---

## 4. Run the dashboard

```bash
uvicorn backend.app:app --reload --port 8000
```

Open **http://127.0.0.1:8000**. The backend serves the frontend, so there's no
CORS setup and no second server.

What happens: a background loop steps through the held-out subject's epochs
one every 2 seconds, classifies each, takes a majority vote over the last 10
predictions (~20 s), and gates changes behind a 25-second dwell time so the
layout can't flicker. The demo subject's recording is low-workload for the
first half and high for the second, so the interface adapts once, roughly 5
epochs after the true switch — that lag is the smoothing, and it's the correct
trade-off to talk about in the viva.

**The two states**

| | Full (low workload) | Simplified (high workload) |
| --- | --- | --- |
| Module rail | visible | removed |
| Side panel (timer, links, notifications) | visible | removed |
| Body type | 17px / 1.68 | 21px / 1.82 |
| Column | 3-column grid | centred, 760px |
| Paragraphs | all equal | non-current dimmed to 34%, current one marked |
| Footer actions | visible | hidden |

**Endpoints**

| Route | Purpose |
| --- | --- |
| `GET /api/state` | current label, confidence, band powers, ground truth |
| `POST /api/override` | `{"mode": "auto" \| "normal" \| "simplified"}` |
| `GET /api/log` | override usage — the secondary evaluation signal |
| `POST /api/pause`, `POST /api/reset` | demo controls |
| `POST /api/predict` | stateless: 59 floats in, label out |

`/api/predict` is the seam. Swap the replay loop for a live Emotiv stream
(via `pylsl` or the Cortex API) and nothing downstream changes.

---

## 5. Usability study (methodology 6.2, step 8)

`GET /api/log` gives you override counts and timestamps per session. Pair
that with SUS after each session with 6–8 participants. The number worth
reporting is **overrides per session** — if people keep forcing the layout back
to Full, the adaptation is firing at the wrong moments regardless of what the
classifier accuracy says.

---
