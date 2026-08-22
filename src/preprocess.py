"""Raw STEW .txt -> cleaned MNE Epochs.

Implements steps 1-5 of methodology 6.1:
acquisition, filtering, artifact handling, re-referencing, epoching.
"""
from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np

from src import config as C

mne.set_log_level("ERROR")

_FNAME = re.compile(r"sub(\d+)_(lo|hi)\.txt$", re.IGNORECASE)


def list_recordings(raw_dir: Path = C.DATA_RAW) -> list[tuple[int, int, Path]]:
    """Return [(subject_id, label, path)] for every STEW recording found.

    label: 0 = low workload (rest), 1 = high workload (SIMKAP multitasking).
    """
    found = []
    for path in sorted(Path(raw_dir).glob("*.txt")):
        m = _FNAME.search(path.name)
        if not m:
            continue  # skips ratings.txt and any readme
        found.append((int(m.group(1)), 0 if m.group(2).lower() == "lo" else 1, path))
    # (subject, label) order puts low before high, so the demo replay starts
    # calm and ramps into the high-workload block.
    found.sort(key=lambda t: (t[0], t[1]))
    if not found:
        raise FileNotFoundError(
            f"No STEW recordings in {raw_dir}. Expected files named "
            "sub01_lo.txt, sub01_hi.txt, ... See README section 1."
        )
    return found


def load_raw(path: Path) -> mne.io.RawArray:
    """Load one recording into an MNE Raw object."""
    arr = np.loadtxt(path)                    # (n_samples, 14)
    if arr.ndim != 2 or arr.shape[1] != len(C.CHANNELS):
        raise ValueError(f"{path.name}: expected 14 columns, got {arr.shape}")

    data = arr.T * C.COUNTS_TO_VOLTS          # (14, n_samples), volts
    info = mne.create_info(C.CHANNELS, C.SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020", on_missing="warn")
    return raw


def clean(raw: mne.io.RawArray, use_ica: bool = False) -> mne.io.RawArray:
    """Band-pass, optional ICA artifact removal, average re-reference."""
    raw = raw.copy().filter(C.L_FREQ, C.H_FREQ, fir_design="firwin")

    if use_ica:
        ica = mne.preprocessing.ICA(n_components=12, method="fastica",
                                    random_state=42, max_iter="auto")
        # ICA is fit on a 1 Hz high-passed copy (standard practice).
        ica.fit(raw)
        bad = []
        # AF3/AF4 sit over the frontal pole -> usable as pseudo-EOG channels.
        try:
            eog_idx, _ = ica.find_bads_eog(raw, ch_name=["AF3", "AF4"])
            bad += eog_idx
        except Exception:
            pass
        try:
            mus_idx, _ = ica.find_bads_muscle(raw)
            bad += mus_idx
        except Exception:
            pass
        ica.exclude = sorted(set(bad))
        raw = ica.apply(raw)

    raw.set_eeg_reference("average", projection=False)
    return raw


def epoch(raw: mne.io.RawArray) -> mne.Epochs:
    """Fixed-length overlapping epochs with scale-invariant artifact rejection."""
    ep = mne.make_fixed_length_epochs(
        raw, duration=C.EPOCH_SEC,
        overlap=C.OVERLAP_SEC, preload=True, verbose=False,
    )
    data = ep.get_data()                       # (n_ep, n_ch, n_times)
    p2p = (data.max(-1) - data.min(-1)).max(1)  # worst channel per epoch
    thresh = np.median(p2p) * C.REJECT_FACTOR
    keep = np.where(p2p <= thresh)[0]
    return ep[keep]


def process_recording(path: Path, use_ica: bool = False) -> mne.Epochs:
    return epoch(clean(load_raw(path), use_ica=use_ica))


if __name__ == "__main__":
    recs = list_recordings()
    print(f"{len(recs)} recordings, {len({s for s, _, _ in recs})} subjects")
    sid, lab, path = recs[0]
    ep = process_recording(path)
    print(f"sub{sid:02d} label={C.LABELS[lab]} -> {len(ep)} epochs of "
          f"{C.EPOCH_SEC}s x {len(C.CHANNELS)}ch")
