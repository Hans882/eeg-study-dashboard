"""Band-power features (methodology 6.1, step 6) and dataset assembly.

Feature vector = log10 absolute band power for 4 bands x 14 channels = 56,
plus 3 workload-relevant ratios = 59 total.
"""
from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from src import config as C
from src.preprocess import list_recordings, process_recording

# numpy 2.x renamed trapz -> trapezoid and removed the old name
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def feature_names() -> list[str]:
    names = [f"{ch}_{band}" for band in C.BANDS for ch in C.CHANNELS]
    names += ["ratio_theta_alpha", "ratio_theta_beta", "engagement_index"]
    return names


def band_power(epochs) -> np.ndarray:
    """(n_epochs, 59) feature matrix from an MNE Epochs object."""
    spec = epochs.compute_psd(
        method="welch", fmin=C.L_FREQ, fmax=C.H_FREQ,
        n_fft=C.PSD_NFFT, n_overlap=C.PSD_NOVERLAP, verbose=False,
    )
    psds, freqs = spec.get_data(return_freqs=True)   # (n_ep, n_ch, n_freq)

    raw_bp = {}
    blocks = []
    for band, (lo, hi) in C.BANDS.items():
        idx = (freqs >= lo) & (freqs < hi)
        bp = _trapz(psds[:, :, idx], freqs[idx], axis=-1)   # (n_ep, n_ch)
        raw_bp[band] = bp
        blocks.append(np.log10(bp + 1e-20))

    # Scalar ratios averaged over channels. Theta/alpha rises and the
    # engagement index beta/(alpha+theta) shifts with mental effort -- both are
    # standard workload indices, and they give the SVM features that survive
    # per-subject amplitude differences.
    eps = 1e-20
    theta = raw_bp["theta"].mean(1)
    alpha = raw_bp["alpha"].mean(1)
    beta = raw_bp["beta"].mean(1)
    ratios = np.column_stack([
        np.log10(theta / (alpha + eps) + eps),
        np.log10(theta / (beta + eps) + eps),
        np.log10(beta / (alpha + theta + eps) + eps),
    ])
    return np.hstack(blocks + [ratios])


def build_dataset(use_ica: bool = False):
    """Run every recording through the pipeline. Returns X, y, groups."""
    X, y, groups = [], [], []
    for sid, label, path in tqdm(list_recordings(), desc="preprocessing"):
        ep = process_recording(path, use_ica=use_ica)
        if len(ep) == 0:
            print(f"  warn: sub{sid:02d} {path.name} produced 0 clean epochs")
            continue
        feats = band_power(ep)
        X.append(feats)
        y.append(np.full(len(feats), label))
        groups.append(np.full(len(feats), sid))

    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int8)
    groups = np.concatenate(groups).astype(np.int16)
    return X, y, groups


def main():
    ap = argparse.ArgumentParser(description="Build the feature dataset.")
    ap.add_argument("--ica", action="store_true",
                    help="run ICA artifact removal (slow, ~3x runtime)")
    args = ap.parse_args()

    X, y, groups = build_dataset(use_ica=args.ica)
    out = C.DATA_PROC / "features.npz"
    np.savez_compressed(out, X=X, y=y, groups=groups,
                        names=np.array(feature_names()))
    print(f"\nSaved {out}")
    print(f"  X {X.shape}  |  high={int(y.sum())} low={int((y == 0).sum())}"
          f"  |  {len(np.unique(groups))} subjects")


if __name__ == "__main__":
    main()
