"""Generate STEW-shaped fake recordings so you can test the whole pipeline
before the real dataset finishes downloading.

    python -m src.make_synthetic --subjects 8

Writes data/stew/sub01_lo.txt ... into a SEPARATE folder guard: it refuses to
run if real recordings are already present. Numbers are not physiological
beyond a theta-up / alpha-down shift in the "high" condition -- this is a
plumbing test, not a result. Delete these files before reporting anything.
"""
from __future__ import annotations

import argparse

import numpy as np

from src import config as C

DURATION_S = 150  # STEW keeps 2.5 min per condition


def synth(label: int, rng: np.random.Generator, subject_gain: float) -> np.ndarray:
    n = int(DURATION_S * C.SFREQ)
    t = np.arange(n) / C.SFREQ
    out = np.zeros((n, len(C.CHANNELS)))

    # high workload: more frontal theta, less posterior alpha
    theta_amp = (9.0 if label else 4.5) * subject_gain
    alpha_amp = (4.0 if label else 9.0) * subject_gain

    for ch in range(len(C.CHANNELS)):
        pink = np.cumsum(rng.normal(0, 1, n))
        pink -= pink.mean()
        sig = 6.0 * pink / (np.abs(pink).max() + 1e-9) * 30
        sig += theta_amp * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 6.3))
        sig += alpha_amp * np.sin(2 * np.pi * 10.5 * t + rng.uniform(0, 6.3))
        sig += rng.normal(0, 3.0, n)
        out[:, ch] = sig + 4200  # Emotiv-like DC offset, killed by the 1 Hz HP
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=8)
    args = ap.parse_args()

    existing = list(C.DATA_RAW.glob("sub*_lo.txt"))
    if existing and args.subjects and any(p.stat().st_size > 5_000_000 for p in existing):
        raise SystemExit("Real STEW files look present. Refusing to overwrite.")

    rng = np.random.default_rng(0)
    for s in range(1, args.subjects + 1):
        gain = rng.uniform(0.7, 1.4)  # per-subject amplitude difference
        for label, tag in [(0, "lo"), (1, "hi")]:
            path = C.DATA_RAW / f"sub{s:02d}_{tag}.txt"
            np.savetxt(path, synth(label, rng, gain), fmt="%.2f")
        print(f"wrote sub{s:02d}_lo.txt / sub{s:02d}_hi.txt")
    print(f"\n{args.subjects} synthetic subjects in {C.DATA_RAW}")


if __name__ == "__main__":
    main()
