#!/usr/bin/env python3
"""
Plot empirical CDF of particle energy from initial_particles.txt.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_input(path: str) -> str:
    if not os.path.isdir(path):
        return path
    candidate_direct = os.path.join(path, "initial_particles.txt")
    candidate_particles = os.path.join(path, "Particles", "initial_particles.txt")
    if os.path.isfile(candidate_direct):
        return candidate_direct
    return candidate_particles


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot particle energy CDF")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to initial_particles.txt or output root directory containing Particles/",
    )
    parser.add_argument("--outdir", required=True, help="Output directory for the figure")
    parser.add_argument("--outfile", default="Energy_CDF.png", help="Output image filename")
    args = parser.parse_args()

    infile = _resolve_input(args.input)
    if not os.path.isfile(infile):
        raise FileNotFoundError(f"Input file not found: {infile}")
    os.makedirs(args.outdir, exist_ok=True)

    data = np.loadtxt(infile, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 11:
        raise ValueError("Input file must include energy column at index 10.")

    energy_eV = np.asarray(data[:, 10], dtype=float)
    energy_eV = energy_eV[np.isfinite(energy_eV)]
    if energy_eV.size == 0:
        raise ValueError("No valid energy data found.")

    energy_sorted = np.sort(energy_eV)
    cdf = np.arange(1, energy_sorted.size + 1, dtype=float) / energy_sorted.size

    plt.figure(figsize=(7.0, 5.0))
    plt.step(energy_sorted, cdf, where="post", linewidth=1.8, label=f"Empirical CDF (N={energy_sorted.size})")
    plt.xlabel("Energy (eV)")
    plt.ylabel("CDF")
    plt.title("Particle Energy CDF")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = args.outfile
    if not os.path.isabs(out_path):
        out_path = os.path.join(args.outdir, out_path)
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
