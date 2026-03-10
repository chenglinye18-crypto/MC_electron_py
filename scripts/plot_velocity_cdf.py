#!/usr/bin/env python3
"""
Plot empirical CDF of initial particle speed magnitude.

Velocity model:
  vx = hbar * kx / (mt * m0)
  vy = hbar * ky / (mt * m0)
  vz = hbar * kz / (ml * m0)
  |v| = sqrt(vx^2 + vy^2 + vz^2)
where kx, ky, kz are converted from (pi/a) to (1/m) by k_real = k * (pi/a0).
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
    parser = argparse.ArgumentParser(description="Plot initial particle speed CDF")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to initial_particles.txt or output root directory containing Particles/",
    )
    parser.add_argument("--outdir", required=True, help="Output directory for the figure")
    parser.add_argument("--mt", type=float, required=True, help="Transverse effective mass (m*/m0)")
    parser.add_argument("--ml", type=float, required=True, help="Longitudinal effective mass (m*/m0)")
    parser.add_argument("--a0", type=float, required=True, help="Lattice constant a0 (meters)")
    parser.add_argument("--outfile", default="Velocity_CDF.png", help="Output image filename")
    args = parser.parse_args()

    if args.mt <= 0.0 or args.ml <= 0.0:
        raise ValueError("--mt and --ml must be > 0")
    if args.a0 <= 0.0:
        raise ValueError("--a0 must be > 0")

    infile = _resolve_input(args.input)
    if not os.path.isfile(infile):
        raise FileNotFoundError(f"Input file not found: {infile}")
    os.makedirs(args.outdir, exist_ok=True)

    data = np.loadtxt(infile, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 10:
        raise ValueError("Input file must include kx, ky, kz columns at indices 7, 8, 9.")

    kx = np.asarray(data[:, 7], dtype=float)
    ky = np.asarray(data[:, 8], dtype=float)
    kz = np.asarray(data[:, 9], dtype=float)

    valid = np.isfinite(kx) & np.isfinite(ky) & np.isfinite(kz)
    if not np.any(valid):
        raise ValueError("No valid k data found.")
    kx = kx[valid]
    ky = ky[valid]
    kz = kz[valid]

    m0 = 9.109383e-31
    hbar = 1.054571e-34

    m_t = args.mt * m0
    m_l = args.ml * m0
    k_scale = np.pi / args.a0  # (1/m) per (pi/a)

    kx_real = kx * k_scale
    ky_real = ky * k_scale
    kz_real = kz * k_scale

    vx = hbar * kx_real / m_t
    vy = hbar * ky_real / m_t
    vz = hbar * kz_real / m_l
    v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)

    v_sorted = np.sort(v_mag)
    cdf = np.arange(1, v_sorted.size + 1, dtype=float) / v_sorted.size

    plt.figure(figsize=(7.0, 5.0))
    plt.step(v_sorted, cdf, where="post", linewidth=1.8, label=f"Empirical CDF (N={v_sorted.size})")
    plt.xlabel("Speed |v| (m/s)")
    plt.ylabel("CDF")
    plt.title("Initial Particle Speed CDF")
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
