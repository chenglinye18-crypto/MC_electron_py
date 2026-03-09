#!/usr/bin/env python3
"""
Plot E-v scatter from particle E-k data and compare with analytic E=1/2 m v^2.

Assumptions:
- mt/ml are given in units of m0 (dimensionless effective mass ratios).
- a0 is lattice constant in meters.
- kx/ky/kz in the particle file are in units of (pi/a0).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot E-v scatter and analytic curve")
    parser.add_argument("--input", required=True, help="Path to initial_particles.txt")
    parser.add_argument("--outdir", required=True, help="Output directory for the plot")
    parser.add_argument("--mt", type=float, required=True, help="Transverse effective mass (m*/m0)")
    parser.add_argument("--ml", type=float, required=True, help="Longitudinal effective mass (m*/m0)")
    parser.add_argument("--a0", type=float, required=True, help="Lattice constant a0 (meters)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    os.makedirs(args.outdir, exist_ok=True)

    data = np.loadtxt(args.input, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 11:
        raise ValueError("Input file must include kx, ky, kz, and energy columns.")

    kx = data[:, 7]
    ky = data[:, 8]
    kz = data[:, 9]
    energy_eV = data[:, 10]

    m0 = 9.109383e-31
    hbar = 1.054571e-34
    q_e = 1.602176e-19

    m_eff = (args.ml * args.mt * args.mt) ** (1.0 / 3.0) * m0

    k_scale = 2 * np.pi / args.a0
    k_mag = np.sqrt(kx * kx + ky * ky + kz * kz) * k_scale

    v_mag = hbar * k_mag / m_eff

    e_min = float(np.min(energy_eV))
    e_max = float(np.max(energy_eV))
    e_line = np.linspace(e_min, e_max, 400)
    v_line = np.sqrt(2.0 * e_line * q_e / m_eff)

    plt.figure(figsize=(7.0, 5.0))
    plt.scatter(energy_eV, v_mag, s=2, alpha=0.25, label="MC samples")
    plt.plot(e_line, v_line, color="red", linewidth=2.0, label="E=1/2 m v^2")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Velocity (m/s)")
    plt.title("E-v Scatter vs Analytic")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(args.outdir, "E_v_scatter.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
