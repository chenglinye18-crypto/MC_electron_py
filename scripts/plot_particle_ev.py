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

    m_t = args.mt * m0
    m_l = args.ml * m0

    # Input k is in (pi/a), so physical wavevector is k_real = k * (pi/a0).
    k_scale = np.pi / args.a0
    kx_real = kx * k_scale
    ky_real = ky * k_scale
    kz_real = kz * k_scale

    vx = hbar * kx_real / m_t
    vy = hbar * ky_real / m_t
    vz = hbar * kz_real / m_l
    v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)

    e_min = float(np.min(energy_eV))
    e_max = float(np.max(energy_eV))
    e0 = max(0.0, e_min)
    e_line = np.linspace(e0, e_max, 400) if e_max > e0 else np.array([e0])
    v_line_ml = np.sqrt(2.0 * e_line * q_e / m_l)
    v_line_mt = np.sqrt(2.0 * e_line * q_e / m_t)

    plt.figure(figsize=(7.0, 5.0))
    plt.scatter(energy_eV, v_mag, s=2, alpha=0.25, label="MC samples")
    plt.plot(e_line, v_line_ml, color="red", linewidth=2.0, label="E=1/2 ml v^2")
    plt.plot(e_line, v_line_mt, color="blue", linewidth=2.0, label="E=1/2 mt v^2")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Velocity (m/s)")
    plt.title("E-v Scatter vs Analytic (ml/mt)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(args.outdir, "E_v_scatter.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
