#!/usr/bin/env python3
"""
Generate a 3D parabolic-band DOS table:
  DOS(E) = (1 / (2*pi^2)) * (2*m_d/hbar^2)^(3/2) * sqrt(E)

Output columns:
  Energy(eV) DOS(1/eV/m^3)

Notes:
- E is measured from the conduction-band minimum (E >= 0).
- m_d uses anisotropic effective masses: m_d = (ml * mt^2)^(1/3) * m0.
- Result includes spin degeneracy (g_s = 2), and does not include valley degeneracy.
"""
from __future__ import annotations

import argparse
import math
import os


def _build_energy_grid(emax_eV: float, de_eV: float) -> list[float]:
    n_steps = int(math.floor(emax_eV / de_eV + 1e-12))
    energy = [i * de_eV for i in range(n_steps + 1)]
    if not energy or energy[-1] < emax_eV - 1e-12:
        energy.append(emax_eV)
    return energy


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate DOS table with 3D parabolic approximation."
    )
    parser.add_argument("--emax", type=float, required=True, help="Maximum energy in eV")
    parser.add_argument("--de", type=float, required=True, help="Energy step in eV")
    parser.add_argument("--ml", type=float, required=True, help="Longitudinal effective mass (m*/m0)")
    parser.add_argument("--mt", type=float, required=True, help="Transverse effective mass (m*/m0)")
    parser.add_argument(
        "--out",
        default="data/bands/DOS_parabolic.txt",
        help="Output DOS file path (default: data/bands/DOS_parabolic.txt)",
    )
    args = parser.parse_args()

    if args.emax < 0.0:
        raise ValueError("--emax must be >= 0")
    if args.de <= 0.0:
        raise ValueError("--de must be > 0")
    if args.ml <= 0.0 or args.mt <= 0.0:
        raise ValueError("--ml and --mt must be > 0")

    m0 = 9.109383e-31
    hbar = 1.054571e-34
    q_e = 1.602176e-19

    m_d = (args.ml * args.mt * args.mt) ** (1.0 / 3.0) * m0
    energy_eV = _build_energy_grid(args.emax, args.de)
    prefac_per_J = (1.0 / (2.0 * math.pi * math.pi)) * ((2.0 * m_d) / (hbar * hbar)) ** 1.5
    dos_per_eV = []
    for e_eV in energy_eV:
        e_J = max(e_eV, 0.0) * q_e
        dos_per_eV.append(prefac_per_J * math.sqrt(e_J) * q_e)

    out_path = args.out
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Energy(eV) DOS(1/eV/m^3)\n")
        for e, g in zip(energy_eV, dos_per_eV):
            f.write(f"{e:.12g} {g:.12g}\n")

    print(f"Wrote DOS file to: {out_path}")
    print(f"Rows: {len(energy_eV)}, Emax={energy_eV[-1]:.6g} eV, dE={args.de:.6g} eV")
    print(f"Effective DOS mass m_d/m0 = {m_d / m0:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
