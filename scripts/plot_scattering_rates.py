#!/usr/bin/env python3
"""
Plot scattering rates vs energy from a scattering_rates.txt file.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_input(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "Scatter", "scattering_rates.txt")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot scattering rates vs energy (log y)")
    parser.add_argument(
        "path",
        help="Output root directory (containing Scatter/) or direct path to scattering_rates.txt",
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Output image filename (default: scattering_rates.png next to input file)",
    )
    args = parser.parse_args()

    infile = _resolve_input(args.path)
    if not os.path.isfile(infile):
        print(f"Error: scattering file not found: {infile}")
        return 1

    energies = []
    total = []
    ac = []
    lo_abs = []
    lo_ems = []
    to_abs = []
    to_ems = []

    with open(infile, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            energies.append(float(parts[0]))
            total.append(float(parts[1]))
            ac.append(float(parts[2]))
            lo_abs.append(float(parts[3]))
            lo_ems.append(float(parts[4]))
            to_abs.append(float(parts[5]))
            to_ems.append(float(parts[6]))

    if not energies:
        print("Error: no data rows found in scattering file.")
        return 1

    plt.figure(figsize=(7.5, 5.0))
    plt.semilogy(energies, total, label="Total", linewidth=2.0)
    plt.semilogy(energies, ac, label="AC")
    plt.semilogy(energies, lo_abs, label="LO Abs")
    plt.semilogy(energies, lo_ems, label="LO Em")
    plt.semilogy(energies, to_abs, label="TO Abs")
    plt.semilogy(energies, to_ems, label="TO Em")

    plt.xlabel("Energy (eV)")
    plt.ylabel("Scattering Rate (1/s)")
    plt.title("Scattering Rates vs Energy")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(infile))
    if args.outfile:
        outfile = args.outfile
        if not os.path.isabs(outfile):
            outfile = os.path.join(out_dir, outfile)
    else:
        outfile = os.path.join(out_dir, "scattering_rates.png")

    plt.savefig(outfile, dpi=200)
    print(f"Saved plot to: {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
