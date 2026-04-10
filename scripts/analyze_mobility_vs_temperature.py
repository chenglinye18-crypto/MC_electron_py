#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_scattering_mobility import (
    ROOT,
    _compute_mobility,
    _load_context,
    _load_dos_table,
    _resolve_fermi_level_eV,
    plt,
)


def _resolve_outdir(outdir: str | None, tmin: float, tmax: float, tstep: float) -> Path:
    if outdir:
        path = Path(outdir)
        if not path.is_absolute():
            path = ROOT / path
        return path
    return ROOT / "output" / f"MobilityVsTemperature_{tmin:g}K_{tmax:g}K_step{tstep:g}K"


def _build_temperature_grid(tmin: float, tmax: float, tstep: float) -> np.ndarray:
    if tmin <= 0.0 or tmax <= 0.0 or tstep <= 0.0:
        raise ValueError("Temperature bounds and step must all be > 0.")
    if tmax < tmin:
        raise ValueError("Require tmax >= tmin.")
    n = int(np.floor((tmax - tmin) / tstep + 1.0e-12)) + 1
    temps = tmin + np.arange(n, dtype=float) * tstep
    if temps[-1] < tmax - 1.0e-9:
        temps = np.append(temps, tmax)
    return temps


def _export_csv(out_csv: Path, rows: list[dict[str, float]]) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Temperature_K",
                "mu_iso_m2_per_Vs",
                "mu_iso_cm2_per_Vs",
                "carrier_density_m^-3",
                "carrier_density_cm^-3",
                "defect_density_m^-3",
                "defect_density_cm^-3",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{row['temperature_K']:.12g}",
                    f"{row['mu_iso_m2_per_Vs']:.12g}",
                    f"{row['mu_iso_cm2_per_Vs']:.12g}",
                    f"{row['carrier_density_m3']:.12g}",
                    f"{row['carrier_density_cm3']:.12g}",
                    f"{row['defect_density_m3']:.12g}",
                    f"{row['defect_density_cm3']:.12g}",
                ]
            )


def _plot_mu_vs_t(out_png: Path, rows: list[dict[str, float]]) -> None:
    if plt is None:
        return
    temps = np.asarray([row["temperature_K"] for row in rows], dtype=float)
    mu_cm2 = np.asarray([row["mu_iso_cm2_per_Vs"] for row in rows], dtype=float)
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(temps, mu_cm2, marker="o", linewidth=2.0)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Mobility (cm^2/V/s)")
    plt.title("Mobility vs Temperature")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan a temperature range and export mobility-vs-temperature CSV/plot."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "input" / "input.txt"),
        help="Path to master input file (default: ./input/input.txt)",
    )
    parser.add_argument("--tmin", type=float, required=True, help="Minimum temperature in K.")
    parser.add_argument("--tmax", type=float, required=True, help="Maximum temperature in K.")
    parser.add_argument("--tstep", type=float, required=True, help="Temperature step in K.")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: ./output/MobilityVsTemperature_<range>)",
    )
    parser.add_argument(
        "--dos",
        default=None,
        help="DOS file path override (default: data/bands/DOS_<material>.txt)",
    )
    parser.add_argument(
        "--ef",
        type=float,
        default=None,
        help="Absolute Fermi level in eV override. Default: read from ldg defects block.",
    )
    parser.add_argument(
        "--ec-ref",
        type=float,
        default=None,
        help="Absolute conduction-band reference Ec in eV. Default: defects.ec or Eg_real.",
    )
    parser.add_argument(
        "--ml",
        type=float,
        default=None,
        help="Longitudinal effective mass override (m*/m0) for parabolic velocity.",
    )
    parser.add_argument(
        "--mt",
        type=float,
        default=None,
        help="Transverse effective mass override (m*/m0) for parabolic velocity.",
    )
    args = parser.parse_args()

    temperatures = _build_temperature_grid(float(args.tmin), float(args.tmax), float(args.tstep))
    outdir = _resolve_outdir(args.outdir, float(args.tmin), float(args.tmax), float(args.tstep))
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    dos_path_resolved: Path | None = None

    for temperature in temperatures:
        input_path = str(Path(args.input).resolve())
        config, device, phys_config, band = _load_context(input_path, float(temperature))

        material = str(phys_config["material"])
        dos_path = Path(args.dos).resolve() if args.dos else ROOT / "data" / "bands" / f"DOS_{material}.txt"
        dos_path_resolved = dos_path
        dos_energy_eV, dos_val = _load_dos_table(dos_path)

        ef_abs_eV, ec_ref_eV, ef_rel_eV = _resolve_fermi_level_eV(device, phys_config, args.ef, args.ec_ref)
        total_norm = np.asarray(band.scattering_rate["total"], dtype=float)
        energy_grid_eV = (band.emin + np.arange(total_norm.size, dtype=float) * band.dtable) * band.eV0
        total_rate = total_norm / float(phys_config["scales"]["time0"])
        total_rate_on_dos = np.interp(dos_energy_eV, energy_grid_eV, total_rate, left=total_rate[0], right=total_rate[-1])

        ml = float(args.ml if args.ml is not None else phys_config["ml_val"])
        mt = float(args.mt if args.mt is not None else phys_config["mt_val"])
        mobility = _compute_mobility(
            dos_energy_eV,
            dos_val,
            total_rate_on_dos,
            ef_rel_eV,
            float(temperature),
            ml,
            mt,
            float(phys_config["q_e"]),
            float(phys_config["kb"]),
            float(phys_config["m0"]),
        )

        rows.append(
            {
                "temperature_K": float(temperature),
                "mu_iso_m2_per_Vs": float(mobility["mu_iso_m2_per_Vs"]),
                "mu_iso_cm2_per_Vs": float(mobility["mu_iso_m2_per_Vs"]) * 1.0e4,
                "carrier_density_m3": float(mobility["carrier_density_m3"]),
                "carrier_density_cm3": float(mobility["carrier_density_m3"]) * 1.0e-6,
                "defect_density_m3": float(phys_config.get("defect_density_m3", 0.0)),
                "defect_density_cm3": float(phys_config.get("defect_density_m3", 0.0)) * 1.0e-6,
            }
        )

        print(
            f"[Mobility(T)] T={temperature:.6g} K "
            f"mu={rows[-1]['mu_iso_cm2_per_Vs']:.6g} cm^2/V/s "
            f"n={rows[-1]['carrier_density_cm3']:.6g} cm^-3"
        )

    csv_path = outdir / "mobility_vs_temperature.csv"
    _export_csv(csv_path, rows)
    _plot_mu_vs_t(outdir / "mobility_vs_temperature.png", rows)

    summary_path = outdir / "mobility_vs_temperature_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Mobility vs Temperature Summary\n")
        f.write(f"input = {Path(args.input).resolve()}\n")
        f.write(f"DOS_file = {dos_path_resolved}\n")
        f.write(f"T_min_K = {float(args.tmin):.12g}\n")
        f.write(f"T_max_K = {float(args.tmax):.12g}\n")
        f.write(f"T_step_K = {float(args.tstep):.12g}\n")
        if rows:
            mu_vals = np.asarray([row["mu_iso_cm2_per_Vs"] for row in rows], dtype=float)
            t_vals = np.asarray([row["temperature_K"] for row in rows], dtype=float)
            f.write(f"mu_min_cm2_per_Vs = {float(np.min(mu_vals)):.12g}\n")
            f.write(f"mu_max_cm2_per_Vs = {float(np.max(mu_vals)):.12g}\n")
            f.write(f"T_at_mu_min_K = {float(t_vals[int(np.argmin(mu_vals))]):.12g}\n")
            f.write(f"T_at_mu_max_K = {float(t_vals[int(np.argmax(mu_vals))]):.12g}\n")

    print(f"[Mobility(T)] Output directory: {outdir}")
    print(f"[Mobility(T)] CSV: {csv_path}")
    print(f"[Mobility(T)] Summary: {summary_path}")
    if plt is None:
        print("[Mobility(T)] matplotlib is unavailable; png plot was skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
