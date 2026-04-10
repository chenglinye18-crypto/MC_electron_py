#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from initialization import init_physical_parameters
from physics.band_structure import AnalyticBand
from utils.parser import InputParser


BRANCH_NAMES = (
    "acoustic",
    "lo_abs",
    "lo_ems",
    "to_abs",
    "to_ems",
)


def _resolve_outdir(outdir: str | None, temperature: float) -> Path:
    if outdir:
        path = Path(outdir)
        if not path.is_absolute():
            path = ROOT / path
        return path
    return ROOT / "output" / f"ScatteringAnalysis_T{temperature:g}K"


def _load_context(input_path: str, temperature: float):
    parser = InputParser()
    config = parser.parse_master(input_path)
    base_dir = Path(input_path).resolve().parent
    config["input_dir"] = str(base_dir)
    config["Temperature"] = float(temperature)

    ldg_path = base_dir / str(config["device_file_name"])
    device = parser.parse_ldg(str(ldg_path))

    phys_config = init_physical_parameters(
        config,
        parser.found_semiconductors,
        device.get("defects"),
    )
    phys_config["energy_step_eV"] = float(config["energy_step_eV"])
    phys_config["energy_max_eV"] = float(config["energy_max_eV"])
    phys_config["init_energy_bin_min_eV"] = float(config.get("init_energy_bin_min_eV", 0.0))
    phys_config["init_energy_bin_split_eV"] = float(config.get("init_energy_bin_split_eV", 0.05))
    phys_config["init_energy_bin_step_low_eV"] = float(config.get("init_energy_bin_step_low_eV", 0.0001))
    phys_config["init_energy_bin_step_high_eV"] = float(config.get("init_energy_bin_step_high_eV", 0.002))
    phys_config["init_energy_bin_max_eV"] = float(config.get("init_energy_bin_max_eV", 8.0))

    bands_dir = ROOT / "data" / "bands"
    band = AnalyticBand(phys_config, str(bands_dir))
    band.initialize(output_root=None)
    return config, device, phys_config, band


def _resolve_fermi_level_eV(
    device: dict,
    phys_config: dict,
    ef_override: float | None,
    ec_ref_override: float | None,
) -> tuple[float, float, float]:
    material = str(phys_config["material"]).upper()
    defect_cfg = device.get("defects", {}).get(material, {})

    ef_abs_eV = float(
        ef_override
        if ef_override is not None
        else defect_cfg.get("ef", defect_cfg.get("efermi", phys_config["eg_real"]))
    )
    ec_ref_eV = float(
        ec_ref_override
        if ec_ref_override is not None
        else defect_cfg.get("ec", phys_config["eg_real"])
    )
    ef_rel_eV = ef_abs_eV - ec_ref_eV
    return ef_abs_eV, ec_ref_eV, ef_rel_eV


def _load_dos_table(pathname: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(pathname, skiprows=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 2:
        raise ValueError(f"DOS file must have at least 2 columns: {pathname}")
    energy_eV = np.asarray(raw[:, 0], dtype=float)
    dos = np.asarray(raw[:, 1], dtype=float)
    order = np.argsort(energy_eV)
    return energy_eV[order], dos[order]


def _export_scattering_csv(
    out_csv: Path,
    energy_eV: np.ndarray,
    total_rate: np.ndarray,
    comp_rates: np.ndarray,
) -> None:
    tau = np.full_like(total_rate, np.inf, dtype=float)
    np.divide(1.0, total_rate, out=tau, where=(total_rate > 1.0e-30))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Energy_eV",
                "Total_1_per_s",
                "Acoustic_1_per_s",
                "LO_Abs_1_per_s",
                "LO_Ems_1_per_s",
                "TO_Abs_1_per_s",
                "TO_Ems_1_per_s",
                "Tau_total_s",
            ]
        )
        for idx in range(energy_eV.size):
            writer.writerow(
                [
                    f"{energy_eV[idx]:.12g}",
                    f"{total_rate[idx]:.12g}",
                    f"{comp_rates[0, idx]:.12g}",
                    f"{comp_rates[1, idx]:.12g}",
                    f"{comp_rates[2, idx]:.12g}",
                    f"{comp_rates[3, idx]:.12g}",
                    f"{comp_rates[4, idx]:.12g}",
                    f"{tau[idx]:.12g}",
                ]
            )


def _plot_scattering_rates(
    out_png: Path,
    energy_eV: np.ndarray,
    total_rate: np.ndarray,
    comp_rates: np.ndarray,
    logy: bool,
) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8.0, 5.5))
    plot_fn = plt.semilogy if logy else plt.plot
    plot_fn(energy_eV, total_rate, linewidth=2.2, label="Total")
    plot_fn(energy_eV, comp_rates[0], label="Acoustic")
    plot_fn(energy_eV, comp_rates[1], label="LO Abs")
    plot_fn(energy_eV, comp_rates[2], label="LO Ems")
    plot_fn(energy_eV, comp_rates[3], label="TO Abs")
    plot_fn(energy_eV, comp_rates[4], label="TO Ems")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Scattering Rate (1/s)")
    plt.title("Scattering Rates vs Energy")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _compute_mobility(
    energy_eV: np.ndarray,
    dos_1_per_eV_m3: np.ndarray,
    total_rate_1_per_s: np.ndarray,
    ef_rel_eV: float,
    temperature_K: float,
    ml: float,
    mt: float,
    q_e: float,
    k_b: float,
    m0: float,
) -> dict[str, np.ndarray | float]:
    kbt_eV = (k_b * temperature_K) / q_e
    if kbt_eV <= 0.0:
        raise ValueError("Temperature must be > 0 K.")

    arg = np.clip((energy_eV - ef_rel_eV) / kbt_eV, -300.0, 300.0)
    f0 = 1.0 / (1.0 + np.exp(arg))
    minus_df_dE = f0 * (1.0 - f0) / kbt_eV

    carrier_density = float(np.trapezoid(dos_1_per_eV_m3 * f0, energy_eV))
    tau = np.zeros_like(total_rate_1_per_s, dtype=float)
    np.divide(1.0, total_rate_1_per_s, out=tau, where=(total_rate_1_per_s > 1.0e-30))

    md = (ml * mt * mt) ** (1.0 / 3.0) * m0
    e_J = np.maximum(energy_eV, 0.0) * q_e

    # Isotropic parabolic approximation:
    #   m_eff = (mt*mt*ml)^(1/3)
    #   v^2 = 2E / m_eff
    #   <v_x^2> = v^2 / 3
    v2 = 2.0 * e_J / md
    v2_over_3 = v2 / 3.0

    common = tau * minus_df_dE * dos_1_per_eV_m3
    integrand = v2_over_3 * common

    # The transport integral here is performed over energy in eV with
    # DOS given in 1/(eV·m^3), so the conductivity prefactor reduces to 1/n.
    # A factor q would be needed only if the whole integral were expressed
    # in Joules with DOS per Joule.
    pref = 1.0 / max(carrier_density, 1.0e-300)
    mu_iso = pref * float(np.trapezoid(integrand, energy_eV))

    return {
        "f0": f0,
        "minus_df_dE": minus_df_dE,
        "tau": tau,
        "md_kg": md,
        "v2": v2,
        "v2_over_3": v2_over_3,
        "integrand": integrand,
        "carrier_density_m3": carrier_density,
        "mu_iso_m2_per_Vs": mu_iso,
    }


def _export_mobility_integrand_csv(
    out_csv: Path,
    energy_eV: np.ndarray,
    dos: np.ndarray,
    total_rate: np.ndarray,
    mobility: dict[str, np.ndarray | float],
) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Energy_eV",
                "DOS_1_per_eV_m3",
                "TotalRate_1_per_s",
                "Tau_s",
                "f0",
                "minus_df0_dE_1_per_eV",
                "v2_m2_per_s2",
                "v2_over_3_m2_per_s2",
                "Integrand_iso",
            ]
        )
        for idx in range(energy_eV.size):
            writer.writerow(
                [
                    f"{energy_eV[idx]:.12g}",
                    f"{dos[idx]:.12g}",
                    f"{total_rate[idx]:.12g}",
                    f"{mobility['tau'][idx]:.12g}",
                    f"{mobility['f0'][idx]:.12g}",
                    f"{mobility['minus_df_dE'][idx]:.12g}",
                    f"{mobility['v2'][idx]:.12g}",
                    f"{mobility['v2_over_3'][idx]:.12g}",
                    f"{mobility['integrand'][idx]:.12g}",
                ]
            )


def _plot_mobility_integrands(out_png: Path, energy_eV: np.ndarray, mobility: dict[str, np.ndarray | float]) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8.0, 5.5))
    plt.plot(energy_eV, mobility["integrand"], label="isotropic")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Mobility Integrand")
    plt.title("Mobility Integrand vs Energy")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export phonon scattering rates and estimate mobility from DOS + parabolic velocity."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "input" / "input.txt"),
        help="Path to master input file (default: ./input/input.txt)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Lattice temperature in K used to rebuild scattering rates and mobility integrals.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: ./output/ScatteringAnalysis_T{T}K)",
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

    input_path = str(Path(args.input).resolve())
    config, device, phys_config, band = _load_context(input_path, args.temperature)

    outdir = _resolve_outdir(args.outdir, args.temperature)
    outdir.mkdir(parents=True, exist_ok=True)

    total_norm = np.asarray(band.scattering_rate["total"], dtype=float)
    comp_norm = np.asarray(band.scattering_rate["components"], dtype=float)
    time0 = float(phys_config["scales"]["time0"])
    energy_grid_eV = (band.emin + np.arange(total_norm.size, dtype=float) * band.dtable) * band.eV0
    total_rate = total_norm / time0
    comp_rates = comp_norm / time0

    scattering_csv = outdir / "scattering_rates_vs_energy.csv"
    _export_scattering_csv(scattering_csv, energy_grid_eV, total_rate, comp_rates)
    _plot_scattering_rates(outdir / "scattering_rates_log.png", energy_grid_eV, total_rate, comp_rates, logy=True)
    _plot_scattering_rates(outdir / "scattering_rates_linear.png", energy_grid_eV, total_rate, comp_rates, logy=False)

    material = str(phys_config["material"])
    dos_path = Path(args.dos).resolve() if args.dos else ROOT / "data" / "bands" / f"DOS_{material}.txt"
    dos_energy_eV, dos_val = _load_dos_table(dos_path)

    ef_abs_eV, ec_ref_eV, ef_rel_eV = _resolve_fermi_level_eV(device, phys_config, args.ef, args.ec_ref)
    total_rate_on_dos = np.interp(dos_energy_eV, energy_grid_eV, total_rate, left=total_rate[0], right=total_rate[-1])

    ml = float(args.ml if args.ml is not None else phys_config["ml_val"])
    mt = float(args.mt if args.mt is not None else phys_config["mt_val"])
    mobility = _compute_mobility(
        dos_energy_eV,
        dos_val,
        total_rate_on_dos,
        ef_rel_eV,
        float(args.temperature),
        ml,
        mt,
        float(phys_config["q_e"]),
        float(phys_config["kb"]),
        float(phys_config["m0"]),
    )

    _export_mobility_integrand_csv(
        outdir / "mobility_integrand.csv",
        dos_energy_eV,
        dos_val,
        total_rate_on_dos,
        mobility,
    )
    _plot_mobility_integrands(outdir / "mobility_integrands.png", dos_energy_eV, mobility)

    with (outdir / "mobility_summary.txt").open("w", encoding="utf-8") as f:
        defect_m3 = float(phys_config.get("defect_density_m3", 0.0))
        defect_detail = phys_config.get("defect_model", {}) or {}
        tail_acc_m3 = float(defect_detail.get("tail_acceptor_occupied_m3", 0.0))
        gauss_don_m3 = float(defect_detail.get("gauss_donor_ionized_m3", 0.0))
        f.write("Scattering / Mobility Analysis Summary\n")
        f.write(f"input = {input_path}\n")
        f.write(f"material = {material}\n")
        f.write(f"temperature_K = {args.temperature:.6g}\n")
        f.write(f"DOS_file = {dos_path}\n")
        f.write(f"EF_abs_eV = {ef_abs_eV:.12g}\n")
        f.write(f"Ec_ref_eV = {ec_ref_eV:.12g}\n")
        f.write(f"EF_rel_to_Ec_eV = {ef_rel_eV:.12g}\n")
        f.write(f"ml = {ml:.12g}\n")
        f.write(f"mt = {mt:.12g}\n")
        f.write(f"md_kg = {mobility['md_kg']:.12g}\n")
        f.write(f"md_over_m0 = {mobility['md_kg'] / phys_config['m0']:.12g}\n")
        f.write(f"defect_density_m^-3 = {defect_m3:.12g}\n")
        f.write(f"defect_density_cm^-3 = {defect_m3 * 1.0e-6:.12g}\n")
        f.write(f"tail_acceptor_occupied_m^-3 = {tail_acc_m3:.12g}\n")
        f.write(f"tail_acceptor_occupied_cm^-3 = {tail_acc_m3 * 1.0e-6:.12g}\n")
        f.write(f"gauss_donor_ionized_m^-3 = {gauss_don_m3:.12g}\n")
        f.write(f"gauss_donor_ionized_cm^-3 = {gauss_don_m3 * 1.0e-6:.12g}\n")
        f.write(f"carrier_density_m^-3 = {mobility['carrier_density_m3']:.12g}\n")
        f.write(f"mu_iso_m2_per_Vs = {mobility['mu_iso_m2_per_Vs']:.12g}\n")
        f.write(f"mu_iso_cm2_per_Vs = {mobility['mu_iso_m2_per_Vs'] * 1.0e4:.12g}\n")

    print(f"[Analysis] Output directory: {outdir}")
    print(f"[Analysis] Scattering CSV: {scattering_csv}")
    print(f"[Analysis] Mobility summary: {outdir / 'mobility_summary.txt'}")
    if plt is None:
        print("[Analysis] matplotlib is unavailable; png plots were skipped.")
    print(
        "[Analysis] Mobility (cm^2/V/s): "
        f"mu_iso={mobility['mu_iso_m2_per_Vs'] * 1.0e4:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
