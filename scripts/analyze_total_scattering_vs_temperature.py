#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np

from analyze_scattering_mobility import ROOT, _load_context, plt


def _resolve_outdir(outdir: str | None, temperatures: list[float]) -> Path:
    if outdir:
        path = Path(outdir)
        if not path.is_absolute():
            path = ROOT / path
        return path
    label = "_".join(f"{temp:g}K" for temp in temperatures)
    return ROOT / "output" / f"ScatteringTotalVsTemperature_{label}"


def _export_csv(out_csv: Path, energy_eV: np.ndarray, temps: list[float], total_rates: list[np.ndarray]) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Energy_eV"] + [f"TotalRate_{temp:g}K_1_per_s" for temp in temps]
        writer.writerow(header)
        for idx in range(energy_eV.size):
            row = [f"{energy_eV[idx]:.12g}"]
            for rates in total_rates:
                row.append(f"{rates[idx]:.12g}")
            writer.writerow(row)


def _plot_total_rates(out_png: Path, energy_eV: np.ndarray, temps: list[float], total_rates: list[np.ndarray]) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8.0, 5.5))
    for temp, rates in zip(temps, total_rates):
        plt.semilogy(energy_eV, rates, linewidth=2.0, label=f"{temp:g} K")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Total Scattering Rate (1/s)")
    plt.title("Total Scattering Rate vs Energy")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _plot_total_rates_svg(out_svg: Path, energy_eV: np.ndarray, temps: list[float], total_rates: list[np.ndarray]) -> None:
    width = 980
    height = 640
    margin_left = 90
    margin_right = 30
    margin_top = 40
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    xmin = float(np.min(energy_eV))
    xmax = float(np.max(energy_eV))
    positive_values = np.concatenate([rates[rates > 1.0e-30] for rates in total_rates if np.any(rates > 1.0e-30)])
    if positive_values.size == 0:
        ymin_log = -2.0
        ymax_log = 2.0
    else:
        ymin_log = float(np.floor(np.log10(np.min(positive_values))))
        ymax_log = float(np.ceil(np.log10(np.max(positive_values))))
        if ymax_log <= ymin_log:
            ymax_log = ymin_log + 1.0

    def sx(x: float) -> float:
        if xmax <= xmin:
            return margin_left
        return margin_left + (x - xmin) / (xmax - xmin) * plot_w

    def sy_log(y: float) -> float:
        y_safe = max(float(y), 1.0e-30)
        y_log = np.log10(y_safe)
        return margin_top + (ymax_log - y_log) / (ymax_log - ymin_log) * plot_h

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    lines: list[str] = []

    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="20" font-family="Arial">'
        'Total Scattering Rate vs Energy'
        '</text>'
    )

    x0 = margin_left
    y0 = margin_top + plot_h
    x1 = margin_left + plot_w
    y1 = margin_top
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="black" stroke-width="1.5"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="black" stroke-width="1.5"/>')

    for tick in np.linspace(xmin, xmax, 8):
        xt = sx(float(tick))
        lines.append(f'<line x1="{xt:.2f}" y1="{y0}" x2="{xt:.2f}" y2="{y0+6}" stroke="black" stroke-width="1"/>')
        lines.append(
            f'<text x="{xt:.2f}" y="{y0+24}" text-anchor="middle" font-size="12" font-family="Arial">{tick:.2f}</text>'
        )

    for exp in range(int(ymin_log), int(ymax_log) + 1):
        yv = 10.0 ** exp
        yt = sy_log(yv)
        lines.append(f'<line x1="{x0-6}" y1="{yt:.2f}" x2="{x0}" y2="{yt:.2f}" stroke="black" stroke-width="1"/>')
        lines.append(
            f'<line x1="{x0}" y1="{yt:.2f}" x2="{x1}" y2="{yt:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x0-10}" y="{yt+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">1e{exp}</text>'
        )

    lines.append(
        f'<text x="{width/2:.1f}" y="{height-20}" text-anchor="middle" font-size="16" font-family="Arial">Energy (eV)</text>'
    )
    lines.append(
        f'<text x="24" y="{height/2:.1f}" text-anchor="middle" font-size="16" font-family="Arial" '
        f'transform="rotate(-90 24 {height/2:.1f})">Total Scattering Rate (1/s)</text>'
    )

    legend_x = x1 - 140
    legend_y = y1 + 16
    for idx, (temp, rates) in enumerate(zip(temps, total_rates)):
        points = " ".join(f"{sx(float(x)):.2f},{sy_log(float(y)):.2f}" for x, y in zip(energy_eV, rates))
        color = colors[idx % len(colors)]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        ly = legend_y + idx * 22
        lines.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+24}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{legend_x+32}" y="{ly+4}" font-size="13" font-family="Arial">{html.escape(f"{temp:g} K")}</text>'
        )

    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export total scattering-rate vs energy curves for multiple temperatures."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "input" / "input.txt"),
        help="Path to master input file (default: ./input/input.txt)",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        required=True,
        help="Temperature list in K, e.g. --temperatures 300 320 350",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: ./output/ScatteringTotalVsTemperature_<temps>)",
    )
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temperatures]
    outdir = _resolve_outdir(args.outdir, temperatures)
    outdir.mkdir(parents=True, exist_ok=True)

    energy_ref = None
    total_rates: list[np.ndarray] = []
    rows: list[tuple[float, float, float, float]] = []

    for temp in temperatures:
        _, _, phys_config, band = _load_context(str(Path(args.input).resolve()), temp)
        total_norm = np.asarray(band.scattering_rate["total"], dtype=float)
        total_rate = total_norm / float(phys_config["scales"]["time0"])
        energy_eV = (band.emin + np.arange(total_norm.size, dtype=float) * band.dtable) * band.eV0

        if energy_ref is None:
            energy_ref = np.asarray(energy_eV, dtype=float)
        else:
            if energy_ref.shape != energy_eV.shape or not np.allclose(energy_ref, energy_eV, rtol=0.0, atol=1e-12):
                raise ValueError("Energy grids differ across temperatures; cannot export combined CSV.")

        total_rates.append(np.asarray(total_rate, dtype=float))
        rows.append((temp, float(np.min(total_rate)), float(np.max(total_rate)), float(total_rate[-1])))
        print(
            f"[Scatter(T)] T={temp:.6g} K "
            f"min={rows[-1][1]:.6e}  max={rows[-1][2]:.6e}  at_Emax={rows[-1][3]:.6e}"
        )

    assert energy_ref is not None
    csv_path = outdir / "total_scattering_vs_energy.csv"
    _export_csv(csv_path, energy_ref, temperatures, total_rates)
    plot_path = outdir / "total_scattering_vs_energy.png"
    plot_svg_path = outdir / "total_scattering_vs_energy.svg"
    _plot_total_rates(plot_path, energy_ref, temperatures, total_rates)
    if plt is None:
        _plot_total_rates_svg(plot_svg_path, energy_ref, temperatures, total_rates)

    summary_path = outdir / "total_scattering_vs_energy_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Total Scattering Rate vs Temperature Summary\n")
        f.write(f"input = {Path(args.input).resolve()}\n")
        f.write(f"temperatures_K = {', '.join(f'{t:g}' for t in temperatures)}\n")
        for temp, rate_min, rate_max, rate_emax in rows:
            f.write(
                f"T={temp:g}K min_1_per_s={rate_min:.12g} "
                f"max_1_per_s={rate_max:.12g} at_Emax_1_per_s={rate_emax:.12g}\n"
            )

    print(f"[Scatter(T)] Output directory: {outdir}")
    print(f"[Scatter(T)] CSV: {csv_path}")
    print(f"[Scatter(T)] Summary: {summary_path}")
    if plt is None:
        print(f"[Scatter(T)] SVG: {plot_svg_path}")
        print("[Scatter(T)] matplotlib is unavailable; png plot was skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
