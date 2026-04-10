#!/usr/bin/env python3
"""
3D Monte Carlo Simulator (Python Refactor Skeleton)
"""

import argparse
import os
import time

from utils.parser import InputParser
from physics.mesh import Mesh
from physics.band_structure import AnalyticBand
from Poisson import PoissonSolver
from initialization import (
    init_physical_parameters,
    init_cell_data,
    init_point_data,
)
from mc import Monte_Carlo_Simulation


def print_banner() -> None:
    print("----------------------------------------------------")
    print("-3D Monte Carlo Simulator for Semiconductor Devices-")
    print("-    Developed by Chenglin Ye, Peking University   -")
    print("-    Email:chenglinye18@gmail.com                  -")
    print("-    Version: 1.1.0, 2026-04-10                    -")
    print("----------------------------------------------------")


def main() -> int:
    parser = argparse.ArgumentParser(description="3DMC Python Refactor Skeleton")
    parser.add_argument(
        "--input",
        default="./input/input.txt",
        help="Path to master input file (default: ./input/input.txt)",
    )
    args = parser.parse_args()

    print_banner()
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    print(f"[Input] {args.input}")
    parser = InputParser()
    config = parser.parse_master(args.input)
    if config:
        dt_val = config["dt"]
        if isinstance(dt_val, (int, float)) and dt_val > 1e-15:
            print("  -> Warning: dt is large for MC; consider a smaller step.")

    base_dir = os.path.dirname(os.path.abspath(args.input))
    config["input_dir"] = base_dir
    ldg_name = config["device_file_name"]
    ldg_path = os.path.join(base_dir, ldg_name)
    device = parser.parse_ldg(ldg_path)
    mats = sorted(parser.found_semiconductors)
    print(
        f"[Device] semiconductors={mats if mats else 'None'} "
        f"regions={len(device['regions'])} contacts={len(device['contacts'])}"
    )

    monitor_file = str(config.get("CurrentMonitorFile", "")).strip()
    if monitor_file:
        monitor_path = os.path.join(base_dir, monitor_file)
        monitors = parser.parse_monitor_file(monitor_path)
        config["current_monitors"] = monitors
        print(f"[Monitor] surfaces={len(monitors)} file={monitor_file}")
    else:
        config["current_monitors"] = []

    grid_name = config["gridFile"]
    grid_path = os.path.join(base_dir, grid_name)
    coords = parser.parse_lgrid(grid_path)
    mesh = Mesh(coords, device["regions"])
    print(f"[Mesh] cells={mesh.nx} x {mesh.ny} x {mesh.nz}")

    print("[Init] Physical parameters")
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

    project_root = os.path.dirname(base_dir)
    output_base = config["output_dir"]
    if not os.path.isabs(output_base):
        output_base = os.path.join(project_root, output_base)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(output_base, run_id)
    os.makedirs(output_root, exist_ok=True)
    config["output_root"] = output_root
    print(f"[Run] output={output_root}")

    print("[Init] Band structure")
    bands_dir = os.path.join(project_root, "data", "bands")
    band_struct = AnalyticBand(phys_config, bands_dir)
    band_struct.initialize(output_root=output_root)

    phys_config["Ni_norm"] = band_struct.Ni_norm
    phys_config["barrier_height_norm"] = band_struct.barrier_height_norm
    phys_config["beta_norm"] = band_struct.beta_norm
    phys_config["difpr"] = band_struct.difpr

    print("[Init] Cell and point data")
    init_cell_data(mesh, config, phys_config, device, input_dir=base_dir)
    init_point_data(mesh, phys_config, device)

    print("[Init] Poisson solver")
    poisson_solver = PoissonSolver(mesh, phys_config, device, build_matrix=True)

    monte_carlo_simulation = Monte_Carlo_Simulation(
        mesh,
        config,
        phys_config,
        band_struct,
        output_root,
        poisson_solver=poisson_solver,
        device_structure=device,
    )
    monte_carlo_simulation.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
