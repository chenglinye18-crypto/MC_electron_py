#!/usr/bin/env python3
"""
3D Monte Carlo Simulator (Python Refactor Skeleton)
"""

import argparse
import os
import sys
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
from Particle import Particle


def print_banner() -> None:
    print("----------------------------------------------------")
    print("-3D Monte Carlo Simulator for Semiconductor Devices-")
    print("-    Developed by Chenglin Ye, Peking University   -")
    print("-    Email:chenglinye18@gmail.com                  -")
    print("-    Version: 1.0.0, 2025-11-01                    -")
    print("----------------------------------------------------")


def initialize(config: dict) -> None:
    """
    Placeholder for initialization stage.
    """
    print("[Init] Base initialization stage (placeholder).")


def run_mc(config: dict) -> None:
    """
    Placeholder for Monte Carlo main loop.
    """
    print("[3/4] Entering MC loop (placeholder)...")
    total_steps = 10
    dt_fs = 0.1
    for step in range(total_steps):
        if step % 5 == 0:
            print(f"  Step {step:5d} | Time {step * dt_fs:8.2f} fs")


def postprocess(config: dict) -> None:
    """
    Placeholder for output stage.
    """
    print("[4/4] Post-processing and saving results (placeholder).")


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

    print(f"[STEP 1] Reading input file: {args.input}")
    parser = InputParser()
    config = parser.parse_master(args.input)
    if config:
        print("  -> gridFile        :", config["gridFile"])
        print("  -> device_file_name:", config["device_file_name"])
        print("  -> total_step      :", config["total_step"])
        print("  -> ElectronNumber  :", config["ElectronNumber"])
        print("  -> Temperature     :", config["Temperature"])
        print("  -> dt              :", config["dt"])
        print("  -> output_dir      :", config["output_dir"])

        dt_val = config["dt"]
        if isinstance(dt_val, (int, float)) and dt_val > 1e-15:
            print("  -> Warning: dt is large for MC; consider a smaller step.")

    base_dir = os.path.dirname(os.path.abspath(args.input))
    ldg_name = config["device_file_name"]
    ldg_path = os.path.join(base_dir, ldg_name)
    print(f"[STEP 2] Reading device file: {ldg_path}")
    device = parser.parse_ldg(ldg_path)
    mats = sorted(parser.found_semiconductors)
    print("  -> Semiconductors  :", mats if mats else "None")
    print("  -> Regions         :", len(device["regions"]))
    print("  -> Contacts        :", len(device["contacts"]))

    grid_name = config["gridFile"]
    grid_path = os.path.join(base_dir, grid_name)
    print(f"[STEP 3] Reading grid file: {grid_path}")
    coords = parser.parse_lgrid(grid_path)
    mesh = Mesh(coords, device["regions"])
    print(f"  -> Mesh cells       : {mesh.nx} x {mesh.ny} x {mesh.nz}")

    # Initialization steps (placeholders)
    print("[STEP 4] Initializing physical parameters")
    phys_config = init_physical_parameters(config, parser.found_semiconductors)
    phys_config["energy_step_eV"] = float(config["energy_step_eV"])  #为了散射表
    phys_config["energy_max_eV"] = float(config["energy_max_eV"])   #为了散射表

    project_root = os.path.dirname(base_dir)
    output_base = config["output_dir"]
    if not os.path.isabs(output_base):
        output_base = os.path.join(project_root, output_base)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(output_base, run_id)
    os.makedirs(output_root, exist_ok=True)
    config["output_root"] = output_root
    print(f"[Main] Output directory created: {output_root}")

    print("[STEP 5] Initializing band structure")

    bands_dir = os.path.join(project_root, "data", "bands")
    band_struct = AnalyticBand(phys_config, bands_dir)
    band_struct.initialize(output_root=output_root)

    phys_config["Ni_norm"] = band_struct.Ni_norm
    phys_config["barrier_height_norm"] = band_struct.barrier_height_norm
    phys_config["beta_norm"] = band_struct.beta_norm
    phys_config["difpr"] = band_struct.difpr

    print("[STEP 6] Initializing cell and point data")
    init_cell_data(mesh, config, phys_config, device, input_dir=base_dir)
    init_point_data(mesh, phys_config)

    print("[STEP 7] Initializing Poisson solver")
    #初始化Poisson求解器，主要构建eps矩阵
    poisson_solver = PoissonSolver(mesh, phys_config, device, build_matrix=True)

    print("[STEP 8] Initializing particle ensemble")
    _ensemble = Particle(mesh, config, phys_config, band_struct, output_root)

    _ = poisson_solver  # placeholder to avoid unused warnings
    initialize(config)
    run_mc(config)
    postprocess(config)

    print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
