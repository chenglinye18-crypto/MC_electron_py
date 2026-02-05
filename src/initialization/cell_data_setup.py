"""
Module: cell_data_setup.py
Description: Initialize per-cell volume, doping, and initial electron charge.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def init_cell_data(mesh, params: dict, phys_config: dict, device_structure: dict,
                   input_dir: Optional[str] = None) -> None:
    """
    Initialize per-cell data needed for electrostatics / carrier initialization.

    Behavior:
    - Always computes cell volume from mesh spacing.
    - Always builds donor/acceptor grids from device structure (if present).
    - If an external initial electron table is provided, this is a TODO hook.
      For now, it falls back to charge-neutrality initialization.
    - Only electrons are considered (holes are intentionally omitted).
    """
    print("[Init] Initializing cell data (volume/doping/electron charge)...")

    _ensure_volume(mesh)
    _ensure_doping_arrays(mesh)
    _fill_doping_from_device(mesh, device_structure)

    init_file = params["InitialConcentrationFile"]
    init_path = _resolve_input_path(init_file, input_dir)
    if init_path and os.path.isfile(init_path):
        print(f"      -> Loading initial electron concentration from: {init_path}")
        print("      [TODO] External concentration loader not implemented yet.")
        print("      -> Falling back to charge-neutrality initialization.")
        _init_electron_charge_neutral(mesh, phys_config)
    else:
        if init_path:
            print(f"      [Warning] File not found: {init_path}. Using charge-neutrality init.")
        else:
            print("      -> Using charge-neutrality initialization.")
        _init_electron_charge_neutral(mesh, phys_config)

    total_charge = float(np.sum(mesh.electron_charge))
    print(f"      -> Total initial electron charge: {total_charge:.4e}")


def init_point_data(mesh, phys_config: dict) -> None:
    """
    Build node-centered quantities from cell-centered fields.

    This mirrors the core C++ `init_point_data` intent:
    - distribute each cell volume and net doping*volume to 8 corner nodes (1/8 each),
    - recover node-average doping,
    - compute node built-in potential contribution `vadd = asinh(Nnet/(2*Ni))`,
    - compute `charge_fac = node_volume * Nc / conc0` (Nc optional for now).
    """
    print("[Init] Initializing point data (cell -> node mapping)...")

    _ensure_volume(mesh)
    _ensure_doping_arrays(mesh)

    sem_mask = _semiconductor_mask(mesh)
    cell_vol = np.where(sem_mask, mesh.volume, 0.0)
    cell_dop_charge = np.where(sem_mask, mesh.doping * mesh.volume, 0.0)

    mesh.node_volume = _distribute_cell_to_nodes(cell_vol)
    mesh.node_doping_charge = _distribute_cell_to_nodes(cell_dop_charge)

    node_dop = np.zeros_like(mesh.node_volume)
    valid = np.abs(mesh.node_volume) > 1e-30
    node_dop[valid] = mesh.node_doping_charge[valid] / mesh.node_volume[valid]
    mesh.node_doping = node_dop

    ni = float(phys_config["Ni_norm"])
    ni_safe = max(abs(ni), 1e-40)
    alpha = node_dop / (2.0 * ni_safe)
    mesh.node_vadd = np.arcsinh(alpha)

    # Keep boundary behavior close to legacy code: copy from nearest interior node.
    if mesh.node_vadd.shape[1] > 2:
        mesh.node_vadd[:, 0, :] = mesh.node_vadd[:, 1, :]
        mesh.node_vadd[:, -1, :] = mesh.node_vadd[:, -2, :]

    Nc_norm = float(phys_config["Nc_norm"])
    mesh.node_charge_fac = mesh.node_volume * Nc_norm

    print(
        "      -> Point data ready. "
        f"vadd range: [{np.min(mesh.node_vadd):.4e}, {np.max(mesh.node_vadd):.4e}]"
    )


def _resolve_input_path(path_value: Optional[str], input_dir: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    path_value = str(path_value).strip()
    if not path_value:
        return None
    if os.path.isabs(path_value) or input_dir is None:
        return path_value
    return os.path.join(input_dir, path_value)


def _ensure_volume(mesh) -> None:
    if mesh.volume is not None and mesh.volume.shape == (mesh.nx, mesh.ny, mesh.nz):
        return
    dx = mesh.dx[:, np.newaxis, np.newaxis]
    dy = mesh.dy[np.newaxis, :, np.newaxis]
    dz = mesh.dz[np.newaxis, np.newaxis, :]
    mesh.volume = dx * dy * dz


def _ensure_doping_arrays(mesh) -> None:
    shape = (mesh.nx, mesh.ny, mesh.nz)
    if mesh.donor is None or mesh.donor.shape != shape:
        mesh.donor = np.zeros(shape, dtype=float)
    if mesh.acceptor is None or mesh.acceptor.shape != shape:
        mesh.acceptor = np.zeros(shape, dtype=float)
    if mesh.doping is None or mesh.doping.shape != shape:
        mesh.doping = np.zeros(shape, dtype=float)
    if mesh.da_total is None or mesh.da_total.shape != shape:
        mesh.da_total = np.zeros(shape, dtype=float)
    if mesh.electron_charge is None or mesh.electron_charge.shape != shape:
        mesh.electron_charge = np.zeros(shape, dtype=float)


def _fill_doping_from_device(mesh, device_structure: dict) -> None:
    mesh.donor.fill(0.0)
    mesh.acceptor.fill(0.0)

    donors = device_structure["donors"]
    acceptors = device_structure["acceptors"]

    for entry in donors:
        bounds = entry["bounds"]
        value = float(entry["value"])
        _apply_bounds(mesh.donor, bounds, value, mesh)

    for entry in acceptors:
        bounds = entry["bounds"]
        value = float(entry["value"])
        _apply_bounds(mesh.acceptor, bounds, value, mesh)

    mesh.doping = mesh.donor - mesh.acceptor
    mesh.da_total = mesh.donor + mesh.acceptor


def _apply_bounds(target: np.ndarray, bounds: list, value: float, mesh) -> None:
    if len(bounds) != 6:
        return
    x1, x2, y1, y2, z1, z2 = bounds
    xs = slice(max(x1, 0), min(x2, mesh.nx - 1) + 1)
    ys = slice(max(y1, 0), min(y2, mesh.ny - 1) + 1)
    zs = slice(max(z1, 0), min(z2, mesh.nz - 1) + 1)
    target[xs, ys, zs] += value


def _init_electron_charge_neutral(mesh, phys_config: dict) -> None:
    """
    C++-style charge-neutrality initialization (electrons only):
      dope = donor - acceptor
      if dope > 0:  n = dope
      if dope < 0:  n = Ni^2 / dope  (negative; keeps electron charge negative)
      if dope == 0: n = Ni
      electron_charge = -n * volume   (note: n can be negative for p-type)
    """
    ni = float(phys_config["Ni_norm"])
    ni_sq = ni * ni

    sem_mask = _semiconductor_mask(mesh)

    mesh.electron_charge.fill(0.0)

    dope = mesh.doping[sem_mask]
    vol = mesh.volume[sem_mask]

    if dope.size == 0:
        return

    charge = np.zeros_like(dope)
    pos = dope > 0.0
    neg = dope < 0.0
    zero = ~(pos | neg)

    charge[pos] = -dope[pos] * vol[pos]
    if np.any(neg):
        charge[neg] = (ni_sq / dope[neg]) * vol[neg]
    if np.any(zero):
        charge[zero] = -ni * vol[zero]

    mesh.electron_charge[sem_mask] = charge


def _semiconductor_mask(mesh) -> np.ndarray:
    ids = mesh.label_map
    sem_ids = []
    for key in ("IGZO", "SILICON", "ZNO", "GA2O3"):
        if key in ids:
            sem_ids.append(ids[key])
    if not sem_ids:
        return np.zeros_like(mesh.material_id, dtype=bool)
    return np.isin(mesh.material_id, sem_ids)


def _distribute_cell_to_nodes(cell_data: np.ndarray) -> np.ndarray:
    """
    Scatter each cell value equally to its 8 corner nodes (factor 1/8).
    """
    nx, ny, nz = cell_data.shape
    out = np.zeros((nx + 1, ny + 1, nz + 1), dtype=cell_data.dtype)
    scaled = cell_data * 0.125

    out[:-1, :-1, :-1] += scaled
    out[1:, :-1, :-1] += scaled
    out[:-1, 1:, :-1] += scaled
    out[:-1, :-1, 1:] += scaled
    out[1:, 1:, :-1] += scaled
    out[1:, :-1, 1:] += scaled
    out[:-1, 1:, 1:] += scaled
    out[1:, 1:, 1:] += scaled

    return out
