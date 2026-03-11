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
    测试完毕，没有问题
    Behavior:
    - Always computes cell volume from mesh spacing.
    - Always builds donor/acceptor grids from device structure (if present).
    - If an external initial electron table is provided, load point data,
      interpolate to cell centers, and initialize electron charge.
    - If loading fails, fall back to charge-neutrality initialization.
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
        try:
            n_cell_m3 = _load_external_electron_concentration_to_cells(init_path, mesh)
            _init_electron_charge_from_concentration(mesh, n_cell_m3)
            _export_initial_concentration(mesh, n_cell_m3, params.get("output_root"))
            print("      -> External concentration initialization applied.")
        except Exception as exc:
            print(f"      [Warning] External concentration loading failed: {exc}")
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
    - compute `charge_fac_real = node_volume * Nc_real`.
    """
    print("[Init] Initializing point data (cell -> node mapping)...")

    sem_mask = _semiconductor_mask(mesh)
    cell_vol = np.where(sem_mask, mesh.volume, 0.0)
    cell_dop_charge = np.where(sem_mask, mesh.doping * mesh.volume, 0.0)

    mesh.node_volume = _distribute_cell_to_nodes(cell_vol)
    mesh.node_doping_charge = _distribute_cell_to_nodes(cell_dop_charge)

    node_dop = np.zeros_like(mesh.node_volume)
    valid = np.abs(mesh.node_volume) > 1e-30
    node_dop[valid] = mesh.node_doping_charge[valid] / mesh.node_volume[valid]
    mesh.node_doping = node_dop

    conc0 = float(phys_config["scales"]["conc0"])
    ni = float(phys_config["Ni_norm"]) * conc0
    ni_safe = max(abs(ni), 1e-40)
    alpha = node_dop / (2.0 * ni_safe)
    mesh.node_vadd = np.arcsinh(alpha)

    # Disabled temporarily per request:
    # Keep boundary behavior close to legacy code by copying from nearest interior node.
    # if mesh.node_vadd.shape[1] > 2:
    #     mesh.node_vadd[:, 0, :] = mesh.node_vadd[:, 1, :]
    #     mesh.node_vadd[:, -1, :] = mesh.node_vadd[:, -2, :]

    Nc_real = float(phys_config["Nc_real"])
    mesh.node_charge_fac_real = mesh.node_volume * Nc_real

    print(
        "      -> Point data ready. "
        f"vadd range: [{np.min(mesh.node_vadd)*phys_config["scales"]["pot0_V"]:.4e}, {np.max(mesh.node_vadd)*phys_config["scales"]["pot0_V"]:.4e}]"
    )


def _resolve_input_path(path_value: Optional[str], input_dir: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    path_value = str(path_value).strip()
    if not path_value:
        return None
    if os.path.isabs(path_value) or input_dir is None:
        resolved = path_value
    else:
        resolved = os.path.join(input_dir, path_value)

    if os.path.isfile(resolved):
        return resolved
    if not resolved.lower().endswith(".dat"):
        dat_path = resolved + ".dat"
        if os.path.isfile(dat_path):
            return dat_path
    return resolved


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


def _init_electron_charge_from_concentration(mesh, n_cell_m3: np.ndarray) -> None:
    """
    Initialize electron charge from externally supplied electron concentration (1/m^3).
    Only semiconductor cells are populated.
    """
    _ensure_volume(mesh)
    _ensure_doping_arrays(mesh)
    if n_cell_m3.shape != (mesh.nx, mesh.ny, mesh.nz):
        raise ValueError("External concentration shape does not match mesh cell shape.")

    sem_mask = _semiconductor_mask(mesh)
    mesh.electron_charge.fill(0.0)
    n_safe = np.where(n_cell_m3 > 0.0, n_cell_m3, 0.0)
    mesh.electron_charge[sem_mask] = -n_safe[sem_mask] * mesh.volume[sem_mask]


def _load_external_electron_concentration_to_cells(path: str, mesh) -> np.ndarray:
    """
    Load TCAD point data file and interpolate electron concentration to cell centers.

    Expected columns: x y z n
    Assumptions:
    - x/y/z are in um and converted to meters by 1e-6.
    - n is in cm^-3 and converted to m^-3 by 1e6.
    """
    raw = np.loadtxt(path, comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 4:
        raise ValueError("Initial concentration file must have at least 4 columns: x y z n.")

    x_um = np.asarray(raw[:, 0], dtype=float)
    y_um = np.asarray(raw[:, 1], dtype=float)
    z_um = np.asarray(raw[:, 2], dtype=float)
    n_cm3 = np.asarray(raw[:, 3], dtype=float)

    valid = np.isfinite(x_um) & np.isfinite(y_um) & np.isfinite(z_um) & np.isfinite(n_cm3)
    if not np.any(valid):
        raise ValueError("No valid rows found in initial concentration file.")

    x_src = x_um[valid] * 1.0e-6
    y_src = y_um[valid] * 1.0e-6
    z_src = z_um[valid] * 1.0e-6
    n_src = n_cm3[valid] * 1.0e6

    x_ticks, y_ticks, z_ticks, n_grid = _build_structured_point_grid(x_src, y_src, z_src, n_src)

    xc = 0.5 * (mesh.x_nodes[:-1] + mesh.x_nodes[1:])
    yc = 0.5 * (mesh.y_nodes[:-1] + mesh.y_nodes[1:])
    zc = 0.5 * (mesh.z_nodes[:-1] + mesh.z_nodes[1:])
    xx, yy, zz = np.meshgrid(xc, yc, zc, indexing="ij")

    n_cell = _trilinear_interp_structured(
        x_ticks,
        y_ticks,
        z_ticks,
        n_grid,
        xx.ravel(),
        yy.ravel(),
        zz.ravel(),
    ).reshape(mesh.nx, mesh.ny, mesh.nz)

    return np.where(np.isfinite(n_cell), n_cell, 0.0)


def _build_structured_point_grid(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_ticks = np.unique(np.round(x, 18))
    y_ticks = np.unique(np.round(y, 18))
    z_ticks = np.unique(np.round(z, 18))

    ix = np.searchsorted(x_ticks, np.round(x, 18))
    iy = np.searchsorted(y_ticks, np.round(y, 18))
    iz = np.searchsorted(z_ticks, np.round(z, 18))

    grid = np.full((x_ticks.size, y_ticks.size, z_ticks.size), np.nan, dtype=float)
    grid[ix, iy, iz] = values

    if np.isnan(grid).any():
        raise ValueError("Point grid is incomplete; cannot build structured concentration grid.")

    return x_ticks, y_ticks, z_ticks, grid


def _trilinear_interp_structured(
    x_ticks: np.ndarray,
    y_ticks: np.ndarray,
    z_ticks: np.ndarray,
    grid: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
    zq: np.ndarray,
) -> np.ndarray:
    xq_c = np.clip(xq, x_ticks[0], x_ticks[-1])
    yq_c = np.clip(yq, y_ticks[0], y_ticks[-1])
    zq_c = np.clip(zq, z_ticks[0], z_ticks[-1])

    ix0 = np.searchsorted(x_ticks, xq_c, side="right") - 1
    iy0 = np.searchsorted(y_ticks, yq_c, side="right") - 1
    iz0 = np.searchsorted(z_ticks, zq_c, side="right") - 1
    ix0 = np.clip(ix0, 0, x_ticks.size - 2)
    iy0 = np.clip(iy0, 0, y_ticks.size - 2)
    iz0 = np.clip(iz0, 0, z_ticks.size - 2)

    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    x0 = x_ticks[ix0]
    x1 = x_ticks[ix1]
    y0 = y_ticks[iy0]
    y1 = y_ticks[iy1]
    z0 = z_ticks[iz0]
    z1 = z_ticks[iz1]

    tx = np.divide(xq_c - x0, x1 - x0, out=np.zeros_like(xq_c), where=(x1 > x0))
    ty = np.divide(yq_c - y0, y1 - y0, out=np.zeros_like(yq_c), where=(y1 > y0))
    tz = np.divide(zq_c - z0, z1 - z0, out=np.zeros_like(zq_c), where=(z1 > z0))

    c000 = grid[ix0, iy0, iz0]
    c100 = grid[ix1, iy0, iz0]
    c010 = grid[ix0, iy1, iz0]
    c110 = grid[ix1, iy1, iz0]
    c001 = grid[ix0, iy0, iz1]
    c101 = grid[ix1, iy0, iz1]
    c011 = grid[ix0, iy1, iz1]
    c111 = grid[ix1, iy1, iz1]

    wx0 = 1.0 - tx
    wy0 = 1.0 - ty
    wz0 = 1.0 - tz
    wx1 = tx
    wy1 = ty
    wz1 = tz

    return (
        c000 * wx0 * wy0 * wz0
        + c100 * wx1 * wy0 * wz0
        + c010 * wx0 * wy1 * wz0
        + c110 * wx1 * wy1 * wz0
        + c001 * wx0 * wy0 * wz1
        + c101 * wx1 * wy0 * wz1
        + c011 * wx0 * wy1 * wz1
        + c111 * wx1 * wy1 * wz1
    )


def _export_initial_concentration(mesh, n_cell_m3: np.ndarray, output_root: Optional[str]) -> None:
    if not output_root:
        return
    conc_dir = os.path.join(output_root, "Concentration")
    os.makedirs(conc_dir, exist_ok=True)
    out_path = os.path.join(conc_dir, "initial_electron_concentration_cells.txt")

    ii, jj, kk = np.meshgrid(
        np.arange(mesh.nx, dtype=np.int32),
        np.arange(mesh.ny, dtype=np.int32),
        np.arange(mesh.nz, dtype=np.int32),
        indexing="ij",
    )
    xc = 0.5 * (mesh.x_nodes[:-1] + mesh.x_nodes[1:])
    yc = 0.5 * (mesh.y_nodes[:-1] + mesh.y_nodes[1:])
    zc = 0.5 * (mesh.z_nodes[:-1] + mesh.z_nodes[1:])
    xx, yy, zz = np.meshgrid(xc, yc, zc, indexing="ij")

    n_cm3 = n_cell_m3 / 1.0e6
    data = np.column_stack(
        (
            ii.ravel(),
            jj.ravel(),
            kk.ravel(),
            xx.ravel(),
            yy.ravel(),
            zz.ravel(),
            n_cell_m3.ravel(),
            n_cm3.ravel(),
        )
    )
    header = "i j k x_center(m) y_center(m) z_center(m) n_init(1/m^3) n_init(1/cm^3)"
    fmt = ["%d", "%d", "%d"] + ["%.6e"] * 5
    np.savetxt(out_path, data, header=header, fmt=fmt)
    print(f"      -> Saved initial concentration to: {out_path}")


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
