"""
Module: particle_ensemble.py
Description: Particle ensemble class with initialization in __init__.
"""
from __future__ import annotations

import os

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _select_k_from_bins(
    n,
    bins,
    ntlist,
    ptlist,
    tlist,
    wcdf,
    wsum,
    k_data,
    kx_out,
    ky_out,
    kz_out,
    k_idx_out,
):
    for i in prange(n):
        itab = bins[i]
        count = ntlist[itab]
        if count > 0:
            start = ptlist[itab]
            total_w = wsum[itab]
            if total_w > 0.0:
                target = np.random.random() * total_w
                left = 0
                right = count - 1
                while left < right:
                    mid = (left + right) // 2
                    if wcdf[start + mid] < target:
                        left = mid + 1
                    else:
                        right = mid
                offset = left
            else:
                offset = np.random.randint(0, count)
            k_idx = tlist[start + offset]
        else:
            k_idx = 0
        k_idx_out[i] = k_idx
        kx_out[i] = k_data[k_idx, 0]
        ky_out[i] = k_data[k_idx, 1]
        kz_out[i] = k_data[k_idx, 2]


def _sample_thermal_k(n_particles, temperature, kb, q_e, band_struct):
    """
    Sample thermal energies from Boltzmann distribution and map to k via analytic bins.
    Returns kx, ky, kz (code units), energy (eV), and ek_data indices.
    """
    kBT = (kb * temperature) / q_e
    if getattr(band_struct, "analytic_bin_edges_eV", None) is None:
        raise ValueError("Energy-bin edges are not initialized.")
    bin_edges_eV = band_struct.analytic_bin_edges_eV
    e_max_eV = float(bin_edges_eV[-1])
    if e_max_eV <= 0.0:
        raise ValueError("Invalid maximum energy for sampling.")

    r = np.random.random(n_particles)
    exp_term = np.exp(-e_max_eV / kBT)
    energy = -kBT * np.log(1.0 - r * (1.0 - exp_term))

    bin_indices = band_struct.map_energy_to_bins(energy)

    ntlist = band_struct.analytic_ntlist  #能量位于哪一个能量bin
    empty_mask = ntlist[bin_indices] <= 0
    if np.any(empty_mask):
        nonempty_bins = band_struct.analytic_nonempty_bins
        if nonempty_bins is None or nonempty_bins.size == 0:
            raise ValueError("All energy bins are empty; cannot sample k states.")

        targets = bin_indices[empty_mask]
        pos = np.searchsorted(nonempty_bins, targets)
        left_idx = np.clip(pos - 1, 0, nonempty_bins.size - 1)
        right_idx = np.clip(pos, 0, nonempty_bins.size - 1)
        left_bins = nonempty_bins[left_idx]
        right_bins = nonempty_bins[right_idx]
        choose_right = np.abs(right_bins - targets) < np.abs(targets - left_bins)
        bin_indices[empty_mask] = np.where(choose_right, right_bins, left_bins)

    if band_struct.analytic_wcdf is None or band_struct.analytic_wsum is None:
        raise ValueError("Weighted k-sampling tables are not initialized.")

    kx = np.empty(n_particles, dtype=np.float64)
    ky = np.empty(n_particles, dtype=np.float64)
    kz = np.empty(n_particles, dtype=np.float64)
    k_idx = np.empty(n_particles, dtype=np.int64)
    _select_k_from_bins(n_particles,
        bin_indices,
        band_struct.analytic_ntlist,
        band_struct.analytic_ptlist,
        band_struct.analytic_tlist,
        band_struct.analytic_wcdf,
        band_struct.analytic_wsum,
        band_struct.ek_data["k_norm"],
        kx,
        ky,
        kz,
        k_idx,
    )

    return kx, ky, kz, energy, k_idx


class Particle:
    """
    Particle ensemble initialized on construction (electrons only).
    """

    def __init__(self, mesh, config: dict, phys_config: dict, band_struct, output_root: str):
        self._initialize(mesh, config, phys_config, band_struct, output_root)

    def _initialize(self, mesh, config: dict, phys_config: dict, band_struct, output_root: str) -> None:
        print("[Init] Building initial particle ensemble")

        total_particles_target = int(config["ElectronNumber"])
        if total_particles_target <= 0:
            raise ValueError("ElectronNumber must be positive.")

        total_charge = -float(np.sum(mesh.electron_charge))
        if total_charge <= 0.0:
            raise ValueError("Total electron charge is non-positive; cannot allocate particles.")

        weight_per_particle = total_charge / total_particles_target
        abs_charge = np.abs(mesh.electron_charge).ravel()
        raw_counts = abs_charge / weight_per_particle
        base_counts = np.floor(raw_counts).astype(np.int64)
        remain = total_particles_target - int(base_counts.sum())

        if remain < 0:
            raise ValueError("Particle count underflow; check charge distribution.")
        if remain > 0:
            frac = raw_counts - base_counts
            idx = np.argpartition(frac, -remain)[-remain:]
            base_counts[idx] += 1

        if int(base_counts.sum()) != total_particles_target:
            raise ValueError("Particle count mismatch after distribution.")

        cell_indices = np.repeat(np.arange(base_counts.size, dtype=np.int64), base_counts)

        cell_charge = mesh.electron_charge.ravel()
        per_cell_charge = np.zeros_like(cell_charge)
        nonzero = base_counts > 0
        per_cell_charge[nonzero] = cell_charge[nonzero] / base_counts[nonzero]
        particle_charge = np.repeat(per_cell_charge, base_counts)

        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
        ci, cj, ck = np.unravel_index(cell_indices, (nx, ny, nz))

        rand_pos = np.random.random((cell_indices.size, 3))
        x = mesh.x_nodes[ci] + rand_pos[:, 0] * mesh.dx[ci]
        y = mesh.y_nodes[cj] + rand_pos[:, 1] * mesh.dy[cj]
        z = mesh.z_nodes[ck] + rand_pos[:, 2] * mesh.dz[ck]

        temperature = phys_config["Temperature"]
        kb = phys_config["kb"]
        q_e = phys_config["q_e"]

        kx, ky, kz, energy, _k_idx = _sample_thermal_k(cell_indices.size, temperature, kb, q_e, band_struct)  #归一化

        to_pi = phys_config["sia0_norm"] / np.pi
        kx_idx = band_struct.get_axis_indices_vectorized(kx * to_pi, axis="x")
        ky_idx = band_struct.get_axis_indices_vectorized(ky * to_pi, axis="y")
        kz_idx = band_struct.get_axis_indices_vectorized(kz * to_pi, axis="z")

        seed = np.arange(cell_indices.size, dtype=np.int64)
        left_time = np.full(cell_indices.size, float(config["dt"]), dtype=float)

        self.size = cell_indices.size
        self.x = x
        self.y = y
        self.z = z
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.energy = energy
        self.charge = particle_charge
        self.i = ci
        self.j = cj
        self.k = ck
        self.seed = seed
        self.left_time = left_time
        self.kx_idx = kx_idx
        self.ky_idx = ky_idx
        self.kz_idx = kz_idx

        export_flag = int(config["export_particles"])
        if export_flag not in (0, 1):
            raise ValueError("export_particles must be 0 or 1.")
        if export_flag == 1:
            self._export_particles(output_root, phys_config)

    def _export_particles(self, output_root: str, phys_config) -> None:
        particle_dir = os.path.join(output_root, "Particles")
        os.makedirs(particle_dir, exist_ok=True)
        out_path = os.path.join(particle_dir, "initial_particles.txt")

        k_scale = np.pi / phys_config["sia0_norm"]

        data = np.column_stack(
            (
                np.arange(self.size, dtype=np.int64),
                self.i,
                self.j,
                self.k,
                self.x,
                self.y,
                self.z,
                self.kx/k_scale,
                self.ky/k_scale,
                self.kz/k_scale,
                self.energy,
                self.charge,
                self.kx_idx,
                self.ky_idx,
                self.kz_idx,
            )
        )
        header = "ID i j k     x(m)         y(m)          z(m)       kx(pi/a)  ky(pi/a)  kz(pi/a)   energy_eV   charge(/q)    kx_idx ky_idx kz_idx"
        fmt = ["%d", "%d", "%d", "%d"] + ["%.6e"] * 3 + ["%.3e"] * 3 + ["%.6e"] * 2 + ["%d", "%d", "%d"]
        np.savetxt(out_path, data, header=header, fmt=fmt)
        print(f"  -> Initial particles exported: {out_path}")
