"""
Module: band_structure.py
Description: Analytic band/phonon/scattering framework placeholder.
             Computes derived constants (Ni, barrier) for Poisson.
"""
from __future__ import annotations

import os
import numpy as np


class AnalyticBand:
    def __init__(self, phys_config: dict, input_path: str):
        self.phys = phys_config
        self.input_path = input_path

        self.dos_norm = None
        self.ek_data = {
            "k_pi": None,
            "k_norm": None,
            "energy": None,
            "velocity": None,
            "weight": None,
            "valley_idx": None,
        }
        self.emin = 0.0
        self.dtable_eV = float(self.phys["energy_step_eV"])
        self.energy_max_eV = float(self.phys["energy_max_eV"])
        self.q_e = self.phys["q_e"]
        self.eV0 = self.phys["scales"]["eV0_J"] / self.q_e    #about 0.026
        self.conc0 = self.phys["scales"]["conc0"]

        self.dtable = self.dtable_eV /self.eV0
        self.mtab = max(1, int(self.energy_max_eV / self.dtable_eV + 0.5))

        self.analytic_ntlist = None
        self.analytic_ptlist = None
        self.analytic_tlist = None
        self.analytic_wcdf = None
        self.analytic_wsum = None
        self.analytic_nonempty_bins = None
        self.analytic_num_bins = 0
        self.analytic_bin_edges_eV = None
        self.analytic_bin_piecewise = None

        self.material = self.phys["material"]
        self.ek_file_override = f"bands_{self.material}.txt"

        self.valley_config = None
        self.axis_lookup_table = None
        self.kx_ticks_pi = None
        self.ky_ticks_pi = None
        self.kz_ticks_pi = None
        self.kx_tick_boundaries = None
        self.ky_tick_boundaries = None
        self.kz_tick_boundaries = None
        self.num_ticks_x = 0
        self.num_ticks_y = 0
        self.num_ticks_z = 0
        self.velocity_grid_real = None
        self.energy_grid_eV_map = None
        self.lookup_valid = None
        self.phonon_spectrum = None
        self.scattering_table = None
        self.scattering_rate = None
        self.hw_min_limit = None
        self.hw_max_limit = None
        self.phonon_hw_cdf = None

        self.Ni_norm = 0.0
        self.barrier_height_norm = 0.0
        self.beta_norm = 0.0
        self.difpr = {}

    def initialize(self, output_root: str | None = None) -> None:
        print("[Band] Building analytic band tables")
        if self.ek_data["k_norm"] is None:
            self.read_analytic_data(ek_file_override=self.ek_file_override) #√
        self.init_valley_configuration() #√
        self.init_axis_lookup_table() #O 判断kid用的
        self.build_analytic_lists() #√
        self.init_phonon_spectrum()
        self.build_analytic_scattering_table(output_root=output_root)
        self.init_derived_constants()
        nk = 0 if self.ek_data["energy_eV"] is None else int(len(self.ek_data["energy_eV"]))
        print(
            "[Band] Ready: "
            f"nk={nk}, "
            f"kgrid={self.num_ticks_x}x{self.num_ticks_y}x{self.num_ticks_z}, "
            f"bins={self.analytic_num_bins}"
        )

    def read_analytic_data(self, ek_file_override: str | None = None) -> None:
        # 1. DOS table 检查完毕
        dos_filename = f"DOS_{self.material}.txt"
        dos_path = os.path.join(self.input_path, dos_filename)
        if os.path.exists(dos_path):
            raw_dos = np.loadtxt(dos_path, skiprows=1)

            self.dos_norm = np.zeros(self.mtab + 1, dtype=float)

            #q_e = self.phys["q_e"]
            #eV0 = self.phys["scales"]["eV0_J"] / q_e
            #conc0 = self.phys["scales"]["conc0"]
            if raw_dos.shape[1] < 2:
                raise ValueError(f"[Band] DOS file {dos_path} must have at least 2 columns (E, DOS).")
            E_eV = raw_dos[:, 0]
            dos_real = raw_dos[:, 1]
            # DOS_norm = DOS_real[1/eV/m^3] * eV0[eV] / conc0[1/m^3]
            dos_norm_table = dos_real * self.eV0 / self.conc0
            E_norm = E_eV / self.eV0
            itab = ((E_norm - self.emin) / self.dtable + 0.5).astype(int) #dos的能量柱序号

            valid_mask = (itab >= 0) & (itab <= self.mtab)
            self.dos_norm[itab[valid_mask]] = dos_norm_table[valid_mask] #实际算出来的和dos_norm一样
        else:
            print(f"      [Warning] DOS file {dos_path} not found.")

        # 2. E-k-v table 检查完毕
        ek_filename = ek_file_override
        ek_path = os.path.join(self.input_path, ek_filename)
        if not os.path.exists(ek_path):
            raise FileNotFoundError(f"Critical: E-k band file not found at {ek_path}")

        data = np.loadtxt(ek_path, skiprows=1)

        kx_pi, ky_pi, kz_pi = data[:, 0], data[:, 1], data[:, 2]  #单位pi/a
        E_eV_in = data[:, 3]   #单位eV
        vx_si, vy_si, vz_si = data[:, 4], data[:, 5], data[:, 6]  #单位m/s

        self.ek_data["k_pi"] = np.column_stack((kx_pi, ky_pi, kz_pi))
        k_scale = np.pi / self.phys["sia0_norm"]
        self.ek_data["k_norm"] = np.column_stack((kx_pi, ky_pi, kz_pi)) * k_scale

        #q_e = self.phys["q_e"]
        #eV0 = self.phys["scales"]["eV0_J"] / q_e
        self.ek_data["energy_eV"] = E_eV_in

        #v_scale = 1.0 / self.phys["scales"]["velo0"]
        self.ek_data["velocity_real"] = np.column_stack((vx_si, vy_si, vz_si))

        wx = self._compute_axis_cell_widths(kx_pi)
        wy = self._compute_axis_cell_widths(ky_pi)
        wz = self._compute_axis_cell_widths(kz_pi)
        self.ek_data["weight"] = wx * wy * wz
        # 目前不需要
        valley_idx = np.full(len(E_eV_in), 2, dtype=int)
        #valley_idx[np.abs(kx_pi) > 1.5] = 0
        #valley_idx[np.abs(ky_pi) > 1.5] = 1
        self.ek_data["valley_idx"] = valley_idx

        #意义不大先注释
        #print(
        #    "      -> Grid step sets (pi/a): "
        #    f"dx={self._format_step_values(self._detect_grid_step_stats(kx_pi))}, "
        #    f"dy={self._format_step_values(self._detect_grid_step_stats(ky_pi))}, "
        #    f"dz={self._format_step_values(self._detect_grid_step_stats(kz_pi))}"
        #)

    def _detect_grid_step_stats(self, arr_pi: np.ndarray) -> np.ndarray:
        """
        Detect distinct grid steps on one axis.
        1) Sort unique ticks.
        2) Compute adjacent diffs.
        3) Keep unique diffs and merge numerically-close values.
        """
        ticks = np.unique(np.round(arr_pi.astype(float), decimals=12))
        if ticks.size <= 1:
            return np.array([1.0], dtype=float)

        ticks.sort()
        diffs = np.diff(ticks)
        valid_diffs = diffs[diffs > 1e-12]
        if valid_diffs.size == 0:
            return np.array([1.0], dtype=float)

        unique_diffs = np.unique(np.round(valid_diffs, decimals=12))
        unique_diffs.sort()

        merged = [float(unique_diffs[0])]
        for d in unique_diffs[1:]:
            if np.isclose(d, merged[-1], rtol=1e-6, atol=1e-12):
                continue
            merged.append(float(d))
        return np.asarray(merged, dtype=float)

    def _format_step_values(self, steps: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.6g}" for v in steps) + "]"

    def _compute_axis_cell_widths(self, arr_pi: np.ndarray) -> np.ndarray:
        """
        Per-point cell width for non-uniform 1D grids using midpoint boundaries.
        If axis has N sampled k-values, construct N+1 boundaries as:
        [k0, (k0+k1)/2, ..., (k_{N-2}+k_{N-1})/2, k_{N-1}],
        then widths are boundary differences (size N).
        """
        arr_round = np.round(arr_pi.astype(float), decimals=12)
        ticks, inv = np.unique(arr_round, return_inverse=True)
        ticks = np.sort(ticks)
        if ticks.size <= 1:
            return np.ones_like(arr_pi, dtype=float)

        boundaries = self._build_axis_boundaries(ticks)
        widths_per_tick = np.diff(boundaries)

        return widths_per_tick[inv]

    def _build_axis_boundaries(self, ticks: np.ndarray) -> np.ndarray:
        """Build midpoint boundaries from sorted ticks (units unchanged)."""
        ticks = np.asarray(ticks, dtype=float)
        if ticks.size <= 0:
            raise ValueError("[Error] Empty ticks for boundary construction.")
        boundaries = np.empty(ticks.size + 1, dtype=float)
        boundaries[0] = ticks[0]
        boundaries[-1] = ticks[-1]
        if ticks.size > 1:
            boundaries[1:-1] = 0.5 * (ticks[:-1] + ticks[1:])
        return boundaries

    #波谷定义，目前不太需要
    def init_valley_configuration(self) -> None:
        mat = self.phys["material"]
        if mat == "IGZO":
            self.valley_config = {"num_valleys": 1, "degeneracy": 1, "type": "GAMMA"}
        else:
            self.valley_config = {"num_valleys": 6, "degeneracy": 6, "type": "X"}
        return

    def init_axis_lookup_table(self) -> None:
        """
        Build axis lookup tables and a structured 3D velocity map in units of pi/a.
        """
        if self.ek_data["k_pi"] is None:
            raise ValueError("[Error] Critical: E-k data must be loaded before building lookup table.")

        raw_k = np.asarray(self.ek_data["k_pi"], dtype=float)
        raw_v = np.asarray(self.ek_data["velocity_real"], dtype=float)
        raw_e = np.asarray(self.ek_data["energy_eV"], dtype=float)

        kx_round = np.round(raw_k[:, 0], decimals=12)
        ky_round = np.round(raw_k[:, 1], decimals=12)
        kz_round = np.round(raw_k[:, 2], decimals=12)

        self.kx_ticks_pi = np.unique(kx_round)
        self.ky_ticks_pi = np.unique(ky_round)
        self.kz_ticks_pi = np.unique(kz_round)
        self.kx_ticks_pi.sort()
        self.ky_ticks_pi.sort()
        self.kz_ticks_pi.sort()

        self.num_ticks_x = int(self.kx_ticks_pi.size)
        self.num_ticks_y = int(self.ky_ticks_pi.size)
        self.num_ticks_z = int(self.kz_ticks_pi.size)
        if self.num_ticks_x == 0 or self.num_ticks_y == 0 or self.num_ticks_z == 0:
            raise ValueError("[Error] No K ticks detected from E-k data.")

        self.kx_tick_boundaries = self._build_axis_boundaries(self.kx_ticks_pi)
        self.ky_tick_boundaries = self._build_axis_boundaries(self.ky_ticks_pi)
        self.kz_tick_boundaries = self._build_axis_boundaries(self.kz_ticks_pi)

        ix = np.searchsorted(self.kx_ticks_pi, kx_round)
        iy = np.searchsorted(self.ky_ticks_pi, ky_round)
        iz = np.searchsorted(self.kz_ticks_pi, kz_round)

        grid_shape = (self.num_ticks_x, self.num_ticks_y, self.num_ticks_z)
        self.velocity_grid_real = np.zeros(grid_shape + (3,), dtype=float)
        self.energy_grid_eV_map = np.zeros(grid_shape, dtype=float)
        self.lookup_valid = np.zeros(grid_shape, dtype=bool)
        self.axis_lookup_table = np.full(grid_shape, -1, dtype=np.int64)

        self.velocity_table = np.zeros(self.num_ticks_x, dtype=float)
        vx_acc = np.zeros(self.num_ticks_x, dtype=float)
        vx_cnt = np.zeros(self.num_ticks_x, dtype=np.int64)

        for row in range(raw_k.shape[0]):
            ii = int(ix[row])
            jj = int(iy[row])
            kk = int(iz[row])
            self.velocity_grid_real[ii, jj, kk, :] = raw_v[row]
            self.energy_grid_eV_map[ii, jj, kk] = raw_e[row]
            self.lookup_valid[ii, jj, kk] = True
            self.axis_lookup_table[ii, jj, kk] = row
            vx_acc[ii] += raw_v[row, 0]
            vx_cnt[ii] += 1

        valid_x = vx_cnt > 0
        self.velocity_table[valid_x] = vx_acc[valid_x] / vx_cnt[valid_x]

        missing = int(np.size(self.lookup_valid) - np.count_nonzero(self.lookup_valid))
        if missing > 0:
            raise ValueError(
                f"[Error] Structured E-k lookup grid is incomplete: missing {missing} k-points."
            )


    def get_axis_indices_vectorized(self, k_values: np.ndarray, axis: str = "x") -> np.ndarray:
        """
        Vectorized lookup for k-axis indices.
        Input k_values must be in units of pi/a.
        """
        axis_key = axis.lower()
        if axis_key == "x":
            boundaries = self.kx_tick_boundaries
            n_ticks = self.num_ticks_x
        elif axis_key == "y":
            boundaries = self.ky_tick_boundaries
            n_ticks = self.num_ticks_y
        elif axis_key == "z":
            boundaries = self.kz_tick_boundaries
            n_ticks = self.num_ticks_z
        else:
            raise ValueError(f"[Error] Unsupported axis '{axis}'.")

        if boundaries is None or n_ticks <= 0:
            raise ValueError("[Error] Axis lookup boundaries are not initialized.")
        if not np.all(np.isfinite(k_values)):
            raise ValueError("[Error] Non-finite k_values detected.")
        idx = np.searchsorted(boundaries, k_values, side="right") - 1
        idx = np.clip(idx, 0, n_ticks - 1)
        return idx.astype(np.int32)

    def get_velocity_real_by_indices(
        self,
        kx_idx: int | np.ndarray,
        ky_idx: int | np.ndarray,
        kz_idx: int | np.ndarray,
    ) -> np.ndarray:
        if self.velocity_grid_real is None:
            raise ValueError("[Error] velocity_grid_real is not initialized.")

        ix = np.clip(np.asarray(kx_idx, dtype=np.int64), 0, self.num_ticks_x - 1)
        iy = np.clip(np.asarray(ky_idx, dtype=np.int64), 0, self.num_ticks_y - 1)
        iz = np.clip(np.asarray(kz_idx, dtype=np.int64), 0, self.num_ticks_z - 1)
        return self.velocity_grid_real[ix, iy, iz]

    def get_energy_eV_by_indices(
        self,
        kx_idx: int | np.ndarray,
        ky_idx: int | np.ndarray,
        kz_idx: int | np.ndarray,
    ) -> np.ndarray:
        if self.energy_grid_eV_map is None:
            raise ValueError("[Error] energy_grid_eV_map is not initialized.")

        ix = np.clip(np.asarray(kx_idx, dtype=np.int64), 0, self.num_ticks_x - 1)
        iy = np.clip(np.asarray(ky_idx, dtype=np.int64), 0, self.num_ticks_y - 1)
        iz = np.clip(np.asarray(kz_idx, dtype=np.int64), 0, self.num_ticks_z - 1)
        return self.energy_grid_eV_map[ix, iy, iz]

    def get_total_phonon_rate_real(self, energy_eV: float | np.ndarray) -> np.ndarray:
        """
        Interpolate total phonon scattering rate from the prebuilt E-rate table.

        Input energy is in eV. Output rate is in 1/s.
        """
        if self.scattering_rate is None or "total" not in self.scattering_rate:
            raise ValueError("[Error] Phonon scattering table is not initialized.")

        total_rate_norm = np.asarray(self.scattering_rate["total"], dtype=float)
        if total_rate_norm.size == 0:
            energy_arr = np.asarray(energy_eV, dtype=float)
            return np.zeros_like(energy_arr, dtype=float)

        time0 = float(self.phys["scales"]["time0"])
        if time0 <= 0.0:
            raise ValueError("[Error] Invalid time0 for rate conversion.")

        energy_grid_eV = (self.emin + np.arange(total_rate_norm.size) * self.dtable) * self.eV0
        energy_arr = np.asarray(energy_eV, dtype=float)
        energy_clipped = np.clip(energy_arr, energy_grid_eV[0], energy_grid_eV[-1])
        rate_norm = np.interp(energy_clipped, energy_grid_eV, total_rate_norm)
        return rate_norm / time0

    def build_analytic_lists(self, debug_output_path: str | None = None) -> None:
        """
        Build energy-bin lookup lists for fast E -> k-index mapping.
        """
        if self.ek_data["energy_eV"] is None:
            raise ValueError("[Error] E-k energy data is missing.")

        q_e = self.phys["q_e"]
        eV0 = self.phys["scales"]["eV0_J"] / q_e
        energies_eV = self.ek_data["energy_eV"]

        # Piecewise energy bins:
        #   [emin, esplit) -> de_low
        #   [esplit, emax] -> de_high
        e_min = float(self.phys.get("init_energy_bin_min_eV", 0.0))
        e_split = float(self.phys.get("init_energy_bin_split_eV", 0.05))
        de_low = float(self.phys.get("init_energy_bin_step_low_eV", 0.0001))
        de_high = float(self.phys.get("init_energy_bin_step_high_eV", 0.002))
        e_max = float(self.phys.get("init_energy_bin_max_eV", 8.0))
        if not (de_low > 0.0 and de_high > 0.0):
            raise ValueError("Energy bin steps must be > 0.")
        if not (e_min <= e_split < e_max):
            raise ValueError("Energy bin bounds must satisfy e_min <= e_split < e_max.")

        low_edges = np.arange(e_min, e_split + 0.5 * de_low, de_low, dtype=float)
        high_edges = np.arange(e_split + de_high, e_max + 0.5 * de_high, de_high, dtype=float)
        edges_eV = np.concatenate((low_edges, high_edges))
        if edges_eV[-1] < e_max - 1e-12:
            edges_eV = np.append(edges_eV, e_max)

        self.analytic_bin_edges_eV = edges_eV
        self.analytic_num_bins = int(edges_eV.size - 1)
        self.analytic_bin_piecewise = (
            e_min,
            e_split,
            de_low,
            de_high,
            int(low_edges.size - 1),
        )

        bin_indices = self.map_energy_to_bins(energies_eV)
        valid_mask = np.isfinite(energies_eV)
        valid_bins = bin_indices[valid_mask]

        self.analytic_ntlist = np.bincount(valid_bins, minlength=self.analytic_num_bins)
        self.analytic_ptlist = np.zeros(self.analytic_num_bins, dtype=np.int32)
        cumsum = np.cumsum(self.analytic_ntlist)
        if self.analytic_num_bins > 1:
            self.analytic_ptlist[1:] = cumsum[:-1]

        valid_indices = np.nonzero(valid_mask)[0]
        self.analytic_tlist = valid_indices[np.argsort(valid_bins, kind="stable")].astype(np.int32)

        # Build per-bin weighted CDF for k-state sampling.
        weights_all = self.ek_data.get("weight", None)  
        if weights_all is None:
            weights_all = np.ones_like(self.ek_data["energy_eV"], dtype=np.float64)
        else:
            weights_all = np.asarray(weights_all, dtype=np.float64)

        self.analytic_wcdf = np.zeros_like(self.analytic_tlist, dtype=np.float64)
        self.analytic_wsum = np.zeros(self.analytic_num_bins, dtype=np.float64)
        for ib in range(self.analytic_num_bins):
            count = int(self.analytic_ntlist[ib])
            if count <= 0:
                continue
            start = int(self.analytic_ptlist[ib])
            end = start + count
            idx_slice = self.analytic_tlist[start:end]
            w_slice = weights_all[idx_slice]
            w_slice = np.where(w_slice > 0.0, w_slice, 0.0)

            cdf = np.cumsum(w_slice)
            total_w = float(cdf[-1]) if cdf.size > 0 else 0.0
            if total_w <= 0.0:
                # Fallback to uniform sampling for this bin.
                cdf = np.arange(1, count + 1, dtype=np.float64)
                total_w = float(count)

            self.analytic_wcdf[start:end] = cdf
            self.analytic_wsum[ib] = total_w

        self.analytic_nonempty_bins = np.flatnonzero(self.analytic_ntlist > 0).astype(np.int32)

        if debug_output_path:
            self._write_debug_bins(debug_output_path, eV0)

    def _write_debug_bins(self, pathname: str, eV0: float) -> None:
        filename = os.path.join(pathname, "debug_analytic_bins.txt")
        print(f"      [Debug] Dumping bin stats to: {filename}")

        empty_bins = 0
        total_bins = self.analytic_num_bins
        with open(filename, "w", encoding="utf-8") as f:
            f.write("Bin_Index Energy_Min(eV) Energy_Max(eV) Count Start_Index\n")
            for i in range(total_bins):
                count = int(self.analytic_ntlist[i])
                if count == 0:
                    empty_bins += 1
                e_min_eV = self.analytic_bin_edges_eV[i]
                e_max_eV = self.analytic_bin_edges_eV[i + 1]
                start_idx = int(self.analytic_ptlist[i])
                f.write(f"{i} {e_min_eV:.5f} {e_max_eV:.5f} {count} {start_idx}\n")

        print(f"      [Debug] Total Bins: {total_bins}, Empty Bins: {empty_bins}")

    def map_energy_to_bins(self, energies_eV: np.ndarray) -> np.ndarray:
        """
        Map energies (eV) to analytic bin indices.
        Uses O(1) piecewise arithmetic mapping when piecewise settings exist.
        """
        if self.analytic_bin_edges_eV is None or self.analytic_num_bins <= 0:
            raise ValueError("[Error] Energy bin edges are not initialized.")

        energies = np.asarray(energies_eV, dtype=float)
        out = np.zeros(energies.shape, dtype=np.int32)
        finite = np.isfinite(energies)
        if not np.any(finite):
            return out

        e = energies[finite]
        if self.analytic_bin_piecewise is not None:
            e_min, e_split, de_low, de_high, n_low = self.analytic_bin_piecewise
            idx = np.empty(e.shape, dtype=np.int64)
            low_mask = e < e_split
            idx[low_mask] = np.floor((e[low_mask] - e_min) / de_low + 1e-12).astype(np.int64)
            idx[~low_mask] = n_low + np.floor((e[~low_mask] - e_split) / de_high + 1e-12).astype(np.int64)
            idx = np.clip(idx, 0, self.analytic_num_bins - 1)
        else:
            idx = np.searchsorted(self.analytic_bin_edges_eV, e, side="right") - 1
            idx = np.clip(idx, 0, self.analytic_num_bins - 1)

        out[finite] = idx.astype(np.int32)
        return out

    def init_phonon_spectrum(self) -> None:
        """
        Read phonon dispersion data (phonon_dispersion*.txt), parse metadata,
        and normalize omega/velocity tables.
        """
        mat_suffix = "_IGZO" if self.phys["material"] == "IGZO" else ""
        filename = f"phonon_dispersion{mat_suffix}.txt"
        file_path = os.path.join(self.input_path, filename)
        if not os.path.exists(file_path):
            alt_path = os.path.join("data", "phonon", filename)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                print(f"[Warning] Phonon file {filename} not found. Skipping phonon init.")
                return

        self.phonon = {
            "a0": 0.0,
            "qmax": 0.0,
            "dq": 0.0,
            "nq_tab": 0,
            "omega_table": None,
            "vg_table": None,
        }

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        if "a0=" in line:
                            val_str = line.split("a0=")[1].split()[0]
                            self.phonon["a0"] = float(val_str)
                        if "qmax=" in line:
                            val_str = line.split("qmax=")[1].split()[0]
                            self.phonon["qmax"] = float(val_str)
                    else:
                        break
        except Exception as exc:
            print(f"      [Warning] Error parsing metadata: {exc}")

        try:
            raw_data = np.loadtxt(file_path, comments="#")
        except ValueError:
            print("      [Error] Failed to load numerical data. Check file format.")
            return

        if raw_data.size == 0:
            print("      [Error] Phonon file is empty.")
            return

        q_vals = raw_data[:, 0]
        w_vals = raw_data[:, 1:5]
        v_vals = raw_data[:, 5:9]

        self.phonon["nq_tab"] = int(len(q_vals))
        max_q_found = float(np.max(q_vals))
        if self.phonon["qmax"] <= 0.0 and max_q_found > 0.0:
            self.phonon["qmax"] = max_q_found

        if self.phonon["nq_tab"] > 1:
            self.phonon["dq"] = self.phonon["qmax"] / (self.phonon["nq_tab"] - 1)

        # File stores angular frequency (rad/s). Convert to energy (eV) first.
        hbar_si = self.phys["hbar"]
        q_si = self.phys["q_e"]
        eV0 = self.phys["scales"]["eV0_J"] / q_si
        velo0 = self.phys["scales"]["velo0"]
        hw_eV = (hbar_si * w_vals) / q_si
        self.phonon["omega_table"] = hw_eV / eV0
        self.phonon["vg_table"] = v_vals / velo0

    def build_analytic_scattering_table(self, output_root: str | None = None) -> None:
        if self.phys["material"] != "IGZO":
            print("[Warning] Scattering table: only IGZO branch implemented.")
            return
        if self.phonon is None or self.phonon["omega_table"] is None:
            print("[Warning] Phonon spectrum not initialized. Skipping scattering table.")
            return

        kB = self.phys["kb"]
        hbar = self.phys["hbar"]
        q_e = self.phys["q_e"]
        m0 = self.phys["m0"]
        pi = np.pi

        eV0_J = self.phys["scales"]["eV0_J"]
        time0 = self.phys["scales"]["time0"]
        T_lattice = float(self.phys["Temperature"])

        rho = self.phys["sirho_real"]
        scat_cfg = self.phys.get("scattering_config", {}) or {}
        scat_flags = scat_cfg.get("flags", {}) or {}
        scat_models = scat_cfg.get("models", {}) or {}
        scat_params = scat_cfg.get("params", {}) or {}

        E_ac_eV = float(scat_params.get("acoustic_deformation_potential_eV", 5.0))
        D_LA = E_ac_eV * q_e
        D_TA = E_ac_eV * q_e
        Dopt_LO = float(scat_params.get("optical_deformation_potential_lo_eV_per_m", 5.0e5)) * q_e
        Dopt_TO = float(scat_params.get("optical_deformation_potential_to_eV_per_m", 5.0e5)) * q_e

        ml = float(self.phys["ml_val"]) * m0
        mt = float(self.phys["mt_val"]) * m0
        md_SI = (ml * mt * mt) ** (1.0 / 3.0)

        alpha_val = float(scat_params.get("nonparabolicity_eV_inv", 0.0))

        a0 = self.phonon["a0"]
        Rs = a0 * (3.0 / (16.0 * pi)) ** (1.0 / 3.0) if a0 > 0.0 else 0.0

        disorder_model = str(scat_models.get("disorder", "none")).strip().lower()
        if disorder_model in {"", "none"}:
            E_tail_eV = 0.0
            E_max_corr_eV = 0.0
        elif disorder_model == "linear_tail_enhancement":
            E_tail_eV = float(scat_params.get("disorder_tail_energy_eV", 0.18))
            E_max_corr_eV = float(scat_params.get("disorder_cutoff_energy_eV", 10.0))
        else:
            raise ValueError(f"[Error] Unsupported disorder_model: {disorder_model}")
        kBT_eV = (kB * T_lattice) / q_e

        num_energy_bins = self.mtab + 1
        energy_grid = self.emin + np.arange(num_energy_bins) * self.dtable

        eV0_eV = eV0_J / q_e
        E_max_table_eV = energy_grid[-1] * eV0_eV
        k_max_val = np.sqrt(2.0 * md_SI * E_max_table_eV * q_e) / hbar
        q_grid_limit = 2.0 * k_max_val
        nq_int = 2000
        dq_int = q_grid_limit / (nq_int - 1)
        q_grid = np.linspace(0.0, q_grid_limit, nq_int)

        omega_table = self.phonon["omega_table"].T.copy()
        vg_table = self.phonon["vg_table"].T.copy()

        hw_min_limit = np.zeros(5, dtype=float)
        hw_max_limit = np.zeros(5, dtype=float)
        for mech_idx, branch_idx in ((1, 2), (3, 3)):
            hw_eV = omega_table[branch_idx] * eV0_eV
            valid = hw_eV[hw_eV > 0.0]
            if valid.size == 0:
                hw_min = 0.05
                hw_max = 0.051
            else:
                hw_min = float(np.min(valid))
                hw_max = float(np.max(valid))
                if hw_max <= hw_min:
                    hw_max = hw_min + 0.001
            hw_min_limit[mech_idx] = hw_min
            hw_max_limit[mech_idx] = hw_max
            hw_min_limit[mech_idx + 1] = hw_min
            hw_max_limit[mech_idx + 1] = hw_max

        HW_BINS = 100

        dose, sumscatt, phonon_hw_cdf = _calc_igzo_scattering_kernel(
            num_energy_bins,
            energy_grid,
            q_grid,
            self.phonon["qmax"],
            dq_int,
            omega_table,
            vg_table,
            md_SI,
            T_lattice,
            rho,
            D_LA,
            D_TA,
            Dopt_LO,
            Dopt_TO,
            Rs,
            alpha_val,
            E_tail_eV,
            E_max_corr_eV,
            kBT_eV,
            HW_BINS,
            hbar,
            kB,
            q_e,
            pi,
            eV0_J,
            time0,
            hw_min_limit,
            hw_max_limit,
        )

        if not bool(scat_flags.get("acoustic", True)):
            dose[0, :] = 0.0
        if not bool(scat_flags.get("lo_abs", True)):
            dose[1, :] = 0.0
        if not bool(scat_flags.get("lo_ems", True)):
            dose[2, :] = 0.0
        if not bool(scat_flags.get("to_abs", True)):
            dose[3, :] = 0.0
        if not bool(scat_flags.get("to_ems", True)):
            dose[4, :] = 0.0
        sumscatt = np.sum(dose, axis=0)

        self.scattering_rate = {
            "total": sumscatt,
            "components": dose,
            "cdf": phonon_hw_cdf,
        }
        self.hw_min_limit = hw_min_limit
        self.hw_max_limit = hw_max_limit
        self.phonon_hw_cdf = phonon_hw_cdf

        if output_root:
            scatter_dir = os.path.join(output_root, "Scatter")
            os.makedirs(scatter_dir, exist_ok=True)
            out_file = os.path.join(scatter_dir, "scattering_rates.txt")

            eV0_eV = eV0_J / q_e
            with open(out_file, "w", encoding="utf-8") as handle:
                handle.write("# IGZO Analytical Scattering Rates\n")
                handle.write(f"# Temperature: {T_lattice:.2f} K\n")
                handle.write(f"# Scaling Time0: {time0:.4e} s\n")
                handle.write(
                    "# Columns: Energy(eV) Total(1/s) AC(1/s) "
                    "LO_Abs(1/s) LO_Em(1/s) TO_Abs(1/s) TO_Em(1/s)\n"
                )
                for idx in range(num_energy_bins):
                    energy_eV = energy_grid[idx] * eV0_eV
                    r_tot = sumscatt[idx] / time0
                    r_ac = dose[0, idx] / time0
                    r_lo_abs = dose[1, idx] / time0
                    r_lo_ems = dose[2, idx] / time0
                    r_to_abs = dose[3, idx] / time0
                    r_to_ems = dose[4, idx] / time0
                    handle.write(
                        f"{energy_eV:.6f} {r_tot:.6e} {r_ac:.6e} "
                        f"{r_lo_abs:.6e} {r_lo_ems:.6e} {r_to_abs:.6e} {r_to_ems:.6e}\n"
                    )
    def init_derived_constants(self) -> None:
        eV0_J = self.phys["scales"]["eV0_J"]
        spr0 = self.phys["scales"]["spr0"]
        conc0 = self.phys["scales"]["conc0"]
        field0 = self.phys["scales"]["pot0_V"] / spr0

        kB = self.phys["kb"]
        q_e = self.phys["q_e"]
        T0 = eV0_J / kB
        mat = self.phys["material"]
        sieg_norm = self.phys["sieg_norm"]

        if mat == "IGZO":
            self.barrier_height_norm = 3.23 / (eV0_J / q_e)
            self.beta_norm = (2.15e-5 / (eV0_J / q_e)) * np.sqrt(field0)
            self.difpr = {"PELEC": 0.20, "PHOLE": 0.35, "POXEL": 0.20}

            A_ref = 2.10e16
            ni_calc_cm3 = A_ref * (T0 ** 1.5) * np.exp(-sieg_norm / 2.0)
            ni_real_cm3 = ni_calc_cm3 * 0.12
            ni_real_m3 = ni_real_cm3 * 1.0e6
            self.Ni_norm = ni_real_m3 / conc0
        else:
            self.barrier_height_norm = 3.2 / (eV0_J / q_e)
            self.beta_norm = (2.15e-5 / (eV0_J / q_e)) * np.sqrt(field0)
            self.difpr = {"PELEC": 0.16, "PHOLE": 0.35, "POXEL": 0.16}

            A_ref = 3.87e16
            ni_calc_cm3 = A_ref * (T0 ** 1.5) * np.exp(-sieg_norm / 2.0)
            ni_real_cm3 = ni_calc_cm3 * 0.1534
            ni_real_m3 = ni_real_cm3 * 1.0e6
            self.Ni_norm = ni_real_m3 / conc0


try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None


if njit is not None:
    @njit(fastmath=True)
    def _calc_igzo_scattering_kernel(
        num_energy_bins,
        energy_grid,
        q_grid,
        qmax,
        dq_int,
        omega_table,
        vg_table,
        md_SI,
        T_lattice,
        rho,
        D_LA,
        D_TA,
        Dopt_LO,
        Dopt_TO,
        Rs,
        alpha_val,
        E_tail_eV,
        E_max_corr_eV,
        kBT_eV,
        HW_BINS,
        HBAR_SI,
        KB_SI,
        Q_SI,
        PI_SI,
        eV0_J,
        time0,
        hw_min_limit,
        hw_max_limit,
    ):
        dose = np.zeros((5, num_energy_bins))
        sumscatt = np.zeros(num_energy_bins)
        phonon_hw_cdf = np.zeros((5, num_energy_bins, HW_BINS))

        nq = q_grid.shape[0]
        n_ph = omega_table.shape[1]
        dq_ph = qmax / (n_ph - 1) if n_ph > 1 else 0.0

        for itab in range(num_energy_bins):
            E_norm = energy_grid[itab]
            E_eV = E_norm * (eV0_J / Q_SI)

            S_disorder = 1.0
            if kBT_eV > 0.0 and E_eV < E_max_corr_eV:
                delta_E = E_tail_eV * (1.0 - (E_eV / E_max_corr_eV))
                expo = delta_E / kBT_eV
                if expo > 700.0:
                    expo = 700.0
                S_disorder = np.exp(expo)

            term_k = E_eV * (1.0 + alpha_val * E_eV)
            if term_k < 0.0:
                term_k = 0.0
            ks = np.sqrt(2.0 * md_SI * term_k * Q_SI) / HBAR_SI

            if ks <= 1e-10 or dq_int <= 0.0 or n_ph < 2:
                continue

            sum_LA = 0.0
            sum_TA = 0.0
            sum_LO_abs = 0.0
            sum_LO_ems = 0.0
            sum_TO_abs = 0.0
            sum_TO_ems = 0.0

            hw_weights = np.zeros((5, HW_BINS))

            for iq in range(nq):
                q = q_grid[iq]
                if q < 1e-12:
                    continue

                q_mapped = q
                if qmax > 0.0:
                    q_period = 2.0 * qmax
                    q_mod = q % q_period
                    q_mapped = qmax - np.abs(q_mod - qmax)

                q_strength = q_mapped if q_mapped > 1e-12 else 1e-12

                idx_f = q_strength / dq_ph
                idx_i = int(idx_f)
                if idx_i >= n_ph - 1:
                    idx_i = n_ph - 2
                frac = idx_f - idx_i

                w_norm_LA = omega_table[0, idx_i] * (1.0 - frac) + omega_table[0, idx_i + 1] * frac
                w_norm_TA = omega_table[1, idx_i] * (1.0 - frac) + omega_table[1, idx_i + 1] * frac
                w_norm_LO = omega_table[2, idx_i] * (1.0 - frac) + omega_table[2, idx_i + 1] * frac
                w_norm_TO = omega_table[3, idx_i] * (1.0 - frac) + omega_table[3, idx_i + 1] * frac

                E_ph_LA = w_norm_LA * eV0_J
                E_ph_TA = w_norm_TA * eV0_J
                E_ph_LO = w_norm_LO * eV0_J
                E_ph_TO = w_norm_TO * eV0_J
                if E_ph_LA <= 0.0 or E_ph_TA <= 0.0 or E_ph_LO <= 0.0 or E_ph_TO <= 0.0:
                    continue

                w_LA_real = E_ph_LA / HBAR_SI
                w_TA_real = E_ph_TA / HBAR_SI
                w_LO_real = E_ph_LO / HBAR_SI
                w_TO_real = E_ph_TO / HBAR_SI

                N_LA = 1.0 / (np.exp(E_ph_LA / (KB_SI * T_lattice)) - 1.0)
                N_TA = 1.0 / (np.exp(E_ph_TA / (KB_SI * T_lattice)) - 1.0)
                N_LO = 1.0 / (np.exp(E_ph_LO / (KB_SI * T_lattice)) - 1.0)
                N_TO = 1.0 / (np.exp(E_ph_TO / (KB_SI * T_lattice)) - 1.0)

                Iq = 1.0
                if Rs > 0.0:
                    qr = q_strength * Rs
                    if qr > 1e-12:
                        Iq = 3.0 * (np.sin(qr) - qr * np.cos(qr)) / (qr * qr * qr)
                Iq2 = Iq * Iq

                base_term = Iq2 * (q_mapped * q_mapped * q_mapped)
                base_opt = Iq2 * (q_strength * q_strength)

                def check_q(E, ks, q, hw_eV, sign, alpha, md, Q_SI, HBAR_SI):
                    E_final = E + sign * hw_eV
                    if E_final < 0.0:
                        return False
                    term_kp = E_final * (1.0 + alpha * E_final)
                    kp = np.sqrt(2.0 * md * term_kp * Q_SI) / HBAR_SI
                    return (q >= np.abs(ks - kp)) and (q <= (ks + kp))

                hw_LA_eV = E_ph_LA / Q_SI
                hw_TA_eV = E_ph_TA / Q_SI

                if check_q(E_eV, ks, q, hw_LA_eV, 1.0, alpha_val, md_SI, Q_SI, HBAR_SI):
                    sum_LA += (1.0 / w_LA_real) * N_LA * base_term
                if check_q(E_eV, ks, q, hw_LA_eV, -1.0, alpha_val, md_SI, Q_SI, HBAR_SI):
                    if E_eV > hw_LA_eV:
                        sum_LA += (1.0 / w_LA_real) * (N_LA + 1.0) * base_term

                if check_q(E_eV, ks, q, hw_TA_eV, 1.0, alpha_val, md_SI, Q_SI, HBAR_SI):
                    sum_TA += (1.0 / w_TA_real) * N_TA * base_term
                if check_q(E_eV, ks, q, hw_TA_eV, -1.0, alpha_val, md_SI, Q_SI, HBAR_SI):
                    if E_eV > hw_TA_eV:
                        sum_TA += (1.0 / w_TA_real) * (N_TA + 1.0) * base_term

                if q <= 2.0 * ks:
                    hw_LO_eV = E_ph_LO / Q_SI
                    hw_TO_eV = E_ph_TO / Q_SI

                    lo_min = hw_min_limit[1]
                    lo_max = hw_max_limit[1]
                    to_min = hw_min_limit[3]
                    to_max = hw_max_limit[3]

                    inv_bw_lo = (HW_BINS - 1) / (lo_max - lo_min) if lo_max > lo_min else 0.0
                    inv_bw_to = (HW_BINS - 1) / (to_max - to_min) if to_max > to_min else 0.0

                    bin_lo = int((hw_LO_eV - lo_min) * inv_bw_lo + 0.5)
                    if bin_lo < 0:
                        bin_lo = 0
                    if bin_lo >= HW_BINS:
                        bin_lo = HW_BINS - 1

                    bin_to = int((hw_TO_eV - to_min) * inv_bw_to + 0.5)
                    if bin_to < 0:
                        bin_to = 0
                    if bin_to >= HW_BINS:
                        bin_to = HW_BINS - 1

                    w_abs_lo = (1.0 / w_LO_real) * N_LO * base_opt
                    sum_LO_abs += w_abs_lo
                    hw_weights[1, bin_lo] += w_abs_lo

                    if E_eV > hw_LO_eV:
                        w_em_lo = (1.0 / w_LO_real) * (N_LO + 1.0) * base_opt
                        sum_LO_ems += w_em_lo
                        hw_weights[2, bin_lo] += w_em_lo

                    w_abs_to = (1.0 / w_TO_real) * N_TO * base_opt
                    sum_TO_abs += w_abs_to
                    hw_weights[3, bin_to] += w_abs_to

                    if E_eV > hw_TO_eV:
                        w_em_to = (1.0 / w_TO_real) * (N_TO + 1.0) * base_opt
                        sum_TO_ems += w_em_to
                        hw_weights[4, bin_to] += w_em_to

            pre = md_SI / (4.0 * PI_SI * rho * HBAR_SI * HBAR_SI * ks)
            Rate_AC = pre * (D_LA * D_LA * sum_LA + D_TA * D_TA * sum_TA) * dq_int
            Rate_LO_abs = pre * (Dopt_LO * Dopt_LO) * sum_LO_abs * dq_int
            Rate_LO_ems = pre * (Dopt_LO * Dopt_LO) * sum_LO_ems * dq_int
            Rate_TO_abs = pre * (Dopt_TO * Dopt_TO) * sum_TO_abs * dq_int
            Rate_TO_ems = pre * (Dopt_TO * Dopt_TO) * sum_TO_ems * dq_int

            dose[0, itab] = Rate_AC * S_disorder * time0
            dose[1, itab] = Rate_LO_abs * S_disorder * time0
            dose[2, itab] = Rate_LO_ems * S_disorder * time0
            dose[3, itab] = Rate_TO_abs * S_disorder * time0
            dose[4, itab] = Rate_TO_ems * S_disorder * time0
            sumscatt[itab] = (
                dose[0, itab]
                + dose[1, itab]
                + dose[2, itab]
                + dose[3, itab]
                + dose[4, itab]
            )

            totals = np.array((0.0, sum_LO_abs, sum_LO_ems, sum_TO_abs, sum_TO_ems))
            for m in range(1, 5):
                tot = totals[m]
                if tot > 1e-60:
                    cum = 0.0
                    for b in range(HW_BINS):
                        cum += hw_weights[m, b]
                        phonon_hw_cdf[m, itab, b] = cum / tot
                    phonon_hw_cdf[m, itab, HW_BINS - 1] = 1.0
                else:
                    for b in range(HW_BINS):
                        phonon_hw_cdf[m, itab, b] = (b + 1) / HW_BINS

        return dose, sumscatt, phonon_hw_cdf
else:
    def _calc_igzo_scattering_kernel(*_args, **_kwargs):
        raise RuntimeError("Numba is required for scattering kernel but is not available.")
