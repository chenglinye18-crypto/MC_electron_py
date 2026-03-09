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

        self.dos_table = None
        self.ek_data = {
            "k": None,
            "energy": None,
            "velocity": None,
            "weight": None,
            "valley_idx": None,
        }
        self.emin = 0.0
        self.dtable_eV = float(self.phys["energy_step_eV"])
        self.energy_max_eV = float(self.phys["energy_max_eV"])
        q_e = self.phys["q_e"]
        eV0_eV = self.phys["scales"]["eV0_J"] / q_e
        if eV0_eV <= 0.0:
            eV0_eV = 1.0
        self.dtable = self.dtable_eV / eV0_eV
        self.mtab = max(1, int(self.energy_max_eV / self.dtable_eV + 0.5))
        self.mwle_ana = 4000
        self.dlist = 0.0

        self.analytic_ntlist = None
        self.analytic_ptlist = None
        self.analytic_tlist = None

        self.material = self.phys["material"]
        self.ek_file_override = f"bands_{self.material}.txt"

        self.valley_config = None
        self.axis_lookup_table = None
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

        print("[Band] Initializing Analytic Band Structure framework...")

    def initialize(self, output_root: str | None = None) -> None:
        if self.ek_data["k"] is None:
            self.read_analytic_data(ek_file_override=self.ek_file_override)
        self.init_valley_configuration()
        self.init_axis_lookup_table()
        self.build_analytic_lists()
        self.init_phonon_spectrum()
        self.build_analytic_scattering_table(output_root=output_root)
        self.init_derived_constants()

    def read_analytic_data(self, ek_file_override: str | None = None) -> None:
        # 1. DOS table
        dos_filename = f"analytic_dos{self.material}.txt"
        dos_path = os.path.join(self.input_path, dos_filename)
        if os.path.exists(dos_path):
            print(f"[Band] Reading DOS data from: {dos_path}")
            raw_dos = np.loadtxt(dos_path, skiprows=1)

            self.dos_table = np.zeros(self.mtab + 1, dtype=float)

            q_e = self.phys["q_e"]
            eV0 = self.phys["scales"]["eV0_J"] / q_e
            conc0 = self.phys["scales"]["conc0"]
            if raw_dos.shape[1] < 2:
                raise ValueError(f"[Band] DOS file {dos_path} must have at least 2 columns (E, DOS).")
            E_eV = raw_dos[:, 0]
            dos_real = raw_dos[:, 1]
            # DOS_norm = DOS_real[1/eV/m^3] * eV0[eV] / conc0[1/m^3]
            dos_norm = dos_real * eV0 / conc0
            E_norm = E_eV / eV0
            itab = ((E_norm - self.emin) / self.dtable + 0.5).astype(int) #dos的能量柱序号

            valid_mask = (itab >= 0) & (itab <= self.mtab)
            self.dos_table[itab[valid_mask]] = dos_norm[valid_mask] #实际算出来的和dos_norm一样
            print(f"      -> DOS table loaded. Max DOS: {np.max(self.dos_table / eV0 * conc0):.4e} eV^-1 m^-3")
        else:
            print(f"      [Warning] DOS file {dos_path} not found.")

        # 2. E-k-v table
        ek_filename = ek_file_override
        ek_path = os.path.join(self.input_path, ek_filename)
        if not os.path.exists(ek_path):
            raise FileNotFoundError(f"Critical: E-k band file not found at {ek_path}")

        print(f"[Band] Reading E-k data from: {ek_path}")
        data = np.loadtxt(ek_path, skiprows=1)

        kx_pi, ky_pi, kz_pi = data[:, 0], data[:, 1], data[:, 2]
        E_eV_in = data[:, 3]
        vx_si, vy_si, vz_si = data[:, 4], data[:, 5], data[:, 6]

        if "sia0_norm" in self.phys:
            k_scale = np.pi / self.phys["sia0_norm"]
        else:
            k_scale = 1.0
            print("      [Warning] sia0_norm missing, using raw k-vectors.")
        self.ek_data["k"] = np.column_stack((kx_pi, ky_pi, kz_pi)) * k_scale

        q_e = self.phys["q_e"]
        eV0 = self.phys["scales"]["eV0_J"] / q_e
        self.ek_data["energy"] = E_eV_in / eV0

        v_scale = 1.0 / self.phys["scales"]["velo0"]
        self.ek_data["velocity"] = np.column_stack((vx_si, vy_si, vz_si)) * v_scale

        step_x = self._detect_grid_step(kx_pi)
        step_y = self._detect_grid_step(ky_pi)
        step_z = self._detect_grid_step(kz_pi)
        weight_val = step_x * step_y * step_z
        self.ek_data["weight"] = np.full(len(E_eV_in), weight_val)

        valley_idx = np.full(len(E_eV_in), 2, dtype=int)
        valley_idx[np.abs(kx_pi) > 1.5] = 0
        valley_idx[np.abs(ky_pi) > 1.5] = 1
        self.ek_data["valley_idx"] = valley_idx

        print(f"      -> E-k table loaded. Points: {len(E_eV_in)}")
        print(
            f"      -> Grid step detected: dx={step_x:.3f}, dy={step_y:.3f}, dz={step_z:.3f} (pi/a units)"
        )

    def _detect_grid_step(self, arr_pi: np.ndarray) -> float:
        unique_vals = np.unique(np.abs(arr_pi))
        if len(unique_vals) > 1:
            diffs = np.diff(unique_vals)
            valid_diffs = diffs[diffs > 1e-6]
            if len(valid_diffs) > 0:
                return float(np.min(valid_diffs))
        return 1.0


    def init_valley_configuration(self) -> None:
        mat = self.phys["material"]
        if mat == "IGZO":
            self.valley_config = {"num_valleys": 1, "degeneracy": 1, "type": "GAMMA"}
        else:
            self.valley_config = {"num_valleys": 6, "degeneracy": 6, "type": "X"}
        print(f"      -> Valley Config: {self.valley_config['type']} "
              f"({self.valley_config['num_valleys']} valleys)")

    def init_axis_lookup_table(self) -> None:
        """
        Build full-range (-K to +K) axis lookup map and a signed velocity table.
        """
        if self.ek_data["k"] is None:
            raise ValueError("[Error] Critical: E-k data must be loaded before building lookup table.")

        print("[Band] Building Full-Range Axis Lookup Table & Velocity Map...")

        raw_kx = self.ek_data["k"][:, 0]
        raw_vx = self.ek_data["velocity"][:, 0]

        unique_abs_k = np.unique(np.round(np.abs(raw_kx), decimals=6))
        negative_ticks = -np.flip(unique_abs_k[unique_abs_k > 0])
        self.ticks = np.concatenate((negative_ticks, unique_abs_k))

        self.num_ticks = int(len(self.ticks))
        if self.num_ticks == 0:
            raise ValueError("[Error] No K ticks detected from E-k data.")

        print(f"      -> Generated {self.num_ticks} symmetric grid ticks.")
        # print(f"      -> Range: [{self.ticks[0]:.4f}, {self.ticks[-1]:.4f}] (pi/a units)") 这打印的是归一化的值，避免歧义去掉打印

        self.velocity_table = np.zeros(self.num_ticks)
        sort_indices = np.argsort(np.abs(raw_kx))
        sorted_abs_k = np.abs(raw_kx)[sort_indices]
        sorted_abs_v = np.abs(raw_vx)[sort_indices]

        abs_v_on_ticks = np.interp(np.abs(self.ticks), sorted_abs_k, sorted_abs_v)
        self.velocity_table = abs_v_on_ticks * np.sign(self.ticks)
        print("      -> Velocity table populated with signed values.")

        max_k = float(np.max(unique_abs_k))
        self.k_map_max = max_k * 1.01
        self.k_map_min = -self.k_map_max
        self.k_map_res = 0.001
        self.k_map_scale = 1.0 / self.k_map_res

        map_size = int((self.k_map_max - self.k_map_min) * self.k_map_scale) + 1
        k_query = self.k_map_min + np.arange(map_size) * self.k_map_res

        idx_candidates = np.searchsorted(self.ticks, k_query)
        idx_candidates = np.clip(idx_candidates, 1, self.num_ticks - 1)

        val_upper = self.ticks[idx_candidates]
        val_lower = self.ticks[idx_candidates - 1]

        mask_closer_to_lower = (np.abs(k_query - val_lower)) < (np.abs(val_upper - k_query))
        final_indices = np.where(mask_closer_to_lower, idx_candidates - 1, idx_candidates)

        self.axis_map = final_indices.astype(np.int32)
        print(f"      -> Lookup Map built. Size: {map_size}, Resolution: {self.k_map_res}")

    def get_axis_indices_vectorized(self, k_values: np.ndarray) -> np.ndarray:
        """
        Vectorized lookup for k-axis indices using precomputed axis_map.
        """
        if not hasattr(self, "axis_map"):
            raise ValueError("[Error] axis_map is not initialized.")
        if not np.all(np.isfinite(k_values)):
            raise ValueError("[Error] Non-finite k_values detected.")
        map_idx_float = (k_values - self.k_map_min) * self.k_map_scale
        map_idx_float = np.clip(map_idx_float, 0, self.axis_map.size - 1)
        map_idx = map_idx_float.astype(np.int32)
        return self.axis_map[map_idx]

    def build_analytic_lists(self, debug_output_path: str | None = None) -> None:
        """
        Build energy-bin lookup lists for fast E -> k-index mapping.
        """
        if self.ek_data["energy"] is None:
            raise ValueError("[Error] E-k energy data is missing.")

        print("[Band] Building Analytic Lists (Energy Binning)...")

        q_e = self.phys["q_e"]
        eV0 = self.phys["scales"]["eV0_J"] / q_e
        if self.dlist <= 0.0:
            target_res_eV = 0.002
            self.dlist = eV0 / target_res_eV
        print(f"      -> Energy bin resolution: 2 meV (dlist={self.dlist:.2f})")

        energies = self.ek_data["energy"]
        bin_indices = ((energies - self.emin) * self.dlist).astype(np.int32)

        max_idx = int(np.max(bin_indices)) if len(bin_indices) > 0 else 0
        if max_idx >= self.mwle_ana:
            print(f"      [Warning] Expanding MWLE_ana to {max_idx + 1}.")
            self.mwle_ana = max_idx + 1

        valid_mask = (bin_indices >= 0) & (bin_indices < self.mwle_ana)
        valid_bins = bin_indices[valid_mask]

        self.analytic_ntlist = np.bincount(valid_bins, minlength=self.mwle_ana)
        self.analytic_ptlist = np.zeros(self.mwle_ana, dtype=np.int32)
        cumsum = np.cumsum(self.analytic_ntlist)
        if self.mwle_ana > 1:
            self.analytic_ptlist[1:] = cumsum[:-1]

        valid_indices = np.nonzero(valid_mask)[0]
        self.analytic_tlist = valid_indices[np.argsort(valid_bins, kind="stable")].astype(np.int32)

        print(f"      -> Indexed {len(energies)} states into {self.mwle_ana} bins.")

        if debug_output_path:
            self._write_debug_bins(debug_output_path, eV0)

    def _write_debug_bins(self, pathname: str, eV0: float) -> None:
        filename = os.path.join(pathname, "debug_analytic_bins.txt")
        print(f"      [Debug] Dumping bin stats to: {filename}")

        empty_bins = 0
        total_bins = self.mwle_ana
        with open(filename, "w", encoding="utf-8") as f:
            f.write("Bin_Index Energy_Min(eV) Energy_Max(eV) Count Start_Index\n")
            for i in range(total_bins):
                count = int(self.analytic_ntlist[i])
                if count == 0:
                    empty_bins += 1
                e_min_eV = (i / self.dlist + self.emin) * eV0
                e_max_eV = ((i + 1) / self.dlist + self.emin) * eV0
                start_idx = int(self.analytic_ptlist[i])
                f.write(f"{i} {e_min_eV:.5f} {e_max_eV:.5f} {count} {start_idx}\n")

        print(f"      [Debug] Total Bins: {total_bins}, Empty Bins: {empty_bins}")

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

        print(f"[Band] Reading Phonon Spectrum from: {file_path}")

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

        print(f"      -> Loaded {self.phonon['nq_tab']} points.")
        print(f"      -> a0 = {self.phonon['a0']:.4e}, qmax = {self.phonon['qmax']:.4f}")
        print(f"      -> Data normalized by eV0={eV0:.4f} eV, velo0={velo0:.2e} m/s")

    def build_analytic_scattering_table(self, output_root: str | None = None) -> None:
        if self.phys["material"] != "IGZO":
            print("[Warning] Scattering table: only IGZO branch implemented.")
            return
        if self.phonon is None or self.phonon["omega_table"] is None:
            print("[Warning] Phonon spectrum not initialized. Skipping scattering table.")
            return

        print("[Band] Building Analytic Scattering Table (IGZO + Numba)...")

        kB = self.phys["kb"]
        hbar = self.phys["hbar"]
        q_e = self.phys["q_e"]
        m0 = self.phys["m0"]
        pi = np.pi

        eV0_J = self.phys["scales"]["eV0_J"]
        time0 = self.phys["scales"]["time0"]
        T_lattice = float(self.phys["Temperature"])

        rho = self.phys["sirho_real"]
        E_ac_eV = 5.0
        D_LA = E_ac_eV * q_e
        D_TA = E_ac_eV * q_e
        Dopt_LO = 5e5 * q_e
        Dopt_TO = 5e5 * q_e

        ml = float(self.phys["ml_val"]) * m0
        mt = float(self.phys["mt_val"]) * m0
        md_SI = (ml * mt * mt) ** (1.0 / 3.0)

        alpha_val = 0.0

        a0 = self.phonon["a0"]
        Rs = a0 * (3.0 / (16.0 * pi)) ** (1.0 / 3.0) if a0 > 0.0 else 0.0

        E_tail_eV = 0.18
        E_max_corr_eV = 10.0
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

        self.scattering_rate = {
            "total": sumscatt,
            "components": dose,
            "cdf": phonon_hw_cdf,
        }
        self.hw_min_limit = hw_min_limit
        self.hw_max_limit = hw_max_limit
        self.phonon_hw_cdf = phonon_hw_cdf

        print(f"      -> Scattering built. Max rate = {np.max(sumscatt):.4e}")

        if output_root:
            scatter_dir = os.path.join(output_root, "Scatter")
            os.makedirs(scatter_dir, exist_ok=True)
            out_file = os.path.join(scatter_dir, "scattering_rates.txt")
            print(f"      -> Writing scattering table to: {out_file}")

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
            print("      -> Scattering table export complete.")

    def init_derived_constants(self) -> None:
        print("      -> Calculating derived constants (Ni, Barrier)...")

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
            print(f"         [IGZO] Ni (Real): {ni_real_cm3:.2e} cm^-3")
        else:
            self.barrier_height_norm = 3.2 / (eV0_J / q_e)
            self.beta_norm = (2.15e-5 / (eV0_J / q_e)) * np.sqrt(field0)
            self.difpr = {"PELEC": 0.16, "PHOLE": 0.35, "POXEL": 0.16}

            A_ref = 3.87e16
            ni_calc_cm3 = A_ref * (T0 ** 1.5) * np.exp(-sieg_norm / 2.0)
            ni_real_cm3 = ni_calc_cm3 * 0.1534
            ni_real_m3 = ni_real_cm3 * 1.0e6
            self.Ni_norm = ni_real_m3 / conc0
            print(f"         [Si] Ni (Real): {ni_real_cm3:.2e} cm^-3")

        print(f"         -> Ni (Normalized): {self.Ni_norm:.4e}")


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
