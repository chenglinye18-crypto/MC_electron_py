"""
Helpers for phonon-scattering clocks and event handling.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None


PHONON_ACOUSTIC = 0
PHONON_LO_ABS = 1
PHONON_LO_EMS = 2
PHONON_TO_ABS = 3
PHONON_TO_EMS = 4


if njit is not None:
    @njit(fastmath=True, cache=True)
    def _searchsorted_left_kernel(arr, value):
        left = 0
        right = len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < value:
                left = mid + 1
            else:
                right = mid
        return left


    @njit(fastmath=True, cache=True)
    def _searchsorted_left_range_kernel(arr, start, count, value):
        left = 0
        right = count
        while left < right:
            mid = (left + right) // 2
            if arr[start + mid] < value:
                left = mid + 1
            else:
                right = mid
        return left


    @njit(fastmath=True, cache=True)
    def _searchsorted_right_kernel(arr, value):
        left = 0
        right = len(arr)
        while left < right:
            mid = (left + right) // 2
            if value < arr[mid]:
                right = mid
            else:
                left = mid + 1
        return left


    @njit(fastmath=True, cache=True)
    def _interp_1d_clipped_kernel(x, xp, fp):
        n = len(xp)
        if n == 0:
            return 0.0
        if n == 1:
            return float(fp[0])

        if x <= xp[0]:
            return float(fp[0])
        if x >= xp[n - 1]:
            return float(fp[n - 1])

        idx = _searchsorted_right_kernel(xp, x) - 1
        if idx < 0:
            idx = 0
        elif idx > n - 2:
            idx = n - 2

        x0 = xp[idx]
        x1 = xp[idx + 1]
        y0 = fp[idx]
        y1 = fp[idx + 1]
        if x1 <= x0:
            return float(y0)
        t = (x - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))


    @njit(fastmath=True, cache=True)
    def _map_energy_to_bin_kernel(energy_eV, analytic_num_bins, e_min, e_split, de_low, de_high, n_low):
        if energy_eV < e_split:
            idx = int(np.floor((energy_eV - e_min) / de_low + 1.0e-12))
        else:
            idx = int(n_low + np.floor((energy_eV - e_split) / de_high + 1.0e-12))
        if idx < 0:
            idx = 0
        elif idx > analytic_num_bins - 1:
            idx = analytic_num_bins - 1
        return idx


    @njit(fastmath=True, cache=True)
    def _nearest_nonempty_bin_kernel(bin_idx, ntlist, nonempty):
        if 0 <= bin_idx < ntlist.size and ntlist[bin_idx] > 0:
            return int(bin_idx)
        if nonempty.size == 0:
            return -1

        pos = _searchsorted_left_kernel(nonempty, bin_idx)
        left_idx = pos - 1
        if left_idx < 0:
            left_idx = 0
        right_idx = pos
        if right_idx > nonempty.size - 1:
            right_idx = nonempty.size - 1

        left_bin = int(nonempty[left_idx])
        right_bin = int(nonempty[right_idx])
        if abs(right_bin - bin_idx) < abs(bin_idx - left_bin):
            return right_bin
        return left_bin


    @njit(fastmath=True, cache=True)
    def _weighted_offset_kernel(count, start, total_w, wcdf, rand_weight, rand_uniform):
        if total_w > 0.0:
            target = rand_weight * total_w
            offset = _searchsorted_left_range_kernel(wcdf, start, count, target)
            if offset < 0:
                offset = 0
            elif offset > count - 1:
                offset = count - 1
            return int(offset)

        offset = int(rand_uniform * count)
        if offset < 0:
            offset = 0
        elif offset > count - 1:
            offset = count - 1
        return int(offset)


    @njit(fastmath=True, cache=True)
    def _axis_index_kernel(value_pi, boundaries, n_ticks):
        idx = _searchsorted_right_kernel(boundaries, value_pi) - 1
        if idx < 0:
            idx = 0
        elif idx > n_ticks - 1:
            idx = n_ticks - 1
        return int(idx)


    @njit(fastmath=True, cache=True)
    def _component_rates_kernel(energy_eV, energy_grid_eV, comp_norm, time0):
        rates = np.empty(5, dtype=np.float64)
        for mech in range(5):
            rate = _interp_1d_clipped_kernel(energy_eV, energy_grid_eV, comp_norm[mech]) / time0
            rates[mech] = rate if rate > 0.0 else 0.0
        return rates


    @njit(fastmath=True, cache=True)
    def _mechanism_index_kernel(rates, mech_pick):
        accum = 0.0
        mech_idx = 4
        for i in range(rates.size):
            accum += rates[i]
            if mech_pick <= accum:
                mech_idx = i
                break
        if mech_idx < 0:
            mech_idx = 0
        elif mech_idx > 4:
            mech_idx = 4
        return int(mech_idx)


    @njit(fastmath=True, cache=True)
    def _hw_from_cdf_kernel(cdf_row, rand_value, hw_min, hw_max):
        if cdf_row.size == 0:
            return 0.0
        b = _searchsorted_left_kernel(cdf_row, rand_value)
        if b < 0:
            b = 0
        elif b > cdf_row.size - 1:
            b = cdf_row.size - 1
        if hw_max <= hw_min:
            return hw_min if hw_min > 0.0 else 0.0
        frac = (b + 0.5) / cdf_row.size
        return float(hw_min + frac * (hw_max - hw_min))
else:
    def _searchsorted_left_kernel(arr, value):
        return int(np.searchsorted(arr, value, side="left"))


    def _searchsorted_left_range_kernel(arr, start, count, value):
        return int(np.searchsorted(arr[start:start + count], value, side="left"))


    def _searchsorted_right_kernel(arr, value):
        return int(np.searchsorted(arr, value, side="right"))


    def _interp_1d_clipped_kernel(x, xp, fp):
        if len(xp) == 0:
            return 0.0
        x_c = float(np.clip(x, xp[0], xp[-1]))
        return float(np.interp(x_c, xp, fp))


    def _map_energy_to_bin_kernel(energy_eV, analytic_num_bins, e_min, e_split, de_low, de_high, n_low):
        if energy_eV < e_split:
            idx = int(np.floor((energy_eV - e_min) / de_low + 1.0e-12))
        else:
            idx = int(n_low + np.floor((energy_eV - e_split) / de_high + 1.0e-12))
        return int(np.clip(idx, 0, analytic_num_bins - 1))


    def _nearest_nonempty_bin_kernel(bin_idx, ntlist, nonempty):
        if 0 <= bin_idx < ntlist.size and ntlist[bin_idx] > 0:
            return int(bin_idx)
        if nonempty.size == 0:
            return -1
        pos = int(np.searchsorted(nonempty, bin_idx, side="left"))
        left_idx = max(pos - 1, 0)
        right_idx = min(pos, nonempty.size - 1)
        left_bin = int(nonempty[left_idx])
        right_bin = int(nonempty[right_idx])
        if abs(right_bin - bin_idx) < abs(bin_idx - left_bin):
            return right_bin
        return left_bin


    def _weighted_offset_kernel(count, start, total_w, wcdf, rand_weight, rand_uniform):
        if total_w > 0.0:
            target = rand_weight * total_w
            offset = int(np.searchsorted(wcdf[start:start + count], target, side="left"))
            return int(np.clip(offset, 0, count - 1))
        return int(np.clip(int(rand_uniform * count), 0, count - 1))


    def _axis_index_kernel(value_pi, boundaries, n_ticks):
        idx = int(np.searchsorted(boundaries, value_pi, side="right") - 1)
        return int(np.clip(idx, 0, n_ticks - 1))


    def _component_rates_kernel(energy_eV, energy_grid_eV, comp_norm, time0):
        rates = np.empty(5, dtype=np.float64)
        for mech in range(5):
            rate = _interp_1d_clipped_kernel(energy_eV, energy_grid_eV, comp_norm[mech]) / time0
            rates[mech] = rate if rate > 0.0 else 0.0
        return rates


    def _mechanism_index_kernel(rates, mech_pick):
        mech_cdf = np.cumsum(rates)
        return int(np.clip(np.searchsorted(mech_cdf, mech_pick, side="left"), 0, 4))


    def _hw_from_cdf_kernel(cdf_row, rand_value, hw_min, hw_max):
        if cdf_row.size == 0:
            return 0.0
        b = int(np.searchsorted(cdf_row, rand_value, side="left"))
        b = int(np.clip(b, 0, cdf_row.size - 1))
        if hw_max <= hw_min:
            return max(float(hw_min), 0.0)
        frac = (b + 0.5) / cdf_row.size
        return float(hw_min + frac * (hw_max - hw_min))


def build_kcell_max_phsr_real(band_struct) -> np.ndarray:
    """
    Build a conservative per-k-cell upper-bound table for phonon scattering.

    The user suggested using the cell vertices. To guarantee the bound never
    drops below the current sampled-state rate, this implementation uses the
    local 3x3x3 neighborhood maximum, which is slightly more conservative.
    """
    if band_struct is None or getattr(band_struct, "energy_grid_eV_map", None) is None:
        raise ValueError("Band structure energy grid is not initialized.")

    energy_grid_eV = np.asarray(band_struct.energy_grid_eV_map, dtype=float)
    state_rates_real = np.asarray(band_struct.get_total_phonon_rate_real(energy_grid_eV), dtype=float)
    nx, ny, nz = state_rates_real.shape
    out = np.zeros_like(state_rates_real, dtype=float)

    for ix in range(nx):
        x0 = max(ix - 1, 0)
        x1 = min(ix + 2, nx)
        for iy in range(ny):
            y0 = max(iy - 1, 0)
            y1 = min(iy + 2, ny)
            for iz in range(nz):
                z0 = max(iz - 1, 0)
                z1 = min(iz + 2, nz)
                out[ix, iy, iz] = float(np.max(state_rates_real[x0:x1, y0:y1, z0:z1]))

    return out


def _scattering_energy_grid_eV(band_struct) -> np.ndarray:
    cached = getattr(band_struct, "_scattering_energy_grid_eV_cache", None)
    total = np.asarray(band_struct.scattering_rate["total"], dtype=float)
    if cached is not None and cached.shape[0] == total.size:
        return cached

    energy_grid = (band_struct.emin + np.arange(total.size, dtype=float) * band_struct.dtable) * band_struct.eV0
    band_struct._scattering_energy_grid_eV_cache = np.asarray(energy_grid, dtype=np.float64)
    return band_struct._scattering_energy_grid_eV_cache


def _interpolate_total_rate_real(band_struct, energy_eV: float) -> float:
    if band_struct.scattering_rate is None or "total" not in band_struct.scattering_rate:
        return 0.0

    total_norm = np.asarray(band_struct.scattering_rate["total"], dtype=np.float64)
    if total_norm.size == 0:
        return 0.0

    time0 = float(band_struct.phys["scales"]["time0"])
    if time0 <= 0.0:
        return 0.0

    energy_grid_eV = np.asarray(_scattering_energy_grid_eV(band_struct), dtype=np.float64)
    rate = _interp_1d_clipped_kernel(float(energy_eV), energy_grid_eV, total_norm) / time0
    return float(rate) if rate > 0.0 else 0.0


def _interpolate_component_rates_real(band_struct, energy_eV: float) -> np.ndarray:
    if band_struct.scattering_rate is None or "components" not in band_struct.scattering_rate:
        return np.zeros(5, dtype=float)

    comp_norm = np.asarray(band_struct.scattering_rate["components"], dtype=np.float64)
    if comp_norm.ndim != 2 or comp_norm.shape[0] < 5:
        return np.zeros(5, dtype=float)

    time0 = float(band_struct.phys["scales"]["time0"])
    if time0 <= 0.0:
        return np.zeros(5, dtype=float)

    energy_grid_eV = np.asarray(_scattering_energy_grid_eV(band_struct), dtype=np.float64)
    return _component_rates_kernel(float(energy_eV), energy_grid_eV, comp_norm, float(time0))


def _sample_hw_eV(band_struct, mech_idx: int, energy_eV: float) -> float:
    cdf_table = np.asarray(band_struct.scattering_rate["cdf"], dtype=np.float64)
    if cdf_table.ndim != 3:
        return 0.0

    itab = int(
        np.clip(
            np.round((float(energy_eV) / band_struct.eV0 - band_struct.emin) / band_struct.dtable),
            0,
            cdf_table.shape[1] - 1,
        )
    )
    cdf_row = cdf_table[mech_idx, itab]
    if cdf_row.size == 0:
        return 0.0

    r = float(np.random.random())

    hw_min = float(band_struct.hw_min_limit[mech_idx])
    hw_max = float(band_struct.hw_max_limit[mech_idx])
    return float(_hw_from_cdf_kernel(cdf_row, r, hw_min, hw_max))


def _nearest_nonempty_bin(band_struct, bin_idx: int) -> int:
    ntlist = band_struct.analytic_ntlist
    if ntlist is None:
        raise ValueError("Analytic bin counts are not initialized.")
    nonempty = band_struct.analytic_nonempty_bins
    if nonempty is None or nonempty.size == 0:
        raise ValueError("No non-empty analytic bins are available for phonon scattering.")
    out = _nearest_nonempty_bin_kernel(int(bin_idx), ntlist, nonempty)
    if out < 0:
        raise ValueError("No non-empty analytic bins are available for phonon scattering.")
    return int(out)


def sample_k_state_from_energy(band_struct, energy_eV: float) -> tuple[float, float, float, float, int, int, int]:
    """
    Sample a new k-state from the analytic energy-bin lookup using the existing
    weighted in-bin CDF.
    """
    if band_struct.analytic_bin_piecewise is None:
        raise ValueError("Analytic piecewise bin definition is missing.")
    e_min, e_split, de_low, de_high, n_low = band_struct.analytic_bin_piecewise
    bin_idx = _map_energy_to_bin_kernel(
        float(energy_eV),
        int(band_struct.analytic_num_bins),
        float(e_min),
        float(e_split),
        float(de_low),
        float(de_high),
        int(n_low),
    )
    bin_idx = _nearest_nonempty_bin(band_struct, bin_idx)

    count = int(band_struct.analytic_ntlist[bin_idx])
    start = int(band_struct.analytic_ptlist[bin_idx])
    if count <= 0:
        raise ValueError("Selected phonon-scatter energy bin is empty.")

    total_w = float(band_struct.analytic_wsum[bin_idx])
    offset = _weighted_offset_kernel(
        int(count),
        int(start),
        float(total_w),
        np.asarray(band_struct.analytic_wcdf, dtype=np.float64),
        float(np.random.random()),
        float(np.random.random()),
    )

    ek_idx = int(np.asarray(band_struct.analytic_tlist, dtype=np.int32)[start + offset])
    k_norm = np.asarray(band_struct.ek_data["k_norm"][ek_idx], dtype=np.float64)
    k_pi = np.asarray(band_struct.ek_data["k_pi"][ek_idx], dtype=np.float64)
    energy_out = float(band_struct.ek_data["energy_eV"][ek_idx])

    kx_idx = _axis_index_kernel(float(k_pi[0]), np.asarray(band_struct.kx_tick_boundaries, dtype=np.float64), int(band_struct.num_ticks_x))
    ky_idx = _axis_index_kernel(float(k_pi[1]), np.asarray(band_struct.ky_tick_boundaries, dtype=np.float64), int(band_struct.num_ticks_y))
    kz_idx = _axis_index_kernel(float(k_pi[2]), np.asarray(band_struct.kz_tick_boundaries, dtype=np.float64), int(band_struct.num_ticks_z))

    return (
        float(k_norm[0]),
        float(k_norm[1]),
        float(k_norm[2]),
        energy_out,
        kx_idx,
        ky_idx,
        kz_idx,
    )


def handle_phonon_scatter_event(
    simulation,
    energy_eV: float,
    kx_idx: int,
    ky_idx: int,
    kz_idx: int,
) -> tuple[float, float, float, float, int, int, int] | None:
    """
    Resolve EVENT_PHONON_SCATTER using a k-cell upper-bound clock plus
    rejection-based self-scattering.

    Standard logic is used:
      X = Random() * Gamma_max
      X <= Gamma_real -> real scatter
      X >  Gamma_real -> self scatter
    """
    band_struct = simulation.band_struct
    if band_struct is None or simulation.kcell_max_phsr_real is None:
        return None

    gamma_max = float(simulation.kcell_max_phsr_real[kx_idx, ky_idx, kz_idx])
    if gamma_max <= 1.0e-30 or not np.isfinite(gamma_max):
        return None

    gamma_real = _interpolate_total_rate_real(band_struct, energy_eV)
    if gamma_real <= 1.0e-30 or not np.isfinite(gamma_real):
        return None

    x_rand = float(np.random.random()) * gamma_max
    if x_rand > gamma_real:
        simulation.phonon_scatter_stats["self"] += 1
        return None

    mech_rates = _interpolate_component_rates_real(band_struct, energy_eV)
    rate_sum = float(np.sum(mech_rates))
    if rate_sum <= 1.0e-30:
        simulation.phonon_scatter_stats["self"] += 1
        return None

    mech_pick = float(np.random.random()) * rate_sum
    mech_idx = _mechanism_index_kernel(np.asarray(mech_rates, dtype=np.float64), mech_pick)

    if mech_idx == PHONON_ACOUSTIC:
        new_state = sample_k_state_from_energy(band_struct, energy_eV)
        simulation.phonon_scatter_stats["acoustic"] += 1
        simulation.step_scatter_stats["phonon"] += 1
        simulation.total_scatter_stats["phonon"] += 1
    else:
        hw_eV = _sample_hw_eV(band_struct, mech_idx, energy_eV)
        if mech_idx in (PHONON_LO_ABS, PHONON_TO_ABS):
            new_energy = energy_eV + hw_eV
            simulation.phonon_scatter_stats["absorbed_eV"] += hw_eV
            simulation.phonon_scatter_stats["optical_abs"] += 1
            simulation.step_scatter_stats["phonon"] += 1
            simulation.total_scatter_stats["phonon"] += 1
        else:
            new_energy = energy_eV - hw_eV
            if new_energy <= 0.0:
                simulation.phonon_scatter_stats["self"] += 1
                return None
            simulation.phonon_scatter_stats["emitted_eV"] += hw_eV
            simulation.phonon_scatter_stats["optical_ems"] += 1
            simulation.step_scatter_stats["phonon"] += 1
            simulation.total_scatter_stats["phonon"] += 1

        new_state = sample_k_state_from_energy(band_struct, new_energy)

    return new_state
