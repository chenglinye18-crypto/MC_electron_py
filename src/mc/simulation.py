"""
Runtime simulation workflow for 3D Monte Carlo.
"""
from __future__ import annotations

import os
import time

import numpy as np

from Particle import Particle
from Poisson import PoissonSolver
from .impurity_scattering import compute_impurity_scatter_time, handle_impurity_scatter_event
from .phonon_scattering import (
    build_kcell_max_phsr_real,
    handle_phonon_scatter_event,
    _interp_1d_clipped_kernel,
    _component_rates_kernel,
    _mechanism_index_kernel,
    _hw_from_cdf_kernel,
    _map_energy_to_bin_kernel,
    _nearest_nonempty_bin_kernel,
    _weighted_offset_kernel,
    _axis_index_kernel,
)
from .surface_scattering import compute_surface_scatter_time, handle_surface_scatter_event

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional dependency
    njit = None
    prange = range

EVENT_HIT_KGRID = 1
EVENT_HIT_CELL = 2
EVENT_PHONON_SCATTER = 3
EVENT_IMPURITY_SCATTER = 4
EVENT_SURFACE_SCATTER = 5
EVENT_TIME_STEP_END = 6

RULE_PASS = 0
RULE_CATCH = 1
RULE_GENERATE = 2
RULE_REFLECT = 3
RULE_SCATTOX = 4

FLY_STATUS_DONE = 0
FLY_STATUS_CATCH = 1
FLY_STATUS_GENERATE = 2
FLY_STATUS_ERROR_NON_MC = 3


if njit is not None:
    @njit(fastmath=True, cache=True)
    def _accumulate_monitor_crossing_kernel(
        x,
        y,
        z,
        hit_dir,
        charge_c,
        monitor_bounds_m,
        monitor_faces,
        monitor_charge_c_sum,
        monitor_crossing_count,
    ):
        tol = 1.0e-15
        for idx in range(monitor_faces.size):
            if int(monitor_faces[idx]) != int(hit_dir):
                continue
            b = monitor_bounds_m[idx]
            if (
                (b[0] - tol) <= x <= (b[1] + tol)
                and (b[2] - tol) <= y <= (b[3] + tol)
                and (b[4] - tol) <= z <= (b[5] + tol)
            ):
                monitor_charge_c_sum[idx] += charge_c
                monitor_crossing_count[idx] += 1


    @njit(fastmath=True, cache=True)
    def _select_next_event_kernel(tet_tf, cell_tf, ph_tf, imp_tf, surf_tf, left_time):
        tf = tet_tf
        event_flag = EVENT_HIT_KGRID
        if cell_tf < tf:
            tf = cell_tf
            event_flag = EVENT_HIT_CELL
        if ph_tf < tf:
            tf = ph_tf
            event_flag = EVENT_PHONON_SCATTER
        if imp_tf < tf:
            tf = imp_tf
            event_flag = EVENT_IMPURITY_SCATTER
        if surf_tf < tf:
            tf = surf_tf
            event_flag = EVENT_SURFACE_SCATTER
        if left_time < tf:
            tf = left_time
            event_flag = EVENT_TIME_STEP_END
        return float(tf), int(event_flag)


    @njit(fastmath=True, cache=True)
    def _compute_cell_time_kernel(x, y, z, vx, vy, vz, x0, x1, y0, y1, z0, z1, time0):
        if abs(vx) <= 1.0e-30:
            tx = np.inf
            dir_x = -1
        else:
            if vx > 0.0:
                delta_l = x1 - x
                dir_x = 0
            else:
                delta_l = x - x0
                dir_x = 1
            if delta_l < 0.0 and abs(delta_l) < 1.0e-18:
                delta_l = 0.0
            if delta_l < 0.0:
                tx = np.inf
                dir_x = -1
            else:
                tx = delta_l / abs(vx)

        if abs(vy) <= 1.0e-30:
            ty = np.inf
            dir_y = -1
        else:
            if vy > 0.0:
                delta_l = y1 - y
                dir_y = 2
            else:
                delta_l = y - y0
                dir_y = 3
            if delta_l < 0.0 and abs(delta_l) < 1.0e-18:
                delta_l = 0.0
            if delta_l < 0.0:
                ty = np.inf
                dir_y = -1
            else:
                ty = delta_l / abs(vy)

        if abs(vz) <= 1.0e-30:
            tz = np.inf
            dir_z = -1
        else:
            if vz > 0.0:
                delta_l = z1 - z
                dir_z = 4
            else:
                delta_l = z - z0
                dir_z = 5
            if delta_l < 0.0 and abs(delta_l) < 1.0e-18:
                delta_l = 0.0
            if delta_l < 0.0:
                tz = np.inf
                dir_z = -1
            else:
                tz = delta_l / abs(vz)

        time_real = tx
        hit_dir = dir_x
        if ty < time_real:
            time_real = ty
            hit_dir = dir_y
        if tz < time_real:
            time_real = tz
            hit_dir = dir_z

        if np.isfinite(time_real):
            time_norm = time_real / time0
        else:
            time_norm = np.inf
        return float(time_real), int(hit_dir), float(time_norm)


    @njit(fastmath=True, cache=True)
    def _compute_kgrid_time_kernel(
        kx_real,
        ky_real,
        kz_real,
        dkx_dt_real,
        dky_dt_real,
        dkz_dt_real,
        kx_idx,
        ky_idx,
        kz_idx,
        kx_boundaries,
        ky_boundaries,
        kz_boundaries,
        boundary_real_scale,
        time0,
    ):
        nx = len(kx_boundaries) - 1
        ny = len(ky_boundaries) - 1
        nz = len(kz_boundaries) - 1

        if abs(dkx_dt_real) <= 1.0e-30 or nx <= 1:
            tx = np.inf
            dir_x = -1
        else:
            idx = kx_idx
            if idx < 0:
                idx = 0
            elif idx > nx - 2:
                idx = nx - 2
            if dkx_dt_real > 0.0:
                k_wall_real = kx_boundaries[idx + 1] * boundary_real_scale
                delta_k = k_wall_real - kx_real
                dir_x = 0
            else:
                k_wall_real = kx_boundaries[idx] * boundary_real_scale
                delta_k = kx_real - k_wall_real
                dir_x = 1
            if delta_k < 0.0 and abs(delta_k) < 1.0e-18:
                delta_k = 0.0
            if delta_k < 0.0:
                tx = np.inf
                dir_x = -1
            else:
                tx = delta_k / abs(dkx_dt_real)

        if abs(dky_dt_real) <= 1.0e-30 or ny <= 1:
            ty = np.inf
            dir_y = -1
        else:
            idx = ky_idx
            if idx < 0:
                idx = 0
            elif idx > ny - 2:
                idx = ny - 2
            if dky_dt_real > 0.0:
                k_wall_real = ky_boundaries[idx + 1] * boundary_real_scale
                delta_k = k_wall_real - ky_real
                dir_y = 2
            else:
                k_wall_real = ky_boundaries[idx] * boundary_real_scale
                delta_k = ky_real - k_wall_real
                dir_y = 3
            if delta_k < 0.0 and abs(delta_k) < 1.0e-18:
                delta_k = 0.0
            if delta_k < 0.0:
                ty = np.inf
                dir_y = -1
            else:
                ty = delta_k / abs(dky_dt_real)

        if abs(dkz_dt_real) <= 1.0e-30 or nz <= 1:
            tz = np.inf
            dir_z = -1
        else:
            idx = kz_idx
            if idx < 0:
                idx = 0
            elif idx > nz - 2:
                idx = nz - 2
            if dkz_dt_real > 0.0:
                k_wall_real = kz_boundaries[idx + 1] * boundary_real_scale
                delta_k = k_wall_real - kz_real
                dir_z = 4
            else:
                k_wall_real = kz_boundaries[idx] * boundary_real_scale
                delta_k = kz_real - k_wall_real
                dir_z = 5
            if delta_k < 0.0 and abs(delta_k) < 1.0e-18:
                delta_k = 0.0
            if delta_k < 0.0:
                tz = np.inf
                dir_z = -1
            else:
                tz = delta_k / abs(dkz_dt_real)

        time_real = tx
        hit_dir = dir_x
        if ty < time_real:
            time_real = ty
            hit_dir = dir_y
        if tz < time_real:
            time_real = tz
            hit_dir = dir_z

        if np.isfinite(time_real):
            time_norm = time_real / time0
        else:
            time_norm = np.inf
        return float(time_real), int(hit_dir), float(time_norm)


    @njit(fastmath=True, cache=True)
    def _advance_particle_drift_kernel(
        x,
        y,
        z,
        kx,
        ky,
        kz,
        energy,
        vx,
        vy,
        vz,
        ex,
        ey,
        ez,
        tf,
        phrnl,
        imprnl,
        ssnl,
        ph_gamma_real,
        imp_gamma_real,
        surf_gamma_real,
        charge_sign,
        hbar,
        q_e,
        spr0,
    ):
        dkx_dt_real = charge_sign * q_e * ex / hbar
        dky_dt_real = charge_sign * q_e * ey / hbar
        dkz_dt_real = charge_sign * q_e * ez / hbar
        return (
            float(x + vx * tf),
            float(y + vy * tf),
            float(z + vz * tf),
            float(kx + dkx_dt_real * tf * spr0),
            float(ky + dky_dt_real * tf * spr0),
            float(kz + dkz_dt_real * tf * spr0),
            float(energy + charge_sign * (vx * ex + vy * ey + vz * ez) * tf),
            float(max(0.0, phrnl - ph_gamma_real * tf)),
            float(max(0.0, imprnl - imp_gamma_real * tf)),
            float(max(0.0, ssnl - surf_gamma_real * tf)),
        )
else:
    def _select_next_event_kernel(tet_tf, cell_tf, ph_tf, imp_tf, surf_tf, left_time):
        candidates = (
            (float(tet_tf), EVENT_HIT_KGRID),
            (float(cell_tf), EVENT_HIT_CELL),
            (float(ph_tf), EVENT_PHONON_SCATTER),
            (float(imp_tf), EVENT_IMPURITY_SCATTER),
            (float(surf_tf), EVENT_SURFACE_SCATTER),
            (float(left_time), EVENT_TIME_STEP_END),
        )
        tf, event_flag = min(candidates, key=lambda item: item[0])
        return float(tf), int(event_flag)


    def _compute_cell_time_kernel(x, y, z, vx, vy, vz, x0, x1, y0, y1, z0, z1, time0):
        def axis_hit_time(pos, vel, lower, upper, dir_pos, dir_neg):
            if abs(vel) <= 1.0e-30:
                return np.inf, -1
            if vel > 0.0:
                delta_l = upper - pos
                hit_dir = dir_pos
            else:
                delta_l = pos - lower
                hit_dir = dir_neg
            if delta_l < 0.0 and abs(delta_l) < 1.0e-18:
                delta_l = 0.0
            if delta_l < 0.0:
                return np.inf, -1
            return float(delta_l / abs(vel)), hit_dir

        tx, dir_x = axis_hit_time(x, vx, x0, x1, 0, 1)
        ty, dir_y = axis_hit_time(y, vy, y0, y1, 2, 3)
        tz, dir_z = axis_hit_time(z, vz, z0, z1, 4, 5)
        candidates = ((tx, dir_x), (ty, dir_y), (tz, dir_z))
        time_real, hit_dir = min(candidates, key=lambda item: item[0])
        time_norm = float(time_real / time0) if np.isfinite(time_real) else np.inf
        return float(time_real), int(hit_dir), float(time_norm)


    def _compute_kgrid_time_kernel(
        kx_real,
        ky_real,
        kz_real,
        dkx_dt_real,
        dky_dt_real,
        dkz_dt_real,
        kx_idx,
        ky_idx,
        kz_idx,
        kx_boundaries,
        ky_boundaries,
        kz_boundaries,
        boundary_real_scale,
        time0,
    ):
        def axis_hit_time(k_real, k_idx, dk_dt_real, boundaries_pi, dir_pos, dir_neg):
            if abs(dk_dt_real) <= 1.0e-30:
                return np.inf, -1
            idx = int(np.clip(k_idx, 0, len(boundaries_pi) - 2))
            if dk_dt_real > 0.0:
                k_wall_real = float(boundaries_pi[idx + 1]) * boundary_real_scale
                delta_k = k_wall_real - k_real
                hit_dir = dir_pos
            else:
                k_wall_real = float(boundaries_pi[idx]) * boundary_real_scale
                delta_k = k_real - k_wall_real
                hit_dir = dir_neg
            if delta_k < 0.0 and abs(delta_k) < 1.0e-18:
                delta_k = 0.0
            if delta_k < 0.0:
                return np.inf, -1
            return float(delta_k / abs(dk_dt_real)), hit_dir

        tx, dir_x = axis_hit_time(kx_real, kx_idx, dkx_dt_real, kx_boundaries, 0, 1)
        ty, dir_y = axis_hit_time(ky_real, ky_idx, dky_dt_real, ky_boundaries, 2, 3)
        tz, dir_z = axis_hit_time(kz_real, kz_idx, dkz_dt_real, kz_boundaries, 4, 5)
        candidates = ((tx, dir_x), (ty, dir_y), (tz, dir_z))
        time_real, hit_dir = min(candidates, key=lambda item: item[0])
        time_norm = float(time_real / time0) if np.isfinite(time_real) else np.inf
        return float(time_real), int(hit_dir), float(time_norm)


    def _advance_particle_drift_kernel(
        x,
        y,
        z,
        kx,
        ky,
        kz,
        energy,
        vx,
        vy,
        vz,
        ex,
        ey,
        ez,
        tf,
        phrnl,
        imprnl,
        ssnl,
        ph_gamma_real,
        imp_gamma_real,
        surf_gamma_real,
        charge_sign,
        hbar,
        q_e,
        spr0,
    ):
        dkx_dt_real = charge_sign * q_e * ex / hbar
        dky_dt_real = charge_sign * q_e * ey / hbar
        dkz_dt_real = charge_sign * q_e * ez / hbar
        return (
            float(x + vx * tf),
            float(y + vy * tf),
            float(z + vz * tf),
            float(kx + dkx_dt_real * tf * spr0),
            float(ky + dky_dt_real * tf * spr0),
            float(kz + dkz_dt_real * tf * spr0),
            float(energy + charge_sign * (vx * ex + vy * ey + vz * ez) * tf),
            float(max(0.0, phrnl - ph_gamma_real * tf)),
            float(max(0.0, imprnl - imp_gamma_real * tf)),
            float(max(0.0, ssnl - surf_gamma_real * tf)),
        )


if njit is not None:
    @njit(fastmath=True, cache=True)
    def _motion_rule_code_kernel(x, y, z, hit_dir, motion_bounds_m, motion_faces, motion_rules):
        tol = 1.0e-15
        for idx in range(motion_faces.size):
            if motion_faces[idx] != hit_dir:
                continue
            b = motion_bounds_m[idx]
            if (
                (b[0] - tol) <= x <= (b[1] + tol)
                and (b[2] - tol) <= y <= (b[3] + tol)
                and (b[4] - tol) <= z <= (b[5] + tol)
            ):
                return int(motion_rules[idx])
        return RULE_PASS


    @njit(fastmath=True, cache=True)
    def _wrap_axis_periodic_kernel(k_norm, k_idx, k_period_norm, n_ticks):
        if n_ticks <= 0:
            return float(k_norm), int(k_idx)
        if k_idx >= n_ticks:
            return float(k_norm - k_period_norm), 0
        if k_idx < 0:
            return float(k_norm + k_period_norm), n_ticks - 1
        return float(k_norm), int(k_idx)


    @njit(fastmath=True, cache=True)
    def _reflect_k_state_kernel(
        kx,
        ky,
        kz,
        kx_idx,
        ky_idx,
        kz_idx,
        hit_dir,
        to_pi,
        kx_boundaries,
        ky_boundaries,
        kz_boundaries,
        num_ticks_x,
        num_ticks_y,
        num_ticks_z,
    ):
        kx_ref = kx
        ky_ref = ky
        kz_ref = kz
        if hit_dir == 0 or hit_dir == 1:
            kx_ref = -kx_ref
        elif hit_dir == 2 or hit_dir == 3:
            ky_ref = -ky_ref
        elif hit_dir == 4 or hit_dir == 5:
            kz_ref = -kz_ref

        kx_idx_new = _axis_index_kernel(kx_ref * to_pi, kx_boundaries, num_ticks_x)
        ky_idx_new = _axis_index_kernel(ky_ref * to_pi, ky_boundaries, num_ticks_y)
        kz_idx_new = _axis_index_kernel(kz_ref * to_pi, kz_boundaries, num_ticks_z)
        return (
            float(kx_ref),
            float(ky_ref),
            float(kz_ref),
            int(kx_idx_new),
            int(ky_idx_new),
            int(kz_idx_new),
        )


    @njit(fastmath=True, cache=True)
    def _sample_k_state_from_energy_kernel(
        energy_eV,
        analytic_num_bins,
        e_min,
        e_split,
        de_low,
        de_high,
        n_low,
        ntlist,
        ptlist,
        tlist,
        wcdf,
        wsum,
        nonempty,
        ek_k_norm,
        ek_k_pi,
        ek_energy_eV,
        kx_boundaries,
        ky_boundaries,
        kz_boundaries,
        num_ticks_x,
        num_ticks_y,
        num_ticks_z,
    ):
        bin_idx = _map_energy_to_bin_kernel(
            energy_eV,
            analytic_num_bins,
            e_min,
            e_split,
            de_low,
            de_high,
            n_low,
        )
        bin_idx = _nearest_nonempty_bin_kernel(bin_idx, ntlist, nonempty)
        if bin_idx < 0:
            return (
                False,
                0.0,
                0.0,
                0.0,
                energy_eV,
                0,
                0,
                0,
            )

        count = int(ntlist[bin_idx])
        if count <= 0:
            return (
                False,
                0.0,
                0.0,
                0.0,
                energy_eV,
                0,
                0,
                0,
            )

        start = int(ptlist[bin_idx])
        total_w = float(wsum[bin_idx])
        offset = _weighted_offset_kernel(
            count,
            start,
            total_w,
            wcdf,
            np.random.random(),
            np.random.random(),
        )
        ek_idx = int(tlist[start + offset])
        k_norm = ek_k_norm[ek_idx]
        k_pi = ek_k_pi[ek_idx]

        return (
            True,
            float(k_norm[0]),
            float(k_norm[1]),
            float(k_norm[2]),
            float(ek_energy_eV[ek_idx]),
            int(_axis_index_kernel(float(k_pi[0]), kx_boundaries, num_ticks_x)),
            int(_axis_index_kernel(float(k_pi[1]), ky_boundaries, num_ticks_y)),
            int(_axis_index_kernel(float(k_pi[2]), kz_boundaries, num_ticks_z)),
        )


    @njit(fastmath=True, cache=True)
    def _phonon_event_kernel(
        energy_eV,
        kx_idx,
        ky_idx,
        kz_idx,
        gamma_max,
        total_energy_grid_eV,
        total_rate_norm,
        component_rate_norm,
        time0,
        cdf_table,
        hw_min_limit,
        hw_max_limit,
        analytic_num_bins,
        e_min,
        e_split,
        de_low,
        de_high,
        n_low,
        ntlist,
        ptlist,
        tlist,
        wcdf,
        wsum,
        nonempty,
        ek_k_norm,
        ek_k_pi,
        ek_energy_eV,
        kx_boundaries,
        ky_boundaries,
        kz_boundaries,
        num_ticks_x,
        num_ticks_y,
        num_ticks_z,
        emin_eV,
        dtable_eV,
    ):
        if gamma_max <= 1.0e-30:
            return (
                False, False,
                0.0, 0.0, 0.0, energy_eV, kx_idx, ky_idx, kz_idx,
                0, 0, 0, 0, 0.0, 0.0,
            )

        gamma_real = _interp_1d_clipped_kernel(energy_eV, total_energy_grid_eV, total_rate_norm) / time0
        if gamma_real <= 1.0e-30 or not np.isfinite(gamma_real):
            return (
                False, False,
                0.0, 0.0, 0.0, energy_eV, kx_idx, ky_idx, kz_idx,
                0, 0, 0, 0, 0.0, 0.0,
            )

        x_rand = np.random.random() * gamma_max
        if x_rand > gamma_real:
            return (
                False, True,
                0.0, 0.0, 0.0, energy_eV, kx_idx, ky_idx, kz_idx,
                0, 0, 0, 0, 0.0, 0.0,
            )

        mech_rates = _component_rates_kernel(energy_eV, total_energy_grid_eV, component_rate_norm, time0)
        rate_sum = 0.0
        for i in range(mech_rates.size):
            rate_sum += mech_rates[i]
        if rate_sum <= 1.0e-30:
            return (
                False, True,
                0.0, 0.0, 0.0, energy_eV, kx_idx, ky_idx, kz_idx,
                0, 0, 0, 0, 0.0, 0.0,
            )

        mech_idx = _mechanism_index_kernel(mech_rates, np.random.random() * rate_sum)
        if mech_idx == 0:
            ok, new_kx, new_ky, new_kz, new_energy, new_kx_idx, new_ky_idx, new_kz_idx = _sample_k_state_from_energy_kernel(
                energy_eV,
                analytic_num_bins, e_min, e_split, de_low, de_high, n_low,
                ntlist, ptlist, tlist, wcdf, wsum, nonempty,
                ek_k_norm, ek_k_pi, ek_energy_eV,
                kx_boundaries, ky_boundaries, kz_boundaries,
                num_ticks_x, num_ticks_y, num_ticks_z,
            )
            if not ok:
                return (
                    False, True,
                    0.0, 0.0, 0.0, energy_eV, kx_idx, ky_idx, kz_idx,
                    0, 0, 0, 0, 0.0, 0.0,
                )
            return (
                True, False,
                new_kx, new_ky, new_kz, new_energy, new_kx_idx, new_ky_idx, new_kz_idx,
                1, 0, 0, 0, 0.0, 0.0,
            )

        if dtable_eV <= 0.0:
            return (
                False, True,
                0.0, 0.0, 0.0, energy_eV, kx_idx, ky_idx, kz_idx,
                0, 0, 0, 0, 0.0, 0.0,
            )
        itab = int(np.round((energy_eV - emin_eV) / dtable_eV))
        if itab < 0:
            itab = 0
        elif itab > cdf_table.shape[1] - 1:
            itab = cdf_table.shape[1] - 1
        cdf_row = cdf_table[mech_idx, itab]
        hw_eV = _hw_from_cdf_kernel(cdf_row, np.random.random(), hw_min_limit[mech_idx], hw_max_limit[mech_idx])

        absorbed_eV = 0.0
        emitted_eV = 0.0
        optical_abs_inc = 0
        optical_ems_inc = 0
        if mech_idx == 1 or mech_idx == 3:
            new_energy_eV = energy_eV + hw_eV
            absorbed_eV = hw_eV
            optical_abs_inc = 1
        else:
            new_energy_eV = energy_eV - hw_eV
            if new_energy_eV <= 0.0:
                return (
                    False, True,
                    0.0, 0.0, 0.0, energy_eV, kx_idx, ky_idx, kz_idx,
                    0, 0, 0, 0, 0.0, 0.0,
                )
            emitted_eV = hw_eV
            optical_ems_inc = 1

        ok, new_kx, new_ky, new_kz, sampled_energy, new_kx_idx, new_ky_idx, new_kz_idx = _sample_k_state_from_energy_kernel(
            new_energy_eV,
            analytic_num_bins, e_min, e_split, de_low, de_high, n_low,
            ntlist, ptlist, tlist, wcdf, wsum, nonempty,
            ek_k_norm, ek_k_pi, ek_energy_eV,
            kx_boundaries, ky_boundaries, kz_boundaries,
            num_ticks_x, num_ticks_y, num_ticks_z,
        )
        if not ok:
            return (
                False, True,
                0.0, 0.0, 0.0, energy_eV, kx_idx, ky_idx, kz_idx,
                0, 0, 0, 0, 0.0, 0.0,
            )
        return (
            True, False,
            new_kx, new_ky, new_kz, sampled_energy, new_kx_idx, new_ky_idx, new_kz_idx,
            0, optical_abs_inc, optical_ems_inc, 0, absorbed_eV, emitted_eV,
        )


    @njit(fastmath=True, cache=True)
    def _particle_fly_single_kernel(
        x, y, z, kx, ky, kz, energy,
        i, j, k,
        kx_idx, ky_idx, kz_idx,
        left_time, charge,
        cell_ex, cell_ey, cell_ez,
        x_nodes, y_nodes, z_nodes,
        material_id, mc_mat_id,
        velocity_grid,
        kx_boundaries, ky_boundaries, kz_boundaries,
        num_ticks_x, num_ticks_y, num_ticks_z,
        k_period_norm_x, k_period_norm_y, k_period_norm_z,
        hbar, q_e, spr0, time0, sia0_norm, a_real,
        kcell_max_phsr_real,
        total_energy_grid_eV, total_rate_norm, component_rate_norm, cdf_table,
        hw_min_limit, hw_max_limit,
        phonon_emin_eV, phonon_dtable_eV,
        analytic_num_bins, e_min, e_split, de_low, de_high, n_low,
        ntlist, ptlist, tlist, wcdf, wsum, nonempty,
        ek_k_norm, ek_k_pi, ek_energy_eV,
        motion_bounds_m, motion_faces, motion_rules,
        monitor_bounds_m, monitor_faces,
        monitor_charge_c_sum, monitor_crossing_count,
        max_loops,
    ):
        get_tet_time = True
        get_cell_time = True
        get_ph_time = True
        get_imp_time = True
        get_surf_time = True

        tet_tf = np.inf
        cell_tf = np.inf
        ph_tf = np.inf
        imp_tf = np.inf
        surf_tf = np.inf
        phrnl = 0.0
        imprnl = 0.0
        ssnl = 0.0
        ph_gamma_real = 0.0
        imp_gamma_real = 0.0
        surf_gamma_real = 0.0

        charge_sign = -1.0
        if charge > 0.0:
            charge_sign = 1.0
        charge_c = charge * q_e

        catch_stat_inc = 0
        escape_total_inc = 0
        escape_xp = 0
        escape_xm = 0
        escape_yp = 0
        escape_ym = 0
        escape_zp = 0
        escape_zm = 0

        phonon_self_inc = 0
        phonon_acoustic_inc = 0
        phonon_opt_abs_inc = 0
        phonon_opt_ems_inc = 0
        phonon_absorbed_eV = 0.0
        phonon_emitted_eV = 0.0
        step_ph_inc = 0

        spawn_requested = False
        spawn_x = 0.0
        spawn_y = 0.0
        spawn_z = 0.0
        spawn_kx = 0.0
        spawn_ky = 0.0
        spawn_kz = 0.0
        spawn_energy = 0.0
        spawn_kx_idx = 0
        spawn_ky_idx = 0
        spawn_kz_idx = 0
        spawn_left_time = 0.0
        spawn_i = 0
        spawn_j = 0
        spawn_k = 0

        to_pi = sia0_norm / np.pi
        boundary_real_scale = np.pi / a_real
        k_real_scale = 1.0 / spr0

        status = FLY_STATUS_DONE
        err_i = 0
        err_j = 0
        err_k = 0

        for _loop in range(max_loops):
            if left_time <= 0.0:
                status = FLY_STATUS_DONE
                break

            if i < 0 or i >= material_id.shape[0] or j < 0 or j >= material_id.shape[1] or k < 0 or k >= material_id.shape[2]:
                status = FLY_STATUS_ERROR_NON_MC
                err_i = i
                err_j = j
                err_k = k
                break
            if material_id[i, j, k] != mc_mat_id:
                status = FLY_STATUS_ERROR_NON_MC
                err_i = i
                err_j = j
                err_k = k
                break

            ex = cell_ex[i, j, k]
            ey = cell_ey[i, j, k]
            ez = cell_ez[i, j, k]
            vel = velocity_grid[kx_idx, ky_idx, kz_idx]
            vx = vel[0]
            vy = vel[1]
            vz = vel[2]

            if get_tet_time:
                fx_real = charge_sign * q_e * ex
                fy_real = charge_sign * q_e * ey
                fz_real = charge_sign * q_e * ez
                kx_real = kx * k_real_scale
                ky_real = ky * k_real_scale
                kz_real = kz * k_real_scale
                dkx_dt_real = fx_real / hbar
                dky_dt_real = fy_real / hbar
                dkz_dt_real = fz_real / hbar
                tet_tf, hit_dir, _ = _compute_kgrid_time_kernel(
                    kx_real, ky_real, kz_real,
                    dkx_dt_real, dky_dt_real, dkz_dt_real,
                    kx_idx, ky_idx, kz_idx,
                    kx_boundaries, ky_boundaries, kz_boundaries,
                    boundary_real_scale, time0,
                )
                last_k_col_dir = hit_dir
                get_tet_time = False
            else:
                last_k_col_dir = -1

            if get_cell_time:
                cell_tf, hit_dir, _ = _compute_cell_time_kernel(
                    x, y, z, vx, vy, vz,
                    x_nodes[i], x_nodes[i + 1],
                    y_nodes[j], y_nodes[j + 1],
                    z_nodes[k], z_nodes[k + 1],
                    time0,
                )
                last_cell_col_dir = hit_dir
                get_cell_time = False
            else:
                last_cell_col_dir = -1

            gamma_max = 0.0
            if get_ph_time:
                if kcell_max_phsr_real.size > 0:
                    gamma_max = kcell_max_phsr_real[kx_idx, ky_idx, kz_idx]
                if gamma_max > 1.0e-30 and np.isfinite(gamma_max):
                    rand = np.random.random()
                    if rand < np.finfo(np.float64).tiny:
                        rand = np.finfo(np.float64).tiny
                    phrnl = -np.log(rand)
                    ph_tf = phrnl / gamma_max
                    ph_gamma_real = phrnl / ph_tf if ph_tf > 0.0 else 0.0
                else:
                    ph_tf = np.inf
                    phrnl = 0.0
                    ph_gamma_real = 0.0
                get_ph_time = False
            elif kcell_max_phsr_real.size > 0:
                gamma_max = kcell_max_phsr_real[kx_idx, ky_idx, kz_idx]

            if get_imp_time:
                imp_tf = np.inf
                imprnl = 0.0
                imp_gamma_real = 0.0
                get_imp_time = False
            if get_surf_time:
                surf_tf = np.inf
                ssnl = 0.0
                surf_gamma_real = 0.0
                get_surf_time = False

            tf, event_flag = _select_next_event_kernel(tet_tf, cell_tf, ph_tf, imp_tf, surf_tf, left_time)
            if tf < 0.0:
                status = FLY_STATUS_CATCH
                break

            tet_tf -= tf
            cell_tf -= tf
            ph_tf -= tf
            imp_tf -= tf
            surf_tf -= tf
            left_time -= tf

            x, y, z, kx, ky, kz, energy, phrnl, imprnl, ssnl = _advance_particle_drift_kernel(
                x, y, z, kx, ky, kz, energy,
                vx, vy, vz, ex, ey, ez, tf,
                phrnl, imprnl, ssnl,
                ph_gamma_real, imp_gamma_real, surf_gamma_real,
                charge_sign, hbar, q_e, spr0,
            )

            if event_flag == EVENT_TIME_STEP_END:
                status = FLY_STATUS_DONE
                break

            if event_flag == EVENT_HIT_KGRID:
                if last_k_col_dir == 0:
                    kx_idx += 1
                elif last_k_col_dir == 1:
                    kx_idx -= 1
                elif last_k_col_dir == 2:
                    ky_idx += 1
                elif last_k_col_dir == 3:
                    ky_idx -= 1
                elif last_k_col_dir == 4:
                    kz_idx += 1
                elif last_k_col_dir == 5:
                    kz_idx -= 1
                kx, kx_idx = _wrap_axis_periodic_kernel(kx, kx_idx, k_period_norm_x, num_ticks_x)
                ky, ky_idx = _wrap_axis_periodic_kernel(ky, ky_idx, k_period_norm_y, num_ticks_y)
                kz, kz_idx = _wrap_axis_periodic_kernel(kz, kz_idx, k_period_norm_z, num_ticks_z)
                get_tet_time = True
                get_cell_time = True
                continue

            if event_flag == EVENT_HIT_CELL:
                hit_dir = last_cell_col_dir
                rule_code = _motion_rule_code_kernel(x, y, z, hit_dir, motion_bounds_m, motion_faces, motion_rules)

                if rule_code == RULE_CATCH:
                    _accumulate_monitor_crossing_kernel(
                        x, y, z, hit_dir, charge_c,
                        monitor_bounds_m, monitor_faces,
                        monitor_charge_c_sum, monitor_crossing_count,
                    )
                    catch_stat_inc += 1
                    status = FLY_STATUS_CATCH
                    break

                if rule_code == RULE_GENERATE:
                    _accumulate_monitor_crossing_kernel(
                        x, y, z, hit_dir, charge_c,
                        monitor_bounds_m, monitor_faces,
                        monitor_charge_c_sum, monitor_crossing_count,
                    )
                    spawn_requested = True
                    spawn_x = x
                    spawn_y = y
                    spawn_z = z
                    spawn_kx = kx
                    spawn_ky = ky
                    spawn_kz = kz
                    spawn_energy = energy
                    spawn_kx_idx = kx_idx
                    spawn_ky_idx = ky_idx
                    spawn_kz_idx = kz_idx
                    spawn_left_time = left_time
                    spawn_i = i
                    spawn_j = j
                    spawn_k = k
                    if hit_dir == 0:
                        spawn_i += 1
                    elif hit_dir == 1:
                        spawn_i -= 1
                    elif hit_dir == 2:
                        spawn_j += 1
                    elif hit_dir == 3:
                        spawn_j -= 1
                    elif hit_dir == 4:
                        spawn_k += 1
                    elif hit_dir == 5:
                        spawn_k -= 1

                    kx, ky, kz, kx_idx, ky_idx, kz_idx = _reflect_k_state_kernel(
                        kx, ky, kz, kx_idx, ky_idx, kz_idx, hit_dir,
                        to_pi, kx_boundaries, ky_boundaries, kz_boundaries,
                        num_ticks_x, num_ticks_y, num_ticks_z,
                    )
                    status = FLY_STATUS_GENERATE
                    break

                if rule_code == RULE_REFLECT or rule_code == RULE_SCATTOX:
                    kx, ky, kz, kx_idx, ky_idx, kz_idx = _reflect_k_state_kernel(
                        kx, ky, kz, kx_idx, ky_idx, kz_idx, hit_dir,
                        to_pi, kx_boundaries, ky_boundaries, kz_boundaries,
                        num_ticks_x, num_ticks_y, num_ticks_z,
                    )
                    get_tet_time = True
                    get_cell_time = True
                    if rule_code == RULE_SCATTOX:
                        get_surf_time = True
                    continue

                if hit_dir == 0:
                    i += 1
                elif hit_dir == 1:
                    i -= 1
                elif hit_dir == 2:
                    j += 1
                elif hit_dir == 3:
                    j -= 1
                elif hit_dir == 4:
                    k += 1
                elif hit_dir == 5:
                    k -= 1

                if i < 0 or i >= material_id.shape[0] or j < 0 or j >= material_id.shape[1] or k < 0 or k >= material_id.shape[2]:
                    _accumulate_monitor_crossing_kernel(
                        x, y, z, hit_dir, charge_c,
                        monitor_bounds_m, monitor_faces,
                        monitor_charge_c_sum, monitor_crossing_count,
                    )
                    escape_total_inc += 1
                    if hit_dir == 0:
                        escape_xp += 1
                    elif hit_dir == 1:
                        escape_xm += 1
                    elif hit_dir == 2:
                        escape_yp += 1
                    elif hit_dir == 3:
                        escape_ym += 1
                    elif hit_dir == 4:
                        escape_zp += 1
                    elif hit_dir == 5:
                        escape_zm += 1
                    left_time = 0.0
                    status = FLY_STATUS_CATCH
                    break

                if material_id[i, j, k] != mc_mat_id:
                    status = FLY_STATUS_ERROR_NON_MC
                    err_i = i
                    err_j = j
                    err_k = k
                    break

                _accumulate_monitor_crossing_kernel(
                    x, y, z, hit_dir, charge_c,
                    monitor_bounds_m, monitor_faces,
                    monitor_charge_c_sum, monitor_crossing_count,
                )
                get_tet_time = True
                get_cell_time = True
                continue

            if event_flag == EVENT_PHONON_SCATTER:
                did_real, did_self, new_kx, new_ky, new_kz, new_energy, new_kx_idx, new_ky_idx, new_kz_idx, acoustic_inc, opt_abs_inc, opt_ems_inc, _dummy_zero, absorbed_eV, emitted_eV = _phonon_event_kernel(
                    energy, kx_idx, ky_idx, kz_idx,
                    gamma_max,
                    total_energy_grid_eV, total_rate_norm, component_rate_norm, time0,
                    cdf_table, hw_min_limit, hw_max_limit,
                    analytic_num_bins, e_min, e_split, de_low, de_high, n_low,
                    ntlist, ptlist, tlist, wcdf, wsum, nonempty,
                    ek_k_norm, ek_k_pi, ek_energy_eV,
                    kx_boundaries, ky_boundaries, kz_boundaries,
                    num_ticks_x, num_ticks_y, num_ticks_z,
                    phonon_emin_eV, phonon_dtable_eV,
                )
                if did_self:
                    phonon_self_inc += 1
                if did_real:
                    kx = new_kx
                    ky = new_ky
                    kz = new_kz
                    energy = new_energy
                    kx_idx = new_kx_idx
                    ky_idx = new_ky_idx
                    kz_idx = new_kz_idx
                    step_ph_inc += 1
                    phonon_acoustic_inc += acoustic_inc
                    phonon_opt_abs_inc += opt_abs_inc
                    phonon_opt_ems_inc += opt_ems_inc
                    phonon_absorbed_eV += absorbed_eV
                    phonon_emitted_eV += emitted_eV
                get_tet_time = True
                get_cell_time = True
                get_ph_time = True
                continue

            if event_flag == EVENT_IMPURITY_SCATTER:
                get_tet_time = True
                get_cell_time = True
                get_imp_time = True
                continue

            if event_flag == EVENT_SURFACE_SCATTER:
                get_tet_time = True
                get_cell_time = True
                get_surf_time = True
                continue

        return (
            int(status),
            int(err_i), int(err_j), int(err_k),
            float(x), float(y), float(z),
            float(kx), float(ky), float(kz),
            float(energy),
            int(i), int(j), int(k),
            int(kx_idx), int(ky_idx), int(kz_idx),
            float(left_time),
            bool(spawn_requested),
            float(spawn_x), float(spawn_y), float(spawn_z),
            float(spawn_kx), float(spawn_ky), float(spawn_kz),
            float(spawn_energy),
            int(spawn_kx_idx), int(spawn_ky_idx), int(spawn_kz_idx),
            float(spawn_left_time),
            int(spawn_i), int(spawn_j), int(spawn_k),
            int(catch_stat_inc),
            int(escape_total_inc),
            int(escape_xp), int(escape_xm),
            int(escape_yp), int(escape_ym),
            int(escape_zp), int(escape_zm),
            int(step_ph_inc),
            int(phonon_self_inc),
            int(phonon_acoustic_inc),
            int(phonon_opt_abs_inc),
            int(phonon_opt_ems_inc),
            float(phonon_absorbed_eV),
            float(phonon_emitted_eV),
        )


    @njit(fastmath=True, cache=True)
    def _particle_fly_batch_kernel(
        active_indices,
        x_arr, y_arr, z_arr, kx_arr, ky_arr, kz_arr, energy_arr,
        i_arr, j_arr, k_arr, kx_idx_arr, ky_idx_arr, kz_idx_arr,
        left_time_arr, charge_arr,
        cell_ex, cell_ey, cell_ez,
        x_nodes, y_nodes, z_nodes,
        material_id, mc_mat_id,
        velocity_grid,
        kx_boundaries, ky_boundaries, kz_boundaries,
        num_ticks_x, num_ticks_y, num_ticks_z,
        k_period_norm_x, k_period_norm_y, k_period_norm_z,
        hbar, q_e, spr0, time0, sia0_norm, a_real,
        kcell_max_phsr_real,
        total_energy_grid_eV, total_rate_norm, component_rate_norm, cdf_table,
        hw_min_limit, hw_max_limit,
        phonon_emin_eV, phonon_dtable_eV,
        analytic_num_bins, e_min, e_split, de_low, de_high, n_low,
        ntlist, ptlist, tlist, wcdf, wsum, nonempty,
        ek_k_norm, ek_k_pi, ek_energy_eV,
        motion_bounds_m, motion_faces, motion_rules,
        monitor_bounds_m, monitor_faces,
        monitor_charge_c_sum, monitor_crossing_count,
        max_loops,
        status_out,
        err_i_out, err_j_out, err_k_out,
        x_out, y_out, z_out, kx_out, ky_out, kz_out, energy_out,
        i_out, j_out, k_out, kx_idx_out, ky_idx_out, kz_idx_out,
        left_time_out,
        spawn_requested_out,
        spawn_x_out, spawn_y_out, spawn_z_out,
        spawn_kx_out, spawn_ky_out, spawn_kz_out,
        spawn_energy_out,
        spawn_kx_idx_out, spawn_ky_idx_out, spawn_kz_idx_out,
        spawn_left_time_out,
        spawn_i_out, spawn_j_out, spawn_k_out,
        catch_stat_inc_out,
        escape_total_inc_out,
        escape_xp_out, escape_xm_out, escape_yp_out, escape_ym_out, escape_zp_out, escape_zm_out,
        step_ph_inc_out,
        phonon_self_inc_out,
        phonon_acoustic_inc_out,
        phonon_opt_abs_inc_out,
        phonon_opt_ems_inc_out,
        phonon_absorbed_eV_out,
        phonon_emitted_eV_out,
    ):
        n_active = active_indices.size
        for m in range(n_active):
            par_idx = int(active_indices[m])
            out = _particle_fly_single_kernel(
                x_arr[par_idx], y_arr[par_idx], z_arr[par_idx],
                kx_arr[par_idx], ky_arr[par_idx], kz_arr[par_idx], energy_arr[par_idx],
                i_arr[par_idx], j_arr[par_idx], k_arr[par_idx],
                kx_idx_arr[par_idx], ky_idx_arr[par_idx], kz_idx_arr[par_idx],
                left_time_arr[par_idx], charge_arr[par_idx],
                cell_ex, cell_ey, cell_ez,
                x_nodes, y_nodes, z_nodes,
                material_id, mc_mat_id,
                velocity_grid,
                kx_boundaries, ky_boundaries, kz_boundaries,
                num_ticks_x, num_ticks_y, num_ticks_z,
                k_period_norm_x, k_period_norm_y, k_period_norm_z,
                hbar, q_e, spr0, time0, sia0_norm, a_real,
                kcell_max_phsr_real,
                total_energy_grid_eV, total_rate_norm, component_rate_norm, cdf_table,
                hw_min_limit, hw_max_limit,
                phonon_emin_eV, phonon_dtable_eV,
                analytic_num_bins, e_min, e_split, de_low, de_high, n_low,
                ntlist, ptlist, tlist, wcdf, wsum, nonempty,
                ek_k_norm, ek_k_pi, ek_energy_eV,
                motion_bounds_m, motion_faces, motion_rules,
                monitor_bounds_m, monitor_faces,
                monitor_charge_c_sum, monitor_crossing_count,
                max_loops,
            )
            (
                status_out[m],
                err_i_out[m], err_j_out[m], err_k_out[m],
                x_out[m], y_out[m], z_out[m],
                kx_out[m], ky_out[m], kz_out[m],
                energy_out[m],
                i_out[m], j_out[m], k_out[m],
                kx_idx_out[m], ky_idx_out[m], kz_idx_out[m],
                left_time_out[m],
                spawn_requested_out[m],
                spawn_x_out[m], spawn_y_out[m], spawn_z_out[m],
                spawn_kx_out[m], spawn_ky_out[m], spawn_kz_out[m],
                spawn_energy_out[m],
                spawn_kx_idx_out[m], spawn_ky_idx_out[m], spawn_kz_idx_out[m],
                spawn_left_time_out[m],
                spawn_i_out[m], spawn_j_out[m], spawn_k_out[m],
                catch_stat_inc_out[m],
                escape_total_inc_out[m],
                escape_xp_out[m], escape_xm_out[m], escape_yp_out[m], escape_ym_out[m], escape_zp_out[m], escape_zm_out[m],
                step_ph_inc_out[m],
                phonon_self_inc_out[m],
                phonon_acoustic_inc_out[m],
                phonon_opt_abs_inc_out[m],
                phonon_opt_ems_inc_out[m],
                phonon_absorbed_eV_out[m],
                phonon_emitted_eV_out[m],
            ) = out
else:
    def _motion_rule_code_kernel(x, y, z, hit_dir, motion_bounds_m, motion_faces, motion_rules):
        tol = 1.0e-15
        for idx in range(motion_faces.size):
            if int(motion_faces[idx]) != int(hit_dir):
                continue
            b = motion_bounds_m[idx]
            if (
                (b[0] - tol) <= x <= (b[1] + tol)
                and (b[2] - tol) <= y <= (b[3] + tol)
                and (b[4] - tol) <= z <= (b[5] + tol)
            ):
                return int(motion_rules[idx])
        return RULE_PASS


    def _wrap_axis_periodic_kernel(k_norm, k_idx, k_period_norm, n_ticks):
        if k_idx >= n_ticks:
            return float(k_norm - k_period_norm), 0
        if k_idx < 0:
            return float(k_norm + k_period_norm), n_ticks - 1
        return float(k_norm), int(k_idx)


    def _reflect_k_state_kernel(
        kx, ky, kz, kx_idx, ky_idx, kz_idx, hit_dir,
        to_pi, kx_boundaries, ky_boundaries, kz_boundaries,
        num_ticks_x, num_ticks_y, num_ticks_z,
    ):
        kx_ref, ky_ref, kz_ref = float(kx), float(ky), float(kz)
        if hit_dir in (0, 1):
            kx_ref = -kx_ref
        elif hit_dir in (2, 3):
            ky_ref = -ky_ref
        elif hit_dir in (4, 5):
            kz_ref = -kz_ref
        return (
            kx_ref,
            ky_ref,
            kz_ref,
            _axis_index_kernel(kx_ref * to_pi, kx_boundaries, num_ticks_x),
            _axis_index_kernel(ky_ref * to_pi, ky_boundaries, num_ticks_y),
            _axis_index_kernel(kz_ref * to_pi, kz_boundaries, num_ticks_z),
        )


    def _particle_fly_single_kernel(*args, **kwargs):  # pragma: no cover - no-numba fallback
        raise RuntimeError("Numba is required for the compiled particle_fly core.")


    def _particle_fly_batch_kernel(*args, **kwargs):  # pragma: no cover - no-numba fallback
        raise RuntimeError("Numba is required for the compiled particle_fly batch core.")
def _resolve_input_path(path_value: str | None, input_dir: str | None) -> str | None:
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


def _build_structured_point_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
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
        raise ValueError("Potential point grid is incomplete; cannot build structured potential grid.")

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


def _load_external_potential_to_nodes(path: str, mesh) -> np.ndarray:
    """
    Load TCAD point potential file and interpolate it onto simulation nodes.

    Expected columns: x y z phi
    Assumptions:
    - x/y/z are in um and converted to meters by 1e-6.
    - phi is in volts.
    """
    raw = np.loadtxt(path, comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 4:
        raise ValueError("Initial potential file must have at least 4 columns: x y z phi.")

    x_um = np.asarray(raw[:, 0], dtype=float)
    y_um = np.asarray(raw[:, 1], dtype=float)
    z_um = np.asarray(raw[:, 2], dtype=float)
    phi_v = np.asarray(raw[:, 3], dtype=float)

    valid = np.isfinite(x_um) & np.isfinite(y_um) & np.isfinite(z_um) & np.isfinite(phi_v)
    if not np.any(valid):
        raise ValueError("No valid rows found in initial potential file.")

    x_src = x_um[valid] * 1.0e-6
    y_src = y_um[valid] * 1.0e-6
    z_src = z_um[valid] * 1.0e-6
    phi_src = phi_v[valid]

    x_ticks, y_ticks, z_ticks, phi_grid = _build_structured_point_grid(x_src, y_src, z_src, phi_src)

    xx, yy, zz = np.meshgrid(mesh.x_nodes, mesh.y_nodes, mesh.z_nodes, indexing="ij")
    phi_nodes = _trilinear_interp_structured(
        x_ticks,
        y_ticks,
        z_ticks,
        phi_grid,
        xx.ravel(),
        yy.ravel(),
        zz.ravel(),
    ).reshape(mesh.nx + 1, mesh.ny + 1, mesh.nz + 1)

    return np.where(np.isfinite(phi_nodes), phi_nodes, 0.0)


class Monte_Carlo_Simulation:
    """
    Execute Poisson/MC iteration flow after base initialization is complete.
    """

    def __init__(
        self,
        mesh,
        config: dict,
        phys_config: dict,
        band_struct,
        output_root: str,
        poisson_solver: PoissonSolver | None = None,
        device_structure: dict | None = None,
    ) -> None:
        self.mesh = mesh
        self.config = config
        self.phys_config = phys_config
        self.band_struct = band_struct
        self.output_root = output_root
        self.device_structure = device_structure or {}
        self.input_dir = str(config.get("input_dir", "")).strip() or None

        self.poisson_solver = poisson_solver
        if self.poisson_solver is None:
            self.poisson_solver = PoissonSolver(
                self.mesh,
                self.phys_config,
                self.device_structure,
                build_matrix=False,
            )

        self.particle_ensemble: Particle | None = None
        self.poisson_matrix_A = None
        self.poisson_vector_B: np.ndarray | None = None
        self.initial_potential_from_file = False
        self._poisson_mode_notice_shown = False
        self.catch_par_num = 0
        self.gen_par = 0
        self.last_k_col_dir = -1
        self.last_k_col_time_real = np.inf
        self.last_k_col_time_norm = np.inf
        self.last_cell_col_dir = -1
        self.last_cell_col_time_real = np.inf
        self.last_cell_col_time_norm = np.inf
        self.kcell_max_phsr_real = None
        self.phonon_scatter_stats = {
            "self": 0,
            "acoustic": 0,
            "optical_abs": 0,
            "optical_ems": 0,
            "absorbed_eV": 0.0,
            "emitted_eV": 0.0,
        }
        self.step_scatter_stats = {
            "phonon": 0,
            "impurity": 0,
            "surface": 0,
        }
        self.total_scatter_stats = {
            "phonon": 0,
            "impurity": 0,
            "surface": 0,
        }
        self.particle_event_stats = {
            "catch": 0,
            "generate": 0,
            "escape_total": 0,
            "escape_+X": 0,
            "escape_-X": 0,
            "escape_+Y": 0,
            "escape_-Y": 0,
            "escape_+Z": 0,
            "escape_-Z": 0,
        }
        self.current_monitors = list(self.config.get("current_monitors", []))
        self.monitor_names: list[str] = []
        self.monitor_bounds_m = np.zeros((0, 6), dtype=np.float64)
        self.monitor_faces = np.zeros(0, dtype=np.int32)
        self.step_monitor_charge_c = np.zeros(0, dtype=np.float64)
        self.step_monitor_crossings = np.zeros(0, dtype=np.int64)
        self.output_monitor_charge_c = np.zeros(0, dtype=np.float64)
        self.output_monitor_crossings = np.zeros(0, dtype=np.int64)
        self.output_monitor_step_count = 0
        self.monitor_currents_csv_path: str | None = None
        self.motion_bounds_m = np.zeros((0, 6), dtype=np.float64)
        self.motion_faces = np.zeros(0, dtype=np.int32)
        self.motion_rules = np.zeros(0, dtype=np.int32)
        self._prepare_motion_plane_kernel_tables()
        self._prepare_current_monitor_kernel_tables()
        self._init_current_monitor_outputs()
        self._prepare_compiled_band_tables()
        if self.band_struct is not None and getattr(self.band_struct, "scattering_rate", None) is not None:
            try:
                self.kcell_max_phsr_real = build_kcell_max_phsr_real(self.band_struct)
                print(
                    "[MC] k-cell phonon-rate upper bound ready. "
                    f"max={float(np.max(self.kcell_max_phsr_real)):.4e} 1/s"
                )
            except Exception as exc:
                print(f"[MC] [Warning] Failed to build k-cell phonon max-rate table: {exc}")
                self.kcell_max_phsr_real = None

    def _prepare_motion_plane_kernel_tables(self) -> None:
        face_code = {
            "+X": 0,
            "-X": 1,
            "+Y": 2,
            "-Y": 3,
            "+Z": 4,
            "-Z": 5,
        }
        rule_code = {
            "PASS": RULE_PASS,
            "CATCH": RULE_CATCH,
            "GENERATE": RULE_GENERATE,
            "REFLECT": RULE_REFLECT,
            "SCATTOX": RULE_SCATTOX,
        }

        bounds_list: list[list[float]] = []
        faces_list: list[int] = []
        rules_list: list[int] = []
        for entry in self.device_structure.get("motion_planes", []):
            face = str(entry.get("face", "")).upper()
            rule = str(entry.get("rule", "PASS")).upper()
            bounds = entry.get("bounds", [])
            if face not in face_code or rule not in rule_code or len(bounds) != 6:
                continue
            x1, x2, y1, y2, z1, z2 = [float(v) * 1.0e-9 for v in bounds]
            bounds_list.append(
                [
                    min(x1, x2), max(x1, x2),
                    min(y1, y2), max(y1, y2),
                    min(z1, z2), max(z1, z2),
                ]
            )
            faces_list.append(face_code[face])
            rules_list.append(rule_code[rule])

        if bounds_list:
            self.motion_bounds_m = np.asarray(bounds_list, dtype=np.float64)
            self.motion_faces = np.asarray(faces_list, dtype=np.int32)
            self.motion_rules = np.asarray(rules_list, dtype=np.int32)

    def _prepare_current_monitor_kernel_tables(self) -> None:
        face_code = {
            "+X": 0,
            "-X": 1,
            "+Y": 2,
            "-Y": 3,
            "+Z": 4,
            "-Z": 5,
        }
        bounds_list: list[list[float]] = []
        faces_list: list[int] = []
        names_list: list[str] = []
        for idx, entry in enumerate(self.current_monitors, start=1):
            face = str(entry.get("face", "")).upper()
            bounds = entry.get("bounds", [])
            if face not in face_code or len(bounds) != 6:
                continue
            x1, x2, y1, y2, z1, z2 = [float(v) * 1.0e-9 for v in bounds]
            bounds_list.append(
                [
                    min(x1, x2), max(x1, x2),
                    min(y1, y2), max(y1, y2),
                    min(z1, z2), max(z1, z2),
                ]
            )
            faces_list.append(face_code[face])
            names_list.append(str(entry.get("name", f"M{idx:03d}")))

        if bounds_list:
            self.monitor_names = names_list
            self.monitor_bounds_m = np.asarray(bounds_list, dtype=np.float64)
            self.monitor_faces = np.asarray(faces_list, dtype=np.int32)
            self.step_monitor_charge_c = np.zeros(len(bounds_list), dtype=np.float64)
            self.step_monitor_crossings = np.zeros(len(bounds_list), dtype=np.int64)
            self.output_monitor_charge_c = np.zeros(len(bounds_list), dtype=np.float64)
            self.output_monitor_crossings = np.zeros(len(bounds_list), dtype=np.int64)

    def _init_current_monitor_outputs(self) -> None:
        if self.monitor_bounds_m.shape[0] <= 0:
            return
        monitor_dir = os.path.join(self.output_root, "Monitors")
        os.makedirs(monitor_dir, exist_ok=True)
        info_path = os.path.join(monitor_dir, "monitor_surfaces.csv")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write("name,x1_nm,x2_nm,y1_nm,y2_nm,z1_nm,z2_nm,face\n")
            for idx, name in enumerate(self.monitor_names):
                b = self.monitor_bounds_m[idx] * 1.0e9
                face = self._cell_hit_dir_to_face_label(int(self.monitor_faces[idx])) or "NA"
                f.write(
                    f"{name},{b[0]:.6f},{b[1]:.6f},{b[2]:.6f},{b[3]:.6f},{b[4]:.6f},{b[5]:.6f},{face}\n"
                )

        self.monitor_currents_csv_path = os.path.join(monitor_dir, "monitor_currents.csv")
        with open(self.monitor_currents_csv_path, "w", encoding="utf-8") as f:
            header = ["Step", "Time_fs", "AveragingSteps"]
            for name in self.monitor_names:
                header.append(f"{name}_A")
                header.append(f"{name}_count")
            f.write(",".join(header) + "\n")

    @staticmethod
    def _distribute_cell_to_nodes(cell_data: np.ndarray) -> np.ndarray:
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

    def _build_cell_electron_concentration_from_particles(self) -> np.ndarray:
        shape = (self.mesh.nx, self.mesh.ny, self.mesh.nz)
        out = np.zeros(shape, dtype=np.float64)
        if self.particle_ensemble is None or self.particle_ensemble.size <= 0:
            return out

        par = self.particle_ensemble
        valid = (
            (par.i >= 0) & (par.i < self.mesh.nx)
            & (par.j >= 0) & (par.j < self.mesh.ny)
            & (par.k >= 0) & (par.k < self.mesh.nz)
        )
        if not np.any(valid):
            return out

        np.add.at(
            out,
            (par.i[valid], par.j[valid], par.k[valid]),
            -np.asarray(par.charge[valid], dtype=np.float64),
        )
        vol_safe = np.where(self.mesh.volume > 1.0e-30, self.mesh.volume, 1.0)
        out = out / vol_safe
        out[self.mesh.volume <= 1.0e-30] = 0.0
        return out

    def _build_node_electron_concentration_from_cells(self, n_cell_m3: np.ndarray) -> np.ndarray:
        mc_label = str(self.phys_config.get("material", "")).upper()
        mc_mat_id = self.mesh.label_map.get(mc_label, None)
        if mc_mat_id is None:
            return np.zeros((self.mesh.nx + 1, self.mesh.ny + 1, self.mesh.nz + 1), dtype=np.float64)

        mc_mask = self.mesh.material_id == int(mc_mat_id)
        cell_vol = np.where(mc_mask, self.mesh.volume, 0.0)
        cell_charge = np.where(mc_mask, n_cell_m3 * self.mesh.volume, 0.0)
        node_vol = self._distribute_cell_to_nodes(cell_vol)
        node_charge = self._distribute_cell_to_nodes(cell_charge)
        out = np.zeros_like(node_vol, dtype=np.float64)
        valid = node_vol > 1.0e-30
        out[valid] = node_charge[valid] / node_vol[valid]
        return out

    def _export_particle_snapshot(self, snapshot_dir: str) -> None:
        if self.particle_ensemble is None:
            return
        par = self.particle_ensemble
        valid = np.asarray(par.i >= 0)
        out_path = os.path.join(snapshot_dir, "particles.txt")
        if not np.any(valid):
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("# No active particles\n")
            return

        k_scale = np.pi / float(self.phys_config["sia0_norm"])
        ids = np.flatnonzero(valid).astype(np.int64)
        data = np.column_stack(
            (
                ids,
                par.i[valid],
                par.j[valid],
                par.k[valid],
                par.x[valid],
                par.y[valid],
                par.z[valid],
                par.kx[valid] / k_scale,
                par.ky[valid] / k_scale,
                par.kz[valid] / k_scale,
                par.energy[valid],
                par.charge[valid],
                par.kx_idx[valid],
                par.ky_idx[valid],
                par.kz_idx[valid],
            )
        )
        header = "ID i j k x(m) y(m) z(m) kx(pi/a) ky(pi/a) kz(pi/a) energy_eV charge(/q) kx_idx ky_idx kz_idx"
        fmt = ["%d", "%d", "%d", "%d"] + ["%.6e"] * 3 + ["%.6e"] * 3 + ["%.6e"] * 2 + ["%d", "%d", "%d"]
        np.savetxt(out_path, data, header=header, fmt=fmt)

    def _export_snapshot(self, step: int, time_fs: float) -> None:
        snapshot_root = os.path.join(self.output_root, "Snapshots")
        os.makedirs(snapshot_root, exist_ok=True)
        snapshot_dir = os.path.join(snapshot_root, f"step_{step + 1:07d}")
        os.makedirs(snapshot_dir, exist_ok=True)

        n_cell_m3 = self._build_cell_electron_concentration_from_particles()
        n_node_m3 = self._build_node_electron_concentration_from_cells(n_cell_m3)

        xc = 0.5 * (self.mesh.x_nodes[:-1] + self.mesh.x_nodes[1:])
        yc = 0.5 * (self.mesh.y_nodes[:-1] + self.mesh.y_nodes[1:])
        zc = 0.5 * (self.mesh.z_nodes[:-1] + self.mesh.z_nodes[1:])
        ii, jj, kk = np.meshgrid(
            np.arange(self.mesh.nx, dtype=np.int32),
            np.arange(self.mesh.ny, dtype=np.int32),
            np.arange(self.mesh.nz, dtype=np.int32),
            indexing="ij",
        )
        xxc, yyc, zzc = np.meshgrid(xc, yc, zc, indexing="ij")
        cell_data = np.column_stack(
            (
                ii.ravel(), jj.ravel(), kk.ravel(),
                xxc.ravel(), yyc.ravel(), zzc.ravel(),
                n_cell_m3.ravel(), (n_cell_m3 / 1.0e6).ravel(),
            )
        )
        np.savetxt(
            os.path.join(snapshot_dir, "electron_concentration_cells.txt"),
            cell_data,
            header="i j k x_center(m) y_center(m) z_center(m) n(1/m^3) n(1/cm^3)",
            fmt=["%d", "%d", "%d"] + ["%.6e"] * 5,
        )

        ip, jp, kp = np.meshgrid(
            np.arange(self.mesh.nx + 1, dtype=np.int32),
            np.arange(self.mesh.ny + 1, dtype=np.int32),
            np.arange(self.mesh.nz + 1, dtype=np.int32),
            indexing="ij",
        )
        xxp, yyp, zzp = np.meshgrid(self.mesh.x_nodes, self.mesh.y_nodes, self.mesh.z_nodes, indexing="ij")
        node_conc_data = np.column_stack(
            (
                ip.ravel(), jp.ravel(), kp.ravel(),
                xxp.ravel(), yyp.ravel(), zzp.ravel(),
                n_node_m3.ravel(), (n_node_m3 / 1.0e6).ravel(),
            )
        )
        np.savetxt(
            os.path.join(snapshot_dir, "electron_concentration_nodes.txt"),
            node_conc_data,
            header="i j k x_node(m) y_node(m) z_node(m) n(1/m^3) n(1/cm^3)",
            fmt=["%d", "%d", "%d"] + ["%.6e"] * 5,
        )

        phi_real = np.asarray(self.mesh.node_potential_real, dtype=np.float64)
        pot_data = np.column_stack(
            (
                ip.ravel(), jp.ravel(), kp.ravel(),
                xxp.ravel(), yyp.ravel(), zzp.ravel(),
                phi_real.ravel(),
            )
        )
        np.savetxt(
            os.path.join(snapshot_dir, "potential_nodes.txt"),
            pot_data,
            header="i j k x_node(m) y_node(m) z_node(m) potential(V)",
            fmt=["%d", "%d", "%d"] + ["%.6e"] * 4,
        )

        with open(os.path.join(snapshot_dir, "snapshot_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"step={step + 1}\n")
            f.write(f"time_fs={float(time_fs):.6f}\n")

        self._export_particle_snapshot(snapshot_dir)

    def _accumulate_monitor_output_block(self) -> None:
        if self.step_monitor_charge_c.size <= 0:
            return
        self.output_monitor_charge_c += self.step_monitor_charge_c
        self.output_monitor_crossings += self.step_monitor_crossings
        self.output_monitor_step_count += 1

    def _write_current_monitor_output_block(self, step: int, time_fs: float, dt_s: float) -> None:
        if self.monitor_currents_csv_path is None or self.step_monitor_charge_c.size <= 0:
            return
        avg_dt = max(float(dt_s) * max(int(self.output_monitor_step_count), 1), 1.0e-30)
        currents_a = self.output_monitor_charge_c / avg_dt
        row = [str(int(step)), f"{float(time_fs):.6f}", str(int(self.output_monitor_step_count))]
        for idx in range(currents_a.size):
            row.append(f"{float(currents_a[idx]):.12e}")
            row.append(str(int(self.output_monitor_crossings[idx])))
        with open(self.monitor_currents_csv_path, "a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")
        self.output_monitor_charge_c.fill(0.0)
        self.output_monitor_crossings.fill(0)
        self.output_monitor_step_count = 0

    def _prepare_compiled_band_tables(self) -> None:
        zero_1d = np.zeros(0, dtype=np.float64)
        zero_2d = np.zeros((0, 0), dtype=np.float64)
        zero_3d = np.zeros((0, 0, 0), dtype=np.float64)
        zero_4d = np.zeros((0, 0, 0, 3), dtype=np.float64)

        if self.band_struct is None:
            self.velocity_grid_real_kernel = zero_4d
            self.kx_tick_boundaries_kernel = zero_1d
            self.ky_tick_boundaries_kernel = zero_1d
            self.kz_tick_boundaries_kernel = zero_1d
            self.total_energy_grid_eV_kernel = zero_1d
            self.total_rate_norm_kernel = zero_1d
            self.component_rate_norm_kernel = np.zeros((5, 0), dtype=np.float64)
            self.phonon_cdf_kernel = np.zeros((5, 0, 0), dtype=np.float64)
            self.hw_min_limit_kernel = np.zeros(5, dtype=np.float64)
            self.hw_max_limit_kernel = np.zeros(5, dtype=np.float64)
            self.phonon_emin_eV_kernel = 0.0
            self.phonon_dtable_eV_kernel = 0.0
            self.analytic_num_bins_kernel = 0
            self.analytic_bin_piecewise_kernel = (0.0, 0.0, 1.0, 1.0, 0)
            self.analytic_ntlist_kernel = np.zeros(0, dtype=np.int32)
            self.analytic_ptlist_kernel = np.zeros(0, dtype=np.int32)
            self.analytic_tlist_kernel = np.zeros(0, dtype=np.int32)
            self.analytic_wcdf_kernel = zero_1d
            self.analytic_wsum_kernel = zero_1d
            self.analytic_nonempty_bins_kernel = np.zeros(0, dtype=np.int32)
            self.ek_k_norm_kernel = np.zeros((0, 3), dtype=np.float64)
            self.ek_k_pi_kernel = np.zeros((0, 3), dtype=np.float64)
            self.ek_energy_eV_kernel = zero_1d
            return

        band = self.band_struct
        self.velocity_grid_real_kernel = np.asarray(
            getattr(band, "velocity_grid_real", zero_4d), dtype=np.float64
        )
        self.kx_tick_boundaries_kernel = np.asarray(
            getattr(band, "kx_tick_boundaries", zero_1d), dtype=np.float64
        )
        self.ky_tick_boundaries_kernel = np.asarray(
            getattr(band, "ky_tick_boundaries", zero_1d), dtype=np.float64
        )
        self.kz_tick_boundaries_kernel = np.asarray(
            getattr(band, "kz_tick_boundaries", zero_1d), dtype=np.float64
        )

        total_rate_norm = np.asarray(
            getattr(getattr(band, "scattering_rate", None), "get", lambda *_: zero_1d)("total", zero_1d),
            dtype=np.float64,
        )
        self.total_rate_norm_kernel = total_rate_norm
        if total_rate_norm.size > 0:
            emin = float(getattr(band, "emin", 0.0))
            dtable = float(getattr(band, "dtable", 0.0))
            eV0 = float(getattr(band, "eV0", 0.0))
            self.total_energy_grid_eV_kernel = (emin + np.arange(total_rate_norm.size, dtype=float) * dtable) * eV0
            self.phonon_emin_eV_kernel = emin * eV0
            self.phonon_dtable_eV_kernel = dtable * eV0
        else:
            self.total_energy_grid_eV_kernel = zero_1d
            self.phonon_emin_eV_kernel = 0.0
            self.phonon_dtable_eV_kernel = 0.0

        scat = getattr(band, "scattering_rate", None) or {}
        self.component_rate_norm_kernel = np.asarray(scat.get("components", np.zeros((5, 0))), dtype=np.float64)
        self.phonon_cdf_kernel = np.asarray(scat.get("cdf", np.zeros((5, 0, 0))), dtype=np.float64)
        self.hw_min_limit_kernel = np.asarray(getattr(band, "hw_min_limit", np.zeros(5)), dtype=np.float64)
        self.hw_max_limit_kernel = np.asarray(getattr(band, "hw_max_limit", np.zeros(5)), dtype=np.float64)

        piecewise = getattr(band, "analytic_bin_piecewise", None)
        if piecewise is None:
            self.analytic_num_bins_kernel = 0
            self.analytic_bin_piecewise_kernel = (0.0, 0.0, 1.0, 1.0, 0)
        else:
            self.analytic_num_bins_kernel = int(getattr(band, "analytic_num_bins", 0))
            self.analytic_bin_piecewise_kernel = (
                float(piecewise[0]),
                float(piecewise[1]),
                float(piecewise[2]),
                float(piecewise[3]),
                int(piecewise[4]),
            )

        self.analytic_ntlist_kernel = np.asarray(getattr(band, "analytic_ntlist", np.zeros(0)), dtype=np.int32)
        self.analytic_ptlist_kernel = np.asarray(getattr(band, "analytic_ptlist", np.zeros(0)), dtype=np.int32)
        self.analytic_tlist_kernel = np.asarray(getattr(band, "analytic_tlist", np.zeros(0)), dtype=np.int32)
        self.analytic_wcdf_kernel = np.asarray(getattr(band, "analytic_wcdf", zero_1d), dtype=np.float64)
        self.analytic_wsum_kernel = np.asarray(getattr(band, "analytic_wsum", zero_1d), dtype=np.float64)
        self.analytic_nonempty_bins_kernel = np.asarray(
            getattr(band, "analytic_nonempty_bins", np.zeros(0)), dtype=np.int32
        )

        ek_data = getattr(band, "ek_data", {}) or {}
        self.ek_k_norm_kernel = np.asarray(ek_data.get("k_norm", np.zeros((0, 3))), dtype=np.float64)
        self.ek_k_pi_kernel = np.asarray(ek_data.get("k_pi", np.zeros((0, 3))), dtype=np.float64)
        self.ek_energy_eV_kernel = np.asarray(ek_data.get("energy_eV", zero_1d), dtype=np.float64)

    @staticmethod
    def _cell_hit_dir_to_face_label(hit_dir: int) -> str | None:
        mapping = {
            0: "+X",
            1: "-X",
            2: "+Y",
            3: "-Y",
            4: "+Z",
            5: "-Z",
        }
        return mapping.get(int(hit_dir))

    def _material_name_from_id(self, mat_id: int) -> str:
        for label, idx in self.mesh.label_map.items():
            if int(idx) == int(mat_id):
                return str(label)
        return f"id={mat_id}"

    def _assert_particle_in_mc_region(
        self,
        i: int,
        j: int,
        k: int,
        par_idx: int,
        x: float,
        y: float,
        z: float,
    ) -> None:
        mc_label = str(self.phys_config.get("material", "")).upper()
        mc_mat_id = self.mesh.label_map.get(mc_label, None)
        if mc_mat_id is None:
            return
        if i < 0 or i >= self.mesh.nx or j < 0 or j >= self.mesh.ny or k < 0 or k >= self.mesh.nz:
            raise RuntimeError(
                f"Particle moved outside mesh before MC-region check: par_idx={par_idx}, "
                f"cell=({i},{j},{k}), pos=({x:.6e},{y:.6e},{z:.6e})"
            )

        mat_id = int(self.mesh.material_id[i, j, k])
        if mat_id != int(mc_mat_id):
            raise RuntimeError(
                "Particle entered non-MC region: "
                f"par_idx={par_idx}, cell=({i},{j},{k}), "
                f"material={self._material_name_from_id(mat_id)}, "
                f"expected={mc_label}, "
                f"pos=({x:.6e},{y:.6e},{z:.6e})"
            )

    def _record_escape(self, hit_dir: int) -> None:
        face = self._cell_hit_dir_to_face_label(hit_dir)
        self.particle_event_stats["escape_total"] += 1
        if face is not None:
            key = f"escape_{face}"
            if key in self.particle_event_stats:
                self.particle_event_stats[key] += 1

    def _update_electric_field_from_potential(self) -> None:
        """
        Update electric field from node potential in real units.
        - Node-centered field is kept for compatibility.
        - Cell-centered field is built directly from node potential and used by MC.
        """
        phi_real = self.mesh.node_potential_real
        if phi_real is None:
            raise ValueError("node_potential_real is not initialized.")

        edge_order = 2
        if min(phi_real.shape) < 3:
            edge_order = 1

        dphi_dx, dphi_dy, dphi_dz = np.gradient(
            phi_real,
            self.mesh.x_nodes,
            self.mesh.y_nodes,
            self.mesh.z_nodes,
            edge_order=edge_order,
        )
        self.mesh.node_electric_field_x_real = -dphi_dx
        self.mesh.node_electric_field_y_real = -dphi_dy
        self.mesh.node_electric_field_z_real = -dphi_dz

        dx = np.asarray(self.mesh.dx, dtype=float)[:, None, None]
        dy = np.asarray(self.mesh.dy, dtype=float)[None, :, None]
        dz = np.asarray(self.mesh.dz, dtype=float)[None, None, :]

        # Cell-centered field from nodal potential differences averaged over the
        # four edges parallel to each axis in the hexahedral cell.
        ex_cell = -0.25 * (
            (phi_real[1:, :-1, :-1] - phi_real[:-1, :-1, :-1])
            + (phi_real[1:, 1:, :-1] - phi_real[:-1, 1:, :-1])
            + (phi_real[1:, :-1, 1:] - phi_real[:-1, :-1, 1:])
            + (phi_real[1:, 1:, 1:] - phi_real[:-1, 1:, 1:])
        ) / dx
        ey_cell = -0.25 * (
            (phi_real[:-1, 1:, :-1] - phi_real[:-1, :-1, :-1])
            + (phi_real[1:, 1:, :-1] - phi_real[1:, :-1, :-1])
            + (phi_real[:-1, 1:, 1:] - phi_real[:-1, :-1, 1:])
            + (phi_real[1:, 1:, 1:] - phi_real[1:, :-1, 1:])
        ) / dy
        ez_cell = -0.25 * (
            (phi_real[:-1, :-1, 1:] - phi_real[:-1, :-1, :-1])
            + (phi_real[1:, :-1, 1:] - phi_real[1:, :-1, :-1])
            + (phi_real[:-1, 1:, 1:] - phi_real[:-1, 1:, :-1])
            + (phi_real[1:, 1:, 1:] - phi_real[1:, 1:, :-1])
        ) / dz

        self.mesh.cell_electric_field_x_real = ex_cell
        self.mesh.cell_electric_field_y_real = ey_cell
        self.mesh.cell_electric_field_z_real = ez_cell

    def initialize(self) -> None:
        print("[Init] Potential distribution")
        self.init_pot_distribution()

        print("[Init] Particle ensemble")
        self.particle_ensemble = Particle(
            self.mesh,
            self.config,
            self.phys_config,
            self.band_struct,
            self.output_root,
        )

    def init_pot_distribution(self) -> None:
        """
        Initialize node potential for the MC stage.
        Prefer TCAD input; otherwise leave a Poisson TODO placeholder.
        """
        shape = (self.mesh.nx + 1, self.mesh.ny + 1, self.mesh.nz + 1)
        pot0 = float(self.phys_config["scales"]["pot0_V"])

        pot_file = self.config.get("InitialPotentialFile")
        pot_path = _resolve_input_path(pot_file, self.input_dir)

        if pot_path and os.path.isfile(pot_path):
            phi_real = _load_external_potential_to_nodes(pot_path, self.mesh)
            phi_norm = phi_real / pot0

            self.mesh.node_potential_real = phi_real
            self.mesh.node_potential_norm = phi_norm
            self.poisson_solver.phi = phi_norm.ravel().copy()
            self._update_electric_field_from_potential()
            self.initial_potential_from_file = True

            print(
                "      -> Potential ready from external file. "
                f"range=[{np.min(phi_real):.4e}, {np.max(phi_real):.4e}] V"
            )
            return

        if pot_path:
            print(f"      [Warning] Initial potential file not found: {pot_path}")
        else:
            print("      -> No initial potential file specified.")

        print("      [TODO] Initial Poisson solve is not implemented yet. Using zero potential.")
        self.mesh.node_potential_real = np.zeros(shape, dtype=float)
        self.mesh.node_potential_norm = np.zeros(shape, dtype=float)
        self.poisson_solver.phi = self.mesh.node_potential_norm.ravel().copy()
        self._update_electric_field_from_potential()

    def iterate_poisson(self) -> None:
        """
        Handle the Poisson stage for the current MC iteration.
        If TCAD potential is provided, keep potential/electric field fixed.
        Otherwise leave a Poisson-solve TODO placeholder.
        """
        if self.initial_potential_from_file:
            if not self._poisson_mode_notice_shown:
                print("[Poisson] External potential is active; keeping potential/electric field fixed.")
                self._poisson_mode_notice_shown = True
            return

        if not self._poisson_mode_notice_shown:
            print("[Poisson] [TODO] No external potential file. Poisson solve is not implemented yet.")
            self._poisson_mode_notice_shown = True

    def _sample_local_field_real(self, i: int, j: int, k: int) -> tuple[float, float, float]:
        """
        Return the local cell-centered electric field in real units.
        """
        ex_cell = self.mesh.cell_electric_field_x_real
        ey_cell = self.mesh.cell_electric_field_y_real
        ez_cell = self.mesh.cell_electric_field_z_real
        if ex_cell is None or ey_cell is None or ez_cell is None:
            return 0.0, 0.0, 0.0

        return (
            float(ex_cell[i, j, k]),
            float(ey_cell[i, j, k]),
            float(ez_cell[i, j, k]),
        )

    def _evaluate_particle_velocity_real(
        self,
        kx_idx: int,
        ky_idx: int,
        kz_idx: int,
    ) -> tuple[float, float, float]:
        """
        Evaluate particle velocity from the structured k-grid indices.
        Energy is not refreshed here; it is evolved only by drift or scattering.
        """
        if self.band_struct is None:
            return 0.0, 0.0, 0.0

        vel_grid = getattr(self.band_struct, "velocity_grid_real", None)
        if vel_grid is None:
            return 0.0, 0.0, 0.0

        ix = int(np.clip(kx_idx, 0, self.band_struct.num_ticks_x - 1))
        iy = int(np.clip(ky_idx, 0, self.band_struct.num_ticks_y - 1))
        iz = int(np.clip(kz_idx, 0, self.band_struct.num_ticks_z - 1))
        vel = vel_grid[ix, iy, iz]
        return float(vel[0]), float(vel[1]), float(vel[2])

    def _compute_kgrid_time(
        self,
        par_idx: int,
        x: float,
        y: float,
        z: float,
        kx: float,
        ky: float,
        kz: float,
        vx: float,
        vy: float,
        vz: float,
        ex: float,
        ey: float,
        ez: float,
        energy: float,
        kx_idx: int,
        ky_idx: int,
        kz_idx: int,
    ) -> float:
        """
        Compute time to the nearest k-space cell boundary.

        The physical relation is:
          hbar * dk/dt = F = q * E
          t_hit = delta_k * hbar / F

        Here we evaluate everything in SI units first. The current particle
        driver (`left_time`, `dt`) still uses seconds, so this function returns
        time in seconds for consistency. The normalized time is stored in
        `self.last_k_col_time_norm` for future use when the MC core is fully
        normalized.
        """
        _ = (x, y, z, vx, vy, vz, energy)
        if self.particle_ensemble is None:
            raise RuntimeError("Particle ensemble is not initialized.")
        if self.band_struct is None:
            self.last_k_col_dir = -1
            self.last_k_col_time_real = np.inf
            self.last_k_col_time_norm = np.inf
            return np.inf

        hbar = float(self.phys_config["hbar"])
        q_e = float(self.phys_config["q_e"])
        a_real = float(self.phys_config["sia0_real"])
        spr0 = float(self.phys_config["scales"]["spr0"])
        time0 = float(self.phys_config["scales"]["time0"])

        par = self.particle_ensemble
        charge_sign = float(np.sign(par.charge[par_idx]))
        if charge_sign == 0.0:
            charge_sign = -1.0

        k_real_scale = 1.0 / spr0
        boundary_real_scale = np.pi / a_real
        kx_real = float(kx) * k_real_scale
        ky_real = float(ky) * k_real_scale
        kz_real = float(kz) * k_real_scale

        # Force is determined by the particle charge sign:
        # electron -> F = -qE, hole -> F = +qE.
        fx_real = charge_sign * q_e * float(ex)
        fy_real = charge_sign * q_e * float(ey)
        fz_real = charge_sign * q_e * float(ez)

        dkx_dt_real = fx_real / hbar
        dky_dt_real = fy_real / hbar
        dkz_dt_real = fz_real / hbar

        time_real, hit_dir, time_norm = _compute_kgrid_time_kernel(
            float(kx_real),
            float(ky_real),
            float(kz_real),
            float(dkx_dt_real),
            float(dky_dt_real),
            float(dkz_dt_real),
            int(kx_idx),
            int(ky_idx),
            int(kz_idx),
            np.asarray(self.band_struct.kx_tick_boundaries, dtype=float),
            np.asarray(self.band_struct.ky_tick_boundaries, dtype=float),
            np.asarray(self.band_struct.kz_tick_boundaries, dtype=float),
            float(boundary_real_scale),
            float(time0),
        )

        self.last_k_col_dir = int(hit_dir)
        self.last_k_col_time_real = float(time_real)
        self.last_k_col_time_norm = float(time_norm)

        return float(time_real)

    def _compute_cell_time(
        self,
        par_idx: int,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        i: int,
        j: int,
        k: int,
    ) -> float:
        """
        Compute time to the nearest real-space cell boundary.

        The current particle coordinates are in meters and velocities are in m/s,
        so the returned value is in seconds. The normalized time is stored in
        `self.last_cell_col_time_norm` for future use.
        """
        _ = par_idx
        time0 = float(self.phys_config["scales"]["time0"])

        time_real, hit_dir, time_norm = _compute_cell_time_kernel(
            float(x),
            float(y),
            float(z),
            float(vx),
            float(vy),
            float(vz),
            float(self.mesh.x_nodes[i]),
            float(self.mesh.x_nodes[i + 1]),
            float(self.mesh.y_nodes[j]),
            float(self.mesh.y_nodes[j + 1]),
            float(self.mesh.z_nodes[k]),
            float(self.mesh.z_nodes[k + 1]),
            float(time0),
        )

        self.last_cell_col_dir = int(hit_dir)
        self.last_cell_col_time_real = float(time_real)
        self.last_cell_col_time_norm = float(time_norm)

        return float(time_real)

    def _compute_phonon_scatter_time(
        self,
        par_idx: int,
        energy: float,
        kx_idx: int,
        ky_idx: int,
        kz_idx: int,
    ) -> tuple[float, float]:
        """
        Sample next phonon-scattering time from the per-k-cell upper-bound rate.
        Returns (time_real_s, -log(rand)).
        """
        _ = (par_idx, energy)
        if self.band_struct is None or self.kcell_max_phsr_real is None:
            return np.inf, 0.0

        gamma_ph_real = float(self.kcell_max_phsr_real[int(kx_idx), int(ky_idx), int(kz_idx)])
        if gamma_ph_real <= 1.0e-30 or not np.isfinite(gamma_ph_real):
            return np.inf, 0.0

        rand = max(float(np.random.random()), np.finfo(float).tiny)
        phrnl = -np.log(rand)
        return float(phrnl / gamma_ph_real), float(phrnl)

    def _compute_impurity_scatter_time(
        self,
        par_idx: int,
        energy: float,
        doping: float,
        charge_density: float,
    ) -> tuple[float, float]:
        """
        Placeholder impurity-scattering clock.
        Returns (time, random_log_cache).
        """
        return compute_impurity_scatter_time(self, par_idx, energy, doping, charge_density)

    def _compute_surface_scatter_time(
        self,
        par_idx: int,
        x: float,
        y: float,
        z: float,
        i: int,
        j: int,
        k: int,
    ) -> tuple[float, float]:
        """
        Placeholder surface-scattering clock.
        Returns (time, random_log_cache).
        """
        return compute_surface_scatter_time(self, par_idx, x, y, z, i, j, k)

    def _select_next_event(
        self,
        tet_tf: float,
        cell_tf: float,
        ph_tf: float,
        imp_tf: float,
        surf_tf: float,
        left_time: float,
    ) -> tuple[float, int]:
        """
        Select the next particle event from candidate clocks.
        """
        return _select_next_event_kernel(
            float(tet_tf),
            float(cell_tf),
            float(ph_tf),
            float(imp_tf),
            float(surf_tf),
            float(left_time),
        )

    def _advance_particle_drift(
        self,
        par_idx: int,
        x: float,
        y: float,
        z: float,
        kx: float,
        ky: float,
        kz: float,
        energy: float,
        vx: float,
        vy: float,
        vz: float,
        ex: float,
        ey: float,
        ez: float,
        tf: float,
        phrnl: float,
        imprnl: float,
        ssnl: float,
        ph_gamma_real: float,
        imp_gamma_real: float,
        surf_gamma_real: float,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        """
        Advance real-space and k-space coordinates over free-flight time tf.
        Field/force updates are evaluated in SI units first, then mapped back to
        the code's normalized k representation.
        """
        if self.particle_ensemble is None:
            raise RuntimeError("Particle ensemble is not initialized.")

        hbar = float(self.phys_config["hbar"])
        q_e = float(self.phys_config["q_e"])
        spr0 = float(self.phys_config["scales"]["spr0"])

        charge_sign = float(np.sign(self.particle_ensemble.charge[par_idx]))
        if charge_sign == 0.0:
            charge_sign = -1.0

        return _advance_particle_drift_kernel(
            float(x),
            float(y),
            float(z),
            float(kx),
            float(ky),
            float(kz),
            float(energy),
            float(vx),
            float(vy),
            float(vz),
            float(ex),
            float(ey),
            float(ez),
            float(tf),
            float(phrnl),
            float(imprnl),
            float(ssnl),
            float(ph_gamma_real),
            float(imp_gamma_real),
            float(surf_gamma_real),
            float(charge_sign),
            float(hbar),
            float(q_e),
            float(spr0),
        )

    def _wrap_k_axis_periodic(
        self,
        k_norm: float,
        k_idx: int,
        ticks_pi: np.ndarray | None,
    ) -> tuple[float, int]:
        """
        Apply periodic wrap on one k-axis when the index exits the sampled BZ.

        To keep the state self-consistent:
        - the discrete k-cell index is wrapped to the opposite side,
        - the continuous k value is shifted by one full BZ period on that axis.
        """
        if ticks_pi is None:
            return float(k_norm), int(k_idx)

        n_ticks = int(len(ticks_pi))
        if n_ticks <= 0:
            return float(k_norm), int(k_idx)

        k_period_norm = float(ticks_pi[-1] - ticks_pi[0]) * np.pi / float(self.phys_config["sia0_norm"])
        if k_idx >= n_ticks:
            return float(k_norm - k_period_norm), 0
        if k_idx < 0:
            return float(k_norm + k_period_norm), n_ticks - 1
        return float(k_norm), int(k_idx)

    @staticmethod
    def _point_in_nm_bounds(bounds_nm: list, x: float, y: float, z: float, tol: float = 1.0e-15) -> bool:
        if len(bounds_nm) != 6:
            return False
        x1, x2, y1, y2, z1, z2 = [float(v) * 1.0e-9 for v in bounds_nm]
        xlo, xhi = (x1, x2) if x1 <= x2 else (x2, x1)
        ylo, yhi = (y1, y2) if y1 <= y2 else (y2, y1)
        zlo, zhi = (z1, z2) if z1 <= z2 else (z2, z1)
        return (
            (xlo - tol) <= x <= (xhi + tol)
            and (ylo - tol) <= y <= (yhi + tol)
            and (zlo - tol) <= z <= (zhi + tol)
        )

    def _resolve_motion_rule(self, x: float, y: float, z: float, hit_dir: int) -> str:
        """
        Resolve real-space boundary action from explicit motionplane rules.
        If no rule matches, default to PASS.
        """
        face_label = self._cell_hit_dir_to_face_label(hit_dir)
        if face_label is None:
            return "PASS"

        for entry in self.device_structure.get("motion_planes", []):
            if str(entry.get("face", "")).upper() != face_label:
                continue
            bounds = entry.get("bounds", [])
            if self._point_in_nm_bounds(bounds, x, y, z):
                return str(entry.get("rule", "PASS")).upper()

        return "PASS"

    @staticmethod
    def _advance_cell_indices(i: int, j: int, k: int, hit_dir: int) -> tuple[int, int, int]:
        if hit_dir == 0:
            return i + 1, j, k
        if hit_dir == 1:
            return i - 1, j, k
        if hit_dir == 2:
            return i, j + 1, k
        if hit_dir == 3:
            return i, j - 1, k
        if hit_dir == 4:
            return i, j, k + 1
        if hit_dir == 5:
            return i, j, k - 1
        return i, j, k

    def _reflect_k_state(
        self,
        kx: float,
        ky: float,
        kz: float,
        kx_idx: int,
        ky_idx: int,
        kz_idx: int,
        hit_dir: int,
    ) -> tuple[float, float, float, int, int, int]:
        """
        Specular reflection placeholder: only reverse the k component normal to
        the hit face, then recompute structured k-cell indices from the
        reflected continuous k.
        """
        kx_ref = float(kx)
        ky_ref = float(ky)
        kz_ref = float(kz)

        if hit_dir in (0, 1):
            kx_ref = -kx_ref
        elif hit_dir in (2, 3):
            ky_ref = -ky_ref
        elif hit_dir in (4, 5):
            kz_ref = -kz_ref

        if self.band_struct is None:
            return (
                kx_ref,
                ky_ref,
                kz_ref,
                int(kx_idx),
                int(ky_idx),
                int(kz_idx),
            )

        to_pi = float(self.phys_config["sia0_norm"]) / np.pi
        kx_pi = kx_ref * to_pi
        ky_pi = ky_ref * to_pi
        kz_pi = kz_ref * to_pi
        kx_idx_new = int(np.searchsorted(self.band_struct.kx_tick_boundaries, kx_pi, side="right") - 1)
        ky_idx_new = int(np.searchsorted(self.band_struct.ky_tick_boundaries, ky_pi, side="right") - 1)
        kz_idx_new = int(np.searchsorted(self.band_struct.kz_tick_boundaries, kz_pi, side="right") - 1)
        kx_idx_new = int(np.clip(kx_idx_new, 0, self.band_struct.num_ticks_x - 1))
        ky_idx_new = int(np.clip(ky_idx_new, 0, self.band_struct.num_ticks_y - 1))
        kz_idx_new = int(np.clip(kz_idx_new, 0, self.band_struct.num_ticks_z - 1))

        return kx_ref, ky_ref, kz_ref, kx_idx_new, ky_idx_new, kz_idx_new

    def _append_generated_particle(
        self,
        par_idx: int,
        x: float,
        y: float,
        z: float,
        kx: float,
        ky: float,
        kz: float,
        energy: float,
        kx_idx: int,
        ky_idx: int,
        kz_idx: int,
        left_time: float,
        i: int,
        j: int,
        k: int,
        hit_dir: int,
    ) -> bool:
        """
        Duplicate the current particle state and let the new particle execute a
        PASS transition into the neighboring cell.
        """
        if self.particle_ensemble is None:
            return False

        new_i, new_j, new_k = self._advance_cell_indices(
            int(i),
            int(j),
            int(k),
            int(hit_dir),
        )
        if (
            new_i < 0 or new_i >= self.mesh.nx
            or new_j < 0 or new_j >= self.mesh.ny
            or new_k < 0 or new_k >= self.mesh.nz
        ):
            return False

        self._assert_particle_in_mc_region(
            new_i,
            new_j,
            new_k,
            int(par_idx),
            float(x),
            float(y),
            float(z),
        )

        par = self.particle_ensemble

        def _append_scalar_array(name: str, value) -> None:
            arr = getattr(par, name)
            dtype = arr.dtype
            setattr(par, name, np.concatenate((arr, np.asarray([value], dtype=dtype))))

        _append_scalar_array("x", float(x))
        _append_scalar_array("y", float(y))
        _append_scalar_array("z", float(z))
        _append_scalar_array("kx", float(kx))
        _append_scalar_array("ky", float(ky))
        _append_scalar_array("kz", float(kz))
        _append_scalar_array("energy", float(energy))
        _append_scalar_array("charge", float(par.charge[int(par_idx)]))
        _append_scalar_array("i", int(new_i))
        _append_scalar_array("j", int(new_j))
        _append_scalar_array("k", int(new_k))
        _append_scalar_array("kx_idx", int(kx_idx))
        _append_scalar_array("ky_idx", int(ky_idx))
        _append_scalar_array("kz_idx", int(kz_idx))
        _append_scalar_array("left_time", float(left_time))

        if hasattr(par, "seed") and par.seed is not None:
            next_seed = int(par.seed[-1]) + 1 if par.seed.size > 0 else 0
            _append_scalar_array("seed", next_seed)

        if hasattr(par, "flag") and par.flag is not None:
            _append_scalar_array("flag", 0)

        par.size = int(par.size) + 1
        self.gen_par += 1
        self.particle_event_stats["generate"] += 1
        return True

    def _append_particle_state(
        self,
        parent_idx: int,
        x: float,
        y: float,
        z: float,
        kx: float,
        ky: float,
        kz: float,
        energy: float,
        kx_idx: int,
        ky_idx: int,
        kz_idx: int,
        left_time: float,
        i: int,
        j: int,
        k: int,
    ) -> bool:
        if self.particle_ensemble is None:
            return False
        if (
            i < 0 or i >= self.mesh.nx
            or j < 0 or j >= self.mesh.ny
            or k < 0 or k >= self.mesh.nz
        ):
            return False

        self._assert_particle_in_mc_region(
            int(i),
            int(j),
            int(k),
            int(parent_idx),
            float(x),
            float(y),
            float(z),
        )

        par = self.particle_ensemble

        def _append_scalar_array(name: str, value) -> None:
            arr = getattr(par, name)
            dtype = arr.dtype
            setattr(par, name, np.concatenate((arr, np.asarray([value], dtype=dtype))))

        _append_scalar_array("x", float(x))
        _append_scalar_array("y", float(y))
        _append_scalar_array("z", float(z))
        _append_scalar_array("kx", float(kx))
        _append_scalar_array("ky", float(ky))
        _append_scalar_array("kz", float(kz))
        _append_scalar_array("energy", float(energy))
        _append_scalar_array("charge", float(par.charge[int(parent_idx)]))
        _append_scalar_array("i", int(i))
        _append_scalar_array("j", int(j))
        _append_scalar_array("k", int(k))
        _append_scalar_array("kx_idx", int(kx_idx))
        _append_scalar_array("ky_idx", int(ky_idx))
        _append_scalar_array("kz_idx", int(kz_idx))
        _append_scalar_array("left_time", float(left_time))

        if hasattr(par, "seed") and par.seed is not None:
            next_seed = int(par.seed[-1]) + 1 if par.seed.size > 0 else 0
            _append_scalar_array("seed", next_seed)

        if hasattr(par, "flag") and par.flag is not None:
            _append_scalar_array("flag", 0)

        par.size = int(par.size) + 1
        self.gen_par += 1
        self.particle_event_stats["generate"] += 1
        return True

    def _append_particle_states_batch(
        self,
        parent_indices: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        kx: np.ndarray,
        ky: np.ndarray,
        kz: np.ndarray,
        energy: np.ndarray,
        kx_idx: np.ndarray,
        ky_idx: np.ndarray,
        kz_idx: np.ndarray,
        left_time: np.ndarray,
        i: np.ndarray,
        j: np.ndarray,
        k: np.ndarray,
    ) -> int:
        if self.particle_ensemble is None:
            return 0

        n_new = int(parent_indices.size)
        if n_new <= 0:
            return 0

        valid = (
            (i >= 0) & (i < self.mesh.nx)
            & (j >= 0) & (j < self.mesh.ny)
            & (k >= 0) & (k < self.mesh.nz)
        )
        if not np.all(valid):
            parent_indices = parent_indices[valid]
            x = x[valid]
            y = y[valid]
            z = z[valid]
            kx = kx[valid]
            ky = ky[valid]
            kz = kz[valid]
            energy = energy[valid]
            kx_idx = kx_idx[valid]
            ky_idx = ky_idx[valid]
            kz_idx = kz_idx[valid]
            left_time = left_time[valid]
            i = i[valid]
            j = j[valid]
            k = k[valid]
            n_new = int(parent_indices.size)
            if n_new <= 0:
                return 0

        mc_label = str(self.phys_config.get("material", "")).upper()
        mc_mat_id = int(self.mesh.label_map.get(mc_label, -1))
        mat_ok = self.mesh.material_id[i, j, k] == mc_mat_id
        if not np.all(mat_ok):
            bad_idx = int(np.flatnonzero(~mat_ok)[0])
            self._assert_particle_in_mc_region(
                int(i[bad_idx]),
                int(j[bad_idx]),
                int(k[bad_idx]),
                int(parent_indices[bad_idx]),
                float(x[bad_idx]),
                float(y[bad_idx]),
                float(z[bad_idx]),
            )

        par = self.particle_ensemble

        def _append_array(name: str, values: np.ndarray) -> None:
            arr = getattr(par, name)
            setattr(par, name, np.concatenate((arr, np.asarray(values, dtype=arr.dtype))))

        _append_array("x", x)
        _append_array("y", y)
        _append_array("z", z)
        _append_array("kx", kx)
        _append_array("ky", ky)
        _append_array("kz", kz)
        _append_array("energy", energy)
        _append_array("charge", par.charge[parent_indices])
        _append_array("i", i)
        _append_array("j", j)
        _append_array("k", k)
        _append_array("kx_idx", kx_idx)
        _append_array("ky_idx", ky_idx)
        _append_array("kz_idx", kz_idx)
        _append_array("left_time", left_time)

        if hasattr(par, "seed") and par.seed is not None:
            start_seed = int(par.seed[-1]) + 1 if par.seed.size > 0 else 0
            seeds = np.arange(start_seed, start_seed + n_new, dtype=par.seed.dtype)
            _append_array("seed", seeds)

        if hasattr(par, "flag") and par.flag is not None:
            flags = np.zeros(n_new, dtype=par.flag.dtype)
            _append_array("flag", flags)

        par.size = int(par.size) + n_new
        self.gen_par += n_new
        self.particle_event_stats["generate"] += n_new
        return n_new

    def _handle_particle_event(
        self,
        par_idx: int,
        event_flag: int,
        x: float,
        y: float,
        z: float,
        kx: float,
        ky: float,
        kz: float,
        energy: float,
        i: int,
        j: int,
        k: int,
        kx_idx: int,
        ky_idx: int,
        kz_idx: int,
        left_time: float,
        phrnl: float,
        imprnl: float,
        ssnl: float,
    ) -> tuple[
        float, float, float,
        float, float, float,
        float,
        int, int, int,
        int, int, int,
        float, float, float, float,
        bool, bool, bool, bool, bool, bool, bool
    ]:
        """
        Event dispatcher without dict allocation in the hot path.
        """
        refresh_all_spatial = event_flag in (
            EVENT_HIT_KGRID,
            EVENT_HIT_CELL,
            EVENT_PHONON_SCATTER,
            EVENT_IMPURITY_SCATTER,
            EVENT_SURFACE_SCATTER,
        )
        flag_catch = False
        fly_too_far = False
        refresh_tet_time = refresh_all_spatial
        refresh_cell_time = refresh_all_spatial
        refresh_ph_time = event_flag == EVENT_PHONON_SCATTER
        refresh_imp_time = event_flag == EVENT_IMPURITY_SCATTER
        refresh_surf_time = event_flag == EVENT_SURFACE_SCATTER

        if event_flag == EVENT_HIT_KGRID:
            hit_dir = int(self.last_k_col_dir)
            if hit_dir == 0:
                kx_idx += 1
            elif hit_dir == 1:
                kx_idx -= 1
            elif hit_dir == 2:
                ky_idx += 1
            elif hit_dir == 3:
                ky_idx -= 1
            elif hit_dir == 4:
                kz_idx += 1
            elif hit_dir == 5:
                kz_idx -= 1

            if self.band_struct is not None:
                kx, kx_idx = self._wrap_k_axis_periodic(kx, kx_idx, self.band_struct.kx_ticks_pi)
                ky, ky_idx = self._wrap_k_axis_periodic(ky, ky_idx, self.band_struct.ky_ticks_pi)
                kz, kz_idx = self._wrap_k_axis_periodic(kz, kz_idx, self.band_struct.kz_ticks_pi)
        elif event_flag == EVENT_HIT_CELL:
            hit_dir = int(self.last_cell_col_dir)
            rule = self._resolve_motion_rule(
                float(x),
                float(y),
                float(z),
                hit_dir,
            )

            if rule == "CATCH":
                self.particle_event_stats["catch"] += 1
                flag_catch = True
                return (
                    x, y, z, kx, ky, kz, energy,
                    i, j, k, kx_idx, ky_idx, kz_idx,
                    left_time, phrnl, imprnl, ssnl,
                    flag_catch, fly_too_far,
                    refresh_tet_time, refresh_cell_time, refresh_ph_time, refresh_imp_time, refresh_surf_time,
                )

            if rule == "GENERATE":
                _ = self._append_generated_particle(
                    par_idx, x, y, z, kx, ky, kz, energy,
                    kx_idx, ky_idx, kz_idx, left_time, i, j, k, hit_dir,
                )
                kx, ky, kz, kx_idx, ky_idx, kz_idx = self._reflect_k_state(
                    kx, ky, kz, kx_idx, ky_idx, kz_idx, hit_dir
                )
                return (
                    x, y, z, kx, ky, kz, energy,
                    i, j, k, kx_idx, ky_idx, kz_idx,
                    left_time, phrnl, imprnl, ssnl,
                    flag_catch, fly_too_far,
                    refresh_tet_time, refresh_cell_time, refresh_ph_time, refresh_imp_time, refresh_surf_time,
                )

            if rule in {"REFLECT", "SCATTOX"}:
                kx, ky, kz, kx_idx, ky_idx, kz_idx = self._reflect_k_state(
                    kx, ky, kz, kx_idx, ky_idx, kz_idx, hit_dir
                )
                if rule == "SCATTOX":
                    refresh_surf_time = True
                return (
                    x, y, z, kx, ky, kz, energy,
                    i, j, k, kx_idx, ky_idx, kz_idx,
                    left_time, phrnl, imprnl, ssnl,
                    flag_catch, fly_too_far,
                    refresh_tet_time, refresh_cell_time, refresh_ph_time, refresh_imp_time, refresh_surf_time,
                )

            i, j, k = self._advance_cell_indices(i, j, k, hit_dir)

            if i < 0 or i >= self.mesh.nx or j < 0 or j >= self.mesh.ny or k < 0 or k >= self.mesh.nz:
                self._record_escape(hit_dir)
                flag_catch = True
                left_time = 0.0
                return (
                    x, y, z, kx, ky, kz, energy,
                    i, j, k, kx_idx, ky_idx, kz_idx,
                    left_time, phrnl, imprnl, ssnl,
                    flag_catch, fly_too_far,
                    refresh_tet_time, refresh_cell_time, refresh_ph_time, refresh_imp_time, refresh_surf_time,
                )

            self._assert_particle_in_mc_region(
                i,
                j,
                k,
                int(par_idx),
                float(x),
                float(y),
                float(z),
            )
        elif event_flag == EVENT_PHONON_SCATTER:
            new_state = handle_phonon_scatter_event(self, energy, kx_idx, ky_idx, kz_idx)
            if new_state is not None:
                kx, ky, kz, energy, kx_idx, ky_idx, kz_idx = new_state
        elif event_flag == EVENT_IMPURITY_SCATTER:
            handle_impurity_scatter_event(self, par_idx)
        elif event_flag == EVENT_SURFACE_SCATTER:
            handle_surface_scatter_event(self, par_idx)

        return (
            x, y, z, kx, ky, kz, energy,
            i, j, k, kx_idx, ky_idx, kz_idx,
            left_time, phrnl, imprnl, ssnl,
            flag_catch, fly_too_far,
            refresh_tet_time, refresh_cell_time, refresh_ph_time, refresh_imp_time, refresh_surf_time,
        )

    def _particle_fly_single(self, par_idx: int) -> None:
        if self.particle_ensemble is None:
            raise RuntimeError("Particle ensemble is not initialized.")
        if njit is None:
            raise RuntimeError("Numba is required for _particle_fly_single.")

        par = self.particle_ensemble
        band = self.band_struct
        if band is None or self.velocity_grid_real_kernel.size == 0:
            raise RuntimeError("Band structure velocity grid is not initialized for compiled particle flight.")

        mc_label = str(self.phys_config.get("material", "")).upper()
        mc_mat_id = int(self.mesh.label_map.get(mc_label, -1))
        if mc_mat_id < 0:
            raise RuntimeError(f"MC material '{mc_label}' is not present in mesh.label_map.")

        x = float(par.x[par_idx])
        y = float(par.y[par_idx])
        z = float(par.z[par_idx])
        kx = float(par.kx[par_idx])
        ky = float(par.ky[par_idx])
        kz = float(par.kz[par_idx])
        energy = float(par.energy[par_idx])
        i = int(par.i[par_idx])
        j = int(par.j[par_idx])
        k = int(par.k[par_idx])
        kx_idx = int(par.kx_idx[par_idx])
        ky_idx = int(par.ky_idx[par_idx])
        kz_idx = int(par.kz_idx[par_idx])
        left_time = float(par.left_time[par_idx])
        charge = float(par.charge[par_idx])

        sia0_norm = float(self.phys_config["sia0_norm"])
        a_real = float(self.phys_config["sia0_real"])
        hbar = float(self.phys_config["hbar"])
        q_e = float(self.phys_config["q_e"])
        spr0 = float(self.phys_config["scales"]["spr0"])
        time0 = float(self.phys_config["scales"]["time0"])
        max_loops = int(self.config.get("max_particle_subloops", 10000))
        cell_ex = self.mesh.cell_electric_field_x_real
        cell_ey = self.mesh.cell_electric_field_y_real
        cell_ez = self.mesh.cell_electric_field_z_real
        x_nodes = self.mesh.x_nodes
        y_nodes = self.mesh.y_nodes
        z_nodes = self.mesh.z_nodes
        material_id = self.mesh.material_id

        k_period_norm_x = 0.0
        k_period_norm_y = 0.0
        k_period_norm_z = 0.0
        if getattr(band, "kx_ticks_pi", None) is not None and len(band.kx_ticks_pi) > 0:
            k_period_norm_x = float(band.kx_ticks_pi[-1] - band.kx_ticks_pi[0]) * np.pi / sia0_norm
        if getattr(band, "ky_ticks_pi", None) is not None and len(band.ky_ticks_pi) > 0:
            k_period_norm_y = float(band.ky_ticks_pi[-1] - band.ky_ticks_pi[0]) * np.pi / sia0_norm
        if getattr(band, "kz_ticks_pi", None) is not None and len(band.kz_ticks_pi) > 0:
            k_period_norm_z = float(band.kz_ticks_pi[-1] - band.kz_ticks_pi[0]) * np.pi / sia0_norm

        kcell_max = self.kcell_max_phsr_real
        if kcell_max is None:
            kcell_max = np.zeros((0, 0, 0), dtype=np.float64)

        while True:
            out = _particle_fly_single_kernel(
                x, y, z, kx, ky, kz, energy,
                i, j, k,
                kx_idx, ky_idx, kz_idx,
                left_time, charge,
                cell_ex,
                cell_ey,
                cell_ez,
                x_nodes,
                y_nodes,
                z_nodes,
                material_id,
                mc_mat_id,
                self.velocity_grid_real_kernel,
                self.kx_tick_boundaries_kernel,
                self.ky_tick_boundaries_kernel,
                self.kz_tick_boundaries_kernel,
                int(getattr(band, "num_ticks_x", 0)),
                int(getattr(band, "num_ticks_y", 0)),
                int(getattr(band, "num_ticks_z", 0)),
                float(k_period_norm_x),
                float(k_period_norm_y),
                float(k_period_norm_z),
                hbar, q_e, spr0, time0, sia0_norm, a_real,
                np.asarray(kcell_max, dtype=np.float64),
                self.total_energy_grid_eV_kernel,
                self.total_rate_norm_kernel,
                self.component_rate_norm_kernel,
                self.phonon_cdf_kernel,
                self.hw_min_limit_kernel,
                self.hw_max_limit_kernel,
                float(self.phonon_emin_eV_kernel),
                float(self.phonon_dtable_eV_kernel),
                int(self.analytic_num_bins_kernel),
                float(self.analytic_bin_piecewise_kernel[0]),
                float(self.analytic_bin_piecewise_kernel[1]),
                float(self.analytic_bin_piecewise_kernel[2]),
                float(self.analytic_bin_piecewise_kernel[3]),
                int(self.analytic_bin_piecewise_kernel[4]),
                self.analytic_ntlist_kernel,
                self.analytic_ptlist_kernel,
                self.analytic_tlist_kernel,
                self.analytic_wcdf_kernel,
                self.analytic_wsum_kernel,
                self.analytic_nonempty_bins_kernel,
                self.ek_k_norm_kernel,
                self.ek_k_pi_kernel,
                self.ek_energy_eV_kernel,
                self.motion_bounds_m,
                self.motion_faces,
                self.motion_rules,
                self.monitor_bounds_m,
                self.monitor_faces,
                self.step_monitor_charge_c,
                self.step_monitor_crossings,
                max_loops,
            )

            (
                status,
                err_i,
                err_j,
                err_k,
                x,
                y,
                z,
                kx,
                ky,
                kz,
                energy,
                i,
                j,
                k,
                kx_idx,
                ky_idx,
                kz_idx,
                left_time,
                spawn_requested,
                spawn_x,
                spawn_y,
                spawn_z,
                spawn_kx,
                spawn_ky,
                spawn_kz,
                spawn_energy,
                spawn_kx_idx,
                spawn_ky_idx,
                spawn_kz_idx,
                spawn_left_time,
                spawn_i,
                spawn_j,
                spawn_k,
                catch_stat_inc,
                escape_total_inc,
                escape_xp,
                escape_xm,
                escape_yp,
                escape_ym,
                escape_zp,
                escape_zm,
                step_ph_inc,
                phonon_self_inc,
                phonon_acoustic_inc,
                phonon_opt_abs_inc,
                phonon_opt_ems_inc,
                phonon_absorbed_eV,
                phonon_emitted_eV,
            ) = out

            self.step_scatter_stats["phonon"] += int(step_ph_inc)
            self.total_scatter_stats["phonon"] += int(step_ph_inc)
            self.phonon_scatter_stats["self"] += int(phonon_self_inc)
            self.phonon_scatter_stats["acoustic"] += int(phonon_acoustic_inc)
            self.phonon_scatter_stats["optical_abs"] += int(phonon_opt_abs_inc)
            self.phonon_scatter_stats["optical_ems"] += int(phonon_opt_ems_inc)
            self.phonon_scatter_stats["absorbed_eV"] += float(phonon_absorbed_eV)
            self.phonon_scatter_stats["emitted_eV"] += float(phonon_emitted_eV)

            self.particle_event_stats["escape_total"] += int(escape_total_inc)
            self.particle_event_stats["escape_+X"] += int(escape_xp)
            self.particle_event_stats["escape_-X"] += int(escape_xm)
            self.particle_event_stats["escape_+Y"] += int(escape_yp)
            self.particle_event_stats["escape_-Y"] += int(escape_ym)
            self.particle_event_stats["escape_+Z"] += int(escape_zp)
            self.particle_event_stats["escape_-Z"] += int(escape_zm)
            self.particle_event_stats["catch"] += int(catch_stat_inc)

            if spawn_requested:
                self._append_particle_state(
                    int(par_idx),
                    float(spawn_x),
                    float(spawn_y),
                    float(spawn_z),
                    float(spawn_kx),
                    float(spawn_ky),
                    float(spawn_kz),
                    float(spawn_energy),
                    int(spawn_kx_idx),
                    int(spawn_ky_idx),
                    int(spawn_kz_idx),
                    float(spawn_left_time),
                    int(spawn_i),
                    int(spawn_j),
                    int(spawn_k),
                )

            if int(status) == FLY_STATUS_ERROR_NON_MC:
                self._assert_particle_in_mc_region(
                    int(err_i),
                    int(err_j),
                    int(err_k),
                    int(par_idx),
                    float(x),
                    float(y),
                    float(z),
                )
                raise RuntimeError(
                    f"Particle entered invalid region: par_idx={par_idx}, cell=({err_i},{err_j},{err_k})"
                )

            if int(status) == FLY_STATUS_CATCH:
                if int(catch_stat_inc) == 0:
                    self.particle_event_stats["catch"] += 1
                par.i[par_idx] = -9999
                par.left_time[par_idx] = 0.0
                if hasattr(par, "flag") and par.flag is not None:
                    par.flag[par_idx] = 1
                self.catch_par_num += 1
                return

            if int(status) == FLY_STATUS_DONE:
                par.x[par_idx] = x
                par.y[par_idx] = y
                par.z[par_idx] = z
                par.kx[par_idx] = kx
                par.ky[par_idx] = ky
                par.kz[par_idx] = kz
                par.energy[par_idx] = energy
                par.i[par_idx] = i
                par.j[par_idx] = j
                par.k[par_idx] = k
                par.kx_idx[par_idx] = kx_idx
                par.ky_idx[par_idx] = ky_idx
                par.kz_idx[par_idx] = kz_idx
                par.left_time[par_idx] = left_time
                if hasattr(par, "flag") and par.flag is not None:
                    par.flag[par_idx] = 1
                return

            if int(status) != FLY_STATUS_GENERATE:
                raise RuntimeError(f"Unexpected particle_fly kernel status: {status}")

    def particle_fly(self, active_mask: np.ndarray) -> None:
        if self.particle_ensemble is None:
            raise RuntimeError("Particle ensemble is not initialized.")
        if njit is None:
            raise RuntimeError("Numba is required for particle_fly.")
        if self.band_struct is None or self.velocity_grid_real_kernel.size == 0:
            raise RuntimeError("Band structure velocity grid is not initialized for compiled particle flight.")

        if active_mask.size != self.particle_ensemble.size:
            raise ValueError("active_mask shape does not match particle ensemble size.")
        if not np.any(active_mask):
            return

        par = self.particle_ensemble
        band = self.band_struct
        active_indices = np.flatnonzero(active_mask).astype(np.int64)
        n_active = int(active_indices.size)
        if n_active <= 0:
            return

        mc_label = str(self.phys_config.get("material", "")).upper()
        mc_mat_id = int(self.mesh.label_map.get(mc_label, -1))
        if mc_mat_id < 0:
            raise RuntimeError(f"MC material '{mc_label}' is not present in mesh.label_map.")

        sia0_norm = float(self.phys_config["sia0_norm"])
        a_real = float(self.phys_config["sia0_real"])
        hbar = float(self.phys_config["hbar"])
        q_e = float(self.phys_config["q_e"])
        spr0 = float(self.phys_config["scales"]["spr0"])
        time0 = float(self.phys_config["scales"]["time0"])
        max_loops = int(self.config.get("max_particle_subloops", 10000))

        k_period_norm_x = 0.0
        k_period_norm_y = 0.0
        k_period_norm_z = 0.0
        if getattr(band, "kx_ticks_pi", None) is not None and len(band.kx_ticks_pi) > 0:
            k_period_norm_x = float(band.kx_ticks_pi[-1] - band.kx_ticks_pi[0]) * np.pi / sia0_norm
        if getattr(band, "ky_ticks_pi", None) is not None and len(band.ky_ticks_pi) > 0:
            k_period_norm_y = float(band.ky_ticks_pi[-1] - band.ky_ticks_pi[0]) * np.pi / sia0_norm
        if getattr(band, "kz_ticks_pi", None) is not None and len(band.kz_ticks_pi) > 0:
            k_period_norm_z = float(band.kz_ticks_pi[-1] - band.kz_ticks_pi[0]) * np.pi / sia0_norm

        kcell_max = self.kcell_max_phsr_real
        if kcell_max is None:
            kcell_max = np.zeros((0, 0, 0), dtype=np.float64)

        status_out = np.empty(n_active, dtype=np.int32)
        err_i_out = np.empty(n_active, dtype=np.int32)
        err_j_out = np.empty(n_active, dtype=np.int32)
        err_k_out = np.empty(n_active, dtype=np.int32)
        x_out = np.empty(n_active, dtype=np.float64)
        y_out = np.empty(n_active, dtype=np.float64)
        z_out = np.empty(n_active, dtype=np.float64)
        kx_out = np.empty(n_active, dtype=np.float64)
        ky_out = np.empty(n_active, dtype=np.float64)
        kz_out = np.empty(n_active, dtype=np.float64)
        energy_out = np.empty(n_active, dtype=np.float64)
        i_out = np.empty(n_active, dtype=np.int32)
        j_out = np.empty(n_active, dtype=np.int32)
        k_out = np.empty(n_active, dtype=np.int32)
        kx_idx_out = np.empty(n_active, dtype=np.int32)
        ky_idx_out = np.empty(n_active, dtype=np.int32)
        kz_idx_out = np.empty(n_active, dtype=np.int32)
        left_time_out = np.empty(n_active, dtype=np.float64)
        spawn_requested_out = np.empty(n_active, dtype=np.bool_)
        spawn_x_out = np.empty(n_active, dtype=np.float64)
        spawn_y_out = np.empty(n_active, dtype=np.float64)
        spawn_z_out = np.empty(n_active, dtype=np.float64)
        spawn_kx_out = np.empty(n_active, dtype=np.float64)
        spawn_ky_out = np.empty(n_active, dtype=np.float64)
        spawn_kz_out = np.empty(n_active, dtype=np.float64)
        spawn_energy_out = np.empty(n_active, dtype=np.float64)
        spawn_kx_idx_out = np.empty(n_active, dtype=np.int32)
        spawn_ky_idx_out = np.empty(n_active, dtype=np.int32)
        spawn_kz_idx_out = np.empty(n_active, dtype=np.int32)
        spawn_left_time_out = np.empty(n_active, dtype=np.float64)
        spawn_i_out = np.empty(n_active, dtype=np.int32)
        spawn_j_out = np.empty(n_active, dtype=np.int32)
        spawn_k_out = np.empty(n_active, dtype=np.int32)
        catch_stat_inc_out = np.empty(n_active, dtype=np.int32)
        escape_total_inc_out = np.empty(n_active, dtype=np.int32)
        escape_xp_out = np.empty(n_active, dtype=np.int32)
        escape_xm_out = np.empty(n_active, dtype=np.int32)
        escape_yp_out = np.empty(n_active, dtype=np.int32)
        escape_ym_out = np.empty(n_active, dtype=np.int32)
        escape_zp_out = np.empty(n_active, dtype=np.int32)
        escape_zm_out = np.empty(n_active, dtype=np.int32)
        step_ph_inc_out = np.empty(n_active, dtype=np.int32)
        phonon_self_inc_out = np.empty(n_active, dtype=np.int32)
        phonon_acoustic_inc_out = np.empty(n_active, dtype=np.int32)
        phonon_opt_abs_inc_out = np.empty(n_active, dtype=np.int32)
        phonon_opt_ems_inc_out = np.empty(n_active, dtype=np.int32)
        phonon_absorbed_eV_out = np.empty(n_active, dtype=np.float64)
        phonon_emitted_eV_out = np.empty(n_active, dtype=np.float64)

        _particle_fly_batch_kernel(
            active_indices,
            par.x, par.y, par.z, par.kx, par.ky, par.kz, par.energy,
            par.i, par.j, par.k, par.kx_idx, par.ky_idx, par.kz_idx,
            par.left_time, par.charge,
            self.mesh.cell_electric_field_x_real,
            self.mesh.cell_electric_field_y_real,
            self.mesh.cell_electric_field_z_real,
            self.mesh.x_nodes,
            self.mesh.y_nodes,
            self.mesh.z_nodes,
            self.mesh.material_id,
            mc_mat_id,
            self.velocity_grid_real_kernel,
            self.kx_tick_boundaries_kernel,
            self.ky_tick_boundaries_kernel,
            self.kz_tick_boundaries_kernel,
            int(getattr(band, "num_ticks_x", 0)),
            int(getattr(band, "num_ticks_y", 0)),
            int(getattr(band, "num_ticks_z", 0)),
            float(k_period_norm_x),
            float(k_period_norm_y),
            float(k_period_norm_z),
            hbar, q_e, spr0, time0, sia0_norm, a_real,
            np.asarray(kcell_max, dtype=np.float64),
            self.total_energy_grid_eV_kernel,
            self.total_rate_norm_kernel,
            self.component_rate_norm_kernel,
            self.phonon_cdf_kernel,
            self.hw_min_limit_kernel,
            self.hw_max_limit_kernel,
            float(self.phonon_emin_eV_kernel),
            float(self.phonon_dtable_eV_kernel),
            int(self.analytic_num_bins_kernel),
            float(self.analytic_bin_piecewise_kernel[0]),
            float(self.analytic_bin_piecewise_kernel[1]),
            float(self.analytic_bin_piecewise_kernel[2]),
            float(self.analytic_bin_piecewise_kernel[3]),
            int(self.analytic_bin_piecewise_kernel[4]),
            self.analytic_ntlist_kernel,
            self.analytic_ptlist_kernel,
            self.analytic_tlist_kernel,
            self.analytic_wcdf_kernel,
            self.analytic_wsum_kernel,
            self.analytic_nonempty_bins_kernel,
            self.ek_k_norm_kernel,
            self.ek_k_pi_kernel,
            self.ek_energy_eV_kernel,
            self.motion_bounds_m,
            self.motion_faces,
            self.motion_rules,
            self.monitor_bounds_m,
            self.monitor_faces,
            self.step_monitor_charge_c,
            self.step_monitor_crossings,
            max_loops,
            status_out,
            err_i_out, err_j_out, err_k_out,
            x_out, y_out, z_out, kx_out, ky_out, kz_out, energy_out,
            i_out, j_out, k_out, kx_idx_out, ky_idx_out, kz_idx_out,
            left_time_out,
            spawn_requested_out,
            spawn_x_out, spawn_y_out, spawn_z_out,
            spawn_kx_out, spawn_ky_out, spawn_kz_out,
            spawn_energy_out,
            spawn_kx_idx_out, spawn_ky_idx_out, spawn_kz_idx_out,
            spawn_left_time_out,
            spawn_i_out, spawn_j_out, spawn_k_out,
            catch_stat_inc_out,
            escape_total_inc_out,
            escape_xp_out, escape_xm_out, escape_yp_out, escape_ym_out, escape_zp_out, escape_zm_out,
            step_ph_inc_out,
            phonon_self_inc_out,
            phonon_acoustic_inc_out,
            phonon_opt_abs_inc_out,
            phonon_opt_ems_inc_out,
            phonon_absorbed_eV_out,
            phonon_emitted_eV_out,
        )

        self.step_scatter_stats["phonon"] += int(np.sum(step_ph_inc_out, dtype=np.int64))
        self.total_scatter_stats["phonon"] += int(np.sum(step_ph_inc_out, dtype=np.int64))
        self.phonon_scatter_stats["self"] += int(np.sum(phonon_self_inc_out, dtype=np.int64))
        self.phonon_scatter_stats["acoustic"] += int(np.sum(phonon_acoustic_inc_out, dtype=np.int64))
        self.phonon_scatter_stats["optical_abs"] += int(np.sum(phonon_opt_abs_inc_out, dtype=np.int64))
        self.phonon_scatter_stats["optical_ems"] += int(np.sum(phonon_opt_ems_inc_out, dtype=np.int64))
        self.phonon_scatter_stats["absorbed_eV"] += float(np.sum(phonon_absorbed_eV_out, dtype=np.float64))
        self.phonon_scatter_stats["emitted_eV"] += float(np.sum(phonon_emitted_eV_out, dtype=np.float64))

        self.particle_event_stats["escape_total"] += int(np.sum(escape_total_inc_out, dtype=np.int64))
        self.particle_event_stats["escape_+X"] += int(np.sum(escape_xp_out, dtype=np.int64))
        self.particle_event_stats["escape_-X"] += int(np.sum(escape_xm_out, dtype=np.int64))
        self.particle_event_stats["escape_+Y"] += int(np.sum(escape_yp_out, dtype=np.int64))
        self.particle_event_stats["escape_-Y"] += int(np.sum(escape_ym_out, dtype=np.int64))
        self.particle_event_stats["escape_+Z"] += int(np.sum(escape_zp_out, dtype=np.int64))
        self.particle_event_stats["escape_-Z"] += int(np.sum(escape_zm_out, dtype=np.int64))
        self.particle_event_stats["catch"] += int(np.sum(catch_stat_inc_out, dtype=np.int64))

        catch_count = 0
        for m in range(n_active):
            par_idx = int(active_indices[m])
            status = int(status_out[m])
            if status == FLY_STATUS_ERROR_NON_MC:
                self._assert_particle_in_mc_region(
                    int(err_i_out[m]),
                    int(err_j_out[m]),
                    int(err_k_out[m]),
                    int(par_idx),
                    float(x_out[m]),
                    float(y_out[m]),
                    float(z_out[m]),
                )
                raise RuntimeError(
                    f"Particle entered invalid region: par_idx={par_idx}, "
                    f"cell=({int(err_i_out[m])},{int(err_j_out[m])},{int(err_k_out[m])})"
                )

            if status == FLY_STATUS_CATCH:
                if int(catch_stat_inc_out[m]) == 0:
                    self.particle_event_stats["catch"] += 1
                par.i[par_idx] = -9999
                par.left_time[par_idx] = 0.0
                if hasattr(par, "flag") and par.flag is not None:
                    par.flag[par_idx] = 1
                catch_count += 1
                continue

            par.x[par_idx] = x_out[m]
            par.y[par_idx] = y_out[m]
            par.z[par_idx] = z_out[m]
            par.kx[par_idx] = kx_out[m]
            par.ky[par_idx] = ky_out[m]
            par.kz[par_idx] = kz_out[m]
            par.energy[par_idx] = energy_out[m]
            par.i[par_idx] = i_out[m]
            par.j[par_idx] = j_out[m]
            par.k[par_idx] = k_out[m]
            par.kx_idx[par_idx] = kx_idx_out[m]
            par.ky_idx[par_idx] = ky_idx_out[m]
            par.kz_idx[par_idx] = kz_idx_out[m]
            par.left_time[par_idx] = left_time_out[m]
            if hasattr(par, "flag") and par.flag is not None:
                par.flag[par_idx] = 1

        self.catch_par_num += catch_count

        if np.any(spawn_requested_out):
            spawn_mask = np.asarray(spawn_requested_out, dtype=bool)
            self._append_particle_states_batch(
                active_indices[spawn_mask],
                spawn_x_out[spawn_mask],
                spawn_y_out[spawn_mask],
                spawn_z_out[spawn_mask],
                spawn_kx_out[spawn_mask],
                spawn_ky_out[spawn_mask],
                spawn_kz_out[spawn_mask],
                spawn_energy_out[spawn_mask],
                spawn_kx_idx_out[spawn_mask],
                spawn_ky_idx_out[spawn_mask],
                spawn_kz_idx_out[spawn_mask],
                spawn_left_time_out[spawn_mask],
                spawn_i_out[spawn_mask],
                spawn_j_out[spawn_mask],
                spawn_k_out[spawn_mask],
            )

    def migrate_particles(self) -> None:
        """
        Placeholder for future particle migration / domain exchange logic.
        Single-process Python path currently does nothing.
        """
        return

    def update_particles(self) -> None:
        """
        Advance all particles for one global MC timestep.

        This mirrors the C++ control flow:
        1. Repeatedly fly unfinished particles.
        2. Count how many remain unfinished.
        3. Migrate particles between domains.

        The Python skeleton uses flat particle arrays plus boolean masks rather
        than per-cell linked lists so the hot path can later be replaced by a
        Numba-parallel particle_fly kernel without changing the driver logic.
        """
        if self.particle_ensemble is None:
            raise RuntimeError("Particle ensemble is not initialized.")

        par = self.particle_ensemble
        npar = int(par.size)
        if npar <= 0:
            return

        dt = float(self.config.get("dt", 1e-16))
        max_loops = int(self.config.get("max_particle_subloops", 10000))

        self.catch_par_num = 0
        self.gen_par = 0
        self.step_scatter_stats["phonon"] = 0
        self.step_scatter_stats["impurity"] = 0
        self.step_scatter_stats["surface"] = 0
        if self.step_monitor_charge_c.size > 0:
            self.step_monitor_charge_c.fill(0.0)
            self.step_monitor_crossings.fill(0)

        # Start a fresh MC timestep: all particles have full dt remaining and
        # are marked unfinished.
        if getattr(par, "left_time", None) is None or par.left_time.shape != (npar,):
            par.left_time = np.full(npar, dt, dtype=float)
        else:
            par.left_time.fill(dt)

        if getattr(par, "flag", None) is None or par.flag.shape != (npar,):
            par.flag = np.zeros(npar, dtype=np.int8)
        else:
            par.flag.fill(0)

        loop = 0
        while True:
            loop += 1

            active_mask = (par.i >= 0) & (par.left_time > 0.0)
            if loop > 1:
                active_mask &= (par.flag == 0)

            if not np.any(active_mask):
                break

            self.particle_fly(active_mask)

            unfinished_mask = (par.i >= 0) & (par.flag == 0) & (par.left_time > 0.0)
            unfinished_count = int(np.count_nonzero(unfinished_mask))

            self.migrate_particles()

            if unfinished_count <= 0:
                break
            if loop >= max_loops:
                print(
                    "      [Warning] Reached max particle subloops before all particles finished. "
                    "Stopping placeholder update."
                )
                break

    def run_mc(self) -> None:
        """
        Placeholder MC loop with Poisson hook.
        """
        print("[3/4] Entering MC loop")
        total_steps = int(self.config.get("total_step", 10))
        dt_fs = float(self.config.get("dt", 1e-16)) * 1e15
        print_interval = max(
            1,
            int(self.config.get("print_interval", self.config.get("stat_interval", 1))),
        )
        output_interval = max(1, int(self.config.get("output_interval", 1)))

        for step in range(total_steps):
            self.iterate_poisson()
            self.update_particles()
            self._accumulate_monitor_output_block()
            n_particles = 0
            if self.particle_ensemble is not None and getattr(self.particle_ensemble, "i", None) is not None:
                n_particles = int(np.count_nonzero(self.particle_ensemble.i >= 0))
            monitor_text = ""
            if self.step_monitor_charge_c.size > 0:
                currents_a = self.step_monitor_charge_c / max(float(self.config.get("dt", 1e-16)), 1.0e-30)
                parts = []
                for idx, name in enumerate(self.monitor_names[:4]):
                    parts.append(f"{name}={currents_a[idx]:.3e}A")
                if len(self.monitor_names) > 4:
                    parts.append("...")
                monitor_text = " | " + " ".join(parts)
            if ((step + 1) % output_interval) == 0 or step == total_steps - 1:
                self._write_current_monitor_output_block(step + 1, (step + 1) * dt_fs, float(self.config.get("dt", 1e-16)))
                self._export_snapshot(step, (step + 1) * dt_fs)
            if ((step + 1) % print_interval) == 0 or step == total_steps - 1:
                print(
                    f"  Step {step + 1:5d} | Time {(step + 1) * dt_fs:8.2f} fs | "
                    f"N={n_particles:7d} | "
                    f"Scatter(ph/imp/surf)="
                    f"{self.step_scatter_stats['phonon']}/"
                    f"{self.step_scatter_stats['impurity']}/"
                    f"{self.step_scatter_stats['surface']} | "
                    f"Generate={self.gen_par} | Catch={self.catch_par_num}"
                    f"{monitor_text}"
                )

    def postprocess(self) -> None:
        print("[4/4] Post-processing and saving results (placeholder).")

    def run(self) -> None:
        self.initialize()
        self.run_mc()
        self.postprocess()
        print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
