from __future__ import annotations

import copy
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mc.simulation as sim_module
from mc.simulation import (
    EVENT_HIT_CELL,
    EVENT_HIT_KGRID,
    EVENT_IMPURITY_SCATTER,
    EVENT_PHONON_SCATTER,
    EVENT_SURFACE_SCATTER,
    EVENT_TIME_STEP_END,
    Monte_Carlo_Simulation,
)


class DummyBand:
    def __init__(self) -> None:
        self.kx_ticks_pi = np.asarray([-1.0, 0.0, 1.0], dtype=float)
        self.ky_ticks_pi = np.asarray([-1.0, 0.0, 1.0], dtype=float)
        self.kz_ticks_pi = np.asarray([-1.0, 0.0, 1.0], dtype=float)
        self.kx_tick_boundaries = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=float)
        self.ky_tick_boundaries = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=float)
        self.kz_tick_boundaries = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=float)
        self.num_ticks_x = 3
        self.num_ticks_y = 3
        self.num_ticks_z = 3
        self.scattering_rate = None


class DummyMesh:
    def __init__(self) -> None:
        self.nx = 3
        self.ny = 3
        self.nz = 3
        self.label_map = {"IGZO": 1, "OXIDE": 2}
        self.material_id = np.ones((3, 3, 3), dtype=int)


class HandleParticleEventConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = DummyMesh()
        self.band = DummyBand()
        self.device_structure = {
            "motion_planes": [
                {"bounds": [0, 0, 0, 0, 0, 0], "face": "-Y", "rule": "REFLECT"},
                {"bounds": [1, 1, 1, 1, 1, 1], "face": "+X", "rule": "CATCH"},
                {"bounds": [2, 2, 2, 2, 2, 2], "face": "+Y", "rule": "GENERATE"},
                {"bounds": [3, 3, 3, 3, 3, 3], "face": "+Z", "rule": "SCATTOX"},
            ]
        }
        self.sim = Monte_Carlo_Simulation(
            self.mesh,
            config={},
            phys_config={"material": "IGZO", "sia0_norm": 1.0},
            band_struct=self.band,
            output_root=str(ROOT / "output"),
            poisson_solver=types.SimpleNamespace(phi=None),
            device_structure=self.device_structure,
        )
        self.sim.particle_ensemble = self._make_particle_ensemble()

    @staticmethod
    def _make_particle_ensemble():
        par = types.SimpleNamespace()
        par.x = np.asarray([9.0e-9], dtype=float)
        par.y = np.asarray([9.0e-9], dtype=float)
        par.z = np.asarray([9.0e-9], dtype=float)
        par.kx = np.asarray([0.75 * math.pi], dtype=float)
        par.ky = np.asarray([0.75 * math.pi], dtype=float)
        par.kz = np.asarray([0.75 * math.pi], dtype=float)
        par.energy = np.asarray([1.0], dtype=float)
        par.charge = np.asarray([-1.0], dtype=float)
        par.i = np.asarray([1], dtype=int)
        par.j = np.asarray([1], dtype=int)
        par.k = np.asarray([1], dtype=int)
        par.kx_idx = np.asarray([2], dtype=int)
        par.ky_idx = np.asarray([2], dtype=int)
        par.kz_idx = np.asarray([2], dtype=int)
        par.left_time = np.asarray([1.0e-15], dtype=float)
        par.seed = np.asarray([7], dtype=int)
        par.flag = np.asarray([0], dtype=np.int8)
        par.size = 1
        return par

    @staticmethod
    def _base_state() -> dict:
        return {
            "x": 9.0e-9,
            "y": 9.0e-9,
            "z": 9.0e-9,
            "kx": 0.75 * math.pi,
            "ky": 0.75 * math.pi,
            "kz": 0.75 * math.pi,
            "energy": 1.0,
            "i": 1,
            "j": 1,
            "k": 1,
            "kx_idx": 2,
            "ky_idx": 2,
            "kz_idx": 2,
            "left_time": 1.0e-15,
            "phrnl": 0.11,
            "imprnl": 0.22,
            "ssnl": 0.33,
        }

    def _reference_handle(self, sim, par_idx: int, event_flag: int, state: dict):
        state = dict(state)
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
            hit_dir = int(sim.last_k_col_dir)
            if hit_dir == 0:
                state["kx_idx"] += 1
            elif hit_dir == 1:
                state["kx_idx"] -= 1
            elif hit_dir == 2:
                state["ky_idx"] += 1
            elif hit_dir == 3:
                state["ky_idx"] -= 1
            elif hit_dir == 4:
                state["kz_idx"] += 1
            elif hit_dir == 5:
                state["kz_idx"] -= 1

            if sim.band_struct is not None:
                state["kx"], state["kx_idx"] = sim._wrap_k_axis_periodic(
                    state["kx"], state["kx_idx"], sim.band_struct.kx_ticks_pi
                )
                state["ky"], state["ky_idx"] = sim._wrap_k_axis_periodic(
                    state["ky"], state["ky_idx"], sim.band_struct.ky_ticks_pi
                )
                state["kz"], state["kz_idx"] = sim._wrap_k_axis_periodic(
                    state["kz"], state["kz_idx"], sim.band_struct.kz_ticks_pi
                )

        elif event_flag == EVENT_HIT_CELL:
            hit_dir = int(sim.last_cell_col_dir)
            rule = sim._resolve_motion_rule(state["x"], state["y"], state["z"], hit_dir)

            if rule == "CATCH":
                sim.particle_event_stats["catch"] += 1
                flag_catch = True
            elif rule == "GENERATE":
                sim._append_generated_particle(
                    par_idx,
                    state["x"],
                    state["y"],
                    state["z"],
                    state["kx"],
                    state["ky"],
                    state["kz"],
                    state["energy"],
                    state["kx_idx"],
                    state["ky_idx"],
                    state["kz_idx"],
                    state["left_time"],
                    state["i"],
                    state["j"],
                    state["k"],
                    hit_dir,
                )
                (
                    state["kx"],
                    state["ky"],
                    state["kz"],
                    state["kx_idx"],
                    state["ky_idx"],
                    state["kz_idx"],
                ) = sim._reflect_k_state(
                    state["kx"],
                    state["ky"],
                    state["kz"],
                    state["kx_idx"],
                    state["ky_idx"],
                    state["kz_idx"],
                    hit_dir,
                )
            elif rule in {"REFLECT", "SCATTOX"}:
                (
                    state["kx"],
                    state["ky"],
                    state["kz"],
                    state["kx_idx"],
                    state["ky_idx"],
                    state["kz_idx"],
                ) = sim._reflect_k_state(
                    state["kx"],
                    state["ky"],
                    state["kz"],
                    state["kx_idx"],
                    state["ky_idx"],
                    state["kz_idx"],
                    hit_dir,
                )
                if rule == "SCATTOX":
                    refresh_surf_time = True
            else:
                state["i"], state["j"], state["k"] = sim._advance_cell_indices(
                    state["i"], state["j"], state["k"], hit_dir
                )
                if (
                    state["i"] < 0
                    or state["i"] >= sim.mesh.nx
                    or state["j"] < 0
                    or state["j"] >= sim.mesh.ny
                    or state["k"] < 0
                    or state["k"] >= sim.mesh.nz
                ):
                    sim._record_escape(hit_dir)
                    flag_catch = True
                    state["left_time"] = 0.0
                else:
                    sim._assert_particle_in_mc_region(
                        state["i"],
                        state["j"],
                        state["k"],
                        int(par_idx),
                        float(state["x"]),
                        float(state["y"]),
                        float(state["z"]),
                    )
        elif event_flag == EVENT_PHONON_SCATTER:
            new_state = sim_module.handle_phonon_scatter_event(
                sim, state["energy"], state["kx_idx"], state["ky_idx"], state["kz_idx"]
            )
            if new_state is not None:
                (
                    state["kx"],
                    state["ky"],
                    state["kz"],
                    state["energy"],
                    state["kx_idx"],
                    state["ky_idx"],
                    state["kz_idx"],
                ) = new_state
        elif event_flag == EVENT_IMPURITY_SCATTER:
            sim_module.handle_impurity_scatter_event(sim, par_idx)
        elif event_flag == EVENT_SURFACE_SCATTER:
            sim_module.handle_surface_scatter_event(sim, par_idx)

        return (
            state["x"],
            state["y"],
            state["z"],
            state["kx"],
            state["ky"],
            state["kz"],
            state["energy"],
            state["i"],
            state["j"],
            state["k"],
            state["kx_idx"],
            state["ky_idx"],
            state["kz_idx"],
            state["left_time"],
            state["phrnl"],
            state["imprnl"],
            state["ssnl"],
            flag_catch,
            fly_too_far,
            refresh_tet_time,
            refresh_cell_time,
            refresh_ph_time,
            refresh_imp_time,
            refresh_surf_time,
        )

    def _run_pair(self, event_flag: int, state: dict, *, patch_phonon=None):
        sim_ref = copy.deepcopy(self.sim)
        sim_new = copy.deepcopy(self.sim)
        par_idx = 0

        ctx = patch(
            "mc.simulation.handle_phonon_scatter_event",
            patch_phonon,
        ) if patch_phonon is not None else nullcontext()

        with ctx:
            expected = self._reference_handle(sim_ref, par_idx, event_flag, state)
            actual = sim_new._handle_particle_event(
                par_idx,
                event_flag,
                state["x"],
                state["y"],
                state["z"],
                state["kx"],
                state["ky"],
                state["kz"],
                state["energy"],
                state["i"],
                state["j"],
                state["k"],
                state["kx_idx"],
                state["ky_idx"],
                state["kz_idx"],
                state["left_time"],
                state["phrnl"],
                state["imprnl"],
                state["ssnl"],
            )

        self._assert_tuple_equal(expected, actual)
        self._assert_sim_side_effects_equal(sim_ref, sim_new)

    def _assert_tuple_equal(self, expected, actual) -> None:
        self.assertEqual(len(expected), len(actual))
        for exp, act in zip(expected, actual):
            if isinstance(exp, float):
                self.assertTrue(np.isclose(exp, act, rtol=1.0e-12, atol=1.0e-18), msg=f"{exp} != {act}")
            else:
                self.assertEqual(exp, act)

    def _assert_sim_side_effects_equal(self, sim_ref, sim_new) -> None:
        self.assertEqual(sim_ref.gen_par, sim_new.gen_par)
        self.assertEqual(sim_ref.catch_par_num, sim_new.catch_par_num)
        self.assertEqual(sim_ref.particle_event_stats, sim_new.particle_event_stats)
        self.assertEqual(sim_ref.step_scatter_stats, sim_new.step_scatter_stats)
        self.assertEqual(sim_ref.total_scatter_stats, sim_new.total_scatter_stats)

        par_ref = sim_ref.particle_ensemble
        par_new = sim_new.particle_ensemble
        self.assertEqual(par_ref.size, par_new.size)
        for name in (
            "x", "y", "z", "kx", "ky", "kz", "energy", "charge",
            "i", "j", "k", "kx_idx", "ky_idx", "kz_idx", "left_time",
            "seed", "flag",
        ):
            arr_ref = getattr(par_ref, name)
            arr_new = getattr(par_new, name)
            self.assertEqual(arr_ref.shape, arr_new.shape, msg=name)
            self.assertTrue(np.array_equal(arr_ref, arr_new), msg=name)

    def test_hit_kgrid_wrap_consistent(self) -> None:
        state = self._base_state()
        self.sim.last_k_col_dir = 0
        self._run_pair(EVENT_HIT_KGRID, state)

    def test_hit_cell_pass_consistent(self) -> None:
        state = self._base_state()
        self.sim.last_cell_col_dir = 0
        self._run_pair(EVENT_HIT_CELL, state)

    def test_hit_cell_reflect_consistent(self) -> None:
        state = self._base_state()
        state.update({"x": 0.0, "y": 0.0, "z": 0.0})
        self.sim.last_cell_col_dir = 3
        self._run_pair(EVENT_HIT_CELL, state)

    def test_hit_cell_catch_consistent(self) -> None:
        state = self._base_state()
        state.update({"x": 1.0e-9, "y": 1.0e-9, "z": 1.0e-9})
        self.sim.last_cell_col_dir = 0
        self._run_pair(EVENT_HIT_CELL, state)

    def test_hit_cell_generate_consistent(self) -> None:
        state = self._base_state()
        state.update({"x": 2.0e-9, "y": 2.0e-9, "z": 2.0e-9})
        self.sim.last_cell_col_dir = 2
        self._run_pair(EVENT_HIT_CELL, state)

    def test_hit_cell_scattox_consistent(self) -> None:
        state = self._base_state()
        state.update({"x": 3.0e-9, "y": 3.0e-9, "z": 3.0e-9})
        self.sim.last_cell_col_dir = 4
        self._run_pair(EVENT_HIT_CELL, state)

    def test_phonon_event_consistent(self) -> None:
        state = self._base_state()

        def phonon_stub(simulation, energy_eV, kx_idx, ky_idx, kz_idx):
            _ = (simulation, energy_eV, kx_idx, ky_idx, kz_idx)
            return (0.1, 0.2, 0.3, 1.23, 0, 1, 2)

        self._run_pair(EVENT_PHONON_SCATTER, state, patch_phonon=phonon_stub)

    def test_impurity_event_consistent(self) -> None:
        state = self._base_state()
        self._run_pair(EVENT_IMPURITY_SCATTER, state)

    def test_surface_event_consistent(self) -> None:
        state = self._base_state()
        self._run_pair(EVENT_SURFACE_SCATTER, state)

    def test_time_step_end_consistent(self) -> None:
        state = self._base_state()
        self._run_pair(EVENT_TIME_STEP_END, state)


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    unittest.main()
