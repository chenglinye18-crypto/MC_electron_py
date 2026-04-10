from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc.simulation import Monte_Carlo_Simulation


class DummyBandMinimal:
    def __init__(self) -> None:
        self.kx_ticks_pi = np.asarray([0.0], dtype=np.float64)
        self.ky_ticks_pi = np.asarray([0.0], dtype=np.float64)
        self.kz_ticks_pi = np.asarray([0.0], dtype=np.float64)
        self.kx_tick_boundaries = np.asarray([-1.0, 1.0], dtype=np.float64)
        self.ky_tick_boundaries = np.asarray([-1.0, 1.0], dtype=np.float64)
        self.kz_tick_boundaries = np.asarray([-1.0, 1.0], dtype=np.float64)
        self.num_ticks_x = 1
        self.num_ticks_y = 1
        self.num_ticks_z = 1
        self.velocity_grid_real = np.zeros((1, 1, 1, 3), dtype=np.float64)
        self.scattering_rate = None


class DummyBandGenerate:
    def __init__(self) -> None:
        self.kx_ticks_pi = np.asarray([-0.5, 0.5], dtype=np.float64)
        self.ky_ticks_pi = np.asarray([0.0], dtype=np.float64)
        self.kz_ticks_pi = np.asarray([0.0], dtype=np.float64)
        self.kx_tick_boundaries = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
        self.ky_tick_boundaries = np.asarray([-1.0, 1.0], dtype=np.float64)
        self.kz_tick_boundaries = np.asarray([-1.0, 1.0], dtype=np.float64)
        self.num_ticks_x = 2
        self.num_ticks_y = 1
        self.num_ticks_z = 1
        self.velocity_grid_real = np.zeros((2, 1, 1, 3), dtype=np.float64)
        self.velocity_grid_real[0, 0, 0, 0] = -1.0e5
        self.velocity_grid_real[1, 0, 0, 0] = 1.0e5
        self.scattering_rate = None


class DummyMeshBase:
    def __init__(self, x_nodes, y_nodes, z_nodes) -> None:
        self.x_nodes = np.asarray(x_nodes, dtype=float)
        self.y_nodes = np.asarray(y_nodes, dtype=float)
        self.z_nodes = np.asarray(z_nodes, dtype=float)
        self.dx = np.diff(self.x_nodes)
        self.dy = np.diff(self.y_nodes)
        self.dz = np.diff(self.z_nodes)
        self.nx = self.x_nodes.size - 1
        self.ny = self.y_nodes.size - 1
        self.nz = self.z_nodes.size - 1
        self.label_map = {"IGZO": 1, "OXIDE": 2}
        self.material_id = np.ones((self.nx, self.ny, self.nz), dtype=np.int32)
        self.cell_electric_field_x_real = np.zeros((self.nx, self.ny, self.nz), dtype=float)
        self.cell_electric_field_y_real = np.zeros((self.nx, self.ny, self.nz), dtype=float)
        self.cell_electric_field_z_real = np.zeros((self.nx, self.ny, self.nz), dtype=float)


class ParticleFlySingleNumbaTest(unittest.TestCase):
    def _make_particle(self, **kwargs):
        par = types.SimpleNamespace()
        par.x = np.asarray([kwargs["x"]], dtype=float)
        par.y = np.asarray([kwargs["y"]], dtype=float)
        par.z = np.asarray([kwargs["z"]], dtype=float)
        par.kx = np.asarray([kwargs["kx"]], dtype=float)
        par.ky = np.asarray([kwargs["ky"]], dtype=float)
        par.kz = np.asarray([kwargs["kz"]], dtype=float)
        par.energy = np.asarray([kwargs.get("energy", 0.2)], dtype=float)
        par.charge = np.asarray([kwargs.get("charge", -1.0)], dtype=float)
        par.i = np.asarray([kwargs["i"]], dtype=int)
        par.j = np.asarray([kwargs["j"]], dtype=int)
        par.k = np.asarray([kwargs["k"]], dtype=int)
        par.kx_idx = np.asarray([kwargs["kx_idx"]], dtype=int)
        par.ky_idx = np.asarray([kwargs.get("ky_idx", 0)], dtype=int)
        par.kz_idx = np.asarray([kwargs.get("kz_idx", 0)], dtype=int)
        par.left_time = np.asarray([kwargs["left_time"]], dtype=float)
        par.seed = np.asarray([0], dtype=np.int64)
        par.flag = np.asarray([0], dtype=np.int8)
        par.size = 1
        return par

    def test_compiled_particle_fly_time_step_end(self) -> None:
        mesh = DummyMeshBase([0.0, 1.0e-8], [0.0, 1.0e-8], [0.0, 1.0e-8])
        band = DummyBandMinimal()
        sim = Monte_Carlo_Simulation(
            mesh,
            config={"max_particle_subloops": 8},
            phys_config={
                "material": "IGZO",
                "sia0_norm": 1.0,
                "sia0_real": 1.0e-9,
                "hbar": 1.054571817e-34,
                "q_e": 1.602176634e-19,
                "scales": {"spr0": 1.0, "time0": 1.0},
            },
            band_struct=band,
            output_root=str(ROOT / "output"),
            poisson_solver=types.SimpleNamespace(phi=None),
            device_structure={},
        )
        sim.particle_ensemble = self._make_particle(
            x=5.0e-9,
            y=5.0e-9,
            z=5.0e-9,
            kx=0.0,
            ky=0.0,
            kz=0.0,
            i=0,
            j=0,
            k=0,
            kx_idx=0,
            left_time=2.0e-16,
        )

        sim._particle_fly_single(0)

        par = sim.particle_ensemble
        self.assertEqual(int(par.flag[0]), 1)
        self.assertEqual(int(par.i[0]), 0)
        self.assertTrue(np.isclose(float(par.left_time[0]), 0.0, atol=1.0e-30))
        self.assertEqual(sim.catch_par_num, 0)
        self.assertEqual(sim.gen_par, 0)

    def test_compiled_particle_fly_generate_reflects_and_spawns(self) -> None:
        mesh = DummyMeshBase([0.0, 1.0e-8, 2.0e-8], [0.0, 1.0e-8], [0.0, 1.0e-8])
        band = DummyBandGenerate()
        sim = Monte_Carlo_Simulation(
            mesh,
            config={"max_particle_subloops": 16},
            phys_config={
                "material": "IGZO",
                "sia0_norm": 1.0,
                "sia0_real": 1.0e-9,
                "hbar": 1.054571817e-34,
                "q_e": 1.602176634e-19,
                "scales": {"spr0": 1.0, "time0": 1.0},
            },
            band_struct=band,
            output_root=str(ROOT / "output"),
            poisson_solver=types.SimpleNamespace(phi=None),
            device_structure={
                "motion_planes": [
                    {"bounds": [10.0, 10.0, 0.0, 10.0, 0.0, 10.0], "face": "+X", "rule": "GENERATE"},
                ]
            },
        )
        sim.particle_ensemble = self._make_particle(
            x=9.0e-9,
            y=5.0e-9,
            z=5.0e-9,
            kx=0.5 * math.pi,
            ky=0.0,
            kz=0.0,
            i=0,
            j=0,
            k=0,
            kx_idx=1,
            left_time=2.0e-14,
        )

        sim._particle_fly_single(0)

        par = sim.particle_ensemble
        self.assertEqual(par.size, 2)
        self.assertEqual(sim.gen_par, 1)
        self.assertEqual(sim.particle_event_stats["generate"], 1)

        self.assertEqual(int(par.i[0]), 0)
        self.assertEqual(int(par.kx_idx[0]), 0)
        self.assertLess(float(par.kx[0]), 0.0)
        self.assertEqual(int(par.flag[0]), 1)

        self.assertEqual(int(par.i[1]), 1)
        self.assertEqual(int(par.flag[1]), 0)
        self.assertGreaterEqual(float(par.left_time[1]), 0.0)

    def test_particle_fly_batch_flushes_generated_particles_once(self) -> None:
        mesh = DummyMeshBase([0.0, 1.0e-8, 2.0e-8], [0.0, 1.0e-8], [0.0, 1.0e-8])
        band = DummyBandGenerate()
        sim = Monte_Carlo_Simulation(
            mesh,
            config={"max_particle_subloops": 16},
            phys_config={
                "material": "IGZO",
                "sia0_norm": 1.0,
                "sia0_real": 1.0e-9,
                "hbar": 1.054571817e-34,
                "q_e": 1.602176634e-19,
                "scales": {"spr0": 1.0, "time0": 1.0},
            },
            band_struct=band,
            output_root=str(ROOT / "output"),
            poisson_solver=types.SimpleNamespace(phi=None),
            device_structure={
                "motion_planes": [
                    {"bounds": [10.0, 10.0, 0.0, 10.0, 0.0, 10.0], "face": "+X", "rule": "GENERATE"},
                ]
            },
        )
        sim.particle_ensemble = self._make_particle(
            x=9.0e-9,
            y=5.0e-9,
            z=5.0e-9,
            kx=0.5 * math.pi,
            ky=0.0,
            kz=0.0,
            i=0,
            j=0,
            k=0,
            kx_idx=1,
            left_time=2.0e-14,
        )

        sim.particle_fly(np.asarray([True], dtype=bool))

        par = sim.particle_ensemble
        self.assertEqual(par.size, 2)
        self.assertEqual(sim.gen_par, 1)
        self.assertEqual(sim.particle_event_stats["generate"], 1)

        self.assertEqual(int(par.i[0]), 0)
        self.assertEqual(int(par.kx_idx[0]), 0)
        self.assertLess(float(par.kx[0]), 0.0)
        self.assertEqual(int(par.flag[0]), 1)

        self.assertEqual(int(par.i[1]), 1)
        self.assertEqual(int(par.flag[1]), 0)
        self.assertGreaterEqual(float(par.left_time[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
