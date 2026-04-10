from __future__ import annotations

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

from mc.phonon_scattering import (
    _interpolate_component_rates_real,
    handle_phonon_scatter_event,
    sample_k_state_from_energy,
)


class DummyBand:
    def __init__(self) -> None:
        self.scattering_rate = {
            "total": np.asarray([2.0, 4.0, 6.0], dtype=np.float64),
            "components": np.asarray(
                [
                    [2.0, 4.0, 6.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            "cdf": np.zeros((5, 3, 4), dtype=np.float64),
        }
        self.phys = {"scales": {"time0": 2.0}}
        self.emin = 0.0
        self.dtable = 1.0
        self.eV0 = 0.5
        self.hw_min_limit = np.zeros(5, dtype=np.float64)
        self.hw_max_limit = np.zeros(5, dtype=np.float64)

        self.analytic_bin_piecewise = (0.0, 0.5, 0.5, 0.5, 1)
        self.analytic_num_bins = 2
        self.analytic_ntlist = np.asarray([1, 1], dtype=np.int32)
        self.analytic_ptlist = np.asarray([0, 1], dtype=np.int32)
        self.analytic_tlist = np.asarray([0, 1], dtype=np.int32)
        self.analytic_wcdf = np.asarray([1.0, 1.0], dtype=np.float64)
        self.analytic_wsum = np.asarray([1.0, 1.0], dtype=np.float64)
        self.analytic_nonempty_bins = np.asarray([0, 1], dtype=np.int32)

        self.ek_data = {
            "k_norm": np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
            "k_pi": np.asarray([[-0.2, -0.1, 0.2], [0.4, 0.3, -0.2]], dtype=np.float64),
            "energy_eV": np.asarray([0.25, 0.75], dtype=np.float64),
        }

        self.kx_tick_boundaries = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
        self.ky_tick_boundaries = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
        self.kz_tick_boundaries = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
        self.num_ticks_x = 2
        self.num_ticks_y = 2
        self.num_ticks_z = 2


class PhononScatteringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.band = DummyBand()
        self.simulation = types.SimpleNamespace(
            band_struct=self.band,
            kcell_max_phsr_real=np.full((2, 2, 2), 3.0, dtype=np.float64),
            phonon_scatter_stats={
                "self": 0,
                "acoustic": 0,
                "optical_abs": 0,
                "optical_ems": 0,
                "absorbed_eV": 0.0,
                "emitted_eV": 0.0,
            },
            step_scatter_stats={"phonon": 0, "impurity": 0, "surface": 0},
            total_scatter_stats={"phonon": 0, "impurity": 0, "surface": 0},
        )

    def test_component_rate_interp_matches_expected(self) -> None:
        rates = _interpolate_component_rates_real(self.band, 0.5)
        expected = np.asarray([2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.assertTrue(np.allclose(rates, expected, rtol=1.0e-12, atol=1.0e-18))

    def test_sample_k_state_from_energy_single_entry_bin(self) -> None:
        state = sample_k_state_from_energy(self.band, 0.25)
        expected = (0.1, 0.2, 0.3, 0.25, 0, 0, 1)
        for exp, act in zip(expected, state):
            if isinstance(exp, float):
                self.assertTrue(np.isclose(exp, act, rtol=1.0e-12, atol=1.0e-18))
            else:
                self.assertEqual(exp, act)

    def test_self_scatter_does_not_increment_printed_phonon_count(self) -> None:
        with patch("numpy.random.random", side_effect=[0.9]):
            out = handle_phonon_scatter_event(self.simulation, 0.25, 0, 0, 1)

        self.assertIsNone(out)
        self.assertEqual(self.simulation.phonon_scatter_stats["self"], 1)
        self.assertEqual(self.simulation.step_scatter_stats["phonon"], 0)
        self.assertEqual(self.simulation.total_scatter_stats["phonon"], 0)

    def test_real_acoustic_scatter_counts_once(self) -> None:
        with patch("numpy.random.random", side_effect=[0.1, 0.0, 0.0, 0.0]):
            out = handle_phonon_scatter_event(self.simulation, 0.25, 0, 0, 1)

        self.assertIsNotNone(out)
        expected = (0.1, 0.2, 0.3, 0.25, 0, 0, 1)
        for exp, act in zip(expected, out):
            if isinstance(exp, float):
                self.assertTrue(np.isclose(exp, act, rtol=1.0e-12, atol=1.0e-18))
            else:
                self.assertEqual(exp, act)

        self.assertEqual(self.simulation.phonon_scatter_stats["self"], 0)
        self.assertEqual(self.simulation.phonon_scatter_stats["acoustic"], 1)
        self.assertEqual(self.simulation.step_scatter_stats["phonon"], 1)
        self.assertEqual(self.simulation.total_scatter_stats["phonon"], 1)


if __name__ == "__main__":
    unittest.main()
