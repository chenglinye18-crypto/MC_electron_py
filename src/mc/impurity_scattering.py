"""
Placeholder helpers for impurity-scattering clocks and events.
"""
from __future__ import annotations


def compute_impurity_scatter_time(
    simulation,
    par_idx: int,
    energy: float,
    doping: float,
    charge_density: float,
) -> tuple[float, float]:
    """
    Placeholder impurity-scattering clock.
    Returns a very large time so the event is effectively disabled for now.
    """
    _ = (simulation, par_idx, energy, doping, charge_density)
    return 1.0e99, 0.0


def handle_impurity_scatter_event(simulation, par_idx: int) -> None:
    """
    Placeholder impurity-scattering event handler.
    Current behavior is equivalent to self-scattering / no state change.
    """
    _ = par_idx
    simulation.step_scatter_stats["impurity"] += 1
    simulation.total_scatter_stats["impurity"] += 1
    return None
