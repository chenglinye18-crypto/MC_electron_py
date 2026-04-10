"""
Placeholder helpers for surface-scattering clocks and events.
"""
from __future__ import annotations


def compute_surface_scatter_time(
    simulation,
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
    Returns a very large time so the event is effectively disabled for now.
    """
    _ = (simulation, par_idx, x, y, z, i, j, k)
    return 1.0e99, 0.0


def handle_surface_scatter_event(simulation, par_idx: int) -> None:
    """
    Placeholder surface-scattering event handler.
    Current behavior is equivalent to self-scattering / no state change.
    """
    _ = par_idx
    simulation.step_scatter_stats["surface"] += 1
    simulation.total_scatter_stats["surface"] += 1
    return None
