"""
Module: particle_setup.py
Description: Allocate and initialize particle ensemble (placeholder).
"""

import numpy as np


def init_particles(params: dict, mesh, device_structure: dict) -> dict:
    """
    params: parsed input.txt dictionary
    mesh: Mesh instance
    device_structure: parsed ldg device dictionary
    """
    num_p = int(params.get("ElectronNumber", 0))
    print(f"[Init] Allocating particles: {num_p} (placeholder).")

    ensemble = {
        "pos": np.zeros((num_p, 3), dtype=float),
        "k": np.zeros((num_p, 3), dtype=float),
        "energy": np.zeros(num_p, dtype=float),
        "active_mask": np.ones(num_p, dtype=bool),
    }

    return ensemble
