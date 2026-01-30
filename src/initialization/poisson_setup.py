"""
Module: poisson_setup.py
Description: Prepare Poisson solver environment (skeleton).
"""

from physics.poisson_solver import PoissonSolver


def init_poisson(mesh, phys_config: dict, device_structure: dict) -> dict:
    """
    mesh: Mesh instance
    phys_config: physical parameters dictionary
    device_structure: parsed ldg device dictionary
    """
    print("[Init] Preparing Poisson solver environment (skeleton)...")

    solver = PoissonSolver(mesh, phys_config, device_structure, build_matrix=False)

    poisson_env = {
        "solver": solver,
        "matrix_L": None,
        "boundary_mask": None,
        "initial_phi": None,
    }

    return poisson_env
