"""
Module: poisson_solver.py
Description: Nonlinear Poisson solver (skeleton).
             Mirrors C++ Picard iteration flow but keeps matrix assembly TODO.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
except ImportError:  # Optional at this stage
    sparse = None
    spsolve = None


class PoissonSolver:
    def __init__(self, mesh, phys_config: dict, device_structure: dict, build_matrix: bool = False):
        self.mesh = mesh
        self.phys = phys_config
        self.structure = device_structure

        self.nx, self.ny, self.nz = mesh.nx, mesh.ny, mesh.nz
        self.num_cells = self.nx * self.ny * self.nz

        self._init_epsilon_map()

        self.matrix_A = None
        if build_matrix:
            self.matrix_A = self._build_laplacian_matrix()

        self.phi = np.zeros(self.num_cells)

    def _idx(self, i: int, j: int, k: int) -> int:
        return (i * self.ny + j) * self.nz + k

    def _init_epsilon_map(self) -> None:
        eps_vac = self.phys.get("eps_vacuum_norm", 1.0)
        eps_ox = self.phys.get("eps_oxide_norm", eps_vac)
        eps_semi = self.phys.get("eps_semi_norm", eps_ox)

        mat = self.mesh.material_id
        ids = self.mesh.label_map
        vac_id = ids.get("VACUUM", 0)
        ox_id = ids.get("OXIDE", 1)
        igzo_id = ids.get("IGZO", 3)
        si_id = ids.get("SILICON", 2)

        eps = np.full_like(mat, eps_vac, dtype=float)
        eps[mat == ox_id] = eps_ox
        eps[(mat == igzo_id) | (mat == si_id)] = eps_semi
        self.eps_grid = eps

    def _build_laplacian_matrix(self):
        """
        TODO: Port C++ init_matrix FVM assembly.
        Placeholder raises to avoid silent misuse.
        """
        raise NotImplementedError("Laplacian assembly not implemented yet.")

    def calculate_contact_bc(self, contact: dict) -> float:
        """
        Return Dirichlet value for a contact.
        Uses C++-style PhiMS + Vapp convention.
        """
        vapp = float(contact.get("vapp", 0.0))
        phi_ms = float(contact.get("phi_ms", 0.0))
        return vapp + phi_ms

    def solve_nonlinear(self, particle_rho: np.ndarray) -> np.ndarray:
        if self.matrix_A is None:
            raise RuntimeError("Poisson matrix not assembled.")
        if sparse is None or spsolve is None:
            raise RuntimeError("scipy is required for Poisson solve.")

        max_iter = 100
        alpha = 0.05
        tol = 1e-5

        phi_old = self.phi.copy()

        for _ in range(max_iter):
            rhs = self._compute_rhs(phi_old, particle_rho)

            # NOTE: Dirichlet BC handling is not wired yet.
            phi_star = spsolve(self.matrix_A, rhs)

            phi_new = (1.0 - alpha) * phi_old + alpha * phi_star
            diff = np.linalg.norm(phi_new - phi_old) / (np.linalg.norm(phi_new) + 1e-12)
            phi_old = phi_new
            if diff < tol:
                break

        self.phi = phi_old
        return self.phi

    def _compute_rhs(self, phi: np.ndarray, particle_rho: np.ndarray) -> np.ndarray:
        """
        TODO: Add doping + Ni*exp(-phi) term and volume scaling.
        Placeholder passes through particle_rho.
        """
        return particle_rho.copy()
