"""
Module: poisson_solver.py
Description: Nonlinear Poisson solver (skeleton).
             Mirrors C++ Picard iteration flow but keeps matrix assembly TODO.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve, splu
except ImportError:  # Optional at this stage
    sparse = None
    spsolve = None
    splu = None


class PoissonSolver:
    def __init__(self, mesh, phys_config: dict, device_structure: dict, build_matrix: bool = False):
        self.mesh = mesh
        self.phys = phys_config
        self.structure = device_structure

        self.nx, self.ny, self.nz = mesh.nx, mesh.ny, mesh.nz
        self.num_cells = self.nx * self.ny * self.nz

        self._init_epsilon_map()

        self.matrix_A = None
        self.linear_solver = None

        self.phi = np.zeros(self.num_cells)
        self._init_poisson_matrix(build_matrix=build_matrix)

    def _idx(self, i: int, j: int, k: int) -> int:
        return (i * self.ny + j) * self.nz + k

    def _init_epsilon_map(self) -> None:
        eps_vac_norm = self.phys["eps_vacuum_norm"]
        eps_ox_norm = self.phys["eps_oxide_norm"]
        eps_semi_norm = self.phys["eps_semi_norm"]

        mat = self.mesh.material_id
        ids = self.mesh.label_map
        vac_id = ids["VACUUM"]
        ox_id = ids["OXIDE"]
        igzo_id = ids["IGZO"]
        si_id = ids["SILICON"]

        eps_norm = np.full_like(mat, eps_vac_norm, dtype=float)
        eps_norm[mat == ox_id] = eps_ox_norm
        eps_norm[(mat == igzo_id) | (mat == si_id)] = eps_semi_norm
        self.eps_grid_norm = eps_norm

    def _build_laplacian_matrix(self):
        """
        TODO: Port C++ init_matrix FVM assembly.
        Placeholder raises to avoid silent misuse.
        """
        raise NotImplementedError("Laplacian assembly not implemented yet.")

    def _init_poisson_matrix(self, build_matrix: bool) -> None:
        """
        Initialize Poisson matrix and optional direct solver in constructor.
        Mirrors C++ init_poisson_matrix + Amesos solver setup flow.
        """
        if not build_matrix:
            print("[Poisson] Matrix assembly deferred (build_matrix=False).")
            return

        try:
            self.matrix_A = self._build_laplacian_matrix()
        except NotImplementedError as exc:
            print(f"[Poisson] {exc} Matrix setup deferred.")
            self.matrix_A = None
            return

        if sparse is None or splu is None:
            print("[Poisson] scipy.sparse is unavailable; direct LU not initialized.")
            return

        # Keep matrix in CSC format so repeated solves can reuse the factorization.
        self.matrix_A = self.matrix_A.tocsc()
        print("[Poisson] Pre-computing LU factorization...")
        self.linear_solver = splu(self.matrix_A)
        print("[Poisson] LU factorization complete.")

    def calculate_contact_bc(self, contact: dict) -> float:
        """
        Return Dirichlet value for a contact.
        Uses C++-style PhiMS + Vapp convention.
        """
        vapp = float(contact["vapp"])
        phi_ms = float(contact["phi_ms"])
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
            if self.linear_solver is not None:
                phi_star = self.linear_solver.solve(rhs)
            else:
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
