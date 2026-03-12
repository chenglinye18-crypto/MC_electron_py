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

        # Cell-centered mesh sizes from lgrid intervals.
        self.nx_cell, self.ny_cell, self.nz_cell = mesh.nx, mesh.ny, mesh.nz
        self.num_cells = self.nx_cell * self.ny_cell * self.nz_cell

        # Point-centered unknown sizes for Poisson potential.
        self.nx_point = self.nx_cell + 1
        self.ny_point = self.ny_cell + 1
        self.nz_point = self.nz_cell + 1
        self.num_points = self.nx_point * self.ny_point * self.nz_point

        self._init_epsilon_map()

        self.matrix_A = None
        self.linear_solver = None

        self.phi = np.zeros(self.num_points)
        self._init_poisson_matrix(build_matrix=build_matrix)

    def _idx(self, i: int, j: int, k: int) -> int:
        return (i * self.ny_point + j) * self.nz_point + k

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

    def _build_flux_operator_matrix(self):
        """
        Build sparse Poisson matrix A with a 7-point FVM stencil
        on point-centered potential unknowns (cell-corner nodes).
        Boundary conditions are intentionally left for later.
        """

        if sparse is None:
            raise RuntimeError("scipy.sparse is required for matrix assembly.")

        nx_cell, ny_cell, nz_cell = self.nx_cell, self.ny_cell, self.nz_cell
        nx_point, ny_point, nz_point = self.nx_point, self.ny_point, self.nz_point
        spr0 = float(self.phys["scales"]["spr0"])
        if spr0 <= 0.0:
            raise ValueError("Invalid normalization length scale spr0.")

        dx_norm = np.asarray(self.mesh.dx, dtype=float) / spr0
        dy_norm = np.asarray(self.mesh.dy, dtype=float) / spr0
        dz_norm = np.asarray(self.mesh.dz, dtype=float) / spr0
        eps_norm = np.asarray(self.eps_grid_norm, dtype=float)

        if dx_norm.shape != (nx_cell,):
            raise ValueError("dx shape mismatch")
        if dy_norm.shape != (ny_cell,):
            raise ValueError("dy shape mismatch")
        if dz_norm.shape != (nz_cell,):
            raise ValueError("dz shape mismatch")
        if eps_norm.shape != (nx_cell, ny_cell, nz_cell):
            raise ValueError("eps_grid_norm shape mismatch")

        # Face coefficients (vectorized), preserving current FVM logic.
        # x-faces: shape (nx_cell, ny_point, nz_point)
        wx = 0.25 * eps_norm * dy_norm[None, :, None] * dz_norm[None, None, :]
        num_x = np.zeros((nx_cell, ny_point, nz_point), dtype=float)
        num_x[:, :-1, :-1] += wx
        num_x[:, 1:, :-1] += wx
        num_x[:, :-1, 1:] += wx
        num_x[:, 1:, 1:] += wx
        cx = num_x / dx_norm[:, None, None]

        # y-faces: shape (nx_point, ny_cell, nz_point)
        wy = 0.25 * eps_norm * dx_norm[:, None, None] * dz_norm[None, None, :]
        num_y = np.zeros((nx_point, ny_cell, nz_point), dtype=float)
        num_y[:-1, :, :-1] += wy
        num_y[1:, :, :-1] += wy
        num_y[:-1, :, 1:] += wy
        num_y[1:, :, 1:] += wy
        cy = num_y / dy_norm[None, :, None]

        # z-faces: shape (nx_point, ny_point, nz_cell)
        wz = 0.25 * eps_norm * dx_norm[:, None, None] * dy_norm[None, :, None]
        num_z = np.zeros((nx_point, ny_point, nz_cell), dtype=float)
        num_z[:-1, :-1, :] += wz
        num_z[1:, :-1, :] += wz
        num_z[:-1, 1:, :] += wz
        num_z[1:, 1:, :] += wz
        cz = num_z / dz_norm[None, None, :]

        idx = np.arange(self.num_points, dtype=np.int64).reshape(nx_point, ny_point, nz_point)

        # x-direction faces
        px = idx[:-1, :, :].ravel()
        qx = idx[1:, :, :].ravel()
        cxv = cx.ravel()
        rows_x = np.concatenate([px, px, qx, qx])
        cols_x = np.concatenate([px, qx, px, qx])
        data_x = np.concatenate([cxv, -cxv, -cxv, cxv])

        # y-direction faces
        py = idx[:, :-1, :].ravel()
        qy = idx[:, 1:, :].ravel()
        cyv = cy.ravel()
        rows_y = np.concatenate([py, py, qy, qy])
        cols_y = np.concatenate([py, qy, py, qy])
        data_y = np.concatenate([cyv, -cyv, -cyv, cyv])

        # z-direction faces
        pz = idx[:, :, :-1].ravel()
        qz = idx[:, :, 1:].ravel()
        czv = cz.ravel()
        rows_z = np.concatenate([pz, pz, qz, qz])
        cols_z = np.concatenate([pz, qz, pz, qz])
        data_z = np.concatenate([czv, -czv, -czv, czv])

        rows = np.concatenate([rows_x, rows_y, rows_z])
        cols = np.concatenate([cols_x, cols_y, cols_z])
        data = np.concatenate([data_x, data_y, data_z])

        return sparse.coo_matrix((data, (rows, cols)), shape=(self.num_points, self.num_points)).tocsr()

    def _init_poisson_matrix(self, build_matrix: bool) -> None:
        """
        Initialize Poisson matrix and optional direct solver in constructor.
        Mirrors C++ init_poisson_matrix + Amesos solver setup flow.
        """
        if not build_matrix:
            print("[Poisson] Matrix assembly deferred (build_matrix=False).")
            return

        self.matrix_A = self._build_flux_operator_matrix()

        if sparse is None or splu is None:
            print("[Poisson] scipy.sparse is unavailable; direct LU not initialized.")
            return

        # Keep matrix in CSC format so repeated solves can reuse the factorization.
        self.matrix_A = self.matrix_A.tocsc()
        print("[Poisson] Pre-computing LU factorization...")
        try:
            self.linear_solver = splu(self.matrix_A)
            print("[Poisson] LU factorization complete.")
        except Exception as exc:
            # Without boundary conditions A can be singular; keep matrix and continue.
            self.linear_solver = None
            print(f"[Poisson] LU factorization skipped: {exc}")

    def assemble_matrix_A(self, rebuild: bool = False):
        """
        Public entry for assembling/reusing the Poisson system matrix A.
        """
        if self.matrix_A is not None and not rebuild:
            return self.matrix_A

        if rebuild:
            self.matrix_A = None
            self.linear_solver = None

        self._init_poisson_matrix(build_matrix=True)
        return self.matrix_A

    def assemble_vector_B(
        self,
        particle_rho: np.ndarray | None = None,
        phi: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Public entry for assembling RHS vector B.
        """
        if particle_rho is None:
            rho = np.zeros(self.num_points, dtype=float)
        else:
            rho = np.asarray(particle_rho, dtype=float).reshape(self.num_points)

        if phi is None:
            phi_eval = self.phi
        else:
            phi_eval = np.asarray(phi, dtype=float).reshape(self.num_points)

        return self._compute_rhs(phi_eval, rho)

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
            rhs = self.assemble_vector_B(particle_rho=particle_rho, phi=phi_old)

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
