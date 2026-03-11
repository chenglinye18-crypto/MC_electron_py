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

    def _build_laplacian_matrix(self):
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

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        def _avg_step(step_arr: np.ndarray) -> np.ndarray:
            # For n_cell segments, build n_point averaged lengths:
            # interior: average adjacent segments; boundaries: half segment.
            avg = np.empty(step_arr.size + 1, dtype=float)
            avg[0] = 0.5 * step_arr[0]
            avg[-1] = 0.5 * step_arr[-1]
            if step_arr.size > 1:
                avg[1:-1] = 0.5 * (step_arr[:-1] + step_arr[1:])
            return avg

        avedx = _avg_step(dx_norm)
        avedy = _avg_step(dy_norm)
        avedz = _avg_step(dz_norm)

        def _eps_face_x(i_face: int, j_point: int, k_point: int) -> float:
            """
            Area-weighted epsilon on x-normal face for point-control-volume FVM.
            Interior uses four surrounding cells; boundaries naturally reduce to
            two/one terms by availability.
            """
            num = 0.0

            j_candidates = []
            if j_point < ny_cell:
                j_candidates.append(j_point)
            if j_point > 0:
                j_candidates.append(j_point - 1)

            k_candidates = []
            if k_point < nz_cell:
                k_candidates.append(k_point)
            if k_point > 0:
                k_candidates.append(k_point - 1)

            for jc in j_candidates:
                for kc in k_candidates:
                    num += 0.25 * eps_norm[i_face, jc, kc] * dy_norm[jc] * dz_norm[kc]

            den = avedy[j_point] * avedz[k_point]
            if den <= 1e-30:
                return 0.0
            return num / den

        def _eps_face_y(i_point: int, j_face: int, k_point: int) -> float:
            """
            Area-weighted epsilon on y-normal face for point-control-volume FVM.
            """
            num = 0.0

            i_candidates = []
            if i_point < nx_cell:
                i_candidates.append(i_point)
            if i_point > 0:
                i_candidates.append(i_point - 1)

            k_candidates = []
            if k_point < nz_cell:
                k_candidates.append(k_point)
            if k_point > 0:
                k_candidates.append(k_point - 1)

            for ic in i_candidates:
                for kc in k_candidates:
                    num += 0.25 * eps_norm[ic, j_face, kc] * dx_norm[ic] * dz_norm[kc]

            den = avedx[i_point] * avedz[k_point]
            if den <= 1e-30:
                return 0.0
            return num / den

        def _eps_face_z(i_point: int, j_point: int, k_face: int) -> float:
            """
            Area-weighted epsilon on z-normal face for point-control-volume FVM.
            """
            num = 0.0

            i_candidates = []
            if i_point < nx_cell:
                i_candidates.append(i_point)
            if i_point > 0:
                i_candidates.append(i_point - 1)

            j_candidates = []
            if j_point < ny_cell:
                j_candidates.append(j_point)
            if j_point > 0:
                j_candidates.append(j_point - 1)

            for ic in i_candidates:
                for jc in j_candidates:
                    num += 0.25 * eps_norm[ic, jc, k_face] * dx_norm[ic] * dy_norm[jc]

            den = avedx[i_point] * avedy[j_point]
            if den <= 1e-30:
                return 0.0
            return num / den

        for i in range(nx_point):
            for j in range(ny_point):
                for k in range(nz_point):
                    p = self._idx(i, j, k)
                    diag = 0.0

                    # x- neighbor
                    if i > 0:
                        dist = dx_norm[i - 1]
                        area = avedy[j] * avedz[k]
                        eps_face = _eps_face_x(i - 1, j, k)
                        c = eps_face * area / dist
                        rows.append(p)
                        cols.append(self._idx(i - 1, j, k))
                        data.append(-c)
                        diag += c

                    # x+ neighbor
                    if i < nx_point - 1:
                        dist = dx_norm[i]
                        area = avedy[j] * avedz[k]
                        eps_face = _eps_face_x(i, j, k)
                        c = eps_face * area / dist
                        rows.append(p)
                        cols.append(self._idx(i + 1, j, k))
                        data.append(-c)
                        diag += c

                    # y- neighbor
                    if j > 0:
                        dist = dy_norm[j - 1]
                        area = avedx[i] * avedz[k]
                        eps_face = _eps_face_y(i, j - 1, k)
                        c = eps_face * area / dist
                        rows.append(p)
                        cols.append(self._idx(i, j - 1, k))
                        data.append(-c)
                        diag += c

                    # y+ neighbor
                    if j < ny_point - 1:
                        dist = dy_norm[j]
                        area = avedx[i] * avedz[k]
                        eps_face = _eps_face_y(i, j, k)
                        c = eps_face * area / dist
                        rows.append(p)
                        cols.append(self._idx(i, j + 1, k))
                        data.append(-c)
                        diag += c

                    # z- neighbor
                    if k > 0:
                        dist = dz_norm[k - 1]
                        area = avedx[i] * avedy[j]
                        eps_face = _eps_face_z(i, j, k - 1)
                        c = eps_face * area / dist
                        rows.append(p)
                        cols.append(self._idx(i, j, k - 1))
                        data.append(-c)
                        diag += c

                    # z+ neighbor
                    if k < nz_point - 1:
                        dist = dz_norm[k]
                        area = avedx[i] * avedy[j]
                        eps_face = _eps_face_z(i, j, k)
                        c = eps_face * area / dist
                        rows.append(p)
                        cols.append(self._idx(i, j, k + 1))
                        data.append(-c)
                        diag += c

                    rows.append(p)
                    cols.append(p)
                    data.append(diag)

        mat = sparse.csr_matrix((data, (rows, cols)), shape=(self.num_points, self.num_points))
        return mat

    def _init_poisson_matrix(self, build_matrix: bool) -> None:
        """
        Initialize Poisson matrix and optional direct solver in constructor.
        Mirrors C++ init_poisson_matrix + Amesos solver setup flow.
        """
        if not build_matrix:
            print("[Poisson] Matrix assembly deferred (build_matrix=False).")
            return

        self.matrix_A = self._build_laplacian_matrix()

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
