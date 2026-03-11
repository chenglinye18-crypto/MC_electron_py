"""
Runtime simulation workflow for 3D Monte Carlo.
"""
from __future__ import annotations

import time

import numpy as np

from Particle import Particle
from Poisson import PoissonSolver


class Monte_Carlo_Simulation:
    """
    Execute Poisson/MC iteration flow after base initialization is complete.
    """

    def __init__(
        self,
        mesh,
        config: dict,
        phys_config: dict,
        band_struct,
        output_root: str,
        poisson_solver: PoissonSolver | None = None,
        device_structure: dict | None = None,
    ) -> None:
        self.mesh = mesh
        self.config = config
        self.phys_config = phys_config
        self.band_struct = band_struct
        self.output_root = output_root
        self.device_structure = device_structure or {}

        self.poisson_solver = poisson_solver
        if self.poisson_solver is None:
            self.poisson_solver = PoissonSolver(
                self.mesh,
                self.phys_config,
                self.device_structure,
                build_matrix=False,
            )

        self.particle_ensemble: Particle | None = None
        self.poisson_matrix_A = None
        self.poisson_vector_B: np.ndarray | None = None

    def initialize(self) -> None:
        print("[STEP 8] Initializing particle ensemble")
        self.particle_ensemble = Particle(
            self.mesh,
            self.config,
            self.phys_config,
            self.band_struct,
            self.output_root,
        )

    def iterate_poisson(self) -> None:
        """
        Placeholder Poisson iteration entry.
        Keeps A/B assembly in one place for later nonlinear solve integration.
        """
        print("[Poisson] Assembling A/B for current iteration...")
        self.poisson_matrix_A = self.poisson_solver.assemble_matrix_A(rebuild=False)
        self.poisson_vector_B = self.poisson_solver.assemble_vector_B()

        if self.poisson_matrix_A is None:
            print("[Poisson] Matrix A is not ready yet (placeholder assembly).")
        else:
            print(f"[Poisson] Matrix A shape: {self.poisson_matrix_A.shape}")
        print(f"[Poisson] Vector B size : {self.poisson_vector_B.size}")

    def run_mc(self) -> None:
        """
        Placeholder MC loop with Poisson hook.
        """
        print("[3/4] Entering MC loop (placeholder)...")
        total_steps = int(self.config.get("total_step", 10))
        total_steps = min(total_steps, 10)
        dt_fs = float(self.config.get("dt", 1e-16)) * 1e15

        for step in range(total_steps):
            if step == 0:
                self.iterate_poisson()
            if step % 5 == 0:
                print(f"  Step {step:5d} | Time {step * dt_fs:8.2f} fs")

    def postprocess(self) -> None:
        print("[4/4] Post-processing and saving results (placeholder).")

    def run(self) -> None:
        self.initialize()
        self.run_mc()
        self.postprocess()
        print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
