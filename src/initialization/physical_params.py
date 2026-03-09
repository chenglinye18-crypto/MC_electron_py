"""
Module: physical_params.py
Description: Initializing physical constants, material properties, and scaling system.
             Supports multi-material detection and legacy C++ normalization logic.
"""
import numpy as np


def init_physical_parameters(params, materials_found):
    """
    Args:
        params (dict): Parameters from input.txt (Temperature, dt, etc.)
        materials_found (list): Semiconductors identified from ldg.txt (e.g., ['IGZO'])
    Returns:
        dict: Physical configuration containing both real and normalized values.
    """
    print(f"[Init] Initializing physical parameters for: {materials_found}")

    # =========================================================================
    # 1. Fundamental Physical Constants (SI Units)
    # =========================================================================
    BOLTZ = 1.380649e-23  # Boltzmann constant [J/K]
    EM = 9.109383e-31     # Electron rest mass [kg]
    PLANCK = 1.054571e-34 # Reduced Planck constant (hbar) [J*s]
    EC = 1.602176e-19     # Elementary charge [C]
    CLIGHT = 2.997924e8   # Speed of light [m/s]
    FSC = 1 / 137.036     # Fine structure constant
    PI = np.pi

    T0 = params["Temperature"]

    # =========================================================================
    # 2. Scaling Factor System (Reference units for normalization)
    # =========================================================================
    eV0 = BOLTZ * T0      # Energy scale [J] (thermal energy k_B*T)
    em0 = EM              # Mass scale [kg]
    hq0 = PLANCK          # Action scale [J*s]
    ec0 = EC              # Charge scale [C]

    # Derived scales
    rmom0 = np.sqrt(em0 * eV0)   # Momentum scale
    spr0 = hq0 / rmom0           # Length scale (r-space)
    spk0 = 1.0 / spr0            # Wavevector scale (k-space)
    time0 = hq0 / eV0            # Time scale
    velo0 = spr0 / time0         # Velocity scale

    pot0 = eV0 / ec0             # Potential scale [V]
    field0 = pot0 / spr0         # Electric field scale [V/m]
    conc0 = spk0 ** 3            # Concentration scale [1/m^3]
    dens0 = em0 * conc0          # Mass density scale [kg/m^3]

    cvr_norm = CLIGHT / velo0    # Normalized speed of light

    # =========================================================================
    # 3. Material Specific Parameters (Real Physical Values)
    # =========================================================================
    # Default to the first detected semiconductor or Silicon
    if materials_found:
        primary_mat = next(iter(materials_found))
    else:
        primary_mat = "SILICON"

    # Initialization with default values
    sia0_real = 5.43e-10   # Lattice constant [m]
    sirho_real = 2.33e3    # Mass density [kg/m^3]
    siul_real = 9.05e3     # Longitudinal sound velocity [m/s]
    siut_real = 9.05e3     # Transverse sound velocity [m/s]
    psi_real = 4.05        # Electron Affinity [eV]
    eg_real = 1.12         # Band gap [eV]
    eps_rel = 11.7         # Relative dielectric constant
    alpha_val = 0.5        # Non-parabolicity [1/eV]

    if primary_mat == "IGZO":
        # IGZO specific parameters
        psi_real = 4.16     # Electron Affinity for IGZO
        # Lattice constant calculation: a = 9.6 * sqrt(3) * 1e-10
        sia0_real = 9.6 * np.sqrt(3.0) / 2 * 1.0e-10
        sirho_real = 6.10e3
        siul_real = 6.00e3
        siut_real = 6.00e3
        eg_real = 3.33      # Fixed wide bandgap for IGZO
        eps_rel = 10.0      # IGZO relative permittivity
        # Effective mass logic preserved from legacy defaults
        ml_val = 0.268
        mt_val = 0.254

    elif primary_mat == "SILICON":
        # Silicon specific parameters with temperature dependence
        psi_real = 4.05
        sia0_real = 5.43e-10
        sirho_real = 2.33e3
        siul_real = 9.05e3
        siut_real = 9.05e3
        # Bandgap temperature fitting
        if T0 < 190.0:
            eg_real = 1.170 + 1.059e-5 * T0 - 6.05e-7 * T0 ** 2
        elif T0 < 250.0:
            eg_real = 1.17850 - 9.025e-5 * T0 - 3.05e-7 * T0 ** 2
        else:
            eg_real = 1.2060 - 2.730e-4 * T0
        eps_rel = 11.7
        ml_val = 0.9163
        mt_val = 0.1905

    # =========================================================================
    # 4. Derived density of states (conduction band)
    # =========================================================================
    h_planck = 2.0 * PI * PLANCK
    md_eff = (ml_val * mt_val * mt_val) ** (1.0 / 3.0) * EM
    Nc_real = 2.0 * ((2.0 * PI * md_eff * BOLTZ * T0) / (h_planck * h_planck)) ** 1.5
    Nc_norm = Nc_real / conc0

    # =========================================================================
    # 5. Final Normalization (Suffix: _norm)
    # =========================================================================
    phys_config = {
        # Meta info and Scales
        "material": primary_mat,
        "scales": {
            "spr0": spr0,
            "time0": time0,
            "velo0": velo0,
            "eV0_J": eV0,
            "pot0_V": pot0,
            "conc0": conc0,
        },

        # Fundamental constants (for direct reuse elsewhere)
        "kb": BOLTZ,
        "m0": EM,
        "hbar": PLANCK,
        "q_e": EC,

        # real Physical Parameters
        "Temperature": T0,
        "eg_real": eg_real,
        "eps_rel": eps_rel,
        "sirho_real": sirho_real,
        "ml_val": ml_val,
        "mt_val": mt_val,
        "Nc_real": Nc_real,

        # Normalized Bulk Parameters
        "sia0_norm": sia0_real / spr0,
        "sirho_norm": sirho_real / dens0,
        "siul_norm": siul_real / velo0,
        "siut_norm": siut_real / velo0,
        "sieg_norm": eg_real / (eV0 / EC),  # Eg normalized to kBT

        # Surface/Contact Parameters (Real values for Schottky logic)
        "psi_real": psi_real,

        # 其他材料归一化介电系数 Dielectric Constants (Normalized for Poisson solver)
        "eps_vacuum_norm": 1.0 / (4 * PI * cvr_norm * FSC),
        "eps_oxide_norm": 3.9 / (4 * PI * cvr_norm * FSC),
        "eps_semi_norm": eps_rel / (4 * PI * cvr_norm * FSC),

        # Transport Parameters
        "ml_norm": ml_val,
        "mt_norm": mt_val,
        "alpha_norm": alpha_val * (eV0 / EC),  # alpha[1/eV] * kBT[eV]
        "Nc_norm": Nc_norm,

        # Coefficients
        "a0pi_norm": (2.0 * PI) / (sia0_real / spr0),
        "QuantumPotentialCoef_norm": (hq0 ** 2 * ec0)
        / (12.0 * 0.26 * em0 * spr0 ** 2 * pot0),
    }

    print(f"      -> Normalization complete. [Eg={eg_real:.2f}eV, eps_r={eps_rel}]")
    return phys_config
