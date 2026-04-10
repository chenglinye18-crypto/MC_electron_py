"""
Module: physical_params.py
Description: Initializing physical constants, material properties, and scaling system.
             Supports multi-material detection and legacy C++ normalization logic.
"""
import numpy as np


def _resolve_material_scattering_config(params: dict, material: str) -> dict:
    material_blocks = params.get("material_blocks", {}) or {}
    block = material_blocks.get(str(material).upper(), {}) or {}

    def _flag_enabled(name: str, default: bool) -> bool:
        raw_flags = block.get("scattering_flags", None)
        if raw_flags is None:
            return default
        if not isinstance(raw_flags, list):
            raw_flags = [raw_flags]
        flags = {str(item).strip().lower() for item in raw_flags}
        return name.lower() in flags

    return {
        "material": str(material).upper(),
        "flags": {
            "acoustic": _flag_enabled("acoustic", True),
            "lo_abs": _flag_enabled("lo_abs", True),
            "lo_ems": _flag_enabled("lo_ems", True),
            "to_abs": _flag_enabled("to_abs", True),
            "to_ems": _flag_enabled("to_ems", True),
        },
        "models": {
            "acoustic": str(block.get("acoustic_model", "deformation_potential_acoustic")),
            "optical_lo": str(block.get("optical_lo_model", block.get("optical_model", "deformation_potential_optical"))),
            "optical_to": str(block.get("optical_to_model", block.get("optical_model", "deformation_potential_optical"))),
            "disorder": str(block.get("disorder_model", "none")),
        },
        "params": {
            "acoustic_deformation_potential_eV": float(block.get("acoustic_deformation_potential_eV", 5.0)),
            "optical_deformation_potential_lo_eV_per_m": float(
                block.get("optical_deformation_potential_lo_eV_per_m", 5.0e5)
            ),
            "optical_deformation_potential_to_eV_per_m": float(
                block.get("optical_deformation_potential_to_eV_per_m", 5.0e5)
            ),
            "nonparabolicity_eV_inv": float(block.get("nonparabolicity_eV_inv", 0.0)),
            "disorder_tail_energy_eV": float(block.get("disorder_tail_energy_eV", 0.18)),
            "disorder_cutoff_energy_eV": float(block.get("disorder_cutoff_energy_eV", 10.0)),
        },
    }


def _compute_igzo_defect_density_m3(
    defect_params: dict,
    eg_real_eV: float,
    temperature_K: float,
) -> tuple[float, dict]:
    """
    Compute net defect charge source density [/m^3] for IGZO.

    Sign convention:
    - tail states are treated as acceptor-like: occupied fraction f(E)
      contributes negative charge magnitude N_A^-.
    - Gaussian vacancy states are treated as donor-like: ionized fraction
      1-f(E) contributes positive charge magnitude N_D^+.

    Returned value is the net defect charge source in /q units:
      N_defect_net = N_D^+ - N_A^-
    """
    if not defect_params:
        return 0.0, {}

    nta_cm3_eV = max(float(defect_params.get("nta", 0.0)), 0.0)
    nga_cm3_eV = max(float(defect_params.get("nga", 0.0)), 0.0)
    ea_eV = max(float(defect_params.get("wta", defect_params.get("ea", 0.0))), 1e-6)
    ega_eV = float(defect_params.get("ega", eg_real_eV))
    wga_eV = max(float(defect_params.get("wga", 0.0)), 1e-6)

    # Optional energy reference overrides in ldg defects block.
    ec_eV = float(defect_params.get("ec", eg_real_eV))
    ef_eV = float(
        defect_params.get(
            "ef",
            defect_params.get("efermi", ec_eV),
        )
    )

    if nta_cm3_eV <= 0.0 and nga_cm3_eV <= 0.0:
        return 0.0, {
            "ec_eV": ec_eV,
            "ef_eV": ef_eV,
            "ea_eV": ea_eV,
            "ega_eV": ega_eV,
            "wga_eV": wga_eV,
            "energy_min_eV": 0.0,
            "energy_max_eV": 0.0,
            "num_points": 0,
        }

    kBT_eV = 8.617333262145e-5 * float(temperature_K)
    kBT_eV = max(kBT_eV, 1e-6)

    # Build a finite integration window wide enough for both DOS terms.
    e_min = min(ec_eV - 40.0 * ea_eV, ega_eV - 8.0 * wga_eV, ef_eV - 40.0 * kBT_eV)
    e_max = max(ec_eV + 20.0 * kBT_eV, ega_eV + 8.0 * wga_eV, ef_eV + 20.0 * kBT_eV)
    if e_max <= e_min:
        e_max = e_min + 1.0

    dE_target = min(ea_eV, wga_eV, kBT_eV) / 40.0
    dE_target = max(dE_target, 1.0e-5)
    n_energy = int(np.ceil((e_max - e_min) / dE_target)) + 1
    n_energy = int(np.clip(n_energy, 2001, 200001))
    energy_eV = np.linspace(e_min, e_max, n_energy, dtype=float)

    # Convert DOS prefactors from cm^-3 eV^-1 to m^-3 eV^-1.
    nta_m3_eV = nta_cm3_eV * 1.0e6
    nga_m3_eV = nga_cm3_eV * 1.0e6

    # Tail states are taken below Ec; above Ec they are not part of the tail DOS.
    dos_tail = np.zeros_like(energy_eV)
    tail_mask = energy_eV <= ec_eV
    tail_arg = np.clip((energy_eV[tail_mask] - ec_eV) / ea_eV, -300.0, 300.0)
    dos_tail[tail_mask] = nta_m3_eV * np.exp(tail_arg)
    gauss_arg = ((energy_eV - ega_eV) / wga_eV) ** 2
    dos_gauss = nga_m3_eV * np.exp(-gauss_arg)

    fermi_arg = np.clip((energy_eV - ef_eV) / kBT_eV, -300.0, 300.0)
    fermi = 1.0 / (1.0 + np.exp(fermi_arg))
    one_minus_fermi = 1.0 - fermi

    tail_acceptor_occupied_m3 = float(np.trapezoid(dos_tail * fermi, energy_eV))
    gauss_donor_ionized_m3 = float(np.trapezoid(dos_gauss * one_minus_fermi, energy_eV))
    defect_density_m3 = gauss_donor_ionized_m3 - tail_acceptor_occupied_m3

    detail = {
        "ec_eV": ec_eV,
        "ef_eV": ef_eV,
        "ea_eV": ea_eV,
        "ega_eV": ega_eV,
        "wga_eV": wga_eV,
        "energy_min_eV": e_min,
        "energy_max_eV": e_max,
        "num_points": n_energy,
        "tail_acceptor_occupied_m3": tail_acceptor_occupied_m3,
        "gauss_donor_ionized_m3": gauss_donor_ionized_m3,
        "net_defect_density_m3": defect_density_m3,
    }
    return float(defect_density_m3), detail


def init_physical_parameters(params, materials_found, defects_by_material: dict | None = None):
    """
    Args:
        params (dict): Parameters from input.txt (Temperature, dt, etc.)
        materials_found (list): Semiconductors identified from ldg.txt (e.g., ['IGZO'])
        defects_by_material (dict | None): Parsed defects blocks from ldg.txt.
    Returns:
        dict: Physical configuration containing both real and normalized values.
    """
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
        # Lattice constant calculation: a = 9.6 * sqrt(3)/2 * 1e-10 
        # 周期性为2pi，故k空间取值范围为[-pi/a, pi/a]，具体可以看日志说明
        sia0_real = 8.313845876e-10
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

    # IGZO defect density from defects block (if provided).
    defect_density_m3 = 0.0
    defect_detail = {}
    if primary_mat == "IGZO":
        mat_defect_cfg = {}
        if defects_by_material is not None:
            mat_defect_cfg = defects_by_material.get("IGZO", {})
        defect_density_m3, defect_detail = _compute_igzo_defect_density_m3(
            mat_defect_cfg,
            eg_real,
            T0,
        )
    defect_density_norm = defect_density_m3 / conc0
    scattering_config = _resolve_material_scattering_config(params, primary_mat)

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
        "sia0_real":sia0_real,
        "defect_density_m3": defect_density_m3,

        # Normalized Bulk Parameters
        "sia0_norm": sia0_real / spr0,   #晶格常数归一化
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
        "defect_density_norm": defect_density_norm,
        "scattering_config": scattering_config,

        # Coefficients
        "a0pi_norm": PI / (sia0_real / spr0),    #k空间归一化
        "QuantumPotentialCoef_norm": (hq0 ** 2 * ec0)
        / (12.0 * 0.26 * em0 * spr0 ** 2 * pot0),
        "defect_model": defect_detail,
    }

    summary = f"[Init] material={primary_mat} Eg={eg_real:.2f} eV eps_r={eps_rel}"
    if primary_mat == "IGZO":
        summary += f" defect={defect_density_m3:.4e} m^-3"
    print(summary)
    return phys_config
