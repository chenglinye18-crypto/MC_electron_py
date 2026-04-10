# Scattering Model Configuration

This project reads scattering-model settings from `input/input.txt` through a
material block such as `IGZO { ... }`.

The main simulation and the analysis script
[`scripts/analyze_scattering_mobility.py`](/home/ic/3dmc_Si_ylx_mod/3DMCpy/scripts/analyze_scattering_mobility.py)
use the same parser and the same `AnalyticBand` scattering build path.

## Input Format

Example:

```txt
IGZO {
  scattering_flags = acoustic, lo_abs, lo_ems, to_abs, to_ems

  acoustic_model = deformation_potential_acoustic
  optical_lo_model = deformation_potential_optical
  optical_to_model = deformation_potential_optical
  disorder_model = linear_tail_enhancement

  acoustic_deformation_potential_eV = 5.0
  optical_deformation_potential_lo_eV_per_m = 5.0e5
  optical_deformation_potential_to_eV_per_m = 5.0e5
  nonparabolicity_eV_inv = 0.0
  disorder_tail_energy_eV = 0.18
  disorder_cutoff_energy_eV = 10.0
}
```

If a material block is omitted, the code falls back to built-in defaults.

## Supported Flags

`scattering_flags` is a comma-separated list. Supported values:

- `acoustic`
- `lo_abs`
- `lo_ems`
- `to_abs`
- `to_ems`

Any flag not listed is disabled.

Example: disable LO absorption only

```txt
IGZO {
  scattering_flags = acoustic, lo_ems, to_abs, to_ems
}
```

## Supported Models

### 1. Acoustic Phonon

Key:

```txt
acoustic_model = deformation_potential_acoustic
```

Current status:

- This is the only implemented acoustic model.
- The code uses one acoustic deformation-potential strength for both LA and TA
  branches.

Parameter:

```txt
acoustic_deformation_potential_eV = 5.0
```

Used in code as:

```text
D_LA = D_TA = acoustic_deformation_potential_eV * q
```

### 2. Optical Phonon

Keys:

```txt
optical_lo_model = deformation_potential_optical
optical_to_model = deformation_potential_optical
```

Current status:

- This is the only implemented optical model.
- LO and TO use independent coupling strengths.

Parameters:

```txt
optical_deformation_potential_lo_eV_per_m = 5.0e5
optical_deformation_potential_to_eV_per_m = 5.0e5
```

Used in code as:

```text
Dopt_LO = optical_deformation_potential_lo_eV_per_m * q
Dopt_TO = optical_deformation_potential_to_eV_per_m * q
```

### 3. Band Nonparabolicity

Parameter:

```txt
nonparabolicity_eV_inv = 0.0
```

Used in the kinetic-energy correction:

```text
k(E) from E (1 + alpha E)
```

with:

```text
alpha = nonparabolicity_eV_inv
```

### 4. Disorder Enhancement

Key:

```txt
disorder_model = ...
```

Supported values:

- `none`
- `linear_tail_enhancement`

#### `disorder_model = none`

No disorder enhancement is applied.

Equivalent factor:

```text
S_disorder(E) = 1
```

This is also the default if `disorder_model` is omitted.

#### `disorder_model = linear_tail_enhancement`

Current formula:

```text
S_disorder(E) = exp(delta_E / kBT)
delta_E = E_tail * (1 - E / E_cutoff),   for E < E_cutoff
delta_E = 0,                             for E >= E_cutoff
```

Parameters:

```txt
disorder_tail_energy_eV = 0.18
disorder_cutoff_energy_eV = 10.0
```

Interpretation:

- `disorder_tail_energy_eV` sets the low-energy enhancement scale.
- `disorder_cutoff_energy_eV` sets the energy range over which the enhancement
  decays linearly to zero.

## Minimal Examples

### Fully Enabled Phonon Scattering, No Disorder

```txt
IGZO {
  scattering_flags = acoustic, lo_abs, lo_ems, to_abs, to_ems
  acoustic_model = deformation_potential_acoustic
  optical_lo_model = deformation_potential_optical
  optical_to_model = deformation_potential_optical
  disorder_model = none

  acoustic_deformation_potential_eV = 5.0
  optical_deformation_potential_lo_eV_per_m = 5.0e5
  optical_deformation_potential_to_eV_per_m = 5.0e5
  nonparabolicity_eV_inv = 0.0
}
```

### Enable Disorder Enhancement

```txt
IGZO {
  scattering_flags = acoustic, lo_abs, lo_ems, to_abs, to_ems
  acoustic_model = deformation_potential_acoustic
  optical_lo_model = deformation_potential_optical
  optical_to_model = deformation_potential_optical
  disorder_model = linear_tail_enhancement

  acoustic_deformation_potential_eV = 5.0
  optical_deformation_potential_lo_eV_per_m = 5.0e5
  optical_deformation_potential_to_eV_per_m = 5.0e5
  nonparabolicity_eV_inv = 0.0
  disorder_tail_energy_eV = 0.18
  disorder_cutoff_energy_eV = 10.0
}
```

## Current Scope

At the moment, only the IGZO phonon-scattering branch is implemented in the
analytic scattering table builder. Impurity and surface scattering are still
handled separately and are currently placeholders in the MC flow.
