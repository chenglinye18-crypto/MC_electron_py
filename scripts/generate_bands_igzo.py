#!/usr/bin/env python3
"""
Generate bands_IGZO.txt on a user-specified k-grid (units: pi/a).
Grid and material parameters are read from a config text file.

Output format matches existing bands_IGZO.txt:
  kx(pi/a) ky(pi/a) kz(pi/a) Energy(eV) vx(m/s) vy(m/s) vz(m/s)

Model: isotropic parabolic band (E = hbar^2 k^2 / (2 m_eff)).
Velocity: v = (1/hbar) dE/dk = hbar * k / m_eff.
生成IGZO表格
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List

import numpy as np


def _parse_segments(spec: str) -> List[float]:
    """
    Parse a spec like "-1:0.1:-0.3, -0.29:0.02:0.29, 0.3:0.1:1" into a list.
    Format per segment: start:step:stop (inclusive stop).
    """
    values: List[float] = []
    for seg in spec.split(","):
        seg = seg.strip()
        if not seg:
            continue
        parts = seg.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad segment format: '{seg}'. Expected start:step:stop")
        start = float(parts[0])
        step = float(parts[1])
        stop = float(parts[2])
        if step == 0.0:
            raise ValueError("step cannot be 0")
        n = int(round((stop - start) / step))
        if n < 0:
            raise ValueError(f"Segment step sign does not reach stop: '{seg}'")
        # inclusive stop
        seg_vals = [start + i * step for i in range(n + 1)]
        # avoid duplication when concatenating segments
        if values and abs(values[-1] - seg_vals[0]) < 1e-12:
            seg_vals = seg_vals[1:]
        values.extend(seg_vals)
    return values


def _parse_config(path: str) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].split("//", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"Bad config line (missing '='): {line}")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(f"Bad config line (empty key): {line}")
            cfg[key] = value
    return cfg


def _eval_number(expr: str) -> float:
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["pi"] = math.pi
    return float(eval(expr, {"__builtins__": {}}, allowed))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate bands_IGZO.txt on custom k-grid")
    parser.add_argument("--config", required=True, help="Path to k-grid config file")
    args = parser.parse_args()

    cfg = _parse_config(args.config)

    for key in ("kx", "ky", "kz", "mt", "ml", "a", "out"):
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")

    kx_vals = _parse_segments(cfg["kx"])
    ky_vals = _parse_segments(cfg["ky"])
    kz_vals = _parse_segments(cfg["kz"])

    mt = _eval_number(cfg["mt"])
    ml = _eval_number(cfg["ml"])
    a0 = _eval_number(cfg["a"])
    out_path = cfg["out"]
    k_scale = math.pi / a0  # (1/m) per (pi/a)

    if not kx_vals or not ky_vals or not kz_vals:
        raise ValueError("kx/ky/kz grid cannot be empty")

    m0 = 9.109383e-31
    hbar = 1.054571e-34
    q_e = 1.602176e-19

    m_eff = (ml * mt * mt) ** (1.0 / 3.0) * m0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("kx(pi/a) ky(pi/a) kz(pi/a) Energy(eV) vx(m/s) vy(m/s) vz(m/s)\n")
        for kx in kx_vals:
            for ky in ky_vals:
                for kz in kz_vals:
                
                    kx_real = kx * k_scale
                    ky_real = ky * k_scale
                    kz_real = kz * k_scale
        
                    vx = hbar * kx_real / (mt*m0)
                    vy = hbar * ky_real / (mt*m0)
                    vz = hbar * kz_real / (ml*m0)
        
                    energy = (hbar*hbar/2.0) * (
                            kx_real*kx_real/(mt*m0)
                          + ky_real*ky_real/(mt*m0)
                          + kz_real*kz_real/(ml*m0)
                    )
        
                    energy_eV = energy / q_e
                    f.write(f"{kx:.6f} {ky:.6f} {kz:.6f} {energy_eV:.6f} {vx:.6e} {vy:.6e} {vz:.6e}\n")

    print(f"Wrote bands file to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
