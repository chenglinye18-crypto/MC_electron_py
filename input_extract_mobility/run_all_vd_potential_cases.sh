#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
inputs=(
  "${SCRIPT_DIR}/input_vd0.1.txt"
  "${SCRIPT_DIR}/input_vd0.25.txt"
  "${SCRIPT_DIR}/input_vd0.5.txt"
  "${SCRIPT_DIR}/input_vd1.txt"
  "${SCRIPT_DIR}/input_vd2.txt"
  "${SCRIPT_DIR}/input_vd4.txt"
)
for input_file in "${inputs[@]}"; do
  echo "[Run] ${input_file}"
  (cd "${REPO_ROOT}" && "${PYTHON_BIN}" ./src/main.py --input "${input_file}")
done
