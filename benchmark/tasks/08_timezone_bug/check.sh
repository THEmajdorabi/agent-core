#!/usr/bin/env bash
# Exits 0 when the task is solved.
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q test_solution.py
