#!/usr/bin/env bash
# Build on Linux/WSL and run the GPU experiment. CPU experiments are run separately (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/build_linux.sh > /dev/null
python3 -c "import matplotlib" 2>/dev/null || dnf install -y -q python3-matplotlib || python3 -m pip install --quiet matplotlib
python3 bench/heatbench.py gpu --build build-linux
