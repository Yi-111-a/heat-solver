#!/usr/bin/env bash
# Build the solver on Linux or WSL, including the CUDA back end if nvcc is available.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux -j
