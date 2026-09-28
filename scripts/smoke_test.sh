#!/usr/bin/env bash
# Quick check that every back end runs and agrees with the exact solution.
set -euo pipefail
cd "$(dirname "$0")/.."
H=./build-linux/heat
for b in serial omp cuda; do
  for p in double float; do $H --n 64 --t-end 0.01 --precision $p --backend $b; done
done
for b in omp cuda; do $H --n 2048 --steps 400 --backend $b --precision float; done
