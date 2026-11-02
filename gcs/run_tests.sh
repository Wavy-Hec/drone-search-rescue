#!/usr/bin/env bash
# Run the whole test suite in the conda env 'vision' (one command).
#   ./run_tests.sh                      # everything, ~25 s, CPU only (the YOLO test does 2 CPU inferences)
#   DRONE_GCS_SKIP_E2E=1 ./run_tests.sh # skip the fake-drone -> GCS end-to-end test
#   ./run_tests.sh tests.test_mapping   # one module
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${CONDA_DEFAULT_ENV:-}" != "vision" ]]; then
  CONDA_BASE=$(conda info --base) || { echo "conda not found: run 'conda activate vision' first" >&2; exit 1; }
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate vision
fi
if [[ $# -gt 0 ]]; then
  exec python -m unittest -v "$@"
fi
exec python -m unittest discover -s tests -t . -v
