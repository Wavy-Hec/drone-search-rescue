#!/usr/bin/env bash
# Lint and CPU tests of the three projects: the same checks as CI (.github/workflows/tests.yml).
#
#   python -m pip install -r .github/requirements-ci.txt   # Python 3.12
#   scripts/check.sh                                       # every step
#   scripts/check.sh lint gcs                              # selected steps: lint, sim, gcs, quadruped
#
# Isaac Lab, PyTorch and Ultralytics are optional: the tests that need them skip when they cannot be imported,
# and run in the full environments of the README.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
PYTHON=${PYTHON:-python}

run_lint() {
  ruff check .
  ruff format --check .
}

run_sim() { (cd sim && "$PYTHON" -m pytest -q -rs -p no:cacheprovider tests); }

run_gcs() { (cd gcs && "$PYTHON" -m pytest -q -rs -p no:cacheprovider tests); }

run_quadruped() {
  # Every quadruped test needs Isaac Lab. Without it pytest skips them all and exits with 5 ("no tests ran").
  local rc=0
  (cd quadruped && PYTHONPATH=src "$PYTHON" -m pytest -q -rs -p no:cacheprovider tests) || rc=$?
  if ((rc == 5)); then
    echo "quadruped: every test was skipped (they need Isaac Lab)"
    rc=0
  fi
  return "$rc"
}

steps=("$@")
((${#steps[@]})) || steps=(lint sim gcs quadruped)
failed=()
for step in "${steps[@]}"; do
  case $step in
    lint | sim | gcs | quadruped) ;;
    *) echo "unknown step '$step' (choose from: lint sim gcs quadruped)" >&2 && exit 2 ;;
  esac
  printf '\n== %s\n' "$step"
  if "run_$step"; then
    echo "== $step: passed"
  else
    echo "== $step: FAILED"
    failed+=("$step")
  fi
done
if ((${#failed[@]})); then
  printf '\nFailed: %s\n' "${failed[*]}"
  exit 1
fi
printf '\nAll passed: %s\n' "${steps[*]}"
