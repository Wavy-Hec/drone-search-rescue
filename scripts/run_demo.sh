#!/usr/bin/env bash
# Run the search-and-rescue demo end to end: the ground station starts first and waits for the drone, the mission
# flies in Isaac Sim, the ground station writes its report after the landing; then, optionally, the custom quadruped
# walks to a person the drone found.
#
#   scripts/download_assets.sh               # once: the flight policy, the detector and the quadruped policy
#   scripts/run_demo.sh                      # Isaac Sim and ground-station windows (needs a display)
#   scripts/run_demo.sh --headless --video   # no windows; records both sides and composes the demo video
#   scripts/run_demo.sh --finale             # the mission, then the rescue finale on its report
#   scripts/run_demo.sh --dry-run            # print the commands only
#
# Options: --name NAME (run folder, default demo), --seed N (site, default 0), --video (record the Isaac Sim view and
# compose the side-by-side demo video), --headless, --finale, --dry-run.
# Environment: SIM_ENV (conda env with Isaac Sim + Isaac Lab, default env_isaaclab), GCS_ENV (ground station, default
# vision), CONDA_SH (conda's profile script, found automatically).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SIM_ENV=${SIM_ENV:-env_isaaclab}
GCS_ENV=${GCS_ENV:-vision}
NAME=demo SEED=0 HEADLESS=0 VIDEO=0 FINALE=0 DRY=0

usage() { sed -n '2,17s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; }
while (($#)); do
  case $1 in
    --name) NAME=${2:?--name needs a value}; shift ;;
    --seed) SEED=${2:?--seed needs a value}; shift ;;
    --headless) HEADLESS=1 ;;
    --video) VIDEO=1 ;;
    --finale) FINALE=1 ;;
    --dry-run) DRY=1 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

POLICY=$ROOT/downloads/arl-robot-1_policy.pt
WEIGHTS=$ROOT/downloads/drone_sar_yolo26n.pt
QUAD_POLICY=$ROOT/quadruped/checkpoints_2x/height_scanner_model_rslrl5.pt

if [[ -z ${CONDA_SH:-} ]]; then
  for c in "$(conda info --base 2>/dev/null || true)" "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3"; do
    [[ -n $c && -f $c/etc/profile.d/conda.sh ]] && CONDA_SH=$c/etc/profile.d/conda.sh && break
  done
fi

# in_env ENV DIR CMD...: run CMD in DIR inside the conda env ENV (a fresh bash, so the two envs never mix)
in_env() {
  local env=$1 dir=$2
  shift 2
  if ((DRY)); then
    local cmd
    cmd=$(printf '%q ' "$@")
    printf '(cd %q && conda activate %q && %s)\n' "$dir" "$env" "${cmd% }"
    return 0
  fi
  bash -c 'source "$1" && conda activate "$2" && cd "$3" && shift 3 && exec "$@"' _ "$CONDA_SH" "$env" "$dir" "$@"
}

if ((!DRY)); then
  [[ -n ${CONDA_SH:-} && -f $CONDA_SH ]] || { echo "conda not found: set CONDA_SH=<conda>/etc/profile.d/conda.sh" >&2; exit 1; }
  needed=("$POLICY" "$WEIGHTS") missing=()
  ((FINALE)) && needed+=("$QUAD_POLICY")
  for f in "${needed[@]}"; do [[ -f $f ]] || missing+=("${f#"$ROOT"/}"); done
  if ((${#missing[@]})); then
    echo "missing: ${missing[*]}; run scripts/download_assets.sh first" >&2
    exit 1
  fi
fi

gcs_cmd=(python scripts/gcs.py --name "$NAME" --weights "$WEIGHTS" --exit-on-end)
sim_cmd=(python scripts/sar_mission.py --checkpoint "$POLICY" --seed "$SEED" --out "$NAME")
if ((HEADLESS)); then gcs_cmd+=(--headless); else sim_cmd+=(--viz kit); fi
((VIDEO)) && sim_cmd+=(--video)

echo "== 1/3 ground station ($GCS_ENV): waits for the drone, writes gcs/runs/$NAME/ after the landing"
if ((DRY)); then
  in_env "$GCS_ENV" "$ROOT/gcs" "${gcs_cmd[@]}"
else
  in_env "$GCS_ENV" "$ROOT/gcs" "${gcs_cmd[@]}" &
  gcs_pid=$!
  trap 'kill "$gcs_pid" 2>/dev/null || true' EXIT
fi

echo "== 2/3 mission in Isaac Sim ($SIM_ENV): take-off, survey, landing (about 6 minutes of flight)"
in_env "$SIM_ENV" "$ROOT/sim" "${sim_cmd[@]}"
if ((!DRY)); then
  wait "$gcs_pid"
  trap - EXIT
  echo "   report: gcs/runs/$NAME/report.md"
fi

if ((VIDEO)); then
  echo "== demo video: both recordings side by side, in step"
  in_env "$GCS_ENV" "$ROOT/gcs" python scripts/compose_video.py --gcs "runs/$NAME" --sim "$ROOT/sim/runs/$NAME"
fi

if ((FINALE)); then
  echo "== 3/3 rescue finale ($SIM_ENV): the custom quadruped walks to a person the drone found"
  if ((!DRY)) && ! in_env "$SIM_ENV" "$ROOT" python -c "import quadruped_lab" 2>/dev/null; then
    echo "quadruped_lab is not installed in $SIM_ENV: python -m pip install --no-deps -e quadruped" >&2
    exit 1
  fi
  in_env "$SIM_ENV" "$ROOT/sim" python scripts/rescue_finale.py --report "$ROOT/gcs/runs/$NAME" \
    --checkpoint "$QUAD_POLICY" --video --out "finale_$NAME"
fi
