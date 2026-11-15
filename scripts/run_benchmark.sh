#!/usr/bin/env bash
# Fly the search-and-rescue mission on several sites, one after another, and put the scorecards in one table.
# One mission on one site says little about how often people are found: this gives the rate and its spread.
#
#   scripts/run_benchmark.sh --seeds 0-9                         # ten sites on the standard terrain
#   scripts/run_benchmark.sh --seeds 0-4 --terrain-seeds 17,3,5  # every site seed on every terrain seed
#   scripts/run_benchmark.sh --seeds 0-9 --record                # keep each mission's link stream for replays
#   scripts/run_benchmark.sh --seeds 0-2 --dry-run               # print the commands only
#
# Options: --seeds LIST (site seeds, e.g. 0-9 or 0,3,7; required), --terrain-seeds LIST (default: the standard
# terrain only), --name NAME (prefix of the run folders, default bench), --record, --dry-run.
#
# Every mission is one headless scripts/run_demo.sh run named <name>_s<seed> (with _t<terrain seed> when terrain
# seeds are given); a mission that fails is reported and the benchmark goes on. At the end
# gcs/scripts/aggregate_runs.py writes gcs/runs/<name>_benchmark/benchmark.md, .csv and .json.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=bench SEEDS="" TERRAIN_SEEDS="" RECORD=0 DRY=0

usage() { sed -n '2,15s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; }
while (($#)); do
  case $1 in
    --seeds) SEEDS=${2:?--seeds needs a list}; shift ;;
    --terrain-seeds) TERRAIN_SEEDS=${2:?--terrain-seeds needs a list}; shift ;;
    --name) NAME=${2:?--name needs a value}; shift ;;
    --record) RECORD=1 ;;
    --dry-run) DRY=1 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# expand_list "0-2,7" -> "0 1 2 7"
expand_list() {
  local part parts lo hi i out=()
  IFS=, read -ra parts <<<"$1"
  for part in "${parts[@]}"; do
    if [[ $part =~ ^([0-9]+)-([0-9]+)$ ]]; then
      lo=$((10#${BASH_REMATCH[1]})) hi=$((10#${BASH_REMATCH[2]})) # base 10: 08 is eight, not an octal error
      ((lo <= hi)) || { echo "bad range '$part'" >&2; return 1; }
      for ((i = lo; i <= hi; i++)); do out+=("$i"); done
    elif [[ $part =~ ^[0-9]+$ ]]; then
      out+=("$((10#$part))")
    else
      echo "bad seed '$part' (use e.g. 0-9 or 0,3,7)" >&2
      return 1
    fi
  done
  # a seed twice would run its mission twice into <name>_2, and the table would read the first run twice
  local twice
  twice=$(printf '%s\n' "${out[@]}" | sort -n | uniq -d)
  if [[ -n $twice ]]; then
    echo "seed listed twice: ${twice//$'\n'/ }" >&2
    return 1
  fi
  echo "${out[*]}"
}

[[ -n $SEEDS ]] || { echo "--seeds is required" >&2; usage >&2; exit 2; }
list=$(expand_list "$SEEDS") || exit 2
read -ra seeds <<<"$list"
terrain_seeds=("")
if [[ -n $TERRAIN_SEEDS ]]; then
  list=$(expand_list "$TERRAIN_SEEDS") || exit 2
  read -ra terrain_seeds <<<"$list"
fi

names=()
for t in "${terrain_seeds[@]}"; do
  for s in "${seeds[@]}"; do names+=("${NAME}_s${s}${t:+_t$t}"); done
done
if ((!DRY)); then
  for n in "${names[@]}"; do
    # the ground station never overwrites a run folder (it would write <name>_2), and the table would read the old one
    if [[ -d $ROOT/gcs/runs/$n && -n $(ls -A "$ROOT/gcs/runs/$n") ]]; then
      echo "gcs/runs/$n exists already: choose another --name or remove it" >&2
      exit 1
    fi
  done
fi

echo "== benchmark '$NAME': ${#names[@]} missions (site seeds ${seeds[*]}; terrain seeds ${TERRAIN_SEEDS:-standard})"
failed=() k=0
for t in "${terrain_seeds[@]}"; do
  for s in "${seeds[@]}"; do
    k=$((k + 1))
    name=${NAME}_s${s}${t:+_t$t}
    args=(--headless --seed "$s" --name "$name")
    [[ -n $t ]] && args+=(--terrain-seed "$t")
    ((RECORD)) && args+=(--record)
    ((DRY)) && args+=(--dry-run)
    echo "== mission $k/${#names[@]}: $name"
    "$ROOT/scripts/run_demo.sh" "${args[@]}" || failed+=("$name")
  done
done

run_dirs=()
for n in "${names[@]}"; do run_dirs+=("$ROOT/gcs/runs/$n"); done
table=(python3 "$ROOT/gcs/scripts/aggregate_runs.py" "${run_dirs[@]}" --out "$ROOT/gcs/runs/${NAME}_benchmark")
if ((DRY)); then
  printf '%q ' "${table[@]}"
  echo
  exit 0
fi
((${#failed[@]})) && echo "== failed missions: ${failed[*]}"
"${table[@]}"
