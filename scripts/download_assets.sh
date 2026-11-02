#!/usr/bin/env bash
# Download the release assets (trained models, demo video) and check them against the release's SHA256SUMS.txt.
#
#   scripts/download_assets.sh                    # every asset of the latest release
#   scripts/download_assets.sh --tag v1.0.0       # a given release
#   scripts/download_assets.sh --only drone_sar_yolo26n.pt
#   scripts/download_assets.sh --list             # show what goes where; downloads nothing
#
# Uses the GitHub CLI (gh) when it is installed and logged in, otherwise curl on the public download URLs.
# Files that are already in place with the right checksum are kept.
set -euo pipefail
REPO=Wavy-Hec/drone-search-rescue
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# release asset | destination (relative to the repository root) | what it is
ASSETS=(
  "arl-robot-1_policy.pt|downloads/arl-robot-1_policy.pt|drone flight policy (TorchScript) for sim/scripts/sar_mission.py"
  "drone_sar_yolo26n.pt|downloads/drone_sar_yolo26n.pt|fine-tuned YOLO26n detector for gcs/scripts/gcs.py (AGPL-3.0)"
  "height_scanner_model_rslrl5.pt|quadruped/checkpoints_2x/height_scanner_model_rslrl5.pt|custom quadruped walking policy for the rescue finale"
  "height_scanner_model.pt|quadruped/checkpoints_2x/height_scanner_model.pt|the same policy as trained in Isaac Lab 2.x (input of the checkpoint conversion)"
  "drone-sar-demo.mp4|downloads/drone-sar-demo.mp4|demo video: Isaac Sim and the ground station side by side"
)

usage() { sed -n '2,10s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; }
TAG="" LIST=0 ONLY=()
while (($#)); do
  case $1 in
    --tag) TAG=${2:?--tag needs a release tag}; shift ;;
    --only) ONLY+=("${2:?--only needs an asset name}"); shift ;;
    --list) LIST=1 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

selected=()
for entry in "${ASSETS[@]}"; do
  name=${entry%%|*}
  if ((${#ONLY[@]} == 0)) || [[ " ${ONLY[*]} " == *" $name "* ]]; then
    selected+=("$entry")
  fi
done
for want in "${ONLY[@]}"; do
  [[ " ${selected[*]} " == *"$want|"* ]] || { echo "unknown asset: $want (see --list)" >&2; exit 2; }
done

if ((LIST)); then
  for entry in "${selected[@]}"; do
    IFS='|' read -r name dest what <<<"$entry"
    printf '%-32s -> %-56s %s\n' "$name" "$dest" "$what"
  done
  exit 0
fi

if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  fetch() { gh release download ${TAG:+"$TAG"} --repo "$REPO" --pattern "$1" --dir "$2" --clobber; }
  via="gh"
else
  command -v curl >/dev/null || { echo "neither gh nor curl is available" >&2; exit 1; }
  base="https://github.com/$REPO/releases/${TAG:+download/$TAG}"
  [[ -n $TAG ]] || base="https://github.com/$REPO/releases/latest/download"
  fetch() { curl -fsSL --retry 3 -o "$2/$1" "$base/$1"; }
  via="curl"
fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/dsr-assets.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
echo "release: ${TAG:-latest} of $REPO (via $via)"
fetch SHA256SUMS.txt "$tmp" || { echo "could not download SHA256SUMS.txt of release ${TAG:-latest}" >&2; exit 1; }

sum_of() { awk -v n="$1" '$2 == n || $2 == "*"n {print $1}' "$tmp/SHA256SUMS.txt"; }
for entry in "${selected[@]}"; do
  IFS='|' read -r name dest _ <<<"$entry"
  want=$(sum_of "$name")
  [[ -n $want ]] || { echo "$name is not listed in SHA256SUMS.txt" >&2; exit 1; }
  if [[ -f $ROOT/$dest ]] && [[ $(sha256sum "$ROOT/$dest" | cut -d' ' -f1) == "$want" ]]; then
    echo "ok        $dest (already up to date)"
    continue
  fi
  fetch "$name" "$tmp"
  got=$(sha256sum "$tmp/$name" | cut -d' ' -f1)
  [[ $got == "$want" ]] || { echo "checksum mismatch for $name: expected $want, got $got" >&2; exit 1; }
  mkdir -p "$(dirname "$ROOT/$dest")"
  mv -f "$tmp/$name" "$ROOT/$dest"
  echo "verified  $dest"
done
