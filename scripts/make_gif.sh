#!/usr/bin/env bash
# Make a GIF from a video with ffmpeg, in the quality of the README GIFs.
#
#   scripts/make_gif.sh IN OUT [--start S] [--duration D] [--crop W:H:X:Y] [--width W] [--fps F] [--speed X]
#                              [--colors N] [--dither bayer|sierra|none] [--no-denoise] [--dashboard]
#
# Rendered footage (default): every frame gets a palette of its own, a light ordered dither replaces the colour
# banding of one shared palette, and a mild denoise runs first (video compression noise costs GIF bytes and shows
# as grain). --dashboard, for footage of the ground station's dashboard: one palette for the whole clip and only
# the changed part of each frame stored (about 16 KB a frame instead of 200), 256 colours, no dither, no blur.
#
# --start / --duration: the part of the video [s]; --crop: the part of the frame before scaling; --width: of the GIF
# (the height follows); --fps: GIF frame rate, at most that of the video; --speed: GIF playback speed against the
# video (0.5 = half speed, e.g. to undo part of a 3x time-lapse); --colors: palette size (96, --dashboard 256).
# Defaults: the whole video, no crop, width 800, 15 fps, speed 1.
#
# The README GIFs (docs/media/), from the videos a mission writes on the computer that flew it:
#   flight.gif     scripts/make_gif.sh downloads/drone-sar-demo.mp4 docs/media/flight.gif \
#                      --crop 1200:676:0:0 --start 2.6 --duration 2.5 --speed 0.5     # the release's demo video
#                  scripts/make_gif.sh sim/runs/<mission>/mission.mp4 docs/media/flight.gif --start S --duration 5
#   detection.gif  scripts/make_gif.sh gcs/runs/<mission>/dashboard.mp4 docs/media/detection.gif \
#                      --crop 642:508:11:49 --width 642 --fps 12 --dashboard --start S --duration 6   # camera panel
#   hero.gif       scripts/make_gif.sh gcs/runs/<mission>/dashboard.mp4 docs/media/hero.gif \
#                      --crop 937:851:663:49 --width 937 --fps 12 --dashboard --start S --duration 15  # map panel
#   finale.gif     scripts/make_gif.sh sim/runs/<finale>/rescue.mp4 docs/media/finale.gif --width 884 --start S \
#                      --duration 10
# The two dashboard crops are the camera panel and the map panel with the HUD below it in the 1600 x 900 dashboard
# (drone_gcs/dashboard.py, Dashboard.__init__).
set -euo pipefail

usage() { sed -n '2,31s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; }
[[ ${1:-} == -h || ${1:-} == --help ]] && { usage; exit 0; }
(($# >= 2)) || { usage >&2; exit 2; }
IN=$1 OUT=$2
shift 2
START=0 DURATION="" CROP="" WIDTH=800 FPS=15 SPEED=1 COLORS="" DITHER="" DENOISE=1 DASHBOARD=0
while (($#)); do
  case $1 in
    --start) START=${2:?--start needs seconds}; shift ;;
    --duration) DURATION=${2:?--duration needs seconds}; shift ;;
    --crop) CROP=${2:?--crop needs W:H:X:Y}; shift ;;
    --width) WIDTH=${2:?--width needs pixels}; shift ;;
    --fps) FPS=${2:?--fps needs a frame rate}; shift ;;
    --speed) SPEED=${2:?--speed needs a factor}; shift ;;
    --colors) COLORS=${2:?--colors needs a number}; shift ;;
    --dither) DITHER=${2:?--dither needs bayer, sierra or none}; shift ;;
    --no-denoise) DENOISE=0 ;;
    --dashboard) DASHBOARD=1 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if ((DASHBOARD)); then COLORS=${COLORS:-256} DITHER=${DITHER:-none}; else COLORS=${COLORS:-96} DITHER=${DITHER:-bayer}; fi
[[ -f $IN ]] || { echo "no such video: $IN" >&2; exit 1; }
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || { echo "ffmpeg and ffprobe are needed" >&2; exit 1; }
((COLORS >= 2 && COLORS <= 256)) || { echo "--colors must be 2 to 256" >&2; exit 2; }
case $DITHER in
  bayer) dither="dither=bayer:bayer_scale=3" ;;
  sierra) dither="dither=sierra2_4a" ;;
  none) dither="dither=none" ;;
  *) echo "--dither must be bayer, sierra or none" >&2; exit 2 ;;
esac

# the GIF cannot have more frames per second than the video shows at the chosen speed
src_fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$IN")
FPS=$(python3 -c "from fractions import Fraction as F; print(float(min(F('$FPS'), F('$src_fps') * F('$SPEED'))))")

filters=""
[[ -n $CROP ]] && filters+="crop=$CROP:exact=1," # exact: odd sizes and offsets are not rounded to even ones
filters+="setpts=PTS/$SPEED,fps=$FPS,"
if ((DENOISE && DASHBOARD)); then
  filters+="hqdn3d=0:0:6:6," # over time only: steadies the static panels, keeps the text sharp
elif ((DENOISE)); then
  filters+="hqdn3d=4:3:6:4.5,gblur=sigma=0.5,"
fi
filters+="scale=$WIDTH:-1:flags=lanczos,split[a][b];"
if ((DASHBOARD)); then # one palette, and each frame stores only the rectangle that changed
  filters+="[a]palettegen=stats_mode=full:max_colors=$COLORS[p];[b][p]paletteuse=$dither:diff_mode=rectangle"
else # a palette per frame: the moving camera changes every pixel anyway
  filters+="[a]palettegen=stats_mode=single:max_colors=$COLORS[p];[b][p]paletteuse=new=1:$dither"
fi

input=(-ss "$START")
[[ -n $DURATION ]] && input+=(-t "$DURATION")
ffmpeg -v error -y "${input[@]}" -i "$IN" -filter_complex "$filters" "$OUT"
frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$OUT")
size=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$OUT")
echo "$OUT: $size, $frames frames at $FPS fps, $(du -h "$OUT" | cut -f1)"
