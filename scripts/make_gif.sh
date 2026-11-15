#!/usr/bin/env bash
# Make a GIF from a video with ffmpeg, in the quality of the README GIFs.
#
#   scripts/make_gif.sh IN OUT [--start S] [--duration D] [--crop W:H:X:Y] [--width W] [--fps F] [--speed X]
#                              [--colors N] [--dither bayer|bayer0-5|sierra|none] [--no-denoise] [--dashboard]
#                              [--hold S]
#
# Rendered footage (default): every frame gets a palette of its own, a light ordered dither replaces the colour
# banding of one shared palette, and a mild denoise runs first (video compression noise costs GIF bytes and shows
# as grain). --dashboard, for footage of the ground station's dashboard: one palette for the whole clip and only
# the changed part of each frame stored (about 16 KB a frame instead of 200), 256 colours, no dither, no blur.
#
# --start / --duration: the part of the video [s]; --crop: the part of the frame before scaling; --width: of the GIF
# (the height follows); --fps: GIF frame rate, at most that of the video; --speed: GIF playback speed against the
# video (0.5 = half speed, e.g. to undo part of a 3x time-lapse); --colors: palette size (96, --dashboard 256);
# --dither: bayerN sets the pattern's scale, from 0 (strongest) to 5 (lightest; bayer = bayer3); --hold: seconds the
# last frame stays up longer before the GIF starts over. Defaults: the whole video, no crop, width 800, 15 fps,
# speed 1, no hold.
#
# The README GIFs (docs/media/) come from one mission. Run the commands that
# scripts/run_demo.sh --dry-run --headless --video --record --finale --name gifs prints, with these changes:
#   - sar_mission.py: add --video_size 1920 1080;
#   - rescue_finale.py: add --goal_pin 34 (the person lying in the crater) and --out finale_gifs_720, and keep its
#     default 1280 x 720 (its HUD and minimap have a fixed size in pixels: rendered at 1920, the HUD text would be
#     8 px high in the 884 px GIF);
#   - then, in gcs/, replay the recorded link into a bigger dashboard: python scripts/gcs.py --replay
#     runs/gifs/link.dgcrec --replay-speed 1 --size 2050x1350 --name gifs_2050 --weights ... --headless.
# The dashboard GIFs come from a lossless copy of that dashboard video, dashboard_lossless.mkv, which gcs.py cannot
# write: its frames, also piped into ffmpeg -f rawvideo -pix_fmt bgr24 ... -c:v libx264rgb -qp 0 by a wrapper. Made
# from gcs.py's dashboard.mp4 (mp4v, then H.264), detection.gif is 11 dB lower in PSNR against those frames.
# GIF frames of rendered footage are costly: at the videos' 1920 width, flight.gif and finale.gif are 88 MB each.
#   flight.gif     scripts/make_gif.sh sim/runs/gifs/mission.mp4 docs/media/flight.gif --width 944 --colors 64 \
#                      --dither bayer5 --start 9.4 --duration 7.5 --speed 1.5             # sim t 11.0-18.4 s
#   detection.gif  scripts/make_gif.sh gcs/runs/gifs_2050/dashboard_lossless.mkv docs/media/detection.gif \
#                      --dashboard --no-denoise --crop 642:508:11:49 --width 642 --fps 12 --start 97.08 \
#                      --duration 8.58 --speed 1.43                         # camera panel, sim t 70.5-79.2 s
#   hero.gif       scripts/make_gif.sh gcs/runs/gifs_2050/dashboard_lossless.mkv docs/media/hero.gif \
#                      --dashboard --no-denoise --crop 1387:1301:663:49 --width 1387 --fps 12 --start 28.75 \
#                      --speed 24 --hold 1                                  # map panel, the pad to the landing
#   finale.gif     scripts/make_gif.sh sim/runs/finale_gifs_720/rescue.mp4 docs/media/finale.gif --width 884 \
#                      --colors 32 --dither bayer5 --start 17 --duration 10.28 --fps 12.5 --hold 1
#                                                                           # the walk's last 10 s, to the end card
# The dashboard crops of a W x H dashboard from W = 1560 up, where the camera panel is 640 wide (drone_gcs/dashboard.py,
# Dashboard.__init__): the camera panel 642:508:11:49 and the map panel with the HUD below it (W-663):(H-49):663:49
# (937:851:663:49 at the default 1600 x 900). From W = H + 674 up the square map fills the map panel's height.
set -euo pipefail

usage() { sed -n '2,/^[^#]/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; }
[[ ${1:-} == -h || ${1:-} == --help ]] && { usage; exit 0; }
(($# >= 2)) || { usage >&2; exit 2; }
IN=$1 OUT=$2
shift 2
START=0 DURATION="" CROP="" WIDTH=800 FPS=15 SPEED=1 COLORS="" DITHER="" DENOISE=1 DASHBOARD=0 HOLD=0
while (($#)); do
  case $1 in
    --start) START=${2:?--start needs seconds}; shift ;;
    --duration) DURATION=${2:?--duration needs seconds}; shift ;;
    --crop) CROP=${2:?--crop needs W:H:X:Y}; shift ;;
    --width) WIDTH=${2:?--width needs pixels}; shift ;;
    --fps) FPS=${2:?--fps needs a frame rate}; shift ;;
    --speed) SPEED=${2:?--speed needs a factor}; shift ;;
    --colors) COLORS=${2:?--colors needs a number}; shift ;;
    --dither) DITHER=${2:?--dither needs bayer, bayer0-5, sierra or none}; shift ;;
    --no-denoise) DENOISE=0 ;;
    --dashboard) DASHBOARD=1 ;;
    --hold) HOLD=${2:?--hold needs seconds}; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if ((DASHBOARD)); then COLORS=${COLORS:-256} DITHER=${DITHER:-none}; else COLORS=${COLORS:-96} DITHER=${DITHER:-bayer}; fi
[[ -f $IN ]] || { echo "no such video: $IN" >&2; exit 1; }
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || { echo "ffmpeg and ffprobe are needed" >&2; exit 1; }
((COLORS >= 2 && COLORS <= 256)) || { echo "--colors must be 2 to 256" >&2; exit 2; }
[[ $HOLD =~ ^([0-9]+[.]?[0-9]*|[.][0-9]+)$ ]] && python3 -c "import sys; sys.exit(float(sys.argv[1]) > 650)" "$HOLD" ||
  { echo "--hold must be 0 to 650 seconds" >&2; exit 2; }
case $DITHER in
  bayer) dither="dither=bayer:bayer_scale=3" ;;
  bayer[0-5]) dither="dither=bayer:bayer_scale=${DITHER#bayer}" ;;
  sierra) dither="dither=sierra2_4a" ;;
  none) dither="dither=none" ;;
  *) echo "--dither must be bayer, bayer0-5, sierra or none" >&2; exit 2 ;;
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
# the last frame's delay [1/100 s]: its own frame time plus the hold (-1 = the muxer's default: the previous delay)
final_delay=$(python3 -c 'import sys; h, f = map(float, sys.argv[1:]); print(round(100 / f + 100 * h) if h > 0 else -1)' \
  "$HOLD" "$FPS")
ffmpeg -v error -y "${input[@]}" -i "$IN" -filter_complex "$filters" -final_delay "$final_delay" "$OUT"
frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$OUT")
size=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$OUT")
echo "$OUT: $size, $frames frames at $FPS fps, $(du -h "$OUT" | cut -f1)"
