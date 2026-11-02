# models/

`detector.pt` (Git LFS) and `detector_metrics.json` are written here by `scripts/train_detector.py` (fine-tuned
YOLO on simulator images). Use it with `python scripts/gcs.py --weights models/detector.pt`.

Without `--weights` the GCS takes `$DRONE_GCS_WEIGHTS`, else the fine-tuned
`~/isaac_ws/assets/models/drone_sar_yolo26n.pt` when it exists, else the COCO nano weights `yolo26n.pt` next to
it (neither is copied here). Ultralytics weights and models fine-tuned from them are AGPL-3.0 (see the project
README).
