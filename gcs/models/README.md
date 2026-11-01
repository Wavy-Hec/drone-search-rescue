# models/

`detector.pt` (Git LFS) and `detector_metrics.json` are written here by `scripts/train_detector.py` (fine-tuned
YOLO on simulator images). Use it with `python scripts/gcs.py --weights models/detector.pt`.

Until then the GCS uses the COCO nano weights `~/isaac_ws/assets/models/yolo26n.pt` (not copied here).
Ultralytics weights and models fine-tuned from them are AGPL-3.0 (see the project README).
