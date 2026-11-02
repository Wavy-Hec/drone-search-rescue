# models/

`scripts/train_detector.py` writes its weights here by default (`detector.pt`, with `detector_metrics.json` and the
training curves next to it); `--out` picks another file. Use them with
`python scripts/gcs.py --weights models/detector.pt`.

Without `--weights` the GCS takes `$DRONE_GCS_WEIGHTS`, else the default of `drone_gcs/detector.py`: the
fine-tuned `drone_sar_yolo26n.pt` in its models folder when it exists, else the COCO nano weights `yolo26n.pt`
there. The fine-tuned detector of the demo is a release asset (`../scripts/download_assets.sh` puts it in
`../downloads/`). Ultralytics weights and models fine-tuned from them are AGPL-3.0 (see the project README).
