# QB Detection & KPIs (YOLO Pipeline)

This repo contains a YOLO-based pipeline to detect defenders/QB/ball/receivers, run a ball specialist, and compute basic QB KPIs. Code lives in `demo/`, while datasets/config live in `at-it6/`.

If you just want to run inference with the best models, start with **Quickstart (Best Models)** below.

## Repo Layout (short)
- at-it6/
  - data.yaml, data*.yaml: YOLO configs (paths currently point to this folder on Windows).
  - data/: full-frame YOLO dataset (`images/{train,val,train_oversampled,_dropped}`, `labels/{train,val,train_oversampled,_orphaned}`).
  - data/ball_patches/: patch dataset for the ball (`images/labels/{train,val}`, optional `hard_negs`, `ball_patches.yaml`).
  - ball_crops/, ball_crops_aug/: legacy cropped/augmented ball patch sets.
  - split_strict_log_*.csv, train.txt: split audit logs and image list.
- demo/
  - best_models/: curated best weights for sharing/repro (`ball_specialist_best.pt`, `general_players_ball_best.pt`).
  - data.yaml + train.txt: training list for the full model.
  - train.py: train the 4-class full-frame model on `at-it6/data`.
  - train_ball.py: two-stage fine-tune of the ball patch specialist (defaults to resume from `runs/detect/.../best.pt`).
  - make_split.py: strict train/val split builder; drops unlabeled into `data/images/_dropped`, updates `data.yaml`.
  - oversample.py: writes `train_oversampled` subsets weighted by ball presence.
  - make_ball_only_ds.py: crops ball + near-negative patches into `data/ball_patches`.
  - ball_jitter.py: jittered positive patches for recall; keeps size in sync with training.
  - mine_hard_negs.py: mines false positives from full frames into `ball_patches/hard_negs`.
  - add_background_patch.py: injects background-only negatives into `ball_patches`.
  - run_ball_pipeline.py: convenience runner chaining the mining/background/train/predict steps (edit the `steps` list as needed).
  - predict.py: single-model inference that keeps only one QB per frame; saves overlays/CSV to `demo/pred_vis`.
  - predict_combo.py: merges general + ball-specialist detections (tiling/TTA) -> `demo/pred_vis_combo/{vis,detections_combo.csv}`.
  - ensemble_track_kpis.py: tracks QB/ball from detections_combo.csv, optional homography, emits tracks_combo.csv + kpis.json.
  - EDA.py, map_field.py, homography.json: data sanity checks and field mapping helpers.
  - QB_project.md: broader project goals/notes.
  - yolov8*/yolo11n.pt: base checkpoints; `runs/` and `pred_vis*` are model outputs (gitignored).
- .gitignore: keeps data, runs, weights, and large blobs out of git.

## Best Models (curated)
The current best weights are hosted on Google Drive (selected by highest final mAP50-95 in `results.csv`).
Request access if needed:
```
https://drive.google.com/drive/folders/1Y0he9NJhIuzeVqcaVsOzFTNKy9VpK3Nu?usp=sharing
```

If you want to use different runs, edit paths in the scripts.

## Quickstart (Best Models)
1) Create a Python env and install deps:
   - `python -m venv .venv && .\.venv\Scripts\activate`
   - `pip install -U ultralytics torch torchvision torchaudio opencv-python-headless pandas numpy matplotlib pillow pyyaml`
2) Put inference images in a folder, e.g. `C:\data\frames`.
3) Run combined inference:
   - `python demo/predict_combo.py --images C:\data\frames --out demo\pred_vis_combo`
4) (Optional) KPIs: run `python demo/ensemble_track_kpis.py` after `predict_combo.py`.

Notes:
- Run scripts from the repo root or `demo/`. Relative paths assume `demo/` sits next to `at-it6/`.
- `demo/pred_vis*` and `demo/runs/` are outputs and are ignored by git.

## Training / Reproduction (full)
- Populate `at-it6/data` with YOLO-format images/labels; class map is `{0: Defender, 1: QB, 2: Ball, 3: Receiver}`. Update `data.yaml` paths if you are not on Windows or the root changes.
- For the patch pipeline, ensure `at-it6/data/ball_patches` exists (run `make_ball_only_ds.py` + `ball_jitter.py` and optionally `mine_hard_negs.py`/`add_background_patch.py`).
- Train full model: `python demo/train.py`.
- Build/refresh ball patches: `python demo/make_ball_only_ds.py` -> `python demo/ball_jitter.py` -> optionally `python demo/mine_hard_negs.py --weights <full_best.pt> --conf 0.25 --iou 0.60 --tp_iou 0.20 --max_per_image 3 --limit 1500 --imgsz 1536` -> `python demo/add_background_patch.py`.
- Train specialist: adjust `RESUME_WEIGHTS`/`USE_RESUME` in `demo/train_ball.py` then run `python demo/train_ball.py`.
- Optional: `demo/homography.json` with `H` matrix and `scale_px_per_yd` for KPI projection.
