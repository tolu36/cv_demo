# CV Demo (QB Detection & KPIs)

This repo hosts a YOLO-based pipeline to detect defenders/QB/ball/receivers, specialize on the ball, and compute simple QB KPIs. Two main roots: `at-it6` for datasets/configs and `demo` for training/inference scripts plus outputs.

## Layout
- at-it6/
  - data.yaml, data*.yaml: YOLO configs (paths currently point to this folder on Windows).
  - data/: full-frame YOLO dataset (`images/{train,val,train_oversampled,_dropped}`, `labels/{train,val,train_oversampled,_orphaned}`).
  - data/ball_patches/: patch dataset for the ball (`images/labels/{train,val}`, optional `hard_negs`, `ball_patches.yaml`).
  - ball_crops/, ball_crops_aug/: legacy cropped/augmented ball patch sets.
  - split_strict_log_*.csv, train.txt: split audit logs and image list.
- demo/
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

## Data & weights to restore
- Populate `at-it6/data` with YOLO-format images/labels; class map is `{0: Defender, 1: QB, 2: Ball, 3: Receiver}`. Update `data.yaml` paths if you are not on Windows or the root changes.
- For the patch pipeline, ensure `at-it6/data/ball_patches` exists (run `make_ball_only_ds.py` + `ball_jitter.py` and optionally `mine_hard_negs.py`/`add_background_patch.py`).
- Provide trained weights or edit paths in scripts:
  - General model default: `demo/runs/detect/qbcv_v8x_ballfocus_p23/weights/best.pt`.
  - Ball specialist default: `demo/runs/detect/ball_patches_l_p2_tinyobj_1024_lr5e-4_nomosaic_coslr_w5_ls0012/weights/best.pt`.
- Optional: `demo/homography.json` with `H` matrix and `scale_px_per_yd` for KPI projection.

## Quickstart
1) Python env: `python -m venv .venv && .\.venv\Scripts\activate` then `pip install -U ultralytics torch torchvision torchaudio opencv-python-headless pandas numpy matplotlib pillow pyyaml`.
2) Place data under `at-it6/data` (and `data/ball_patches` if using the specialist); fix `data.yaml`/`ball_patches.yaml` paths if needed.
3) Train full model: `python demo/train.py`.
4) Build/refresh ball patches: `python demo/make_ball_only_ds.py` -> `python demo/ball_jitter.py` -> optionally `python demo/mine_hard_negs.py --weights <full_best.pt> --conf 0.25 --iou 0.60 --tp_iou 0.20 --max_per_image 3 --limit 1500 --imgsz 1536` -> `python demo/add_background_patch.py`.
5) Train specialist: adjust `RESUME_WEIGHTS`/`USE_RESUME` in `demo/train_ball.py` then run `python demo/train_ball.py`.
6) Run combined inference: `python demo/predict_combo.py --images <folder> --out demo/pred_vis_combo`.
7) Track + KPIs: `python demo/ensemble_track_kpis.py` (uses detections_combo.csv; uses homography.json if present).
8) One-shot runner: `python demo/run_ball_pipeline.py` from `demo/` to execute the scripted steps.

Notes: Run scripts from the `demo` folder so relative `../at-it6` paths resolve; Ultralytics will write outputs under `demo/runs/` and `demo/pred_vis*` which are ignored by git.
