# run_ball_pipeline.py
import os, subprocess, sys
from pathlib import Path

# Point to your folder (where this file lives)
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# Your scripts assume data lives at ../at-it6/data from this file.
AT_IT6_DATA = (BASE_DIR / ".." / "at-it6" / "data").resolve()
if not AT_IT6_DATA.exists():
    raise SystemExit(
        f"[ERROR] Expected data folder not found: {AT_IT6_DATA}\n"
        "Your scripts use AT = os.path.abspath('../at-it6'). "
        "Move this runner into the folder that sits alongside 'at-it6', "
        "or adjust the scripts' AT path in each script."
    )

steps = [
    # ("make_split.py", "Strict split & drop unlabeled (builds data.yaml)"),
    # ("make_ball_only_ds.py", "Build ball_patches (pos + near negs)"),
    # ("ball_jitter.py", "Add jittered positive patches to ball_patches"),
    (
        "mine_hard_negs.py --weights runs/detect/ball_patches_l_p2_tinyobj_1024_lr5e-4_nomosaic_coslr_w5_ls0012/weights/best.pt --conf 0.25 --iou 0.60 --tp_iou 0.20 --max_per_image 3 --limit 1500 --imgsz 1536",
        "Mine hard negatives from full frames with latest 1024 specialist",
    ),
    ("add_background_patch.py", "Add pure background negatives to ball_patches"),
    ("train_ball.py", "Train/Re-train the patch specialist on ball_patches"),
    ("predict_combo.py", "Run combined predictions (general + ball specialist)"),
]


def run_step(script, desc):
    print(f"\n=== Running: {script} — {desc} ===")
    parts = script.split()  # allow args
    result = subprocess.run([sys.executable, *parts], cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {script} (exit code {result.returncode})")


if __name__ == "__main__":
    print(f"[INFO] Working dir: {BASE_DIR}")
    print(f"[INFO] Data root (from scripts): {AT_IT6_DATA}")
    for script, desc in steps:
        p = BASE_DIR / script
        if p.exists():
            run_step(script, desc)
        else:
            print(f"Skipping missing script: {script}")
    print("\nPipeline complete.")
