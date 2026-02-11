# run_ball_pipeline.py
import argparse
import csv
import os
import subprocess
import sys
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

RUNS_DIR = (BASE_DIR / "runs" / "detect").resolve()


def pick_best_ball_weights(runs_dir):
    pref_cols = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
        "metrics/mAP50(B)",
        "metrics/mAP50",
    ]
    prefixes = ("ball_patches_", "ball_crops_", "ball_specialist")
    best = None
    if not runs_dir.is_dir():
        return None
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith(prefixes):
            continue
        pt = p / "weights" / "best.pt"
        res = p / "results.csv"
        if not (pt.is_file() and res.is_file()):
            continue
        try:
            with res.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue
        col = next((c for c in pref_cols if c in rows[0]), None)
        if not col:
            continue
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col, "")))
            except Exception:
                pass
        if not vals:
            continue
        score = max(vals)
        if (best is None) or (score > best["score"]):
            best = {"pt": str(pt), "score": score}
    return best["pt"] if best else None


def run_step(parts, desc):
    print(f"\n=== Running: {' '.join(parts)} - {desc} ===")
    result = subprocess.run([sys.executable, *parts], cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {' '.join(parts)} (exit code {result.returncode})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-split", action="store_true", help="Run make_split.py")
    ap.add_argument(
        "--with-hard-negs", action="store_true", help="Run mine_hard_negs.py step"
    )
    ap.add_argument(
        "--hard-neg-weights",
        type=str,
        default="",
        help="Weights for mine_hard_negs.py (defaults to best ball specialist)",
    )
    ap.add_argument("--skip-predict", action="store_true", help="Skip predict_combo.py")
    args = ap.parse_args()

    print(f"[INFO] Working dir: {BASE_DIR}")
    print(f"[INFO] Data root (from scripts): {AT_IT6_DATA}")

    steps = []
    if args.with_split:
        steps.append((["make_split.py"], "Strict split & drop unlabeled (builds data.yaml)"))

    steps.append((["make_ball_only_ds.py"], "Build ball_patches (pos + near negs)"))
    steps.append((["ball_jitter.py"], "Add jittered positive patches to ball_patches"))

    if args.with_hard_negs:
        weights = args.hard_neg_weights or pick_best_ball_weights(RUNS_DIR)
        if not weights:
            raise SystemExit(
                "[ERROR] --with-hard-negs needs weights. Pass --hard-neg-weights or "
                "train a ball specialist first."
            )
        steps.append(
            (
                [
                    "mine_hard_negs.py",
                    "--weights",
                    weights,
                    "--conf",
                    "0.30",
                    "--iou",
                    "0.60",
                    "--tp_iou",
                    "0.20",
                    "--max_per_image",
                    "3",
                    "--limit",
                    "1500",
                    "--imgsz",
                    "1536",
                ],
                "Mine hard negatives from full frames",
            )
        )

    steps.append((["add_background_patch.py"], "Add pure background negatives to ball_patches"))
    steps.append((["train_ball.py"], "Train/Re-train the patch specialist on ball_patches"))
    if not args.skip_predict:
        steps.append((["predict_combo.py"], "Run combined predictions (general + ball specialist)"))

    for parts, desc in steps:
        script_path = BASE_DIR / parts[0]
        if script_path.exists():
            run_step(parts, desc)
        else:
            print(f"Skipping missing script: {parts[0]}")
    print("\nPipeline complete.")
