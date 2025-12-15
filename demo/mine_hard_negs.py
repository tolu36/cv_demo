# mine_hard_negs.py
"""
Mine hard negatives from full frames using a trained model.
- Runs inference on ../at-it6/data/images/{train,val}
- Keeps detections that DON'T match GT balls (IoU < TP_IOU)
- Crops around them and saves into ball_patches/hard_negs/{train,val}/
- Next, run add_background_patch.py (or your pipeline) to import them as backgrounds.

Usage:
  python mine_hard_negs.py --weights path\to\best.pt --conf 0.30 --iou 0.60
"""

import os, glob, math, argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---- paths (match your project) ----
AT = os.path.abspath("../at-it6")
DATA = os.path.join(AT, "data")
IM_FULL = {s: os.path.join(DATA, "images", s) for s in ("train", "val")}
LB_FULL = {s: os.path.join(DATA, "labels", s) for s in ("train", "val")}
PATCH_ROOT = os.path.join(DATA, "ball_patches")
HNM_DIR = {s: os.path.join(PATCH_ROOT, "hard_negs", s) for s in ("train", "val")}
for s in ("train", "val"):
    os.makedirs(HNM_DIR[s], exist_ok=True)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
FULL_BALL_ID = 2  # your full-dataset ball class id


def list_images(folder):
    imgs = []
    for ext in IMG_EXTS:
        imgs += glob.glob(os.path.join(folder, f"*{ext}"))
    return sorted(imgs)


def read_gt_balls(lbl_path, W, H):
    """Read YOLO labels and return ball boxes in xyxy pixels."""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            c, xc, yc, w, h = map(float, ln.split())
            if int(c) != FULL_BALL_ID:
                continue
            bw, bh = w * W, h * H
            cx, cy = xc * W, yc * H
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            boxes.append([x1, y1, x2, y2])
    return (
        np.array(boxes, dtype=np.float32)
        if boxes
        else np.zeros((0, 4), dtype=np.float32)
    )


def iou_matrix(a, b):
    """IoU between two [N,4] and [M,4] sets of xyxy boxes."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.maximum(0, inter_x2 - inter_x1)
    ih = np.maximum(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = np.maximum(0, (ax2 - ax1)) * np.maximum(0, (ay2 - ay1))
    area_b = np.maximum(0, (bx2 - bx1)) * np.maximum(0, (by2 - by1))
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def choose_square_crop(x1, y1, x2, y2, W, H, expand=1.4):
    """Square crop centered on detection, expanded a bit, clamped to image."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * expand
    side = max(24.0, min(side, max(W, H)))  # sane bounds
    sx1 = int(max(0, min(W - 1, round(cx - side / 2))))
    sy1 = int(max(0, min(H - 1, round(cy - side / 2))))
    sx2 = int(max(sx1 + 1, min(W, round(sx1 + side))))
    sy2 = int(max(sy1 + 1, min(H, round(sy1 + side))))
    return sx1, sy1, sx2, sy2


def infer_patch_size():
    # look for an existing patch; fallback to 768 if none
    for split in ("train", "val"):
        pats = list_images(os.path.join(PATCH_ROOT, "images", split))
        if pats:
            im = cv2.imread(pats[0])
            if im is not None:
                h, w = im.shape[:2]
                return h, w
    return 768, 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights", type=str, required=True, help="Path to trained weights (best.pt)"
    )
    ap.add_argument(
        "--conf", type=float, default=0.30, help="Confidence threshold for mining"
    )
    ap.add_argument(
        "--iou", type=float, default=0.60, help="NMS IoU used during mining"
    )
    ap.add_argument(
        "--tp_iou",
        type=float,
        default=0.20,
        help="IoU ≥ this with GT ball = true positive (ignore)",
    )
    ap.add_argument(
        "--max_per_image", type=int, default=3, help="Max hard negs to keep per image"
    )
    ap.add_argument("--limit", type=int, default=2000, help="Max hard negs per split")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference size for mining")
    args = ap.parse_args()

    ph, pw = infer_patch_size()
    print(f"[HNM] Using patch size: {pw}x{ph}")

    model = YOLO(args.weights)

    for split in ("train", "val"):
        out_dir = Path(HNM_DIR[split])
        out_dir.mkdir(parents=True, exist_ok=True)

        made = 0
        ims = list_images(IM_FULL[split])
        for ip in ims:
            if made >= args.limit:
                break

            img = cv2.imread(ip)
            if img is None:
                continue
            H, W = img.shape[:2]

            # run prediction
            pred = model.predict(
                source=img,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
                save=False,
                device=0 if model.device.type == "cuda" else "cpu",
            )
            if not pred or not pred[0].boxes:
                continue

            det = pred[0].boxes
            if det.xyxy is None or len(det.xyxy) == 0:
                continue

            # keep only detections that are NOT balls (per GT overlap)
            det_xyxy = det.xyxy.cpu().numpy().astype(np.float32)
            det_scores = det.conf.cpu().numpy().astype(np.float32)
            gt = read_gt_balls(
                os.path.join(LB_FULL[split], Path(ip).stem + ".txt"), W, H
            )
            ious = (
                iou_matrix(det_xyxy, gt)
                if gt.size
                else np.zeros((det_xyxy.shape[0], 0), dtype=np.float32)
            )
            max_iou_vs_gt = (
                ious.max(axis=1)
                if ious.size
                else np.zeros((det_xyxy.shape[0],), dtype=np.float32)
            )
            keep_idx = np.where(max_iou_vs_gt < args.tp_iou)[0]  # FP candidates

            # sort by score high->low and keep up to max_per_image
            keep_idx = keep_idx[np.argsort(-det_scores[keep_idx])]
            keep_idx = keep_idx[: args.max_per_image]

            for j in keep_idx:
                x1, y1, x2, y2 = det_xyxy[j]
                sx1, sy1, sx2, sy2 = choose_square_crop(
                    x1, y1, x2, y2, W, H, expand=1.4
                )
                patch = img[sy1:sy2, sx1:sx2]
                if patch.size == 0:
                    continue
                patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
                name = (
                    f"hnm_{Path(ip).stem}_{int(x1)}_{int(y1)}_{det_scores[j]:.2f}.png"
                )
                cv2.imwrite(str(out_dir / name), patch)
                made += 1
                if made >= args.limit:
                    break

        print(f"[HNM] {split}: saved {made} hard negatives to {out_dir}")


if __name__ == "__main__":
    main()
