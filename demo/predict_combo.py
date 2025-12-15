"""
predict_combo.py — Combine general model (QB/Def/Rec) with ball specialist.

Runs per-image inference using:
- General model for classes: Defender(0), QB(1), Receiver(3)
- Ball specialist for class Ball(2) only (remapped from specialist's class 0)

Outputs:
- Overlays to demo/pred_vis_combo/ images
- Combined detections CSV at demo/pred_vis_combo/detections_combo.csv
"""

from ultralytics import YOLO
import os, glob, warnings, argparse
import torch
import cv2
import pandas as pd
import numpy as np

# ---- env / perf ----
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
warnings.filterwarnings("ignore", message=".*Deterministic behavior was enabled.*")

torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---- paths ----
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = str(BASE_DIR.parent)

# General model (QB/Def/Rec/Ball trained on full data)
GENERAL_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "demo",
    "runs",
    "detect",
    "qbcv_v8x_ballfocus_p23",
    "weights",
    "best.pt",
)

# Ball specialist at 1024 (latest complete run)
BALL_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "demo",
    "runs",
    "detect",
    "ball_patches_l_p2_tinyobj_1024_lr5e-4_nomosaic_coslr_w5_ls0012",
    "weights",
    "best.pt",
)

# Image directory (try _dropped fallback to val)
IMG_DIR_A = os.path.join(PROJECT_ROOT, "at-it6", "data", "images", "_dropped")
IMG_DIR_B = os.path.join(PROJECT_ROOT, "at-it6", "data", "images", "val")
IMG_DIR = IMG_DIR_A if os.path.isdir(IMG_DIR_A) else IMG_DIR_B

OUT_DIR = os.path.join(PROJECT_ROOT, "demo", "pred_vis_combo")
OUT_VIS = os.path.join(OUT_DIR, "vis")
os.makedirs(OUT_VIS, exist_ok=True)

# ---- thresholds / sizing ----
# Use earlier PR scan guidance and increase imgsz so the ball gets enough pixels
GEN_CONF, GEN_IOU, GEN_IMGSZ = 0.30, 0.50, 1536
BALL_CONF, BALL_IOU, BALL_IMGSZ = 0.25, 0.50, 1536
BALL_TTA = True  # enable simple TTA for specialist (scales/flips)
FALLBACK_GENERAL_BALL = True  # if specialist finds none, keep general's ball detections

# Optional tile-based inference for ball specialist to boost recall
BALL_TILE = True
BALL_TILE_SIZE = 1024
BALL_TILE_OVERLAP = 0.30  # 30% overlap to avoid edge misses

# ---- class map/colors ----
# Full model classes: {0: Defender, 1: QB, 2: Ball, 3: Receiver}
NAMES = {0: "Defender", 1: "QB", 2: "Ball", 3: "Receiver"}
COLORS = {0: (255, 160, 60), 1: (60, 160, 255), 2: (60, 255, 80), 3: (255, 80, 160)}


def parse_bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, default=IMG_DIR, help="Folder of images to predict")
    ap.add_argument("--out", type=str, default=OUT_DIR, help="Output folder root")
    ap.add_argument("--gen_conf", type=float, default=GEN_CONF)
    ap.add_argument("--gen_iou", type=float, default=GEN_IOU)
    ap.add_argument("--gen_imgsz", type=int, default=GEN_IMGSZ)
    ap.add_argument("--ball_conf", type=float, default=BALL_CONF)
    ap.add_argument("--ball_iou", type=float, default=BALL_IOU)
    ap.add_argument("--ball_imgsz", type=int, default=BALL_IMGSZ)
    ap.add_argument("--ball_tta", type=str, default=str(BALL_TTA))
    ap.add_argument("--ball_tile", type=str, default=str(BALL_TILE))
    ap.add_argument("--ball_tile_size", type=int, default=BALL_TILE_SIZE)
    ap.add_argument("--ball_tile_overlap", type=float, default=BALL_TILE_OVERLAP)
    ap.add_argument("--fallback_general_ball", type=str, default=str(FALLBACK_GENERAL_BALL))
    args = ap.parse_args()

    images_dir = args.images
    out_dir = args.out
    out_vis = os.path.join(out_dir, "vis")
    os.makedirs(out_vis, exist_ok=True)

    gen_conf, gen_iou, gen_imgsz = args.gen_conf, args.gen_iou, args.gen_imgsz
    ball_conf, ball_iou, ball_imgsz = args.ball_conf, args.ball_iou, args.ball_imgsz
    ball_tta = parse_bool(args.ball_tta)
    ball_tile = parse_bool(args.ball_tile)
    ball_tile_size = args.ball_tile_size
    ball_tile_overlap = args.ball_tile_overlap
    fallback_general_ball = parse_bool(args.fallback_general_ball)
    device = 0 if (torch.cuda.is_available() and torch.cuda.device_count() >= 1) else "cpu"
    print("Using device:", device)
    print("General:", GENERAL_WEIGHTS)
    print("Ball specialist:", BALL_WEIGHTS)
    print("Images from:", images_dir)

    if not os.path.isfile(GENERAL_WEIGHTS):
        raise SystemExit(f"Missing GENERAL_WEIGHTS: {GENERAL_WEIGHTS}")
    if not os.path.isfile(BALL_WEIGHTS):
        raise SystemExit(f"Missing BALL_WEIGHTS: {BALL_WEIGHTS}")
    if not os.path.isdir(images_dir):
        raise SystemExit(f"Missing images folder: {images_dir}")

    gen = YOLO(GENERAL_WEIGHTS)
    ball = YOLO(BALL_WEIGHTS)

    # collect common image types
    IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    images = []
    for pat in IMG_EXTS:
        images += glob.glob(os.path.join(images_dir, pat))
    images = sorted(images)
    rows = []

    # simple NMS for numpy arrays
    def nms_boxes(boxes, scores, iou_thr=0.5):
        if boxes.size == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return keep

    def tile_predict_ball(im_bgr):
        H, W = im_bgr.shape[:2]
        ts = BALL_TILE_SIZE
        stride = int(ts * (1.0 - BALL_TILE_OVERLAP))
        xs = list(range(0, max(1, W - ts + 1), max(1, stride)))
        ys = list(range(0, max(1, H - ts + 1), max(1, stride)))
        if xs and xs[-1] + ts < W:
            xs.append(W - ts)
        if ys and ys[-1] + ts < H:
            ys.append(H - ts)
        all_boxes = []
        all_scores = []
        for y0 in ys:
            for x0 in xs:
                tile = im_bgr[y0 : y0 + ts, x0 : x0 + ts]
                if tile.size == 0:
                    continue
                r = ball.predict(
                    source=tile,
                    conf=BALL_CONF,
                    iou=BALL_IOU,
                    imgsz=BALL_IMGSZ,
                    augment=BALL_TTA,
                    device=device,
                    verbose=False,
                    save=False,
                )[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                bx = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                bs = r.boxes.conf.cpu().numpy().astype(np.float32)
                if bx.size == 0:
                    continue
                # map back to full image coords
                bx[:, [0, 2]] += float(x0)
                bx[:, [1, 3]] += float(y0)
                all_boxes.append(bx)
                all_scores.append(bs)
        if not all_boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        keep = nms_boxes(boxes, scores, iou_thr=BALL_IOU)
        return boxes[keep], scores[keep]

    for p in images:
        im = cv2.imread(p)
        if im is None:
            continue

        gres = gen.predict(
            source=im,
            conf=gen_conf,
            iou=gen_iou,
            imgsz=gen_imgsz,
            device=device,
            verbose=False,
            save=False,
        )[0]

        # Keep non-ball from general (0,1,3)
        g_xyxy = np.zeros((0, 4), dtype=np.float32)
        g_cls = np.zeros((0,), dtype=np.int32)
        g_conf = np.zeros((0,), dtype=np.float32)
        if gres.boxes is not None and len(gres.boxes) > 0:
            gx_all = gres.boxes.xyxy.cpu().numpy().astype(np.float32)
            gc_all = gres.boxes.cls.cpu().numpy().astype(np.int32)
            gs_all = gres.boxes.conf.cpu().numpy().astype(np.float32)
            keep = np.isin(gc_all, np.array([0, 1, 3], dtype=np.int32))
            gx, gc, gs = gx_all[keep], gc_all[keep], gs_all[keep]

            # Enforce only one QB (class 1): keep highest-confidence QB if multiple
            qb_idx = np.where(gc == 1)[0]
            if qb_idx.size > 1:
                best_qb_local = qb_idx[np.argmax(gs[qb_idx])]
                mask = np.ones(gc.shape[0], dtype=bool)
                mask[qb_idx] = False
                mask[best_qb_local] = True
                gx, gc, gs = gx[mask], gc[mask], gs[mask]

            g_xyxy, g_cls, g_conf = gx, gc, gs

        # Ball from specialist: class 0 -> remap to class 2
        b_xyxy = np.zeros((0, 4), dtype=np.float32)
        b_cls = np.zeros((0,), dtype=np.int32)
        b_conf = np.zeros((0,), dtype=np.float32)
        if ball_tile:
            # override local tile settings
            nonlocal_tile_size = ball_tile_size
            nonlocal_tile_overlap = ball_tile_overlap

            # temporarily use closures via local variables
            ts, ov = nonlocal_tile_size, nonlocal_tile_overlap

            def tile_predict_ball_local(im_bgr):
                H, W = im_bgr.shape[:2]
                stride = int(ts * (1.0 - ov))
                xs = list(range(0, max(1, W - ts + 1), max(1, stride)))
                ys = list(range(0, max(1, H - ts + 1), max(1, stride)))
                if xs and xs[-1] + ts < W:
                    xs.append(W - ts)
                if ys and ys[-1] + ts < H:
                    ys.append(H - ts)
                all_boxes, all_scores = [], []
                for y0 in ys:
                    for x0 in xs:
                        tile = im_bgr[y0 : y0 + ts, x0 : x0 + ts]
                        if tile.size == 0:
                            continue
                        r = ball.predict(
                            source=tile,
                            conf=ball_conf,
                            iou=ball_iou,
                            imgsz=ball_imgsz,
                            augment=ball_tta,
                            device=device,
                            verbose=False,
                            save=False,
                        )[0]
                        if r.boxes is None or len(r.boxes) == 0:
                            continue
                        bx = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                        bs = r.boxes.conf.cpu().numpy().astype(np.float32)
                        if bx.size == 0:
                            continue
                        bx[:, [0, 2]] += float(x0)
                        bx[:, [1, 3]] += float(y0)
                        all_boxes.append(bx)
                        all_scores.append(bs)
                if not all_boxes:
                    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
                boxes = np.vstack(all_boxes)
                scores = np.concatenate(all_scores)
                keep = nms_boxes(boxes, scores, iou_thr=ball_iou)
                return boxes[keep], scores[keep]

            bx, bs = tile_predict_ball_local(im)
            if bx.size:
                b_xyxy = bx
                b_conf = bs
                b_cls = np.full((bx.shape[0],), 2, dtype=np.int32)
        else:
            bres = ball.predict(
                source=im,
                conf=ball_conf,
                iou=ball_iou,
                imgsz=ball_imgsz,
                augment=ball_tta,
                device=device,
                verbose=False,
                save=False,
            )[0]
            if bres.boxes is not None and len(bres.boxes) > 0:
                bx = bres.boxes.xyxy.cpu().numpy().astype(np.float32)
                bs = bres.boxes.conf.cpu().numpy().astype(np.float32)
                b_xyxy = bx
                b_cls = np.full((bx.shape[0],), 2, dtype=np.int32)
                b_conf = bs
        # Fallback: if specialist found none, optionally include general model's ball detections
        if fallback_general_ball and (b_xyxy.size == 0) and (
            gres.boxes is not None and len(gres.boxes) > 0
        ):
            # fallback: include general model's ball detections if specialist found none
            try:
                # reuse gx_all/gc_all/gs_all if available; else recompute
                gx_all, gc_all, gs_all
            except NameError:
                gx_all = gres.boxes.xyxy.cpu().numpy().astype(np.float32)
                gc_all = gres.boxes.cls.cpu().numpy().astype(np.int32)
                gs_all = gres.boxes.conf.cpu().numpy().astype(np.float32)
            mask_ball = gc_all == 2
            if mask_ball.any():
                b_xyxy = gx_all[mask_ball]
                b_cls = np.full((b_xyxy.shape[0],), 2, dtype=np.int32)
                b_conf = gs_all[mask_ball]

        # Combine
        xyxy = np.vstack([g_xyxy, b_xyxy]) if g_xyxy.size or b_xyxy.size else np.zeros((0, 4))
        cls = np.concatenate([g_cls, b_cls]) if g_cls.size or b_cls.size else np.zeros((0,), dtype=np.int32)
        conf = (
            np.concatenate([g_conf, b_conf]) if g_conf.size or b_conf.size else np.zeros((0,), dtype=np.float32)
        )

        # Draw
        img0 = im.copy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
            color = COLORS.get(int(c), (0, 255, 255))
            pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img0, pt1, pt2, color, 2)
            label = f"{NAMES.get(int(c), str(c))} {s:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ylab = max(pt1[1] - 6, th + 6)
            cv2.rectangle(img0, (pt1[0], ylab - th - 6), (pt1[0] + tw + 4, ylab), color, -1)
            cv2.putText(
                img0,
                label,
                (pt1[0] + 2, ylab - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        out_path = os.path.join(out_vis, os.path.basename(p))
        cv2.imwrite(out_path, img0)

        # Rows for CSV
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
            rows.append(
                {
                    "frame": os.path.basename(p),
                    "cls": int(c),
                    "conf": float(s),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )

    df = pd.DataFrame(rows).sort_values(["frame", "cls", "x1", "y1"]) if rows else pd.DataFrame()
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "detections_combo.csv")
    df.to_csv(csv_path, index=False)
    print("Saved detections:", csv_path)
    print("Overlays saved to:", out_vis)


if __name__ == "__main__":
    run()
