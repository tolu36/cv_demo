# build_ball_patches.py
import os, glob, json, math, random, shutil
import cv2
import numpy as np
from pathlib import Path
import yaml

random.seed(1337)

# ---- your project paths ----
AT = os.path.abspath("../at-it6")
DATA = os.path.join(AT, "data")
IM_TRAIN = os.path.join(DATA, "images", "train")
IM_VAL = os.path.join(DATA, "images", "val")
LB_TRAIN = os.path.join(DATA, "labels", "train")
LB_VAL = os.path.join(DATA, "labels", "val")

OUT = os.path.join(DATA, "ball_patches")  # new dataset root
IMG_T = os.path.join(OUT, "images", "train")
IMG_V = os.path.join(OUT, "images", "val")
LB_T = os.path.join(OUT, "labels", "train")
LB_V = os.path.join(OUT, "labels", "val")
for d in (IMG_T, IMG_V, LB_T, LB_V):
    os.makedirs(d, exist_ok=True)

# settings
BALL_ID = 2  # your Ball class id in the full-frame labels
MARGIN = 3.0  # crop side = MARGIN * max(ball_w_px, ball_h_px)
SIDE_MIN, SIDE_MAX = 96, 384  # clamp crop side in px (keeps scale reasonable)
NEGS_PER_POS = 1  # number of negative patches per ball (near negatives)
VAL_FRACTION = 0.2
RAND = random.Random(42)


def find_image(basename):
    for r in (IM_TRAIN, IM_VAL):
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            p = os.path.join(r, basename + ext)
            if os.path.exists(p):
                return p
    return None


def load_labels(txt):
    out = []
    if not os.path.exists(txt):
        return out
    for ln in open(txt, "r", encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        c, x, y, w, h = map(float, ln.split())
        out.append((int(c), x, y, w, h))
    return out


def save_patch(dst_img, dst_txt, patch, box_norm_or_None):
    cv2.imwrite(dst_img, patch)
    if box_norm_or_None is None:
        # negative => write empty file (YOLO accepts empty label file)
        open(dst_txt, "w").close()
    else:
        c, x, y, w, h = box_norm_or_None
        with open(dst_txt, "w") as f:
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def clamp(a, lo, hi):
    return max(lo, min(hi, a))


def crop_one(img, H, W, cx, cy, side):
    x1 = clamp(int(cx - side / 2), 0, W - 1)
    y1 = clamp(int(cy - side / 2), 0, H - 1)
    x2 = clamp(x1 + int(side), 1, W)
    y2 = clamp(y1 + int(side), 1, H)
    crop = img[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def box_to_norm(px_box, crop_rect):
    # px_box: (cx,cy,w,h) in px; crop_rect: (x1,y1,x2,y2)
    x1, y1, x2, y2 = crop_rect
    cw, ch = (x2 - x1), (y2 - y1)
    cx, cy, w, h = px_box
    nx = (cx - x1) / cw
    ny = (cy - y1) / ch
    nw = w / cw
    nh = h / ch
    return nx, ny, nw, nh


def iou_rect_rect(rectA, rectB):
    # rect=(x1,y1,x2,y2) in px
    ax1, ay1, ax2, ay2 = rectA
    bx1, by1, bx2, by2 = rectB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = areaA + areaB - inter + 1e-9
    return inter / union


def process_split(label_dir, img_out_dir, lab_out_dir):
    items = glob.glob(os.path.join(label_dir, "*.txt"))
    RAND.shuffle(items)
    kept = 0
    negs = 0
    for lab_path in items:
        base = os.path.splitext(os.path.basename(lab_path))[0]
        im_path = find_image(base)
        if not im_path:
            continue
        img = cv2.imread(im_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        labels = load_labels(lab_path)

        # collect all balls in this image
        balls = [(c, x, y, w, h) for (c, x, y, w, h) in labels if c == BALL_ID]
        if not balls:
            continue

        # choose split by image hash to avoid leakage
        into_val = (hash(base) % 100) < int(VAL_FRACTION * 100)
        IMG_DIR = IMG_V if into_val else IMG_T
        LAB_DIR = LB_V if into_val else LB_T

        for i, (c, x, y, w, h) in enumerate(balls):
            # ball box in pixels
            bw, bh = w * W, h * H
            cx, cy = x * W, y * H

            side = clamp(MARGIN * max(bw, bh), SIDE_MIN, SIDE_MAX)
            patch, rect = crop_one(img, H, W, cx, cy, side)

            # normalized box within the patch
            nx, ny, nw, nh = box_to_norm((cx, cy, bw, bh), rect)

            # keep if center is inside; gently clip to retain more positives
            if not (0.0 < nx < 1.0 and 0.0 < ny < 1.0 and nw > 0 and nh > 0):
                continue
            eps = 1e-5
            nx = max(eps, min(1 - eps, nx))
            ny = max(eps, min(1 - eps, ny))
            nw = max(eps, min(1.0, nw))
            nh = max(eps, min(1.0, nh))

            tag = f"{base}_b{i}"
            dst_img = os.path.join(IMG_DIR, tag + ".png")
            dst_txt = os.path.join(LAB_DIR, tag + ".txt")
            save_patch(dst_img, dst_txt, patch, (0, nx, ny, nw, nh))  # class 0 = ball
            kept += 1

            # negatives near the ball (offsets), ensure they do NOT include the ball
            # Build the full-image ball rect in px for IoU check
            ball_rect = (
                int(cx - bw / 2),
                int(cy - bh / 2),
                int(cx + bw / 2),
                int(cy + bh / 2),
            )
            for k in range(NEGS_PER_POS):
                dx = RAND.randint(-int(1.0 * side), int(1.0 * side))
                dy = RAND.randint(-int(1.0 * side), int(1.0 * side))
                patchN, rectN = crop_one(img, H, W, cx + dx, cy + dy, side)

                # 1) skip if ball center lands inside this negative patch
                x1, y1, x2, y2 = rectN
                if (x1 <= cx <= x2) and (y1 <= cy <= y2):
                    continue
                # 2) skip if IoU with ball bbox is non-trivial
                if iou_rect_rect(rectN, ball_rect) > 0.01:
                    continue

                tagN = f"{base}_n{i}_{k}"
                save_patch(
                    os.path.join(IMG_DIR, tagN + ".png"),
                    os.path.join(LAB_DIR, tagN + ".txt"),
                    patchN,
                    None,  # empty label
                )
                negs += 1
    print(f"done: +{kept} positives, +{negs} negatives from {label_dir}")


process_split(LB_TRAIN, IMG_T, LB_T)
process_split(LB_VAL, IMG_V, LB_V)

cfg = {
    "path": Path(OUT).as_posix(),
    "train": "images/train",
    "val": "images/val",
    "nc": 1,
    "names": ["ball"],
}

with open(os.path.join(OUT, "ball_patches.yaml"), "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Wrote", os.path.join(OUT, "ball_patches.yaml"))
