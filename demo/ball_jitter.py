import os, glob, cv2, random
from pathlib import Path

random.seed(1337)

# ----- paths -----
AT = os.path.abspath("../at-it6")
DATA = os.path.join(AT, "data")
IM_FULL = {s: os.path.join(DATA, "images", s) for s in ("train", "val")}
LB_FULL = {s: os.path.join(DATA, "labels", s) for s in ("train", "val")}
PATCH_ROOT = os.path.join(DATA, "ball_patches")
IM_PATCH = {s: os.path.join(PATCH_ROOT, "images", s) for s in ("train", "val")}
LB_PATCH = {s: os.path.join(PATCH_ROOT, "labels", s) for s in ("train", "val")}
for d in [*IM_PATCH.values(), *LB_PATCH.values()]:
    os.makedirs(d, exist_ok=True)

# ----- toggle: tighten jitter/centering when chasing precision -----
TIGHT_JITTER = False  # set True to use tighter jitter (J=0.15, margin=0.10)

# ----- config -----
BALL_ID_FULL = 2
SAVE_SIZE = 768  # keep in sync with training imgsz
if TIGHT_JITTER:
    JITTER = 0.15  # tighter: helps precision
    MARGIN_FRAC = 0.10  # keep ball further from patch borders
else:
    JITTER = 0.25  # baseline: slightly looser for recall
    MARGIN_FRAC = 0.05

EXPAND_MIN, EXPAND_MAX = 1.6, 3.6  # context around the ball before resize
EPS = 1e-6
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def yolo_to_px_boxes(lbl_path, W, H, wanted_cls=BALL_ID_FULL):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            c, xc, yc, w, h = map(float, ln.split())
            if int(c) != wanted_cls:
                continue
            bw, bh = w * W, h * H
            cx, cy = xc * W, yc * H
            boxes.append((cx, cy, bw, bh))
    return boxes


def sample_patch(cx, cy, bw, bh, W, H):
    """Choose a square crop around the ball with jitter + variable context."""
    side = max(bw, bh) * random.uniform(EXPAND_MIN, EXPAND_MAX)
    dx = (random.uniform(-JITTER, JITTER)) * side
    dy = (random.uniform(-JITTER, JITTER)) * side
    pcx, pcy = cx + dx, cy + dy

    x1 = int(round(pcx - side / 2))
    y1 = int(round(pcy - side / 2))
    x2 = int(round(pcx + side / 2))
    y2 = int(round(pcy + side / 2))

    # clamp to image
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > W:
        x1 -= x2 - W
        x2 = W
    if y2 > H:
        y1 -= y2 - H
        y2 = H

    # keep ball comfortably inside
    margin = MARGIN_FRAC * min(x2 - x1, y2 - y1)
    bx1, by1, bx2, by2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    shift_x = max(0, (x1 + margin) - bx1) + min(0, (x2 - margin) - bx2)
    shift_y = max(0, (y1 + margin) - by1) + min(0, (y2 - margin) - by2)
    x1 = int(max(0, min(W - 1, x1 - shift_x)))
    x2 = int(max(1, min(W, x2 - shift_x)))
    y1 = int(max(0, min(H - 1, y1 - shift_y)))
    y2 = int(max(1, min(H, y2 - shift_y)))

    if x2 <= x1 + 1 or y2 <= y1 + 1:
        side = int(max(bw, bh) * 2.0)
        x1 = int(max(0, min(W - side, cx - side / 2)))
        y1 = int(max(0, min(H - side, cy - side / 2)))
        x2, y2 = x1 + side, y1 + side

    return x1, y1, x2, y2


def write_patch(img, x1, y1, x2, y2, cx, cy, bw, bh, out_img, out_lbl):
    patch = img[y1:y2, x1:x2]
    ph, pw = patch.shape[:2]
    px = (cx - x1) / max(1, pw)
    py = (cy - y1) / max(1, ph)
    pw_norm = bw / max(1, pw)
    ph_norm = bh / max(1, ph)
    px = min(max(px, EPS), 1.0 - EPS)
    py = min(max(py, EPS), 1.0 - EPS)
    pw_norm = min(max(pw_norm, EPS), 1.0)
    ph_norm = min(max(ph_norm, EPS), 1.0)

    patch = cv2.resize(patch, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_img, patch)
    with open(out_lbl, "w", encoding="utf-8") as f:
        f.write(f"0 {px:.6f} {py:.6f} {pw_norm:.6f} {ph_norm:.6f}\n")


def build_split(split):
    out_count = 0
    imgs = []
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        imgs += glob.glob(os.path.join(IM_FULL[split], f"*{ext}"))
    imgs = sorted(imgs)

    for ip in imgs:
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        lp = os.path.join(LB_FULL[split], Path(ip).stem + ".txt")
        balls = yolo_to_px_boxes(lp, W, H)
        for k, (cx, cy, bw, bh) in enumerate(balls):
            x1, y1, x2, y2 = sample_patch(cx, cy, bw, bh, W, H)
            base = f"pos_{Path(ip).stem}_{k}"
            out_img = os.path.join(IM_PATCH[split], base + ".png")
            out_lbl = os.path.join(LB_PATCH[split], base + ".txt")
            write_patch(img, x1, y1, x2, y2, cx, cy, bw, bh, out_img, out_lbl)
            out_count += 1
    print(f"{split}: wrote {out_count} positive patches")


if __name__ == "__main__":
    build_split("train")
    build_split("val")
    print("Done making jittered ball patches.")
