# add_background_patches.py
import os, glob, random, cv2, shutil
from pathlib import Path

SEED = 1337
random.seed(SEED)
RNG = random.Random(SEED)

# --- paths ---
AT = os.path.abspath("../at-it6")
DATA = os.path.join(AT, "data")
IM_FULL = {s: os.path.join(DATA, "images", s) for s in ("train", "val")}
LB_FULL = {s: os.path.join(DATA, "labels", s) for s in ("train", "val")}
PATCH_ROOT = os.path.join(DATA, "ball_patches")
IM_PATCH = {s: os.path.join(PATCH_ROOT, "images", s) for s in ("train", "val")}
LB_PATCH = {s: os.path.join(PATCH_ROOT, "labels", s) for s in ("train", "val")}
HNM_ROOT = {
    s: os.path.join(PATCH_ROOT, "hard_negs", s) for s in ("train", "val")
}  # optional

for s in ("train", "val"):
    os.makedirs(IM_PATCH[s], exist_ok=True)
    os.makedirs(LB_PATCH[s], exist_ok=True)
    os.makedirs(HNM_ROOT[s], exist_ok=True)  # so you can drop FP crops here

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# infer patch size
def infer_patch_size():
    for split in ("train", "val"):
        for ext in IMG_EXTS:
            pats = glob.glob(os.path.join(IM_PATCH[split], f"*{ext}"))
            if pats:
                im = cv2.imread(pats[0])
                if im is not None:
                    h, w = im.shape[:2]
                    return h, w
    fallback = int(os.environ.get("SAVE_SIZE", "768"))
    return fallback, fallback


ph, pw = infer_patch_size()

FULL_BALL_ID = 2

# backgrounds per positive (env override: set BG_NEG_PER_POS=3 to try 3:1)
NEG_PER_POS = int(os.environ.get("BG_NEG_PER_POS", "2"))
MAX_NEG_PER_POS_CAP = 2.5  # don’t exceed ~2.5x neg per pos overall


def yolo_to_px(lbl_path, img_w, img_h):
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
            x = (xc - w / 2) * img_w
            y = (yc - h / 2) * img_h
            boxes.append([x, y, w * img_w, h * img_h])
    return boxes


def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter + 1e-9
    return inter / union


def list_images(folder):
    imgs = []
    for ext in IMG_EXTS:
        imgs += glob.glob(os.path.join(folder, f"*{ext}"))
    return imgs


def count_positive_patches(split):
    pos = 0
    for p in Path(LB_PATCH[split]).glob("*.txt"):
        if p.stat().st_size > 0:
            pos += 1
    return pos


def count_negative_patches(split):
    neg = 0
    for p in Path(LB_PATCH[split]).glob("*.txt"):
        if p.stat().st_size == 0:
            neg += 1
    return neg


def import_hard_negs(split):
    """Optional: copy any images in ball_patches/hard_negs/<split>/ as backgrounds."""
    src = Path(HNM_ROOT[split])
    if not src.exists():
        return 0
    made = 0
    for p in list_images(str(src)):
        img = cv2.imread(p)
        if img is None:
            continue
        # resize to patch size
        patch = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
        base = f"hnm_{Path(p).stem}"
        out_img = os.path.join(IM_PATCH[split], base + ".png")
        out_lbl = os.path.join(LB_PATCH[split], base + ".txt")
        cv2.imwrite(out_img, patch)
        open(out_lbl, "w").close()
        made += 1
    if made:
        print(f"{split}: imported {made} hard negatives from {src}")
    return made


def add_negs(split):
    pos_count = count_positive_patches(split)
    if pos_count == 0:
        print(f"{split}: No positives found; skipping negatives.")
        return

    target_negs_total = int(
        min(NEG_PER_POS * pos_count, MAX_NEG_PER_POS_CAP * pos_count)
    )
    existing_negs = count_negative_patches(split)

    # import any user-supplied hard negatives first (doesn’t overrun target)
    hn_made = import_hard_negs(split)
    existing_negs += hn_made

    remaining = max(0, target_negs_total - existing_negs)
    if remaining == 0:
        print(
            f"{split}: backgrounds already at/over target ({existing_negs}/{target_negs_total}); skipping."
        )
        return

    made = 0
    full_imgs = sorted(list_images(IM_FULL[split]))
    RNG.shuffle(full_imgs)

    for img_path in full_imgs:
        if made >= remaining:
            break
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        if W < pw or H < ph:
            continue
        lbl_path = os.path.join(LB_FULL[split], Path(img_path).stem + ".txt")
        balls = yolo_to_px(lbl_path, W, H)

        for _ in range(30):
            if made >= remaining:
                break
            x = RNG.randint(0, W - pw)
            y = RNG.randint(0, H - ph)
            crop = [x, y, pw, ph]
            if any(iou_xywh(crop, b) > 0.01 for b in balls):
                continue

            patch = img[y : y + ph, x : x + pw]
            base = f"neg_{Path(img_path).stem}_{x}_{y}"
            out_img = os.path.join(IM_PATCH[split], base + ".png")
            out_lbl = os.path.join(LB_PATCH[split], base + ".txt")
            cv2.imwrite(out_img, patch)
            open(out_lbl, "w").close()
            made += 1

    print(
        f"{split}: added {made} negatives (target total {target_negs_total}, existing {existing_negs})"
    )


if __name__ == "__main__":
    add_negs("train")
    add_negs("val")
    print("Done adding negatives.")
