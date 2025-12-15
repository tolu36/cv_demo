# %%
import os, glob, random, shutil, yaml, csv
from datetime import datetime

# ---- CONFIG ----
AT_IT6_DIR = os.path.abspath("../at-it6")
DATA_ROOT = os.path.join(AT_IT6_DIR, "data")

IMG_TRAIN = os.path.join(DATA_ROOT, "images", "train")
IMG_VAL = os.path.join(DATA_ROOT, "images", "val")
LAB_TRAIN = os.path.join(DATA_ROOT, "labels", "train")
LAB_VAL = os.path.join(DATA_ROOT, "labels", "val")

DROP_DIR = os.path.join(DATA_ROOT, "images", "_dropped")  # unlabeled images go here
ORPHAN_DIR = os.path.join(
    DATA_ROOT, "labels", "_orphaned"
)  # labels without images (optional)

DATA_YAML = os.path.join(AT_IT6_DIR, "data.yaml")

VAL_RATIO = 0.20  # 20% of images -> val
RANDOM_SEED = 42
MOVE_FILES = True  # True: move; False: copy
DROP_IF_MISSING_LABEL = True  # <-- STRICT: images without labels are dropped

KEEP_EXISTING_NAMES = True  # pull names from existing YAML if present


# ---------- helpers ----------
def ensure_dirs():
    for d in (IMG_TRAIN, IMG_VAL, LAB_TRAIN, LAB_VAL, DROP_DIR, ORPHAN_DIR):
        os.makedirs(d, exist_ok=True)


def list_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ]
    )


def build_label_index(folder):
    """Map lowercase basename -> label path for fast lookup."""
    idx = {}
    for p in glob.glob(os.path.join(folder, "*.txt")):
        base = os.path.splitext(os.path.basename(p))[0].lower()
        idx[base] = p
    return idx


def label_for_image(img_path, label_index):
    base = os.path.splitext(os.path.basename(img_path))[0].lower()
    p = label_index.get(base, None)
    if p and os.path.exists(p):
        return p
    # Fallback: sidecar next to image
    sidecar = os.path.join(os.path.dirname(img_path), base + ".txt")
    return sidecar if os.path.exists(sidecar) else None


def safe_move_or_copy(src, dst_dir, move=True):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst, "skipped_same_path"
    if os.path.exists(dst):
        return dst, "exists_skip"
    if move:
        shutil.move(src, dst)
        return dst, "moved"
    else:
        shutil.copy2(src, dst)
        return dst, "copied"


def load_existing_yaml_names():
    if os.path.exists(DATA_YAML):
        try:
            with open(DATA_YAML, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            names = cfg.get("names", None)
            if isinstance(names, (dict, list)):
                return names
        except Exception:
            pass
    return {0: "Defender X,Y Pos", 1: "QB", 2: "Ball", 3: "Receiver"}


# ---------- main ----------
def main():
    ensure_dirs()

    # 1) Snapshot images currently in images/train
    imgs = list_images(IMG_TRAIN)
    if not imgs:
        raise RuntimeError(
            f"No images found under {IMG_TRAIN}. Put your dataset in folder mode first."
        )

    # 2) Build label index from labels/train (what you said you have)
    lbl_index = build_label_index(LAB_TRAIN)

    # 3) Split
    random.seed(RANDOM_SEED)
    random.shuffle(imgs)
    n = len(imgs)
    n_val = max(1, int(round(n * VAL_RATIO)))
    val_candidates = imgs[:n_val]
    keep_candidates = imgs[n_val:]

    print(f"Total images in images/train BEFORE split: {n}")
    print(
        f"Candidate val: {len(val_candidates)} | Remaining train candidates: {len(keep_candidates)}"
    )

    log_rows = []
    moved_val = 0
    moved_val_labels = 0
    dropped_to_review = 0

    # 4) Materialize VAL: only include images with labels; otherwise DROP (if strict)
    for img in val_candidates:
        lab = label_for_image(img, lbl_index)
        if lab:
            # move/copy image to images/val and label to labels/val
            _, img_status = safe_move_or_copy(img, IMG_VAL, MOVE_FILES)
            _, lab_status = safe_move_or_copy(lab, LAB_VAL, MOVE_FILES)
            moved_val += 1
            if lab_status in ("moved", "copied", "exists_skip"):
                moved_val_labels += 1
            log_rows.append(
                {
                    "split": "val",
                    "image": os.path.basename(img),
                    "label": os.path.basename(lab),
                    "image_action": img_status,
                    "label_action": lab_status,
                    "dropped": 0,
                }
            )
        else:
            if DROP_IF_MISSING_LABEL:
                _, st = safe_move_or_copy(img, DROP_DIR, MOVE_FILES)
                dropped_to_review += 1
                log_rows.append(
                    {
                        "split": "val",
                        "image": os.path.basename(img),
                        "label": "<missing>",
                        "image_action": st,
                        "label_action": "missing",
                        "dropped": 1,
                    }
                )
            else:
                # relax: still put image in val without label (not recommended for training)
                _, img_status = safe_move_or_copy(img, IMG_VAL, MOVE_FILES)
                log_rows.append(
                    {
                        "split": "val",
                        "image": os.path.basename(img),
                        "label": "<missing>",
                        "image_action": img_status,
                        "label_action": "missing",
                        "dropped": 0,
                    }
                )

    # 5) Enforce TRAIN consistency too: drop any remaining train images w/o labels
    kept_imgs = list_images(IMG_TRAIN)
    kept_miss = 0
    for img in kept_imgs:
        lab = label_for_image(img, lbl_index)
        if not lab and DROP_IF_MISSING_LABEL:
            _, st = safe_move_or_copy(img, DROP_DIR, MOVE_FILES)
            kept_miss += 1
            log_rows.append(
                {
                    "split": "train",
                    "image": os.path.basename(img),
                    "label": "<missing>",
                    "image_action": st,
                    "label_action": "missing",
                    "dropped": 1,
                }
            )

    # 6) Optional: move orphan labels (labels/train without a train image) out of the way
    train_basenames = {
        os.path.splitext(os.path.basename(p))[0].lower() for p in list_images(IMG_TRAIN)
    }
    for lab in glob.glob(os.path.join(LAB_TRAIN, "*.txt")):
        base = os.path.splitext(os.path.basename(lab))[0].lower()
        if base not in train_basenames:
            # if there's also a val image for this base, it should now be in LAB_VAL already
            _, _ = safe_move_or_copy(lab, ORPHAN_DIR, MOVE_FILES)

    print(
        f"\nVAL -> images kept: {moved_val} | labels moved/copied: {moved_val_labels} | images DROPPED: {dropped_to_review}"
    )
    print(f"TRAIN -> unlabeled images DROPPED: {kept_miss}")

    # 7) Write a small CSV log
    log_csv = os.path.join(
        AT_IT6_DIR, f"split_strict_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image",
                "label",
                "image_action",
                "label_action",
                "dropped",
            ],
        )
        w.writeheader()
        w.writerows(log_rows)
    print(f"Log written to: {log_csv}")

    # 8) Rewrite data.yaml -> folder mode with path root
    names_block = (
        load_existing_yaml_names()
        if KEEP_EXISTING_NAMES
        else {0: "Defender X,Y Pos", 1: "QB", 2: "Ball", 3: "Receiver"}
    )
    cfg = {
        "path": DATA_ROOT.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names_block,
    }
    # backup then write
    if os.path.exists(DATA_YAML):
        bk = DATA_YAML + ".bak"
        if os.path.exists(bk):
            try:
                os.remove(bk)
            except:
                pass
        os.replace(DATA_YAML, bk)
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("\nWrote data.yaml:", cfg)

    # 9) Final counts
    def count_imgs(path):
        return (
            len(glob.glob(os.path.join(path, "*.png")))
            + len(glob.glob(os.path.join(path, "*.jpg")))
            + len(glob.glob(os.path.join(path, "*.jpeg")))
        )

    ti, vi = count_imgs(IMG_TRAIN), count_imgs(IMG_VAL)
    tl, vl = len(glob.glob(os.path.join(LAB_TRAIN, "*.txt"))), len(
        glob.glob(os.path.join(LAB_VAL, "*.txt"))
    )
    print(
        f"\nFinal counts -> train images: {ti} | train labels: {tl} | val images: {vi} | val labels: {vl}"
    )
    print(
        f"Dropped images for review: {len(os.listdir(DROP_DIR)) if os.path.exists(DROP_DIR) else 0}"
    )
    print(
        f"Orphan labels moved: {len(os.listdir(ORPHAN_DIR)) if os.path.exists(ORPHAN_DIR) else 0}"
    )


# %%
if __name__ == "__main__":
    main()

# %%
