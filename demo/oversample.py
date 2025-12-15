# make_oversampled_train_folder.py
import os, glob, yaml, shutil

AT = os.path.abspath("../at-it6")
DATA = os.path.join(AT, "data")
IM_TRAIN = os.path.join(DATA, "images", "train")
LB_TRAIN = os.path.join(DATA, "labels", "train")
DATA_YAML = os.path.join(AT, "data.yaml")

# Output folders (created fresh)
IM_OUT = os.path.join(DATA, "images", "train_oversampled")
LB_OUT = os.path.join(DATA, "labels", "train_oversampled")

BALL_ID = 2
W_HAS_BALL = 3  # weight if a frame has any ball
W_TINY_BALL_BONUS = 2  # extra weight if the ball is tiny
TINY_AREA = 0.001  # "tiny" threshold on normalized (w*h)

USE_HARDLINKS = True  # saves disk; falls back to copy if not supported


# ---------------- helpers ----------------
def resolve_image(base):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(IM_TRAIN, base + ext)
        if os.path.exists(p):
            return p
    return None


def parse_label(path):
    has_ball, tiny = False, False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                c, x, y, w, h = map(float, ln.split())
                if int(c) == BALL_ID:
                    has_ball = True
                    if (w * h) <= TINY_AREA:
                        tiny = True
    except FileNotFoundError:
        pass
    return has_ball, tiny


def safe_empty_dir(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


def unique_basename(base, ext, taken):
    """Ensure unique filename when duplicating oversampled items."""
    name = f"{base}{ext}"
    i = 1
    while name in taken:
        name = f"{base}__dup{i}{ext}"
        i += 1
    taken.add(name)
    return name


def copy_with_match(img_src, lbl_src, img_dst, lbl_dst):
    if USE_HARDLINKS:
        try:
            os.link(img_src, img_dst)
        except Exception:
            shutil.copy2(img_src, img_dst)
        try:
            os.link(lbl_src, lbl_dst)
        except Exception:
            shutil.copy2(lbl_src, lbl_dst)
    else:
        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)


# ---------------- build weighted list ----------------
pairs = []  # list of (abs_img_path, abs_lbl_path, weight)
missing_label = 0
for lab in glob.glob(os.path.join(LB_TRAIN, "*.txt")):
    base = os.path.splitext(os.path.basename(lab))[0]
    img = resolve_image(base)
    if not img:
        continue
    has_ball, tiny = parse_label(lab)
    w = 1 + (W_HAS_BALL if has_ball else 0) + (W_TINY_BALL_BONUS if tiny else 0)
    if not os.path.exists(lab):
        missing_label += 1
        continue
    pairs.append((img, lab, w))

if not pairs:
    raise RuntimeError("No (image,label) pairs found in train.")

# ---------------- prepare output dirs ----------------
safe_empty_dir(IM_OUT)
safe_empty_dir(LB_OUT)

# ---------------- copy (oversample) ----------------
taken_names = set()
total_copies = 0
unique_imgs = set()

for img, lab, w in pairs:
    base = os.path.splitext(os.path.basename(img))[0]
    ext = os.path.splitext(os.path.basename(img))[1].lower()
    unique_imgs.add(img)

    for _ in range(w):
        # Create unique filename on each duplication
        out_name = unique_basename(base, ext, taken_names)
        img_dst = os.path.join(IM_OUT, out_name)
        lbl_dst = os.path.join(LB_OUT, os.path.splitext(out_name)[0] + ".txt")
        copy_with_match(img, lab, img_dst, lbl_dst)
        total_copies += 1

# ---------------- patch data.yaml ----------------
with open(DATA_YAML, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

backup = DATA_YAML + ".bak"
shutil.copy2(DATA_YAML, backup)

cfg["path"] = DATA.replace("\\", "/")
cfg["train"] = "images/train_oversampled"  # folder mode
cfg["val"] = cfg.get("val", "images/val")

with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Unique source train images: {len(unique_imgs)}")
print(f"Total oversampled copies written: {total_copies}")
print(f"Output images dir: {IM_OUT}")
print(f"Output labels dir: {LB_OUT}")
print(f"Backed up data.yaml -> {backup}")
print(f"Updated data.yaml: train = images/train_oversampled, val = {cfg['val']}")
