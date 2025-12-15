# %%
import os, yaml, random, glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

DATA_YAML = os.path.abspath("../at-it6/data.yaml")

with open(DATA_YAML, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# %%
train_list = cfg.get("train", None)
assert train_list, "data.yaml must define 'train' (txt file or folder)."

if train_list.endswith(".txt"):
    with open(os.path.join(os.path.abspath("../at-it6"), train_list), "r") as f:
        img_paths = [
            os.path.join(os.path.abspath("../at-it6"), p.strip())
            for p in f
            if p.strip()
        ]
else:
    # if a folder, list images
    img_paths = glob.glob(os.path.join(train_list, "**/*.png"), recursive=True)

names = cfg.get("names", {})
print("Class map:", names)
print("Samples:", len(img_paths))


def draw_yolo(image_path, labels_path):
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    if os.path.exists(labels_path):
        for line in open(labels_path):
            c, cx, cy, w, h = line.split()
            c = int(float(c))
            cx = float(cx)
            cy = float(cy)
            w = float(w)
            h = float(h)
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            draw.rectangle([x1, y1, x2, y2], width=3)
            draw.text((x1, y1 - 12), names.get(c, str(c)))
    return im


# show 4 random images
sampled = random.sample(img_paths, min(4, len(img_paths)))
plt.figure(figsize=(12, 9))
for i, p in enumerate(sampled, 1):
    lp = os.path.splitext(p)[0] + ".txt"
    im = draw_yolo(p, lp)
    plt.subplot(2, 2, i)
    plt.imshow(im)
    plt.axis("off")
    plt.title(os.path.basename(p))
plt.tight_layout()
plt.show()

# %%
