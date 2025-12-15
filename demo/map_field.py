import os, cv2, json, glob, random
import numpy as np
from pathlib import Path

# ---- your paths (as provided) ----
AT = os.path.abspath("../at-it6")
DATA = os.path.join(AT, "data")
IM_TRAIN = os.path.join(DATA, "images", "train")
LB_TRAIN = os.path.join(DATA, "labels", "train")
DATA_YAML = os.path.join(AT, "data.yaml")

# where to save the calibration
DEMO_DIR = os.path.abspath("../demo")
os.makedirs(DEMO_DIR, exist_ok=True)
OUT_JSON = os.path.join(DEMO_DIR, "homography.json")

# ---- centered coordinate system (recommended) ----
# X (length):  -50 … 0 … +50  (goal lines at ±50, midfield at 0)
# Y (width):     0 … 53.3      (bottom sideline=0, top sideline=53.3)
FIELD_W = 53.3  # yards (sideline to sideline)
GOAL_X = 50.0  # |x| at goal line from center
END_X = 60.0  # |x| at end line (back of end zone)
SCALE = 10.0  # pixels per yard in the top-down canvas
GRID_STEP_YDS = 5  # draw yardlines every 5 yards to match field marks


# -------------- helpers ----------------
def choose_images(root, k=5, seed=42):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, "**", e), recursive=True))
    if len(files) < k:
        raise RuntimeError(f"Found only {len(files)} images in {root}")
    random.seed(seed)
    random.shuffle(files)
    return files[:k]


def click_points(img, title):
    """Collect 4–8 clicked image points (yardline × sideline intersections)."""
    pts = []
    vis = img.copy()

    def cb(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                vis,
                str(len(pts)),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(title, vis)

    cv2.imshow(title, vis)
    cv2.setMouseCallback(title, cb)
    print(
        "\nLeft-click 4–8 IMAGE points on",
        title,
        "\nTip: click yard-line × sideline intersections. Press ENTER when done.",
    )
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 13:  # ENTER
            break
    cv2.destroyWindow(title)
    return np.array(pts, np.float32)


# ---- friendly input parser (centered axis) ----
def parse_centered_label(s: str, scale: float, field_w: float):
    """
    Accepts either:
      - numeric: 'x,y' where x ∈ [-50..+50], y ∈ [0..53.3]
      - tokens:  'R45 top', 'L10 bottom', 'center top', 'goal right top', 'end left bottom'
    Returns [x_px, y_px] in top-down canvas pixels.
    """
    t = s.strip().lower()

    # numeric 'x,y'
    if "," in t:
        x, y = map(float, t.split(","))
        return [x * scale, y * scale]

    toks = t.replace("_", " ").split()
    # y (sideline)
    y = field_w if "top" in toks else 0.0 if "bottom" in toks else None

    # center / goal / end keywords
    if "center" in toks or "c" in toks:
        x = 0.0
    elif "goal" in toks or "gl" in toks:
        # goal line at ±50 from center
        if "right" in toks or "r" in toks:
            x = +GOAL_X
        elif "left" in toks or "l" in toks:
            x = -GOAL_X
        else:
            raise ValueError("Specify 'goal left' or 'goal right'.")
    elif "end" in toks or "el" in toks:
        # end line at ±60 from center
        if "right" in toks or "r" in toks:
            x = +END_X
        elif "left" in toks or "l" in toks:
            x = -END_X
        else:
            raise ValueError("Specify 'end left' or 'end right'.")
    else:
        # patterns like 'R45 top', 'L10 bottom', or '45 right'
        side = None
        yards = None
        # collect tokens like r45/l10
        for tok in toks:
            if tok.startswith("r") and tok[1:].isdigit():
                side, yards = "right", float(tok[1:])
            elif tok.startswith("l") and tok[1:].isdigit():
                side, yards = "left", float(tok[1:])
        # fallback forms '45 right' or '10 left'
        if yards is None:
            for tok in toks:
                if tok.isdigit():
                    yards = float(tok)
            if "right" in toks or "r" in toks:
                side = side or "right"
            if "left" in toks or "l" in toks:
                side = side or "left"

        if side is None or yards is None:
            raise ValueError(
                "Use formats like 'R5 top', 'L45 bottom', 'center top', 'goal right top', or numeric 'x,y'."
            )
        x = +yards if side == "right" else -yards

    if y is None:
        raise ValueError("Specify 'top' or 'bottom' for the sideline.")
    return [x * scale, y * scale]


def ask_field_coords(n):
    print("\nCentered axis input:")
    print(
        "  - 'R5 top', 'L45 bottom', 'center top', 'goal right top', 'end left bottom'"
    )
    print("  - Or numeric 'x,y' with x∈[-50..+50], y∈[0..53.3]")
    fpts = []
    for i in range(n):
        while True:
            s = input(f"Field coords for point {i}: ").strip()
            try:
                x_px, y_px = parse_centered_label(s, SCALE, FIELD_W)
                fpts.append([x_px, y_px])
                break
            except Exception as e:
                print("  Parse error:", e)
    return np.array(fpts, np.float32)


def overlay_yardlines(img, H, every=GRID_STEP_YDS):
    """
    Draw projected yard lines back onto the input image for a quick sanity check.
    Centered axis: x in [-50, +50], draw every 5 yards (default).
    """
    Hinv = np.linalg.inv(H)
    for x_yd in range(-50, 51, every):
        pts_field = np.float32(
            [[x_yd * SCALE, 0 * SCALE], [x_yd * SCALE, FIELD_W * SCALE]]
        ).reshape(-1, 1, 2)
        pts_img = cv2.perspectiveTransform(pts_field, Hinv).reshape(-1, 2).astype(int)
        a, b = tuple(pts_img[0]), tuple(pts_img[1])

        # thicker line for goal lines and center line
        thickness = 3 if x_yd in (-50, 0, 50) else 2
        color = (0, 0, 255) if x_yd in (-50, 50) else (0, 0, 200)
        cv2.line(img, a, b, color, thickness)

        # label every 10 yards; mark center and goals
        if x_yd == 0:
            cv2.putText(img, "C", a, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif x_yd in (-50, 50):
            cv2.putText(img, "GL", a, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif x_yd % 10 == 0:
            cv2.putText(
                img, f"{x_yd}", a, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
    return img


# -------------- main ----------------
def main():
    imgs = choose_images(IM_TRAIN, k=5, seed=42)
    print("Selected images:\n - " + "\n - ".join(imgs))

    img_pts_all, fld_pts_all = [], []

    for i, ip in enumerate(imgs):
        img = cv2.imread(ip)
        if img is None:
            print("  ! Could not read", ip)
            continue
        title = f"image_{i+1}/{len(imgs)}: {Path(ip).name}"
        ipts = click_points(img, title)
        if len(ipts) < 4:
            print("  ! Need at least 4 points; skipping this image.")
            continue
        fpts = ask_field_coords(len(ipts))
        img_pts_all.append(ipts)
        fld_pts_all.append(fpts)

    if not img_pts_all:
        raise RuntimeError("No points collected.")

    img_pts = np.vstack(img_pts_all)
    fld_pts = np.vstack(fld_pts_all)

    # compute homography image -> top-down field pixels (centered axis)
    H, mask = cv2.findHomography(img_pts, fld_pts, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("Homography fit failed. Try re-running with cleaner clicks.")

    # save
    payload = {
        "coordinate_system": "centered_length_axis",  # x in [-50,+50], y in [0,53.3]
        "grid_step_yards": GRID_STEP_YDS,
        "goal_x": GOAL_X,
        "end_x": END_X,
        "H": H.tolist(),
        "scale": SCALE,
        "field_w": FIELD_W,
        "images_used": imgs,
        "num_correspondences": int(len(img_pts)),
        # handy converters if you ever want other axes later:
        "conversions": {
            "centered_to_abs120": "x_abs = x_centered + 60.0",
            "abs120_to_centered": "x_centered = x_abs - 60.0",
            "centered_to_play100": "x_play = x_centered + 50.0",
            "play100_to_centered": "x_centered = x_play - 50.0",
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✓ Saved homography to {OUT_JSON}")

    # write overlay checks
    for ip in imgs:
        img = cv2.imread(ip)
        if img is None:
            continue
        out = overlay_yardlines(img.copy(), H, every=GRID_STEP_YDS)
        op = os.path.join(DEMO_DIR, f"overlay_check_{Path(ip).stem}.png")
        cv2.imwrite(op, out)
        print("  wrote", op)

    print("\nDone. If red lines hug the painted yard lines, calibration is good.")


if __name__ == "__main__":
    main()
