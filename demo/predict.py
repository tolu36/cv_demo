# predict.py  — keep at most ONE QB per frame

from ultralytics import YOLO
import os, glob, math, time, warnings
import torch
import pandas as pd
import cv2
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
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "demo",
    "runs",
    "detect",
    "qbcv_v8x_ballfocus_p23",
    "weights",
    "best.pt",
)
IMG_DIR = os.path.join(PROJECT_ROOT, "at-it6", "data", "images", "_dropped")
OUT_DIR = os.path.join(PROJECT_ROOT, "demo", "pred_vis")
OUT_VIS = os.path.join(OUT_DIR, "vis_only_one_qb")
os.makedirs(OUT_VIS, exist_ok=True)

# ---- constants ----
QB_ID = 1  # class id for QB per your names: {0: Defender, 1: QB, 2: Ball, 3: Receiver}
NMS_CONF = 0.25
NMS_IOU = 0.50

# (optional) nice colors BGR
COLORS = {
    0: (255, 160, 60),  # Defender
    1: (60, 160, 255),  # QB (blue)
    2: (60, 255, 80),  # Ball
    3: (255, 80, 160),  # Receiver
}
NAMES = {0: "Defender", 1: "QB", 2: "Ball", 3: "Receiver"}

# ---- load model ----
device = 0 if (torch.cuda.is_available() and torch.cuda.device_count() >= 1) else "cpu"
print("Using device:", device)
model = YOLO(WEIGHTS)

# ---- predict & enforce single QB ----
rows = []
images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))

for p in images:
    # run prediction (no built-in saving)
    res = model.predict(
        p, conf=NMS_CONF, iou=NMS_IOU, device=device, save=False, verbose=False
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        continue

    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()

    # ---- keep only top-1 QB by confidence ----
    keep_idx = np.arange(len(cls))
    qb_idx = np.where(cls == QB_ID)[0]
    if len(qb_idx) > 1:
        # choose the single QB to keep
        best_qb_local = qb_idx[np.argmax(conf[qb_idx])]
        mask = np.ones(len(cls), dtype=bool)
        mask[qb_idx] = False
        mask[best_qb_local] = True
        keep_idx = np.where(mask)[0]

    # filtered detections
    xyxy = xyxy[keep_idx]
    cls = cls[keep_idx]
    conf = conf[keep_idx]

    # ---- draw our own overlay with only-one-QB rule ----
    img0 = cv2.imread(p)
    h, w = img0.shape[:2]
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

    out_path = os.path.join(OUT_VIS, os.path.basename(p))
    cv2.imwrite(out_path, img0)

    # ---- rows for CSV (already filtered) ----
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

# detections csv
df = pd.DataFrame(rows).sort_values(["frame", "cls", "x1", "y1"])
os.makedirs(OUT_DIR, exist_ok=True)
csv_path = os.path.join(OUT_DIR, "detections_one_qb.csv")
df.to_csv(csv_path, index=False)
print("Saved detections:", csv_path)


# ---- tiny per-class tracker (works on filtered detections) ----
def center(row):
    return ((row.x1 + row.x2) / 2.0, (row.y1 + row.y2) / 2.0)


tracks = []
if not df.empty:
    for cls_id in sorted(df.cls.unique()):
        g = df[df.cls == cls_id].copy()
        g["cx"] = (g.x1 + g.x2) / 2.0
        g["cy"] = (g.y1 + g.y2) / 2.0
        frames = sorted(g.frame.unique())
        track_id = 0
        last_points = {}
        for fr in frames:
            cur = g[g.frame == fr].copy()
            new_last = {}
            for _, row in cur.iterrows():
                # greedy nearest-neighbor
                best_tid, best_d = None, 1e9
                for tid, (pcx, pcy) in last_points.items():
                    d = math.hypot(row.cx - pcx, row.cy - pcy)
                    if d < best_d:
                        best_d, best_tid = d, tid
                if best_tid is not None and best_d < 80:
                    tracks.append((fr, cls_id, best_tid, row.cx, row.cy))
                    new_last[best_tid] = (row.cx, row.cy)
                else:
                    track_id += 1
                    tracks.append((fr, cls_id, track_id, row.cx, row.cy))
                    new_last[track_id] = (row.cx, row.cy)
            last_points = new_last

trk = pd.DataFrame(tracks, columns=["frame", "cls", "track_id", "cx", "cy"])
trk_path = os.path.join(OUT_DIR, "tracks_one_qb.csv")
trk.to_csv(trk_path, index=False)
print("Saved tracks:", trk_path)
print("Overlays (only one QB) saved to:", OUT_VIS)
