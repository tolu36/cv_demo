# %%
from ultralytics import YOLO
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)  # make sure it's not set

import warnings

warnings.filterwarnings("ignore", message=".*Deterministic behavior was enabled.*")

import torch

torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import time
import glob, math
import pandas as pd

# %%
print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA is available ✅")
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available ❌ (check drivers / CUDA / PyTorch install)")


# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)
t0 = time.time()
c = a @ b
torch.cuda.synchronize() if device == "cuda" else None
print(device, "OK in", round(time.time() - t0, 3), "s", "| norm:", float(c.norm()))
# %%


# Path to data.yaml
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_YAML = os.path.join(PROJECT_ROOT, "at-it6", "data.yaml")

# model = YOLO("yolov8x.pt")

# pick device automatically (0 if at least one CUDA GPU, else 'cpu')
device = 0 if (torch.cuda.is_available() and torch.cuda.device_count() >= 1) else "cpu"
print("Using device:", device)

# initial training model config
# model.train(
#     data=DATA_YAML,
#     epochs=150,
#     imgsz=1408,
#     batch=2,  # auto batch to fill your VRAM
#     device=device,  # <-- single GPU
#     workers=0,
#     cache="disk",
#     amp=True,
#     name="qbcv_yolov8x_1536",
#     patience=25,
# )
# %%
# training ball focus model
# --- try to load x-p2 weights; fallback to YAML + seed with yolov8x.pt ---
weights = "yolov8x-p2.pt"
try:
    model = YOLO(weights)
    print(f"Loaded weights: {weights}")
except Exception as e:
    print(
        f"Could not load '{weights}' ({e}). Building P2 model from YAML and seeding with yolov8x.pt..."
    )
    # Build P2 architecture; seed from x weights (partial transfer is expected)
    model = YOLO("yolov8x-p2.yaml")
    # If you don't have yolov8x.pt locally Ultralytics will auto-download it
    model.load(
        "yolov8x.pt"
    )  # partial load -> you'll see "Transferred N/M items" message

model.train(
    data=DATA_YAML,
    epochs=200,
    patience=40,
    imgsz=1536,  # try 1536 later (batch=1) if stable
    batch=1,  # explicit to avoid OOM during autobatch probing
    device=device,
    workers=0,  # Windows-stable
    cache="disk",
    amp=True,
    # — small-object–friendly augs —
    mosaic=0.5,
    close_mosaic=10,
    copy_paste=0.5,
    mixup=0.1,
    scale=0.5,  # limit downscaling (range 0.5..1.0)
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.5,
    fliplr=0.5,
    flipud=0.0,
    degrees=0.0,
    translate=0.1,
    shear=0.0,
    # ————————————————
    name="qbcv_v8x_ballfocus_p2",
)
