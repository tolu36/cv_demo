# train_ball.py  (1024px single-pass fine-tune with stabilizers)
import os, torch, gc
from ultralytics import YOLO

BALL_PATCHES_YAML = os.path.abspath(r"..\at-it6\data\ball_patches\ball_patches.yaml")

# Be nicer to CUDA allocator to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = 0 if (torch.cuda.is_available() and torch.cuda.device_count() >= 1) else "cpu"
print("Using device:", device)

def build_preferred_model():
    """Prefer x-P2 backbone (more capacity for recall). Fallbacks are handled locally."""
    try:
        return YOLO("yolov8x-p2.pt")
    except Exception:
        try:
            m = YOLO("yolov8x-p2.yaml")
            try:
                m.load("yolov8x.pt")
            except Exception:
                pass
            return m
        except Exception:
            pass
    # fall back to L-P2
    try:
        return YOLO("yolov8l-p2.pt")
    except Exception:
        m = YOLO("yolov8l-p2.yaml")
        try:
            m.load("yolov8l.pt")
        except Exception:
            pass
        return m


# --- Choose weights ---
USE_RESUME = True  # resume from Stage 1 best
RESUME_WEIGHTS = os.path.abspath(
    r"runs\detect\ball_patches_l_p2_tinyobj_1024_lr5e-4_nomosaic_coslr_w5_ls0013\weights\best.pt"
)

if USE_RESUME and os.path.isfile(RESUME_WEIGHTS):
    print(f"Loading resume weights: {RESUME_WEIGHTS}")
    model = YOLO(RESUME_WEIGHTS)
else:
    print("Building preferred base model (x-p2 preferred, with x weights if available)...")
    model = build_preferred_model()

# Control which stages to run
RUN_S1 = False  # set True to re-run Stage 1 @1024


def tinyobj_train_args(imgsz=1024, run_name_suffix="1024"):
    return dict(
        data=BALL_PATCHES_YAML,
        imgsz=imgsz,
        epochs=80,  # one pass at higher res
        patience=40,
        batch=-1,  # auto; if OOM, try 8 or 4 explicitly
        device=device,
        workers=0,  # Windows-stable for train
        amp=True,
        cache=False,
        # LR & regularization
        lr0=5e-4,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        cos_lr=True,
        warmup_epochs=5,
        label_smoothing=0.01,
        # Augs (gentle for tiny objs)
        mosaic=0.0,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        scale=0.2,
        translate=0.05,
        degrees=2.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.25,
        flipud=0.0,
        # (Optional) try focal if class imbalance FNs persist:
        # fl_gamma=1.5,
        name=f"ball_patches_l_p2_tinyobj_{run_name_suffix}_lr5e-4_nomosaic_coslr_w5_ls001",
    )


def safe_scalar(x, default=0.0):
    try:
        if isinstance(x, (list, tuple)):
            return float(x[0])
        if hasattr(x, "shape"):
            return float(x.mean())
        return float(x)
    except Exception:
        return float(default)


def quick_val_scan(model, imgsz=1024, iou=0.60, confs=(0.20, 0.25, 0.30, 0.40)):
    print("\n[PR scan over confidence thresholds]")
    rows = []
    for c in confs:
        r = model.val(
            data=BALL_PATCHES_YAML,
            imgsz=imgsz,
            device=device,
            conf=c,
            iou=iou,
            plots=False,
            verbose=False,
            workers=0,  # <-- important on Windows
        )
        rows.append(
            (c, safe_scalar(r.box.p), safe_scalar(r.box.r), safe_scalar(r.box.map50))
        )
    for c, p, r_, m in rows:
        print(f"conf={c:>4.2f} | P={p:6.3f}  R={r_:6.3f}  mAP50={m:6.3f}")
    r0 = model.val(
        data=BALL_PATCHES_YAML,
        imgsz=imgsz,
        device=device,
        conf=0.001,
        iou=iou,
        plots=False,
        verbose=False,
        workers=0,  # <-- also here
    )
    print(
        "\n[Reference @ conf=0.001] ",
        {
            "mAP50": safe_scalar(r0.box.map50),
            "mAP50-95": safe_scalar(r0.box.map),
            "Precision": safe_scalar(r0.box.p),
            "Recall": safe_scalar(r0.box.r),
        },
    )


def main():
    # Stage 1: Train at 1024 (balanced compute)
    if RUN_S1:
        args_1024 = tinyobj_train_args(imgsz=1024, run_name_suffix="1024")
        print("\n[Train S1] Full fine-tune @1024 with cosine LR, warmup=5, ls=0.01")
        model.train(**{**args_1024, "freeze": None})
        quick_val_scan(model, imgsz=1024, iou=0.60, confs=(0.20, 0.25, 0.30, 0.40))
    else:
        print("\n[Train S1] Skipped (using provided resume weights for S2)")
        # Build dummy for referencing epochs in S2; use defaults
        args_1024 = tinyobj_train_args(imgsz=1024, run_name_suffix="1024")

    # Stage 2 (optional): push recall with 1536
    RUN_1536 = True
    if RUN_1536:
        # Free cached memory between stages
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        print("\n[Train S2] Fine-tune at higher res to lift recall")
        # Explicit, stable attempts starting with your requested 1536 batch=1
        # Use 'nbs' (nominal batch size) to keep LR scaling stable across small batches
        trials = [
            {"imgsz": 1536, "batch": 1, "nbs": 4, "run_name_suffix": "1536_b1"},
            {"imgsz": 1536, "batch": 2, "nbs": 4, "run_name_suffix": "1536_b2_nbs4"},
            {"imgsz": 1408, "batch": 2, "nbs": 4, "run_name_suffix": "1408_b2"},
            {"imgsz": 1280, "batch": 4, "nbs": 4, "run_name_suffix": "1280"},
            {"imgsz": 1280, "batch": 2, "nbs": 4, "run_name_suffix": "1280_b2_nbs4"},
        ]

        trained = False
        last_err = None
        for t in trials:
            try:
                print(f"[S2 Attempt] imgsz={t['imgsz']} batch={t['batch']} nbs={t['nbs']}")
                args_hi = tinyobj_train_args(
                    imgsz=t["imgsz"], run_name_suffix=str(t["run_name_suffix"])
                )
                args_hi = {
                    **args_hi,
                    "epochs": max(80, args_1024["epochs"]),
                    "patience": 50,
                    "batch": t["batch"],
                    "nbs": t["nbs"],
                }
                model.train(**args_hi)
                quick_val_scan(
                    model,
                    imgsz=t["imgsz"],
                    iou=0.60,
                    confs=(0.20, 0.25, 0.30, 0.40),
                )
                trained = True
                break
            except RuntimeError as e:
                msg = str(e).lower()
                last_err = e
                print(f"[S2 Attempt Failed] {e}")
                if "out of memory" not in msg and "cuda" not in msg:
                    # not a CUDA OOM; re-raise
                    raise
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
                continue
        if not trained:
            raise last_err if last_err else RuntimeError("Stage 2 training failed")


if __name__ == "__main__":
    # from multiprocessing import freeze_support; freeze_support()  # optional
    main()
