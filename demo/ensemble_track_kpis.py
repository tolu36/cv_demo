"""
ensemble_track_kpis.py — Recall-first ensemble + tracking + field KPIs.

Pipeline:
1) Ensure combined detections CSV exists (from predict_combo.py). If not, run it.
2) Track QB (cls=1) and Ball (cls=2) across frames (greedy nearest-neighbor).
3) If homography is available, map (cx, cy) image points to field yards.
4) Compute simple KPIs: ball travel (yds), QB path/footwork (yds), QB drop depth (yds),
   approximate time-to-throw (frames), average speeds.

Outputs:
- Tracks CSV: demo/pred_vis_combo/tracks_combo.csv
- KPIs JSON:  demo/pred_vis_combo/kpis.json
"""

import os, json, glob, subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

COMBO_DIR = PROJECT_ROOT / "demo" / "pred_vis_combo"
COMBO_VIS = COMBO_DIR / "vis"
COMBO_CSV = COMBO_DIR / "detections_combo.csv"

H_JSON_CANDIDATES = [
    PROJECT_ROOT / "demo" / "homography.json",
    PROJECT_ROOT / "at-it6" / "homography.json",
]


def ensure_combo_csv():
    if COMBO_CSV.exists():
        return
    # Run combo predictor with defaults (tiling + 1536)
    cmd = ["python", str(BASE_DIR / "predict_combo.py"), "--out", str(COMBO_DIR)]
    print("[INFO] detections_combo.csv not found — running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        raise SystemExit(f"Failed to produce detections_combo.csv: {e}")


def load_homography():
    for p in H_JSON_CANDIDATES:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Expect keys like 'H' (3x3) and optional 'scale_px_per_yd'.
            H = np.array(cfg.get("H", []), dtype=np.float32) if cfg.get("H") else None
            s = float(cfg.get("scale_px_per_yd", cfg.get("SCALE", 10.0)))
            return H, s, str(p)
    return None, 10.0, None


def centers(df):
    return (df["x1"] + df["x2"]) / 2.0, (df["y1"] + df["y2"]) / 2.0


def track_greedy(frames_rows, max_dist=120.0):
    """Greedy nearest-neighbor tracker over a list of per-frame rows (with cx, cy)."""
    tracks = []
    last_points = {}  # tid -> (cx, cy)
    next_tid = 1
    for fr, rows in frames_rows:
        new_last = {}
        for _, r in rows.iterrows():
            cx, cy = float(r.cx), float(r.cy)
            best_tid, best_d = None, 1e9
            for tid, (px, py) in last_points.items():
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_tid = d, tid
            if best_tid is not None and best_d <= max_dist:
                tracks.append((fr, int(r.cls), best_tid, cx, cy))
                new_last[best_tid] = (cx, cy)
            else:
                tid = next_tid
                next_tid += 1
                tracks.append((fr, int(r.cls), tid, cx, cy))
                new_last[tid] = (cx, cy)
        last_points = new_last
    cols = ["frame", "cls", "track_id", "cx", "cy"]
    return pd.DataFrame(tracks, columns=cols)


def to_field_points(H, pts_xy, scale_px_per_yd=10.0):
    """Project image points to top-down; return yards if scale provided.
    pts_xy: Nx2 float array (x,y in image pixels)
    """
    if H is None or pts_xy.size == 0:
        return None
    pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    # convert px to yards using scale (px per yard)
    return proj / float(scale_px_per_yd)


def kpi_from_tracks(trk_df, H, scale_px_per_yd):
    out = {}
    # helper: path length in yards
    def path_len_yds(df):
        if len(df) < 2:
            return 0.0
        xy = df[["cx", "cy"]].to_numpy(dtype=np.float32)
        if H is not None:
            xy_field = to_field_points(H, xy, scale_px_per_yd)
            d = np.sqrt(((np.diff(xy_field, axis=0)) ** 2).sum(axis=1)).sum()
        else:
            # pixel path; convert to yards approximately if scale provided
            d_px = np.sqrt(((np.diff(xy, axis=0)) ** 2).sum(axis=1)).sum()
            d = d_px / (scale_px_per_yd if scale_px_per_yd else 10.0)
        return float(d)

    # Top-1 QB track (longest frames)
    qb = trk_df[trk_df.cls == 1].copy()
    qb_len = qb.groupby("track_id").size()
    qb_tid = int(qb_len.idxmax()) if not qb_len.empty else None
    qb_track = qb[qb.track_id == qb_tid].sort_values("frame") if qb_tid else qb.iloc[0:0]

    # Ball: choose main track similarly
    ball = trk_df[trk_df.cls == 2].copy()
    ball_len = ball.groupby("track_id").size()
    ball_tid = int(ball_len.idxmax()) if not ball_len.empty else None
    ball_track = ball[ball.track_id == ball_tid].sort_values("frame") if ball_tid else ball.iloc[0:0]

    # KPIs
    out["qb_track_id"] = qb_tid
    out["ball_track_id"] = ball_tid
    out["qb_path_yds"] = round(path_len_yds(qb_track), 2)
    out["ball_travel_yds"] = round(path_len_yds(ball_track), 2)

    # QB drop depth (first N frames delta-x in yards if H provided)
    N = 12
    if len(qb_track) >= 2:
        xy = qb_track[["cx", "cy"]].to_numpy(dtype=np.float32)
        if H is not None:
            xy_field = to_field_points(H, xy, scale_px_per_yd)
            # interpret x as length axis if your homography follows centered coords
            x0 = xy_field[0, 0]
            xN = xy_field[min(N - 1, len(xy_field) - 1), 0]
            out["qb_drop_depth_yds"] = round(float(xN - x0), 2)
        else:
            out["qb_drop_depth_yds"] = None
    else:
        out["qb_drop_depth_yds"] = None

    # Time to throw (approx): first frame where ball exists and distance QB->Ball > 3 yds
    ttt = None
    if not qb_track.empty and not ball_track.empty:
        qb_xy = qb_track[["frame", "cx", "cy"]].to_numpy()
        ball_xy = ball_track[["frame", "cx", "cy"]].to_numpy()
        # align by frame
        f2qb = {int(fr): (float(x), float(y)) for fr, x, y in qb_xy}
        if H is not None:
            # pre-project all points
            qb_pts = to_field_points(H, np.array([(x, y) for _, x, y in qb_xy], np.float32), scale_px_per_yd)
            ball_pts = to_field_points(H, np.array([(x, y) for _, x, y in ball_xy], np.float32), scale_px_per_yd)
            frs_qb = [int(fr) for fr, _, _ in qb_xy]
            frs_ball = [int(fr) for fr, _, _ in ball_xy]
            fr2qb_pt = {fr: pt for fr, pt in zip(frs_qb, qb_pts)}
            for fr, pt in zip(frs_ball, ball_pts):
                if fr in fr2qb_pt:
                    d = float(np.linalg.norm(pt - fr2qb_pt[fr]))
                    if d > 3.0:
                        ttt = fr - int(qb_track.iloc[0].frame)
                        break
        else:
            # pixel heuristic
            for fr, bx, by in ball_xy:
                fr = int(fr)
                if fr in f2qb:
                    qx, qy = f2qb[fr]
                    d_px = ((bx - qx) ** 2 + (by - qy) ** 2) ** 0.5
                    if d_px > 3.0 * scale_px_per_yd:
                        ttt = fr - int(qb_track.iloc[0].frame)
                        break
    out["time_to_throw_frames"] = ttt

    return out


def main():
    ensure_combo_csv()
    df = pd.read_csv(COMBO_CSV)
    if df.empty:
        raise SystemExit("detections_combo.csv is empty — run predict_combo.py first.")

    # Compute centers
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0

    # Build per-frame lists per class
    frames = sorted(df["frame"].unique())

    # Track QB and Ball separately using greedy nearest neighbor
    tracks_all = []
    for cls_id in (1, 2):
        g = df[df.cls == cls_id].copy().sort_values(["frame", "cx", "cy"])  # stable
        frames_rows = []
        for fr in frames:
            rows = g[g.frame == fr]
            if len(rows):
                frames_rows.append((fr, rows))
        trk = track_greedy(frames_rows, max_dist=120.0 if cls_id == 1 else 160.0)
        tracks_all.append(trk)
    trk_df = pd.concat(tracks_all, ignore_index=True) if tracks_all else pd.DataFrame(columns=["frame","cls","track_id","cx","cy"])

    # Save tracks
    COMBO_DIR.mkdir(parents=True, exist_ok=True)
    tracks_csv = COMBO_DIR / "tracks_combo.csv"
    trk_df.to_csv(tracks_csv, index=False)
    print("Saved tracks:", tracks_csv)

    # Homography
    H, scale_px_per_yd, src = load_homography()
    if H is not None:
        print(f"Using homography from: {src} | scale_px_per_yd={scale_px_per_yd}")
    else:
        print("Homography not found — KPIs will approximate using pixel->yard scale only.")

    kpis = kpi_from_tracks(trk_df, H, scale_px_per_yd)
    kpi_path = COMBO_DIR / "kpis.json"
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2)
    print("Saved KPIs:", kpi_path)
    print("KPIs:", json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()

