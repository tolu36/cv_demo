QB Analytics Pipeline — From Raw Footage to Coaching KPIs

Goal: Turn raw American football footage into structured Quarterback (QB) KPIs (air yards + footwork) that an LLM can use to generate coaching suggestions.

0) Project Objective (One-liner)

Build an end-to-end computer-vision pipeline that detects key entities (ball, QB, receivers/defenders), tracks motion over time, maps trajectories to field coordinates, and extracts QB mechanics (base width, steps, plant stability, hip–shoulder timing, time-to-release). Output per-play metrics with quality flags, ready for LLM analysis.

1) Why This Matters

Analysts/coaches need reliable, automated measurements without frame-by-frame tagging.

Air-yards and footwork mechanics provide objective context for QB decision-making and technique.

Structured KPIs + quality flags let an LLM focus on what to coach and how to improve.

2) Folder Layout (suggested)
data/
  raw/                 # full games or clips (mp4)
  plays/               # auto-split plays (mp4 + metadata)
  frames/              # optional, sampled frames per play
models/
  detect/              # YOLO weights (players, ball)
  pose/                # 2D pose model weights
outputs/
  detections/
  tracks/
  homography/
  kpis/


Single entry point: process_play.py(play.mp4) -> outputs/kpis/<play_id>.json

3) Pipeline Overview (What / Why / How)
3.1 Detection (Per Frame)

What: Run detectors for QB, receivers/defenders, and (ideally) the ball.
Why: High-recall detections feed stable tracks. Missing ball near release/arrival is costly.
How:

Player YOLO (already good), Ball specialist at conf ≈ 0.22–0.30, NMS IoU ≈ 0.60–0.65.

Use 2×2 tiled inference on wide shots (~25% overlap).

Output per frame: [{cls, conf, xyxy}]

3.2 Tracking (Temporal Association)

What: Convert per-frame detections into persistent tracks (IDs) for QB, receivers, defenders, and ball.
Why: Continuous trajectories unlock release, arrival, air yards, and temporal KPIs.
How: OC-SORT / BYTE / IoU+Kalman, with gates:

Min track length ≥ 10 frames

Max gap ≤ 5 frames (linear/KF interpolation)

Speed jump ratio < 1.5×, area change < 2×

3.3 Field Homography (Pixel → Yards)

What: Estimate a planar mapping H from image pixels to field yard coordinates.
Why: You can’t compute yards without grounding to the field.
How:

Detect yard lines and hashes, match to NFL geometry.

Solve with RANSAC (DLT). Re-estimate per segment if camera pans/zooms.

Accept if inliers ≥ 6 and mean reproj ≤ 3 px (HD).

3.4 Event Timing (Release / Arrival)

With Ball:

Release: first sustained forward/upward velocity; detaches from QB hand zone.

Arrival: deceleration or near receiver/ground.
Ball-Optional (fallback):

Release proxy: QB arm-angle peak + shoulder angular velocity surge.

Arrival proxy: receiver catch animation or downfield decel near target.

3.5 KPIs

Throw/Air Metrics (needs homography; ball optional*):

Air yards (downfield + total 2D) from release→arrival.

Time-to-release (TTR): snap/first QB movement → release.

Pocket time: inside pocket polygon (OL hull or fixed box) until release.

Throw lateral yards: lateral component at release.
* If no ball, estimate intended air yards from receiver target track + release proxy; flag estimation_mode="no_ball".

QB Footwork (pose on −30…+5 frames around release/proxy):

Base width (ankle-to-ankle) in yards.

Step count & length (foot displacement peaks).

Plant stability (lead foot variance in last ~10 pre-release frames).

Hip–shoulder separation (angle(shoulders) − angle(hips)); peak timing vs release.

Pressure gap: min QB-to-nearest defender distance (yards).

3.6 Quality Flags (always attach)

Homography: {ok, inliers, reproj_px}

Ball track: {length, jump_ratio}

Pose: {coverage}

Estimation mode: "ball" or "no_ball" (fallback)

4) JSON Schemas (LLM-ready)
4.1 tracks.json (intermediate)
{
  "fps": 30,
  "ball": [{"t": 0.33, "cx": 812, "cy": 412, "w": 18, "h": 18, "conf": 0.62}],
  "qb":   [{"t": 0.00, "xyxy": [x1,y1,x2,y2], "id": 3}],
  "rec":  [{"t": 0.00, "xyxy": [..], "id": 7}],
  "def":  [{"t": 0.00, "xyxy": [..], "id": 11}]
}

4.2 homography.json
{"H": [[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]],
 "inliers": 12, "reproj_px": 1.8, "ok": true}

4.3 events.json
{"release_t": 1.24, "arrival_t": 2.01, "has_ball": true}

4.4 kpis.json (final per-play record)
{
  "air_yards": 24.1,
  "air_yards_mode": "ball",
  "ttr_s": 2.18,
  "pocket_time_s": 2.10,
  "throw_lateral_yd": -3.2,

  "base_width_yd": 1.8,
  "steps": 1,
  "step_lengths_yd": [0.9],
  "plant_stability": 0.11,
  "hip_shoulder_sep_deg": 22,
  "hip_shoulder_peak_dt_s": -0.08,
  "pressure_gap_yd_min": 2.4,

  "quality": {
    "homography_ok": true,
    "homography_inliers": 12,
    "homography_reproj_px": 1.6,
    "ball_track_len": 36,
    "ball_jump_ratio": 1.2,
    "pose_coverage": 0.76,
    "estimation_mode": "ball"
  }
}

5) LLM Integration Pattern

System prompt (sketch):

You are a QB mechanics assistant. Given play KPIs and quality flags, (1) summarize the throw, (2) assess mechanics using the rubric (base width, sequencing, stability, TTR, pressure), and (3) recommend 1–3 concrete improvements. Use hedge language if quality is low.

I/O contract: Provide the play JSON above; ask for a JSON+text response:

{
  "assessment": {
    "mechanics": {"base_width": "slightly narrow", "sequencing": "late hip-shoulder"},
    "timing": {"ttr": "slow", "pocket_time": "adequate"},
    "context": {"pressure": "moderate", "rollout": false}
  },
  "recommendations": [
    "Widen base by ~0.3 yd before plant.",
    "Start hip rotation ~80–120 ms earlier to lead shoulder.",
    "Maintain firmer plant in the last 10 frames pre-release."
  ],
  "confidence": 0.74,
  "narrative": "On this play, the QB ..."
}

6) Quick Wins & Fallbacks

Ball weak? Enable tiling, lower conf by ~0.03, mine hard negatives, short retrain (40–60 epochs).

Ship without ball: deliver TTR + footwork KPIs now; add air yards later.

Keep a fixed “golden set” of ~50 plays to regression-test changes.

7) Minimal Labeling Plan

200 plays: release & arrival timestamps (or release proxy + catch).

200 frames: yard-line/hash correspondences for homography.

~150 QB windows: light pose sanity checks (no dense labels).

8) Success Metrics (initial targets)

Ball specialist @ deployment conf: Recall ≥ 0.60, Precision ≥ 0.70

Ball track acceptance rate (meets gates) ≥ 85%

Homography: ≥ 6 inliers, mean reprojection ≤ 3 px

Air-yards error vs manual: ≤ 2 yards (median)

Pose coverage in release window: ≥ 70% frames usable

9) Getting Started (script skeleton)

detect_play.py → write frame-wise detections JSON.

track_play.py → write tracks JSON.

fit_homography.py → write H + metrics JSON.

find_events.py → write release/arrival JSON.

compute_kpis.py → merge to final per-play KPIs JSON.

aggregate.py → JSONL/Parquet collection for LLM.

Each step takes <play_id>.mp4 and produces a file in outputs/ with the same <play_id> for traceability.

10) Notes & Defaults

Ball operating point: conf 0.22–0.30, NMS IoU 0.60–0.65, max_det=10, 2×2 tiles if needed.

Tracker defaults: allow ≤5-frame gaps; smooth with KF; reject tracks with length <10.

Homography acceptance: inliers ≥6, reproj ≤3 px; otherwise flag homography_ok=false.

Pose: require keypoint conf >0.3; temporal smoothing and crop margin.

All outputs carry quality so downstream logic can filter/hedge.

11) Roadmap (2–3 weeks to V1.5)

V1 (no ball): TTR (proxy), base width, steps, plant stability, hip–shoulder timing, pressure gap + LLM coaching output.

V1.5 (with ball): add air yards, true release/arrival; refine coaching with trajectory context.

V2: pocket detection from OL hulls, route classification for top WR, throw on the move vs set, pressure attribution.

Prepared for: QB Analytics MVP — raw footage → KPIs → LLM coaching.