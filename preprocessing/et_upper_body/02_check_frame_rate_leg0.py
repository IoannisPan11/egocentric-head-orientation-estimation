"""
02_check_frame_rate_leg0.py

Checks the frame rate and the timing regularity of the leg0 recordings from their
world_timestamps.csv files.

Thesis step : Frame-rate check (Section 3.2.1)
Dataset     : ET - Upper Body (explored, not used for the final model)
Run on      : Windows workstation at TU Wien
Note        : "leg0" is the naming used in the code and folders; it corresponds to
              "video0 (leg1)" in the thesis.

Purpose
- Perform a comprehensive temporal consistency analysis on all
  world_timestamps.csv files within the ET–Upper Body dataset.
- Validate frame rate stability and detect timing irregularities
  before synchronization with IMU data.

Context
- Each recording contains world_timestamps.csv with:
      timestamps [seconds], pts
- These timestamps define the temporal backbone of the video stream.
- Any instability directly affects IMU–video synchronization accuracy.

Input
- ROOT_DIR:
      Dataset root directory (recursively scanned).
- CSV_NAME:
      Expected file name: world_timestamps.csv

Configuration Parameters
- STEP_S:
      Sliding window step size for dense 1-second FPS estimation.
      (0.01 = high temporal resolution)
- GAP_THRESH_S:
      Threshold for detecting abnormal temporal gaps (default: 0.5s)

Processing Strategy
For each world_timestamps.csv:

1) Read timestamps (seconds) and PTS values.
2) Compute sliding 1-second window frame counts:
      - Estimate empirical FPS statistics:
            • mean FPS
            • min FPS
            • max FPS
            • std FPS
3) Compute inter-frame interval statistics (Δt):
      - mean Δt
      - median Δt
      - std Δt
      - min Δt
      - max Δt
4) Detect abnormal gaps:
      - Count intervals where Δt > GAP_THRESH_S
5) Aggregate all metrics into a structured summary table.

Methodological Rationale
- Sliding window FPS detects:
      • local frame drops
      • unstable capture rates
      • temporal jitter
- Δt statistics detect:
      • irregular frame spacing
      • recording interruptions
      • pipeline corruption
- Large gap detection (>0.5s) flags severe temporal discontinuities.

Outputs
- Console summary table.
- CSV report saved to:
      <ROOT_DIR>/_logs/fps_sanity_check_summary_sliding_1s.csv

Robustness
- Handles empty or malformed files.
- Captures exceptions per recording without interrupting batch processing.
- Does not modify original data.

Interpretation Guide
- mean_fps_1swin ≈ expected FPS → stable recording.
- high std_fps_1swin → temporal jitter.
- large dt_std_s → irregular frame pacing.
- num_big_gaps_dt_gt_0p5s > 0 → potential recording interruptions.
- NaN values → corrupted or invalid export.

Position in Processing Pipeline
- Execute after pl-rec-export.
- Execute before offset estimation and IMU alignment.
- Recommended mandatory validation step in synchronization workflow.

Author Workflow Note
- Designed for batch validation across large datasets.
- Ensures temporal reliability before cross-sensor integration.
"""


import os
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", None)

# ================= CONFIG =================
ROOT_DIR = r"D:\upper_body_sync"
CSV_NAME = "world_timestamps.csv"

TIMESTAMP_SEC_COL = "timestamps [seconds]"
PTS_COL = "pts"

STEP_S = 0.01          # sliding window step (0.01 = very dense)
GAP_THRESH_S = 0.5     # flag large gaps (adjust if needed)

LOG_DIR = os.path.join(ROOT_DIR, "_logs")
OUT_CSV = os.path.join(LOG_DIR, "fps_sanity_check_summary_sliding_1s.csv")
# ==========================================


def sliding_counts_1s(t_sec: np.ndarray, step_s: float = 0.1) -> np.ndarray:
    """
    Count how many timestamps fall inside sliding 1-second windows [x, x+1s),
    where x moves from 0 to T-1 with step = step_s.
    """
    t_sec = np.sort(t_sec.astype(np.float64))
    if t_sec.size < 2:
        return np.array([], dtype=int)

    t0 = t_sec[0]
    t = t_sec - t0

    duration = t[-1]
    if duration < 1.0:
        return np.array([], dtype=int)

    starts = np.arange(0.0, duration - 1.0 + 1e-12, step_s, dtype=np.float64)

    # searchsorted trick (on float seconds)
    idx_start = np.searchsorted(t, starts, side="left")
    idx_end = np.searchsorted(t, starts + 1.0, side="left")
    return (idx_end - idx_start).astype(int)


def read_world_timestamps(csv_path: str):
    """
    File format:
    # timestamps [seconds],pts
    1.667976,0
    2.496537,54299
    ...
    """
    df = pd.read_csv(
        csv_path,
        comment="#",
        header=None,
        names=[TIMESTAMP_SEC_COL, PTS_COL]
    )
    if df.empty:
        raise RuntimeError("Empty timestamps file")

    t_sec = df[TIMESTAMP_SEC_COL].to_numpy(dtype=np.float64)
    pts = df[PTS_COL].to_numpy(dtype=np.int64)
    return t_sec, pts, len(df)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    csv_paths = []
    for root, _, files in os.walk(ROOT_DIR):
        if CSV_NAME in files:
            csv_paths.append(os.path.join(root, CSV_NAME))

    results = []

    for csv_path in csv_paths:
        rel = os.path.relpath(os.path.dirname(csv_path), ROOT_DIR)

        # try to pick participant/recording ids from path (best-effort)
        parts = rel.split(os.sep)
        participant = parts[0] if len(parts) > 0 else ""
        recording = parts[-1] if len(parts) > 0 else ""

        try:
            t_sec, pts, n_samples = read_world_timestamps(csv_path)

            # sliding 1s counts (FPS-ish)
            counts_1s = sliding_counts_1s(t_sec, step_s=STEP_S)

            if counts_1s.size == 0:
                mean_fps = min_fps = max_fps = std_fps = np.nan
                note = "Too short (<1s) or empty counts"
            else:
                mean_fps = float(counts_1s.mean())
                min_fps = int(counts_1s.min())
                max_fps = int(counts_1s.max())
                std_fps = float(counts_1s.std())
                note = ""

            # dt statistics (seconds)
            t_sorted = np.sort(t_sec)
            dt = np.diff(t_sorted)
            if dt.size == 0:
                dt_mean = dt_median = dt_std = dt_min = dt_max = np.nan
                big_gaps = 0
            else:
                dt_mean = float(dt.mean())
                dt_median = float(np.median(dt))
                dt_std = float(dt.std())
                dt_min = float(dt.min())
                dt_max = float(dt.max())
                big_gaps = int((dt > GAP_THRESH_S).sum())

            results.append({
                "rel_dir": rel,
                "participant": participant,
                "recording": recording,
                "mean_fps_1swin": mean_fps,
                "min_fps_1swin": min_fps,
                "max_fps_1swin": max_fps,
                "std_fps_1swin": std_fps,
                "dt_mean_s": dt_mean,
                "dt_median_s": dt_median,
                "dt_std_s": dt_std,
                "dt_min_s": dt_min,
                "dt_max_s": dt_max,
                "num_big_gaps_dt_gt_0p5s": big_gaps,
                "num_samples": n_samples,
                "note": note,
                "csv_path": csv_path
            })

        except Exception as e:
            results.append({
                "rel_dir": rel,
                "participant": participant,
                "recording": recording,
                "mean_fps_1swin": np.nan,
                "min_fps_1swin": np.nan,
                "max_fps_1swin": np.nan,
                "std_fps_1swin": np.nan,
                "dt_mean_s": np.nan,
                "dt_median_s": np.nan,
                "dt_std_s": np.nan,
                "dt_min_s": np.nan,
                "dt_max_s": np.nan,
                "num_big_gaps_dt_gt_0p5s": np.nan,
                "num_samples": np.nan,
                "note": str(e),
                "csv_path": csv_path
            })

    results_df = pd.DataFrame(results).sort_values(["participant", "recording", "rel_dir"])
    print(results_df)
    print(f"TOTAL recordings found: {len(results_df)}")

    results_df.to_csv(OUT_CSV, index=False)
    print(f"Saved summary to: {OUT_CSV}")


if __name__ == "__main__":
    main()