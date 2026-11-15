"""
04_check_frame_rate.py

Checks the frame rate of every ET - Upper Body recording from the frame timestamps in
world_timestamps.csv.

Thesis step : Frame-rate check (Section 3.2.1)
Dataset     : ET - Upper Body (explored, not used for the final model)
Run on      : Windows workstation at TU Wien

Purpose
- Perform a frame rate sanity check on all exported world_timestamps.csv
  files of the ET–Upper Body dataset.
- Estimate effective FPS statistics using dense sliding 1-second windows.

Context
- After pl-rec-export, each recording contains world_timestamps.csv
  with timestamps in nanoseconds.
- Instead of assuming a fixed FPS, this script computes empirical FPS
  using sliding time windows to detect:
    • dropped frames
    • timing irregularities
    • unstable recording rates
    • corrupted exports

Input
- ROOT_DIR:
    Root directory containing ET recordings
    (expects nested folders with world_timestamps.csv files)

- TIMESTAMP_COL:
    Name of the timestamp column in nanoseconds
    (default: "timestamp [ns]")

Processing Strategy
1) Recursively scan ROOT_DIR to locate all world_timestamps.csv files.
2) For each file:
   - Load timestamps
   - Convert to relative time
   - Apply sliding 1-second window counting
   - Compute:
        • mean FPS
        • min FPS
        • max FPS
        • std FPS
        • total samples
3) Store results in a summary dataframe.
4) Sort by participant and recording.
5) Export summary to CSV.

Methodology – Sliding Window FPS Estimation
- A 1-second window [t, t+1s) is moved across the recording.
- Window step is configurable (default: 0.01s for dense sampling).
- FPS = number of frames inside each 1-second window.
- Uses efficient numpy.searchsorted (O(n log n)) instead of naive looping.

Outputs
- Console summary table.
- CSV file:
    fps_sanity_check_summary_sliding_1s.csv

Safety / Robustness
- Handles missing timestamp column.
- Handles recordings shorter than 1 second.
- Does not modify any original data.

Interpretation Guide
- mean_fps_1swin ~ expected FPS (about 20 fps for this dataset)
- std_fps_1swin ≈ 0 → stable recording
- large min/max spread → timing instability
- NaN values → corrupted or incomplete recording

Notes
- step_s can be reduced (e.g., 0.01) for denser FPS estimation.
- Smaller step increases temporal sensitivity but also computation time.
- Recommended as a preprocessing validation step before synchronization.

Saved output
- `outputs/fps_sanity_check_summary_sliding_1s.csv` is the summary written by this script.

Author Workflow Position
- Should be executed after pl-rec-export
- Should be executed before offset computation or IMU alignment
"""

import os
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", None)

ROOT_DIR = r"E:\john\ET-IMU-and-upper-body\et"
TIMESTAMP_COL = "timestamp [ns]"

def sliding_counts_1s(ts_ns: np.ndarray, step_s: float = 0.1) -> np.ndarray:
    """
    Count how many timestamps fall inside sliding 1-second windows [x, x+1s),
    where x moves from 0 to T-1 with step = step_s.
    step_s controls how dense the windows are (0.1s = 10 windows/sec).
    """
    ts_ns = np.sort(ts_ns)
    if ts_ns.size < 2:
        return np.array([], dtype=int)

    # relative time in ns from start
    t0 = ts_ns[0]
    t_ns = ts_ns - t0

    one_sec_ns = int(1e9)
    step_ns = int(step_s * 1e9)

    duration_ns = int(t_ns[-1])
    if duration_ns < one_sec_ns:
        return np.array([], dtype=int)

    starts = np.arange(0, duration_ns - one_sec_ns + 1, step_ns, dtype=np.int64)

    # Two-pointer / searchsorted trick:
    # count in [start, start+1s) = idx_end - idx_start
    idx_start = np.searchsorted(t_ns, starts, side="left")
    idx_end = np.searchsorted(t_ns, starts + one_sec_ns, side="left")
    return (idx_end - idx_start).astype(int)

results = []
csv_paths = []

for root, dirs, files in os.walk(ROOT_DIR):
    if "world_timestamps.csv" in files:
        csv_paths.append(os.path.join(root, "world_timestamps.csv"))

for csv_path in csv_paths:
    parts = csv_path.split(os.sep)
    observer_id = parts[-3]
    recording_id = parts[-2]

    df = pd.read_csv(csv_path)

    if TIMESTAMP_COL not in df.columns:
        results.append({
            "observer": observer_id,
            "recording": recording_id,
            "mean_fps_1swin": np.nan,
            "min_fps_1swin": np.nan,
            "max_fps_1swin": np.nan,
            "std_fps_1swin": np.nan,
            "num_samples": len(df),
            "note": f"Missing column: {TIMESTAMP_COL}",
            "csv_path": csv_path
        })
        continue

    ts_ns = df[TIMESTAMP_COL].to_numpy(dtype=np.int64)

    counts_1s = sliding_counts_1s(ts_ns, step_s=0.01)  # <-- step of 0.01 s gives a dense estimate

    if counts_1s.size == 0:
        mean_v = min_v = max_v = std_v = np.nan
        note = "Too short (<1s) or empty counts"
    else:
        mean_v = float(counts_1s.mean())
        min_v = int(counts_1s.min())
        max_v = int(counts_1s.max())
        std_v = float(counts_1s.std())
        note = ""

    results.append({
        "observer": observer_id,
        "recording": recording_id,
        "mean_fps_1swin": mean_v,
        "min_fps_1swin": min_v,
        "max_fps_1swin": max_v,
        "std_fps_1swin": std_v,
        "num_samples": len(df),
        "note": note,
        "csv_path": csv_path
    })

results_df = pd.DataFrame(results)

results_df["observer_num"] = results_df["observer"].str.replace("p", "", regex=False).astype(int)
results_df = results_df.sort_values(["observer_num", "recording"]).drop(columns="observer_num")

print(results_df)
print(f"TOTAL recordings found: {len(results_df)}")

out_csv = "fps_sanity_check_summary_sliding_1s.csv"
results_df.to_csv(out_csv, index=False)
print(f"Saved summary to: {out_csv}")