"""
02_sanity_check.py

Purpose
- Sanity-check the effective video/world sampling rate (FPS) per recording by analyzing the exported
  `export/world.csv` timeline and computing counts inside sliding 1-second windows.

What it does
- Loads a "master" CSV (e.g., offset_master.csv / *master*.csv) and filters rows using `FILTER_COL`
  (default: 'frame_no_phone') to include only recordings/rows marked as "recorded".
- For each included master row, resolves the corresponding recording base directory and locates:
    <recording_base>/export/world.csv
- Reads only `TIME_COL` from world.csv (default: 'recording offset [secs]'), interprets it as seconds,
  and computes the number of samples falling in sliding 1s windows over the recording (step = STEP_S).
- Produces per-recording summary statistics over these 1s-window counts:
    mean/min/max/std and total rows read.

Inputs / Assumptions
- DATASET_ROOT points to the dataset root containing recordings like `004_LS47/...`.
- MASTER_HINT points to either:
    (a) the master CSV file itself, OR
    (b) a directory containing `offset_master.csv` (or another *master*.csv).
- Each recording has `export/world.csv` and it contains `TIME_COL` as a seconds timeline.
- `FILTER_COL` exists in the master CSV; only rows where it evaluates to "recorded" are processed.

Outputs
- Writes a summary CSV (OUT_CSV) containing:
    participant (full recording base path), mean_fps_1swin, min_fps_1swin, max_fps_1swin,
    std_fps_1swin, num_rows, note, world_csv
- Prints counts of resolved paths, missing world.csv, and missing TIME_COL, plus the full table.

Notes / Pitfalls
- This script does NOT validate actual video FPS from the MP4; it estimates sampling density from the
  world.csv time axis.
- The sliding-window counts reflect irregularities/gaps in timestamps (e.g., segment boundaries).
- Base-path resolution is heuristic: it tries to detect a full recording path from any master row cell,
  otherwise falls back to constructing DATASET_ROOT/participant/trial/timestamp_folder if available.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", None)

# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT = Path(r"E:\john\ET-IMU-and-head-mounted-imu")  # contains 004_LS47, 005_..., etc.

# You can point this either to:
#   - the exact CSV file, OR
#   - the "results" folder that contains offset_master.csv
MASTER_HINT = Path(r"D:\john.panagiotopoulos\ET-IMU-and-head-mounted-imu\results")

FILTER_COL = "frame_no_phone"         # only rows "recorded" here are INCLUDED; others excluded
WORLD_CSV_REL = Path("export") / "world.csv"

# For "video fps" of the exported world mp4 timeline:
TIME_COL = "recording offset [secs]"      # as you described in world.csv
STEP_S = 0.01                         # 0.01 dense; 0.1 faster

OUT_CSV = "fps_sanity_check_summary_head_world_sliding_1s_MASTER_FILTERED_based_on_recording_offset.csv"


# ============================================================
# HELPERS
# ============================================================
def is_recorded_frame_no_phone(v) -> bool:
    """Include only rows where frame_no_phone indicates recorded (robust)."""
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass

    # numeric -> include if > 0
    try:
        fv = float(v)
        return fv > 0
    except Exception:
        # string-like -> include if not empty/false-ish
        s = str(v).strip().lower()
        if s in ("", "0", "false", "no", "nan", "none"):
            return False
        return True


def sliding_counts_1s_from_seconds(t_s: np.ndarray, step_s: float = 0.1) -> np.ndarray:
    """Counts in sliding 1-second windows on a seconds axis."""
    t_s = pd.to_numeric(pd.Series(t_s), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if t_s.size < 2:
        return np.array([], dtype=int)

    t_s = np.sort(t_s)
    t_rel = t_s - t_s[0]
    duration = float(t_rel[-1])

    if duration < 1.0:
        return np.array([], dtype=int)

    starts = np.arange(0.0, duration - 1.0 + 1e-12, step_s, dtype=np.float64)
    idx_start = np.searchsorted(t_rel, starts, side="left")
    idx_end = np.searchsorted(t_rel, starts + 1.0, side="left")
    return (idx_end - idx_start).astype(int)


def participant_sort_key(path_str: str):
    """Sort by numeric prefix from '\004_LS' inside the path."""
    s = str(path_str)
    m = re.search(r"\\(\d+)_LS", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", s)
    return int(m2.group(1)) if m2 else 10**9


def resolve_master_csv(master_hint: Path) -> Path:
    """
    Resolve the master CSV path robustly.
    Accepts:
      - exact file path to offset_master.csv
      - directory path (e.g., ...\results) containing offset_master.csv
    Also tries common fallbacks and finally searches some roots.
    """
    # 1) If hint is a file, use it (if exists).
    if master_hint.suffix.lower() == ".csv":
        if master_hint.exists():
            return master_hint
        raise FileNotFoundError(f"MASTER CSV not found: {master_hint}")

    # 2) If hint is a directory, try offset_master.csv inside
    if master_hint.exists() and master_hint.is_dir():
        direct = master_hint / "offset_master.csv"
        if direct.exists():
            return direct

        # try any master-like csv inside the directory
        cands = sorted(master_hint.glob("*master*.csv"))
        if cands:
            return cands[0]

        raise FileNotFoundError(
            f"MASTER_HINT is a directory but no offset_master.csv (or *master*.csv) inside:\n{master_hint}"
        )

    # 3) Common candidate absolute locations (your usual patterns)
    candidates = [
        Path(r"E:\john.panagiotopoulos\ET-IMU-and-head-mounted-imu\results\offset_master.csv"),
        Path(r"E:\john\ET-IMU-and-head-mounted-imu\results\offset_master.csv"),
        Path(r"D:\john.panagiotopoulos\ET-IMU-and-head-mounted-imu\results\offset_master.csv"),
        Path(r"D:\john.panagiotopoulos\DERIVED_ET_IMU_head\results\offset_master.csv"),
        Path(__file__).resolve().parent / "results" / "offset_master.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # 4) Last resort: search in some roots (can take a bit, but safe)
    search_roots = [
        DATASET_ROOT,
        Path(r"E:\john"),
        Path(r"D:\john.panagiotopoulos"),
        Path.home(),
    ]
    for sr in search_roots:
        if sr.exists():
            hits = list(sr.rglob("offset_master.csv"))
            if hits:
                return hits[0]

    raise FileNotFoundError(
        "offset_master.csv not found.\n"
        f"MASTER_HINT was: {master_hint}\n"
        "Tried candidates:\n  - " + "\n  - ".join(str(c) for c in candidates) + "\n"
        "Also searched roots:\n  - " + "\n  - ".join(str(sr) for sr in search_roots)
    )


def find_recording_base_path_from_row(row: pd.Series) -> str | None:
    """
    Try to locate a full base recording path in any cell:
    ...\ET-IMU-and-head-mounted-imu\004_LS47\trial_...\2021-10-...\(maybe export)
    """
    for v in row.values:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        s = s.replace("/", "\\")
        if "ET-IMU-and-head-mounted-imu" not in s:
            continue

        # if file -> dir
        if s.lower().endswith(".csv") or s.lower().endswith(".mp4"):
            s = str(Path(s).parent)

        # if endswith export -> go up
        if s.lower().endswith("\\export"):
            s = str(Path(s).parent)

        s = s.rstrip("\\/")

        # accept existing dir
        if Path(s).is_dir():
            return s

        # accept if it matches expected pattern even if not on disk (rare)
        if re.search(r"\\\d+_LS[^\\]+\\trial_[^\\]+\\\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", s):
            return s

    return None


def fallback_build_base_path(row: pd.Series) -> str | None:
    """
    If the master doesn't contain a full base path, try to build:
    DATASET_ROOT\participant\trial\timestamp_folder
    from common column names.
    """
    cols = {c.lower(): c for c in row.index}

    def get_any(*names):
        for n in names:
            if n in cols:
                val = row[cols[n]]
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if not pd.isna(val):
                    return str(val).strip()
        return None

    participant = get_any("participant", "observer", "subject", "person", "id")
    trial = get_any("trial", "recording", "recording_id", "session")
    ts_folder = get_any("timestamp_folder", "folder", "datetime", "export_folder", "timestamp")

    if participant and trial and ts_folder:
        base = DATASET_ROOT / participant / trial / ts_folder
        return str(base)

    return None


# ============================================================
# MAIN
# ============================================================
master_csv_path = resolve_master_csv(MASTER_HINT)
print("Using MASTER_CSV:", master_csv_path)

master = pd.read_csv(master_csv_path)

if FILTER_COL not in master.columns:
    raise KeyError(
        f"'{FILTER_COL}' not found in master CSV.\n"
        f"Columns are: {list(master.columns)}"
    )

# Filter included rows
mask = master[FILTER_COL].apply(is_recorded_frame_no_phone)
master_inc = master[mask].copy()

print(f"MASTER rows: {len(master)} | INCLUDED by '{FILTER_COL}': {len(master_inc)}")

results = []
resolved_base = 0
missing_world_csv = 0
missing_time_col = 0

for _, row in master_inc.iterrows():
    base_path = find_recording_base_path_from_row(row)
    if base_path is None:
        base_path = fallback_build_base_path(row)

    if base_path is None:
        results.append({
            "participant": "UNKNOWN_BASE_PATH",
            "mean_fps_1swin": np.nan,
            "min_fps_1swin": np.nan,
            "max_fps_1swin": np.nan,
            "std_fps_1swin": np.nan,
            "num_rows": np.nan,
            "note": "Could not resolve base recording path from master row",
            "world_csv": ""
        })
        continue

    resolved_base += 1

    base_path_p = Path(base_path)
    world_csv = base_path_p / WORLD_CSV_REL  # export/world.csv

    if not world_csv.exists():
        missing_world_csv += 1
        results.append({
            "participant": str(base_path_p),  # full recording path as requested
            "mean_fps_1swin": np.nan,
            "min_fps_1swin": np.nan,
            "max_fps_1swin": np.nan,
            "std_fps_1swin": np.nan,
            "num_rows": np.nan,
            "note": f"Missing world.csv at: {world_csv}",
            "world_csv": str(world_csv)
        })
        continue

    # Read only the needed column
    try:
        df = pd.read_csv(world_csv, usecols=[TIME_COL])
    except ValueError:
        missing_time_col += 1
        cols = list(pd.read_csv(world_csv, nrows=0).columns)
        results.append({
            "participant": str(base_path_p),
            "mean_fps_1swin": np.nan,
            "min_fps_1swin": np.nan,
            "max_fps_1swin": np.nan,
            "std_fps_1swin": np.nan,
            "num_rows": np.nan,
            "note": f"Missing '{TIME_COL}'. Columns: {cols}",
            "world_csv": str(world_csv)
        })
        continue
    except Exception as e:
        results.append({
            "participant": str(base_path_p),
            "mean_fps_1swin": np.nan,
            "min_fps_1swin": np.nan,
            "max_fps_1swin": np.nan,
            "std_fps_1swin": np.nan,
            "num_rows": np.nan,
            "note": f"Read failed: {type(e).__name__}: {e}",
            "world_csv": str(world_csv)
        })
        continue

    counts_1s = sliding_counts_1s_from_seconds(df[TIME_COL].to_numpy(), step_s=STEP_S)

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
        "participant": str(base_path_p),  # full recording path
        "mean_fps_1swin": mean_v,
        "min_fps_1swin": min_v,
        "max_fps_1swin": max_v,
        "std_fps_1swin": std_v,
        "num_rows": int(len(df)),
        "note": note,
        "world_csv": str(world_csv)
    })

results_df = pd.DataFrame(results)

print(f"Resolved base paths: {resolved_base}/{len(master_inc)}")
print(f"Missing export/world.csv: {missing_world_csv}")
print(f"Missing TIME_COL '{TIME_COL}': {missing_time_col}")

if not results_df.empty:
    results_df["participant_num"] = results_df["participant"].apply(participant_sort_key)
    results_df = results_df.sort_values(["participant_num", "participant"]).drop(columns="participant_num")

print(results_df)

results_df.to_csv(OUT_CSV, index=False)
print("Saved summary to:", OUT_CSV)