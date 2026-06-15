"""
07_add_head_epoch_all.py

Add epoch timestamps to synced head-IMU CSVs using the ET/world recording start time.

Purpose
- Convert per-recording head IMU timestamps that are currently in a recording-relative time axis
  (`timestamp_ns`) into an epoch-based time axis (`head_epoch_ns`) by anchoring them to the first
  world timestamp of the corresponding recording.

Core idea
- Each head CSV already contains `timestamp_ns` (head time aligned to the ET timeline, but typically
  relative / zero-based within the recording).
- For the same recording, `export/world.csv` contains epoch timestamps for video frames.
- We use the first world timestamp as the recording epoch anchor:
    world_start_epoch_ns = first_row(world.csv)["timestamp [ns]"]
  Then for each head sample:
    head_epoch_ns = world_start_epoch_ns + timestamp_ns

What it does
- Scans `--head_root` recursively for per-recording head CSVs (expects layout: <LS>/<trial>/<file>.csv).
- For each head CSV:
  1) Derives `rec_rel = <LS>/<trial>` from its relative path under `--head_root`.
  2) Locates the matching `world.csv` under the dataset root using:
       <root>/<LS>/<trial>/ET/*/export/world.csv
     (the ET timestamp-folder name varies, hence the glob).
  3) Reads ONLY the first data row of world.csv and extracts `timestamp [ns]` as int64 epoch ns.
  4) Streams the head CSV in chunks and writes a new CSV containing:
       - `keep_cols` (default: timestamp_ns, roll, pitch, yaw)
       - plus a new column `head_epoch_ns = timestamp_ns + world_start_epoch_ns`
     Output path mirrors the recording structure under `--out_root`.

Inputs / Assumptions
- Head input CSVs contain `timestamp_ns` and (optionally) roll/pitch/yaw columns as specified by `--keep_cols`.
- `timestamp_ns` must be fully numeric and finite (converted to int64; script raises if NaNs appear).
- `world.csv` must contain column `timestamp [ns]` and have at least one data row.
- Using the *first* world timestamp as anchor assumes `timestamp_ns` is defined relative to that same
  recording start reference (i.e., 0 ns corresponds to the first world frame timestamp).

Outputs
- Writes `<out_root>/<LS>/<trial>/<original_name>_epoch.csv`.
- Default behavior: skip if output exists; `--overwrite` forces regeneration.

Notes / Pitfalls
- If `timestamp_ns` is not actually relative to the first world timestamp (e.g., has an internal offset),
  the resulting `head_epoch_ns` will be shifted by that constant. This script intentionally does not
  attempt to estimate or correct such a bias.
- The helper `process_one_recording()` is currently unused (leftover stub); main() performs the loop.
"""


import argparse
import csv
from pathlib import Path

import pandas as pd


def norm_key(p: Path) -> str:
    # Normalize to forward slashes, no drive letter
    s = str(p).replace("\\", "/")
    # Try to drop drive like "E:/"
    if len(s) >= 3 and s[1:3] == ":/":
        s = s[3:]
    return s.strip("/")


def find_world_csv(root: Path, rec_rel: Path) -> Path | None:
    """
    For a recording like root/038_LS176/trial_1_om,
    world.csv is expected under:
      root/038_LS176/trial_1_om/ET/*/export/world.csv
    """
    rec_dir = root / rec_rel
    # ET might exist, timestamp folder name varies, so glob:
    candidates = sorted(rec_dir.glob("ET/*/export/world.csv"))
    if not candidates:
        return None
    # If multiple (rare), pick the first sorted (stable)
    return candidates[0]


def read_world_start_epoch_ns(world_csv: Path) -> int:
    """
    Read only the first row's 'timestamp [ns]' as integer.
    Uses csv module for speed and low memory.
    """
    colname = "timestamp [ns]"
    with world_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            idx = header.index(colname)
        except ValueError:
            raise ValueError(f"world.csv missing column '{colname}'. Found: {header}")

        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"world.csv is empty: {world_csv}")

        val = first_row[idx].strip()
        if val == "" or val.lower() == "nan":
            raise ValueError(f"world start timestamp is empty/NaN in: {world_csv}")

        # world timestamps are huge (epoch ns), keep as int
        return int(float(val))  # float(...) handles if it was written as 1.639e18


def ensure_int64(series: pd.Series, name: str) -> pd.Series:
    # robust convert; raise on non-finite
    x = pd.to_numeric(series, errors="raise")
    if x.isna().any():
        bad = x.isna().sum()
        raise ValueError(f"Column '{name}' has {bad} NaNs after conversion.")
    return x.astype("int64")


def process_one_recording(
    root: Path,
    head_csv: Path,
    out_csv: Path,
    keep_cols: list[str],
    overwrite: bool,
    chunksize: int,
) -> None:
    if out_csv.exists() and not overwrite:
        return

    # Determine recording relative key:
    # head_csv like: head_root/038_LS176/trial_1_om/MT_..._synced.csv
    # rec_rel = 038_LS176/trial_1_om
    rec_rel = head_csv.parent.relative_to(head_csv.parents[2])  # wrong if depth changes
    # Safer: head_root is parents? We'll pass head_root and compute outside.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root, e.g. E:\\john\\ET-IMU-and-head-mounted-imu")
    ap.add_argument(
        "--head_root",
        required=True,
        help="Head synced CSV root, e.g. E:\\john\\ET-IMU-and-head-mounted-imu\\head_synced_csv",
    )
    ap.add_argument(
        "--out_root",
        required=True,
        help="Output root, e.g. E:\\john\\ET-IMU-and-head-mounted-imu\\head_synced_csv_epoch",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument(
        "--keep_cols",
        default="timestamp_ns,roll,pitch,yaw",
        help="Comma-separated columns to keep from head CSV (head_epoch_ns is always added).",
    )
    args = ap.parse_args()

    root = Path(args.root)
    head_root = Path(args.head_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
    if "timestamp_ns" not in keep_cols:
        raise ValueError("keep_cols must include 'timestamp_ns' (your relative time column).")

    head_files = sorted(head_root.rglob("*.csv"))
    # Only process the per-recording files (skip any summary stuff if exists)
    # Heuristic: must contain MT_ and _synced in name, but keep it permissive:
    head_files = [p for p in head_files if p.is_file()]

    print(f"Found head synced CSVs: {len(head_files)} under {head_root}")

    ok = skip = fail = 0

    for i, head_csv in enumerate(head_files, 1):
        try:
            # recording relative path is two levels under head_root: LS / trial
            # head_csv: head_root/<LS>/<trial>/<file>.csv
            try:
                rel = head_csv.relative_to(head_root)
            except ValueError:
                raise RuntimeError(f"Head file not under head_root? {head_csv}")

            parts = rel.parts
            if len(parts) < 3:
                # Not in LS/trial/file.csv layout -> ignore
                skip += 1
                continue

            rec_rel = Path(parts[0]) / Path(parts[1])  # e.g. 038_LS176/trial_1_om

            world_csv = find_world_csv(root, rec_rel)
            if world_csv is None:
                raise FileNotFoundError(
                    f"No world.csv found for rec={rec_rel}. Expected: {root/rec_rel}/ET/*/export/world.csv"
                )

            world_start = read_world_start_epoch_ns(world_csv)

            out_dir = out_root / rec_rel
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / head_csv.name.replace(".csv", "_epoch.csv")

            if out_csv.exists() and not args.overwrite:
                skip += 1
                if i % 10 == 0:
                    print(f"[{i}/{len(head_files)}] SKIP exists: {out_csv}")
                continue

            # Stream read/write in chunks
            first = True
            for chunk in pd.read_csv(head_csv, chunksize=args.chunksize):
                missing = [c for c in keep_cols if c not in chunk.columns]
                if missing:
                    raise ValueError(f"{head_csv} missing columns: {missing}. Found: {list(chunk.columns)}")

                out = chunk[keep_cols].copy()

                ts_rel = ensure_int64(out["timestamp_ns"], "timestamp_ns")
                out["head_epoch_ns"] = ts_rel + int(world_start)

                # write
                out.to_csv(out_csv, index=False, mode="w" if first else "a", header=first)
                first = False

            ok += 1
            if i % 1 == 0:
                print(f"[{i}/{len(head_files)}] OK -> {out_csv} (world_start={world_start})")

        except Exception as e:
            fail += 1
            print(f"[{i}/{len(head_files)}] FAIL {head_csv}: {e}")

    print("\nDONE")
    print(f"OK={ok} SKIP={skip} FAIL={fail}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()