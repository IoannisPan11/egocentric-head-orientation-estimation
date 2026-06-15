r"""
06_fill_offsets_master_upper_body.py

Purpose
- Fill / compute synchronization offsets between:
  (A) manual phone time (entered by you as phone_time_str)
  (B) ET world timeline timestamps exported by pl-rec-export (world.csv)
- Produces a completed offsets CSV:
    offsets_master_upper.csv -> offsets_master_upper_done.csv

What this script does (per row of offsets_master_upper.csv)
1) Convert phone_time_str to Unix epoch nanoseconds (manual_imu_ts_ns).
   - PHONE_FMT controls the parsing format.
   - PHONE_TIME_IS_LOCAL controls whether phone_time_str is interpreted as local time
     on the current Windows machine (then converted to UTC) vs already-UTC.
2) Resolve the "canonical" ET timestamps file for this participant:
   - Search under:
       D:\upper_body_sync_2\pXX\...\*_WORK\export\world.csv
   - We only accept candidates where:
       ...\<timestamp>_WORK\export\world.csv
   - If recording_id contains a datetime (e.g. p83_2024_07_30_10_51_25_....),
     the script prefers the candidate whose parent folder name contains:
       2024-07-30-10-51-25
     Otherwise it falls back to a stable lexicographic choice.
   - Important: This script does NOT use older pupil_player_format paths anymore.
3) Use frame_no_phone to pick the row from world.csv:
   - world.csv format (new, pl-rec-export):
       file, timestamp [ns], recording offset [secs], video offset [secs]
   - idx0 = frame_no_phone - FRAME_BASE
     * FRAME_BASE = 1 means frame 1 corresponds to the FIRST data row (row index 0 in pandas).
4) Compute:
     offset_ns  = manual_imu_ts_ns - world_ts_ns
     offset_sec = offset_ns / 1e9
   Then write these columns back into the output CSV.

Inputs
- PROJECT_ROOT\offsets_master_upper.csv
  Required columns:
    rec_path, recording_id, frame_no_phone, phone_time_str,
    manual_imu_ts_ns, world_ts_ns, offset_ns, offset_sec, notes

Outputs
- PROJECT_ROOT\offsets_master_upper_done.csv
  Same columns as input, but filled for rows that can be computed.
- PROJECT_ROOT\offsets_fill_log.txt
  Human-readable log of which world.csv was chosen per participant and row-level OK/ERR.

Safety / Re-run behavior
- No raw ET files are modified.
- FORCE_RECALC:
  - True  -> recompute even if output columns already look filled
  - False -> skip rows that already have manual_imu_ts_ns + world_ts_ns + offset_ns
- Output CSV is written safely via a temp file and atomic replace.
  If Excel locks the output, a timestamped alternative file is written and an error is raised.

Common failure modes (and what they mean)
- "Cannot parse participant from recording_id":
  recording_id must start with "pXX_" (e.g. p137_...)
- "No world.csv found for pXX":
  pl-rec-export might not have run, or the folder is missing the *_WORK\export structure.
- "world.csv missing 'timestamp [ns]'":
  the file is not in the expected pl-rec-export format.
- "world.csv too short for idx0":
  frame_no_phone is larger than available rows (wrong frame number or wrong FRAME_BASE).

Notes
- The column rec_path is kept for reference but is NOT used to locate world.csv.
  The script chooses a single canonical world.csv per participant using the rules above.
"""


from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ========================= CONFIG =========================

PROJECT_ROOT = Path(r"D:\upper_body_sync_2")

INPUT_CSV = PROJECT_ROOT / "offsets_master_upper.csv"
OUTPUT_CSV = PROJECT_ROOT / "offsets_master_upper_done.csv"
LOG_FILE = PROJECT_ROOT / "offsets_fill_log.txt"

EXPORT_DIR_NAME = "export"
WORLD_CSV_NAME = "world.csv"

FRAME_BASE = 1  # 1 = frame 1 corresponds to first data row
PHONE_TIME_IS_LOCAL = True
PHONE_FMT = "%Y-%m-%d %H:%M:%S.%f"

FORCE_RECALC = True


# ==========================================================


def log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def ok(v) -> bool:
    return v is not None and str(v).strip() != ""


def is_filled(row: dict) -> bool:
    return ok(row.get("manual_imu_ts_ns")) and ok(row.get("world_ts_ns")) and ok(row.get("offset_ns"))


def phone_time_to_unix_ns(phone_str: str, is_local: bool) -> int:
    phone_str = str(phone_str).strip()
    dt_naive = datetime.strptime(phone_str, PHONE_FMT)

    if is_local:
        dt_local = dt_naive.astimezone()
        dt_utc = dt_local.astimezone(timezone.utc)
    else:
        dt_utc = dt_naive.replace(tzinfo=timezone.utc)

    return int(dt_utc.timestamp() * 1_000_000_000)


def participant_from_recording_id(recording_id: str) -> str:
    m = re.match(r"^(p\d+)_", recording_id.strip())
    if not m:
        raise ValueError(f"Invalid recording_id format: {recording_id}")
    return m.group(1)


def datetime_key_from_recording_id(recording_id: str) -> Optional[str]:
    m = re.search(r"_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_", recording_id + "_")
    if not m:
        return None
    yyyy, mm, dd, hh, mi, ss = m.groups()
    return f"{yyyy}-{mm}-{dd}-{hh}-{mi}-{ss}"


def choose_candidate_by_datetime(candidates: List[Path], dt_key: Optional[str]) -> Path:
    if not candidates:
        raise ValueError("No world.csv candidates found")

    if dt_key:
        for p in candidates:
            rec_folder = p.parent.parent.name
            if dt_key in rec_folder:
                return p

    return sorted(candidates, key=lambda x: x.as_posix())[0]


def find_exported_world_csv(recording_id: str) -> Path:
    participant = participant_from_recording_id(recording_id)
    participant_root = PROJECT_ROOT / participant

    if not participant_root.exists():
        raise FileNotFoundError(f"Participant folder not found: {participant_root}")

    candidates = []

    for world_csv in participant_root.rglob(WORLD_CSV_NAME):
        if world_csv.parent.name.lower() == EXPORT_DIR_NAME:
            if world_csv.parent.parent.name.endswith("_WORK"):
                candidates.append(world_csv)

    if not candidates:
        raise FileNotFoundError(f"No world.csv found under {participant_root}")

    dt_key = datetime_key_from_recording_id(recording_id)
    chosen = choose_candidate_by_datetime(candidates, dt_key)

    log(f"INFO {participant} | world_csv={chosen.as_posix()}")
    return chosen


def read_world_ts_ns(world_csv: Path, frame_no: int) -> int:
    import pandas as pd

    idx0 = frame_no - FRAME_BASE
    if idx0 < 0:
        raise ValueError("Negative frame index after FRAME_BASE adjustment")

    df = pd.read_csv(world_csv)

    cols = {c.strip().lower(): c for c in df.columns}
    if "timestamp [ns]" not in cols:
        raise ValueError(f"'timestamp [ns]' column not found in {world_csv}")

    ts_col = cols["timestamp [ns]"]

    if idx0 >= len(df):
        raise ValueError(f"Frame index {idx0} exceeds world.csv length {len(df)}")

    return int(df.iloc[idx0][ts_col])


def write_csv_safely(out_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    tmp_path = out_path.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    os.replace(tmp_path, out_path)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    skipped = 0
    errors = 0

    for row in rows:
        rid = row.get("recording_id", "").strip()
        frame_str = row.get("frame_no_phone", "").strip()
        phone_str = row.get("phone_time_str", "").strip()

        if not rid or not frame_str or not phone_str:
            skipped += 1
            continue

        try:
            if is_filled(row) and not FORCE_RECALC:
                skipped += 1
                continue

            frame_no = int(frame_str)

            world_csv = find_exported_world_csv(rid)

            manual_ns = phone_time_to_unix_ns(phone_str, PHONE_TIME_IS_LOCAL)
            world_ns = read_world_ts_ns(world_csv, frame_no)

            offset_ns = manual_ns - world_ns
            offset_sec = offset_ns / 1_000_000_000

            row["manual_imu_ts_ns"] = str(manual_ns)
            row["world_ts_ns"] = str(world_ns)
            row["offset_ns"] = str(offset_ns)
            row["offset_sec"] = f"{offset_sec:.6f}"

            updated += 1
            print(f"OK {rid} | frame={frame_no} | offset_sec={offset_sec:.6f}")

        except Exception as e:
            errors += 1
            row["notes"] = f"ERROR: {e}"
            print(f"ERR {rid} | {e}")

    write_csv_safely(OUTPUT_CSV, fieldnames, rows)

    print(f"Done. updated={updated} skipped={skipped} errors={errors}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()