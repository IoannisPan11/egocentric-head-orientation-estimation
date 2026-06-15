"""

01_fill_offsets_master.py
Fill offsets_master.csv (auto) from a manually picked phone timestamp and frame index.

Purpose
- Populate (or complete) per-recording synchronization offsets in `results/offsets_master.csv` by:
  1) converting a manually recorded phone timestamp string (`phone_time_str`) to Unix epoch ns, and
  2) looking up the corresponding world/video epoch timestamp in `export/world.csv` at the given frame,
  3) computing offset_ns = manual_imu_ts_ns - world_ts_ns (and offset_sec).

Core logic
- manual_imu_ts_ns:
    Derived from `phone_time_str` using PHONE_FMT and either local->UTC conversion or direct UTC.
- world_ts_ns:
    Read from `<rec_path>/export/world.csv` at row corresponding to `frame_no_phone`.
    Assumes 1 header row, then 1 row per frame:
      line_number = (frame_no_phone - FRAME_BASE) + 2
    and timestamp is stored in column index 1 (second CSV field).
- offset_ns:
    offset_ns = manual_imu_ts_ns - world_ts_ns
  (positive means the manual IMU time is ahead of the world timestamp for that frame).

Inputs
- MASTER_CSV: `PROJECT_ROOT/results/offsets_master.csv`
  Required columns (must exist in header):
    recording_id, rec_path, frame_no_phone, phone_time_str,
    manual_imu_ts_ns, world_ts_ns, offset_ns, offset_sec, notes
- For each row, computation runs only if BOTH `frame_no_phone` and `phone_time_str` are present.

Outputs
- Updates `offsets_master.csv` in-place (atomic write via temp file + os.replace).
- Writes a detailed run log to `results/offsets_fill_log.txt`.

Parameters / Assumptions
- FRAME_BASE:
    0 if `frame_no_phone` is 0-based, 1 if it matches exported filenames (frame_000001.png).
- PHONE_TIME_IS_LOCAL:
    True  -> interpret `phone_time_str` in the local OS timezone and convert to UTC.
    False -> interpret `phone_time_str` as UTC directly.
- PHONE_FMT:
    Expected phone timestamp format, default "%Y-%m-%d %H:%M:%S.%f".

Recompute policy
- If FORCE_RECALC is False and a row already has (manual_imu_ts_ns, world_ts_ns, offset_ns),
  the script will NOT recompute them; it will only fill `offset_sec` if missing.
- If FORCE_RECALC is True, all derived fields are recomputed for rows that have inputs.

Notes / Pitfalls
- `get_world_ts_ns()` reads the target line by counting lines (no full CSV load). This assumes
  `export/world.csv` row order is stable and aligned with frame indices.
- The script reads the world timestamp from the second CSV field (index 1). If world.csv format changes,
  this must be updated to a header-based lookup.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

# ============================
# Fill offsets_master.csv (auto)
# ============================

PROJECT_ROOT = Path(r"D:\john.panagiotopoulos\ET-IMU-and-head-mounted-imu")
MASTER_CSV = PROJECT_ROOT / "results" / "offsets_master.csv"
LOG_FILE = PROJECT_ROOT / "results" / "offsets_fill_log.txt"

# How YOU write frame_no_phone:
# 0 = you write 0-based frame indices
# 1 = you write 1-based (matches frame_000001.png)
FRAME_BASE = 1

# If True: recompute even if manual_imu_ts_ns/world_ts_ns/offset_ns already exist
FORCE_RECALC = False

# If phone_time_str is in local time of THIS Windows machine, keep True.
# If phone_time_str is already UTC, set False.
PHONE_TIME_IS_LOCAL = False

# Expected phone_time_str format
PHONE_FMT = "%Y-%m-%d %H:%M:%S.%f"


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def phone_time_to_unix_ns(phone_str: str, is_local: bool) -> int:
    """
    phone_str example: "2021-10-04 09:36:12.345"
    Returns Unix epoch in nanoseconds.
    """
    phone_str = phone_str.strip()
    try:
        dt_naive = datetime.strptime(phone_str, PHONE_FMT)
    except ValueError as e:
        raise ValueError(
            f"Bad phone_time_str format: '{phone_str}' (expected yyyy-mm-dd HH:MM:SS.mmm)"
        ) from e

    if is_local:
        # interpret as local time -> convert to UTC
        dt_local = dt_naive.astimezone()  # attaches local tz
        dt_utc = dt_local.astimezone(timezone.utc)
    else:
        # interpret as UTC
        dt_utc = dt_naive.replace(tzinfo=timezone.utc)

    return int(dt_utc.timestamp() * 1_000_000_000)


def get_world_ts_ns(rec_path: Path, frame_no_phone: int, frame_base: int) -> int:
    """
    world.csv mapping:
      line 1 = header
      line (frame_index_0based + 2) = that frame's row
    """
    world_csv = rec_path / "export" / "world.csv"
    if not world_csv.exists():
        raise FileNotFoundError(f"world.csv not found: {world_csv}")

    idx0 = frame_no_phone - frame_base
    if idx0 < 0:
        raise ValueError(
            f"frame_no_phone={frame_no_phone} with FRAME_BASE={frame_base} gives negative index"
        )

    target_line_no = idx0 + 2  # 1-based line number in file

    line = None
    with open(world_csv, "r", encoding="utf-8", newline="") as f:
        for i, ln in enumerate(f, start=1):
            if i == target_line_no:
                line = ln.rstrip("\n")
                break

    if line is None:
        raise ValueError(
            f"world.csv too short for frame index {idx0} (needs line {target_line_no})"
        )

    parts = line.split(",")
    if len(parts) < 2:
        raise ValueError(f"Unexpected world.csv line: {line}")

    return int(parts[1])


def ok(v) -> bool:
    return v is not None and str(v).strip() != ""


def is_filled(row: dict) -> bool:
    # θεωρούμε "filled" αν έχει offset_ns κλπ.
    # αλλά αν offset_sec λείπει, θα το συμπληρώσουμε χωρίς να ξανακάνουμε compute όλα.
    return ok(row.get("manual_imu_ts_ns")) and ok(row.get("world_ts_ns")) and ok(row.get("offset_ns"))


def main() -> None:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"MASTER not found: {MASTER_CSV}")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== START {datetime.now().isoformat()} ===\n")
        f.write(f"MASTER={MASTER_CSV}\n")
        f.write(f"FRAME_BASE={FRAME_BASE} FORCE_RECALC={FORCE_RECALC} PHONE_TIME_IS_LOCAL={PHONE_TIME_IS_LOCAL}\n")

    with open(MASTER_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("MASTER_CSV has no header")

        required = [
            "recording_id",
            "rec_path",
            "frame_no_phone",
            "phone_time_str",
            "manual_imu_ts_ns",
            "world_ts_ns",
            "offset_ns",
            "offset_sec",   # <--- NEW
            "notes",
        ]
        for c in required:
            if c not in fieldnames:
                raise ValueError(f"MASTER_CSV missing column: {c}")

        rows = list(reader)

    updated = 0
    skipped = 0
    errors = 0

    for row in rows:
        rid = row.get("recording_id", "").strip()
        rec_path_str = row.get("rec_path", "").strip()
        frame_str = (row.get("frame_no_phone") or "").strip()
        phone_str = (row.get("phone_time_str") or "").strip()

        has_inputs = frame_str != "" and phone_str != ""

        # περίπτωση 1: δεν έχεις βάλει ακόμα frame + phone → skip
        if not has_inputs:
            skipped += 1
            continue

        try:
            # Αν έχει ήδη offset_ns κλπ και δεν κάνουμε FORCE, απλά συμπλήρωσε offset_sec αν λείπει
            if is_filled(row) and not FORCE_RECALC:
                if not ok(row.get("offset_sec")):
                    off_ns = int(row["offset_ns"])
                    row["offset_sec"] = f"{off_ns / 1_000_000_000:.6f}"
                    msg = f"OK  {rid} | filled offset_sec only | offset_ns={off_ns} | offset_sec={row['offset_sec']}"
                    print(msg)
                    log(msg)
                    updated += 1
                else:
                    skipped += 1
                continue

            # αλλιώς πλήρης υπολογισμός
            rec_path = Path(rec_path_str)
            if not rec_path.exists():
                raise FileNotFoundError(f"rec_path not found: {rec_path}")

            frame_no = int(frame_str)

            imu_ns = phone_time_to_unix_ns(phone_str, PHONE_TIME_IS_LOCAL)
            world_ns = get_world_ts_ns(rec_path, frame_no, FRAME_BASE)
            off_ns = int(imu_ns - world_ns)
            off_sec = off_ns / 1_000_000_000

            row["manual_imu_ts_ns"] = str(imu_ns)
            row["world_ts_ns"] = str(world_ns)
            row["offset_ns"] = str(off_ns)
            row["offset_sec"] = f"{off_sec:.6f}"

            if row.get("notes") is None:
                row["notes"] = ""

            msg = (
                f"OK  {rid} | frame={frame_no} | imu_ns={imu_ns} | world_ns={world_ns} | "
                f"offset_ns={off_ns} | offset_sec={row['offset_sec']}"
            )
            print(msg)
            log(msg)
            updated += 1

        except Exception as e:
            errors += 1
            msg = f"ERR {rid} | {e}"
            print(msg)
            log(msg)

            notes = (row.get("notes") or "").strip()
            if notes:
                row["notes"] = f"{notes} | ERROR: {e}"
            else:
                row["notes"] = f"ERROR: {e}"

    tmp_path = MASTER_CSV.with_suffix(".csv.tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp_path, MASTER_CSV)

    end_msg = f"=== END {datetime.now().isoformat()} === updated={updated} skipped={skipped} errors={errors}"
    print(end_msg)
    log(end_msg)
    print(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()