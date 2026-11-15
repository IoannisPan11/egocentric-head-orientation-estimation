"""
07_compute_angles_from_quaternions.py

Computes roll, pitch and yaw from the head-sensor quaternions of the aligned upper-body IMU files
(output of step 06).

Thesis step : Angle extraction (Section 3.2.3)
Dataset     : ET - Upper Body (explored, not used for the final model)
Run on      : Windows workstation at TU Wien

INPUT:
- Root folder with participant subfolders:
    D:\\upper_body_sync_2\\imu\\pXX\\leg_k_allInOne_aligned.csv

OUTPUT:
- For each input file, writes (OVERWRITE if exists):
    leg_k_allInOne_aligned_rpy.csv
  in the SAME participant folder.

What it does:
- Finds every existing leg_{2,3,4}_allInOne_aligned.csv under IN_ROOT
- Applies participant/leg rules (NO leg1 ever; special p33/p103)
- Reads file
- Extracts quaternion columns: orie_Head_q0..q3
- Assumes quaternion order is WXYZ by default:
    w=q0, x=q1, y=q2, z=q3
- Normalizes quaternions
- Converts to roll/pitch/yaw using ZYX (yaw-pitch-roll) convention
- Adds radians + degrees columns
- Writes output CSV
- Logs a run summary to: D:\\upper_body_sync_2\\imu\\_rpy_build_log.csv

Notes:
- If the quaternion order is XYZW, set QUAT_ORDER="xyzw".
- If yaw sign seems flipped, it is usually axis convention, not “wrong math”.
"""

from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
IN_ROOT = Path(r"D:\upper_body_sync_2\imu")

# Quaternion order in your CSV columns q0..q3
# - "wxyz": q0=w, q1=x, q2=y, q3=z  (DEFAULT)
# - "xyzw": q0=x, q1=y, q2=z, q3=w
QUAT_ORDER = "wxyz"

# Minimum rows to consider “valid”
MIN_ROWS = 10

# Process only these inputs
# We will search recursively under IN_ROOT/pXX/
INPUT_GLOB = "leg_*_allInOne_aligned.csv"

# Allowed legs per participant (NEVER leg1)
DEFAULT_ALLOWED_LEGS = {2, 3, 4}
ALLOWED_LEGS_OVERRIDE = {
    "p33": {2, 3},   # p33: only leg2+leg3
    "p103": {3, 4},  # p103: only leg3+leg4
}

LEG_RE = re.compile(r"^leg_(\d+)_allInOne_aligned\.csv$", re.IGNORECASE)


# =========================
# Quaternion -> RPY
# =========================
def quat_to_rpy_zyx(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert quaternion (w,x,y,z) to roll/pitch/yaw using ZYX (yaw-pitch-roll) convention.
    Returns radians arrays: roll, pitch, yaw
    """
    # roll (x-axis)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # yaw (z-axis)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return roll, pitch, yaw


def extract_wxyz(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract quaternion arrays from dataframe columns orie_Head_q0..q3 according to QUAT_ORDER.
    """
    required = ["orie_Head_q0", "orie_Head_q1", "orie_Head_q2", "orie_Head_q3"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing quaternion columns: {missing}")

    q0 = pd.to_numeric(df["orie_Head_q0"], errors="coerce").to_numpy()
    q1 = pd.to_numeric(df["orie_Head_q1"], errors="coerce").to_numpy()
    q2 = pd.to_numeric(df["orie_Head_q2"], errors="coerce").to_numpy()
    q3 = pd.to_numeric(df["orie_Head_q3"], errors="coerce").to_numpy()

    if QUAT_ORDER.lower() == "wxyz":
        w, x, y, z = q0, q1, q2, q3
    elif QUAT_ORDER.lower() == "xyzw":
        x, y, z, w = q0, q1, q2, q3
    else:
        raise ValueError("QUAT_ORDER must be 'wxyz' or 'xyzw'.")

    return w, x, y, z


def normalize_quat(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize quaternion to unit norm (helps if the data is slightly off).
    """
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n = np.where(n == 0, np.nan, n)
    return w / n, x / n, y / n, z / n


# =========================
# Input selection
# =========================
def participant_from_path(in_csv: Path) -> str:
    # expects ...\imu\pXX\file.csv
    return in_csv.parent.name


def leg_num_from_filename(name: str) -> Optional[int]:
    m = LEG_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def is_allowed(participant: str, leg_num: int) -> bool:
    if leg_num == 1:
        return False
    allowed = ALLOWED_LEGS_OVERRIDE.get(participant, DEFAULT_ALLOWED_LEGS)
    return leg_num in allowed


# =========================
# Main
# =========================
def main() -> None:
    if not IN_ROOT.exists():
        raise FileNotFoundError(f"IN_ROOT not found: {IN_ROOT}")

    # Find all aligned leg files (existing only)
    candidates = sorted(IN_ROOT.rglob(INPUT_GLOB), key=lambda p: str(p).lower())

    # Filter by filename pattern + allowed legs per participant
    inputs: List[Path] = []
    for p in candidates:
        leg_num = leg_num_from_filename(p.name)
        if leg_num is None:
            continue
        participant = participant_from_path(p)
        if is_allowed(participant, leg_num):
            inputs.append(p)

    print(f"Found candidate aligned files: {len(candidates)}")
    print(f"Will process (allowed) files:  {len(inputs)}")

    log_rows: List[Dict] = []
    ok = skip = fail = 0

    for in_csv in inputs:
        participant = participant_from_path(in_csv)
        leg_num = leg_num_from_filename(in_csv.name)

        try:
            df = pd.read_csv(in_csv, engine="python")

            if len(df) < MIN_ROWS:
                log_rows.append({
                    "status": "SKIP",
                    "reason": f"too few rows (<{MIN_ROWS})",
                    "participant": participant,
                    "leg": leg_num,
                    "in_csv": str(in_csv),
                    "out_csv": "",
                    "rows_in": len(df),
                    "rows_out": "",
                    "quat_order": QUAT_ORDER,
                })
                skip += 1
                print(f"[SKIP] {in_csv} | too few rows: {len(df)}")
                continue

            # Extract and normalize quaternions
            w, x, y, z = extract_wxyz(df)
            w, x, y, z = normalize_quat(w, x, y, z)

            # Compute angles
            roll, pitch, yaw = quat_to_rpy_zyx(w, x, y, z)

            # Add columns (radians + degrees)
            out_df = df.copy()
            out_df["roll_rad"] = roll
            out_df["pitch_rad"] = pitch
            out_df["yaw_rad"] = yaw

            deg = 180.0 / math.pi
            out_df["roll_deg"] = out_df["roll_rad"] * deg
            out_df["pitch_deg"] = out_df["pitch_rad"] * deg
            out_df["yaw_deg"] = out_df["yaw_rad"] * deg

            # Output path (overwrite if exists)
            out_csv = in_csv.with_name(in_csv.stem + "_rpy.csv")
            out_df.to_csv(out_csv, index=False)

            ok += 1
            log_rows.append({
                "status": "OK",
                "reason": "",
                "participant": participant,
                "leg": leg_num,
                "in_csv": str(in_csv),
                "out_csv": str(out_csv),
                "rows_in": len(df),
                "rows_out": len(out_df),
                "quat_order": QUAT_ORDER,
            })

            rel = in_csv.relative_to(IN_ROOT)
            print(f"[OK] {rel} -> {out_csv.name} | rows={len(df)} | quat={QUAT_ORDER}")

        except Exception as e:
            fail += 1
            log_rows.append({
                "status": "FAIL",
                "reason": repr(e),
                "participant": participant,
                "leg": leg_num,
                "in_csv": str(in_csv),
                "out_csv": "",
                "rows_in": "",
                "rows_out": "",
                "quat_order": QUAT_ORDER,
            })
            print(f"[FAIL] {in_csv}: {e}")

    # Write run log
    log_df = pd.DataFrame(log_rows)
    log_path = IN_ROOT / "_rpy_build_log.csv"
    log_df.to_csv(log_path, index=False)

    print("\n==== DONE ====")
    print(f"INPUT CANDIDATES: {len(candidates)}")
    print(f"PROCESSED       : {len(inputs)}")
    print(f"OK              : {ok}")
    print(f"SKIP            : {skip}")
    print(f"FAIL            : {fail}")
    print(f"LOG             : {log_path}")


if __name__ == "__main__":
    main()