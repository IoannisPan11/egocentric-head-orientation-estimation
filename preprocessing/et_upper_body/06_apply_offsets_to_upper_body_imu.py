"""
06_apply_offsets_to_upper_body_imu.py

Applies the synchronization offsets of step 05 to the Xsens Awinda upper-body IMU files and adds a
timestamp on the eye-tracker timeline (`upper_world_epoch_ns`). Writes filtered copies; the
original files are not modified.

Thesis step : Time synchronization (Section 3.2.4)
Dataset     : ET - Upper Body (explored, not used for the final model)
Run on      : Windows workstation at TU Wien

Purpose
- Read offsets from: D:\\upper_body_sync_2\\offsets_master_upper_DONE.csv
- For each participant pXX:
    - pick the FIRST valid offset_ns (file order)
    - apply it to every *allowed* leg_k_allInOne.csv under:
        E:\\john\\ET-IMU-and-upper-body\\imu\\pXX\\leg_k_allInOne.csv
    - write a DUPLICATE filtered CSV under:
        D:\\upper_body_sync_2\\imu\\pXX\\leg_k_allInOne_aligned.csv
      containing ONLY the whitelist columns + new:
        upper_world_epoch_ns = ms*1e6 + offset_ns

Leg selection
- NEVER process leg_1 for ANY participant.
- Default allowed legs: {2,3,4}.
- Special cases:
    - p33  -> only {2,3}
    - p103 -> only {3,4}

Notes
- Skips participants with no valid offset_ns.
- Skips legs where input allInOne CSV does not exist.
- Never modifies originals in E:\\...
- Writes log CSV: D:\\upper_body_sync_2\\imu\\_apply_offset_log.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
ALLINONE_ROOT = Path(r"E:\john\ET-IMU-and-upper-body\imu")  # READ
OFFSETS_CSV = Path(r"D:\upper_body_sync_2\offsets_master_upper_DONE.csv")

OUT_ROOT = Path(r"D:\upper_body_sync_2\imu")               # WRITE (duplicates)
LOG_CSV = OUT_ROOT / "_apply_offset_log.csv"

NEW_COL = "upper_world_epoch_ns"
SKIP_IF_EXISTS = True

# candidates for ms column in allInOne
MS_COL_CANDIDATES = ["ms", "timestamp_ms", "timestamp [ms]", "timestamp", "time_ms", "ts_ms"]

# ONLY keep these columns (if present) + NEW_COL will be appended
KEEP_COLS = [
    "leg",
    "fn",
    "frame_idx",
    "ms",
    "jC1Head_x", "jC1Head_y", "jC1Head_z",
    "orie_Head_q0", "orie_Head_q1", "orie_Head_q2", "orie_Head_q3",
    "velo_Head_x", "velo_Head_y", "velo_Head_z",
    "acce_Head_x", "acce_Head_y", "acce_Head_z",
    "angV_Head_x", "angV_Head_y", "angV_Head_z",
    "angA_Head_x", "angA_Head_y", "angA_Head_z",
]

# Allowed legs per participant (NEVER leg1)
DEFAULT_ALLOWED_LEGS = {2, 3, 4}
ALLOWED_LEGS_OVERRIDE = {
    "p33": {2, 3},   # p33: only leg2+leg3
    "p103": {3, 4},  # p103: only leg3+leg4
}

# =========================
# Helpers
# =========================
def safe_numeric_int(v) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        s = str(v).strip()
        if s == "" or s.lower() in ("nan", "none", "<null>"):
            return None
        return int(float(s))
    except Exception:
        return None


def get_participant_from_recording_id(recording_id: str) -> Optional[str]:
    rid = (recording_id or "").strip()
    m = re.match(r"^(p\d+)_", rid)
    return m.group(1) if m else None


def build_first_valid_offset_map(offsets_df: pd.DataFrame) -> Dict[str, int]:
    """
    Return map: participant -> offset_ns
    FIRST valid offset_ns per participant wins (file order).
    """
    required = {"recording_id", "offset_ns"}
    missing = [c for c in required if c not in offsets_df.columns]
    if missing:
        raise ValueError(f"OFFSETS_CSV missing columns {missing}. Found={list(offsets_df.columns)}")

    out: Dict[str, int] = {}
    for _, row in offsets_df.iterrows():
        participant = get_participant_from_recording_id(str(row.get("recording_id", "")))
        if not participant:
            continue

        off = safe_numeric_int(row.get("offset_ns", None))
        if off is None:
            continue

        if participant not in out:
            out[participant] = off

    return out


def pick_ms_column(df: pd.DataFrame) -> str:
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in MS_COL_CANDIDATES:
        k = cand.strip().lower()
        if k in cols:
            return cols[k]
    raise ValueError(f"Cannot find ms column. Candidates={MS_COL_CANDIDATES} Found={list(df.columns)}")


def find_leg_allinone_files(pdir: Path, participant: str) -> List[Tuple[str, Path]]:
    """
    Returns list of (leg_k, csv_path) e.g. ("leg_2", "...\\leg_2_allInOne.csv")
    Filtered by allowed legs per participant.
    Never returns leg1.
    """
    allowed = ALLOWED_LEGS_OVERRIDE.get(participant, DEFAULT_ALLOWED_LEGS)

    out: List[Tuple[str, Path]] = []
    for p in sorted(pdir.glob("leg_*_allInOne.csv")):
        m = re.match(r"^(leg_(\d+))_allInOne\.csv$", p.name)
        if not m:
            continue

        leg_str = m.group(1)      # "leg_2"
        leg_num = int(m.group(2)) # 2

        # Hard rule: never leg1
        if leg_num == 1:
            continue

        # Participant-specific rule
        if leg_num not in allowed:
            continue

        out.append((leg_str, p))

    return out


def subset_keep_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Keep only KEEP_COLS that exist in df, preserving order, plus NEW_COL at the end.
    Returns (filtered_df, kept_cols_list).
    """
    existing = set(df.columns)
    kept = [c for c in KEEP_COLS if c in existing]

    # Ensure essentials if present
    for essential in ("ms", "frame_idx"):
        if essential not in kept and essential in existing:
            kept.append(essential)

    return df[kept].copy(), kept


# =========================
# Main
# =========================
def main() -> None:
    if not ALLINONE_ROOT.exists():
        raise FileNotFoundError(f"ALLINONE_ROOT not found: {ALLINONE_ROOT}")
    if not OFFSETS_CSV.exists():
        raise FileNotFoundError(f"OFFSETS_CSV not found: {OFFSETS_CSV}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    offsets_df = pd.read_csv(OFFSETS_CSV)
    offset_map = build_first_valid_offset_map(offsets_df)

    print(f"Offsets rows: total={len(offsets_df)} | participants_with_valid_offset={len(offset_map)}")

    participant_dirs = sorted(
        [p for p in ALLINONE_ROOT.iterdir() if p.is_dir() and p.name.lower().startswith("p")],
        key=lambda p: p.name.lower(),
    )
    print(f"Participants under allInOne root: {len(participant_dirs)}")

    log_rows = []
    ok = skip = fail = 0

    for pdir in participant_dirs:
        participant = pdir.name

        if participant not in offset_map:
            log_rows.append({
                "participant": participant,
                "leg": "",
                "status": "SKIP",
                "reason": "no valid offset_ns for participant",
                "in_csv": "",
                "out_csv": "",
                "offset_ns": "",
                "rows": "",
            })
            skip += 1
            continue

        offset_ns = offset_map[participant]

        leg_files = find_leg_allinone_files(pdir, participant)
        if not leg_files:
            allowed = ALLOWED_LEGS_OVERRIDE.get(participant, DEFAULT_ALLOWED_LEGS)
            log_rows.append({
                "participant": participant,
                "leg": "",
                "status": "SKIP",
                "reason": f"no allowed leg_*_allInOne.csv found (allowed_legs={sorted(allowed)})",
                "in_csv": "",
                "out_csv": "",
                "offset_ns": offset_ns,
                "rows": "",
            })
            skip += 1
            continue

        for leg, in_csv in leg_files:
            # Extra safety guard (never hurt):
            leg_num = int(leg.split("_")[1])
            allowed = ALLOWED_LEGS_OVERRIDE.get(participant, DEFAULT_ALLOWED_LEGS)
            if leg_num == 1 or leg_num not in allowed:
                log_rows.append({
                    "participant": participant,
                    "leg": leg,
                    "status": "SKIP",
                    "reason": f"leg not allowed by rules (allowed_legs={sorted(allowed)})",
                    "in_csv": str(in_csv),
                    "out_csv": "",
                    "offset_ns": offset_ns,
                    "rows": "",
                })
                skip += 1
                print(f"[SKIP] {participant} {leg}: not allowed")
                continue

            out_dir = OUT_ROOT / participant
            out_dir.mkdir(parents=True, exist_ok=True)

            out_csv = out_dir / f"{leg}_allInOne_aligned.csv"

            if SKIP_IF_EXISTS and out_csv.exists():
                log_rows.append({
                    "participant": participant,
                    "leg": leg,
                    "status": "SKIP",
                    "reason": "output already exists",
                    "in_csv": str(in_csv),
                    "out_csv": str(out_csv),
                    "offset_ns": offset_ns,
                    "rows": "",
                })
                skip += 1
                print(f"[SKIP] {participant} {leg}: output exists")
                continue

            try:
                df = pd.read_csv(in_csv, engine="python")

                # Find ms column (robust) but output column should be "ms"
                ms_col = pick_ms_column(df)
                ms = pd.to_numeric(df[ms_col], errors="coerce")
                if ms.isna().all():
                    raise ValueError(f"All values in ms column are NaN after numeric conversion. ms_col='{ms_col}'")

                # Work on filtered subset
                out_df, _kept_cols = subset_keep_cols(df)

                # Ensure output has "ms"
                if "ms" not in out_df.columns:
                    out_df["ms"] = df[ms_col]

                # Compute NEW_COL using numeric ms
                out_df[NEW_COL] = (ms * 1_000_000.0 + float(offset_ns)).round().astype("Int64")

                # Final column order: whitelist (that exists) + NEW_COL at end
                final_cols = [c for c in KEEP_COLS if c in out_df.columns]

                # ensure essentials if present
                for essential in ("frame_idx", "ms"):
                    if essential in out_df.columns and essential not in final_cols:
                        final_cols.append(essential)

                final_cols.append(NEW_COL)
                out_df = out_df[final_cols]

                out_df.to_csv(out_csv, index=False)

                log_rows.append({
                    "participant": participant,
                    "leg": leg,
                    "status": "OK",
                    "reason": "",
                    "in_csv": str(in_csv),
                    "out_csv": str(out_csv),
                    "offset_ns": offset_ns,
                    "rows": len(out_df),
                    "ms_col_used": ms_col,
                    "kept_cols": ",".join(final_cols),
                })
                ok += 1
                print(f"[OK] {participant} {leg} -> {out_csv.name} | rows={len(out_df)} | offset_ns={offset_ns}")

            except Exception as e:
                log_rows.append({
                    "participant": participant,
                    "leg": leg,
                    "status": "FAIL",
                    "reason": repr(e),
                    "in_csv": str(in_csv),
                    "out_csv": str(out_csv),
                    "offset_ns": offset_ns,
                    "rows": "",
                })
                fail += 1
                print(f"[FAIL] {participant} {leg}: {e}")

    pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False)

    print("\n==== DONE ====")
    print(f"OK   : {ok}")
    print(f"SKIP : {skip}")
    print(f"FAIL : {fail}")
    print(f"LOG  : {LOG_CSV}")
    print(f"OUT  : {OUT_ROOT}")


if __name__ == "__main__":
    main()