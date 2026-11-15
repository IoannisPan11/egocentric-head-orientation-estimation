"""
report_first_timestamps.py

Diagnostic: collects the first timestamp of the frames, of the eye-tracker IMU and of the upper-body
IMU for every participant and session, to compare their start times. Not part of the pipeline.

Dataset     : ET - Upper Body (explored, not used for the final model)
Run on      : Windows workstation at TU Wien

What it does
1) Lists the participants (folders named pXX) in FRAMES_WORLD_ROOT and their session folders,
   sorted by name.
2) Maps the sessions in order to the upper-body legs 2, 3, 4 (SESSION_TO_UPPER_LEG_K) and numbers
   them as legs 1, 2, 3 in the output (LEG_START = 1).
3) For each session, reads:
   - the timestamp of the first extracted frame (from the frame file name);
   - the first timestamp of the best-matching eye-tracker IMU CSV (epoch ns);
   - the first timestamp of the raw upper-body IMU file leg_<k>_allInOne.csv (epoch ms -> ns).
   Column names are detected from lists of candidates.
4) Writes one row per session to OUT_CSV (final_first_timestamps.csv) with the three timestamps,
   the files used, and a status text that records which values were found.

Notes
- The upper-body timestamps are read from the raw files, before the offsets of step 06 are applied.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict


# ============================================================
# FINAL CONFIG (EDIT ONLY THESE PATHS IF NEEDED)
# ============================================================

FRAMES_WORLD_ROOT = Path(r"D:\john.panagiotopoulos\DERIVED_ET_IMU_upper_body\frames_world")

# ET IMU epoch-ns CSVs root
# Expected (typical): ET_IMU_ROOT\pXX\<session_folder>\...*.csv
ET_IMU_ROOT = Path(r"E:\john\ET-IMU-and-upper-body\et")

# Upper IMU root (RAW or your produced folder) BUT:
# In YOUR case: UPPER_IMU_ROOT\pXX\leg_2_allInOne.csv (no session folders)
UPPER_IMU_ROOT = Path(r"E:\john\ET-IMU-and-upper-body\imu")

OUT_CSV = Path(r"D:\john.panagiotopoulos\DERIVED_ET_IMU_upper_body\final_first_timestamps.csv")

# Sessions (sorted) map to upper legs:
# session 1 -> leg_2, session 2 -> leg_3, session 3 -> leg_4
SESSION_TO_UPPER_LEG_K = [2, 3, 4]

# Output leg numbering you want in CSV: 1,2,3
LEG_START = 1

FRAME_EXTS = (".jpg", ".jpeg", ".png")


# ============================================================
# Column candidates
# ============================================================

# ET IMU timestamp candidates (epoch ns)
ET_TS_NS_COL_CANDIDATES = [
    "ts_ns_best",
    "timestamp [ns]",
    "timestamp_ns",
    "ts_ns",
    "timestamp",
]

# Upper IMU time column candidates (RAW files are epoch ms)
UPPER_MS_COL_CANDIDATES = [
    "ms",
    "timestamp [ms]",
    "timestamp_ms",
    "time_ms",
    "ts_ms",
    "timestamp",
]


# ============================================================
# Helpers
# ============================================================

def parse_participant_from_path(p: Path) -> Optional[str]:
    m = re.search(r"(p\d+)", str(p))
    return m.group(1) if m else None


def sniff_dialect(path: Path) -> csv.Dialect:
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
    except csv.Error:
        class _D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return _D()


def parse_number(s: str) -> float:
    """
    Handles decimal comma:
      "33724332,949951" -> 33724332.949951
    """
    s = (s or "").strip()
    if s == "":
        raise ValueError("empty_number")

    if "," in s and "." not in s:
        s = s.replace(",", ".")
    return float(s)


def list_sessions_for_participant_frames(p: str) -> List[Path]:
    base = FRAMES_WORLD_ROOT / p
    if not base.exists():
        return []
    sessions = [d for d in base.iterdir() if d.is_dir()]
    sessions.sort(key=lambda x: x.name)
    return sessions


def find_first_frame_timestamp_ns(session_dir: Path) -> Tuple[Optional[int], Optional[Path], str]:
    """
    Finds the first image file and extracts a timestamp-like integer from filename.
    Returns (ts_ns, frame_path, note).
    """
    imgs = [p for p in session_dir.iterdir() if p.is_file() and p.suffix.lower() in FRAME_EXTS]
    if not imgs:
        imgs = [p for p in session_dir.rglob("*") if p.is_file() and p.suffix.lower() in FRAME_EXTS]

    if not imgs:
        return None, None, "no_frames_found"

    imgs.sort(key=lambda x: x.name)
    first_img = imgs[0]

    nums = re.findall(r"(\d{13,19})", first_img.stem)
    if not nums:
        return None, first_img, "no_timestamp_in_frame_name"

    tok = nums[0]
    ts = int(tok)

    # if ms (13 digits), convert to ns
    if len(tok) == 13:
        ts *= 1_000_000
        return ts, first_img, "frame_ts_was_ms_converted_to_ns"

    return ts, first_img, "ok"


def find_col_idx(header: List[str], candidates: List[str]) -> int:
    hl = [h.strip().lower() for h in header]
    for c in candidates:
        c0 = c.strip().lower()
        if c0 in hl:
            return hl.index(c0)
    raise ValueError(f"Could not find timestamp column. Header={header}")


def read_first_timestamp_ns_from_et_csv(csv_path: Path) -> int:
    """
    Reads first non-empty timestamp from ET csv (epoch ns).
    """
    dialect = sniff_dialect(csv_path)
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, dialect)
        header = next(reader, None)
        if header is None:
            raise ValueError("empty_csv_header")

        idx = find_col_idx(header, ET_TS_NS_COL_CANDIDATES)

        for row in reader:
            if not row:
                continue
            if idx >= len(row):
                continue
            raw = str(row[idx]).strip()
            if raw == "":
                continue
            v = int(float(raw))
            return v

    raise ValueError("no_data_rows_with_timestamp")


def read_first_timestamp_ns_from_upper_csv(csv_path: Path) -> int:
    """
    Reads first non-empty timestamp from Upper allInOne (epoch ms) and converts to ns.
    """
    dialect = sniff_dialect(csv_path)
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, dialect)
        header = next(reader, None)
        if header is None:
            raise ValueError("empty_csv_header")

        idx = find_col_idx(header, UPPER_MS_COL_CANDIDATES)

        for row in reader:
            if not row:
                continue
            if idx >= len(row):
                continue
            raw = str(row[idx]).strip()
            if raw == "":
                continue

            ms = parse_number(raw)          # supports comma decimals
            ns = int(ms * 1_000_000.0)      # ms -> ns
            return ns

    raise ValueError("no_data_rows_with_timestamp")


def find_best_et_csv(participant: str, session_name: str) -> Tuple[Optional[Path], str]:
    """
    Find an ET csv under ET_IMU_ROOT\pXX\session_name\...
    Picks first csv whose header contains an ET timestamp candidate.
    """
    base = ET_IMU_ROOT / participant / session_name
    if not base.exists():
        return None, "et_session_folder_missing"

    csvs = sorted(base.rglob("*.csv"))
    if not csvs:
        return None, "no_et_csvs_found"

    for c in csvs:
        try:
            dialect = sniff_dialect(c)
            with open(c, "r", encoding="utf-8", errors="replace", newline="") as f:
                r = csv.reader(f, dialect)
                header = next(r, None)
                if not header:
                    continue
                hl = [h.strip().lower() for h in header]
                if any(cc.lower() in hl for cc in ET_TS_NS_COL_CANDIDATES):
                    return c, "ok"
        except Exception:
            continue

    return csvs[0], "fallback_first_csv"


def upper_allinone_path(participant: str, leg_k: int) -> Path:
    return UPPER_IMU_ROOT / participant / f"leg_{leg_k}_allInOne.csv"


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not FRAMES_WORLD_ROOT.exists():
        raise FileNotFoundError(f"FRAMES_WORLD_ROOT not found: {FRAMES_WORLD_ROOT}")

    participants = sorted([
        d.name for d in FRAMES_WORLD_ROOT.iterdir()
        if d.is_dir() and re.fullmatch(r"p\d+", d.name)
    ])

    if not participants:
        raise ValueError(f"No participants found under {FRAMES_WORLD_ROOT}")

    rows_out: List[Dict[str, str]] = []

    total_sessions = 0
    for p in participants:
        total_sessions += len(list_sessions_for_participant_frames(p))

    print(f"Found {len(participants)} participants, {total_sessions} sessions total.\n")

    sess_counter = 0

    for p in participants:
        sessions = list_sessions_for_participant_frames(p)
        if not sessions:
            continue

        n_map = min(len(sessions), len(SESSION_TO_UPPER_LEG_K))

        for i in range(n_map):
            session_dir = sessions[i]
            session_name = session_dir.name
            leg = LEG_START + i
            leg_k = SESSION_TO_UPPER_LEG_K[i]

            sess_counter += 1
            print(f"[{sess_counter}/{total_sessions}] {p} | leg={leg} | session={session_name} | upper_leg_k={leg_k}",
                  flush=True)

            status_parts = []

            # ---- Frame ----
            frame_ts_ns, frame_path, frame_note = find_first_frame_timestamp_ns(session_dir)
            status_parts.append(f"frame:{frame_note}")

            # ---- ET ----
            et_ts_ns = None
            et_csv, et_note = find_best_et_csv(p, session_name)
            status_parts.append(f"et_csv:{et_note}")

            if et_csv is not None:
                try:
                    et_ts_ns = read_first_timestamp_ns_from_et_csv(et_csv)
                    status_parts.append("et_ts:ok")
                except Exception as e:
                    status_parts.append(f"et_ts:ERR({e})")
            else:
                status_parts.append("et_ts:missing_csv")

            # ---- Upper ----
            upper_ts_ns = None
            upper_csv = upper_allinone_path(p, leg_k)

            if not upper_csv.exists():
                status_parts.append("upper_csv:missing")
            else:
                status_parts.append("upper_csv:ok")
                try:
                    upper_ts_ns = read_first_timestamp_ns_from_upper_csv(upper_csv)
                    status_parts.append("upper_ts:ok")
                except Exception as e:
                    status_parts.append(f"upper_ts:ERR({e})")

            status = " | ".join(status_parts)

            rows_out.append({
                "participant": p,
                "leg": str(leg),
                "session": session_name,
                "upper_leg_k": str(leg_k),
                "frame_ts_ns": "" if frame_ts_ns is None else str(frame_ts_ns),
                "et_imu_ts_ns": "" if et_ts_ns is None else str(et_ts_ns),
                "upper_imu_ts_ns": "" if upper_ts_ns is None else str(upper_ts_ns),
                "frame_file": "" if frame_path is None else str(frame_path),
                "et_csv": "" if et_csv is None else str(et_csv),
                "upper_csv": str(upper_csv) if upper_csv.exists() else "",
                "status": status,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "participant", "leg", "session", "upper_leg_k",
        "frame_ts_ns", "et_imu_ts_ns", "upper_imu_ts_ns",
        "frame_file", "et_csv", "upper_csv", "status"
    ]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print("\nDONE.")
    print(f"Output CSV: {OUT_CSV}")
    print(f"Rows: {len(rows_out)}")


if __name__ == "__main__":
    main()