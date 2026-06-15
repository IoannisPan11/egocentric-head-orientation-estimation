r"""
01_extract_world_frames_upper_body.py

Purpose
- Extract per-frame images from the Neon "world" video for every recording under ROOT,
  using the timestamps exported by `pl-rec-export` in: <recording_root>\export\world.csv.

Why this version (fixes your exact errors)
- Your `pl-rec-export` world.csv is the NEW format with a real header row and 4 columns:
    file, timestamp [ns], recording offset [secs], video offset [secs]
  (NOT the older "timestamps [seconds],pts" format).
- This script reads that NEW format correctly and *also* supports the older 2-column format as fallback.
- It fixes the “Cannot open video ... .time” bug by **never treating *.time as a video**.
  If the CSV 'file' points to a .time, we auto-map it to the matching .mp4 (same basename).

Expected layout
- pl-rec-export output:
    <recording_root>\export\world.csv
- The video typically lives in <recording_root>\world.mp4 or <recording_root>\Neon Scene Camera v1 ps1.mp4
  BUT we primarily use the full path stored in world.csv column "file" (if it points to an .mp4).

Outputs
- Frames:
    <recording_root>\frames_world\frame_000001_t_1.234567s_ts_1722....jpg
- Log:
    <ROOT>\_logs\frame_extraction_log.csv

Safety / Re-run
- Does not modify raw data.
- Skips a recording if frames_world already contains files (prevents mixing outputs).
  (You can override with FORCE_REEXTRACT = True.)

Dependencies
- Python packages: opencv-python (cv2), numpy, pandas
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# ================= CONFIG =================
ROOT = r"D:\upper_body_sync_2"

EXPORT_DIR_NAME = "export"
WORLD_CSV_NAME = "world.csv"

# Fallback video names if the CSV "file" column is missing/unusable
VIDEO_NAME_PRIMARY = "Neon Scene Camera v1 ps1.mp4"
VIDEO_NAME_FALLBACK = "world.mp4"

OUT_FRAMES_SUBDIR = "frames_world"
LOG_DIRNAME = "_logs"
LOG_CSV_NAME = "frame_extraction_log.csv"

IMG_EXT = "jpg"
JPG_QUALITY = 95

# Re-run behavior
FORCE_REEXTRACT = False  # True => delete existing frames and re-extract cleanly

# Debug
DEBUG_MAX_RECORDINGS = None  # e.g. 1
DEBUG_MAX_FRAMES = None  # e.g. 50
# ==========================================


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_nonempty_files(dir_path: str) -> bool:
    if not os.path.isdir(dir_path):
        return False
    try:
        return any(os.path.isfile(os.path.join(dir_path, f)) for f in os.listdir(dir_path))
    except Exception:
        return False


def find_world_video_fallback(recording_root: str) -> Optional[str]:
    p1 = os.path.join(recording_root, VIDEO_NAME_PRIMARY)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(recording_root, VIDEO_NAME_FALLBACK)
    if os.path.exists(p2):
        return p2
    return None


def normalize_video_path_from_csv(p: str) -> Optional[str]:
    """
    CSV column 'file' should ideally point to an .mp4.
    If it points to a .time, map it to .mp4 with the same basename.
    Also return None if empty/NaN-like.
    """
    if p is None:
        return None
    p = str(p).strip()
    if not p or p.lower() == "nan":
        return None

    lower = p.lower()
    if lower.endswith(".mp4"):
        return p
    if lower.endswith(".time"):
        # Convert "...something.time" -> "...something.mp4"
        return p[:-5] + ".mp4"

    # If it is some other extension, we do NOT treat it as a video path.
    return None


def read_world_csv_any_format(world_csv_path: str) -> Tuple[Optional[str], np.ndarray, np.ndarray]:
    """
    Returns:
      video_path_from_csv (str|None),
      t_sec (float64) -> video timeline seconds (prefer "video offset [secs]" if available),
      ts_ns (int64)   -> epoch timestamps per frame (ns) if available, else synthesized.

    Supported formats:
    1) NEW format (your case) with header columns:
       file, timestamp [ns], recording offset [secs], video offset [secs]
    2) OLD format:
       (optional commented header starting with '#')
       seconds, pts
    """

    # ---------- Try NEW format (header) ----------
    try:
        df = pd.read_csv(world_csv_path)  # header=0 by default

        # Normalize column names for robust matching
        norm = {c: c.strip().lower() for c in df.columns}
        col_by_norm = {v: k for k, v in norm.items()}

        has_new = (
            "file" in col_by_norm
            and "timestamp [ns]" in col_by_norm
            and "video offset [secs]" in col_by_norm
        )

        if has_new:
            file_col = col_by_norm["file"]
            ts_col = col_by_norm["timestamp [ns]"]
            vsec_col = col_by_norm["video offset [secs]"]

            ts_ns = pd.to_numeric(df[ts_col], errors="coerce")
            t_sec = pd.to_numeric(df[vsec_col], errors="coerce")

            valid = ts_ns.notna() & t_sec.notna()
            ts_ns_arr = ts_ns[valid].to_numpy(dtype=np.int64)
            t_sec_arr = t_sec[valid].to_numpy(dtype=np.float64)

            # Choose a usable .mp4 from the file column
            video_path = None
            if file_col in df.columns:
                for raw in df[file_col].tolist():
                    cand = normalize_video_path_from_csv(raw)
                    if cand and os.path.exists(cand):
                        video_path = cand
                        break

            if ts_ns_arr.size == 0 or t_sec_arr.size == 0:
                raise RuntimeError("No valid numeric rows in new-format world.csv")

            return video_path, t_sec_arr, ts_ns_arr

    except Exception:
        # Fall through to old-format parsing
        pass

    # ---------- OLD format (2 columns, maybe commented header) ----------
    df = pd.read_csv(world_csv_path, comment="#", header=None)
    if df.empty:
        raise RuntimeError(f"Empty world.csv: {world_csv_path}")

    # If the first row is actually a header-like row (strings), drop it
    # (This handles the case where someone read header rows as data)
    if df.shape[0] > 0:
        first = df.iloc[0].astype(str).str.lower().tolist()
        if any("timestamp" in x or "recording offset" in x or "video offset" in x for x in first):
            df = df.iloc[1:].reset_index(drop=True)

    if df.shape[1] < 1:
        raise RuntimeError(f"world.csv has no columns: {world_csv_path}")

    # Take the first column as seconds
    sec = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    valid = sec.notna()
    sec = sec[valid].to_numpy(dtype=np.float64)

    if sec.size == 0:
        raise RuntimeError(f"No valid numeric seconds rows in old-format world.csv: {world_csv_path}")

    # Synthesize ns relative to first sample (old format has no epoch ns)
    ts_ns = ((sec - sec[0]) * 1e9).round().astype(np.int64)

    return None, sec, ts_ns


def extract_frames(video_path: str, world_csv_path: str, out_dir: str) -> Tuple[int, int, int, float, str]:
    """
    Extract sequential frames and name them with:
      frame_%06d_t_%0.6fs_ts_%dns.jpg

    Notes:
    - We do a sequential read (cap.read()) which is reliable and faster than random seeking.
    - Pairing rule: best-effort 1 frame per timestamps row (use min of video frame count and csv rows).
    """
    video_from_csv, t_sec, ts_ns = read_world_csv_any_format(world_csv_path)

    # Prefer the CSV-embedded video path if it is a real .mp4
    if video_from_csv and os.path.exists(video_from_csv):
        video_path = video_from_csv

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Probe read to fail early on codec issues
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(
            "OpenCV opened the video but cannot read the first frame (codec/decoding issue). "
            f"Video: {video_path}"
        )

    # rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Pairing: 1 frame per timestamp row (best-effort)
    n = min(video_frames if video_frames > 0 else len(ts_ns), len(ts_ns))
    if DEBUG_MAX_FRAMES is not None:
        n = min(n, int(DEBUG_MAX_FRAMES))

    params = [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY]

    saved = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break

        out_path = os.path.join(
            out_dir,
            f"frame_{i:06d}_t_{t_sec[i]:.6f}s_ts_{ts_ns[i]}.{IMG_EXT}",
        )
        cv2.imwrite(out_path, frame, params)
        saved += 1

    cap.release()
    duration_sec = saved / fps if fps and fps > 0 else float("nan")
    return saved, video_frames, len(ts_ns), duration_sec, video_path


def main() -> None:
    log_dir = os.path.join(ROOT, LOG_DIRNAME)
    safe_makedirs(log_dir)
    log_csv = os.path.join(log_dir, LOG_CSV_NAME)

    # Find all export folders that contain world.csv
    export_dirs = []
    for r, _, files in os.walk(ROOT):
        if WORLD_CSV_NAME in files and os.path.basename(r).lower() == EXPORT_DIR_NAME:
            export_dirs.append(r)

    export_dirs.sort()

    if DEBUG_MAX_RECORDINGS is not None:
        export_dirs = export_dirs[: int(DEBUG_MAX_RECORDINGS)]

    total = len(export_dirs)
    print(f"Found {total} recordings under: {ROOT}", flush=True)

    log_rows = []
    start_all = time.time()

    for idx, export_dir in enumerate(export_dirs, start=1):
        t0 = time.time()
        rel = os.path.relpath(export_dir, ROOT)
        print(f"\n--- [{idx}/{total}] START {rel}", flush=True)

        recording_root = os.path.dirname(export_dir)
        world_csv_path = os.path.join(export_dir, WORLD_CSV_NAME)
        out_dir = os.path.join(recording_root, OUT_FRAMES_SUBDIR)

        status = "OK"
        note = ""

        saved = video_frames = ts_count = float("nan")
        duration = float("nan")
        used_video = ""

        try:
            # Decide the video path:
            # 1) Try usable .mp4 from CSV 'file'
            video_from_csv, _, _ = read_world_csv_any_format(world_csv_path)
            if video_from_csv and os.path.exists(video_from_csv):
                video_path = video_from_csv
            else:
                # 2) Fallback to expected names in the recording root
                video_path = find_world_video_fallback(recording_root)

            if not video_path:
                status = "SKIPPED"
                note = (
                    "world video not found: "
                    "CSV 'file' path missing/unusable (or not .mp4), "
                    f"and no '{VIDEO_NAME_PRIMARY}' or '{VIDEO_NAME_FALLBACK}' in recording root."
                )
            else:
                safe_makedirs(out_dir)

                # Re-run policy
                if list_nonempty_files(out_dir):
                    if FORCE_REEXTRACT:
                        # Clean and re-extract to avoid mixing partial outputs
                        for f in os.listdir(out_dir):
                            try:
                                os.remove(os.path.join(out_dir, f))
                            except Exception:
                                pass
                    else:
                        status = "SKIPPED"
                        note = "output already exists"
                if status == "OK":
                    saved, video_frames, ts_count, duration, used_video = extract_frames(
                        video_path, world_csv_path, out_dir
                    )

        except Exception as e:
            status = "ERROR"
            note = str(e)

        elapsed = time.time() - t0

        mismatch = False
        if status == "OK":
            try:
                mismatch = abs(int(video_frames) - int(ts_count)) > 2
            except Exception:
                mismatch = False

        log_rows.append(
            {
                "rel_export_dir": rel,
                "recording_root": recording_root,
                "status": status,
                "saved_frames": saved,
                "video_frame_count": video_frames,
                "timestamp_rows": ts_count,
                "mismatch": mismatch,
                "duration_sec": duration,
                "processing_time_sec": elapsed,
                "world_csv_path": world_csv_path,
                "video_path_used": used_video if used_video else "",
                "out_dir": out_dir,
                "note": note,
            }
        )

        done = idx
        pct = 100.0 * done / total if total else 100.0
        elapsed_all = time.time() - start_all
        eta = (elapsed_all / done) * (total - done) if done > 0 else 0.0

        print(
            f"[{done}/{total} | {pct:5.1f}%] "
            f"{rel} | {status} | "
            f"elapsed={elapsed:.1f}s | ETA={eta/60:.1f} min | "
            f"note={note}",
            flush=True,
        )

    pd.DataFrame(log_rows).to_csv(log_csv, index=False)
    print(f"\nLOG saved to: {log_csv}", flush=True)


if __name__ == "__main__":
    main()