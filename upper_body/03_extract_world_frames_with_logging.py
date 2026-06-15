"""
02_extract_world_frames_raw_to_derived.py

Purpose
- Extract per-frame images from each recording’s world video (world.mp4),
  using the corresponding world_timestamps.csv file.
- Strictly enforce RAW (read-only) and DERIVED (write-only) separation.
- Ensure that no file is ever written inside the RAW dataset.

Dataset Structure Assumptions
- RAW_ROOT contains the original ET recordings:
    <RAW_ROOT>/<observer>/<recording>/
        world.mp4
        world_timestamps.csv

- DERIVED_ROOT is a completely separate directory used only for outputs:
    <DERIVED_ROOT>/frames_world/<observer>/<recording>/
        frame_000001_ts_XXXXXXXXXXXX.jpg

Core Logic
1) Recursively scan RAW_ROOT for folders containing world_timestamps.csv.
2) For each recording:
   - Read timestamp [ns] column from world_timestamps.csv.
   - Open world.mp4 using OpenCV.
   - Extract frames sequentially.
   - Pair frames with timestamps (1 frame per timestamp row, best-effort).
   - Save images into DERIVED_ROOT preserving relative folder structure.
3) Log metadata and sanity metrics into:
       <DERIVED_ROOT>/logs/frame_extraction_log.csv

Safety Guarantees
- RAW_ROOT is treated as strictly read-only.
- safe_makedirs() asserts that no output path ever resides inside RAW_ROOT.
- All outputs are redirected into DERIVED_ROOT.
- If output folder already contains frames, extraction is skipped to avoid mixing runs.

Output Structure
- Frames:
    <DERIVED_ROOT>/frames_world/<observer>/<recording>/
        frame_000001_ts_<epoch_ns>.jpg

- Log:
    <DERIVED_ROOT>/logs/frame_extraction_log.csv

Frame–Timestamp Pairing Strategy
- The script uses sequential frame reading via cv2.VideoCapture.read().
- Number of extracted frames:
      n = min(video_frame_count, number_of_timestamp_rows)
- This ensures:
      - No overflow beyond available timestamps
      - No attempt to read beyond video length

Sanity Metrics Logged
- saved_frames
- video_frame_count
- timestamp_rows
- mismatch (True if abs(video_frames - timestamp_rows) > 2)
- duration_sec (estimated via saved_frames / FPS)
- processing_time_sec

Re-run Policy
- If output folder already contains files → SKIPPED.
- To re-extract, manually delete the output folder.

Performance Notes
- Sequential frame reading is faster and safer than random seeking.
- JPEG compression quality controlled via JPG_QUALITY.
- Designed for large-scale batch extraction.

Dependencies
- Python
- numpy
- pandas
- opencv-python (cv2)

Typical Use Case
- After synchronization and timestamp sanity checks.
- Before building frame–IMU pairing datasets.
- For downstream deep learning or temporal alignment experiments.

Author Workflow Context
- Used in ET–Upper Body synchronization pipeline.
- Maintains strict separation between RAW recordings and derived artifacts.
"""


import os
import cv2
import time
import numpy as np
import pandas as pd

# ================= CONFIG =================
RAW_ROOT = r"E:\john\ET-IMU-and-upper-body\et"          # READ-ONLY
DERIVED_ROOT = r"E:\john\DERIVED_ET_IMU_upper_body"     # WRITE ONLY HERE

TIMESTAMPS_CSV_NAME = "world_timestamps.csv"
VIDEO_NAME = "world.mp4"
TIMESTAMP_COL = "timestamp [ns]"

OUT_FRAMES_SUBDIR = "frames_world"
LOG_DIR = os.path.join(DERIVED_ROOT, "logs")
LOG_CSV = os.path.join(LOG_DIR, "frame_extraction_log.csv")

IMG_EXT = "jpg"
JPG_QUALITY = 95
# ==========================================

def safe_makedirs(path: str):
    raw_abs = os.path.abspath(RAW_ROOT)
    out_abs = os.path.abspath(path)
    assert not out_abs.startswith(raw_abs), f"FATAL: attempt to write inside RAW: {path}"
    os.makedirs(path, exist_ok=True)

def make_out_dir(raw_recording_dir: str) -> str:
    rel = os.path.relpath(raw_recording_dir, RAW_ROOT)
    return os.path.join(DERIVED_ROOT, OUT_FRAMES_SUBDIR, rel)

def extract_frames(video_path, csv_path, out_dir):
    df = pd.read_csv(csv_path)
    ts = df[TIMESTAMP_COL].to_numpy(dtype=np.int64)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    n = min(video_frames if video_frames > 0 else len(ts), len(ts))
    params = [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY]

    saved = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(out_dir, f"frame_{i:06d}_ts_{ts[i]}.{IMG_EXT}")
        cv2.imwrite(out_path, frame, params)
        saved += 1

    cap.release()
    duration_sec = saved / fps if fps > 0 else np.nan
    return saved, video_frames, len(ts), duration_sec

def main():
    safe_makedirs(LOG_DIR)

    # collect all recordings first
    recordings = []
    for root, _, files in os.walk(RAW_ROOT):
        if TIMESTAMPS_CSV_NAME in files:
            recordings.append(root)

    total = len(recordings)
    print(f"Found {total} recordings")

    log_rows = []
    start_all = time.time()

    for idx, rec_dir in enumerate(recordings, start=1):
        t0 = time.time()

        observer = os.path.basename(os.path.dirname(rec_dir))
        recording = os.path.basename(rec_dir)

        csv_path = os.path.join(rec_dir, TIMESTAMPS_CSV_NAME)
        video_path = os.path.join(rec_dir, VIDEO_NAME)
        out_dir = make_out_dir(rec_dir)

        status = "OK"
        note = ""

        try:
            if not os.path.exists(video_path):
                status = "SKIPPED"
                note = "world.mp4 not found"
                saved = video_frames = ts_count = duration = np.nan
            else:
                safe_makedirs(out_dir)

                # skip if already extracted
                if os.listdir(out_dir):
                    status = "SKIPPED"
                    note = "output already exists"
                    saved = video_frames = ts_count = duration = np.nan
                else:
                    saved, video_frames, ts_count, duration = extract_frames(
                        video_path, csv_path, out_dir
                    )
        except Exception as e:
            status = "ERROR"
            note = str(e)
            saved = video_frames = ts_count = duration = np.nan

        t1 = time.time()
        elapsed = t1 - t0

        mismatch = (
            False if status != "OK"
            else abs(video_frames - ts_count) > 2
        )

        log_rows.append({
            "observer": observer,
            "recording": recording,
            "status": status,
            "saved_frames": saved,
            "video_frame_count": video_frames,
            "timestamp_rows": ts_count,
            "mismatch": mismatch,
            "duration_sec": duration,
            "processing_time_sec": elapsed,
            "csv_path": csv_path,
            "video_path": video_path,
            "out_dir": out_dir,
            "note": note
        })

        # ---------- PROGRESS ----------
        done = idx
        pct = 100.0 * done / total
        elapsed_all = time.time() - start_all
        eta = (elapsed_all / done) * (total - done)

        print(
            f"[{done}/{total} | {pct:5.1f}%] "
            f"{observer}/{recording} | {status} | "
            f"elapsed={elapsed:.1f}s | ETA={eta/60:.1f} min"
        )

    # save log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_CSV, index=False)
    print(f"\nLOG saved to: {LOG_CSV}")

if __name__ == "__main__":
    main()