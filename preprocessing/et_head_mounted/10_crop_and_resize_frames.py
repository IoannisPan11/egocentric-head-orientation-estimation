"""
10_crop_and_resize_frames.py

Center-crops every labeled frame to a square and resizes it to 512 x 512 pixels, the input size of
the model.

Thesis step : Crop and resize (Section 3.2.6)
Dataset     : ET - Head Mounted
Run on      : TU Wien dataLAB cluster, as a SLURM CPU job (see 10_crop_and_resize_frames.sh).
              Unlike steps 00-09, this step ran on the cluster, after the extracted frames and the
              paired_frames.csv files had been uploaded there.

Purpose
- Convert the 1088 x 1080 scene-camera frames into 512 x 512 model inputs. Only frames that have a
  label in paired_frames.csv are processed.

What it does
1) Finds every `paired_frames.csv` (output of step 09) with the layout
     <BASE>/<LS>/<trial>/paired_frames.csv
2) For each file, reads `recording_name` (<LS>/<trial>) and `frame_name`, and looks up the matching
   frames folder in `results/frames/` (folder name starting with "<LS>_<trial>_", as created by step 00).
3) Creates one task per labeled frame that exists on disk.
4) Processes the tasks in parallel with multiprocessing (up to 32 workers). For each frame:
   - skips it if the output file already exists;
   - crops the central square of side min(height, width) (1088 x 1080 -> 1080 x 1080);
   - resizes it to 512 x 512 with OpenCV INTER_AREA interpolation;
   - writes it to `results/frames_512/<frames folder>/<frame_name>`.

Inputs
- BASE = /share/e12507305/preprocessed/DERIVED_ET_IMU_head_mounted (dataLAB storage)
- <BASE>/<LS>/<trial>/paired_frames.csv
- <BASE>/results/frames/<frames folder>/frame_XXXXXX.png

Outputs
- <BASE>/results/frames_512/<frames folder>/frame_XXXXXX.png, read by the training code.
- A progress message every 10,000 processed frames.

Notes
- If several frames folders match a recording, the first match is used.
- Recordings without a matching frames folder, and paired_frames.csv files without the
  `recording_name` / `frame_name` columns, are skipped without a message.
"""
from pathlib import Path
import pandas as pd
import cv2
from multiprocessing import Pool, cpu_count

BASE = Path("/share/e12507305/preprocessed/DERIVED_ET_IMU_head_mounted")
FRAMES_ROOT = BASE / "results" / "frames"
OUT_ROOT = BASE / "results" / "frames_512"

def build_tasks():
    tasks = []
    csv_files = list(BASE.glob("*/*/paired_frames.csv"))

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        if "recording_name" not in df.columns or "frame_name" not in df.columns:
            continue

        recording_name = str(df["recording_name"].iloc[0])
        participant, trial = recording_name.split("/")

        candidates = list(FRAMES_ROOT.glob(f"{participant}_{trial}_*"))
        if not candidates:
            continue

        frame_dir = candidates[0]
        out_dir = OUT_ROOT / frame_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for frame_name in df["frame_name"].dropna().astype(str):
            src = frame_dir / frame_name
            dst = out_dir / frame_name
            if src.exists():
                tasks.append((src, dst))

    return tasks

def process_one(task):
    src, dst = task

    if dst.exists():
        return 0

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return 0

    h, w = img.shape[:2]
    side = min(h, w)

    y0 = (h - side) // 2
    x0 = (w - side) // 2

    cropped = img[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_AREA)

    ok = cv2.imwrite(str(dst), resized)
    return 1 if ok else 0

if __name__ == "__main__":
    tasks = build_tasks()
    print(f"Total labeled frames to process: {len(tasks)}")

    workers = min(32, cpu_count())  # adjust if needed
    print(f"Using workers: {workers}")

    with Pool(processes=workers) as pool:
        done = 0
        for result in pool.imap_unordered(process_one, tasks, chunksize=256):
            done += result
            if done % 10000 == 0 and done > 0:
                print(f"Processed {done} frames")