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

    workers = min(32, cpu_count())  # άλλαξέ το αν θες
    print(f"Using workers: {workers}")

    with Pool(processes=workers) as pool:
        done = 0
        for result in pool.imap_unordered(process_one, tasks, chunksize=256):
            done += result
            if done % 10000 == 0 and done > 0:
                print(f"Processed {done} frames")