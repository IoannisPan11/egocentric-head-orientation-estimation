"""
inspect_recording_windows.py

Counts the training windows of every recording and writes `recording_window_counts.csv`.
It is an inspection script; the split scripts compute their own counts.

Thesis step : Data splitting (Section 4.3)
Run on      : TU Wien dataLAB cluster (SLURM, NVIDIA A100)

What it does
- Finds the recordings with `dataset.discover_recordings`.
- For each recording, drops the rows with missing NN angles and counts the windows of
  20 past + 1 + 5 future frames with stride 8.
- Prints the recordings sorted by window count and writes the table to
  `recording_window_counts.csv` (participant, trial, recording, paired CSV, frame folder,
  number of windows).

Note
- The included `recording_window_counts.csv` was written by an earlier run of this script,
  when the paired files were still named `paired_frames.csv`.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from dataset import discover_recordings

BASE = "/share/e12507305/preprocessed/DERIVED_ET_IMU_head_mounted"

PAST_FRAMES = 20
FUTURE_FRAMES = 5
STRIDE = 8
SEQ_LEN = PAST_FRAMES + 1 + FUTURE_FRAMES


def count_windows_for_recording(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)

    required = ["frame_name", "timestamp_ns", "roll_nn", "pitch_nn", "yaw_nn"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    df = df.dropna(subset=["roll_nn", "pitch_nn", "yaw_nn"]).reset_index(drop=True)

    if len(df) < SEQ_LEN:
        return 0

    max_start = len(df) - SEQ_LEN
    n_windows = (max_start // STRIDE) + 1
    return int(n_windows)


def main():
    items = discover_recordings(BASE, frames_subdir="results/frames_512")

    rows = []
    for item in items:
        n_windows = count_windows_for_recording(item.paired_csv)
        rows.append({
            "participant": item.participant,
            "trial": item.trial,
            "recording_name": item.recording_name,
            "paired_csv": str(item.paired_csv),
            "frames_dir": str(item.frames_dir),
            "n_windows": n_windows,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("n_windows", ascending=False).reset_index(drop=True)

    print("\n=== RECORDINGS SORTED BY NUMBER OF WINDOWS ===\n")
    print(df[["participant", "trial", "recording_name", "n_windows"]].to_string(index=False))

    out_csv = Path("recording_window_counts.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()