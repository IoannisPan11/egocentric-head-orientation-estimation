"""
09b_pair_frames_with_imus_recording_032.py

Variant of step 09 for a single recording (032_LS168/trial_1_om) whose folder layout differs from the
other recordings. This recording is part of the final dataset.

Thesis step : Frame-label pairs (Section 3.2.5)
Dataset     : ET - Head Mounted
Run on      : Windows workstation at TU Wien

Purpose
- Produce `paired_frames.csv` for recording 032_LS168/trial_1_om with the same pairing logic and the
  same output columns as `09_pair_frames_with_imus.py`. The recording contains more than one
  eye-tracker session, and its frames folder is not found automatically, so both are given
  explicitly on the command line.

Differences from step 09
- Processes only the recording given by `--rec_key` (default: 032_LS168/trial_1_om).
- The frames folder is passed directly with `--frames_dir`. `--frames_root` is only printed.
- `--et_session` selects the eye-tracker session folder (e.g. 2021-12-06-08-40-37) when the
  recording contains several sessions.
- There is no `--tolerance_ns` option: every frame always receives the nearest IMU sample.
- The default ET IMU column names are `roll [deg]` and `pitch [deg]` (step 09: `roll`, `pitch`).

What it does
1) Locates the head-IMU epoch CSV in `--head_root/<rec_key>/` (the largest file if there are several).
2) Locates the eye-tracker export folder (world.csv and imu.csv), using `--et_session` if given.
3) Pairs the world.csv frame timestamps with the sorted frame images by index
   (n = min(number of timestamps, number of images)).
4) Optionally drops frames outside the common ET + head time range (`--drop_outside_overlap`).
5) For every frame, takes the nearest eye-tracker sample (roll, pitch) and the nearest head sample
   (yaw), and the arithmetic mean of `2 * half_window + 1` samples around each of them.
6) Writes `<out_root>/<rec_key>/paired_frames.csv` with the columns:
     recording_name, frame_name, timestamp_ns,
     roll_nn, pitch_nn, roll_win, pitch_win, yaw_nn, yaw_win
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_nearest_indices(sorted_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Nearest neighbor indices in sorted_ts for each query_ts (both ns)."""
    idx = np.searchsorted(sorted_ts, query_ts, side="left")
    idx = np.clip(idx, 0, len(sorted_ts))

    idx0 = np.clip(idx - 1, 0, len(sorted_ts) - 1)
    idx1 = np.clip(idx,     0, len(sorted_ts) - 1)

    d0 = np.abs(sorted_ts[idx0] - query_ts)
    d1 = np.abs(sorted_ts[idx1] - query_ts)

    use1 = d1 < d0
    return np.where(use1, idx1, idx0)


def window_mean(arr: np.ndarray, centers: np.ndarray, half: int) -> np.ndarray:
    """
    Mean over window size = 2*half+1 around each center (center included).
    Boundary-safe with clipping.
    """
    n = len(arr)
    out = np.empty(len(centers), dtype=np.float64)
    for k, i in enumerate(centers):
        lo = 0 if i - half < 0 else i - half
        hi = n if i + half + 1 > n else i + half + 1
        w = arr[lo:hi]
        out[k] = float(np.mean(w)) if len(w) else np.nan
    return out


def find_et_export_dir(root: Path, rec_key: str, et_session: str | None) -> Path:
    """
    Find: root/<rec_key>/ET/<date>/export
    If et_session is provided, use that folder name exactly.
    """
    rec_dir = root / rec_key
    et_dir = rec_dir / "ET"
    if not et_dir.exists():
        raise FileNotFoundError(f"Missing ET folder: {et_dir}")

    if et_session is not None:
        d = et_dir / et_session / "export"
        if not (d / "world.csv").exists() or not (d / "imu.csv").exists():
            raise FileNotFoundError(
                f"ET session export not found or missing files:\n"
                f"  {d}\n"
                f"Expected world.csv and imu.csv"
            )
        return d

    # fallback: pick the first folder containing export/world.csv + export/imu.csv
    for d0 in sorted(et_dir.glob("*")):
        exp = d0 / "export"
        if (exp / "world.csv").exists() and (exp / "imu.csv").exists():
            return exp

    raise FileNotFoundError(f"Could not find any ET export dir under: {et_dir}")


def list_images_sorted(frames_dir: Path) -> list[Path]:
    imgs = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort(key=lambda p: p.name)
    return imgs


def main():
    ap = argparse.ArgumentParser()

    # fixed target (only 032)
    ap.add_argument("--rec_key", default="032_LS168/trial_1_om",
                    help="Recording key (default: 032_LS168/trial_1_om)")

    ap.add_argument("--root", required=True, help="Dataset root (contains recordings folders)")
    ap.add_argument("--head_root", required=True, help="Root of head synced csv epoch (recording subfolders)")

    # you keep this as the global frames root
    ap.add_argument("--frames_root", required=True, help="Global frames root (not per-recording folder)")

    # but for tricky ones you override directly
    ap.add_argument("--frames_dir", required=True,
                    help="DIRECT path to THIS recording frames folder (images).")

    ap.add_argument("--out_root", required=True, help="Output root for paired CSVs")

    # IMPORTANT: for recordings with multiple ET sessions
    ap.add_argument("--et_session", default=None,
                    help="Optional ET session folder name (e.g. 2021-12-06-08-40-37).")

    ap.add_argument("--drop_outside_overlap", action="store_true",
                    help="Drop frames outside overlap of ET+HEAD time ranges.")

    ap.add_argument("--half_window", type=int, default=5, help="Half window (5 => window=11)")

    # Column names
    ap.add_argument("--world_ts_col", default="timestamp [ns]", help="world.csv timestamp column")
    ap.add_argument("--et_ts_col", default="timestamp [ns]", help="ET imu.csv timestamp column")
    ap.add_argument("--et_roll_col", default="roll [deg]", help="ET roll col")
    ap.add_argument("--et_pitch_col", default="pitch [deg]", help="ET pitch col")
    ap.add_argument("--head_ts_col", default="head_epoch_ns", help="Head epoch ts column in head csv")
    ap.add_argument("--head_yaw_col", default="yaw", help="Head yaw col")

    args = ap.parse_args()

    root = Path(args.root)
    head_root = Path(args.head_root)
    frames_root = Path(args.frames_root)  # kept for consistency / logging
    frames_dir = Path(args.frames_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rec_key = args.rec_key

    print(f"Running ONLY: {rec_key}")
    print(f"Frames root: {frames_root}")
    print(f"Frames dir : {frames_dir}")
    print(f"ET session : {args.et_session}")
    print(f"Window: {2*args.half_window+1} rows (half={args.half_window}, center included)")
    print(f"Strict overlap drop: {args.drop_outside_overlap}")

    # --- locate head csv (there should be exactly 1 per rec_key)
    head_dir = head_root / rec_key
    head_csvs = sorted(head_dir.glob("*.csv"))
    if not head_csvs:
        raise FileNotFoundError(f"Missing head csv in {head_dir}")
    if len(head_csvs) > 1:
        head_csvs.sort(key=lambda x: x.stat().st_size, reverse=True)
    head_csv = head_csvs[0]

    # --- locate ET export dir (world.csv + imu.csv)
    et_export = find_et_export_dir(root, rec_key, args.et_session)
    world_csv = et_export / "world.csv"
    et_imu_csv = et_export / "imu.csv"

    # --- validate frames dir
    if not frames_dir.exists() or not frames_dir.is_dir():
        raise FileNotFoundError(f"frames_dir does not exist or is not a directory: {frames_dir}")

    imgs = list_images_sorted(frames_dir)
    if not imgs:
        raise FileNotFoundError(f"No images found in frames dir: {frames_dir}")

    # --- load world (frame timestamps)
    world = pd.read_csv(world_csv)
    if args.world_ts_col not in world.columns:
        raise ValueError(f"world.csv missing '{args.world_ts_col}'. cols={list(world.columns)}")
    frame_ts = world[args.world_ts_col].to_numpy(dtype=np.int64)

    # pair frame_ts with image names by index
    n = min(len(frame_ts), len(imgs))
    if n <= 0:
        raise ValueError("No overlap between world rows and images.")
    frame_ts = frame_ts[:n]
    frame_names = np.array([p.name for p in imgs[:n]], dtype=object)

    # --- load ET imu
    et = pd.read_csv(et_imu_csv)
    for c in (args.et_ts_col, args.et_roll_col, args.et_pitch_col):
        if c not in et.columns:
            raise ValueError(f"ET imu missing '{c}'. cols={list(et.columns)}")

    et_ts = et[args.et_ts_col].to_numpy(dtype=np.int64)
    et_roll = et[args.et_roll_col].to_numpy(dtype=np.float64)
    et_pitch = et[args.et_pitch_col].to_numpy(dtype=np.float64)

    ord_et = np.argsort(et_ts)
    et_ts = et_ts[ord_et]
    et_roll = et_roll[ord_et]
    et_pitch = et_pitch[ord_et]

    # --- load Head imu (epoch)
    head = pd.read_csv(head_csv)
    for c in (args.head_ts_col, args.head_yaw_col):
        if c not in head.columns:
            raise ValueError(f"Head csv missing '{c}'. cols={list(head.columns)}")

    head_ts = head[args.head_ts_col].to_numpy(dtype=np.int64)
    head_yaw = head[args.head_yaw_col].to_numpy(dtype=np.float64)

    ord_h = np.argsort(head_ts)
    head_ts = head_ts[ord_h]
    head_yaw = head_yaw[ord_h]

    # --- strict overlap drop (ONLY trims beginning/end, never skips middle)
    if args.drop_outside_overlap:
        lo = max(et_ts[0], head_ts[0])
        hi = min(et_ts[-1], head_ts[-1])
        keep = (frame_ts >= lo) & (frame_ts <= hi)
        frame_ts = frame_ts[keep]
        frame_names = frame_names[keep]

    if len(frame_ts) == 0:
        raise ValueError(
            "After overlap drop, no frames remain.\n"
            "This usually means you picked the wrong ET session.\n"
            "Try --et_session with the correct folder (e.g. 2021-12-06-08-40-37)."
        )

    # --- Pair ET roll/pitch to frames (NO tolerance => NEVER NaN)
    et_nn = find_nearest_indices(et_ts, frame_ts)
    roll_nn = et_roll[et_nn]
    pitch_nn = et_pitch[et_nn]
    roll_win = window_mean(et_roll, et_nn, half=int(args.half_window))
    pitch_win = window_mean(et_pitch, et_nn, half=int(args.half_window))

    # --- Pair HEAD yaw to frames (NO tolerance => NEVER NaN)
    h_nn = find_nearest_indices(head_ts, frame_ts)
    yaw_nn = head_yaw[h_nn]
    yaw_win = window_mean(head_yaw, h_nn, half=int(args.half_window))

    out = pd.DataFrame({
        "recording_name": rec_key,
        "frame_name": frame_names,
        "timestamp_ns": frame_ts,  # epoch ns (from world.csv)
        "roll_nn": roll_nn,
        "pitch_nn": pitch_nn,
        "roll_win": roll_win,
        "pitch_win": pitch_win,
        "yaw_nn": yaw_nn,
        "yaw_win": yaw_win,
    })

    out_path = out_root / rec_key / "paired_frames.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("\nDONE")
    print(f"Wrote: {out_path}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()