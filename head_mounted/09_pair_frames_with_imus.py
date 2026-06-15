"""
09_pair_frames_with_imus.py

Purpose
- Create a per-frame training/analysis table by aligning each exported video frame (world camera)
  to the nearest ET IMU samples (roll/pitch) and head IMU samples (yaw), all on a common epoch
  nanosecond timeline.

What it does
- Discovers recordings from `--head_root` using the folder layout `<LS>/<trial>/*.csv`.
- For each recording:
  1) Loads frame timestamps from `ET/<date>/export/world.csv` and pairs them by index with the
     sorted list of extracted frame image filenames in `--frames_root`.
  2) Loads ET IMU signals from `ET/<date>/export/imu.csv` (timestamp + roll/pitch) and head IMU
     signals from the head epoch CSV (head_epoch_ns + yaw), ensuring both time axes are sorted.
  3) For every frame timestamp, finds the nearest-neighbor IMU index (binary search) and exports:
       - NN features: roll_nn, pitch_nn, yaw_nn
       - Smoothed features: roll_win, pitch_win, yaw_win (mean over a window around the NN index)
     Optionally enforces a max time error (`--tolerance_ns`) and can drop frames outside the strict
     overlap of ET+HEAD time ranges (`--drop_outside_overlap`).
  4) Writes `paired_frames.csv` under `--out_root/<LS>/<trial>/`.

Inputs / Assumptions
- `world.csv` provides epoch nanosecond timestamps for each frame (column `--world_ts_col`).
- Extracted frame images exist and are sortable by filename (e.g., `frame_000001.png`), so index
  order matches `world.csv` row order; the script uses `min(len(world_ts), len(images))`.
- ET IMU CSV contains `{timestamp, roll, pitch}` columns; head CSV contains `{head_epoch_ns, yaw}`.
- Window smoothing is index-based around the nearest IMU sample (not time-weighted).

Outputs
- `paired_frames.csv` with:
    recording_name, frame_name, timestamp_ns,
    roll_nn, pitch_nn, roll_win, pitch_win,
    yaw_nn, yaw_win

Notes / Pitfalls
- If `--tolerance_ns` is set, samples exceeding tolerance are set to NaN (can reduce usable rows).
- If `--drop_outside_overlap` is enabled, frames outside the common ET+HEAD time span are removed.
- The correctness of frame↔timestamp pairing depends on your frame extraction preserving order
  and producing one image per `world.csv` timestamp in the same sequence.
"""


from __future__ import annotations

import argparse
import time
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


def eta_str(done: int, total: int, elapsed_s: float) -> str:
    if done <= 0:
        return "?"
    rate = elapsed_s / done
    remaining = (total - done) * rate
    m = int(remaining // 60)
    s = int(remaining % 60)
    return f"{m}m{s:02d}s"


def first_existing(pats: list[Path]) -> Path | None:
    for p in pats:
        if p.exists():
            return p
    return None


def find_et_export_dir(root: Path, rec_key: str) -> Path | None:
    """
    Find: root/<rec_key>/ET/<date>/export
    rec_key = "004_LS47/trial_1_fcn"
    """
    rec_dir = root / rec_key
    et_dir = rec_dir / "ET"
    if not et_dir.exists():
        return None
    # choose the first directory that contains export/world.csv
    for d in sorted(et_dir.glob("*")):
        exp = d / "export"
        if (exp / "world.csv").exists() and (exp / "imu.csv").exists():
            return exp
    return None


def find_frames_dir(frames_root: Path, rec_key: str) -> Path | None:
    """
    Frames are images. We try common layouts:
      - frames_root/<rec_key>/
      - frames_root/<LS>/<trial>/
      - frames_root/<LS>_<trial>/  (rare)
    """
    a, b = rec_key.split("/", 1)

    candidates = [
        frames_root / a / b,
        frames_root / rec_key,
        frames_root / f"{a}_{b}",
    ]
    d = first_existing(candidates)
    if d is not None and d.is_dir():
        return d

    # fallback: search any dir whose path contains both tokens
    hits = []
    for p in frames_root.rglob("*"):
        if p.is_dir():
            s = str(p).replace("\\", "/")
            if a in s and b in s:
                hits.append(p)
    if not hits:
        return None
    hits.sort(key=lambda x: len(str(x)))
    return hits[0]


def list_images_sorted(frames_dir: Path) -> list[Path]:
    imgs = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    # sort by name; if names are frame_000001.jpg this is correct
    imgs.sort(key=lambda p: p.name)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains recordings folders)")
    ap.add_argument("--head_root", required=True, help="Root of head synced csv epoch (recording subfolders)")
    ap.add_argument("--frames_root", required=True, help="Root folder containing extracted frame images")
    ap.add_argument("--out_root", required=True, help="Output root for paired CSVs")

    ap.add_argument("--drop_outside_overlap", action="store_true",
                    help="Drop frames that are outside [max(starts), min(ends)] across ET+HEAD sensors.")
    ap.add_argument("--half_window", type=int, default=5, help="Half window (5 => window=11)")
    ap.add_argument("--tolerance_ns", type=np.int64, default=None,
                    help="Optional tolerance for NN pairing; if exceeded, NaN. (Usually keep None)")

    # Column names
    ap.add_argument("--world_ts_col", default="timestamp [ns]", help="world.csv timestamp column")
    ap.add_argument("--et_ts_col", default="timestamp [ns]", help="ET imu.csv timestamp column")
    ap.add_argument("--et_roll_col", default="roll", help="ET roll col")
    ap.add_argument("--et_pitch_col", default="pitch", help="ET pitch col")
    ap.add_argument("--head_ts_col", default="head_epoch_ns", help="Head epoch ts column in head csv")
    ap.add_argument("--head_yaw_col", default="yaw", help="Head yaw col")

    args = ap.parse_args()

    root = Path(args.root)
    head_root = Path(args.head_root)
    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # recordings discovered from head_root structure: <head_root>/<LS>/<trial>/*.csv
    rec_keys = []
    for p in sorted(head_root.rglob("*.csv")):
        rel = p.relative_to(head_root)
        if len(rel.parts) >= 3:
            rec_key = f"{rel.parts[0]}/{rel.parts[1]}"
            if rec_key not in rec_keys:
                rec_keys.append(rec_key)

    total = len(rec_keys)
    print(f"Discovered recordings (from head_root): {total}")
    print(f"Window: {2*args.half_window+1} rows (half={args.half_window}, center included)")
    print(f"Strict overlap drop: {args.drop_outside_overlap}")
    print(f"Tolerance_ns: {args.tolerance_ns}")

    ok = skip = fail = 0
    t0 = time.time()

    for i, rec_key in enumerate(rec_keys, start=1):
        step_t0 = time.time()
        prefix = f"[{i:3d}/{total}]"

        try:
            # --- locate head csv (there should be exactly 1 per rec_key)
            head_dir = head_root / rec_key
            head_csvs = sorted(head_dir.glob("*.csv"))
            if not head_csvs:
                raise FileNotFoundError(f"Missing head csv in {head_dir}")
            if len(head_csvs) > 1:
                # pick largest
                head_csvs.sort(key=lambda x: x.stat().st_size, reverse=True)
            head_csv = head_csvs[0]

            # --- locate ET export dir (world.csv + imu.csv)
            et_export = find_et_export_dir(root, rec_key)
            if et_export is None:
                raise FileNotFoundError(f"Missing ET export dir with world.csv+imu.csv for {rec_key}")

            world_csv = et_export / "world.csv"
            et_imu_csv = et_export / "imu.csv"

            # --- locate frames images dir
            frames_dir = find_frames_dir(frames_root, rec_key)
            if frames_dir is None:
                raise FileNotFoundError(f"Missing frames images dir for {rec_key} under {frames_root}")

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

            # ensure sorted
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

            # --- strict overlap drop
            if args.drop_outside_overlap:
                lo = max(et_ts[0], head_ts[0])
                hi = min(et_ts[-1], head_ts[-1])
                keep = (frame_ts >= lo) & (frame_ts <= hi)
                frame_ts = frame_ts[keep]
                frame_names = frame_names[keep]

            if len(frame_ts) == 0:
                raise ValueError("After overlap drop, no frames remain.")

            # --- Pair ET roll/pitch to frames
            et_nn = find_nearest_indices(et_ts, frame_ts)
            if args.tolerance_ns is not None:
                dt_et = np.abs(et_ts[et_nn] - frame_ts)
                ok_et = dt_et <= np.int64(args.tolerance_ns)
            else:
                ok_et = np.ones(len(frame_ts), dtype=bool)

            roll_nn = np.where(ok_et, et_roll[et_nn], np.nan)
            pitch_nn = np.where(ok_et, et_pitch[et_nn], np.nan)

            roll_win = window_mean(et_roll, et_nn, half=int(args.half_window))
            pitch_win = window_mean(et_pitch, et_nn, half=int(args.half_window))
            roll_win = np.where(ok_et, roll_win, np.nan)
            pitch_win = np.where(ok_et, pitch_win, np.nan)

            # --- Pair HEAD yaw to frames
            h_nn = find_nearest_indices(head_ts, frame_ts)
            if args.tolerance_ns is not None:
                dt_h = np.abs(head_ts[h_nn] - frame_ts)
                ok_h = dt_h <= np.int64(args.tolerance_ns)
            else:
                ok_h = np.ones(len(frame_ts), dtype=bool)

            yaw_nn = np.where(ok_h, head_yaw[h_nn], np.nan)
            yaw_win = window_mean(head_yaw, h_nn, half=int(args.half_window))
            yaw_win = np.where(ok_h, yaw_win, np.nan)


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

            ok += 1
            elapsed = time.time() - t0
            eta = eta_str(i, total, elapsed)
            step_s = time.time() - step_t0
            print(f"{prefix} OK  | ok={ok} skip={skip} fail={fail} | {step_s:.2f}s | ETA {eta} | {rec_key}")
            # print(f"    -> {out_path} (rows={len(out)})")

        except Exception as e:
            fail += 1
            elapsed = time.time() - t0
            eta = eta_str(i, total, elapsed)
            step_s = time.time() - step_t0
            print(f"{prefix} FAIL | ok={ok} skip={skip} fail={fail} | {step_s:.2f}s | ETA {eta} | {rec_key}")
            print(f"    -> {e}")

    print("\nDONE")
    print(f"OK={ok} SKIP={skip} FAIL={fail}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()