from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

TS_RE = re.compile(r"_ts_(\d+)")
FRAME_NUM_RE = re.compile(r"frame_(\d+)_ts_", re.IGNORECASE)

# =========================
# Participant special cases
# =========================
DEFAULT_ALLOWED_LEGS = {2, 3, 4}
ALLOWED_LEGS_OVERRIDE = {
    "p33": {2, 3},
    "p103": {3, 4},
}

TWO_FOLDER_OVERRIDES = {
    "p33":  [2, 3],
    "p103": [3, 4],
}

# =========================
# NN + window
# =========================
def find_nearest_indices(sorted_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_ts, query_ts, side="left")
    idx = np.clip(idx, 0, len(sorted_ts))

    idx0 = np.clip(idx - 1, 0, len(sorted_ts) - 1)
    idx1 = np.clip(idx,     0, len(sorted_ts) - 1)

    d0 = np.abs(sorted_ts[idx0] - query_ts)
    d1 = np.abs(sorted_ts[idx1] - query_ts)

    use1 = d1 < d0
    return np.where(use1, idx1, idx0)


def window_mean(arr: np.ndarray, centers: np.ndarray, half: int) -> np.ndarray:
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


# =========================
# Frames helpers (FAST + SAFE)
# =========================
def extract_frame_ts_ns(name: str) -> Optional[int]:
    m = TS_RE.search(name)
    return int(m.group(1)) if m else None


def extract_frame_number(name: str) -> Optional[int]:
    m = FRAME_NUM_RE.search(name)
    return int(m.group(1)) if m else None


def list_images_with_ts(frames_dir: Path) -> List[Dict]:
    """
    Faster + safer than Path.iterdir() + is_file() stat() loops.
    Skips entries that raise errors.
    """
    out: List[Dict] = []
    frames_dir_str = str(frames_dir)

    try:
        with os.scandir(frames_dir_str) as it:
            for entry in it:
                try:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    ext = Path(name).suffix.lower()
                    if ext not in IMG_EXTS:
                        continue
                    ts = extract_frame_ts_ns(name)
                    fn = extract_frame_number(name)
                    if ts is None or fn is None:
                        continue
                    out.append({"path": str(Path(frames_dir_str) / name), "name": name, "frame_num": fn, "ts_ns": ts})
                except (OSError, FileNotFoundError):
                    # in case stat fails on a single entry, just skip it
                    continue
    except FileNotFoundError:
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

    if not out:
        raise FileNotFoundError(f"No valid frames found in: {frames_dir}")

    out.sort(key=lambda d: d["frame_num"])
    return out


def list_subfolders_sorted(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def build_folder_index_to_leg(pid: str, subfolders: List[Path]) -> Dict[int, int]:
    n = len(subfolders)
    if n >= 3:
        return {0: 2, 1: 3, 2: 4}
    if n == 2:
        legs = TWO_FOLDER_OVERRIDES.get(pid, [2, 3])
        return {0: legs[0], 1: legs[1]}
    raise FileNotFoundError(f"{pid}: expected 2 or 3+ folders, found {n}")


def allowed_legs_for_participant(pid: str) -> set[int]:
    return ALLOWED_LEGS_OVERRIDE.get(pid, DEFAULT_ALLOWED_LEGS)


# =========================
# External (aligned_rpy) loader
# =========================
def external_csv_path(external_root: Path, pid: str, leg: int) -> Path:
    return external_root / pid / f"leg_{leg}_allInOne_aligned_rpy.csv"


def load_external_rpy(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, engine="python")

    needed = ["upper_world_epoch_ns", "roll_deg", "pitch_deg", "yaw_deg"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"External RPY missing columns {missing} in {csv_path}. Found={list(df.columns)}")

    ts = pd.to_numeric(df["upper_world_epoch_ns"], errors="coerce").to_numpy(dtype=np.float64)
    roll = pd.to_numeric(df["roll_deg"], errors="coerce").to_numpy(dtype=np.float64)
    pitch = pd.to_numeric(df["pitch_deg"], errors="coerce").to_numpy(dtype=np.float64)
    yaw = pd.to_numeric(df["yaw_deg"], errors="coerce").to_numpy(dtype=np.float64)

    ok = ~np.isnan(ts)
    ts = ts[ok].astype(np.int64)
    roll = roll[ok]
    pitch = pitch[ok]
    yaw = yaw[ok]

    order = np.argsort(ts)
    return ts[order], roll[order], pitch[order], yaw[order]


# =========================
# ET IMU loader (NO export/)
# =========================
def et_rec_dir_for_leg(et_root: Path, pid: str, leg: int) -> Path:
    pid_dir = et_root / pid
    if not pid_dir.exists():
        raise FileNotFoundError(f"Missing ET participant folder: {pid_dir}")

    subdirs = list_subfolders_sorted(pid_dir)
    if len(subdirs) < 2:
        raise FileNotFoundError(f"ET participant folder has <2 subfolders: {pid_dir}")

    folder_to_leg = build_folder_index_to_leg(pid, subdirs)
    leg_to_folder = {L: idx for idx, L in folder_to_leg.items()}
    if leg not in leg_to_folder:
        raise FileNotFoundError(f"{pid}: leg{leg} not available by ET folder mapping. subdirs={[d.name for d in subdirs]}")

    return subdirs[leg_to_folder[leg]]


def load_et_roll_pitch(et_rec_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    imu_csv = et_rec_dir / "imu.csv"
    if not imu_csv.exists():
        raise FileNotFoundError(f"Missing ET imu.csv: {imu_csv}")

    df = pd.read_csv(imu_csv, engine="python")

    ts_col = "timestamp [ns]"
    if ts_col not in df.columns:
        raise ValueError(f"ET imu.csv missing '{ts_col}' in {imu_csv}. Found={list(df.columns)}")

    # Your actual columns are: roll [deg], pitch [deg]
    # Keep fallback to previous names just in case.
    roll_candidates = ["roll [deg]", "roll_deg", "roll"]
    pitch_candidates = ["pitch [deg]", "pitch_deg", "pitch"]

    roll_col = next((c for c in roll_candidates if c in df.columns), None)
    pitch_col = next((c for c in pitch_candidates if c in df.columns), None)

    if roll_col is None or pitch_col is None:
        raise ValueError(
            f"ET imu.csv missing roll/pitch columns in {imu_csv}. "
            f"Tried roll={roll_candidates}, pitch={pitch_candidates}. Found={list(df.columns)}"
        )

    ts = pd.to_numeric(df[ts_col], errors="coerce").to_numpy(dtype=np.float64)
    roll = pd.to_numeric(df[roll_col], errors="coerce").to_numpy(dtype=np.float64)
    pitch = pd.to_numeric(df[pitch_col], errors="coerce").to_numpy(dtype=np.float64)

    ok = ~np.isnan(ts)
    ts = ts[ok].astype(np.int64)
    roll = roll[ok]
    pitch = pitch[ok]

    order = np.argsort(ts)
    return ts[order], roll[order], pitch[order], imu_csv


# =========================
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external_root", default=r"D:\upper_body_sync_2\imu")
    ap.add_argument("--et_root", default=r"E:\john\ET-IMU-and-upper-body\et")
    ap.add_argument("--frames_root", default=r"D:\john.panagiotopoulos\DERIVED_ET_IMU_upper_body\frames_world")
    ap.add_argument("--out_root", default=r"D:\upper_body_sync_2\paired_frames")

    ap.add_argument("--half_window", type=int, default=5, help="Half window (5 => window=11)")
    ap.add_argument("--tolerance_ns", type=np.int64, default=None)
    ap.add_argument("--drop_outside_overlap", action="store_true", default=True)

    args = ap.parse_args()

    external_root = Path(args.external_root)
    et_root = Path(args.et_root)
    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    participants = sorted([p.name for p in frames_root.iterdir() if p.is_dir() and p.name.lower().startswith("p")],
                          key=str.lower)

    print(f"Participants in frames_root: {len(participants)}")
    print(f"Window = {2*args.half_window+1} samples (half={args.half_window})")
    print(f"drop_outside_overlap = {args.drop_outside_overlap}")
    print(f"tolerance_ns = {args.tolerance_ns}")

    tasks: List[Tuple[str, int, Path]] = []
    for pid in participants:
        pid_frames_dir = frames_root / pid
        subdirs = list_subfolders_sorted(pid_frames_dir)
        if len(subdirs) < 2:
            continue
        folder_to_leg = build_folder_index_to_leg(pid, subdirs)
        allowed = allowed_legs_for_participant(pid)
        for idx, leg in folder_to_leg.items():
            if leg not in allowed:
                continue
            tasks.append((pid, leg, subdirs[idx]))

    total = len(tasks)
    print(f"Planned tasks (pid, leg): {total}")

    ok = fail = 0
    t0 = time.time()

    for i, (pid, leg, frames_dir) in enumerate(tasks, start=1):
        prefix = f"[{i:3d}/{total}]"
        step_t0 = time.time()

        try:
            frames = list_images_with_ts(frames_dir)
            frame_ts = np.array([f["ts_ns"] for f in frames], dtype=np.int64)
            frame_names = np.array([f["name"] for f in frames], dtype=object)

            ext_csv = external_csv_path(external_root, pid, leg)
            if not ext_csv.exists():
                raise FileNotFoundError(f"Missing external RPY CSV: {ext_csv}")
            ext_ts, ext_roll, ext_pitch, ext_yaw = load_external_rpy(ext_csv)

            et_rec_dir = et_rec_dir_for_leg(et_root, pid, leg)
            et_ts, et_roll, et_pitch, et_imu_csv = load_et_roll_pitch(et_rec_dir)

            if args.drop_outside_overlap:
                lo = max(et_ts[0], ext_ts[0])
                hi = min(et_ts[-1], ext_ts[-1])
                keep = (frame_ts >= lo) & (frame_ts <= hi)
                frame_ts = frame_ts[keep]
                frame_names = frame_names[keep]

            if len(frame_ts) == 0:
                raise ValueError("No frames remain after overlap filtering (or no frames).")

            et_nn = find_nearest_indices(et_ts, frame_ts)
            ext_nn = find_nearest_indices(ext_ts, frame_ts)

            dt_et = np.abs(et_ts[et_nn] - frame_ts)
            dt_ext = np.abs(ext_ts[ext_nn] - frame_ts)

            if args.tolerance_ns is not None:
                ok_et = dt_et <= np.int64(args.tolerance_ns)
                ok_ext = dt_ext <= np.int64(args.tolerance_ns)
            else:
                ok_et = np.ones(len(frame_ts), dtype=bool)
                ok_ext = np.ones(len(frame_ts), dtype=bool)

            roll_et_nn = np.where(ok_et, et_roll[et_nn], np.nan)
            pitch_et_nn = np.where(ok_et, et_pitch[et_nn], np.nan)
            roll_et_win = window_mean(et_roll, et_nn, half=int(args.half_window))
            pitch_et_win = window_mean(et_pitch, et_nn, half=int(args.half_window))
            roll_et_win = np.where(ok_et, roll_et_win, np.nan)
            pitch_et_win = np.where(ok_et, pitch_et_win, np.nan)

            roll_ext_nn = np.where(ok_ext, ext_roll[ext_nn], np.nan)
            pitch_ext_nn = np.where(ok_ext, ext_pitch[ext_nn], np.nan)
            yaw_ext_nn = np.where(ok_ext, ext_yaw[ext_nn], np.nan)

            roll_ext_win = window_mean(ext_roll, ext_nn, half=int(args.half_window))
            pitch_ext_win = window_mean(ext_pitch, ext_nn, half=int(args.half_window))
            yaw_ext_win = window_mean(ext_yaw, ext_nn, half=int(args.half_window))
            roll_ext_win = np.where(ok_ext, roll_ext_win, np.nan)
            pitch_ext_win = np.where(ok_ext, pitch_ext_win, np.nan)
            yaw_ext_win = np.where(ok_ext, yaw_ext_win, np.nan)

            out = pd.DataFrame({
                "participant": pid,
                "leg": leg,
                "frames_dir": str(frames_dir),
                "frame_name": frame_names,
                "frame_ts_ns": frame_ts,

                "roll_et_nn": roll_et_nn,
                "pitch_et_nn": pitch_et_nn,
                "roll_et_win": roll_et_win,
                "pitch_et_win": pitch_et_win,

                "roll_external_nn": roll_ext_nn,
                "pitch_external_nn": pitch_ext_nn,
                "yaw_external_nn": yaw_ext_nn,
                "roll_external_win": roll_ext_win,
                "pitch_external_win": pitch_ext_win,
                "yaw_external_win": yaw_ext_win,

                "dt_et_nn_ns": dt_et,
                "dt_external_nn_ns": dt_ext,

                "et_imu_csv": str(et_imu_csv),
                "external_rpy_csv": str(ext_csv),
                "et_rec_dir": str(et_rec_dir),
            })

            out_path = out_root / pid / f"leg_{leg}" / "paired_frames.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False)

            ok += 1
            elapsed = time.time() - t0
            eta = eta_str(i, total, elapsed)
            step_s = time.time() - step_t0
            print(f"{prefix} OK  | {step_s:.2f}s | ETA {eta} | {pid} leg{leg} | rows={len(out)}")

        except Exception as e:
            fail += 1
            elapsed = time.time() - t0
            eta = eta_str(i, total, elapsed)
            step_s = time.time() - step_t0
            print(f"{prefix} FAIL | {step_s:.2f}s | ETA {eta} | {pid} leg{leg}")
            print(f"    -> {e}")

    print("\nDONE")
    print(f"OK={ok} FAIL={fail}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()