from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# =========================
# CONFIG
# =========================
IMU_ROOT = Path(r"D:\upper_body_sync_2\imu")
FRAMES_ROOT = Path(r"D:\john.panagiotopoulos\DERIVED_ET_IMU_upper_body\frames_world")

# default legs we care about (we will pick subset per participant)
DEFAULT_LEGS = [2, 3, 4]

# participants with only 2 recording folders and special leg availability
# meaning: if there are 2 folders, map folder1->these[0], folder2->these[1]
TWO_FOLDER_OVERRIDES = {
    "p33":  [2, 3],  # p33 has only legs 1&2, so use leg2+leg3
    "p103": [3, 4],  # p103 has only legs 2&3, so use leg3+leg4
}

# frame filename example: frame_000002_ts_1722604262503000000
TS_RE = re.compile(r"_ts_(\d+)")
FRAME_NUM_RE = re.compile(r"frame_(\d+)_ts_")


# =========================
# HELPERS
# =========================
def read_first_row_last_col_as_int(csv_path: Path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV (no header): {csv_path}")
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"No data rows in CSV: {csv_path}")
        val = first_row[-1].strip()
        return int(val)


def extract_ts_ns(name: str) -> Optional[int]:
    m = TS_RE.search(name)
    return int(m.group(1)) if m else None


def extract_frame_number(name: str) -> Optional[int]:
    m = FRAME_NUM_RE.search(name)
    return int(m.group(1)) if m else None


def list_recording_folders(frames_pid_dir: Path) -> List[Path]:
    """Sorted subfolders under frames_world\pXXX\ ."""
    subdirs = sorted([p for p in frames_pid_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    return subdirs


def scan_frames(frames_dir: Path) -> List[Dict]:
    frames: List[Dict] = []
    for p in frames_dir.iterdir():
        if not p.is_file():
            continue
        ts = extract_ts_ns(p.name)
        fn = extract_frame_number(p.name)
        if ts is None or fn is None:
            continue
        frames.append({"path": p, "name": p.name, "frame_num": fn, "ts_ns": ts})

    if not frames:
        raise ValueError(f"No valid frames found in: {frames_dir}")

    frames.sort(key=lambda d: d["frame_num"])
    return frames


def find_closest_frame(frames: List[Dict], target_ns: int) -> Dict:
    best = None
    best_diff = None
    for fr in frames:
        diff = abs(fr["ts_ns"] - target_ns)
        if best_diff is None or diff < best_diff:
            best = fr
            best_diff = diff
    out = dict(best)
    out["abs_diff_ns"] = int(best_diff)
    return out


def build_folder_to_leg_map(pid: str, recording_folders: List[Path]) -> Dict[int, int]:
    """
    Returns mapping: folder_index -> leg
    folder_index is 0-based.
    """
    n = len(recording_folders)

    if n >= 3:
        # normal: folder1->leg2, folder2->leg3, folder3->leg4
        return {0: 2, 1: 3, 2: 4}

    if n == 2:
        if pid in TWO_FOLDER_OVERRIDES:
            legs = TWO_FOLDER_OVERRIDES[pid]
            return {0: legs[0], 1: legs[1]}
        else:
            # sensible default if some other participant also has 2 folders:
            # assume leg2, leg3
            return {0: 2, 1: 3}

    raise FileNotFoundError(f"Participant {pid} has {n} recording folders in {FRAMES_ROOT/pid}, expected >=2.")


def format_table(rows: List[Dict], cols: List[Tuple[str, str, int]]) -> str:
    """
    cols: list of (key, header, width)
    """
    def clip(s: str, w: int) -> str:
        if len(s) <= w:
            return s
        return s[: max(0, w - 3)] + "..."

    lines = []
    header = " ".join(f"{h:<{w}}" for _, h, w in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        line = " ".join(
            f"{clip(str(r.get(k, '')), w):<{w}}"
            for k, _, w in cols
        )
        lines.append(line)
    return "\n".join(lines)


# =========================
# MAIN
# =========================
def main() -> None:
    # common participants between IMU and FRAMES
    imu_pids = sorted([p.name for p in IMU_ROOT.iterdir() if p.is_dir() and p.name.lower().startswith("p")],
                      key=str.lower)
    pids = [pid for pid in imu_pids if (FRAMES_ROOT / pid).exists()]

    if not pids:
        raise RuntimeError("No common participants between IMU_ROOT and FRAMES_ROOT.")

    results: List[Dict] = []

    for pid in pids:
        frames_pid_dir = FRAMES_ROOT / pid
        recording_folders = list_recording_folders(frames_pid_dir)

        if len(recording_folders) < 2:
            print(f"[SKIP] {pid}: has <2 recording folders: {frames_pid_dir}")
            continue

        folder_to_leg = build_folder_to_leg_map(pid, recording_folders)

        # For each folder index we will look for the corresponding leg CSV
        for folder_idx, leg in folder_to_leg.items():
            if folder_idx >= len(recording_folders):
                continue  # safety

            frames_dir = recording_folders[folder_idx]
            imu_csv = IMU_ROOT / pid / f"leg_{leg}_allInOne_aligned.csv"
            if not imu_csv.exists():
                print(f"[SKIP] {pid}: missing CSV for mapped leg{leg}: {imu_csv}")
                continue

            target_ns = read_first_row_last_col_as_int(imu_csv)
            frames = scan_frames(frames_dir)

            first_fr = frames[0]
            last_fr = frames[-1]
            closest = find_closest_frame(frames, target_ns)

            results.append({
                "participant": pid,
                "leg": leg,

                "first_row_ns": target_ns,

                "closest_frame_num": closest["frame_num"],
                "closest_frame_ts": closest["ts_ns"],

                "first_frame_num": first_fr["frame_num"],
                "first_frame_ts": first_fr["ts_ns"],

                "last_frame_num": last_fr["frame_num"],
                "last_frame_ts": last_fr["ts_ns"],

                "csv_path": str(imu_csv),
                "frames_dir": str(frames_dir),
                "closest_frame_path": str(closest["path"]),

                "abs_diff_ns": closest["abs_diff_ns"],
                "abs_diff_sec": closest["abs_diff_ns"] / 1e9,
            })

    if not results:
        print("No results.")
        return

    # sort for stable output
    results.sort(key=lambda r: (r["participant"].lower(), r["leg"]))

    print("\n================ FINAL SUMMARY TABLE ================\n")

    cols = [
        ("participant", "participant", 10),
        ("leg", "leg", 4),
        ("first_row_ns", "first_row_ns", 19),

        ("closest_frame_num", "closest#", 9),
        ("closest_frame_ts", "closest_ts", 19),

        ("first_frame_num", "first#", 7),
        ("first_frame_ts", "first_ts", 19),

        ("last_frame_num", "last#", 7),
        ("last_frame_ts", "last_ts", 19),
    ]
    print(format_table(results, cols))

    print("\n================ PATHS (per row) ================\n")
    for r in results:
        print(f"{r['participant']} leg{r['leg']}")
        print(f"  CSV          : {r['csv_path']}")
        print(f"  Frames dir   : {r['frames_dir']}")
        print(f"  Closest frame: {r['closest_frame_path']}")
        print(f"  Abs diff     : {r['abs_diff_ns']} ns ({r['abs_diff_sec']:.6f} sec)")
        print()

if __name__ == "__main__":
    main()