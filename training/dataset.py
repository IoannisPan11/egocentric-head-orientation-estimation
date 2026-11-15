"""
dataset.py

Dataset module used by the two Smooth L1 runs of the loss-function experiment (Section 5.2),
which train on the nearest-neighbour (NN) labels, and by `inspect_recording_windows.py`.

Thesis step : Problem formulation and targets (Sections 4.1, 4.4.2)
Run on      : TU Wien dataLAB cluster (SLURM, NVIDIA A100)

Notes
- The code is identical to `dataset_win11.py`: it reads `paired_frames11.csv`. The NN runs
  pass angle_cols=("roll_nn", "pitch_nn", "yaw_nn"), so they use the nearest-sample
  columns of that file, which do not depend on the window size.

Contents
- `RecordingItem`: participant, trial, recording name, paired CSV and frame folder of a recording.
- `discover_recordings`: finds every `<participant>/<trial>/paired_frames11.csv` under the base
  folder and the matching frame folder under `results/frames_512/`.
- `build_participant_split`: a random participant-level split (not used by the experiments, which
  read the fixed split files).
- `HeadMountedSequenceDataset`: sliding windows of past_frames + 1 + future_frames consecutive
  frames with a given stride. Rows with missing angles are dropped.

Target of each window
- The change between the previous frame and the current frame (the last past frame and the
  frame after it): [delta_roll / 180, delta_pitch / 180, sin(delta_yaw), cos(delta_yaw)].
- The yaw change is wrapped to [-180, 180) before the sin/cos encoding.

Each sample is a dict with the frame tensor [T, 3, H, W], the target and metadata
(recording, frame names and timestamps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

@dataclass
class RecordingItem:
    participant: str
    trial: str
    recording_name: str
    paired_csv: Path
    frames_dir: Path

def discover_recordings(
    base_dir: str | Path,
    frames_subdir: str = "results/frames_512",
) -> List[RecordingItem]:
    """
    Discover valid recordings.

    Expected structure:
        BASE/
            004_LS47/trial_1_fcn/paired_frames*.csv
            ...
            results/frames_512/004_LS47_trial_1_fcn_ET_.../frame_xxxxx.png
    """
    base = Path(base_dir)
    frames_root = base / frames_subdir

    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_root}")

    csv_files = sorted(base.glob("*/*/paired_frames11.csv"))
    items: List[RecordingItem] = []

    for csv_path in csv_files:
        participant = csv_path.parent.parent.name
        trial = csv_path.parent.name
        recording_name = f"{participant}/{trial}"

        matches = sorted(frames_root.glob(f"{participant}_{trial}_*"))
        if len(matches) == 0:
            print(f"[WARN] No matching frame directory for {recording_name}")
            continue
        if len(matches) > 1:
            print(f"[WARN] Multiple frame directories for {recording_name}, using first one")

        items.append(
            RecordingItem(
                participant=participant,
                trial=trial,
                recording_name=recording_name,
                paired_csv=csv_path,
                frames_dir=matches[0],
            )
        )

    if not items:
        raise RuntimeError("No valid recordings found.")

    print(f"[Discovery] valid recordings found: {len(items)}")
    return items

def build_participant_split(
    items: List[RecordingItem],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    subset_ratio: float = 1.0,
    seed: int = 42,
) -> Dict[str, List[RecordingItem]]:
    """
    Participant-level split.

    subset_ratio < 1.0 means use only a subset of participants
    for quick experiments, e.g. 0.30 ~ 30% of total participants.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    participants = sorted({x.participant for x in items})
    rng = random.Random(seed)
    rng.shuffle(participants)

    if subset_ratio < 1.0:
        n_subset = max(1, round(len(participants) * subset_ratio))
        participants = participants[:n_subset]

    n_total = len(participants)
    n_train = max(1, round(n_total * train_ratio))
    n_val = max(1, round(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1

    train_participants = set(participants[:n_train])
    val_participants = set(participants[n_train:n_train + n_val])
    test_participants = set(participants[n_train + n_val:])

    split = {
        "train": [x for x in items if x.participant in train_participants],
        "val": [x for x in items if x.participant in val_participants],
        "test": [x for x in items if x.participant in test_participants],
    }

    print("[Split]")
    print(f"  total participants used: {n_total}")
    print(f"  train participants: {len(train_participants)}")
    print(f"  val participants:   {len(val_participants)}")
    print(f"  test participants:  {len(test_participants)}")
    print(f"  train recordings: {len(split['train'])}")
    print(f"  val recordings:   {len(split['val'])}")
    print(f"  test recordings:  {len(split['test'])}")

    print("  train IDs:", sorted(train_participants))
    print("  val IDs:  ", sorted(val_participants))
    print("  test IDs: ", sorted(test_participants))

    return split

class HeadMountedSequenceDataset(Dataset):
    """
    Sequence dataset for head-mounted ET frames.

    Target:
        delta = current_angles - previous_angles
        using roll_nn, pitch_nn, yaw_nn or roll_win, pitch_win, yaw_win
        
    Sequence layout:
        [past_frames ... current_frame ... future_frames]
    """

    def __init__(
        self,
        recordings: List[RecordingItem],
        past_frames: int = 20,
        future_frames: int = 5,
        stride: int = 4,
        frame_name_col: str = "frame_name",
        ts_col: str = "timestamp_ns",
        angle_cols: Tuple[str, str, str] = ("roll_win", "pitch_win", "yaw_win"),
        transform=None,
        verify_files: bool = True,
        max_missing_ratio: float = 0.0,
        divide_by_255: bool = True,
        max_windows: Optional[int] = None,
        subset_recordings: Optional[int] = None,
    ):
        self.recordings = recordings
        if subset_recordings is not None:
            self.recordings = self.recordings[:subset_recordings]

        self.past_frames = int(past_frames)
        self.future_frames = int(future_frames)
        self.seq_len = self.past_frames + 1 + self.future_frames
        self.stride = int(stride)

        self.frame_name_col = frame_name_col
        self.ts_col = ts_col
        self.angle_cols = angle_cols

        self.transform = transform
        self.verify_files = verify_files
        self.max_missing_ratio = float(max_missing_ratio)
        self.divide_by_255 = divide_by_255
        self.max_windows = max_windows

        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.index: List[Dict] = []

        self._build_index()

    def _build_index(self) -> None:
        total_added = 0

        for rec in self.recordings:
            df = pd.read_csv(rec.paired_csv)

            required = [self.frame_name_col, self.ts_col, *self.angle_cols]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in {rec.paired_csv}: {missing}")

            df = df.sort_values(self.ts_col).reset_index(drop=True)
            df = df.dropna(subset=list(self.angle_cols)).reset_index(drop=True)

            if len(df) < self.seq_len:
                print(f"[WARN] Too few rows for sequence length in {rec.paired_csv}")
                continue

            self.dataframes[str(rec.paired_csv)] = df
            max_start = len(df) - self.seq_len

            for start in range(0, max_start + 1, self.stride):
                sl = df.iloc[start:start + self.seq_len]
                frame_names = sl[self.frame_name_col].tolist()

                if self.verify_files:
                    missing_count = sum(
                        1 for name in frame_names if not (rec.frames_dir / name).exists()
                    )
                    if missing_count > 0:
                        missing_ratio = missing_count / len(frame_names)
                        if missing_ratio > self.max_missing_ratio:
                            continue

                self.index.append(
                    {
                        "paired_csv": str(rec.paired_csv),
                        "frames_dir": str(rec.frames_dir),
                        "participant": rec.participant,
                        "trial": rec.trial,
                        "recording_name": rec.recording_name,
                        "start": int(start),
                    }
                )

                total_added += 1
                if self.max_windows is not None and total_added >= self.max_windows:
                    break

            if self.max_windows is not None and total_added >= self.max_windows:
                break

        if not self.index:
            raise RuntimeError("Dataset index is empty.")

        print(
            f"[Dataset] windows={len(self.index)} | "
            f"past={self.past_frames} current=1 future={self.future_frames} | "
            f"seq_len={self.seq_len} | stride={self.stride}"
        )
        print(f"[Dataset] verify_files={self.verify_files} | max_windows={self.max_windows}")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, object]:
        info = self.index[idx]

        paired_csv = info["paired_csv"]
        frames_dir = Path(info["frames_dir"])
        start = info["start"]

        df = self.dataframes[paired_csv]
        sl = df.iloc[start:start + self.seq_len]

        current_idx = self.past_frames
        prev_idx = current_idx - 1

        current_row = sl.iloc[current_idx]
        prev_row = sl.iloc[prev_idx]

        current_angles = torch.tensor(
            [float(current_row[c]) for c in self.angle_cols],
            dtype=torch.float32
        )
        prev_angles = torch.tensor(
            [float(prev_row[c]) for c in self.angle_cols],
            dtype=torch.float32
        )

        delta_angles = current_angles - prev_angles
        
        delta_roll = delta_angles[0] / 180.0
        delta_pitch = delta_angles[1] / 180.0
        
        # wrap yaw difference to [-180, 180]
        delta_yaw = ((delta_angles[2] + 180.0) % 360.0) - 180.0
        
        delta_yaw_rad = torch.deg2rad(delta_yaw)
        
        target = torch.stack([
            delta_roll,
            delta_pitch,
            torch.sin(delta_yaw_rad),
            torch.cos(delta_yaw_rad),
        ], dim=0)

        frames = []
        used_names = []

        for name in sl[self.frame_name_col].tolist():
            img_path = frames_dir / name

            if not img_path.exists():
                raise FileNotFoundError(img_path)

            img = self._load_rgb(img_path)

            if self.transform is not None:
                x = self.transform(img)
            else:
                arr = np.array(img, dtype=np.uint8, copy=True)
                x = torch.from_numpy(arr).permute(2, 0, 1).float()
                if self.divide_by_255:
                    x = x / 255.0

            frames.append(x)
            used_names.append(name)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        return {
            "frames": frames,
            "target": target,
            "delta_angles_raw": delta_angles,
            "current_angles": current_angles,
            "prev_angles": prev_angles,
            "meta": {
                "participant": info["participant"],
                "trial": info["trial"],
                "recording_name": info["recording_name"],
                "paired_csv": paired_csv,
                "frames_dir": str(frames_dir),
                "start": start,
                "t0": int(sl[self.ts_col].iloc[0]),
                "t_current": int(current_row[self.ts_col]),
                "t1": int(sl[self.ts_col].iloc[-1]),
                "frame_names": used_names,
            },
        }