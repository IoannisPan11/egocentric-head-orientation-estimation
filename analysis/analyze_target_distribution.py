"""
analyze_target_distribution.py

Describes the frame-to-frame orientation changes that the model predicts, and the error of a
no-change prediction that assumes zero head motion. The results are printed to the console.

Thesis step : Data splitting and reference performance (Sections 4.3.2 and 5.1)
Thesis      : Table 4 (preliminary subset vs. full dataset), Table 7 (no-change Mean RMSE)
Run on      : TU Wien dataLAB cluster
Launcher    : `run_target_distribution.sh`

What it does
- Uses the baseline configuration (Section 4.4.1): nearest-neighbour angles (`roll_nn`,
  `pitch_nn`, `yaw_nn` of `paired_frames11.csv`) and windows of 20 past + 1 + 5 future
  frames with stride 8, as the training scripts build them.
- For every window, takes the change in roll, pitch and yaw between the previous and the
  current frame, wrapped to [-180, 180) degrees.
- For each angle, prints the mean and median absolute change, the standard deviation, the
  maximum absolute change and the RMSE of predicting zero change. The mean of the three
  RMSE values is the no-change Mean RMSE.
- Repeats this for the whole dataset (all recordings under `BASE`), for each subset of the
  preliminary split (`fixed_split.json`) and for the preliminary subset as a whole, and for
  each subset of the final split (`final_split.json`).

Thesis values
- Table 4: mean absolute change and standard deviation of "PRELIMINARY ALL" and
  "WHOLE DATASET".
- Table 7: no-change Mean RMSE of "PRELIMINARY VAL", "FINAL VAL" and "FINAL TEST".

How to run
- Needs the output of preprocessing step 09 under `BASE` (the data are not public).
- From the `training/` folder, where the split files are:
    python ../analysis/analyze_target_distribution.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path("/share/e12507305/preprocessed/DERIVED_ET_IMU_head_mounted")

PAST_FRAMES = 20
FUTURE_FRAMES = 5
SEQ_LEN = PAST_FRAMES + 1 + FUTURE_FRAMES
STRIDE = 8

ANGLE_COLS = ["roll_nn", "pitch_nn", "yaw_nn"]
TS_COL = "timestamp_ns"


def wrap_angle_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def collect_deltas(csv_files):
    all_deltas = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df = df.sort_values(TS_COL).reset_index(drop=True)
        df = df.dropna(subset=ANGLE_COLS).reset_index(drop=True)

        if len(df) < SEQ_LEN:
            continue

        max_start = len(df) - SEQ_LEN

        for start in range(0, max_start + 1, STRIDE):
            prev_idx = start + PAST_FRAMES - 1
            curr_idx = start + PAST_FRAMES

            prev = df.loc[prev_idx, ANGLE_COLS].astype(float)
            curr = df.loc[curr_idx, ANGLE_COLS].astype(float)

            d_roll = wrap_angle_deg(curr["roll_nn"] - prev["roll_nn"])
            d_pitch = wrap_angle_deg(curr["pitch_nn"] - prev["pitch_nn"])
            d_yaw = wrap_angle_deg(curr["yaw_nn"] - prev["yaw_nn"])

            all_deltas.append([d_roll, d_pitch, d_yaw])

    return np.array(all_deltas)


def print_stats(name, deltas):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Total windows: {len(deltas)}")

    axes = ["roll", "pitch", "yaw"]
    zero_rmses = []

    for i, axis in enumerate(axes):
        x = deltas[:, i]
        abs_x = np.abs(x)
        zero_rmse = np.sqrt(np.mean(x ** 2))
        zero_rmses.append(zero_rmse)

        print(f"\n{axis.upper()}")
        print(f"mean abs   : {abs_x.mean():.4f} deg")
        print(f"median abs : {np.median(abs_x):.4f} deg")
        print(f"std        : {x.std():.4f} deg")
        print(f"zero RMSE  : {zero_rmse:.4f} deg")
        print(f"max abs    : {abs_x.max():.4f} deg")

    print(f"\nZERO-MOTION BASELINE mean RMSE: {np.mean(zero_rmses):.4f} deg")


def recording_names_to_csv(recording_names):
    csv_files = []

    for rec_name in recording_names:
        participant, trial = rec_name.split("/")
        csv_path = BASE / participant / trial / "paired_frames11.csv"

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        csv_files.append(csv_path)

    return csv_files


# =========================================================
# 1) Whole dataset (38 recordings)
# =========================================================
all_csv_files = sorted(BASE.glob("*/*/paired_frames11.csv"))

all_deltas = collect_deltas(all_csv_files)
print_stats("WHOLE DATASET TARGET DISTRIBUTION", all_deltas)

print(f"Whole dataset CSV files found: {len(all_csv_files)}")


# =========================================================
# 2) Preliminary fixed split (13 recordings)
# =========================================================
with open("fixed_split.json", "r") as f:
    prelim_split_ids = json.load(f)

prelim_all_csv_files = []

for split_name in ["train", "val", "test"]:
    split_csv_files = recording_names_to_csv(prelim_split_ids[split_name])
    prelim_all_csv_files.extend(split_csv_files)

    split_deltas = collect_deltas(split_csv_files)
    print_stats(f"PRELIMINARY {split_name.upper()} TARGET DISTRIBUTION", split_deltas)
    print(f"\nPreliminary {split_name.upper()} CSV files found: {len(split_csv_files)}")

prelim_all_deltas = collect_deltas(prelim_all_csv_files)
print_stats("PRELIMINARY ALL TARGET DISTRIBUTION", prelim_all_deltas)
print(f"\nPreliminary ALL CSV files found: {len(prelim_all_csv_files)}")


# =========================================================
# 3) Final fixed split (38 recordings)
# =========================================================
with open("final_split.json", "r") as f:
    final_split_ids = json.load(f)

for split_name in ["train", "val", "test"]:
    split_csv_files = recording_names_to_csv(final_split_ids[split_name])

    split_deltas = collect_deltas(split_csv_files)
    print_stats(f"FINAL {split_name.upper()} TARGET DISTRIBUTION", split_deltas)

    print(f"\nFinal {split_name.upper()} CSV files found: {len(split_csv_files)}")
