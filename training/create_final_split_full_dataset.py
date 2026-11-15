"""
create_final_split_full_dataset.py

Creates the final participant-level split of all 38 recordings (26 / 6 / 6) and writes it
to `final_split.json`. The final model (Section 5.9) reads this file.

Thesis step : Final experimental split (Section 4.3.3)
Run on      : TU Wien dataLAB cluster (SLURM, NVIDIA A100)

What it does
1. Finds the recordings and counts the windows of each one with the final temporal context
   (40 past + 1 + 5 future frames, stride 8).
2. Extends the intermediate split: the 13 recordings of `fixed_split.json` keep their set,
   and every other recording of the same participants goes to the same set.
3. Runs a seeded random search (300,000 iterations) that assigns the remaining participants,
   with all their recordings, to train / val / test until the sets hold 26 / 6 / 6 recordings.
4. Keeps the candidate whose window ratios are closest to 70 / 15 / 15 %.
5. Writes `final_split.json`, checks that the intermediate split is contained in it and that
   no participant appears in two sets, and prints the window counts and ratios.

Input / output
- Input : `fixed_split.json` in the working directory.
- Output: `final_split.json` in the working directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_win31 import discover_recordings


BASE = "/share/e12507305/preprocessed/DERIVED_ET_IMU_head_mounted"

FIXED_SPLIT_JSON = "fixed_split.json"
OUT_JSON = "final_split.json"

PAST_FRAMES = 40
FUTURE_FRAMES = 5
STRIDE = 8

ANGLE_COLS = ("roll_win", "pitch_win", "yaw_win")

RANDOM_SEED = 42
N_ITER = 300000

TARGET_COUNTS = {
    "train": 26,
    "val": 6,
    "test": 6,
}


def count_windows_for_item(
    item,
    past_frames: int,
    future_frames: int,
    stride: int = 8,
    angle_cols=("roll_win", "pitch_win", "yaw_win"),
) -> int:
    df = pd.read_csv(item.paired_csv)

    required = ["frame_name", "timestamp_ns", *angle_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {item.paired_csv}: {missing}")

    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    df = df.dropna(subset=list(angle_cols)).reset_index(drop=True)

    seq_len = past_frames + 1 + future_frames

    if len(df) < seq_len:
        return 0

    max_start = len(df) - seq_len
    return int((max_start // stride) + 1)


def split_score(train_windows: int, val_windows: int, test_windows: int):
    total = train_windows + val_windows + test_windows

    train_ratio = train_windows / total
    val_ratio = val_windows / total
    test_ratio = test_windows / total

    score = (
        (train_ratio - 0.70) ** 2
        + (val_ratio - 0.15) ** 2
        + (test_ratio - 0.15) ** 2
    )

    return score, train_ratio, val_ratio, test_ratio


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    items = discover_recordings(BASE, frames_subdir="results/frames_512")

    with open(FIXED_SPLIT_JSON, "r") as f:
        fixed_split = json.load(f)

    fixed_train = set(fixed_split["train"])
    fixed_val = set(fixed_split["val"])
    fixed_test = set(fixed_split["test"])
    fixed_all = fixed_train | fixed_val | fixed_test

    rows = []

    for item in items:
        n_windows = count_windows_for_item(
            item=item,
            past_frames=PAST_FRAMES,
            future_frames=FUTURE_FRAMES,
            stride=STRIDE,
            angle_cols=ANGLE_COLS,
        )

        if n_windows > 0:
            rows.append(
                {
                    "recording_name": item.recording_name,
                    "participant": item.participant,
                    "trial": item.trial,
                    "n_windows": n_windows,
                }
            )

    if len(rows) != 38:
        print(f"[WARN] Expected 38 valid recordings, found {len(rows)}")

    row_by_name = {x["recording_name"]: x for x in rows}

    missing_fixed = sorted([x for x in fixed_all if x not in row_by_name])
    if missing_fixed:
        raise ValueError(f"Fixed split recordings not found in final rows: {missing_fixed}")

    # -----------------------------------------------------
    # Participant constraints from the intermediate split
    # -----------------------------------------------------
    fixed_subset_by_participant = {}

    for split_name, names in [
        ("train", fixed_train),
        ("val", fixed_val),
        ("test", fixed_test),
    ]:
        for name in names:
            participant = row_by_name[name]["participant"]

            if participant in fixed_subset_by_participant:
                old = fixed_subset_by_participant[participant]
                if old != split_name:
                    raise ValueError(
                        f"Participant {participant} appears in both {old} and {split_name}"
                    )

            fixed_subset_by_participant[participant] = split_name

    # -----------------------------------------------------
    # Forced rows:
    # 1. the original 13 recordings
    # 2. any additional recording from a participant already fixed
    # -----------------------------------------------------
    forced = {
        "train": [],
        "val": [],
        "test": [],
    }

    free_rows = []

    for row in rows:
        participant = row["participant"]

        if participant in fixed_subset_by_participant:
            split_name = fixed_subset_by_participant[participant]
            forced[split_name].append(row)
        else:
            free_rows.append(row)

    # Group free rows by participant
    free_groups = {}
    for row in free_rows:
        free_groups.setdefault(row["participant"], []).append(row)

    free_participants = sorted(free_groups.keys())

    print("[Forced assignments]", flush=True)
    for split_name in ["train", "val", "test"]:
        print(
            f"  {split_name}: recordings={len(forced[split_name])} | "
            f"participants={len({x['participant'] for x in forced[split_name]})}",
            flush=True,
        )

    print(f"\n[Free participants to assign]: {len(free_participants)}", flush=True)

    best_solution = None
    best_score = float("inf")
    valid_candidates = 0

    for _ in range(N_ITER):
        candidate = {
            "train": list(forced["train"]),
            "val": list(forced["val"]),
            "test": list(forced["test"]),
        }

        shuffled_participants = list(free_participants)
        rng.shuffle(shuffled_participants)

        for participant in shuffled_participants:
            group = free_groups[participant]
            possible_splits = []

            for split_name in ["train", "val", "test"]:
                if len(candidate[split_name]) + len(group) <= TARGET_COUNTS[split_name]:
                    possible_splits.append(split_name)

            if not possible_splits:
                break

            chosen_split = rng.choice(possible_splits)
            candidate[chosen_split].extend(group)

        if not all(len(candidate[s]) == TARGET_COUNTS[s] for s in ["train", "val", "test"]):
            continue

        train_p = {x["participant"] for x in candidate["train"]}
        val_p = {x["participant"] for x in candidate["val"]}
        test_p = {x["participant"] for x in candidate["test"]}

        if (
            not train_p.isdisjoint(val_p)
            or not train_p.isdisjoint(test_p)
            or not val_p.isdisjoint(test_p)
        ):
            continue

        train_windows = sum(x["n_windows"] for x in candidate["train"])
        val_windows = sum(x["n_windows"] for x in candidate["val"])
        test_windows = sum(x["n_windows"] for x in candidate["test"])

        score, train_ratio, val_ratio, test_ratio = split_score(
            train_windows,
            val_windows,
            test_windows,
        )

        valid_candidates += 1

        if score < best_score:
            best_score = score
            best_solution = {
                "split": candidate,
                "train_windows": train_windows,
                "val_windows": val_windows,
                "test_windows": test_windows,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "score": score,
            }

    if best_solution is None:
        raise RuntimeError("No valid final split found. Increase N_ITER or inspect constraints.")

    split = best_solution["split"]

    # Safety checks: original 9/2/2 retained in same subsets
    assert fixed_train.issubset({x["recording_name"] for x in split["train"]})
    assert fixed_val.issubset({x["recording_name"] for x in split["val"]})
    assert fixed_test.issubset({x["recording_name"] for x in split["test"]})

    final_json = {
        "metadata": {
            "past_frames": PAST_FRAMES,
            "future_frames": FUTURE_FRAMES,
            "stride": STRIDE,
            "angle_cols": list(ANGLE_COLS),
            "selection": "final_extension_of_fixed_intermediate_split_participant_constrained",
            "target_ratio": [0.70, 0.15, 0.15],
            "target_recording_counts": TARGET_COUNTS,
            "fixed_split_json": FIXED_SPLIT_JSON,
            "random_seed": RANDOM_SEED,
            "n_iter": N_ITER,
            "valid_candidates": valid_candidates,
            "score": best_solution["score"],
        },
        "train": sorted([x["recording_name"] for x in split["train"]]),
        "val": sorted([x["recording_name"] for x in split["val"]]),
        "test": sorted([x["recording_name"] for x in split["test"]]),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(final_json, f, indent=2)

    print("\n[Final split saved]", flush=True)
    print(f"Path: {Path(OUT_JSON).resolve()}", flush=True)

    print("\n[Final split statistics]", flush=True)
    print(
        f"train recordings: {len(final_json['train'])} | "
        f"windows={best_solution['train_windows']} | "
        f"ratio={best_solution['train_ratio']:.3f}",
        flush=True,
    )
    print(
        f"val recordings:   {len(final_json['val'])} | "
        f"windows={best_solution['val_windows']} | "
        f"ratio={best_solution['val_ratio']:.3f}",
        flush=True,
    )
    print(
        f"test recordings:  {len(final_json['test'])} | "
        f"windows={best_solution['test_windows']} | "
        f"ratio={best_solution['test_ratio']:.3f}",
        flush=True,
    )
    print(f"score={best_solution['score']:.8f}", flush=True)

    print("\ntrain IDs:", final_json["train"], flush=True)
    print("val IDs:", final_json["val"], flush=True)
    print("test IDs:", final_json["test"], flush=True)

    assert len(final_json["train"]) == 26
    assert len(final_json["val"]) == 6
    assert len(final_json["test"]) == 6


if __name__ == "__main__":
    main()