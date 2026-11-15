"""
create_fixed_split_preliminary.py

Creates the fixed participant-level split of the intermediate experimentation subset
(9 / 2 / 2 recordings) and writes it to `fixed_split.json`. All experiments of Sections
5.2-5.8 read this file.

Thesis step : Intermediate experimentation subset (Section 4.3.2)
Run on      : TU Wien dataLAB cluster (SLURM, NVIDIA A100)

What it does
1. Finds the recordings (via `dataset_win31.discover_recordings`) and counts the windows of
   each one (20 past + 1 + 5 future frames, stride 8).
2. Sorts the recordings by window count and divides them into four quartiles.
3. Runs a seeded random search (100,000 iterations): picks 3 / 3 / 3 / 4 recordings from the
   four quartiles and splits the 13 recordings randomly into 9 / 2 / 2.
4. Rejects candidates in which a participant appears in more than one set, and keeps the one
   whose window ratios are closest to 70 / 15 / 15 % (sum of squared differences).
5. Writes `fixed_split.json` with the recording names and the search settings, and prints
   the window counts and ratios.

Output
- `fixed_split.json` in the working directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_win31 import discover_recordings


BASE = "/share/e12507305/preprocessed/DERIVED_ET_IMU_head_mounted"
OUT_JSON = "fixed_split.json"

PAST_FRAMES = 20
FUTURE_FRAMES = 5
STRIDE = 8

ANGLE_COLS = ("roll_win", "pitch_win", "yaw_win")

RANDOM_SEED = 42
N_ITER = 100000


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


def split_score(train_windows: int, val_windows: int, test_windows: int) -> tuple[float, float, float, float]:
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


def has_participant_overlap(train_rows, val_rows, test_rows) -> bool:
    train_p = {x["participant"] for x in train_rows}
    val_p = {x["participant"] for x in val_rows}
    test_p = {x["participant"] for x in test_rows}

    return (
        not train_p.isdisjoint(val_p)
        or not train_p.isdisjoint(test_p)
        or not val_p.isdisjoint(test_p)
    )


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    items = discover_recordings(BASE, frames_subdir="results/frames_512")

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

    rows = sorted(rows, key=lambda x: x["n_windows"])
    quartiles = [list(q) for q in np.array_split(rows, 4)]

    subset_per_quartile = [3, 3, 3, 4]

    print("[Quartiles]", flush=True)
    for i, q in enumerate(quartiles, start=1):
        q_sorted = sorted(q, key=lambda x: x["n_windows"])
        print(
            f"  Q{i}: total={len(q_sorted)} | "
            f"range={q_sorted[0]['n_windows']}–{q_sorted[-1]['n_windows']}",
            flush=True,
        )

    best_solution = None
    best_score = float("inf")
    valid_candidates = 0

    print(f"\n[Randomized constrained search] iterations={N_ITER}", flush=True)

    for _ in range(N_ITER):
        selected = []

        # -------------------------------------------------
        # Select 3/3/3/4 recordings from Q1/Q2/Q3/Q4
        # -------------------------------------------------
        for q, n_select in zip(quartiles, subset_per_quartile):
            idx = rng.choice(len(q), size=n_select, replace=False)
            selected.extend([q[i] for i in idx])

        if len(selected) != 13:
            continue

        # -------------------------------------------------
        # Random 9/2/2 split of the selected recordings
        # -------------------------------------------------
        perm = rng.permutation(len(selected))

        train_rows = [selected[i] for i in perm[:9]]
        val_rows = [selected[i] for i in perm[9:11]]
        test_rows = [selected[i] for i in perm[11:13]]

        if has_participant_overlap(train_rows, val_rows, test_rows):
            continue

        train_windows = sum(x["n_windows"] for x in train_rows)
        val_windows = sum(x["n_windows"] for x in val_rows)
        test_windows = sum(x["n_windows"] for x in test_rows)

        score, train_ratio, val_ratio, test_ratio = split_score(
            train_windows,
            val_windows,
            test_windows,
        )

        valid_candidates += 1

        if score < best_score:
            best_score = score
            best_solution = {
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
                "train_windows": train_windows,
                "val_windows": val_windows,
                "test_windows": test_windows,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "score": score,
            }

    if best_solution is None:
        raise RuntimeError("No valid split found. Increase N_ITER or check constraints.")

    print(f"[Search completed] valid_candidates={valid_candidates}", flush=True)

    split_json = {
        "metadata": {
            "past_frames": PAST_FRAMES,
            "future_frames": FUTURE_FRAMES,
            "stride": STRIDE,
            "angle_cols": list(ANGLE_COLS),
            "selection": "fixed_length_stratified_quartile_randomized_constrained",
            "target_ratio": [0.70, 0.15, 0.15],
            "subset_per_quartile": subset_per_quartile,
            "random_seed": RANDOM_SEED,
            "n_iter": N_ITER,
            "score": best_solution["score"],
        },
        "train": sorted([x["recording_name"] for x in best_solution["train_rows"]]),
        "val": sorted([x["recording_name"] for x in best_solution["val_rows"]]),
        "test": sorted([x["recording_name"] for x in best_solution["test_rows"]]),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(split_json, f, indent=2)

    print("\n[Fixed split saved]", flush=True)
    print(f"Path: {Path(OUT_JSON).resolve()}", flush=True)

    print("\n[Final split]", flush=True)
    print(
        f"train recordings: 9 | windows={best_solution['train_windows']} | "
        f"ratio={best_solution['train_ratio']:.3f}",
        flush=True,
    )
    print(
        f"val recordings:   2 | windows={best_solution['val_windows']} | "
        f"ratio={best_solution['val_ratio']:.3f}",
        flush=True,
    )
    print(
        f"test recordings:  2 | windows={best_solution['test_windows']} | "
        f"ratio={best_solution['test_ratio']:.3f}",
        flush=True,
    )
    print(f"score={best_solution['score']:.8f}", flush=True)

    print("\ntrain IDs:", split_json["train"], flush=True)
    print("val IDs:", split_json["val"], flush=True)
    print("test IDs:", split_json["test"], flush=True)

    train_p = {x["participant"] for x in best_solution["train_rows"]}
    val_p = {x["participant"] for x in best_solution["val_rows"]}
    test_p = {x["participant"] for x in best_solution["test_rows"]}

    assert train_p.isdisjoint(val_p)
    assert train_p.isdisjoint(test_p)
    assert val_p.isdisjoint(test_p)

    assert len(split_json["train"]) == 9
    assert len(split_json["val"]) == 2
    assert len(split_json["test"]) == 2


if __name__ == "__main__":
    main()