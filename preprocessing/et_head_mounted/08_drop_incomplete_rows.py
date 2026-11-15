"""
08_drop_incomplete_rows.py

Removes head-IMU rows with missing angles or missing epoch timestamps (output of step 07).

Thesis step : Time synchronization (Section 3.2.4)
Dataset     : ET - Head Mounted
Run on      : Windows workstation at TU Wien

Purpose
- Clean head IMU epoch-aligned CSV files by removing rows with missing orientation
  or epoch timestamp values, ensuring downstream modeling/analysis operates only
  on fully valid samples.

What it does
- Recursively scans `--head_root` for all `.csv` files (typically outputs of the
  epoch-alignment step).
- For each CSV:
  1) Verifies that required columns exist:
         roll, pitch, yaw, head_epoch_ns
  2) Drops rows where any of these columns is NaN.
  3) Writes the cleaned result:
         - In-place if `--overwrite` is set.
         - Otherwise as `<original_name>_clean.csv`.

Inputs / Assumptions
- Input CSVs are per-recording head files already containing `head_epoch_ns`.
- Orientation columns (roll, pitch, yaw) are numeric and may contain NaNs from
  earlier processing steps.
- Only NaN-based filtering is applied (no range validation or outlier detection).

Outputs
- Cleaned CSV files preserving original column structure.
- Console summary reporting per-file removals and global totals.

Notes / Pitfalls
- This script loads entire CSVs into memory (no chunking), which is acceptable
  for moderate file sizes but may need refactoring for very large datasets.
- Row counts may decrease compared to previous steps (e.g., after offset application
  or timestamp repair), as this step enforces strict completeness.
"""


from pathlib import Path
import pandas as pd
import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_root", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    head_root = Path(args.head_root)

    csv_files = list(head_root.rglob("*.csv"))
    print(f"Found {len(csv_files)} head epoch CSVs")

    total_removed = 0
    total_rows = 0

    for i, csv_path in enumerate(csv_files, 1):
        df = pd.read_csv(csv_path)

        before = len(df)

        required_cols = ["roll", "pitch", "yaw", "head_epoch_ns"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"{csv_path} missing column '{col}'")

        df_clean = df.dropna(subset=required_cols)

        after = len(df_clean)
        removed = before - after

        total_removed += removed
        total_rows += before

        if args.overwrite:
            df_clean.to_csv(csv_path, index=False)
        else:
            out_path = csv_path.parent / (csv_path.stem + "_clean.csv")
            df_clean.to_csv(out_path, index=False)

        print(f"[{i}/{len(csv_files)}] {csv_path.name} | removed={removed}")

    print("\nDONE")
    print(f"Total rows before: {total_rows}")
    print(f"Total removed: {total_removed}")


if __name__ == "__main__":
    main()