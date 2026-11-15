"""
inspect_imu_columns.py

Diagnostic: quick schema inspection of one upper-body IMU all-in-one CSV. Not part of the pipeline.

Dataset     : ET - Upper Body (explored, not used for the final model)
Run on      : Windows workstation at TU Wien

Purpose
- Explore and validate the column structure of a merged upper-body IMU export
  (e.g. leg_1_allInOne.csv) before downstream processing or feature selection.

What it does
- Loads a single CSV file into a pandas DataFrame.
- Prints:
    1) Total number of columns.
    2) Full list of column names (raw schema view).
    3) Columns likely related to head kinematics (name contains "Head" or "C1").
    4) Quaternion orientation columns for the head (prefix "orie_Head").
    5) Angular velocity columns for the head (prefix "angV_Head").
    6) Basic dataset dimensions (rows × columns).

Intended use
- Manual inspection / exploratory step to:
    - Confirm expected column naming conventions.
    - Identify which signals correspond to head orientation and motion.
    - Decide which fields should be kept for synchronization or modeling.
    - Detect schema inconsistencies across participants or legs.

Notes
- This script performs no transformation, filtering, or writing.
- It loads the entire CSV into memory; suitable for inspection but not optimized
  for very large files.
- String-based filtering assumes consistent naming (e.g., "orie_Head_*").
"""


import pandas as pd

CSV = r"E:\john\ET-IMU-and-upper-body\imu\p20\leg_1_allInOne.csv"

df = pd.read_csv(CSV)

print("\n===== TOTAL COLUMNS =====")
print(len(df.columns))

print("\n===== ALL COLUMNS =====")
for c in df.columns:
    print(c)

print("\n===== HEAD-RELATED COLUMNS =====")
for c in df.columns:
    if "Head" in c or "C1" in c:
        print(c)

print("\n===== ORIENTATION (quaternion) COLUMNS =====")
for c in df.columns:
    if c.startswith("orie_Head"):
        print(c)

print("\n===== ANGULAR VELOCITY COLUMNS =====")
for c in df.columns:
    if c.startswith("angV_Head"):
        print(c)

print("\n===== BASIC INFO =====")
print("Rows:", len(df))
print("Columns:", df.shape[1])