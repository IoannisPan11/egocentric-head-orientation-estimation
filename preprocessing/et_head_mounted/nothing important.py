import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================

CSV_PATH = Path(r"D:\john.panagiotopoulos\DERIVED_ET_IMU_upper_body\allinone_offset_applied\p16\leg_2\video1_all_in_one.csv")

# =========================
# MAIN
# =========================

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"File not found:\n{CSV_PATH}")

    print(f"\nLoading CSV:\n{CSV_PATH}\n")

    # Read CSV
    df = pd.read_csv(CSV_PATH)

    print("=" * 60)
    print("BASIC INFO")
    print("=" * 60)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\n" + "=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:02d}. {col}")

    print("\n" + "=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)
    print(df.head())

    print("\nDone.")


if __name__ == "__main__":
    main()