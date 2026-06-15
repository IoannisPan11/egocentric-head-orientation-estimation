"""
06_sync_head_imu_to_et_with_offsets.py

Purpose
- Synchronize head-mounted IMU time to the ET/world timeline using per-recording offsets from
  `offsets_master.csv`, and export a lightweight per-recording CSV (timestamp + Euler angles).

Core definition (sign convention)
- The offsets are defined as:
    offset_ns = t_et - t_head
  Therefore head timestamps aligned to the ET timeline are computed as:
    t_head_sync = t_head + offset_ns

What it does
- Builds an offset lookup table from `--offsets_csv` by normalizing `rec_path` strings and parsing
  `offset_ns` as int64 (only finite offsets are kept).
- Recursively scans `--root` for head IMU files:
    *_AHS_ON_merged_fixed.parquet
  and extracts a recording key from the path structure:
    <LS_dir>/<trial_dir>   (e.g., "004_LS47/trial_1_fcn")
- Resolves the correct offset for each recording using a robust matching strategy:
    1) exact match
    2) contains match (rec_key is substring of rec_path)
    3) reverse-contains match (rare fallback)
  with tie-breaking based on minimal path-length difference.
- Streams each parquet in Arrow batches (`--batch_size`) and writes a CSV containing:
    timestamp_ns, roll, pitch, yaw
  where:
    timestamp_ns = int(ts_ns_best) + offset_ns
- Drops rows where `ts_ns_best` is not finite (NaN/inf) before applying the offset.

Inputs / Assumptions
- Input parquet files come from the earlier pipeline steps and contain:
    ts_ns_best, Roll, Pitch, Yaw
  (`ts_ns_best` should already be repaired/created by the "*_merged_fixed" step if UTC was missing).
- `offsets_master.csv` must contain columns: rec_path, offset_ns.
- Directory layout under `--root` is expected to start with:
    <participant_LS>/<trial>/...

Outputs
- One synced CSV per parquet, written under `--out_root` mirroring the recording key:
    <out_root>/<LS>/<trial>/<stem>_synced.csv
  Overwrite behavior:
    - default: skip if output exists
    - `--overwrite`: regenerate output CSVs
- Optional `--dry_run` to print planned actions without writing.

Notes / Pitfalls
- If `ts_ns_best` is based on a relative timeline (SampleTimeFine fallback), the resulting `timestamp_ns`
  is only meaningful after offset application; ensure offsets were computed against the same reference.
- Row counts can decrease due to dropping non-finite `ts_ns_best` values.
"""


import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def fmt_s(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    s2 = s - 60 * m
    return f"{m}m{s2:04.1f}s"


def normalize_pathlike(x: str) -> str:
    """
    Normalize any path-like string to forward slashes, no trailing slash.
    Strips quotes/spaces.
    """
    if x is None:
        return ""
    s = str(x).strip().strip('"').strip("'")
    s = s.replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    s = s.rstrip("/")
    return s


def find_parquets(root: Path):
    return sorted(root.rglob("*_AHS_ON_merged_fixed.parquet"))


def extract_recording_rel_path(file: Path, root: Path) -> str:
    """
    From:
      root/004_LS47/trial_1_fcn/IMU/parquet_parts/MT_..._merged_fixed.parquet
    Return normalized:
      004_LS47/trial_1_fcn
    """
    rel = file.relative_to(root)
    parts = rel.parts
    if len(parts) < 2:
        raise RuntimeError(f"Unexpected path structure: {rel}")
    return normalize_pathlike(str(Path(parts[0]) / parts[1]))


def build_offset_table(offsets_csv: str) -> pd.DataFrame:
    df = pd.read_csv(offsets_csv)

    need = {"rec_path", "offset_ns"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"offsets_master.csv missing columns: {sorted(missing)}")

    df = df.copy()
    df["rec_path_norm"] = df["rec_path"].apply(normalize_pathlike)
    df["offset_ns"] = pd.to_numeric(df["offset_ns"], errors="coerce")

    df_valid = df[
        (df["rec_path_norm"].notna())
        & (df["rec_path_norm"].str.len() > 0)
        & np.isfinite(df["offset_ns"].to_numpy())
    ].copy()

    if df_valid.empty:
        raise RuntimeError("No valid (finite) offset_ns rows found in offsets_master.csv")

    # IMPORTANT:
    # do NOT drop duplicates aggressively here, because the same rec_key might appear as:
    # 004_LS47/trial_1_fcn
    # 004_LS47/trial_1_fcn/IMU
    # ... and we want the "best" one later.
    df_valid["offset_ns"] = df_valid["offset_ns"].round().astype("int64")

    print(f"Offsets rows: total={len(df)} | valid(rec_path+finite offset_ns)={len(df_valid)}")
    return df_valid[["rec_path_norm", "offset_ns"]].reset_index(drop=True)


def lookup_offset_ns(rec_key_norm: str, offsets_df: pd.DataFrame) -> tuple[int | None, str | None]:
    """
    Matching strategy (in order):
    1) exact: rec_path_norm == rec_key_norm
    2) contains: rec_key_norm is substring of rec_path_norm  (covers rec_path with extra subfolders)
    3) reverse-contains: rec_path_norm is substring of rec_key_norm (rare, but safe)

    If multiple matches:
      - Prefer exact
      - Else prefer the match with minimal extra length (closest / most specific)
        i.e. minimize abs(len(rec_path_norm) - len(rec_key_norm))
      - Tie-breaker: shorter rec_path_norm (less junk)
    Returns (offset_ns, matched_rec_path_norm)
    """
    # 1) exact
    m = offsets_df[offsets_df["rec_path_norm"] == rec_key_norm]
    if len(m) >= 1:
        # if multiple exact rows, take first
        return int(m["offset_ns"].iloc[0]), str(m["rec_path_norm"].iloc[0])

    # 2) contains
    m = offsets_df[offsets_df["rec_path_norm"].str.contains(rec_key_norm, regex=False)]
    if len(m) >= 1:
        mm = m.copy()
        mm["_diff"] = (mm["rec_path_norm"].str.len() - len(rec_key_norm)).abs()
        mm["_len"] = mm["rec_path_norm"].str.len()
        mm = mm.sort_values(by=["_diff", "_len"], ascending=[True, True])
        return int(mm["offset_ns"].iloc[0]), str(mm["rec_path_norm"].iloc[0])

    # 3) reverse-contains (very unlikely, but ok)
    m = offsets_df[offsets_df["rec_path_norm"].apply(lambda x: x in rec_key_norm)]
    if len(m) >= 1:
        mm = m.copy()
        mm["_diff"] = (len(rec_key_norm) - mm["rec_path_norm"].str.len()).abs()
        mm["_len"] = mm["rec_path_norm"].str.len()
        mm = mm.sort_values(by=["_diff", "_len"], ascending=[True, False])
        return int(mm["offset_ns"].iloc[0]), str(mm["rec_path_norm"].iloc[0])

    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--offsets_csv", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--batch_size", type=int, default=300000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--every_rows_log", type=int, default=1_000_000)
    ap.add_argument("--every", type=int, default=1, help="progress every N files")
    ap.add_argument("--debug_keys", action="store_true", help="print example keys from offsets CSV")
    ap.add_argument("--debug_fail_examples", type=int, default=5, help="how many fails to print with suggestions")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)

    offsets_df = build_offset_table(args.offsets_csv)

    if args.debug_keys:
        print("\nExample rec_path_norm from CSV (first 25):")
        for x in offsets_df["rec_path_norm"].head(25).tolist():
            print("  ", x)
        print()

    files = find_parquets(root)
    print(f"Found {len(files)} head merged_fixed parquets under: {root}")

    t0 = time.time()
    ok = 0
    skip = 0
    fail = 0
    shown_fail = 0

    for i, p in enumerate(files, 1):
        rec_key = extract_recording_rel_path(p, root)  # e.g. 004_LS47/trial_1_fcn

        offset_ns, matched = lookup_offset_ns(rec_key, offsets_df)
        if offset_ns is None:
            fail += 1
            print(f"[{i}/{len(files)}] FAIL: no offset for rec_key='{rec_key}' | file={p.name}")
            if shown_fail < args.debug_fail_examples:
                shown_fail += 1
                # show a few candidates that contain LSxx or trial token
                token = rec_key.split("/")[0]  # e.g. 004_LS47
                cand = offsets_df[offsets_df["rec_path_norm"].str.contains(token, regex=False)].head(8)
                if len(cand):
                    print("    candidates in CSV containing", token, ":")
                    for c in cand["rec_path_norm"].tolist():
                        print("      -", c)
            continue

        out_csv = out_root / rec_key / (p.stem.replace("_merged_fixed", "") + "_synced.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        if out_csv.exists() and not args.overwrite:
            skip += 1
            if i % args.every == 0:
                elapsed = time.time() - t0
                rate = i / max(1e-9, elapsed)
                eta = (len(files) - i) / max(1e-9, rate)
                print(f"[{i}/{len(files)}] OK={ok} SKIP={skip} FAIL={fail} | elapsed={fmt_s(elapsed)} ETA={fmt_s(eta)} -> SKIP exists")
            continue

        if args.dry_run:
            ok += 1
            if i % args.every == 0:
                elapsed = time.time() - t0
                rate = i / max(1e-9, elapsed)
                eta = (len(files) - i) / max(1e-9, rate)
                print(f"[{i}/{len(files)}] DRY OK | {p.name} -> {out_csv} | offset_ns={offset_ns} matched='{matched}' | elapsed={fmt_s(elapsed)} ETA={fmt_s(eta)}")
            continue

        pf = pq.ParquetFile(p)
        rows_written = 0
        t_file0 = time.time()

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ns", "roll", "pitch", "yaw"])

            for batch in pf.iter_batches(
                batch_size=args.batch_size,
                columns=["ts_ns_best", "Roll", "Pitch", "Yaw"],
            ):
                dfb = batch.to_pandas()

                ts = pd.to_numeric(dfb["ts_ns_best"], errors="coerce").to_numpy(dtype="float64")
                good = np.isfinite(ts)
                if not np.all(good):
                    dfb = dfb.loc[good].copy()
                    ts = ts[good]

                ts_sync = ts.astype("int64") + np.int64(offset_ns)

                w.writerows(
                    zip(
                        ts_sync,
                        dfb["Roll"].to_numpy(),
                        dfb["Pitch"].to_numpy(),
                        dfb["Yaw"].to_numpy(),
                    )
                )

                rows_written += len(dfb)
                if rows_written % args.every_rows_log < len(dfb):
                    print(f"    ... {p.name}: wrote {rows_written:,} rows")

        ok += 1
        if i % args.every == 0:
            elapsed = time.time() - t0
            rate = i / max(1e-9, elapsed)
            eta = (len(files) - i) / max(1e-9, rate)
            print(
                f"[{i}/{len(files)}] OK={ok} SKIP={skip} FAIL={fail} | "
                f"{p.name} ({fmt_s(time.time()-t_file0)}) -> {out_csv.name} | "
                f"rows={rows_written:,} offset_ns={offset_ns} matched='{matched}' | elapsed={fmt_s(elapsed)} ETA={fmt_s(eta)}"
            )

    print("\nDONE")
    print(f"OK={ok} SKIP={skip} FAIL={fail}")
    print("Total time:", fmt_s(time.time() - t0))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)