"""
05_fix_ts_ns_best_all.py

Purpose
- Ensure every `*_merged.parquet` file has a usable time axis in `ts_ns_best`.
  If absolute UTC timestamps are unavailable (all-null), fallback to a relative timeline derived
  from `SampleTimeFine`, and write a new `*_merged_fixed.parquet` without modifying the original.

When it changes a file
- If `ts_ns_best` already contains any non-null values: SKIP (no changes).
- Else:
  - If `utc_ts_ns` exists and contains any non-null values: set `ts_ns_best <- utc_ts_ns`.
  - Otherwise: set `ts_ns_best <- ts_rel_ns`, where:
        ts_rel_ns = (SampleTimeFine - first_non_null(SampleTimeFine)) * 1_000_000
    (i.e., a recording-relative timeline in nanoseconds).

What it does (implementation details)
- Scans `--root` recursively for files ending with `_merged.parquet`.
- Reads only the minimal columns needed to decide and to compute new arrays
  (`SampleTimeFine`, and optionally `ts_ns_best`, `utc_ts_ns`).
- Rewrites the full parquet by streaming row-groups from the input file and replacing/adding:
    - `ts_ns_best` (replaced if exists, appended if missing)
    - `ts_rel_ns`  (added if missing, replaced if exists)
- Outputs next to the original file:
    <name>_merged_fixed.parquet
  unless `--overwrite` is provided (in-place overwrite is supported but not recommended).

Inputs / Assumptions
- `SampleTimeFine` must exist to build the fallback timeline.
- The scaling `* 1_000_000` assumes SampleTimeFine is approximately in milliseconds.
  (This yields nanoseconds and preserves sample spacing; it is not an absolute epoch time.)

Outputs
- New parquet files with corrected/created `ts_ns_best` and `ts_rel_ns`.
- Progress logging and optional `--dry_run` to report intended actions without writing.

Notes / Pitfalls
- The relative fallback timeline is only valid within each recording; it cannot be compared across
  recordings without an external anchor/offset.
- `all_null()` checks for Arrow nulls; NaNs are handled by casting to float64 where needed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


MERGED_SUFFIX = "_merged.parquet"
FIXED_SUFFIX = "_merged_fixed.parquet"


def human_seconds(x: float) -> str:
    if x < 60:
        return f"{x:.1f}s"
    m = int(x // 60)
    s = x - 60 * m
    if m < 60:
        return f"{m}m{s:04.1f}s"
    h = m // 60
    m = m % 60
    return f"{h}h{m:02d}m{s:04.1f}s"


def find_merged_parquets(root: Path) -> List[Path]:
    return sorted(root.rglob(f"*{MERGED_SUFFIX}"))


def get_first_non_null(arr: pa.Array) -> Optional[float]:
    # Works for numeric arrays; returns python scalar or None.
    nn = pc.drop_null(arr)
    if len(nn) == 0:
        return None
    return nn[0].as_py()


def all_null(arr: pa.Array) -> bool:
    # True if all values are null
    return pc.sum(pc.is_null(arr)).as_py() == len(arr)


def ensure_float64(arr: pa.Array) -> pa.Array:
    # Convert to float64 to support NaN and arithmetic
    if pa.types.is_float64(arr.type):
        return arr
    # Cast integers/float32/etc to float64
    return pc.cast(arr, pa.float64())


def build_ts_rel_ns(sample_time_fine: pa.Array) -> pa.Array:
    """
    ts_rel_ns = (SampleTimeFine - first_non_null) * 1_000_000
    returns float64 (NaN where STF is null)
    """
    stf = ensure_float64(sample_time_fine)
    first = get_first_non_null(stf)
    if first is None:
        # all null -> all null array
        return pa.array([None] * len(stf), type=pa.float64())

    # (stf - first) * 1e6
    rel = pc.multiply(pc.subtract(stf, pa.scalar(float(first), pa.float64())), pa.scalar(1_000_000.0, pa.float64()))
    return rel


def fix_one_file(
    in_path: Path,
    overwrite: bool = False,
    verbose: bool = False,
) -> Tuple[str, Optional[Path], str]:
    """
    Returns: (status, out_path_or_none, message)
    status in {"OK", "SKIP", "FAIL"}
    """
    try:
        out_path = in_path if overwrite else in_path.with_name(in_path.name.replace(MERGED_SUFFIX, FIXED_SUFFIX))

        if (not overwrite) and out_path.exists():
            return "SKIP", out_path, "fixed already exists"

        pf = pq.ParquetFile(in_path)
        schema = pf.schema_arrow
        cols = [f.name for f in schema]

        # Must have SampleTimeFine to fallback
        if "SampleTimeFine" not in cols:
            return "SKIP", None, "missing SampleTimeFine"

        # Read needed columns only (efficient)
        need = ["SampleTimeFine"]
        if "ts_ns_best" in cols:
            need.append("ts_ns_best")
        if "utc_ts_ns" in cols:
            need.append("utc_ts_ns")

        table = pq.read_table(in_path, columns=need)

        stf = table["SampleTimeFine"]
        ts_best = table["ts_ns_best"] if "ts_ns_best" in table.column_names else None
        utc = table["utc_ts_ns"] if "utc_ts_ns" in table.column_names else None

        # Determine if we need to fix
        ts_best_all_null = True
        if ts_best is not None:
            ts_best_all_null = all_null(ts_best)

        utc_all_null = True
        if utc is not None:
            utc_all_null = all_null(utc)

        # If ts_ns_best exists and has some non-null values -> nothing to do
        if (ts_best is not None) and (not ts_best_all_null):
            if not overwrite:
                # still write a copy? no, skip.
                return "SKIP", out_path, "ts_ns_best already populated"
            else:
                return "SKIP", out_path, "overwrite requested but ts_ns_best already populated"

        # If utc has values but ts_best is null, we can set ts_best=utc (best practice)
        # Otherwise fallback to SampleTimeFine relative
        if (utc is not None) and (not utc_all_null):
            new_ts_best = ensure_float64(utc)
            reason = "ts_ns_best <- utc_ts_ns"
        else:
            new_ts_best = build_ts_rel_ns(stf)
            reason = "ts_ns_best <- relative(SampleTimeFine)*1e6"

        # Also keep a dedicated relative column for clarity
        ts_rel_ns = build_ts_rel_ns(stf)

        # Now rewrite full parquet: read full table (stream by row groups) and write out
        # We avoid loading entire file in pandas; we stream row-groups and replace/add columns.
        writer: Optional[pq.ParquetWriter] = None
        total_rows = 0
        rg_written = 0

        for rg in range(pf.num_row_groups):
            rg_table = pf.read_row_group(rg)

            # Build mapping for current row-group slice
            start = total_rows
            end = total_rows + rg_table.num_rows

            # Slice the computed arrays to match row-group
            new_ts_best_rg = new_ts_best.slice(start, rg_table.num_rows)
            ts_rel_rg = ts_rel_ns.slice(start, rg_table.num_rows)

            # If ts_ns_best exists in file, replace; else append
            colnames = rg_table.column_names
            new_cols = []
            new_names = []

            for name in colnames:
                if name == "ts_ns_best":
                    new_cols.append(new_ts_best_rg)
                    new_names.append("ts_ns_best")
                else:
                    new_cols.append(rg_table[name])
                    new_names.append(name)

            if "ts_ns_best" not in colnames:
                new_cols.append(new_ts_best_rg)
                new_names.append("ts_ns_best")

            # Add ts_rel_ns if missing (or replace if exists)
            if "ts_rel_ns" in colnames:
                # replace existing ts_rel_ns
                fixed_cols = []
                fixed_names = []
                for n, c in zip(new_names, new_cols):
                    if n == "ts_rel_ns":
                        fixed_cols.append(ts_rel_rg)
                        fixed_names.append("ts_rel_ns")
                    else:
                        fixed_cols.append(c)
                        fixed_names.append(n)
                new_cols, new_names = fixed_cols, fixed_names
            else:
                new_cols.append(ts_rel_rg)
                new_names.append("ts_rel_ns")

            fixed_rg = pa.table(new_cols, names=new_names)

            if writer is None:
                writer = pq.ParquetWriter(out_path, fixed_rg.schema, compression="zstd")

            writer.write_table(fixed_rg)
            total_rows += rg_table.num_rows
            rg_written += 1

        if writer is not None:
            writer.close()

        # Quick metadata read
        out_pf = pq.ParquetFile(out_path)
        msg = f"{reason} | row_groups={out_pf.num_row_groups} rows={out_pf.metadata.num_rows if out_pf.metadata else total_rows}"
        return "OK", out_path, msg

    except Exception as e:
        return "FAIL", None, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fix ts_ns_best across all *_merged.parquet files under --root. Writes *_merged_fixed.parquet."
    )
    ap.add_argument("--root", required=True, help="Root folder to scan (e.g. E:\\john\\ET-IMU-and-head-mounted-imu)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite *_merged.parquet in-place (NOT recommended).")
    ap.add_argument("--every", type=int, default=1, help="Print progress every N files (default: 1).")
    ap.add_argument("--dry_run", action="store_true", help="Scan and report what would happen, without writing.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    files = find_merged_parquets(root)
    if not files:
        print(f"No {MERGED_SUFFIX} found under: {root}")
        return

    print(f"Found {len(files)} merged parquets under: {root}")

    t0 = time.time()
    ok = 0
    skip = 0
    fail = 0

    for i, p in enumerate(files, 1):
        t1 = time.time()

        if args.dry_run:
            # minimal inspection: just check existence of SampleTimeFine and ts_ns_best null-ness
            try:
                pf = pq.ParquetFile(p)
                cols = [f.name for f in pf.schema_arrow]
                if "SampleTimeFine" not in cols:
                    status = "SKIP"
                    msg = "missing SampleTimeFine"
                else:
                    need = ["SampleTimeFine"]
                    if "ts_ns_best" in cols:
                        need.append("ts_ns_best")
                    if "utc_ts_ns" in cols:
                        need.append("utc_ts_ns")
                    tbl = pq.read_table(p, columns=need)
                    tsb = tbl["ts_ns_best"] if "ts_ns_best" in tbl.column_names else None
                    utc = tbl["utc_ts_ns"] if "utc_ts_ns" in tbl.column_names else None
                    tsb_all = True if tsb is None else all_null(tsb)
                    utc_all = True if utc is None else all_null(utc)
                    if (tsb is not None) and (not tsb_all):
                        status, msg = "SKIP", "ts_ns_best already populated"
                    else:
                        status = "OK"
                        msg = "would set ts_ns_best from utc_ts_ns" if (utc is not None and not utc_all) else "would set ts_ns_best from SampleTimeFine relative"
            except Exception as e:
                status, msg = "FAIL", str(e)

            out_path = None
        else:
            status, out_path, msg = fix_one_file(p, overwrite=args.overwrite)

        dt = time.time() - t1

        if status == "OK":
            ok += 1
        elif status == "SKIP":
            skip += 1
        else:
            fail += 1

        if (i % args.every) == 0 or i == 1 or i == len(files):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(files) - i) / rate if rate > 0 else 0.0
            name = p.name
            print(
                f"[{i:4d}/{len(files)}] OK={ok} SKIP={skip} FAIL={fail} | "
                f"{rate:.2f} files/s | elapsed={human_seconds(elapsed)} | ETA={human_seconds(eta)} | "
                f"{name} ({dt:.1f}s) -> {status}"
            )
            if status == "OK":
                if out_path is not None:
                    print(f"    -> {out_path}")
                print(f"    -> {msg}")
            elif status != "SKIP":
                print(f"    -> {msg}")

    total = time.time() - t0
    print("\nDONE")
    print(f"Scanned: {len(files)} | OK: {ok} | SKIP: {skip} | FAIL: {fail}")
    print(f"Total time: {human_seconds(total)}")


if __name__ == "__main__":
    main()