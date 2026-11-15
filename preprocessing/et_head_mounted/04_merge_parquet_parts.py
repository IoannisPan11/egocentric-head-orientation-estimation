"""
04_merge_parquet_parts.py

Merges the Parquet part files of each Xsens recording (step 03) into a single Parquet file.

Thesis step : Angle extraction (Section 3.2.3)
Dataset     : ET - Head Mounted
Run on      : Windows workstation at TU Wien

Purpose
- Merge Parquet "part" files (produced by `03_convert_xsens_txt_to_parquet.py`) into a single merged Parquet
  per recording/stream, using streaming writes to avoid loading everything into memory.

What it does
- Scans `--root` for Parquet parts matching `--pattern` (default: `*.part-*.parquet`), typically under
  directories named `--parts_subdir` (default: `parquet_parts`).
- Groups part files by:
    (stem, parts_dir)
  where `stem` is the original source stem before ".part-xxxxx" (e.g., "MT_..._AHS_ON"), and `parts_dir`
  is the directory containing the parts. This avoids collisions when the same stem exists in multiple folders.
- For each group:
  1) Validates that all parts share the exact same Arrow schema (strict check).
  2) Streams row-groups from each part into a single output Parquet:
       <parts_dir>/{stem}_merged.parquet  (configurable via --out_name)
     using `ParquetWriter` with configurable compression and row_group_size.

Inputs / Assumptions
- Part files follow the naming convention: <stem>.part-00000.parquet, <stem>.part-00001.parquet, ...
- All parts within a group have identical schema; otherwise the group fails with "Schema mismatch".
- Order is determined by lexicographic sorting of the part filenames (part-00000, part-00001, ...).

Outputs
- One merged Parquet file per group, written next to the parts (inside `parts_dir` by default).
- Optional cleanup: if `--cleanup` is enabled, deletes the part files after a successful merge.

Notes / Pitfalls
- This script concatenates parts; it does not re-sort or validate timestamps.
- Strict schema validation is intentional to prevent silent column/type drift across parts.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_ROOT = r"E:\john\ET-IMU-and-head-mounted-imu"
DEFAULT_PARTS_SUBDIR = "parquet_parts"
DEFAULT_GLOB = "*.part-*.parquet"
COMPRESSION = "zstd"  # output compression


def group_parts(root: Path, parts_subdir: str, pattern: str) -> Dict[Tuple[str, Path], List[Path]]:
    """
    Groups parts by (stem, parts_dir). stem = original txt stem (e.g. MT_..._AHS_ON).
    """
    groups: Dict[Tuple[str, Path], List[Path]] = defaultdict(list)

    # find all part files
    if parts_subdir:
        part_dirs = list(root.rglob(parts_subdir))
        for d in part_dirs:
            for p in d.glob(pattern):
                stem = p.name.split(".part-")[0]  # <stem>.part-00000.parquet
                groups[(stem, d)].append(p)
    else:
        for p in root.rglob(pattern):
            stem = p.name.split(".part-")[0]
            groups[(stem, p.parent)].append(p)

    # sort each group's parts
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k])

    return groups


def load_schema_and_validate(parts: List[Path]) -> pa.Schema:
    base_schema = pq.ParquetFile(str(parts[0])).schema_arrow
    for p in parts[1:]:
        sch = pq.ParquetFile(str(p)).schema_arrow
        if sch != base_schema:
            raise ValueError(f"Schema mismatch: {parts[0]} vs {p}")
    return base_schema


def merge_parts_streaming(parts: List[Path], out_path: Path, compression: str, row_group_size: int = 256_000) -> Tuple[int, int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pq.ParquetFile(str(parts[0])).schema_arrow
    num_cols = len(schema)

    writer = pq.ParquetWriter(str(out_path), schema=schema, compression=(None if compression == "none" else compression))

    total_rows = 0
    row_groups_written = 0
    try:
        for part_path in parts:
            pf = pq.ParquetFile(str(part_path))
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                if table.schema != schema:
                    table = table.cast(schema)
                writer.write_table(table, row_group_size=row_group_size)
                total_rows += table.num_rows
                row_groups_written += 1
    finally:
        writer.close()

    return row_groups_written, total_rows, num_cols


def quick_info(out_path: Path) -> None:
    pf = pq.ParquetFile(str(out_path))
    cols = [f.name for f in pf.schema_arrow]
    rows = pf.metadata.num_rows if pf.metadata is not None else None
    print(f"    -> row_groups={pf.num_row_groups} rows={rows} cols={len(cols)}")


def main():
    ap = argparse.ArgumentParser(description="Merge parquet part files to one merged parquet per recording.")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Root folder to scan recursively.")
    ap.add_argument("--parts_subdir", default=DEFAULT_PARTS_SUBDIR, help="Subdir that contains the parts.")
    ap.add_argument("--pattern", default=DEFAULT_GLOB, help="Glob pattern for parts inside parts_subdir.")
    ap.add_argument("--out_name", default="{stem}_merged.parquet", help="Output name template.")
    ap.add_argument("--compression", default=COMPRESSION, choices=["zstd", "snappy", "gzip", "brotli", "none"])
    ap.add_argument("--cleanup", action="store_true", help="Delete parts after successful merge.")
    ap.add_argument("--every", type=int, default=1, help="Print each N merges.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    groups = group_parts(root, args.parts_subdir, args.pattern)
    items = sorted(groups.items(), key=lambda kv: (str(kv[0][1]), kv[0][0]))  # sort by folder then stem
    total = len(items)

    if total == 0:
        print("No part groups found.")
        return

    print(f"Found {total} part-groups under: {root}")
    t_all = time.perf_counter()
    done = 0
    fail = 0

    for i, ((stem, parts_dir), parts) in enumerate(items, 1):
        t0 = time.perf_counter()
        try:
            if not parts:
                continue

            out_name = args.out_name.format(stem=stem)
            out_path = parts_dir / out_name

            # validate schema
            _ = load_schema_and_validate(parts)

            # merge
            rg, rows, cols = merge_parts_streaming(parts, out_path, compression=args.compression)
            done += 1

            # info
            if (done % args.every) == 0 or i == total:
                elapsed = time.perf_counter() - t_all
                rate = done / max(elapsed, 1e-9)
                eta = (total - i) / max(rate, 1e-9)
                print(f"[{i:4d}/{total}] merged={done} fail={fail} | {rate:.2f} merges/s | elapsed={elapsed:.1f}s | ETA={eta:.1f}s | {out_path.name} ({time.perf_counter()-t0:.1f}s)")
                quick_info(out_path)

            if args.cleanup:
                for p in parts:
                    try:
                        p.unlink()
                    except Exception:
                        pass

        except Exception as e:
            fail += 1
            print(f"[{i:4d}/{total}] FAIL stem={stem} dir={parts_dir} reason={e}")

    total_time = time.perf_counter() - t_all
    print("\nDONE")
    print(f"Groups: {total} | merged: {done} | fail: {fail}")
    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()