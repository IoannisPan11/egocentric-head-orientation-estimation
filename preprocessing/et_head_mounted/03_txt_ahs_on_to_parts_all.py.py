"""
03_txt_ahs_on_to_parts_all.py

Purpose
- Convert raw Xsens text exports (`*_AHS_ON.txt`) into Parquet "part" files using streaming reads,
  while keeping only the required sensor columns and generating standardized timestamp fields.

What it does
- Recursively scans `--root` for files matching `--pattern` (default: `*_AHS_ON.txt`).
- For each txt file:
  1) Locates the real CSV header by searching for the first line starting with "PacketCounter,".
  2) Reads the file in pandas chunks (`CHUNKSIZE`) using that header and `usecols=WANTED`.
  3) Coerces non-PacketCounter columns to numeric (invalid values become NaN).
  4) Computes timestamp helper columns:
     - `utc_ts_ns`: Unix epoch time in ns when `UTC_Valid == 1` (else NaN), built from UTC_* fields.
     - `sample_time_fine`: copy of `SampleTimeFine` (kept for fallback timing downstream).
     - `ts_ns_best`: initialized as `utc_ts_ns` (may remain NaN if UTC is invalid).
  5) Writes each chunk as a Parquet part:
       <txt_dir>/<parts_subdir>/<stem>.part-00000.parquet, part-00001.parquet, ...
     using Arrow/Parquet with configurable compression.

Skip / Overwrite behavior
- Writes a marker file `<stem>.DONE` inside the parts directory after successful conversion.
- If the DONE marker exists and `--overwrite` is not set, the file is skipped.
- If `--overwrite` is set, old parts and the DONE marker are removed and parts are regenerated.

Inputs / Assumptions
- Raw txt files contain a comma-separated header line starting with "PacketCounter,".
- The WANTED column names must match the header names in the txt export (duplicates are made unique).
- UTC_* fields represent UTC calendar time; `UTC_Nano` is added on top of the second-level timestamp.

Outputs
- Parquet parts per txt under `PARTS_SUBDIR` (default: `parquet_parts/`).
- A run-level CSV report (`ahs_txt_to_parts_report.csv` by default) listing status per input file.

Notes / Pitfalls
- `ts_ns_best` is NOT guaranteed to be populated here; if UTC is invalid it will be NaN and must be
  repaired later (e.g., by constructing a relative timeline from SampleTimeFine).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ========= USER DEFAULTS (edit once, then just run) =========
DEFAULT_ROOT = r"E:\john\ET-IMU-and-head-mounted-imu"
DEFAULT_PATTERN = "*_AHS_ON.txt"

CHUNKSIZE = 200_000
COMPRESSION = "zstd"  # zstd/snappy/gzip/brotli/none (none -> no compression)

# Pick what you actually need (must match header names in txt)
WANTED = [
    "PacketCounter",
    "SampleTimeFine",
    "UTC_Year", "UTC_Month", "UTC_Day", "UTC_Hour", "UTC_Minute", "UTC_Second", "UTC_Nano",
    "UTC_Valid",
    "Quat_q0", "Quat_q1", "Quat_q2", "Quat_q3",
    "Roll", "Pitch", "Yaw",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z",
    "StatusWord",
]

# Where to write the parts (inside each IMU folder, usually)
PARTS_SUBDIR = "parquet_parts"  # set "" to write next to txt


# ========= helpers =========

def make_unique(names: List[str]) -> List[str]:
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
    return out


def find_header_line(path: Path) -> Tuple[int, str]:
    """
    Returns (line_index_0_based, header_line_string).
    Header is the first line that starts with 'PacketCounter,' (comma-separated).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("PacketCounter,"):
                return i, line.strip()
    raise RuntimeError("NO_HEADER_PACKETCOUNTER_COMMA")


def build_usecols(all_cols: List[str], wanted: List[str]) -> List[str]:
    wanted_set = set(wanted)
    return [c for c in all_cols if c in wanted_set]


def utc_to_ns(df: pd.DataFrame) -> pd.Series:
    """
    Build UTC timestamp in ns when UTC_Valid == 1 and fields exist, else NaN.
    """
    if "UTC_Valid" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")

    valid = pd.to_numeric(df["UTC_Valid"], errors="coerce").astype("float64")
    mask = (valid == 1.0)

    # build datetime (UTC)
    dt = pd.to_datetime(
        dict(
            year=pd.to_numeric(df.get("UTC_Year"), errors="coerce"),
            month=pd.to_numeric(df.get("UTC_Month"), errors="coerce"),
            day=pd.to_numeric(df.get("UTC_Day"), errors="coerce"),
            hour=pd.to_numeric(df.get("UTC_Hour"), errors="coerce"),
            minute=pd.to_numeric(df.get("UTC_Minute"), errors="coerce"),
            second=pd.to_numeric(df.get("UTC_Second"), errors="coerce"),
        ),
        errors="coerce",
        utc=True,
    )

    # dt.view("int64") is deprecated -> use astype
    ns = dt.astype("int64")  # NaT -> min int64
    ns = pd.Series(ns, index=df.index).astype("float64")

    # NaT should become NaN
    ns[ns < 0] = np.nan

    nano = pd.to_numeric(df.get("UTC_Nano"), errors="coerce").astype("float64")
    ns = ns + nano

    ns[~mask] = np.nan
    return ns


@dataclass
class FileResult:
    path: str
    status: str  # OK / SKIP / FAIL
    reason: str
    n_parts: int
    rows_written: int
    seconds: float
    parts_dir: str


def arrow_write_part(df: pd.DataFrame, out_path: Path, compression: Optional[str]) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression=(None if compression == "none" else compression))


def convert_one_txt_to_parts(
    txt_path: Path,
    chunksize: int,
    wanted: List[str],
    parts_subdir: str,
    compression: str,
    overwrite: bool,
) -> FileResult:
    t0 = time.perf_counter()

    try:
        header_idx, header_line = find_header_line(txt_path)
        raw_cols = header_line.split(",")
        cols = make_unique(raw_cols)
        usecols = build_usecols(cols, wanted)

        if not usecols:
            return FileResult(
                path=str(txt_path), status="SKIP", reason="NO_WANTED_COLS_MATCHED",
                n_parts=0, rows_written=0, seconds=time.perf_counter() - t0, parts_dir=""
            )

        base_dir = txt_path.parent
        parts_dir = (base_dir / parts_subdir) if parts_subdir else base_dir
        parts_dir.mkdir(parents=True, exist_ok=True)

        # marker file to skip re-processing
        done_marker = parts_dir / (txt_path.stem + ".DONE")
        if done_marker.exists() and not overwrite:
            return FileResult(
                path=str(txt_path), status="SKIP", reason="ALREADY_DONE",
                n_parts=0, rows_written=0, seconds=time.perf_counter() - t0, parts_dir=str(parts_dir)
            )

        # clean old parts if overwrite
        if overwrite:
            for old in sorted(parts_dir.glob(f"{txt_path.stem}.part-*.parquet")):
                try:
                    old.unlink()
                except Exception:
                    pass
            if done_marker.exists():
                try:
                    done_marker.unlink()
                except Exception:
                    pass

        reader = pd.read_csv(
            txt_path,
            sep=",",
            skiprows=header_idx + 1,  # skip everything up to and incl header
            names=cols,
            usecols=usecols,
            engine="c",
            chunksize=chunksize,
            low_memory=True,
        )

        n_parts = 0
        rows_written = 0
        last_print = time.perf_counter()

        for chunk_i, chunk in enumerate(reader):
            # numeric conversion (avoid object dtype)
            for c in chunk.columns:
                if c != "PacketCounter":
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            # computed columns
            chunk["utc_ts_ns"] = utc_to_ns(chunk)
            chunk["sample_time_fine"] = chunk.get("SampleTimeFine")
            chunk["ts_ns_best"] = chunk["utc_ts_ns"]  # keep NaN if invalid

            part_path = parts_dir / f"{txt_path.stem}.part-{n_parts:05d}.parquet"
            arrow_write_part(chunk, part_path, compression=compression)

            n_parts += 1
            rows_written += int(chunk.shape[0])

            # per-file progress every ~1s
            now = time.perf_counter()
            if now - last_print >= 1.0:
                elapsed = now - t0
                rps = rows_written / max(elapsed, 1e-9)
                print(f"    -> parts={n_parts} rows={rows_written} | {rps:,.0f} rows/s | elapsed={elapsed:.1f}s")
                last_print = now

        # write done marker
        done_marker.write_text(f"OK parts={n_parts} rows={rows_written}\n", encoding="utf-8")

        return FileResult(
            path=str(txt_path), status="OK", reason="",
            n_parts=n_parts, rows_written=rows_written, seconds=time.perf_counter() - t0, parts_dir=str(parts_dir)
        )

    except Exception as e:
        return FileResult(
            path=str(txt_path), status="FAIL", reason=str(e),
            n_parts=0, rows_written=0, seconds=time.perf_counter() - t0, parts_dir=""
        )


def main():
    ap = argparse.ArgumentParser(description="Convert all *_AHS_ON.txt to parquet parts (streaming).")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Root folder to scan recursively.")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern (default: *_AHS_ON.txt)")
    ap.add_argument("--chunksize", type=int, default=CHUNKSIZE, help="Pandas chunksize (rows).")
    ap.add_argument("--compression", default=COMPRESSION, choices=["zstd", "snappy", "gzip", "brotli", "none"])
    ap.add_argument("--parts_subdir", default=PARTS_SUBDIR, help="Subfolder for part parquet files.")
    ap.add_argument("--overwrite", action="store_true", help="Recreate parts even if DONE marker exists.")
    ap.add_argument("--every", type=int, default=1, help="Print each N files.")
    ap.add_argument("--out_csv", default=None, help="Report CSV path.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    txt_files = sorted(root.rglob(args.pattern))
    total = len(txt_files)
    if total == 0:
        print("No files found.")
        return

    out_csv = Path(args.out_csv) if args.out_csv else (root / "ahs_txt_to_parts_report.csv")

    print(f"Scanning {total} files under: {root}")

    t_all = time.perf_counter()
    ok = skip = fail = 0
    results: List[FileResult] = []

    for i, p in enumerate(txt_files, 1):
        t_file = time.perf_counter()
        res = convert_one_txt_to_parts(
            txt_path=p,
            chunksize=args.chunksize,
            wanted=WANTED,
            parts_subdir=args.parts_subdir,
            compression=args.compression,
            overwrite=args.overwrite,
        )
        results.append(res)

        if res.status == "OK":
            ok += 1
        elif res.status == "SKIP":
            skip += 1
        else:
            fail += 1

        if (i % args.every) == 0 or i == total:
            elapsed = time.perf_counter() - t_all
            fps = i / max(elapsed, 1e-9)
            eta = (total - i) / max(fps, 1e-9)
            short = p.name
            print(
                f"[{i:5d}/{total}] OK={ok} SKIP={skip} FAIL={fail} | "
                f"{fps:.2f} files/s | elapsed={elapsed:.1f}s | ETA={eta:.1f}s | "
                f"{short} ({time.perf_counter()-t_file:.1f}s) -> {res.status}"
            )

    # write report
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "status", "reason", "n_parts", "rows_written", "seconds", "parts_dir"])
        for r in results:
            w.writerow([r.path, r.status, r.reason, r.n_parts, r.rows_written, f"{r.seconds:.3f}", r.parts_dir])

    total_time = time.perf_counter() - t_all
    print("\nDONE")
    print(f"Scanned: {total} | OK: {ok} | SKIP: {skip} | FAIL: {fail}")
    print(f"Report: {out_csv}")
    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()