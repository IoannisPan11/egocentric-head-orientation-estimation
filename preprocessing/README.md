# Preprocessing

This folder turns the raw eye-tracking recordings into **frame–label pairs**: every scene-camera frame is paired with its roll, pitch and yaw. Thesis reference: Chapter 3.

Two datasets were preprocessed. **Only ET – Head Mounted was used to train and evaluate the model.**

| Folder | Dataset | Setup | Used for the model |
|---|---|---|---|
| [`et_head_mounted/`](et_head_mounted/) | **ET – Head Mounted** (pedestrians) | Pupil Labs Invisible glasses + Xsens MTi-300 IMU on a cap | ✅ Yes |
| [`et_upper_body/`](et_upper_body/) | **ET – Upper Body** (cyclists) | Pupil Labs Invisible glasses + Xsens Awinda (11 upper-body sensors) | ❌ No (excluded) |

## Why ET – Upper Body was excluded

Thesis Section 3.2.7 gives three reasons:

1. **Synchronization (main reason).** The IMU orientation angles could not be reliably aligned with the video timestamps, so the frame–label pairs were not trustworthy. The same synchronization method worked for ET – Head Mounted.
2. **Different frame rate:** 20 fps instead of 30 fps.
3. **Gait compensation.** The upper-body IMUs apply it; the eye-tracker IMU and the head-mounted IMU do not.

The scripts are still published for transparency. They document the work that was done and could be a starting point for future work.

## ET – Head Mounted pipeline

The scripts run in numerical order. Each script has a docstring at the top that describes its inputs, outputs, and logic.

| Step | Script | What it does | Thesis step |
|---|---|---|---|
| 00 | `00_export_world_frames_safe.ps1` | Exports every frame of the scene-camera video as PNG using FFmpeg (no-overwrite mode) | Frame extraction |
| 01 | `01_fill_offsets_master.py` | Computes the per-recording clock offset from a manually read on-screen clock frame | Time synchronization |
| 02 | `02_sanity_check.py` | Checks the effective video frame rate in sliding 1-second windows | Frame-rate check |
| 03 | `03_txt_ahs_on_to_parts_all.py.py` | Converts raw Xsens text exports (`*_AHS_ON.txt`) into Parquet parts | Angle extraction |
| 04 | `04_merge_all_parts_to_merged_parquet.py` | Merges the Parquet parts into one file per recording | Angle extraction |
| 05 | `05_fix_ts_ns_best_all.py` | Ensures each merged file has a usable time axis | Time synchronization |
| 06 | `06_sync_head_imu_to_et_with_offsets.py` | Shifts the head-IMU timestamps onto the eye-tracker timeline using the offsets | Time synchronization |
| 07 | `07_add_head_epoch_all.py` | Converts the synchronized head-IMU time to epoch timestamps | Time synchronization |
| 08 | `08_clean_head_epoch_csvs.py` | Removes rows with missing angles or timestamps | Cleaning |
| 09 | `09_pair_frames_with_imus.py` | Pairs every frame with roll/pitch (eye-tracker IMU) and yaw (Xsens), using the nearest sample or a window mean | Frame–label pairs |

The last preprocessing step, **crop and resize** (1088×1080 → 1080×1080 → 512×512), ran on the TU Wien dataLAB cluster together with training. Its scripts are in [`../training/frame_resizing/`](../training/frame_resizing/).

Other files in `et_head_mounted/`:

- `09_pair_single_032_window.py`: a variant of step 09 used for a single recording.
- `nothing important.py`: a small ad-hoc script that inspects one CSV; it is not part of the pipeline.
- `fps_sanity_check_summary_*.csv`: outputs of step 02 (frame-rate summaries per recording).

## ET – Upper Body scripts

These scripts follow the same idea (frame export, frame-rate check, synchronization offsets, angle computation from quaternions, frame–label pairing), adapted to the Awinda data and the per-participant "legs" of this dataset. The folder also contains several ad-hoc check scripts written while investigating the synchronization problem (`check_*.py`, `minicheck_*.py`, `kontinoteroframecheck.py`, `final.py`).

> Some comments in these scripts refer to the eye tracker as "Neon". The device used was the **Pupil Labs Invisible**.

## Running environment

The preprocessing ran on a **Windows workstation at TU Wien** with Python, pandas/pyarrow, OpenCV, and FFmpeg. The scripts contain the original absolute Windows paths (`E:\…`, `D:\…`), and some scripts also accept them as command-line arguments (`--root`, `--head_root`, …). The scripts are published as they were run. The data are not public (see the main [README](../README.md#data-availability)).
