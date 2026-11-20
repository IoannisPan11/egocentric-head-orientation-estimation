# Data availability

The data used in this thesis are **not publicly available**, and none of them, raw or derived, are included in this repository.

The recordings belong to the **[Spatial Human-Computer Interaction Lab](https://geoinfo.geo.tuwien.ac.at/spatial-hci-lab/)** of the **Research Group Geoinformation at TU Wien**, which collected them. They are first-person videos of public urban spaces in Vienna and may show bystanders who can be identified, so they cannot be published for privacy and data-protection reasons.

The author was given access to the data for the purpose of the thesis and never held a local copy. All processing ran on TU Wien machines: the preprocessing on a workstation of the lab, and the frame resizing, training and evaluation on the TU Wien dataLAB cluster. The data therefore stayed at TU Wien, and the author is not in a position to share them. Requests for access should be directed to the [Spatial Human-Computer Interaction Lab](https://geoinfo.geo.tuwien.ac.at/spatial-hci-lab/); its web page lists the contacts of the team.

## What the repository offers without the data

The repository documents the complete method and every experiment of the thesis. Without the data it is possible to:

- read and follow the full pipeline, from the raw recordings to the final model;
- inspect the training history and metrics of every run in [`results/`](../results/);
- redraw the appendix figures from these histories (see [`analysis/README.md`](../analysis/README.md#appendix-figures));
- use the weights of the final model (GitHub Release, see the main [README](../README.md#pretrained-model)).

Rerunning the preprocessing or the training, and thereby verifying the reported results independently, requires the data, which have to be requested from the group at TU Wien.

## Data layout expected by the code

The scripts use the original absolute paths of the machines they ran on (see [`reproducibility.md`](reproducibility.md#hard-coded-paths)). The layouts below describe what has to be under these paths.

### Raw recordings (input of the preprocessing)

```
<dataset root>/                                  e.g. E:\john\ET-IMU-and-head-mounted-imu
└── <participant>/                               e.g. 004_LS47
    └── <trial>/                                 e.g. trial_1_fcn
        ├── ET/<timestamp folder>/
        │   ├── PI world*.mp4                    scene-camera video (step 00)
        │   └── export/
        │       ├── world.csv                    one row per video frame, `timestamp [ns]` (steps 01, 07, 09)
        │       └── imu.csv                      eye-tracker IMU: `timestamp [ns]`, `roll`, `pitch` (step 09)
        └── ... *_AHS_ON.txt                     Xsens MTi-300 text export, header starting with `PacketCounter,` (step 03)
```

In addition, step 01 reads `results/offsets_master.csv`, in which two values were entered by hand for every recording: the frame in which the phone clock is visible (`frame_no_phone`) and the time it shows (`phone_time_str`).

### Derived data (input of the training)

All training scripts, step 10 of the preprocessing and `analysis/analyze_target_distribution.py` read from one base folder, which on the cluster was `/share/e12507305/preprocessed/DERIVED_ET_IMU_head_mounted`:

```
<BASE>/
├── <participant>/<trial>/
│   ├── paired_frames.csv                        output of step 09, read by step 10
│   ├── paired_frames11.csv                      labels averaged over 11 IMU samples (dataset.py, dataset_win11.py, analysis)
│   ├── paired_frames21.csv                      labels averaged over 21 IMU samples (dataset_win21.py)
│   └── paired_frames31.csv                      labels averaged over 31 IMU samples (dataset_win31.py)
└── results/
    ├── frames/<participant>_<trial>_<...>/frame_XXXXXX.png       frames of step 00
    └── frames_512/<participant>_<trial>_<...>/frame_XXXXXX.png   512×512 frames of step 10
```

The three `paired_framesNN.csv` files are outputs of step 09 with a half window of 5, 10 and 15 samples (`--half_window`). The dataset modules match each recording to its frame folder by the prefix `<participant>_<trial>_`.

Columns of the paired CSV files:

| Column | Meaning | Used by |
|---|---|---|
| `recording_name` | Name of the recording | Step 10 |
| `frame_name` | File name of the frame, e.g. `frame_000001.png` | Step 10, training |
| `timestamp_ns` | Epoch timestamp of the frame in nanoseconds | Training (sort order) |
| `roll_nn`, `pitch_nn`, `yaw_nn` | Angles of the nearest IMU sample, in degrees | Nearest-neighbour runs (`dataset.py`) |
| `roll_win`, `pitch_win`, `yaw_win` | Angles averaged over the window around the nearest sample, in degrees | Window 11 / 21 / 31 runs and all later experiments |

The participant-level splits (`training/fixed_split.json`, `training/final_split.json`) list the recordings as `<participant>/<trial>`, e.g. `005_LS50/trial_2_fcn_asking`.
