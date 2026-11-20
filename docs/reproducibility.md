# Reproducibility notes

This page describes where each part of the code ran, the software environment, and how far the results can be reproduced. The data are not public (see [`data_availability.md`](data_availability.md)), so a full rerun is only possible with access to them.

## Where each part ran

| Part | Machine | Tools |
|---|---|---|
| Preprocessing, steps 00–09 | Windows workstation of the lab at TU Wien | Python, PowerShell, FFmpeg |
| Preprocessing step 10 (crop and resize) | TU Wien dataLAB cluster | SLURM, Python |
| Training and evaluation | TU Wien dataLAB cluster, one NVIDIA A100 per job (partition `GPU-a100s`) | SLURM, micromamba environment `thesis-gpu` |
| Analysis (`analysis/`) | TU Wien dataLAB cluster | micromamba environment `thesis-gpu` |

Every training script has its original SLURM launcher next to it. The launchers request one GPU, change into the original working directory on the cluster and start the script with `micromamba run -n thesis-gpu`.

## Environment

The original environments are no longer accessible, and the `thesis-gpu` environment was never exported, so the exact package versions are not known. `pyproject.toml` and `uv.lock` define a reconstructed environment with two dependency groups:

```bash
uv sync --group preprocessing   # steps 00–10
uv sync --group training        # training, evaluation and analysis
```

The PyTorch version follows from the code and the saved checkpoint: the training scripts use `torch.amp.GradScaler("cuda")`, which exists from PyTorch 2.3, and the checkpoint metadata point to a 2.x release older than 2.6. The environment therefore uses PyTorch 2.5.1 and torchvision 0.20.1. Versions of the other packages may differ slightly from the original ones.

FFmpeg and PowerShell are needed in addition for the `.ps1` script of step 00.

## Version of the shared modules

The modules shared by the training scripts (`dataset_win31.py`, `model_resnet_transformer.py`) received small additions while the experiments were running. The repository contains their latest version, which is the one used to train the final model.

## Hard-coded paths

The code is published as it was run, so the scripts contain the absolute paths of the original machines:

| Code | Path constant | Original value |
|---|---|---|
| Preprocessing steps 00–09 | `$projectRoot`, `$outRoot`, `PROJECT_ROOT`, `DATASET_ROOT`, `DEFAULT_ROOT`, or command-line arguments such as `--root` | `E:\john\ET-IMU-and-head-mounted-imu`, `D:\john.panagiotopoulos\ET-IMU-and-head-mounted-imu` |
| Preprocessing step 10, all training scripts, `analysis/analyze_target_distribution.py` | `BASE` | `/share/e12507305/preprocessed/DERIVED_ET_IMU_head_mounted` |
| SLURM launchers | `cd`, `--output`, `--error` | `/home/e12507305/...` |

To run a script on another machine, either place the data at the original path or change the path constant in the script. The layout that has to exist under each path is described in [`data_availability.md`](data_availability.md#data-layout-expected-by-the-code).

The training code also uses paths relative to the working directory (`fixed_split.json`, `final_split.json`, `runs/`), because on the cluster all files were in one folder. Training scripts are therefore run from `training/`, with this folder on the Python path:

```bash
cd training
PYTHONPATH=. python experiments/08_final_model/train_final_checkpoint.py
```

The scripts in `analysis/` are run the same way (see [`analysis/README.md`](../analysis/README.md)).

## Order of execution

1. **Preprocessing** (`preprocessing/et_head_mounted/`): steps 00–09 in numerical order on the raw recordings, then step 10 on the frames and the paired CSV files. Step 09 was run with a half window of 5, 10 and 15 samples to produce the Window 11, 21 and 31 labels.
2. **Splits** (`training/`): `create_fixed_split_preliminary.py` writes `fixed_split.json` (intermediate subset, 9 / 2 / 2 recordings) and `create_final_split_full_dataset.py` writes `final_split.json` (26 / 6 / 6 recordings). Both files are included, so this step is only needed to check them.
3. **Experiments** (`training/experiments/`): folders 01–07 in order, each run as one script. Every step builds on the configuration selected in the previous one (see [`training/README.md`](../training/README.md#how-the-experiments-were-run)).
4. **Final model** (`training/experiments/08_final_model/`): the selected configuration retrained on the full split.
5. **Analysis** (`analysis/`): the target distribution with the no-change reference, and the appendix figures.

## Determinism

All training scripts set the random seed to 42 for Python, NumPy and PyTorch. Training on the GPU is nevertheless not fully deterministic: it uses mixed precision (`torch.amp.autocast`), and cuDNN selects its algorithms without a deterministic setting. A rerun with the same data and configuration may therefore give values that differ slightly from those of the thesis. The values reported in the thesis are those recorded in [`results/`](../results/).

## What can be reproduced without the data

| Output | Reproducible without the data | How |
|---|---|---|
| Training histories and metrics of every run (thesis Chapter 5) | Yes, as recorded | [`results/`](../results/) |
| Appendix figures A.1–A.12 | Yes, redrawn from the histories | [`analysis/README.md`](../analysis/README.md#appendix-figures) |
| Preprocessing, training, evaluation, no-change reference | No, requires the data | — |
