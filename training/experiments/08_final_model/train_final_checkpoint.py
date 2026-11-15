"""
train_final_checkpoint.py

Trains the final model: the configuration selected by the experiments of Sections 5.2-5.8,
retrained on the full dataset with the final participant-level split.

Thesis step : Final optimized configuration (Section 5.9)
Run folder  : runs/Final_Full_Dataset__Final_Config_All_Recordings
Run on      : TU Wien dataLAB cluster (SLURM, NVIDIA A100)
Launcher    : `run_training_train_final.sh`

Configuration
- Labels          : Window 31: `dataset_win31.py`, columns roll_win, pitch_win, yaw_win
- Input sequence  : 40 past + 1 current + 5 future frames, stride 8 between windows
- Split           : `final_split.json` (26 / 6 / 6 recordings, participant-level)
- Batch size      : 12
- Loss            : Smooth L1, beta 0.1
- Optimizer       : AdamW, learning rate 1e-4 (backbone 1e-5), weight decay 0
- Schedule        : cosine annealing over 50 epochs down to 1e-6
- Regularization  : dropout 0.2, gradient clipping 1.0
- Transformer     : d_model 64, 2 layers, 4 heads, feed-forward 512
- Early stopping  : patience 4 epochs on the validation loss

What the script does
- The same training and evaluation procedure as the experiment scripts (frozen early ResNet
  layers, mixed precision, `best_model.pth` at the lowest validation loss, `history.csv`,
  plots, test evaluation and `runs/experiments_summary.csv`).
- In addition, it saves `last_checkpoint.pth` after every epoch (model, optimizer, scheduler,
  gradient scaler, early-stopping state and history). If that file exists at start-up,
  training resumes from the next epoch, so the job can be restarted after a time limit.

Output
- The `best_model.pth` of this run is the published model weights.

How to run
- From the `training/` folder:
    PYTHONPATH=. python experiments/08_final_model/train_final_checkpoint.py
"""

from __future__ import annotations

from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import amp

from dataset_win31 import (
    discover_recordings,
    HeadMountedSequenceDataset,
)
from model_resnet_transformer import ResNetTransformerRegressor
import json


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Helpers
# =========================================================
def wrap_angle_deg(x: torch.Tensor) -> torch.Tensor:
    return (x + 180.0) % 360.0 - 180.0


def yaw_deg_from_sincos(sin_yaw: torch.Tensor, cos_yaw: torch.Tensor) -> torch.Tensor:
    yaw_rad = torch.atan2(sin_yaw, cos_yaw)
    return torch.rad2deg(yaw_rad)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    pred_roll = preds[:, 0] * 180.0
    pred_pitch = preds[:, 1] * 180.0
    true_roll = targets[:, 0] * 180.0
    true_pitch = targets[:, 1] * 180.0

    pred_yaw = yaw_deg_from_sincos(preds[:, 2], preds[:, 3])
    true_yaw = yaw_deg_from_sincos(targets[:, 2], targets[:, 3])

    roll_err = pred_roll - true_roll
    pitch_err = pred_pitch - true_pitch
    yaw_err = wrap_angle_deg(pred_yaw - true_yaw)

    roll_abs = torch.abs(roll_err)
    pitch_abs = torch.abs(pitch_err)
    yaw_abs = torch.abs(yaw_err)

    roll_mae = torch.mean(roll_abs).item()
    pitch_mae = torch.mean(pitch_abs).item()
    yaw_mae = torch.mean(yaw_abs).item()

    roll_rmse = torch.sqrt(torch.mean(roll_err ** 2)).item()
    pitch_rmse = torch.sqrt(torch.mean(pitch_err ** 2)).item()
    yaw_rmse = torch.sqrt(torch.mean(yaw_err ** 2)).item()

    roll_medae = torch.median(roll_abs).item()
    pitch_medae = torch.median(pitch_abs).item()
    yaw_medae = torch.median(yaw_abs).item()

    mean_mae = (roll_mae + pitch_mae + yaw_mae) / 3.0
    mean_rmse = (roll_rmse + pitch_rmse + yaw_rmse) / 3.0
    mean_medae = (roll_medae + pitch_medae + yaw_medae) / 3.0

    return {
        "mae_roll_deg": roll_mae,
        "mae_pitch_deg": pitch_mae,
        "mae_yaw_deg": yaw_mae,
        "mae_mean_deg": mean_mae,
        "rmse_roll_deg": roll_rmse,
        "rmse_pitch_deg": pitch_rmse,
        "rmse_yaw_deg": yaw_rmse,
        "rmse_mean_deg": mean_rmse,
        "medae_roll_deg": roll_medae,
        "medae_pitch_deg": pitch_medae,
        "medae_yaw_deg": yaw_medae,
        "medae_mean_deg": mean_medae,
    }


def freeze_early_resnet_layers(model: ResNetTransformerRegressor):
    """
    Freeze the early layers of the ResNet50 backbone:
      conv1, bn1, layer1, layer2

    Keep trainable:
      layer3, layer4, projection, transformer, head
    """
    backbone_net = model.backbone.net

    frozen_modules = [
        backbone_net.conv1,
        backbone_net.bn1,
        backbone_net.layer1,
        backbone_net.layer2,
    ]

    for module in frozen_modules:
        for p in module.parameters():
            p.requires_grad = False

    print("[Backbone freezing] conv1, bn1, layer1, layer2 frozen", flush=True)


# =========================================================
# Data
# =========================================================
def load_fixed_split(items, split_json_path: str):
    with open(split_json_path, "r") as f:
        split_ids = json.load(f)

    item_by_name = {item.recording_name: item for item in items}

    missing = []
    for split_name in ["train", "val", "test"]:
        for recording_name in split_ids[split_name]:
            if recording_name not in item_by_name:
                missing.append(recording_name)

    if missing:
        raise ValueError(f"Recordings from fixed split not found in discovered items: {missing}")

    split = {
        "train": [item_by_name[x] for x in split_ids["train"]],
        "val": [item_by_name[x] for x in split_ids["val"]],
        "test": [item_by_name[x] for x in split_ids["test"]],
    }

    train_p = {x.participant for x in split["train"]}
    val_p = {x.participant for x in split["val"]}
    test_p = {x.participant for x in split["test"]}

    assert train_p.isdisjoint(val_p), "Participant overlap between train and val"
    assert train_p.isdisjoint(test_p), "Participant overlap between train and test"
    assert val_p.isdisjoint(test_p), "Participant overlap between val and test"

    assert len(split["train"]) == 26, f"Expected 26 train recordings, got {len(split['train'])}"
    assert len(split["val"]) == 6, f"Expected 6 val recordings, got {len(split['val'])}"
    assert len(split["test"]) == 6, f"Expected 6 test recordings, got {len(split['test'])}"

    print("[Fixed split loaded]", flush=True)
    print("  train recordings:", len(split["train"]), flush=True)
    print("  val recordings:", len(split["val"]), flush=True)
    print("  test recordings:", len(split["test"]), flush=True)

    print("  train IDs:", split_ids["train"], flush=True)
    print("  val IDs:", split_ids["val"], flush=True)
    print("  test IDs:", split_ids["test"], flush=True)

    return split


def make_loaders(
    base_dir: str,
    batch_size: int = 24,
    num_workers: int = 6,
    seed: int = 42,
    past_frames: int = 20,
    future_frames: int = 5,
    split_json_path: str = "final_split.json",
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    items = discover_recordings(base_dir, frames_subdir="results/frames_512")

    # =====================================================
    # FIXED LENGTH-STRATIFIED SPLIT
    # The split is computed once externally and reused across
    # all experiments to ensure fair comparison.
    # =====================================================
    split = load_fixed_split(
        items=items,
        split_json_path=split_json_path,
    )

    print("[Fixed length-stratified split]", flush=True)
    print("  train recordings:", len(split["train"]), flush=True)
    print("  val recordings:", len(split["val"]), flush=True)
    print("  test recordings:", len(split["test"]), flush=True)

    train_ds = HeadMountedSequenceDataset(
        recordings=split["train"],
        past_frames=past_frames,
        future_frames=future_frames,
        stride=8,
        angle_cols=("roll_win", "pitch_win", "yaw_win"),
        transform=transform,
        verify_files=False,
        max_windows=None,
    )

    val_ds = HeadMountedSequenceDataset(
        recordings=split["val"],
        past_frames=past_frames,
        future_frames=future_frames,
        stride=8,
        angle_cols=("roll_win", "pitch_win", "yaw_win"),
        transform=transform,
        verify_files=False,
        max_windows=None,
    )

    test_ds = HeadMountedSequenceDataset(
        recordings=split["test"],
        past_frames=past_frames,
        future_frames=future_frames,
        stride=8,
        angle_cols=("roll_win", "pitch_win", "yaw_win"),
        transform=transform,
        verify_files=False,
        max_windows=None,
    )

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader


# =========================================================
# Train / Eval loops
# =========================================================
def run_epoch_train(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    scaler,
    accumulation_steps: int = 1,
):
    model.train()

    total_loss = 0.0
    n = 0

    all_preds = []
    all_targets = []

    total_batches = len(loader)

    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader, start=1):
        frames = batch["frames"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with amp.autocast("cuda", enabled=True):
            preds = model(frames)
            loss = loss_fn(preds, targets)
            loss_for_backward = loss / accumulation_steps

        scaler.scale(loss_for_backward).backward()

        should_step = (i % accumulation_steps == 0) or (i == total_batches)

        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        bs = frames.size(0)
        total_loss += loss.item() * bs
        n += bs

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if i % 500 == 0 or i == total_batches:
            print(
                f"[train] batch {i}/{total_batches} | "
                f"loss={loss.item():.5f} | "
                f"accum={accumulation_steps}",
                flush=True,
            )

    avg_loss = total_loss / n

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_preds, all_targets)

    return avg_loss, metrics


@torch.no_grad()
def run_epoch_eval(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    n = 0

    all_preds = []
    all_targets = []

    total_batches = len(loader)

    for i, batch in enumerate(loader, start=1):
        frames = batch["frames"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with amp.autocast("cuda", enabled=True):
            preds = model(frames)
            loss = loss_fn(preds, targets)

        bs = frames.size(0)
        total_loss += loss.item() * bs
        n += bs

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if i % 500 == 0 or i == total_batches:
            print(
                f"[eval] batch {i}/{total_batches} | loss={loss.item():.5f}",
                flush=True
            )

    avg_loss = total_loss / n

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_preds, all_targets)

    return avg_loss, metrics


# =========================================================
# Logging / Saving / Plotting
# =========================================================
def init_history() -> dict:
    return {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_mae_roll_deg": [],
        "train_mae_pitch_deg": [],
        "train_mae_yaw_deg": [],
        "train_mae_mean_deg": [],
        "train_rmse_roll_deg": [],
        "train_rmse_pitch_deg": [],
        "train_rmse_yaw_deg": [],
        "train_rmse_mean_deg": [],
        "train_medae_roll_deg": [],
        "train_medae_pitch_deg": [],
        "train_medae_yaw_deg": [],
        "train_medae_mean_deg": [],
        "val_mae_roll_deg": [],
        "val_mae_pitch_deg": [],
        "val_mae_yaw_deg": [],
        "val_mae_mean_deg": [],
        "val_rmse_roll_deg": [],
        "val_rmse_pitch_deg": [],
        "val_rmse_yaw_deg": [],
        "val_rmse_mean_deg": [],
        "val_medae_roll_deg": [],
        "val_medae_pitch_deg": [],
        "val_medae_yaw_deg": [],
        "val_medae_mean_deg": [],
        "lr_backbone": [],
        "lr_head": [],
        "epoch_time_min": [],
    }


def append_epoch_history(
    history: dict,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_metrics: dict,
    val_metrics: dict,
    lr_backbone: float,
    lr_head: float,
    epoch_time_min: float,
):
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    history["train_mae_roll_deg"].append(train_metrics["mae_roll_deg"])
    history["train_mae_pitch_deg"].append(train_metrics["mae_pitch_deg"])
    history["train_mae_yaw_deg"].append(train_metrics["mae_yaw_deg"])
    history["train_mae_mean_deg"].append(train_metrics["mae_mean_deg"])
    history["train_rmse_roll_deg"].append(train_metrics["rmse_roll_deg"])
    history["train_rmse_pitch_deg"].append(train_metrics["rmse_pitch_deg"])
    history["train_rmse_yaw_deg"].append(train_metrics["rmse_yaw_deg"])
    history["train_rmse_mean_deg"].append(train_metrics["rmse_mean_deg"])
    history["train_medae_roll_deg"].append(train_metrics["medae_roll_deg"])
    history["train_medae_pitch_deg"].append(train_metrics["medae_pitch_deg"])
    history["train_medae_yaw_deg"].append(train_metrics["medae_yaw_deg"])
    history["train_medae_mean_deg"].append(train_metrics["medae_mean_deg"])

    history["val_mae_roll_deg"].append(val_metrics["mae_roll_deg"])
    history["val_mae_pitch_deg"].append(val_metrics["mae_pitch_deg"])
    history["val_mae_yaw_deg"].append(val_metrics["mae_yaw_deg"])
    history["val_mae_mean_deg"].append(val_metrics["mae_mean_deg"])
    history["val_rmse_roll_deg"].append(val_metrics["rmse_roll_deg"])
    history["val_rmse_pitch_deg"].append(val_metrics["rmse_pitch_deg"])
    history["val_rmse_yaw_deg"].append(val_metrics["rmse_yaw_deg"])
    history["val_rmse_mean_deg"].append(val_metrics["rmse_mean_deg"])
    history["val_medae_roll_deg"].append(val_metrics["medae_roll_deg"])
    history["val_medae_pitch_deg"].append(val_metrics["medae_pitch_deg"])
    history["val_medae_yaw_deg"].append(val_metrics["medae_yaw_deg"])
    history["val_medae_mean_deg"].append(val_metrics["medae_mean_deg"])

    history["lr_backbone"].append(lr_backbone)
    history["lr_head"].append(lr_head)
    history["epoch_time_min"].append(epoch_time_min)


def save_history_csv(history: dict, out_dir: Path):
    df = pd.DataFrame(history)
    df.to_csv(out_dir / "history.csv", index=False)


def append_summary_csv(
    summary_csv_path: Path,
    experiment_name: str,
    experiment_value: str,
    best_epoch: int,
    test_loss: float,
    test_metrics: dict,
):
    row = {
        "experiment_name": experiment_name,
        "experiment_value": experiment_value,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_mae_roll_deg": test_metrics["mae_roll_deg"],
        "test_mae_pitch_deg": test_metrics["mae_pitch_deg"],
        "test_mae_yaw_deg": test_metrics["mae_yaw_deg"],
        "test_mae_mean_deg": test_metrics["mae_mean_deg"],
        "test_rmse_roll_deg": test_metrics["rmse_roll_deg"],
        "test_rmse_pitch_deg": test_metrics["rmse_pitch_deg"],
        "test_rmse_yaw_deg": test_metrics["rmse_yaw_deg"],
        "test_rmse_mean_deg": test_metrics["rmse_mean_deg"],
        "test_medae_roll_deg": test_metrics["medae_roll_deg"],
        "test_medae_pitch_deg": test_metrics["medae_pitch_deg"],
        "test_medae_yaw_deg": test_metrics["medae_yaw_deg"],
        "test_medae_mean_deg": test_metrics["medae_mean_deg"],
    }

    row_df = pd.DataFrame([row])

    if summary_csv_path.exists():
        existing = pd.read_csv(summary_csv_path)
        updated = pd.concat([existing, row_df], ignore_index=True)
    else:
        updated = row_df

    updated.to_csv(summary_csv_path, index=False)


def save_plots(history: dict, out_dir: Path):
    epochs = history["epoch"]

    # Plot 1
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_train_vs_val_loss.png", dpi=200)
    plt.close()

    # Plot 2
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_mae_mean_deg"], label="Validation Mean MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Degrees")
    plt.title("Validation Mean MAE per Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_val_mean_mae.png", dpi=200)
    plt.close()

    # Plot 3
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_rmse_mean_deg"], label="Validation Mean RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Degrees")
    plt.title("Validation Mean RMSE per Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_val_mean_rmse.png", dpi=200)
    plt.close()

    # Plot 4
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_medae_mean_deg"], label="Validation Mean MedianAE")
    plt.xlabel("Epoch")
    plt.ylabel("Degrees")
    plt.title("Validation Mean MedianAE per Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_val_mean_medae.png", dpi=200)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    # -----------------------------------------------------
    # Experiment identity
    # -----------------------------------------------------
    EXPERIMENT_NAME = "Final_Full_Dataset"
    EXPERIMENT_VALUE = "Final_Config_All_Recordings"

    PAST_FRAMES = 40
    FUTURE_FRAMES = 5

    
    set_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected")

    device = torch.device("cuda")

    print(f"Experiment: {EXPERIMENT_NAME}", flush=True)
    print(f"Variant: {EXPERIMENT_VALUE}", flush=True)
    print("Torch:", torch.__version__, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    BASE = "/share/e12507305/preprocessed/DERIVED_ET_IMU_head_mounted"

    OUT_DIR = Path(f"runs/{EXPERIMENT_NAME}__{EXPERIMENT_VALUE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_size = 12
    accumulation_steps = 1
    num_workers = 6
    epochs = 50

    train_loader, val_loader, test_loader = make_loaders(
        base_dir=BASE,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=42,
        past_frames=PAST_FRAMES,
        future_frames=FUTURE_FRAMES,
        split_json_path="final_split.json",
    )

    model = ResNetTransformerRegressor( 
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.2,
        pretrained_backbone=True,
        use_cls=True,
        out_dim=4,
    ).to(device)

    freeze_early_resnet_layers(model)

    loss_fn = nn.SmoothL1Loss(beta=0.1)

    trainable_backbone_params = [
        p for p in model.backbone.parameters() if p.requires_grad
    ]
    trainable_backbone_ids = {id(p) for p in trainable_backbone_params}

    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in trainable_backbone_ids
    ]

    base_lr = 1e-4
    backbone_lr = 0.1 * base_lr
    head_lr = base_lr
    
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": head_lr},
        ],
        weight_decay=0.0
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=base_lr * 0.01,
    )

    scaler = amp.GradScaler("cuda", enabled=True)

    early_stopping_patience = 4
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    best_model_path = OUT_DIR / "best_model.pth"
    last_checkpoint_path = OUT_DIR / "last_checkpoint.pth"
    history = init_history()

    start_epoch = 1

    if last_checkpoint_path.exists():
        print("Resuming from last checkpoint...", flush=True)
    
        ckpt = torch.load(last_checkpoint_path, map_location=device)
    
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    
        best_val_loss = ckpt["best_val_loss"]
        best_epoch = ckpt["best_epoch"]
        no_improve = ckpt["no_improve"]
        history = ckpt.get("history", history)
    
        start_epoch = ckpt["epoch"] + 1
    
        print(f"Resuming from epoch {start_epoch}", flush=True)

    if start_epoch > epochs:
        print("Training already completed. Loading best model for test evaluation.", flush=True)
    
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        print(f"\n===== Epoch {epoch:02d} / {epochs} =====", flush=True)

        train_loss, train_metrics = run_epoch_train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
        )

        val_loss, val_metrics = run_epoch_eval(
            model, val_loader, loss_fn, device
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lrs = [group["lr"] for group in optimizer.param_groups]

        append_epoch_history(
            history=history,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr_backbone=current_lrs[0],
            lr_head=current_lrs[1],
            epoch_time_min=epoch_time / 60.0,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_mean_mae={val_metrics['mae_mean_deg']:.3f} | "
            f"val_mean_rmse={val_metrics['rmse_mean_deg']:.3f} | "
            f"val_mean_medae={val_metrics['medae_mean_deg']:.3f} | "
            f"lr_backbone={current_lrs[0]:.2e} | "
            f"lr_head={current_lrs[1]:.2e} | "
            f"time={epoch_time/60:.2f} min",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
        
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                best_model_path,
            )
        else:
            no_improve += 1
        
        
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "history": history,
            },
            last_checkpoint_path,
        )
        
        
        if no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.", flush=True)
            break

    save_history_csv(history, OUT_DIR)
    save_plots(history, OUT_DIR)

    print(f"\nLoading best model from epoch {best_epoch} ...", flush=True)

    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics = run_epoch_eval(
        model, test_loader, loss_fn, device
    )

    print("\nFINAL TEST RESULTS", flush=True)
    print(f"test_loss         = {test_loss:.5f}", flush=True)
    print(f"test_roll_mae     = {test_metrics['mae_roll_deg']:.3f} deg", flush=True)
    print(f"test_pitch_mae    = {test_metrics['mae_pitch_deg']:.3f} deg", flush=True)
    print(f"test_yaw_mae      = {test_metrics['mae_yaw_deg']:.3f} deg", flush=True)
    print(f"test_mean_mae     = {test_metrics['mae_mean_deg']:.3f} deg", flush=True)
    print(f"test_roll_rmse    = {test_metrics['rmse_roll_deg']:.3f} deg", flush=True)
    print(f"test_pitch_rmse   = {test_metrics['rmse_pitch_deg']:.3f} deg", flush=True)
    print(f"test_yaw_rmse     = {test_metrics['rmse_yaw_deg']:.3f} deg", flush=True)
    print(f"test_mean_rmse    = {test_metrics['rmse_mean_deg']:.3f} deg", flush=True)
    print(f"test_roll_medae   = {test_metrics['medae_roll_deg']:.3f} deg", flush=True)
    print(f"test_pitch_medae  = {test_metrics['medae_pitch_deg']:.3f} deg", flush=True)
    print(f"test_yaw_medae    = {test_metrics['medae_yaw_deg']:.3f} deg", flush=True)
    print(f"test_mean_medae   = {test_metrics['medae_mean_deg']:.3f} deg", flush=True)

    summary_csv_path = Path("runs/experiments_summary.csv")
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    append_summary_csv(
        summary_csv_path=summary_csv_path,
        experiment_name=EXPERIMENT_NAME,
        experiment_value=EXPERIMENT_VALUE,
        best_epoch=best_epoch,
        test_loss=test_loss,
        test_metrics=test_metrics,
    )


if __name__ == "__main__":
    main()