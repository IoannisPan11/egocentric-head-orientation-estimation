"""
generate_appendix_rmse_plots.py

Draws the validation Mean RMSE per epoch of all configurations of each experiment, one
figure per experiment, as in Appendix A of the thesis.

Thesis      : Figures A.1-A.12 (Appendix A)
Run on      : TU Wien dataLAB cluster, after the training runs
Input       : `runs/<run folder>/history.csv`, as written by the training scripts
Output      : `appendix_figures/figure_A<n>_<experiment>_rmse_evolution.png` (300 dpi)

What it does
- For each experiment, reads the `val_rmse_mean_deg` column of the runs that the thesis
  compares in the corresponding table of Chapter 5. As there, the reference configuration
  of an experiment is a run of the previous step; for example, "Nearest Neighbor" in
  Figure A.2 is the run `Loss_Function__L1_Beta_0.1`.
- Draws longer curves first, so that shorter curves stay visible on top.
- Saves one figure per experiment. The figures used in the thesis are in `appendix_figures/`.

How to run
- From the folder that contains `runs/`:
    python generate_appendix_rmse_plots.py
- The histories of all runs are also in `results/`; see `analysis/README.md`.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# =========================================================
# Global settings
# =========================================================
METRIC = "val_rmse_mean_deg"

BASE_DIR = Path("runs")
OUTPUT_DIR = Path("appendix_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_rmse_curves(runs: dict[str, Path], title: str, output_file: str) -> None:
    """
    Plot validation mean RMSE evolution across epochs for all configurations
    of a given experiment.
    """
    plt.figure(figsize=(8, 5))

    histories = []

    for label, csv_path in runs.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        history = pd.read_csv(csv_path)

        if METRIC not in history.columns:
            raise ValueError(f"Column '{METRIC}' not found in {csv_path}")

        if "epoch" in history.columns:
            x = history["epoch"]
        else:
            x = range(1, len(history) + 1)

        histories.append((label, x, history[METRIC], len(history)))

    # Plot longer curves first and shorter curves last, so shorter curves remain visible
    histories = sorted(histories, key=lambda item: item[3], reverse=True)

    for label, x, y, n_epochs in histories:
        plt.plot(
            x,
            y,
            linewidth=1.0,
            marker="o",
            markersize=2.5,
            label=label,
            zorder=100 - n_epochs,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation mean RMSE (degrees)")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    max_epoch = max(max(x) for _, x, _, _ in histories)

    if max_epoch <= 15:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    plt.tight_layout()

    out_path = OUTPUT_DIR / output_file
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path.resolve()}")


# =========================================================
# Figure A.1 - Loss Function experiment
# =========================================================
plot_rmse_curves(
    runs={
        "Geodesic loss": BASE_DIR / "Loss_Function__Geodesic" / "history.csv",
        "Smooth L1 loss (β = 0.05)": BASE_DIR / "Loss_Function__L1_Beta_0.05" / "history.csv",
        "Smooth L1 loss (β = 0.1)": BASE_DIR / "Loss_Function__L1_Beta_0.1" / "history.csv",
    },
    title="Loss Function Experiment",
    output_file="figure_A1_loss_function_rmse_evolution.png",
)


# =========================================================
# Figure A.2 - IMU-angle Aggregation experiment
# =========================================================
plot_rmse_curves(
    runs={
        "Nearest Neighbor": BASE_DIR / "Loss_Function__L1_Beta_0.1" / "history.csv",
        "Window 11": BASE_DIR / "IMU_Angle_Experiment__Window_11" / "history.csv",
        "Window 21": BASE_DIR / "IMU_Angle_Experiment__Window_21" / "history.csv",
        "Window 31": BASE_DIR / "IMU_Angle_Experiment__Window_31" / "history.csv",
    },
    title="IMU-Angle Aggregation Experiment",
    output_file="figure_A2_imu_angle_aggregation_rmse_evolution.png",
)


# =========================================================
# Figure A.3 - Temporal Context experiment
# =========================================================
plot_rmse_curves(
    runs={
        "Past 10 - Future 0": BASE_DIR / "Temporal_Contect_Sweep__Past10_Future0" / "history.csv",
        "Past 20 - Future 0": BASE_DIR / "Temporal_Contect_Sweep__Past20_Future0" / "history.csv",
        "Past 20 - Future 5": BASE_DIR / "IMU_Angle_Experiment__Window_31" / "history.csv",
        "Past 40 - Future 5": BASE_DIR / "Temporal_Context_Sweep__Past40_Future5" / "history.csv",
    },
    title="Temporal Context Experiment",
    output_file="figure_A3_temporal_context_rmse_evolution.png",
)


# =========================================================
# Figure A.4 - Batch Size experiment
# =========================================================
plot_rmse_curves(
    runs={
        "Batch size = 12": BASE_DIR / "Batch_Size_12" / "history.csv",
        "Batch size = 24": BASE_DIR / "Temporal_Context_Sweep__Past40_Future5" / "history.csv",
    },
    title="Batch Size Experiment",
    output_file="figure_A4_batch_size_rmse_evolution.png",
)


# =========================================================
# Figure A.5 - Learning Rate experiment
# =========================================================
plot_rmse_curves(
    runs={
        "3 × 10⁻⁵": BASE_DIR / "LR_Sweep__AdamW_diffLR_base3e-5" / "history.csv",
        "1 × 10⁻⁴": BASE_DIR / "Batch_Size_12" / "history.csv",
        "2 × 10⁻⁴": BASE_DIR / "LR_Sweep__AdamW_diffLR_base2e-4" / "history.csv",
        "3 × 10⁻⁴": BASE_DIR / "LR_Sweep__AdamW_diffLR_base3e-4" / "history.csv",
    },
    title="Learning Rate Experiment",
    output_file="figure_A5_learning_rate_rmse_evolution.png",
)


# =========================================================
# Figure A.6 - Dropout experiment
# =========================================================
plot_rmse_curves(
    runs={
        "0.0": BASE_DIR / "Dropout_Sweep__1e-4_dropout00" / "history.csv",
        "0.05": BASE_DIR / "Dropout_Sweep__1e-4_dropout005" / "history.csv",
        "0.1": BASE_DIR / "Batch_Size_12" / "history.csv",
        "0.2": BASE_DIR / "Dropout_Sweep__1e-4_dropout02" / "history.csv",
    },
    title="Dropout Experiment",
    output_file="figure_A6_dropout_rmse_evolution.png",
)


# =========================================================
# Figure A.7 - Weight Decay experiment
# =========================================================
plot_rmse_curves(
    runs={
        "0": BASE_DIR / "Weight_Decay_Sweep__wd_0" / "history.csv",
        "0.001": BASE_DIR / "Weight_Decay_Sweep__wd_0001" / "history.csv",
        "0.01": BASE_DIR / "Dropout_Sweep__1e-4_dropout02" / "history.csv",
        "0.05": BASE_DIR / "Weight_Decay_Sweep__wd_005" / "history.csv",
    },
    title="Weight Decay Experiment",
    output_file="figure_A7_weight_decay_rmse_evolution.png",
)


# =========================================================
# Figure A.8 - Gradient Clipping experiment
# =========================================================
plot_rmse_curves(
    runs={
        "None": BASE_DIR / "Gradient_Clipping_Sweep__gc_none" / "history.csv",
        "0.5": BASE_DIR / "Gradient_Clipping_Sweep__gc_05" / "history.csv",
        "1.0": BASE_DIR / "Weight_Decay_Sweep__wd_0" / "history.csv",
        "5.0": BASE_DIR / "Gradient_Clipping_Sweep__gc_5" / "history.csv",
    },
    title="Gradient Clipping Experiment",
    output_file="figure_A8_gradient_clipping_rmse_evolution.png",
)


# =========================================================
# Figure A.9 - Embedding Dimension experiment
# =========================================================
plot_rmse_curves(
    runs={
        "32": BASE_DIR / "Architecture_Sweep__dmodel_32" / "history.csv",
        "64": BASE_DIR / "Weight_Decay_Sweep__wd_0" / "history.csv",
        "128": BASE_DIR / "Architecture_Sweep__dmodel_128" / "history.csv",
        "256": BASE_DIR / "Architecture_Sweep__dmodel_256" / "history.csv",
    },
    title="Embedding Dimension Experiment",
    output_file="figure_A9_embedding_dimension_rmse_evolution.png",
)


# =========================================================
# Figure A.10 - Transformer Encoder Layers experiment
# =========================================================
plot_rmse_curves(
    runs={
        "1 layer": BASE_DIR / "Architecture_Sweep__layers_1" / "history.csv",
        "2 layers": BASE_DIR / "Weight_Decay_Sweep__wd_0" / "history.csv",
        "3 layers": BASE_DIR / "Architecture_Sweep__layers_3" / "history.csv",
        "4 layers": BASE_DIR / "Architecture_Sweep__layers_4" / "history.csv",
    },
    title="Transformer Encoder Layers Experiment",
    output_file="figure_A10_transformer_layers_rmse_evolution.png",
)

# =========================================================
# Figure A.11 - Attention Heads experiment
# =========================================================
plot_rmse_curves(
    runs={
        "2 attention heads": BASE_DIR / "Architecture_Sweep__head_2" / "history.csv",
        "4 attention heads": BASE_DIR / "Weight_Decay_Sweep__wd_0" / "history.csv",
        "8 attention heads": BASE_DIR / "Architecture_Sweep__head_8" / "history.csv",
    },
    title="Attention Heads Experiment",
    output_file="figure_A11_attention_heads_rmse_evolution.png",
)


# =========================================================
# Figure A.12 - Feed-Forward Dimension experiment
# =========================================================
plot_rmse_curves(
    runs={
        "FFN dimension = 256": BASE_DIR / "Architecture_Sweep__ffn_256" / "history.csv",
        "FFN dimension = 512": BASE_DIR / "Weight_Decay_Sweep__wd_0" / "history.csv",
        "FFN dimension = 1024": BASE_DIR / "Architecture_Sweep__ffn_1024" / "history.csv",
        "FFN dimension = 2048": BASE_DIR / "Architecture_Sweep__ffn_2048" / "history.csv",
    },
    title="Feed-Forward Dimension Experiment",
    output_file="figure_A12_feed_forward_dimension_rmse_evolution.png",
)