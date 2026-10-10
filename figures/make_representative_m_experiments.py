from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ln_rms_activation_family_sweep_raw.csv"
PANEL_OUTPUTS = [
    ROOT / "representative_experiment_a",
    ROOT / "representative_experiment_b",
    ROOT / "representative_experiment_c",
]
LEGEND_OUTPUT = ROOT / "representative_experiments_legend"

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

LN_COLOR = "#0077BB"
RMS_COLOR = "#CC3377"
REFERENCE_COLOR = "#555555"
REPRESENTATIVE_WIDTH = 16

PANELS = [
    ("post_unbounded", "relu"),
    ("affine_pre", "tanh"),
    ("homogeneous_pre", "relu"),
]


def mean_curve(frame: pd.DataFrame, normalization: str) -> tuple[np.ndarray, np.ndarray]:
    curve = (
        frame.loc[
            frame["normalization"] == normalization, ["parameter", "sup_error"]
        ]
        .groupby("parameter", as_index=False)["sup_error"]
        .mean()
        .sort_values("parameter")
    )
    return curve["parameter"].to_numpy(), curve["sup_error"].to_numpy()


def add_reference(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    slope: int,
    linestyle: str,
) -> None:
    anchor = min(3, len(x) - 1)
    reference = y[anchor] * (x / x[anchor]) ** slope
    axis.plot(
        x,
        reference,
        color=REFERENCE_COLOR,
        linestyle=linestyle,
        linewidth=1.0,
        zorder=1,
    )


data = pd.read_csv(DATA)

for panel_index, ((case, activation), output) in enumerate(zip(PANELS, PANEL_OUTPUTS)):
    figure, axis = plt.subplots(figsize=(2.25, 2.0))
    subset = data.loc[
        (data["case"] == case)
        & (data["activation"] == activation)
        & (data["teacher_width"] == REPRESENTATIVE_WIDTH)
    ]
    if subset.empty:
        raise ValueError(
            f"Missing targets at N={REPRESENTATIVE_WIDTH} for {case}/{activation}."
        )
    if subset["teacher_id"].nunique() != 6:
        raise ValueError(f"Expected six targets for {case}/{activation}.")
    x_ln, y_ln = mean_curve(subset, "LayerNorm")
    x_rms, y_rms = mean_curve(subset, "RMSNorm")

    axis.plot(
        x_ln,
        y_ln,
        color=LN_COLOR,
        marker="o",
        markersize=3.0,
        linewidth=1.25,
        label="LayerNorm",
        zorder=3,
    )
    axis.plot(
        x_rms,
        y_rms,
        color=RMS_COLOR,
        marker="s",
        markersize=3.0,
        linewidth=1.25,
        linestyle="--",
        label="RMSNorm",
        zorder=3,
    )

    if panel_index == 0:
        add_reference(axis, x_ln, y_ln, -1, "--")
        add_reference(axis, x_rms, y_rms, -2, ":")
    else:
        common = np.sqrt(y_ln * y_rms)
        add_reference(axis, x_ln, common, -2, ":")

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks([2**1, 2**3, 2**5, 2**7, 2**9])
    axis.set_xticklabels([r"$2^1$", r"$2^3$", r"$2^5$", r"$2^7$", r"$2^9$"])
    axis.grid(True, which="both", color="#D9D9D9", linewidth=0.35, alpha=0.75)
    axis.set_xlabel(r"augmentation magnitude $M$")
    axis.set_ylabel("test sup error (log scale)")
    figure.tight_layout(pad=0.15)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

legend_handles = [
    Line2D([0], [0], color=LN_COLOR, marker="o", markersize=4, linewidth=1.3, label="LayerNorm"),
    Line2D(
        [0],
        [0],
        color=RMS_COLOR,
        marker="s",
        markersize=4,
        linewidth=1.3,
        linestyle="--",
        label="RMSNorm",
    ),
    Line2D([0], [0], color=REFERENCE_COLOR, linewidth=1.0, linestyle="--", label=r"$O(M^{-1})$"),
    Line2D([0], [0], color=REFERENCE_COLOR, linewidth=1.0, linestyle=":", label=r"$O(M^{-2})$"),
]

legend_figure = plt.figure(figsize=(4.2, 0.28))
legend_figure.legend(
    handles=legend_handles,
    loc="center",
    ncol=4,
    frameon=False,
    handlelength=2.4,
    columnspacing=1.6,
)
legend_figure.savefig(
    LEGEND_OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01
)
legend_figure.savefig(
    LEGEND_OUTPUT.with_suffix(".png"), bbox_inches="tight", pad_inches=0.01
)
plt.close(legend_figure)
