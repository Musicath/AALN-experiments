"""Plot construction-only LN/RMSNorm sweeps for three activation families."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from theory_core import (
    PARAMETERS,
    TEACHERS,
    WIDTHS,
    Teacher,
    activate,
    exact_ln,
    make_teacher_parameters,
    two_anchor_values,
)


FAMILIES = (
    (
        "post_unbounded",
        "AN, unbounded",
        "M",
        -1,
        -2,
        (
            ("relu", "ReLU"),
            ("x_sin_x", r"$x\,\sin x$"),
            ("exponential", r"$e^x$"),
        ),
    ),
    (
        "post_general",
        "AN, bounded/general",
        "r",
        -1,
        -1,
        (
            ("tanh", r"$\tanh x$"),
            ("sinc", r"$\sin x/x$"),
            ("gaussian", r"$e^{-x^2}$"),
        ),
    ),
    (
        "affine_pre",
        "PN with affine parameters",
        "M",
        -2,
        -2,
        (
            ("tanh", r"$\tanh x$"),
            ("sinc", r"$\sin x/x$"),
            ("gaussian", r"$e^{-x^2}$"),
        ),
    ),
    (
        "homogeneous_pre",
        "PN, homogeneous",
        "M",
        -2,
        -2,
        (
            ("relu", "ReLU"),
            ("leaky_relu", "Leaky ReLU (0.01)"),
            ("absolute", r"$|x|$"),
        ),
    ),
    (
        "general_pre",
        "PN, general",
        "r",
        -1,
        -1,
        (
            ("tanh", r"$\tanh x$"),
            ("sinc", r"$\sin x/x$"),
            ("gaussian", r"$e^{-x^2}$"),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--test-points", type=int, default=8193)
    return parser.parse_args()


def ln_prediction(
    case: str,
    activation: str,
    parameter: int,
    teacher: Teacher,
    x: torch.Tensor,
) -> torch.Tensor:
    z = teacher.preactivation(x)
    readout = teacher.readout
    output_bias = teacher.output_bias
    width = z.shape[1]

    if case == "post_unbounded":
        features = activate(z, activation)
        zero = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        magnitude = torch.full_like(zero, float(parameter))
        augmented = torch.cat((features, zero, magnitude), dim=1)
        augmented_readout = torch.cat(
            (
                readout,
                -readout.sum().reshape(1),
                torch.zeros(1, dtype=x.dtype, device=x.device),
            )
        )
        scale = parameter * math.sqrt(width + 1) / (width + 2)
        return scale * (exact_ln(augmented) @ augmented_readout[:, None]) + output_bias

    if case == "post_general":
        repeats = parameter
        features = activate(z, activation)
        p_value, q_value = two_anchor_values(activation)
        p_values = torch.full(
            (x.shape[0], repeats), p_value, dtype=x.dtype, device=x.device
        )
        q_values = torch.full(
            (x.shape[0], repeats), q_value, dtype=x.dtype, device=x.device
        )
        augmented = torch.cat((features, p_values, q_values), dim=1)
        readout_sum = readout.sum()
        augmented_readout = torch.zeros(
            width + 2 * repeats, dtype=x.dtype, device=x.device
        )
        augmented_readout[:width] = readout
        augmented_readout[width] = q_value / (p_value - q_value) * readout_sum
        augmented_readout[width + repeats] = (
            -p_value / (p_value - q_value) * readout_sum
        )
        total_width = width + 2 * repeats
        baseline_variance = (
            repeats * (p_value**2 + q_value**2) / total_width
            - repeats**2 * (p_value + q_value) ** 2 / total_width**2
        )
        return (
            math.sqrt(baseline_variance)
            * (exact_ln(augmented) @ augmented_readout[:, None])
            + output_bias
        )

    if case in {"affine_pre", "homogeneous_pre"}:
        magnitude = float(parameter)
        coordinate_sum = z.sum(dim=1, keepdim=True)
        lifted = torch.cat(
            (z, magnitude - 0.5 * coordinate_sum, -magnitude - 0.5 * coordinate_sum),
            dim=1,
        )
        normalized_signal = exact_ln(lifted)[:, :width]
        scale = math.sqrt(2.0) * magnitude / math.sqrt(width + 2)
        if case == "affine_pre":
            return (
                activate(scale * normalized_signal, activation)
                @ readout[:, None]
                + output_bias
            )
        return (
            scale * (activate(normalized_signal, activation) @ readout[:, None])
            + output_bias
        )

    if case == "general_pre":
        repeats = parameter
        minus = -torch.ones((x.shape[0], repeats), dtype=x.dtype, device=x.device)
        plus = torch.ones((x.shape[0], repeats), dtype=x.dtype, device=x.device)
        augmented = torch.cat((z, minus, plus), dim=1)
        features = activate(exact_ln(augmented)[:, :width], activation)
        return features @ readout[:, None] + output_bias

    raise ValueError(case)


def rms_prediction(
    case: str,
    activation: str,
    parameter: int,
    teacher: Teacher,
    x: torch.Tensor,
) -> torch.Tensor:
    z = teacher.preactivation(x)
    readout = teacher.readout
    output_bias = teacher.output_bias

    if case == "post_unbounded":
        features = activate(z, activation)
        signal = features @ readout[:, None]
        denominator = torch.sqrt(
            (features * features).sum(dim=1, keepdim=True) + parameter**2
        )
        return parameter * signal / denominator + output_bias

    if case == "post_general":
        features = activate(z, activation)
        signal = features @ readout[:, None]
        value, _ = two_anchor_values(activation)
        denominator = torch.sqrt(
            (features * features).sum(dim=1, keepdim=True)
            + parameter * value**2
        )
        return abs(value) * math.sqrt(parameter) * signal / denominator + output_bias

    if case in {"affine_pre", "homogeneous_pre"}:
        denominator = torch.sqrt((z * z).sum(dim=1, keepdim=True) + parameter**2)
        restored = parameter * z / denominator
        return activate(restored, activation) @ readout[:, None] + output_bias

    if case == "general_pre":
        width = z.shape[1]
        denominator = torch.sqrt((z * z).sum(dim=1, keepdim=True) + parameter)
        normalized = math.sqrt(width + parameter) * z / denominator
        return activate(normalized, activation) @ readout[:, None] + output_bias

    raise ValueError(case)


def compute_rows(seed: int, test_points: int) -> pd.DataFrame:
    x = torch.linspace(-1.0, 1.0, test_points, dtype=torch.float64)[:, None]
    rows: list[dict] = []
    for case, family, parameter_name, _, _, activations in FAMILIES:
        for activation, activation_label in activations:
            for width in WIDTHS:
                for teacher_id in TEACHERS:
                    teacher_seed = seed + 1000 * teacher_id
                    parameters = make_teacher_parameters(teacher_seed, width)
                    teacher = Teacher(*parameters, activation, torch.float64)
                    with torch.no_grad():
                        target = teacher(x)
                        for parameter in PARAMETERS:
                            for normalization, prediction_fn in (
                                ("LayerNorm", ln_prediction),
                                ("RMSNorm", rms_prediction),
                            ):
                                prediction = prediction_fn(
                                    case, activation, parameter, teacher, x
                                )
                                error = float((prediction - target).abs().max().item())
                                rows.append(
                                    {
                                        "normalization": normalization,
                                        "case": case,
                                        "family": family,
                                        "activation": activation,
                                        "activation_label": activation_label,
                                        "parameter_name": parameter_name,
                                        "teacher_width": width,
                                        "teacher_id": teacher_id,
                                        "parameter": parameter,
                                        "sup_error": error,
                                    }
                                )
    return pd.DataFrame(rows)


def summarize(values: pd.DataFrame) -> pd.DataFrame:
    return (
        values.groupby("parameter")["sup_error"]
        .agg(
            count="count",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
        .sort_values("parameter")
    )


def tail_slopes(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = [
        "normalization",
        "case",
        "family",
        "activation",
        "teacher_width",
        "teacher_id",
    ]
    for key, group in values.groupby(keys):
        tail = group.sort_values("parameter").tail(6)
        slope = float(
            np.polyfit(
                np.log2(tail["parameter"].to_numpy(dtype=float)),
                np.log2(tail["sup_error"].to_numpy(dtype=float)),
                1,
            )[0]
        )
        rows.append({**dict(zip(keys, key)), "tail_slope": slope})
    return pd.DataFrame(rows)


def plot(rows: pd.DataFrame, output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.2,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    # The appendix places the full 5-by-3 sweep at text width.  A compact
    # portrait layout keeps every panel readable while allowing the figure to
    # share a page with its short interpretation instead of forcing an almost
    # empty preceding page.
    figure, axes = plt.subplots(len(FAMILIES), 3, figsize=(7.15, 7.0))
    colors = {"LayerNorm": "#0077BB", "RMSNorm": "#EE3377"}
    markers = {"LayerNorm": "o", "RMSNorm": "s"}
    linestyles = {"LayerNorm": "-", "RMSNorm": "-."}

    for row_index, (
        case,
        family,
        parameter_name,
        ln_order,
        rms_order,
        activations,
    ) in enumerate(FAMILIES):
        for column_index, (activation, activation_label) in enumerate(activations):
            ax = axes[row_index, column_index]
            panel = rows[(rows["case"] == case) & (rows["activation"] == activation)]
            summaries: dict[str, pd.DataFrame] = {}
            for normalization in ("LayerNorm", "RMSNorm"):
                summary = summarize(panel[panel["normalization"] == normalization])
                summaries[normalization] = summary
                x = summary["parameter"].to_numpy(dtype=float)
                median = summary["median"].to_numpy(dtype=float)
                q25 = summary["q25"].to_numpy(dtype=float)
                q75 = summary["q75"].to_numpy(dtype=float)
                ax.fill_between(
                    x,
                    q25,
                    q75,
                    color=colors[normalization],
                    alpha=0.10,
                    linewidth=0,
                )
                ax.plot(
                    x,
                    median,
                    color=colors[normalization],
                    marker=markers[normalization],
                    linestyle=linestyles[normalization],
                    markersize=2.6,
                    linewidth=1.15,
                    label=f"{normalization} construction",
                )

            x = summaries["LayerNorm"]["parameter"].to_numpy(dtype=float)
            ln_endpoint = float(summaries["LayerNorm"]["median"].iloc[-1])
            ax.plot(
                x,
                ln_endpoint * (x / x[-1]) ** ln_order,
                color="#4A4A4A",
                linestyle="--",
                linewidth=0.8,
                label="LayerNorm/common predicted rate",
            )
            if rms_order != ln_order:
                rms_endpoint = float(summaries["RMSNorm"]["median"].iloc[-1])
                ax.plot(
                    x,
                    rms_endpoint * (x / x[-1]) ** rms_order,
                    color="#4A4A4A",
                    linestyle=":",
                    linewidth=0.9,
                    label="sharper RMSNorm rate (unbounded AN)",
                )

            panel_letter = chr(97 + 3 * row_index + column_index)
            ax.set_title(f"({panel_letter}) {family} + {activation_label}", pad=2)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="both", linewidth=0.3, alpha=0.26)
            parameter_description = (
                "augmentation magnitude" if parameter_name == "M"
                else "augmentation repetitions"
            )
            ax.set_xlabel(f"{parameter_description} ${parameter_name}$")
            if column_index == 0:
                ax.set_ylabel("test sup error")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=len(labels),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 1), w_pad=0.7, h_pad=0.55)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_compact(rows: pd.DataFrame, output_prefix: Path) -> None:
    """Plot a main-text summary with three activations in each theorem panel."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.75))
    axes_flat = axes.ravel()
    activation_colors = ("#0077BB", "#EE7733", "#009988")
    norm_styles = {
        "LayerNorm": ("-", "o"),
        "RMSNorm": ("-.", "s"),
    }

    for panel_index, (
        case,
        family,
        parameter_name,
        ln_order,
        rms_order,
        activations,
    ) in enumerate(FAMILIES):
        ax = axes_flat[panel_index]
        endpoint_values: dict[str, list[float]] = {
            "LayerNorm": [],
            "RMSNorm": [],
        }
        x_reference: np.ndarray | None = None
        for activation_index, (activation, activation_label) in enumerate(activations):
            panel = rows[(rows["case"] == case) & (rows["activation"] == activation)]
            color = activation_colors[activation_index]
            for normalization in ("LayerNorm", "RMSNorm"):
                summary = summarize(panel[panel["normalization"] == normalization])
                x = summary["parameter"].to_numpy(dtype=float)
                median = summary["median"].to_numpy(dtype=float)
                linestyle, marker = norm_styles[normalization]
                ax.plot(
                    x,
                    median,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=2.3,
                    linewidth=1.05,
                    label=activation_label if normalization == "LayerNorm" else None,
                )
                endpoint_values[normalization].append(float(median[-1]))
                x_reference = x

        assert x_reference is not None
        ln_endpoint = float(np.median(endpoint_values["LayerNorm"]))
        ax.plot(
            x_reference,
            ln_endpoint * (x_reference / x_reference[-1]) ** ln_order,
            color="#4A4A4A",
            linestyle="--",
            linewidth=0.8,
        )
        if rms_order != ln_order:
            rms_endpoint = float(np.median(endpoint_values["RMSNorm"]))
            ax.plot(
                x_reference,
                rms_endpoint * (x_reference / x_reference[-1]) ** rms_order,
                color="#4A4A4A",
                linestyle=":",
                linewidth=0.9,
            )

        panel_letter = chr(97 + panel_index)
        ax.set_title(f"({panel_letter}) {family}", pad=3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.26)
        parameter_description = (
            "augmentation magnitude" if parameter_name == "M"
            else "augmentation repetitions"
        )
        ax.set_xlabel(f"{parameter_description} ${parameter_name}$")
        ax.set_ylabel("test sup error")

    # Panel (f): fixed shifted-ReLU counterexamples.  For each threshold a, at
    # normalized width d <= a^2 every coordinate is at most a, so all hidden
    # activations vanish.  The displayed no-affine PN constructions become
    # accurate once d is sufficiently large.
    lower_axis = axes_flat[-1]
    thresholds = (10.0, 30.0, 100.0)
    teacher_slope = 2.0
    x_grid = np.linspace(-1.0, 1.0, 8193, dtype=np.float64)

    width_grid = np.unique(
        np.concatenate(
            (
                np.rint(np.geomspace(2.0, 1.0e10, 92)).astype(np.int64),
                np.array(
                    [99, 100, 101, 899, 900, 901, 9999, 10000, 10001],
                    dtype=np.int64,
                ),
            )
        )
    )
    rms_widths = width_grid[width_grid >= 2]
    ln_widths = width_grid[width_grid >= 3]
    ln_widths = np.unique(ln_widths + (ln_widths % 2 == 0).astype(np.int64))

    for threshold, color in zip(thresholds, activation_colors):
        preactivation = threshold + teacher_slope * x_grid
        target = np.maximum(preactivation - threshold, 0.0)

        rms_errors = []
        for width in rms_widths:
            normalized = (
                np.sqrt(width / (preactivation**2 + width - 1)) * preactivation
            )
            prediction = np.maximum(normalized - threshold, 0.0)
            rms_errors.append(float(np.max(np.abs(prediction - target))))

        ln_errors = []
        for width in ln_widths:
            mean = preactivation / width
            variance = (preactivation**2 + width - 1) / width - mean**2
            normalized = (preactivation - mean) / np.sqrt(variance)
            prediction = np.maximum(normalized - threshold, 0.0)
            ln_errors.append(float(np.max(np.abs(prediction - target))))

        lower_axis.plot(
            rms_widths,
            rms_errors,
            color=color,
            linestyle="-.",
            marker="s",
            markevery=12,
            markersize=2.1,
            linewidth=1.0,
        )
        lower_axis.plot(
            ln_widths,
            ln_errors,
            color=color,
            linestyle="-",
            marker="o",
            markevery=12,
            markersize=2.1,
            linewidth=1.0,
        )
        lower_axis.axvline(
            threshold**2, color=color, linestyle=":", linewidth=0.65, alpha=0.8
        )

    lower_axis.text(3.0, 0.55, r"$d\leq a^2$: constant only", fontsize=6.2)
    lower_axis.set_title(r"(f) Shifted-ReLU width barriers", pad=3)
    lower_axis.set_xscale("log")
    lower_axis.set_yscale("log")
    lower_axis.set_xlim(2.0, 1.0e10)
    lower_axis.set_ylim(3.0e-8, 3.0)
    lower_axis.set_xlabel(r"normalized width $d$")
    lower_axis.set_ylabel("uniform error")
    lower_axis.grid(True, which="both", linewidth=0.3, alpha=0.26)

    # The activation names differ by regime, so describe colors positionally.
    activation_handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.5, label=f"activation {index}")
        for index, color in enumerate(activation_colors, start=1)
    ]
    norm_handles = [
        plt.Line2D([0], [0], color="#222222", linestyle="-", marker="o", label="LayerNorm"),
        plt.Line2D([0], [0], color="#222222", linestyle="-.", marker="s", label="RMSNorm"),
        plt.Line2D([0], [0], color="#4A4A4A", linestyle="--", label="predicted order"),
    ]
    figure.legend(
        handles=activation_handles + norm_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        frameon=False,
        ncol=6,
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1), w_pad=0.8, h_pad=0.9)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = compute_rows(args.seed, args.test_points)
    plot(rows, args.output_prefix)
    plot_compact(
        rows,
        args.output_prefix.with_name(args.output_prefix.name + "_compact"),
    )
    rows.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_raw.csv"), index=False)
    slopes = tail_slopes(rows)
    slopes.to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_tail_slopes.csv"),
        index=False,
    )
    slope_summary = (
        slopes.groupby(["normalization", "case", "family", "activation"])[
            "tail_slope"
        ]
        .agg(
            count="count",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    slope_summary.to_csv(
        args.output_prefix.with_name(
            args.output_prefix.name + "_tail_slope_summary.csv"
        ),
        index=False,
    )
    print(slope_summary.to_string(index=False))


if __name__ == "__main__":
    main()
