"""Illustrate PN constructions for phi(t)=ReLU(t-100)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--test-points", type=int, default=8193)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--teacher-slope", type=float, default=2.0)
    return parser.parse_args()


def shifted_relu(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.maximum(values - threshold, 0.0)


def rms_no_gain(t: np.ndarray, width: int, threshold: float) -> np.ndarray:
    repeats = width - 1
    normalized = np.sqrt(width / (t * t + repeats)) * t
    return shifted_relu(normalized, threshold)


def ln_no_affine(t: np.ndarray, width: int, threshold: float) -> np.ndarray:
    if width < 3 or width % 2 == 0:
        raise ValueError("the symmetric LN construction requires odd width >= 3")
    # N=1 and width=1+2r for the replicated [-1_r,+1_r] construction.
    mean = t / width
    variance = (t * t + width - 1) / width - mean * mean
    normalized = (t - mean) / np.sqrt(variance)
    return shifted_relu(normalized, threshold)


def rms_with_gain(t: np.ndarray, magnitude: float, threshold: float) -> np.ndarray:
    restored = magnitude * t / np.sqrt(t * t + magnitude * magnitude)
    return shifted_relu(restored, threshold)


def ln_with_affine(t: np.ndarray, magnitude: float, threshold: float) -> np.ndarray:
    # For the balanced width-3 lift [t, M-t/2, -M-t/2], affine LN restores
    # lambda_M(t)t with lambda_M(t)=(1+3t^2/(4M^2))^{-1/2}.
    restored = t / np.sqrt(1.0 + 0.75 * t * t / magnitude**2)
    return shifted_relu(restored, threshold)


def odd_widths(values: np.ndarray) -> np.ndarray:
    rounded = np.rint(values).astype(np.int64)
    rounded += (rounded % 2 == 0).astype(np.int64)
    return np.unique(np.maximum(rounded, 3))


def sup_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(prediction - target)))


def main() -> None:
    args = parse_args()
    threshold = args.threshold
    teacher_slope = args.teacher_slope
    x = np.linspace(-1.0, 1.0, args.test_points, dtype=np.float64)
    t = threshold + teacher_slope * x
    target = shifted_relu(t, threshold)

    coordinate_bound_width = threshold**2
    t_max = threshold + abs(teacher_slope)
    rms_turn_on_width = (
        threshold**2 * (t_max**2 - 1.0) / (t_max**2 - threshold**2)
    )
    ln_turn_on_width = (
        t_max**2 * (threshold**2 + 1.0) / (t_max**2 - threshold**2)
    )
    construction_turn_on_width = max(rms_turn_on_width, ln_turn_on_width)

    base_widths = np.geomspace(2.0, 1.0e8, 90)
    rms_widths = np.unique(
        np.concatenate(
            (
                np.rint(base_widths).astype(np.int64),
                np.array(
                    [
                        max(2, int(coordinate_bound_width) - 1),
                        max(2, int(coordinate_bound_width)),
                        int(coordinate_bound_width) + 1,
                        int(coordinate_bound_width) + 2,
                        int(np.ceil(rms_turn_on_width)) + 1,
                        int(10 * coordinate_bound_width) + 1,
                    ]
                ),
            )
        )
    )
    rms_widths = rms_widths[rms_widths >= 2]
    ln_widths = odd_widths(
        np.concatenate(
            (
                base_widths,
                np.array(
                    [
                        max(3, int(coordinate_bound_width) - 1),
                        int(coordinate_bound_width) + 1,
                        int(coordinate_bound_width) + 3,
                        int(np.ceil(ln_turn_on_width)) + 1,
                        int(10 * coordinate_bound_width) + 1,
                    ]
                ),
            )
        )
    )
    magnitudes = np.unique(
        np.concatenate(
            (
                np.geomspace(1.0, 1.0e7, 90),
                np.array([100.0, 1.0e3, 1.0e4, 1.0e5]),
            )
        )
    )

    rows: list[dict] = []
    rms_errors = []
    for width in rms_widths:
        error = sup_error(rms_no_gain(t, int(width), threshold), target)
        rms_errors.append(error)
        rows.append(
            {
                "construction": "RMSNorm no-gain",
                "control": "width",
                "value": int(width),
                "sup_error": error,
            }
        )
    ln_errors = []
    for width in ln_widths:
        error = sup_error(ln_no_affine(t, int(width), threshold), target)
        ln_errors.append(error)
        rows.append(
            {
                "construction": "LayerNorm no-affine",
                "control": "width",
                "value": int(width),
                "sup_error": error,
            }
        )

    rms_gain_errors = []
    ln_affine_errors = []
    for magnitude in magnitudes:
        rms_error = sup_error(rms_with_gain(t, float(magnitude), threshold), target)
        ln_error = sup_error(ln_with_affine(t, float(magnitude), threshold), target)
        rms_gain_errors.append(rms_error)
        ln_affine_errors.append(ln_error)
        rows.extend(
            (
                {
                    "construction": "RMSNorm with gain",
                    "control": "magnitude",
                    "value": float(magnitude),
                    "sup_error": rms_error,
                },
                {
                    "construction": "LayerNorm with affine",
                    "control": "magnitude",
                    "value": float(magnitude),
                    "sup_error": ln_error,
                },
            )
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "legend.fontsize": 7.7,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.3))
    colors = ["#0077BB", "#EE3377", "#009988", "#CC3311"]
    selected_widths = [
        int(coordinate_bound_width) + 1,
        int(10 * coordinate_bound_width) + 1,
        int(100 * coordinate_bound_width) + 1,
        int(10000 * coordinate_bound_width) + 1,
    ]
    selected_widths = [width if width % 2 == 1 else width + 1 for width in selected_widths]

    ax = axes[0, 0]
    ax.plot(
        x,
        target,
        color="black",
        linewidth=1.5,
        label="one-hidden-layer MLP target",
    )
    for color, width in zip(colors, selected_widths):
        ax.plot(
            x,
            rms_no_gain(t, width, threshold),
            color=color,
            linewidth=1.0,
            label=rf"$d={width:,}$",
        )
    ax.set_title(r"(a) RMS-PN without affine parameters")
    ax.set_xlabel(r"input $x$")
    ax.set_ylabel("network output")
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    ax.plot(
        x,
        target,
        color="black",
        linewidth=1.5,
        label="one-hidden-layer MLP target",
    )
    for color, width in zip(colors, selected_widths):
        ax.plot(
            x,
            ln_no_affine(t, width, threshold),
            color=color,
            linewidth=1.0,
            label=rf"$d={width:,}$",
        )
    ax.set_title(r"(b) LN-PN without affine parameters")
    ax.set_xlabel(r"input $x$")
    ax.set_ylabel("network output")
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 0]
    ax.plot(
        rms_widths,
        rms_errors,
        color="#0077BB",
        linewidth=1.25,
        label="RMSNorm construction",
    )
    ax.plot(
        ln_widths,
        ln_errors,
        color="#EE3377",
        linestyle="-.",
        linewidth=1.25,
        label="LayerNorm construction",
    )
    ax.axvline(coordinate_bound_width, color="#555555", linestyle=":", linewidth=1.0)
    ax.text(
        1.15 * coordinate_bound_width,
        0.65 * float(np.max(target)),
        rf"coordinate-bound threshold $d\approx{coordinate_bound_width:.0f}$",
        color="#555555",
        fontsize=7.5,
    )
    ax.axvline(construction_turn_on_width, color="#996633", linestyle=":", linewidth=1.0)
    ax.text(
        1.15 * construction_turn_on_width,
        0.2 * float(np.max(target)),
        rf"this construction turns on at $d\approx{construction_turn_on_width:.0f}$",
        color="#996633",
        fontsize=7.5,
    )
    reference_widths = np.asarray(rms_widths[-20:], dtype=float)
    reference_error = rms_errors[-1] * rms_widths[-1] / reference_widths
    ax.plot(
        reference_widths,
        reference_error,
        color="#444444",
        linestyle="--",
        linewidth=0.9,
        label=r"reference $O(d^{-1})$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"(c) No-gain/no-affine error versus normalized width")
    ax.set_xlabel(r"normalized width $d$")
    ax.set_ylabel("test sup error")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.28)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1, 1]
    ax.plot(
        magnitudes,
        rms_gain_errors,
        color="#0077BB",
        linewidth=1.25,
        label="RMSNorm with affine parameters (width 2)",
    )
    ax.plot(
        magnitudes,
        ln_affine_errors,
        color="#EE3377",
        linestyle="-.",
        linewidth=1.25,
        label="LayerNorm with affine (width 3)",
    )
    reference_magnitudes = magnitudes[-20:]
    reference_error = rms_gain_errors[-1] * (magnitudes[-1] / reference_magnitudes) ** 2
    ax.plot(
        reference_magnitudes,
        reference_error,
        color="#444444",
        linestyle="--",
        linewidth=0.9,
        label=r"reference $O(M^{-2})$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"(d) Affine parameters remove the width barrier")
    ax.set_xlabel(r"augmentation magnitude $M$")
    ax.set_ylabel("test sup error")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.28)
    ax.legend(frameon=False)

    figure.suptitle(
        rf"Shifted-ReLU activation $\phi(t)=\operatorname{{ReLU}}(t-{threshold:g})$; "
        rf"target $f(x)=\operatorname{{ReLU}}({teacher_slope:g}x)$",
        y=1.01,
        fontsize=9.5,
    )
    figure.tight_layout(w_pad=1.0, h_pad=1.2)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    pd.DataFrame(rows).to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_raw.csv"),
        index=False,
    )
    print(
        {
            "target_sup_norm": float(np.max(np.abs(target))),
            "coordinate_bound_width": coordinate_bound_width,
            "rms_turn_on_width_strictly_above": rms_turn_on_width,
            "ln_turn_on_width_strictly_above": ln_turn_on_width,
            "rms_error_at_bound": sup_error(
                rms_no_gain(t, max(2, int(coordinate_bound_width)), threshold), target
            ),
            "ln_error_at_bound": sup_error(
                ln_no_affine(t, int(coordinate_bound_width) + 1, threshold), target
            ),
            "rms_error_at_width_1e8": sup_error(
                rms_no_gain(t, 100000000, threshold), target
            ),
            "ln_error_at_width_100000001": sup_error(
                ln_no_affine(t, 100000001, threshold), target
            ),
        }
    )


if __name__ == "__main__":
    main()
