"""Test shifted-ReLU PN constructions on random shallow MLP teachers.

The teacher preactivations are t(x) = threshold + z(x), where z is sampled
exactly as in run_paper_grid.py.  Hence ReLU(t-threshold) = ReLU(z), so every
teacher is an ordinary random ReLU MLP written in shifted-ReLU coordinates.
No optimization or training is used in this experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from theory_core import make_teacher_parameters


WIDTHS = (4, 8, 16, 32)
TEACHERS = tuple(range(6))
BASE_SEED = 20260911


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--test-points", type=int, default=8193)
    return parser.parse_args()


def shifted_relu(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.maximum(values - threshold, 0.0)


def rms_no_gain(t: np.ndarray, total_width: int, threshold: float) -> np.ndarray:
    teacher_width = t.shape[1]
    repeats = total_width - teacher_width
    if repeats < 1:
        raise ValueError("RMS construction requires at least one auxiliary coordinate")
    restored = np.sqrt(total_width / (np.sum(t * t, axis=1, keepdims=True) + repeats)) * t
    return shifted_relu(restored, threshold)


def ln_no_affine(t: np.ndarray, total_width: int, threshold: float) -> np.ndarray:
    teacher_width = t.shape[1]
    difference = total_width - teacher_width
    if difference < 2 or difference % 2:
        raise ValueError("LN construction requires total_width = N + 2r")
    repeats = difference // 2
    coordinate_sum = np.sum(t, axis=1, keepdims=True)
    variance = (
        (np.sum(t * t, axis=1, keepdims=True) + 2.0 * repeats) / total_width
        - (coordinate_sum / total_width) ** 2
    )
    restored = (t - coordinate_sum / total_width) / np.sqrt(variance)
    return shifted_relu(restored, threshold)


def rms_with_gain(t: np.ndarray, magnitude: float, threshold: float) -> np.ndarray:
    norm_squared = np.sum(t * t, axis=1, keepdims=True)
    restored = magnitude * t / np.sqrt(norm_squared + magnitude * magnitude)
    return shifted_relu(restored, threshold)


def ln_with_affine(t: np.ndarray, magnitude: float, threshold: float) -> np.ndarray:
    norm_squared = np.sum(t * t, axis=1, keepdims=True)
    coordinate_sum = np.sum(t, axis=1, keepdims=True)
    denominator = np.sqrt(
        norm_squared + 2.0 * magnitude * magnitude + 0.5 * coordinate_sum**2
    )
    restored = np.sqrt(2.0) * magnitude * t / denominator
    return shifted_relu(restored, threshold)


def sup_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(prediction - target)))


def teacher_data(
    x: np.ndarray, width: int, teacher_id: int, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = BASE_SEED + 1000 * teacher_id
    first_weight, first_bias, readout, _ = make_teacher_parameters(seed, width)
    z = x @ first_weight.T + first_bias[None, :]
    t = threshold + z
    target = shifted_relu(t, threshold) @ readout
    return t, readout, target


def even_width_grid(minimum: int) -> np.ndarray:
    base = 2 * np.rint(np.geomspace(max(34, minimum) / 2.0, 5.0e6, 90)).astype(np.int64)
    diagnostic = np.array(
        [34, 40, 64, 80, 90, 98, 100, 102, 110, 128, 160, 200, 256, 330,
         512, 1000, 10000, 100000, 1000000, 10000000],
        dtype=np.int64,
    )
    values = np.unique(np.concatenate((base, diagnostic)))
    return values[values >= minimum]


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["construction", "control", "value"], as_index=False)
        .agg(
            median_relative_sup_error=("relative_sup_error", "median"),
            q25_relative_sup_error=("relative_sup_error", lambda x: x.quantile(0.25)),
            q75_relative_sup_error=("relative_sup_error", lambda x: x.quantile(0.75)),
            min_relative_sup_error=("relative_sup_error", "min"),
            max_relative_sup_error=("relative_sup_error", "max"),
            teacher_count=("relative_sup_error", "count"),
        )
        .sort_values(["control", "construction", "value"])
    )


def plot_band(
    ax: plt.Axes,
    frame: pd.DataFrame,
    construction: str,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    selected = frame[frame["construction"] == construction].sort_values("value")
    x_values = selected["value"].to_numpy(dtype=float)
    median = selected["median_relative_sup_error"].to_numpy(dtype=float)
    q25 = selected["q25_relative_sup_error"].to_numpy(dtype=float)
    q75 = selected["q75_relative_sup_error"].to_numpy(dtype=float)
    ax.plot(x_values, median, color=color, linestyle=linestyle, linewidth=1.35, label=label)
    ax.fill_between(x_values, q25, q75, color=color, alpha=0.16, linewidth=0)


def main() -> None:
    args = parse_args()
    if args.threshold <= 0:
        raise ValueError("threshold must be positive")
    if args.test_points < 3:
        raise ValueError("test-points must be at least 3")

    x = np.linspace(-1.0, 1.0, args.test_points, dtype=np.float64)[:, None]
    total_widths = even_width_grid(max(WIDTHS) + 2)
    magnitudes = np.unique(
        np.concatenate((np.geomspace(1.0, 1.0e7, 90), np.array([10.0, 100.0, 1.0e3, 1.0e4])))
    )

    rows: list[dict[str, float | int | str]] = []
    example: dict[str, np.ndarray] = {}
    example_width = 8
    example_teacher = 0
    selected_example_widths = (102, 512, 10000, 1000000)

    for width in WIDTHS:
        for teacher_id in TEACHERS:
            t, readout, target = teacher_data(x, width, teacher_id, args.threshold)
            target_scale = float(np.max(np.abs(target)))
            if target_scale <= 1.0e-12:
                raise RuntimeError(f"degenerate teacher N={width}, id={teacher_id}")

            if width == example_width and teacher_id == example_teacher:
                example["target"] = target

            for total_width in total_widths:
                if total_width <= width:
                    continue
                rms_prediction = rms_no_gain(t, int(total_width), args.threshold) @ readout
                ln_prediction = ln_no_affine(t, int(total_width), args.threshold) @ readout
                for construction, prediction in (
                    ("RMS-PN no gain", rms_prediction),
                    ("LN-PN no affine", ln_prediction),
                ):
                    error = sup_error(prediction, target)
                    rows.append(
                        {
                            "construction": construction,
                            "control": "total_width",
                            "value": int(total_width),
                            "teacher_width": width,
                            "teacher_id": teacher_id,
                            "teacher_seed": BASE_SEED + 1000 * teacher_id,
                            "target_sup_norm": target_scale,
                            "sup_error": error,
                            "relative_sup_error": error / target_scale,
                        }
                    )

                if width == example_width and teacher_id == example_teacher and total_width in selected_example_widths:
                    example[f"rms_{total_width}"] = rms_prediction
                    example[f"ln_{total_width}"] = ln_prediction

            for magnitude in magnitudes:
                rms_prediction = rms_with_gain(t, float(magnitude), args.threshold) @ readout
                ln_prediction = ln_with_affine(t, float(magnitude), args.threshold) @ readout
                for construction, prediction in (
                    ("RMS-PN with gain", rms_prediction),
                    ("LN-PN with affine", ln_prediction),
                ):
                    error = sup_error(prediction, target)
                    rows.append(
                        {
                            "construction": construction,
                            "control": "magnitude",
                            "value": float(magnitude),
                            "teacher_width": width,
                            "teacher_id": teacher_id,
                            "teacher_seed": BASE_SEED + 1000 * teacher_id,
                            "target_sup_norm": target_scale,
                            "sup_error": error,
                            "relative_sup_error": error / target_scale,
                        }
                    )

    raw = pd.DataFrame(rows)
    summary = summarize(raw)

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
    x_flat = x[:, 0]

    for axis, prefix, title in (
        (axes[0, 0], "rms", r"(a) Random target: RMS-PN without affine"),
        (axes[0, 1], "ln", r"(b) Random target: LN-PN without affine"),
    ):
        axis.plot(
            x_flat,
            example["target"],
            color="black",
            linewidth=1.5,
            label="one-hidden-layer MLP target",
        )
        for color, total_width in zip(colors, selected_example_widths):
            axis.plot(
                x_flat,
                example[f"{prefix}_{total_width}"],
                color=color,
                linewidth=1.0,
                label=rf"$d={total_width:,}$",
            )
        axis.set_title(title)
        axis.set_xlabel(r"input $x$")
        axis.set_ylabel("network output")
        axis.grid(True, linewidth=0.35, alpha=0.28)
        axis.legend(frameon=False, ncol=2)

    width_summary = summary[summary["control"] == "total_width"]
    ax = axes[1, 0]
    plot_band(ax, width_summary, "RMS-PN no gain", "#0077BB", "RMS-PN median (IQR)")
    plot_band(ax, width_summary, "LN-PN no affine", "#EE3377", "LN-PN median (IQR)", "-.")
    ax.axvline(args.threshold**2, color="#555555", linestyle=":", linewidth=1.0)
    ax.text(1.12 * args.threshold**2, 0.52, rf"coordinate bound $d={args.threshold**2:g}$", color="#555555", fontsize=7.5)
    reference_widths = np.array([1.0e5, 1.0e6, 1.0e7])
    rms_tail = width_summary[width_summary["construction"] == "RMS-PN no gain"].sort_values("value")
    tail_constant = float((rms_tail.iloc[-1]["median_relative_sup_error"]) * rms_tail.iloc[-1]["value"])
    ax.plot(reference_widths, tail_constant / reference_widths, color="#444444", linestyle="--", linewidth=0.9, label=r"reference $O(d^{-1})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"(c) Across 24 random targets")
    ax.set_xlabel(r"normalized total width $d$")
    ax.set_ylabel("relative test sup error")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.28)
    ax.legend(frameon=False, loc="lower left")

    magnitude_summary = summary[summary["control"] == "magnitude"]
    ax = axes[1, 1]
    plot_band(ax, magnitude_summary, "RMS-PN with gain", "#0077BB", "RMS-PN median (IQR)")
    plot_band(ax, magnitude_summary, "LN-PN with affine", "#EE3377", "LN-PN median (IQR)", "-.")
    reference_magnitudes = np.array([1.0e4, 1.0e5, 1.0e6])
    rms_tail = magnitude_summary[magnitude_summary["construction"] == "RMS-PN with gain"].sort_values("value")
    tail_constant = float((rms_tail.iloc[-1]["median_relative_sup_error"]) * rms_tail.iloc[-1]["value"] ** 2)
    ax.plot(reference_magnitudes, tail_constant / reference_magnitudes**2, color="#444444", linestyle="--", linewidth=0.9, label=r"reference $O(M^{-2})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"(d) Affine fixed-width constructions")
    ax.set_xlabel(r"augmentation magnitude $M$")
    ax.set_ylabel("relative test sup error")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.28)
    ax.legend(frameon=False)

    figure.suptitle(
        rf"Shifted-ReLU $\phi(t)=\operatorname{{ReLU}}(t-{args.threshold:g})$; random one-hidden-layer MLP targets",
        y=1.01,
        fontsize=9.5,
    )
    figure.tight_layout(w_pad=1.0, h_pad=1.2)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    raw_path = args.output_prefix.with_name(args.output_prefix.name + "_raw.csv")
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    at_bound = raw[(raw["control"] == "total_width") & (raw["value"] <= args.threshold**2)]
    large_width = summary[(summary["control"] == "total_width") & (summary["value"] == total_widths[-1])]
    print(
        {
            "teachers": len(WIDTHS) * len(TEACHERS),
            "raw_rows": len(raw),
            "coordinate_bound": args.threshold**2,
            "max_deviation_from_unit_relative_error_below_bound": float(np.max(np.abs(at_bound["relative_sup_error"] - 1.0))),
            "largest_width": int(total_widths[-1]),
            "median_relative_error_at_largest_width": {
                row["construction"]: float(row["median_relative_sup_error"])
                for _, row in large_width.iterrows()
            },
            "nonfinite_values": int(raw.select_dtypes(include=[np.number]).isna().sum().sum() + np.isinf(raw.select_dtypes(include=[np.number]).to_numpy()).sum()),
        }
    )


if __name__ == "__main__":
    main()
