"""Check the committed numerical tables used by the published plots."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


FIGURES = Path(__file__).resolve().parents[1] / "figures"


class PublicationDataTests(unittest.TestCase):
    def test_activation_sweep_layout(self) -> None:
        data = pd.read_csv(FIGURES / "ln_rms_activation_family_sweep_raw.csv")
        self.assertEqual(len(data), 7920)
        self.assertEqual(
            data.groupby("normalization").size().to_dict(),
            {"LayerNorm": 3960, "RMSNorm": 3960},
        )
        self.assertEqual(sorted(data["teacher_width"].unique().tolist()), [4, 8, 16, 32])
        self.assertEqual(sorted(data["teacher_id"].unique().tolist()), list(range(6)))
        self.assertTrue((data["sup_error"] >= 0).all())
        self.assertTrue(data["sup_error"].notna().all())

    def test_figure_three_has_six_targets_per_curve(self) -> None:
        data = pd.read_csv(FIGURES / "ln_rms_activation_family_sweep_raw.csv")
        selected = (
            ("post_unbounded", "relu"),
            ("affine_pre", "tanh"),
            ("homogeneous_pre", "relu"),
        )
        for case, activation in selected:
            subset = data.loc[
                (data["case"] == case)
                & (data["activation"] == activation)
                & (data["teacher_width"] == 16)
            ]
            counts = subset.groupby(["normalization", "parameter"])["teacher_id"].nunique()
            self.assertEqual(len(counts), 22)
            self.assertTrue((counts == 6).all())


if __name__ == "__main__":
    unittest.main()
