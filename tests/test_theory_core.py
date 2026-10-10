"""Small invariant checks for the exact-normalization paper code."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from theory_core import Teacher, exact_ln, exact_rms, make_teacher_parameters


class TheoryCoreTests(unittest.TestCase):
    def test_zero_and_constant_inputs(self) -> None:
        zero = torch.zeros((2, 4), dtype=torch.float64)
        constant = torch.full((2, 4), 3.0, dtype=torch.float64)
        self.assertTrue(torch.equal(exact_ln(zero), zero))
        self.assertTrue(torch.equal(exact_rms(zero), zero))
        self.assertTrue(torch.equal(exact_ln(constant), zero))

    def test_normalization_geometry(self) -> None:
        values = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
        ln = exact_ln(values)
        rms = exact_rms(values)
        self.assertAlmostEqual(float(ln.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(ln.square().mean()), 1.0, places=12)
        self.assertAlmostEqual(float(rms.square().mean()), 1.0, places=12)

    def test_teacher_seed_is_reproducible(self) -> None:
        first = make_teacher_parameters(20260911, 4)
        second = make_teacher_parameters(20260911, 4)
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(first[3], second[3])
        teacher = Teacher(*first, "relu", torch.float64)
        output = teacher(torch.zeros((3, 1), dtype=torch.float64))
        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
