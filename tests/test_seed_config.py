from __future__ import annotations

import os
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from app.seed_config import DEFAULT_SEED, derive_seed, resolve_seed, seed_everything


class TestSeedConfig(unittest.TestCase):
    def test_resolve_seed_prefers_explicit_seed(self):
        with patch.dict(os.environ, {"RAG_SEED": "99"}):
            self.assertEqual(resolve_seed(123), 123)

    def test_resolve_seed_uses_environment_variable(self):
        with patch.dict(os.environ, {"RAG_SEED": "99"}):
            self.assertEqual(resolve_seed(), 99)

    def test_resolve_seed_falls_back_to_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_seed(), DEFAULT_SEED)

    def test_seed_everything_reproduces_random_numpy_and_torch(self):
        seed_everything(321)
        first_values = (
            random.random(),
            np.random.rand(),
            torch.rand(1).item(),
        )

        seed_everything(321)
        second_values = (
            random.random(),
            np.random.rand(),
            torch.rand(1).item(),
        )

        self.assertEqual(first_values[0], second_values[0])
        self.assertEqual(first_values[1], second_values[1])
        self.assertEqual(first_values[2], second_values[2])

    def test_derive_seed_is_stable_and_context_sensitive(self):
        first = derive_seed(42, "query-1", 1, 1)
        second = derive_seed(42, "query-1", 1, 1)
        changed = derive_seed(42, "query-1", 1, 2)

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
