"""Tests for procedural texture generation."""

import unittest

import numpy as np

from tests import helpers  # noqa: F401  (sets up sys.path)
import textures


class TestTextures(unittest.TestCase):
    def test_all_textures_shape_and_dtype(self):
        for fn in (textures.brick, textures.floor_stone, textures.ceiling, textures.goal):
            img = fn()
            self.assertEqual(img.shape, (textures._SIZE, textures._SIZE, 3), fn.__name__)
            self.assertEqual(img.dtype, np.uint8, fn.__name__)

    def test_values_in_byte_range(self):
        for fn in (textures.brick, textures.floor_stone, textures.ceiling, textures.goal):
            img = fn()
            self.assertGreaterEqual(int(img.min()), 0)
            self.assertLessEqual(int(img.max()), 255)

    def test_brick_is_deterministic(self):
        a = textures.brick(seed=5)
        b = textures.brick(seed=5)
        np.testing.assert_array_equal(a, b)

    def test_goal_is_bright(self):
        # The exit marker must be vivid so it shows through the fog.
        img = textures.goal()
        self.assertGreater(int(img.max()), 200)


if __name__ == "__main__":
    unittest.main()
