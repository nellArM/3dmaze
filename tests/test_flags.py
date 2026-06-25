"""Tests for the FlagSet placement/pickup state."""

import unittest

from tests import helpers  # noqa: F401  (sets up sys.path)
from flags import FlagSet


class TestFlagSet(unittest.TestCase):
    def test_starts_empty_with_full_remaining(self):
        f = FlagSet()
        self.assertEqual(f.cells, [])
        self.assertEqual(f.remaining, FlagSet.MAX)

    def test_place_adds_and_decrements_remaining(self):
        f = FlagSet()
        self.assertEqual(f.toggle((1, 1)), "placed")
        self.assertIn((1, 1), f.cells)
        self.assertEqual(f.remaining, FlagSet.MAX - 1)

    def test_toggle_same_cell_removes_it(self):
        f = FlagSet()
        f.toggle((3, 5))
        self.assertEqual(f.toggle((3, 5)), "removed")
        self.assertNotIn((3, 5), f.cells)
        self.assertEqual(f.remaining, FlagSet.MAX)

    def test_cannot_place_more_than_max(self):
        f = FlagSet()
        for i in range(FlagSet.MAX):
            self.assertEqual(f.toggle((i, i)), "placed")
        self.assertEqual(f.remaining, 0)
        # A fourth distinct cell is rejected; state is unchanged.
        self.assertEqual(f.toggle((99, 99)), "full")
        self.assertEqual(len(f.cells), FlagSet.MAX)
        self.assertNotIn((99, 99), f.cells)

    def test_can_pick_up_when_full(self):
        f = FlagSet()
        for i in range(FlagSet.MAX):
            f.toggle((i, i))
        # Standing on an existing flag still works even at capacity.
        self.assertEqual(f.toggle((1, 1)), "removed")
        self.assertEqual(f.remaining, 1)

    def test_clear_empties(self):
        f = FlagSet()
        f.toggle((1, 1))
        f.toggle((2, 2))
        f.clear()
        self.assertEqual(f.cells, [])
        self.assertEqual(f.remaining, FlagSet.MAX)


if __name__ == "__main__":
    unittest.main()
