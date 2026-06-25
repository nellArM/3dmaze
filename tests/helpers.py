"""Shared test helpers: make the project root importable and build tiny mazes."""

import os
import sys

# Make the project root (parent of tests/) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W = True   # wall
O = False  # open


def grid_from_ascii(rows):
    """Build an occupancy grid from ASCII rows ('#' = wall, ' '/'.' = open)."""
    return [[ch == "#" for ch in row] for row in rows]
