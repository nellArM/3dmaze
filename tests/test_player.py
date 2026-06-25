"""Tests for player camera state, movement, and wall collision."""

import unittest
import math

import numpy as np

from tests.helpers import grid_from_ascii
import player
from player import Player
import renderer


def step(p, grid, intents, dt=0.1, frames=20):
    for _ in range(frames):
        p.update(dt, intents, grid)


class TestCamera(unittest.TestCase):
    def test_spawn_centered_in_cell(self):
        p = Player((3, 5))  # (z, x)
        self.assertEqual(p.x, 5.5)
        self.assertEqual(p.z, 3.5)
        self.assertEqual(p.y, player.EYE_HEIGHT)

    def test_cell_property(self):
        p = Player((3, 5))
        self.assertEqual(p.cell, (3, 5))

    def test_forward_is_unit_vector(self):
        p = Player((0, 0))
        for yaw in (0.0, 0.7, 1.9, 3.0, -2.0):
            p.yaw = yaw
            fx, fz = p.forward()
            self.assertAlmostEqual(math.hypot(fx, fz), 1.0, places=6)

    def test_forward_yaw_zero_points_plus_z(self):
        p = Player((0, 0))
        p.yaw = 0.0
        fx, fz = p.forward()
        self.assertAlmostEqual(fx, 0.0, places=6)
        self.assertAlmostEqual(fz, 1.0, places=6)

    def test_a_d_rotate_opposite_directions(self):
        grid = grid_from_ascii(["###", "# #", "###"])
        p = Player((1, 1))
        p.update(0.1, {"left": True}, grid)
        p2 = Player((1, 1))
        p2.update(0.1, {"right": True}, grid)
        # A and D must rotate by equal, opposite amounts (convention-independent).
        self.assertLess(p.yaw * p2.yaw, 0.0)
        self.assertAlmostEqual(p.yaw, -p2.yaw, places=6)


class TestRotationDirection(unittest.TestCase):
    """Verify A/D turn the *view* the correct way, not just opposite ways.

    Uses the same look_at the renderer uses. Panning the camera right makes the
    world shift to screen-left (negative view-space x) and vice-versa.
    """

    EYE = (0.0, player.EYE_HEIGHT, 0.0)
    LANDMARK = np.array([0.0, player.EYE_HEIGHT, 5.0, 1.0])  # straight ahead at yaw 0

    def _view_x_after(self, intent):
        grid = grid_from_ascii(["###", "# #", "###"])
        p = Player((1, 1))
        p.x, p.z, p.yaw = self.EYE[0], self.EYE[2], 0.0
        for _ in range(5):
            p.update(0.05, {intent: True}, grid)
        fwd = (p.x + math.sin(p.yaw), p.y, p.z + math.cos(p.yaw))
        view = renderer.look_at((p.x, p.y, p.z), fwd, (0, 1, 0))
        return (view @ self.LANDMARK)[0]

    def test_pressing_right_turns_view_right(self):
        # Turn right -> world slides to screen-left -> view_x < 0.
        self.assertLess(self._view_x_after("right"), -0.1)

    def test_pressing_left_turns_view_left(self):
        # Turn left -> world slides to screen-right -> view_x > 0.
        self.assertGreater(self._view_x_after("left"), 0.1)


class TestMovement(unittest.TestCase):
    def _open_room(self):
        # 5x5 fully open interior.
        return grid_from_ascii([
            "#####",
            "#   #",
            "#   #",
            "#   #",
            "#####",
        ])

    def test_moves_forward_in_open_space(self):
        grid = self._open_room()
        p = Player((2, 1))      # centre (1.5, 2.5)
        p.yaw = math.pi / 2     # face +X
        start_x = p.x
        step(p, grid, {"forward": True}, dt=0.05, frames=5)
        self.assertGreater(p.x, start_x)

    def test_backward_moves_opposite(self):
        grid = self._open_room()
        p = Player((2, 2))
        p.yaw = math.pi / 2     # face +X
        start_x = p.x
        step(p, grid, {"back": True}, dt=0.05, frames=3)
        self.assertLess(p.x, start_x)


class TestCollision(unittest.TestCase):
    def test_cannot_pass_through_wall(self):
        # Corridor open at x=1; wall at x=2. Walking +X must not enter the wall.
        grid = grid_from_ascii([
            "#####",
            "# ###",
            "#####",
        ])
        p = Player((1, 1))
        p.yaw = math.pi / 2     # face +X toward the wall
        step(p, grid, {"forward": True}, frames=40)
        self.assertEqual(p.cell, (1, 1))               # never left the open cell
        self.assertLess(p.x, 2.0 - player.RADIUS + 1e-6)  # stopped before the wall
        self.assertFalse(grid[int(p.z)][int(p.x)])     # not standing inside a wall

    def test_out_of_bounds_is_blocked(self):
        grid = grid_from_ascii(["###", "# #", "###"])
        p = Player((1, 1))
        self.assertTrue(p._blocked(-0.5, 1.5, grid))
        self.assertTrue(p._blocked(1.5, 99.0, grid))

    def test_slides_along_wall(self):
        # East blocked by a wall at (1,2); south open. Moving diagonally (yaw 45)
        # should slide: Z advances while X stays pinned by the wall.
        grid = grid_from_ascii([
            "#####",
            "# # #",
            "# # #",
            "# # #",
            "#####",
        ])
        p = Player((1, 1))      # centre (1.5, 1.5)
        p.yaw = math.pi / 4     # forward = (+X, +Z)
        start_z = p.z
        step(p, grid, {"forward": True}, dt=0.1, frames=15)
        self.assertGreater(p.z, start_z + 0.3)          # slid south
        self.assertLess(p.x, 2.0 - player.RADIUS + 1e-6)  # X pinned by the wall
        self.assertFalse(grid[int(p.z)][int(p.x)])      # still in open space


if __name__ == "__main__":
    unittest.main()
