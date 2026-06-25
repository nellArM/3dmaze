"""Tests for batched mesh construction and the exit pillar."""

import unittest
import math

import numpy as np

from tests.helpers import grid_from_ascii
import maze
import geometry


class TestBuildMesh(unittest.TestCase):
    def test_vertex_stride_is_eight(self):
        g = maze.generate_maze(10, seed=0)
        walls, floor, ceil = geometry.build_mesh(g)
        for arr in (walls, floor, ceil):
            self.assertEqual(arr.shape[1], 8)
            self.assertEqual(arr.dtype, np.float32)
            # Quads -> triangle pairs -> multiple of 6 vertices.
            self.assertEqual(arr.shape[0] % 6, 0)

    def test_floor_and_ceiling_match_open_blocks(self):
        n = 20
        g = maze.generate_maze(n, seed=1)
        open_blocks = sum(1 for row in g for cell in row if not cell)
        walls, floor, ceil = geometry.build_mesh(g)
        self.assertEqual(floor.shape[0], open_blocks * 6)
        self.assertEqual(ceil.shape[0], open_blocks * 6)

    def test_normals_are_unit_axis_aligned(self):
        g = maze.generate_maze(8, seed=2)
        walls, floor, ceil = geometry.build_mesh(g)
        for arr in (walls, floor, ceil):
            normals = arr[:, 3:6]
            lengths = np.linalg.norm(normals, axis=1)
            np.testing.assert_allclose(lengths, 1.0, atol=1e-5)
            # Each normal is one of the 6 axis directions (one component == ±1).
            self.assertTrue(np.all(np.isclose(np.abs(normals).sum(axis=1), 1.0)))

    def test_floor_normals_point_up_ceiling_down(self):
        g = maze.generate_maze(6, seed=3)
        _, floor, ceil = geometry.build_mesh(g)
        self.assertTrue(np.all(floor[:, 4] == 1.0))   # floor normal +Y
        self.assertTrue(np.all(ceil[:, 4] == -1.0))   # ceiling normal -Y

    def test_positions_within_bounds(self):
        n = 12
        g = maze.generate_maze(n, seed=4)
        size = len(g)
        walls, floor, ceil = geometry.build_mesh(g)
        for arr in (walls, floor, ceil):
            pos = arr[:, 0:3]
            self.assertGreaterEqual(pos.min(), 0.0)
            self.assertLessEqual(pos[:, 0].max(), size)  # X
            self.assertLessEqual(pos[:, 2].max(), size)  # Z
            self.assertLessEqual(pos[:, 1].max(), geometry.WALL_H)

    def test_exposed_face_count_simple(self):
        # A single open cell surrounded by walls exposes exactly 4 wall faces.
        g = grid_from_ascii([
            "###",
            "# #",
            "###",
        ])
        walls, floor, ceil = geometry.build_mesh(g)
        self.assertEqual(walls.shape[0], 4 * 6)   # 4 faces
        self.assertEqual(floor.shape[0], 1 * 6)    # 1 open block
        self.assertEqual(ceil.shape[0], 1 * 6)


class TestPillar(unittest.TestCase):
    def test_pillar_has_four_faces(self):
        p = geometry.build_pillar((5, 7))
        self.assertEqual(p.shape, (24, 8))  # 4 faces * 6 verts

    def test_pillar_centered_on_cell(self):
        z, x = 5, 7
        p = geometry.build_pillar((z, x), radius=0.3)
        xs, zs = p[:, 0], p[:, 2]
        self.assertAlmostEqual((xs.min() + xs.max()) / 2, x + 0.5, places=5)
        self.assertAlmostEqual((zs.min() + zs.max()) / 2, z + 0.5, places=5)
        self.assertAlmostEqual(xs.max() - xs.min(), 0.6, places=5)

    def test_pillar_height_param(self):
        p = geometry.build_pillar((2, 2), height=0.5)
        self.assertAlmostEqual(float(p[:, 1].max()), 0.5, places=5)


class TestFlags(unittest.TestCase):
    def test_empty_is_zero_by_eight(self):
        f = geometry.build_flags([])
        self.assertEqual(f.shape, (0, 8))

    def test_vertex_count_scales_with_cells(self):
        for k in (1, 2, 3):
            cells = [(2 * i + 1, 2 * i + 1) for i in range(k)]
            f = geometry.build_flags(cells)
            self.assertEqual(f.shape, (k * 24, 8))  # 4 faces * 6 verts per post

    def test_flag_post_is_short(self):
        f = geometry.build_flags([(1, 1)], height=0.7)
        self.assertAlmostEqual(float(f[:, 1].max()), 0.7, places=5)


if __name__ == "__main__":
    unittest.main()
