"""Tests for maze generation, BFS verification, and start/end placement."""

import unittest

from tests.helpers import grid_from_ascii
import maze


class TestGeneration(unittest.TestCase):
    def test_dimensions(self):
        for n in (1, 5, 10, 50):
            g = maze.generate_maze(n, seed=0)
            self.assertEqual(len(g), 2 * n + 1)
            self.assertTrue(all(len(row) == 2 * n + 1 for row in g))

    def test_outer_border_all_walls(self):
        n = 12
        g = maze.generate_maze(n, seed=1)
        size = 2 * n + 1
        for i in range(size):
            self.assertTrue(g[0][i] and g[size - 1][i], "top/bottom border must be wall")
            self.assertTrue(g[i][0] and g[i][size - 1], "left/right border must be wall")

    def test_all_cells_open(self):
        # Every logical cell sits at odd occupancy coords and must be a passage.
        n = 10
        g = maze.generate_maze(n, seed=2)
        for r in range(n):
            for c in range(n):
                z, x = maze.cell_to_occ(r, c)
                self.assertFalse(g[z][x], f"cell {(r, c)} should be open")

    def test_perfect_maze_open_count(self):
        # A perfect maze is a spanning tree: n*n cells + (n*n - 1) carved walls.
        n = 30
        g = maze.generate_maze(n, seed=3)
        open_blocks = sum(1 for row in g for cell in row if not cell)
        self.assertEqual(open_blocks, 2 * n * n - 1)

    def test_deterministic_with_seed(self):
        a = maze.generate_maze(20, seed=123)
        b = maze.generate_maze(20, seed=123)
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        a = maze.generate_maze(20, seed=1)
        b = maze.generate_maze(20, seed=2)
        self.assertNotEqual(a, b)


class TestBFS(unittest.TestCase):
    def test_generated_maze_fully_reachable(self):
        for seed in range(5):
            g = maze.generate_maze(40, seed=seed)
            self.assertTrue(maze.verify_bfs(g, 40), f"seed {seed} should be connected")

    def test_disconnected_maze_fails_verification(self):
        # Two isolated open cells, no connecting passage -> not all reachable.
        g = grid_from_ascii([
            "#####",
            "# ###",
            "#####",
            "### #",
            "#####",
        ])
        self.assertFalse(maze.verify_bfs(g, 2))

    def test_bfs_distances_basic(self):
        # Straight corridor in a square grid: distances increase by 1 per step.
        g = grid_from_ascii([
            "#####",
            "#   #",
            "#####",
            "#####",
            "#####",
        ])
        dist = maze.bfs_distances(g, (1, 1))
        self.assertEqual(dist[1][1], 0)
        self.assertEqual(dist[1][2], 1)
        self.assertEqual(dist[1][3], 2)
        self.assertEqual(dist[1][0], -1)  # wall is unreachable

    def test_bfs_distances_walls_are_minus_one(self):
        g = maze.generate_maze(15, seed=7)
        dist = maze.bfs_distances(g, maze.cell_to_occ(0, 0))
        # Every wall block stays -1; every open block gets a non-negative distance.
        for z, row in enumerate(g):
            for x, is_wall in enumerate(row):
                if is_wall:
                    self.assertEqual(dist[z][x], -1)
                else:
                    self.assertGreaterEqual(dist[z][x], 0)


class TestPlacement(unittest.TestCase):
    def test_start_and_end_valid(self):
        n = 40
        g = maze.generate_maze(n, seed=9)
        start, end = maze.pick_start_and_end(g, n, seed=9)
        # Both must be open passage blocks.
        self.assertFalse(g[start[0]][start[1]])
        self.assertFalse(g[end[0]][end[1]])
        # They must differ and end must be reachable from start.
        self.assertNotEqual(start, end)
        dist = maze.bfs_distances(g, start)
        self.assertGreater(dist[end[0]][end[1]], 0)

    def test_end_is_far(self):
        # End should land in the far region, not right next to the player.
        n = 40
        g = maze.generate_maze(n, seed=11)
        start, end = maze.pick_start_and_end(g, n, seed=11)
        dist = maze.bfs_distances(g, start)
        reachable = [d for row in dist for d in row if d > 0]
        median = sorted(reachable)[len(reachable) // 2]
        self.assertGreater(dist[end[0]][end[1]], median)


if __name__ == "__main__":
    unittest.main()
