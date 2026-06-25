"""Tests for the matrix helpers in renderer.py (no GL context required)."""

import unittest

import numpy as np

from tests import helpers  # noqa: F401  (sets up sys.path)
import renderer


class TestPerspective(unittest.TestCase):
    def test_shape_and_finiteness(self):
        m = renderer.perspective(70.0, 4 / 3, 0.05, 60.0)
        self.assertEqual(m.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(m)))

    def test_structure(self):
        m = renderer.perspective(90.0, 1.0, 1.0, 100.0)
        # fovy 90, aspect 1 -> f = 1, so m[0,0] == m[1,1] == 1.
        self.assertAlmostEqual(m[0, 0], 1.0, places=5)
        self.assertAlmostEqual(m[1, 1], 1.0, places=5)
        self.assertAlmostEqual(m[3, 2], -1.0, places=5)  # perspective divide row
        self.assertEqual(m[3, 3], 0.0)


class TestLookAt(unittest.TestCase):
    def test_rotation_is_orthonormal(self):
        v = renderer.look_at((1.5, 0.5, 1.5), (1.5, 0.5, 5.0), (0, 1, 0))
        rot = v[:3, :3]
        # Rows should be orthonormal (R @ R.T == I).
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-6)

    def test_eye_maps_to_origin(self):
        eye = np.array([2.0, 0.5, 3.0])
        target = eye + np.array([0.0, 0.0, 1.0])
        v = renderer.look_at(eye, target, (0, 1, 0))
        p = v @ np.array([eye[0], eye[1], eye[2], 1.0])
        # Camera position transforms to the view-space origin.
        np.testing.assert_allclose(p[:3], [0.0, 0.0, 0.0], atol=1e-6)

    def test_point_ahead_has_negative_z(self):
        # OpenGL view space looks down -Z, so a point in front of the camera
        # must have negative view-space z.
        eye = np.array([0.0, 0.5, 0.0])
        target = np.array([0.0, 0.5, 1.0])  # looking toward +Z world
        v = renderer.look_at(eye, target, (0, 1, 0))
        ahead = v @ np.array([0.0, 0.5, 5.0, 1.0])
        self.assertLess(ahead[2], 0.0)


if __name__ == "__main__":
    unittest.main()
