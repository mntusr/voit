import math
import unittest
from re import S

import numpy as np

from voit import Vec2, Vec2i, Vec3


class TestVec3(unittest.TestCase):
    def test_cross(self):
        v1 = Vec3(6, 43, 9)
        v2 = Vec3(4.5, -3, 0.9)

        actual_result = v1.cross(v2)
        expected_result = Vec3(65.7, 35.1, -211.5)

        self.assertAlmostEqual(actual_result.x, expected_result.x)
        self.assertAlmostEqual(actual_result.y, expected_result.y)
        self.assertAlmostEqual(actual_result.z, expected_result.z)

    def test_vec_len(self):
        vec = Vec3(-3, 2, 1)

        actual_len = vec.vec_len()
        expected_len = math.sqrt(14)

        self.assertEqual(actual_len, expected_len)

    def test_normalize(self):
        vec = Vec3(20, 13, -9.5)

        actual_normalized = vec.normalize()
        expected_normalized = Vec3(0.7789, 0.5063, -0.3700)

        self.assertAlmostEqual(actual_normalized.x, expected_normalized.x, places=3)
        self.assertAlmostEqual(actual_normalized.y, expected_normalized.y, places=3)
        self.assertAlmostEqual(actual_normalized.z, expected_normalized.z, places=3)
