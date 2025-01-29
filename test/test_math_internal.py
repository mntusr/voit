import unittest

import numpy as np
import scipy.ndimage
from panda3d.core import NodePath

from voit import Vec2i, Vec3
from voit._math_internal import (
    _get_polar_coordinates_for_directions,
    _sample_envmap,
    get_ab_from_near_far_and_zbuf_range,
    get_corresponding_z_for_transformed_local_y,
    get_points_im_on_rectangle,
    get_transformation_matrix,
    sample_envmap_at_given_directions,
    zbuf_2_depth,
)

from .testutil import VoitTestBase


class TestFunctions(VoitTestBase):
    def test_get_transformation_matrix__happy_path(self):
        new_y = Vec3(0, 0, 2)
        new_x = Vec3(0, 2, 0)
        obj_pos_in_view_space = Vec3(1.5, -0.5, 0.7)

        with self.assertNoLogs(self.voit_logger):
            transf_mat = get_transformation_matrix(
                y_dir_in_view_space=new_y,
                x_dir_in_view_space=new_x,
                obj_pos=obj_pos_in_view_space,
            )

        actual_transformed_point = transf_mat @ np.array(
            [
                [3.3],
                [9],
                [-3.2],
                [1],
            ],
            dtype=np.float32,
        )

        new_y_arr = np.array(
            [
                [0],
                [0],
                [1],
            ],
            dtype=np.float32,
        )
        new_x_arr = np.array(
            [
                [0],
                [1],
                [0],
            ],
            dtype=np.float32,
        )
        new_z_arr = np.array(
            [
                [1],
                [0],
                [0],
            ],
            dtype=np.float32,
        )
        obj_pos_arr = np.array(
            [
                [1.5],
                [-0.5],
                [0.7],
            ],
            dtype=np.float32,
        )

        expected_transformed_point = (
            new_x_arr * 3.3 + new_y_arr * 9 + new_z_arr * (-3.2) + obj_pos_arr
        )
        expected_transformed_point = np.concatenate(
            [expected_transformed_point, np.array([[1]], dtype=np.float32)]
        )

        self.assertAllclose(actual_transformed_point, expected_transformed_point)

    def get_transformation_matrix__non_orth(self):
        new_y = Vec3(0, 0, 1)
        new_x = Vec3(0, 1, 1)
        floor_pos_in_view_space = Vec3(1.5, -0.5, 0.7)

        with self.assertLogs(self.voit_logger) as cm:
            get_transformation_matrix(
                x_dir_in_view_space=new_x,
                y_dir_in_view_space=new_y,
                obj_pos=floor_pos_in_view_space,
            )
            self.assertIn("orthogonal", cm.output[0])
            self.assertIn(str(new_y), cm.output[0])
            self.assertIn(str(new_x), cm.output[0])

    def test_get_ab_from_near_far_and_zbuf_range(self):
        n, f = (10, 100)
        z0, z1 = (-2.5, 6.7)
        a, b = get_ab_from_near_far_and_zbuf_range(
            near_far=(n, f), zbuf_from_to=(z0, z1)
        )

        projected_near = (a * n + b) / n
        projected_far = (a * f + b) / f

        self.assertAlmostEqual(projected_near, z0)
        self.assertAlmostEqual(projected_far, z1)

    def test_zbuf_2_depth(self):
        a = 5
        b = 20

        zbuf = np.ones((1, 10, 13), dtype=np.float32)

        depth = zbuf_2_depth(a_b=(a, b), zbuf_data=zbuf)

        restored_zbuf = (a * depth + b) / depth

        self.assertAllclose(zbuf, restored_zbuf)

    def test_get_polar_coordinates_for_directions(self):
        vectors = np.array(
            [
                [3, 16, 4],
                [-3, 16, 4],
                [3, 16, -4],
                [-3, 16, -4],
                [3, -16, 4],
            ],
            dtype=np.float32,
        )

        # r_flat = 5
        expected_elevations = np.array(
            [
                1.2679,  # asin((16)/sqrt(16^2+5^2))
                1.2679,
                1.2679,
                1.2679,
                -1.2679,
            ],
            dtype=np.float32,
        )

        expected_azimuths = np.array(
            [
                0.6435,  # acos(4/5)
                -0.6435,
                np.pi - 0.6435,
                -np.pi + 0.6435,
                0.6435,
            ],
            dtype=np.float32,
        )
        expected_radiuses = np.array(
            [
                16.7631,  # sqrt(3^2+16^2+4^2)
            ]
            * 5,
            dtype=np.float32,
        )

        azims, elevs, radiuses = _get_polar_coordinates_for_directions(vectors)

        self.assertAllclose(azims, expected_azimuths, atol=1e-3)
        self.assertAllclose(elevs, expected_elevations, atol=1e-3)
        self.assertAllclose(radiuses, expected_radiuses, atol=1e-3)

    def test_sample_envmap__happy_path(self):
        envmap_r = np.array(
            [
                [3, 2, 6, 5, 9, 1],
                [8, 7, 3, 4, 2, 3],
                [10, 5, 7, 2, 7, 9],
                [2, 1, 0, 1, 10, 3],
            ],
            dtype=np.float32,
        )
        envmap_g = (envmap_r**2) / 5
        envmap_b = (envmap_r * 2 + envmap_g) / 3
        n_samples = 5

        envmap = np.stack([envmap_r, envmap_g, envmap_b], dtype=np.float32)

        azimuth_steps = np.linspace(-np.pi, np.pi, envmap.shape[2])
        corresponding_rel_x_steps = np.linspace(0, envmap.shape[2] - 1, envmap.shape[2])
        elevation_steps = np.linspace(np.pi / 2, -np.pi / 2, envmap.shape[1])
        corresponding_rel_y_steps = np.linspace(0, envmap.shape[1] - 1, envmap.shape[1])

        azimuth_index = 4
        elevation_index = -2

        azimuth = azimuth_steps[azimuth_index]
        elevation = elevation_steps[elevation_index]
        with self.assertNoLogs(self.voit_logger):
            actual_sample = _sample_envmap(
                envmap=envmap,
                azimuths=np.array([azimuth] * n_samples),
                elevations=np.array([elevation] * n_samples),
            )

        self.assertEqual(actual_sample.shape, (5, 3))

        # calculate the expected sample
        transformed_coordinates = np.array(
            [
                [
                    corresponding_rel_x_steps[azimuth_index],
                    corresponding_rel_y_steps[elevation_index],
                ]
            ]
            * 5
        ).T
        expected_sample_r = scipy.ndimage.map_coordinates(
            envmap[0], transformed_coordinates[::-1]
        )
        expected_sample_g = scipy.ndimage.map_coordinates(
            envmap[1], transformed_coordinates[::-1]
        )
        expected_sample_b = scipy.ndimage.map_coordinates(
            envmap[2], transformed_coordinates[::-1]
        )
        expected_sample = np.stack(
            [expected_sample_r, expected_sample_g, expected_sample_b], axis=0
        ).T

        self.assertAllclose(actual_sample, expected_sample)

    def test_sample_envmap__azim_clamp(self):
        envmap_r = np.array(
            [
                [3, 2, 6, 5, 9, 1],
                [8, 7, 3, 4, 2, 3],
                [10, 5, 7, 2, 7, 9],
                [2, 1, 0, 1, 10, 3],
            ],
            dtype=np.float32,
        )
        envmap_g = (envmap_r**2) / 5
        envmap_b = (envmap_r * 2 + envmap_g) / 3

        envmap = np.stack([envmap_r, envmap_g, envmap_b], dtype=np.float32)

        azimuth = -1.31 * np.pi
        elev = np.pi / 2

        with self.assertLogs(self.voit_logger) as cm:
            actual_sample = _sample_envmap(
                envmap=envmap,
                azimuths=np.array([azimuth], dtype=np.float32),
                elevations=np.array([elev], dtype=np.float32),
            )
            self.assertIn("azimuths", cm.output[0])
        expected_sample = _sample_envmap(
            envmap=envmap,
            azimuths=np.array([-np.pi], dtype=np.float32),
            elevations=np.array([elev], dtype=np.float32),
        )

        self.assertAllclose(actual_sample, expected_sample)

    def test_sample_envmap__elev_clamp(self):
        envmap_r = np.array(
            [
                [3, 2, 6, 5, 9, 1],
                [8, 7, 3, 4, 2, 3],
                [10, 5, 7, 2, 7, 9],
                [2, 1, 0, 1, 10, 3],
            ],
            dtype=np.float32,
        )
        envmap_g = (envmap_r**2) / 5
        envmap_b = (envmap_r * 2 + envmap_g) / 3

        envmap = np.stack([envmap_r, envmap_g, envmap_b], dtype=np.float32)

        azim = np.pi / 3
        elev = np.pi / 2 * 1.3

        with self.assertLogs(self.voit_logger) as cm:
            actual_sample = _sample_envmap(
                envmap=envmap,
                azimuths=np.array([azim], dtype=np.float32),
                elevations=np.array([elev], dtype=np.float32),
            )
            self.assertIn("elevations", cm.output[0])
        expected_sample = _sample_envmap(
            envmap=envmap,
            azimuths=np.array([azim], dtype=np.float32),
            elevations=np.array([np.pi / 2], dtype=np.float32),
        )

        self.assertAllclose(actual_sample, expected_sample)

    def test_sample_envmap_at_given_directions__partial_results_correctly_combined(
        self,
    ):
        envmap = np.stack(
            [
                np.full((5, 10), 2),
                np.full((5, 10), 3),
                np.full((5, 10), 4),
            ],
            dtype=np.float32,
        )

        with self.assertNoLogs(self.voit_logger):
            actual_samples = sample_envmap_at_given_directions(
                envmap=envmap,
                directions=np.array(
                    [
                        [2, 5, 3],
                        [4, 6, 8],
                        [-5, 3, -1],
                        [-5, -3, -1],
                    ],
                    dtype=np.float32,
                ),
            )
            expected_samples = np.array([[2, 3, 4]] * 4, dtype=np.float32)
            self.assertAllclose(actual_samples, expected_samples)

    def test_get_corresponding_z_for_transformed_local_y(self):
        local_y = Vec3(3, 5, 2)
        local_z = get_corresponding_z_for_transformed_local_y(local_y)

        self.assertAlmostEqual(local_y.normalize().dot(local_z), 0)

        cross1_norm = local_y.cross(local_z).normalize()
        cross2_norm = local_y.cross(Vec3(0, 0, 1)).normalize()
        self.assertVec3Allclose(cross1_norm, cross2_norm)

    def test_get_points_im_on_plane(self):
        plane_x = Vec3(2, -9, 5)
        plane_y = Vec3(3.23722993, -0.76277007, -0.76277007)
        plane_center_pos = Vec3(5, 4, -1)
        points = get_points_im_on_rectangle(
            n_steps=Vec2i(19, 23),
            plane_center_pos=Vec3(5, 4, -1),
            plane_x=plane_x,
            plane_y=plane_y,
        )

        self.assertEqual(points.shape, (3, 23, 19))

        # bottom left corner
        actual_bottom_left = Vec3.from_npy_col_vec(points[:, -1, [0]])
        expected_bottom_left = plane_center_pos.vec_minus(
            plane_x.mul_with_scalar(0.5)
        ).vec_minus(plane_y.mul_with_scalar(0.5))
        self.assertVec3Allclose(actual_bottom_left, expected_bottom_left)

        # bottom right corner
        actual_bottom_left = Vec3.from_npy_col_vec(points[:, -1, [-1]])
        expected_bottom_left = plane_center_pos.vec_plus(
            plane_x.mul_with_scalar(0.5)
        ).vec_minus(plane_y.mul_with_scalar(0.5))
        self.assertVec3Allclose(actual_bottom_left, expected_bottom_left)

        # top left corner
        actual_bottom_left = Vec3.from_npy_col_vec(points[:, 0, [0]])
        expected_bottom_left = plane_center_pos.vec_minus(
            plane_x.mul_with_scalar(0.5)
        ).vec_plus(plane_y.mul_with_scalar(0.5))
        self.assertVec3Allclose(actual_bottom_left, expected_bottom_left)

        # top right corner
        actual_bottom_left = Vec3.from_npy_col_vec(points[:, 0, [-1]])
        expected_bottom_left = plane_center_pos.vec_plus(
            plane_x.mul_with_scalar(0.5)
        ).vec_plus(plane_y.mul_with_scalar(0.5))
        self.assertVec3Allclose(actual_bottom_left, expected_bottom_left)
