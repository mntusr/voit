from pathlib import Path

import cv2 as cv
import numpy as np
from panda3d.core import LVector4f, Texture

from voit import Vec3, VoitError
from voit._envmap_internal import (
    VoitEnvmap,
    _cubemap_faces_2_cubemap_texture,
    _equirectangular_2_cubemap_textures,
    _raise_if_not_cubemap_faces,
)

from .testutil import VoitTestBase


class TestFunction(VoitTestBase):
    def test_raise_if_not_cubemap_faces__invalid_shape(self):
        invalid_shapes: list[tuple[str, list[int], int]] = [
            ("invalid_dim_count", [6, 3], 2),
            ("invalid_face_count", [5, 3, 165, 165], 5),
            ("invalid_channel_count", [6, 4, 165, 165], 4),
            ("invalid_face_count", [6, 3, 13, 165], 13),
        ]
        for case_name, shape_spec, key_num in invalid_shapes:
            with self.subTest(case_name):
                with self.assertRaises(VoitError) as cm:
                    array = np.ones(shape_spec, dtype=np.float32)
                    _raise_if_not_cubemap_faces(array)

                msg = str(cm.exception)
                self.assertIn(str(key_num), msg)

    def test_raise_if_not_cubemap_faces__invalid_dtype(self):
        with self.assertRaises(VoitError):
            array = np.ones((6, 4, 9), dtype=np.uint8)
            _raise_if_not_cubemap_faces(array)

    def test_cubemap_faces_2_cubemap_texture__happy_path(self):
        # setup
        expected_r = 1.7
        expected_g = 9.9
        expected_b = 3.1
        cubemap_size = 240
        expected_color = Vec3(expected_r, expected_g, expected_b)
        data = np.zeros((6, 3, cubemap_size, cubemap_size), dtype=np.float32)
        data[:, 0, :, :] = expected_r
        data[:, 1, :, :] = expected_g
        data[:, 2, :, :] = expected_b

        # run
        tex = _cubemap_faces_2_cubemap_texture(data)

        # texture format checks
        self.assertEqual(tex.get_x_size(), cubemap_size)
        self.assertEqual(tex.get_y_size(), cubemap_size)
        self.assertEqual(tex.get_z_size(), 6)
        self.assertEqual(tex.format, Texture.F_rgba16)
        self.assertEqual(tex.component_type, Texture.T_float)

        # serialization checks
        peeker = tex.peek()

        color_samples: list[Vec3] = []
        peek_positins: list[tuple[int, int, int]] = [
            (13, 9, 5),
            (13, 9, 4),
            (13, 8, 5),
            (12, 9, 5),
        ]
        for peek_pos in peek_positins:
            peeked = LVector4f(0, 0, 0, 1)
            peeker.fetch_pixel(peeked, *peek_pos)
            self.assertAlmostEqual(peeked.get_w(), 1, places=4)
            color_samples.append(Vec3(peeked.x, peeked.y, peeked.z))

        for color_sample in color_samples:
            self.assertVec3Allclose(color_sample, expected_color)

    def test_cubemap_faces_2_cubemap_texture__invalid_cubemap(self):
        with self.assertRaises(VoitError):
            _cubemap_faces_2_cubemap_texture(
                cubemap_faces=np.ones((6, 3, 20, 20), dtype=np.uint8)
            )

    def test_equirectangular_2_cubemap_textures(self):
        equirectangular = _load_proj_test_envmap(self.get_test_data_dir("projection"))
        expected_cubemap_size = 250

        coordinates: list[tuple[int, tuple[int, int]]] = [
            (0, (83, 152)),
            (0, (185, 173)),
            (1, (83, 152)),
            (1, (185, 173)),
            (2, (128, 89)),
            (2, (128, 170)),
            (3, (128, 89)),
            (3, (128, 170)),
            (4, (128, 89)),
            (4, (128, 170)),
            (5, (128, 89)),
            (5, (128, 170)),
        ]

        expected_colors: list[Vec3] = [
            Vec3(0.145, 0.18, 0.584),
            Vec3(0.796, 0.635, 0.459),
            Vec3(0.831, 0.431, 0.345),
            Vec3(0.576, 0.173, 0.518),
            Vec3(0.451, 0.485, 0.808),
            Vec3(0.578, 0.49, 0.0),
            Vec3(0.545, 0.727, 0.761),
            Vec3(0.181, 0.126, 0.255),
            Vec3(0.482, 0.467, 0.808),
            Vec3(0.0157, 0.478, 0.18),
            Vec3(0.592, 0.769, 0.557),
            Vec3(0.2, 0.0422, 0.18),
        ]
        actual_faces = _equirectangular_2_cubemap_textures(
            cubemap_size=expected_cubemap_size, envmap_hdr=equirectangular
        )
        self.assertEqual(
            actual_faces.shape, (6, 3, expected_cubemap_size, expected_cubemap_size)
        )

        # for actual_face in actual_faces:
        #     plt.imshow(actual_face.transpose([1, 2, 0]))
        #     plt.show(block=True)
        #     plt.close()

        for (face, (pos_x, pos_y)), expected_color in zip(coordinates, expected_colors):
            actual_color = Vec3.from_npy_col_vec(
                np.expand_dims(actual_faces[face, :, pos_y, pos_x], axis=1)
            )

            self.assertVec3Allclose(actual_color, expected_color, atol=1e-2)


class TestEnvmap(VoitTestBase):
    def test_from_image__linear(self):
        envmap_im1 = _load_proj_test_envmap(self.get_test_data_dir("projection"))
        envmap_im2 = envmap_im1 * 2

        envmap1 = VoitEnvmap.from_image(envmap=envmap_im1, envmap_linear=True)
        envmap2 = VoitEnvmap.from_image(envmap=envmap_im2, envmap_linear=True)

        for coeff1, coeff2 in zip(
            envmap1._env_map_internal.sh_coefficients,
            envmap2._env_map_internal.sh_coefficients,
        ):
            self.assertAlmostEqual(coeff1.x * 2, coeff2.x)
            self.assertAlmostEqual(coeff1.y * 2, coeff2.y)
            self.assertAlmostEqual(coeff1.z * 2, coeff2.z)

    def test_from_image__srgb(self):
        envmap_im1 = _load_proj_test_envmap(self.get_test_data_dir("projection"))
        envmap_im2 = envmap_im1 * 2

        envmap1 = VoitEnvmap.from_image(envmap=envmap_im1, envmap_linear=False)
        envmap2 = VoitEnvmap.from_image(envmap=envmap_im2, envmap_linear=False)

        for coeff1, coeff2 in zip(
            envmap1._env_map_internal.sh_coefficients,
            envmap2._env_map_internal.sh_coefficients,
        ):
            self.assertNotAlmostEqual(coeff1.x * 2, coeff2.x)
            self.assertNotAlmostEqual(coeff1.y * 2, coeff2.y)
            self.assertNotAlmostEqual(coeff1.z * 2, coeff2.z)


def _load_proj_test_envmap(test_data_dir: Path) -> np.ndarray:
    """
    Load the test images for the cubemap conversion test.

    Returns
    -------
    equirect
        The equirectangular image. Format: ``Im::RGBEnvmapLike``
    cubemap
        The cubemap. Format: ``Cubemap::RGBLike``
    """

    im_path = test_data_dir / "equirectangular.png"
    im: np.ndarray = cv.imread(str(im_path))
    im = im[:, :, ::-1].transpose([2, 0, 1])
    im = im.astype(np.float32) / 255
    return im
