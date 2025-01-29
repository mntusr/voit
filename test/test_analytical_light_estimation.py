import cv2 as cv
import numpy as np

from voit import Vec3, VoitError
from voit._analytical_light_estimation import (
    _get_intensity_based_weights,
    _get_principal_light,
    _new_uniform_points_on_the_sphere,
)

from .testutil import VoitTestBase


class TestFunctions(VoitTestBase):
    def test_get_intensity_based_weights__happy_path(self):
        rgb_intensities = np.array(
            [
                [9, 3, 2],
                [18, 6, 4],
            ],
            dtype=np.float32,
        )
        weights = _get_intensity_based_weights(rgb_intensities=rgb_intensities)

        self.assertAlmostEqual(weights.sum(), 1, places=4)
        self.assertEqual(weights.shape, (2,))
        self.assertAlmostEqual(weights[0] / weights[1], 0.5, places=4)

    def test_get_intensity_based_weights__invalid_shape(self):
        with self.assertRaises(VoitError):
            _get_intensity_based_weights(rgb_intensities=np.zeros((3,)))

    def test_new_uniform_points_on_the_sphere__happy_path(self):
        vectors = _new_uniform_points_on_the_sphere()
        vector_lens = np.linalg.norm(vectors, axis=1, ord=2, keepdims=False)

        self.assertEqual(vectors.shape, (1321, 3))
        self.assertTrue(np.issubdtype(vectors.dtype, np.float32))
        self.assertAllAlmostEqual(vector_lens, 1)
        self.assertAllclose(
            vectors.mean(axis=0), np.array([0, 0.4868, 0], dtype=np.float32), atol=1e-4
        )

    def test_get_principal_light(self):
        sg_mu_dir = Vec3(5, 3, 4)
        envmap_hdr = self.create_sg_envmap(
            n_horiz_steps=100, a=Vec3(0, 1.1, 0.9), lam=7, mu_dir=sg_mu_dir
        )

        envmap_hdr = envmap_hdr + 0.2

        light_dir, light_intensity = _get_principal_light(envmap=envmap_hdr)

        light_dir_pred_normalized = light_dir.normalize()
        actual_light_dir_normalized = sg_mu_dir.normalize().neg()

        pred_gt_cos = light_dir_pred_normalized.dot(actual_light_dir_normalized)
        pred_gt_angle = np.arccos(pred_gt_cos) / np.pi * 180

        self.assertAlmostEqual(pred_gt_cos, 1.0, delta=1e-3)
        self.assertAlmostEqual(light_intensity.x, 0.2 * 2 * np.pi, delta=1e-4)
        self.assertGreater(light_intensity.z, light_intensity.x)
        self.assertGreater(light_intensity.y, light_intensity.z)

    def create_sg_envmap(
        self, n_horiz_steps: int, mu_dir: Vec3, lam: float, a: Vec3
    ) -> np.ndarray:
        """
        Create an environment map of a single spherical gaussian.

        Parameters
        ----------
        n_horiz_steps
            The number of the horizontal steps in the created environment map.
        mu_dir
            The direction vector of the spherical gaussian. This does not have to be normalized, since the function normalizes it.
        lam
            The sharpness of the SG.
        a
            The amplitude of the SG.

        Returns
        -------
        v
            The created environment map. Format: ``Im::RGBEnvmapLike``
        """
        azimuth_steps = np.linspace(-np.pi, np.pi, n_horiz_steps)
        elev_steps = np.linspace(np.pi / 2, -np.pi / 2, n_horiz_steps // 2)
        azimuth_im, elev_im = np.meshgrid(azimuth_steps, elev_steps)

        v_im = self.get_v_im(azimuth_im=azimuth_im, elev_im=elev_im)

        sg_im = self.get_spherical_gaussian_im(v_im=v_im, a=a, lam=lam, mu_dir=mu_dir)
        return sg_im

    def get_v_im(self, azimuth_im: np.ndarray, elev_im: np.ndarray) -> np.ndarray:
        """
        Calculate the direction vectors for the given azimuths and elevations.

        Parameters
        ----------
        azimuth_im
            An image, where each pixel describes an azimuth value. Format: ``Im::Scalar``
        elev_im
            An image, where each pixel describes an elevation value. Format: ``Im::Scalar``

        Returns
        -------
        v
            An image, where each pixel describes the direction vector for the corresponding azimuths and elevations. Format: ``Im::RGBLike``
        """
        r = 1

        y = np.sin(elev_im) * r

        r_flat = np.cos(elev_im) * r

        x = np.sin(azimuth_im) * r_flat
        z = np.cos(azimuth_im) * r_flat

        return np.stack([x, y, z], axis=0)

    def get_spherical_gaussian_im(
        self, v_im: np.ndarray, mu_dir: Vec3, lam: float, a: Vec3
    ) -> np.ndarray:
        """
        Evaluate the spherical gaussians on an image of vectors.

        Parameters
        ----------
        v_im
            An image, where every pixel describes a vector that specifies a direction. Format: ``Im::RGBLike``
        mu_dir
            The direction vector of the spherical gaussian. This does not have to be normalized, since the function normalizes it.
        lam
            The sharpness of the SG.
        a
            The amplitude of the SG.

        Returns
        -------
        v
            The SG function evaluated at each pixel. Format: ``Cubemap::RGBLike``
        """
        # from https://therealmjp.github.io/posts/sg-series-part-2-spherical-gaussians-101/
        mu_dir = mu_dir.normalize()
        v_im = v_im / np.linalg.norm(v_im, axis=0, ord=2)

        v_dot_mu_im = v_im[[0]] * mu_dir.x + v_im[[1]] * mu_dir.y + v_im[[2]] * mu_dir.z

        pow_expr = np.exp(lam * (v_dot_mu_im - 1))

        sg_im = np.concatenate(
            [
                a.x * pow_expr,
                a.y * pow_expr,
                a.z * pow_expr,
            ],
            dtype=np.float32,
        )

        return sg_im
