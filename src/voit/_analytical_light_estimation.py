from pathlib import Path
from typing import Any, Literal, Union

import numpy as np

from ._envmap_internal import VoitEnvmap
from ._errors import VoitError
from ._light_conf_internal import LightConf
from ._math_internal import sample_envmap_at_given_directions
from ._vectors import Vec3


def extract_analytical_light_from_envmap(
    envmap: np.ndarray,
) -> LightConf:
    pl_dir, pl_intensity = _get_principal_light(envmap=envmap)

    return LightConf(
        envmap=VoitEnvmap.from_image(envmap=envmap * 0.5, envmap_linear=True),
        dir_light_dir_and_color=(pl_dir, pl_intensity.mul_with_scalar(0.5)),
    )


def _get_principal_light(
    envmap: np.ndarray,
) -> tuple[Vec3, Vec3]:
    """
    Estimate the direction and intensity of the principal light. This function assumes that the light source is in the upper hemisphere and the lower hemisphere is not important.

    The intensity of the principal light: the mean of the intensities for each channel*2*pi

    Parameters
    ----------
    envmap
        The environment map to use. It uses equirectangular projection. The transfer function of the environment map is linear. Format: ``Im_RGBLike``

    Returns
    -------
    principal_dir
        The direction of the principal light (this vector is ``-l`` in the shaders).
    principal_intensity
        The intensity of the principal light for the red, green and plue color.
    """
    uniform_points = _new_uniform_points_on_the_sphere()
    intensities = sample_envmap_at_given_directions(
        envmap=envmap, directions=uniform_points
    )
    mean_intensities_per_channel = intensities.mean(axis=0)

    over_mean_intensities = np.maximum(intensities - mean_intensities_per_channel, 0)

    weights = _get_intensity_based_weights(over_mean_intensities)
    weights = np.expand_dims(weights, axis=1)

    final_dir = np.sum(uniform_points * weights, axis=0, keepdims=True).T

    final_dir = Vec3.from_npy_col_vec(-final_dir)
    final_intensity = Vec3(
        mean_intensities_per_channel[0] * 2 * np.pi,
        mean_intensities_per_channel[1] * 2 * np.pi,
        mean_intensities_per_channel[2] * 2 * np.pi,
    )

    return final_dir, final_intensity


def _get_intensity_based_weights(rgb_intensities: np.ndarray) -> np.ndarray:
    """
    Calculate the weights for the specified vectors that describe lighting. The properties of the weights:

    * The sum of the weights is 1.
    * The weight of a vector is proportional to the L1 norm of the 3 intensities.

    Parameters
    ----------
    rgb_intensities
        The corresponding intensities in the red, green and blue channels. Format: ``Points::ThreeD``

    Raises
    ------
    VoitError
        If the array is not two-dimensional.
    """
    if len(rgb_intensities.shape) != 2:
        raise VoitError("The array is not two dimensional.")

    weights: np.ndarray = np.linalg.norm(rgb_intensities, axis=1, ord=1, keepdims=False)
    weights = weights / weights.sum()
    return weights


_uniform_points_on_sphere: Union[np.ndarray, Any] = None


def _new_uniform_points_on_the_sphere() -> np.ndarray:
    """
    Get the vertices of an ico sphere with radious 1 and subdivision count 5.

    Number of vertices: 2562

    Returns
    -------
    v
        The generated points. Format: ``Points::ThreeD[$float32]``
    """
    global _uniform_points_on_sphere
    points_path = (
        Path(__file__).resolve().parent / "resources" / "sphere_surface_lut.npy"
    )

    if _uniform_points_on_sphere is None:
        points = np.load(str(points_path)).astype(np.float32)
        _uniform_points_on_sphere = points
        return points
    else:
        return _uniform_points_on_sphere.copy()
