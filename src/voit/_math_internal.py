import math

import depth_tools
import numpy as np
import scipy.ndimage

from ._errors import VoitError
from ._logging_internal import VOIT_LOGGER
from ._vectors import Vec2i, Vec3


def get_transformation_matrix(
    obj_pos: Vec3,
    y_dir_in_view_space: Vec3,
    x_dir_in_view_space: Vec3,
) -> np.ndarray:
    """
    Calculate the transformation matrix that transforms an object to the view space. This function implements rotation and translation, but not scaling.

    Parameters
    ----------
    obj_pos
        The position of the object in the view space.
    y_dir_in_view_space
        The Y axis of the model space in the view space.
    x_dir_in_view_space
        The X axis of the model space in the view space.

    Returns
    -------
    v
        The model view matrix of the object. Format: ``Transform::Homog[4x4]``
    """
    y_dir_in_view_space = y_dir_in_view_space.normalize()
    x_dir_in_view_space = x_dir_in_view_space.normalize()

    xy_dot = x_dir_in_view_space.dot(y_dir_in_view_space)

    if abs(xy_dot) > 1e-6:
        VOIT_LOGGER.warning(
            f"The transformed local Y ({y_dir_in_view_space}) and X ({x_dir_in_view_space}) axis of the object are not orthogonal. Dot product: {xy_dot}, threshold: 1e-6"
        )

    fy = y_dir_in_view_space
    fx = x_dir_in_view_space
    fz = x_dir_in_view_space.cross(y_dir_in_view_space).normalize()

    rot_arr = np.array(
        [
            [fx.x, fy.x, fz.x, 0],
            [fx.y, fy.y, fz.y, 0],
            [fx.z, fy.z, fz.z, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    transl_arr = np.array(
        [
            [1, 0, 0, obj_pos.x],
            [0, 1, 0, obj_pos.y],
            [0, 0, 1, obj_pos.z],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return transl_arr @ rot_arr


def zbuf_2_depth(
    zbuf_data: np.ndarray,
    a_b: tuple[float, float],
) -> np.ndarray:
    """
    Convert the zbuffer to the depth map.

    This function assumes that the content of the zbuffer is calculated using the following formula: ``(a*d+b)/d``, where the ``d`` is the depth at the corresponding pixel.

    Parameters
    ----------
    zbuf_data
        The data in the zbuffer. Format: ``Im::Scalar``
    near_far
        The ``(near plane depth, far plane depth)`` tuple.
    zbuf_from_to
        The ``(minimal zbuffer value, maximal zbuffer value)`` tuple.

    Returns
    -------
    v
        The depth map. Format: ``Im::Scalar``
    """
    a, b = a_b
    true_depth_data = b / (zbuf_data - a)

    return true_depth_data


def get_ab_from_near_far_and_zbuf_range(
    near_far: tuple[float, float], zbuf_from_to: tuple[float, float]
) -> tuple[float, float]:
    """
    Get the ``a, b`` parameters to calculate the zbuffer from the depth using the following formula: ``z(d)=(a*d+b)/d``, where the ``d`` is the depth at the corresponding pixel.

    This function assumes that the zbuffer is not reversed.

    Parameters
    ----------
    near_far
        A tuple of the depth of the near and the far plane.
    zbuf_from_to
        A tuple containing the minimal and maximal value supported by the zbuffer. In case of OpenGL, this is (-1, 1).

    Returns
    -------
    a
        The ``a`` parameter.
    b
        The ``b`` parameter.
    """

    n, f = near_far
    z_0, z_1 = zbuf_from_to

    a = (f * z_1 - n * z_0) / (f - n)
    b = (f * n * z_0 - f * n * z_1) / (f - n)

    return a, b


def sample_envmap_at_given_directions(
    directions: np.ndarray, envmap: np.ndarray
) -> np.ndarray:
    """
    Read the specified environment map at the given directions.

    This function assumes an Y-up left handed coordinate system.

    Parameters
    ----------
    directions
        The vectors that specify the directions to read. These vectors do not have to be normalized. Format: ``Points::ThreeD``
    envmap
        The environment map or half environment map to read. Format: ``Im::RGBEnvmapLike``

    Returns
    -------
    v
        The got samples. Format: ``Vecs3``
    """
    azim, elev, _ = _get_polar_coordinates_for_directions(vectors=directions)

    results = _sample_envmap(envmap=envmap, azimuths=azim, elevations=elev)
    return results


def _get_polar_coordinates_for_directions(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the polar coordinates for the given vectors.

    Parameters
    ----------
    directions
        The specified vectors.

    Returns
    -------
    azim
        The azimuths of the vectors. Format: ``Scalars::Real``
    elev
        The elevations of the vectors. Format: ``Scalars::Real``
    radius
        The radiuses of the vectors. Format: ``Scalars::Real``
    """

    x_coords = vectors[:, 0]
    y_coords = vectors[:, 1]
    z_coords = vectors[:, 2]
    rs = np.linalg.norm(vectors, ord=2, axis=1, keepdims=False)

    r_flats = np.sqrt(x_coords**2 + z_coords**2)
    r_flats[r_flats < 1e-30] = 1e-30

    c = y_coords / rs

    elevations = np.arcsin(y_coords / rs)
    azimuths = np.arccos(z_coords / r_flats)
    azimuths[x_coords < 0] *= -1

    return azimuths, elevations, rs


def _sample_envmap(
    envmap: np.ndarray, azimuths: np.ndarray, elevations: np.ndarray
) -> np.ndarray:
    """
    Sample the environment map at the specified azimuths and elevations.

    Parameters
    ----------
    envmap
        The sampled environment map. Format: ``Im::RGBEnvmapLike``
    azimuths
        The azimuths. Format: ``Scalars::Real``
    elevations
        The elevations. Format: ``Scalars::Real``

    Returns
    -------
    v
        The sampled environment map. Format: ``Points::ThreeD``
    """
    if np.any(abs(azimuths) > np.pi):
        VOIT_LOGGER.warn(
            "The array of azimuths contains some values that have absolute values greater than pi. These values will be clamped."
        )
        azimuths = np.clip(azimuths, -np.pi, np.pi)
    if np.any(abs(elevations) > np.pi / 2):
        VOIT_LOGGER.warn(
            "The array of elevations contains some values that have absolute values greater than pi/2. These values will be clamped."
        )
        elevations = np.clip(elevations, -np.pi / 2, np.pi / 2)

    envmap_width = envmap.shape[2]
    envmap_height = envmap.shape[1]
    half_pi = np.pi / 2

    coord_im_x = (azimuths + np.pi) / (2 * np.pi) * (envmap_width - 1)
    coord_im_y = (1 - (elevations + half_pi) / np.pi) * (envmap_height - 1)

    coords = np.stack([coord_im_y, coord_im_x], axis=0)

    results_r = scipy.ndimage.map_coordinates(envmap[0], coords, order=1)
    results_g = scipy.ndimage.map_coordinates(envmap[1], coords, order=1)
    results_b = scipy.ndimage.map_coordinates(envmap[2], coords, order=1)

    results = np.stack([results_r, results_g, results_b], axis=1)

    return results


def get_corresponding_z_for_transformed_local_y(new_y_dir: Vec3) -> Vec3:
    """
    This function calculates a vector that can describe the local X axis of a model in the world space without applying unnecessary rotation. This is basically the same transform as in a look-at matrix. This function uses ``(0, 0, 1)`` as the "up"-value in the look-at matrix calculation.

    Parameters
    ----------
    new_y_dir
        The direction of the new Y axis. This does not have to be normalized.

    Returns
    -------
    v
        The new X axis in the previous space.
    """
    new_y_dir = new_y_dir.normalize()

    if abs(new_y_dir.x) < 1e-18 and abs(new_y_dir.y) < 1e-18:
        old_z = Vec3(0, 1, 0)
    else:
        old_z = Vec3(0, 0, 1)

    new_z = old_z.vec_minus(new_y_dir.mul_with_scalar(old_z.dot(new_y_dir)))

    return new_z.normalize()


def get_points_im_on_rectangle(
    plane_center_pos: Vec3,
    plane_x: Vec3,
    plane_y: Vec3,
    n_steps: Vec2i,
) -> np.ndarray:
    """
    Calculate the points on a rectangle in the space.

    The function creates an RGB image, where each pixel denotes the position of the corresponding point using the ``rgb->xyz`` mapping.

    Parameters
    ----------
    plane_center_pos
        The position of the center of the plane.
    plane_x
        ``plane_x = bottom_right_pos - bottom_left_pos``
    plane_y
        ``plane_y = top_left_pos-bottom_right_pos``

    Returns
    -------
    v
        The calculated texture.
    """
    x_steps = np.linspace(-0.5, 0.5, n_steps.x)
    y_steps = np.linspace(0.5, -0.5, n_steps.y)

    x_weight_mesh, y_weight_mesh = np.meshgrid(x_steps, y_steps)

    dx_plane = np.stack(
        [
            x_weight_mesh * plane_x.x,
            x_weight_mesh * plane_x.y,
            x_weight_mesh * plane_x.z,
        ],
        dtype=np.float32,
    )
    dy_plane = np.stack(
        [
            y_weight_mesh * plane_y.x,
            y_weight_mesh * plane_y.y,
            y_weight_mesh * plane_y.z,
        ],
        dtype=np.float32,
    )
    delta_plane = dx_plane + dy_plane

    plane_center_pre_expand = [
        plane_center_pos.x,
        plane_center_pos.y,
        plane_center_pos.z,
    ]
    total_plane = delta_plane + np.expand_dims(
        np.expand_dims(plane_center_pre_expand, -1), -1
    )
    return total_plane


def px_2_vs_unchecked(
    camera: depth_tools.CameraIntrinsics, pos_px: Vec2i, depth: float, im_size: Vec2i
) -> Vec3:
    """
    Convert the given pixel-depth pair to a view space point.

    This function does not check its arguments.

    Parameters
    ----------
    camera
        The camera intrinsics.
    pos_px
        The position of the pixel.
    depth
        The depth value for the pixel.
    """
    point_h = np.array(
        [
            [pos_px.x * depth],
            [
                ((im_size.y - 1) - pos_px.y) * depth
            ],  # the y coordinate of the camera intrinsic description is flipped relative to the pixel indices
            [depth],
        ],
        dtype=np.float32,
    )
    point_restored = camera.get_intrinsic_mat_inv() @ point_h

    return Vec3(
        point_restored[0, 0],
        point_restored[1, 0],
        point_restored[2, 0],
    )


def linear_2_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Assuming that the input array contains the channel values of a linear RGB image, this function converts these values to the corresponding sRGB values.

    This function uses the simple, but not completely correct ``x**(1/2.2)`` formula for the conversion.
    """
    return linear ** (1 / 2.2)


def srgb_2_linear(linear: np.ndarray) -> np.ndarray:
    """
    Assuming that the input array contains the channel values of an sRGB image, this function converts these values to the corresponding linear RGB values.

    This function uses the simple, but not completely correct ``x**(2.2)`` formula for the conversion.
    """
    return linear**2.2
