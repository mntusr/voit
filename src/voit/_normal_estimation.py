import math
from typing import overload

import depth_tools
import numpy as np
import scipy.linalg

from ._errors import VoitError
from ._fmt_checks_internal import DepthmapLike, Im_Mask, Transform_3x3
from ._math_internal import px_2_vs_unchecked
from ._vectors import Vec2, Vec2i, Vec3


def get_pos_normal_from_3_points(
    *,
    camera: depth_tools.CameraIntrinsics,
    depth: np.ndarray,
    depth_mask: np.ndarray,
    center_point_px: Vec2i,
    aux_points_px: tuple[Vec2i, Vec2i],
) -> tuple[Vec3, Vec3] | None:
    """
    Get the position of a point and the surface normal from two auxillary points.

    Parameters
    ----------
    t_proj_mat
        The theoretical projection matrix. Format: ``TProjMat``
    depth
        The depth values. Format: ``DepthmapLike``
    depth_mask
        The mask that selects the valid depth values. Format: ``Im_Mask``
    center_point_px
        The pixel position of the point "center point".
    aux_points_px
        A tuple that gives the pixel positions of the two other points required for the normal calculation.

    Returns
    -------
    center_point_vs
        The view space position of the center point.
    n_vs
        The estimated normal vector in the view space.
    """
    DepthmapLike.check_arg(depth, name="depth")
    Im_Mask.check_arg(depth_mask, name="depth_mask")
    if depth.shape != depth_mask.shape:
        raise VoitError(
            f"The shape of the specified depth map ({depth.shape}) does not match to the shape of the depth mask ({depth_mask.shape})."
        )

    _, height, width = depth.shape
    im_size = Vec2i(width, height)

    if not depth_mask[0, center_point_px.y, center_point_px.x]:
        return None

    if not depth_mask[0, aux_points_px[0].y, aux_points_px[0].x]:
        return None

    if not depth_mask[0, aux_points_px[1].y, aux_points_px[1].x]:
        return None

    center_point_vs = px_2_vs_unchecked(
        camera=camera,
        depth=depth[0, center_point_px.y, center_point_px.x],
        im_size=im_size,
        pos_px=center_point_px,
    )
    aux0_point_vs = px_2_vs_unchecked(
        camera=camera,
        depth=depth[0, aux_points_px[0].y, aux_points_px[0].x],
        im_size=im_size,
        pos_px=aux_points_px[0],
    )
    aux1_point_vs = px_2_vs_unchecked(
        camera=camera,
        depth=depth[0, aux_points_px[1].y, aux_points_px[1].x],
        im_size=im_size,
        pos_px=aux_points_px[1],
    )

    dir_c0 = aux0_point_vs.vec_minus(center_point_vs).normalize()
    dir_c1 = aux1_point_vs.vec_minus(center_point_vs).normalize()

    n_vs = dir_c0.cross(dir_c1).normalize()

    if n_vs.dot(center_point_vs.neg()) < 0:
        n_vs = n_vs.neg()

    return center_point_vs, n_vs
