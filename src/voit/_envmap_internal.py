import numpy as np
import scipy.ndimage
from panda3d.core import Texture, TexturePool

from ._complexerpbr_internal import EnvMap
from ._errors import VoitError
from ._math_internal import (
    get_points_im_on_rectangle,
    sample_envmap_at_given_directions,
)
from ._vectors import Vec2i, Vec3


class VoitEnvmap:
    """
    An environment map for the render pipeline.

    The constructor is not intended for public use.
    """

    def __init__(self, env_map_internal: EnvMap):
        self._env_map_internal = env_map_internal

    @staticmethod
    def from_image(envmap: np.ndarray, envmap_linear: bool) -> "VoitEnvmap":
        """
        Calculate the environment map data structures from an environment map with equirectangular projection.

        Parameters
        ----------
        envmap
            The environment map image. Format: ``Im_RGBLike``
        envmap_linear
            If true, then it is assumed that the transfer function of the environment map is linear. Otherwise it is treated as sRGB.

        Returns
        -------
        v
            The created environment map object.
        """
        if not envmap_linear:
            envmap = envmap**2.2

        cube_faces = _equirectangular_2_cubemap_textures(envmap, cubemap_size=100)
        texture = _cubemap_faces_2_cubemap_texture(cube_faces)
        env_map_internal = EnvMap(cubemap=texture)
        return VoitEnvmap(env_map_internal)


def _cubemap_faces_2_cubemap_texture(cubemap_faces: np.ndarray) -> Texture:
    """
    Create a Panda3d texture from the faces of a HDR cubemap.

    The properties of the created texture:

    * general texture type: cube map
    * format: `panda3d.core.Texture.F_rgba16`
    * component type: `panda3d.core.Texture.T_float`
    * x size = y size = x size of the faces

    Parameters
    ----------
    cubemap_faces
        The faces of the cubemap. The order and orientation of the faces should be the same as the order and orientation of the faces of the cubemaps in Panda3d.

    Returns
    -------
    v
        The created texture. This is a cubemap texture.

    Raises
    ------
    VoitError
        If the given array does not contain the faces of a cubemap
    """
    _raise_if_not_cubemap_faces(cubemap_faces)
    height = cubemap_faces.shape[2]

    tex = Texture()
    tex.setup_cube_map(height, Texture.T_float, Texture.F_rgba16)
    cubemap_faces = np.concatenate(
        [cubemap_faces, np.ones((6, 1, height, height))],
        axis=1,
        dtype=cubemap_faces.dtype,
    )
    cubemap_faces = cubemap_faces.astype("<f")
    # the textures always use BGRA format internally (see https://discourse.panda3d.org/t/opencv-videocapture-to-panda3d-texture/25981/4)
    # rgba-> bgra; schw -> shwc
    binary_data = (
        cubemap_faces[:, [2, 1, 0, 3]].transpose([0, 2, 3, 1]).tobytes(order="C")
    )
    # expected_size = tex.get_expected_ram_image_size()

    tex.set_ram_image(binary_data, Texture.CM_off)

    return tex


def _raise_if_not_cubemap_faces(array: np.ndarray):
    """
    Raises
    ------
    VoitError
        If the given array does not contain the faces of a cubemap.
    """
    if len(array.shape) != 4:
        raise VoitError(
            f"The number of dimensions ({ len(array.shape) }) of the array is not equal to 4."
        )

    if array.shape[0] != 6:
        raise VoitError(
            f"The number of faces ({array.shape[0]}) of the array is not equal to 6."
        )
    if array.shape[1] != 3:
        raise VoitError(
            f"The number of channels ({array.shape[1]}) in the images of the array is not equal to 3."
        )
    if array.shape[2] != array.shape[3]:
        raise VoitError(
            f"The images in the array are not square images. Width: {array.shape[3]}, height: {array.shape[2]}"
        )

    if not np.issubdtype(array.dtype, np.floating):
        raise VoitError(
            f"The dtype of the array ({array.dtype}) is not a subdtype of float32."
        )


def _equirectangular_2_cubemap_textures(
    envmap_hdr: np.ndarray,
    cubemap_size: int,
) -> np.ndarray:
    """
    Calculate the cubemap texutres from an equirectangluar environment map.

    Parameters
    ----------
    envmap_hdr
        The environment map. Format: ``Im::RGBEnvmapLike``
    cubemap_size
        The width and height of each face of the cubemap.

    Returns
    -------
    v
        The cubemap textures. Format: ``Cubemap::RGBLike``

    Developer notes
    ---------------
    Unlike most Panda3d related functions, this function uses the Y-up left handed coordinate system instead of the default of Panda3d.
    """

    forward = Vec3(0, 0, 1)
    up = Vec3(0, 1, 0)
    right = Vec3(1, 0, 0)
    left = right.neg()
    back = forward.neg()
    down = up.neg()

    # (face_forward, face_right, face_up)
    face_directions = [
        (right, down, forward),  # 0
        (left, up, forward),  # 1
        (forward, right, down),  # 2
        (back, right, up),  # 3
        (up, right, forward),  # 4
        (down, left, forward),  # 5
    ]

    face_steps = Vec2i(cubemap_size, cubemap_size)
    faces: list[np.ndarray] = []
    for face_forward, face_right, face_up in face_directions:
        face_center_pos = face_forward.mul_with_scalar(0.5)
        face_local_y = face_up
        face_local_z = face_forward
        face_local_x = face_right
        directions = get_points_im_on_rectangle(
            plane_center_pos=face_center_pos,
            plane_x=face_local_x,
            plane_y=face_local_y,
            n_steps=face_steps,
        )
        directions = directions.reshape((3, -1)).T
        samples = sample_envmap_at_given_directions(
            directions=directions, envmap=envmap_hdr
        )
        face = samples.T.reshape((3, cubemap_size, cubemap_size))
        faces.append(face)

    faces_array = np.stack(faces, axis=0)

    return faces_array
