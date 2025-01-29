import copy
import math
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, TypedDict, TypeGuard, cast

import depth_tools
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Camera,
    DirectionalLight,
    FrameBufferProperties,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayData,
    GeomVertexData,
    GeomVertexFormat,
    GraphicsOutput,
    GraphicsPipe,
    GraphicsWindow,
    InternalName,
    Lens,
    LMatrix3f,
    LMatrix4f,
    LoaderOptions,
    LPoint2f,
    LPoint3f,
    LVector3f,
    Material,
    MaterialAttrib,
    MatrixLens,
    NodePath,
    PandaNode,
    PerspectiveLens,
    Texture,
    TextureAttrib,
    TextureStage,
    Vec3F,
    WindowProperties,
    loadPrcFileData,
)

from ._complexerpbr_internal import Pipeline
from ._envmap_internal import VoitEnvmap
from ._errors import VoitError
from ._logging_internal import VOIT_LOGGER
from ._math_internal import get_ab_from_near_far_and_zbuf_range
from ._vectors import Vec2i, Vec3


class RendererShowBase(ShowBase):
    """
    Implements a ShowBase that does the rendering of the floor and the inserted object using Panda3d.

    Parameters
    ----------
    proj_mat
        The projection matrix. This should have the same format as the user matrix of `panda3d.core.MatrixLens`. Format: ``Transform::Homog[4x4]``
    im_size
        The size of the manipulated images.
    reflective_plane_screen_texcoord
        If this parameter is true, then the shader will ignore the texcoord values for the vertices of the reflective plane. It will use the screen texture coordinate instead.
    """

    def __init__(
        self,
        proj_mat: np.ndarray,
        im_size: Vec2i,
        offscreen: bool,
        show_envmap_as_skybox: bool,
        reflective_plane_screen_texcoord: bool,
    ):

        prc_data = f"""win-size {im_size.x} {im_size.y}
color-bits 8 8 8
depth-bits 24
framebuffer-float 0
gl-depth-zero-to-one 0
load-display pandagl
aux-display pandagl
"""
        loadPrcFileData("", prc_data)
        super().__init__(windowType="offscreen" if offscreen else None)
        assert self.cam is not None

        cam_lens = new_lens_from_proj_mat(proj_mat)
        cam_node = self.cam.node()
        cam_node = cast(Camera, cam_node)
        cam_node.set_lens(cam_lens)

        got_fbprops = self.win.getFbProperties()  # type: ignore
        assert _has_unsigned_byte_rgba_format(
            got_fbprops
        ), "The buffer does not have the expected format (unsigned byte RGBA color; 24 bytes depth). See the console warnings for the got format."

        self.disableMouse()
        self.pipeline = Pipeline(
            base=self,
            enable_shadows=True,
            use_normal_maps=False,
            use_occlusion_maps=True,
            max_lights=1,
            sdr_lut_factor=10,
            exposure=1,
            show_envmap_as_skybox=show_envmap_as_skybox,
            reflective_plane_screen_texcoord=reflective_plane_screen_texcoord,
        )

    def set_reflective_plane_visibility(self, new_visibility: bool) -> None:
        self.pipeline.show_reflective_plane = new_visibility

    def write_cube_buffer(self, half_envmap: VoitEnvmap | None):
        if half_envmap is not None:
            self.pipeline.env_map = half_envmap._env_map_internal
        else:
            self.pipeline.env_map = None

    def render_single_RGBB_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Render a single new frame to a numpy array.

        This function makes sure that the returned numpy array contains
        the rendered frame (and not the previous one).

        Returns
        -------
        rgb_image
            The captured RGB image. Format: ``Im::RGBLike``
        zbuf_image
            The captured zbuffer image. Format: ``Im::Scalar``

        Notes
        -----
        This function instructs Panda3d to render only one (or two for technical
        reasons) frame. This means that if something has 1-2 frame delay, then that
        may be broken in the rendered frame.
        """
        self.pipeline.update_manual()
        self._render_frame_sync()
        rgb_image = self._capture_last_RGB()
        zbuf_image = self._capture_last_zbuffer()

        return rgb_image, zbuf_image

    def set_envmap_transform_inv(self, ibl_transform_inv: np.ndarray) -> None:
        """
        Set the transform that converts the world coordinate system to the local coordinate system of the environment map.

        Parameters
        ----------
        ibl_transform_inv
            The transform. Format: ``Transform::*[3x3]``

        Raises
        ------
        VoitError
            If the format of the array is incorrect.
        """
        if ibl_transform_inv.shape != (3, 3):
            raise VoitError(
                "The inverse transform of the environment map should be a 3x3 matrix."
            )

        ibl_transform_mat_tmp = np.concatenate(
            [
                ibl_transform_inv,
                np.array([[0, 0, 0]], dtype=ibl_transform_inv.dtype),
            ],
            axis=0,
        )
        ibl_transform_mat_tmp = np.concatenate(
            [
                ibl_transform_mat_tmp,
                np.array(
                    [
                        [0],
                        [0],
                        [0],
                        [1],
                    ],
                    dtype=ibl_transform_mat_tmp.dtype,
                ),
            ],
            axis=1,
        )
        ibl_transform_inv_mat = np_mat4x4_2_panda_mat(ibl_transform_mat_tmp.T)
        self.pipeline.world_2_env_mat = ibl_transform_inv_mat

    def set_envmap(self, half_env_map: VoitEnvmap) -> None:
        self.pipeline.env_map = half_env_map._env_map_internal

    def _render_frame_sync(self) -> None:
        """
        Instruct Panda3d to render a single frame.

        This function blocks until the rendering is actually done
        (`panda3d.core.GraphicsEngine.renderFrame` is asynchronous).
        -----
        """
        # initiate rendering
        self.graphicsEngine.renderFrame()
        # another rendering (Panda3d commonly calls
        # this function twice)
        # see https://discourse.panda3d.org/t/forcing-render-to-update/5750/3
        self.graphicsEngine.renderFrame()
        # wait for the render to complete
        self.graphicsEngine.syncFrame()

    def _capture_last_RGB(self) -> np.ndarray:
        """
        Get the last rendered RGB image as a Numpy array.

        It assumes without checking that the frame buffer
        uses unsigned bytes to represent colors.

        Returns
        -------
        v
            Format: ``Im::RGBs[Single]``
        """
        # This function is somewhat hacky, since the frame buffer properties
        # expected by this function are hardware-dependent.
        assert self.win is not None
        screenshot = self.win.get_screenshot()
        return read_2d_rgba_texture(screenshot)

    def _capture_last_zbuffer(self) -> np.ndarray:
        """
        Get the last Z-buffer data as a Numpy array.

        Returns
        -------
        v
            Format: ``Im::ZBuffers[Single]``
        """
        return read_depth_texture(self.pipeline.depth_texture)


def get_cw_rot_mat_around_z(deg: float) -> np.ndarray:
    """
    Get a rotation matrix that applies a rotation around the Z-axis.

    The direction of the rotation is clockwise if the camera looks at ``(0, 0, 0)`` from ``(0, 0, 1)`` with up vector ``(0, 1, 0)`` and no mirroring is applied.

    Parameters
    ----------
    deg
        The amount of rotation in degrees.

    Returns
    -------
    v
        The rotation matrix. Format: ``Transform_4x4``

    Developer notes
    ---------------
    Solely based on this functionality, this function would belong to the mathematical module. However, the clockwise-ness of the rotations actually depend on the coordinate system, which is specific to the Panda3d code.
    """
    m_rad = -deg / 180 * math.pi

    return np.array(
        [
            [math.cos(m_rad), -math.sin(m_rad), 0, 0],
            [math.sin(m_rad), math.cos(m_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _has_unsigned_byte_rgba_format(got_buffer_props: FrameBufferProperties) -> bool:
    r_bits = got_buffer_props.getRedBits()
    g_bits = got_buffer_props.getGreenBits()
    b_bits = got_buffer_props.getBlueBits()
    depth_bits = got_buffer_props.getDepthBits()
    color_bits = got_buffer_props.getColorBits()
    float_color = got_buffer_props.getFloatColor()

    return (
        (r_bits == 8)
        and (g_bits == 8)
        and (b_bits == 8)
        and (depth_bits == 24)
        and (color_bits == 24)
        and (float_color == False)
    )


def _get_properties_copy(buffer: GraphicsOutput) -> WindowProperties:
    """
    Get a copy of the `panda3d.core.WindowProperties` of the specified buffer.

    If the buffer is a `panda3d.core.GraphicsWindow`, then it uses the copy constructor of `panda3d.core.WindowProperties`. Otherwise, it creates a new `panda3d.core.WindowProperties` from the width and the height of the buffer.

    Parameters
    ----------
    buffer
        The buffer to use.

    Returns
    -------
    v
        The created `panda3d.core.WindowProperties`.
    """
    if isinstance(buffer, GraphicsWindow):
        return WindowProperties(buffer.getProperties())
    else:
        props = WindowProperties()
        props.setSize(x_size=buffer.getXSize(), y_size=buffer.getYSize())
        return props


def new_lens_from_proj_mat(proj_mat: np.ndarray) -> MatrixLens:
    """
    Create a `panda3d.core.MatrixLens` from the given projection matrix.

    This function assumes that the given projection matrix is designed to transforms column vectors. This function internally transposes the given projection matrix to transform row vectors instead.

    Parameters
    ----------
    proj_mat
        The projection matrix. Format: ``Transform::Homog[4x4]``
    """
    lens = MatrixLens()
    user_mat = np_mat4x4_2_panda_mat(proj_mat.T)
    lens.set_user_mat(user_mat)
    return lens


def put_obj(path: NodePath, new_obj: PandaNode) -> None:
    """
    Replace the object at the specified node path.

    This function uses the `panda3d.core.NodePath.removeNode` to remove the node at the specified node path.

    Parameters
    ----------
    path
        The node path to remove.
    new_obj
        The object to add to the parent of `path`.

    Raises
    ------
    Panda3dAssumptionViolation
        If the object does not have any parent.
    """
    if not path.hasParent():
        raise Panda3dAssumptionViolation("The object does not have any parent.")
    parent: NodePath = path.getParent()
    path.removeNode()
    parent.attachNewNode(new_obj)


def is_geom_node_obj(obj: NodePath) -> "TypeGuard[NodePath[GeomNode]]":
    node = obj.node()
    return isinstance(node, GeomNode)


def set_col_in_vertex_data(
    vertex_data: GeomVertexData,
    col_name: "VtxColName",
    new_values: np.ndarray,
) -> None:
    """
    Transform the values of the specfied column of a `panda3d.core.GeomVertexData`.

    This function assumes without checking that all columns use the float data type.

    Parameters
    ----------
    vertex_data
        The `panda3d.core.GeomVertexData` to modify.
    col_name
        The name of the column to modify.
    new_values
        The array of the new values. This is a two dimensional array.

    Raises
    ------
    Panda3dAssumptionViolation
        If the column does not exist or the data types of all columns in the specified geom vertex data are not float32.
    """

    col_info = _get_column_info(col_name, vertex_data)
    v_array = vertex_data.modifyArray(col_info.array_index_in_geom_vertex_data)
    float_view = _get_memoryview_for_pure_float_array_data(v_array)
    np_array = np.asarray(float_view).reshape((v_array.getNumRows(), -1))
    np_array[:, col_info.index_range_in_array] = new_values.astype(np.float32)
    float_view[:] = np_array.reshape((-1,))  # type: ignore


def get_col_in_vertex_data(
    vertex_data: GeomVertexData,
    col_name: "VtxColName",
) -> np.ndarray:
    """
    Get the values of the specfied column of a `panda3d.core.GeomVertexData`.

    This function assumes without checking that all columns use the float data type.

    Parameters
    ----------
    vertex_data
        The `panda3d.core.GeomVertexData` to modify.
    col_name
        The name of the column to modify.
    new_values
        The array of the new values. This is a two dimensional array.

    Raises
    ------
    Panda3dAssumptionViolation
        If the column does not exist or the data types of all columns in the specified geom vertex data are not float32.
    """

    col_info = _get_column_info(col_name, vertex_data)
    v_array = vertex_data.modifyArray(col_info.array_index_in_geom_vertex_data)
    float_view = _get_memoryview_for_pure_float_array_data(v_array)
    np_array = np.asarray(float_view).reshape((v_array.getNumRows(), -1))
    return np_array[:, col_info.index_range_in_array].copy()


class VtxColName(Enum):
    Normal = ({str(InternalName.get_normal())},)
    Vertex = ({str(InternalName.get_vertex())},)
    Texcoord = (
        {str(InternalName.get_texcoord_name("0")), str(InternalName.get_texcoord())},
    )

    def __init__(self, panda3d_names: set[str]):
        self.panda3d_names = frozenset(panda3d_names)


def _get_column_info(
    col_name: VtxColName, geom_vertex_data: GeomVertexData
) -> "_ColumnInfo":
    for i_arr, arr in enumerate(geom_vertex_data.getArrays()):
        arr_format = arr.getArrayFormat()
        columns = arr_format.getColumns()
        len_acc = 0
        for col in columns:
            if str(col.getName()) in col_name.panda3d_names:
                array_idxs = list(np.arange(0, col.getNumValues()) + len_acc)
                return _ColumnInfo(
                    index_range_in_array=array_idxs,
                    array_index_in_geom_vertex_data=i_arr,
                )
            else:
                len_acc += col.getNumValues()
    else:
        raise Panda3dAssumptionViolation(f'The column "{col_name}" was not found.')


def _get_memoryview_for_pure_float_array_data(
    array_data: GeomVertexArrayData,
) -> memoryview:
    raw_view = memoryview(array_data)  # type: ignore
    total_len = raw_view.nbytes // 4
    float_view = raw_view.cast("B").cast("f", (total_len,))

    return float_view  # type: ignore


def load_model_from_local_file(base: ShowBase, model_path: Path) -> NodePath:
    """
    A simple function to load a 3d model using Panda3d from a local model file.

    Unlike the original Panda3d model loader function, this function by default provides more information, disables model caching and supports Windows-style paths.

    This function keeps the hierarchy of the objects in the model.

    Parameters
    ----------
    base
        The `direct.showbase.ShowBase.ShowBase` to load the model.
    model_path
        The path of the model.

    Returns
    -------
    v
        The `panda3d.core.NodePath` of the loaded model.

    Raises
    ------
    VoitError
        If the model was not loadable.
    """
    options = LoaderOptions(LoaderOptions.LFReportErrors | LoaderOptions.LFNoCache)
    unix_style_path = convert_path_to_unix_style(model_path)
    model: NodePath | None = base.loader.loadModel(unix_style_path, loaderOptions=options)  # type: ignore

    if model is None:
        raise VoitError("Failed to load the model.")

    return model


def convert_path_to_unix_style(path: Path) -> str:
    """
    Convert the path to the Unix-style style required by Panda3d on Windows. This function makes the specified path absolute.

    This function does not change the paths in all platforms, but Windows.

    Parameters
    ----------
    path
        The path to convert.

    Returns
    -------
    v
        The path as a string.
    """
    if sys.platform != "win32":
        return str(path)

    path_parts = list(path.resolve().parts)

    assert len(path_parts) > 0
    if path_parts[0].endswith(":/") or path_parts[0].endswith(":\\"):
        path_parts[0] = path_parts[0][:-2].lower()

    full_path = "/" + ("/".join(path_parts))
    return full_path


def panda_mat4_2_np_mat(mat: LMatrix4f) -> np.ndarray:
    """
    Convert the given Panda3d matrix to a Numpy matrix.

    Returns
    -------
    v
        The created Numpy matrix. Format: ``Transform_4x4``
    """
    return np.array(
        [
            [
                mat.get_cell(0, 0),
                mat.get_cell(0, 1),
                mat.get_cell(0, 2),
                mat.get_cell(0, 3),
            ],
            [
                mat.get_cell(1, 0),
                mat.get_cell(1, 1),
                mat.get_cell(1, 2),
                mat.get_cell(1, 3),
            ],
            [
                mat.get_cell(2, 0),
                mat.get_cell(2, 1),
                mat.get_cell(2, 2),
                mat.get_cell(2, 3),
            ],
            [
                mat.get_cell(3, 0),
                mat.get_cell(3, 1),
                mat.get_cell(3, 2),
                mat.get_cell(3, 3),
            ],
        ],
        dtype=np.float32,
    )


def np_mat3x3_2_panda_mat(np_mat: np.ndarray) -> LMatrix3f:
    """
    Convert the given 3x3 numpy matrix to a Panda3d matrix. This function does not transpose the matrix.

    Parameters
    ----------
    np_mat
        The numpy matrix. Format: ``Transform::*[3x3]``

    Returns
    -------
    v
        The created Panda3d matrix.

    Raises
    ------
    VoitError
        If the shape of the array is not equal to ``(3, 3)``.
    """
    if np_mat.shape != (3, 3):
        raise VoitError(f"The shape ({np_mat.shape}) of the array is not (3, 3).")

    return LMatrix3f(
        (np_mat[0, 0], np_mat[0, 1], np_mat[0, 2]),
        (np_mat[1, 0], np_mat[1, 1], np_mat[1, 2]),
        (np_mat[2, 0], np_mat[2, 1], np_mat[2, 2]),
    )


def np_mat4x4_2_panda_mat(np_mat: np.ndarray) -> LMatrix4f:
    """
    Convert the given 4x4 numpy matrix to a Panda3d matrix. This function does not transpose the matrix.

    Parameters
    ----------
    np_mat
        The numpy matrix. Format: ``Transform::*[3x3]``

    Returns
    -------
    v
        The created Panda3d matrix.

    Raises
    ------
    VoitError
        If the shape of the array is not equal to ``(4, 4)``.
    """
    if np_mat.shape != (4, 4):
        raise VoitError(f"The shape ({np_mat.shape}) of the array is not (3, 3).")

    return LMatrix4f(
        (np_mat[0, 0], np_mat[0, 1], np_mat[0, 2], np_mat[0, 3]),
        (np_mat[1, 0], np_mat[1, 1], np_mat[1, 2], np_mat[1, 3]),
        (np_mat[2, 0], np_mat[2, 1], np_mat[2, 2], np_mat[2, 3]),
        (np_mat[3, 0], np_mat[3, 1], np_mat[3, 2], np_mat[3, 3]),
    )


def read_depth_texture(texture: Texture) -> np.ndarray:
    if texture.format != Texture.F_depth_component32:
        raise VoitError(
            f"The format of the texture is not (Texture.F_depth_component32). Current format code: {texture.format}"
        )
    ram_image = texture.get_ram_image()
    if ram_image is None:
        raise VoitError(f"Failed to read the RAM image of the texture.")
    width = texture.get_x_size()
    height = texture.get_y_size()
    ram_image_memview = memoryview(ram_image)  # type: ignore
    ram_image_memview_casted = ram_image_memview.cast("f", (1, height, width))
    im = np.array(ram_image_memview_casted)
    im = im * 2 - 1
    im = im[:, ::-1, :]
    return im.copy()


def read_2d_rgba_texture(texture: Texture) -> np.ndarray:
    """
    Read an RGBA texture to an RGB image.

    Parameters
    ----------
    texture
        The texture to read.

    Returns
    -------
    v
        The image of the texture. Format: ``Im_RGBLike``

    Raises
    ------
    VoitError
        If the texture does not have `panda3d.core.Texture.F_rgba8` or `panda3d.core.Texture.F_rgba` format.

        If the texture has `panda3d.core.Texture.F_rgba` format and the number of the actual bits per channel is not 8.
    """
    texture_is_rgba8 = texture.format != Texture.F_rgba8
    texture_is_any_rgba = texture.format != Texture.F_rgba
    if texture_is_rgba8 and texture_is_any_rgba:
        raise VoitError(
            f"The format of the texture is not (Texture.F_rgba8 or Texture.F_rgba). Current format code: {texture.format}"
        )
    ram_image = texture.get_ram_image()
    if ram_image is None:
        raise VoitError(f"Failed to read the RAM image of the texture.")
    width = texture.get_x_size()
    height = texture.get_y_size()
    ram_image_memview = memoryview(ram_image)  # type: ignore
    if ram_image_memview.nbytes != height * width * 4:
        raise VoitError(f"Only the reading of 8-bit channels are supported.")
    ram_image_memview_casted = ram_image_memview.cast("B", (height, width, 4))
    im = np.array(ram_image_memview_casted)[:, :, [2, 1, 0]]
    im = im.astype(np.float32) / 255.0
    im = im.transpose([2, 0, 1])
    # flip the Y axis of the image (see https://github.com/panda3d/panda3d/issues/789)
    im = im[:, ::-1, :]
    return im.copy()


def write_2d_rgba_texture(texture: Texture, im: np.ndarray):
    """
    Write the given RGB image to the given texture.

    The function clips the channel values of the input image to the ``[0, 1]`` range without warning.

    Parameters
    ----------
    texture
        The texture to modify.
    im
        The new image of the texture. Format: ``Im_RGBLike``
    """
    if texture.format != Texture.F_rgba8:
        raise VoitError(
            f"The format of the texture is not (Texture.F_rgba8). Current format code: {texture.format}"
        )

    if texture.component_type != Texture.T_unsigned_byte:
        raise VoitError(
            f"The format of the components of the texture is not  (Texture.T_byte). Current component type code: {texture.component_type}"
        )

    expected_shape = (3, texture.get_y_size(), texture.get_x_size())
    if im.shape != expected_shape:
        raise VoitError(
            f"The shape of the texture ({im.shape}) is not equal to {expected_shape}."
        )
    if not np.issubdtype(im.dtype, np.floating):
        raise VoitError(
            f'The dtype ({im.dtype}) of the array is not a subdtype of "numpy.float32".'
        )

    # make sure that the channel values are in the [0, 1] range
    # (the .astype(np.uint8) would produce 0 for those values)
    im = np.clip(im, 0, 1)

    # the textures always use BGRA format internally (see https://discourse.panda3d.org/t/opencv-videocapture-to-panda3d-texture/25981/4)
    # rgba->bgra; chw -> hwc; array -> bytes
    im = np.concatenate([im, np.ones((1, im.shape[1], im.shape[2]), dtype=np.float32)])
    im_bytes = (
        (im[[2, 1, 0, 3], ::-1, :].transpose([1, 2, 0]) * 255)
        .astype(np.uint8)
        .tobytes(order="C")
    )
    texture.set_ram_image(im_bytes, Texture.CM_off)


def set_simple_material(
    obj: NodePath,
    texture_size: Vec2i,
) -> tuple[Texture, Texture]:
    """
    Configure a new material on the object with the following properties:

    * The metallic value is 0.
    * The base color and roughness is set from textures.
    * There is no emission.

    Parameters
    ----------
    obj
        The object to configure.
    texture_size
        The size of the base color and roughness textures.

    Returns
    -------
    base_color_tex
        The texture for the base color.
    roughness_tex
        The texture for the roughness.
    """
    # do we need this? https://docs.panda3d.org/1.10/python/programming/texturing/choosing-a-texture-size#padded-textures

    mat = Material(obj.name + "_material")
    mat.set_base_color((1.0, 1.0, 1.0, 1.0))
    mat.set_metallic(1.0)
    mat.set_roughness(1.0)

    base_color_tex = Texture("color")
    base_color_tex.setup_2d_texture(
        texture_size.x, texture_size.y, Texture.T_unsigned_byte, Texture.F_rgba8
    )
    metallic_roughness_tex = Texture("roughness")
    metallic_roughness_tex.setup_2d_texture(
        texture_size.x, texture_size.y, Texture.T_unsigned_byte, Texture.F_rgba8
    )

    tex_attrib = _new_texture_attrib(
        base_color_tex=base_color_tex, metallic_roughness_tex=metallic_roughness_tex
    )

    obj.set_attrib(MaterialAttrib.make(mat))
    obj.set_attrib(tex_attrib)

    return base_color_tex, metallic_roughness_tex


def _new_texture_attrib(base_color_tex: Texture, metallic_roughness_tex: Texture):
    tex_attrib = TextureAttrib.make()

    base_color_stage = TextureStage("base_color")
    base_color_stage.set_sort(0)
    base_color_stage.set_texcoord_name(InternalName.getTexcoord())
    base_color_stage.set_mode(TextureStage.M_modulate)

    tex_attrib = tex_attrib.add_on_stage(base_color_stage, base_color_tex)

    metallic_roughness_stage = TextureStage("roughness")
    metallic_roughness_stage.set_sort(1)
    metallic_roughness_stage.set_texcoord_name(InternalName.getTexcoord())
    # M_selector = (occlusion-)metallic-roughness (https://github.com/panda3d/panda3d/issues/1388)
    metallic_roughness_stage.set_mode(TextureStage.M_selector)

    tex_attrib = tex_attrib.add_on_stage(
        metallic_roughness_stage, metallic_roughness_tex
    )

    return tex_attrib


def general_vec3_2_panda3d_coord_sys(vec: Vec3):
    return Vec3(vec.x, vec.z, vec.y)


def camera_2_panda3d_proj_mat(
    camera: depth_tools.CameraIntrinsics,
    near_far: tuple[float, float],
    im_size: Vec2i,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Calculate the user matrix expected by `panda3d.core.MatrixLens` from the specified intrinsic matrix and near and far plane distances. This function assumes that the film size is 2 (this is the default).

    This function does not make any assumption about the order or validity of the near and far planes.

    This function also explicitly gives the necessary ``(a, b)`` tuple to restore the depth from the zbuffer. The values in the zbuffer are calculated using the following formula: ``zbuf = (a*depth+b)/depth``.

    Parameters
    ----------
    intrinsic_mat
        The intrinsic matrix. Format: ``Transform::*[3x3]``
    near_far
        A tuple specifying the distance of the near and far planes.

    Returns
    -------
    proj_mat
        The calculated projection matrix.
    a_b
        The mentioned ``(a, b)`` tuple.

    """
    zup_rh_2_yup_lh = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    a, b = get_ab_from_near_far_and_zbuf_range(
        near_far=near_far,
        zbuf_from_to=(-1, 1),
    )

    MATRIX_LENS_FILM_SIZE = 2

    lh_proj_mat = np.array(
        [
            [
                camera.f_x * MATRIX_LENS_FILM_SIZE / im_size.x,
                0,
                camera.c_x * MATRIX_LENS_FILM_SIZE / im_size.x - 1,
                0,
            ],
            [
                0,
                camera.f_y * MATRIX_LENS_FILM_SIZE / im_size.y,
                camera.c_y * MATRIX_LENS_FILM_SIZE / im_size.y - 1,
                0,
            ],
            [0, 0, a, b],
            [0, 0, 1, 0],
        ],
    )

    return lh_proj_mat @ zup_rh_2_yup_lh, (a, b)


def add_shadow_casting_dir_light(
    lighted_objects_root: NodePath, shadow_map_size: int, parent: NodePath, name: str
) -> "NodePath[DirectionalLight]":
    """
    Add a new directional light that casts shadows.

    This function does not configure the color of the light.

    Parameters
    ----------
    lighted_objects_root
        This object (and its descendants) will be lighted by this light.
    shadow_map_size
        The size of the shadow map.
    parent
        The parent object of the light.
    name
        The name of the object.

    Returns
    -------
    v
        The new directional light.
    """
    dir_light_node = DirectionalLight(name)
    dir_light = NodePath(dir_light_node)
    dir_light_node.set_shadow_caster(True, shadow_map_size, shadow_map_size)
    dir_light.reparent_to(parent)
    lighted_objects_root.set_light(dir_light)
    return dir_light


def update_shadow_casting_dir_light(
    related_objects_root: NodePath,
    dir_light: "NodePath[DirectionalLight]",
    direction: Vec3,
) -> None:
    """
    Update a shadow casting directional light to properly cast the shadows to the given direction.

    The function creates a warning if the length of the direction is almost zero or the root object of the lighting related objects does not have any vertex, including its children.

    The shadow casting does not have to be actually enabled on the updated directional light, but some steps executed by this function are unnecessary in that case.

    Parameters
    ----------
    related_objects_root
        The root of all objects that are related to the shadow casting in any way (cast shadows or casted shadows onto them).
    dir_light
        The directional light to be updated.
    direction
        A direction vector that describes the (new) direction of the directional light in the coordinate system of its parent. This is given in the Z-up right handed coordinate sytem of Panda3d. The vector does not have to be normalized, because the function normalizes it before further processing.

    Raises
    ------
    VoitError
        If the light has no parent.
    """
    if direction.vec_len() < 1e-30:
        VOIT_LOGGER.warning(
            f"The configured relative direction vector of the directional light ({dir_light}) is almost equal to the zero vector. Using failback value ({Vec3(1, 1, 1)}) instead. Direction vector: {direction}; Length threshold: 1e-30"
        )
        direction = Vec3(1, 1, 1)

    direction = direction.normalize()

    # calculate the position of the light
    dir_light_parent = dir_light.parent
    if dir_light_parent is None:
        raise VoitError(f"The directional light {dir_light} has no parent.")
    tight_bounds = related_objects_root.get_tight_bounds(dir_light_parent)
    if tight_bounds is not None:
        old_bmin, old_bmax = tight_bounds
        old_b_diameter = (old_bmax - old_bmin).length()
    else:
        VOIT_LOGGER.warning(
            f"The object, {related_objects_root} specified as the root of the objects related to shadow casting does not contain any vertex, including its children."
        )
        old_b_diameter = 1.0
    related_objects_root_rel_pos = related_objects_root.get_pos(dir_light_parent)

    direction_p_vec = LVector3f(direction.x, direction.y, direction.z)
    light_pos = -direction_p_vec * old_b_diameter + related_objects_root_rel_pos
    dir_light.set_pos(light_pos)

    dir_light.look_at(other=dir_light_parent, point=related_objects_root_rel_pos)

    # update shadow configurations
    tight_bounds = related_objects_root.get_tight_bounds(dir_light)
    if tight_bounds is not None:
        bmin, bmax = tight_bounds

        light_lens = dir_light.node().get_lens()

        light_lens.set_film_offset((bmin.xz + bmax.xz) * 0.5)
        light_lens.set_film_size(bmax.xz - bmin.xz)
        light_lens.set_near_far(bmin.y, bmax.y)


@dataclass
class _ColumnInfo:
    index_range_in_array: list[int]
    array_index_in_geom_vertex_data: int


class Panda3dAssumptionViolation(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
