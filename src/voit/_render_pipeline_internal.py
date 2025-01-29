from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict, cast

import numpy as np
from direct.showbase.ShowBase import MatrixLens, ShowBase
from panda3d.core import (
    Camera,
    DirectionalLight,
    FrameBufferProperties,
    GeomNode,
    GraphicsOutput,
    LVector3f,
    NodePath,
    PandaNode,
    Texture,
)
import depth_tools

from ._errors import VoitError
from ._light_conf_internal import LightConf
from ._logging_internal import VOIT_LOGGER
from ._math_internal import get_transformation_matrix, zbuf_2_depth
from ._panda3d_util_internal import (
    RendererShowBase,
    VtxColName,
    add_shadow_casting_dir_light,
    general_vec3_2_panda3d_coord_sys,
    get_cw_rot_mat_around_z,
    load_model_from_local_file,
    new_lens_from_proj_mat,
    np_mat4x4_2_panda_mat,
    set_col_in_vertex_data,
    set_simple_material,
    camera_2_panda3d_proj_mat,
    update_shadow_casting_dir_light,
    write_2d_rgba_texture,
)
from ._vectors import Vec2, Vec2i, Vec3


class RasterizingPipeline:
    """
    Implement a Panda3d-based rasterizing render pipeline.

    Parameters
    ----------
    t_proj_mat
        The theoretical projection matrix of the camera.
    im_size
        The size of the image to modify.
    floor_proxy_size
        The size of the floor proxy rectangle.
    shadow_map_size
        The resolution of the shadow map.
    near_far
        A tuple containing the depth values for the near and far planes in ``(near depth, far depth)`` form.
    debug_window
        If this argument is true, then the rendering will not be offscreen. It will be shown on a window instead. This is useful for debugging purposes.

    Raises
    ------
    VoitError
        If the following statement does not hold: ``0 < near plane depth < far plane depth``.
        If the shape of the modified image contains non-positive values.
        If the specified floor proxy size contains non-positive values.
        If the size of the shadow maps is non-positive.
        If the specified intrinsic matrix is not in ``R^{3x3}`` or it is not invertable.

    Developer notes
    ---------------
    Internally, this class creates a full Panda3d scene and manipulates it.

    Scene structure: ::

        render
        |- floor_parent
           |- floor_proxy_obj
           |- inserted_obj (if present)
        |- dir_light_parent
           |- dir_light
        |- camera

    The individual elements:

    * ``floor_proxy_obj``: The proxy mesh of the floor. The transform of the floor is applied on this object.
    * ``dir_light_parent``: The parent of the directional light. It is transformed to have the same orientation as the environment map and to have the same position as the floor.
    * ``dir_light``: The directional light.
    * ``inserted_obj``: The inserted object. The class does not change the relative transform of this object.
    * ``camera``: The camera. It is not transformed.

    The class internally uses a mixture of Panda3d conventions and global conventions. It uses Z-up right handed coordinate system, just like in Panda3d. However, it uses column vectors instead of the row vector convention of Panda3d. The arguments of the public functions of the class do not use the Panda3d conventions, they use the global conventions and do the necessary conversions themselves.

    The pipeline uses occlusion maps.
    """

    def __init__(
        self,
        camera: depth_tools.CameraIntrinsics,
        im_size: Vec2i,
        floor_proxy_size: Vec2,
        shadow_map_size: int,
        near_far: tuple[float, float],
        debug_window: bool = False,
    ):
        if not im_size.is_positive():
            raise VoitError(
                f"The size of the modified image ({im_size}) is not positive in all axes."
            )
        if not floor_proxy_size.is_positive():
            raise VoitError(
                f"The size of the floor proxy ({floor_proxy_size}) is not positive in all axes."
            )
        if not (0 < near_far[0] < near_far[1]):
            raise VoitError(
                f"The '0 < near_plane < far_plane' constraint does not hold. Near plane: {near_far[0]}, far plane: {near_far[1]}"
            )
        if shadow_map_size < 0:
            raise VoitError(
                f"The size of the shadow map ({shadow_map_size}) is not positive."
            )

        proj_mat, a_b = camera_2_panda3d_proj_mat(
            camera=camera, near_far=near_far, im_size=im_size
        )

        self._proj_mat = proj_mat
        self._show_base = RendererShowBase(
            proj_mat=proj_mat,
            im_size=im_size,
            offscreen=not debug_window,
            show_envmap_as_skybox=False,
            reflective_plane_screen_texcoord=True,
        )

        self._scene_structure = _create_scene(
            floor_proxy_size=floor_proxy_size,
            shadow_map_size=shadow_map_size,
            im_size=im_size,
            reflective_plane=self._show_base.pipeline.reflective_plane,
            render=self._show_base.render,
            reflective_plane_parent=self._show_base.pipeline.reflective_plane_parent,
        )
        self._show_base.pipeline.reflected_plane_local_normal = LVector3f(0, 0, 1)
        self._near_far = near_far
        self._inserted_obj: NodePath | None = None
        self._a_b = a_b

    def render_floor_and_obj(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render the floor and the object within the already configured scene.

        Returns
        -------
        full_rgb
            The rendered RGB image. Format: ``Im::RGBLike``
        full_depth
            The corresponding depth map. Format: ``Im::Scalar``
        floor_obj_mask
            The mask that selects both the object and the floor. Format: ``Im::Mask``

        Raises
        ------
        RuntimeError
            If the scene is not yet configured.
        """
        if self._inserted_obj is None:
            raise RuntimeError("The scene is not yet configured.")

        self._inserted_obj.show()
        full_rgb, floor_zbuf = self._show_base.render_single_RGBB_frame()
        full_depth = zbuf_2_depth(a_b=self._a_b, zbuf_data=floor_zbuf)
        floor_obj_mask = self._get_mask_for_depth(full_depth)

        return full_rgb, full_depth, floor_obj_mask

    def render_floor(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render the floor without the object. In this case, the object is completely hidden and does not cast any shadow.

        Returns
        -------
        full_rgb
            The rendered RGB image. Format: ``Im::RGBLike``
        full_depth
            The corresponding depth map. Format: ``Im::Scalar``
        floor_obj_mask
            The mask that selects the floor. Format: ``Im::Mask``

        Raises
        ------
        RuntimeError
            If the scene is not yet configured.
        """
        if self._inserted_obj is None:
            raise RuntimeError("The scene is not yet configured.")

        self._inserted_obj.hide()
        floor_rgb, floor_zbuf = self._show_base.render_single_RGBB_frame()
        floor_depth = zbuf_2_depth(a_b=self._a_b, zbuf_data=floor_zbuf)
        floor_mask = self._get_mask_for_depth(floor_depth)
        self._inserted_obj.show()
        return floor_rgb, floor_depth, floor_mask

    def _get_mask_for_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Calculate the mask that select the objects appearing on a scene based on the depth map.

        Parameters
        ----------
        depth
            The depth map. Format: ``Im::Scalar``

        Returns
        -------
        v
            The mask that selects the objects.
        """
        near, far = self._near_far
        max_relevant_depth = near + (far - near) * 0.99
        return depth < max_relevant_depth

    def configure_scene(
        self,
        obj: NodePath,
        floor_pos_in_view_space: Vec3,
        floor_y_in_view_space: Vec3,
        floor_z_in_view_space: Vec3,
        envmap_y_in_view_space: Vec3,
        envmap_z_in_view_space: Vec3,
        light_conf: LightConf,
        bg_im_base_color_map: np.ndarray,
        bg_im_metallic_roughness: np.ndarray,
        rotation_around_floor_normal_cw: float,
    ) -> NodePath:
        """
        Configure the scene.

        This function expects its arguments in the general coordinate system (Y-up left handed).

        The function warns you in the following cases:

        * The z coordinate of the Y axis of the floor in the view space is positive.
        * The floor is placed in front of the near plane or behind the far plane.

        The orientation of the lighting is spearately defined from the orientation of the inserted object, because it highly depends on the lighting estimation algorithm.

        Parameters
        ----------
        obj
            The inserted object.
        floor_pos_in_view_space
            The position of the floor in the view space.
        floor_y_in_view_space
            The Y axis of the floor in the view space.
        floor_z_in_view_space
            The z axis of the floor in the view space.
        envmap_y_in_view_space
            The Y axis of the local coordinate system of the environment map in the view space.
        envmap_z_in_view_space
            The Z axis of the local coordinate system of the environment map in the view space.
        light_conf
            The configuration of the lighting of the floor. The coordinate system of the lighting is given by ``envmap_y_in_view_space`` and ``envmap_z_in_view_space``.
        bg_im_base_color_map
            The base color map of the background. This specifies the base color for each pixel. Format: ``Im::RGBLike``
        bg_im_metallic_roughness
            The metallic-roguhness map of the background. The channel ``G`` is the roughness and the channel ``B`` is the metallic value. Both channels range from 0 to 1. Format: ``Im::RGBLike``
        rotation_around_floor_normal_cw
            The amount of clockwise rotation around the normal vector of the floor. The angle is in degree.

        Raises
        ------
        VoitError
            If the maps do not have the correct format.
        """
        if self._inserted_obj is not None:
            self._inserted_obj.detach_node()
            self._inserted_obj = None

        floor_y_in_view_space = _get_normalized_dir_vec_arg(
            floor_y_in_view_space, "the local Y axis of the floor"
        )
        floor_z_in_view_space = _get_normalized_dir_vec_arg(
            floor_z_in_view_space, "the local Z axis of the floor"
        )
        envmap_y_in_view_space = _get_normalized_dir_vec_arg(
            envmap_y_in_view_space, "the local Y axis of the environment map"
        )
        envmap_z_in_view_space = _get_normalized_dir_vec_arg(
            envmap_z_in_view_space, "the local Z axis of the environment map"
        )

        if floor_pos_in_view_space.z < self._near_far[0]:
            VOIT_LOGGER.warning(
                f"The floor proxy is placed in front of the near plane of the camera. Z coordinate of the floor: {floor_pos_in_view_space.z}. Near plane: {self._near_far[0]}"
            )
        if floor_pos_in_view_space.z > self._near_far[1]:
            VOIT_LOGGER.warning(
                f"The floor proxy is placed behind the far plane of the camera. Z coordinate of the floor: {floor_pos_in_view_space.z}. Far plane: {self._near_far[1]}"
            )
        if floor_pos_in_view_space.neg().dot(floor_y_in_view_space) < 0:
            VOIT_LOGGER.warning(
                f"The dot product of the vector pointing to the camera from the the origin of the inserted object and the normal vector of the floor is negative. This means that the camera is below the level of the floor in the coordinate system of the inserted object. This is not yet well implemented."
            )

        floor_yz_dot = floor_y_in_view_space.dot(floor_z_in_view_space)
        envmap_yz_dot = envmap_y_in_view_space.dot(envmap_z_in_view_space)

        if abs(floor_yz_dot) > 1e-6:
            VOIT_LOGGER.warning(
                f"The local Y and Z axis of the floor in the view coordinate system are not orthogonal. Local Y: {floor_y_in_view_space}; Local Z: {floor_z_in_view_space}; absolute dot product threshold: 1e-6."
            )
        if abs(envmap_yz_dot) > 1e-6:
            VOIT_LOGGER.warning(
                f"The local Y and Z axis of the environment map in the view coordinate system are not orthogonal. Local Y: {floor_y_in_view_space}; Local Z: {floor_z_in_view_space}; absolute dot product threshold: 1e-6."
            )

        # convert enverything to the coordinate system of Panda3d
        p3d_floor_z_in_view_space = general_vec3_2_panda3d_coord_sys(
            floor_y_in_view_space
        )
        p3d_floor_y_in_view_space = general_vec3_2_panda3d_coord_sys(
            floor_z_in_view_space
        )
        p3d_envmap_z_in_view_space = general_vec3_2_panda3d_coord_sys(
            envmap_y_in_view_space
        )
        p3d_envmap_y_in_view_space = general_vec3_2_panda3d_coord_sys(
            envmap_z_in_view_space
        )

        p3d_floor_pos_in_view_space = general_vec3_2_panda3d_coord_sys(
            floor_pos_in_view_space
        )

        # calculate the X axis of the floor in the view space
        p3d_floor_x_in_view_space = p3d_floor_y_in_view_space.cross(
            p3d_floor_z_in_view_space
        ).normalize()
        p3d_envmap_x_in_view_space = p3d_envmap_y_in_view_space.cross(
            p3d_envmap_z_in_view_space
        ).normalize()

        # do the actual configuration
        light_conf = LightConf(
            envmap=light_conf.envmap,
            dir_light_dir_and_color=(
                (
                    general_vec3_2_panda3d_coord_sys(
                        light_conf.dir_light_dir_and_color[0]
                    ),
                    light_conf.dir_light_dir_and_color[1],
                )
                if light_conf.dir_light_dir_and_color is not None
                else None
            ),
        )

        if light_conf.envmap is not None:
            self._show_base.set_envmap(light_conf.envmap)

        # add the object to the scene
        obj.reparent_to(self._scene_structure.floor_parent)
        floor_transf_mat = get_transformation_matrix(
            x_dir_in_view_space=p3d_floor_x_in_view_space,
            y_dir_in_view_space=p3d_floor_y_in_view_space,
            obj_pos=p3d_floor_pos_in_view_space,
        )
        obj_transf_mat = get_cw_rot_mat_around_z(rotation_around_floor_normal_cw)
        envmap_transf_mat = get_transformation_matrix(
            x_dir_in_view_space=p3d_envmap_x_in_view_space,
            y_dir_in_view_space=p3d_envmap_y_in_view_space,
            obj_pos=p3d_floor_pos_in_view_space,
        )
        p3d_floor_transf_mat_T = np_mat4x4_2_panda_mat(floor_transf_mat.T)
        self._scene_structure.floor_parent.set_mat(p3d_floor_transf_mat_T)
        p3d_envmap_transf_mat_T = np_mat4x4_2_panda_mat(envmap_transf_mat.T)
        self._scene_structure.dir_light_parent.set_mat(p3d_envmap_transf_mat_T)
        p3d_obj_transf_mat_T = np_mat4x4_2_panda_mat(obj_transf_mat.T)
        obj.set_mat(p3d_obj_transf_mat_T)

        # update floor object textures
        write_2d_rgba_texture(
            im=bg_im_base_color_map, texture=self._scene_structure.bg_base_color_tex
        )
        write_2d_rgba_texture(
            im=bg_im_metallic_roughness,
            texture=self._scene_structure.bg_metallic_roughness_tex,
        )

        # update directional light
        # set the directional light position based on its direction
        dir_light = self._scene_structure.dir_light
        if light_conf.dir_light_dir_and_color is None:
            dir_light.node().set_color((0, 0, 0, 1))
        else:
            new_dir, new_color = light_conf.dir_light_dir_and_color
            update_shadow_casting_dir_light(
                dir_light=self._scene_structure.dir_light,
                direction=new_dir,
                related_objects_root=self._scene_structure.floor_parent,
            )

            dir_light.node().set_color((new_color.x, new_color.y, new_color.z, 1))

        # update environment map
        if light_conf.envmap is None:
            self._show_base.write_cube_buffer(None)
        else:
            self._show_base.write_cube_buffer(light_conf.envmap)
            enmvmap_transform_inv = envmap_transf_mat[:3, :3].T
            self._show_base.set_envmap_transform_inv(enmvmap_transform_inv)

        # store the inserted object
        self._inserted_obj = obj

        return obj

    def destroy(self):
        self._show_base.destroy()

    def load_model(self, model_path: Path) -> "NodePath[PandaNode]":
        return load_model_from_local_file(base=self._show_base, model_path=model_path)


def _get_normalized_dir_vec_arg(direction: Vec3, description: str) -> Vec3:
    """
    Processs an argument that should specify a direction vector.

    If the length of the direction vector candidate is almost zero, then this function gives a warning. Message: ``The euclidean norm of {description} is almost zero. Using ({Vec3(1, 1, 1)}) instead. Vector: {dir_vec}. Threshold: 1e-30.``

    Otherwise, it returns with the normalized vector.
    """
    if direction.vec_len() < 1e-30:
        VOIT_LOGGER.warning(
            f"The euclidean norm of {description} is almost zero. Using ({Vec3(1, 1, 1)}) instead. Vector: {direction}. Threshold: 1e-30."
        )
        return Vec3(1, 1, 1).normalize()
    else:
        return direction.normalize()


def _create_scene(
    floor_proxy_size: Vec2,
    im_size: Vec2i,
    shadow_map_size: int,
    reflective_plane_parent: "NodePath",
    reflective_plane: "NodePath[GeomNode]",
    render: NodePath,
) -> "_SceneStructure":
    floor_proxy_obj = reflective_plane
    albedo_tex, roughness_tex = set_simple_material(
        obj=floor_proxy_obj, texture_size=im_size
    )

    floor_vertex_pos_table = np.array(
        [
            [-1, 1, 0],
            [1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
        ],
        dtype=np.float32,
    )
    floor_vertex_pos_table[:, 0] *= floor_proxy_size.x / 2
    floor_vertex_pos_table[:, 1] *= floor_proxy_size.y / 2
    floor_vertex_data = floor_proxy_obj.node().modify_geom(0).modify_vertex_data()

    if floor_vertex_data.get_num_rows() != 4:
        raise RuntimeError(
            "Internal error. The number of vertices in the floor proxy mesh is not equal to 4."
        )
    set_col_in_vertex_data(
        vertex_data=floor_proxy_obj.node().modify_geom(0).modify_vertex_data(),
        col_name=VtxColName.Vertex,
        new_values=floor_vertex_pos_table,
    )

    floor_proxy_obj.set_scale((floor_proxy_size.x, 1, floor_proxy_size.y))

    dir_light_parent = NodePath("dir_light_parent")
    dir_light_parent.reparent_to(render)

    dir_light = add_shadow_casting_dir_light(
        lighted_objects_root=render,
        parent=dir_light_parent,
        shadow_map_size=shadow_map_size,
        name="dir_light",
    )

    return _SceneStructure(
        dir_light=dir_light,
        floor_proxy_obj=floor_proxy_obj,
        bg_base_color_tex=albedo_tex,
        bg_metallic_roughness_tex=roughness_tex,
        dir_light_parent=dir_light_parent,
        floor_parent=reflective_plane_parent,
    )


@dataclass
class _SceneStructure:
    dir_light_parent: "NodePath"
    dir_light: "NodePath[DirectionalLight]"
    floor_proxy_obj: "NodePath[GeomNode]"
    bg_base_color_tex: Texture
    bg_metallic_roughness_tex: Texture
    floor_parent: NodePath
