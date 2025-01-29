import logging
import unittest
from typing import Literal, cast

import depth_tools
import numpy as np
from direct.showbase.ShowBase import NodePath, PandaNode
from panda3d.core import LVector4f, Texture, TextureAttrib, TextureStage

from voit import Vec2, Vec2i, Vec3, VoitError
from voit._analytical_light_estimation import VoitEnvmap
from voit._panda3d_util_internal import (
    RendererShowBase,
    VtxColName,
    camera_2_panda3d_proj_mat,
    general_vec3_2_panda3d_coord_sys,
    get_col_in_vertex_data,
    read_2d_rgba_texture,
)
from voit._render_pipeline_internal import LightConf, RasterizingPipeline

from .testutil import VoitTestBase


class RenderPipelineTest(VoitTestBase):
    def setUp(self) -> None:
        self.camera = depth_tools.CameraIntrinsics(f_x=200, f_y=200, c_x=100, c_y=79)
        self.floor_proxy_size = Vec2(0.9, 0.5)
        self.im_size = Vec2i(200, 159)
        self.near_far = (0.3, 113)
        self.shadow_map_size = 1024

    def tearDown(self) -> None:
        self.destroy_showbase()

    def test_init__happy_path(self):
        with self.assertNoLogs(self.voit_logger):
            pipeline = RasterizingPipeline(
                camera=self.camera,
                floor_proxy_size=self.floor_proxy_size,
                im_size=self.im_size,
                near_far=self.near_far,
                shadow_map_size=self.shadow_map_size,
            )
        self.assertEqual(pipeline._near_far, self.near_far)
        self.assertIsInstance(pipeline._show_base, RendererShowBase)
        self.assertAncestorOf(
            pipeline._show_base.render, pipeline._scene_structure.dir_light
        )
        self.assertParentOf(
            pipeline._show_base.render,
            pipeline._scene_structure.floor_parent,
        )
        self.assertParentOf(
            pipeline._scene_structure.floor_parent,
            pipeline._scene_structure.floor_proxy_obj,
        )
        self.assertIsNone(pipeline._inserted_obj)

        floor_node = pipeline._scene_structure.floor_proxy_obj.node()
        self.assertEqual(floor_node.get_num_geoms(), 1)

        floor_geom = floor_node.get_geom(0)
        self.assertTrue(floor_geom.get_primitive(0).is_indexed())
        self.assertEqual(floor_node.get_nested_vertices(), 6)

        self.assertTrue(
            pipeline._show_base.render.has_light(pipeline._scene_structure.dir_light)
        )

        expected_pipeline_proj_mat, _ = camera_2_panda3d_proj_mat(
            camera=self.camera, im_size=self.im_size, near_far=self.near_far
        )

        self.assertAllclose(expected_pipeline_proj_mat, pipeline._proj_mat)
        self.assertIsNone(pipeline._show_base.pipeline.env_map)

        # reflective plane textures
        actual_base_color_tex = self._get_texture(
            pipeline._scene_structure.floor_proxy_obj, "base_color"
        )
        actual_metallic_roughness_tex = self._get_texture(
            pipeline._scene_structure.floor_proxy_obj, "metallic_roughness"
        )
        expected_base_color_tex = pipeline._scene_structure.bg_base_color_tex
        expected_metallic_roughness_tex = (
            pipeline._scene_structure.bg_metallic_roughness_tex
        )
        self.assertEqual(actual_base_color_tex, expected_base_color_tex)
        self.assertEqual(actual_metallic_roughness_tex, expected_metallic_roughness_tex)
        self.assertNotEqual(actual_base_color_tex, actual_metallic_roughness_tex)

        # important pipeline-features
        self.assertTrue(pipeline._show_base.pipeline.use_occlusion_maps)
        self.assertTrue(pipeline._show_base.pipeline.enable_shadows)
        self.assertTrue(pipeline._show_base.pipeline.reflective_plane_screen_texcoord)

    def _get_texture(
        self, obj: NodePath, texture_type: Literal["base_color", "metallic_roughness"]
    ) -> Texture:
        texture_attrib = cast(TextureAttrib, obj.get_attrib(TextureAttrib))
        expected_mode = {
            "base_color": TextureStage.M_modulate,
            "metallic_roughness": TextureStage.M_selector,
        }[texture_type]
        for stage in texture_attrib.get_on_stages():
            if stage.get_mode() == expected_mode:
                return obj.get_texture(stage)
        else:
            self.fail()

    def test_init__im_size_non_positive(self):
        im_size = Vec2i(0, 300)
        with self.assertRaises(VoitError) as cm:
            RasterizingPipeline(
                camera=self.camera,
                floor_proxy_size=self.floor_proxy_size,
                im_size=im_size,
                near_far=self.near_far,
                shadow_map_size=self.shadow_map_size,
            )
        msg = str(cm.exception)
        self.assertIn(str(im_size), msg)

    def test_init__fps_non_positive(self):
        floor_proxy_size = Vec2(3, 0)
        with self.assertRaises(VoitError) as cm:
            RasterizingPipeline(
                camera=self.camera,
                floor_proxy_size=floor_proxy_size,
                im_size=self.im_size,
                near_far=self.near_far,
                shadow_map_size=self.shadow_map_size,
            )
        msg = str(cm.exception)
        self.assertIn(str(floor_proxy_size), msg)

    def test_init__invalid_near(self):
        with self.assertRaises(VoitError) as cm:
            RasterizingPipeline(
                camera=self.camera,
                floor_proxy_size=self.floor_proxy_size,
                im_size=self.im_size,
                near_far=(-0.00001, 10000),
                shadow_map_size=self.shadow_map_size,
            )
        msg = str(cm.exception)
        self.assertIn(str(-0.00001), msg)
        self.assertIn(str(10000), msg)

    def test_init__invalid_far(self):
        with self.assertRaises(VoitError) as cm:
            RasterizingPipeline(
                camera=self.camera,
                floor_proxy_size=self.floor_proxy_size,
                im_size=self.im_size,
                near_far=(0.2, 0.1),
                shadow_map_size=self.shadow_map_size,
            )
        msg = str(cm.exception)
        self.assertIn(str(0.2), msg)
        self.assertIn(str(0.1), msg)

    def test_configure_scene__happy_path(self):
        im_shape_tuple = (3, self.im_size.y, self.im_size.x)
        pipeline = RasterizingPipeline(
            camera=self.camera,
            floor_proxy_size=self.floor_proxy_size,
            im_size=self.im_size,
            near_far=self.near_far,
            shadow_map_size=self.shadow_map_size,
        )
        bg_im_base_color_map = np.full(im_shape_tuple, 0.5)
        bg_im_metallic_roughness = np.full(im_shape_tuple, 0.1)
        envmap = np.full((3, 10, 20), 0.9)
        expected_light_dir = Vec3(1, 1.5, 2.5)
        expected_light_color = Vec3(0.9, 0.8, 0.3)
        inserted_obj = NodePath(PandaNode("node1"))

        floor_y_in_view_space = Vec3(-3, 0, -1)
        floor_z_in_view_space = Vec3(-1, 1, 3)
        envmap_y_in_view_space = Vec3(3, 9, -2)
        envmap_z_in_view_space = Vec3(10, -4, -3)
        floor_pos_in_view_space = Vec3(1, 5, 2)

        with self.assertNoLogs(self.voit_logger, level="WARNING"):
            pipeline.configure_scene(
                obj=inserted_obj,
                bg_im_base_color_map=bg_im_base_color_map,
                bg_im_metallic_roughness=bg_im_metallic_roughness,
                floor_pos_in_view_space=floor_pos_in_view_space,
                floor_z_in_view_space=floor_z_in_view_space,
                floor_y_in_view_space=floor_y_in_view_space,
                envmap_y_in_view_space=envmap_y_in_view_space,
                envmap_z_in_view_space=envmap_z_in_view_space,
                light_conf=LightConf(
                    envmap=VoitEnvmap.from_image(envmap=envmap, envmap_linear=True),
                    dir_light_dir_and_color=(expected_light_dir, expected_light_color),
                ),
                rotation_around_floor_normal_cw=32.7,
            )

        self.assertIsNotNone(pipeline._inserted_obj)
        assert pipeline._inserted_obj is not None  # for type checkers
        self.assertEqual(pipeline._inserted_obj, inserted_obj)

        # transforms
        floor_parent_transform_mat = self.get_col_vec_transform_mat(
            pipeline._scene_structure.floor_parent
        )
        floor_transform_mat = self.get_col_vec_transform_mat(
            pipeline._scene_structure.floor_proxy_obj
        )
        dir_lights_root_transform_mat = self.get_col_vec_transform_mat(
            pipeline._scene_structure.dir_light_parent
        )
        inserted_obj_transform_mat = self.get_col_vec_transform_mat(
            pipeline._inserted_obj
        )
        assert pipeline._show_base.camera is not None
        camera_transform_mat = self.get_col_vec_transform_mat(
            pipeline._show_base.camera
        )

        # scene structure
        self.assertParentOf(
            pipeline._show_base.render,
            pipeline._scene_structure.dir_light_parent,
        )
        self.assertParentOf(
            pipeline._scene_structure.dir_light_parent,
            pipeline._scene_structure.dir_light,
        )
        self.assertParentOf(
            pipeline._scene_structure.floor_parent,
            pipeline._scene_structure.floor_proxy_obj,
        )
        self.assertParentOf(
            pipeline._scene_structure.floor_parent,
            pipeline._inserted_obj,
        )

        # transforms
        identity_transform = np.eye(4)
        self.assertAllclose(camera_transform_mat, identity_transform)

        self.assertLocalZMatches(
            floor_parent_transform_mat,
            general_vec3_2_panda3d_coord_sys(floor_y_in_view_space),
        )
        self.assertLocalYMatches(
            floor_parent_transform_mat,
            general_vec3_2_panda3d_coord_sys(floor_z_in_view_space),
        )
        self.assertPosMatches(
            floor_parent_transform_mat,
            general_vec3_2_panda3d_coord_sys(floor_pos_in_view_space),
        )

        self.assertLocalZMatches(
            dir_lights_root_transform_mat,
            general_vec3_2_panda3d_coord_sys(envmap_y_in_view_space),
        )
        self.assertLocalYMatches(
            dir_lights_root_transform_mat,
            general_vec3_2_panda3d_coord_sys(envmap_z_in_view_space),
        )
        self.assertPosMatches(
            dir_lights_root_transform_mat,
            general_vec3_2_panda3d_coord_sys(floor_pos_in_view_space),
        )

        expected_floor_transform_mat = np.array(
            [
                [self.floor_proxy_size.x, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, self.floor_proxy_size.y, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.assertAllclose(floor_transform_mat, expected_floor_transform_mat)

        self.assertAllclose(
            inserted_obj_transform_mat
            @ np.array(
                [
                    [1],
                    [2],
                    [3],
                    [1],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [1.9220],
                    [1.1428],
                    [3],
                    [1],
                ],
                dtype=np.float32,
            ),
            atol=1e-4,
        )

        # dir light position
        rel_light_pos = pipeline._scene_structure.dir_light.get_pos(
            pipeline._scene_structure.dir_light_parent
        )
        rel_obj_pos = pipeline._scene_structure.floor_proxy_obj.get_pos(
            pipeline._scene_structure.dir_light_parent
        )
        vec_from_light_to_obj = rel_obj_pos - rel_light_pos
        vec_from_light_to_obj = Vec3(
            vec_from_light_to_obj.x, vec_from_light_to_obj.y, vec_from_light_to_obj.z
        )
        vec_from_light_to_obj = vec_from_light_to_obj.normalize()
        normalized_expected_light_dir_p3d = general_vec3_2_panda3d_coord_sys(
            expected_light_dir.normalize()
        )
        light_dir_diff_cos = normalized_expected_light_dir_p3d.dot(
            vec_from_light_to_obj
        )
        self.assertAlmostEqual(light_dir_diff_cos, 1, delta=1e-4)

        # dir light intensity
        actual_light_color = pipeline._scene_structure.dir_light.node().get_color()
        self.assertAlmostEqual(actual_light_color.x, expected_light_color.x)
        self.assertAlmostEqual(actual_light_color.y, expected_light_color.y)
        self.assertAlmostEqual(actual_light_color.z, expected_light_color.z)

        # ibl
        self.assertIsNotNone(pipeline._show_base.pipeline.env_map)
        actual_ibl_transform_inv = self.panda3d_mat_2_mat(
            pipeline._show_base.pipeline.world_2_env_mat
        )
        expected_ibl_transform_inv = np.linalg.inv(
            dir_lights_root_transform_mat[:3, :3]
        ).T
        self.assertAllclose(
            actual_ibl_transform_inv[:3, :3], expected_ibl_transform_inv
        )

        # floor texcoord
        vtx_data = (
            pipeline._scene_structure.floor_proxy_obj.node()
            .modify_geom(0)
            .modify_vertex_data()
        )

    def test_get_mask_for_depth(self):
        near, far = self.near_far
        pipeline = RasterizingPipeline(
            camera=self.camera,
            floor_proxy_size=self.floor_proxy_size,
            im_size=self.im_size,
            near_far=self.near_far,
            shadow_map_size=self.shadow_map_size,
        )

        v_in = near + (far + near) / 2
        v_out = far + 5

        expected_mask = np.array(
            [
                [
                    [True, True, True, False],
                    [True, False, False, True],
                    [True, False, False, False],
                ]
            ],
            dtype=np.bool_,
        )
        depths = np.zeros_like(expected_mask, dtype=np.float32)

        depths[expected_mask] = v_in
        depths[~expected_mask] = v_out

        actual_mask = pipeline._get_mask_for_depth(depths)
        self.assertTrue(np.array_equal(expected_mask, actual_mask))

    def assertLocalYMatches(
        self, transform: np.ndarray, expected_local_y_dir: Vec3, atol: float = 1e-4
    ) -> None:
        self.assertAllclose(transform[3], np.array([0, 0, 0, 1], dtype=np.float32))

        expected_local_y_dir = expected_local_y_dir.normalize()

        transformed_origin = transform @ np.array(
            [
                [0],
                [0],
                [0],
                [1],
            ],
            dtype=np.float32,
        )
        transformed_origin = transformed_origin[:3]
        transformed_010 = transform @ np.array(
            [
                [0],
                [1],
                [0],
                [1],
            ],
            dtype=np.float32,
        )
        transformed_010 = transformed_010[:3]

        diff = transformed_010 - transformed_origin
        diff_dir_vec = Vec3.from_npy_col_vec(diff).normalize()

        self.assertVec3Allclose(diff_dir_vec, expected_local_y_dir, atol=atol)

    def assertLocalZMatches(
        self, transform: np.ndarray, expected_local_z_dir: Vec3, atol: float = 1e-4
    ) -> None:
        self.assertAllclose(transform[3], np.array([0, 0, 0, 1], dtype=np.float32))

        expected_local_z_dir = expected_local_z_dir.normalize()

        transformed_origin = transform @ np.array(
            [
                [0],
                [0],
                [0],
                [1],
            ],
            dtype=np.float32,
        )
        transformed_origin = transformed_origin[:3]
        transformed_001 = transform @ np.array(
            [
                [0],
                [0],
                [1],
                [1],
            ],
            dtype=np.float32,
        )
        transformed_001 = transformed_001[:3]

        diff = transformed_001 - transformed_origin
        diff_dir_vec = Vec3.from_npy_col_vec(diff).normalize()

        self.assertVec3Allclose(diff_dir_vec, expected_local_z_dir, atol=atol)

    def assertPosMatches(
        self, transform: np.ndarray, expected_pos: Vec3, atol: float = 1e-4
    ):
        transformed_origin = transform[:, 3]
        transformed_origin_vec = Vec3(
            float(transformed_origin[0]),
            float(transformed_origin[1]),
            float(transformed_origin[2]),
        )
        self.assertVec3Allclose(transformed_origin_vec, expected_pos, atol=atol)
        self.assertVec3Allclose(transformed_origin_vec, expected_pos, atol=atol)
