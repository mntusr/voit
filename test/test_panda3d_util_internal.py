import math
from typing import Union, cast

import cv2 as cv
import depth_tools
import numpy as np
from direct.showbase.ShowBase import CardMaker, NodePath, ShowBase
from matplotlib import pyplot as plt
from panda3d.core import (
    DirectionalLight,
    GeomNode,
    Lens,
    LMatrix3f,
    LMatrix4f,
    LVector4f,
    ModelNode,
    Shader,
    Texture,
    TextureAttrib,
)
from voit import Vec2, Vec2i, Vec3, VoitError
from voit._math_internal import zbuf_2_depth
from voit._panda3d_util_internal import (
    LoaderOptions,
    RendererShowBase,
    VtxColName,
    add_shadow_casting_dir_light,
    camera_2_panda3d_proj_mat,
    get_col_in_vertex_data,
    load_model_from_local_file,
    new_lens_from_proj_mat,
    np_mat3x3_2_panda_mat,
    np_mat4x4_2_panda_mat,
    read_2d_rgba_texture,
    update_shadow_casting_dir_light,
    write_2d_rgba_texture,
)

from .testutil import VoitTestBase


class TestFunctions(VoitTestBase):
    def setUp(self) -> None:
        self.models_dir = self.get_test_data_dir("panda3d_util")
        rsb = RendererShowBase(
            proj_mat=np.eye(4),
            im_size=Vec2i(500, 250),
            offscreen=True,
            show_envmap_as_skybox=False,
            reflective_plane_screen_texcoord=False,
        )
        try:
            lit_object = load_model_from_local_file(
                rsb, self.models_dir / "pink_cube.glb"
            )
            self.test_obj = load_model_from_local_file(
                rsb, self.models_dir / "pink_cube.glb"
            )
        finally:
            rsb.destroy()

        self.light_upd_ctx = DirLightUpdContext(lit_object)
        self.test_data_dir = self.get_test_data_dir("panda3d_util")

    def test_new_lens_from_proj_mat(self):
        proj_mat = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
            dtype=np.float32,
        )
        lens = new_lens_from_proj_mat(proj_mat=proj_mat)

        self.assertAlmostEqual(lens.film_size.x, 2)
        self.assertAlmostEqual(lens.film_size.y, 2)
        self.assertAlmostEqual(lens.user_mat.get_cell(2, 3), 14)

    def test_np_mat3x3_2_panda_mat__happy_path(self):
        np_mat = np.arange(0, 9).reshape((3, 3))
        panda_mat = np_mat3x3_2_panda_mat(np_mat=np_mat)

        self.assertNpMatAlmostEqualToPandaMat(np_mat, panda_mat, atol=1e-4)

    def test_np_mat3x3_2_panda_mat__invalid_shape(self):
        with self.assertRaises(VoitError):
            np_mat3x3_2_panda_mat(np_mat=np.ones((4, 4)))

    def test_np_mat4x4_2_panda_mat__happy_path(self):
        np_mat = np.arange(0, 16).reshape((4, 4))
        panda_mat = np_mat4x4_2_panda_mat(np_mat=np_mat)

        self.assertNpMatAlmostEqualToPandaMat(np_mat, panda_mat, atol=1e-4)

    def test_np_mat4x4_2_panda_mat__invalid_shape(self):
        with self.assertRaises(VoitError):
            np_mat4x4_2_panda_mat(np_mat=np.ones((3, 3)))

    def test_write_2d_rgba_texture__happy_path(self):
        image = np.zeros((3, 16, 23), dtype=np.float32)
        image[0, -1, -1] = 0.3
        image[1, 0, -1] = 1
        image[2, 0, 0] = 7
        image[0, -1, 0] = 0.7
        image[1, -1, 0] = 12
        texture = Texture()
        texture.setup_2d_texture(23, 16, Texture.T_unsigned_byte, Texture.F_rgba8)
        write_2d_rgba_texture(im=image, texture=texture)

        peeker = texture.peek()

        fetched_pixels: list[Vec3] = []
        expected_colors: list[Vec3] = []
        locations: list[tuple[int, int]] = [(0, 0), (22, 0), (22, 15), (0, 15)]
        for location in locations:
            fetch_out = LVector4f()
            peeker.fetch_pixel(fetch_out, location[0], location[1], 0)

            self.assertEqual(fetch_out.get_w(), 1.0)
            fetched_pixels.append(Vec3(fetch_out.x, fetch_out.y, fetch_out.z))
            # see https://github.com/panda3d/panda3d/issues/789
            expected_colors.append(
                Vec3(
                    min(max(image[0, 15 - location[1], location[0]], 0), 1),
                    min(max(image[1, 15 - location[1], location[0]], 0), 1),
                    min(max(image[2, 15 - location[1], location[0]], 0), 1),
                )
            )

        for fetched_pixel, expected_color in zip(fetched_pixels, expected_colors):
            self.assertVec3Allclose(fetched_pixel, expected_color, atol=1e-2)

    def test_write_2d_rgba_texture__invalid_shape(self):
        texture = Texture()
        texture.setup_2d_texture(23, 16, Texture.T_unsigned_byte, Texture.F_rgba8)

        cases = [
            ("too_many_dims", (3, 16, 23, 1)),
            ("incorrect_width", (3, 16, 1)),
        ]

        for case_name, shape in cases:
            with self.subTest(case_name):
                with self.assertRaises(VoitError):
                    im = np.ones(shape, dtype=np.float32)
                    write_2d_rgba_texture(im=im, texture=texture)

    def test_write_2d_rgba_texture__invalid_dtype(self):
        texture = Texture()
        texture.setup_2d_texture(23, 16, Texture.T_unsigned_byte, Texture.F_rgba8)

        with self.assertRaises(VoitError):
            im = np.ones((3, 16, 23), dtype=np.uint8)
            write_2d_rgba_texture(im=im, texture=texture)

    def test_intrinsic_mat_2_panda3d_proj_mat__happy_path(self):
        camera = depth_tools.CameraIntrinsics(f_x=500, f_y=500, c_x=400, c_y=300)
        im_size = Vec2i(800, 600)
        near = 0.3
        far = 705
        proj_mat, a_b = camera_2_panda3d_proj_mat(
            camera=camera, near_far=(near, far), im_size=im_size
        )

        initial_point_general = Vec3(3.9, 7.5, far)

        # project the point using the intrinsics matrix provided by the camera
        general_projected_point = camera.get_intrinsic_mat() @ np.array(
            [
                [initial_point_general.x],
                [initial_point_general.y],
                [initial_point_general.z],
            ],
            dtype=np.float32,
        )
        general_projected_point = (
            general_projected_point / general_projected_point[-1, 0]
        )[[0, 1]]

        # project the point using the calculated projection matrix
        panda3d_projected_point = proj_mat @ np.array(
            [
                [initial_point_general.x],
                [initial_point_general.z],  # ] transform the coordinate system
                [initial_point_general.y],  # ]
                [1],
            ],
            dtype=np.float32,
        )
        panda3d_projected_point = (
            panda3d_projected_point / panda3d_projected_point[-1, 0]
        )[[0, 1]]
        panda3d_projected_point[0] = (panda3d_projected_point[0] + 1) / 2 * im_size.x
        panda3d_projected_point[1] = (panda3d_projected_point[1] + 1) / 2 * im_size.y

        # compare the two results
        self.assertAllclose(panda3d_projected_point, general_projected_point)

        near_zbuf = (a_b[0] * near + a_b[1]) / near
        self.assertAlmostEqual(near_zbuf, -1, delta=1e-4)

    def assertColumnsAre(
        self, obj_node: GeomNode, expected_columns: dict[VtxColName, np.ndarray]
    ):
        vtx_data = obj_node.get_geom(0).get_vertex_data()

        actual_columns: dict[VtxColName, np.ndarray] = dict()
        for col_name in expected_columns.keys():
            actual_columns[col_name] = get_col_in_vertex_data(
                col_name=col_name,
                vertex_data=vtx_data,
            )

        for col_name in expected_columns.keys():
            expected_column = expected_columns[col_name]
            actual_column = actual_columns[col_name]

            self.assertAllclose(expected_column, actual_column)

    def test_add_shadow_casting_dir_light(self):
        lighted_objects_root = NodePath("lighted_objects_root")
        light_parent = NodePath("light_parent")
        expected_light_name = "light65"

        dir_light = add_shadow_casting_dir_light(
            lighted_objects_root=lighted_objects_root,
            name=expected_light_name,
            parent=light_parent,
            shadow_map_size=567,
        )
        self.assertParentOf(light_parent, dir_light)
        self.assertEqual(dir_light.name, expected_light_name)
        self.assertTrue(dir_light.node().is_shadow_caster())
        self.assertTrue(lighted_objects_root.has_light(dir_light))

    def test_update_shadow_casting_dir_light__happy_path(self):
        direction = Vec3(5, 2, -3)
        self.light_upd_ctx.lights_root.set_pos((-3, 2, 1))
        self.light_upd_ctx.lights_root.look_at((15, 9, 2))
        self.light_upd_ctx.lit_objects_root.set_pos((2, 3, 5))

        with self.assertNoLogs(self.voit_logger):
            update_shadow_casting_dir_light(
                dir_light=self.light_upd_ctx.dir_light,
                direction=direction,
                related_objects_root=self.light_upd_ctx.lit_objects_root,
            )

        # check light direction
        light_pos_in_parent_coord_sys = self.get_pos_in_the_coordinate_system(
            point=Vec3(0, 0, 0),
            src_coord_sys=self.light_upd_ctx.dir_light,
            target_coord_sys=self.light_upd_ctx.lights_root,
        )
        y1_pos_in_parent_coord_sys = self.get_pos_in_the_coordinate_system(
            point=Vec3(0, 1, 0),
            src_coord_sys=self.light_upd_ctx.dir_light,
            target_coord_sys=self.light_upd_ctx.lights_root,
        )
        actual_normalized_light_dir_in_parent_coord_sys = (
            y1_pos_in_parent_coord_sys.vec_minus(
                light_pos_in_parent_coord_sys
            ).normalize()
        )
        expected_normalized_light_dir_in_parent_coord_sys = direction.normalize()
        self.assertVec3Allclose(
            actual_normalized_light_dir_in_parent_coord_sys,
            expected_normalized_light_dir_in_parent_coord_sys,
        )

        # check light position
        lit_obj_pos_in_parent_coord_sys = self.get_pos_in_the_coordinate_system(
            point=Vec3(0, 0, 0),
            src_coord_sys=self.light_upd_ctx.lit_object,
            target_coord_sys=self.light_upd_ctx.lights_root,
        )
        vector_pointing_to_the_lit_obj_in_light_parent_coord_sys = (
            lit_obj_pos_in_parent_coord_sys.vec_minus(light_pos_in_parent_coord_sys)
        )
        normalized_vector_pointing_to_the_lit_obj_in_light_parent_coord_sys = (
            vector_pointing_to_the_lit_obj_in_light_parent_coord_sys.normalize()
        )
        light_dir_cos = (
            normalized_vector_pointing_to_the_lit_obj_in_light_parent_coord_sys.dot(
                actual_normalized_light_dir_in_parent_coord_sys
            )
        )
        self.assertAlmostEqual(light_dir_cos, 1.0, delta=1e-4)
        light_distance_from_lit_obj = (
            vector_pointing_to_the_lit_obj_in_light_parent_coord_sys.vec_len()
        )
        self.assertGreater(light_distance_from_lit_obj, math.sqrt(12) - 0.001)

        # check lens preferences
        tight_bounds = self.light_upd_ctx.lit_objects_root.get_tight_bounds(
            self.light_upd_ctx.dir_light
        )
        assert tight_bounds is not None
        bmin, bmax = tight_bounds
        light_lens = self.light_upd_ctx.dir_light.node().get_lens()
        light_near = light_lens.get_near()
        light_far = light_lens.get_far()
        self.assertAlmostEqual(bmin.y, light_near)
        self.assertAlmostEqual(bmax.y, light_far)
        self.assertAlmostEqual(light_lens.film_size.x, bmax.x - bmin.x)
        self.assertAlmostEqual(light_lens.film_size.y, bmax.z - bmin.z)
        self.assertAlmostEqual(light_lens.film_offset.x, (bmin.x + bmax.x) / 2)
        self.assertAlmostEqual(light_lens.film_offset.y, (bmin.z + bmax.z) / 2)

    def test_update_shadow_casting_dir_light__no_light_parent(self):
        dir_light = self.light_upd_ctx.dir_light
        self.light_upd_ctx.dir_light.detach_node()

        with self.assertRaises(VoitError):
            update_shadow_casting_dir_light(
                dir_light=self.light_upd_ctx.dir_light,
                direction=Vec3(4, 0, 2),
                related_objects_root=self.light_upd_ctx.lit_objects_root,
            )

    def test_update_shadow_casting_dir_light__zero_len_dir(self):
        with self.assertLogs(self.voit_logger, level="WARNING") as cm:
            update_shadow_casting_dir_light(
                dir_light=self.light_upd_ctx.dir_light,
                direction=Vec3(0, 0, 0),
                related_objects_root=self.light_upd_ctx.lit_objects_root,
            )
        msg = cm.output[0]
        self.assertIn("zero vector", msg)
        self.assertIn(str(Vec3(0, 0, 0)), msg)
        self.assertIn("1e-30", msg)
        self.assertIn(str(Vec3(1, 1, 1)), msg)

    def test_update_shadow_casting_dir_light__empty_lit(self):
        self.light_upd_ctx.lit_object.remove_node()
        with self.assertLogs(self.voit_logger, level="WARNING") as cm:
            update_shadow_casting_dir_light(
                dir_light=self.light_upd_ctx.dir_light,
                direction=Vec3(5, 2, 1),
                related_objects_root=self.light_upd_ctx.lit_objects_root,
            )
        msg = cm.output[0]
        self.assertIn("any vertex", msg)
        self.assertIn(str(self.light_upd_ctx.lit_objects_root), msg)

    def get_pos_in_the_coordinate_system(
        self, point: Vec3, src_coord_sys: NodePath, target_coord_sys: NodePath
    ) -> Vec3:
        probe = NodePath(ModelNode("probe"))
        probe.reparent_to(src_coord_sys)
        probe.set_pos((point.x, point.y, point.z))
        got_pos = probe.get_pos(target_coord_sys)
        probe.detach_node()
        return Vec3(got_pos.x, got_pos.y, got_pos.z)

    def test_read_2d_rgba_texture(self):
        tex = Texture()
        im_path = self.test_data_dir / "im.jpg"
        tex.read(
            im_path,
            options=LoaderOptions.LF_no_cache | LoaderOptions.LF_report_errors,  # type: ignore
            alpha_fullpath=im_path,
            alpha_file_channel=0,
            primary_file_num_channels=0,
        )
        tex.set_format(Texture.F_rgba8)
        actual_im = read_2d_rgba_texture(tex)
        expected_im = cv.imread(str(im_path))
        expected_im = (expected_im.astype(np.float32) / 255.0).transpose([2, 0, 1])[
            ::-1, :, :
        ]
        self.assertAllclose(actual_im, expected_im, atol=0.1)

    def _get_v_texcoord_shader(self) -> Shader:
        vs = """#version 300 es
        precision highp float;

        uniform mat4 p3d_ModelViewProjectionMatrix;

        in vec4 p3d_Vertex;
        in vec2 p3d_MultiTexCoord0;

        out vec2 v_texcoord;

        void main() {
            v_texcoord = p3d_MultiTexCoord0;
            gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
        }
        """
        fs = """#version 300 es
        precision highp float;

        in vec2 v_texcoord;

        out vec4 out_color;

        void main(){
            out_color = vec4(v_texcoord, 0, 1);
        }
        """
        total = Shader.make(Shader.SL_GLSL, vs, fs)
        return total

    def assertNpMatAlmostEqualToPandaMat(
        self, np_mat: np.ndarray, panda_mat: Union[LMatrix3f, LMatrix4f], atol: float
    ):
        if isinstance(panda_mat, LMatrix3f):
            n_panda_mat_cols = 3
            n_panda_mat_rows = 3
        elif isinstance(panda_mat, LMatrix4f):
            n_panda_mat_cols = 4
            n_panda_mat_rows = 4

        self.assertEqual(np_mat.shape[0], n_panda_mat_rows)
        self.assertEqual(np_mat.shape[1], n_panda_mat_cols)

        for row_idx in range(n_panda_mat_rows):
            for col_idx in range(n_panda_mat_cols):
                np_val = np_mat[row_idx, col_idx]
                panda_val = panda_mat.get_cell(row_idx, col_idx)
                self.assertAlmostEqual(np_val, panda_val, delta=atol)


class DirLightUpdContext:
    def __init__(self, lit_object: NodePath):
        self.common_root = NodePath("common_root")
        self.lit_objects_root = NodePath("lit_objects_root")
        self.lights_root = NodePath("lights_root")
        self.dir_light = NodePath(DirectionalLight("dir_light"))
        self.lit_object = lit_object

        self.lit_objects_root.reparent_to(self.common_root)
        self.lights_root.reparent_to(self.common_root)
        self.dir_light.reparent_to(self.lights_root)
        self.lit_object = lit_object
        self.lit_object.reparent_to(self.lit_objects_root)


class TestRendererShowBase(VoitTestBase):
    def setUp(self) -> None:
        self.models_dir = self.get_test_data_dir("panda3d_util")

    def tearDown(self) -> None:
        self.destroy_showbase()

    def test_render(self):
        im_size = Vec2i(500, 250)

        camera = depth_tools.CameraIntrinsics(f_x=500, f_y=500, c_x=305, c_y=90)
        far = 1000
        near = 0.1
        proj_mat, a_b = camera_2_panda3d_proj_mat(
            camera=camera, near_far=(near, far), im_size=im_size
        )

        rsb = RendererShowBase(
            proj_mat=proj_mat,
            im_size=im_size,
            offscreen=True,
            show_envmap_as_skybox=False,
            reflective_plane_screen_texcoord=False,
        )
        rsb.set_reflective_plane_visibility(False)

        self.assertTrue(rsb.pipeline.is_render_node_shader_successfully_compiled())
        self.assertTrue(rsb.pipeline.is_postproc_shader_successfully_compiled())

        pink_cube = load_model_from_local_file(rsb, self.models_dir / "pink_cube.glb")

        cube_center_distance_from_camera = 7
        expected_cube_depth = cube_center_distance_from_camera - 1

        cam = rsb.cam
        assert cam is not None
        cam.set_pos((0, -cube_center_distance_from_camera, 0))
        cam.look_at((0, 0, 0))
        pink_cube.reparent_to(rsb.render)
        pink_cube.set_pos((0, 0, 0))
        dir_light_node = DirectionalLight("dir_light")
        dir_light_node.set_color((0.5, 0.5, 0.5, 1.0))
        dir_light = NodePath(dir_light_node)
        dir_light.reparent_to(rsb.render)
        dir_light.set_pos((1, -7, 5))
        dir_light.look_at((0, 0, 0))
        rsb.render.set_light(dir_light)

        captured_rgb, captured_zbuf = rsb.render_single_RGBB_frame()

        captured_depth = zbuf_2_depth(a_b=a_b, zbuf_data=captured_zbuf)

        # plt.imshow(captured_rgb.transpose([1, 2, 0]))
        # plt.show(block=True)
        # plt.imshow(captured_depth[0])
        # plt.show(block=True)

        # check depth correctness
        is_pixel_in_cube = (captured_depth < (expected_cube_depth + 0.5)) & (
            captured_depth > (expected_cube_depth - 0.5)
        )
        self.assertGreater(int(np.sum(is_pixel_in_cube)), 20)
        self.assertAllAlmostEqual(captured_depth[is_pixel_in_cube], expected_cube_depth)
        self.assertTrue(np.all(captured_depth[~is_pixel_in_cube] > 999))

        # check color
        mean_colors = self.get_obj_mean_color(
            rgb_im=captured_rgb, pixel_mask=is_pixel_in_cube
        )

        self.assertGreaterWithMargin(mean_colors.x, mean_colors.z, 0.03)
        self.assertGreaterWithMargin(mean_colors.z, mean_colors.y, 0.03)

        # check object center
        obj_center = self.get_obj_center_in_px_xy(pixel_mask=is_pixel_in_cube)
        self.assertVec2Allclose(obj_center, Vec2(camera.c_x, camera.c_y), atol=1)

    def get_obj_mean_color(self, rgb_im: np.ndarray, pixel_mask: np.ndarray) -> Vec3:
        cube_area_r = rgb_im[[0]][pixel_mask]
        cube_area_g = rgb_im[[1]][pixel_mask]
        cube_area_b = rgb_im[[2]][pixel_mask]

        mean_r = float(np.mean(cube_area_r))
        mean_g = float(np.mean(cube_area_g))
        mean_b = float(np.mean(cube_area_b))

        return Vec3(mean_r, mean_g, mean_b)

    def get_obj_center_in_px_xy(self, pixel_mask: np.ndarray) -> Vec2:
        width = pixel_mask.shape[2]
        height = pixel_mask.shape[1]

        x_steps = np.arange(width)
        y_steps = np.arange(height - 1, -1, -1)

        x, y = np.meshgrid(x_steps, y_steps)

        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        x = x[pixel_mask]
        y = y[pixel_mask]

        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        return Vec2(x_center, y_center)

    def assertGreaterWithMargin(self, a: float, b: float, margin: float):
        self.assertTrue(
            a - b > margin,
            f"The value a ({a}) is not greater than b ({b}) with margin {margin}.",
        )
