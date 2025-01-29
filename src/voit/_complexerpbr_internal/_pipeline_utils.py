import copy
from dataclasses import dataclass
from typing import Any, Final, Literal, Sequence, TypedDict

import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase


def render_scene_into_ext(
    base: ShowBase,
    cameras: "Sequence[p3d.NodePath[p3d.Camera]]",
    intermediate_out_specs: "Sequence[GraphicsOutputSpec]",
) -> "RenderSystem":
    assert base.cam is not None
    assert base.win is not None

    display_regions_to_modify: list[p3d.DisplayRegion] = []

    for cam in cameras:
        display_regions_to_modify.extend(cam.node().get_display_regions())

    min_out_sort = min(region.get_sort() for region in display_regions_to_modify)

    scenes: list[Scene] = [Scene.from_base(base, cam) for cam in cameras]
    intermediate_outs: list[GraphicsOutputWithTextures] = []

    for i in range(len(intermediate_out_specs)):
        if i < len(scenes):
            out_cam = scenes[i].cam
        else:
            postproc_scene = Scene.new_postproc()
            out_cam = postproc_scene.cam
            scenes.append(postproc_scene)
        im_out = GraphicsOutputWithTextures.from_spec(
            spec=intermediate_out_specs[i],
            render_buffer_sort=min_out_sort - len(intermediate_out_specs) + i,
            src_win=base.win,
            cam=out_cam,
        )
        intermediate_outs.append(im_out)

    scenes.append(Scene.new_postproc())

    for region in display_regions_to_modify:
        region.set_camera(scenes[-1].cam)

    return RenderSystem(intermediate_outputs=intermediate_outs, scenes=scenes)


class ReflectedCamera:
    def __init__(
        self,
        base: ShowBase,
        reflective_plane_hide_mask: p3d.DrawMask,
        reflective_plane_obj: p3d.NodePath,
    ):
        cam = base.cam
        assert cam is not None
        assert base.pipe is not None

        cam_lens = cam.node().get_lens()
        assert isinstance(cam_lens, p3d.PerspectiveLens) or isinstance(
            cam_lens, p3d.MatrixLens
        )

        new_lens = copy.deepcopy(cam_lens)
        new_lens.set_film_size(
            (cam_lens.get_film_size().x, -cam_lens.get_film_size().y)
        )

        self._base = base

        self.original_cam: Final = cam
        self.reflection_cam: Final = p3d.NodePath(p3d.Camera("reflective_cam"))
        self.reflection_cam.node().set_lens(new_lens)
        self.reflection_cam.reparent_to(base.render)
        self.reflection_cam.node().set_camera_mask(reflective_plane_hide_mask)

        self._reflective_plane_obj: Final = reflective_plane_obj
        self._reflective_plane_obj.hide(reflective_plane_hide_mask)

        self.reflected_plane_local_normal = p3d.LVector3f(0, 0, 1)

    @staticmethod
    def _reflect_point_to_plane(
        plane_n: p3d.LVector3f,
        plane_point: p3d.LVector3f,
        point_to_reflect: p3d.LVector3f,
    ) -> p3d.LVector3f:
        return (
            point_to_reflect
            - plane_n * plane_n.dot(point_to_reflect - plane_point) * 2.0
        )

    def update(self) -> None:
        plane_point = p3d.LVector3f(
            self._reflective_plane_obj.get_pos(self._base.render)
        )
        plane_n = self._get_reflective_plane_normal()
        self._update_impl(plane_n=plane_n, plane_point=plane_point)

    def _get_reflective_plane_normal(self) -> p3d.LVector3f:
        model_world_mat = self._reflective_plane_obj.get_mat(self._base.render)

        n1_in_world = model_world_mat.xform_point(self.reflected_plane_local_normal)
        n0_in_world = model_world_mat.xform_point(p3d.LVector3f(0, 0, 0))

        result = n1_in_world - n0_in_world
        result.normalize()

        return p3d.LVector3f(result)

    def _update_impl(self, plane_n: p3d.LVector3f, plane_point: p3d.LVector3f) -> None:
        cam_transform = self.original_cam.get_transform(self._base.render).get_mat()

        local_x1_in_render = cam_transform.xform_point((1, 0, 0))
        local_y1_in_render = cam_transform.xform_point((0, 1, 0))
        local_z1_in_render = cam_transform.xform_point((0, 0, 1))
        local_orig_in_render = cam_transform.xform_point((0, 0, 0))

        reflected_x1_in_render = ReflectedCamera._reflect_point_to_plane(
            plane_n=plane_n,
            plane_point=plane_point,
            point_to_reflect=ReflectedCamera.vbase_2_vec(local_x1_in_render),
        )
        reflected_y1_in_render = ReflectedCamera._reflect_point_to_plane(
            plane_n=plane_n,
            plane_point=plane_point,
            point_to_reflect=ReflectedCamera.vbase_2_vec(local_y1_in_render),
        )
        reflected_z1_in_render = ReflectedCamera._reflect_point_to_plane(
            plane_n=plane_n,
            plane_point=plane_point,
            point_to_reflect=ReflectedCamera.vbase_2_vec(local_z1_in_render),
        )
        reflected_orig_in_render = ReflectedCamera._reflect_point_to_plane(
            plane_n=plane_n,
            plane_point=plane_point,
            point_to_reflect=ReflectedCamera.vbase_2_vec(local_orig_in_render),
        )

        translate_mat = p3d.LMatrix4f(
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (
                reflected_orig_in_render.x,
                reflected_orig_in_render.y,
                reflected_orig_in_render.z,
                1,
            ),
        )

        local_x_axis_in_render = local_x1_in_render - local_orig_in_render
        local_y_axis_in_render = local_y1_in_render - local_orig_in_render
        local_z_axis_in_render = local_z1_in_render - local_orig_in_render
        reflected_x_axis_in_render = reflected_x1_in_render - reflected_orig_in_render
        reflected_y_axis_in_render = reflected_y1_in_render - reflected_orig_in_render
        reflected_z_axis_in_render = reflected_z1_in_render - reflected_orig_in_render

        reflected_x_axis_in_render.normalize()
        reflected_y_axis_in_render.normalize()
        reflected_z_axis_in_render.normalize()

        rotate_mat = p3d.LMatrix4f(
            (
                reflected_x_axis_in_render.x,
                reflected_x_axis_in_render.y,
                reflected_x_axis_in_render.z,
                0,
            ),
            (
                reflected_y_axis_in_render.x,
                reflected_y_axis_in_render.y,
                reflected_y_axis_in_render.z,
                0,
            ),
            (
                reflected_z_axis_in_render.x,
                reflected_z_axis_in_render.y,
                reflected_z_axis_in_render.z,
                0,
            ),
            (0, 0, 0, 1),
        )

        flip_z_mat = p3d.LMatrix4f(
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, -1, 0),
            (0, 0, 0, 1),
        )

        total_mat = p3d.LMatrix4f()
        total_mat.multiply(rotate_mat, translate_mat)

        self.reflection_cam.set_mat(total_mat)
        self._reflective_cam_transform = total_mat

    @staticmethod
    def vbase_2_vec(vbase: p3d.LVecBase3f) -> p3d.LVector3f:
        return p3d.LVector3f(vbase.x, vbase.y, vbase.z)

    def get_reflection_model_view_matrix(self) -> p3d.LMatrix4f:
        model_2_world = self._reflective_plane_obj.get_mat(self._base.render)
        cam_2_world = self.reflection_cam.get_mat(self._base.render)
        world_2_cam = p3d.LMatrix4f()
        world_2_cam.invert_from(cam_2_world)

        return model_2_world * world_2_cam

    @staticmethod
    def make_reflective_plane_obj() -> "p3d.NodePath[p3d.GeomNode]":
        cm = p3d.CardMaker("reflective_plane")
        cm.set_frame((-0.5, 0.5, 0.5, -0.5))
        return p3d.NodePath(cm.generate())  # type: ignore


@dataclass
class RenderSystem:
    intermediate_outputs: "list[GraphicsOutputWithTextures]"
    scenes: "list[Scene]"


class GraphicsOutputSpec(TypedDict):
    n_aux_outs: int
    depth_bits: int
    color_bits: int
    win_size: tuple[int, int]
    copyable_to_ram: set[
        Literal["depth", "color", "aux_out0", "aux_out1", "aux_out2", "aux_out3"]
    ]


@dataclass
class GraphicsOutputWithTextures:
    graphics_output: p3d.GraphicsOutput
    color_out_tex: p3d.Texture
    depth_tex: p3d.Texture
    aux_out_texs: list[p3d.Texture]
    relevant_region: p3d.DisplayRegion

    @staticmethod
    def from_spec(
        spec: "GraphicsOutputSpec",
        cam: "p3d.NodePath[p3d.Camera]",
        src_win: p3d.GraphicsOutput,
        render_buffer_sort: int,
    ):
        color_out_tex = p3d.Texture("color_out_tex")
        depth_out_tex = p3d.Texture("depth_out_tex")
        graphics_engine = src_win.get_gsg().get_engine()

        rtm_map: dict[bool, Any] = {
            True: p3d.GraphicsOutput.RTM_copy_ram,
            False: p3d.GraphicsOutput.RTM_bind_or_copy,
        }

        aux_out_tex_flags: list[Any] = [
            p3d.GraphicsOutput.RTP_aux_rgba_0,
            p3d.GraphicsOutput.RTP_aux_rgba_1,
            p3d.GraphicsOutput.RTP_aux_rgba_2,
            p3d.GraphicsOutput.RTP_aux_rgba_3,
        ][: spec["n_aux_outs"]]

        # create render buffer
        winprops = p3d.WindowProperties()
        winprops.setSize(spec["win_size"])
        props = p3d.FrameBufferProperties()
        props.set_depth_bits(spec["depth_bits"])
        props.set_stereo(False)
        props.set_aux_rgba(spec["n_aux_outs"])
        props.set_aux_float(0)
        props.set_aux_hrgba(0)
        props.set_rgba_bits(
            spec["color_bits"],
            spec["color_bits"],
            spec["color_bits"],
            spec["color_bits"],
        )
        props.set_force_hardware(True)
        render_buffer = graphics_engine.make_output(
            src_win.get_pipe(),
            "render_preproc_output",
            -1,
            props,
            winprops,
            p3d.GraphicsPipe.BF_refuse_window | p3d.GraphicsPipe.BF_resizeable,
            src_win.get_gsg(),
            src_win,
        )
        render_buffer.add_render_texture(
            depth_out_tex,
            rtm_map["depth" in spec["copyable_to_ram"]],
            p3d.GraphicsOutput.RTP_depth,
        )
        render_buffer.add_render_texture(
            color_out_tex,
            rtm_map["color" in spec["copyable_to_ram"]],
            p3d.GraphicsOutput.RTP_color,
        )

        aux_out_texs: list[p3d.Texture] = []
        for i_aux_tex, rtp_flag in enumerate(aux_out_tex_flags):
            out_tex = p3d.Texture(f"aux_out_tex_{i_aux_tex}")
            render_buffer.add_render_texture(
                out_tex,
                rtm_map[f"aux_out{i_aux_tex}" in spec["copyable_to_ram"]],
                rtp_flag,
            )
            aux_out_texs.append(out_tex)
        render_buffer.set_sort(render_buffer_sort)
        render_buffer.disable_clears()

        all_flags = [
            p3d.GraphicsOutput.RTP_color,
            p3d.GraphicsOutput.RTP_depth,
        ] + aux_out_tex_flags
        for flag in all_flags:
            render_buffer.set_clear_active(flag, True)

        new_region = render_buffer.make_display_region()
        new_region.set_camera(cam)

        return GraphicsOutputWithTextures(
            aux_out_texs=aux_out_texs,
            color_out_tex=color_out_tex,
            depth_tex=depth_out_tex,
            graphics_output=render_buffer,
            relevant_region=new_region,
        )


@dataclass
class Scene:
    cam: "p3d.NodePath[p3d.Camera]"
    render: p3d.NodePath

    @staticmethod
    def from_base(base: ShowBase, cam: "p3d.NodePath[p3d.Camera]") -> "Scene":
        assert base.cam is not None
        return Scene(cam=cam, render=base.render)

    @staticmethod
    def new_postproc() -> "Scene":
        # create the quad
        cm = p3d.CardMaker("filter-base-quad")
        cm.setFrameFullscreenQuad()
        quad = p3d.NodePath(cm.generate())
        quad.setDepthTest(False)
        quad.setDepthWrite(False)
        quad.setColor(1, 0.5, 0.5, 1)

        # create the camera and connect it to the quad
        quad_cam = p3d.Camera("filter-quad-cam")
        lens = p3d.OrthographicLens()
        lens.setFilmSize(2, 2)
        lens.setFilmOffset(0, 0)
        lens.setNearFar(-1000, 1000)
        quad_cam.setLens(lens)
        quadcam = quad.attachNewNode(quad_cam)

        return Scene(cam=quadcam, render=quad)


def new_1x1_indexed_plane_with_texcoords_and_up_normals() -> (
    "p3d.NodePath[p3d.GeomNode]"
):
    vertex_data = p3d.GeomVertexData(
        "reflective_plane_vtx_data",
        p3d.GeomVertexFormat.getV3n3t2(),
        p3d.GeomVertexData.UH_dynamic,
    )
    vertex_writer = p3d.GeomVertexWriter(vertex_data, p3d.InternalName.getVertex())
    normal_writer = p3d.GeomVertexWriter(vertex_data, p3d.InternalName.getNormal())
    texcoord_writer = p3d.GeomVertexWriter(vertex_data, p3d.InternalName.getTexcoord())

    vertices = [
        p3d.LVector3f(-0.5, 0.5, 0),
        p3d.LVector3f(0.5, 0.5, 0),
        p3d.LVector3f(0.5, -0.5, 0),
        p3d.LVector3f(-0.5, -0.5, 0),
    ]
    texcoords = [
        p3d.LVector2f(0, 1),
        p3d.LVector2f(1, 1),
        p3d.LVector2f(1, 0),
        p3d.LVector2f(0, 0),
    ]

    for vertex, texcoord in zip(vertices, texcoords):
        vertex_writer.add_data3(vertex)
        normal_writer.add_data3(p3d.LVector3f(0, 0, 1))
        texcoord_writer.add_data2(texcoord)

    prim = p3d.GeomTristrips(p3d.GeomPrimitive.UH_static)
    prim.add_vertices(0, 3, 1)
    prim.add_vertices(3, 1, 2)
    prim.close_primitive()

    geom = p3d.Geom(vertex_data)
    geom.add_primitive(prim)

    node = p3d.GeomNode("reflective_plane")
    node.add_geom(geom)

    obj = p3d.NodePath(node)
    return obj


@dataclass
class PostprocComponents:
    postquad: p3d.NodePath
    original_cam: p3d.NodePath
    color_out_tex: p3d.Texture
    depth_out_tex: p3d.Texture
    aux_out_texs: list[p3d.Texture]
    new_render_buffer: p3d.GraphicsOutput
    quad_cam: "p3d.NodePath[p3d.Camera]"
