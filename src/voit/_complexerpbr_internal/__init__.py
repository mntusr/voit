from __future__ import annotations

import builtins
import functools
import math
import os
from dataclasses import MISSING, Field, InitVar, dataclass, field
from typing import Final, TypedDict

import panda3d.core as p3d
from direct.filter.FilterManager import FilterManager
from direct.showbase.ShowBase import ShowBase
from direct.task.Task import TaskManager
from typing_extensions import Any, ClassVar, Literal, TypeAlias, TypeVar

from . import _shaderutils as shaderutils
from . import logging, utils
from ._pipeline_utils import (
    ReflectedCamera,
    new_1x1_indexed_plane_with_texcoords_and_up_normals,
    render_scene_into_ext,
)
from .envmap import EnvMap
from .envpool import EnvPool
from .version import __version__

try:
    from .textures import textures  # type: ignore
except ImportError:
    textures = None


__all__ = [
    "init",
    "Pipeline",
    "EnvMap",
    "EnvPool",
    "utils",
]

ShaderDefinesType: TypeAlias = "dict[str, Any]"


def _load_texture(texturepath: str) -> p3d.Texture:
    full_texture_path = p3d.Filename.from_os_specific(
        os.path.realpath(
            os.path.join(os.path.dirname(__file__), "textures", texturepath)
        )
    )
    texture = p3d.TexturePool.load_texture(full_texture_path)

    return texture


def _get_showbase_attr(attr: str) -> Any:
    showbase: ShowBase = builtins.base  # type: ignore[attr-defined]
    return getattr(showbase, attr)


def _get_default_330() -> bool:
    cvar = p3d.ConfigVariableInt("gl-version")
    gl_version = [cvar.get_word(i) for i in range(cvar.get_num_words())]
    if len(gl_version) >= 2 and gl_version[0] >= 3 and gl_version[1] >= 2:
        # Not exactly accurate, but setting this variable to '3 2' is common for disabling
        # the fixed-function pipeline and 3.2 support likely means 3.3 support as well.
        return True

    return False


TypeT = TypeVar("TypeT", bound=type)


def add_prc_fields(cls: TypeT) -> TypeT:
    prc_types = {
        "int": p3d.ConfigVariableInt,
        "bool": p3d.ConfigVariableBool,
        "float": p3d.ConfigVariableDouble,
        "str": p3d.ConfigVariableString,
    }

    def factoryfn(attrname: str, attrtype: str, default_value: Any) -> Any:
        name = f'simplepbr-{attrname.replace("_", "-")}'
        if isinstance(default_value, Field):
            if default_value.default_factory is not MISSING:
                default_value = default_value.default_factory()
            elif default_value.default is not MISSING:
                default_value = default_value.default
        return prc_types[attrtype](
            name=name,
            default_value=default_value,
        ).value

    def wrap(cls: type) -> type:
        annotations = cls.__dict__.get("__annotations__", {})
        for attrname, attrtype in annotations.items():
            if attrname.startswith("_"):
                # Private member, skip
                continue
            if attrtype == "ShowBase":
                continue

            default_value = getattr(cls, attrname)

            if attrtype.startswith("Literal") and isinstance(default_value, int):
                attrtype = "int"

            if attrtype not in prc_types:
                # Not a currently supported type, skip
                continue

            # pylint:disable-next=invalid-field-call
            setattr(
                cls,
                attrname,
                field(
                    default_factory=functools.partial(
                        factoryfn, attrname, attrtype, default_value
                    )
                ),
            )
        return cls

    return wrap(cls)  # type: ignore


@dataclass()
@add_prc_fields
class Pipeline:
    """
    The default properties of the reflective plane:

    * shape: rectangle
    * primitive: indexed
    * vertex columns: vertex, normal, texcoord
    * size: 2x2
    * normals = (0, 0, 1)

    Transfer functions of the textures:

    * environment map: linear
    * base color: linear
    * metallic-roughness: linear
    """

    # Class variables
    _EMPTY_ENV_MAP: ClassVar[EnvMap] = EnvMap.create_empty()
    _BRDF_LUT: ClassVar[p3d.Texture] = _load_texture("brdf_lut.txo")
    _PBR_VARS: ClassVar[list[str]] = [
        "enable_fog",
        "enable_hardware_skinning",
        "shadow_bias",
        "max_lights",
        "use_emission_maps",
        "use_normal_maps",
        "use_occlusion_maps",
        "calculate_normalmap_blue",
        "world_2_env_mat",
        "reflected_plane_local_normal",
        "show_reflective_plane",
        "reflective_plane_screen_texcoord",
    ]
    _POST_PROC_VARS: ClassVar[list[str]] = [
        "camera_node",
        "msaa_samples",
        "sdr_lut",
        "window",
        "world_2_env_mat",
        "show_envmap_as_skybox",
    ]
    base: ShowBase

    # Public instance variables
    show_envmap_as_skybox: bool = field(default=False)
    taskmgr: TaskManager = field(default_factory=lambda: _get_showbase_attr("task_mgr"))
    msaa_samples: Literal[0, 2, 4, 8, 16] = 4
    max_lights: int = 8
    use_normal_maps: bool = False
    use_emission_maps: bool = True
    use_occlusion_maps: bool = False
    exposure: float = 1.0
    enable_shadows: bool = True
    shadow_bias: float = 0.005
    enable_fog: bool = False
    use_hardware_skinning: InitVar[bool | None] = None
    enable_hardware_skinning: bool = True
    sdr_lut: p3d.Texture | None = None
    sdr_lut_factor: float = 1.0
    env_map: EnvMap | str | None = None
    calculate_normalmap_blue: bool = True
    world_2_env_mat: p3d.LMatrix4f = field(
        default_factory=lambda: p3d.LMatrix4f(
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    reflected_plane_local_normal: p3d.LVector3f = field(
        default_factory=lambda: p3d.LVector3f(0, 0, 1)
    )
    show_reflective_plane: bool = field(default=True)
    reflective_plane_screen_texcoord: bool = field(default=False)

    # Private instance variables
    _shader_ready: bool = False
    _render_node: p3d.NodePath[p3d.PandaNode] = field(init=False)
    _window: p3d.GraphicsOutput = field(init=False)
    _camera_node: p3d.NodePath[p3d.Camera] = field(init=False)
    _reflected_camera_mask: p3d.DrawMask = field(init=False)

    def __post_init__(
        self,
        use_hardware_skinning: bool | None,
    ) -> None:
        assert self.base.win is not None
        assert self.base.cam is not None

        self._render_node = self.base.render
        self._window = self.base.win
        self._camera_node = self.base.cam

        self._shader_ready = False

        # Do not force power-of-two textures
        p3d.Texture.set_textures_power_2(p3d.ATS_none)

        # Make sure we have AA for if/when MSAA is enabled
        self._render_node.set_antialias(p3d.AntialiasAttrib.M_off)

        # Add a default/fallback material
        fallback_material = p3d.Material("simplepbr-fallback")
        self._render_node.set_material(fallback_material)

        # create the basic structure of the rendering system
        self.reflective_plane = new_1x1_indexed_plane_with_texcoords_and_up_normals()
        self.reflective_plane_parent = self._render_node.attach_new_node(
            p3d.ModelNode("reflective_plane_parent")
        )
        self.reflective_plane_parent.set_mat(p3d.LMatrix4f.ident_mat())
        self.reflective_plane_parent.reparent_to(self._render_node)
        self.reflective_plane.reparent_to(self.reflective_plane_parent)

        self._reflected_camera_mask = p3d.DrawMask(2)
        self._reflected_camera = ReflectedCamera(
            base=self.base,
            reflective_plane_obj=self.reflective_plane,
            reflective_plane_hide_mask=self._reflected_camera_mask,
        )
        render_system = render_scene_into_ext(
            base=self.base,
            cameras=[
                self._reflected_camera.reflection_cam,
                self._reflected_camera.original_cam,
            ],
            intermediate_out_specs=[
                {
                    "depth_bits": 32,
                    "n_aux_outs": 2,
                    "win_size": (
                        self.base.win.get_x_size(),
                        self.base.win.get_y_size(),
                    ),
                    "color_bits": 16,
                    "copyable_to_ram": set(),
                },
                {
                    "depth_bits": 32,
                    "n_aux_outs": 2,
                    "win_size": (
                        self.base.win.get_x_size(),
                        self.base.win.get_y_size(),
                    ),
                    "color_bits": 16,
                    "copyable_to_ram": {"depth"},
                },
            ],
        )
        self._reflection_out = render_system.intermediate_outputs[0]
        self._regular_out = render_system.intermediate_outputs[1]
        self._postquad = render_system.scenes[2].render

        # PBR Shader
        if use_hardware_skinning is None:
            use_hardware_skinning = True
        self.enable_hardware_skinning = use_hardware_skinning
        self._recompile_pbr()

        # Tonemapping
        self._setup_tonemapping()

        # Do updates based on scene changes
        self.taskmgr.add(self._update, "simplepbr update", sort=49)

        self._shader_ready = True

        self._BRDF_LUT.wrap_u = p3d.SamplerState.WM_clamp
        self._BRDF_LUT.wrap_v = p3d.SamplerState.WM_clamp
        self._BRDF_LUT.minfilter = p3d.SamplerState.FT_linear
        self._BRDF_LUT.magfilter = p3d.SamplerState.FT_linear

        self.update_manual()

    def is_render_node_shader_successfully_compiled(self) -> bool:
        return not self._render_node.get_shader().get_error_flag()

    def is_postproc_shader_successfully_compiled(self) -> bool:
        return not self._postquad.get_shader().get_error_flag()

    @property
    def depth_texture(self) -> p3d.Texture:
        return self._regular_out.depth_tex

    def __setattr__(self, name: str, value: Any) -> None:
        prev_value = getattr(self, name, None)
        super().__setattr__(name, value)

        if not self._shader_ready:
            return

        elif name == "env_map":
            self._set_env_map_uniforms()
        elif name == "shadow_bias":
            self._render_node.set_shader_input("global_shadow_bias", self.shadow_bias)

        if name == "show_reflective_plane":
            if self.show_reflective_plane:
                self.reflective_plane.show()
                self.reflective_plane.hide(self._reflected_camera_mask)
            else:
                self.reflective_plane.hide()

        if name in self._PBR_VARS and prev_value != value:
            self._recompile_pbr()

        if name in self._POST_PROC_VARS and prev_value != value:
            self._setup_tonemapping()

    def _set_env_map_uniforms(self) -> None:
        env_map = self.env_map
        if env_map is None:
            env_map = self._EMPTY_ENV_MAP
        elif isinstance(env_map, str):
            env_map = EnvPool.ptr().load(env_map)
        self._render_node.set_shader_input("sh_coeffs", env_map.sh_coefficients)
        self._render_node.set_shader_input("brdf_lut", self._BRDF_LUT)
        filtered_env_map = env_map.filtered_env_map
        self._render_node.set_shader_input("filtered_env_map", filtered_env_map)
        self._render_node.set_shader_input(
            "max_reflection_lod", filtered_env_map.num_loadable_ram_mipmap_images
        )
        wolrld_2_env_mat_4x4 = p3d.LMatrix4f(self.world_2_env_mat)
        self._render_node.set_shader_input("CPUWorld2Envmap", wolrld_2_env_mat_4x4)
        self._render_node.set_shader_input("skybox_tex", env_map.cubemap)
        self._postquad.set_shader_input("skybox_tex", env_map.cubemap)

    def _recompile_pbr(self) -> None:
        pbr_defines = {
            "MAX_LIGHTS": self.max_lights,
            "USE_NORMAL_MAP": self.use_normal_maps,
            "USE_EMISSION_MAP": self.use_emission_maps,
            "ENABLE_SHADOWS": self.enable_shadows,
            "ENABLE_FOG": self.enable_fog,
            "USE_OCCLUSION_MAP": self.use_occlusion_maps,
            "ENABLE_SKINNING": self.enable_hardware_skinning,
            "CALC_NORMAL_Z": self.calculate_normalmap_blue,
        }

        pbrshader = shaderutils.make_shader(
            "pbr", "simplepbr.vert", "simplepbr.frag", pbr_defines
        )
        attr = p3d.ShaderAttrib.make(pbrshader)
        if self.enable_hardware_skinning:
            attr = attr.set_flag(p3d.ShaderAttrib.F_hardware_skinning, True)
        self._render_node.set_attrib(attr)
        self._render_node.set_shader_input("global_shadow_bias", self.shadow_bias)
        self.reflective_plane.set_shader_input("is_reflective_plane", True)
        self._render_node.set_shader_input("is_reflective_plane", False)
        self._render_node.set_shader_input(
            "reflection_color_and_alpha_tex", self._reflection_out.color_out_tex
        )
        self._render_node.set_shader_input(
            "reflected_plane_local_normal", self.reflected_plane_local_normal
        )
        self._render_node.set_shader_input(
            "show_reflective_plane", self.show_reflective_plane
        )
        self._render_node.set_shader_input(
            "reflective_plane_screen_texcoord", self.reflective_plane_screen_texcoord
        )
        self._set_env_map_uniforms()

    def _setup_tonemapping(self) -> None:
        defines = {
            "USE_SDR_LUT": bool(self.sdr_lut),
        }

        tonemap_shader = shaderutils.make_shader(
            "tonemap", "post.vert", "tonemap.frag", defines
        )
        self._postquad.set_shader(tonemap_shader)
        self._postquad.set_shader_input("tex", self._regular_out.color_out_tex)
        self._postquad.set_shader_input("exposure", self.exposure)
        if self.sdr_lut:
            self._postquad.set_shader_input("sdr_lut", self.sdr_lut)
            self._postquad.set_shader_input("sdr_lut_factor", self.sdr_lut_factor)

        world_2_env_mat = p3d.LMatrix4f(self.world_2_env_mat)
        self._postquad.set_shader_input(
            "orig_CPUWorld2Envmap",
            world_2_env_mat,
        )
        self._postquad.set_shader_input(
            "reflected_color_tex",
            self._regular_out.aux_out_texs[0],
        )
        self._postquad.set_shader_input(
            "reflection_weight_and_roughness_tex", self._regular_out.aux_out_texs[1]
        )
        self._postquad.set_shader_input(
            "show_envmap_as_skybox", self.show_envmap_as_skybox
        )

    def get_all_casters(self) -> list[p3d.LightLensNode]:
        engine = p3d.GraphicsEngine.get_global_ptr()
        cameras = [
            dispregion.camera
            for win in engine.windows
            for dispregion in win.active_display_regions
        ]

        def is_caster(node: p3d.NodePath[p3d.PandaNode]) -> bool:
            if node.is_empty():
                return False

            pandanode = node.node()
            return (
                hasattr(pandanode, "is_shadow_caster") and pandanode.is_shadow_caster()  # type: ignore
            )

        return [i.node() for i in cameras if is_caster(i)]  # type: ignore

    def _create_shadow_shader_attrib(self) -> p3d.ShaderAttrib:
        defines = {
            "ENABLE_SKINNING": self.enable_hardware_skinning,
        }
        shader = shaderutils.make_shader(
            "shadow", "shadow.vert", "shadow.frag", defines
        )
        attr = p3d.ShaderAttrib.make(shader)
        if self.enable_hardware_skinning:
            attr = attr.set_flag(p3d.ShaderAttrib.F_hardware_skinning, True)
        return attr

    def _update(self, task: p3d.PythonTask) -> int:
        self.update_manual()
        return task.DS_cont

    def update_manual(self):
        assert self.base.cam is not None
        assert self.base.win is not None
        recompile = False
        # Use a simpler, faster shader for shadows
        for caster in self.get_all_casters():
            if isinstance(caster, p3d.PointLight):
                logging.warning(
                    f"PointLight shadow casters are not supported, disabling {caster.name}"
                )
                caster.set_shadow_caster(False)
                recompile = True
                continue
            state = caster.get_initial_state()
            if not state.has_attrib(p3d.ShaderAttrib):
                attr = self._create_shadow_shader_attrib()
                state = state.add_attrib(attr, 1)
                state = state.remove_attrib(p3d.CullFaceAttrib)
                caster.set_initial_state(state)

        if recompile:
            self._recompile_pbr()

        self._render_node.set_shader_input(
            "camera_world_position", self._camera_node.get_pos(self._render_node)
        )

        # update the camera
        self._reflected_camera.update()

        # update the post-processing shader
        proj_mat_inv_in = self.base.cam.node().get_lens().get_projection_mat_inv()
        proj_mat_in = self.base.cam.node().get_lens().get_projection_mat()
        cam_coord_sys_transform_mat = self._get_view_2_world_cs_matrix(
            render=self.base.render, cam=self.base.cam
        )
        self._postquad.set_shader_input(
            "orig_CPUProjectionMatrixInverse",
            proj_mat_inv_in,
        )
        self._postquad.set_shader_input(
            "orig_CPUProjectionMatrix",
            proj_mat_in,
        )
        self._postquad.set_shader_input(
            "orig_CPUView2WorldCSMatrix", cam_coord_sys_transform_mat
        )

        # update the reflective plane preferences
        reflection_model_view_mat = (
            self._reflected_camera.get_reflection_model_view_matrix()
        )
        self._render_node.set_shader_input(
            "CPUView2WorldCSMatrix", cam_coord_sys_transform_mat
        )
        self._render_node.set_shader_input(
            "reflection_model_view_mat", reflection_model_view_mat
        )
        self._render_node.set_shader_input("texture_size", self.base.win.get_size())
        self.reflective_plane.set_shader_input("texture_size", self.base.win.get_size())
        self._render_node.set_shader_input("from_reflective_cam", True)
        self._postquad.set_shader_input("texture_size", self.base.win.get_size())

        reflective_plane_mat = self.reflective_plane.get_mat(self._render_node)
        reflective_plane_mat_inv = p3d.LMatrix4f(reflective_plane_mat)
        reflective_plane_mat_inv.invert_in_place()
        self._render_node.set_shader_input(
            "reflective_plane_world_model_mat", reflective_plane_mat_inv
        )

    @staticmethod
    def _get_view_2_world_cs_matrix(
        render: p3d.NodePath, cam: p3d.NodePath
    ) -> p3d.LMatrix4f:
        cam_mat = cam.get_mat(render)

        transform_mat = p3d.LMatrix4f(cam_mat)
        transform_mat.invert_in_place()
        transform_mat.transpose_in_place()

        return transform_mat

    def verify_shaders(self) -> None:
        gsg = self._window.gsg

        def check_shader(shader: p3d.Shader) -> None:
            shader = p3d.Shader(shader)
            shader.prepare_now(gsg.prepared_objects, gsg)
            assert shader.is_prepared(gsg.prepared_objects)
            assert not shader.get_error_flag()

        check_shader(self._render_node.get_shader())
        check_shader(self._postquad.get_shader())

        attr = self._create_shadow_shader_attrib()
        check_shader(attr.get_shader())


init = Pipeline  # pylint: disable=invalid-name
