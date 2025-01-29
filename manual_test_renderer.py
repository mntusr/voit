import argparse
import math
import sys
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import cv2 as cv
import depth_tools
import matplotlib.pyplot as plt
import numpy as np
from direct.showbase.ShowBase import NodePath
from panda3d.core import DirectionalLight, LoaderOptions, PythonTask, Texture
from voit import Vec2i, Vec3
from voit._panda3d_util_internal import (
    RendererShowBase,
    VoitEnvmap,
    VtxColName,
    add_shadow_casting_dir_light,
    camera_2_panda3d_proj_mat,
    convert_path_to_unix_style,
    load_model_from_local_file,
    set_col_in_vertex_data,
    set_simple_material,
    update_shadow_casting_dir_light,
    write_2d_rgba_texture,
)

FloorMaterial = Literal["metal-checked", "metal-smooth", "diel-checked"]
FloorTextureValues: Final = ("metal-checked", "metal-smooth", "diel-checked")

RenderMode = Literal["offscreen", "move"]
RenderModeValues: Final = ("offscreen", "move")


def main(args: "ParsedArgs") -> None:
    print("This test uses the coordinate system of Panda3D (Z-up right handed).")
    im_size = Vec2i(800, 600)
    c = 500

    camera = depth_tools.CameraIntrinsics(f_x=c, f_y=c, c_x=400, c_y=300)
    proj_mat, _ = camera_2_panda3d_proj_mat(
        camera=camera, near_far=(0.01, 1000), im_size=Vec2i(800, 600)
    )
    envmap_im = load_image(args.envmap)
    envmap = VoitEnvmap.from_image(envmap=envmap_im, envmap_linear=False)
    ibl_transform = get_envmap_transform_mat()

    report_reflective_plane_texcoord(args.rpst)
    rsb = RendererShowBase(
        proj_mat=proj_mat,
        im_size=im_size,
        offscreen=args.render == "offscreen",
        show_envmap_as_skybox=True,
        reflective_plane_screen_texcoord=args.rpst,
    )
    rsb.set_envmap(envmap)
    rsb.set_envmap_transform_inv(ibl_transform.T)
    scene = load_model_from_local_file(rsb, args.model)
    scene.set_pos((0, 0, 0))
    scene.reparent_to(rsb.render)
    load_material_on_reflective_plane(
        material=args.floor, obj=rsb.pipeline.reflective_plane
    )

    rsb.pipeline.reflective_plane.set_scale((15, 15, 1))
    cam = rsb.cam
    assert cam is not None
    set_initial_cam_pos_rot(cam, args.render)

    dir_light = add_shadow_casting_dir_light(
        lighted_objects_root=rsb.render,
        parent=rsb.render,
        shadow_map_size=1024,
        name="dir_light",
    )
    dir_light.node().set_color((0.3 * 10, 0.3 * 10, 0.3 * 10, 1.0))
    update_shadow_casting_dir_light(
        dir_light=dir_light,
        related_objects_root=rsb.render,
        direction=Vec3(-9.6151, -5.15416, -7.94988),
    )

    if args.render == "offscreen":
        rendered_rgb, _ = rsb.render_single_RGBB_frame()
        rendered_rgb = rendered_rgb ** (1 / 2.2)

        rendered_rgb_hwc = rendered_rgb.transpose([1, 2, 0])

        cv.imwrite("actual_render_result.png", rendered_rgb_hwc[:, :, ::-1])
        plt.imshow(rendered_rgb_hwc)
        plt.title("Rendered image")
        plt.show(block=True)
    else:

        def cam_upd(task: PythonTask):
            cam_pos = cam.get_pos()
            cam.set_pos((cam_pos.x, cam_pos.y, cam_pos.z + 0.1))
            cam.look_at((0, 0, 0))
            return task.DS_again

        rsb.task_mgr.do_method_later(0.1, cam_upd, "camera_update")
        rsb.run()


def report_reflective_plane_texcoord(rpst: bool) -> None:
    if rpst:
        print(
            "Using the on-screen position as texture coordinate on the reflective plane."
        )
    else:
        print(
            "Using the texture coordinates of the vertces as texture coordinate on the reflective plane."
        )


def load_image(path: Path) -> np.ndarray:
    im = cv.imread(str(path))
    im = im.transpose([2, 0, 1])
    im = im[::-1]
    im = im.astype(np.float32) / 255
    return im


manual_test_data_dir = Path(__file__).resolve().parent / "manual_test_data"


def get_envmap_transform_mat() -> np.ndarray:
    print(
        "Environment map transform: Rotate clockwise with 90° around the Z axis (practical meaning: make the original -X side of the environment map appear at +Y)."
    )
    return np.array(
        [
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def set_initial_cam_pos_rot(cam: NodePath, render: RenderMode):
    cam_pos = (7.25885, 8.61993, 1.38215) if render == "offscreen" else (0, -15, 0)

    print(f"Initial camera position: {cam_pos}")
    print(f"Initial camera rotation: look at (0, 0, 0)")
    cam.set_pos(cam_pos)
    cam.look_at((0, 0, 0))


def load_material_on_reflective_plane(material: FloorMaterial, obj: NodePath) -> None:
    floor_base_color = (0.7, 1.0, 0.7)
    smooth_roughness = 0.4
    floor_metallic = 1 if material == "metal-checked" else 0

    print(f"Floor base color: {floor_base_color}")
    print(f"Floor metallic: {floor_metallic}")

    if material.endswith("checked"):
        print("Floor roughness: checkerboard with size 50x50, min: 0.4, max: 1")
    else:
        print(f"Floor roughness: {smooth_roughness}")

    tex_size = Vec2i(500, 500)
    base_color_tex, metallic_roughness_tex = set_simple_material(
        obj=obj, texture_size=tex_size
    )
    base_color_tex.set_clear_color((0.7, 1.0, 0.7, 1.0))

    if material == "metal-smooth":
        metallic_roughness_tex.set_clear_color((0.7, 0.4, 1.0, 1.0))
    elif material == "diel-checked" or material == "metal-checked":
        x_steps = np.arange(0, tex_size.x, dtype=np.float32)
        y_steps = np.arange(0, tex_size.y, dtype=np.float32)
        x, y = np.meshgrid(x_steps, y_steps)
        check_pattern = np.full((tex_size.y, tex_size.x), 0.4, dtype=np.float32)
        check_pattern[(x % 100 > 50) ^ (y % 100 > 50)] = 1
        metallic = np.full(
            check_pattern.shape,
            1 if material == "metal-checked" else 0,
            dtype=check_pattern.dtype,
        )
        mr_im = np.stack(
            [
                np.ones_like(check_pattern),
                check_pattern,
                metallic,
            ]
        )
        write_2d_rgba_texture(texture=metallic_roughness_tex, im=mr_im)


def parse_args() -> "ParsedArgs":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=Path, help="The path of the model to show.", required=True
    )
    parser.add_argument(
        "--floor",
        type=str,
        choices=FloorTextureValues,
        help="The texture of the reflective plane.",
        required=True,
    )
    parser.add_argument(
        "--render",
        type=str,
        choices=["offscreen", "move"],
        help="How the screen should be rendered.",
        required=True,
    )
    parser.add_argument(
        "--envmap", type=Path, help="The environment map to be used.", required=True
    )
    parser.add_argument(
        "--rpst",
        action="store_true",
        help="If this flag is set, then the reflective plane will ignore the texture coordinates configured for its vertices, and it will use the on-screen texture coordinates instead. The flag abbreviates: reflective plane screen texcoord",
    )

    parsed = parser.parse_args()
    return cast(ParsedArgs, parsed)


class ParsedArgs(Protocol):
    envmap: Path
    model: Path
    floor: FloorMaterial
    render: RenderMode
    rpst: bool


if __name__ == "__main__":
    main(parse_args())
