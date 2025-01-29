import time
from typing import Any, cast

import depth_tools
import matplotlib.pyplot as plt
import numpy as np
import torch
from panda3d.core import MaterialAttrib, TextureStage
from voit import EnvSpecParam, Inserter, InsertionResult, Vec2, Vec2i, Vec3, VoitError
from voit._insertion_main import (
    _binary_erode_mask_disk2,
    _diffuse_color_2_base_color,
    _get_envmap,
    _roughness_2_metallic_roughness,
)

from .testutil import VoitTestBase


class TestInserter(VoitTestBase):
    def setUp(self):
        self.input_im_color = Vec3(0.2, 0.5, 0.7)
        self.camera = depth_tools.CameraIntrinsics(f_x=500, f_y=500, c_x=400, c_y=300)
        self.inserter = Inserter(
            floor_proxy_size=Vec2(5, 5),
            im_size=Vec2i(600, 400),
            pt_device=torch.device("cuda"),
            camera=self.camera,
            debug_window=False,
        )

        self.input_im = np.zeros((3, self.inserter.im_size.y, self.inserter.im_size.x))
        self.input_im[0] = self.input_im_color.x
        self.input_im[1] = self.input_im_color.y
        self.input_im[2] = self.input_im_color.z

        self.input_depth_depth = np.full(
            (1, self.inserter.im_size.y, self.inserter.im_size.x), 9, dtype=np.float32
        )
        self.input_depth_mask = np.full(
            (1, self.inserter.im_size.y, self.inserter.im_size.x), True
        )
        self.input_depth_mask[0, 1, 0] = False
        test_obj_path = self.get_test_data_dir("insertion") / "rect_1x1x2.glb"
        load_successful = False
        try:
            self.test_obj = self.inserter.load_model(test_obj_path)
            load_successful = True
        finally:
            if not load_successful:
                self.inserter.destroy()

    def tearDown(self) -> None:
        self.inserter.destroy()

    def test_color(self):
        cases = [
            {
                "name": "both_linear",
                "input_im_linear": True,
                "output_im_linear": True,
                "input_im_power": 1,
            },
            {
                "name": "both_srgb",
                "input_im_linear": False,
                "output_im_linear": False,
                "input_im_power": 1,
            },
            {
                "name": "linear_2_srgb",
                "input_im_linear": True,
                "output_im_linear": False,
                "input_im_power": 1 / 2.2,
            },
            {
                "name": "srgb_2_linear",
                "input_im_linear": False,
                "output_im_linear": True,
                "input_im_power": 2.2,
            },
        ]

        for case in cases:
            with self.subTest(case["name"]):
                out = self._generate_output(
                    at_arg_change={
                        "input_im_linear": case["input_im_linear"],
                    },
                    other_arg_change={
                        "output_im_linear": case["output_im_linear"],
                    },
                )
                top_left_px_color = Vec3(
                    out.im[0, 0, 0],
                    out.im[1, 0, 0],
                    out.im[2, 0, 0],
                )
                self.assertAlmostEqual(
                    top_left_px_color.x, self.input_im_color.x ** case["input_im_power"]
                )
                self.assertAlmostEqual(
                    top_left_px_color.y, self.input_im_color.y ** case["input_im_power"]
                )
                self.assertAlmostEqual(
                    top_left_px_color.z, self.input_im_color.z ** case["input_im_power"]
                )

    def test_obj_mask(self):
        out = self._generate_output()

        # plt.imshow(out.obj_mask[0])
        # plt.show(block=True)

        self.assertFalse(out.obj_mask[0, 0, 0])
        self.assertTrue(out.obj_mask[0, 290, 200])

    def test_color_determinism(self):
        outs = [
            self._generate_output(),
            self._generate_output(),
            self._generate_output(),
        ]
        time.sleep(1)
        outs.append(self._generate_output())
        px_pos = Vec2i(200, 290)

        first_color = outs[0].im[:, px_pos.y, px_pos.y]

        for i in range(1, len(outs)):
            # plt.imshow(outs[i].im.transpose([1, 2, 0]))
            # plt.title(str(i))
            # plt.show(block=True)
            # plt.close()
            current_color = outs[i].im[:, px_pos.y, px_pos.y]

            self.assertAllclose(first_color, current_color)

    def test_depth(self):
        no_depth = self._generate_output(at_arg_change={"input_depth": None})
        depth_present = self._generate_output()

        # plt.imshow(depth_present.obj_mask[0])
        # plt.show(block=True)

        # top left depth (depth present)
        depth_at_top_left = depth_present.depth[0, 0, 0]
        self.assertAlmostEqual(depth_at_top_left, self.input_depth_depth[0, 0, 0])
        self.assertTrue(depth_present.depth_mask[0, 0, 0])
        self.assertFalse(depth_present.depth_mask[0, 1, 0])

        # top left depth (no depth)
        depth_at_top_left = no_depth.depth[0, 0, 0]
        self.assertAlmostEqual(depth_at_top_left, 0)
        self.assertFalse(no_depth.depth_mask[0, 0, 0])
        self.assertFalse(no_depth.depth_mask[0, 1, 0])

        # object depth (depth present)
        self.assertAlmostEqual(depth_present.depth[0, 290, 200], 5 - 1, delta=1e-4)
        self.assertAlmostEqual(no_depth.depth[0, 290, 200], 5 - 1, delta=1e-4)

        # no floor depth
        self.assertAlmostEqual(
            depth_present.depth[0, 360, 200],
            self.input_depth_depth[0, 360, 200],
            delta=1e-4,
        )
        self.assertTrue(depth_present.depth_mask[0, 360, 200])
        self.assertAlmostEqual(no_depth.depth[0, 360, 200], 0, delta=1e-4)
        self.assertFalse(no_depth.depth_mask[0, 360, 200])

    def test_normal(self):
        normal_points_to_camera = self._generate_output(
            other_arg_change={"normal_vs": Vec3(0, 0, -1)}
        )

        # plt.imshow(np.clip(normal_points_to_camera.depth, -1, 10)[0])
        # plt.show(block=True)

        self.assertAlmostEqual(
            normal_points_to_camera.depth[0, 290, 200], 5 - 2, delta=1e-3
        )

    def test_depth_occlusion__object_occluded(self):
        out = self._generate_output(
            at_arg_change={
                "pos_depth": float(self.input_depth_depth[0, 0, 0]) + 5,
            },
            other_arg_change={
                "depth_occlusion": {
                    "threshold": 0.1,
                    "dense_depth": self.input_depth_depth,
                },
            },
        )

        self.assertAllAlmostEqual(
            out.depth[out.depth_mask], self.input_depth_depth[0, 0, 0]
        )
        self.assertTrue(np.all(~out.obj_mask))

    def test_depth_occlusion__object_not_occluded(self):
        out = self._generate_output(
            other_arg_change={
                "depth_occlusion": {
                    "threshold": 0.1,
                    "dense_depth": self.input_depth_depth,
                },
            }
        )

        self.assertTrue(
            np.all(out.depth[out.obj_mask] < self.input_depth_depth[0, 0, 0])
        )
        self.assertFalse(np.all(~out.obj_mask))

    def test_error_handling(self):
        with self.subTest("invalid_im_shape"):
            with self.assertRaises(VoitError):
                self._generate_output(
                    at_arg_change={
                        "input_im": np.array((3, 100, 205), dtype=self.input_im.dtype)
                    }
                )
        with self.subTest("invalid_im_dtype"):
            with self.assertRaises(VoitError):
                self._generate_output(
                    at_arg_change={"input_im": self.input_im.astype(np.uint8)}
                )
        with self.subTest("invalid_depth_depth_shape"):
            with self.assertRaises(VoitError):
                self._generate_output(
                    at_arg_change={
                        "input_depth": {
                            "depth": np.array(
                                (1, 100, 205), dtype=self.input_depth_depth.dtype
                            ),
                            "mask": self.input_depth_mask,
                        }
                    }
                )
        with self.subTest("invalid_depth_mask_dtype"):
            with self.assertRaises(VoitError):
                self._generate_output(
                    at_arg_change={
                        "input_depth": {
                            "depth": self.input_depth_depth.astype(np.uint8),
                            "mask": self.input_depth_mask,
                        }
                    }
                )
        with self.subTest("invalid_depth_mask_shape"):
            with self.assertRaises(VoitError):
                self._generate_output(
                    at_arg_change={
                        "input_depth": {
                            "depth": self.input_depth_depth,
                            "mask": np.full((1, 100, 65), False),
                        }
                    }
                )
        with self.subTest("invalid_depth_mask_dtype"):
            with self.assertRaises(VoitError):
                self._generate_output(
                    at_arg_change={
                        "input_depth": {
                            "depth": self.input_depth_depth,
                            "mask": self.input_depth_mask.astype(np.uint8),
                        }
                    }
                )
        with self.subTest("invalid_depth_occlusion_depth_shape"):
            with self.assertRaises(VoitError):
                self._generate_output(
                    other_arg_change={
                        "depth_occlusion": {
                            "dense_depth": np.full(
                                (1, 200, 300), 0.7, dtype=self.input_depth_depth.dtype
                            ),
                        }
                    }
                )
        with self.subTest("invalid_depth_occlusion_depth_dtype"):
            with self.assertRaises(VoitError):
                self._generate_output(
                    other_arg_change={
                        "depth_occlusion": {
                            "dense_depth": self.input_depth_depth.astype(np.uint8),
                        }
                    }
                )

    def test_caching(self):
        env_name = "a1"
        at: EnvSpecParam = {
            "input_im_linear": True,
            "input_depth": {
                "depth": self.input_depth_depth,
                "mask": self.input_depth_mask,
            },
            "pos_px": Vec2i(200, 300),
            "pos_depth": 4,
            "input_im": self.input_im,
        }
        self.inserter.bake(
            at=at,
            name=env_name,
        )
        self.assertEqual(self.inserter.cache_names, frozenset({env_name}))

        out1 = self._generate_output(at_arg_change=env_name)
        out2 = self._generate_output(at_arg_change=cast(dict[str, Any], at))

        # account for nondeterminisms during the inverse rendering phase
        n_differing_pixels = (abs(out2.im - out1.im) > 1e-4).sum()
        # if abs(out2.im - out1.im).max() > 1e-4:
        #    plt.imshow(abs(out2.im - out1.im).transpose([1, 2, 0]))
        #    plt.title("|out2.im-out1.im|")
        #    plt.show(block=True)
        #    plt.close()
        #
        #    fig, axs = plt.subplots(nrows=1, ncols=3)
        #    axs[0].imshow(out1.im.transpose([1, 2, 0]))
        #    axs[0].set_title("out1.im")
        #    axs[1].imshow(out2.im.transpose([1, 2, 0]))
        #    axs[1].set_title("out2.im")
        #    axs[2].imshow(abs(out2.im - out1.im).transpose([1, 2, 0]))
        #    axs[2].set_title("|out2.im-out1.im|")
        #    plt.show(block=True)
        #    plt.close()

        self.assertLess(n_differing_pixels, 10)
        self.assertAllclose(out1.depth, out2.depth)
        self.assertArrayEqual(out1.depth_mask, out2.depth_mask)
        self.assertArrayEqual(out1.obj_mask, out2.obj_mask)

    def _generate_output(
        self,
        *,
        at_arg_change: dict[str, Any] | str | None = None,
        other_arg_change: dict[str, Any] | None = None,
    ) -> InsertionResult:
        """
        Do an object insertion and give back the result.

        By default the function uses the following call: ::

            self.inserter.insert(
                at= {
                    "input_im_linear": True,
                    "input_depth": {
                        "depth": self.input_depth_depth,
                        "mask": self.input_depth_mask,
                    },
                    "pos_px": Vec2i(200, 300),
                    "pos_depth": 5,
                    "input_im": self.input_im,
                },
                normal_vs=Vec3(0, 1, 0),
                output_im_linear=True,
                obj=self.test_obj,
            )
        """
        at_args: EnvSpecParam | str = {
            "input_im_linear": True,
            "input_depth": {
                "depth": self.input_depth_depth,
                "mask": self.input_depth_mask,
            },
            "pos_px": Vec2i(200, 300),
            "pos_depth": 5,
            "input_im": self.input_im,
        }
        other_args = {
            "normal_vs": Vec3(0, 1, 0),
            "output_im_linear": True,
            "obj": self.test_obj,
        }

        if at_arg_change is not None:
            if isinstance(at_arg_change, str):
                at_args = at_arg_change
            else:
                for key, value in at_arg_change.items():
                    if key not in at_args.keys():
                        raise ValueError(f'Unknown key "{key}" for argument "at".')
                    at_args[key] = value

        if other_arg_change is not None:
            for key, value in other_arg_change.items():
                if key == "at":
                    raise ValueError(
                        'The "at" argument is not changeable using argument "other_args". Use "at_arg_change" instead.'
                    )
                other_args[key] = value

        return self.inserter.insert(at=at_args, **other_args)


class TestFunctions(VoitTestBase):
    def test_diffuse_color_2_base_color(self):
        expected_base_color = np.full(shape=(3, 15, 20), fill_value=0.5)
        F0 = 0.04
        m = 0.0
        albedo = (expected_base_color * (1 - F0)) * (1 - m)

        actual_base_color = _diffuse_color_2_base_color(albedo=albedo)

        self.assertAllclose(actual_base_color, expected_base_color)

    def test_binary_erode_mask_disk2__happy_path(self):
        mask = np.ones((1, 13, 13), dtype=np.bool_)
        mask[0, 1, 6] = 0

        expected_result = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ],
            dtype=np.bool_,
        )

        actual_result = _binary_erode_mask_disk2(mask)

        self.assertTrue(np.issubdtype(actual_result.dtype, np.bool_))
        self.assertArrayEqual(actual_result, expected_result)

    def test_binary_erode_mask_disk2__invalid_mask_dtype(self):
        mask = np.ones((1, 13, 13), dtype=np.uint8)
        with self.assertRaises(VoitError) as cm:
            _binary_erode_mask_disk2(mask)

        msg = str(cm.exception)
        self.assertIn("dtype", msg)
        self.assertIn("uint8", msg)
        self.assertIn("bool", msg)
        self.assertIn("mask", msg)

    def test_binary_erode_mask_disk2__invalid_mask_shape(self):
        mask = np.ones((1, 13), dtype=np.bool_)
        with self.assertRaises(VoitError) as cm:
            _binary_erode_mask_disk2(mask)

        msg = str(cm.exception)
        self.assertIn("(1, 13)", msg)
        self.assertIn("mask", msg)
        self.assertIn("shape", msg)

    def test_get_envmap__happy_path(self):
        half_envmaps = np.zeros((100, 150, 8, 16, 3), dtype=np.float32)
        half_envmaps[23, 15] = 1

        im_size = Vec2i(300, 200)
        pixel_pos = Vec2i(30, 46)

        got_envmap = _get_envmap(
            half_envmaps=half_envmaps,
            original_im_size=im_size,
            original_px_pos=pixel_pos,
        )
        self.assertEqual(got_envmap.shape, (3, 512, 1024))
        self.assertAllAlmostEqual(got_envmap[:, :256, :], 1)
        self.assertAllAlmostEqual(got_envmap[:, 256:, :], 0)

    def test_get_envmap__zero_px_pos_x(self):
        half_envmaps = np.zeros((100, 150, 3, 8, 16), dtype=np.float32)

        im_size = Vec2i(300, 200)
        pixel_pos = Vec2i(0, 46)

        _get_envmap(
            half_envmaps=half_envmaps,
            original_im_size=im_size,
            original_px_pos=pixel_pos,
        )

    def test_get_envmap__zero_px_pos_y(self):
        half_envmaps = np.zeros((100, 150, 3, 8, 16), dtype=np.float32)

        im_size = Vec2i(300, 200)
        pixel_pos = Vec2i(30, 0)

        _get_envmap(
            half_envmaps=half_envmaps,
            original_im_size=im_size,
            original_px_pos=pixel_pos,
        )

    def test_get_envmap__invalid_im_size(self):
        half_envmaps = np.zeros((100, 150, 3, 8, 16), dtype=np.float32)

        im_size = Vec2i(-1, 200)
        pixel_pos = Vec2i(30, 46)

        with self.assertRaises(VoitError):
            _get_envmap(
                half_envmaps=half_envmaps,
                original_im_size=im_size,
                original_px_pos=pixel_pos,
            )

    def test_get_envmap__invalid_pixel_pos(self):
        half_envmaps = np.zeros((100, 150, 3, 8, 16), dtype=np.float32)

        im_size = Vec2i(300, 200)
        pos_vals = [
            Vec2i(3, -1),
            Vec2i(-3, 1),
            Vec2i(im_size.x, 1),
            Vec2i(3, im_size.y),
            Vec2i(im_size.x + 1, 1),
            Vec2i(3, im_size.y + 1),
        ]

        for pixel_pos in pos_vals:
            with self.subTest(f"({pixel_pos.x},{pixel_pos.y})"):
                with self.assertRaises(VoitError):
                    _get_envmap(
                        half_envmaps=half_envmaps,
                        original_im_size=im_size,
                        original_px_pos=pixel_pos,
                    )

    def test_roughness_2_metallic_roughness__happy_path(self):
        roughness = np.full((1, 20, 15), 0.7, dtype=np.float32)
        metallic_roughness = _roughness_2_metallic_roughness(roughness)

        self.assertAllclose(metallic_roughness[[1]], roughness)
        self.assertAllclose(metallic_roughness[[2]], np.zeros_like(roughness))

    def test_roughness_2_metallic_roughness__invalid_shape(self):
        cases: list[tuple[str, tuple[int, ...]]] = [
            ("incorrect_dim_count", (2, 3)),
            ("incorrect_channel_count", (4, 9, 5)),
        ]
        for case_name, shape in cases:
            with self.subTest(case_name):
                with self.assertRaises(VoitError):
                    _roughness_2_metallic_roughness(np.ones(shape, dtype=np.float32))

    def test_roughness_2_metallic_roughness__invalid_dtype(self):
        with self.assertRaises(VoitError):
            _roughness_2_metallic_roughness(np.ones((3, 8, 5), dtype=np.uint8))
