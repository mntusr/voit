import json
import unittest
import unittest.mock as mock
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import cv2
import depth_tools
import matplotlib.pyplot as plt
import npy_unittest
import numpy as np
import torch
import voit.serialized

from .testutil import VoitTestBase


class TestDatasetBasedInserter(VoitTestBase):
    def setUp(self):
        self.original_sample_count = 9
        self.im_size = voit.Vec2i(73, 18)

        self.camera = depth_tools.CameraIntrinsics(
            f_x=30, f_y=30, c_x=73 // 2, c_y=18 // 2
        )

        rng = np.random.default_rng(15)
        self.values: dict[int, depth_tools.Sample] = {
            5: {
                "rgb": np.zeros((3, self.im_size.y, self.im_size.x)),
                "depth": np.ones((1, self.im_size.y, self.im_size.x)),
                "mask": rng.uniform(0, 1, size=(1, self.im_size.y, self.im_size.x))
                > 0.5,
                "name": "sample5",
                "camera": self.camera,
            },
            2: {
                "rgb": np.zeros((3, self.im_size.y, self.im_size.x)),
                "depth": np.ones((1, self.im_size.y, self.im_size.x)),
                "mask": rng.uniform(0, 1, size=(1, self.im_size.y, self.im_size.x))
                > 0.5,
                "name": "sample2",
                "camera": self.camera,
            },
            0: {
                "rgb": np.full((3, self.im_size.y, self.im_size.x), 0.7),
                "depth": np.ones((1, self.im_size.y, self.im_size.x)) * 2.5,
                "mask": rng.uniform(0, 1, size=(1, self.im_size.y, self.im_size.x))
                > 0.5,
                "name": "sample0",
                "camera": self.camera,
            },
        }
        self.values[5]["mask"][0, 7, 3] = True
        self.values[2]["mask"][0, 7, 3] = True
        self.values[0]["mask"][0, 7, 3] = True

        self.original_samples_mock = _new_original_samples_mock(
            original_sample_count=self.original_sample_count,
            values=self.values,
        )

        self.inserter_mock = mock.Mock(name="inserter")
        self.inserter_mock.destroy = mock.Mock(name="inserter.destroy")

        self.dataset_based_inserter = (
            self._new_dataset_based_inserter_with_test_defaults(
                is_depth_usable_for_depth_maps=True
            )
        )

    def _new_dataset_based_inserter_with_test_defaults(
        self, is_depth_usable_for_depth_maps: bool
    ):
        return voit.serialized.DatasetBasedInserter(
            original_samples={
                "dataset": self.original_samples_mock,
                "global_camera": self.values[5]["camera"],
                "global_im_size": self.im_size,
                "is_depth_usable_for_depth_maps": is_depth_usable_for_depth_maps,
                "is_im_linear": False,
            },
            internal_inserter=self.inserter_mock,
            objs_and_keys={"foo": Path("foo.glb")},
            debug_window=False,
            floor_proxy_size=voit.Vec2(4, 6),
            output_im_linear=False,
            post_insertion_hook=None,
            pt_device=torch.device("cpu"),
        )

    def test_destroy(self):
        self.inserter_mock.destroy.assert_not_called()
        self.dataset_based_inserter.destroy()
        self.inserter_mock.destroy.assert_called()

    def test_get_im_size(self):
        self.assertEqual(self.dataset_based_inserter.im_size, self.im_size)

    def test_insert_to__invalid_pos(self):
        cases = [
            (3, -7),
            (-3, 7),
            (self.dataset_based_inserter.im_size.x, 2),
            (2, self.dataset_based_inserter.im_size.y),
        ]
        for pos_x, pos_y in cases:
            with self.subTest(f"{pos_x=};{pos_y=}"):
                with self.assertRaises(voit.VoitError) as cm:
                    self.dataset_based_inserter.insert_to(
                        {
                            "depth_occl_thr": 2.0,
                            "normal_x": 0,
                            "normal_y": 1,
                            "normal_z": 0,
                            "obj_key": "foo",
                            "pos_x": pos_x,
                            "pos_y": pos_y,
                            "rotation": 13.1,
                            "src_idx": 2,
                        }
                    )

                self.assertIn("outside of the image", str(cm.exception))

    def test_insert_to__invalid_obj_key(self):
        obj_key = "unknown_obj"
        with self.assertRaises(voit.VoitError) as cm:
            self.dataset_based_inserter.insert_to(
                {
                    "depth_occl_thr": 2.0,
                    "normal_x": 0,
                    "normal_y": 1,
                    "normal_z": 0,
                    "obj_key": "unknown_obj",
                    "pos_x": 2,
                    "pos_y": 1,
                    "rotation": 13.1,
                    "src_idx": 2,
                }
            )

        msg = str(cm.exception)
        self.assertIn("no object key", msg)
        self.assertIn(obj_key, msg)

    def test_insert_to__happy_path(self):
        im_size = voit.Vec2i(20, 30)
        rotation_around_floor_normal_cw = -42.3
        src_idx = 2
        for just_inserted_obj_mask_val in [True, False]:
            for depth_maps_in in [True, False]:
                dataset_based_inserter = (
                    self._new_dataset_based_inserter_with_test_defaults(
                        is_depth_usable_for_depth_maps=depth_maps_in
                    )
                )

                with self.subTest(f"{just_inserted_obj_mask_val=},{depth_maps_in=}"):
                    self.dataset_based_inserter._original_samples[
                        "is_depth_usable_for_depth_maps"
                    ] = depth_maps_in
                    insertion_result_depth_mask = self.values[src_idx]["mask"].copy()
                    insertion_result_depth_mask[:3] = False
                    insertion_result_im = np.full((3, im_size.y, im_size.x), 0.7)
                    obj_mask = np.full((1, im_size.y, im_size.x), False)
                    obj_mask[:, :, :15] = True
                    expected_pre_hook_result = voit.InsertionResult(
                        depth=self.values[src_idx]["depth"] * 3,
                        depth_mask=insertion_result_depth_mask,
                        im=insertion_result_im,
                        obj_mask=obj_mask,
                    )

                    # self test: check that the masks and depths are different
                    self.assertFalse(
                        np.allclose(
                            self.values[src_idx]["depth"],
                            expected_pre_hook_result.depth,
                            atol=1e-4,
                        )
                    )
                    self.assertFalse(
                        np.array_equal(
                            self.values[src_idx]["mask"],
                            expected_pre_hook_result.depth_mask,
                        )
                    )
                    self.assertFalse(
                        np.array_equal(
                            self.values[src_idx]["mask"],
                            expected_pre_hook_result.obj_mask,
                        )
                    )

                    env_names: list[str] = []

                    def inserter_bake(**kwargs):
                        at_arg = kwargs["at"]
                        self.assertIs(at_arg["input_im_linear"], False)
                        self.assertEqual(at_arg["pos_px"].x, 3)
                        self.assertEqual(at_arg["pos_px"].y, 7)
                        self.assertEqual(
                            at_arg["pos_depth"], self.values[src_idx]["depth"][0, 7, 3]
                        )
                        self.assertEqual(
                            at_arg["input_im"].shape, self.values[src_idx]["rgb"].shape
                        )
                        if depth_maps_in:
                            self.assertAllclose(
                                at_arg["input_depth"]["depth"], self.values[2]["depth"]
                            )
                            self.assertArrayEqual(
                                at_arg["input_depth"]["mask"], self.values[2]["mask"]
                            )
                        else:
                            self.assertIsNone(at_arg["input_depth"])
                        self.assertEqual(kwargs["name"], "last_insert_serialized")
                        env_names.append(kwargs["name"])

                    # configure the mocks
                    def inserter_insert(**kwargs):
                        self.assertIs(kwargs["output_im_linear"], False)
                        self.assertAlmostEqual(
                            kwargs["depth_occlusion"]["threshold"], 2
                        )
                        self.assertAlmostEqual(kwargs["normal_vs"].x, 0, delta=1e-5)
                        self.assertAlmostEqual(kwargs["normal_vs"].y, 1, delta=1e-5)
                        self.assertAlmostEqual(kwargs["normal_vs"].z, -1, delta=1e-5)
                        self.assertAlmostEqual(
                            kwargs["rotation_around_floor_normal_cw"],
                            rotation_around_floor_normal_cw,
                        )
                        self.assertEqual(env_names[0], kwargs["at"])
                        return expected_pre_hook_result

                    expected_post_hook_result = mock.Mock("post_hook_result")

                    def post_insertion_hook(
                        result: voit.InsertionResult,
                    ) -> voit.InsertionResult:
                        self.assertAllclose(result.im, expected_pre_hook_result.im)
                        self.assertAllclose(
                            result.depth, expected_pre_hook_result.depth
                        )
                        self.assertArrayEqual(
                            result.depth_mask, expected_pre_hook_result.depth_mask
                        )

                        if just_inserted_obj_mask_val:
                            self.assertArrayEqual(
                                result.obj_mask, expected_pre_hook_result.obj_mask
                            )

                        return expected_post_hook_result

                    self.inserter_mock.insert = mock.Mock(
                        name="inserter.insert", side_effect=inserter_insert
                    )
                    self.inserter_mock.bake = mock.Mock(
                        name="inserter.bake", side_effect=inserter_bake
                    )

                    dataset_based_inserter.post_insertion_hook = post_insertion_hook
                    actual_post_hook_result = dataset_based_inserter.insert_to(
                        {
                            "depth_occl_thr": 2.0,
                            "normal_x": 0,
                            "normal_y": 1,
                            "normal_z": -1,
                            "obj_key": "foo",
                            "pos_x": 3,
                            "pos_y": 7,
                            "rotation": rotation_around_floor_normal_cw,
                            "src_idx": 2,
                        }
                    )
                    self.assertIs(actual_post_hook_result, expected_post_hook_result)

    def test_insert_to__caching(self):
        inserter_input: voit.serialized.InsertionSpec = {
            "depth_occl_thr": 2.0,
            "normal_x": 0,
            "normal_y": 1,
            "normal_z": -1,
            "obj_key": "foo",
            "pos_x": 3,
            "pos_y": 7,
            "rotation": 0.7,
            "src_idx": 2,
        }
        self.inserter_mock.bake = mock.Mock(name="inserter.bake")
        self.dataset_based_inserter.insert_to(inserter_input)
        self.assertEqual(self.inserter_mock.bake.call_count, 1)

        self.dataset_based_inserter.insert_to(inserter_input)
        self.assertEqual(self.inserter_mock.bake.call_count, 1)

        inserter_input["src_idx"] += 3
        self.dataset_based_inserter.insert_to(inserter_input)
        self.assertEqual(self.inserter_mock.bake.call_count, 2)

        inserter_input["pos_x"] += 1
        self.dataset_based_inserter.insert_to(inserter_input)
        self.assertEqual(self.inserter_mock.bake.call_count, 3)

        inserter_input["pos_y"] += 1
        self.dataset_based_inserter.insert_to(inserter_input)
        self.assertEqual(self.inserter_mock.bake.call_count, 4)

        inserter_input["depth_occl_thr"] += 1
        self.dataset_based_inserter.insert_to(inserter_input)
        self.assertEqual(self.inserter_mock.bake.call_count, 4)


def _new_original_samples_mock(
    original_sample_count: int,
    values: dict[int, depth_tools.Sample],
):
    def get_sample(idx: int) -> depth_tools.Sample:
        return values[idx]

    t_proj_mat = np.eye(3)
    original_samples = mock.Mock(name="original_samples")
    original_samples.get_t_proj_mat = mock.Mock(
        name="original_samples.get_t_proj_mat", side_effect=lambda: t_proj_mat
    )
    original_samples.__len__ = mock.Mock(
        name="original_samples.__len__",
        side_effect=lambda: original_sample_count,
    )
    original_samples.__getitem__ = mock.Mock(
        name="original_samples.__getitem__",
        side_effect=get_sample,
    )
    original_samples.is_depth_usable_for_depth_maps = mock.Mock(
        name="original_samples.is_depth_usable_for_depth_maps", return_value=True
    )
    return original_samples


class TestDatasetWithObjectsInserted(npy_unittest.NpyTestCase):
    @classmethod
    def setUpClass(cls):
        cls.insertion_specs: list[voit.serialized.InsertionSpec] = [
            {
                "depth_occl_thr": 3,
                "normal_x": 5,
                "normal_y": 9,
                "normal_z": 22,
                "obj_key": "key1",
                "pos_x": 43,
                "pos_y": 26,
                "rotation": 35,
                "src_idx": 2,
            },
            {
                "depth_occl_thr": 7,
                "normal_x": 2,
                "normal_y": 6,
                "normal_z": 43,
                "obj_key": "key1",
                "pos_x": 41,
                "pos_y": 31,
                "rotation": 8,
                "src_idx": 5,
            },
        ]
        cls.src_idx_reverse_lookup = {
            cls.insertion_specs[i]["src_idx"]: i
            for i in range(len(cls.insertion_specs))
        }
        rng = np.random.default_rng(30)
        cls.rgbs = cls.reconvert_ims(
            rng.uniform(0, 1, size=(len(cls.insertion_specs), 3, 400, 360))
        )

        cls.depths = rng.uniform(
            1, 120, size=(len(cls.insertion_specs), 3, 400, 360)
        ).astype(np.float32)
        cls.depth_masks = (
            rng.uniform(1, 120, size=(len(cls.insertion_specs), 3, 400, 360)) > 0.5
        )
        cls.obj_masks = (
            rng.uniform(1, 120, size=(len(cls.insertion_specs), 3, 400, 360)) > 0.5
        )

        cls.inserter_mock = mock.Mock("inserter")
        cls.inserter_mock.insert_to = mock.Mock(
            "inserter.insert_to",
            side_effect=lambda insertion_spec: cls.insertion_mock_fn(insertion_spec),
        )
        cls.inserter_mock.dataset = mock.Mock("inserter.dataset")
        cls.inserter_mock.dataset.__getitem__ = mock.Mock(
            "inserter.dataset.__getitem__", side_effect=cls.get_orig_sample_at
        )

        cls.sample_names = [f"name{i}" for i in range(len(cls.insertion_specs))]
        cls.cameras = [
            depth_tools.CameraIntrinsics(2, 3, 4, 5),
            depth_tools.CameraIntrinsics(6, 7, 8, 9),
        ]

        cls.cache_dir_obj = TemporaryDirectory()
        voit.serialized.DatasetWithObjectsInserted.generate(
            cache_dir=Path(cls.cache_dir_obj.name),
            report_progress_tqdm=False,
            inserter=cls.inserter_mock,
            insertion_specs=cls.insertion_specs,
        )
        cls.cache_dir = Path(cls.cache_dir_obj.name)

    @staticmethod
    def reconvert_ims(im: np.ndarray) -> np.ndarray:
        im = im.astype(np.float32)
        im = im * 255
        im = im.astype(np.uint8)
        im = im.astype(np.float32)
        im = im / 255
        return im

    @classmethod
    def tearDownClass(cls):
        cls.cache_dir_obj.cleanup()

    @classmethod
    def get_orig_sample_at(cls, orig_sample_idx: int, /) -> depth_tools.Sample:
        insertion_sample_idx = cls.src_idx_reverse_lookup[orig_sample_idx]
        return {
            "rgb": cls.rgbs[insertion_sample_idx],
            "depth": cls.depths[insertion_sample_idx],
            "mask": cls.depth_masks[insertion_sample_idx],
            "name": cls.sample_names[insertion_sample_idx],
            "camera": cls.cameras[insertion_sample_idx],
        }

    @classmethod
    def insertion_mock_fn(
        cls,
        insertion_spec: voit.serialized.InsertionSpec,
    ) -> voit.InsertionResult:
        spec_idx = cls.src_idx_reverse_lookup[insertion_spec["src_idx"]]
        return voit.InsertionResult(
            im=cls.rgbs[spec_idx],
            depth=cls.depths[spec_idx],
            depth_mask=cls.depth_masks[spec_idx],
            obj_mask=cls.obj_masks[spec_idx],
        )

    def assertSampleBelongsToInsertionSpec(
        self,
        sample: depth_tools.Sample,
        obj_mask: np.ndarray,
        retrieved_name: str,
        retrieved_original_sample_index: int,
        spec_idx: int,
    ) -> None:
        max_diff = abs(sample["rgb"] - self.rgbs[spec_idx]).max()

        self.assertAllclose(sample["rgb"], self.rgbs[spec_idx])
        self.assertAllclose(sample["depth"], self.depths[spec_idx], atol=1e-4)
        self.assertArrayEqual(sample["mask"], self.depth_masks[spec_idx])
        self.assertArrayEqual(obj_mask, self.obj_masks[spec_idx])
        self.assertEqual(retrieved_name, self.sample_names[spec_idx])
        self.assertEqual(
            retrieved_original_sample_index, self.insertion_specs[spec_idx]["src_idx"]
        )

        self.assertAlmostEqual(sample["camera"].f_x, self.cameras[spec_idx].f_x)
        self.assertAlmostEqual(sample["camera"].f_y, self.cameras[spec_idx].f_y)
        self.assertAlmostEqual(sample["camera"].c_x, self.cameras[spec_idx].c_x)
        self.assertAlmostEqual(sample["camera"].c_y, self.cameras[spec_idx].c_y)

    def test_len(self):
        dataset = voit.serialized.DatasetWithObjectsInserted(cache_dir=self.cache_dir)

        self.assertEqual(len(dataset), len(self.insertion_specs))

    def test_simple_sample_reading(self) -> None:
        dataset = voit.serialized.DatasetWithObjectsInserted(cache_dir=self.cache_dir)
        sample0 = dataset[0]
        self.assertSampleBelongsToInsertionSpec(
            sample=sample0,
            spec_idx=0,
            obj_mask=dataset.get_obj_mask(0),
            retrieved_name=dataset.original_names[0],
            retrieved_original_sample_index=dataset.original_indices[0],
        )

        sample1 = dataset[1]
        self.assertSampleBelongsToInsertionSpec(
            sample=sample1,
            spec_idx=1,
            obj_mask=dataset.get_obj_mask(1),
            retrieved_name=dataset.original_names[1],
            retrieved_original_sample_index=dataset.original_indices[1],
        )

    def test_oob_obj_mask_idx(self):
        dataset = voit.serialized.DatasetWithObjectsInserted(cache_dir=self.cache_dir)
        with self.assertRaises(IndexError):
            dataset.get_obj_mask(100000)

        with self.assertRaises(IndexError):
            dataset.get_obj_mask(-100000)

    def test_oob_general_sample_idx(self):
        dataset = voit.serialized.DatasetWithObjectsInserted(cache_dir=self.cache_dir)

        with self.assertRaises(IndexError):
            dataset[10000]

        with self.assertRaises(IndexError):
            dataset[-110000]

    def test_negative_general_sample_idx(self):
        dataset = voit.serialized.DatasetWithObjectsInserted(cache_dir=self.cache_dir)
        sample_pos = dataset[len(dataset) - 1]
        sample_neg = dataset[-1]

        # self test
        self.assertGreater(len(dataset), 1)

        # real checks
        self.assertAllclose(sample_pos["depth"], sample_neg["depth"])
        self.assertAllclose(sample_pos["rgb"], sample_neg["rgb"])
        self.assertArrayEqual(sample_pos["mask"], sample_neg["mask"])
        self.assertEqual(sample_pos["name"], sample_neg["name"])
        self.assertEqual(sample_pos["camera"], sample_neg["camera"])

    def test_negative_obj_mask_idx(self):
        dataset = voit.serialized.DatasetWithObjectsInserted(cache_dir=self.cache_dir)

        mask_pos = dataset.get_obj_mask(len(dataset) - 1)
        mask_neg = dataset.get_obj_mask(-1)

        self.assertArrayEqual(mask_pos, mask_neg)
