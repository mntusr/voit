from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch.cuda import is_available

from voit._ctx_estimation_internal import CtxPredictor

from .testutil import VoitTestBase


class TestCtxPredictor(VoitTestBase):
    def test_predict(self):
        test_device = self.get_test_device()
        predictor = CtxPredictor()
        checkpoint_path = Path(".").resolve() / "voit_ctx_estim_weights"
        predictor.load_checkpoint(checkpoint_path)
        predictor.to(test_device)

        test_ims = self.get_ctx_pred_test_images()
        actual_albedo, actual_roughness, actual_envmap_preds, _ = predictor.predict(
            test_ims.input_im
        )

        actual_albedo = cv.resize(
            actual_albedo.transpose([1, 2, 0]),
            (actual_albedo.shape[2] // 4, actual_albedo.shape[1] // 4),
        ).transpose([2, 0, 1])
        actual_roughness = np.expand_dims(
            cv.resize(
                actual_roughness.transpose([1, 2, 0]),
                (actual_roughness.shape[2] // 4, actual_roughness.shape[1] // 4),
            ),
            axis=0,
        )
        actual_chosen_half_envmap = actual_envmap_preds[-20, -35]

        self.assertAll(np.mean((actual_albedo - test_ims.expected_albedo) ** 2) < 0.1)  # type: ignore
        self.assertAll(
            np.mean((actual_roughness - test_ims.expected_roughness) ** 2) < 0.1  # type: ignore
        )
        self.assertTrue(
            np.all(
                np.mean(
                    (actual_chosen_half_envmap - test_ims.expected_chosen_half_envmap)
                    ** 2
                )
                < 0.1
            )
        )

    def get_ctx_pred_test_images(self) -> "CtxPredTestImages":
        testdata_dir = self.get_test_data_dir("inverse_rendering")

        def load_test_im(stem: str) -> np.ndarray:
            """
            Load the test image by stem.

            Parameters
            ----------
            stem
                The stem of the filename of the loaded image.

            Returns
            -------
            v
                The loaded image. Format: ``Im::RGBLike``
            """
            file_path = testdata_dir / f"{stem}.png"
            im = cv.imread(str(file_path))
            im = im.astype(np.float32) / 255
            im = im[:, :, ::-1]
            im = im.transpose([2, 0, 1])
            return im

        albedo_im = load_test_im("0001_albedoBS1") ** 2.2
        rough_im = load_test_im("0001_roughBS1")
        rough_im = rough_im[[0]]
        input_im = load_test_im("0001") ** 2.2
        half_envmap_preds = np.load(testdata_dir / "0001_envmap1_m20_m35.npy")
        return CtxPredTestImages(
            expected_albedo=albedo_im,
            expected_chosen_half_envmap=half_envmap_preds,
            expected_roughness=rough_im,
            input_im=input_im,
        )

    def get_test_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


@dataclass
class CtxPredTestImages:
    input_im: np.ndarray
    expected_roughness: np.ndarray
    expected_albedo: np.ndarray
    expected_chosen_half_envmap: np.ndarray
