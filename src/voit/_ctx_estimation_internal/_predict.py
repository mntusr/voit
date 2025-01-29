import itertools
import traceback
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model
from torch.autograd import Variable

from .._errors import VoitError
from .._logging_internal import VOIT_LOGGER
from .BilateralLayer import BilateralLayer
from .models import (
    LSregressDiffSpec,
    decoder0,
    decoderLight,
    encoder0,
    encoderLight,
    output2env,
    renderingLayer,
)


class CtxPredictor:
    _LEVEL = 2
    _SG_NUM = 12
    _EXPERIMENTS = ["check_cascadeNYU0", "check_cascadeNYU1"]
    _NEPOCHS = [2, 3]
    _IS_LIGHT = True
    _EXPERIMENTS_LIGHT = [
        "check_cascadeLight0_sg12_offset1",
        "check_cascadeLight1_sg12_offset1",
    ]
    _NEPOCHS_LIGHT = [10, 10]
    _IS_BS = True
    _EXPERIMENTS_BS = ["checkBs_cascade0_w320_h240", "checkBs_cascade1_w320_h240"]
    _NEPOCHS_BS = [15, 8]
    _NITERS_BS = [1000, 4500]
    _IM_WIDTHS = [320, 320]
    _IM_HEIGHTS = [240, 240]
    _ENV_ROW = 120
    _ENV_COL = 160

    _IBL_MAP_WIDTH = 16
    _IBL_MAP_HEIGHT = 8

    def __init__(self) -> None:

        self.encoders: list[encoder0] = []
        self.albedoDecoders: list[decoder0] = []
        self.normalDecoders: list[decoder0] = []
        self.roughDecoders: list[decoder0] = []
        self.depthDecoders: list[decoder0] = []

        self.lightEncoders: list[encoderLight] = []
        self.axisDecoders: list[decoderLight] = []
        self.lambDecoders: list[decoderLight] = []
        self.weightDecoders: list[decoderLight] = []

        self.albedoBSs: list[BilateralLayer] = []
        self.depthBSs: list[BilateralLayer] = []
        self.roughBSs: list[BilateralLayer] = []

        for n in range(0, CtxPredictor._LEVEL):
            # BRDF Predictioins
            self.encoders.append(encoder0(cascadeLevel=n).eval())
            self.albedoDecoders.append(decoder0(mode=0).eval())
            self.normalDecoders.append(decoder0(mode=1).eval())
            self.roughDecoders.append(decoder0(mode=2).eval())
            self.depthDecoders.append(decoder0(mode=4).eval())

            # Light networks
            self.lightEncoders.append(
                encoderLight(cascadeLevel=n, SGNum=CtxPredictor._SG_NUM).eval()
            )
            self.axisDecoders.append(
                decoderLight(mode=0, SGNum=CtxPredictor._SG_NUM).eval()
            )
            self.lambDecoders.append(
                decoderLight(mode=1, SGNum=CtxPredictor._SG_NUM).eval()
            )
            self.weightDecoders.append(
                decoderLight(mode=2, SGNum=CtxPredictor._SG_NUM).eval()
            )

            # BS network
            self.albedoBSs.append(BilateralLayer(mode=0))
            self.roughBSs.append(BilateralLayer(mode=2))
            self.depthBSs.append(BilateralLayer(mode=4))

        self.__device = torch.device("cpu")

    def _get_all_modules(
        self,
    ) -> list[encoder0 | decoder0 | encoderLight | decoderLight | BilateralLayer]:
        return list(
            itertools.chain(
                self.encoders,
                self.albedoDecoders,
                self.normalDecoders,
                self.roughDecoders,
                self.depthDecoders,
                self.lightEncoders,
                self.axisDecoders,
                self.lambDecoders,
                self.weightDecoders,
                self.albedoBSs,
                self.depthBSs,
                self.roughBSs,
            )
        )

    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        self._make_sure_checkpoints_available(checkpoint_dir)

        for n in range(0, CtxPredictor._LEVEL):
            # BRDF Predictioins
            self.encoders.append(encoder0(cascadeLevel=n).eval())
            self.albedoDecoders.append(decoder0(mode=0).eval())
            self.normalDecoders.append(decoder0(mode=1).eval())
            self.roughDecoders.append(decoder0(mode=2).eval())
            self.depthDecoders.append(decoder0(mode=4).eval())

            # Load weight
            _load_model_from(
                model=self.encoders[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/encoder{1}_{2}.safetensors".format(
                    CtxPredictor._EXPERIMENTS[n], n, CtxPredictor._NEPOCHS[n] - 1
                ),
            )
            _load_model_from(
                model=self.albedoDecoders[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/albedo{1}_{2}.safetensors".format(
                    CtxPredictor._EXPERIMENTS[n], n, CtxPredictor._NEPOCHS[n] - 1
                ),
            )
            _load_model_from(
                model=self.normalDecoders[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/normal{1}_{2}.safetensors".format(
                    CtxPredictor._EXPERIMENTS[n], n, CtxPredictor._NEPOCHS[n] - 1
                ),
            )
            _load_model_from(
                model=self.roughDecoders[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/rough{1}_{2}.safetensors".format(
                    CtxPredictor._EXPERIMENTS[n], n, CtxPredictor._NEPOCHS[n] - 1
                ),
            )
            _load_model_from(
                model=self.depthDecoders[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/depth{1}_{2}.safetensors".format(
                    CtxPredictor._EXPERIMENTS[n], n, CtxPredictor._NEPOCHS[n] - 1
                ),
            )

            for param in self.encoders[n].parameters():
                param.requires_grad = False
            for param in self.albedoDecoders[n].parameters():
                param.requires_grad = False
            for param in self.normalDecoders[n].parameters():
                param.requires_grad = False
            for param in self.roughDecoders[n].parameters():
                param.requires_grad = False
            for param in self.depthDecoders[n].parameters():
                param.requires_grad = False

            # Light networks
            self.lightEncoders.append(
                encoderLight(cascadeLevel=n, SGNum=CtxPredictor._SG_NUM).eval()
            )
            self.axisDecoders.append(
                decoderLight(mode=0, SGNum=CtxPredictor._SG_NUM).eval()
            )
            self.lambDecoders.append(
                decoderLight(mode=1, SGNum=CtxPredictor._SG_NUM).eval()
            )
            self.weightDecoders.append(
                decoderLight(mode=2, SGNum=CtxPredictor._SG_NUM).eval()
            )

            _load_model_from(
                model=self.lightEncoders[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/lightEncoder{1}_{2}.safetensors".format(
                    CtxPredictor._EXPERIMENTS_LIGHT[n],
                    n,
                    CtxPredictor._NEPOCHS_LIGHT[n] - 1,
                ),
            )
            _load_model_from(
                model=self.axisDecoders[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/axisDecoder{1}_{2}.safetensors".format(
                    CtxPredictor._EXPERIMENTS_LIGHT[n],
                    n,
                    CtxPredictor._NEPOCHS_LIGHT[n] - 1,
                ),
            )
            _load_model_from(
                model=self.lambDecoders[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/lambDecoder{1}_{2}.safetensors".format(
                    CtxPredictor._EXPERIMENTS_LIGHT[n],
                    n,
                    CtxPredictor._NEPOCHS_LIGHT[n] - 1,
                ),
            )
            _load_model_from(
                model=self.weightDecoders[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/weightDecoder{1}_{2}.safetensors".format(
                    CtxPredictor._EXPERIMENTS_LIGHT[n],
                    n,
                    CtxPredictor._NEPOCHS_LIGHT[n] - 1,
                ),
            )

            # BS network
            _load_model_from(
                model=self.albedoBSs[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/albedoBs{1}_{2}_{3}.safetensors".format(
                    CtxPredictor._EXPERIMENTS_BS[n],
                    n,
                    CtxPredictor._NEPOCHS_BS[n] - 1,
                    CtxPredictor._NITERS_BS[n],
                ),
            )
            _load_model_from(
                model=self.roughBSs[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/roughBs{1}_{2}_{3}.safetensors".format(
                    CtxPredictor._EXPERIMENTS_BS[n],
                    n,
                    CtxPredictor._NEPOCHS_BS[n] - 1,
                    CtxPredictor._NITERS_BS[n],
                ),
            )
            _load_model_from(
                model=self.depthBSs[n],
                dir_path=checkpoint_dir,
                subdirs="{0}/depthBs{1}_{2}_{3}.safetensors".format(
                    CtxPredictor._EXPERIMENTS_BS[n],
                    n,
                    CtxPredictor._NEPOCHS_BS[n] - 1,
                    CtxPredictor._NITERS_BS[n],
                ),
            )

    def _make_sure_checkpoints_available(self, checkpoint_dir: Path) -> None:
        if not checkpoint_dir.is_dir():
            VOIT_LOGGER.info(
                f"The environment estimation weights directory ({checkpoint_dir}) was not found. Downloading weights."
            )
            self._download_checkpoint_data(checkpoint_dir)

        version_data = (checkpoint_dir / "version.txt").read_text().strip()
        if version_data != "0.1":
            raise VoitError(
                f'The environment estimation weight cache has an unsupported version ("{version_data}") expected: "0.1"'
            )

    def _download_checkpoint_data(self, checkpoint_dir: Path) -> None:
        """
        Download the checkpoints data from GitHub.

        Parameters
        ----------
        checkpoint_dir
            The directory to which the checkpoints should be downloaded.

        Raises
        ------
        OSError
            If the downloading fails due to any file-related error.
        """
        checkpoint_dir.mkdir()

        checkpoints_zip_path = checkpoint_dir / "environment_estim_checkpoints.zip"
        # weights_url = "http://localhost:9000/environment_estim_checkpoints.zip"
        weights_url = "https://github.com/mntusr/voit/releases/download/v0_1/environment_estim_checkpoints.zip"
        try:
            _, msg = urllib.request.urlretrieve(
                weights_url,
                str(checkpoints_zip_path),
            )
        except:
            raise VoitError(
                f"Failed to download the weights from {weights_url}. Error: {traceback.format_exc()}"
            )
        if not checkpoints_zip_path.is_file():
            raise VoitError("Failed to load the checkpoints.")

        with zipfile.ZipFile(checkpoints_zip_path) as zip_file:
            zip_file.extractall(checkpoint_dir)

    def to(self, device: torch.device) -> None:
        for module in self._get_all_modules():
            module.to(device)

        self.__device = device

    def predict(
        self, im: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Do the non-processed inverse rendering.

        Parameters
        ----------
        im
            The input image on which the inverse rendering is done. This should be in the sRGB color data. Color data: linear RGB. Format: ``Im::RGBLike``

        Returns
        -------
        lastAlbedoBSPred
            The predicted albedos. Color data: linear RGB. Format: ``Im_RGBLike``
        lastRoughBSPred
            The predicted roughnesses. Format: ``Im_Scalar``
        lastEnvmapsPredImage
            The predicted environment maps for each pixel. Format: ``HalfEnvmapLikes``
        depthmap
            The predicted depth map for each pixel. Format: ``DepthmapLike``
        """
        im = im.astype(np.float32)
        with torch.no_grad():
            imBatches: list[torch.Tensor] = []

            # chw -> hwc
            im_cpu = im.transpose([1, 2, 0])
            nh, nw = im_cpu.shape[0], im_cpu.shape[1]
            # Resize Input Images
            newImWidth = []
            newImHeight = []
            for n in range(0, CtxPredictor._LEVEL):
                if nh < nw:
                    newW = CtxPredictor._IM_WIDTHS[n]
                    newH = int(float(CtxPredictor._IM_WIDTHS[n]) / float(nw) * nh)
                else:
                    newH = CtxPredictor._IM_HEIGHTS[n]
                    newW = int(float(CtxPredictor._IM_HEIGHTS[n]) / float(nh) * nw)

                if nh < newH:
                    im = cv2.resize(im_cpu, (newW, newH), interpolation=cv2.INTER_AREA)
                else:
                    im = cv2.resize(
                        im_cpu, (newW, newH), interpolation=cv2.INTER_LINEAR
                    )

                newImWidth.append(newW)
                newImHeight.append(newH)

                im = (np.transpose(im, [2, 0, 1]))[np.newaxis, :, :, :]
                im = im / im.max()
                imBatches.append(Variable(torch.from_numpy(im)).cuda())

            nh, nw = newImHeight[-1], newImWidth[-1]

            newEnvWidth, newEnvHeight, fov = 0, 0, 0.0
            if nh < nw:
                fov = 57.0
                newW = CtxPredictor._ENV_COL
                newH = int(float(CtxPredictor._ENV_COL) / float(nw) * nh)
            else:
                fov = 42.75
                newH = CtxPredictor._ENV_ROW
                newW = int(float(CtxPredictor._ENV_ROW) / float(nh) * nw)

            if nh < newH:
                im = cv2.resize(im_cpu, (newW, newH), interpolation=cv2.INTER_AREA)
            else:
                im = cv2.resize(im_cpu, (newW, newH), interpolation=cv2.INTER_LINEAR)

            newEnvWidth = newW
            newEnvHeight = newH

            im = (np.transpose(im, [2, 0, 1]))[np.newaxis, :, :, :]
            imBatchSmall = Variable(torch.from_numpy(im)).cuda()
            renderLayer = renderingLayer(
                imWidth=newEnvWidth,
                imHeight=newEnvHeight,
                fov=fov,
                envWidth=CtxPredictor._IBL_MAP_WIDTH,
                envHeight=CtxPredictor._IBL_MAP_HEIGHT,
            )
            renderLayer.to(self.__device)

            out_2_env = output2env(
                envWidth=CtxPredictor._IBL_MAP_WIDTH,
                envHeight=CtxPredictor._IBL_MAP_HEIGHT,
                SGNum=CtxPredictor._SG_NUM,
            )
            out_2_env.to(self.__device)

            ########################################################
            # Build the cascade network architecture #
            albedoPreds, normalPreds, roughPreds, depthPreds = [], [], [], []
            albedoBSPreds, roughBSPreds, depthBSPreds = [], [], []
            envmapsPreds, envmapsPredImages = [], []
            cAlbedos = []
            cLights = []

            ################# BRDF Prediction ######################
            inputBatch = imBatches[0]
            x1, x2, x3, x4, x5, x6 = self.encoders[0](inputBatch)

            albedoPred = 0.5 * (
                self.albedoDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6) + 1
            )
            normalPred = self.normalDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6)
            roughPred = self.roughDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6)
            depthPred = 0.5 * (
                self.depthDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6) + 1
            )

            # Normalize Albedo and depth
            bn, ch, nrow, ncol = albedoPred.size()
            albedoPred = albedoPred.view(bn, -1)
            albedoPred = (
                albedoPred
                / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1)
                / 3.0
            )
            albedoPred = albedoPred.view(bn, ch, nrow, ncol)

            bn, ch, nrow, ncol = depthPred.size()
            depthPred = depthPred.view(bn, -1)
            depthPred = (
                depthPred
                / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1)
                / 3.0
            )
            depthPred = depthPred.view(bn, ch, nrow, ncol)

            albedoPreds.append(albedoPred)
            normalPreds.append(normalPred)
            roughPreds.append(roughPred)
            depthPreds.append(depthPred)

            ################# Lighting Prediction ###################
            # Interpolation
            imBatchLarge = F.interpolate(
                imBatches[0],
                [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4],
                mode="bilinear",
            )
            albedoPredLarge = F.interpolate(
                albedoPreds[0],
                [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4],
                mode="bilinear",
            )
            normalPredLarge = F.interpolate(
                normalPreds[0],
                [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4],
                mode="bilinear",
            )
            roughPredLarge = F.interpolate(
                roughPreds[0],
                [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4],
                mode="bilinear",
            )
            depthPredLarge = F.interpolate(
                depthPreds[0],
                [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4],
                mode="bilinear",
            )

            inputBatch = torch.cat(
                [
                    imBatchLarge,
                    albedoPredLarge,
                    0.5 * (normalPredLarge + 1),
                    0.5 * (roughPredLarge + 1),
                    depthPredLarge,
                ],
                dim=1,
            )
            x1, x2, x3, x4, x5, x6 = self.lightEncoders[0](inputBatch)

            # Prediction
            axisPred: torch.Tensor = self.axisDecoders[0](
                x1, x2, x3, x4, x5, x6, imBatchSmall
            )
            lambPred: torch.Tensor = self.lambDecoders[0](
                x1, x2, x3, x4, x5, x6, imBatchSmall
            )
            weightPred: torch.Tensor = self.weightDecoders[0](
                x1, x2, x3, x4, x5, x6, imBatchSmall
            )
            bn, SGNum, _, envRow, envCol = axisPred.size()
            envmapsPred = torch.cat(
                [axisPred.view(bn, SGNum * 3, envRow, envCol), lambPred, weightPred],
                dim=1,
            )
            envmapsPreds.append(envmapsPred)

            envmapsPredImage, axisPred, lambPred, weightPred = out_2_env.output2env(
                axisPred, lambPred, weightPred
            )
            envmapsPredImages.append(envmapsPredImage)

            diffusePred, specularPred = renderLayer.forwardEnv(
                albedoPreds[0], normalPreds[0], roughPreds[0], envmapsPredImages[0]
            )

            diffusePredNew, specularPredNew = LSregressDiffSpec(
                diffusePred, specularPred, imBatchSmall, diffusePred, specularPred
            )
            renderedPred = diffusePredNew + specularPredNew

            cDiff, cSpec = (
                torch.sum(diffusePredNew) / torch.sum(diffusePred)
            ).data.item(), (
                (torch.sum(specularPredNew)) / (torch.sum(specularPred))
            ).data.item()
            if cSpec < 1e-3:
                cAlbedo = 1 / albedoPreds[-1].max().data.item()
                cLight = cDiff / cAlbedo
            else:
                cLight = cSpec
                cAlbedo = cDiff / cLight
                cAlbedo = np.clip(cAlbedo, 1e-3, 1 / albedoPreds[-1].max().data.item())
                cLight = cDiff / cAlbedo
            envmapsPredImages[0] = envmapsPredImages[0] * cLight
            cAlbedos.append(cAlbedo)
            cLights.append(cLight)

            diffusePred = diffusePredNew
            specularPred = specularPredNew

            #################### BRDF Prediction ####################
            albedoPredLarge = F.interpolate(
                albedoPreds[0], [newImHeight[1], newImWidth[1]], mode="bilinear"
            )
            normalPredLarge = F.interpolate(
                normalPreds[0], [newImHeight[1], newImWidth[1]], mode="bilinear"
            )
            roughPredLarge = F.interpolate(
                roughPreds[0], [newImHeight[1], newImWidth[1]], mode="bilinear"
            )
            depthPredLarge = F.interpolate(
                depthPreds[0], [newImHeight[1], newImWidth[1]], mode="bilinear"
            )

            diffusePredLarge = F.interpolate(
                diffusePred, [newImHeight[1], newImWidth[1]], mode="bilinear"
            )
            specularPredLarge = F.interpolate(
                specularPred, [newImHeight[1], newImWidth[1]], mode="bilinear"
            )

            inputBatch = torch.cat(
                [
                    imBatches[1],
                    albedoPredLarge,
                    0.5 * (normalPredLarge + 1),
                    0.5 * (roughPredLarge + 1),
                    depthPredLarge,
                    diffusePredLarge,
                    specularPredLarge,
                ],
                dim=1,
            )

            x1, x2, x3, x4, x5, x6 = self.encoders[1](inputBatch)
            albedoPred: torch.Tensor = 0.5 * (
                self.albedoDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6) + 1
            )
            normalPred = self.normalDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6)
            roughPred = self.roughDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6)
            depthPred: torch.Tensor = 0.5 * (
                self.depthDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6) + 1
            )

            # Normalize Albedo and depth
            bn, ch, nrow, ncol = albedoPred.size()
            albedoPred = albedoPred.view(bn, -1)
            albedoPred = (
                albedoPred
                / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1)
                / 3.0
            )
            albedoPred = albedoPred.view(bn, ch, nrow, ncol)

            bn, ch, nrow, ncol = depthPred.size()
            depthPred = depthPred.view(bn, -1)
            depthPred = (
                depthPred
                / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1)
                / 3.0
            )
            depthPred = depthPred.view(bn, ch, nrow, ncol)

            albedoPreds.append(albedoPred)
            normalPreds.append(normalPred)
            roughPreds.append(roughPred)
            depthPreds.append(depthPred)

            ############### Lighting Prediction ######################
            # Interpolation
            imBatchLarge = F.interpolate(
                imBatches[1],
                [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4],
                mode="bilinear",
            )
            albedoPredLarge = F.interpolate(
                albedoPreds[1],
                [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4],
                mode="bilinear",
            )
            normalPredLarge = F.interpolate(
                normalPreds[1],
                [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4],
                mode="bilinear",
            )
            roughPredLarge = F.interpolate(
                roughPreds[1],
                [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4],
                mode="bilinear",
            )
            depthPredLarge = F.interpolate(
                depthPreds[1],
                [imBatchSmall.size(2) * 4, imBatchSmall.size(3) * 4],
                mode="bilinear",
            )

            inputBatch = torch.cat(
                [
                    imBatchLarge,
                    albedoPredLarge,
                    0.5 * (normalPredLarge + 1),
                    0.5 * (roughPredLarge + 1),
                    depthPredLarge,
                ],
                dim=1,
            )
            x1, x2, x3, x4, x5, x6 = self.lightEncoders[1](inputBatch, envmapsPred)

            # Prediction
            axisPred = self.axisDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall)
            lambPred = self.lambDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall)
            weightPred = self.weightDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall)
            bn, SGNum, _, envRow, envCol = axisPred.size()
            envmapsPred = torch.cat(
                [axisPred.view(bn, SGNum * 3, envRow, envCol), lambPred, weightPred],
                dim=1,
            )
            envmapsPreds.append(envmapsPred)

            envmapsPredImage, axisPred, lambPred, weightPred = out_2_env.output2env(
                axisPred, lambPred, weightPred
            )
            envmapsPredImages.append(envmapsPredImage)

            diffusePred, specularPred = renderLayer.forwardEnv(
                albedoPreds[1], normalPreds[1], roughPreds[1], envmapsPredImages[1]
            )

            diffusePredNew, specularPredNew = LSregressDiffSpec(
                diffusePred, specularPred, imBatchSmall, diffusePred, specularPred
            )

            cDiff, cSpec = (
                torch.sum(diffusePredNew) / torch.sum(diffusePred)
            ).data.item(), (
                (torch.sum(specularPredNew)) / (torch.sum(specularPred))
            ).data.item()
            if cSpec == 0:
                cAlbedo = 1 / albedoPreds[-1].max().data.item()
                cLight = cDiff / cAlbedo
            else:
                cLight = cSpec
                cAlbedo = cDiff / cLight
                cAlbedo = np.clip(cAlbedo, 1e-3, 1 / albedoPreds[-1].max().data.item())
                cLight = cDiff / cAlbedo
            envmapsPredImages[-1] = envmapsPredImages[-1] * cLight
            cAlbedos.append(cAlbedo)
            cLights.append(cLight)

            diffusePred = diffusePredNew
            specularPred = specularPredNew

            #################### BilateralLayer ######################
            for n in range(0, CtxPredictor._LEVEL):
                albedoBSPred, albedoConf = self.albedoBSs[n](
                    imBatches[n], albedoPreds[n].detach(), albedoPreds[n]
                )
                albedoBSPreds.append(albedoBSPred)
                roughBSPred, roughConf = self.roughBSs[n](
                    imBatches[n], albedoPreds[n].detach(), 0.5 * (roughPreds[n] + 1)
                )
                roughBSPred = torch.clamp(2 * roughBSPred - 1, -1, 1)
                roughBSPreds.append(roughBSPred)
                depthBSPred, depthConf = self.depthBSs[n](
                    imBatches[n], albedoPreds[n].detach(), depthPreds[n]
                )
                depthBSPreds.append(depthBSPred)

            #################### Output Results #######################
            # Save the albedo (albedoPreds)

            # Save the normal (normalPreds)

            # Save the rough (roughPreds)

            # Save the depth (depthPreds)

            # Save the albedo bs (albedoBSPreds)

            # Save the rough bs (roughBSPreds)

            # Save the depth bs (depthBSPreds)

            # Save the envmapImages (envmapsPredImages, envmapsPreds, cLights)

            lastAlbedoBSPred: np.ndarray = (
                (albedoBSPreds[-1][0] * cAlbedos[-1]).cpu().numpy()
            )
            lastRoughBSPred: np.ndarray = (
                (0.5 * (roughBSPreds[-1][0] + 1)).cpu().numpy()
            )
            lastEnvmapsPredImage: np.ndarray = (
                envmapsPredImages[-1].cpu().numpy().squeeze()
            )
            lastEnvmapsPredImage = lastEnvmapsPredImage.transpose([1, 2, 3, 4, 0])
            lastDepthPred = np.expand_dims(
                depthPreds[-1].data.cpu().numpy().squeeze(), axis=0
            )

            return (
                lastAlbedoBSPred,
                lastRoughBSPred,
                lastEnvmapsPredImage,
                lastDepthPred,
            )


def _load_model_from(model: nn.Module, dir_path: Path, subdirs: str) -> None:
    total_path = str(dir_path / subdirs)
    load_model(model=model, filename=total_path, strict=False)
