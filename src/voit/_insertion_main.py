import copy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Final, Literal, TypedDict

import cv2 as cv
import depth_tools
import numpy as np
import scipy.ndimage
import torch
from panda3d.core import NodePath, PandaNode
from typing_extensions import NotRequired

from ._analytical_light_estimation import (
    LightConf,
    extract_analytical_light_from_envmap,
)
from ._ctx_estimation_internal import CtxPredictor
from ._errors import VoitError
from ._logging_internal import VOIT_LOGGER
from ._math_internal import (
    get_corresponding_z_for_transformed_local_y,
    linear_2_srgb,
    px_2_vs_unchecked,
    srgb_2_linear,
)
from ._render_pipeline_internal import RasterizingPipeline
from ._vectors import Vec2, Vec2i, Vec3


class Inserter:
    """
    This is the main class of the object insertion.

    Parameters
    ----------
    camera
        The description of the camera intrinsics.
    floor_proxy_size
        The size of the floor proxy rectangle mesh.
    pt_device
        The Pytorch device used by the inverse rendering neural networks.
    debug_window
        If this is true, then the GPU-based part of the internal rendering will not be offscreen. It will appear on a window instead. This is useful for debugging purposes.
    ctx_estim_weights
        The directory to which the weights of the lighting estimation weights are downloaded (if they do not exist).

    Raises
    ------
    TBD
    """

    def __init__(
        self,
        /,
        camera: depth_tools.CameraIntrinsics,
        floor_proxy_size: Vec2,
        im_size: Vec2i,
        pt_device: torch.device,
        debug_window: bool = False,
        lighting_estim_weights: Path | None = None,
    ):
        if lighting_estim_weights is None:
            lighting_estim_weights = Path("./voit_ctx_estim_weights")
        self.render_pipeline = RasterizingPipeline(
            camera=camera,
            floor_proxy_size=floor_proxy_size,
            im_size=im_size,
            shadow_map_size=1024,
            near_far=(0.1, 1000),
            debug_window=debug_window,
        )
        self.camera: Final = camera
        self._ctx_predictor = CtxPredictor()
        self._ctx_predictor.load_checkpoint(lighting_estim_weights)
        self._ctx_predictor.to(pt_device)
        self.im_size: Final = im_size
        """
        The size of the modified images.
        """

        self._cache: "dict[str, _CachedEnvironmentAtPoint]" = dict()
        """
        The dict that maps the baked environments to their names.
        """

    def _calculate_environment_at(
        self, at: "EnvSpecParam", defensive_copy: bool
    ) -> "_CachedEnvironmentAtPoint":
        input_im = at["input_im"]
        input_depth = at["input_depth"]
        pos_px = at["pos_px"]
        pos_depth = at["pos_depth"]

        # argument validation
        if input_im.shape != (3, self.im_size.y, self.im_size.x):
            raise VoitError(
                f"The shape of the array containing the input image is not {(3, self.im_size.y, self.im_size.x)}."
            )
        if not np.issubdtype(input_im.dtype, np.floating):
            raise VoitError(
                f"The array containing the input image does not contain floating point data. Its dtype is {input_im.dtype}."
            )
        if input_depth is not None:
            input_depth_depth = input_depth["depth"]
            input_depth_mask = input_depth["mask"]
            if input_depth_depth.shape != (1, self.im_size.y, self.im_size.x):
                raise VoitError(
                    f"The shape of the array containing the input depth values is not {(3, self.im_size.y, self.im_size.x)}."
                )
            if not np.issubdtype(input_depth_depth.dtype, np.floating):
                raise VoitError(
                    f"The array containing the input depth values does not contain floating point data. Its dtype is {input_depth_depth.dtype}."
                )
            if input_depth_mask.shape != (1, self.im_size.y, self.im_size.x):
                raise VoitError(
                    f"The shape of the array containing the input depth mask is not {(3, self.im_size.y, self.im_size.x)}."
                )
            if not np.issubdtype(input_depth_mask.dtype, np.bool_):
                raise VoitError(
                    f"The array containing the input depth mask does not contain boolean data. Its dtype is {input_depth_mask.dtype}."
                )

        if not ((0 <= pos_px.x < self.im_size.x) and (0 <= pos_px.y < self.im_size.y)):
            raise VoitError(
                f"The pixel {pos_px} is outside of the image. Image size: {self.im_size}."
            )

        if pos_depth < 0:
            VOIT_LOGGER.warning(
                "The depth to which the object is inserted is negative."
            )
        if not at["input_im_linear"]:
            input_im_linear = srgb_2_linear(input_im)
        else:
            input_im_linear = input_im

        # calculate all the inputs of the rasterizing renderer
        pos_vs = px_2_vs_unchecked(
            camera=self.camera,
            depth=pos_depth,
            pos_px=pos_px,
            im_size=Vec2i(input_im.shape[2], input_im.shape[1]),
        )

        bg_im_base_color_map, bg_im_metallic_roughness_map, envmap, occl_depthmap = (
            self._estimate_textures_and_lighting_unchecked(
                input_im=input_im_linear,
                lighting_pixel_pos_px=pos_px,
                lighting_pixel_pos_vs=pos_vs,
            )
        )
        lighting = extract_analytical_light_from_envmap(envmap=envmap)
        if not defensive_copy:
            return _CachedEnvironmentAtPoint(
                bg_im_base_color_map=bg_im_base_color_map,
                bg_im_metallic_roughness_map=bg_im_metallic_roughness_map,
                input_im_linear=input_im_linear,
                light_conf=lighting,
                occl_depthmap=occl_depthmap,
                pos_vs=pos_vs,
                input_depth=input_depth,
            )
        else:
            return _CachedEnvironmentAtPoint(
                bg_im_base_color_map=bg_im_base_color_map.copy(),
                bg_im_metallic_roughness_map=bg_im_metallic_roughness_map.copy(),
                input_im_linear=input_im_linear.copy(),
                light_conf=lighting,
                occl_depthmap=occl_depthmap.copy(),
                pos_vs=pos_vs,
                input_depth=copy.deepcopy(input_depth),
            )

    def bake(self, /, name: str, at: "EnvSpecParam") -> None:
        """
        Bake the environment information at the given location on the given image.

        This speeds up insertion, since the inverse rendering part is skipped in this case.

        Parameters
        ----------
        name
            The name of the cached environment.
        at
            The environment to cache.
        """
        env = self._calculate_environment_at(at=at, defensive_copy=True)
        self._cache[name] = env
        self._cache_names = frozenset(self._cache.keys())

    @property
    def cache_names(self) -> frozenset[str]:
        """
        Get the names of the cached environments.
        """
        return self._cache_names

    def insert(
        self,
        /,
        at: "str | EnvSpecParam",
        output_im_linear: bool,
        normal_vs: Vec3,
        obj: NodePath,
        depth_occlusion: "_DepthOcclusion | None" = None,
        rotation_around_floor_normal_cw: float = 0.0,
    ) -> "InsertionResult":
        """
        Do the object insertion.

        By default, the pre-existing objects in the scene do not occlude the inserted object. However, you can configure then to do so, using the ``depth_occlusion`` argument.

        The resulting depth map is created for each pixel using the following rules:

        * If the inserted object appears on a non-masked pixel, then its depth is used. This does not depend on whether an original depth map is given or what its mask is.
        * If the inserted object does not appear on a non-masked pixel, then the original depth is used, regardless of whether the floor proxy mesh appears on that pixel or not.

        The inserter can handle if the transfer function of the input image is sRGB instead of linear. In this case, the inserter uses the ``x**(2.2)`` formula to do the conversion.

        The inserter is able to produce the output image with sRGB trasnfer function. In this case, the inserter uses the ``x**(1/2.2)`` formula to do the conversion for the non-zero pixels.

        The ``at`` parameter describes where the object should be inserted. If it is a dictionary, then the keys are the following:

        * ``input_im``: The image to which the object should be inserted. Format: ``Im_RGBLike``
        * ``input_im_linear``: If this is True, then the transfer function of the input image is assumed to be linear. Otherwise it is assumed to be sRGB.
        * ``pos_px``: The pixe to which the origin of the inserted object should be placed.
        * ``pos_depth``: The depth value for the current pixel, specified at ``pos_depth``.
        * ``input_depth``: The dict that specifies the sparse input depth map. Keys: ``depth``: the depth values, ``mask``: the mask that selects the valid depth values.

        If the ``at`` parameter is a string, then it gives the name of the cached environment to which the object should be inserted. You can cache an environment using the `Inserter.bake` function.

        Parameters
        ----------
        at
            This is where the object should be inserted. For its detailed usage, see the description of the function.
        output_im_linear
            This variable specifies whether the output image should contain linear or sRGB data. This does not affect the depth data.
        normal_vs
            The normal vector of the floor at the given point in the view space.
        obj
            The object to insert.
        depth_occlusion
            If this argument is not None, then the feature, depth occlusion is enabled. This feature makes sure that the nearer objects occlude the inserted object and its reflections and shadows. Keys: ``dense_depth``: the dense depth map for this depth testing; ``threshold``: if the depth difference is less than this threshold, then the occlusion is not applied.

        Returns
        -------
        new_im
            The resulting RGB image of the compositing. Format: ``Im::RGBLike``
        new_depth
            The resulting depth map of the compositing. Format: ``Im::Scalar``
        obj_mask
            The mask that selects the pixels of the inserted object (but not the floor proxy mesh) on ``new_im``. Format: ``Im::Mask``

        Raises
        ------
        VoitError
            If any of the maps do not have the correct shape or dtype.
        """
        if isinstance(at, str):
            environment = self._cache[at]
        else:
            environment = self._calculate_environment_at(at=at, defensive_copy=False)

        if depth_occlusion is not None:
            if "dense_depth" in depth_occlusion.keys():
                if depth_occlusion["dense_depth"].shape != (  # type: ignore
                    1,
                    self.im_size.y,
                    self.im_size.x,
                ):
                    raise VoitError(
                        f"The shape of the array containing the depth values for the depth occlusion is not {(1, self.im_size.y, self.im_size.x)}."
                    )
                dense_depth_dtype = depth_occlusion["dense_depth"].dtype  # type: ignore
                if not np.issubdtype(dense_depth_dtype, np.floating):
                    raise VoitError(
                        f"The array containing the depth values for the depth occlusion does not contain floating point data. Its dtype is {dense_depth_dtype}."
                    )

        if depth_occlusion is not None:
            if "dense_depth" not in depth_occlusion.keys():
                depth_occlusion = depth_occlusion | {"dense_depth": environment.occl_depthmap}  # type: ignore

        # the transformed local coordinate system of the floor is chosen to be the same as
        # the transformed local coordinate system of the environment map
        floor_z_in_view_space = get_corresponding_z_for_transformed_local_y(normal_vs)

        self.render_pipeline.configure_scene(
            obj=obj,
            bg_im_base_color_map=environment.bg_im_base_color_map,
            bg_im_metallic_roughness=environment.bg_im_metallic_roughness_map,
            floor_pos_in_view_space=environment.pos_vs,
            floor_z_in_view_space=floor_z_in_view_space,
            floor_y_in_view_space=normal_vs,
            envmap_z_in_view_space=floor_z_in_view_space,
            envmap_y_in_view_space=normal_vs,
            light_conf=environment.light_conf,
            rotation_around_floor_normal_cw=rotation_around_floor_normal_cw,
        )

        # render
        floor_rendered_rgb, floor_rendered_depth, floor_mask = (
            self.render_pipeline.render_floor()
        )
        full_rendered_rgb, full_rendered_depth, floor_and_obj_mask = (
            self.render_pipeline.render_floor_and_obj()
        )

        # do the compositing
        result = self._composite_unchecked(
            floor_rendered_rgb=floor_rendered_rgb,
            original_rgb=environment.input_im_linear,
            input_depth=environment.input_depth,
            floor_mask=floor_mask,
            full_rendered_rgb=full_rendered_rgb,
            full_rendered_depth=full_rendered_depth,
            floor_and_obj_mask=floor_and_obj_mask,
            depth_occlusion=depth_occlusion,
            obj_mask=floor_rendered_depth > full_rendered_depth,
        )

        if not output_im_linear:
            positive_pixel_mask = result.im > 0
            result.im[positive_pixel_mask] = linear_2_srgb(
                result.im[positive_pixel_mask]
            )

        return result

    def _estimate_textures_and_lighting_unchecked(
        self,
        input_im: np.ndarray,
        lighting_pixel_pos_px: Vec2i,
        lighting_pixel_pos_vs: Vec3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the materials and the lighting of the visible image. The estimation algorithm assumes completely dielectric materials.

        This function *does not* validate its arguments.

        Parameters
        ----------
        input_im
            The visible image. Expected color data: linear RGB. Format: ``Im::RGBLike``
        lighting_pixel_pos
            The position of the pixel of which the lighting should be estimated.

        Returns
        -------
        base_color
            The base color for each pixel. It has the same size as the original image. Color data: linear RGB. Format: ``Im::RGBLike``
        metallic_rough
            The metallic-roughness value for each pixel. It has the same size as the original image. Format: ``Im::RGBLike``
        envmap
            The environment map for the specified pixel. Format: ``Im::RGBEnvmapLike[X=1024, Y=512]``
        depth_map
            The predicted scale invariant depth map for the specified pixel. Format: ``DepthmapLike``
        """
        if self._ctx_predictor is None:
            raise VoitError("The inserter is already destroyed.")

        original_im_size = Vec2i(input_im.shape[2], input_im.shape[1])
        albedo, rough, half_envmaps, depthmap = self._ctx_predictor.predict(input_im)
        base_color = _diffuse_color_2_base_color(albedo)
        base_color = _scale_image_unchecked(
            base_color, original_im_size, interpolation="linear"
        )
        rough = _scale_image_unchecked(rough, original_im_size, interpolation="linear")
        metallic_rough = _roughness_2_metallic_roughness(rough)
        depthmap = _scale_image_unchecked(
            depthmap, original_im_size, interpolation="nearest"
        )

        # rescale the depth map
        depthmap = (
            depthmap
            * lighting_pixel_pos_vs.z
            / depthmap[:, lighting_pixel_pos_px.y, lighting_pixel_pos_px.x]
        )

        envmap = _get_envmap(
            half_envmaps=half_envmaps,
            original_im_size=Vec2i(input_im.shape[2], input_im.shape[1]),
            original_px_pos=lighting_pixel_pos_px,
        )
        return base_color, metallic_rough, envmap, depthmap

    def _composite_unchecked(
        self,
        original_rgb: np.ndarray,
        input_depth: "InputDepthParam | None",
        floor_rendered_rgb: np.ndarray,
        floor_mask: np.ndarray,
        obj_mask: np.ndarray,
        full_rendered_rgb: np.ndarray,
        full_rendered_depth: np.ndarray,
        floor_and_obj_mask: np.ndarray,
        depth_occlusion: "_DepthOcclusion | None",
    ) -> "InsertionResult":
        """
        Do the compositing.

        This function *does not* validate its arguments.

        Parameters
        ----------
        original_rgb
            The image to which the insertion should be done. Format: ``Im_RGBLike``
        original_depth
            The depth map of the original image. Format: ``DepthmapLike``
        floor_rendered_rgb
            The rendered image of the floor, without the inserted object. ``Im_RGBLike``
        floor_mask
            The mask that selects the non-empty pixels on ``floor_rendered_rgb``. Format: ``Im_Mask``
        obj_mask
            The mask that selects the pixels that belong to the object. Format: ``Im_Mask``
        full_rendered_rgb
            The RGB rgb image that conatins both the floor proxy mesh and the inserted object. Format: ``Im_RGBLike``
        full_rendered_depth
            The depth map for ``full_rendered_rgb``. Format: ``DepthmapLike``
        floor_and_obj_mask
            The mask that selects the non-empty pixels on ``full_rendered_rgb``.
        depth_occlusion
            If this argument is not None, then the feature, depth occlusion is enabled. This feature makes sure that the nearer objects occlude the inserted object and its reflections and shadows. Keys: "dense_depth": the dense depth map for this depth testing; "threshold": if the depth difference is less than this threshold, then the occlusion is not applied.

        Returns
        -------
        v
            The result of the insertion.
        """
        floor_diff = full_rendered_rgb - floor_rendered_rgb

        eroded_full_mask = _binary_erode_mask_disk2(floor_and_obj_mask)

        if depth_occlusion is not None:
            assert "dense_depth" in depth_occlusion.keys()
            dense_depth = depth_occlusion["dense_depth"]  # type: ignore
            rendered_depth_based_occl_mask = ~(
                dense_depth - full_rendered_depth < -depth_occlusion["threshold"]
            )
            occluded_obj_mask = rendered_depth_based_occl_mask & obj_mask
            occluded_eroded_full_mask = (
                rendered_depth_based_occl_mask & eroded_full_mask
            )
        else:
            occluded_obj_mask = obj_mask
            occluded_eroded_full_mask = eroded_full_mask

        # plt.imshow((original_rgb).transpose([1, 2, 0]))
        # plt.title("original_rgb")
        # plt.show(block=True)
        # plt.close()
        # plt.imshow(abs(floor_diff).transpose([1, 2, 0]))
        # plt.title("abs(floor_diff)")
        # plt.show(block=True)
        # plt.close()
        # plt.imshow((original_rgb).transpose([1, 2, 0]))
        # plt.title("original_rgb")
        # plt.show(block=True)
        # plt.close()

        new_im = (
            original_rgb * (~occluded_eroded_full_mask)
            + (original_rgb + floor_diff) * occluded_eroded_full_mask
        )
        new_im = full_rendered_rgb * occluded_obj_mask + new_im * (~occluded_obj_mask)

        if input_depth is not None:
            new_depth = (
                input_depth["depth"] * (~occluded_obj_mask)
                + full_rendered_depth * occluded_obj_mask
            )
            new_mask = occluded_obj_mask | input_depth["mask"]
        else:
            new_depth = full_rendered_depth * occluded_obj_mask
            new_mask = occluded_obj_mask

        new_im = np.clip(new_im, 0, 1)

        return InsertionResult(
            im=new_im, depth=new_depth, depth_mask=new_mask, obj_mask=occluded_obj_mask
        )

    def load_model(self, model_path: Path) -> NodePath:
        """
        Load a model from the give file.

        Parameters
        ----------
        model_path
            The path of the model to load.

        Returns
        -------
        v
            The Panda3d object that contains the loaded model.
        """
        return self.render_pipeline.load_model(model_path)

    def destroy(self) -> None:
        del self._ctx_predictor
        self._ctx_predictor = None
        self.render_pipeline.destroy()


def _get_vs_pos_for_px_unchecked(
    t_proj_mat: np.ndarray, pos_px: Vec2i, im_size: Vec2i, pos_depth: float
) -> Vec3:
    """
    Calculate the view space position for a pixel with a depth.

    This function *does not* check its arguments.

    Parameters
    ----------
    t_proj_mat
        The theoretical projection matrix. Format: ``TProjMat``
    px
        The position of the pixel.
    im_size
        The size of the image.
    depth
        The depth value for the pixel.

    Returns
    -------
    v
        The calculated view space position.
    """
    pos_ndc = np.array(
        [
            [pos_px.x / im_size.x * 2 - 1],
            [1 - pos_px.y / im_size.y * 2],
            [1],
        ],
        dtype=np.float32,
    )

    pos_clip = pos_ndc * pos_depth
    t_proj_mat_inv = np.linalg.inv(t_proj_mat)

    pos_vs = t_proj_mat_inv @ pos_clip

    return Vec3.from_npy_col_vec(pos_vs)


class _DepthOcclusion(TypedDict):
    threshold: float
    dense_depth: NotRequired[np.ndarray]


@dataclass
class InsertionResult:
    im: np.ndarray
    """
    The resulting image. Format: ``Im_RGBLike``
    """

    obj_mask: np.ndarray
    """
    The mask that selects the inserted object. Format: ``Im_Mask``
    """

    depth: np.ndarray
    """
    The depth values of the sparse depth map created during insertion. Format: ``DepthmapLike``
    """

    depth_mask: np.ndarray
    """
    The mask that selects the valid depth values. Format: ``Im_Mask``
    """


class InputDepthParam(TypedDict):
    depth: np.ndarray
    mask: np.ndarray


def _get_envmap(
    half_envmaps: np.ndarray, original_im_size: Vec2i, original_px_pos: Vec2i
) -> np.ndarray:
    """
    Get an environment map from the array that specifies a half environment map for each pixel.

    Parameters
    ----------
    half_envmaps
        The array that specifies a half environment map for each pixel of a scaled version of an image. Format: ``PerPixelHalfEnvmaps::RGBLike``
    original_im_size
        The size of the original image.
    original_px_pos
        The position of the relevant pixel in the original image.
    envmap_px_pos
        The position of the relevant pixel.

    Return
    ------
    v
        The loaded environment map. Format: ``Im::RGBEnvmapLike[W=1024, H=512]``

    Raises
    ------
    VoitError
        If the specified size of the original image is not positive in all dimensions.
        If the specified position of the relevant pixel is negative in at least one dimension.
    """
    if not original_im_size.is_positive():
        raise VoitError(
            f"The specified size of the original image is not positive in all dimensions. Original image size: {original_im_size}"
        )
    if (
        (original_px_pos.x < 0)
        or (original_px_pos.y < 0)
        or (original_px_pos.x >= original_im_size.x)
        or (original_px_pos.y >= original_im_size.y)
    ):
        raise VoitError(
            f"The specified position of the relevant pixel is outside of the orignal image. Specified position: {original_px_pos}; original image size: {original_im_size}"
        )
    half_envmap_map_width = half_envmaps.shape[1]
    half_envmap_map_height = half_envmaps.shape[0]

    envmap_px_x = int(original_px_pos.x / original_im_size.x * half_envmap_map_width)
    envmap_px_y = int(original_px_pos.y / original_im_size.y * half_envmap_map_height)

    half_envmap: np.ndarray = half_envmaps[envmap_px_y, envmap_px_x].transpose(
        [2, 0, 1]
    )
    half_envmap = _scale_image_unchecked(
        half_envmap, Vec2i(1024, 256), interpolation="linear"
    )
    return np.concatenate([half_envmap, np.zeros_like(half_envmap)], axis=1)


def _roughness_2_metallic_roughness(rough: np.ndarray) -> np.ndarray:
    """
    Convert a map that contains only roughness values to a metallic-roughness map.

    The occlusion values will be 1.

    Parameters
    ----------
    rough
        The roughness map. Format: ``Im::Scalar[$float32]``

    Returns
    -------
    v
        The metallic-roughness map. Format: ``Im::RGBLike``

    Raises
    ------
    VoitError
        If the shape specified roughness is not an ``Im::Scalar[$float32]``.
    """
    if len(rough.shape) != 3:
        raise VoitError(
            f"The specified roughness array does not contain a valid image, because it is not three-dimensional. Shape: {rough.shape}"
        )
    if rough.shape[0] != 1:
        raise VoitError(
            f"The specified roughness image has more than one channels. Shape: {rough.shape}"
        )
    if not np.issubdtype(rough.dtype, np.floating):
        raise VoitError(
            f"The dtype of the specified roughness image is not a subdtype of float."
        )

    width = rough.shape[2]
    height = rough.shape[1]
    return np.concatenate(
        [
            np.ones((1, height, width), dtype=np.float32),
            rough,
            np.zeros((1, height, width), dtype=np.float32),
        ],
        axis=0,
    )


def _diffuse_color_2_base_color(albedo: np.ndarray) -> np.ndarray:
    """
    Calculate the GLTF-like base color from the given diffuse color for completely non-metallic materials.

    Unlike in GLTF, the base color is in linear space and uses float32 instead of uint8.

    Key assumptions about the material:

    * ``metallic = 0``
    * ``F0 = 0.04`` per GLTF specification

    Parameters
    ----------
    albedo
        A texture specifying the diffuse colors. Format: ``Im::RGBLike``

    Returns
    -------
    v
        The texture specifying the equivalent base color. Format: ``Im::RGBLike``
    """
    # a = (b*(1-F0))*(1-m)

    F0 = 0.04
    m = 0.0
    a = albedo
    base_color_linear = a / (F0 * m - F0 - m + 1)
    return base_color_linear


def _binary_erode_mask_disk2(mask: np.ndarray) -> np.ndarray:
    """
    Apply the erosion operation on the given mask with a strucutre with the following structure::

       np.array([
            [False, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
            [False, True, True, False],
       ], dtype=np.bool_)

    This function internally uses `scipy.ndimage.binary_erosion` with the mentioned structure and ``iterations=1``.

    Parameters
    ----------
    mask
        The specified mask. Format: ``Im::Mask``

    Returns
    -------
    v
        The result of the erosion. Format: ``Im::Mask``
    """
    if not np.issubdtype(mask.dtype, np.bool_):
        raise VoitError(
            f"The dtype of the mask array ({mask.dtype}) is not a subdtype of {np.bool_}."
        )

    if len(mask.shape) != 3 or (mask.shape[0] != 1):
        raise VoitError(
            f"The shape of the mask array is not valid, because it does not belong to an image. Array shape: {mask.shape}"
        )
    mask = mask[0].astype(np.uint8)
    structure = np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    eroded_mask = scipy.ndimage.binary_erosion(mask, structure=structure, iterations=1)
    eroded_mask = np.expand_dims(eroded_mask, axis=0)
    return eroded_mask


class EnvSpecParam(TypedDict):
    input_im: np.ndarray
    input_depth: InputDepthParam | None
    input_im_linear: bool
    pos_px: Vec2i
    pos_depth: float


@dataclass
class _CachedEnvironmentAtPoint:
    pos_vs: Vec3
    """
    The view space position of the inserted object.
    """

    light_conf: LightConf
    """
    The configuration of the lighting of the scene.
    """

    input_im_linear: np.ndarray
    """
    The input image. Transfer function: linear. Format: ``Im_RGBLike``
    """

    bg_im_base_color_map: np.ndarray
    """
    The base color map of the given input image. Format: ``Im_RGBLike``
    """

    bg_im_metallic_roughness_map: np.ndarray
    """
    The metallic-roughness map of the input image. Transfer function: linear. Format: ``Im_RGBLike``
    """

    occl_depthmap: np.ndarray
    """
    The dense depth map used for depth occlusion. Format: ``Im_Scalar``
    """

    input_depth: InputDepthParam | None
    """
    The input depth map and the corresponding mask.
    """


def _scale_image_unchecked(
    im: np.ndarray, target_size: Vec2i, interpolation: Literal["linear", "nearest"]
) -> np.ndarray:
    """
    Scale an image to the given size using linear interpolation.

    Parameters
    ----------
    im
        The specified image. Format: ``Im`` with floating point type.
    target_size
        The new image size.

    Returns
    -------
    v
        The scaled image.
    """
    match interpolation:
        case "linear":
            cv_interp = cv.INTER_LINEAR
        case "nearest":
            cv_interp = cv.INTER_NEAREST

    if im.shape[0] == 1:
        result = im[0]
    else:
        result = im.transpose([1, 2, 0])
    result = cv.resize(result, (target_size.x, target_size.y), interpolation=cv_interp)

    if im.shape[0] == 1:
        result = np.expand_dims(result, axis=0)
    else:
        result = result.transpose([2, 0, 1])
    return result
