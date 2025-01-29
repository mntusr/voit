import copy
import json
import operator
import re
import tkinter as tk
import tkinter.messagebox
import webbrowser
from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    SupportsIndex,
    TypedDict,
    TypeVar,
)

import cv2
import depth_tools
import numpy as np
import torch
from PIL import Image, ImageTk
from typing_extensions import TypeVar

from .._errors import VoitError
from .._insertion_main import InputDepthParam, Inserter, InsertionResult
from .._normal_estimation import get_pos_normal_from_3_points
from .._vectors import Vec2, Vec2i, Vec3


def run_editor(
    *,
    editor_inserter: "DatasetBasedInserter",
    max_displayed_points: int | None = None,
    jump_next: bool = False,
    fig_tmp_path: Path | None = None,
    on_apply_hook: "Callable[[list[InsertionSpec]], str | None] | None" = None,
    insertion_specs: "list[InsertionSpec] | None" = None,
) -> "list[InsertionSpec]":
    """
    Run a GUI-based editor to interactively author a list of `InsertionSpec`-s.

    Parameters
    ----------
    editor_inserter
        The inserter used by the interactive editor.
    original_samples
        The object that describes the original samples.
    max_displayed_points
        The maximal number of displayed points in the depth point cloud viewer.
    jump_next
        If true, then the editor jumps to the next samle when the user clicks the "apply" button. Otherwise the editor jumps to a new sample.
    fig_tmp_path
        The path of the temporary directory to which the HTML file containing the displayed point cloud is written when the user wants to view the depth values as point cloud.
    on_apply_hook
        An additional hook to be run when the user clicks the "apply" button. It can, for example save the data. If the hook returns with a string, it is displayed as an error message.
    insertion_specs
        The list of the initial insertion specifications, for example loaded from a file.

    Returns
    -------
    v
        The created insertion configurations.
    """
    gui = EditorGUI(
        editor_inserter=editor_inserter,
        jump_next=jump_next,
        fig_tmp_path=fig_tmp_path,
        insertion_specs=insertion_specs,
        max_displayed_points=max_displayed_points,
        on_apply_hook=on_apply_hook,
    )
    gui.run()
    return gui.get_insertion_specs()


class DatasetWithObjectsInserted:
    """
    A dataset that contains images that contain inserted objects.

    The format in which the dataset stores the results with objects inserted in the cache dir is an implementation detail and might change between versions.

    Parameters
    ----------
    cache_dir
        The directory that contains the generated dataset cache.

    Raises
    ------
    IOError
        If the cache directory does not contain some of the necessary files or they are not readable.
    json.JSONDecodeError
        If the metadata file does not contain valid json.
    """

    def __init__(
        self,
        cache_dir: Path,
    ) -> None:
        self.cache_dir: Final = cache_dir

        metadata = json.loads((self.cache_dir / "metadata.json").read_text())

        self._name_lookup_table = metadata["original_names"]
        self._index_lookup_table = metadata["original_indices"]
        self._camera_params = np.load(self.cache_dir / "camera_params.npy")

        self.original_indices: Final[tuple[int, ...]] = tuple(
            metadata["original_indices"]
        )
        """
        The tuple containing the indices of the original samples. The i-th element is the original index of the i-th sample.
        """

        self.original_names: Final[tuple[str, ...]] = tuple(metadata["original_names"])
        """
        The tuple containing the names of the original samples. The i-th element is the original name of the i-th sample.
        """

    def __len__(self) -> int:
        return len(self._name_lookup_table)

    def __getitem__(self, idx: SupportsIndex, /) -> depth_tools.Sample:
        """
        Get a sample with an object inserted.

        Parameters
        ----------
        idx
            The index of the sample.

        Returns
        -------
        v
            The mask. Format: ``Im_Mask``

        Raises
        ------
        IndexError
            If the sample index is out of bounds. Negative indices are supported.
        """
        idx = self._process_index(idx)

        image = cv2.imread(str(self.cache_dir / f"rgb_{idx:010}.png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose([2, 0, 1])
        image = image.astype(np.float32) / 255

        f_x: int = self._camera_params[idx][0].item()
        f_y: int = self._camera_params[idx][1].item()
        c_x: int = self._camera_params[idx][2].item()
        c_y: int = self._camera_params[idx][3].item()

        camera = depth_tools.CameraIntrinsics(f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)

        return {
            "rgb": image,
            "depth": np.load(self.cache_dir / f"depth_{idx:010}.npy"),
            "mask": np.load(self.cache_dir / f"depth_mask_{idx:010}.npy"),
            "name": self._name_lookup_table[idx],
            "camera": camera,
        }

    def get_obj_mask(self, idx: int, /) -> np.ndarray:
        """
        Get the mask that selects the pixels of the given object.

        Parameters
        ----------
        idx
            The index of the sample.

        Returns
        -------
        v
            The mask. Format: ``Im_Mask``

        Raises
        ------
        IndexError
            If the mask index is out of bounds. Negative indices are supported.
        """
        idx = self._process_index(idx)

        return np.load(self.cache_dir / f"obj_mask_{idx:010}.npy")

    def _process_index(self, idx: SupportsIndex) -> int:
        """
        Validate the index and take care of negative indexing.

        Parameters
        ----------
        idx
            The index to process.

        Returns
        -------
        v
            The original index if the index is non-negative. Otherwise the negative index converted to positive index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """

        real_idx = operator.index(idx)

        if real_idx > len(self):
            raise IndexError(f"The index {idx} is out of bounds.")

        if real_idx < 0:
            if real_idx < -len(self):
                raise IndexError(f"The index {idx} is out of bounds.")
            else:
                real_idx = len(self) + real_idx
        return real_idx

    @staticmethod
    def generate_or_load(
        *,
        insertion_data_factory: "Callable[[], tuple[DatasetBasedInserter, list[InsertionSpec]]]",
        cache_dir: Path,
        report_tqdm_progress: bool,
    ) -> "DatasetWithObjectsInserted":
        if cache_dir.is_dir():
            cache_dir.mkdir(parents=True)
            return DatasetWithObjectsInserted(cache_dir=cache_dir)
        else:
            inserter, specs = insertion_data_factory()
            DatasetWithObjectsInserted.generate(
                inserter=inserter,
                cache_dir=cache_dir,
                insertion_specs=specs,
                report_progress_tqdm=report_tqdm_progress,
            )

            return DatasetWithObjectsInserted(cache_dir)

    @staticmethod
    def generate(
        *,
        inserter: "DatasetBasedInserter",
        cache_dir: Path,
        insertion_specs: "list[InsertionSpec]",
        report_progress_tqdm: bool,
    ) -> None:
        if report_progress_tqdm:
            from tqdm import tqdm  # type: ignore
        else:
            tqdm = lambda x: x

        idx_spec_pairs = list(enumerate(insertion_specs))

        original_names: list[str] = []
        original_indices: list[int] = []

        camera_data_array = np.zeros((len(insertion_specs), 4), dtype=np.float32)

        for idx, spec in tqdm(idx_spec_pairs):
            result = inserter.insert_to(spec)

            np.save(cache_dir / f"depth_{idx:010}.npy", result.depth)
            np.save(cache_dir / f"depth_mask_{idx:010}.npy", result.depth_mask)
            np.save(cache_dir / f"obj_mask_{idx:010}.npy", result.obj_mask)

            saveable_im = result.im
            saveable_im = saveable_im * 255
            saveable_im = saveable_im.astype(np.uint8)
            saveable_im = saveable_im.transpose([1, 2, 0])
            saveable_im = cv2.cvtColor(saveable_im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(cache_dir / f"rgb_{idx:010}.png"), saveable_im)

            original_sample = inserter.dataset[spec["src_idx"]]

            original_indices.append(spec["src_idx"])
            original_names.append(original_sample["name"])

            camera = original_sample["camera"]

            camera_data_array[idx][0] = camera.f_x
            camera_data_array[idx][1] = camera.f_y
            camera_data_array[idx][2] = camera.c_x
            camera_data_array[idx][3] = camera.c_y

        json_data = json.dumps(
            {"original_indices": original_indices, "original_names": original_names}
        )
        (cache_dir / "metadata.json").write_text(json_data)

        np.save(cache_dir / "camera_params.npy", camera_data_array)

    def __iter__(self) -> Iterator[depth_tools.Sample]:
        for i in range(len(self)):
            yield self[i]


class _OriginalSamples(TypedDict):
    dataset: depth_tools.Dataset
    is_im_linear: bool
    global_camera: depth_tools.CameraIntrinsics
    global_im_size: Vec2i
    is_depth_usable_for_depth_maps: bool


class DatasetBasedInserter:
    """
    Do the object insertion based on a pre-existing configuration that can be serialized to JSON.

    The inserter uses the cache functionality of its internal inserter, so the subsequent insertions are faster if the same source index and origin position is used. The name of the cached environment is: ``last_insert_serialized``.

    Parameters
    ----------
    original_samples
        The object that describes the original samples to modify.
    floor_proxy_size
        The size of the floor proxy object.
    obj_keys
        A dictionary that maps the actual pathso of the insertable objects to saveable and displayable string keys.
    output_im_linear
        The if this is true, then the transfer function of the resulting insertions will be linear. Otherwise they will be sRGB. Note that this does not affect the assumed transfer function of the input images, since that is described by the source samples.
    post_insertion_hook
        If this function is configured, then a transform is applied on the result of the insertion. This is useful if you want to edit it, for example to add a black border.
    debug_window
        If this is true, then the debugging window of the internal inserter will be visible. Only useful for debugging purposes.
    internal_inserter
        If this is given, then this inserter will be used. Useful for testing purposes.
    """

    def __init__(
        self,
        original_samples: _OriginalSamples,
        floor_proxy_size: Vec2,
        objs_and_keys: Mapping[str, Path],
        pt_device: torch.device,
        output_im_linear: bool,
        post_insertion_hook: Callable[[InsertionResult], InsertionResult] | None = None,
        debug_window: bool = False,
        internal_inserter: Inserter | None = None,
    ) -> None:
        self.post_insertion_hook = post_insertion_hook
        self._original_samples = original_samples.copy()

        if internal_inserter is None:
            self._internal_inserter: Final = Inserter(
                debug_window=debug_window,
                floor_proxy_size=floor_proxy_size,
                im_size=original_samples["global_im_size"],
                pt_device=pt_device,
                camera=original_samples["global_camera"],
            )
        else:
            self._internal_inserter = internal_inserter  # type: ignore
        init_successful = False
        try:
            self._objs_and_keys = {
                name: self._internal_inserter.load_model(path)
                for name, path in objs_and_keys.items()
            }
            init_successful = True
        finally:
            if not init_successful:
                self._internal_inserter.destroy()
        self.output_im_linear = output_im_linear
        self.obj_keys: Final = frozenset(self._objs_and_keys.keys())
        """
        A set that contains the string keys for the insertable objects.
        """

        self.im_size: Final = self._original_samples["global_im_size"]
        self._last_insert: dict[str, Any] | None = None

    @property
    def dataset(self) -> "depth_tools.Dataset":
        return self._original_samples["dataset"]

    def insert_to(self, insertion_spec: "InsertionSpec", /) -> InsertionResult:
        """
        Do the insertion based on the given specification.

        This class internally uses `voit.Inserter.insert` to do the insertion.

        Parameters
        ----------
        insertion_spec
            The specification that controls the insertion.

        Returns
        -------
        v
            The result of the insertion.

        Raises
        ------
        VoitError
            If the point to which the objet should be inserted is outside of the image.

            If the key of the inserted object is not one of the configured object keys.

            There is no depth at which the object should be inserted.


        """
        src_idx = insertion_spec["src_idx"]
        pos_px = Vec2i(insertion_spec["pos_x"], insertion_spec["pos_y"])
        obj_key = insertion_spec["obj_key"]

        if (not (0 <= pos_px.x < self.im_size.x)) or (
            not (0 <= pos_px.y < self.im_size.y)
        ):
            raise VoitError(
                f"The pixel to which the object should be inserted is outside of the image. Pixel: {pos_px}; image size: {self.im_size}"
            )

        original_dataset = self._original_samples["dataset"]

        if obj_key not in self.obj_keys:
            raise VoitError(f'There is no object key "{obj_key}" in {self.obj_keys}')

        src_sample = original_dataset[src_idx]
        if not src_sample["mask"][0, pos_px.y, pos_px.x]:
            raise VoitError(f"There is no depth at pixel {pos_px}.")

        if not (0 <= src_idx < len(original_dataset)):
            raise VoitError(
                f"The index of the source sample is outside of the bounds of the source samples. Index: {src_idx}, number of samples: {len(original_dataset)}."
            )

        input_depth: InputDepthParam | None = (
            {"depth": src_sample["depth"], "mask": src_sample["mask"]}
            if self._original_samples["is_depth_usable_for_depth_maps"]
            else None
        )

        new_last_insert = {"src_idx": src_idx, "pos_px": pos_px}
        if self._last_insert != new_last_insert:
            self._last_insert = new_last_insert
            self._internal_inserter.bake(
                at={
                    "input_im": src_sample["rgb"],
                    "input_im_linear": self._original_samples["is_im_linear"],
                    "pos_px": pos_px,
                    "pos_depth": src_sample["depth"][0, pos_px.y, pos_px.x],
                    "input_depth": input_depth,
                },
                name="last_insert_serialized",
            )

        insertion_result = self._internal_inserter.insert(
            at="last_insert_serialized",
            normal_vs=Vec3(
                insertion_spec["normal_x"],
                insertion_spec["normal_y"],
                insertion_spec["normal_z"],
            ),
            obj=self._objs_and_keys[obj_key],
            rotation_around_floor_normal_cw=insertion_spec["rotation"],
            output_im_linear=self.output_im_linear,
            depth_occlusion={"threshold": insertion_spec["depth_occl_thr"]},
        )

        if self.post_insertion_hook is not None:
            insertion_result = self.post_insertion_hook(insertion_result)
        return insertion_result

    def get_camera(self) -> depth_tools.CameraIntrinsics:
        return self._internal_inserter.camera

    def destroy(self) -> None:
        return self._internal_inserter.destroy()


def load_insertion_specs(src: Path) -> "list[InsertionSpec]":
    return json.loads(src.read_text())


def save_insertion_specs(dst: Path, insertion_specs: "Sequence[InsertionSpec]") -> None:
    insertion_specs = list(insertion_specs)
    dst.write_text(json.dumps(insertion_specs))


class InsertionSpec(TypedDict):
    src_idx: int
    obj_key: str
    pos_x: int
    pos_y: int
    normal_x: float
    normal_y: float
    normal_z: float
    depth_occl_thr: float
    rotation: float


class EditorGUI:
    def __init__(
        self,
        editor_inserter: DatasetBasedInserter,
        jump_next: bool,
        max_displayed_points: int | None,
        fig_tmp_path: Path | None,
        insertion_specs: "list[InsertionSpec] | None",
        on_apply_hook: Callable[[list[InsertionSpec]], str | None] | None,
    ):

        self._dataset: Final = editor_inserter.dataset
        self.max_displayed_points = max_displayed_points
        im_size = editor_inserter.im_size

        self._inserter = editor_inserter
        model_options = sorted(self._inserter.obj_keys)

        self._editor_state = EditorState(
            error_handler=self._show_error,
            initial_obj_key=model_options[0],
            jump_next=jump_next,
            samples=insertion_specs if insertion_specs is not None else [],
            on_apply_hook=(
                on_apply_hook if (on_apply_hook is not None) else (lambda _: None)
            ),
        )

        if fig_tmp_path is not None:
            self._fig_path = fig_tmp_path / "rgbd.html"
        else:
            self._fig_path = None

        self._root = tk.Tk()

        self._tkinter_im_anchor = ImageTk.PhotoImage(
            Image.fromarray(np.ones((im_size.y, im_size.x, 3), dtype=np.uint8))
        )

        tk.Label(self._root, text="Sample index").grid(row=0, column=0)
        self._sample_in = tk.Entry(self._root)
        self._sample_in.grid(row=0, column=1)
        self._sample_in.insert(0, "next")
        self._sample_in.bind("<Return>", self._on_sample_idx_change)
        self._sample_in.bind("<FocusOut>", self._on_sample_idx_change)

        tk.Label(self._root, text="NYUv2 sample").grid(row=1, column=0)
        self._nyu_in = tk.Entry(self._root)
        self._nyu_in.grid(row=1, column=1)
        self._nyu_in.insert(0, "0")
        self._nyu_in.bind("<Return>", self._on_src_idx_change)
        self._nyu_in.bind("<FocusOut>", self._on_src_idx_change)

        tk.Label(self._root, text="Select object").grid(row=2, column=0)
        self._selected_model = tk.StringVar()
        self._selected_model.set(model_options[0])
        self._model_selector = tk.OptionMenu(
            self._root, self._selected_model, *model_options
        )
        self._model_selector.grid(row=2, column=1)
        self._selected_model.trace_add(
            "write", lambda *args: self._on_selected_model_change()
        )

        tk.Label(self._root, text="Depth occl. thr.").grid(row=3, column=0)
        self._depth_occl_thr_in = tk.Entry(self._root)
        self._depth_occl_thr_in.grid(row=3, column=1)
        self._depth_occl_thr_in.insert(0, "0")
        self._depth_occl_thr_in.bind("<Return>", self._on_depth_occl_thr_change)
        self._depth_occl_thr_in.bind("<FocusOut>", self._on_depth_occl_thr_change)

        tk.Label(self._root, text="CW rotation (°)").grid(row=4, column=0)
        self._rot_in = tk.Entry(self._root)
        self._rot_in.grid(row=4, column=1)
        self._rot_in.insert(0, "0")
        self._rot_in.bind("<Return>", self._on_rotation_change)
        self._rot_in.bind("<FocusOut>", self._on_rotation_change)

        tk.Label(self._root, text="Normal est.").grid(row=5, column=0)
        self._estimate_normal_btn = tk.Button(
            self._root,
            text="Estimate N",
            command=lambda *args: self._estimate_normal_click(),
        )
        self._estimate_normal_btn.grid(row=5, column=1)

        self._canvas = tk.Canvas(self._root, width=im_size.x, height=im_size.y)
        self._canvas.create_image((0, 0), anchor="nw", image=self._tkinter_im_anchor)
        self._canvas.grid(row=6, column=0, columnspan=2)
        self._canvas.bind("<ButtonRelease-1>", self._on_im_viewer_click)

        self._add_and_next = tk.Button(
            self._root,
            text="View RGBD",
            command=lambda *args: self._on_show_rgbd_click(),
        )
        self._add_and_next.grid(row=7, column=0)

        self._assumed_normal_out = tk.Label(self._root)
        self._assumed_normal_out.grid(row=7, column=1)

        self._add_and_next = tk.Button(
            self._root, text="Apply", command=lambda *args: self._on_add_and_next()
        )
        self._add_and_next.grid(row=7, column=3)
        self._last_current = None

    def run(self):
        self._update_selection_ui()
        self._root.mainloop()

    def get_insertion_specs(self) -> list[InsertionSpec]:
        return copy.deepcopy(self._editor_state.samples)

    def _update_selection_ui(self) -> None:
        self._sample_in.delete(0, tk.END)
        self._sample_in.insert(0, str(self._editor_state.sample_idx))

        self._nyu_in.delete(0, tk.END)
        self._nyu_in.insert(0, str(self._editor_state.current_sample["src_idx"]))
        self._selected_model.set(self._editor_state.current_sample["obj_key"])

        self._depth_occl_thr_in.delete(0, tk.END)
        self._depth_occl_thr_in.insert(
            0, str(self._editor_state.current_sample["depth_occl_thr"])
        )

        self._rot_in.delete(0, tk.END)
        self._rot_in.insert(0, str(self._editor_state.current_sample["rotation"]))

        if self._editor_state.normal_estimation_state is not None:
            self._estimate_normal_btn["text"] = "Cancel"
            self._sample_in["state"] = "disabled"
            self._nyu_in["state"] = "disabled"
            self._depth_occl_thr_in["state"] = "disabled"
            self._rot_in["state"] = "disabled"
        else:
            self._estimate_normal_btn["text"] = "Estimate"
            self._sample_in["state"] = "normal"
            self._nyu_in["state"] = "normal"
            self._depth_occl_thr_in["state"] = "normal"
            self._rot_in["state"] = "normal"

        new_current = copy.deepcopy(self._editor_state.current_sample)
        if new_current != self._last_current:
            self._last_current = new_current
            if (
                self._editor_state.current_sample["pos_x"] == -1
                or self._editor_state.current_sample["pos_y"] == -1
            ):
                self._add_and_next["state"] = "disabled"
                shown_sample: depth_tools.Sample = self._dataset[new_current["src_idx"]]
                self._assumed_normal_out["text"] = "N=-"
                self._estimate_normal_btn["state"] = "disabled"
            else:
                assumed_normal = Vec3(
                    x=self._editor_state.current_sample["normal_x"],
                    y=self._editor_state.current_sample["normal_y"],
                    z=self._editor_state.current_sample["normal_z"],
                )
                insertion_result = self._inserter.insert_to(
                    self._editor_state.current_sample
                )
                self._add_and_next["state"] = "normal"
                self._assumed_normal_out["text"] = (
                    f"N=({assumed_normal.x:.2f}, {assumed_normal.y:.2f}, {assumed_normal.z:.2f})"
                )
                self._estimate_normal_btn["state"] = "normal"
                shown_sample: depth_tools.Sample = {
                    "rgb": insertion_result.im,
                    "depth": insertion_result.depth,
                    "mask": insertion_result.depth_mask,
                    "name": str(self._editor_state.current_sample["src_idx"]),
                    "camera": self._inserter.get_camera(),
                }

            self._shown_sample: depth_tools.Sample = shown_sample
            self._tkinter_im_anchor = ImageTk.PhotoImage(
                Image.fromarray(
                    (shown_sample["rgb"].transpose([1, 2, 0]) * 255.0).astype(np.uint8)
                )
            )
            self._canvas.create_image(
                (0, 0), anchor="nw", image=self._tkinter_im_anchor
            )

    def _on_add_and_next(self) -> None:
        self._editor_state.apply_save_next()
        self._update_selection_ui()

    def _on_im_viewer_click(self, evt) -> None:
        pos_x = evt.x
        pos_y = evt.y
        old_sample = self._dataset[self._editor_state.current_sample["src_idx"]]

        if self._editor_state.normal_estimation_state is None:
            self._editor_state.current_sample["pos_x"] = pos_x
            self._editor_state.current_sample["pos_y"] = pos_y
        elif len(self._editor_state.normal_estimation_state) < 1:
            self._editor_state.normal_estimation_state.append(Vec2i(pos_x, pos_y))
        else:
            self._editor_state.normal_estimation_state.append(Vec2i(pos_x, pos_y))

            estimation = get_pos_normal_from_3_points(
                aux_points_px=(
                    self._editor_state.normal_estimation_state[0],
                    self._editor_state.normal_estimation_state[1],
                ),
                center_point_px=Vec2i(
                    self._editor_state.current_sample["pos_x"],
                    self._editor_state.current_sample["pos_y"],
                ),
                depth=old_sample["depth"],
                depth_mask=old_sample["mask"],
                camera=self._inserter.get_camera(),
            )
            assert estimation is not None

            _, n_vs = estimation
            self._editor_state.current_sample["normal_x"] = n_vs.x
            self._editor_state.current_sample["normal_y"] = n_vs.y
            self._editor_state.current_sample["normal_z"] = n_vs.z

            self._editor_state.normal_estimation_state = None

        self._update_selection_ui()

    def _on_selected_model_change(self) -> None:
        self._editor_state.current_sample["obj_key"] = self._selected_model.get()
        self._update_selection_ui()

    def _on_src_idx_change(self, evt) -> None:
        src_idx_new = self._nyu_in.get()
        if not src_idx_new.isdigit():
            self._show_error(
                f"The index in the NYUv2 dataset ({src_idx_new}) is not an integer."
            )
            self._nyu_in.delete(0, tk.END)
            self._nyu_in.insert(0, str(self._editor_state.current_sample["src_idx"]))
            return
        src_idx_new = int(src_idx_new)
        if src_idx_new == self._editor_state.current_sample["src_idx"]:
            return
        dataset_len = len(self._dataset)
        if not (0 <= src_idx_new < dataset_len):
            self._show_error(
                f"The index in the NYUv2 dataset ({src_idx_new}) is greater than the length of the dataset ({dataset_len}) or non-positive."
            )
            self._nyu_in.delete(0, tk.END)
            self._nyu_in.insert(0, str(self._editor_state.current_sample["src_idx"]))
            return

        self._editor_state.set_src_idx_and_reset_everything_else(
            int(self._nyu_in.get())
        )
        self._update_selection_ui()

    def _on_sample_idx_change(self, evt) -> None:
        sample_in_new = self._sample_in.get()
        if sample_in_new == "next" or sample_in_new == "":
            new_sample_idx = len(self._editor_state.samples)
        else:
            if sample_in_new.isdigit():
                new_sample_idx = int(sample_in_new)
            else:
                self._sample_in.delete(0, tk.END)
                self._sample_in.insert(0, str(self._editor_state.sample_idx))
                self._show_error('The new sample index is neither a number nor "next".')
                return
        if new_sample_idx == self._editor_state.sample_idx:
            return

        if not (0 <= new_sample_idx <= len(self._editor_state.samples)):
            self._show_error(
                f"The shown sample index is outside of range 0 and {len(self._editor_state.samples)} (both including)."
            )
            self._sample_in.delete(0, tk.END)
            self._sample_in.insert(0, str(self._editor_state.sample_idx))
            return

        self._editor_state.select(new_sample_idx)
        self._update_selection_ui()

    def _on_depth_occl_thr_change(self, evt) -> None:
        depth_occl_thr_new = self._depth_occl_thr_in.get()
        if not re.match(r"^\+?[0-9]+(\.[0-9]+)?$", depth_occl_thr_new):
            self._show_error(
                f"The new depth occlusion threshold ({depth_occl_thr_new}) is not a positive floating point number."
            )
            self._depth_occl_thr_in.delete(0, tk.END)
            self._depth_occl_thr_in.insert(
                0, str(self._editor_state.current_sample["depth_occl_thr"])
            )
            return
        depth_occl_thr_new = float(depth_occl_thr_new)
        self._editor_state.current_sample["depth_occl_thr"] = depth_occl_thr_new
        self._update_selection_ui()

    def _on_rotation_change(self, evt) -> None:
        rot_new = self._rot_in.get()
        if not re.match(r"^(\+|-)?[0-9]+(\.[0-9]+)?$", rot_new):
            self._show_error(
                f"The new rotation ({rot_new}) is not a floating point number."
            )
            self._rot_in.delete(0, tk.END)
            self._rot_in.insert(0, str(self._editor_state.current_sample["rotation"]))
            return
        rot_new = float(rot_new)
        self._editor_state.current_sample["rotation"] = rot_new
        self._update_selection_ui()

    def _estimate_normal_click(self) -> None:
        if self._editor_state.normal_estimation_state is None:
            self._editor_state.normal_estimation_state = []
        else:
            self._editor_state.normal_estimation_state = None
        self._update_selection_ui()

    def _on_show_rgbd_click(self):
        assert self._fig_path is not None

        if self.max_displayed_points is not None:
            fig = depth_tools.depths_2_plotly_fig(
                intrinsics=self._inserter.get_camera(),
                depth_maps=[
                    {
                        "color": self._shown_sample["rgb"],
                        "depth_map": self._shown_sample["depth"],
                        "depth_mask": self._shown_sample["mask"],
                        "name": "RGBD",
                        "size": 2,
                    }
                ],
                subsample={"max_num": self.max_displayed_points},
                coord_sys=depth_tools.CoordSys.LH_YUp,
            )
        else:
            fig = depth_tools.depths_2_plotly_fig(
                intrinsics=self._inserter.get_camera(),
                depth_maps=[
                    {
                        "color": self._shown_sample["rgb"],
                        "depth_map": self._shown_sample["depth"],
                        "depth_mask": self._shown_sample["mask"],
                        "name": "RGBD",
                        "size": 2,
                    }
                ],
                coord_sys=depth_tools.CoordSys.LH_YUp,
            )
        fig.update_layout(height=900)
        fig.write_html(self._fig_path)
        webbrowser.open("file://" + str(self._fig_path.resolve()))

    def _show_error(self, msg: str) -> None:
        tkinter.messagebox.showerror("Error", msg)


class EditorState:
    def __init__(
        self,
        samples: list[InsertionSpec],
        error_handler: Callable[[str], None],
        initial_obj_key: str,
        jump_next: bool,
        on_apply_hook: Callable[[list[InsertionSpec]], str | None],
    ):
        self.initial_obj_key = initial_obj_key
        self.jump_next = jump_next
        self.samples = copy.deepcopy(samples)
        self._sample_idx = len(self.samples)
        self.current_sample: InsertionSpec = {
            "src_idx": 0,
            "obj_key": initial_obj_key,
            "pos_x": -1,
            "pos_y": -1,
            "depth_occl_thr": 2.0,
            "rotation": 0,
            "normal_x": 0,
            "normal_y": 1,
            "normal_z": 0,
        }
        self.error_handler: Callable[[str], None] = error_handler
        self.normal_estimation_state: list[Vec2i] | None = None
        self.on_apply_hook = on_apply_hook

    def set_src_idx_and_reset_everything_else(self, new_src_idx: int) -> None:
        self.current_sample = {
            "src_idx": new_src_idx,
            "obj_key": self.initial_obj_key,
            "pos_x": -1,
            "pos_y": -1,
            "depth_occl_thr": 2.0,
            "rotation": 0,
            "normal_x": 0,
            "normal_y": 1,
            "normal_z": 0,
        }

    @property
    def sample_idx(self) -> int:
        return self._sample_idx

    def save(self) -> None:
        self.on_apply_hook(self.samples)

    def select(self, sample_idx: int) -> None:
        if not (0 <= sample_idx <= len(self.samples)):
            self.error_handler(f"The sample index {sample_idx} is out of bounds.")
            return

        self.normal_estimation_state = None

        self._sample_idx = sample_idx
        if sample_idx < len(self.samples):
            self.current_sample = self.samples[sample_idx]
        else:
            self.current_sample = {
                "src_idx": 0,
                "obj_key": self.initial_obj_key,
                "pos_x": -1,
                "pos_y": -1,
                "depth_occl_thr": 2.0,
                "rotation": 0,
                "normal_x": 0,
                "normal_y": 1,
                "normal_z": 0,
            }

    def apply_save_next(self) -> None:
        if self._sample_idx == len(self.samples):
            self.samples.append(self.current_sample)
        else:
            self.samples[self._sample_idx] = self.current_sample
        self.save()
        if self.jump_next:
            self.select(min(self._sample_idx + 1, len(self.samples)))
        else:
            self.select(len(self.samples))
