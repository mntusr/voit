import builtins
import unittest
from logging import Logger
from pathlib import Path
from typing import Any

import npy_unittest
import numpy as np
from panda3d.core import LMatrix3f, LMatrix4f, NodePath

from voit import Vec2, Vec3
from voit._logging_internal import VOIT_LOGGER


class VoitTestBase(npy_unittest.NpyTestCase):
    def get_test_data_dir(self, dir_name: str) -> Path:
        if ("/" in dir_name) or ("\\" in dir_name):
            raise ValueError(
                f'The directory name "({dir_name})" contains "/" or "\\\\"'
            )
        return Path(__file__).resolve().parent / "test_data" / dir_name

    def assertVec2Allclose(self, vec1: Vec2, vec2: Vec2, atol: float = 1e-4) -> None:
        def almost_equal(v1: float, v2: float) -> bool:
            return abs(v1 - v2) < atol

        if not (almost_equal(vec1.x, vec2.x) and almost_equal(vec1.y, vec2.y)):
            raise self.failureException(
                f"The absolute difference between {repr(vec1)} and {repr(vec2)} is greater than {atol}."
            )

    def assertVec3Allclose(self, vec1: Vec3, vec2: Vec3, atol: float = 1e-4) -> None:
        def almost_equal(v1: float, v2: float) -> bool:
            return abs(v1 - v2) < atol

        if not (
            almost_equal(vec1.x, vec2.x)
            and almost_equal(vec1.y, vec2.y)
            and almost_equal(vec1.z, vec2.z)
        ):
            raise self.failureException(
                f"The absolute difference between {repr(vec1)} and {repr(vec2)} is greater than {atol}."
            )

    def assertAllAlmostEqual(
        self, array: np.ndarray, value: float, atol: float = 1e-4
    ) -> None:
        if not np.all(abs(array - value) < atol):
            raise self.failureException(
                f"Some elements of {array} are not equal to {value} within tolerance {atol}."
            )

    def assertAllAlmostIn(
        self, array: np.ndarray, values: set[float], atol: float = 1e-4
    ) -> None:
        val_found = False
        not_found_elements: list[Any] = []
        for array_item in array.flatten():
            for value in values:
                if abs(array_item - value) < atol:
                    val_found = True

            if not val_found:
                not_found_elements.append(array_item)

        if len(not_found_elements) > 0:
            raise self.failureException(
                f"The following elements were not found: {not_found_elements}. Supported values: {values}."
            )

    def assertParentOf(self, parent: NodePath, child: NodePath):
        if child not in parent.children:
            raise self.failureException(
                f"The object {parent} is not the parent of {child}."
            )

    @property
    def voit_logger(self):
        return VOIT_LOGGER

    def destroy_showbase(self):
        if hasattr(builtins, "base"):
            getattr(builtins, "base").destroy()

    def assertAncestorOf(self, ancestor: NodePath, descendant: NodePath):
        if not ancestor.isAncestorOf(descendant):
            raise self.failureException(
                f"The object {ancestor} is not an ancestor of {descendant}."
            )

    def panda3d_mat_2_mat(self, mat: LMatrix4f | LMatrix3f) -> np.ndarray:
        if isinstance(mat, LMatrix4f):
            return np.array(
                [
                    [
                        mat.get_cell(0, 0),
                        mat.get_cell(0, 1),
                        mat.get_cell(0, 2),
                        mat.get_cell(0, 3),
                    ],
                    [
                        mat.get_cell(1, 0),
                        mat.get_cell(1, 1),
                        mat.get_cell(1, 2),
                        mat.get_cell(1, 3),
                    ],
                    [
                        mat.get_cell(2, 0),
                        mat.get_cell(2, 1),
                        mat.get_cell(2, 2),
                        mat.get_cell(2, 3),
                    ],
                    [
                        mat.get_cell(3, 0),
                        mat.get_cell(3, 1),
                        mat.get_cell(3, 2),
                        mat.get_cell(3, 3),
                    ],
                ],
                dtype=np.float32,
            )
        elif isinstance(mat, LMatrix3f):
            return np.array(
                [
                    [
                        mat.get_cell(0, 0),
                        mat.get_cell(0, 1),
                        mat.get_cell(0, 2),
                    ],
                    [
                        mat.get_cell(1, 0),
                        mat.get_cell(1, 1),
                        mat.get_cell(1, 2),
                    ],
                    [
                        mat.get_cell(2, 0),
                        mat.get_cell(2, 1),
                        mat.get_cell(2, 2),
                    ],
                ],
                dtype=np.float32,
            )

    def get_col_vec_transform_mat(self, obj: NodePath) -> np.ndarray:
        """
        Get the transform matrix of the given object. Keep in mind that this function only acquires the transform of the object itself, not the transform of the parents or children.

        Since Panda3d uses row vectors and we use column vectors this function also transposes the got matrix to be usable with column vectors.

        Parameters
        ----------
        obj
            The object with the relevant transform.

        Returns
        -------
        v
            The transform. Format: ``Transform::Homog[4x4]``
        """
        panda3d_mat_rowvec = obj.get_transform().get_mat()
        npy_mat_rowvec = self.panda3d_mat_2_mat(panda3d_mat_rowvec)
        return npy_mat_rowvec.T
