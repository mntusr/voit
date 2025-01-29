import math
import struct
import sys
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from panda3d.core import LPoint3f


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    @staticmethod
    def from_npy_col_vec(npy_vec: np.ndarray) -> "Vec3":
        expected_shape = (3, 1)
        actual_shape = tuple(npy_vec.shape)

        if actual_shape == (3, 1):
            return Vec3(
                float(npy_vec[0, 0]),
                float(npy_vec[1, 0]),
                float(npy_vec[2, 0]),
            )
        elif actual_shape == (4, 1):
            if abs(npy_vec[3, 0] - 1) < 1e-6:
                raise ValueError("The homogen coordinate is not equal to 1.")
            return Vec3(
                float(npy_vec[0, 0]),
                float(npy_vec[1, 0]),
                float(npy_vec[2, 0]),
            )
        else:
            raise ValueError(
                f"The shape of the array is {actual_shape} instead of {expected_shape}"
            )

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def normalize(self) -> "Vec3":
        vec_len = self.vec_len()
        return Vec3(
            self.x / vec_len,
            self.y / vec_len,
            self.z / vec_len,
        )

    def mul_with_scalar(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def vec_len(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def vec_plus(self, b: "Vec3") -> "Vec3":
        return Vec3(self.x + b.x, self.y + b.y, self.z + b.z)

    def vec_minus(self, b: "Vec3") -> "Vec3":
        return Vec3(self.x - b.x, self.y - b.y, self.z - b.z)

    def neg(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def to_panda_point(self) -> LPoint3f:
        return LPoint3f(self.x, self.y, self.z)


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def is_positive(self) -> bool:
        return self.x > 0 and self.y > 0

    def dot(self, other: "Vec2 | Vec2i") -> float:
        return self.x * other.x + self.y * other.y

    def mul_with_scalar(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def vec_minus(self, b: "Vec2") -> "Vec2":
        return Vec2(self.x - b.x, self.y - b.y)


@dataclass(frozen=True)
class Vec2i:
    x: int
    y: int

    def is_positive(self) -> bool:
        return self.x > 0 and self.y > 0

    def vec_minus(self, b: "Vec2i") -> "Vec2i":
        return Vec2i(self.x - b.x, self.y - b.y)

    def dot(self, other: "Vec2i") -> float:
        return self.x * other.x + self.y * other.y

    def vec_len(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self) -> "Vec2":
        vec_len = self.vec_len()
        return Vec2(
            self.x / vec_len,
            self.y / vec_len,
        )

    def vec_plus(self, b: "Vec2i") -> "Vec2i":
        return Vec2i(self.x + b.x, self.y + b.y)
