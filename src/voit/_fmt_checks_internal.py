from typing import Callable, Final, cast

import numpy as np

from ._errors import VoitError


def _fmt_check(name: str):
    def impl(check_fn: Callable[[np.ndarray], bool]) -> "_FmtCheckImpl":
        return _FmtCheckImpl(check_fn, name)

    return impl


class _FmtCheckImpl:
    def __init__(self, check_fn: Callable[[np.ndarray], bool], name: str) -> None:
        self.has_fmt: Final = check_fn
        self._name: Final = name

    def check_arg(self, arg: np.ndarray, name: str | None = None) -> None:
        if not self.has_fmt(arg):
            if name is not None:
                raise VoitError(f"The array {arg} does not have format {self._name}")
            else:
                raise VoitError(
                    f'The argument "{name}" does not have format {self._name}'
                )


@_fmt_check("Transform_3x3")
def Transform_3x3(array: np.ndarray) -> bool:
    return array.shape == (3, 3) and cast(bool, np.issubdtype(array.dtype, np.floating))


@_fmt_check("DepthmapLike")
def DepthmapLike(array: np.ndarray) -> bool:
    return (
        len(array.shape) == 3
        and array.shape[0] == 1
        and cast(bool, np.issubdtype(array.dtype, np.floating))
    )


@_fmt_check("Im_Mask")
def Im_Mask(array: np.ndarray) -> bool:
    return (
        len(array.shape) == 3
        and array.shape[0] == 1
        and cast(bool, np.issubdtype(array.dtype, np.bool_))
    )
