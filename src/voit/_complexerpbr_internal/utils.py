from __future__ import annotations

import math

import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase

from . import _shaderutils as shaderutils


def load_sdr_lut(filename: str) -> p3d.Texture:
    """Load an SDR color LUT embedded in a screenshot"""
    path = p3d.Filename(filename)
    vfs = p3d.VirtualFileSystem.get_global_ptr()
    failed = (
        not vfs.resolve_filename(path, p3d.get_model_path().value)
        or not path.is_regular_file()
    )
    if failed:
        raise RuntimeError(f"Failed to find file {filename}")

    image = p3d.PNMImage(path)

    lutdim = 64
    xsize, ysize = image.get_size()
    tiles_per_row = xsize // lutdim
    num_rows = math.ceil(lutdim / tiles_per_row)
    ysize -= num_rows * lutdim

    texture = p3d.Texture()
    texture.setup_3d_texture(
        lutdim, lutdim, lutdim, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgb8
    )
    texture.minfilter = p3d.Texture.FT_linear
    texture.magfilter = p3d.Texture.FT_linear

    for tileidx in range(lutdim):
        xstart = tileidx % tiles_per_row * lutdim
        ystart = tileidx // tiles_per_row * lutdim + ysize
        islice = p3d.PNMImage(lutdim, lutdim, 3, 255)
        islice.copy_sub_image(image, 0, 0, xstart, ystart, lutdim, lutdim)
        texture.load(islice, tileidx, 0)
    return texture


def sdr_lut_screenshot(  # type: ignore[no-untyped-def]
    showbase: ShowBase, *args, **kwargs
) -> str | None:
    """Take a screenshot with an embedded SDR color LUT"""
    filename = showbase.screenshot(*args, **kwargs)

    if not filename:
        return filename  # type: ignore

    lutdim = 64
    stepsize = 256 // lutdim

    image = p3d.PNMImage(filename)
    xsize, ysize = image.get_size()
    tiles_per_row = xsize // lutdim
    num_rows = math.ceil(lutdim / tiles_per_row)

    image.expand_border(0, 0, num_rows * lutdim, 0, (0, 0, 0, 1))

    steps = list(range(0, 256, stepsize))
    maxoffset = len(steps) - 1

    for tileidx, bcol in enumerate(steps):
        xbase = tileidx % tiles_per_row * lutdim
        ybase = tileidx // tiles_per_row * lutdim + ysize
        for xoff, rcol in enumerate(steps):
            xcoord = xbase + xoff
            for yoff, gcol in enumerate(steps):
                ycoord = ybase + maxoffset - yoff
                image.set_xel_val(xcoord, ycoord, rcol, gcol, bcol)

    image.write(filename)

    return filename  # type: ignore
