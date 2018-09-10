#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gdal, osr
import numpy as np
from math import ceil
from .wrapper import array, full
from .gdalio import _getDataset, _fromDataset
from .gdalspatial import _Projection, _Transformer

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

_RESAMPLING = {
    # A documentation would be nice.
    # There seem to be more functions in GDAL > 2
    "average"     : gdal.GRA_Average,
    "bilinear"    : gdal.GRA_Bilinear,
    "cubic"       : gdal.GRA_Cubic,
    "cubicspline" : gdal.GRA_CubicSpline,
    "lanczos"     : gdal.GRA_Lanczos,
    "mode"        : gdal.GRA_Mode,
    "nearest"     : gdal.GRA_NearestNeighbour,
    # only available in GDAL > 2
    "max"         : getattr(gdal, "GRA_Max", None),
    "min"         : getattr(gdal, "GRA_Min", None),
}

def _warpTo(source, target, func, max_error=0.125):

    if func is None:
        raise TypeError("Resampling method {:} not available in your GDAL version".format(func))

    target = np.atleast_2d(target)
    if target.ndim < source.ndim:
        target = np.broadcast_to(
            target, source.shape[:-len(target.shape)]+target.shape, subok=True)

    target = np.ma.array(
        target,
        mask  = target.data==target.fill_value,
        dtype = source.dtype,
        copy  = True,
        subok = True)

    target[target.mask] = source.fill_value
    target.fill_value = source.fill_value

    out = _getDataset(target, True)

    gdal.Warp(out, _getDataset(source),
              resampleAlg=_RESAMPLING[func],
              errorThreshold=max_error,
              # geoloc=True,
              # transformerOptions=["SRC_METHOD=NO_GEOTRANSFORM"]
    )

    # gdal.ReprojectImage(
    #     _getDataset(source), out,
    #     None, None,
    #     _RESAMPLING[func],
    #     0.0, max_error)
    # print _fromDataset(out)

    return _fromDataset(out)


def project(grid, proj, cellsize=None, func="nearest", max_error=0.125):
    # type: (GeoArray, _Projection, Optional[float], str, float) -> GeoArray

    bbox = grid.bbox
    proj = _Projection(proj)
    trans = _Transformer(grid.proj, proj)
    uly, ulx = trans(bbox["ymax"], bbox["xmin"])
    lry, lrx = trans(bbox["ymin"], bbox["xmax"])
    ury, urx = trans(bbox["ymax"], bbox["xmax"])
    lly, llx = trans(bbox["ymin"], bbox["xmin"])

    # Calculate cellsize, i.e. same number of cells along the diagonal.
    if cellsize is None:
        src_diag = np.sqrt(grid.nrows**2 + grid.ncols**2)
        trg_diag = np.sqrt((lly - ury)**2 + (llx - urx)**2)
        cellsize = trg_diag/src_diag

    # number of cells
    ncols = int(abs(ceil((max(urx, lrx, ulx, llx) - min(urx, lrx, ulx, llx))/cellsize)))
    nrows = int(abs(ceil((max(ury, lry,  uly, lly) - min(ury, lry, uly, lly))/cellsize)))

    target = array(
        data       = np.full((grid.nbands, nrows, ncols), grid.fill_value, grid.dtype),
        fill_value = grid.fill_value,
        dtype      = grid.dtype,
        yorigin    = max(uly, ury, lly, lry),
        xorigin    = min(ulx, urx, llx, lrx),
        origin     = "ul",
        ycellsize  = abs(cellsize) * -1,
        xcellsize  = cellsize,
        # cellsize   = cellsize,
        proj       = proj,
        mode       = grid.mode,
    )

    out = resample(
        source    = grid,
        target    = target,
        func      = func,
        max_error = max_error)
    return out.trim()


def resample(source, target, func="nearest", max_error=0.125):
    return _warpTo(
        source=source,
        target=target,
        func=func,
        max_error=max_error)


def rescale(source, scaling_factor, func="nearest"):
    shape = (list(source.shape[:-2]) +
             [int(s / scaling_factor) for s in source.shape[-2:]])
    cellsize = [c * scaling_factor for c in source.cellsize]
    scaled_grid = full(shape, source.fill_value,
                       xorigin=source.xorigin, yorigin=source.yorigin,
                       cellsize=cellsize, dtype=source.dtype)
    return resample(source, scaled_grid, func=func)
