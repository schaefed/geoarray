#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author
------
David Schaefer

Purpose
-------
This module provides initializer function for core.GeoArray
"""

import numpy as np
from .core import GeoArray
from .gdalio import _fromFile, _fromDataset
from .gdaltrans import _Projection, _Geotrans
# from typing import Optional, Union, Tuple, Any, Mapping, AnyStr


def array(data,               # type: Union[np.ndarray, GeoArray]
          dtype      = None,  # type: Optional[Union[AnyStr, np.dtype]]
          yorigin    = 0,     # type: Optional[Union[int, float]]
          xorigin    = 0,     # type: Optional[float]
          origin     = "ul",  # type: Optional[floatAnyStr]
          fill_value = None,  # type: Optional[Union[int, float]]
          cellsize   = 1,     # type: Optional[Union[int, float]]]
          ycellsize  = None,  # Optional[Union[int, float]]
          xcellsize  = None,  # Optional[Union[int, float]]
          yparam     = 0,     # Optional[Union[int, float]]
          xparam     = 0,     # Optional[Union[int, float]]
          proj       = None,  # type: Mapping[AnyStr, Union[AnyStr, float]]
          mode       = "r",   # type: AnyStr
          color_mode = "L",   # type: AnyStr
          copy       = False, # type: bool
          fobj       = None,  # type: Optional[osgeo.gdal.Dataset]
):                            # type: (...) -> GeoArray
    """
    Arguments
    ---------
    data         : numpy.ndarray  # data to wrap

    Optional Arguments
    ------------------
    dtype        : str/np.dtype                  # type of the returned grid
    yorigin      : int/float, default: 0         # y-value of the grid's origin
    xorigin      : int/float, default: 0         # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"},        # position of the origin. One of:
                   default: "ul"                 #     "ul" : upper left corner
                                                 #     "ur" : upper right corner
                                                 #     "ll" : lower left corner
                                                 #     "lr" : lower right corner
    fill_value   : inf/float                     # fill or fill value
    cellsize     : int/float or 2-tuple of those # cellsize, cellsizes in y and x direction
    proj         : dict/None                     # proj4 projection parameters
    copy         : bool                          # create a copy of the given data

    Returns
    -------
    GeoArray

    Purpose
    -------
    Create a GeoArray from data.
    """
    if ycellsize is None:
        ycellsize = cellsize
        if origin[0] == "u":
            ycellsize *= -1

    if xcellsize is None:
        xcellsize = cellsize
        if origin[1] == "r":
            xcellsize *= -1

    proj = _Projection(proj)

    if isinstance(data, GeoArray):
        return GeoArray(
            dtype      = dtype or data.dtype,
            yorigin    = data.yorigin,
            xorigin    = data.xorigin,
            ycellsize  = data.ycellsize,
            xcellsize  = data.xcellsize,
            yparam     = data.yparam,
            xparam     = data.xparam,
            fill_value = fill_value or data.fill_value,
            proj       = proj or data.proj,
            mode       = mode or data.mode,
            color_mode = color_mode or data.color_mode,
            fobj       = data.fobj,
            data       = data.data)

    return GeoArray(
        data       = np.array(data, dtype=dtype, copy=copy),
        yorigin    = yorigin,
        xorigin    = xorigin,
        ycellsize  = ycellsize,
        xcellsize  = xcellsize,
        yparam     = yparam,
        xparam     = xparam,
        fill_value = fill_value,
        proj       = proj,
        mode       = mode,
        color_mode = color_mode,
        fobj       = fobj,)


def zeros(shape, dtype=np.float64, *args, **kwargs):
    """
    Arguments
    ---------
    shape        : tuple          # shape of the returned grid

    Optional Arguments
    ------------------
    see array

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with zeros.
    """

    return array(data=np.zeros(shape, dtype), *args, **kwargs)


def ones(shape, dtype=np.float64, *args, **kwargs):
    """
    Arguments
    ---------
    shape        : tuple          # shape of the returned grid

    Optional Arguments
    ------------------
    see array

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with ones.
    """

    return array(data=np.ones(shape, dtype), *args, **kwargs)


def full(shape, value, dtype=np.float64, *args, **kwargs):
    """
    Arguments
    ---------
    shape        : tuple          # shape of the returned grid
    fill_value   : scalar         # fille value

    Optional Arguments
    ------------------
    see array

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with fill_value.
    """

    return array(data=np.full(shape, value, dtype), *args, **kwargs)


def empty(shape, dtype=np.float64, *args, **kwargs):
    """
    Arguments
    ----------
    shape        : tuple          # shape of the returned grid

    Optional Arguments
    ------------------
    see array

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new empty GeoArray of given shape and type
    """

    return array(data=np.empty(shape, dtype), *args, **kwargs)


def _likeArgs(arr):

    out = {}
    if isinstance(arr, GeoArray):
        out["yorigin"]    = arr.yorigin
        out["xorigin"]    = arr.xorigin
        out["origin"]     = arr.origin
        out["fill_value"] = arr.fill_value
        out["cellsize"]   = arr.cellsize
        out["proj"]       = arr.proj
        out["mode"]       = "r"
        out["color_mode"] = arr.color_mode

    return out


def zeros_like(arr, dtype=None):
    args = _likeArgs(arr)
    return zeros(shape=arr.shape, dtype=dtype or arr.dtype, **args)


def ones_like(arr, dtype=None):
    args = _likeArgs(arr)
    return ones(shape=arr.shape, dtype=dtype or arr.dtype, **args)


def full_like(arr, value, dtype=None):
    args = _likeArgs(arr)
    return full(shape=arr.shape, value=value, dtype=dtype or arr.dtype, **args)


def fromdataset(ds):
    return array(**_fromDataset(ds))


def fromfile(fname, mode="r"):
    """
    Arguments
    ---------
    fname : str  # file name

    Returns
    -------
    GeoArray

    Purpose
    -------
    Create GeoArray from file

    """
    return _fromFile(fname, mode)
