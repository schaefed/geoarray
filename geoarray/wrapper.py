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
# from typing import Optional, Union, Tuple, Any, Mapping, AnyStr


def array(data,               # type: Union[np.ndarray, GeoArray]       
          dtype      = None,  # type: Optional[Union[AnyStr, np.dtype]]
          yorigin    = None,  # type: Optional[float]
          xorigin    = None,  # type: Optional[float]
          origin     = None,  # type: Optional[AnyStr]
          fill_value = None,  # type: Optional[float]
          cellsize   = None,  # type: Optional[Union[float, Tuple[float, float]]]
          proj       = None,  # type: Mapping[AnyStr, Union[AnyStr, float]]
          mode       = None,  # type: AnyStr
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

    if isinstance(data, GeoArray):
        dtype      = dtype or data.dtype
        yorigin    = yorigin or data.yorigin
        xorigin    = xorigin or data.xorigin
        origin     = origin or data.origin
        fill_value = fill_value or data.fill_value
        cellsize   = cellsize or data.cellsize
        proj       = proj or data.proj
        mode       = mode or data.mode
        fobj       = data.fobj
        data       = data.data
        
    return GeoArray(
        data       = np.array(data, dtype=dtype, copy=copy), 
        yorigin    = yorigin or 0,
        xorigin    = xorigin or 0,
        origin     = origin or "ul",
        fill_value = fill_value,
        cellsize   = cellsize or (1,1),
        proj       = proj,
        mode       = mode,
        fobj       = fobj,
    )


def _likeArgs(arr):

    out = {}
    if isinstance(arr, GeoArray):
        out["yorigin"]    = arr.yorigin
        out["xorigin"]     = arr.xorigin
        out["origin"]     = arr.origin
        out["fill_value"] = arr.fill_value
        out["cellsize"]   = arr.cellsize
        out["proj"]       = arr.proj
        out["mode"]       = arr.mode

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


def zeros(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
          fill_value=None, cellsize=1, proj=None, mode=None):
    """
    Arguments
    ---------
    shape        : tuple          # shape of the returned grid

    Optional Arguments
    ------------------
    dtype        : str/np.dtype                  # type of the returned grid
    yorigin      : int/float                     # y-value of the grid's origin
    xorigin      : int/float                     # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"}         # position of the origin. One of:
                                                 #     "ul" : upper left corner
                                                 #     "ur" : upper right corner
                                                 #     "ll" : lower left corner
                                                 #     "lr" : lower right corner
    fill_value   : inf/float                     # fill or fill value
    cellsize     : int/float or 2-tuple of those # cellsize, cellsizes in y and x direction
    proj         : dict/None                     # proj4 projection parameters

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with zeros.
    """

    return GeoArray(
        data       = np.zeros(shape, dtype),
        yorigin    = yorigin,
        xorigin    = xorigin,
        origin     = origin,
        fill_value = fill_value,
        cellsize   = cellsize,
        proj       = proj,
        mode       = mode,
    )


def ones(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
         fill_value=None, cellsize=1, proj=None, mode=None):
    """
    Arguments
    ---------
    shape        : tuple          # shape of the returned grid

    Optional Arguments
    ------------------
    dtype        : str/np.dtype                  # type of the returned grid
    yorigin      : int/float                     # y-value of the grid's origin
    xorigin      : int/float                     # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"}         # position of the origin. One of:
    fill_value   : inf/float                     # fill or fill value
    cellsize     : int/float or 2-tuple of those # cellsize, cellsizes in y and x direction
    proj         : dict/None                     # proj4 projection parameters

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with ones.
    """

    return GeoArray(
        data       = np.ones(shape, dtype),
        yorigin    = yorigin,
        xorigin    = xorigin,
        origin     = origin,
        fill_value = fill_value,
        cellsize   = cellsize,
        proj       = proj,
        mode       = mode,
    )


def full(shape, value, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
         fill_value=None, cellsize=1, proj=None, mode=None):
    """
    Arguments
    ---------
    shape        : tuple          # shape of the returned grid
    fill_value   : scalar         # fille value

    Optional Arguments
    ------------------
    dtype        : str/np.dtype                  # type of the returned grid
    yorigin      : int/float                     # y-value of the grid's origin
    xorigin      : int/float                     # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"}         # position of the origin. One of:
    fill_value   : inf/float                     # fill or fill value
    cellsize     : int/float or 2-tuple of those # cellsize, cellsizes in y and x direction
    proj         : dict/None                     # proj4 projection parameters

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with fill_value.
    """

    return GeoArray(
        data       = np.full(shape, value, dtype),
        yorigin    = yorigin,
        xorigin    = xorigin,
        origin     = origin,
        fill_value = fill_value,
        cellsize   = cellsize,
        proj       = proj,
        mode       = mode)


def empty(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
          fill_value=None, cellsize=1, proj=None, mode=None):
    """
    Arguments
    ----------
    shape        : tuple          # shape of the returned grid

    Optional Arguments
    ------------------
    dtype        : str/np.dtype                  # type of the returned grid
    yorigin      : int/float                     # y-value of the grid's origin
    xorigin      : int/float                     # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"}         # position of the origin. One of:
    fill_value   : inf/float                     # fill or fill value
    cellsize     : int/float or 2-tuple of those # cellsize, cellsizes in y and x direction
    proj         : dict/None                     # proj4 projection parameters

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new empty GeoArray of given shape and type
    """

    return GeoArray(
        data       = np.empty(shape, dtype),
        yorigin    = yorigin,
        xorigin    = xorigin,
        origin     = origin,
        fill_value = fill_value,
        cellsize   = cellsize,
        proj       = proj,
        mode       = mode,
    )


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
    
    return array(**_fromFile(fname, mode))

