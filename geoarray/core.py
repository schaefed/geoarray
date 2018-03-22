#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author
------
David Schaefer

Purpose
-------
This module provides a numpy.ma.MaskedArray as a
wrapper around gdal raster functionality

"""
import os
import copy
import numpy as np
import warnings
from numpy.ma import MaskedArray
from .utils import _broadcastedMeshgrid, _broadcastTo
from .gdalspatial import _Projection
from .gdalio import _getDataset, _toFile, _writeData
from .trans import GeotransMixin
from .spatial import SpatialMixin


# Possible positions of the grid origin
ORIGINS = (
    "ul",  # "ul" -> upper left
    "ur",  # "ur" -> upper right
    "ll",  # "ll" -> lower left
    "lr",  # "lr" -> lower right
)

_METHODS = (
    # comparison
    "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__", "__nonzero__",

    # unary
    "__neg__", "__pos__", "__abs__", "__invert__",

    # arithmetic
    "__add__", "__sub__", "__mul__", "__div__", "__truediv__",
    "__floordiv__", "__mod__", "__divmod__", "__pow__", "__lshift__",
    "__rshift__", "__and__", "__or__", "__xor__",  # "__matmul__",

    # arithmetic, in-place
    "__iadd__", "__isub__", "__imul__", "__idiv__", "__itruediv__",
    "__ifloordiv__", "__imod__", "__ipow__", "__ilshift__", "__irshift__",
    "__iand__", "__ior__", "__ixor__",  # "__imatmul__"
)


def _checkMatch(func):
    def inner(*args):
        if len({a.proj for a in args if isinstance(a, GeoArray)}) > 1:
            warnings.warn("Incompatible map projections!", RuntimeWarning)
        if len({a.cellsize for a in args if isinstance(a, GeoArray)}) != 1:
            warnings.warn("Incompatible cellsizes", RuntimeWarning)
        if len({a.getCorner("ul") for a in args
                if isinstance(a, GeoArray)}) != 1:
            warnings.warn("Incompatible origins", RuntimeWarning)
        return func(*args)
    return inner


class GeoArrayMeta(object):
    def __new__(cls, name, bases, attrs):
        for key in _METHODS:
            attrs[key] = _checkMatch(getattr(MaskedArray, key))
        return type(name, bases, attrs)


class GeoArray(GeotransMixin, SpatialMixin, MaskedArray):
    """
    Arguments
    ----------
    TODO

    Purpose
    -------
    This numpy.ndarray subclass adds geographic context to data.
    A (hopfully growing) number of operations on the data I/O to/from
    different file formats (see the variable gdalfuncs._DRIVER_DICT)
    is supported.

    Restrictions
    ------------
    Adding the geographic information to the data does (at the moment)
    not imply any additional logic. If the shapes of two grids allow
    the succesful execution of a certain operator/function your program
    will continue. It is within the responsability of the user to check
    whether a given operation makes sense within a geographic context
    (e.g. grids cover the same spatial domain, share a common projection,
    etc.) or not
    """

    __metaclass__ = GeoArrayMeta

    def __new__(
            cls, data, yorigin=0, xorigin=0, ycellsize=-1, xcellsize=1, yparam=0, xparam=0,
            proj=None, fill_value=None, fobj=None, color_mode=None,  # mask=None,
            yvalues=None, xvalues=None, mode="r", *args, **kwargs):

        # NOTE: The mask will always be calculated, even if its
        #       already present or not needed at all...
        mask = (np.zeros_like(data, np.bool)
                if fill_value is None else data == fill_value)


        self = MaskedArray.__new__(
            cls, data=data, fill_value=fill_value, mask=mask, *args, **kwargs)
        self.unshare_mask()

        self.yorigin = yorigin
        self.xorigin = xorigin

        self.ycellsize = ycellsize
        self.xcellsize = xcellsize

        self.yparam = yparam
        self.xparam = xparam

        self.proj = _Projection(proj)

        self.color_mode = color_mode
        self.mode = mode

        self._fobj = fobj
        self._yvalues = yvalues
        self._xvalues = xvalues
        self._geolocation = None

        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super(GeoArray, self).__array_finalize__(obj)
        self._update_from(obj)

    def _update_from(self, obj):

        super(GeoArray, self)._update_from(obj)

        # TODO: move into a datastructure
        self.yorigin = getattr(obj, "yorigin", None)
        self.xorigin = getattr(obj, "xorigin", None)
        self.ycellsize = getattr(obj, "ycellsize", None)
        self.xcellsize = getattr(obj, "xcellsize", None)
        self.yparam = getattr(obj, "yparam", None)
        self.xparam = getattr(obj, "xparam", None)

        self.proj = getattr(obj, "proj", None)

        self.color_mode = getattr(obj, "color_mode", None)
        self.mode = getattr(obj, "mode", None)

        self._fobj = getattr(obj, "_fobj", None)
        self._yvalues = getattr(obj, "_yvalues", None)
        self._xvalues = getattr(obj, "_xvalues", None)
        self._geolocation = getattr(obj, "_geolocation", None)

    @property
    def header(self):
        out = self._getArgs()
        del out["data"]
        return out

    @property
    def nbands(self):
        try:
            return self.shape[-3]
        except IndexError:
            return 1

    @property
    def nrows(self):
        try:
            return self.shape[-2] or 1
        except IndexError:
            return 1

    @property
    def ncols(self):
        try:
            return self.shape[-1] or 1
        except IndexError:
            return 1

    @property
    def fobj(self):
        if self._fobj is None:
            self._fobj = _getDataset(self, mem=True)
        return self._fobj

    def getFillValue(self):
        return super(GeoArray, self).get_fill_value()

    def setFillValue(self, value):
        # change fill_value and update mask
        super(GeoArray, self).set_fill_value(value)
        self.mask = self == value
        if value != self.fill_value:
            warnings.warn("Data types not compatible. New fill_value is: {:}"
                          .format(self.fill_value))

    # decorating the methods did not work out...
    fill_value = property(fget=getFillValue, fset=setFillValue)

    def __getattribute__(self, key):
        "Make descriptors work"
        v = object.__getattribute__(self, key)
        if hasattr(v, '__get__'):
            return v.__get__(None, self)
        return v

    def __setattr__(self, key, value):
        "Make descriptors work"
        try:
            object.__getattribute__(self, key).__set__(self, key, value)
        except AttributeError:
            object.__setattr__(self, key, value) 

    @property
    def fobj(self):
        if self._fobj is None:
            self._fobj = _getDataset(self, mem=True)
        return self._fobj

    def _getArgs(self, data=None, fill_value=None,
                 yorigin=None, xorigin=None,
                 ycellsize=None, xcellsize=None,
                 yparam=None, xparam=None,
                 mode=None, color_mode=None,
                 proj=None, fobj=None):

        return {
            "data"       : data if data is not None else self.data,
            "yorigin"    : yorigin if yorigin is not None else self.yorigin,
            "xorigin"    : xorigin if xorigin is not None else self.xorigin,
            "ycellsize"  : ycellsize if ycellsize is not None else self.ycellsize,
            "xcellsize"  : xcellsize if xcellsize is not None else self.xcellsize,
            "yparam"     : yparam if yparam is not None else self.yparam,
            "xparam"     : xparam if xparam is not None else self.xparam,
            "proj"       : proj if proj is not None else self.proj,
            "fill_value" : fill_value if fill_value is not None else self.fill_value,
            "mode"       : mode if mode is not None else self.mode,
            "color_mode" : color_mode if color_mode is not None else self.color_mode,
            "fobj"       : fobj if fobj is not None else self._fobj
        }


    def fill(self, fill_value):
        """
        works similar to MaskedArray.filled(value) but also changes
        the fill_value and returns an GeoArray instance
        """
        return GeoArray(
            self._getArgs(data=self.filled(fill_value), fill_value=fill_value))

    def __copy__(self):
        return GeoArray(**self._getArgs())

    def __deepcopy__(self, memo):
        return GeoArray(**self._getArgs(data=self.data.copy()))

    def __getitem__(self, slc):

        data = super(self.__class__, self).__getitem__(slc)

        # empty array
        if data.size == 0:
            return data

        y, x = self.coordinates
        # x, y = _broadcastedMeshgrid(*self.coordinates[::-1])

        bbox = []
        for arr, idx in zip((y, x), (-2, -1)):
            arr = np.array(
                _broadcastTo(arr, self.shape, (-2, -1))[slc],
                copy=False, ndmin=abs(idx)
            )
            s = [0] * arr.ndim
            s[idx] = slice(None, None, None)
            bbox.append((arr[s][0], arr[s][-1]))

        try:
            ystart, ystop = sorted(bbox[0], reverse=data.origin[0] == "u")
            xstart, xstop = sorted(bbox[1], reverse=data.origin[1] == "r")
        except AttributeError:
            # scalar
            return data

        nrows, ncols = ((1, 1) + data.shape)[-2:]
        cellsize = (
            float(ystop-ystart)/(nrows-1) if nrows > 1 else self.cellsize[-2],
            float(xstop-xstart)/(ncols-1) if ncols > 1 else self.cellsize[-1],
        )

        return GeoArray(
            **self._getArgs(data=data.data, yorigin=ystart, xorigin=xstart,
                            ycellsize=cellsize[0], xcellsize=cellsize[1]))

    def flush(self):
        fobj = self._fobj
        if fobj is not None:
            fobj.FlushCache()
            if self.mode == "a":
                _writeData(self)

    def close(self):
        self.__del__()

    def __del__(self):
        # the virtual memory mapping needs to be released BEFORE the fobj
        self.flush()
        self._fobj = None

    def tofile(self, fname):
        _toFile(self, fname)
