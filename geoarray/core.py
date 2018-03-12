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
            mode="r", *args, **kwargs):

        # NOTE: The mask will always be calculated, even if its
        #       already present or not needed at all...
        mask = (np.zeros_like(data, np.bool)
                if fill_value is None else data == fill_value)


        self = MaskedArray.__new__(
            cls, data=data, fill_value=fill_value, mask=mask, *args, **kwargs)
        self.unshare_mask()

        self._optinfo["yorigin"] = yorigin
        self._optinfo["xorigin"] = xorigin
        
        self._optinfo["ycellsize"] = ycellsize
        self._optinfo["xcellsize"] = xcellsize

        self._optinfo["yparam"] = yparam
        self._optinfo["xparam"] = xparam

        self._optinfo["proj"] = _Projection(proj)
        self._optinfo["fill_value"] = fill_value
        self._optinfo["color_mode"] = color_mode
        self._optinfo["mode"] = mode
        self._optinfo["_fobj"] = fobj

        return self


    @property
    def header(self):
        """
        Arguments
        ---------
        None

        Returns
        -------
        dict

        Purpose
        -------
        Return the basic definition of the grid. Together
        with a numpy.ndarray this information
        can be passed to any of the factory functions.
        """

        return {
            "yorigin": self.yorigin,
            "xorigin": self.xorigin,
            "origin": self.origin,
            "fill_value": self.fill_value,
            "cellsize": self.cellsize,
            "proj": self.proj,
            "color_mode": self.color_mode}

    @property
    def nbands(self):
        """
        Arguments
        ---------
        None

        Returns
        -------
        int

        Purpose
        -------
        Return the number of bands in the dataset, i.e. the third last element
        in the shape tuple.
        """

        try:
            return self.shape[-3]
        except IndexError:
            return 1

    @property
    def nrows(self):
        """
        Arguments
        ---------
        None

        Returns
        -------
        int

        Purpose
        -------
        Return the number of rows in the dataset, i.e. the second last element
        in the shape tuple.
        """

        try:
            return self.shape[-2] or 1
        except IndexError:
            return 1

    @property
    def ncols(self):
        """
        Arguments
        ---------
        None

        Returns
        -------
        int

        Purpose
        -------
        Return the number of columns in the dataset, i.e. the last element
        in the shape tuple.
        """

        try:
            return self.shape[-1] or 1
        except IndexError:
            return 1

    @property
    def fill_value(self):
        return self._optinfo["fill_value"]

    # Work around a bug in np.ma.core present at least until version 1.13.0:
    # The _optinfo dictionary is not updated when calling __eq__/__ne__
    # numpy PR: 9279
    def _comparison(self, other, compare):
        out = super(self.__class__, self)._comparison(other, compare)
        out._update_from(self)
        return out

    @fill_value.setter
    def fill_value(self, value):
        # change fill_value and update mask
        self._optinfo["fill_value"] = value
        self.mask = self == value

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

    def __getattr__(self, key):
        try:
            value = self._optinfo[key]
            # make descriptors work
            if hasattr(value, "__get__"):
                value = value.__get__(None, self)
            return value
        except KeyError:
            raise AttributeError("'{:}' object has no attribute {:}"
                                 .format(self.__class__.__name__, key))

    def __setattr__(self, key, value):
        try:
            attr = getattr(self, key)
            if hasattr(attr, "__set__"):
                attr.__set__(None, value)
            else:
                super(self.__class__, self).__setattr__(key, value)
        except AttributeError:
            self._optinfo[key] = value

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
        fobj = self._optinfo.get("_fobj")
        if fobj is not None:
            fobj.FlushCache()
            if self.mode == "a":
                _writeData(self)

    def close(self):
        self.__del__()

    def __del__(self):
        # the virtual memory mapping needs to be released BEFORE the fobj
        self.flush()
        self._optinfo["data"] = None
        self._optinfo["_fobj"] = None

    def tofile(self, fname):
        _toFile(self, fname)
