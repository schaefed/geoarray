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
from .gdalio import _toDataset, _toFile, _writeData
from .geotrans import _Geotrans
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


class GeoArray(SpatialMixin, MaskedArray):
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
            cls, data, geotrans=None,
            proj=None, fill_value=None, fobj=None, color_mode=None,  # mask=None,
            yvalues=None, xvalues=None, mode="r", *args, **kwargs):

        # NOTE: The mask will always be calculated, even if its
        #       already present or not needed at all...
        mask = (np.zeros_like(data, np.bool)
                if fill_value is None else data == fill_value)

        self = MaskedArray.__new__(
            cls, data=data, fill_value=fill_value, mask=mask, *args, **kwargs)
        self.unshare_mask()

        self.__dict__["geotrans"] = geotrans
        self.__dict__["proj"] = _Projection(proj)

        self.__dict__["color_mode"] = color_mode
        self.__dict__["mode"] = mode

        self.__dict__["_fobj"] = fobj
        self.__dict__["_yvalues"] = yvalues
        self.__dict__["_xvalues"] = xvalues

        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super(GeoArray, self).__array_finalize__(obj)
        self._update_from(obj)

    def _update_from(self, obj):

        super(GeoArray, self)._update_from(obj)

        self.__dict__["geotrans"] = getattr(obj, "geotrans", None)
        self.__dict__["proj"] = getattr(obj, "proj", None)
        self.__dict__["color_mode"] = getattr(obj, "color_mode", None)
        self.__dict__["mode"] = getattr(obj, "mode", None)
        self.__dict__["_fobj"] = getattr(obj, "_fobj", None)
        self.__dict__["_yvalues"] = getattr(obj, "_yvalues", None)
        self.__dict__["_xvalues"] = getattr(obj, "_xvalues", None)
        self.__dict__["_geolocation"] = getattr(obj, "_geolocation", None)

    @property
    def header(self):
        out = self._getArgs()
        out.update(out.pop("geotrans")._todict())
        del out["data"]
        return out

    def _getShapeProperty(self, idx):
        try:
            return self.shape[idx] or 1
        except IndexError:
            return 1

    @property
    def nbands(self):
        return self._getShapeProperty(-3)

    @property
    def nrows(self):
        return self._getShapeProperty(-2)

    @property
    def ncols(self):
        return self._getShapeProperty(-1)

    @property
    def fobj(self):
        if self._fobj is None:
            self._fobj = _toDataset(self, mem=True)
        return self._fobj

    def getFillValue(self):
        return super(GeoArray, self).get_fill_value()

    def setFillValue(self, value):
        # change fill_value and update mask
        super(GeoArray, self).set_fill_value(value)
        self.mask = self.data == value
        if value != self.fill_value:
            warnings.warn(
                "Data types not compatible. New fill_value is: {:}"
                .format(self.fill_value))

    # decorating the methods did not work out...
    fill_value = property(fget=getFillValue, fset=setFillValue)

    # def __getattribute__(self, key):
    #     "Make descriptors work"
    #     v = object.__getattribute__(self, key)
    #     if hasattr(v, '__get__'):
    #         return v.__get__(None, self)
    #     return v

    # def __setattr__(self, key, value):
    #     "Make descriptors work"
    #     try:
    #         object.__getattribute__(self, key).__set__(self, value)
    #     except AttributeError:
    #         object.__setattr__(self, key, value) 

    def __setattr__(self, key, value):
        instance = self.geotrans if hasattr(self.geotrans, key) else self
        object.__setattr__(instance, key, value) 


    def __getattr__(self, key):
        try:
            # return object.__getattribute__(self.geotrans, key)
            return getattr(self.geotrans, key)
        except AttributeError:
            raise AttributeError(
                "'GeoArray' object has no attribute '{:}'".format(key))

    @property
    def fobj(self):
        if self._fobj is None:
            self._fobj = _toDataset(self, mem=True)
        return self._fobj

    def _getArgs(self, data=None, fill_value=None,
                 geotrans=None, mode=None, color_mode=None,
                 proj=None, fobj=None):

        return {
            "data"       : data if data is not None else self.data,
            "geotrans"   : geotrans if geotrans is not None else self.geotrans,
            "proj"       : proj if proj is not None else self.proj,
            "fill_value" : fill_value if fill_value is not None else self.fill_value,
            "mode"       : mode if mode is not None else self.mode,
            "color_mode" : color_mode if color_mode is not None else self.color_mode,
            "fobj"       : fobj if fobj is not None else self._fobj}


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

        data = MaskedArray.__getitem__(self, slc)
        
        # empty array
        if data.size == 0 or np.isscalar(data):
            return data

        geotrans = self.geotrans._getitem(slc)

        return GeoArray(**self._getArgs(data=data.data, geotrans=geotrans))

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
