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
from math import floor, ceil
from .utils import _broadcastedMeshgrid, _broadcastTo
from .gdaltrans import _Projection, _Geotrans
from .gdalio import _getDataset, _toFile, _writeData


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

class GeotransMixin(object):

    @property
    def yorigin(self):
        return self.geotrans.yorigin

    @property
    def xorigin(self):
        return self.geotrans.xorigin

    @property
    def cellsize(self):
        return (self.geotrans.ycellsize, self.geotrans.xcellsize)

    @property
    def ycellsize(self):
        return self.geotrans.ycellsize

    @property
    def xcellsize(self):
        return self.geotrans.xcellsize

    @property
    def origin(self):
        return "".join(
            ["l" if self.ycellsize > 0 else "u",
             "l" if self.xcellsize > 0 else "r"])

    @property
    def bbox(self):

        corners = np.array(self.getCorners())
        ymin, xmin = np.min(corners, axis=0)
        ymax, xmax = np.max(corners, axis=0)

        return {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax}


    def _calcCoordinate(self, row, col):
        yval = (self.geotrans.yorigin
                + col * self.geotrans.yparam
                + row * self.geotrans.ycellsize)
        xval = (self.geotrans.xorigin
                + col * self.geotrans.xcellsize
                + row * self.geotrans.xparam)
        return yval, xval

    @property
    def coordinates(self):
        # NOTE: rather costly, should be cached
        xdata, ydata = np.meshgrid(
            np.arange(self.ncols, dtype=float),
            np.arange(self.nrows, dtype=float))
        return self._calcCoordinate(ydata, xdata)

    def getCorners(self):
        corners = [(0, 0), (self.nrows, 0),
                   (0, self.ncols), (self.nrows, self.ncols)]
        return [self._calcCoordinate(*idx) for idx in corners]

    def getCorner(self, corner=None):
        """
        Arguments
        ---------
        corner : str/None

        Returns
        -------
        (scalar, scalar)

        Purpose
        -------
        Return the grid's corner coordinates. Defaults to the origin
        corner. Any other corner may be specifed with the 'origin' argument,
        which should be one of: 'ul','ur','ll','lr'.
        """

        if not corner:
            corner = self.origin

        bbox = self.bbox
        return (
            bbox["ymax"] if corner[0] == "u" else bbox["ymin"],
            bbox["xmax"] if corner[1] == "r" else bbox["xmin"],)


class GeoArray(GeotransMixin, MaskedArray):
    """
    Arguments
    ----------
    data         : np.ndarray/list/tuple
    yorigin      : scalar                # y-coordinate of origin
    xorigin      : scalar                # x-coordinate of origin
    origin       : {"ul","ur","ll","lr"} # position of the grid origin
    fill_value   : scalar
    cellsize     : (scalar, scalar)
    fobj         : return object from gdal.Open or None
    proj         : _Projection           # projection information
    color_mode   : string
    mode         : AnyStr

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
            cls, data, geotrans,
            proj=None, fill_value=None, fobj=None, color_mode=None,  # mask=None,
            mode="r", *args, **kwargs):

        # NOTE: The mask will always be calculated, even if its
        #       already present or not needed at all...
        mask = (np.zeros_like(data, np.bool)
                if fill_value is None else data == fill_value)

        obj = MaskedArray.__new__(
            cls, data=data, fill_value=fill_value, mask=mask, *args, **kwargs)
        obj.unshare_mask()

        obj._optinfo["geotrans"] = _Geotrans(**geotrans)
        obj._optinfo["proj"] = _Projection(proj)
        obj._optinfo["fill_value"] = fill_value
        obj._optinfo["color_mode"] = color_mode
        obj._optinfo["mode"] = mode
        obj._optinfo["_fobj"] = fobj

        return obj

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

    def coordinatesOf(self, y_idx, x_idx):
        """
        Arguments
        ---------
        y_idx, x_idx :  int

        Returns
        -------
        (scalar, scalar)

        Purpose
        -------
        Return the coordinates of the grid cell definied by the given
        row and column index values. The cell corner to which the returned
        values belong is definied
        by the attribute origin:
            "ll": lower-left corner
            "lr": lower-right corner
            "ul": upper-left corner
            "ur": upper-right corner
        """

        if ((y_idx < 0 or x_idx < 0)
            or (y_idx >= self.nrows
                or x_idx >= self.ncols)):
            raise ValueError("Index out of bounds !")

        yorigin, xorigin = self.getCorner("ul")
        return (
            yorigin - y_idx * abs(self.cellsize[0]),
            xorigin + x_idx * abs(self.cellsize[1]))

    def indexOf(self, ycoor, xcoor):
        """
        Arguments
        ---------
        ycoor, xcoor : scalar

        Returns
        -------
        (int, int)

        Purpose
        -------
        Find the grid cell into which the given coordinates
        fall and return its row/column index values.
        """

        yorigin, xorigin = self.getCorner("ul")
        cellsize = np.abs(self.cellsize)
        yidx = int(floor((yorigin - ycoor) / float(cellsize[0])))
        xidx = int(floor((xcoor - xorigin) / float(cellsize[1])))

        if yidx < 0 or yidx >= self.nrows or xidx < 0 or xidx >= self.ncols:
            raise ValueError("Given Coordinates not within the grid domain!")

        return yidx, xidx

    def fill(self, fill_value):
        """
        works similar to MaskedArray.filled(value) but also changes
        the fill_value and returns an GeoArray instance
        """
        return GeoArray(
            data=self.filled(fill_value),
            geotrans=geotrans,
            proj=self.proj,
            fill_value=fill_value,
            mode = self.mode,
            color_mode=self.color_mode)

    def trim(self):
        """
        Arguments
        ---------
        None

        Returns
        -------
        GeoArray

        Purpose
        -------
        Removes rows and columns from the margins of the
        grid if they contain only fill values.
        """

        try:
            y_idx, x_idx = np.where(self.data != self.fill_value)
            return self.removeCells(
                top=min(y_idx), bottom=self.nrows - max(y_idx) - 1,
                left=min(x_idx), right=self.ncols - max(x_idx) - 1)
        except ValueError:
            return self

    def removeCells(self, top=0, left=0, bottom=0, right=0):
        """
        Arguments
        ---------
        top, left, bottom, right : int

        Returns
        -------
        GeoArray

        Purpose
        -------
        Remove the number of given cells from the respective
        margin of the grid.
        """

        top = int(max(top, 0))
        left = int(max(left, 0))
        bottom = self.nrows - int(max(bottom, 0))
        right = self.ncols - int(max(right, 0))

        return self[..., top:bottom, left:right]

    def shrink(self, ymin=None, ymax=None, xmin=None, xmax=None):
        """
        Arguments
        ---------
        ymin, ymax, xmin, xmax : scalar

        Returns
        -------
        GeoArray

        Purpose
        -------
        Shrinks the grid in a way that the given bbox is still
        within the grid domain.

        BUG:
        ------------
        For bbox with both negative and postive values
        """
        bbox = {
            "ymin": ymin if ymin is not None else self.bbox["ymin"],
            "ymax": ymax if ymax is not None else self.bbox["ymax"],
            "xmin": xmin if xmin is not None else self.bbox["xmin"],
            "xmax": xmax if xmax is not None else self.bbox["xmax"],
            }

        cellsize = [float(abs(cs)) for cs in self.cellsize]
        top = floor((self.bbox["ymax"] - bbox["ymax"]) / cellsize[0])
        left = floor((bbox["xmin"] - self.bbox["xmin"]) / cellsize[1])
        bottom = floor((bbox["ymin"] - self.bbox["ymin"]) / cellsize[0])
        right = floor((self.bbox["xmax"] - bbox["xmax"]) / cellsize[1])

        return self.removeCells(
            max(top, 0), max(left, 0), max(bottom, 0), max(right, 0))

    def addCells(self, top=0, left=0, bottom=0, right=0):
        """
        Arguments
        ---------
        top, left, bottom, right : int

        Returns
        -------
        GeoArray

        Purpose
        -------
        Add the number of given cells to the respective margin of the grid.
        """

        top = int(max(top, 0))
        left = int(max(left, 0))
        bottom = int(max(bottom, 0))
        right = int(max(right, 0))

        if self.origin[0] == "l":
            top, bottom = bottom, top
        if self.origin[1] == "r":
            left, right = right, left

        shape = list(self.shape)
        shape[-2:] = self.nrows + top + bottom, self.ncols + left + right

        try:
            data = np.full(shape, self.fill_value, self.dtype)
        except TypeError:
            # fill_value is set to none
            raise AttributeError(
                "Valid fill_value needed, actual value is {:}"
                .format(self.fill_value))

        geotrans = self.geotrans.copy(
            yorigin=self.geotrans.yorigin + top*self.geotrans.ycellsize * -1,
            xorigin=self.geotrans.xorigin + left*self.geotrans.xcellsize * -1)

        out = GeoArray(
            data=data,
            dtype=self.dtype,
            geotrans=geotrans,
            fill_value=self.fill_value,
            mode="r",
            fobj=None,
            proj=self.proj,
            color_mode=self.color_mode)

        # the Ellipsis ensures that the function works
        # for arrays with more than two dimensions
        out[..., top:top+self.nrows, left:left+self.ncols] = self
        return out

    def enlarge(self, ymin=None, ymax=None, xmin=None, xmax=None):
        """
        Arguments
        ---------
        ymin, ymax, xmin, xmax : scalar

        Returns
        -------
        None

        Purpose
        -------
        Enlarge the grid in a way that the given coordinates will
        be part of the grid domain. Added rows/cols are filled with
        the grid's fill value.
        """

        bbox = {
            "ymin": ymin if ymin is not None else self.bbox["ymin"],
            "ymax": ymax if ymax is not None else self.bbox["ymax"],
            "xmin": xmin if xmin is not None else self.bbox["xmin"],
            "xmax": xmax if xmax is not None else self.bbox["xmax"],}

        cellsize = [float(abs(cs)) for cs in self.cellsize]

        top = ceil((bbox["ymax"] - self.bbox["ymax"]) / cellsize[0])
        left = ceil((self.bbox["xmin"] - bbox["xmin"]) / cellsize[1])
        bottom = ceil((self.bbox["ymin"] - bbox["ymin"]) / cellsize[0])
        right = ceil((bbox["xmax"] - self.bbox["xmax"]) / cellsize[1])

        return self.addCells(
            max(top, 0), max(left, 0), max(bottom, 0), max(right, 0))

    # def snap(self,target):
    #     """
    #     Arguments
    #     ---------
    #     target : GeoArray

    #     Returns
    #     -------
    #     None

    #     Purpose
    #     -------
    #     Shift the grid origin that it matches the nearest cell origin in target.

    #     Restrictions
    #     ------------
    #     The shift will only alter the grid coordinates. No changes to the
    #     data will be done. In case of large shifts the physical integrety
    #     of the data might be disturbed!

    #     diff = np.array(self.getCorner()) - np.array(target.getCorner(self.origin))
    #     dy, dx = abs(diff)%target.cellsize * np.sign(diff)

    #     if abs(dy) > self.cellsize[0]/2.:
    #         dy += self.cellsize[0]

    #     if abs(dx) > self.cellsize[1]/2.:
    #         dx += self.cellsize[1]

    #     self.xorigin -= dx
    #     self.yorigin -= dy

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
        return GeoArray(
            data=self.data,
            geotrans=copy.deepcopy(self.geotrans),
            proj=copy.deepcopy(self.proj),
            fill_value=self.fill_value,
            mode=self.mode,
            color_mode=self.color_mode)

    def __deepcopy__(self, memo):
        return GeoArray(
            data=self.data.copy(),
            geotrans=copy.deepcopy(self.geotrans),
            proj=copy.deepcopy(self.proj),
            fill_value=self.fill_value,
            mode="r",
            color_mode=self.color_mode)

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
        geotrans = self.geotrans.copy(
            yorigin=ystart, xorigin=xstart, ycellsize=cellsize[0], xcellsize=cellsize[1])

        return GeoArray(
            data       = data.data,
            geotrans   = geotrans,
            origin     = self.origin,
            proj       = self.proj,
            fill_value = self.fill_value,
            mode       = self.mode,
            color_mode = self.color_mode)

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
