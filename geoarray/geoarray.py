#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Purpose
-------
This module provides a numpy.ma.MaskedArray subclass and a number of wrapper 
functions to easen the work with array-like data in geographically explicit 
context. The Python GDAL bindings are used for I/O and offer the possibility 
for a future extension of the map projection handling.

Requirements
------------
GDAL >= 1.11
numpy >= 1.8

License
-------

Copyright (C) 2015 David Schaefer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import re, os, sys
import gdal, osr
import numpy as np
from math import floor, ceil
from slicing import getSlices
from gdalfuncs import _fromFile, _toFile, _getDataset, _fromDataset, _Projection, _Transformer, _warp, _warpTo

try:
    xrange
except NameError: # python 3
    xrange = range

# Possible positions of the grid origin
ORIGINS = (
    "ul",    #     "ul" -> upper left
    "ur",    #     "ur" -> upper right
    "ll",    #     "ll" -> lower left
    "lr",    #     "lr" -> lower right
)

def array(data, dtype=None, yorigin=None, xorigin=None, origin=None,
          fill_value=None, cellsize=None, proj=None, mode=None, copy=False):
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
        

    return _factory(
        data       = np.array(data, dtype=dtype, copy=copy), 
        yorigin    = yorigin or 0,
        xorigin    = xorigin or 0,
        origin     = origin or "ul",
        fill_value = fill_value,
        cellsize   = cellsize or (1,1),
        proj       = proj,
        mode       = mode,
    )

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

    return _factory(
        data       = np.zeros(shape, dtype),
        yorigin    = yorigin,
        xorigin    = xorigin,
        origin     = origin,
        fill_value = fill_value,
        cellsize   = cellsize,
        proj       = proj,
        mode       = mode,
    )

def zeros_like(a, *args, **kwargs):
    """
    Arguments
    ---------
    a           : np.ndarray         # the array to derive the shape and dtype attributes

    Optional Arguments
    ------------------
    dtype       : str/np.dtype      # overrides the data stype of the result
    order       : {"C","F","A","K"} # overrides the memory layout of the result
    subok       : bool              # If True, then the newly created array will use the
                                    # sub-class type of ‘a’, otherwise it will be a base-class
                                    # array
    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a GeoArray of zeros with the same shape and type as a given array.

    """

    try:
        return array(data = np.zeros_like(a, *args, **kwargs), **a.header)
    except AttributeError:
        return array(data = np.zeros_like(a, *args, **kwargs))


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

    return _factory(
        data       = np.ones(shape, dtype),
        yorigin    = yorigin,
        xorigin    = xorigin,
        origin     = origin,
        fill_value = fill_value,
        cellsize   = cellsize,
        proj       = proj,
        mode       = mode,
    )

def ones_like(a, *args, **kwargs):
    """
    Arguments
    ---------
    a           : np.ndarray         # the array to derive the shape and dtype attributes

    Optional Arguments
    -------------------
    dtype       : str/np.dtype       # overrides the data stype of the result
    order       : {"C","F","A","K"}  # overrides the memory layout of the result
    subok       : bool               # If True, then the newly created array will use the
                                     # sub-class type of ‘a’, otherwise it will be a base-class
                                     # array
    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a GeoArray of ones with the same shape and type as a given array.
    """

    try:
        return array(data = np.ones_like(a,*args,**kwargs), **a.header)
    except AttributeError:
        return array(data = np.ones_like(a,*args,**kwargs))


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

    return _factory(
        data       = np.full(shape, value, dtype),
        yorigin    = yorigin,
        xorigin    = xorigin,
        origin     = origin,
        fill_value = fill_value,
        cellsize   = cellsize,
        proj       = proj,
        mode       = mode,
    )

def full_like(a, fill_value, *args, **kwargs):
    """
    Arguments
    ----------
    a           : np.ndarray         # the array to derive the shape and dtype attributes
    fill_value  : scalar             # fill value

    Optional Arguments
    -------------------
    dtype       : str/np.dtype      # overrides the data stype of the result
    order       : {"C","F","A","K"} # overrides the memory layout of the result
    subok       : bool              # If True, then the newly created array will use the
                                    # sub-class type of ‘a’, otherwise it will be a base-class
                                    # array
    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a full GeoArray with the same shape and type as a given array.
    """

    try:
        return array(data = np.full_like(a, fill_value, *args, **kwargs), **a.header)
    except AttributeError:
        return array(data = np.full_like(a, fill_value, *args, **kwargs))

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

    return _factory(
        data       = np.empty(shape, dtype),
        yorigin    = yorigin,
        xorigin    = xorigin,
        origin     = origin,
        fill_value = fill_value,
        cellsize   = cellsize,
        proj       = proj,
        mode       = mode,
    )

def empty_like(a, *args, **kwargs):
    """
    Arguments
    ----------
    a           : np.ndarray         # the array to derive the shape and dtype attributes

    Optional Arguments
    -------------------
    dtype       : str/np.dtype       # overrides the data stype of the result
    order       : {"C","F","A","K"}  # overrides the memory layout of the result
    subok       : bool               # If True, then the newly created array will use the
                                     # sub-class type of ‘a’, otherwise it will be a base-class
                                     # array
    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a empty GeoArray with the same shape and type as a given array.
    """

    try:
        return array(data = np.full_like(a, a.fill_value, *args, **kwargs), **a.header)
    except AttributeError:
        return array(data = np.full_like(a, -9999), *args, **kwargs)


def _factory(data, yorigin, xorigin, origin, fill_value, cellsize, proj, fobj=None, mode=None):
    if origin not in ORIGINS:
        raise TypeError("Argument 'origin' must be one of '{:}'".format(ORIGINS))
    try:
        origin = "".join(
            ("l" if cellsize[0] > 0 else "u",
             "l" if cellsize[1] > 0 else "r")
        )
    except (IndexError, TypeError):
        cs = abs(cellsize)
        cellsize = (
            cs if origin[0] == "l" else -cs,
            cs if origin[1] == "l" else -cs
        )
        
    if fill_value is None:
        mask = np.zeros_like(data, np.bool)
    else:
        mask = data==fill_value
    
    return GeoArray(
        data       = data,
        yorigin    = yorigin,
        xorigin    = xorigin,
        origin     = origin,
        cellsize   = cellsize,
        proj       = _Projection(proj),
        fill_value = fill_value,
        mask       = np.zeros_like(data, np.bool) if fill_value is None else data == fill_value,
        mode       = mode or "L",
        fobj       = fobj,
    )

def fromfile(fname):
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
    return _fromFile(fname)

def _dtypeInfo(dtype):
    try:
        tinfo = np.finfo(dtype)
    except ValueError:
        tinfo = np.iinfo(dtype)

    return {"min": tinfo.min, "max": tinfo.max}


class GeoArray(np.ma.MaskedArray):
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
    proj_params  : _Projection           # Projection Instance holding projection information

    Purpose
    -------
    This numpy.ndarray subclass adds geographic context to data.
    A (hopfully growing) number of operations on the data and the writing
    to different file formats (see the variable _DRIVER_DICT) is supported.

    Restrictions
    ------------
    1. A GeoArray instance can be passed to any numpy function expecting a
       numpy.ndarray as argument and, in theory, all these functions should also return
       an object of the same type. In practice however not all functions do so and some
       will return a numpy.ndarray.
    2. Adding the geographic information to the data does (at the moment) not imply
       any additional logic. If the shapes of two grids allow the succesful execution
       of a certain operator/function your program will continue. It is within the responsability
       of the user to check whether a given operation makes sense within a geographic context
       (e.g. grids cover the same spatial domain, share a common projection, etc.) or not.
       Overriding the operators could fix this.
    """

    def __new__(
            cls, data, yorigin, xorigin, origin, cellsize,
            proj=None, fill_value=None, fobj=None, mode=None,
            *args, **kwargs
    ):
        obj = np.ma.MaskedArray.__new__(cls, data, fill_value=fill_value, *args, **kwargs)

        obj._optinfo["yorigin"]    = yorigin
        obj._optinfo["xorigin"]    = xorigin
        obj._optinfo["origin"]     = origin
        obj._optinfo["cellsize"]   = cellsize
        obj._optinfo["proj"]       = proj
        obj._optinfo["fill_value"] = fill_value if fill_value else _dtypeInfo(obj.dtype)["min"]
        obj._optinfo["mode"]       = mode
        obj._optinfo["_fobj"]      = fobj
        
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
            "yorigin"     : self.yorigin,
            "xorigin"     : self.xorigin,
            "origin"      : self.origin,
            "fill_value"  : self.fill_value,
            "cellsize"    : self.cellsize,
            "proj"        : self.proj.getProj4()
        }

    @property
    def bbox(self):
        """
        Arguments
        ---------
        None

        Returns
        -------
        dict

        Purpose
        -------
        Return the grid's bounding box.
        """
        
        yvals = (self.yorigin, self.yorigin + self.nrows*self.cellsize[0])
        xvals = (self.xorigin, self.xorigin + self.ncols*self.cellsize[1])
        return {
            "ymin": min(yvals), "ymax": max(yvals),
            "xmin": min(xvals), "xmax": max(xvals),
        }

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
            return self.shape[-2]
        except IndexError:
            return 0

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
            return self.shape[-1]
        except IndexError:
            return 0

    @property
    def fill_value(self):
        return self._optinfo["fill_value"]
        
    @fill_value.setter
    def fill_value(self, value):
        
        self._optinfo["fill_value"] = value
        self.mask = self == value

    def getOrigin(self, origin=None):
        """
        Arguments
        ---------
        origin : str/None

        Returns
        -------
        (scalar, scalar)

        Purpose
        -------
        Return the grid's corner coordinates. Defaults to the origin
        corner. Any other corner may be specifed with the 'origin' argument,
        which should be one of: 'ul','ur','ll','lr'.
        """

        if not origin:
            origin = self.origin

        bbox = self.bbox
        return (
            bbox["ymax"] if origin[0] == "u" else bbox["ymin"],
            bbox["xmax"] if origin[1] == "r" else bbox["xmin"],
        )

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
        Return the coordinates of the grid cell definied by the given row and column
        index values. The cell corner to which the returned values belong is definied
        by the attribute origin:
            "ll": lower-left corner
            "lr": lower-right corner
            "ul": upper-left corner
            "ur": upper-right corner
        """

        if (y_idx < 0 or x_idx < 0) or (y_idx >= self.nrows or x_idx >= self.ncols):
            raise ValueError("Index out of bounds !")

        yorigin, xorigin = self.getOrigin("ul")
        return (
            yorigin - y_idx * abs(self.cellsize[0]),
            xorigin + x_idx * abs(self.cellsize[1]),
        )
        
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

        yorigin, xorigin = self.getOrigin("ul")
        cellsize = np.abs(self.cellsize)
        yidx = int(floor((yorigin - ycoor)/float(cellsize[0])))
        xidx = int(floor((xcoor - xorigin )/float(cellsize[1])))
        
        if yidx < 0 or yidx >= self.nrows or xidx < 0 or xidx >= self.ncols:
            raise ValueError("Given Coordinates not within the grid domain!")

        return yidx, xidx

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
        grid if they contain only fill value.
        """

        try:
            y_idx, x_idx = np.where(self.data != self.fill_value)
            return self.removeCells(
                top  = min(y_idx), bottom = self.nrows-max(y_idx)-1,
                left = min(x_idx), right  = self.ncols-max(x_idx)-1
            )
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
        Remove the number of given cells from the respective margin of the grid.
        """

        top    = int(max(top,0))
        left   = int(max(left,0))
        bottom = self.nrows - int(max(bottom,0))
        right  = self.ncols - int(max(right,0))

        return self[...,top:bottom,left:right]

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
        Shrinks the grid in a way that the given bbox is still within the grid domain.
        """

        bbox = {
            "ymin": ymin if ymin else self.bbox["ymin"],
            "ymax": ymax if ymax else self.bbox["ymax"],
            "xmin": xmin if xmin else self.bbox["xmin"],
            "xmax": xmax if xmax else self.bbox["xmax"],
            }

        cellsize = map(float, self.cellsize)
        top    = floor((self.bbox["ymax"] - bbox["ymax"])/cellsize[0])
        left   = floor((bbox["xmin"] - self.bbox["xmin"])/cellsize[1])
        bottom = floor((bbox["ymin"] - self.bbox["ymin"])/cellsize[0])
        right  = floor((self.bbox["xmax"] - bbox["xmax"])/cellsize[1])

        return self.removeCells(max(top,0),max(left,0),max(bottom,0),max(right,0))

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
        
        top    = int(max(top,0))
        left   = int(max(left,0))
        bottom = int(max(bottom,0))
        right  = int(max(right,0))

        shape = list(self.shape)
        shape[-2:] = self.nrows + top  + bottom, self.ncols + left + right
        yorigin, xorigin = self.getOrigin("ul")

        out = full(
            shape       = shape,
            value       = self.fill_value,
            dtype       = self.dtype,
            yorigin     = yorigin - top*abs(self.cellsize[0]),
            xorigin     = xorigin - left*abs(self.cellsize[1]),
            origin      = "ul",
            fill_value  = self.fill_value,
            cellsize    = (abs(self.cellsize[0])*-1, abs(self.cellsize[1])),
            proj = self.proj,
        )
        # the Ellipsis ensures that the function works
        # for arrays with more than two dimensions
        out[Ellipsis, top:top+self.nrows, left:left+self.ncols] = self
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
            "ymin": ymin if ymin else self.bbox["ymin"],
            "ymax": ymax if ymax else self.bbox["ymax"],
            "xmin": xmin if xmin else self.bbox["xmin"],
            "xmax": xmax if xmax else self.bbox["xmax"],
            }

        cellsize = map(float, self.cellsize)
        top    = ceil((bbox["ymax"] - self.bbox["ymax"])/cellsize[0])
        left   = ceil((self.bbox["xmin"] - bbox["xmin"])/cellsize[1])
        bottom = ceil((self.bbox["ymin"] - bbox["ymin"])/cellsize[0])
        right  = ceil((bbox["xmax"] - self.bbox["xmax"])/cellsize[1])

        return self.addCells(max(top,0),max(left,0),max(bottom,0),max(right,0))

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

    #     diff = np.array(self.getOrigin()) - np.array(target.getOrigin(self.origin))
    #     dy, dx = abs(diff)%target.cellsize * np.sign(diff)

    #     if abs(dy) > self.cellsize[0]/2.:
    #         dy += self.cellsize[0]

    #     if abs(dx) > self.cellsize[1]/2.:
    #         dx += self.cellsize[1]

    #     self.xorigin -= dx
    #     self.yorigin -= dy

    def basicMatch(self, grid):
        """
        Arguments
        ---------
        grid : GeoArray

        Returns
        -------
        bool

        Purpose
        -------
        Check if two grids are broadcastable.
        """
        return (
            (self.proj == grid.proj) and
            (self.getOrigin() == grid.getOrigin(self.origin)) and
            (self.cellsize == grid.cellsize)
        )

    def __repr__(self):
        return super(self.__class__,self).__repr__()

    def __str__(self):
        out = super(self.__class__,self).__str__()
        name = self.__class__.__name__
        pad = " "*(len(name)+1)
        return "{:}({:})".format(
            name, os.linesep.join(["{:}{:}".format(pad,l) for l in out.split(os.linesep)]).strip()
        )

    def __getattr__(self, name):
        try:
            return self._optinfo[name]
        except KeyError:
            raise AttributeError(
                "'{:}' object has no attribute {:}".format (self.__class__.__name__, name)
            )

    def __deepcopy__(self, memo):
        return array(
            data       = self.data.copy(),
            dtype      = self.dtype,
            yorigin    = self.yorigin,
            xorigin    = self.xorigin,
            origin     = self.origin,
            fill_value = self.fill_value,
            cellsize   = self.cellsize,
            proj       = self.proj.getWkt()
        )
        
    def __getitem__(self, slc):
        out = super(GeoArray, self).__getitem__(slc)
        slices = getSlices(slc, self.shape)
        try:
            yorigin, xorigin = self.getOrigin("ul")
            if self.origin[0] == "u":
                if slices[-2].start:
                    out.yorigin = yorigin + slices[-2].start * self.cellsize[0]
            else:
                if slices[-2].stop:
                    out.yorigin = yorigin - (slices[-2].stop + 1) * self.cellsize[0]
            if self.origin[1] == "l":
                if slices[-1].start:
                    out.xorigin = xorigin + slices[-1].start * self.cellsize[1]
            else:
                if slices[-1].stop:
                    out.xorigin = xorigin - (slices[-1].stop + 1) * self.cellsize[1]

        except AttributeError: # out is scalar
            pass

        return out

    warp   = _warp
    warpTo = _warpTo
    tofile = _toFile
    
if __name__ == "__main__":

    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
