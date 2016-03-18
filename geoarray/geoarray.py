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
from gdalfuncs import _fromFile, _toFile, _memDataset, _fromDataset, _Projection, _Transformer

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

def array(data, dtype=None, yorigin=0, xorigin=0, origin="ul",
          fill_value=None, cellsize=(1,1), proj=None):
    """
    Arguments
    ---------
    data         : numpy.ndarray  # data to wrap

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
    proj  : dict/None                     # proj4 projection parameters

    Returns
    -------
    GeoArray

    Purpose
    -------
    Create a GeoArray from data.

    Examples
    --------
    >>> import geoarray as ga

    >>> yorigin      = 63829.3
    >>> xorigin      = 76256.6
    >>> origin       = "ul"
    >>> fill_value = -9
    >>> cellsize     = 55
    >>> data         = np.array([[-9 ,-9, -9, -9, -9, -9],
    ...                          [-9 , 4,  4,  0,  2, -9],
    ...                          [-9 , 0,  5,  8,  5, -9],
    ...                          [-9 , 0,  0,  1,  0, -9],
    ...                          [-9 , 2,  3,  3,  3, -9],
    ...                          [-9 , 0,  1,  0,  6, -9],
    ...                          [-9 , 0,  3,  3,  3, -9],
    ...                          [-9 , 4,  6,  2,  4, -9],
    ...                          [-9 , 2,  1,  0,  1, -9],
    ...                          [-9 ,-9, -9, -9, -9, -9],])

    >>> grid = ga.array(data,yorigin=yorigin,xorigin=xorigin,fill_value=fill_value,cellsize=cellsize)
    >>> print(grid)
    GeoArray([[-- -- -- -- -- --]
              [-- 4 4 0 2 --]
              [-- 0 5 8 5 --]
              [-- 0 0 1 0 --]
              [-- 2 3 3 3 --]
              [-- 0 1 0 6 --]
              [-- 0 3 3 3 --]
              [-- 4 6 2 4 --]
              [-- 2 1 0 1 --]
              [-- -- -- -- -- --]])

    """
    return _factory(
        np.asarray(data) if not dtype else np.asarray(data, dtype),
        yorigin, xorigin, origin, fill_value, cellsize, proj
    )

def zeros(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
          fill_value=None, cellsize=1, proj=None):
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

    Examples
    --------
    >>> import geoarray as ga
    >>> print(ga.zeros((4,4)))
    GeoArray([[0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]])
    """
    return _factory(
        np.zeros(shape, dtype), yorigin, xorigin,
        origin, fill_value, cellsize, proj
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

    Examples
    --------
    >>> import numpy as np
    >>> import geoarray as ga

    >>> x = np.arange(6).reshape((3,2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5]])

    >>> print(ga.zeros_like(x))
    GeoArray([[0 0]
              [0 0]
              [0 0]])

    >>> y = ga.array(x,yorigin=-5555,xorigin=4444,cellsize=42)
    >>> print(y)
    GeoArray([[0 1]
              [2 3]
              [4 5]])

    >>> z = ga.zeros_like(y)
    >>> print(z)
    GeoArray([[0 0]
              [0 0]
              [0 0]])

    >>> z.header == y.header
    True

    """
    try:
        return array(np.zeros_like(a, *args, **kwargs), **a.header)
    except AttributeError:
        return array(np.zeros_like(a, *args, **kwargs))


def ones(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
         fill_value=None, cellsize=1, proj=None):
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

    Examples
    --------
    >>> import geoarray as ga
    >>> print(ga.ones((4,4)))
    GeoArray([[1.0 1.0 1.0 1.0]
              [1.0 1.0 1.0 1.0]
              [1.0 1.0 1.0 1.0]
              [1.0 1.0 1.0 1.0]])
    """

    return _factory(
        np.ones(shape,dtype), yorigin, xorigin,
        origin, fill_value, cellsize, proj
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

    Examples
    --------
    >>> import numpy as np
    >>> import geoarray as ga

    >>> x = np.arange(6).reshape((3,2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5]])

    >>> print(ga.ones_like(x))
    GeoArray([[1 1]
              [1 1]
              [1 1]])

    >>> y = ga.array(x,yorigin=-5555,xorigin=4444,cellsize=42)
    >>> print(y)
    GeoArray([[0 1]
              [2 3]
              [4 5]])

    >>> z = ga.ones_like(y)
    >>> print(z)
    GeoArray([[1 1]
              [1 1]
              [1 1]])

    >>> z.header == y.header
    True

    """
    try:
        return array(np.ones_like(a,*args,**kwargs),**a.header)
    except AttributeError:
        return array(np.ones_like(a,*args,**kwargs))


def full(shape, value, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
         fill_value=None, cellsize=1, proj=None):
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
    proj  : dict/None                     # proj4 projection parameters

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with fill_value.

    Examples
    --------
    >>> import geoarray as ga
    >>> print(ga.full((4,4), 42))
    GeoArray([[42.0 42.0 42.0 42.0]
              [42.0 42.0 42.0 42.0]
              [42.0 42.0 42.0 42.0]
              [42.0 42.0 42.0 42.0]])
    """
    return _factory(
        np.full(shape, value, dtype), yorigin, xorigin,
        origin, fill_value, cellsize, proj
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

    Examples
    --------
    >>> import numpy as np
    >>> import geoarray as ga

    >>> x = np.arange(6).reshape((3,2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5]])

    >>> print(ga.full_like(x,42))
    GeoArray([[42 42]
              [42 42]
              [42 42]])

    >>> y = (ga.array(x,yorigin=-5555,xorigin=4444,cellsize=42))
    >>> print(y)
    GeoArray([[0 1]
              [2 3]
              [4 5]])

    >>> z = ga.full_like(y,42)
    >>> print(z)
    GeoArray([[42 42]
              [42 42]
              [42 42]])

    >>> z.header == y.header
    True

    """
    try:
        return array(np.full_like(a,fill_value,*args,**kwargs),**a.header)
    except AttributeError:
        return array(np.full_like(a,fill_value,*args,**kwargs))

def empty(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
          fill_value=None, cellsize=1, proj=None):
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
        np.empty(shape, dtype), yorigin, xorigin,
        origin, fill_value, cellsize, proj
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
        return array(np.full_like(a,a.fill_value,*args,**kwargs),**a.header)
    except AttributeError:
        return array(np.full_like(a,-9999),*args,**kwargs)


def _factory(data, yorigin, xorigin, origin, fill_value, cellsize, proj):
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
        data, yorigin, xorigin, origin, cellsize,
        _Projection(proj),
        mask=mask, fill_value=fill_value
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
    return _factory(**_fromFile(fname))
    
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
    
    --------
    array : construct a GeoArray.
    zeros : construct a GeoArray, each element of which is zero.
    ones  : construct a GeoArray, each element of which is one.
    empty : construct a GeoArray, each element of which is fill_value.
    full  : construct a GeoArray, each element of which is a given value.

    Examples
    --------
    >>> import numpy as np
    >>> import geoarray as ga

    >>> data = np.array([[-9 ,-9, -9, -9, -9],
    ...                  [-9 , 0,  5,  8, -9],
    ...                  [-9 , 0,  0,  1, -9],
    ...                  [-9 , 2,  3,  3, -9],
    ...                  [-9 , 0,  1,  0, -9],
    ...                  [-9 , 0,  3,  3, -9],
    ...                  [-9 ,-9, -9, -9, -9],])

    >>> grid = ga.GeoArray(data,yorigin=63829.3,xorigin=76256.6,origin="ul",
    ...                    fill_value=-9,cellsize=(55, 55))

    >>> print(grid)
    GeoArray([[-9 -9 -9 -9 -9]
              [-9  0  5  8 -9]
              [-9  0  0  1 -9]
              [-9  2  3  3 -9]
              [-9  0  1  0 -9]
              [-9  0  3  3 -9]
              [-9 -9 -9 -9 -9]])

    # all numpy operators are supported
    >>> print(grid + 5)
    GeoArray([[-4 -4 -4 -4 -4]
              [-4  5 10 13 -4]
              [-4  5  5  6 -4]
              [-4  7  8  8 -4]
              [-4  5  6  5 -4]
              [-4  5  8  8 -4]
              [-4 -4 -4 -4 -4]])

    # all numpy methods are supported
    >>> print(np.exp(grid).astype(np.int64))
    GeoArray([[0 0 0 0 0]
              [0 1 148 2980 0]
              [0 1 1 2 0]
              [0 7 20 20 0]
              [0 1 2 1 0]
              [0 1 20 20 0]
              [0 0 0 0 0]])

    >>> np.sum(grid)
    -151

    # all np.ndarray methods are supported
    >>> grid.max()
    8

    >>> grid.argsort()
    array([ 0, 32, 31, 30, 29, 25, 24, 20, 19, 33, 15, 14, 34, 10,  1,  9,  2,
            3,  4,  5, 23, 11, 21, 26, 12,  6, 13, 22, 16, 27, 28, 18, 17,  7,
            8])

    >>> print(grid.clip(0,3))
    GeoArray([[0 0 0 0 0]
              [0 0 3 3 0]
              [0 0 0 1 0]
              [0 2 3 3 0]
              [0 0 1 0 0]
              [0 0 3 3 0]
              [0 0 0 0 0]])

    # all np.ndarray attributes are supported
    >>> print(grid.T)
    GeoArray([[-9 -9 -9 -9 -9 -9 -9]
              [-9  0  0  2  0  0 -9]
              [-9  5  0  3  1  3 -9]
              [-9  8  1  3  0  3 -9]
              [-9 -9 -9 -9 -9 -9 -9]])

    >>> grid.nbytes
    280

    >>> grid.shape
    (7, 5)
    """

    def __new__(cls, data, yorigin, xorigin, origin,
                cellsize, proj=None, *args, **kwargs):

        obj = np.ma.MaskedArray.__new__(cls, data, *args, **kwargs)

        obj._optinfo["yorigin"]     = yorigin
        obj._optinfo["xorigin"]     = xorigin
        obj._optinfo["origin"]      = origin
        obj._optinfo["cellsize"]    = cellsize
        obj._optinfo["proj"] = proj

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

        Examples
        --------
        >>> import geoarray as ga
        >>> x = ga.full((4,4), 42, yorigin=100, xorigin=55, origin="ur", fill_value=-9999)
        >>> x.header
        {'origin': 'ur', 'fill_value': -9999.0, 'cellsize': (-1, -1), 'yorigin': 100, 'proj': {}, 'xorigin': 55}
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

        Examples
        --------
        >>> import geoarray as ga
        >>> x = ga.full((4,4),42,yorigin=100,xorigin=55,origin="ur")
        >>> x.bbox
        {'xmin': 51, 'ymin': 96, 'ymax': 100, 'xmax': 55}
        """
        
        # print self.cellsize
        yvals = (self.yorigin, self.yorigin + self.nrows*self.cellsize[0])
        xvals = (self.xorigin, self.xorigin + self.ncols*self.cellsize[1])
        return {
            "ymin": min(yvals), "ymax": max(yvals),
            "xmin": min(xvals), "xmax": max(xvals),
        }

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

        Examples
        --------
        >>> import numpy as np
        >>> import geoarray as ga

        >>> x = np.arange(40).reshape((2,4,5))
        >>> grid = ga.array(x)
        >>> grid.shape
        (2, 4, 5)
        >>> grid.nbands
        2

        >>> x = np.arange(120).reshape((3,2,4,5))
        >>> grid = ga.array(x)
        >>> grid.shape
        (3, 2, 4, 5)
        >>> grid.nbands
        2
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

        Examples
        --------
        >>> import numpy as np
        >>> import geoarray as ga

        >>> x = np.arange(40).reshape((2,4,5))
        >>> grid = ga.array(x)
        >>> grid.shape
        (2, 4, 5)
        >>> grid.nrows
        4

        >>> x = np.arange(120).reshape((3,2,4,5))
        >>> grid = ga.array(x)
        >>> grid.shape
        (3, 2, 4, 5)
        >>> grid.nrows
        4
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

        Examples
        --------
        >>> import numpy as np
        >>> import geoarray as ga

        >>> x = np.arange(40).reshape((2,4,5))
        >>> grid = ga.array(x)
        >>> grid.shape
        (2, 4, 5)
        >>> grid.ncols
        5

        >>> x = np.arange(120).reshape((3,2,4,5))
        >>> grid = ga.array(x)
        >>> grid.shape
        (3, 2, 4, 5)
        >>> grid.ncols
        5
        """

        try:
            return self.shape[-1]
        except IndexError:
            return 0

    def tofile(self,fname):
        """
        Arguments
        ---------
        fname : str  # file name

        Returns
        -------
        None

        Purpose
        -------
        Write GeoArray to file. The output dataset type is derived from
        the file name extension. See _DRIVER_DICT for implemented formats.
        """
        _toFile(fname, self)

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

        Examples
        --------
        >>> import numpy as np
        >>> import geoarray as ga
        >>> data = np.array([[-9 ,-9, -9, -9, -9],
        ...                  [-9 , 4,  0,  2, -9],
        ...                  [-9 , 3,  3,  3, -9],
        ...                  [-9 , 1,  0,  6, -9],
        ...                  [-9 , 1,  0,  1, -9],
        ...                  [-9 ,-9, -9, -9, -9],])

        >>> grid = ga.array(data,fill_value=-9)
        >>> print(grid)
        GeoArray([[-- -- -- -- --]
                  [-- 4 0 2 --]
                  [-- 3 3 3 --]
                  [-- 1 0 6 --]
                  [-- 1 0 1 --]
                  [-- -- -- -- --]])

        >>> print(grid.trim())
        GeoArray([[4 0 2]
                  [3 3 3]
                  [1 0 6]
                  [1 0 1]])
        """

        y_idx, x_idx = np.where(self.data != self.fill_value)
        try:
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

        Example
        -------
        >>> import geoarray as ga
        >>> x = ga.array(np.arange(36).reshape(6,6))
        >>> print(x)
        GeoArray([[0 1 2 3 4 5]
                  [6 7 8 9 10 11]
                  [12 13 14 15 16 17]
                  [18 19 20 21 22 23]
                  [24 25 26 27 28 29]
                  [30 31 32 33 34 35]])

        >>> print(x.removeCells(top=1,left=2,bottom=2,right=1))
        GeoArray([[8 9 10]
                  [14 15 16]
                  [20 21 22]])
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

        Examples
        --------
        >>> import numpy as np
        >>> import geoarray as ga

        >>> x = np.arange(20).reshape((4,5))
        >>> grid = ga.array(x,origin="ll",cellsize=20)

        >>> grid.bbox
        {'xmin': 0, 'ymin': 0, 'ymax': 80, 'xmax': 100}
        >>> grid.shape
        (4, 5)
        >>> print(grid)
        GeoArray([[0 1 2 3 4]
                  [5 6 7 8 9]
                  [10 11 12 13 14]
                  [15 16 17 18 19]])

        >>> shrinked = grid.shrink(ymin=18,ymax=56,xmin=22,xmax=92)
        >>> shrinked.bbox
        {'xmin': 20, 'ymin': 0, 'ymax': 60, 'xmax': 100}
        >>> shrinked.shape
        (3, 4)
        >>> print(shrinked)
        GeoArray([[6 7 8 9]
                  [11 12 13 14]
                  [16 17 18 19]])
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

        Example
        -------
        >>> import geoarray as ga

        >>> grid = ga.full((4,5), 42, fill_value=-9)
        >>> print(grid)
        GeoArray([[42.0 42.0 42.0 42.0 42.0]
                  [42.0 42.0 42.0 42.0 42.0]
                  [42.0 42.0 42.0 42.0 42.0]
                  [42.0 42.0 42.0 42.0 42.0]])

        >>> print(grid.addCells(top=1,left=1,bottom=2))
        GeoArray([[-- -- -- -- -- --]
                  [-- 42.0 42.0 42.0 42.0 42.0]
                  [-- 42.0 42.0 42.0 42.0 42.0]
                  [-- 42.0 42.0 42.0 42.0 42.0]
                  [-- 42.0 42.0 42.0 42.0 42.0]
                  [-- -- -- -- -- --]
                  [-- -- -- -- -- --]])
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

        Examples
        --------
        >>> import numpy as np
        >>> import geoarray as ga

        >>> x = np.arange(20).reshape((4,5))
        >>> grid = ga.array(x, yorigin=100, xorigin=200, origin="ll", cellsize=20, fill_value=-9)

        >>> grid.bbox
        {'xmin': 200, 'ymin': 100, 'ymax': 180, 'xmax': 300}
        >>> grid.shape
        (4, 5)
        >>> print(grid)
        GeoArray([[0 1 2 3 4]
                  [5 6 7 8 9]
                  [10 11 12 13 14]
                  [15 16 17 18 19]])

        >>> enlarged = grid.enlarge(xmin=130, xmax=200, ymin=66)
        >>> enlarged.bbox
        {'xmin': 120, 'ymin': 60, 'ymax': 180, 'xmax': 300}

        >>> print(enlarged)
        GeoArray([[-- -- -- -- 0 1 2 3 4]
                  [-- -- -- -- 5 6 7 8 9]
                  [-- -- -- -- 10 11 12 13 14]
                  [-- -- -- -- 15 16 17 18 19]
                  [-- -- -- -- -- -- -- -- --]
                  [-- -- -- -- -- -- -- -- --]])
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

    #     Examples
    #     --------
    #     >>> import numpy as np
    #     >>> import geoarray as ga

    #     >>> x = np.arange(20).reshape((4,5))
    #     >>> grid1 = ga.array(x,origin="ll",cellsize=25)
    #     >>> grid1.bbox
    #     {'xmin': 0, 'ymin': 0, 'ymax': 100, 'xmax': 125}

    #     >>> grid2 = ga.array(x,yorigin=3,xorigin=1.24,origin="ll",cellsize=18.67)
    #     >>> grid2.bbox
    #     {'xmin': 1.24, 'ymin': 3, 'ymax': 77.68, 'xmax': 94.59}

    #     >>> grid2.snap(grid1)
    #     >>> grid2.bbox
    #     {'xmin': 0.0, 'ymin': 0.0, 'ymax': 74.680000000000007, 'xmax': 93.350000000000009}
    #     """

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

    def warp(self, proj, max_error=0.125):
        """
        Arguments
        ---------
        proj       : dict   -> proj4 parameters of the target coordinate system
        max_error  : float  -> Maximum error (in pixels) allowed in transformation
                               approximation (default: value of gdalwarp)
        
        Return
        ------
        GeoArray
        
        Todo
        ----
        - Make the resampling strategy an optional argument
        """

        # transform corners
        bbox = self.bbox
        trans = _Transformer(self.proj, _Projection(proj))
        uly, ulx = trans(bbox["ymax"], bbox["xmin"])
        lry, lrx = trans(bbox["ymin"], bbox["xmax"])
        ury, urx = trans(bbox["ymax"], bbox["xmax"])
        lly, llx = trans(bbox["ymin"], bbox["xmin"])

        # Calculate cellsize, i.e. same number of cells along the diagonal.
        sdiag = np.sqrt(self.nrows**2 + self.ncols**2)
        # tdiag = np.sqrt((uly - lry)**2 + (lrx - ulx)**2)
        tdiag = np.sqrt((lly - ury)**2 + (llx - urx)**2)
        tcellsize = tdiag/sdiag

        # number of cells
        ncols = int(abs(round((max(urx, lrx) - min(ulx, llx))/tcellsize)))
        nrows = int(abs(round((max(ury, lry) - min(uly, lly))/tcellsize)))
        
        target = full(
            shape      = (self.nbands, nrows, ncols),
            value      = self.fill_value,
            fill_value = self.fill_value,
            dtype      = self.dtype,
            yorigin    = max(uly, ury, lly, lry),
            xorigin    = min(ulx, urx, llx, lrx),
            origin     = "ul",
            cellsize   = (-tcellsize, tcellsize),
            proj       = proj
        )

        return self.warpTo(target, max_error)
        
    def warpTo(self, grid, max_error=0.125):
        """
        Arguments
        ---------
        grid: GeoArray

        Return
        ------
        GeoArray

        Purpose
        -------
        Interpolates self to the target grid, including
        coordinate transformations if necessary.
        """
        
        out = _memDataset(grid)
        resampling = gdal.GRA_NearestNeighbour
           
        res = gdal.ReprojectImage(
            _memDataset(self), out,
            None, None,
            resampling, 
            0.0, max_error)
        
        return _factory(**_fromDataset(out))
  
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
            self.data.copy(), self.dtype, self.yorigin, self.xorigin, self.origin,
            self.fill_value, self.cellsize, self.proj.getWkt()
        )
        
    def __getitem__(self, slc):

        out = super(GeoArray,self).__getitem__(slc)
        slices = getSlices(slc,self.shape)

        try:
            yorigin, xorigin = self.getOrigin("ul")
            if self.origin[0] == "u":
                if slices[-2].first:
                    out.yorigin = yorigin + slices[-2].first * self.cellsize[0]
            else:
                if slices[-2].last:
                    out.yorigin = yorigin - (slices[-2].last + 1) * self.cellsize[0]
            if self.origin[1] == "l":
                if slices[-1].first:
                    out.xorigin = xorigin + slices[-1].first * self.cellsize[1]
            else:
                if slices[-1].last:
                    out.xorigin = xorigin - (slices[-1].last + 1) * self.cellsize[1]

        except AttributeError: # out is scalar
            pass

        return out

    
if __name__ == "__main__":

    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
