#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from geoarray import GeoArray, Projer

# Possible positions of the grid origin
ORIGINS = (
    "ul",    #     "ul" -> upper left
    "ur",    #     "ur" -> upper right
    "ll",    #     "ll" -> lower left
    "lr",    #     "lr" -> lower right
)

def array(data, dtype=None, yorigin=0, xorigin=0, origin="ul",
          fill_value=-9999, cellsize=(1,1), proj=None):
    """
    Parameters
    ----------
    data         : numpy.ndarray  # data to wrap

    Optional Parameters
    -------------------
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
    return _factory(np.asarray(data) if not dtype else np.asarray(data, dtype),
                    yorigin, xorigin, origin, fill_value, cellsize, proj, None)

def zeros(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
          fill_value=-9999, cellsize=1, proj=None):
    """
    Parameters
    ----------
    shape        : tuple          # shape of the returned grid

    Optional Parameters
    -------------------
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
    return _factory(np.zeros(shape, dtype), yorigin, xorigin,
                    origin, fill_value, cellsize, proj, None)

def zeros_like(a, *args, **kwargs):
    """
    Parameters
    ----------
    a           : np.ndarray         # the array to derive the shape and dtype attributes

    Optional Parameters
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
         fill_value=-9999, cellsize=1, proj=None):
    """
    Parameters
    ----------
    shape        : tuple          # shape of the returned grid

    Optional Parameters
    -------------------
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

    return _factory(np.ones(shape,dtype), yorigin, xorigin,
                    origin, fill_value, cellsize, proj, None)

def ones_like(a, *args, **kwargs):
    """
    Parameters
    ----------
    a           : np.ndarray         # the array to derive the shape and dtype attributes

    Optional Parameters
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
         fill_value=-9999, cellsize=1, proj=None):
    """
    Parameters
    ----------
    shape        : tuple          # shape of the returned grid
    fill_value   : scalar         # fille value

    Optional Parameters
    -------------------
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
    return _factory(np.full(shape, value, dtype), yorigin, xorigin,
                    origin, fill_value, cellsize, proj, None)

def full_like(a, fill_value, *args, **kwargs):
    """
    Parameters
    ----------
    a           : np.ndarray         # the array to derive the shape and dtype attributes
    fill_value  : scalar             # fill value

    Optional Parameters
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
          fill_value=-9999, cellsize=1, proj=None):
    """
    Parameters
    ----------
    shape        : tuple          # shape of the returned grid

    Optional Parameters
    ------------------
    dtype        : str/np.dtype                  # type of the returned grid
    yorigin      : int/float                     # y-value of the grid's origin
    xorigin      : int/float                     # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"}         # position of the origin. One of:
                                                 #    "ul" : upper left corner
                                                 #    "ur" : upper right corner
                                                 #    "ll" : lower left corner
                                                 #    "lr" : lower right corner
    fill_value   : inf/float                     # fill or fill value
    cellsize     : int/float or 2-tuple of those # cellsize, cellsizes in y and x direction
    proj  : dict/None                     # proj4 projection parameters

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with fill_value

    Examples
    --------
    >>> import geoarray as ga

    >>> print(ga.empty((4,4)))
    GeoArray([[-- -- -- --]
              [-- -- -- --]
              [-- -- -- --]
              [-- -- -- --]])

    >>> print(ga.empty((4,4),fill_value=32))
    GeoArray([[-- -- -- --]
              [-- -- -- --]
              [-- -- -- --]
              [-- -- -- --]])
    """

    return _factory(np.full(shape, fill_value, dtype), yorigin, xorigin,
                    origin, fill_value, cellsize, proj, None)

def empty_like(a,*args,**kwargs):
    """
    Parameters
    ----------
    a           : np.ndarray         # the array to derive the shape and dtype attributes

    Optional Parameters
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
    Return a empty GeoArray with the same shape and type as a given array.

    Examples
    --------
    >>> import numpy as np
    >>> import geoarray as ga

    >>> x = np.arange(6).reshape((3,2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5]])

    >>> print(ga.empty_like(x))
    GeoArray([[-- --]
              [-- --]
              [-- --]])

    >>> y = ga.array(x,yorigin=-5555,xorigin=4444,fill_value=42)
    >>> print(y)
    GeoArray([[0 1]
              [2 3]
              [4 5]])

    >>> z = ga.empty_like(y)
    >>> print(z)
    GeoArray([[-- --]
              [-- --]
              [-- --]])

    >>> z.header == y.header
    True
    """

    try:
        return array(np.full_like(a,a.fill_value,*args,**kwargs),**a.header)
    except AttributeError:
        return array(np.full_like(a,-9999),*args,**kwargs)


def _factory(data, yorigin, xorigin, origin, fill_value, cellsize, proj, fobj):
    if origin not in ORIGINS:
        raise TypeError("Argument 'origin' must be one of '{:}'".format(ORIGINS))
    try:
        # cellsize[1]
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
        Projer(proj),
        mask=mask, fill_value=fill_value, fobj=fobj
    )

