#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""

Purpose
-------
This module provides a numpy.ndarray subclass to work with arrays in a
geographically explicit context.

Requirements
------------
GDAL >= 1.11
numpy >= 1.8

License
-------

This Python module is free software: you can redistribute it and/or modify                                                                                                                             
it under the terms of the GNU Lesser General Public License as published by                                                                                                                                
the Free Software Foundation, either version 3 of the License, or                                                                                                                                          
(at your option) any later version.                                                                                                                                                                        

The UFZ Python package is distributed in the hope that it will be useful,                                                                                                                                  
but WITHOUT ANY WARRANTY; without even the implied warranty of                                                                                                                                             
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                                                                                                                                               
GNU Lesser General Public License for more details.                                                                                                                                                        


Author
------
David Schaefer
"""

import re, os, sys
import gdal, osr
import numpy as np
from math import floor, ceil

MAX_PRECISION  = 10

# could be extended, for available options see:
# http://www.gdal.org/formats_list.html
_DRIVER_DICT = {
    ".tif" : "GTiff",
    ".asc" : "AAIGrid",
    ".img" : "HFA",
}

# type mapping
DTYPE2GDAL = {
    "uint8"      : 1,
    "int8"       : 1,
    "uint16"     : 2,
    "int16"      : 3,
    "uint32"     : 4,
    "int32"      : 5,
    "float32"    : 6,
    "float64"    : 7,
    "complex64"  : 10,
    "complex128" : 11,
}

GDAL2DTYPE = {v:k for k,v in DTYPE2GDAL.items()}

# The open gdal file objects need to outlive their GeoArray
# instance. Therefore they are stored globally.
_FILEREFS = []

gdal.PushErrorHandler('CPLQuietErrorHandler')

def array(data, dtype=None, yorigin=0, xorigin=0, origin="ul",
          fill_value=-9999, cellsize=1, proj_params=None):
    """
    Parameters
    ----------
    data         : numpy.ndarray  # data to wrap

    Optional Parameters
    -------------------
    dtype        : str/np.dtype          # type of the returned grid
    yorigin      : int/float             # y-value of the grid's origin
    xorigin      : int/float             # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"} # position of the origin. One of:
                                         #     "ul" : upper left corner
                                         #     "ur" : upper right corner
                                         #     "ll" : lower left corner
                                         #     "lr" : lower right corner
    fill_value : inf/float             # fill or fill value
    cellsize     : int/float             # cellsize
    proj_params  : dict/None             # proj4 projection parameters

    Returns
    -------
    GeoArray

    Purpose
    -------
    Create a GeoArray from data.
    
    Examples
    --------
    >>> import geogrid as gg

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
    
    >>> gg.array(data,yorigin=yorigin,xorigin=xorigin,fill_value=fill_value,cellsize=cellsize)
    GeoArray([[-9, -9, -9, -9, -9, -9],
              [-9,  4,  4,  0,  2, -9],
              [-9,  0,  5,  8,  5, -9],
              [-9,  0,  0,  1,  0, -9],
              [-9,  2,  3,  3,  3, -9],
              [-9,  0,  1,  0,  6, -9],
              [-9,  0,  3,  3,  3, -9],
              [-9,  4,  6,  2,  4, -9],
              [-9,  2,  1,  0,  1, -9],
              [-9, -9, -9, -9, -9, -9]])
 
    """
    return _factory(np.asarray(data) if not dtype else np.asarray(data,dtype),
                    yorigin,xorigin,origin,fill_value,cellsize,proj_params)
        
def zeros(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
          fill_value=-9999, cellsize=1, proj_params=None):
    """
    Parameters
    ----------
    shape        : tuple          # shape of the returned grid 

    Optional Parameters
    -------------------
    dtype        : str/np.dtype          # type of the returned grid
    yorigin      : int/float             # y-value of the grid's origin
    xorigin      : int/float             # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"} # position of the origin. One of:
                                         #     "ul" : upper left corner
                                         #     "ur" : upper right corner
                                         #     "ll" : lower left corner
                                         #     "lr" : lower right corner
    fill_value : inf/float             # fill or fill value
    cellsize     : int/float             # cellsize
    proj_params  : dict/None             # proj4 projection parameters
    
    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with zeros.
    
    Examples
    --------
    >>> import geogrid as gg
    >>> gg.zeros((4,4))
    GeoArray([[ 0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.]])
    """
    return _factory(np.zeros(shape,dtype),yorigin,xorigin,
                    origin,fill_value,cellsize,proj_params)

def zeros_like(a,*args,**kwargs):
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
    >>> import geogrid as gg

    >>> x = np.arange(6).reshape((3,2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5]])

    >>> gg.zeros_like(x)
    GeoArray([[0, 0],
              [0, 0],
              [0, 0]])
    
    >>> y = gg.array(x,yorigin=-5555,xorigin=4444,cellsize=42)
    >>> y
    GeoArray([[0, 1],
              [2, 3],
              [4, 5]])

    >>> z = gg.zeros_like(y)
    >>> z
    GeoArray([[0, 0],
              [0, 0],
              [0, 0]])

    >>> z.header == y.header
    True

    """
    try:
        return array(np.zeros_like(a,*args,**kwargs),**a.header)
    except AttributeError:
        return array(np.zeros_like(a,*args,**kwargs))
        

def ones(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
         fill_value=-9999, cellsize=1, proj_params=None):
    """
    Parameters
    ----------
    shape        : tuple          # shape of the returned grid 

    Optional Parameters
    -------------------
    dtype        : str/np.dtype          # type of the returned grid
    yorigin      : int/float             # y-value of the grid's origin
    xorigin      : int/float             # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"} # position of the origin. One of:
                                         #     "ul" : upper left corner
                                         #     "ur" : upper right corner
                                         #     "ll" : lower left corner
                                         #     "lr" : lower right corner
    fill_value : inf/float             # fill or fill value
    cellsize     : int/float             # cellsize
    proj_params  : dict/None             # proj4 projection parameters

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with ones.

    Examples
    --------
    >>> import geogrid as gg
    >>> gg.ones((4,4))
    GeoArray([[ 1.,  1.,  1.,  1.],
              [ 1.,  1.,  1.,  1.],
              [ 1.,  1.,  1.,  1.],
              [ 1.,  1.,  1.,  1.]])
    """

    return _factory(np.ones(shape,dtype),yorigin,xorigin,
                    origin,fill_value,cellsize,proj_params)

def ones_like(a,*args,**kwargs):
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
    >>> import geogrid as gg

    >>> x = np.arange(6).reshape((3,2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5]])

    >>> gg.ones_like(x)
    GeoArray([[1, 1],
              [1, 1],
              [1, 1]])
    
    >>> y = gg.array(x,yorigin=-5555,xorigin=4444,cellsize=42)
    >>> y
    GeoArray([[0, 1],
              [2, 3],
              [4, 5]])

    >>> z = gg.ones_like(y)
    >>> z
    GeoArray([[1, 1],
              [1, 1],
              [1, 1]])

    >>> z.header == y.header
    True

    """
    try:
        return array(np.ones_like(a,*args,**kwargs),**a.header)
    except AttributeError:
        return array(np.ones_like(a,*args,**kwargs))


def full(shape, value, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
         fill_value=-9999, cellsize=1, proj_params=None):
    """
    Parameters
    ----------
    shape        : tuple          # shape of the returned grid  
    fill_value   : scalar         # fille value

    Optional Parameters
    -------------------
    dtype        : str/np.dtype          # type of the returned grid
    yorigin      : int/float             # y-value of the grid's origin
    xorigin      : int/float             # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"} # position of the origin. One of:
                                         #     "ul" : upper left corner
                                         #     "ur" : upper right corner
                                         #     "ll" : lower left corner
                                         #     "lr" : lower right corner
    fill_value : inf/float             # fill or fill value
    cellsize     : int/float             # cellsize
    proj_params  : dict/None             # proj4 projection parameters

    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with fill_value.

    Examples
    --------
    >>> import geogrid as gg
    >>> gg.full((4,4), 42)
    GeoArray([[ 42.,  42.,  42.,  42.],
              [ 42.,  42.,  42.,  42.],
              [ 42.,  42.,  42.,  42.],
              [ 42.,  42.,  42.,  42.]])
    """
    return _factory(np.full(shape,value,dtype),yorigin,xorigin,
                    origin,fill_value,cellsize,proj_params)

def full_like(a,fill_value,*args,**kwargs):
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
    >>> import geogrid as gg

    >>> x = np.arange(6).reshape((3,2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5]])

    >>> gg.full_like(x,42)
    GeoArray([[42, 42],
              [42, 42],
              [42, 42]])
    
    >>> y = gg.array(x,yorigin=-5555,xorigin=4444,cellsize=42)
    >>> y
    GeoArray([[0, 1],
              [2, 3],
              [4, 5]])

    >>> z = gg.full_like(y,42)
    >>> z
    GeoArray([[42, 42],
              [42, 42],
              [42, 42]])

    >>> z.header == y.header
    True

    """
    try:
        return array(np.full_like(a,fill_value,*args,**kwargs),**a.header)
    except AttributeError:
        return array(np.full_like(a,fill_value,*args,**kwargs))

def empty(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
          fill_value=-9999, cellsize=1, proj_params=None):
    """
    Parameters
    ----------
    shape        : tuple          # shape of the returned grid  
 
    Optional Parameters
    ------------------
    dtype        : str/np.dtype          # type of the returned grid
    yorigin      : int/float             # y-value of the grid's origin
    xorigin      : int/float             # x-value of the grid's origin
    origin       : {"ul","ur","ll","lr"} # position of the origin. One of:
                                         #    "ul" : upper left corner
                                         #    "ur" : upper right corner
                                         #    "ll" : lower left corner
                                         #    "lr" : lower right corner
    fill_value : inf/float             # fill or fill value
    cellsize     : int/float             # cellsize
    proj_params  : dict/None             # proj4 projection parameters
    
    Returns
    -------
    GeoArray

    Purpose
    -------
    Return a new GeoArray of given shape and type, filled with fill_value

    Examples
    --------
    >>> import geogrid as gg

    >>> gg.empty((4,4))
    GeoArray([[-9999., -9999., -9999., -9999.],
              [-9999., -9999., -9999., -9999.],
              [-9999., -9999., -9999., -9999.],
              [-9999., -9999., -9999., -9999.]])
    
    >>> gg.empty((4,4),fill_value=32)
    GeoArray([[ 32.,  32.,  32.,  32.],
              [ 32.,  32.,  32.,  32.],
              [ 32.,  32.,  32.,  32.],
              [ 32.,  32.,  32.,  32.]])
   """

    return _factory(np.full(shape,fill_value,dtype),yorigin,xorigin,
                    origin,fill_value,cellsize,proj_params)

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
    >>> import geogrid as gg

    >>> x = np.arange(6).reshape((3,2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5]])

    >>> gg.empty_like(x)
    GeoArray([[-9999, -9999],
              [-9999, -9999],
              [-9999, -9999]])
    
    >>> y = gg.array(x,yorigin=-5555,xorigin=4444,fill_value=42)
    >>> y
    GeoArray([[0, 1],
              [2, 3],
              [4, 5]])

    >>> z = gg.empty_like(y)
    >>> z
    GeoArray([[42, 42],
              [42, 42],
              [42, 42]])

    >>> z.header == y.header
    True
    """

    try:
        return array(np.full_like(a,a.fill_value,*args,**kwargs),**a.header)
    except AttributeError:
        return array(np.full_like(a,-9999),*args,**kwargs)

    # return np.empty_like(a,*args,**kwargs)


def _factory(data, yorigin, xorigin, origin, fill_value, cellsize, proj_params):
    origins = ("ul","ur","ll","lr")
    if origin not in origins:
        raise TypeError("Argument 'origin' must be on of '{:}'".format(origins))
    mask = data==fill_value
    if not np.any(mask):
        mask = False
    return GeoArray(data,yorigin,xorigin,origin,cellsize,proj_params,mask=mask,fill_value=fill_value)


def fromfile(fname):
    """
    Parameters
    ----------
    fname : str  # file name

    Returns
    -------
    GeoArray

    Purpose
    -------
    Create GeoArray from file

    Examples
    --------
    >>> import numpy as np
    >>> import geogrid as gg
    
    """

    def _openFile(fname):
        fobj = gdal.OpenShared(fname)
        if fobj:
            return fobj
        raise IOError("Could not open file")    

    def _cellsize(geotrans):       
        if abs(geotrans[1]) == abs(geotrans[5]):
            return abs(geotrans[1])
        raise NotImplementedError(
            "Diverging cellsizes in x and y direction are not allowed yet!")    

    def _projParams(fobj):
        srs = osr.SpatialReference()
        srs.ImportFromWkt(fobj.GetProjection())
        proj_params = filter(None,re.split("[+= ]",srs.ExportToProj4()))
        return dict(zip(proj_params[0::2],proj_params[1::2]))

    global _FILEREFS
    
    fobj = _openFile(fname)

    rasterband = fobj.GetRasterBand(1)
    geotrans   = fobj.GetGeoTransform()

    nrows      = fobj.RasterYSize        
    ncols      = fobj.RasterXSize
    nbands     = fobj.RasterCount

    dtype      = np.dtype(GDAL2DTYPE[rasterband.DataType])

    if "linux" in sys.platform:
        data       = fobj.GetVirtualMemArray(
            gdal.GF_Write, cache_size = nbands*nrows*ncols*dtype.itemsize
        )
        _FILEREFS.append(fobj)
    else:
        data = fobj.ReadAsArray()

    return _factory(data=data,yorigin=geotrans[3],xorigin=geotrans[0],
                    origin="ul",fill_value=rasterband.GetNoDataValue(),
                    cellsize=_cellsize(geotrans),proj_params=_projParams(fobj))


def _tofile(fname,geogrid):
    def _fnameExtension(fname):
        return os.path.splitext(fname)[-1].lower()

    def _projection(grid):
        params = None 
        if grid.proj_params:
            params =  "+{:}".format(" +".join(
                ["=".join(pp) for pp in grid.proj_params.items()])                                    
                )
        srs = osr.SpatialReference()
        srs.ImportFromProj4(params)
        return srs.ExportToWkt()

    def _getDriver(fext):
        if fext in _DRIVER_DICT:
            driver = gdal.GetDriverByName(_DRIVER_DICT[fext])
            metadata = driver.GetMetadata_Dict()
            if "YES" == metadata.get("DCAP_CREATE",metadata.get("DCAP_CREATECOPY")):
                return driver
            raise IOError("Datatype canot be written")            
        raise IOError("No driver found for filenmae extension '{:}'".format(fext))
    
    def _writeGdalMemory(grid,projection):
        driver = gdal.GetDriverByName("MEM")
        out = driver.Create(
            "",grid.ncols,grid.nrows,grid.nbands,
            DTYPE2GDAL[str(grid.dtype)]
        )
        out.SetGeoTransform(
            (grid.xorigin, grid.cellsize,0,
             grid.yorigin, 0, grid.cellsize)
        )
        out.SetProjection(projection)
        for n in xrange(grid.nbands):
            band = out.GetRasterBand(n+1)
            band.SetNoDataValue(float(grid.fill_value))
            banddata = grid[(n,Ellipsis) if grid.nbands > 1 else (Ellipsis)]
            band.WriteArray(banddata)
        out.FlushCache()
        return out
            
    memset = _writeGdalMemory(geogrid, _projection(geogrid))
    outdriver = _getDriver(_fnameExtension(fname))
    out = outdriver.CreateCopy(fname,memset,0)
    errormsg = gdal.GetLastErrorMsg()
    if errormsg or not out:
        raise IOError(errormsg)

        
class GeoArray(np.ma.MaskedArray):
    """
    Parameters
    ----------
    data         : np.ndarray/list/tuple     
    yorigin      : scalar                # y-coordinate of origin
    xorigin      : scalar                # x-coordinate of origin
    origin       : {"ul","ur","ll","lr"} # position of the grid origin
                                         #     "ul" -> upper left
                                         #     "ur" -> upper right
                                         #     "ll" -> lower left
                                         #     "lr" -> lower right
    fill_value : scalar 
    cellsize     : scalar
    
    Optional Parameters
    -------------------
    proj_params  : dict/None             # Proj4 projection parameters
        
    Purpose
    -------
    This numpy.ndarray subclass adds geographic context to data.
    A (hopfully growing) number of operations on the data and the writing
    to different file formats (see the variable _DRIVER_DICT) is supported.
    
    Details
    -------
    The Python GDAL bindings are used as backend. On Linux the GDAL virtual memory 
    mapping, i.e. data is only read from storage when it is actually accessed
    For other OS this mechanism is not implemented by GDAL and all data is read
    during initialization. 

    Restrictions
    ------------
    1. A GeoArray instance can be passed to any numpy function expecting a
       numpy.ndarray as argument and, in theory, all these functions should also return
       an object of the same type. In practice however not all functions do so and some
       will still return a numpy.ndarray.
    2. Adding the geographic information to the data does (at the moment) not imply
       any additional logic. If the shapes of two grids allow the succesful execution 
       of a certain operator/function your program will continue. It is within the responsability
       of the user to check whether a given operation makes sense within a geographic context 
       (e.g. grids cover the same spatial domain, share a common projection, etc.) or not.
       Overriding the __array_prepare__ method to implement the necessary checks would solve that
       issue for all operators and most functions. But there are a few edge cases, where numpy 
       functions are not routed through the __array_prepare__/__array_wrap__ mechanism. Implementing
       implicit checks would still mean, that there are some unchecked calls, beside
       pretending a (geographically) safe environment.
    
    See also
    --------
    array : construct a GeoArray.
    zeros : construct a GeoArray, each element of which is zero.
    ones  : construct a GeoArray, each element of which is one.
    empty : construct a GeoArray, each element of which is fill_value.
    full  : construct a GeoArray, each element of which is a given value.

    Examples
    --------
    >>> import numpy as np
    >>> import geogrid as gg
    
    >>> data = np.array([[-9 ,-9, -9, -9, -9],
    ...                  [-9 , 0,  5,  8, -9],
    ...                  [-9 , 0,  0,  1, -9],
    ...                  [-9 , 2,  3,  3, -9],
    ...                  [-9 , 0,  1,  0, -9],
    ...                  [-9 , 0,  3,  3, -9],
    ...                  [-9 ,-9, -9, -9, -9],])
    
    >>> grid = gg.GeoArray(data,yorigin=63829.3,xorigin=76256.6,origin="ul",
    ...                    fill_value=-9,cellsize=55)

    >>> grid
    GeoArray([[-9, -9, -9, -9, -9],
              [-9,  0,  5,  8, -9],
              [-9,  0,  0,  1, -9],
              [-9,  2,  3,  3, -9],
              [-9,  0,  1,  0, -9],
              [-9,  0,  3,  3, -9],
              [-9, -9, -9, -9, -9]])
 
    # all numpy operators are supported
    >>> grid + 5    
    GeoArray([[-4, -4, -4, -4, -4],
           [-4,  5, 10, 13, -4],
           [-4,  5,  5,  6, -4],
           [-4,  7,  8,  8, -4],
           [-4,  5,  6,  5, -4],
           [-4,  5,  8,  8, -4],
           [-4, -4, -4, -4, -4]])

    # all numpy methods are supported
    >>> np.exp(grid).astype(np.int64)
    GeoArray([[   0,    0,    0,    0,    0],
              [   0,    1,  148, 2980,    0],
              [   0,    1,    1,    2,    0],
              [   0,    7,   20,   20,    0],
              [   0,    1,    2,    1,    0],
              [   0,    1,   20,   20,    0],
              [   0,    0,    0,    0,    0]])

    >>> np.sum(grid)
    -151
    
    # all np.ndarray methods are supported
    >>> grid.max()
    8
    >>> grid.argsort()
    GeoArray([[0, 1, 2, 3, 4],
              [0, 4, 1, 2, 3],
              [0, 4, 1, 2, 3],
              [0, 4, 1, 2, 3],
              [0, 4, 1, 3, 2],
              [0, 4, 1, 2, 3],
              [0, 1, 2, 3, 4]])
    >>> grid.clip(0,3)
    GeoArray([[0, 0, 0, 0, 0],
              [0, 0, 3, 3, 0],
              [0, 0, 0, 1, 0],
              [0, 2, 3, 3, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 3, 3, 0],
              [0, 0, 0, 0, 0]])

    # all np.ndarray attributes are supported
    >>> grid.T
    GeoArray([[-9, -9, -9, -9, -9, -9, -9],
              [-9,  0,  0,  2,  0,  0, -9],
              [-9,  5,  0,  3,  1,  3, -9],
              [-9,  8,  1,  3,  0,  3, -9],
              [-9, -9, -9, -9, -9, -9, -9]])
    >>> grid.nbytes
    280
    >>> grid.shape
    (7, 5)
    """
   
    def __new__(cls, data, yorigin, xorigin, origin, 
                cellsize, proj_params=None,*args,**kwargs):

        obj = np.ma.MaskedArray.__new__(cls,data, *args, **kwargs)
        obj._optinfo['yorigin'] = yorigin
        obj._optinfo['xorigin'] = xorigin
        obj._optinfo['origin'] = origin
        obj._optinfo['cellsize'] = cellsize
        obj._optinfo['proj_params'] = proj_params

        return obj

    @property
    def header(self):
        """
        Parameters
        ----------
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
        >>> import geogrid as gg
        >>> x = gg.full((4,4),42,yorigin=100,xorigin=55,origin="ur")
        >>> x.header
        {'origin': 'ur', 'proj_params': None, 'cellsize': 1,
         'yorigin': 100, 'xorigin': 55, 'fill_value': -9999.0}
        """

        return {
            "yorigin"      : self.yorigin,
            "xorigin"      : self.xorigin,
            "origin"       : self.origin,
            "fill_value" : self.fill_value,            
            "cellsize"     : self.cellsize,
            "proj_params"  : self.proj_params
        }
        
    @property
    def bbox(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        dict

        Purpose
        -------
        Return the grid's bounding box.

        Examples
        --------
        >>> import geogrid as gg
        >>> x = gg.full((4,4),42,yorigin=100,xorigin=55,origin="ur")
        >>> x.bbox
        {'xmin': 51, 'ymin': 96, 'ymax': 100, 'xmax': 55}
        """

        yopp = self.nrows * self.cellsize
        xopp = self.ncols * self.cellsize
        return { 
            "ymin": self.yorigin if self.origin[0] == "l" else self.yorigin - yopp,
            "ymax": self.yorigin if self.origin[0] == "u" else self.yorigin + yopp,
            "xmin": self.xorigin if self.origin[1] == "l" else self.xorigin - xopp,
            "xmax": self.xorigin if self.origin[1] == "r" else self.xorigin + xopp,
        }


    def getOrigin(self,origin=None):
        """
        Parameters
        ----------
        None

        Parameters
        ----------
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
        Parameters
        ----------
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
        >>> import geogrid as gg
        
        >>> x = np.arange(40).reshape((2,4,5))
        >>> grid = gg.array(x)
        >>> grid.shape
        (2, 4, 5)
        >>> grid.nbands
        2
        
        >>> x = np.arange(120).reshape((3,2,4,5))
        >>> grid = gg.array(x)
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
        Parameters
        ----------
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
        >>> import geogrid as gg
        
        >>> x = np.arange(40).reshape((2,4,5))
        >>> grid = gg.array(x)
        >>> grid.shape
        (2, 4, 5)
        >>> grid.nrows
        4
        
        >>> x = np.arange(120).reshape((3,2,4,5))
        >>> grid = gg.array(x)
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
        Parameters
        ----------
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
        >>> import geogrid as gg
        
        >>> x = np.arange(40).reshape((2,4,5))
        >>> grid = gg.array(x)
        >>> grid.shape
        (2, 4, 5)
        >>> grid.ncols
        5
        
        >>> x = np.arange(120).reshape((3,2,4,5))
        >>> grid = gg.array(x)
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
        Parameters
        ----------
        fname : str  # file name

        Returns
        -------
        None

        Purpose
        -------
        Write GeoArray to file. The output dataset type is derived from
        the file name extension. See _DRIVER_DICT for implemented formats.
        """
        _tofile(fname, self)
        
    def indexCoordinates(self,y_idx,x_idx):
        """
        Parameters
        ----------
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
        y_coor =  yorigin - y_idx * self.cellsize        
        x_coor =  xorigin + x_idx * self.cellsize
        return y_coor, x_coor

    def coordinateIndex(self,y_coor,x_coor):
        """
        Parameters
        ----------
        y_coor, x_coor : scalar

        Returns
        -------
        (int, int)

        Purpose
        -------
        Find the grid cell into which the given coordinates 
        fall and return its row/column index values.
        """

        yorigin, xorigin = self.getOrigin("ul")

        y_idx = int(floor((yorigin - y_coor)/float(self.cellsize))) 

        x_idx = int(floor((x_coor - xorigin )/float(self.cellsize)))
        if y_idx < 0 or y_idx >= self.nrows or x_idx < 0 or x_idx >= self.ncols:
            raise ValueError("Given Coordinates not within the grid domain!")
        return y_idx,x_idx

    def trim(self):
        """
        Parameters
        ----------
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
        >>> import geogrid as gg
        >>> data         = np.array([[-9 ,-9, -9, -9, -9],
        ...                          [-9 , 4,  0,  2, -9],
        ...                          [-9 , 3,  3,  3, -9],
        ...                          [-9 , 1,  0,  6, -9],
        ...                          [-9 , 1,  0,  1, -9],
        ...                          [-9 ,-9, -9, -9, -9],])
    
        >>> grid = gg.array(data,fill_value=-9)
        >>> grid
        GeoArray([[-9, -9, -9, -9, -9],
                  [-9,  4,  0,  2, -9],
                  [-9,  3,  3,  3, -9],
                  [-9,  1,  0,  6, -9],
                  [-9,  1,  0,  1, -9],
                  [-9, -9, -9, -9, -9]])

        >>> grid.trim()
        GeoArray([[4, 0, 2],
                  [3, 3, 3],
                  [1, 0, 6],
                  [1, 0, 1]])
        """

        y_idx, x_idx = np.where(self.data != self.fill_value)
        try:
            return self.removeCells(
                top=min(y_idx),bottom=self.nrows-max(y_idx)-1,
                left=min(x_idx),right=self.ncols-max(x_idx)-1)
        except ValueError:
            return self

    def removeCells(self,top=0,left=0,bottom=0,right=0):
        """
        Parameters
        ----------
        top, left, bottom, right : int

        Returns
        -------
        GeoArray

        Purpose
        -------
        Remove the number of given cells from the respective margin of the grid.

        Example
        -------
        >>> import geogrid as gg
        >>> x = gg.array(np.arange(36).reshape(6,6))
        >>> x
        GeoArray([[ 0,  1,  2,  3,  4,  5],
                  [ 6,  7,  8,  9, 10, 11],
                  [12, 13, 14, 15, 16, 17],
                  [18, 19, 20, 21, 22, 23],
                  [24, 25, 26, 27, 28, 29],
                  [30, 31, 32, 33, 34, 35]]) 
        
        >>> x.removeCells(top=1,left=2,bottom=2,right=1)
        GeoArray([[ 8,  9, 10],
                  [14, 15, 16],
                  [20, 21, 22]])
        """

        top    = int(max(top,0))
        left   = int(max(left,0))
        bottom = int(max(bottom,0))
        right  = int(max(right,0))

        nrows = self.nrows - top  - bottom
        ncols = self.ncols - left - right

        yorigin, xorigin = self.getOrigin("ul")

        return array(
            data        = self[Ellipsis, top:top+nrows, left:left+ncols],
            dtype       = self.dtype,
            yorigin     = yorigin - top*self.cellsize ,
            xorigin     = xorigin + left*self.cellsize,
            origin      = "ul", 
            fill_value  = self.fill_value,
            cellsize    = self.cellsize,
            proj_params = self.proj_params,
        )

    def shrink(self,ymin=None,ymax=None,xmin=None,xmax=None):
        """
        Parameters
        ----------
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
        >>> import geogrid as gg
        
        >>> x = np.arange(20).reshape((4,5))
        >>> grid = gg.array(x,origin="ll",cellsize=20)

        >>> grid.bbox
        {'xmin': 0, 'ymin': 0, 'ymax': 80, 'xmax': 100}
        >>> grid.shape
        (4, 5)
        >>> grid
        GeoArray([[ 0,  1,  2,  3,  4],
                  [ 5,  6,  7,  8,  9],
                  [10, 11, 12, 13, 14],
                  [15, 16, 17, 18, 19]])


        >>> shrinked = grid.shrink(ymin=18,ymax=56,xmin=22,xmax=92)
        >>> shrinked.bbox
        {'xmin': 20, 'ymin': 0, 'ymax': 60, 'xmax': 100}
        >>> shrinked.shape
        (3, 4)
        >>> shrinked
        GeoArray([[ 6,  7,  8,  9],
                  [11, 12, 13, 14],
                  [16, 17, 18, 19]])
        """

        bbox = {
            "ymin": ymin if ymin else self.bbox["ymin"],
            "ymax": ymax if ymax else self.bbox["ymax"],
            "xmin": xmin if xmin else self.bbox["xmin"],
            "xmax": xmax if xmax else self.bbox["xmax"],
            }

        cellsize = float(self.cellsize)
        top    = floor(round((self.bbox["ymax"] - bbox["ymax"])
                             /cellsize, MAX_PRECISION))
        left   = floor(round((bbox["xmin"] - self.bbox["xmin"])
                            /cellsize, MAX_PRECISION))
        bottom = floor(round((bbox["ymin"] - self.bbox["ymin"])
                            /cellsize, MAX_PRECISION))
        right  = floor(round((self.bbox["xmax"] - bbox["xmax"])
                            /cellsize, MAX_PRECISION))

        return self.removeCells(max(top,0),max(left,0),max(bottom,0),max(right,0))        

    def addCells(self,top=0,left=0,bottom=0,right=0):
        """
        Parameters
        ----------
        top, left, bottom, right : int 

        Returns
        -------
        GeoArray

        Purpose
        -------
        Add the number of given cells to the respective margin of the grid.
        
        Example
        -------
        >>> import geogrid as gg

        >>> grid = gg.full((4,5),42,fill_value=-9)
        >>> grid
        GeoArray([[ 42.,  42.,  42.,  42.,  42.],
                  [ 42.,  42.,  42.,  42.,  42.],
                  [ 42.,  42.,  42.,  42.,  42.],
                  [ 42.,  42.,  42.,  42.,  42.]])

        >>> grid.addCells(top=1,left=1,bottom=2)
        GeoArray([[ -9.,  -9.,  -9.,  -9.,  -9.,  -9.],
                  [ -9.,  42.,  42.,  42.,  42.,  42.],
                  [ -9.,  42.,  42.,  42.,  42.,  42.],
                  [ -9.,  42.,  42.,  42.,  42.,  42.],
                  [ -9.,  42.,  42.,  42.,  42.,  42.],
                  [ -9.,  -9.,  -9.,  -9.,  -9.,  -9.],
                  [ -9.,  -9.,  -9.,  -9.,  -9.,  -9.]])
        """

        top    = int(max(top,0))
        left   = int(max(left,0))
        bottom = int(max(bottom,0))
        right  = int(max(right,0))

        shape = list(self.shape)
        shape[-2:] = self.nrows + top  + bottom, self.ncols + left + right
        yorigin,xorigin = self.getOrigin("ul")

        out = empty(
            shape        = shape,
            dtype        = self.dtype,        
            yorigin      = yorigin + top*self.cellsize ,
            xorigin      = xorigin - left*self.cellsize,
            origin       = "ul",
            fill_value = self.fill_value,
            cellsize     = self.cellsize,
            proj_params  = self.proj_params,
        )

        # the Ellipsis ensures that the function works 
        # for arrays with more than two dimensions
        out[Ellipsis, top:top+self.nrows, left:left+self.ncols] = self
        return out

    def enlarge(self,ymin=None,ymax=None,xmin=None,xmax=None):
        """
        Parameters
        ----------
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
        >>> import geogrid as gg
        
        >>> x = np.arange(20).reshape((4,5))
        >>> grid = gg.array(x,yorigin=100,xorigin=200,origin="ll",cellsize=20,fill_value=-9)

        >>> grid.bbox
        {'xmin': 200, 'ymin': 100, 'ymax': 180, 'xmax': 300}
        >>> grid.shape
        (4, 5)
        >>> grid
        GeoArray([[ 0,  1,  2,  3,  4],
                  [ 5,  6,  7,  8,  9],
                  [10, 11, 12, 13, 14],
                  [15, 16, 17, 18, 19]])

        >>> enlarged = grid.enlarge(xmin=130,xmax=200,ymin=66)
        >>> enlarged.bbox
        {'xmin': 120, 'ymin': 60, 'ymax': 180, 'xmax': 300}
        
        >>> enlarged
        GeoArray([[-9, -9, -9, -9,  0,  1,  2,  3,  4],
                  [-9, -9, -9, -9,  5,  6,  7,  8,  9],
                  [-9, -9, -9, -9, 10, 11, 12, 13, 14],
                  [-9, -9, -9, -9, 15, 16, 17, 18, 19],
                  [-9, -9, -9, -9, -9, -9, -9, -9, -9],
                  [-9, -9, -9, -9, -9, -9, -9, -9, -9]])
        """

        bbox = {
            "ymin": ymin if ymin else self.bbox["ymin"],
            "ymax": ymax if ymax else self.bbox["ymax"],
            "xmin": xmin if xmin else self.bbox["xmin"],
            "xmax": xmax if xmax else self.bbox["xmax"],
            }
        cellsize = float(self.cellsize)
        top    = ceil(round((bbox["ymax"] - self.bbox["ymax"])
                            /cellsize,MAX_PRECISION))
        left   = ceil(round((self.bbox["xmin"] - bbox["xmin"])
                            /cellsize,MAX_PRECISION))
        bottom = ceil(round((self.bbox["ymin"] - bbox["ymin"])
                            /cellsize,MAX_PRECISION))
        right  = ceil(round((bbox["xmax"] - self.bbox["xmax"])
                            /cellsize,MAX_PRECISION))

        return self.addCells(max(top,0),max(left,0),max(bottom,0),max(right,0))        

    def snap(self,target):
        """
        Parameters
        ----------
        target : GeoArray

        Returns
        -------
        None
        
        Purpose
        -------
        Shift the grid origin that it matches the nearest cell origin in target.

        Restrictions
        ------------
        The shift will only alter the grid coordinates. No changes to the
        data will be done. In case of large shifts the physical integrety 
        of the data might be disturbed!
        
        Examples
        --------
        >>> import numpy as np
        >>> import geogrid as gg
        
        >>> x = np.arange(20).reshape((4,5))
        >>> grid1 = gg.array(x,origin="ll",cellsize=25)
        >>> grid1.bbox
        {'xmin': 0, 'ymin': 0, 'ymax': 100, 'xmax': 125}
        
        >>> grid2 = gg.array(x,yorigin=3,xorigin=1.24,origin="ll",cellsize=18.67)
        >>> grid2.bbox
        {'xmin': 1.24, 'ymin': 3, 'ymax': 77.68, 'xmax': 94.59}

        >>> grid2.snap(grid1)
        >>> grid2.bbox
        {'xmin': 0.0, 'ymin': 0.0, 'ymax': 74.680000000000007, 'xmax': 93.350000000000009}
        """

        diff = np.array(self.getOrigin()) - np.array(target.getOrigin(self.origin))
        dy,dx = abs(diff)%target.cellsize * np.sign(diff)

        if abs(dx) > self.cellsize/2.:
            dx += self.cellsize

        if abs(dy) > self.cellsize/2.:
            dy += self.cellsize

        self.xorigin -= dx
        self.yorigin -= dy

    def __getattr__(self,name):
        try:
            return self._optinfo[name]
        except KeyError:
            raise AttributeError(
                "'{:}' object has no attribute {:}".format (self.__class__.__name__, name)
            )

   
if __name__ == "__main__":

    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    
