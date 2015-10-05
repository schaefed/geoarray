#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""

This module provides a numpy.ndarray subclass to work with arrays in a
geographically explicit context and a number of factory functions.

Requirements:
    - GDAL >= 1.11
    - numpy

>>> import numpy as np
>>> import geogrid as gg

>>> yorigin      = 63829.3
>>> xorigin      = 76256.6
>>> origin       = "ul"
>>> nodata_value = -9
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

>>> grid = gg.array(data,yorigin=yorigin,xorigin=xorigin,nodata_value=nodata_value,cellsize=cellsize)

# cast back
>>> np.array(grid)
array([[-9, -9, -9, -9, -9, -9],
       [-9,  4,  4,  0,  2, -9],
       [-9,  0,  5,  8,  5, -9],
       [-9,  0,  0,  1,  0, -9],
       [-9,  2,  3,  3,  3, -9],
       [-9,  0,  1,  0,  6, -9],
       [-9,  0,  3,  3,  3, -9],
       [-9,  4,  6,  2,  4, -9],
       [-9,  2,  1,  0,  1, -9],
       [-9, -9, -9, -9, -9, -9]])

# geogrid supports all the numpy operators ... 
>>> grid + 5    
GeoArray([[-4, -4, -4, -4, -4, -4],
          [-4,  9,  9,  5,  7, -4],
          [-4,  5, 10, 13, 10, -4],
          [-4,  5,  5,  6,  5, -4],
          [-4,  7,  8,  8,  8, -4],
          [-4,  5,  6,  5, 11, -4],
          [-4,  5,  8,  8,  8, -4],
          [-4,  9, 11,  7,  9, -4],
          [-4,  7,  6,  5,  6, -4],
          [-4, -4, -4, -4, -4, -4]])

# ... and functions
>>> np.exp(grid).astype(np.int64)
GeoArray([[   0,    0,    0,    0,    0,    0],
           [   0,   54,   54,    1,    7,    0],
           [   0,    1,  148, 2980,  148,    0],
           [   0,    1,    1,    2,    1,    0],
           [   0,    7,   20,   20,   20,    0],
           [   0,    1,    2,    1,  403,    0],
           [   0,    1,   20,   20,   20,    0],
           [   0,   54,  403,    7,   54,    0],
           [   0,    7,    2,    1,    2,    0],
           [   0,    0,    0,    0,    0,    0]])

>>> np.sum(grid)
-176

# all arguments are accessible
>>> grid.yorigin == yorigin
True

>>> grid.nodata_value == nodata_value
True

# currently the amount of grid related functinality is limited.
# There are functions to:
# 1. increase the size of the grid, padding it with nodatdata values
#    1.1 increase by an amount of cells
>>> grid.addCells(top=1,bottom=2,left=1)    
GeoArray([[-9, -9, -9, -9, -9, -9, -9],
           [-9, -9, -9, -9, -9, -9, -9],
           [-9, -9,  4,  4,  0,  2, -9],
           [-9, -9,  0,  5,  8,  5, -9],
           [-9, -9,  0,  0,  1,  0, -9],
           [-9, -9,  2,  3,  3,  3, -9],
           [-9, -9,  0,  1,  0,  6, -9],
           [-9, -9,  0,  3,  3,  3, -9],
           [-9, -9,  4,  6,  2,  4, -9],
           [-9, -9,  2,  1,  0,  1, -9],
           [-9, -9, -9, -9, -9, -9, -9],
           [-9, -9, -9, -9, -9, -9, -9],
           [-9, -9, -9, -9, -9, -9, -9]])

#    1.2 increase to a given extend
>>> grid.enlargeGrid(xmin=grid.xorigin-3.5*cellsize,xmax=grid.xorigin+(grid.ncols+2)*grid.cellsize)
GeoArray([[-9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9],
           [-9, -9, -9, -9, -9,  4,  4,  0,  2, -9, -9, -9],
           [-9, -9, -9, -9, -9,  0,  5,  8,  5, -9, -9, -9],
           [-9, -9, -9, -9, -9,  0,  0,  1,  0, -9, -9, -9],
           [-9, -9, -9, -9, -9,  2,  3,  3,  3, -9, -9, -9],
           [-9, -9, -9, -9, -9,  0,  1,  0,  6, -9, -9, -9],
           [-9, -9, -9, -9, -9,  0,  3,  3,  3, -9, -9, -9],
           [-9, -9, -9, -9, -9,  4,  6,  2,  4, -9, -9, -9],
           [-9, -9, -9, -9, -9,  2,  1,  0,  1, -9, -9, -9],
           [-9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9]])

# 2. decrease the size of a grid
#    2.1 decrease by an amount of cells
>>> grid.removeCells(top=2,bottom=2)
GeoArray([[-9,  0,  5,  8,  5, -9],
           [-9,  0,  0,  1,  0, -9],
           [-9,  2,  3,  3,  3, -9],
           [-9,  0,  1,  0,  6, -9],
           [-9,  0,  3,  3,  3, -9],
           [-9,  4,  6,  2,  4, -9]])

#    2.2 decrease to a given extend
>>> grid.shrinkGrid(ymax=grid.yorigin-1.9*grid.cellsize)
GeoArray([[-9,  4,  4,  0,  2, -9],
           [-9,  0,  5,  8,  5, -9],
           [-9,  0,  0,  1,  0, -9],
           [-9,  2,  3,  3,  3, -9],
           [-9,  0,  1,  0,  6, -9],
           [-9,  0,  3,  3,  3, -9],
           [-9,  4,  6,  2,  4, -9],
           [-9,  2,  1,  0,  1, -9],
           [-9, -9, -9, -9, -9, -9]])

#   3 Remove framing nodata_value
>>> grid.trimGrid()
GeoArray([[4, 4, 0, 2],
           [0, 5, 8, 5],
           [0, 0, 1, 0],
           [2, 3, 3, 3],
           [0, 1, 0, 6],
           [0, 3, 3, 3],
           [4, 6, 2, 4],
           [2, 1, 0, 1]])

#  4. Match two grids
>>> grid2 = empty_like(grid)
>>> grid2.yorigin -= 76872.8
>>> grid2.xorigin += 34792.2
>>> grid2.snapGrid(grid)
>>> (np.array(grid2.getOrigin()) - np.array(grid.getOrigin()))/grid.cellsize
array([-1398.,   631.])

#  5. Get the grid position of coordinates
>>> yll,xll = grid.getOrigin("ll")
>>> yll += grid.cellsize*3.3
>>> xll += grid.cellsize*1.2
>>> grid.coordinateIndex(yll,xll)
(6, 1)

#  6. Get the coordinates of a grid cell
>>> grid.indexCoordinates(4,4)
(63609.3, 76476.6)

"""

import re, os
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
          nodata_value=-9999, cellsize=1, proj_params=None):
    """
    Parameters
    ----------
        data         : numpy.ndarray  # data to wrap

    Optional Parameters
    -------------------
        dtype        : str/np.dtype   # type of the returned grid
        yorigin      : int/float      # y-value of the grid's origin
        xorigin      : int/float      # x-value of the grid's origin
        origin       : str            # position of the origin. One of:
                                             "ul" : upper left corner
                                             "ur" : upper right corner
                                             "ll" : lower left corner
                                             "lr" : lower right corner
        nodata_value : inf/float      # nodata or fill value
        cellsize     : int/float      # cellsize
        proj_params  : dict/None      # proj4 projection parameters

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
    >>> nodata_value = -9
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
    
    >>> gg.array(data,yorigin=yorigin,xorigin=xorigin,nodata_value=nodata_value,cellsize=cellsize)
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
                    yorigin,xorigin,origin,nodata_value,cellsize,proj_params)
        
def zeros(shape, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
          nodata_value=-9999, cellsize=1, proj_params=None):
    """
    Parameters
    ----------
        shape        : tuple          # shape of the returned grid 

    Optional Parameters
    -------------------
        dtype        : str/np.dtype   # type of the returned grid
        yorigin      : int/float      # y-value of the grid's origin
        xorigin      : int/float      # x-value of the grid's origin
        origin       : str            # position of the origin. One of:
                                             "ul" : upper left corner
                                             "ur" : upper right corner
                                             "ll" : lower left corner
                                             "lr" : lower right corner
        nodata_value : inf/float      # nodata or fill value
        cellsize     : int/float      # cellsize
        proj_params  : dict/None      # proj4 projection parameters
    
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
                    origin,nodata_value,cellsize,proj_params)

def zeros_like(a,*args,**kwargs):
    """
    Parameters
    ----------
        a           : np.ndarray         # the array to derive the shape and dtype attributes 

    Optional Parameters
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
         nodata_value=-9999, cellsize=1, proj_params=None):
    """
    Parameters
    ----------
        shape        : tuple          # shape of the returned grid 

    Optional Parameters
    -------------------
        dtype        : str/np.dtype   # type of the returned grid
        yorigin      : int/float      # y-value of the grid's origin
        xorigin      : int/float      # x-value of the grid's origin
        origin       : str            # position of the origin. One of:
                                             "ul" : upper left corner
                                             "ur" : upper right corner
                                             "ll" : lower left corner
                                             "lr" : lower right corner
        nodata_value : inf/float      # nodata or fill value
        cellsize     : int/float      # cellsize
        proj_params  : dict/None      # proj4 projection parameters

    Returns
    -------
        GeoArray

    Purpose
    -------
        Return a new GeoArray of given shape and type, filled with ones

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
                    origin,nodata_value,cellsize,proj_params)

def ones_like(a,*args,**kwargs):
    """
    Parameters
    ----------
        a           : np.ndarray         # the array to derive the shape and dtype attributes 

    Optional Parameters
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


def full(shape, fill_value, dtype=np.float64, yorigin=0, xorigin=0, origin="ul",
         nodata_value=-9999, cellsize=1, proj_params=None):
    """
    Parameters
    ----------
        shape        : tuple          # shape of the returned grid  
        fill_value   : scalar         # fille value

    Optional Parameters
    -------------------
        dtype        : str/np.dtype   # type of the returned grid
        yorigin      : int/float      # y-value of the grid's origin
        xorigin      : int/float      # x-value of the grid's origin
        origin       : str            # position of the origin. One of:
                                            "ul" : upper left corner
                                            "ur" : upper right corner
                                            "ll" : lower left corner
                                            "lr" : lower right corner
        nodata_value : inf/float      # nodata or fill value
        cellsize     : int/float      # cellsize
        proj_params  : dict/None      # proj4 projection parameters

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
    return _factory(np.full(shape,fill_value,dtype),yorigin,xorigin,
                    origin,nodata_value,cellsize,proj_params)

def full_like(a,fill_value,*args,**kwargs):
    """
    Parameters
    ----------
        a           : np.ndarray         # the array to derive the shape and dtype attributes 
        fill_value  : scalar             # fill value

    Optional Parameters
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
          nodata_value=-9999, cellsize=1, proj_params=None):
    """
    Parameters
    ----------
        shape        : tuple          # shape of the returned grid  
 
    Optional Parameters
    ------------------
        dtype        : str/np.dtype   # type of the returned grid
        yorigin      : int/float      # y-value of the grid's origin
        xorigin      : int/float      # x-value of the grid's origin
        origin       : str            # position of the origin. One of:
                                            "ul" : upper left corner
                                            "ur" : upper right corner
                                            "ll" : lower left corner
                                            "lr" : lower right corner
        nodata_value : inf/float      # nodata or fill value
        cellsize     : int/float      # cellsize
        proj_params  : dict/None      # proj4 projection parameters

    Returns
    -------
        GeoArray

    Purpose
    -------
        Return a new GeoArray of given shape and type, without initializing entries.
    """
    return _factory(np.full(shape,nodata_value,dtype),yorigin,xorigin,
                    origin,nodata_value,cellsize,proj_params)

def empty_like(a,*args,**kwargs):
    """
    Parameters
    ----------
        a           : np.ndarray         # the array to derive the shape and dtype attributes 

    Optional Parameters
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
    return np.empty_like(a,*args,**kwargs)


def _factory(data, yorigin, xorigin, origin, nodata_value, cellsize, proj_params):
    origins = ("ul","ur","ll","lr")
    if origin not in origins:
        raise TypeError("Argument 'origin' must be on of '{:}'".format(origins))
    return GeoArray(data,yorigin,xorigin,origin,nodata_value,cellsize,proj_params)


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
        Construct an GeoArray from data in a file
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
    _FILEREFS.append(fobj)

    rasterband = fobj.GetRasterBand(1)
    geotrans   = fobj.GetGeoTransform()

    nrows      = fobj.RasterYSize        
    ncols      = fobj.RasterXSize
    nbands     = fobj.RasterCount

    dtype      = np.dtype(GDAL2DTYPE[rasterband.DataType])

    data       = fobj.GetVirtualMemArray(
        gdal.GF_Write, cache_size = nbands*nrows*ncols*dtype.itemsize
    )
    return _factory(data=data,yorigin=geotrans[3],xorigin=geotrans[0],
                     origin="ul",nodata_value=rasterband.GetNoDataValue(),
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
            band.SetNoDataValue(float(grid.nodata_value))
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

        
class GeoArray(np.ndarray):
    """
    Purpose
    -------
        This numpy.ndarray subclass adds geographic context to the data.
        A (hopfully growing) number of operations on the data and the writing
        to different file formats (see the variable _DRIVER_DICT) is supported.
        The Python GDAL bindings are used as backend, the GDAL virtual memory mapping
        to read the dataset. So when a file is opened only its metadata is read from
        storage, the actual data is only accessed when needed.

    Restrictions
    ------------
        1. The Gdal virtual memory mapping is only available on Linux!
        2. A GeoArray instance can be passed to any numpy function expecting a
           numpy.ndarray as argument and, in theory, all these functions should also return
           an object of the same type. In practice however not all functions do so and some
           will still return a numpy.ndarray.
        3. Adding the geographic information to the data does (at the moment) not imply
           any additional logic. If the shapes of two grids allow the succesful execution 
           of a certain operator/function your program will continue. It is within the responsability
           of the user to check whether a given operation makes sense within a geographic context 
           (e.g. grids cover the same spatial domain, share a common projection, etc.) or not.
           Overriding the __array_prepare__ method to implement the necessary checks would solve that
           issue for all operators and most functions. But there are a few edge cases, where numpy 
           functions are not routed through the __array_prepare__/__array_wrap__ mechanism. Implementing
           implicit checks would still mean, that there are some unchecked calls, beside
           pretending a (geographically) safe environment.
    """
   
    def __new__(cls, data, yorigin , xorigin, origin, nodata_value,
                cellsize, proj_params=None):

        obj = np.asarray(data).view(cls)
        obj.yorigin = yorigin
        obj.xorigin = xorigin
        obj.origin = origin
        obj._nodata_value = nodata_value
        obj.cellsize = cellsize
        obj.proj_params = proj_params
        return obj

    def __array_finalize__(self,obj):
        if obj is not None:
            self.xorigin = getattr(obj,'xorigin',None)
            self.yorigin = getattr(obj,'yorigin',None)
            self.origin = getattr(obj,'origin',None)
            self.cellsize = getattr(obj,'cellsize',None)
            self.proj_params = getattr(obj,'proj_params',None)
            self._nodata_value = getattr(obj,'_nodata_value',None)
            
    def __array_wrap__(self,result):
        if result.shape:
            return array(data=result,**self.header)
        return result[0]
    
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
         'yorigin': 100, 'xorigin': 55, 'nodata_value': -9999.0}

        """
        return {
            "yorigin"      : self.yorigin,
            "xorigin"      : self.xorigin,
            "origin"       : self.origin,
            "nodata_value" : self.nodata_value,            
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
            origin : str/None

        Returns
        -------
            2-tuple

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
                
    def _getNodataValue(self):
        """
        Parameters
        ----------
            None

        Returns
        -------
            int/float

        Purpose
        -------
            Return the nodata_value.
        """
        if type(self._nodata_value) != self.dtype.type:
            self._nodata_value = self.dtype.type(self._nodata_value)
        return self._nodata_value

    def _setNodataValue(self,value):
        """
        Parameters
        ----------
            value : int/float

        Returns
        -------
            None

        Purpose
        -------
            Set the nodata_value attribute. All nodata_values in the dataset 
            will be changed accordingly. Stored data will be read from disk, 
            so calling this property may be a costly operation.
        """
        if type(value) != self.dtype.type:
            value = self.dtype.type(value)
        self[self == self.nodata_value] = value
        self._nodata_value = value

    
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
            Return the number of bands in the dataset.
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
            Return the number of rows in the dataset.
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
            Return the number of columns in the dataset.
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
            y_idx :  int/float
            x_idx :  int/float

        Returns
        -------
            y_coor : int/float
            x_coor : int/float

        Purpose
        -------
            Return the coordinates of the grid cell definied by the given index values.
            The cell corner to which the returned values belong is definied by the 
            attribute origin (i.e "ll": lower-left, "ur": upper-right, ...).
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
            y_coor : int/float
            x_coor : int/float

        Returns
        -------
            y_idx  : int 
            x_idx  : int 

        Purpose
        -------
            Find the grid cell into which the given coordinates 
            fall and return its index values.
        """

        yorigin, xorigin = self.getOrigin("ul")

        y_idx = int(floor((yorigin - y_coor)/float(self.cellsize))) 

        x_idx = int(floor((x_coor - xorigin )/float(self.cellsize)))
        if y_idx < 0 or y_idx >= self.nrows or x_idx < 0 or x_idx >= self.ncols:
            raise ValueError("Given Coordinates not within the grid domain!")
        return y_idx,x_idx

    def trimGrid(self):
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
            grid if they contain only nodata value.
        """
        y_idx, x_idx = np.where(self != self.nodata_value)
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
            Removes the number of given cells from the 
            respective margin of the grid.

        Example
        -------
            removeCells(top=1, left=0, bottom=1, right=2)

                        |0|5|8|9| 
                        ---------
                        |1|2|0|0|         |1|2|
                        ---------   -->   -----
                        |3|4|4|1|         |3|4|
                        ---------
                        |0|0|0|1| 

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
            nodata_value  = self.nodata_value,
            cellsize    = self.cellsize,
            proj_params = self.proj_params,
        )

    def shrinkGrid(self,ymin=None,ymax=None,xmin=None,xmax=None):
        """
        Parameters
        ----------
            ymin, ymax, xmin, xmax : int/float

        Returns
        -------
            GeoArray

        Purpose
        -------
            Shrinks the grid in a way that the given bbox is still 
            within the grid domain.        
        """
        bbox = {
            "ymin": ymin if ymin else self.bbox["ymin"],
            "ymax": ymax if ymax else self.bbox["ymax"],
            "xmin": xmin if xmin else self.bbox["xmin"],
            "xmax": xmax if xmax else self.bbox["xmax"],
            }
        top    = floor(round((self.bbox["ymax"] - bbox["ymax"])
                             /self.cellsize, MAX_PRECISION))
        left   = floor(round((bbox["xmin"] - self.bbox["xmin"])
                            /self.cellsize, MAX_PRECISION))
        bottom = floor(round((bbox["ymin"] - self.bbox["ymin"])
                            /self.cellsize, MAX_PRECISION))
        right  = floor(round((self.bbox["xmax"] - bbox["xmax"])
                            /self.cellsize, MAX_PRECISION))

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
            Add the number of given cells to the respective
            margin of the grid.
        
        Example
        -------
            addCells(top=1, left=0, bottom=1, right=2)

                                      |0|0|0|0| 
                                      ---------
                        |1|2|         |1|2|0|0|    
                        ------   -->  ---------
                        |4|3|         |3|4|0|0|
                                      ---------
                                      |0|0|0|0| 
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
            nodata_value = self.nodata_value,
            cellsize     = self.cellsize,
            proj_params  = self.proj_params,
        )

        # the Ellipsis ensures that the function works 
        # for arrays with more than two dimensions
        out[Ellipsis, top:top+self.nrows, left:left+self.ncols] = self
        return out

    def enlargeGrid(self,ymin=None,ymax=None,xmin=None,xmax=None):
        """
        Parameters
        ----------
            ymin, ymax, xmin, xmax : int/float

        Returns
        -------
            None

        Purpose
        -------
            Enlarge the grid in a way that the given coordinates will 
            be part of the grid domain. Added rows/cols are filled with
            the grid's nodata value.
        """
        bbox = {
            "ymin": ymin if ymin else self.bbox["ymin"],
            "ymax": ymax if ymax else self.bbox["ymax"],
            "xmin": xmin if xmin else self.bbox["xmin"],
            "xmax": xmax if xmax else self.bbox["xmax"],
            }
        top    = ceil(round((bbox["ymax"] - self.bbox["ymax"])
                            /self.cellsize,MAX_PRECISION))
        left   = ceil(round((self.bbox["xmin"] - bbox["xmin"])
                            /self.cellsize,MAX_PRECISION))
        bottom = ceil(round((self.bbox["ymin"] - bbox["ymin"])
                            /self.cellsize,MAX_PRECISION))
        right  = ceil(round((bbox["xmax"] - self.bbox["xmax"])
                            /self.cellsize,MAX_PRECISION))
        return self.addCells(max(top,0),max(left,0),max(bottom,0),max(right,0))        

    def snapGrid(self,target):
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
        """
        diff = np.array(self.getOrigin()) - np.array(target.getOrigin(self.origin))
        dy,dx = abs(diff)%target.cellsize * np.sign(diff)

        if abs(dx) > self.cellsize/2.:
            dx += self.cellsize

        if abs(dy) > self.cellsize/2.:
            dy += self.cellsize

        self.xorigin -= dx
        self.yorigin -= dy

    nodata_value = property(fget = _getNodataValue, fset = _setNodataValue)

if __name__ == "__main__":

    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    
