#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re, os
import gdal, osr
import numpy as np

# should be extended, for available options see:
# http://www.gdal.org/formats_list.html
_DRIVER_DICT = {
    ".tif" : "GTiff",
    ".asc" : "AAIGrid",
    ".img" : "HFA",
    ".png" : "PNG",
}

# type mapping: there is no boolean data type in GDAL
TYPEMAP = {
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
    1            : "int8",
    2            : "uint16",
    3            : "int16",
    4            : "uint32",
    5            : "int32",
    6            : "float32",
    7            : "float64",
    10           : "complex64",
    11           : "complex128",
    
}

# The open gdal file objects need to outlive their GeoArray
# instance. Therefore they are stored globaly.
# _FILEREFS = []

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

class _Projection(object):
    def __init__(self, arg):

        self._srs = osr.SpatialReference()

        if isinstance(arg, int):
            self._srs.ImportFromProj4("+init=epsg:{:}".format(arg))
        elif isinstance(arg, dict):
            params =  "+{:}".format(" +".join(
                ["=".join(map(str, pp)) for pp in arg.items()])
            )
            self._srs.ImportFromProj4(params)
        elif isinstance(arg, str):
            self._srs.ImportFromWkt(arg)
        elif isinstance(arg, _Projection):
            self._srs.ImportFromWkt(arg.getWkt())
            
    def getProj4(self):
        tmp = self._srs.ExportToProj4()
        proj = [x for x in re.split("[+= ]", tmp) if x]
        return dict(zip(proj[0::2], proj[1::2]))
        
    def getWkt(self):
        return self._srs.ExportToWkt()

    def getReference(self):
        if str(self._srs):
            return self._srs

    def __str__(self):
        return str(self.getProj4())

    def __repr__(self):
        return str(self.getProj4())
    
class _Transformer(object):
    def __init__(self, sproj, tproj):
        """
        Arguments
        ---------
        sproj, tproj : Projection
        
        Purpose
        -------
        Encapsulates the osr Cordinate Transformation functionality
        """
        self._tx = osr.CoordinateTransformation(
            sproj.getReference(), tproj.getReference()
        )

    def __call__(self, y, x):
        try:
            xt, yt, _ = self._tx.TransformPoint(x, y)
        except NotImplementedError:
            raise AttributeError("Projections not correct or given!")
        return yt, xt
        
def _fromFile(fname):
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

    """
    
    fobj = gdal.OpenShared(fname)
    if fobj:
        return _fromDataset(fobj)
    raise IOError("Could not open file: {:}".format(fname))
       
def _fromDataset(fobj):

    rasterband = fobj.GetRasterBand(1)
    geotrans   = fobj.GetGeoTransform()

    return {
        "data"       : fobj.ReadAsArray(),
        "yorigin"    : geotrans[3],
        "xorigin"    : geotrans[0],
        "origin"     : "ul",
        "fill_value" : rasterband.GetNoDataValue(),
        "cellsize"   : (geotrans[5], geotrans[1]),
        "proj"       : fobj.GetProjection(),
        # "fobj"       : fobj,
    }

def _memDataset(grid): #, projection):

    """
    Create GDAL memory dataset
    """
    
    driver = gdal.GetDriverByName("MEM")
    out = driver.Create(
        "", grid.ncols, grid.nrows, grid.nbands, TYPEMAP[str(grid.dtype)]
    )
    out.SetGeoTransform(
        (
            grid.bbox["xmin"], abs(grid.cellsize[1]), 0,
            grid.bbox["ymax"], 0, abs(grid.cellsize[0])*-1)
    )
    out.SetProjection(grid.proj.getWkt())
    for n in xrange(grid.nbands):
        band = out.GetRasterBand(n+1)
        band.SetNoDataValue(float(grid.fill_value))
        band.WriteArray(grid[n] if grid.ndim>2 else grid)
            
    # out.FlushCache()
    return out

def _adaptPrecision(data, dtype):
    try:
        tinfo = np.finfo(dtype)
    except ValueError:
        tinfo = np.iinfo(dtype)

    # if np.any(data < tinfo.min):

def _toFile(fname, geoarray):
    def _fnameExtension(fname):
        return os.path.splitext(fname)[-1].lower()

    def _getDriver(fext):
        """
        Guess driver from file name extension
        """
        if fext in _DRIVER_DICT:
            driver = gdal.GetDriverByName(_DRIVER_DICT[fext])
            metadata = driver.GetMetadata_Dict()
            if "YES" == metadata.get("DCAP_CREATE",metadata.get("DCAP_CREATECOPY")):
                return driver
            raise IOError("Datatype canot be written")
        raise IOError("No driver found for filename extension '{:}'".format(fext))

    def _getDatatype(driver):
        tnames = tuple(driver.GetMetadata_Dict()["DMD_CREATIONDATATYPES"].split(" "))
        types = tuple(gdal.GetDataTypeByName(t) for t in tnames)
        tdict = tuple((gdal.GetDataTypeSize(t), t) for t in types)
        otype = max(tdict, key=lambda x: x[0])[-1]
        return np.dtype(TYPEMAP[otype])

        
    memset = _memDataset(geoarray) #, _proj2Gdal(geoarray.proj_params))
    driver = _getDriver(_fnameExtension(fname))
    driver.CreateCopy(fname, memset, 0)
    _adaptPrecision(geoarray, np.float32)
    _adaptPrecision(geoarray, _getDatatype(driver))
