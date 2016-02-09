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
}
TYPEMAP.update([reversed(x) for x in TYPEMAP.items()])

# The open gdal file objects need to outlive their GeoArray
# instance. Therefore they are stored globaly.
# _FILEREFS = []

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

class Projection(object):
    def __init__(self, arg):

        self._srs = osr.SpatialReference()

        if isinstance(arg, int):
            self._srs.ImportFromEpsg(arg)
        elif isinstance(arg, dict):
            params =  "+{:}".format(" +".join(
                ["=".join(map(str, pp)) for pp in arg.items()])
            )
            self._srs.ImportFromProj4(params)
        elif isinstance(arg, str):
            self._srs.ImportFromWkt(arg)
            
    def getProj4(self):
        tmp = self._srs.ExportToProj4()
        proj = [x for x in re.split("[+= ]", tmp) if x]
        return dict(zip(proj[0::2], proj[1::2]))
        
    def getWkt(self):
        return self._srs.ExportToWkt()

    def getReference(self):
        if str(self._srs):
            return self._srs

    # def getTransformer(self, tprojer):
    #     tx = osr.CoordinateTransformation(
    #         self.getReference(), tprojer.getReference()
    #     )
    #     return tx.TransformPoint

class Transformer(object):
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
        
def _fromfile(fname):
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
    raise IOError("Could not open file")
       
def _fromDataset(fobj):

    # _FILEREFS.append(fobj)

    rasterband = fobj.GetRasterBand(1)
    geotrans   = fobj.GetGeoTransform()

    nrows      = fobj.RasterYSize
    ncols      = fobj.RasterXSize
    nbands     = fobj.RasterCount

    dtype      = np.dtype(TYPEMAP[rasterband.DataType])

    # if "linux" in sys.platform:
    #     # use GDAL's virtual memmory mappings
    #     data       = fobj.GetVirtualMemArray(
    #         gdal.GF_Write, cache_size = nbands*nrows*ncols*dtype.itemsize
    #     )
    # else:
    #     data = fobj.ReadAsArray()

    data = fobj.ReadAsArray()
    
    return {
        "data":data, "yorigin":geotrans[3], "xorigin":geotrans[0],
        "origin":"ul", "fill_value":rasterband.GetNoDataValue(),
        "cellsize":(geotrans[5], geotrans[1]),
        "proj" : fobj.GetProjection()
    }

# return _factory(
#         data=data, yorigin=geotrans[3], xorigin=geotrans[0],
#         origin="ul", fill_value=rasterband.GetNoDataValue(),
#         cellsize=(geotrans[5], geotrans[1]),
#         proj = fobj.GetProjection(),
#         fobj=fobj
#     )


def _memDataset(grid): #, projection):

    """
    Create GDAL memory dataset
    """
    
    driver = gdal.GetDriverByName("MEM")
    out = driver.Create(
        "", grid.ncols, grid.nrows, grid.nbands, TYPEMAP[str(grid.dtype)]
    )
    out.SetGeoTransform(
        (grid.xorigin, grid.cellsize[1], 0,
         grid.yorigin, 0, grid.cellsize[0])
    )

    out.SetProjection(grid.proj.getWkt())
    for n in xrange(grid.nbands):
        band = out.GetRasterBand(n+1)
        band.SetNoDataValue(float(grid.fill_value))
        band.WriteArray(grid[n] if grid.ndim>2 else grid)
            
    # out.FlushCache()
    return out


def _tofile(fname, geoarray):
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

    memset = _memDataset(geoarray) #, _proj2Gdal(geoarray.proj_params))
    outdriver = _getDriver(_fnameExtension(fname))
    outdriver.CreateCopy(fname, memset, 0)

