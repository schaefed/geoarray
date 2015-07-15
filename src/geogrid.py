#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re, os
import gdal, osr
import numpy as np

# needs to be extended
_DRIVER_DICT = {
    ".tif" : "GTiff",
    ".asc" : "AAIGrid",
    ".bmp" : "BMP",
    ".img" : "HFA",
    ".jpg" : "JPEG",
    ".png" : "PNG"
}

_FILEREF = None

gdal.PushErrorHandler('CPLQuietErrorHandler')

def array(data, dtype=None, yorigin=0, xorigin=0, origin="ul",
          nodata_value=-9999,cellsize=1,proj_params=None):
    
    return _GeoArray(np.asarray(data) if not dtype else np.asarray(data,dtype),
                    yorigin,xorigin,origin,nodata_value,cellsize,proj_params)
        
def zeros(shape,dtype=np.float64,yorigin=0, xorigin=0, origin="ul",
          nodata_value=-9999,cellsize=1,proj_params=None):
    return _GeoArray(np.zeros(shape,dtype),yorigin,xorigin,
                    origin,nodata_value,cellsize,proj_params)

def ones(shape,dtype=np.float64,yorigin=0, xorigin=0, origin="ul",
         nodata_value=-9999,cellsize=1,proj_params=None):
    return _GeoArray(np.ones(shape,dtype),yorigin,xorigin,
                    origin,nodata_value,cellsize,proj_params)

def full(shape,fill_value,dtype=None,yorigin=0, xorigin=0, origin="ul",
         nodata_value=-9999,cellsize=1,proj_params=None):
    return _GeoArray(np.full(shape,fill_value,dtype),yorigin,xorigin,
                    origin,nodata_value,cellsize,proj_params)

def empty(shape,dtype=None,yorigin=0, xorigin=0, origin="ul",
          nodata_value=-9999,cellsize=1,proj_params=None):
    return _GeoArray(np.full(shape,nodata_value,dtype),yorigin,xorigin,
                    origin,nodata_value,cellsize,proj_params)

def fromfile(fname):

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

    # the file object needs to survive 'data'
    global _FILEREF

    _FILEREF   = _openFile(fname)

    rasterband = _FILEREF.GetRasterBand(1)
    geotrans   = _FILEREF.GetGeoTransform()

    nrows      = _FILEREF.RasterYSize        
    ncols      = _FILEREF.RasterXSize
    nbands     = _FILEREF.RasterCount

    dtype      = np.dtype(gdal.GetDataTypeName(rasterband.DataType))

    data       = _FILEREF.GetVirtualMemArray(
        gdal.GF_Write, cache_size = nbands*nrows*ncols*dtype.itemsize
    )
    return _GeoArray(data,geotrans[3],geotrans[0],"ul",_cellsize(geotrans),
                    rasterband.GetNoDataValue(),_projParams(_FILEREF))


def tofile(fname,geogrid):
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
            gdal.GetDataTypeByName(str(grid.dtype))
        )
        out.SetGeoTransform(
            (grid.xorigin, grid.cellsize,0,
             grid.yorigin, 0, grid.cellsize)
        )
        out.SetProjection(projection)
        for n in xrange(grid.nbands):
            band = out.GetRasterBand(n+1)
            band.SetNoDataValue(float(grid.nodata_value)) 
            band.WriteArray(grid[(n,Ellipsis) if grid.nbands > 1 else (Ellipsis)])
        out.FlushCache()
        return out
            
    memset = _writeGdalMemory(geogrid, _projection(geogrid))
    outdriver = _getDriver(_fnameExtension(fname))
    out = outdriver.CreateCopy(fname,memset,0)
    errormsg = gdal.GetLastErrorMsg()
    if errormsg or not out:
        raise IOError(errormsg)
    
class _GeoArray(np.ndarray):
    
    """
    ORIGIN: upper left corner
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
            
    @property
    def header(self):
        return {
            "yorigin"     : self.yorigin,
            "xorigin"     : self.xorigin,
            "origin"      : self.origin,
            "nodata_value"  : self.nodata_value,            
            "cellsize"    : self.cellsize,
            "proj_params" : self.proj_params
        }
        
    @property
    def bbox(self):
        yopp = self.nrows * self.cellsize
        xopp = self.ncols * self.cellsize
        return { 
            "ymax": self.yorigin if self.origin[0] == "u" else self.yorigin - yopp,
            "ymin": self.yorigin if self.origin[0] == "l" else self.yorigin + yopp,
            "xmin": self.xorigin if self.origin[1] == "l" else self.xorigin - xopp,
            "xmax": self.xorigin if self.origin[1] == "r" else self.xorigin + xopp,
        }


    def getOrigin(self,origin=None):
        """
        Return the grid's corner coorindates. Defaults to the origin
        corner, any other corner may be specifed with the 'origin' argument, 
        which should be one of: 'ul','ur','ll','lr'
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
            nodata_value getter
        """
        if type(self._nodata_value) != self.dtype.type:
            self._nodata_value = self.dtype.type(self._nodata_value)
        return self._nodata_value

    def _setNodataValue(self,value):
        """
            nodata_value setter
            All nodata_values in the dataset will be changed accordingly.
            Stored data will be read from disk, so calling this
            property may be a costly operation
        """
        if type(value) != self.dtype.type:
            value = self.dtype.type(value)
        self[self == self.nodata_value] = value
        self._nodata_value = value

    def __array_wrap__(self,result):
        if result.shape:
            return array(data=result,**self.header)
        return result[0]
    
    @property
    def nbands(self):
        try:
            return self.shape[-3]
        except IndexError:
            return 0

    @property
    def nrows(self):
        try:
            return self.shape[-2]
        except IndexError:
            return 0

    @property
    def ncols(self):
        try:
            return self.shape[-1]
        except IndexError:
            return 0
    
    def write(self,fname):
        tofile(fname,self)
        # _GridWriter(self).write(fname)
        
    nodata_value = property(fget = _getNodataValue, fset = _setNodataValue)

    
class _GridWriter(object):

    def __init__(self,fobj):
        self.fobj = fobj
        
    def _fnameExtension(self,fname):
        return os.path.splitext(fname)[-1].lower()

    def _getDriver(self,fext):
        if fext in _DRIVER_DICT:
            driver = gdal.GetDriverByName(_DRIVER_DICT[fext])
            metadata = driver.GetMetadata_Dict()
            if "YES" == metadata.get("DCAP_CREATE",metadata.get("DCAP_CREATECOPY")):
                return driver
            raise IOError("Datatype canot be written")            
        raise IOError("No driver found for filenmae extension '{:}'".format(fext))

    def _proj4String(self):
        params = self.fobj.proj_params
        if params:
            return "+{:}".format(" +".join(
                ["=".join(pp) for pp in params.items()])
                             )

    def _writeGdalMemory(self):
        driver = gdal.GetDriverByName("MEM")
        out = driver.Create(
            "",self.fobj.ncols,self.fobj.nrows,self.fobj.nbands,
            gdal.GetDataTypeByName(str(self.fobj.dtype))
        )
        out.SetGeoTransform(
            (self.fobj.xorigin, self.fobj.cellsize,0,
             self.fobj.yorigin, 0, self.fobj.cellsize)
        )
        srs = osr.SpatialReference()
        srs.ImportFromProj4(self._proj4String())
        out.SetProjection(srs.ExportToWkt())
        for n in xrange(self.fobj.nbands):
            band = out.GetRasterBand(n+1)
            band.SetNoDataValue(float(self.fobj.nodata_value)) 
            band.WriteArray(self.fobj[(n,Ellipsis) if self.fobj.nbands > 1 else (Ellipsis)])
        out.FlushCache()
        return out

    def write(self,fname):
        memset = self._writeGdalMemory()
        fext = self._fnameExtension(fname)        
        outdriver = self._getDriver(fext)
        out = outdriver.CreateCopy(fname,memset,0)
        errormsg = gdal.GetLastErrorMsg()
        if errormsg or not out:
            raise IOError(errormsg)
