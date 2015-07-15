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

gdal.PushErrorHandler('CPLQuietErrorHandler')

        
# "ul", "ll", "ur", "lr"
def GeoGrid(fname=None, data=None, shape=(),
            yorigin=0, xorigin=0, origin="ul",
            # lbound=None, rbound=None, ubound=None, bbound=None,
            dtype=np.float64, fill_value=-9999, cellsize=1,
            proj_params=None):
    """
    expected data shape: (nbands,y,x)
    """
    # read data
    if fname:
        return _GdalGrid(fname)

    # initialize empty data
    if data == None:
        data = np.full(shape, fill_value, dtype=dtype)
        
    # wrap array
    # if 0 < data.ndim < 4:
    nrows, ncols = ((1,1) + data.shape) [-2:]
    if origin in ("ul", "ll", "ur", "lr"):
        return _GeoGrid(data, yorigin, xorigin, origin, cellsize, fill_value, proj_params)
                    
    raise TypeError("Insufficient arguments given!")

class _GeoGrid(np.ndarray):
    
    """
    ORIGIN: upper left corner
    """        
    def __new__(cls, array, yorigin , xorigin, origin,cellsize,
                fill_value, proj_params=None):

        obj = np.asarray(array).view(cls)
        obj.yorigin = yorigin
        obj.xorigin = xorigin
        obj.origin = origin
        obj.cellsize = cellsize
        obj._fill_value = fill_value
        obj.proj_params = proj_params
        return obj

    def __array_finalize__(self,obj):
        # if isinstance(obj,_GeoGrid):
        if obj is not None:
            self.xorigin = getattr(obj,'xorigin',None)
            self.yorigin = getattr(obj,'yorigin',None)
            self.origin = getattr(obj,'origin',None)
            self.cellsize = getattr(obj,'cellsize',None)
            self.proj_params = getattr(obj,'proj_params',None)
            self._fill_value = getattr(obj,'_fill_value',None)
            # self._propagateType()
            
    @property
    def header(self):
        """
        Output:
        Purpose:
            Returns the basic definition of the instance. The values given in
            the optional exclude argument will not be part of the definition
            dict
        Note:
            The output of this method is sufficient to create a new
            albeit empty _GeoGrid using the GeoGrid factory function.
        """
            
        return {
            "shape"       : self.shape,
            "yorigin"     : self.yorigin,
            "xorigin"     : self.xorigin,
            "origin"      : self.origin,
            "dtype"       : self.dtype,
            "fill_value"  : self.fill_value,            
            "cellsize"    : self.cellsize,
            "proj_params" : self.proj_params
        }
        
    @property
    def bbox(self):
        """
        Output:
            {"ymin": int/float, "ymax": int/float,
             "xmin": int/float, "xmax": int/float}
        Purpose:
            Returns the bounding box of the GeoGrid Instance.
        Note:
            Bounding box is here definied as a rectangle entirely enclosing
            the GeoGrid Instance. That means that ymax and xmax values are
            calculated as the coordinates of the last cell + cellsize.
            Trying to acces the point ymax/xmax will therefore fail, as these
            coorindates actually point to the cell nrows+1/ncols+1
        """
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
                
    def _getFillValue(self):
        """
            fill_value getter
        """
        if type(self._fill_value) != self.dtype.type:
            self._fill_value = self.dtype.type(self._fill_value)
        return self._fill_value

    def _setFillValue(self,value):
        """
            fill_value setter
            All fill_values in the dataset will be changed accordingly.
            Stored data will be read from disk, so calling this
            property may be a costly operation
        """
        if type(value) != self.dtype.type:
            value = self.dtype.type(value)
        self[self == self.fill_value] = value
        self._fill_value = value

    def __array_wrap__(self,array):
        if array.shape:
            return GeoGrid(data=array,**self.header)
        return array[0]
    
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
        _GridWriter(self).write(fname)
        
    fill_value    = property(fget=_getFillValue, fset=_setFillValue)
           
class _GdalGrid(_GeoGrid):
        
    def __new__(cls,fname):
        
        fobj        = cls.__open(fname)
        rasterband  = fobj.GetRasterBand(1)
        fill_value  = rasterband.GetNoDataValue()
        geotrans    = fobj.GetGeoTransform()
        proj_params = cls.__projParams(fobj)
        nrows       = fobj.RasterYSize        
        ncols       = fobj.RasterXSize
        nbands      = fobj.RasterCount
        dtype       = np.dtype(gdal.GetDataTypeName(rasterband.DataType))

        data = fobj.GetVirtualMemArray(
                gdal.GF_Write,
                cache_size = nbands*nrows*ncols*dtype.itemsize
            )
        
        obj = super(_GdalGrid,cls).__new__(
            cls,
            array       = data,
            yorigin     = geotrans[3],
            xorigin     = geotrans[0],
            origin      = "ul",
            cellsize    = cls.__cellsize(geotrans),
            fill_value  = fill_value,
            proj_params = proj_params,
        )

        # this reference is needed to avoid the situation that the open
        # file's refcount reaches zeros before the GetVirtualMemArray array
        # does...
        obj._fobj = fobj
        return obj
        
    def __array_finalize__(self,obj):
        if obj is not None:
            self._fobj       = getattr(obj,"_fobj",None)                        
            super(_GdalGrid,self).__array_finalize__(obj)
                
    @classmethod
    def __open(cls,fname):
        fobj = gdal.OpenShared(fname)
        if fobj:
            return fobj
        raise IOError("Could not open file")
           
    @staticmethod
    def __cellsize(geotrans):       
        if abs(geotrans[1]) == abs(geotrans[5]):
            return abs(geotrans[1])
        raise NotImplementedError(
            "Diverging cellsizes in x and y direction are not allowed yet!")    

    @staticmethod
    def __projParams(fobj):
        srs = osr.SpatialReference()
        srs.ImportFromWkt(fobj.GetProjection())
        proj_params = filter(None,re.split("[+= ]",srs.ExportToProj4()))
        return dict(zip(proj_params[0::2],proj_params[1::2]))
    
    
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
            band.SetNoDataValue(float(self.fobj.fill_value)) 
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
