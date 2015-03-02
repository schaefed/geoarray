#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re, os
import gdal, osr
import numpy as np
from numpymember import NumpyMemberBase, COMPARISON_OPERATORS
from slicing import slicingBounds
# import geogridfuncs as ggfuncs

# needs to be extended
_DRIVER_DICT = {
    ".tif" : "GTiff",
    ".asc" : "AAIGrid",
    ".bmp" : "BMP",
    ".img" : "HFA",
    ".jpg" : "JPEG",
}

gdal.PushErrorHandler('CPLQuietErrorHandler')
#gdal.UseExceptions()

# def removeCells(grid,top=0,left=0,bottom=0,right=0):
#     """
#     Input:
#         top, left, bottom, right:    Integer
#     Output:
#         GeoGrid
#     Purpose:
#         Removes the number of given cells from the 
#         respective margin of the grid
#     Example:
#         top=1, left=0, bottom=1, right=2
#         nodata_value  0

#                      |0|5|8|9| 
#                      ---------
#                      |1|2|0|0|         |1|2|
#                      ---------   -->   -----
#                      |3|4|4|1|         |3|4|
#                      ---------
#                      |0|0|0|1| 

#     """

#     top    = int(max(top,0))
#     left   = int(max(left,0))
#     bottom = int(max(bottom,0))
#     right  = int(max(right,0))

#     out = GeoGrid(
#             nrows        = grid.nrows - top  - bottom,
#             ncols        = grid.ncols - left - right,
#             xllcorner    = grid.xllcorner + left*grid.cellsize,
#             yllcorner    = grid.yllcorner + bottom*grid.cellsize,
#             cellsize     = grid.cellsize,
#             data         = None,
#             dtype        = grid.dtype,
#             proj_params  = grid.proj_params,
#             nodata_value = grid.nodata_value,
#     )

#     # the Ellipsis ensures that the function works with 
#     # arrays with more than two dimensions
#     out[:] = grid[Ellipsis, top:top+out.nrows, left:left+out.ncols]
#     return out

def GeoGrid(fname=None,            
            nbands=1, nrows=None, ncols=None,
            xllcorner=0, yllcorner=0, cellsize=1,
            dtype=np.float32, data=None, nodata_value=-9999,
            proj_params=None):
    
    kwargs = {k:v for k,v in locals().iteritems() if v != None} 

    # An existing file will be read -> only dtype and proj_params
    # are accpted as arguments
    if "fname" in kwargs:
        return _GdalGrid(kwargs["fname"])
        
    # data is given
    elif "data" in kwargs:
        kwargs["nbands"], kwargs["ncols"], kwargs["nrows"] = data.shape if data.ndim == 3 else (1,)+data.shape        
        kwargs["dtype"] = kwargs.get("dtype", data.dtype)
        return _DummyGrid(**kwargs)
        
    # nrows and ncols are also good ... 
    elif ("nrows" in kwargs) and ("ncols" in kwargs):
            return _DummyGrid(**kwargs)
            
    raise TypeError("Insufficient arguments given!")

class _GeoGridBase(NumpyMemberBase):
    
    """
    The File Reader/Writer Base class. All reader/writer classes NEED 
    to inherit from _GeoGridBase. Its purpose is to garantee a common 
    interface on which _GeoGrid depends on. It give meaningful defaults
    to the child class attributes and ensure data state consistency.
    
    TODO:
        Implement an input arguments check
    """        
    def __init__(self,
            nbands=None, nrows=None, ncols=None,
            xllcorner=None, yllcorner=None, cellsize=None,
            dtype=None, data=None, nodata_value=None,        
            proj_params=None
    ):

        super(_GeoGridBase,self).__init__(
            self,"data",
            # hooks = {c: lambda x: x._setDataType(np.bool) for c in COMPARISON_OPERATORS}
        )

        self.nbands        = nbands if nbands else 1
        self.nrows         = nrows
        self.ncols         = ncols

        self.xllcorner     = xllcorner
        self.yllcorner     = yllcorner
        self.cellsize      = cellsize
        self.proj_params   = proj_params

        self._nodata_value = nodata_value

        self._dtype        = np.dtype(dtype).type
        self._data         = self._initData(data)
        self._readmask     = self._initReadMask(data)
        self._consistentTypes()

    def _initReadMask(self,data):
        if data == None:
            return np.zeros(self.shape, dtype=np.bool)            
        return np.ones(self.shape,dtype=np.bool)

    def _initData(self,data):
        if data == None:
            data = np.empty(self.shape, dtype=self.dtype)
            data.fill(self.nodata_value)
        return data
        
    def getDefinition(self,exclude=None):
        """
        Input:
            exclude: list/tuple
        Output:
            {"xllcorner" : int/float, "yllcorner" : int/float, "cellsize" : int/float
            {"nodata_value" : int/float, "nrows" : int/float, "ncols": int/float}
        Purpose:
            Returns the basic definition of the instance. The values given in
            the optional exclude argument will not be part of the definition
            dict
        Note:
            The output of this method is sufficient to create a new
            albeit empty GeoGrid instance.
        """
        if not exclude:
            exclude = ()
            
        out = {
            "yllcorner":self.yllcorner,
            "xllcorner":self.xllcorner,
            "cellsize":self.cellsize,
            "nbands":self.nbands,
            "nrows":self.nrows,
            "ncols":self.ncols,
            "dtype":self.dtype,
            "nodata_value":self.nodata_value,
            "proj_params":self.proj_params
        }
        
        for k in exclude:
            del out[k]
        return out
        
    def getCoordinates(self):
        """
        Input:
            None
        Output:
            y: numpy.ndarray (1D)
            x: numpy.ndarray (1D)
        Purpose:
            Returns the coordinates of the lower left corner of all cells as
            two sepererate numpy arrays. 
        """
        x = np.arange(self.xllcorner, self.xllcorner + self.ncols * self.cellsize,\
                      self.cellsize)
        y = np.arange(self.yllcorner, self.yllcorner + self.nrows * self.cellsize,\
                      self.cellsize)
        return y,x

    def getBbox(self):
        """
        Input:
            None
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
        return {"ymin":self.yllcorner,
                "ymax":self.yllcorner + self.nrows * self.cellsize,
                "xmin":self.xllcorner,
                "xmax":self.xllcorner + self.ncols * self.cellsize}

    def _getData(self):
        return self.__getitem__(slice(None))

    def _setData(self,value):
        # print value.shape,self.shape
        if not value.shape == self.shape:
            raise ValueError
        self._data = value
        
    def _consistentTypes(self):
        """
           Reflect dtype changes
        """
        self._nodata_value = self.dtype(self._nodata_value)
        self._data = self[:].astype(self._dtype)
            
    def _squeeze(self,data):
        """
           Squeeze the possibly 1-length first dimension for
           convinient indexing.
        """
        try:
            return np.squeeze(data,0)
        except ValueError:
            return data
        
    def __copy__(self):
        return _DummyGrid(**self.getDefinition())
        
    def __deepcopy__(self,memo=None):
        kwargs = self.getDefinition()
        kwargs["data"] = self.data
        return _DummyGrid(**kwargs)
                
    def _getNodataValue(self):
        """
            nodata_value getter
        """
        return self._nodata_value

    def _setNodataValue(self,value):
        """
            nodata_value setter
            All nodata_values in the dataset will be changed accordingly.
            Stored data will be read from disk, so calling this
            property may be a costly operation
        """
        self[...,self[:] == self._nodata_value] = self.dtype(value)
        self._nodata_value = self.dtype(value)

    def _getShape(self):
        """
            shape getter
            1-length nbands will be skipped
        """
        if self.nbands > 1:
            return self.nbands, self.nrows, self.ncols
        return self.nrows, self.ncols

    def _getDataType(self):
        """
            dtype getter
        """
        return self._dtype

    def _setDataType(self,value):
        """
            dtype setter
            If the data is already read from file its type will be
            changed. _consistentTypes() is invoked
        """
        self._dtype = np.dtype(value).type
        self._consistentTypes()
    
    def __getitem__(self,slc):
        """
            slicing operator invokes the data reading
            TODO: Implement a slice reading from file
        """
        if not np.all(self._readmask[slc]):
            self._data = self.dtype(self._readData())
            self._readmask[slc] = True
        # (top,bottom),(left,right) = slicingBounds(slc,self.shape)[-2:]
        # print self._data.shape
        return self._squeeze(self._data)[slc]
        # return self._data[slc]
        
    def __setitem__(self,slc,value):
        # print self._data.shape
        # print self._readmask.shape
        self._data[slc] = value
        self._readmask[slc] = True
        
    def write(self,fname):
        _GridWriter(self).write(fname)
        
    nodata_value  = property(fget=lambda self:            self._getNodataValue(),
                             fset=lambda self, value:     self._setNodataValue(value))
    shape         = property(fget=lambda self:            self._getShape())
    dtype         = property(fget=lambda self:            self._getDataType(),
                             fset=lambda self, value:     self._setDataType(value))
    data          = property(fget=lambda self:            self._getData(),
                             fset=lambda self, value:     self._setData(value))
    

    
class _DummyGrid(_GeoGridBase):
    """
        A simple dummy data class needed as reader attribute to
        _GeoGrid if data is given on initialisation
    """
    def __init__(self,*args,**kwargs):
        super(_DummyGrid,self).__init__(*args,**kwargs)
        
    def _readData(self):
        """
            returns an array filled with nodata_values
        """
        out = np.empty((self.nbands,self.nrows,self.ncols),dtype=self.dtype)
        out.fill(self.nodata_value)        
        return out
            
                
class _GdalGrid(_GeoGridBase):
    def __init__(self,fname):
        self.fobj = self._open(fname)
        trans = self.fobj.GetGeoTransform()
        band = self.fobj.GetRasterBand(1)
        pparams = self._proj4Params() 
        
        super(_GdalGrid,self).__init__(
            nbands       = self.fobj.RasterCount,
            nrows        = self.fobj.RasterYSize,
            ncols        = self.fobj.RasterXSize,
            xllcorner    = trans[0],
            yllcorner    = self._yllcorner(trans[3],trans[5],self.fobj.RasterYSize),
            cellsize     = self._cellsize(trans[1],trans[5]),
            nodata_value = band.GetNoDataValue(),
            proj_params  = pparams if pparams else {},
            dtype        = gdal.GetDataTypeName(band.DataType),
        )

    def _open(self,fname):
        fobj = gdal.Open(fname)
        if fobj:
            return fobj
        raise IOError("Could not open file")
        
    def _readData(self):
        return self.fobj.ReadAsArray()
        
    def _yllcorner(self,yulcorner,cellsize,nrows):
        if cellsize < 0:
            return float(yulcorner) + (cellsize * nrows)
        return yulcorner
        
    def _cellsize(self,x_cellsize,y_cellsize):
        if abs(x_cellsize) == abs(y_cellsize):
            return abs(x_cellsize)
        raise NotImplementedError(
            "Diverging cellsizes in x and y direction are not allowed yet!")    
        
    def _proj4Params(self):
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.fobj.GetProjection())
        proj_string = srs.ExportToProj4()
        proj_params = filter(None,re.split("[+= ]",proj_string))
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
        return "+{:}".format(" +".join(
            ["=".join(pp) for pp in self.fobj.proj_params.items()])
        )
    

    def _writeGdalMemory(self):
        driver = gdal.GetDriverByName("MEM")
        out = driver.Create(
            "",self.fobj.ncols,self.fobj.nrows,self.fobj.nbands,
            gdal.GetDataTypeByName(self.fobj.dtype.__name__)
        )
        out.SetGeoTransform(
            (self.fobj.xllcorner,self.fobj.cellsize,0,
             self.fobj.yllcorner + (self.fobj.nrows * self.fobj.cellsize),
             0,self.fobj.cellsize*-1)
        )
        srs = osr.SpatialReference()
        srs.ImportFromProj4(self._proj4String())
        out.SetProjection(srs.ExportToWkt())
        for n in xrange(self.fobj.nbands):
            band = out.GetRasterBand(n+1)
            band.SetNoDataValue(float(self.fobj.nodata_value)) 
            band.WriteArray(self.fobj[:])
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
                
    
if __name__ == "__main__":

    pass
    
    
