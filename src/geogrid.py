#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re, os, copy
import gdal, osr
import numpy as np
from numpymember import NumpyMemberBase
from slicing import slicingBounds, fullSlices

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


def gridOrigin(nrows, ncols, cellsize, lbound=None, rbound=None, ubound=None, bbound=None):

    yorigin = ubound if ubound else bbound - cellsize * (nrows + 1),
    xorigin = lbound if lbound else rbound - cellsize * ncols + 1        

    return yorigin, xorigin
        
    
def GeoGrid(fname=None, data=None, shape=(),
            yorigin=0, xorigin=0,
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
    if 1 < data.ndim < 4:
        nrows,ncols = data.shape[-2:]
        # if any((lbound,rbound)) and any((ubound,bbound)):
        #     yorigin,xorigin = gridOrigin(nrows,ncols,lbound,rbound,ubound,bbound)
        return _GeoGrid(data, yorigin, xorigin, cellsize, fill_value, proj_params)
            
        
    raise TypeError("Insufficient arguments given!")

        
class _GeoGrid(NumpyMemberBase):
    
    """
    ORIGIN: upper left corner
    """        
    def __init__(self, data, yorigin, xorigin, cellsize,
                 fill_value, proj_params=None):
        super(_GeoGrid,self).__init__(
            self,"data",
        )
        
        self._data = data        
        self.xorigin = xorigin
        self.yorigin = yorigin
        self.cellsize = cellsize
        self._fill_value = fill_value
        self.proj_params = proj_params
            
        self._propagateType()


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
            albeit empty GeoGrid instance.
        """
            
        return {
            "shape"       : self.shape,
            "yorigin"     : self.yorigin,
            "xorigin"     : self.xorigin,
            "dtype"       : self.dtype,
            "fill_value"  : self.fill_value,            
            "cellsize"    : self.cellsize,
            "proj_params" : self.proj_params
        }
        
    # @property
    # def coordinates(self):
    #     """
    #     Input:
    #         None
    #     Output:
    #         y: numpy.ndarray (1D)
    #         x: numpy.ndarray (1D)
    #     Purpose:
    #         Returns the coordinates of the lower left corner of all cells as
    #         two sepererate numpy arrays. 
    #     """
    #     x = np.arange(self.xorigin, self.xorigin + self.ncols * self.cellsize,\
    #                   self.cellsize)
    #     y = np.arange(self.yllcorner, self.yllcorner + self.nrows * self.cellsize,\
    #                   self.cellsize)
    #     return y,x

#     @property
#     def bbox(self):
#         """
#         Output:
#             {"ymin": int/float, "ymax": int/float,
#              "xmin": int/float, "xmax": int/float}
#         Purpose:
#             Returns the bounding box of the GeoGrid Instance.
#         Note:
#             Bounding box is here definied as a rectangle entirely enclosing
#             the GeoGrid Instance. That means that ymax and xmax values are
#             calculated as the coordinates of the last cell + cellsize.
#             Trying to acces the point ymax/xmax will therefore fail, as these
#             coorindates actually point to the cell nrows+1/ncols+1
#         """
#         return {
#             "ymin":self.yllcorner,
#             "ymax":self.yllcorner + self.nrows * self.cellsize,
#             "xmin":self.xllcorner,
#             "xmax":self.xllcorner + self.ncols * self.cellsize
#         }


    # def _getMask(self):
    #     return self.data == self.fill_value
                
    def _propagateType(self):
        """
           Reflect dtype changes
        """
        dtype = self.data.dtype.type
        self.yorigin = dtype(self.yorigin)
        self.xorigin = dtype(self.xorigin)
        self._fill_value = dtype(self._fill_value)
                
    def __copy__(self):
        return GeoGrid(**self.header)
        
    def __deepcopy__(self,memo=None):
        return GeoGrid(data=self.data.copy(),**self.header)
        
    def _getFillValue(self):
        """
            fill_value getter
        """
        return self._fill_value

    def _setFillValue(self,value):
        """
            fill_value setter
            All fill_values in the dataset will be changed accordingly.
            Stored data will be read from disk, so calling this
            property may be a costly operation
        """
        # In order to create a correct mask the fill_value must
        # be set last        
        # fill_value = self.dtype(value)
        self.data[self.data == self.fill_value] = value
        self._fill_value = value

    def _getData(self):
        return self._data

    def _setData(self,data):
        self._data = data

    @property
    def shape(self):
        """
        """
        return self.data.shape
    
    def _getDataType(self):
        """
            dtype getter
        """
        return self.data.dtype

    def _setDataType(self,value):
        """
            dtype setter
            If the data is already read from file its type will be
            changed. _consistentTypes() is invoked
        """
        self.data = self.data.astype(value)
        self._propagateType()
    
    @property
    def nbands(self):
        return 1 if len(self.shape) == 2 else self.shape[0]

    @property
    def nrows(self):
        return self.shape[-2]

    @property
    def ncols(self):
        return self.shape[-1]
    
    def __getitem__(self,slc):
        return self.data[slc]
        
    def write(self,fname):
        _GridWriter(self).write(fname)
        
    fill_value    = property(fget=lambda self:            self._getFillValue(),
                             fset=lambda self, value:     self._setFillValue(value))
    dtype         = property(fget=lambda self:            self._getDataType(),
                             fset=lambda self, value:     self._setDataType(value))
    data          = property(fget=lambda self:            self._getData(),
                             fset=lambda self, data:     self._setData(data))
    
class _GdalGrid(_GeoGrid):
    def __init__(self,fname,dtype=None):
        self.__fobj       = self.__open(fname)
        self.__geotrans   = self.__fobj.GetGeoTransform()
        self.__rasterband = self.__fobj.GetRasterBand(1)
        self.__fill_value = self.__rasterband.GetNoDataValue()
        self.__shape      = self.__shape()
        self.__dtype      = gdal.GetDataTypeName(self.__rasterband.DataType)
        self.__readmask   = np.zeros(self.__shape,dtype=bool)
        
        super(_GdalGrid,self).__init__(            
            data = np.full(self.__shape,self.__fill_value, self.__dtype),
            yorigin     = self.__geotrans[3],
            xorigin     = self.__geotrans[0],
            cellsize    = self.__cellsize(),
            fill_value   = self.__fill_value,
            proj_params = self.__proj4Params()
        )
        
    def __open(self,fname):
        fobj = gdal.Open(fname)
        if fobj:
            return fobj
        raise IOError("Could not open file")
        
    def __shape(self):
        nbands = self.__fobj.RasterCount
        if nbands > 1:
            return (nbands, self.__fobj.RasterYSize, self.__fobj.RasterXSize)
        return (self.__fobj.RasterYSize, self.__fobj.RasterXSize)
        
    def _getData(self):
        if not np.all(self.__readmask):            
            data = self.__fobj.ReadAsArray()
            self._data[~self.__readmask]  = data[~self.__readmask]
            self.__readmask[~self.__readmask] = True
        return self._data
        
    def __getitem__(self,slc):
        """
            slicing operator invokes the data reading
            TODO: Implement a slice reading from file
        """
        # if not np.all(self._readmask[slc]):
        # slices = slicingBounds(slc,self.shape)
        # if fullSlices(slices,self.shape):
        #     self._getData()
        self._getData()
        return super(_GdalGrid,self).__getitem__(slc)

    def __setitem__(self,slc,value):
        self.__readmask[slc] = True
        super(_GdalGrid,self).__setitem__(slc,value)
                
    def __cellsize(self):
        
        if abs(self.__geotrans[1]) == abs(self.__geotrans[5]):
            return abs(self.__geotrans[1])
        raise NotImplementedError(
            "Diverging cellsizes in x and y direction are not allowed yet!")    
        
    def __proj4Params(self):
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.__fobj.GetProjection())
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
