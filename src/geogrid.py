#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re, os, copy
import gdal, osr
import numpy as np
from numpymember import NumpyMemberBase, COMPARISON_OPERATORS
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

def GeoGrid(fname=None,            
            nbands=1, nrows=None, ncols=None,
            xllcorner=0, yllcorner=0, cellsize=1,
            dtype=None, data=None, fill_value=-9999,
            proj_params=None):
    
    kwargs = {k:v for k,v in locals().iteritems() if v != None} 
    # An existing file will be read -> only dtype and proj_params
    # are accpted as arguments
    if "fname" in kwargs:
        return _GdalGrid(kwargs["fname"], dtype)
        
    # data is given
    elif "data" in kwargs:
        kwargs["nbands"], kwargs["nrows"], kwargs["ncols"] = ((1,1) + data.shape)[-3:]
        kwargs["dtype"] = data.dtype
        return _GeoGrid(**kwargs)
        
    # nrows and ncols are also good ... 
    elif ("nrows" in kwargs) and ("ncols" in kwargs):
        kwargs["dtype"] = kwargs.get("dtype",np.float32)
        return _GeoGrid(**kwargs)
            
    raise TypeError("Insufficient arguments given!")

        
class _GeoGrid(NumpyMemberBase):
    
    """
    The File Reader/Writer Base class. All reader/writer classes NEED 
    to inherit from _GeoGrid. Its purpose is to garantee a common 
    interface on which _GeoGrid depends on. It give meaningful defaults
    to the child class attributes and ensure data state consistency.
    
    TODO:
        Implement an input arguments check
    """        
    def __init__(self,
            nbands=None, nrows=None, ncols=None,
            xllcorner=None, yllcorner=None, cellsize=None,
            dtype=None, data=None, fill_value=None,
            proj_params=None
    ):
        super(_GeoGrid,self).__init__(
            self,"data",
            hooks = {c: lambda x: x._setDataType(bool) for c in COMPARISON_OPERATORS}
        )
                
        self.nbands        = nbands if nbands else 1
        self.nrows         = nrows
        self.ncols         = ncols

        self.xllcorner     = xllcorner
        self.yllcorner     = yllcorner
        self.cellsize      = cellsize
        self.proj_params   = proj_params

        self._fill_value   = fill_value
        self._dtype        = np.dtype(dtype).type
        self._data         = self._initData(data)

        self._consistentTypes()

    def _initData(self,data):
        if data == None:
            data = np.full(self.shape,self.fill_value, dtype=self.dtype)
        return data

    @property
    def header(self):
        """
        Output:
            {"xllcorner" : int/float, "yllcorner" : int/float, "cellsize" : int/float
            {"fill_value" : int/float, "nrows" : int/float, "ncols": int/float}
        Purpose:
            Returns the basic definition of the instance. The values given in
            the optional exclude argument will not be part of the definition
            dict
        Note:
            The output of this method is sufficient to create a new
            albeit empty GeoGrid instance.
        """
            
        return {
            "yllcorner":self.yllcorner,
            "xllcorner":self.xllcorner,
            "cellsize":self.cellsize,
            "nbands":self.nbands,
            "nrows":self.nrows,
            "ncols":self.ncols,
            "dtype":self.dtype,
            "fill_value":self.fill_value,
            "proj_params":self.proj_params
        }
        
    @property
    def coordinates(self):
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
        return {
            "ymin":self.yllcorner,
            "ymax":self.yllcorner + self.nrows * self.cellsize,
            "xmin":self.xllcorner,
            "xmax":self.xllcorner + self.ncols * self.cellsize
        }


    def _getMask(self):
        return self.data == self.fill_value
        
    def _getData(self):
        return self._data

    def _setData(self,value):
        if not value.shape == self.shape:
            raise ValueError(
                "could not broadcast input array from shape {:} into shape {:}"\
                .format(value.shape,self.shape))
        self._data = value
        
    def _consistentTypes(self):
        """
           Reflect dtype changes
        """
        self.yllcorner = self.dtype(self.yllcorner)
        self.xllcorner = self.dtype(self.xllcorner)
        self._fill_value = self.dtype(self._fill_value)
        if np.dtype(self._data.dtype).type != self._dtype:
            self._data = self._data.astype(self._dtype)
        
    def __eq__(self,other):
        super(_GeoGrid,self).__eq__(other)
        
    def __copy__(self):
        return GeoGrid(**self.header)
        
    def __deepcopy__(self,memo=None):
        kwargs = copy.deepcopy(self.header)
        kwargs["data"] = self.data
        return GeoGrid(**kwargs)
        
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
        fill_value = self.dtype(value)
        self._data[self.mask] = fill_value
        self._fill_value = fill_value
        
    @property
    def shape(self):
        """
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
    
                
    def write(self,fname):
        _GridWriter(self).write(fname)
        
    fill_value    = property(fget=lambda self:            self._getFillValue(),
                             fset=lambda self, value:     self._setFillValue(value))
    dtype         = property(fget=lambda self:            self._getDataType(),
                             fset=lambda self, value:     self._setDataType(value))
    data          = property(fget=lambda self:            self._getData(),
                             fset=lambda self, value:     self._setData(value))
    mask          = property(fget=lambda self:            self._getMask())
                        
class _GdalGrid(_GeoGrid):
    def __init__(self,fname,dtype=None):
        self.fobj             = self._open(fname)
        trans                 = self.fobj.GetGeoTransform()
        band                  = self.fobj.GetRasterBand(1)
        pparams               = self._proj4Params()
        # the file's original fill value
        # self._file_fill_value = band.GetNoDataValue()
        
        super(_GdalGrid,self).__init__(
            nbands       = self.fobj.RasterCount,
            nrows        = self.fobj.RasterYSize,
            ncols        = self.fobj.RasterXSize,
            xllcorner    = trans[0],
            yllcorner    = self._yllcorner(trans[3],trans[5],self.fobj.RasterYSize),
            cellsize     = self._cellsize(trans[1],trans[5]),
            fill_value   = band.GetNoDataValue(),
            proj_params  = pparams if pparams else {},
            dtype        = gdal.GetDataTypeName(band.DataType) if dtype == None else dtype,            
        )

        self._readmask     = self._initReadMask()
        
    def _initReadMask(self):
        return np.zeros(self.shape, dtype=np.bool)            
        
    def _open(self,fname):
        fobj = gdal.Open(fname)
        if fobj:
            return fobj
        raise IOError("Could not open file")
        
    def _readData(self):
        """
        Reads all data from file
        """
        return self.fobj.ReadAsArray()

    def _getData(self):
        if not np.all(self._readmask):
            # only set where mask allows it
            self._data[~self._readmask] = self._readData()[~self._readmask]
            self._readmask[~self._readmask] = True
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
        self._readmask[slc] = True
        super(_GdalGrid,self).__setitem__(slc,value)
        
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
        params = self.fobj.proj_params
        if params:
            return "+{:}".format(" +".join(
                ["=".join(pp) for pp in params.items()])
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


class InfoArray(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
                                 order)
        # Finally, we must return the newly created object:
        return obj

    def __array__(self,dtype):
        print "Hallo"
        return super(InfoArray,self).__array__(dtype)
        
    def __array_finalize__(self, obj):
        if obj is None: return

    def __getitem__(self,slc):
        # print type(slc), slc.data.dtype
        out = super(InfoArray,self).__getitem__(slc)
        # print type(slc), slc.data.dtype
        return out
