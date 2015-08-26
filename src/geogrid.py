#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides a numpy.ndarray subclass adding


>>> import numpy as np
>>> import geogrid as gg

>>> arr = np.array([-9.,-9.       ,-9.       ,-9.       ,-9.       ,-9.,
...                 -9.,2.16512767,4.97776467,4.2279204 ,0.        ,-9.,
...                 -9.,8.25658422,0.01238773,5.05858306,8.33503939,-9.,
...                 -9.,7.53470443,7.15304826,9.45150218,8.79359049,-9.,
...                 -9.,0.0536634 ,0.42101194,0.22721601,1.1458486 ,-9.,
...                 -9.,6.79183025,2.50622739,3.76725118,3.97934707,-9.,
...                 -9.,0.        ,0.24743279,1.4627512 ,0.38430722,-9.,
...                 -9.,5.30171261,0.        ,3.17667353,3.80908144,-9.,
...                 -9.,7.12445478,4.83891708,6.10898131,2.93801857,-9.,
...                 -9.,2.56170107,2.54503559,1.72767934,0.        ,-9.,
...                 -9.,-9.       ,-9.       ,-9.       ,-9.       ,-9.,])

# fill value
>>> nodata_value = -9

# x-coordinate of the grid origin
>>> xorigin = 63733

# y-coordinate of the grid origin
>>> yorigin = 78867
>>> cellsize = 23.3

# Set the origin to the upper left corner. One of ("ul", "ll", "ur", "ul" )
>>> origin = "ul"

>>> grid = array(arr, yorigin=yorigin, xorigin=xorigin, origin=origin, cellsize=cellsize)
>>> grid

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
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "float64": 7,
    "complex64": 10,
    "complex128": 11,
}

GDAL2DTYPE = {v:k for k,v in DTYPE2GDAL.items()}

# The open gdal file objects need to outlive its _GeoGrid
# instance. Therefore they are stored globally.
_FILEREFS = []

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
    return _GeoArray(data=data,yorigin=geotrans[3],xorigin=geotrans[0],
                     origin="ul",nodata_value=rasterband.GetNoDataValue(),
                    cellsize=_cellsize(geotrans),proj_params=_projParams(fobj))


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

        
class _GeoArray(np.ndarray):
    
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
            "ymin": self.yorigin if self.origin[0] == "l" else self.yorigin - yopp,
            "ymax": self.yorigin if self.origin[0] == "u" else self.yorigin + yopp,
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
            return 1

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
    
    def tofile(self,fname):
        tofile(fname, self)
        
    def indexCoordinates(self,y_idx,x_idx):
        """
        Input:
            y_idx:  int/float
            x_idx:  int/float
        Output:
            y_coor: int/float
            x_coor: int/float
        Purpose:
            Returns the coordinates of the cell definied by the input indices.
            Works as the reverse of getCoordinateIdx.
        Note:
            The arguments are expected to be numpy arrax indices, i.e. the 
            upper left corner of the grid is the origin. 
        """        
        if (y_idx < 0 or x_idx < 0) or (y_idx >= self.nrows or x_idx >= self.ncols):
            raise ValueError("Index out of bounds !")
        yorigin, xorigin = self.getOrigin("ul")
        y_coor =  yorigin - y_idx * self.cellsize        
        x_coor =  xorigin + x_idx * self.cellsize
        return y_coor, x_coor

    def coordinateIndex(self,y_coor,x_coor):
        """
        Input:
            y_coor: int/float
            x_coor: int/float
        Output:
            y_idx:  int/float
            x_idx:  int/float
        Purpose:
            Returns the indices of the cell in which the coordinates fall        
        """
        yorigin, xorigin = self.getOrigin("ul")

        y_idx = int(floor((yorigin - y_coor)/float(self.cellsize))) 

        x_idx = int(floor((x_coor - xorigin )/float(self.cellsize)))
        if y_idx < 0 or y_idx >= self.nrows or x_idx < 0 or x_idx >= self.ncols:
            raise ValueError("Given Coordinates not within the grid domain!")
        return y_idx,x_idx

    def trimGrid(self):
        """
        Input:
            None
        Output:
            GeoGrid
        Purpose:
            Removes rows/cols containing only nodata values from the
            margins of the GeoGrid Instance
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
        Input:
            top, left, bottom, right:    Integer
        Output:
            GeoGrid
        Purpose:
            Removes the number of given cells from the 
            respective margin of the grid
        Example:
            top=1, left=0, bottom=1, right=2
            fill_value  0

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

    def shrinkGrid(self,bbox):
        """
        Input:
            bbox: {"ymin": int/float, "ymax": int/float,
                "xmin": int/float, "xmax": int/float}
        Output:
            GeoGrid
        Purpose:
            Shrinks the grid in a way that the given bbox is still 
            within the grid domain.        
        """
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
        Input:
            top:    int 
            left:   int 
            bottom: int 
            right:  int 
        Output:
            GeoGrid
        Purpose:
            Adds the number of given cells to the respective
            margin of the grid.
        Example:
            top=1, left=0, bottom=1, right=2
            fill_value  0

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

        # the Ellipsis ensures that the function works with 
        # arrays with more than two dimensions
        out[Ellipsis, top:top+self.nrows, left:left+self.ncols] = self
        return out

    def enlargeGrid(self,bbox):
        """
        Input:
            bbox: {"ymin": int/float, "ymax": int/float,
                "xmin": int/float, "xmax": int/float}
        Output:
            GeoGrid
        Purpose:
            Enlarges the GeoGrid Instance in a way that the given bbox will 
            be part of the grid domain. Added rows/cols ar filled with
            the grid's nodata value
        """
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
        Input:
            GeoGrid
        Output:
            None
        Purpose:
            Shift the grid origin in a way that it matches the nearest corner
            of any gridcell in target
        Restrictions:
            The shift will only alter the grid coordinates.
            No changes to the data will be done. In case of large
            shifts the physical integrety of the data might
            be disturbed!
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

    
