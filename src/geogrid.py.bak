#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from math import ceil, floor
from geogridbase import _DummyGrid, _GdalGrid
from numpymember import NumpyMemberBase
from slicing import slicingBounds

"""
This Module provides a wrapper class around gdal/osr. That means it can
(or easily could) read/write all dataformats supported by gdal.

The returned _GeoGrid instance mimics the behaviour of a numpy.ndarray.
All operators work as expected and a any _GeoGrid instance can be passed
to numpy function (np.sum,np.exp,...)

Definition
----------
    def GeoGrid(fname=None,
                nbands=None, nrows=None, ncols=None,
                xllcorner=None, yllcorner=None, cellsize=None,
                dtype=None, data=None, nodata_value=None,
                proj_params=None)
        -> returns : _GeoGrid instance

Input
-----
There are three different way to create a GeoGrid Object

    1. Read from file
    input:
        fname        (str)                  full path to an Ascii Grid. File extension
                                            needs to be in (".asc","txt")
    optional input:
        dtype        (np.dtype/float/int)   -> default: read from file
        proj_params: (dict)                 -> default: None

    2. Create from existing numpy array
    input:
        data (numpy.array)
    optional input:
        xllcorner    (float/int)            -> default: 0
        yllcorner    (float/int)            -> default: 0
        cellsize     (float/int)            -> default: 1
        nodata_value (float/int)            -> default: -9999
        dtype        (np.dtype/int/float)   -> default: data.dtype
        proj_params  (dict)                 -> default: None

    3. Create an 'empty' grid
    input:
        nrows         (int)
        ncols         (int)
    optional input:
        nbands        (int)                 -> default: 1
        xllcorner     (float/int)           -> default: 0
        yllcorner     (float/int)           -> default: 0
        cellsize      (float/int)           -> default: 1
        nodata_value  (float/int)           -> default: -9999
        dtype         (np.dtype/int/float)  -> default: data.dtype
        proj_params   (dict)                -> default: None



Restrictions
------------
    - The grid coordinate system has to be rectangular and cells to be
      quadratic. Irregual shaped grids are currently not suported.
    
    - Although a grid may store information about map projections and 
      coordinate systems, no implicit transformations are done. 
      Please pay attention to work with consitent coordinate systems in 
      order to obtain correct results.

Note
----
    - The grid as well as the cell origins are defined as the lower left
       corner.

   

History
-------
    Written, DS, 2014/2015




"""

MAX_PRECISION  = 10

def GeoGrid(fname=None,            
            nbands=1, nrows=None, ncols=None,
            xllcorner=0, yllcorner=0, cellsize=1,
            dtype=np.float32, data=None, nodata_value=-9999,
            proj_params=None):

    kwargs = {k:v for k,v in locals().iteritems() if v != None} 
    # An existing file will be read -> only dtype and proj_params
    # are accpted as arguments
    if "fname" in kwargs:
        return _GeoGrid(
            _GdalGrid(
                kwargs["fname"],
                dtype=dtype,
                proj_params=proj_params)
        )
    else:
        # data is given
        if "data" in kwargs:
            kwargs["nbands"], kwargs["ncols"], kwargs["nrows"] = data.shape if data.ndim == 3 else (1,)+data.shape
            kwargs["dtype"] = kwargs.get("dtype",data.dtype)
            return _GeoGrid(
                _DummyGrid(
                    **kwargs)
            )
            
        # nrows and ncols are also good ... 
        elif ("nrows" in kwargs) and ("ncols" in kwargs):
            return _GeoGrid(_DummyGrid(**kwargs))
        else:
            raise TypeError("Insufficient arguments given!")
    
class _GeoGrid(NumpyMemberBase):
    """
    This class serves as a backend for the different reader classes which
    need to derive from DataReaderBase. The idea is to sperate data state
    logic which should be implemented in the aforementioned classes and
    a processing logic which is found here. 

    TODO: Return copies using the GeoGrid factory function 
          istead of creating the DummyGrids here.
    """
    def __init__(self, reader):
        self.__dict__["_reader"] = reader
        super(_GeoGrid,self).__init__(self,"_data")
        
    def __setattr__(self,name,value):
        self._reader.__setattr__(name,value)
        
    def __getattr__(self,name):
        try:
            return self._reader.__getattribute__(name)
        except AttributeError:
            raise AttributeError("'_GeoGrid' object has no attribute '{:}'".format(name))
        
    def __getitem__(self,slc):
        slicingBounds(slc,self.shape)
        return self._reader.__getitem__(slc)

    def __setitem__(self,slc,value):
        self._reader.__setitem__(slc,value)

    def __copy__(self):
        return _GeoGrid(
            _DummyGrid(**self.getDefinition())
        )

    def __deepcopy__(self,memo=None):
        kwargs = self.getDefinition()
        kwargs["data"] = self._data
        return _GeoGrid(_DummyGrid(**kwargs))        
        
    def __gridMatch(f):        
        def decorator(self,grid,*args,**kwargs):
            if self.cellsize != grid.cellsize:
                raise TypeError("Grid cellsizes do not match !!")
            dy = (self.yllcorner-grid.yllcorner)%self.cellsize
            dx = (self.xllcorner-grid.xllcorner)%self.cellsize
            check = (
                (
                    dy < sys.float_info.epsilon or
                    dy-self.cellsize < sys.float_info.epsilon
                    )
                and
                (
                    dx < sys.float_info.epsilon or
                    dx-self.cellsize < sys.float_info.epsilon
                    )
                )         
            if not check:
                raise TypeError("No way to match cell corners !!")
            return f(self,grid,*args,**kwargs)
        return decorator        
        
    @property
    def _data(self):
        # without this property 'wrapper' we end up with an segmentation fault !!
        return self._reader[:]
        
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
            
        out = {"yllcorner":self.yllcorner,
               "xllcorner":self.xllcorner,
               "cellsize":self.cellsize,
               "nbands":self.nbands,
               "nrows":self.nrows,
               "ncols":self.ncols,
               "dtype":self.dtype,
               "nodata_value":self.nodata_value,
               "proj_params":self.proj_params}
        
        for k in exclude:
            del out[k]
        return out
    
    def getIdxCoordinates(self,y_idx,x_idx):
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
        y_coor =  self.yllcorner + (self.nrows - y_idx - 1) * self.cellsize        
        x_coor =  self.xllcorner + x_idx * self.cellsize
        return y_coor,x_coor

    def getCoordinateIdx(self,y_coor,x_coor):
        """
        Input:
            y_coor: int/float
            x_coor: int/float
        Output:
            y_idx:  int/float
            x_idx:  int/float
        Purpose:
            Returns the indices of the cell in which the cooridnates fall        
        """
        y_idx = int(ceil((self.getBbox()["ymax"] - y_coor)/float(self.cellsize))) - 1
        x_idx = int(floor((x_coor - self.xllcorner)/float(self.cellsize))) 
        if y_idx < 0 or y_idx >= self.nrows or x_idx < 0 or x_idx >= self.ncols:
            raise IndexError("Given Coordinates not within the grid domain!")
        return y_idx,x_idx
        

    # def pointInGrid(self,y_coor,x_coor):
    #     """
    #     Input:
    #         y: Float/Integer
    #         x: Float/Integer
    #     Output:
    #         Boolean
    #     Purpose:
    #         Checks if the given coordinates fall within the grid domain
    #         and if the cell values of the respective cell value != nodata_value
    #     """
    #     try:
    #         return self._data[self.getCoordinateIdx(y_coor,x_coor)] != self.nodata_value
    #     except IndexError:
    #         return False

    def addCells(self,top=0,left=0,bottom=0,right=0):
        """
        Input:
            top:    Integer
            left:   Integer
            bottom: Integer
            right:  Integer
        Output:
            GeoGrid
        Purpose:
            Adds the number of given cells to the respective
            margin of the grid.
        Example:
            top=1, left=0, bottom=1, right=2
            nodata_value  0
        
                                       |0|0|0|0| 
                                       ---------
                         |1|2|         |1|2|0|0|    
                         ------   -->  ---------
                         |4|3|         |3|4|0|0|
                                       ---------
                                       |0|0|0|0| 
        
        """
        top    = max(top,0)
        left   = max(left,0) 
        bottom = max(bottom,0)
        right  = max(right,0)

        out = GeoGrid(
                nrows = self.nrows + top  + bottom,
                ncols = self.ncols + left + right,
                xllcorner = self.xllcorner - left*self.cellsize,
                yllcorner = self.yllcorner - bottom*self.cellsize ,
                cellsize = self.cellsize,
                dtype = self.dtype,
                data = None,
                nodata_value = self.nodata_value,
                proj_params = self.proj_params,
        )

        # the Ellipsis ensures that the function works with 
        # arrays with more than two dimensions
        out[Ellipsis, top:top+self.nrows, left:left+self.ncols] = self._data        
        return out

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
            nodata_value  0
        
                         |0|5|8|9| 
                         ---------
                         |1|2|0|0|         |1|2|
                         ---------   -->   -----
                         |3|4|4|1|         |3|4|
                         ---------
                         |0|0|0|1| 
        
        """

        top    = max(top,0)
        left   = max(left,0) 
        bottom = max(bottom,0)
        right  = max(right,0)
        
        out = GeoGrid(
                nrows = self.nrows - top  - bottom,
                ncols = self.ncols - left - right,
                xllcorner = self.xllcorner + left*self.cellsize,
                yllcorner = self.yllcorner + bottom*self.cellsize,
                cellsize = self.cellsize,
                data = None,
                dtype = self.dtype,
                proj_params = self.proj_params,
                nodata_value = self.nodata_value,
        )

        # the Ellipsis ensures that the function works with 
        # arrays with more than two dimensions
        out[:] = self[Ellipsis, top:top+out.nrows, left:left+out.ncols]
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
        self_bbox = self.getBbox()
        top    = ceil(round((bbox["ymax"] - self_bbox["ymax"])
                            /self.cellsize,MAX_PRECISION))
        left   = ceil(round((self_bbox["xmin"] - bbox["xmin"])
                            /self.cellsize,MAX_PRECISION))
        bottom = ceil(round((self_bbox["ymin"] - bbox["ymin"])
                            /self.cellsize,MAX_PRECISION))
        right  = ceil(round((bbox["xmax"] - self_bbox["xmax"])
                            /self.cellsize,MAX_PRECISION))
        return self.addCells(max(top,0),max(left,0),max(bottom,0),max(right,0))        

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
        self_bbox = self.getBbox()
        top    = floor(round((self_bbox["ymax"] - bbox["ymax"])
                             /self.cellsize, MAX_PRECISION))
        left   = floor(round((bbox["xmin"] - self_bbox["xmin"])
                             /self.cellsize, MAX_PRECISION))
        bottom = floor(round((bbox["ymin"] - self_bbox["ymin"])
                             /self.cellsize, MAX_PRECISION))
        right  = floor(round((self_bbox["xmax"] - bbox["xmax"])
                             /self.cellsize, MAX_PRECISION))
        return self.removeCells(max(top,0),max(left,0),max(bottom,0),max(right,0))        

    @__gridMatch
    def mergeGrid(self,grid):
        """
        Input:
            GeoGrid
        Purpose:
            Inserts the data of the argument grid based on its
            geolocation
        Restrictions:
            The cellsizes of the grids must be identical and cells
            in the common area must match
        """
        bbox = self.getBbox()
        y_offset,x_offset = self.getCoordinateIdx(
            bbox["ymax"] - grid.cellsize, bbox["xmin"])
        self[y_offset:y_offset+grid.nrows,x_offset:x_offset+grid.ncols]\
            = grid.data

    @__gridMatch
    def maskGrid(self,grid):
        """
        Input:
            GeoGrid
        Purpose:
            Sets all values in grid to nodata_value where the
            argument contains nodata_value
        """
        bbox = self.getBbox()
        y_offset,x_offset = self.getCoordinateIdx(
            bbox["ymax"] - grid.cellsize, bbox["xmin"])
        y_idx,x_idx = np.where(grid.data == grid.nodata_value)
        self[y_idx+y_offset,x_idx+x_offset] = self.nodata_value

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
        y_idx, x_idx = np.where(self._data != self.nodata_value)
        try:
            return self.removeCells(
                top=min(y_idx),bottom=self.nrows-max(y_idx)-1,
                left=min(x_idx),right=self.ncols-max(x_idx)-1)
        except ValueError:
            return self
        

    def snapGrid(self,grid):
        """
        Input:
            GeoGrid
        Output:
            None
        Purpose:
            Fit the map origin to the lower-left corner
            of the nearest cell in the input grid.
        Restrictions:
            The shift will only alter the grid coordinates.
            No changes to the data will be done. In case of large
            shifts the physical integrety of the data might
            be disturbed!
        """

        dy = (self.yllcorner-grid.yllcorner)%self.cellsize
        dx = (self.xllcorner-grid.xllcorner)%self.cellsize

        if dy > self.cellsize/2.: dy = (self.cellsize-dy) * -1
        if dx > self.cellsize/2.: dx = (self.cellsize-dx) * -1

        self.yllcorner -= dy
        self.xllcorner -= dx
        
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
    
        
if __name__== "__main__":

    pass

    
