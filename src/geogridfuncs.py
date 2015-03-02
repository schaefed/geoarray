#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from math import ceil, floor
from geogrid import GeoGrid

MAX_PRECISION  = 10

def _gridMatch(f):        
    def decorator(grid,other,*args,**kwargs):
        if grid.cellsize != other.cellsize:
            raise TypeError("Grid cellsizes do not match !!")
        dy = (grid.yllcorner-other.yllcorner)%grid.cellsize
        dx = (grid.xllcorner-other.xllcorner)%grid.cellsize
        check = (
            (
                dy < sys.float_info.epsilon or
                dy-grid.cellsize < sys.float_info.epsilon
                )
            and
            (
                dx < sys.float_info.epsilon or
                dx-grid.cellsize < sys.float_info.epsilon
                )
            )         
        if not check:
            raise TypeError("No way to match cell corners !!")
        return f(grid,other,*args,**kwargs)
    return decorator        
        
def _offset(grid,other):
    bbox = grid.getBbox()
    return coordinateIndex(grid, bbox["ymax"] - other.cellsize, bbox["xmin"])
    
def indexCoordinates(grid,y_idx,x_idx):
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
    y_coor =  grid.yllcorner + (grid.nrows - y_idx - 1) * grid.cellsize        
    x_coor =  grid.xllcorner + x_idx * grid.cellsize
    return y_coor,x_coor


def coordinateIndex(grid,y_coor,x_coor):
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
    y_idx = int(ceil((grid.getBbox()["ymax"] - y_coor)/float(grid.cellsize))) - 1
    x_idx = int(floor((x_coor - grid.xllcorner)/float(grid.cellsize))) 
    if y_idx < 0 or y_idx >= grid.nrows or x_idx < 0 or x_idx >= grid.ncols:
        raise IndexError("Given Coordinates not within the grid domain!")
    return y_idx,x_idx

    
def addCells(grid,top=0,left=0,bottom=0,right=0):
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
            nrows        = grid.nrows + top  + bottom,
            ncols        = grid.ncols + left + right,
            xllcorner    = grid.xllcorner - left*grid.cellsize,
            yllcorner    = grid.yllcorner - bottom*grid.cellsize ,
            cellsize     = grid.cellsize,
            dtype        = grid.dtype,
            data         = None,
            nodata_value = grid.nodata_value,
            proj_params  = grid.proj_params,
    )

    # the Ellipsis ensures that the function works with 
    # arrays with more than two dimensions
    out[Ellipsis, top:top+grid.nrows, left:left+grid.ncols] = grid.data
    return out


def removeCells(grid,top=0,left=0,bottom=0,right=0):
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
            nrows        = grid.nrows - top  - bottom,
            ncols        = grid.ncols - left - right,
            xllcorner    = grid.xllcorner + left*grid.cellsize,
            yllcorner    = grid.yllcorner + bottom*grid.cellsize,
            cellsize     = grid.cellsize,
            data         = None,
            dtype        = grid.dtype,
            proj_params  = grid.proj_params,
            nodata_value = grid.nodata_value,
    )

    # the Ellipsis ensures that the function works with 
    # arrays with more than two dimensions
    out[:] = grid[Ellipsis, top:top+out.nrows, left:left+out.ncols]
    return out



def enlargeGrid(grid,bbox):
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
    grid_bbox = grid.getBbox()
    top    = ceil(round((bbox["ymax"] - grid_bbox["ymax"])
                        /grid.cellsize,MAX_PRECISION))
    left   = ceil(round((grid_bbox["xmin"] - bbox["xmin"])
                        /grid.cellsize,MAX_PRECISION))
    bottom = ceil(round((grid_bbox["ymin"] - bbox["ymin"])
                        /grid.cellsize,MAX_PRECISION))
    right  = ceil(round((bbox["xmax"] - grid_bbox["xmax"])
                        /grid.cellsize,MAX_PRECISION))
    return addCells(grid,max(top,0),max(left,0),max(bottom,0),max(right,0))        

def shrinkGrid(grid,bbox):
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
    grid_bbox = grid.getBbox()
    top    = floor(round((grid_bbox["ymax"] - bbox["ymax"])
                         /grid.cellsize, MAX_PRECISION))
    left   = floor(round((bbox["xmin"] - grid_bbox["xmin"])
                         /grid.cellsize, MAX_PRECISION))
    bottom = floor(round((bbox["ymin"] - grid_bbox["ymin"])
                         /grid.cellsize, MAX_PRECISION))
    right  = floor(round((grid_bbox["xmax"] - bbox["xmax"])
                         /grid.cellsize, MAX_PRECISION))
    return removeCells(grid,max(top,0),max(left,0),max(bottom,0),max(right,0))        

        
@_gridMatch
def mergeGrid(grid,other):
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
    y_offset,x_offset = _offset(grid,other)
    grid[y_offset:y_offset+other.nrows,x_offset:x_offset+other.ncols] = grid.data

@_gridMatch
def maskGrid(grid,other):
    """
    Input:
        GeoGrid
    Purpose:
        Sets all values in grid to nodata_value where the
        argument contains nodata_value
    """
    y_offset,x_offset = _offset(grid,other)
    y_idx,x_idx = np.where(grid.data == grid.nodata_value)
    grid[y_idx+y_offset,x_idx+x_offset] = grid.nodata_value

def trimGrid(grid):
    """
    Input:
        None
    Output:
        GeoGrid
    Purpose:
        Removes rows/cols containing only nodata values from the
        margins of the GeoGrid Instance
    """
    y_idx, x_idx = np.where(grid != grid.nodata_value)
    try:
        return removeCells(
            grid,
            top=min(y_idx),bottom=grid.nrows-max(y_idx)-1,
            left=min(x_idx),right=grid.ncols-max(x_idx)-1)
    except ValueError:
        return grid


def snapGrid(grid,other):
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

    dy = (grid.yllcorner-other.yllcorner)%grid.cellsize
    dx = (grid.xllcorner-other.xllcorner)%grid.cellsize

    if dy > grid.cellsize/2.:
        dy = (grid.cellsize-dy) * -1
    if dx > grid.cellsize/2.:
        dx = (grid.cellsize-dx) * -1

    grid.yllcorner -= dy
    grid.xllcorner -= dx


    
