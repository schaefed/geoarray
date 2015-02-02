#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest, copy
import numpy as np
from geogrid import GeoGrid

FNAME = "dem.asc"
PROJ_PARAMS = {
    'lon_0': '148.8',   'lat_0': '0',
    'y_0'  : '3210000', 'x_0': '4321000',
    'units': 'm', 
    'proj' : 'laea',    'ellps': 'WGS84'
}

class TestGeoGrid(unittest.TestCase):
    def setUp(self):
        self.grid = GeoGrid(FNAME,proj_params=PROJ_PARAMS)

    def test_typeConsistency(self):
        def check(grid):
            self.assertTrue(grid.dtype == type(grid.nodata_value))
        checkgrid = copy.deepcopy(self.grid)
        checkgrid.dtype = np.int32
        check(self.grid)
        check(checkgrid)
            
    def test_setNodataValue(self):
        rpcvalue = -2222
        checkgrid = copy.deepcopy(self.grid)
        checkgrid.nodata_value = rpcvalue
        nodatapos1 = np.where(checkgrid == rpcvalue)
        nodatapos2 = np.where(self.grid == self.grid.nodata_value)
        for pos1,pos2 in zip(nodatapos1,nodatapos2):
            self.assertEqual(np.sum(pos1-pos2),0)
        self.assertEqual(checkgrid.nodata_value, rpcvalue)
        
    def test_setDtype(self):
        rpctype = np.float64
        checkgrid = copy.deepcopy(self.grid)
        checkgrid.dtype = rpctype
        self.assertEqual(checkgrid.dtype,rpctype)
        self.assertEqual(checkgrid.dtype,rpctype)
        
    def test_addCells(self):
        padgrid = self.grid.addCells(1,1,1,1)
        self.assertEqual(np.sum(padgrid[1:-1,1:-1] - self.grid),0)

        padgrid = self.grid.addCells(0,0,0,0)
        self.assertEqual(np.sum(padgrid[:] - self.grid),0)
        
    def test_enlargeGrid(self):
        newbbox = {"xmin" : self.grid.bbox["xmin"] - 2.5 * self.grid.cellsize,
                   "ymin" : self.grid.bbox["ymin"] -  .7 * self.grid.cellsize,
                   "xmax" : self.grid.bbox["xmax"] +  .1 * self.grid.cellsize,
                   "ymax" : self.grid.bbox["ymax"] + 6.1 * self.grid.cellsize,}
        enlrgrid = self.grid.enlargeGrid(newbbox)
        self.assertEqual(enlrgrid.nrows, self.grid.nrows + 1 + 7)
        self.assertEqual(enlrgrid.ncols, self.grid.ncols + 3 + 1)

    def test_shrinkGrid(self):
        newbbox = {"xmin" : self.grid.bbox["xmin"] + 2.5 * self.grid.cellsize,
                   "ymin" : self.grid.bbox["ymin"] +  .7 * self.grid.cellsize,
                   "xmax" : self.grid.bbox["xmax"] -  .1 * self.grid.cellsize,
                   "ymax" : self.grid.bbox["ymax"] - 6.1 * self.grid.cellsize,}
        shrgrid = self.grid.shrinkGrid(newbbox)
        self.assertEqual(shrgrid.nrows, self.grid.nrows - 0 - 6)
        self.assertEqual(shrgrid.ncols, self.grid.ncols - 2 - 0)

    def test_removeCells(self):
        rmgrid = self.grid.removeCells(1,1,1,1)
        self.assertEqual(np.sum(rmgrid - self.grid[1:-1,1:-1]) , 0)
        rmgrid = self.grid.removeCells(0,0,0,0)
        self.assertEqual(np.sum(rmgrid - self.grid) , 0)

        
    def test_trimGrid(self):
        trimgrid = self.grid.trimGrid()
        self.assertTrue(np.any(trimgrid[0,...]  != self.grid.nodata_value))
        self.assertTrue(np.any(trimgrid[-1,...] != self.grid.nodata_value))
        self.assertTrue(np.any(trimgrid[...,0]  != self.grid.nodata_value))
        self.assertTrue(np.any(trimgrid[...,-1] != self.grid.nodata_value))

    def test_snapGrid(self):
        checkgrid = copy.deepcopy(self.grid)#.copy()
        checkgrid.xllcorner += 50
        checkgrid.yllcorner -= 75
        checkgrid.snapGrid(self.grid)
        self.assertEqual(checkgrid.xllcorner, self.grid.xllcorner)
        self.assertEqual(checkgrid.yllcorner, self.grid.yllcorner)

        checkgrid.xllcorner += (checkgrid.cellsize * .9)
        checkgrid.yllcorner -= (checkgrid.cellsize * .9)
        checkgrid.snapGrid(self.grid)
        self.assertEqual(checkgrid.xllcorner, self.grid.xllcorner + self.grid.cellsize)
        self.assertEqual(checkgrid.yllcorner, self.grid.yllcorner - self.grid.cellsize)
        
    def test_copy(self):
        self.assertTrue(np.all(copy.copy(self.grid) == self.grid.nodata_value))
        self.assertTrue(np.all(copy.deepcopy(self.grid) == self.grid))

    def test_write(self):
        fnames = ("testout.tif","testout.asc")
        for fname in fnames:
            self.grid.write(fname)
            checkgrid = GeoGrid(fname,proj_params=PROJ_PARAMS)
            self.assertTrue(np.all(checkgrid == self.grid))
            self.assertDictEqual(checkgrid.getDefinition(), self.grid.getDefinition())
            
if __name__== "__main__":
    unittest.main()
