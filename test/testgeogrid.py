#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest, copy, shutil, os
import numpy as np
from geogrid import GeoGrid, _DRIVER_DICT
import geogridfuncs as ggfuncs
import warnings
warnings.filterwarnings("ignore") 

FNAME = os.path.join(os.path.split(__file__)[0],"dem.asc")
#FNAME = "dem.asc"

PROJ_PARAMS = {
    'lon_0': '148.8',   'lat_0': '0',
    'y_0'  : '3210000', 'x_0': '4321000',
    'units': 'm', 
    'proj' : 'laea',    'ellps': 'WGS84'
}

# class TestInitialisation(unittest.TestCase):

#     def test_initReadData(self):
#         grid = GeoGrid(FNAME)
        
#     def test_initWithData(self):
        
#         data = np.arange(48).reshape(2,4,6)
#         grid = GeoGrid(data=data)
#         self.assertEqual(grid.shape, data.shape)

#         # given with to be ignored shape parameters
#         grid = GeoGrid(data=data, shape=(3,3,44))
#         self.assertEqual(grid.shape, data.shape)

#         # given with all other parameters
#         grid = GeoGrid(data=data, dtype=np.float32, shape=(3,3,44),
#                        fill_value=42, yorigin=-15,xorigin=88,cellsize=33.33
#         )
#         # grid[1:-1]
#         self.assertEqual(grid.shape, data.shape)
#         self.assertEqual(grid.dtype, data.dtype)
#         self.assertEqual(grid.fill_value,42)
#         self.assertEqual(grid.yorigin,-15)
#         self.assertEqual(grid.xorigin,88)
#         self.assertEqual(grid.cellsize,33.33)

#     def test_initWithoutData(self):
        
#         grid = GeoGrid(shape=(4,44,66))
#         self.assertEqual(grid.shape,(4,44,66))

#         grid = GeoGrid(shape=(4,44,66),fill_value=42)
#         self.assertEqual(grid.shape,(4,44,66))
#         self.assertTrue(np.all(grid == 42))

#         # given with all other parameters
#         grid = GeoGrid(shape=(4,44,66),dtype=np.float32,
#                        fill_value=42, yorigin=-15,xorigin=88,cellsize=33.33
#         )        
#         self.assertEqual(grid.shape, (4,44,66))
#         self.assertEqual(grid.dtype, np.float32)
#         self.assertEqual(grid.fill_value,42)
#         self.assertEqual(grid.yorigin,-15)
#         self.assertEqual(grid.xorigin,88)
#         self.assertEqual(grid.cellsize,33.33)
    
class TestGeoGrid(unittest.TestCase):
    
    def setUp(self):        
        self.grid = GeoGrid(fname=FNAME,dtype=np.float32)
        self.write_path = "out"

    def tearDown(self):        
        try:
            shutil.rmtree(self.write_path)
        except:
            pass
        
    def test_typeConsistency(self):
        def check(grid):
            self.assertTrue(grid.dtype       == type(grid.fill_value))
            self.assertTrue(grid.dtype       == type(grid.yorigin))            
            self.assertTrue(grid.dtype       == type(grid.xorigin))
            self.assertTrue(grid.data.dtype  == grid.dtype)
            self.assertTrue(grid._data.dtype == grid.dtype)
        #
        checkgrid = copy.deepcopy(self.grid)
        checkgrid.dtype = np.int32
        check(self.grid)
        check(checkgrid)
        
    def test_setFillValue(self):
        rpcvalue = -2222
        checkgrid = GeoGrid(FNAME)
        org_fill_fvalue = checkgrid.fill_value
        checkgrid.fill_value = rpcvalue
        self.assertFalse(np.any(checkgrid == org_fill_fvalue))
        self.assertEqual(checkgrid.fill_value, rpcvalue)
        nodatapos1 = np.where(checkgrid == rpcvalue)
        nodatapos2 = np.where(self.grid == self.grid.fill_value)
        for pos1,pos2 in zip(nodatapos1,nodatapos2):
            self.assertTrue(np.all(pos1 == pos2))
        self.assertEqual(checkgrid.fill_value, rpcvalue)
        
    def test_setDataType(self):
        rpctype = np.int32        
        checkgrid = copy.deepcopy(self.grid)
        checkgrid.dtype = rpctype
        self.assertEqual(checkgrid.dtype,rpctype)
        self.assertEqual(checkgrid.data.dtype,rpctype)
        self.assertEqual(checkgrid._data.dtype,rpctype)
        
    def test_getitem(self):        
        data = self.grid.data.copy()
        slices = (
            self.grid < 3,
            self.grid == 10,
            np.where(self.grid>6),
            (slice(None,None,None),slice(0,4,3)),(1,1),Ellipsis
        )
        idx = np.arange(12,20).reshape(1,-1)
        self.assertTrue(np.all(self.grid[idx] == self.grid[GeoGrid(data=idx)]))
        for i,slc in enumerate(slices):
            try:
                slc = slc.data
            except AttributeError:
                pass
            self.assertTrue(np.all(data[slc] == self.grid[slc]))
        
    def test_setitem(self):
        
        slices = (
            self.grid < 3,
            self.grid == 10,
            np.where(self.grid>6),
            (slice(None,None,None),slice(0,4,3)),(1,1),Ellipsis
        )
        value = 11
        slc = np.arange(12,20).reshape(1,-1)
        
        testgrid = copy.deepcopy(self.grid)
        testgrid[slc] = value
        self.assertTrue(np.all(testgrid[slc] == value))

        testgrid = copy.deepcopy(self.grid)
        testgrid[GeoGrid(data=slc)] = value
        self.assertTrue(np.all(testgrid[slc] == value))

        for slc in slices:
            try:
                slc = slc.data
            except AttributeError:
                pass        
            testgrid = copy.deepcopy(self.grid)
            testgrid[slc] = value
            self.assertTrue(np.all(testgrid[slc] == value))

    def test_write(self):
        
        fnames = ("{:}/testout{:}".format(self.write_path,ext) for ext in _DRIVER_DICT)
        for fname in fnames:
            try:
                self.grid.write(fname)
            except IOError:
                continue                
            checkgrid = GeoGrid(fname,proj_params=PROJ_PARAMS)
            self.assertTrue(np.all(checkgrid == self.grid))
            self.assertDictEqual(checkgrid.getDefinition(), self.grid.getDefinition())

    def test_copy(self):
        deep_copy = copy.deepcopy(self.grid)
        self.assertTrue(deep_copy.header == self.grid.header)
        self.assertTrue(np.all(self.grid == deep_copy))
        self.assertTrue(np.all(copy.copy(self.grid) == self.grid.fill_value))

    def test_numpyFunctions(self):
        
        # funcs tuple could be extended
        funcs = (np.exp,
                 np.sin,np.cos,np.tan,np.arcsinh,
                 np.around,np.rint,np.fix,
                 np.prod,np.sum,
                 # np.cumprod,            # returns an array of changed shape -> does not make sense here
                 np.trapz,
                 np.i0,np.sinc,
                 np.arctanh, np.gradient,                
        )
        compare = self.grid[:]
        for f in funcs:
            np.testing.assert_equal(f(self.grid),f(compare))

    # def test_numpyAttributes(self):
    #     g = GeoGrid(shape=(40,45))

    #     # print g.shape
    #     # print g.ndim
    #     # gg = g.T
    #     # print gg.shape
    #     # gg = g.reshape(1,-1)
    #     # print gg.shape
    #     print np.transpose((g,g)).shape
    #     # print type(np.hstack((g,g)))
    #     # print np.hstack((g,g)).shape
    #     # print np.transpose(g)
    #     # print type(np.transpose((g,g)))
    #     # print type(g.T)
    #     # print type(np.transpose(g))
    #     # print type(self.grid)
    #     # print type(np.hstack((self.grid,self.grid)))
    #     # print self.grid.bbox
    #     # g = np.transpose(self.grid)
    #     # print g.bbox
        
    def test_reading(self):
        value          = 42
        maxrow         = 10
        mincol         = 100
        checkgrid      = copy.deepcopy(self.grid)
        idx            = (slice(0,maxrow),slice(mincol,None))
        idx_inv        = (slice(maxrow,None),slice(0,mincol))
        self.grid[idx] = value
        self.assertTrue(np.all(self.grid[idx] == value))
        self.assertTrue(np.all(self.grid[idx_inv] == checkgrid[idx_inv]))
        
class TestGeoGridFuncs(unittest.TestCase):
    
    def setUp(self):        
        self.grid = GeoGrid(FNAME)        

    def test_addCells(self):
        
        padgrid = ggfuncs.addCells(self.grid, 1, 1, 1, 1)
        self.assertTrue(np.sum(padgrid[1:-1,1:-1] == self.grid))

        padgrid = ggfuncs.addCells(self.grid, 0, 0, 0, 0)
        self.assertTrue(np.sum(padgrid[:] == self.grid))

        padgrid = ggfuncs.addCells(self.grid, 0, 99, 0, 4000)
        self.assertTrue(np.sum(padgrid[...,99:-4000] == self.grid))

        padgrid = ggfuncs.addCells(self.grid, -1000, -4.55, 0, -6765.222)
        self.assertTrue(np.all(padgrid == self.grid))
        
        
    def test_enlargeGrid(self):
        
        bbox = self.grid.bbox
        newbbox = {"xmin" : bbox["xmin"] - 2.5 * self.grid.cellsize,
                   "ymin" : bbox["ymin"] -  .7 * self.grid.cellsize,
                   "xmax" : bbox["xmax"] +  .1 * self.grid.cellsize,
                   "ymax" : bbox["ymax"] + 6.1 * self.grid.cellsize,}
        enlrgrid = ggfuncs.enlargeGrid(self.grid,newbbox)
        self.assertEqual(enlrgrid.nrows, self.grid.nrows + 1 + 7)
        self.assertEqual(enlrgrid.ncols, self.grid.ncols + 3 + 1)

    def test_shrinkGrid(self):
        
        bbox = self.grid.bbox
        newbbox = {"xmin" : bbox["xmin"] + 2.5 * self.grid.cellsize,
                   "ymin" : bbox["ymin"] +  .7 * self.grid.cellsize,
                   "xmax" : bbox["xmax"] -  .1 * self.grid.cellsize,
                   "ymax" : bbox["ymax"] - 6.1 * self.grid.cellsize,}
        shrgrid = ggfuncs.shrinkGrid(self.grid,newbbox)        
        self.assertEqual(shrgrid.nrows, self.grid.nrows - 0 - 6)
        self.assertEqual(shrgrid.ncols, self.grid.ncols - 2 - 0)

    def test_removeCells(self):
        
        rmgrid = ggfuncs.removeCells(self.grid,1,1,1,1)
        self.assertEqual(np.sum(rmgrid - self.grid[1:-1,1:-1]) , 0)
        rmgrid = ggfuncs.removeCells(self.grid,0,0,0,0)
        self.assertEqual(np.sum(rmgrid - self.grid) , 0)

        
    def test_trimGrid(self):
        
        trimgrid = ggfuncs.trimGrid(self.grid)
        self.assertTrue(np.any(trimgrid[0,...]  != self.grid.fill_value))
        self.assertTrue(np.any(trimgrid[-1,...] != self.grid.fill_value))
        self.assertTrue(np.any(trimgrid[...,0]  != self.grid.fill_value))
        self.assertTrue(np.any(trimgrid[...,-1] != self.grid.fill_value))

    def test_snapGrid(self):
        
        checkgrid = copy.deepcopy(self.grid)
        checkgrid.xorigin += 50
        checkgrid.yorigin -= 75
        ggfuncs.snapGrid(checkgrid,self.grid)
        self.assertEqual(checkgrid.xorigin, self.grid.xorigin)
        self.assertEqual(checkgrid.yorigin, self.grid.yorigin)

        checkgrid.xorigin += (checkgrid.cellsize * .9)
        checkgrid.yorigin += (checkgrid.cellsize * 20)
        ggfuncs.snapGrid(checkgrid,self.grid)
        self.assertEqual(checkgrid.xorigin, self.grid.xorigin + self.grid.cellsize)
        self.assertEqual(checkgrid.yorigin, self.grid.yorigin - self.grid.cellsize)


if __name__== "__main__":
    unittest.main()
