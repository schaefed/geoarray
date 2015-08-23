#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest, copy, shutil
import numpy as np
import geogrid as gg
import warnings
warnings.filterwarnings("ignore") 

#FNAME = os.path.join(os.path.split(__file__)[0],"dem.asc")
FNAME = "dem.asc"

PROJ_PARAMS = {
    'lon_0' : '148.8',
    'lat_0' : '0',
    'y_0'   : '3210000',
    'x_0'   : '4321000',
    'units' : 'm', 
    'proj'  : 'laea',
    'ellps' : 'WGS84'
}

class TestInitialisation(unittest.TestCase):

    def test_array(self):
        data = np.arange(48).reshape(2,4,6)
        nodata_value = -42
        yorigin = -15
        xorigin = 72
        cellsize = 33.33
        grid = gg.array(
            data=data, nodata_value=nodata_value,
            yorigin=yorigin, xorigin=xorigin,
            cellsize=cellsize
        )
        self.assertEqual(grid.shape, data.shape)
        self.assertEqual(grid.nodata_value, nodata_value)
        self.assertEqual(grid.yorigin, yorigin)
        self.assertEqual(grid.xorigin, xorigin)
        self.assertEqual(grid.cellsize, cellsize)
        self.assertTrue(np.all(grid == data))

    def test_zeros(self):
        shape = (2,4,6)
        grid = gg.zeros(shape)
        self.assertEqual(grid.shape, shape)
        self.assertTrue(np.all(grid == 0))

    def test_ones(self):
        shape = (2,4,6)
        grid = gg.ones(shape)
        self.assertEqual(grid.shape, shape)
        self.assertTrue(np.all(grid == 1))

    def test_full(self):
        shape = (2,4,6)
        fill_value = 4242
        grid = gg.full(shape,fill_value=fill_value)
        self.assertEqual(grid.shape, shape)
        self.assertTrue(np.all(grid == fill_value))
        
    def test_empty(self):
        shape = (2,4,6)
        nodata_value = 42
        grid = gg.empty(shape,nodata_value=nodata_value)
        self.assertEqual(grid.shape, shape)
        self.assertTrue(np.all(grid == nodata_value))

class TestGeoGrid(unittest.TestCase):
    
    def setUp(self):        
        self.grid = gg.fromfile(fname=FNAME)
        self.write_path = "out"

    def tearDown(self):        
        try:
            shutil.rmtree(self.write_path)
        except:
            pass
                
    def test_setFillValue(self):
        rpcvalue = -2222
        checkgrid = gg.fromfile(FNAME)
        org_fill_fvalue = checkgrid.nodata_value
        checkgrid.nodata_value = rpcvalue
        self.assertFalse(np.any(checkgrid == org_fill_fvalue))
        self.assertEqual(checkgrid.nodata_value, rpcvalue)
        nodatapos1 = np.where(checkgrid == checkgrid.nodata_value)
        nodatapos2 = np.where(self.grid == self.grid.nodata_value)
        for pos1,pos2 in zip(nodatapos1,nodatapos2):
            self.assertItemsEqual(pos1,pos2)
        self.assertEqual(checkgrid.nodata_value, rpcvalue)
        
    def test_setDataType(self):
        rpctype = np.int32
        checkgrid = self.grid.astype(rpctype)
        self.assertEqual(checkgrid.dtype,rpctype)
        
    def test_getitem(self):        
        data = self.grid.copy()
        slices = (
            self.grid < 3,
            self.grid == 10,
            np.where(self.grid>6),
            (slice(None,None,None),slice(0,4,3)),(1,1),Ellipsis
        )
        
        idx = np.arange(12,20).reshape(1,-1)
        self.assertTrue(np.all(self.grid[idx] == self.grid[gg.array(data=idx)]))
        for i,slc in enumerate(slices):
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
        testgrid[gg.array(data=slc)] = value
        self.assertTrue(np.all(testgrid[slc] == value))

        for slc in slices:
            testgrid = copy.deepcopy(self.grid)
            testgrid[slc] = value
            self.assertTrue(np.all(testgrid[slc] == value))

    def test_write(self):
        
        fnames = ("{:}/testout{:}".format(self.write_path,ext) for ext in gg._DRIVER_DICT)
        for fname in fnames:
            try:
                self.grid.write(fname)
            except IOError:
                continue                
            checkgrid = gg.fromfile(fname)
            self.assertTrue(np.all(checkgrid == self.grid))
            self.assertDictEqual(checkgrid.getDefinition(), self.grid.getDefinition())

    def test_copy(self):
        
        deep_copy = copy.deepcopy(self.grid)        
        self.assertTrue(self.grid.header == deep_copy.header)
        self.assertNotEqual(id(self.grid),id(deep_copy))
        self.assertTrue(np.all(self.grid == deep_copy))

        shallow_copy = copy.copy(self.grid)
        self.assertTrue(self.grid.header == shallow_copy.header)
        self.assertNotEqual(id(self.grid),id(shallow_copy))
        self.assertTrue(np.all(self.grid == shallow_copy))


    def test_numpyFunctions(self):
        
        # funcs tuple could be extended
        funcs = (np.exp,
                 np.sin,np.cos,np.tan,np.arcsinh,
                 np.around,np.rint,np.fix,
                 np.prod,np.sum,
                 np.trapz,
                 np.i0,np.sinc,
                 np.arctanh, np.gradient,                
        )
        compare = self.grid[:]
        for f in funcs:
            np.testing.assert_equal(f(self.grid),f(compare))

        
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
        
# class TestGeoGridFuncs(unittest.TestCase):
    
#     def setUp(self):        
#         self.grid = GeoGrid(FNAME)        

#     def test_addCells(self):
        
#         padgrid = ggfuncs.addCells(self.grid, 1, 1, 1, 1)
#         self.assertTrue(np.sum(padgrid[1:-1,1:-1] == self.grid))

#         padgrid = ggfuncs.addCells(self.grid, 0, 0, 0, 0)
#         self.assertTrue(np.sum(padgrid[:] == self.grid))

#         padgrid = ggfuncs.addCells(self.grid, 0, 99, 0, 4000)
#         self.assertTrue(np.sum(padgrid[...,99:-4000] == self.grid))

#         padgrid = ggfuncs.addCells(self.grid, -1000, -4.55, 0, -6765.222)
#         self.assertTrue(np.all(padgrid == self.grid))
        
        
#     def test_enlargeGrid(self):
        
#         bbox = self.grid.bbox
#         newbbox = {"xmin" : bbox["xmin"] - 2.5 * self.grid.cellsize,
#                    "ymin" : bbox["ymin"] -  .7 * self.grid.cellsize,
#                    "xmax" : bbox["xmax"] +  .1 * self.grid.cellsize,
#                    "ymax" : bbox["ymax"] + 6.1 * self.grid.cellsize,}
#         enlrgrid = ggfuncs.enlargeGrid(self.grid,newbbox)
#         self.assertEqual(enlrgrid.nrows, self.grid.nrows + 1 + 7)
#         self.assertEqual(enlrgrid.ncols, self.grid.ncols + 3 + 1)

#     def test_shrinkGrid(self):
        
#         bbox = self.grid.bbox
#         newbbox = {"xmin" : bbox["xmin"] + 2.5 * self.grid.cellsize,
#                    "ymin" : bbox["ymin"] +  .7 * self.grid.cellsize,
#                    "xmax" : bbox["xmax"] -  .1 * self.grid.cellsize,
#                    "ymax" : bbox["ymax"] - 6.1 * self.grid.cellsize,}
#         shrgrid = ggfuncs.shrinkGrid(self.grid,newbbox)        
#         self.assertEqual(shrgrid.nrows, self.grid.nrows - 0 - 6)
#         self.assertEqual(shrgrid.ncols, self.grid.ncols - 2 - 0)

#     def test_removeCells(self):
        
#         rmgrid = ggfuncs.removeCells(self.grid,1,1,1,1)
#         self.assertEqual(np.sum(rmgrid - self.grid[1:-1,1:-1]) , 0)
#         rmgrid = ggfuncs.removeCells(self.grid,0,0,0,0)
#         self.assertEqual(np.sum(rmgrid - self.grid) , 0)

        
#     def test_trimGrid(self):
        
#         trimgrid = ggfuncs.trimGrid(self.grid)
#         self.assertTrue(np.any(trimgrid[0,...]  != self.grid.nodata_value))
#         self.assertTrue(np.any(trimgrid[-1,...] != self.grid.nodata_value))
#         self.assertTrue(np.any(trimgrid[...,0]  != self.grid.nodata_value))
#         self.assertTrue(np.any(trimgrid[...,-1] != self.grid.nodata_value))

#     def test_snapGrid(self):

#         def test(grid,target):
#             yorg,xorg = grid.getOrigin()
#             ggfuncs.snapGrid(grid,target)            

#             xdelta = abs(grid.xorigin - xorg)
#             ydelta = abs(grid.yorigin - yorg)

#             # asure the shift to the next cell
#             self.assertLessEqual(xdelta,target.cellsize/2)
#             self.assertLessEqual(ydelta,target.cellsize/2)
            
#             # grid origin is shifted to a cell multiple of self.grid.origin
#             self.assertEqual((grid.xorigin - grid.xorigin)%grid.cellsize,0)
#             self.assertEqual((grid.yorigin - grid.yorigin)%grid.cellsize,0)

#         offsets = (
#             (-75,-30),
#             (self.grid.cellsize *.9,self.grid.cellsize *20),
#             (self.grid.yorigin * -1.1, self.grid.xorigin * 1.89),
#         )

#         for yoff,xoff in offsets:            
#             checkgrid = copy.deepcopy(self.grid)
#             checkgrid.yorigin -= yoff
#             checkgrid.xorigin -= xoff
#             test(checkgrid,self.grid)



if __name__== "__main__":
    unittest.main()
