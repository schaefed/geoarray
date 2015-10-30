#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest, copy, shutil, os
import numpy as np
import geogrid as gg
import warnings

FNAME = os.path.join(os.path.dirname(__file__), "dem.asc")

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
        fill_value = -42
        yorigin = -15
        xorigin = 72
        cellsize = 33.33
        grid = gg.array(
            data=data, fill_value=fill_value,
            yorigin=yorigin, xorigin=xorigin,
            cellsize=cellsize
        )
        self.assertEqual(grid.shape, data.shape)
        self.assertEqual(grid.fill_value, fill_value)
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
        fill_value = 42
        grid = gg.full(shape,fill_value)
        self.assertEqual(grid.shape, shape)
        self.assertTrue(np.all(grid == fill_value))
        
    def test_empty(self):
        shape = (2,4,6)
        fill_value = 42
        grid = gg.empty(shape,fill_value=fill_value)
        self.assertEqual(grid.shape, shape)
        self.assertTrue(np.all(grid.data == fill_value))

class TestGeoGrid(unittest.TestCase):
    
    def setUp(self):        
        self.grid = gg.fromfile(fname=FNAME)
        self.write_path = "out"
        try:
            os.mkdir(self.write_path)
        except OSError:
            pass
        
    def tearDown(self):        
        try:
            shutil.rmtree(self.write_path)
        except:
            pass

        
    def test_setFillValue(self):
        rpcvalue = -2222
        checkgrid = gg.fromfile(FNAME)
        checkgrid.fill_value = rpcvalue
        # replacing works ...
        self.assertEqual(checkgrid.fill_value, rpcvalue)
        nodatapos1 = np.where(checkgrid == self.grid.fill_value)
        nodatapos2 = np.where(self.grid == self.grid.fill_value)
        for pos1,pos2 in zip(nodatapos1,nodatapos2):
            self.assertItemsEqual(pos1,pos2)
        
    def test_setDataType(self):
        rpctype = np.int32
        checkgrid = self.grid.astype(rpctype)
        self.assertEqual(checkgrid.dtype,rpctype)
        
    def test_getitem(self):        
        grid = self.grid.copy()
        slices = (
            self.grid < 3,
            # self.grid == 10,
            # np.where(self.grid>6),
            # (slice(None,None,None),slice(0,4,3)),(1,1),Ellipsis
        )
        idx = np.arange(12,20)
        self.assertTrue(np.all(grid[idx] == self.grid[gg.array(idx)]))
        for i,s in enumerate(slices):
            slc1 = grid[s]
            slc2 = self.grid[s]
            self.assertTrue(np.all(slc1.data == slc2.data))
            self.assertTrue(np.all(slc1.mask == slc2.mask))

    def test_getitemOrigin(self):
        grid1 = gg.ones((100,100),yorigin=1000,xorigin=1200,origin="ul")
        grid2 = gg.ones((100,100),yorigin=1000,xorigin=1200,origin="ll")
        grid3 = gg.ones((100,100),yorigin=1000,xorigin=1200,origin="ur")
        grid4 = gg.ones((100,100),yorigin=1000,xorigin=1200,origin="lr")

        slices = (
            ( slice(3,4) ),
            ( slice(3,4),slice(55,77,None) ),
            ( slice(None,None,7),slice(55,77,None) ),
            ( -1, ),
        )

        grids = {
            grid1: ( (997,  1200), (997,  1255), (1000, 1255), (901,  1200) ),
            grid2: ( (1096, 1200), (1096, 1255), (1001, 1255), (1000, 1200) ),
            grid3: ( (997,  1200), (997,  1177), (1000, 1177), (901,  1200) ),
            grid4: ( (1096, 1200), (1096, 1177), (1001, 1177), (1000, 1200) )
        }

        for grid in grids:
            for slc,expected in zip( slices, grids[grid] ):
                self.assertTupleEqual( expected, grid[slc].getOrigin() )

    def test_setitem(self):
        
        slices = (
            np.arange(12,20).reshape(1,-1),
            self.grid < 3,
            np.where(self.grid>6),
            (slice(None,None,None),slice(0,4,3)),
            (1,1),
            Ellipsis
        )
        value = 11
        grid = copy.deepcopy(self.grid)
        for slc in slices:
            grid = copy.deepcopy(self.grid)
            grid[slc] = value
            self.assertTrue(np.all(grid[slc] == value))

    def test_tofile(self):
        
        fnames = ("{:}/testout{:}".format(self.write_path,ext) for ext in gg._DRIVER_DICT)

        for fname in fnames:
            self.grid.tofile(fname)
            checkgrid = gg.fromfile(fname)
            self.assertTrue(np.all(checkgrid == self.grid))
            self.assertDictEqual(checkgrid.header, self.grid.header)

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
        
        # Ignore over/underflow warnings in function calls
        warnings.filterwarnings("ignore")
        # funcs tuple could be extended
        funcs = (np.exp,
                 np.sin,np.cos,np.tan,np.arcsinh,
                 np.around,np.rint,np.fix,
                 np.prod,np.sum,
                 np.trapz,
                 np.i0,np.sinc,
                 np.arctanh,
                 np.gradient,                
        )
        grid = self.grid.astype(np.float64)
        compare = grid.copy()
        for f in funcs:
            r1 = f(grid)
            r2 = f(compare)
            try:
                np.testing.assert_equal(r1,r2)
            except AssertionError:
                np.testing.assert_equal(r1.data,r2.data)
                np.testing.assert_equal(r1.mask,r2.mask)
            
        
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
        self.grid = gg.fromfile(FNAME)        

    def test_addCells(self):

        padgrid = self.grid.addCells(1, 1, 1, 1)
        self.assertTrue(np.sum(padgrid[1:-1,1:-1] == self.grid))

        padgrid = self.grid.addCells(0, 0, 0, 0)
        self.assertTrue(np.sum(padgrid[:] == self.grid))

        padgrid = self.grid.addCells(0, 99, 0, 4000)
        self.assertTrue(np.sum(padgrid[...,99:-4000] == self.grid))

        padgrid = self.grid.addCells(-1000, -4.55, 0, -6765.222)
        self.assertTrue(np.all(padgrid == self.grid))
        
        
    def test_enlarge(self):
        
        bbox = self.grid.bbox
        newbbox = {"xmin" : bbox["xmin"] - 2.5 * self.grid.cellsize,
                   "ymin" : bbox["ymin"] -  .7 * self.grid.cellsize,
                   "xmax" : bbox["xmax"] +  .1 * self.grid.cellsize,
                   "ymax" : bbox["ymax"] + 6.1 * self.grid.cellsize,}
        enlrgrid = self.grid.enlarge(**newbbox)
        self.assertEqual(enlrgrid.nrows, self.grid.nrows + 1 + 7)
        self.assertEqual(enlrgrid.ncols, self.grid.ncols + 3 + 1)

    def test_shrink(self):
        
        bbox = self.grid.bbox
        newbbox = {"xmin" : bbox["xmin"] + 2.5 * self.grid.cellsize,
                   "ymin" : bbox["ymin"] +  .7 * self.grid.cellsize,
                   "xmax" : bbox["xmax"] -  .1 * self.grid.cellsize,
                   "ymax" : bbox["ymax"] - 6.1 * self.grid.cellsize,}
        shrgrid = self.grid.shrink(**newbbox)        
        self.assertEqual(shrgrid.nrows, self.grid.nrows - 0 - 6)
        self.assertEqual(shrgrid.ncols, self.grid.ncols - 2 - 0)

    def test_removeCells(self):
        
        rmgrid = self.grid.removeCells(1,1,1,1)
        self.assertEqual(np.sum(rmgrid - self.grid[1:-1,1:-1]) , 0)
        rmgrid = self.grid.removeCells(0,0,0,0)
        self.assertEqual(np.sum(rmgrid - self.grid) , 0)

        
    def test_trim(self):
        
        trimgrid = self.grid.trim()
        self.assertTrue(np.any(trimgrid[0,...]  != self.grid.fill_value))
        self.assertTrue(np.any(trimgrid[-1,...] != self.grid.fill_value))
        self.assertTrue(np.any(trimgrid[...,0]  != self.grid.fill_value))
        self.assertTrue(np.any(trimgrid[...,-1] != self.grid.fill_value))

    def test_snap(self):

        def test(grid,target):
            yorg,xorg = grid.getOrigin()
            grid.snap(target)            

            xdelta = abs(grid.xorigin - xorg)
            ydelta = abs(grid.yorigin - yorg)

            # asure the shift to the next cell
            self.assertLessEqual(xdelta,target.cellsize/2)
            self.assertLessEqual(ydelta,target.cellsize/2)
            
            # grid origin is shifted to a cell multiple of self.grid.origin
            self.assertEqual((grid.xorigin - grid.xorigin)%grid.cellsize,0)
            self.assertEqual((grid.yorigin - grid.yorigin)%grid.cellsize,0)

        offsets = (
            (-75,-30),
            (self.grid.cellsize *.9,self.grid.cellsize *20),
            (self.grid.yorigin * -1.1, self.grid.xorigin * 1.89),
        )

        for yoff,xoff in offsets:            
            checkgrid = copy.deepcopy(self.grid)
            checkgrid.yorigin -= yoff
            checkgrid.xorigin -= xoff
            test(checkgrid,self.grid)

    def test_indexCoordinates(self):

        grid = self.grid
        offset = grid.cellsize
        ulyorigin, ulxorigin = grid.getOrigin("ul")
        uryorigin, urxorigin = grid.getOrigin("ur")
        llyorigin, llxorigin = grid.getOrigin("ll")
        lryorigin, lrxorigin = grid.getOrigin("lr")

        idxs = ((0,0),(grid.nrows-1, grid.ncols-1), (0,grid.ncols-1), (grid.nrows-1,0))
        coords = ((ulyorigin,ulxorigin),(lryorigin+offset, lrxorigin-offset),
                  (uryorigin, urxorigin-offset),(llyorigin+offset, llxorigin))

        for idx,coord in zip(idxs,coords):
            self.assertTupleEqual(grid.indexCoordinates(*idx),coord)


    def test_coordinateIndex(self):

        grid = self.grid
        offset = grid.cellsize
        ulyorigin, ulxorigin = grid.getOrigin("ul")
        uryorigin, urxorigin = grid.getOrigin("ur")
        llyorigin, llxorigin = grid.getOrigin("ll")
        lryorigin, lrxorigin = grid.getOrigin("lr")

        idxs = ((0,0),(grid.nrows-1, grid.ncols-1), (0,grid.ncols-1), (grid.nrows-1,0))
        coords = ((ulyorigin,ulxorigin),(lryorigin+offset, lrxorigin-offset),
                  (uryorigin, urxorigin-offset),(llyorigin+offset, llxorigin))

        for idx,coord in zip(idxs,coords):
            self.assertTupleEqual(grid.coordinateIndex(*coord),idx)

            
if __name__== "__main__":
    unittest.main()
