#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest, copy, shutil, os
import numpy as np
import geoarray as ga
import gdal
import warnings
import subprocess
import tempfile

# from parent directory run 
# python -m unittest test.test_geoarray

PWD = os.path.dirname(__file__)

# path to testfiles
PATH = os.path.join(PWD, "files")
FILES = [os.path.join(PATH, f) for f in os.listdir(PATH)]

TMPPATH = os.path.join(PWD, "out")

# random projection parameters
PROJ_PARAMS = {
    'lon_0' : '148.8',
    'lat_0' : '0',
    'y_0'   : '3210000',
    'x_0'   : '4321000',
    'units' : 'm', 
    'proj'  : 'laea',
    'ellps' : 'WGS84'
}

def readTestFiles():
    return [ga.fromfile(f) for f in FILES]

class TestInitialisation(unittest.TestCase):

    def test_array(self):
        data = np.arange(48).reshape(2,4,6)
        fill_value = -42
        yorigin = -15
        xorigin = 72
        cellsize = (33.33, 33.33)
        grid = ga.array(
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
        grid = ga.zeros(shape)
        self.assertEqual(grid.shape, shape)
        self.assertTrue(np.all(grid == 0))

    def test_ones(self):
        shape = (2,4,6)
        grid = ga.ones(shape)
        self.assertEqual(grid.shape, shape)
        self.assertTrue(np.all(grid == 1))

    def test_full(self):
        shape = (2,4,6)
        fill_value = 42
        grid = ga.full(shape,fill_value)
        self.assertEqual(grid.shape, shape)
        self.assertTrue(np.all(grid == fill_value))
        
    def test_empty(self):
        shape = (2,4,6)
        fill_value = 42
        grid = ga.empty(shape,fill_value=fill_value)
        self.assertEqual(grid.shape, shape)
        # self.assertTrue(np.all(grid.data == fill_value))

class TestGeoArray(unittest.TestCase):
    
    def setUp(self):
        self.grids = readTestFiles()

        try:
            os.mkdir(TMPPATH)
        except OSError:
            pass
        
    def tearDown(self):        
        try:
            shutil.rmtree(TMPPATH)
        except:
            pass

    def test_setFillValue(self):
        rpcvalue = -2222
        for base in self.grids:
            base.fill_value = rpcvalue
            self.assertEqual(base.fill_value, rpcvalue)

    def test_setDataType(self):
        rpctype = np.int32
        for base in self.grids:
            grid = base.astype(rpctype)
            self.assertEqual(grid.dtype,rpctype)
        
    def test_basicMatch(self):
        for base in self.grids:
            grid1, grid2, grid3, grid4 = [base.copy() for _ in xrange(4)]
            grid2.xorigin -= 1
            grid3.cellsize = (grid3.cellsize[0] + 1, grid3.cellsize[0] + 1)
            grid4.proj = {"invalid":"key"}
            self.assertTrue(base.basicMatch(grid1))
            self.assertFalse(base.basicMatch(grid2))
            self.assertFalse(base.basicMatch(grid3))
            self.assertFalse(base.basicMatch(grid4))
           
    def test_getitem(self):        
        for base in self.grids:
            grid = base.copy()
            slices = (
                base < 3,
                base == 10,
                np.where(base>6),
                (slice(None,None,None),slice(0,4,3)),(1,1),Ellipsis
            )
            idx = np.arange(12,20)
            self.assertTrue(np.all(grid[idx].data == base[ga.array(idx)].data))
            for i,s in enumerate(slices):
                slc1 = grid[s]
                slc2 = base[s]
                self.assertTrue(np.all(slc1.data == slc2.data))
                try:
                    self.assertTrue(np.all(slc1.mask == slc2.mask))
                except AttributeError: # __getitem__ returned a scalar
                    pass 
        
    def test_getitemOrigin(self):
        grids = (
            ga.ones((100,100),yorigin=1000,xorigin=1200,origin="ul"),
            ga.ones((100,100),yorigin=1000,xorigin=1200,origin="ll"),
            ga.ones((100,100),yorigin=1000,xorigin=1200,origin="ur"),
            ga.ones((100,100),yorigin=1000,xorigin=1200,origin="lr"),
        )
        slices = (
            ( slice(3,4) ),
            ( slice(3,4),slice(55,77,None) ),
            ( slice(None,None,7),slice(55,77,None) ),
            ( -1, ),
        )
        expected = (
            ( (997,  1200), (997,  1255), (1000, 1255), (901,  1200) ),
            ( (1096, 1200), (1096, 1255), (1001, 1255), (1000, 1200) ),
            ( (997,  1200), (997,  1177), (1000, 1177), (901,  1200) ),
            ( (1096, 1200), (1096, 1177), (1001, 1177), (1000, 1200) )
        )
        for i,grid in enumerate(grids):
            for slc,exp in zip(slices,expected[i]):
                self.assertTupleEqual( exp, grid[slc].getOrigin() )
                break
            break
                
    def test_setitem(self):
        for base in self.grids:
            slices = (
                np.arange(12,20).reshape(1,-1),
                base.data < 3,
                np.where(base>6),
                (slice(None,None,None),slice(0,4,3)),
                (1,1),
                Ellipsis
            )
            value = 11
            grid = copy.deepcopy(base)
            for slc in slices:
                grid = copy.deepcopy(base)
                grid[slc] = value
                self.assertTrue(np.all(grid[slc] == value))

    def test_bbox(self):
        grids = (
            ga.ones((100,100),yorigin=1000,xorigin=1200,origin="ul"),
            ga.ones((100,100),yorigin=1000,xorigin=1200,origin="ll"),
            ga.ones((100,100),yorigin=1000,xorigin=1200,origin="ur"),
            ga.ones((100,100),yorigin=1000,xorigin=1200,origin="lr"),
        )
        expected = (
            {'xmin': 1200, 'ymin': 900, 'ymax': 1000, 'xmax': 1300},
            {'xmin': 1200, 'ymin': 1000, 'ymax': 1100, 'xmax': 1300},
            {'xmin': 1100, 'ymin': 900, 'ymax': 1000, 'xmax': 1200},
            {'xmin': 1100, 'ymin': 1000, 'ymax': 1100, 'xmax': 1200},
        )

        for g, e in zip(grids, expected):
            self.assertDictEqual(g.bbox, e)
        
    def test_tofile(self):
        outfiles = ("{:}/testout{:}".format(TMPPATH, ext) for ext in ga._DRIVER_DICT)

        for base in self.grids:
            for outfile in outfiles:
                if outfile.endswith(".png"):
                    # data type conversion is done and precision lost
                    continue
                base.tofile(outfile)
                checkgrid = ga.fromfile(outfile)
                self.assertTrue(np.all(checkgrid == base))
                self.assertDictEqual(checkgrid.bbox, base.bbox)

    def test_typeConversion(self):
        for base in self.grids:
            with tempfile.NamedTemporaryFile(suffix=".png") as tf:
                base.tofile(tf.name)
                
                
    def test_copy(self):
        for base in self.grids[1:]:

            deep_copy = copy.deepcopy(base)        
            self.assertDictEqual(base.header, deep_copy.header)
            self.assertNotEqual(id(base),id(deep_copy))
            self.assertTrue(np.all(base == deep_copy))
            shallow_copy = copy.copy(base)
            self.assertDictEqual(base.header, shallow_copy.header)
            self.assertNotEqual(id(base),id(shallow_copy))
            self.assertTrue(np.all(base == shallow_copy))

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
        
        for base in self.grids:
            grid = base.copy()
            for f in funcs:
                r1 = f(base)
                r2 = f(grid)
                try:
                    np.testing.assert_equal(r1,r2)
                except AssertionError:
                    # masked array
                    np.testing.assert_equal(r1.data,r2.data)
                    np.testing.assert_equal(r1.mask,r2.mask)
            break
       
class TestGeoArrayFuncs(unittest.TestCase):

    def setUp(self):
        self.grids = readTestFiles()

    #     try:
    #         os.mkdir(TMPPATH)
    #     except OSError:
    #         pass
        
    # def tearDown(self):        
    #     try:
    #         shutil.rmtree(TMPPATH)
    #     except:
    #         pass
        
    def test_addCells(self):
        for base in self.grids:
            padgrid = base.addCells(1, 1, 1, 1)
            self.assertTrue(np.sum(padgrid[1:-1,1:-1] == base))

            padgrid = base.addCells(0, 0, 0, 0)
            self.assertTrue(np.sum(padgrid[:] == base))

            padgrid = base.addCells(0, 99, 0, 4000)
            self.assertTrue(np.sum(padgrid[...,99:-4000] == base))

            padgrid = base.addCells(-1000, -4.55, 0, -6765.222)
            self.assertTrue(np.all(padgrid == base))

    def test_enlarge(self):
        for base in self.grids:
            bbox = base.bbox
            newbbox = {
                "ymin" : bbox["ymin"] -  .7 * base.cellsize[0],
                "xmin" : bbox["xmin"] - 2.5 * base.cellsize[1],
                "ymax" : bbox["ymax"] + 6.1 * base.cellsize[0],
                "xmax" : bbox["xmax"] +  .1 * base.cellsize[1]
            }
            enlrgrid = base.enlarge(**newbbox)
            self.assertEqual(enlrgrid.nrows, base.nrows + 1 + 7)
            self.assertEqual(enlrgrid.ncols, base.ncols + 3 + 1)

        # the doctest
        x = np.arange(20).reshape((4,5))
        grid = ga.array(x, yorigin=100, xorigin=200, origin="ll", cellsize=20, fill_value=-9)
        enlarged = grid.enlarge(xmin=130, xmax=200, ymin=66)
        self.assertDictEqual(
            enlarged.bbox, 
            {'xmin': 120, 'ymin': 60, 'ymax': 180, 'xmax': 300}    
        )
        
    def test_shrink(self):
        for base in self.grids:
            bbox = base.bbox
            newbbox = {
                "ymin" : bbox["ymin"] +  .7 * base.cellsize[0],
                "xmin" : bbox["xmin"] + 2.5 * base.cellsize[1],
                "ymax" : bbox["ymax"] - 6.1 * base.cellsize[0],
                "xmax" : bbox["xmax"] -  .1 * base.cellsize[1],
            }
            shrgrid = base.shrink(**newbbox)        
            self.assertEqual(shrgrid.nrows, base.nrows - 0 - 6)
            self.assertEqual(shrgrid.ncols, base.ncols - 2 - 0)

    def test_removeCells(self):
        for base in self.grids:
            rmgrid = base.removeCells(1,1,1,1)
            self.assertEqual(np.sum(rmgrid - base[1:-1,1:-1]) , 0)
            rmgrid = base.removeCells(0,0,0,0)
            self.assertEqual(np.sum(rmgrid - base) , 0)

    def test_trim(self):
        for base in self.grids:
            trimgrid = base.trim()
            self.assertTrue(np.any(trimgrid[0,...]  != base.fill_value))
            self.assertTrue(np.any(trimgrid[-1,...] != base.fill_value))
            self.assertTrue(np.any(trimgrid[...,0]  != base.fill_value))
            self.assertTrue(np.any(trimgrid[...,-1] != base.fill_value))

    # def test_snap(self):
    #     for base in self.grids:
    #         offsets = (
    #             (-75,-30),
    #             (np.array(base.cellsize) *.9, np.array(base.cellsize) *20),
    #             (base.yorigin * -1.1, base.xorigin * 1.89),
    #         )

    #         for yoff,xoff in offsets:            
    #             grid = copy.deepcopy(base)
    #             grid.yorigin -= yoff
    #             grid.xorigin -= xoff
    #             yorg, xorg = grid.getOrigin()
    #             grid.snap(base)            

    #             xdelta = abs(grid.xorigin - xorg)
    #             ydelta = abs(grid.yorigin - yorg)

    #             # asure the shift to the next cell
    #             self.assertLessEqual(ydelta, base.cellsize[0]/2)
    #             self.assertLessEqual(xdelta, base.cellsize[1]/2)

    #             # grid origin is shifted to a cell multiple of self.grid.origin
    #             self.assertEqual((grid.yorigin - grid.yorigin)%grid.cellsize[0], 0)
    #             self.assertEqual((grid.xorigin - grid.xorigin)%grid.cellsize[1], 0)

    def test_coordinatesOf(self):
        for base in self.grids:
            offset = np.abs(base.cellsize)
            bbox = base.bbox

            idxs = (
                (0,0),
                (base.nrows-1, base.ncols-1),
                (0,base.ncols-1),
                (base.nrows-1,0)
            )
            expected = (
                (bbox["ymax"], bbox["xmin"]),
                (bbox["ymin"]+offset[0], bbox["xmax"]-offset[1]),
                (bbox["ymax"], bbox["xmax"]-offset[1]),
                (bbox["ymin"]+offset[0], bbox["xmin"])
            )

            for idx, e in zip(idxs, expected):
                self.assertTupleEqual(base.coordinatesOf(*idx), e)

    def test_indexOf(self):
        for base in self.grids:
            offset = np.abs(base.cellsize)*.8
            bbox = base.bbox
            
            coodinates = (
                (bbox["ymax"], bbox["xmin"]),
                (bbox["ymin"]+offset[0], bbox["xmax"]-offset[1]),
                (bbox["ymax"], bbox["xmax"]-offset[1]),
                (bbox["ymin"]+offset[0], bbox["xmin"])
            )

            expected = (
                (0,0),
                (base.nrows-1, base.ncols-1),
                (0,base.ncols-1),
                (base.nrows-1,0)
            )

            for c, e in zip(coodinates, expected):
                self.assertTupleEqual(base.indexOf(*c), e)

    def test_warp(self):

        """
        This test fails for gdal versions below 2.0. The warping is correct, but
        the void space around the original image is filled with fill_value in versions
        >= 2.0, else with 0. The tested function behaves like the more recent versions
        of GDAL
        """
        codes = (2062, 3857)
        tmpfile = os.path.join(TMPPATH, "tmp.tif")

        if gdal.VersionInfo().startswith("1"):
            return
        for fname, base in zip(FILES, self.grids):
            if base.proj.getReference():
                for epsg in codes:
                    # gdalwarp flips the warped imagel
                    proj = base[::-1].warp({"init":"epsg:{:}".format(epsg)}, 0)
                    with tempfile.NamedTemporaryFile(suffix=".tif") as tf:
                        subprocess.check_output(
                            "gdalwarp -r 'near' -et 0 -t_srs 'EPSG:{:}' {:} {:}".format(
                                epsg, fname, tf.name
                            ),
                            shell=True
                        )
                        compare = ga.fromfile(tf.name)
                        self.assertTrue(np.all(proj.data == compare.data))
                        self.assertTrue(np.all(proj.mask == compare.mask))
                        self.assertDictEqual(proj.bbox, compare.bbox)
            else:
                self.assertRaises(AttributeError)
                 
               
if __name__== "__main__":
    unittest.main()
