#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest, copy, shutil, os
import numpy as np
import geoarray as ga
import gdal
import warnings
import subprocess
import tempfile
from test_utils import createTestFiles, removeTestFiles, TMPPATH, FILES

# all tests, run from main directory:
# python -m unittest discover test

# this test only, run from parent directory run 
# python -m unittest test.test_methods

class Test(unittest.TestCase):

    def setUp(self):
        self.grids = createTestFiles()
        
    def tearDown(self):        
        removeTestFiles()
       
    # def test_basicMatch(self):
    #     for base in self.grids:
    #         grid1, grid2, grid3, grid4 = [base.copy() for _ in xrange(4)]
    #         grid2.xorigin -= 1
    #         grid3.cellsize = (grid3.cellsize[0] + 1, grid3.cellsize[0] + 1)
    #         grid4.proj = {"invalid":"key"} # sets proj to False
    #         self.assertTrue(base.basicMatch(grid1))
    #         self.assertFalse(base.basicMatch(grid2))
    #         self.assertFalse(base.basicMatch(grid3))
    #         self.assertTrue(base.basicMatch(grid4))
 
    def test_addCells(self):
        for base in self.grids:
             
            try:
                padgrid = base.addCells(1, 1, 1, 1)
                self.assertTrue(np.sum(padgrid[...,1:-1,1:-1] == base))
            except AttributeError:
                # input grid has an invalid fill_value, e.g the used test.png
                continue

            padgrid = base.addCells(0, 0, 0, 0)
            self.assertTrue(np.sum(padgrid[:] == base))

            padgrid = base.addCells(0, 99, 0, 4000)
            self.assertTrue(np.sum(padgrid[...,99:-4000] == base))

            padgrid = base.addCells(-1000, -4.55, 0, -6765.222)
            self.assertTrue(np.all(padgrid == base))

    def test_enlarge(self):
        for base in self.grids[1:]:
            bbox = base.bbox
            if base.fill_value is None:
                base.fill_value = -9999
            cellsize = map(abs, base.cellsize)
            newbbox = {
                "ymin" : bbox["ymin"] -  .7 * cellsize[0],
                "xmin" : bbox["xmin"] - 2.5 * cellsize[1],
                "ymax" : bbox["ymax"] + 6.1 * cellsize[0],
                "xmax" : bbox["xmax"] +  .1 * cellsize[1]
            }
            enlrgrid = base.enlarge(**newbbox)
            self.assertEqual(enlrgrid.nrows, base.nrows + 1 + 7)
            self.assertEqual(enlrgrid.ncols, base.ncols + 3 + 1)

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
            cellsize = map(abs, base.cellsize)
            newbbox = {
                "ymin" : bbox["ymin"] +  .7 * cellsize[0],
                "xmin" : bbox["xmin"] + 2.5 * cellsize[1],
                "ymax" : bbox["ymax"] - 6.1 * cellsize[0],
                "xmax" : bbox["xmax"] -  .1 * cellsize[1],
            }
            shrgrid = base.shrink(**newbbox)        
            self.assertEqual(shrgrid.nrows, base.nrows - 0 - 6)
            self.assertEqual(shrgrid.ncols, base.ncols - 2 - 0)

    def test_removeCells(self):
        for base in self.grids:
            rmgrid = base.removeCells(1,1,1,1)
            self.assertEqual(np.sum(rmgrid - base[...,1:-1,1:-1]) , 0)
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

    def test_project(self):

        """
        This test fails for gdal versions below 2.0. The warping is correct, but
        the void space around the original image is filled with fill_value in versions
        >= 2.0, else with 0. The tested function behaves like the more recent versions
        of GDAL
        """
        codes = (2062, 3857)

        if gdal.VersionInfo().startswith("1"):
            warnings.warn("Skipping incompatible warp test on GDAL versions < 2", RuntimeWarning)
            return
        
        for fname, base in zip(FILES, self.grids):
            # break
            if base.proj:
                for epsg in codes:
                    # gdalwarp flips the warped image
                    proj = ga.project(
                        grid      = base[::-1],
                        proj      = {"init":"epsg:{:}".format(epsg)},
                        max_error = 0
                    )
                    # proj = base[::-1].warp({"init":"epsg:{:}".format(epsg)}, 0)
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
