#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest, copy, shutil, os
import numpy as np
import geoarray as ga
import gdal
import warnings
import subprocess
import tempfile
from test_utils import createTestFiles, removeTestFiles

# all tests, run from main directory:
# python -m unittest discover test

# this test only, run from main directory
# python -m unittest test.test_core

class Test(unittest.TestCase):

    def setUp(self):
        _, self.grids = createTestFiles()

    def tearDown(self):
        removeTestFiles()

    def test_setFillValue(self):
        rpcvalue = -2222
        for base in self.grids:
            base.fill_value = rpcvalue
            self.assertEqual(base.fill_value, base.dtype.type(rpcvalue))

    def test_setDataType(self):
        rpctype = np.int32
        for base in self.grids:
            grid = base.astype(rpctype)
            self.assertEqual(grid.dtype,rpctype)

    def test_getitem(self):
        for base in self.grids:
            # simplifies the tests...
            base = base[0] if base.ndim > 2 else base
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

    def test_getitemCellsize(self):

        grid = ga.ones((100,100), yorigin=1000, xorigin=1200, cellsize=10, origin="ul")

        self.assertTupleEqual(grid[3:4].cellsize, (-10, 10))
        self.assertTupleEqual(grid[0::2, 0::4].cellsize, (-20, 40))

        self.assertTupleEqual(grid[0::3, 0::5].cellsize, (-30, 50))
        self.assertTupleEqual(grid[0::1, 0::7].cellsize, (-10, 70))
        # # # needs to be extended...
        self.assertTupleEqual(grid[[1,2,5]].cellsize, (-20, 10))
        self.assertTupleEqual(grid[[1,2,4,10]].cellsize, (-30, 10))

        self.assertTupleEqual(grid[[0,10,5]].cellsize, (-25, 10))

        grid = ga.ones((100,100), yorigin=1000, xorigin=1200, cellsize=10,origin="ll")
        self.assertTupleEqual(grid[3:4].cellsize, (10, 10))
        self.assertTupleEqual(grid[0::2, 0::4].cellsize, (20, 40))
        self.assertTupleEqual(grid[0::3, 0::5].cellsize, (30, 50))
        self.assertTupleEqual(grid[0::1, 0::7].cellsize, (10, 70))

        grid = ga.ones((100,100), yorigin=1000, xorigin=1200, cellsize=10, origin="lr")
        self.assertTupleEqual(grid[3:4].cellsize, (10, -10))

        grid = ga.ones((100,100), yorigin=1000, xorigin=1200, cellsize=10, origin="ur")
        self.assertTupleEqual(grid[3:4].cellsize, (-10, -10))

        # yvals = np.array(range(1000, 1100, 2) + range(1100, 1250, 3))[::-1]
        # xvals = np.array(range(0, 100, 2) + range(100, 250, 3))
        # xvalues, yvalues = np.meshgrid(yvals, xvals)
        # grid = ga.ones((100, 100), origin="ul", yvalues=yvalues, xvalues=xvalues)


    def test_getitemOrigin(self):
        grids = (
            ga.ones((100, 100), yorigin=1000, xorigin=1200, origin="ul"),
            ga.ones((100, 100), yorigin=1000, xorigin=1200, origin="ll"),
            ga.ones((100, 100), yorigin=1000, xorigin=1200, origin="ur"),
            ga.ones((100, 100), yorigin=1000, xorigin=1200, origin="lr"))
        slices = (
            (slice(3, 4)),
            (slice(3, 4), slice(55, 77, None)),
            (slice(None, None, 7), slice(55, 77, None)),
            (-1, ),)

        expected = (
            ((997,   1200),  (997,   1255),  (1000,  1255),  (901,   1200)),
            ((1096,  1200),  (1096,  1255),  (1001,  1255),  (1000,  1200)),
            ((997,   1200),  (997,   1177),  (1000,  1177),  (901,   1200)),
            ((1096,  1200),  (1096,  1177),  (1001,  1177),  (1000,  1200)))

        for i, grid in enumerate(grids):
            for slc, exp in zip(slices, expected[i]):
                self.assertTupleEqual( exp,  grid[slc].getCorner() )
                break
            break

    def test_setitem(self):
        for base in self.grids:
            # simplifies the tests...
            base = base[0] if base.ndim > 2 else base
            slices = (
                np.arange(12,20).reshape(1,-1),
                base.data < 3,
                np.where(base>6),
                (slice(None,None,None),slice(0,4,3)),
                (1,1),
                Ellipsis
            )
            value = 11
            # grid = copy.deepcopy(base)
            for slc in slices:
                grid = copy.deepcopy(base)
                grid[slc] = value
                self.assertTrue(np.all(grid[slc] == value))


    def test_bbox(self):
        grids = (
            ga.ones((100,100), yorigin=1000, xorigin=1200, origin="ul"),
            ga.ones((100,100), yorigin=1000, xorigin=1200, origin="ll"),
            ga.ones((100,100), yorigin=1000, xorigin=1200, origin="ur"),
            ga.ones((100,100), yorigin=1000, xorigin=1200, origin="lr"),
        )
        expected = (
            {'xmin': 1200, 'ymin': 900,  'ymax': 1000, 'xmax': 1300},
            {'xmin': 1200, 'ymin': 1000, 'ymax': 1100, 'xmax': 1300},
            {'xmin': 1100, 'ymin': 900,  'ymax': 1000, 'xmax': 1200},
            {'xmin': 1100, 'ymin': 1000, 'ymax': 1100, 'xmax': 1200},
        )

        for g, e in zip(grids, expected):
            self.assertDictEqual(g.bbox, e)

    # def test_simplewrite(self):
    #     for infile in FILES:
    #         outfile = os.path.join(TMPPATH, os.path.split(infile)[1])
    #         base = ga.fromfile(infile)

    #         base.tofile(outfile)
    #         checkgrid = ga.fromfile(outfile)

    #         self.assertDictEqual(
    #             base._fobj.GetDriver().GetMetadata_Dict(),
    #             checkgrid._fobj.GetDriver().GetMetadata_Dict()
    #         )

    # def test_tofile(self):
    #     outfiles = (os.path.join(TMPPATH, "file{:}".format(ext)) for ext in ga._DRIVER_DICT)

    #     for base in self.grids:
    #         for outfile in outfiles:
    #             if outfile.endswith(".png"):
    #                 # data type conversion is done and precision lost
    #                 continue
    #             if outfile.endswith(".asc") and base.nbands > 1:
    #                 self.assertRaises(RuntimeError)
    #                 continue
    #             base.tofile(outfile)
    #             checkgrid = ga.fromfile(outfile)
    #             self.assertTrue(np.all(checkgrid == base))
    #             self.assertDictEqual(checkgrid.bbox, base.bbox)

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
                 np.sin, np.cos, np.tan, np.arcsinh,
                 np.around, np.rint, np.fix,
                 np.prod, np.sum,
                 np.trapz,
                 np.i0,
                 np.sinc,
                 np.arctanh,
                 np.gradient)

        for base in self.grids:
            grid = base.copy()
            for f in funcs:
                r1 = f(base)
                r2 = f(grid)

                try:
                    np.testing.assert_equal(r1.data,r2.data)
                    np.testing.assert_equal(r1.mask,r2.mask)
                except AttributeError:
                    np.testing.assert_equal(r1,r2)

if __name__== "__main__":
    unittest.main()
