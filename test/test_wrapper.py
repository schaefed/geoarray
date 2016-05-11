#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest, copy, shutil, os
import numpy as np
import geoarray as ga
import gdal
import warnings
import subprocess
import tempfile

# all tests, run from main directory:
# python -m unittest discover test

# this test only, run from parent directory run 
# python -m unittest test.test_wrapper

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
              
if __name__== "__main__":
    unittest.main()
