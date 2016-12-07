#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import tempfile
import geoarray as ga
import numpy as np
from test_utils import testArray, dtypeInfo

class Test(unittest.TestCase):

    def setUp(self):
        self.grid = testArray()
        
    def test_fromfile(self):
        endings = ga._DRIVER_DICT.keys()
        for ending in endings:
            with tempfile.NamedTemporaryFile(suffix=ending) as tf:
                # write and read again
                self.grid.tofile(tf.name)
                check = ga.fromfile(tf.name)
                # gdal truncates values smaller/larger than the datatype, numpy wraps around.
                # clip array to make things comparable.
                dinfo = dtypeInfo(check.dtype)
                grid = self.grid.clip(dinfo["min"], dinfo["max"])

                np.testing.assert_almost_equal(check, grid)
                self.assertDictEqual(check.bbox, self.grid.bbox)
                self.assertEqual(check.cellsize, self.grid.cellsize)
                self.assertEqual(check.proj, self.grid.proj)
                self.assertEqual(check.fill_value, self.grid.fill_value)
                self.assertEqual(check.mode, self.grid.mode)
