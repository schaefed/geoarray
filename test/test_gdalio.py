#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import tempfile
import geoarray as ga
import numpy as np
from test_utils import testArray, dtypeInfo

class Test(unittest.TestCase):

    def test_io(self):
        test_array = testArray((340, 270))
        endings = ga._DRIVER_DICT.keys()
        for ending in endings:
            with tempfile.NamedTemporaryFile(suffix=ending) as tf:
                # write and read again
                test_array.tofile(tf.name)
                check_array = ga.fromfile(tf.name)

                # gdal truncates values smaller/larger than the datatype, numpy wraps around.
                # clip array to make things comparable.
                dinfo = dtypeInfo(check_array.dtype)
                grid = test_array.clip(dinfo["min"], dinfo["max"])
                fill_value = check_array.dtype.type(test_array.fill_value)

                np.testing.assert_almost_equal(check_array, grid)
                self.assertDictEqual(check_array.bbox, test_array.bbox)
                self.assertEqual(check_array.cellsize, test_array.cellsize)
                self.assertEqual(check_array.proj, test_array.proj)
                self.assertEqual(check_array.fill_value, fill_value)
                self.assertEqual(check_array.color_mode, test_array.color_mode)

    def test_updateio(self):
        test_array = testArray((340, 270))
        slices = slice(1, -1, 3)
        endings = ga._DRIVER_DICT.keys()
        for ending in endings:
            if ending != ".tif":
                continue
            with tempfile.NamedTemporaryFile(suffix=ending) as tf:
                test_array.tofile(tf.name)
                check_file = ga.fromfile(tf.name, "a")
                check_file[slices] = 42
                check_file.close()
                check_file = ga.fromfile(tf.name, "r")
                self.assertTrue((check_file[slices] == 42).all())
