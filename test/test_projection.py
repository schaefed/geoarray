#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import warnings
import geoarray as ga
import numpy as np

class Test(unittest.TestCase):
        
    def test_correctValue(self):
        g = ga.ones((200,300), proj=3857) # WGS 84 / Pseudo-Mercator aka Web-Mercator
        self.assertTrue(len(g.proj) > 0)

    def test_incorrectValue(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # With some Python versions w is empty, so skip the test...
            if w:
                ga.ones((200,300), proj=4444) # invalid epsg code 
                self.assertEqual(str(w[0].message), "Projection not understood")
                self.assertEqual(w[0].category, RuntimeWarning)
