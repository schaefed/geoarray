#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import geoarray as ga
import os

PWD = os.path.abspath(os.path.dirname(__file__))
TMPPATH = os.path.join(PWD, "tmp")
FILES = tuple(os.path.join(TMPPATH, "test{:}".format(e)) for e in ga._DRIVER_DICT)


def testArray():
    shape = (340, 270)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) #np.random.rand(*shape)#.astype(np.float32)
    return ga.array(
        data,
        proj = 9001,
        yorigin = 7235561,
        xorigin = 3820288,
        cellsize = 1000,
        fill_value = -9999,
        mode = "L",
    )

def testFiles(fnames):
    out = []
    test_array = testArray()
    for fname in fnames:
        test_array.tofile(fname)
        out.append(ga.fromfile(fname))
    return out

def dtypeInfo(dtype):
    try:
        tinfo = np.finfo(dtype)
    except ValueError:
        tinfo = np.iinfo(dtype)
    return {"min": tinfo.min, "max": tinfo.max}

