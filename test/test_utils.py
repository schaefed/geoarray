#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import geoarray as ga
import os

PWD = os.path.abspath(os.path.dirname(__file__))
TMPPATH = os.path.join(PWD, "tmp")
FILES = tuple(
    os.path.join(TMPPATH, "test{:}".format(e)) for e in ga._DRIVER_DICT
)

def testArray(shape):
    dinfo = dtypeInfo(np.int32)
    return ga.array(
        data = np.random.randint(dinfo["min"], high=dinfo["max"], size=shape, dtype=np.int64),
        proj = 9001,
        yorigin = 7235561,
        xorigin = 3820288,
        cellsize = 1000,
        fill_value = -9999,
        mode = "L",
    )

def createDirectory(path):
    try:
        os.mkdir(path)
    except OSError:
        pass
  
def removeTestFiles():
    try:
        shutil.rmtree(TMPPATH)
    except:
        pass
 
def createTestFiles():
    createDirectory(TMPPATH)
    test_array = testArray((340, 270))
    out = []
    for fname in FILES:
        test_array.tofile(fname)
        out.append(ga.fromfile(fname))
    return out

def dtypeInfo(dtype):
    try:
        tinfo = np.finfo(dtype)
    except ValueError:
        tinfo = np.iinfo(dtype)
    return {"min": tinfo.min, "max": tinfo.max}

