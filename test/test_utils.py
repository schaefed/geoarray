#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import geoarray as ga
import os

PWD = os.path.abspath(os.path.dirname(__file__))
TMPPATH = os.path.join(PWD, "tmp")
# FILES = tuple(
#     os.path.join(TMPPATH, "test{:}".format(e)) for e in ga._DRIVER_DICT
# )

def testArray(shape):
    dinfo = dtypeInfo(np.int32)
    return ga.array(
        data = np.random.randint(dinfo["min"], high=dinfo["max"], size=shape, dtype=np.int32),
        proj = 9001,
        yorigin = 7235561,
        xorigin = 3820288,
        cellsize = 1000,
        fill_value = -9999,
        color_mode = "L")

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
    arrays = (
        testArray((340, 270)),
        testArray((4, 340, 270))
    )
    files, fnames = [], []
    for ending in ga._DRIVER_DICT:
        for i, arr in enumerate(arrays):
            fname = os.path.join(TMPPATH, "test-{:}{:}".format(i, ending))
            try:
                arr.tofile(fname)
            except RuntimeError:
                continue
            files.append(ga.fromfile(fname))
            fnames.append(fname)
    return tuple(fnames), tuple(files)

def dtypeInfo(dtype):
    try:
        tinfo = np.finfo(dtype)
    except ValueError:
        tinfo = np.iinfo(dtype)
    return {"min": tinfo.min, "max": tinfo.max}
