#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np

class Slice(object):
    """
    Abstracts all possible __getitem__ arguments to a numpy.ndarray into 
    an Object follwing the slice-notation of start/stop/step attributes.

    The idea is to be able to extract this basic slicing information in 
    an unique way for all the different index types.
    
    In order to actually index your object you should however use the
    original argument to __getitem__ !!
    """
    
    def __init__(self,obj,length):
        self.obj    = obj
        self.length = length
        self.data   = self._prepare(obj)

    @property
    def first(self):
        return self.data.start

    @property
    def last(self):
       return self.data.stop

    @property
    def step(self):
        return self.data.step
       
    def _prepare(self,obj):
        try:
            idx = obj.indices(self.length)
            idxrange = xrange(*idx)
            slc = slice(idxrange[0], idxrange[-1], idx[-1])
        except AttributeError: # not a slice object
            slc = np.array(obj,ndmin=1).ravel()
            if slc[0] == Ellipsis:
                slc = slice(0, self.length-1, 1)
            else:
                slc[slc < 0] += self.length 
                diff = np.unique(np.diff(slc))
                if len(diff) == 1: # equally spaced elements -> sorted
                    slc = slice(slc[0],slc[-1],diff[0])
                else:
                    slc = slice(min(slc),max(slc),1)                    
        return slc
            
    def __str__(self):
        return "Slice({}, {}, {})".format(self.first, self.last, self.step)
    
# ADVANCED_INDEXING_TYPES = (list, np.ndarray)
# INT_TYPES = (int, np.int, np.int32, np.int64, np.uint, np.uint32, np.uint64)

def getSlices(slices,shape):
   
    # if ((isinstance(slices, INT_TYPES))
    #     or (isinstance(slices, slice))
    #     or (isinstance(slices, ADVANCED_INDEXING_TYPES))
    #     or (slices is Ellipsis)
    # ):
    #     slices = (slices,)

    try:
        slices = tuple(slices)
    except TypeError:
        slices = (slices,)

    out = []
    for i in xrange(len(shape)):
        try:
            slc = slices[i]
            
            if isinstance(slc, np.ndarray):
                if slc.dtype is np.dtype("bool"):
                    slc = np.nonzero(slices)
                else:
                    if slc.ndim > 1:
                        raise TypeError("Non-boolean arrays with more than one dimension are not supported yet!")
            out.append(Slice(slc,shape[i]))
        except IndexError:
            out.append(Slice(Ellipsis,shape[i]))
    return tuple(out)
        
TESTCASES = (
    Ellipsis,
    0,
    12,
    -1,
    -2,
    -10,
    range(4),
    range(4,10),
    range(-10,-1),
    range(4,100,25),
    range(100,10,-2),
    np.arange(4),
    np.arange(4,10),
    np.arange(-10,-1),
    np.arange(4,100,25),
    np.arange(100,10,-2),
    slice(3,None,None),
    slice(-3,None,None),
    slice(3,-1,None),
    slice(5,44,5),
    slice(None,-1,None),
    slice(-65,-1,-4),
    slice(100,10,-2),
    (5,3,1,3,88,54,-55), # failing in test_use by design
)
        
class TestSlice(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(TestSlice,self).__init__(*args,**kwargs)
        self.length = 120
        self.array = np.arange(self.length)

    def test_start(self):
        results = (
            # Ellipsis
            0,
            # Integers
            0, 12, 119, 118, 110,
            # range
            0, 4, 110, 4, 100,
            # np.arange
            0, 4, 110, 4, 100,
            # slices
            3, 117, 3, 5, 0, 55, 100,
            # list
            1,
        )
        
        for date,expected in zip(TESTCASES,results):
            slc = Slice(date,self.length)
            self.assertEqual(slc.start,expected)

    def test_stop(self):
        results = (
            # Ellipsis
            120,
            # Integers
            1, 13, 120, 119, 111,
            # range
            4, 10, 119, 104, 10,
            # np.arange
            4, 10, 119, 104, 10,
            # slices
            120, 120, 119, 44, 119, 119, 10,
            # list
            89
        )

        for date,expected in zip(TESTCASES,results):
            slc = Slice(date,self.length)
            self.assertEqual(slc.stop,expected)
 
            
    def test_step(self):
        results = (
            # Ellipsis
            1,
            # Integers
            1, 1, 1, 1, 1,
            # range
            1, 1, 1, 25, -2,
            # np.arange
            1, 1, 1, 25, -2,
            # slices
            1, 1, 1, 5, 1, -4, -2,
            # list
            1,
        )
        for date,expected in zip(TESTCASES, results):
            slc = Slice(date, self.length)
            self.assertEqual(slc.step, expected)

  
if __name__== "__main__":

    unittest.main()
