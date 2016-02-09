#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np

try:
    xrange
except NameError:
    xrange = range
    
class Slice(object):
    
    def __init__(self, obj, length):
        self.obj    = obj
        self.length = length
        self.data   = self._prepare(obj)
        
    @property
    def min(self):
        try:
            return min(self.data)
        except ValueError: # empty sequence
            return 
        
    @property
    def max(self):
        try:
            return max(self.data)
        except ValueError: # empty sequence
            return 

    @property
    def first(self):
        try:
            return self.data[0]
        except IndexError:
            return

    @property
    def last(self):
        try:
            return self.data[-1]
        except IndexError:
            return
        
    @property
    def step(self):
        if len(self.data) > 0:
            diff = np.unique(np.diff(self.data))
            if len(diff) == 1:
                return diff[0]
            return 1
        return 0
    
    def isContiguous(self):
        diff = np.unique(np.diff(self.data))
        return len(diff) == 1 or len(self.data) == 1
        
    def _prepare(self,obj):
        """
        obj should be one off:
        - slice
        - Ellipsis
        - list
        - tuple
        - array
        - int
        - range/xrange
        """
        tmp = obj
        if isinstance(obj,slice):
            tmp = np.arange(*obj.indices(self.length))
        elif obj is Ellipsis:
            tmp = np.arange(self.length)
        elif isinstance(obj,(int,np.int,np.int32,np.int64,
                             np.uint,np.uint32,np.uint64)):
            tmp = np.array(obj,ndmin=1)
        elif isinstance(obj,np.ma.MaskedArray):
            tmp = obj.data[~obj.mask]
        else:
            tmp = np.asarray(obj)
            
        out = tmp.copy()
        out[out < 0] += self.length

        illegal =  np.where((out > self.length) | (out < 0))[0]
        if len(illegal):
            raise IndexError("index {:} out of bounds for axis with size {:}".format(tmp[illegal[0]],self.length))

        return out
        

def _tupelizeSlices(slices):
    
    if isinstance(slices,list):
        try:
            slices = np.asarray(slices)
        except ValueError: # heterogenous list, e.g [1,[1,2]]
            slices = tuple(slices)
    
    if isinstance(slices,np.ndarray):
        if slices.dtype is np.dtype("O"): # heteorogenous array
            slices = tuple(slices)
        else: # homogenous array -> indices for first dimension
            slices = (slices,)
            
    if not isinstance(slices,tuple): # integer, slice, Ellipsis
        slices = (slices,)

    return slices

def _handleEllipsis(slices,shape):

    out = []
    found = False
    for i,slc in enumerate(slices):
        if slc is Ellipsis and not found:
            for _ in xrange((len(shape)-len(slices))+1):
                out.append(Ellipsis)
            found = True
            continue
        out.append(slc)
    return out
    
def getSlices(slices,shape):
    slices = _tupelizeSlices(slices)
    slices = _handleEllipsis(slices,shape)
    
    if len(slices) > len(shape):
        raise IndexError("too many indices")

    slices += [Ellipsis]*len(shape)
    out = []
    for i in xrange(len(shape)):
        slc = slices[i]
        if isinstance(slc,np.ndarray):
            if slc.dtype is np.dtype("bool"):
                idx = np.nonzero(slc)
                slc = idx[0]
                for j,other in enumerate(slc[1:]):
                    slices.insert(i+j+1,other)
            if slc.ndim > 1:
                slc = slc.ravel()
                
        out.append(Slice(slc,shape[i]))
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
    #slice(-65,-1,-4),
    slice(100,10,-2),
    #(5,3,1,3,88,54,-55), 
)
        
class TestSlice(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(TestSlice,self).__init__(*args,**kwargs)
        self.length = 120
        self.array = np.arange(self.length)

    def test_first(self):
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
            3, 117, 3, 5, 0, 100,
            # list
            1,
        )
        
        for date,expected in zip(TESTCASES,results):
            slc = Slice(date,self.length)
            self.assertEqual(slc.first,expected)

    def test_last(self):
        results = (
            # Ellipsis
            119,
            # Integers
            0, 12, 119, 118, 110,
            # range
            3, 9, 118, 79, 12,
            # np.arange
            3, 9, 118, 79, 12,
            # slices
            119, 119, 118, 40, 118, 12,
            # # list
            88
        )

        for date,expected in zip(TESTCASES,results):
            slc = Slice(date,self.length)
            self.assertEqual(slc.last,expected)
 
            
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
            1, 1, 1, 5, 1, -2,
            # list
            1,
        )
        for date,expected in zip(TESTCASES, results):
            slc = Slice(date, self.length)
            self.assertEqual(slc.step, expected)

    # def test_handleEllipsis(self):
    #     dates = (
    #         ((0,Ellipsis,1,Ellipsis,Ellipsis), (1,2,6,6)),
    #     )
  
if __name__== "__main__":

    unittest.main()
    
