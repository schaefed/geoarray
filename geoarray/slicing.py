#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np

# Python 3
try:
    xrange
except NameError:
    xrange = range

    
class Slice(object):
    def __init__(self, slc, shape):
        self._slc = slc
        self._data = np.arange(shape)
        self._idx = self._data[self._slc]
        
    @property
    def min(self):
        return min(self._idx)
    
    @property
    def max(self):
        return max(self._idx)

    @property
    def start(self):
        try:
            return self._idx[0]
        except IndexError:
            return 
            
    @property
    def stop(self):
        try:
            return self._idx[-1]
        except IndexError:
            return None
       
    @property
    def step(self):
        diff = np.unique(np.diff(self._idx))
        if len(diff) == 0:
            return None
        if len(diff) == 1:
            return diff[0]
        return 1
   
    def __len__(self):
        return len(self._idx)
    
    # def isContiguous(self):
    #     diff = np.unique(np.diff(self.data))
    #     return len(diff) == 1 or len(self.data) == 1
 
        
class Slices(object):
    def __init__(self, slc, shape):
        self._slc  = getSlices(slc, shape)

    @property
    def ndim(self):
        return len(self._slc)
        
    @property
    def start(self):
        return tuple(s.start for s in self)

    @property
    def stop(self):
        return tuple(s.stop for s in self)

    @property
    def step(self):
        return tuple(s.step for s in self)

    @property
    def indices(self):
        return tuple(slc._idx for slc in self._slc)
    
    # make class iterable
    def __getitem__(self, idx):
        return self._slc[idx]
       
def _tupelizeSlices(slices):
    
    if isinstance(slices, list):
        try:
            slices = np.asarray(slices)
        except ValueError: # heterogenous list, e.g [1,[1,2]]
            slices = tuple(slices)
    
    if isinstance(slices, np.ndarray):
        if slices.dtype is np.dtype("O"): # heteorogenous array
            slices = tuple(slices)
        else: # homogenous array -> indices to only the first dimension
            slices = (slices,)
            
    if not isinstance(slices, tuple): # integer, slice, Ellipsis
        slices = (slices,)

    return slices

def _handleEllipsis(slices, shape):
    out = []
    found = False
    for i, slc in enumerate(slices):
        if slc is Ellipsis and not found:
            # The first ellipsis 'fills' the slice
            for _ in xrange((len(shape)-len(slices))+1):
                out.append(Ellipsis)
            found = True
            continue
        out.append(slc)
    return tuple(out)

def _handleArrays(slices):
    out = []
    for slc in slices:
        if isinstance(slc, np.ndarray):
            if slc.dtype is np.dtype("bool"):
                out.extend(list(np.nonzero(slc)))
            else:
                out.append(slc.ravel())
        else:
            out.append(slc)
    return tuple(out)
         

def getSlices(slices, shape):
    
    slices = _tupelizeSlices(slices)
    slices = _handleArrays(slices)
    slices = _handleEllipsis(slices, shape)

    if len(slices) > len(shape):
        raise IndexError("too many indices")

    slices = slices + (Ellipsis,)*len(shape)
    return tuple(Slice(slc, shp) for slc, shp in zip(slices, shape))
   
# def getSlicesNew(slices, shape):
#     slices = _tupelizeSlices(slices)
#     slices = _handleEllipsis(slices, shape)
    
#     if len(slices) > len(shape):
#         raise IndexError("too many indices")
    
#     slices += [Ellipsis]*len(shape)
#     out = []
#     for i in xrange(len(shape)):
#         slc = slices[i]
#         if isinstance(slc,np.ndarray):
#             if slc.dtype is np.dtype("bool"):
#                 idx = np.nonzero(slc)
#                 slc = idx[0]
#                 for j,other in enumerate(slc[1:]):
#                     slices.insert(i+j+1,other)
#             if slc.ndim > 1:
#                 slc = slc.ravel()
#         # print slc
#         out.append(Slice(slc,shape[i]))
#     return tuple(out)

   
# TESTCASES = (
#     Ellipsis,
#     0,
#     12,
#     -1,
#     -2,
#     -10,
#     range(4),
#     range(4,10),
#     range(-10,-1),
#     range(4,100,25),
#     range(100,10,-2),
#     np.arange(4),
#     np.arange(4,10),
#     np.arange(-10,-1),
#     np.arange(4,100,25),
#     np.arange(100,10,-2),
#     slice(3,None,None),
#     slice(-3,None,None),
#     slice(3,-1,None),
#     slice(5,44,5),
#     slice(None,-1,None),
#     #slice(-65,-1,-4),
#     slice(100,10,-2),
#     #(5,3,1,3,88,54,-55), 
# )
        
# class TestSlice(unittest.TestCase):
#     def __init__(self,*args,**kwargs):
#         super(TestSlice,self).__init__(*args,**kwargs)
#         self.length = 120
#         self.array = np.arange(self.length)

#     def test_first(self):
#         results = (
#             # Ellipsis
#             0,
#             # Integers
#             0, 12, 119, 118, 110,
#             # range
#             0, 4, 110, 4, 100,
#             # np.arange
#             0, 4, 110, 4, 100,
#             # slices
#             3, 117, 3, 5, 0, 100,
#             # list
#             1,
#         )
        
#         for date,expected in zip(TESTCASES,results):
#             slc = Slice(date,self.length)
#             self.assertEqual(slc.first,expected)

#     def test_last(self):
#         results = (
#             # Ellipsis
#             119,
#             # Integers
#             0, 12, 119, 118, 110,
#             # range
#             3, 9, 118, 79, 12,
#             # np.arange
#             3, 9, 118, 79, 12,
#             # slices
#             119, 119, 118, 40, 118, 12,
#             # # list
#             88
#         )

#         for date,expected in zip(TESTCASES,results):
#             slc = Slice(date,self.length)
#             self.assertEqual(slc.last,expected)
 
            
#     def test_step(self):
#         results = (
#             # Ellipsis
#             1,
#             # Integers
#             1, 1, 1, 1, 1,
#             # range
#             1, 1, 1, 25, -2,
#             # np.arange
#             1, 1, 1, 25, -2,
#             # slices
#             1, 1, 1, 5, 1, -2,
#             # list
#             1,
#         )
#         for date,expected in zip(TESTCASES, results):
#             slc = Slice(date, self.length)
#             self.assertEqual(slc.step, expected)

    # def test_handleEllipsis(self):
    #     dates = (
    #         ((0,Ellipsis,1,Ellipsis,Ellipsis), (1,2,6,6)),
    #     )
  
if __name__== "__main__":

    pass
    # unittest.main()
    
