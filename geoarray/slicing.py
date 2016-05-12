#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np

# Python 3
try:
    xrange
except NameError:
    xrange = range

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

def _handleNone(slices, shape):

    outslices = list(slices)
    outshape = list(shape)
    
    for i, slc in enumerate(slices):
        if slc is None:
            outslices[i] = 1
            outshape.insert(i,1)
        
    return tuple(outslices), tuple(outshape)

def getSlices(slices, shape):
    
    slices = _tupelizeSlices(slices)
    slices = _handleArrays(slices)
    slices = _handleEllipsis(slices, shape)
    slices, shape = _handleNone(slices, shape)

    if len(slices) > len(shape):
        raise IndexError("too many indices")

    slices = slices + (Ellipsis,)*len(shape)
    return tuple(Slice(slc, shp) for slc, shp in zip(slices, shape))
   
     
class Slice(object):
    def __init__(self, slc, shape):
        self._slc = slc
        self._data = np.arange(shape)
        self._idx = np.atleast_1d(self._data[self._slc])
        
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

 
       
