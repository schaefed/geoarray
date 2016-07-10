#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author
------
David Schaefer

Purpose
-------
This module provides the GeoArray metaclass.

"""
import warnings
from numpy.ma import MaskedArray

_METHODS = (
    "__add__",
)

def checkProjection(func):
    def inner(*args):
        tmp = set()
        for a in args:
            try:
                tmp.add(a.proj)
            except AttributeError:
                pass
        if len(tmp) > 1:
            warnings.warn("Incompatible map projections!", RuntimeWarning)
        return func(*args)
    return inner

class GeoArrayMeta(object):
    def __new__(cls, name, bases, attrs):
        for key in _METHODS:
            attrs[key] = checkProjection(getattr(MaskedArray, key))
        return type(name, bases, attrs)
    
