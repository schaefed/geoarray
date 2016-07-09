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
from numpy.ma import MaskedArray

_METHODS = (
    "__add__",
)

def checkProjection(func):
    def inner(*args):
        # print "checking"
        return func(*args)
    return inner

class GeoArrayMeta(object):
    def __new__(cls, name, bases, attrs):
        for name in _METHODS:
            attrs[name] = checkProjection(getattr(MaskedArray, name))
        return type(name, bases, attrs)
    
