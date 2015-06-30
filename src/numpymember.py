#! /usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np
# from contextlib import contextmanager

# @contextmanager
# def popInstanceAttribute(obj,name):
#     objcls = obj.__class__
#     attr = getattr(objcls,name)
#     delattr(objcls,name)
#     yield obj
#     setattr(objcls,name,attr)

class NumpyMemberBase(object):


    """
    Purpose:
        An (Abstract)BaseClass which makes its subclasses (mostly) behave
        like a numpy.ndarray.
    
    Arguments
        obj:        instance of the childclass holding the np.ndarray object
        name:       name of the np.ndarray attribute in the childclass

    Restrictions:
        - Slincing a numpy.ndarray using a subclass of NumpyMemberBase with an boolean
          data attribute yields wrong results (__getitem__ actually treats the boolean
          as an integer array and slices it accordingly)
    """
    
    def __init__(self, obj, name):
        
        self.__obj = obj            
        self.__attr = name
                
    @property
    def __array_interface__(self):
        return self._array.__array_interface__

    def __array_prepare__(self,array,context=None):
        """
            Called at the beginning of every ufunc.
            The output of this function is passed to
            the ufunc -> the ndarray needs to be
            returned
        """
        return array

    def __array_wrap__(self,array,context=None):
        """
            Called at the end of every ufunc.
            The place to wrap the ndarray into
            an instance of self.__obj.__class__
        """
        return self.__copy(array)
        
    def _getArray(self):
        """
            return the ndarray from the childclass
        """
        return self.__obj.__getattribute__(self.__attr)

    def _setArray(self,arr):
        self.__obj.__setattr__(self.__attr,arr)

    def __copy(self,data):
        """
        There should be more efficient solution
        """
        out = copy.deepcopy(self.__obj)
        out._array = data
        return out
        
    def __prepObject(self,obj):
        """
        Returns the ndarray member of obj or the obj itself
        if it is not a subclass of NumpyMemberBase
        TODO:
            Check if obj is a subclass of NumpyMemberBase
        """
        if isinstance(obj, NumpyMemberBase):
            return obj._array
        return obj

    def __getitem__(self,slc):
        """
            Redirect the slicing to the ndarray
            attribute of self.__obj
        """
        sliced = self._array.__getitem__(
            self.__prepObject(slc)
        )    
        if isinstance(sliced, self.__class__):
            return sliced
        return self.__copy(sliced)

    def __setitem__(self,slc,value):
        """
            Redirect the slicing to the ndarray
            attribute of self.__obj
        """
        self._array.__setitem__(self.__prepObject(slc),value)

        
    # def __getitem__(self,slc):
    #     """
    #         Redirect the slicing to the ndarray
    #         attribute of self.__obj
    #     """
    #     return self._array[self.__prepObject(slc)]

    # def __setitem__(self,slc,value):
    #     """
    #         Redirect the slicing to the ndarray
    #         attribute of self.__obj
    #     """
    #     self._array[self.__prepObject(slc)] = value
 
    
    def __getslice__(self, start, stop) :
        """
            This solves a subtle bug, where __getitem__ is not 
            called, and all the dimensional checking not done,
            when a slice of only the first dimension is taken, 
            e.g. a[1:3]. From the Python docs:
            Deprecated since version 2.0: Support slice objects
            as parameters to the __getitem__() method. (However,
            built-in types in CPython currently still implement
            __getslice__(). Therefore, you have to override it
            in derived classes when implementing slicing.)
        """
        return self.__getitem__(slice(start, stop))

    def __setslice__(self, start, stop, value) :
        """
            See __getslice__
        """
        return self.__setitem__(slice(start, stop),value)

        
    #Inplace operators
    def __iadd__(self,other):
        self._array = self._array.__iadd__(self.__prepObject(other))                   
        
    def __isub__(self,other):
        self._array = self._array.__isub__(self.__prepObject(other))                   
        
    def __imul__(self,other):
        self._array = self._array.__imul__(self.__prepObject(other))                   
        
    def __idiv__(self,other):
        self._array = self._array.__idiv__(self.__prepObject(other))                   
        
    def __ifloordiv__(self,other):
        self._array = self._array.__ifloordiv__(self.__prepObject(other))                   
        
    def __itruediv__(self,other):
        self._array = self._array.__itruediv__(self.__prepObject(other))                   
        
    def __imod__(self,other):
        self._array = self._array.__imod__(self.__prepObject(other))                   
        
    def __ipow__(self,other):
        self._array = self._array.__ipow__(self.__prepObject(other))                   
        
    def __ilshift__(self,other):
        self._array = self._array.__ilshift__(self.__prepObject(other))                   
        
    def __irshift__(self,other):
        self._array = self._array.__irshift__(self.__prepObject(other))                   
        
    def __iand__(self,other):
        self._array = self._array.__iand__(self.__prepObject(other))                   
        
    def __ior__(self,other):
        self._array = self._array.__ior__(self.__prepObject(other))                   
        
    def __ixor__(self,other):
        self._array = self._array.__ixor__(self.__prepObject(other))                   
        
    # Comparison operators
    def __eq__(self,other):
        return self.__copy(
            self._array.__eq__(self.__prepObject(other)))    

    def __ne__(self,other):
        return self.__copy(
            self._array.__ne__(self.__prepObject(other)))    
    
    def __lt__(self,other):
        return self.__copy(
            self._array.__lt__(self.__prepObject(other)))    
    
    def __gt__(self,other):
        return self.__copy(
            self._array.__gt__(self.__prepObject(other)))    
    
    def __le__(self,other):
        return self.__copy(
            self._array.__le__(self.__prepObject(other)))    
    
    def __ge__(self,other):
        return self.__copy(
            self._array.__ge__(self.__prepObject(other)))    
    
    # Copy operators
    def __add__(self, other):
        return self.__copy(
            self._array.__add__(self.__prepObject(other)))
    
    def __sub__(self, other):
        return self.__copy(
            self._array.__sub__(self.__prepObject(other)))
    
    def __mul__(self, other):
        return self.__copy(
            self._array.__mul__(self.__prepObject(other)))
    
    def __div__(self, other):
        return self.__copy(
            self._array.__div__(self.__prepObject(other)))
    
    def __floordiv__(self, other):    
        return self.__copy(
            self._array.__floordiv__(self.__prepObject(other)))

    def __truediv__(self, other):
        return self.__copy(
            self._array.__truediv__(self.__prepObject(other)))

    def __mod__(self, other):
        return self.__copy(
            self._array.__mod__(self.__prepObject(other)))
    
    def __divmod__(self, other):
        return self.__copy(
            self._array.__divmod__(self.__prepObject(other)))

    def __pow__(self, other):
        return self.__copy(
            self._array.__pow__(self.__prepObject(other)))
    
    def __lshift__(self, other):
        return self.__copy(
            self._array.__lshift__(self.__prepObject(other)))

    def __rshift__(self, other):
        return self.__copy(
            self._array.__rshift__(self.__prepObject(other)))

    def __and__(self, other):
        return self.__copy(
            self._array.__and__(self.__prepObject(other)))
    
    def __or__(self, other):
        return self.__copy(
            self._array.__or__(self.__prepObject(other)))
    
    def __xor__(self, other):
        return self.__copy(
            self._array.__xor__(self.__prepObject(other)))
    
    def __radd__(self, other):
        return self.__copy(
            self._array.__radd__(self.__prepObject(other)))
    
    def __rsub__(self, other):
        return self.__copy(
            self._array.__rsub__(self.__prepObject(other)))
    
    def __rmul__(self, other):
        return self.__copy(
            self._array.__rmul__(self.__prepObject(other)))
    
    def __rdiv__(self, other):
        return self.__copy(
            self._array.__rdiv__(self.__prepObject(other)))
    
    def __rfloordiv__(self, other):
        return self.__copy(
            self._array.__rfloordiv__(self.__prepObject(other)))

    def __rtruediv__(self, other):
        return self.__copy(
            self._array.__rtruediv__(self.__prepObject(other)))

    def __rmod__(self, other):
        return self.__copy(
            self._array.__rmod__(self.__prepObject(other)))
    
    def __rdivmod__(self, other):
        return self.__copy(
            self._array.__rdivmod__(self.__prepObject(other)))

    def __rpow__(self, other):
        return self.__copy(
            self._array.__rpow__(self.__prepObject(other)))
    
    def __rlshift__(self, other):
        return self.__copy(
            self._array.__rlshift__(self.__prepObject(other)))

    def __rrshift__(self, other):
        return self.__copy(
            self._array.__rrshift__(self.__prepObject(other)))

    def __rand__(self, other):
        return self.__copy(
            self._array.__rand__(self.__prepObject(other)))
    
    def __ror__(self, other):
        return self.__copy(
            self._array.__ror__(self.__prepObject(other)))
    
    def __rxor__(self, other):
        return self.__copy(
            self._array.__rxor__(self.__prepObject(other)))
    
    def __repr__(self):
        return self._array.__repr__()
    
    def __str__(self):
        return self._array.__str__()

    _array    = property(fget=lambda self:            self._getArray(),
                         fset=lambda self, value:     self._setArray(value))

