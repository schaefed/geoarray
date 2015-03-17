#! /usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np

# Comparisons
COMPARISON_OPERATORS = (
    "__eq__","__ne__","__lt__","__gt__","__le__","__ge__"
)    

# Operators returning a copy
COPY_OPERATORS = (
    "__add__","__sub__","__mul__","__div__","__floordiv__",
    "__truediv__","__mod__","__divmod__",
    "__pow__","__lshift__",
    "__rshift__","__and__","__or__","__xor__",
    "__radd__","__rsub__","__rmul__","__rdiv__","__rfloordiv__",
    "__rtruediv__","__rmod__","__rdivmod__",
    "__rpow__","__rlshift__",
    "__rrshift__","__rand__","__ror__","__rxor__"
)

# Operators changing data in place
INPLACE_OPERATORS = (
    "__iadd__","__isub__","__imul__","__idiv__","__ifloordiv__",
    "__itruediv__","__imod__","__ipow__","__ilshift__",
    "__irshift__","__iand__","__ior__","__ixor__",
)


class NumpyMemberBase(object):

    # __array_priority__ = 1

    """
    Purpose:
        An (Abstract)BaseClass which makes its subclasses (mostly) behave
        like a numpy.ndarray.
    
    Arguments
        obj:        instance of the childclass holding the np.ndarray object
        name:       name of the np.ndarray attribute in the childclass
        hooks:      dictionary {operator_name: function to call after operator call}

    Restrictions:
        - Performance is most likely not too good as the array view system
          of numpy is not implemented an more copies than necessary are
          created
        - Slincing an numpy.ndarray using a subclass of Numpymember with an boolean
          data attribute yields wrong results (__getitem__ actually treats the boolean
          as an integer array and slices it accordingly)
        - The operators set through the __setOperators method are not overwritable
          in the subclasses. The implemted method in the subclass won't ever be called
          -> A solution would be to implement all these methods seperately ;-(
    """
    
    def __init__(self, obj, name, hooks=None, #implement=False
             ):
        
        self.__obj = obj            
        self.__attr = name
        self.__ndarray = None
        self.__hooks = hooks if hooks else {}
        
        self.__setOperators(
            INPLACE_OPERATORS,
            self.__inplaceOperator,
        )

        self.__setOperators(
            COPY_OPERATORS,
            self.__copyOperator,
        )

        self.__setOperators(
            COMPARISON_OPERATORS,
            self.__copyOperator,
            # self.__comparisonOperators,
        )
                
    @property
    def __array_interface__(self):
        return self.__getArray().__array_interface__
        
    def __getArray(self):
        """
            return the ndarray from the childclass
        """
        if self.__ndarray == None:
            self.__ndarray = self.__obj.__getattribute__(self.__attr)
        return self.__ndarray

        
    def __prepObject(self,obj):
        """
            Returns the ndarray member of obj or the obj itself
            if it is not a subclass of NumpyMemberBase
        """
        try:
            return obj.__getArray()
        except AttributeError:
            return obj

    def __setOperators(self,operators,factory):
        for op in operators:
            setattr(self.__class__,op, factory(op))            

    def __changedCopy(self,array):
        """
            Returns a copy of self.__obj where the ndarray
            attribute is replaced by array. Right now the
            underlying data is first copied, than replaced.
            That is probably not the most efficent way...
        """
        # should be a deepcopy without self.__attr!
        out = copy.copy(self.__obj)
        out.__obj.__setattr__(out.__attr, array)
        return out
        
    def __inplaceOperator(self,opname):
        """
            Wrapper closure for the different inplace operators.
        """
        f = getattr(np.ndarray,opname)            
        def operator(self,other):
            self.__obj.__setattr__(self.__attr,
                    f(self.__getArray(),self.__prepObject(other))
                )
            if opname in self.__hooks:
                self.__hooks[opname](self)
            return self
        return operator
            
    def __copyOperator(self,opname):
        """
            Wrapper closure for the different copy operators.
        """
        f = getattr(np.ndarray,opname)
        def operator(self,other):
            out = self.__changedCopy(
                f(self.__getArray(),
                  self.__prepObject(other))                
            )            
            if opname in self.__hooks:
                self.__hooks[opname](out)
            return out
        return operator
    
    def __repr__(self):
        return self.__getArray().__repr__()
    
    def __str__(self):
        return self.__getArray().__str__()

    def __array_prepare__(cls,array,context=None):
        """
            Called at the beginning of every ufunc.
            The output of this function is passed to
            the ufunc -> the ndarray needs to be
            returned
        """
        if array.shape:
            return cls.__prepObject(array)
        return array[0]

        
    def __array_wrap__(cls,array,context=None):
        """
            Called at the end of every ufunc.
            The place to wrap the ndarray into
            an instance of self.__obj.__class__
        """
        if array.shape:
            return cls.__changedCopy(array)
        return array[0]

    # def __array__(cls,dtype=None):
    #     return cls.__getArray()
        
    def __getitem__(self,slc):
        """
            Redirect the slicing to the ndarray
            attribute of self.__obj
        """
        # return self.__getArray()[slc]
        return self.__getArray()[self.__prepObject(slc)]
            
    def __setitem__(self,slc,value):
        """
            Redirect the slicing to the ndarray
            attribute of self.__obj
        """
        # self.__getArray()[slc] = value
        self.__getArray()[self.__prepObject(slc)] = value
 
    
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

    # def __getattr__(self, name):
    #     if name == "__deepcopy__":
    #         return
    #     else:
    #         if name.startswith("__") or self.__implement:
    #             try:
    #                 return getattr(self.__getArray(),name)
    #             except AttributeError:
    #                 pass
    #         raise AttributeError("{:} object has no attribute '{:}'".format(
    #             self.__obj, name))

    
