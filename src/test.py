#! /usr/bin/env python
# -*- coding: utf-8 -*-

operators = ["add", "mul"]

class OperatorHackiness(object):
  """
  Use this base class if you want your object
  to intercept __add__, __iadd__, __radd__, __mul__ etc.
  using __getattr__.
  __getattr__ will called at most _once_ during the
  lifetime of the object, as the result is cached!
  """

  def __init__(self):
    # create a instance-local base class which we can
    # manipulate to our needs
    self.__class__ = self.meta = type('tmp', (self.__class__,), {})


# add operator methods dynamically, because we are damn lazy.
# This loop is however only called once in the whole program
# (when the module is loaded)
def create_operator(name):
  def dynamic_operator(self, *args):
    # call getattr to allow interception
    # by user
    func = self.__getattr__(name)
    # save the result in the temporary
    # base class to avoid calling getattr twice
    setattr(self.meta, name, func)
    # use provided function to calculate result
    return func(self, *args)
  return dynamic_operator

for op in operators:
  for name in ["__%s__" % op, "__r%s__" % op, "__i%s__" % op]:
    setattr(OperatorHackiness, name, create_operator(name))


# Example user class
class Test(OperatorHackiness):
  def __init__(self, x):
    super(Test, self).__init__()
    self.x = x

  def __getattr__(self, attr):
    print "__getattr__(%s)" % attr
    if attr == "__add__":
      return lambda a, b: a.x + b.x
    elif attr == "__iadd__":
      def iadd(self, other):
        self.x += other.x
        return self
      return iadd
    elif attr == "__mul__":
      return lambda a, b: a.x * b.x
    else:
      raise AttributeError

## Some test code:

a = Test(3)
b = Test(4)

# let's test addition
print a + b # this first call to __add__ will trigger
            # a __getattr__ call
print a + b # this second call will not!

# same for multiplication
print a * b
print a * b

# inplace addition (getattr is also only called once)
a += b
a += b
print a.x # yay!
