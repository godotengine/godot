#
# Copyright (c) 2001 - 2017 The SCons Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
from __future__ import print_function

__revision__ = "src/engine/SCons/Memoize.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

__doc__ = """Memoizer

A decorator-based implementation to count hits and misses of the computed
values that various methods cache in memory.

Use of this modules assumes that wrapped methods be coded to cache their
values in a consistent way. In particular, it requires that the class uses a
dictionary named "_memo" to store the cached values.

Here is an example of wrapping a method that returns a computed value,
with no input parameters::

    @SCons.Memoize.CountMethodCall
    def foo(self):

        try:                                                    # Memoization
            return self._memo['foo']                            # Memoization
        except KeyError:                                        # Memoization
            pass                                                # Memoization

        result = self.compute_foo_value()

        self._memo['foo'] = result                              # Memoization

        return result

Here is an example of wrapping a method that will return different values
based on one or more input arguments::

    def _bar_key(self, argument):                               # Memoization
        return argument                                         # Memoization

    @SCons.Memoize.CountDictCall(_bar_key)
    def bar(self, argument):

        memo_key = argument                                     # Memoization
        try:                                                    # Memoization
            memo_dict = self._memo['bar']                       # Memoization
        except KeyError:                                        # Memoization
            memo_dict = {}                                      # Memoization
            self._memo['dict'] = memo_dict                      # Memoization
        else:                                                   # Memoization
            try:                                                # Memoization
                return memo_dict[memo_key]                      # Memoization
            except KeyError:                                    # Memoization
                pass                                            # Memoization

        result = self.compute_bar_value(argument)

        memo_dict[memo_key] = result                            # Memoization

        return result

Deciding what to cache is tricky, because different configurations
can have radically different performance tradeoffs, and because the
tradeoffs involved are often so non-obvious.  Consequently, deciding
whether or not to cache a given method will likely be more of an art than
a science, but should still be based on available data from this module.
Here are some VERY GENERAL guidelines about deciding whether or not to
cache return values from a method that's being called a lot:

    --  The first question to ask is, "Can we change the calling code
        so this method isn't called so often?"  Sometimes this can be
        done by changing the algorithm.  Sometimes the *caller* should
        be memoized, not the method you're looking at.

    --  The memoized function should be timed with multiple configurations
        to make sure it doesn't inadvertently slow down some other
        configuration.

    --  When memoizing values based on a dictionary key composed of
        input arguments, you don't need to use all of the arguments
        if some of them don't affect the return values.

"""

# A flag controlling whether or not we actually use memoization.
use_memoizer = None

# Global list of counter objects
CounterList = {}

class Counter(object):
    """
    Base class for counting memoization hits and misses.

    We expect that the initialization in a matching decorator will
    fill in the correct class name and method name that represents
    the name of the function being counted.
    """
    def __init__(self, cls_name, method_name):
        """
        """
        self.cls_name = cls_name
        self.method_name = method_name
        self.hit = 0
        self.miss = 0
    def key(self):
        return self.cls_name+'.'+self.method_name
    def display(self):
        print("    {:7d} hits {:7d} misses    {}()".format(self.hit, self.miss, self.key()))
    def __eq__(self, other):
        try:
            return self.key() == other.key()
        except AttributeError:
            return True

class CountValue(Counter):
    """
    A counter class for simple, atomic memoized values.

    A CountValue object should be instantiated in a decorator for each of
    the class's methods that memoizes its return value by simply storing
    the return value in its _memo dictionary.
    """
    def count(self, *args, **kw):
        """ Counts whether the memoized value has already been
            set (a hit) or not (a miss).
        """
        obj = args[0]
        if self.method_name in obj._memo:
            self.hit = self.hit + 1
        else:
            self.miss = self.miss + 1

class CountDict(Counter):
    """
    A counter class for memoized values stored in a dictionary, with
    keys based on the method's input arguments.

    A CountDict object is instantiated in a decorator for each of the
    class's methods that memoizes its return value in a dictionary,
    indexed by some key that can be computed from one or more of
    its input arguments.
    """
    def __init__(self, cls_name, method_name, keymaker):
        """
        """
        Counter.__init__(self, cls_name, method_name)
        self.keymaker = keymaker
    def count(self, *args, **kw):
        """ Counts whether the computed key value is already present
           in the memoization dictionary (a hit) or not (a miss).
        """
        obj = args[0]
        try:
            memo_dict = obj._memo[self.method_name]
        except KeyError:
            self.miss = self.miss + 1
        else:
            key = self.keymaker(*args, **kw)
            if key in memo_dict:
                self.hit = self.hit + 1
            else:
                self.miss = self.miss + 1

def Dump(title=None):
    """ Dump the hit/miss count for all the counters
        collected so far.
    """
    if title:
        print(title)
    for counter in sorted(CounterList):
        CounterList[counter].display()

def EnableMemoization():
    global use_memoizer
    use_memoizer = 1

def CountMethodCall(fn):
    """ Decorator for counting memoizer hits/misses while retrieving
        a simple value in a class method. It wraps the given method
        fn and uses a CountValue object to keep track of the
        caching statistics.
        Wrapping gets enabled by calling EnableMemoization().
    """
    if use_memoizer:
        def wrapper(self, *args, **kwargs):
            global CounterList
            key = self.__class__.__name__+'.'+fn.__name__
            if key not in CounterList:
                CounterList[key] = CountValue(self.__class__.__name__, fn.__name__)
            CounterList[key].count(self, *args, **kwargs)
            return fn(self, *args, **kwargs)
        wrapper.__name__= fn.__name__
        return wrapper
    else:
        return fn

def CountDictCall(keyfunc):
    """ Decorator for counting memoizer hits/misses while accessing
        dictionary values with a key-generating function. Like
        CountMethodCall above, it wraps the given method
        fn and uses a CountDict object to keep track of the
        caching statistics. The dict-key function keyfunc has to
        get passed in the decorator call and gets stored in the
        CountDict instance.
        Wrapping gets enabled by calling EnableMemoization().
    """
    def decorator(fn):
        if use_memoizer:
            def wrapper(self, *args, **kwargs):
                global CounterList
                key = self.__class__.__name__+'.'+fn.__name__
                if key not in CounterList:
                    CounterList[key] = CountDict(self.__class__.__name__, fn.__name__, keyfunc)
                CounterList[key].count(self, *args, **kwargs)
                return fn(self, *args, **kwargs)
            wrapper.__name__= fn.__name__
            return wrapper
        else:
            return fn
    return decorator

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
