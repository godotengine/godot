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

__doc__ = """
SCons compatibility package for old Python versions

This subpackage holds modules that provide backwards-compatible
implementations of various things that we'd like to use in SCons but which
only show up in later versions of Python than the early, old version(s)
we still support.

Other code will not generally reference things in this package through
the SCons.compat namespace.  The modules included here add things to
the builtins namespace or the global module list so that the rest
of our code can use the objects and names imported here regardless of
Python version.

The rest of the things here will be in individual compatibility modules
that are either: 1) suitably modified copies of the future modules that
we want to use; or 2) backwards compatible re-implementations of the
specific portions of a future module's API that we want to use.

GENERAL WARNINGS:  Implementations of functions in the SCons.compat
modules are *NOT* guaranteed to be fully compliant with these functions in
later versions of Python.  We are only concerned with adding functionality
that we actually use in SCons, so be wary if you lift this code for
other uses.  (That said, making these more nearly the same as later,
official versions is still a desirable goal, we just don't need to be
obsessive about it.)

We name the compatibility modules with an initial '_scons_' (for example,
_scons_subprocess.py is our compatibility module for subprocess) so
that we can still try to import the real module name and fall back to
our compatibility module if we get an ImportError.  The import_as()
function defined below loads the module as the "real" name (without the
'_scons'), after which all of the "import {module}" statements in the
rest of our code will find our pre-loaded compatibility module.
"""

__revision__ = "src/engine/SCons/compat/__init__.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os
import sys
import imp  # Use the "imp" module to protect imports from fixers.

PYPY = hasattr(sys, 'pypy_translation_info')


def import_as(module, name):
    """
    Imports the specified module (from our local directory) as the
    specified name, returning the loaded module object.
    """
    dir = os.path.split(__file__)[0]
    return imp.load_module(name, *imp.find_module(module, [dir]))


def rename_module(new, old):
    """
    Attempts to import the old module and load it under the new name.
    Used for purely cosmetic name changes in Python 3.x.
    """
    try:
        sys.modules[new] = imp.load_module(old, *imp.find_module(old))
        return True
    except ImportError:
        return False


# TODO: FIXME
# In 3.x, 'pickle' automatically loads the fast version if available.
rename_module('pickle', 'cPickle')

# Default pickle protocol. Higher protocols are more efficient/featureful
# but incompatible with older Python versions. On Python 2.7 this is 2.
# Negative numbers choose the highest available protocol.
import pickle

# Was pickle.HIGHEST_PROTOCOL
# Changed to 2 so py3.5+'s pickle will be compatible with py2.7.
PICKLE_PROTOCOL = 2

# TODO: FIXME
# In 3.x, 'profile' automatically loads the fast version if available.
rename_module('profile', 'cProfile')

# TODO: FIXME
# Before Python 3.0, the 'queue' module was named 'Queue'.
rename_module('queue', 'Queue')

# TODO: FIXME
# Before Python 3.0, the 'winreg' module was named '_winreg'
rename_module('winreg', '_winreg')

# Python 3 moved builtin intern() to sys package
# To make porting easier, make intern always live
# in sys package (for python 2.7.x)
try:
    sys.intern
except AttributeError:
    # We must be using python 2.7.x so monkey patch
    # intern into the sys package
    sys.intern = intern

# Preparing for 3.x. UserDict, UserList, UserString are in
# collections for 3.x, but standalone in 2.7.x
import collections

try:
    collections.UserDict
except AttributeError:
    exec ('from UserDict import UserDict as _UserDict')
    collections.UserDict = _UserDict
    del _UserDict

try:
    collections.UserList
except AttributeError:
    exec ('from UserList import UserList as _UserList')
    collections.UserList = _UserList
    del _UserList

try:
    collections.UserString
except AttributeError:
    exec ('from UserString import UserString as _UserString')
    collections.UserString = _UserString
    del _UserString


import shutil
try:
    shutil.SameFileError
except AttributeError:
    class SameFileError(Exception):
        pass

    shutil.SameFileError = SameFileError

def with_metaclass(meta, *bases):
    """
    Function from jinja2/_compat.py. License: BSD.

    Use it like this::

        class BaseForm(object):
            pass

        class FormType(type):
            pass

        class Form(with_metaclass(FormType, BaseForm)):
            pass

    This requires a bit of explanation: the basic idea is to make a
    dummy metaclass for one level of class instantiation that replaces
    itself with the actual metaclass.  Because of internal type checks
    we also need to make sure that we downgrade the custom metaclass
    for one level to something closer to type (that's why __call__ and
    __init__ comes back from type etc.).

    This has the advantage over six.with_metaclass of not introducing
    dummy classes into the final MRO.
    """

    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__

        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)

    return metaclass('temporary_class', None, {})


class NoSlotsPyPy(type):
    """
    Workaround for PyPy not working well with __slots__ and __class__ assignment.
    """

    def __new__(meta, name, bases, dct):
        if PYPY and '__slots__' in dct:
            dct.pop('__slots__')
        return super(NoSlotsPyPy, meta).__new__(meta, name, bases, dct)

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
