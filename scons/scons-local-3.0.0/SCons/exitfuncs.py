"""SCons.exitfuncs

Register functions which are executed when SCons exits for any reason.

"""

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

__revision__ = "src/engine/SCons/exitfuncs.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"


import atexit

_exithandlers = []
def _run_exitfuncs():
    """run any registered exit functions

    _exithandlers is traversed in reverse order so functions are executed
    last in, first out.
    """

    while _exithandlers:
        func, targs, kargs =  _exithandlers.pop()
        func(*targs, **kargs)

def register(func, *targs, **kargs):
    """register a function to be executed upon normal program termination

    func - function to be called at exit
    targs - optional arguments to pass to func
    kargs - optional keyword arguments to pass to func
    """
    _exithandlers.append((func, targs, kargs))


# make our exit function get run by python when it exits
atexit.register(_run_exitfuncs)

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
