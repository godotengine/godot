"""engine.SCons.Variables.BoolVariable

This file defines the option type for SCons implementing true/false values.

Usage example::

    opts = Variables()
    opts.Add(BoolVariable('embedded', 'build for an embedded system', 0))
    ...
    if env['embedded'] == 1:
    ...
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

__revision__ = "src/engine/SCons/Variables/BoolVariable.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

__all__ = ['BoolVariable',]

import SCons.Errors

__true_strings  = ('y', 'yes', 'true', 't', '1', 'on' , 'all' )
__false_strings = ('n', 'no', 'false', 'f', '0', 'off', 'none')


def _text2bool(val):
    """
    Converts strings to True/False depending on the 'truth' expressed by
    the string. If the string can't be converted, the original value
    will be returned.

    See '__true_strings' and '__false_strings' for values considered
    'true' or 'false respectively.

    This is usable as 'converter' for SCons' Variables.
    """
    lval = val.lower()
    if lval in __true_strings: return True
    if lval in __false_strings: return False
    raise ValueError("Invalid value for boolean option: %s" % val)


def _validator(key, val, env):
    """
    Validates the given value to be either '0' or '1'.
    
    This is usable as 'validator' for SCons' Variables.
    """
    if not env[key] in (True, False):
        raise SCons.Errors.UserError(
            'Invalid value for boolean option %s: %s' % (key, env[key]))


def BoolVariable(key, help, default):
    """
    The input parameters describe a boolean option, thus they are
    returned with the correct converter and validator appended. The
    'help' text will by appended by '(yes|no) to show the valid
    valued. The result is usable for input to opts.Add().
    """
    return (key, '%s (yes|no)' % help, default,
            _validator, _text2bool)

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
