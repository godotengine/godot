"""SCons.Tool.gnulink

Tool-specific initialization for the gnu linker.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

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

__revision__ = "src/engine/SCons/Tool/gnulink.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons.Util
import SCons.Tool
import os
import sys
import re

from . import link


def generate(env):
    """Add Builders and construction variables for gnulink to an Environment."""
    link.generate(env)

    if env['PLATFORM'] == 'hpux':
        env['SHLINKFLAGS'] = SCons.Util.CLVar('$LINKFLAGS -shared -fPIC')

    # __RPATH is set to $_RPATH in the platform specification if that
    # platform supports it.
    env['RPATHPREFIX'] = '-Wl,-rpath='
    env['RPATHSUFFIX'] = ''
    env['_RPATH'] = '${_concat(RPATHPREFIX, RPATH, RPATHSUFFIX, __env__)}'

    # OpenBSD doesn't usually use SONAME for libraries
    use_soname = not sys.platform.startswith('openbsd')
    link._setup_versioned_lib_variables(env, tool = 'gnulink', use_soname = use_soname)
    env['LINKCALLBACKS'] = link._versioned_lib_callbacks()

    # For backward-compatibility with older SCons versions
    env['SHLIBVERSIONFLAGS'] = SCons.Util.CLVar('-Wl,-Bsymbolic')
    
def exists(env):
    # TODO: sync with link.smart_link() to choose a linker
    linkers = { 'CXX': ['g++'], 'CC': ['gcc'] }
    alltools = []
    for langvar, linktools in linkers.items():
        if langvar in env: # use CC over CXX when user specified CC but not CXX
            return SCons.Tool.FindTool(linktools, env)
        alltools.extend(linktools)
    return SCons.Tool.FindTool(alltools, env) # find CXX or CC

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
