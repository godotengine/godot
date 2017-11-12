"""SCons.Tool.aixlink

Tool-specific initialization for the IBM Visual Age linker.

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

__revision__ = "src/engine/SCons/Tool/aixlink.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os
import os.path

import SCons.Util

from . import aixcc
from . import link

import SCons.Tool.cxx
cplusplus = SCons.Tool.cxx
#cplusplus = __import__('cxx', globals(), locals(), [])


def smart_linkflags(source, target, env, for_signature):
    if cplusplus.iscplusplus(source):
        build_dir = env.subst('$BUILDDIR', target=target, source=source)
        if build_dir:
            return '-qtempinc=' + os.path.join(build_dir, 'tempinc')
    return ''

def generate(env):
    """
    Add Builders and construction variables for Visual Age linker to
    an Environment.
    """
    link.generate(env)

    env['SMARTLINKFLAGS'] = smart_linkflags
    env['LINKFLAGS']      = SCons.Util.CLVar('$SMARTLINKFLAGS')
    env['SHLINKFLAGS']    = SCons.Util.CLVar('$LINKFLAGS -qmkshrobj -qsuppress=1501-218')
    env['SHLIBSUFFIX']    = '.a'

def exists(env):
    # TODO: sync with link.smart_link() to choose a linker
    linkers = { 'CXX': ['aixc++'], 'CC': ['aixcc'] }
    alltools = []
    for langvar, linktools in linkers.items():
        if langvar in env: # use CC over CXX when user specified CC but not CXX
            return SCons.Tool.FindTool(linktools, env)
        alltools.extend(linktools)
    return SCons.Tool.FindTool(alltools, env)

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
