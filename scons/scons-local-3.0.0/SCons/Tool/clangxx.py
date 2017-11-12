# -*- coding: utf-8; -*-

"""SCons.Tool.clang++

Tool-specific initialization for clang++.

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

# __revision__ = "src/engine/SCons/Tool/clangxx.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

# Based on SCons/Tool/g++.py by Paweł Tomulik 2014 as a separate tool.
# Brought into the SCons mainline by Russel Winder 2017.

import os.path
import re
import subprocess
import sys

import SCons.Tool
import SCons.Util
import SCons.Tool.cxx

compilers = ['clang++']

def generate(env):
    """Add Builders and construction variables for clang++ to an Environment."""
    static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

    SCons.Tool.cxx.generate(env)

    env['CXX']        = env.Detect(compilers) or 'clang++'

    # platform specific settings
    if env['PLATFORM'] == 'aix':
        env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS -mminimal-toc')
        env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1
        env['SHOBJSUFFIX'] = '$OBJSUFFIX'
    elif env['PLATFORM'] == 'hpux':
        env['SHOBJSUFFIX'] = '.pic.o'
    elif env['PLATFORM'] == 'sunos':
        env['SHOBJSUFFIX'] = '.pic.o'
    # determine compiler version
    if env['CXX']:
        pipe = SCons.Action._subproc(env, [env['CXX'], '--version'],
                                     stdin='devnull',
                                     stderr='devnull',
                                     stdout=subprocess.PIPE)
        if pipe.wait() != 0: return
        # clang -dumpversion is of no use
        line = pipe.stdout.readline()
        if sys.version_info[0] > 2:
            line = line.decode()
        match = re.search(r'clang +version +([0-9]+(?:\.[0-9]+)+)', line)
        if match:
            env['CXXVERSION'] = match.group(1)

def exists(env):
    return env.Detect(compilers)

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
