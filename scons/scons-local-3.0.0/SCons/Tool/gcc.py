"""SCons.Tool.gcc

Tool-specific initialization for gcc.

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

__revision__ = "src/engine/SCons/Tool/gcc.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

from . import cc
import os
import re
import subprocess

import SCons.Util

compilers = ['gcc', 'cc']

def generate(env):
    """Add Builders and construction variables for gcc to an Environment."""

    if 'CC' not in env:
        env['CC'] = env.Detect(compilers) or compilers[0]

    cc.generate(env)

    if env['PLATFORM'] in ['cygwin', 'win32']:
        env['SHCCFLAGS'] = SCons.Util.CLVar('$CCFLAGS')
    else:
        env['SHCCFLAGS'] = SCons.Util.CLVar('$CCFLAGS -fPIC')
    # determine compiler version
    version = detect_version(env, env['CC'])
    if version:
        env['CCVERSION'] = version

def exists(env):
    # is executable, and is a GNU compiler (or accepts '--version' at least)
    return detect_version(env, env.Detect(env.get('CC', compilers)))

def detect_version(env, cc):
    """Return the version of the GNU compiler, or None if it is not a GNU compiler."""
    cc = env.subst(cc)
    if not cc:
        return None
    version = None
    #pipe = SCons.Action._subproc(env, SCons.Util.CLVar(cc) + ['-dumpversion'],
    pipe = SCons.Action._subproc(env, SCons.Util.CLVar(cc) + ['--version'],
                                 stdin = 'devnull',
                                 stderr = 'devnull',
                                 stdout = subprocess.PIPE)
    # -dumpversion was added in GCC 3.0.  As long as we're supporting
    # GCC versions older than that, we should use --version and a
    # regular expression.
    #line = pipe.stdout.read().strip()
    #if line:
    #    version = line
    line = SCons.Util.to_str(pipe.stdout.readline())
    match = re.search(r'[0-9]+(\.[0-9]+)+', line)
    if match:
        version = match.group(0)
    # Non-GNU compiler's output (like AIX xlc's) may exceed the stdout buffer:
    # So continue with reading to let the child process actually terminate.
    while SCons.Util.to_str(pipe.stdout.readline()):
        pass
    ret = pipe.wait()
    if ret != 0:
        return None
    return version

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
