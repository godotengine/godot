"""engine.SCons.Platform.aix

Platform-specific initialization for IBM AIX systems.

There normally shouldn't be any need to import this module directly.  It
will usually be imported through the generic SCons.Platform.Platform()
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

__revision__ = "src/engine/SCons/Platform/aix.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os
import subprocess

from . import posix

import SCons.Util
import SCons.Action

def get_xlc(env, xlc=None, packages=[]):
    # Use the AIX package installer tool lslpp to figure out where a
    # given xl* compiler is installed and what version it is.
    xlcPath = None
    xlcVersion = None

    if xlc is None:
        xlc = env.get('CC', 'xlc')
    if SCons.Util.is_List(xlc):
        xlc = xlc[0]
    for package in packages:
        # find the installed filename, which may be a symlink as well
        pipe = SCons.Action._subproc(env, ['lslpp', '-fc', package],
                stdin = 'devnull',
                stderr = 'devnull',
                stdout = subprocess.PIPE)
        # output of lslpp is something like this:
        #     #Path:Fileset:File
        #     /usr/lib/objrepos:vac.C 6.0.0.0:/usr/vac/exe/xlCcpp
        #     /usr/lib/objrepos:vac.C 6.0.0.0:/usr/vac/bin/xlc_r -> /usr/vac/bin/xlc
        for line in pipe.stdout:
            if xlcPath:
                continue # read everything to let lslpp terminate
            fileset, filename = line.split(':')[1:3]
            filename = filename.split()[0]
            if ('/' in xlc and filename == xlc) \
            or ('/' not in xlc and filename.endswith('/' + xlc)):
                xlcVersion = fileset.split()[1]
                xlcPath, sep, xlc = filename.rpartition('/')
            pass
        pass
    return (xlcPath, xlc, xlcVersion)

def generate(env):
    posix.generate(env)
    #Based on AIX 5.2: ARG_MAX=24576 - 3000 for environment expansion
    env['MAXLINELENGTH']  = 21576

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
