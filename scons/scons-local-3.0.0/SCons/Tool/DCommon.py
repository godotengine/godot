from __future__ import print_function

"""SCons.Tool.DCommon

Common code for the various D tools.

Coded by Russel Winder (russel@winder.org.uk)
2012-09-06
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

__revision__ = "src/engine/SCons/Tool/DCommon.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os.path


def isD(env, source):
    if not source:
        return 0
    for s in source:
        if s.sources:
            ext = os.path.splitext(str(s.sources[0]))[1]
            if ext == '.d':
                return 1
    return 0


def addDPATHToEnv(env, executable):
    dPath = env.WhereIs(executable)
    if dPath:
        phobosDir = dPath[:dPath.rindex(executable)] + '/../src/phobos'
        if os.path.isdir(phobosDir):
            env.Append(DPATH=[phobosDir])


def allAtOnceEmitter(target, source, env):
    if env['DC'] in ('ldc2', 'dmd'):
        env.SideEffect(str(target[0]) + '.o', target[0])
        env.Clean(target[0], str(target[0]) + '.o')
    return target, source


# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
