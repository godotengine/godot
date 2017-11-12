"""SCons.Tool.midl

Tool-specific initialization for midl (Microsoft IDL compiler).

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

__revision__ = "src/engine/SCons/Tool/midl.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons.Action
import SCons.Builder
import SCons.Defaults
import SCons.Scanner.IDL
import SCons.Util

from .MSCommon import msvc_exists

def midl_emitter(target, source, env):
    """Produces a list of outputs from the MIDL compiler"""
    base, _ = SCons.Util.splitext(str(target[0]))
    tlb = target[0]
    incl = base + '.h'
    interface = base + '_i.c'
    targets = [tlb, incl, interface]

    midlcom = env['MIDLCOM']

    if midlcom.find('/proxy') != -1:
        proxy = base + '_p.c'
        targets.append(proxy)
    if midlcom.find('/dlldata') != -1:
        dlldata = base + '_data.c'
        targets.append(dlldata)

    return (targets, source)

idl_scanner = SCons.Scanner.IDL.IDLScan()

midl_action = SCons.Action.Action('$MIDLCOM', '$MIDLCOMSTR')

midl_builder = SCons.Builder.Builder(action = midl_action,
                                     src_suffix = '.idl',
                                     suffix='.tlb',
                                     emitter = midl_emitter,
                                     source_scanner = idl_scanner)

def generate(env):
    """Add Builders and construction variables for midl to an Environment."""

    env['MIDL']          = 'MIDL.EXE'
    env['MIDLFLAGS']     = SCons.Util.CLVar('/nologo')
    env['MIDLCOM']       = '$MIDL $MIDLFLAGS /tlb ${TARGETS[0]} /h ${TARGETS[1]} /iid ${TARGETS[2]} /proxy ${TARGETS[3]} /dlldata ${TARGETS[4]} $SOURCE 2> NUL'
    env['BUILDERS']['TypeLibrary'] = midl_builder

def exists(env):
    return msvc_exists()

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
