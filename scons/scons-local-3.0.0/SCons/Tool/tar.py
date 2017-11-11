"""SCons.Tool.tar

Tool-specific initialization for tar.

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

__revision__ = "src/engine/SCons/Tool/tar.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons.Action
import SCons.Builder
import SCons.Defaults
import SCons.Node.FS
import SCons.Util

tars = ['tar', 'gtar']

TarAction = SCons.Action.Action('$TARCOM', '$TARCOMSTR')

TarBuilder = SCons.Builder.Builder(action = TarAction,
                                   source_factory = SCons.Node.FS.Entry,
                                   source_scanner = SCons.Defaults.DirScanner,
                                   suffix = '$TARSUFFIX',
                                   multi = 1)


def generate(env):
    """Add Builders and construction variables for tar to an Environment."""
    try:
        bld = env['BUILDERS']['Tar']
    except KeyError:
        bld = TarBuilder
        env['BUILDERS']['Tar'] = bld

    env['TAR']        = env.Detect(tars) or 'gtar'
    env['TARFLAGS']   = SCons.Util.CLVar('-c')
    env['TARCOM']     = '$TAR $TARFLAGS -f $TARGET $SOURCES'
    env['TARSUFFIX']  = '.tar'

def exists(env):
    return env.Detect(tars)

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
