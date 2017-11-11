"""SCons.Tool.gs

Tool-specific initialization for Ghostscript.

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

__revision__ = "src/engine/SCons/Tool/gs.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons.Action
import SCons.Builder
import SCons.Platform
import SCons.Util

# Ghostscript goes by different names on different platforms...
platform = SCons.Platform.platform_default()

if platform == 'os2':
    gs = 'gsos2'
elif platform == 'win32':
    gs = 'gswin32c'
else:
    gs = 'gs'

GhostscriptAction = None

def generate(env):
    """Add Builders and construction variables for Ghostscript to an
    Environment."""
    global GhostscriptAction
    # The following try-except block enables us to use the Tool
    # in standalone mode (without the accompanying pdf.py),
    # whenever we need an explicit call of gs via the Gs()
    # Builder ...
    try:
        if GhostscriptAction is None:
            GhostscriptAction = SCons.Action.Action('$GSCOM', '$GSCOMSTR')
    
        from SCons.Tool import pdf
        pdf.generate(env)
    
        bld = env['BUILDERS']['PDF']
        bld.add_action('.ps', GhostscriptAction)
    except ImportError as e:
        pass

    gsbuilder = SCons.Builder.Builder(action = SCons.Action.Action('$GSCOM', '$GSCOMSTR'))
    env['BUILDERS']['Gs'] = gsbuilder
    
    env['GS']      = gs
    env['GSFLAGS'] = SCons.Util.CLVar('-dNOPAUSE -dBATCH -sDEVICE=pdfwrite')
    env['GSCOM']   = '$GS $GSFLAGS -sOutputFile=$TARGET $SOURCES'


def exists(env):
    if 'PS2PDF' in env:
        return env.Detect(env['PS2PDF'])
    else:
        return env.Detect(gs) or SCons.Util.WhereIs(gs)

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
