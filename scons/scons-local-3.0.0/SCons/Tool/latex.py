"""SCons.Tool.latex

Tool-specific initialization for LaTeX.
Generates .dvi files from .latex or .ltx files

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

__revision__ = "src/engine/SCons/Tool/latex.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons.Action
import SCons.Defaults
import SCons.Scanner.LaTeX
import SCons.Util
import SCons.Tool
import SCons.Tool.tex

def LaTeXAuxFunction(target = None, source= None, env=None):
    result = SCons.Tool.tex.InternalLaTeXAuxAction( SCons.Tool.tex.LaTeXAction, target, source, env )
    if result != 0:
        SCons.Tool.tex.check_file_error_message(env['LATEX'])
    return result

LaTeXAuxAction = SCons.Action.Action(LaTeXAuxFunction,
                              strfunction=SCons.Tool.tex.TeXLaTeXStrFunction)

def generate(env):
    """Add Builders and construction variables for LaTeX to an Environment."""

    env.AppendUnique(LATEXSUFFIXES=SCons.Tool.LaTeXSuffixes)

    from . import dvi
    dvi.generate(env)

    from . import pdf
    pdf.generate(env)

    bld = env['BUILDERS']['DVI']
    bld.add_action('.ltx', LaTeXAuxAction)
    bld.add_action('.latex', LaTeXAuxAction)
    bld.add_emitter('.ltx', SCons.Tool.tex.tex_eps_emitter)
    bld.add_emitter('.latex', SCons.Tool.tex.tex_eps_emitter)

    SCons.Tool.tex.generate_common(env)

def exists(env):
    SCons.Tool.tex.generate_darwin(env)
    return env.Detect('latex')

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
