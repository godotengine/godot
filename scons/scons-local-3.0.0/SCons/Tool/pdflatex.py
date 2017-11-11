"""SCons.Tool.pdflatex

Tool-specific initialization for pdflatex.
Generates .pdf files from .latex or .ltx files

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

__revision__ = "src/engine/SCons/Tool/pdflatex.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons.Action
import SCons.Util
import SCons.Tool.pdf
import SCons.Tool.tex

PDFLaTeXAction = None

def PDFLaTeXAuxFunction(target = None, source= None, env=None):
    result = SCons.Tool.tex.InternalLaTeXAuxAction( PDFLaTeXAction, target, source, env )
    if result != 0:
        SCons.Tool.tex.check_file_error_message(env['PDFLATEX'])
    return result

PDFLaTeXAuxAction = None

def generate(env):
    """Add Builders and construction variables for pdflatex to an Environment."""
    global PDFLaTeXAction
    if PDFLaTeXAction is None:
        PDFLaTeXAction = SCons.Action.Action('$PDFLATEXCOM', '$PDFLATEXCOMSTR')

    global PDFLaTeXAuxAction
    if PDFLaTeXAuxAction is None:
        PDFLaTeXAuxAction = SCons.Action.Action(PDFLaTeXAuxFunction,
                              strfunction=SCons.Tool.tex.TeXLaTeXStrFunction)

    env.AppendUnique(LATEXSUFFIXES=SCons.Tool.LaTeXSuffixes)

    from . import pdf
    pdf.generate(env)

    bld = env['BUILDERS']['PDF']
    bld.add_action('.ltx', PDFLaTeXAuxAction)
    bld.add_action('.latex', PDFLaTeXAuxAction)
    bld.add_emitter('.ltx', SCons.Tool.tex.tex_pdf_emitter)
    bld.add_emitter('.latex', SCons.Tool.tex.tex_pdf_emitter)

    SCons.Tool.tex.generate_common(env)

def exists(env):
    SCons.Tool.tex.generate_darwin(env)
    return env.Detect('pdflatex')

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
