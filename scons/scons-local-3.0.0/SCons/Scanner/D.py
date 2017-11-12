"""SCons.Scanner.D

Scanner for the Digital Mars "D" programming language.

Coded by Andy Friesen
17 Nov 2003

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

__revision__ = "src/engine/SCons/Scanner/D.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons.Scanner

def DScanner():
    """Return a prototype Scanner instance for scanning D source files"""
    ds = D()
    return ds

class D(SCons.Scanner.Classic):
    def __init__ (self):
        SCons.Scanner.Classic.__init__ (
            self,
            name = "DScanner",
            suffixes = '$DSUFFIXES',
            path_variable = 'DPATH',
            regex = '(?:import\s+)([\w\s=,.]+)(?:\s*:[\s\w,=]+)?(?:;)'
        )

    def find_include(self, include, source_dir, path):
        # translate dots (package separators) to slashes
        inc = include.replace('.', '/')

        i = SCons.Node.FS.find_file(inc + '.d', (source_dir,) + path)
        if i is None:
            i = SCons.Node.FS.find_file (inc + '.di', (source_dir,) + path)
        return i, include

    def find_include_names(self, node):
        includes = []
        for iii in self.cre.findall(node.get_text_contents()):
            for jjj in iii.split(','):
                kkk = jjj.split('=')[-1]
                includes.append(kkk.strip())
        return includes

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
