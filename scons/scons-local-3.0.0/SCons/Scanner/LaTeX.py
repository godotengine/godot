"""SCons.Scanner.LaTeX

This module implements the dependency scanner for LaTeX code.

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

__revision__ = "src/engine/SCons/Scanner/LaTeX.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os.path
import re

import SCons.Scanner
import SCons.Util

# list of graphics file extensions for TeX and LaTeX
TexGraphics   = ['.eps', '.ps']
#LatexGraphics = ['.pdf', '.png', '.jpg', '.gif', '.tif']
LatexGraphics = [ '.png', '.jpg', '.gif', '.tif']


# Used as a return value of modify_env_var if the variable is not set.
class _Null(object):
    pass
_null = _Null

# The user specifies the paths in env[variable], similar to other builders.
# They may be relative and must be converted to absolute, as expected
# by LaTeX and Co. The environment may already have some paths in
# env['ENV'][var]. These paths are honored, but the env[var] paths have
# higher precedence. All changes are un-done on exit.
def modify_env_var(env, var, abspath):
    try:
        save = env['ENV'][var]
    except KeyError:
        save = _null
    env.PrependENVPath(var, abspath)
    try:
        if SCons.Util.is_List(env[var]):
            env.PrependENVPath(var, [os.path.abspath(str(p)) for p in env[var]])
        else:
            # Split at os.pathsep to convert into absolute path
            env.PrependENVPath(var, [os.path.abspath(p) for p in str(env[var]).split(os.pathsep)])
    except KeyError:
        pass

    # Convert into a string explicitly to append ":" (without which it won't search system
    # paths as well). The problem is that env.AppendENVPath(var, ":")
    # does not work, refuses to append ":" (os.pathsep).

    if SCons.Util.is_List(env['ENV'][var]):
        env['ENV'][var] = os.pathsep.join(env['ENV'][var])
    # Append the trailing os.pathsep character here to catch the case with no env[var]
    env['ENV'][var] = env['ENV'][var] + os.pathsep

    return save

class FindENVPathDirs(object):
    """
    A class to bind a specific E{*}PATH variable name to a function that
    will return all of the E{*}path directories.
    """
    def __init__(self, variable):
        self.variable = variable
    def __call__(self, env, dir=None, target=None, source=None, argument=None):
        import SCons.PathList
        try:
            path = env['ENV'][self.variable]
        except KeyError:
            return ()

        dir = dir or env.fs._cwd
        path = SCons.PathList.PathList(path).subst_path(env, target, source)
        return tuple(dir.Rfindalldirs(path))



def LaTeXScanner():
    """
    Return a prototype Scanner instance for scanning LaTeX source files
    when built with latex.
    """
    ds = LaTeX(name = "LaTeXScanner",
               suffixes =  '$LATEXSUFFIXES',
               # in the search order, see below in LaTeX class docstring
               graphics_extensions = TexGraphics,
               recursive = 0)
    return ds

def PDFLaTeXScanner():
    """
    Return a prototype Scanner instance for scanning LaTeX source files
    when built with pdflatex.
    """
    ds = LaTeX(name = "PDFLaTeXScanner",
               suffixes =  '$LATEXSUFFIXES',
               # in the search order, see below in LaTeX class docstring
               graphics_extensions = LatexGraphics,
               recursive = 0)
    return ds

class LaTeX(SCons.Scanner.Base):
    """
    Class for scanning LaTeX files for included files.

    Unlike most scanners, which use regular expressions that just
    return the included file name, this returns a tuple consisting
    of the keyword for the inclusion ("include", "includegraphics",
    "input", or "bibliography"), and then the file name itself.  
    Based on a quick look at LaTeX documentation, it seems that we 
    should append .tex suffix for the "include" keywords, append .tex if
    there is no extension for the "input" keyword, and need to add .bib
    for the "bibliography" keyword that does not accept extensions by itself.

    Finally, if there is no extension for an "includegraphics" keyword
    latex will append .ps or .eps to find the file, while pdftex may use .pdf,
    .jpg, .tif, .mps, or .png.
    
    The actual subset and search order may be altered by
    DeclareGraphicsExtensions command. This complication is ignored.
    The default order corresponds to experimentation with teTeX::

        $ latex --version
        pdfeTeX 3.141592-1.21a-2.2 (Web2C 7.5.4)
        kpathsea version 3.5.4

    The order is:
        ['.eps', '.ps'] for latex
        ['.png', '.pdf', '.jpg', '.tif'].

    Another difference is that the search path is determined by the type
    of the file being searched:
    env['TEXINPUTS'] for "input" and "include" keywords
    env['TEXINPUTS'] for "includegraphics" keyword
    env['TEXINPUTS'] for "lstinputlisting" keyword
    env['BIBINPUTS'] for "bibliography" keyword
    env['BSTINPUTS'] for "bibliographystyle" keyword
    env['INDEXSTYLE'] for "makeindex" keyword, no scanning support needed just allows user to set it if needed.

    FIXME: also look for the class or style in document[class|style]{}
    FIXME: also look for the argument of bibliographystyle{}
    """
    keyword_paths = {'include': 'TEXINPUTS',
                     'input': 'TEXINPUTS',
                     'includegraphics': 'TEXINPUTS',
                     'bibliography': 'BIBINPUTS',
                     'bibliographystyle': 'BSTINPUTS',
                     'addbibresource': 'BIBINPUTS',
                     'addglobalbib': 'BIBINPUTS',
                     'addsectionbib': 'BIBINPUTS',
                     'makeindex': 'INDEXSTYLE',
                     'usepackage': 'TEXINPUTS',
                     'lstinputlisting': 'TEXINPUTS'}
    env_variables = SCons.Util.unique(list(keyword_paths.values()))
    two_arg_commands = ['import', 'subimport',
                        'includefrom', 'subincludefrom',
                        'inputfrom', 'subinputfrom']

    def __init__(self, name, suffixes, graphics_extensions, *args, **kw):

        # We have to include \n with the % we exclude from the first part
        # part of the regex because the expression is compiled with re.M.
        # Without the \n,  the ^ could match the beginning of a *previous*
        # line followed by one or more newline characters (i.e. blank
        # lines), interfering with a match on the next line.
        # add option for whitespace before the '[options]' or the '{filename}'
        regex = r'''
            ^[^%\n]*
            \\(
                include
              | includegraphics(?:\s*\[[^\]]+\])?
              | lstinputlisting(?:\[[^\]]+\])?
              | input
              | import
              | subimport
              | includefrom
              | subincludefrom
              | inputfrom
              | subinputfrom
              | bibliography
              | addbibresource
              | addglobalbib
              | addsectionbib
              | usepackage
              )
                  \s*{([^}]*)}       # first arg
              (?: \s*{([^}]*)} )?    # maybe another arg
        '''
        self.cre = re.compile(regex, re.M | re.X)
        self.comment_re = re.compile(r'^((?:(?:\\%)|[^%\n])*)(.*)$', re.M)

        self.graphics_extensions = graphics_extensions

        def _scan(node, env, path=(), self=self):
            node = node.rfile()
            if not node.exists():
                return []
            return self.scan_recurse(node, path)

        class FindMultiPathDirs(object):
            """The stock FindPathDirs function has the wrong granularity:
            it is called once per target, while we need the path that depends
            on what kind of included files is being searched. This wrapper
            hides multiple instances of FindPathDirs, one per the LaTeX path
            variable in the environment. When invoked, the function calculates
            and returns all the required paths as a dictionary (converted into
            a tuple to become hashable). Then the scan function converts it
            back and uses a dictionary of tuples rather than a single tuple
            of paths.
            """
            def __init__(self, dictionary):
                self.dictionary = {}
                for k,n in dictionary.items():
                    self.dictionary[k] = ( SCons.Scanner.FindPathDirs(n),
                                           FindENVPathDirs(n) )

            def __call__(self, env, dir=None, target=None, source=None,
                                    argument=None):
                di = {}
                for k,(c,cENV)  in self.dictionary.items():
                    di[k] = ( c(env, dir=None, target=None, source=None,
                                   argument=None) ,
                              cENV(env, dir=None, target=None, source=None,
                                   argument=None) )
                # To prevent "dict is not hashable error"
                return tuple(di.items())

        class LaTeXScanCheck(object):
            """Skip all but LaTeX source files, i.e., do not scan *.eps,
            *.pdf, *.jpg, etc.
            """
            def __init__(self, suffixes):
                self.suffixes = suffixes
            def __call__(self, node, env):
                current = not node.has_builder() or node.is_up_to_date()
                scannable = node.get_suffix() in env.subst_list(self.suffixes)[0]
                # Returning false means that the file is not scanned.
                return scannable and current

        kw['function'] = _scan
        kw['path_function'] = FindMultiPathDirs(LaTeX.keyword_paths)
        kw['recursive'] = 0
        kw['skeys'] = suffixes
        kw['scan_check'] = LaTeXScanCheck(suffixes)
        kw['name'] = name

        SCons.Scanner.Base.__init__(self, *args, **kw)

    def _latex_names(self, include_type, filename):
        if include_type == 'input':
            base, ext = os.path.splitext( filename )
            if ext == "":
                return [filename + '.tex']
        if include_type in ('include', 'import', 'subimport',
                            'includefrom', 'subincludefrom',
                            'inputfrom', 'subinputfrom'):
            base, ext = os.path.splitext( filename )
            if ext == "":
                return [filename + '.tex']
        if include_type == 'bibliography':
            base, ext = os.path.splitext( filename )
            if ext == "":
                return [filename + '.bib']
        if include_type == 'usepackage':
            base, ext = os.path.splitext( filename )
            if ext == "":
                return [filename + '.sty']
        if include_type == 'includegraphics':
            base, ext = os.path.splitext( filename )
            if ext == "":
                #return [filename+e for e in self.graphics_extensions + TexGraphics]
                # use the line above to find dependencies for the PDF builder
                # when only an .eps figure is present.  Since it will be found
                # if the user tells scons how to make the pdf figure, leave
                # it out for now.
                return [filename+e for e in self.graphics_extensions]
        return [filename]

    def sort_key(self, include):
        return SCons.Node.FS._my_normcase(str(include))

    def find_include(self, include, source_dir, path):
        inc_type, inc_subdir, inc_filename = include
        try:
            sub_paths = path[inc_type]
        except (IndexError, KeyError):
            sub_paths = ((), ())
        try_names = self._latex_names(inc_type, inc_filename)

        # There are three search paths to try:
        #  1. current directory "source_dir"
        #  2. env[var]
        #  3. env['ENV'][var]
        search_paths = [(source_dir,)] + list(sub_paths)

        for n in try_names:
            for search_path in search_paths:
                paths = tuple([d.Dir(inc_subdir) for d in search_path])
                i = SCons.Node.FS.find_file(n, paths)
                if i:
                    return i, include
        return None, include

    def canonical_text(self, text):
        """Standardize an input TeX-file contents.

        Currently:
          * removes comments, unwrapping comment-wrapped lines.
        """
        out = []
        line_continues_a_comment = False
        for line in text.splitlines():
            line,comment = self.comment_re.findall(line)[0]
            if line_continues_a_comment == True:
                out[-1] = out[-1] + line.lstrip()
            else:
                out.append(line)
            line_continues_a_comment = len(comment) > 0
        return '\n'.join(out).rstrip()+'\n'

    def scan(self, node, subdir='.'):
        # Modify the default scan function to allow for the regular
        # expression to return a comma separated list of file names
        # as can be the case with the bibliography keyword.

        # Cache the includes list in node so we only scan it once:
        # path_dict = dict(list(path))
        # add option for whitespace (\s) before the '['
        noopt_cre = re.compile('\s*\[.*$')
        if node.includes != None:
            includes = node.includes
        else:
            text = self.canonical_text(node.get_text_contents())
            includes = self.cre.findall(text)
            # 1. Split comma-separated lines, e.g.
            #      ('bibliography', 'phys,comp')
            #    should become two entries
            #      ('bibliography', 'phys')
            #      ('bibliography', 'comp')
            # 2. Remove the options, e.g., such as
            #      ('includegraphics[clip,width=0.7\\linewidth]', 'picture.eps')
            #    should become
            #      ('includegraphics', 'picture.eps')
            split_includes = []
            for include in includes:
                inc_type = noopt_cre.sub('', include[0])
                inc_subdir = subdir
                if inc_type in self.two_arg_commands:
                    inc_subdir = os.path.join(subdir, include[1])
                    inc_list = include[2].split(',')
                else:
                    inc_list = include[1].split(',')
                for j in range(len(inc_list)):
                    split_includes.append( (inc_type, inc_subdir, inc_list[j]) )
            #
            includes = split_includes
            node.includes = includes

        return includes

    def scan_recurse(self, node, path=()):
        """ do a recursive scan of the top level target file
        This lets us search for included files based on the
        directory of the main file just as latex does"""

        path_dict = dict(list(path))
        
        queue = [] 
        queue.extend( self.scan(node) )
        seen = {}

        # This is a hand-coded DSU (decorate-sort-undecorate, or
        # Schwartzian transform) pattern.  The sort key is the raw name
        # of the file as specifed on the \include, \input, etc. line.
        # TODO: what about the comment in the original Classic scanner:
        # """which lets
        # us keep the sort order constant regardless of whether the file
        # is actually found in a Repository or locally."""
        nodes = []
        source_dir = node.get_dir()
        #for include in includes:
        while queue:
            
            include = queue.pop()
            inc_type, inc_subdir, inc_filename = include

            try:
                if seen[inc_filename] == 1:
                    continue
            except KeyError:
                seen[inc_filename] = 1

            #
            # Handle multiple filenames in include[1]
            #
            n, i = self.find_include(include, source_dir, path_dict)
            if n is None:
                # Do not bother with 'usepackage' warnings, as they most
                # likely refer to system-level files
                if inc_type != 'usepackage':
                    SCons.Warnings.warn(SCons.Warnings.DependencyWarning,
                                        "No dependency generated for file: %s (included from: %s) -- file not found" % (i, node))
            else:
                sortkey = self.sort_key(n)
                nodes.append((sortkey, n))
                # recurse down
                queue.extend( self.scan(n, inc_subdir) )

        return [pair[1] for pair in sorted(nodes)]

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
