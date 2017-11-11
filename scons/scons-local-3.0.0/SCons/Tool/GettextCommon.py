"""SCons.Tool.GettextCommon module

Used by several tools of `gettext` toolset.
"""

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

__revision__ = "src/engine/SCons/Tool/GettextCommon.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons.Warnings
import re


#############################################################################
class XgettextToolWarning(SCons.Warnings.Warning): pass


class XgettextNotFound(XgettextToolWarning): pass


class MsginitToolWarning(SCons.Warnings.Warning): pass


class MsginitNotFound(MsginitToolWarning): pass


class MsgmergeToolWarning(SCons.Warnings.Warning): pass


class MsgmergeNotFound(MsgmergeToolWarning): pass


class MsgfmtToolWarning(SCons.Warnings.Warning): pass


class MsgfmtNotFound(MsgfmtToolWarning): pass


#############################################################################
SCons.Warnings.enableWarningClass(XgettextToolWarning)
SCons.Warnings.enableWarningClass(XgettextNotFound)
SCons.Warnings.enableWarningClass(MsginitToolWarning)
SCons.Warnings.enableWarningClass(MsginitNotFound)
SCons.Warnings.enableWarningClass(MsgmergeToolWarning)
SCons.Warnings.enableWarningClass(MsgmergeNotFound)
SCons.Warnings.enableWarningClass(MsgfmtToolWarning)
SCons.Warnings.enableWarningClass(MsgfmtNotFound)


#############################################################################

#############################################################################
class _POTargetFactory(object):
    """ A factory of `PO` target files.
    
    Factory defaults differ from these of `SCons.Node.FS.FS`.  We set `precious`
    (this is required by builders and actions gettext) and `noclean` flags by
    default for all produced nodes.
    """

    def __init__(self, env, nodefault=True, alias=None, precious=True
                 , noclean=True):
        """ Object constructor.
    
        **Arguments**
    
            - *env* (`SCons.Environment.Environment`)
            - *nodefault* (`boolean`) - if `True`, produced nodes will be ignored
              from default target `'.'`
            - *alias* (`string`) - if provided, produced nodes will be automatically
              added to this alias, and alias will be set as `AlwaysBuild`
            - *precious* (`boolean`) - if `True`, the produced nodes will be set as
              `Precious`.
            - *noclen* (`boolean`) - if `True`, the produced nodes will be excluded
              from `Clean`.
        """
        self.env = env
        self.alias = alias
        self.precious = precious
        self.noclean = noclean
        self.nodefault = nodefault

    def _create_node(self, name, factory, directory=None, create=1):
        """ Create node, and set it up to factory settings. """
        import SCons.Util
        node = factory(name, directory, create)
        node.set_noclean(self.noclean)
        node.set_precious(self.precious)
        if self.nodefault:
            self.env.Ignore('.', node)
        if self.alias:
            self.env.AlwaysBuild(self.env.Alias(self.alias, node))
        return node

    def Entry(self, name, directory=None, create=1):
        """ Create `SCons.Node.FS.Entry` """
        return self._create_node(name, self.env.fs.Entry, directory, create)

    def File(self, name, directory=None, create=1):
        """ Create `SCons.Node.FS.File` """
        return self._create_node(name, self.env.fs.File, directory, create)


#############################################################################

#############################################################################
_re_comment = re.compile(r'(#[^\n\r]+)$', re.M)
_re_lang = re.compile(r'([a-zA-Z0-9_]+)', re.M)


#############################################################################
def _read_linguas_from_files(env, linguas_files=None):
    """ Parse `LINGUAS` file and return list of extracted languages """
    import SCons.Util
    import SCons.Environment
    global _re_comment
    global _re_lang
    if not SCons.Util.is_List(linguas_files) \
            and not SCons.Util.is_String(linguas_files) \
            and not isinstance(linguas_files, SCons.Node.FS.Base) \
            and linguas_files:
        # If, linguas_files==True or such, then read 'LINGUAS' file.
        linguas_files = ['LINGUAS']
    if linguas_files is None:
        return []
    fnodes = env.arg2nodes(linguas_files)
    linguas = []
    for fnode in fnodes:
        contents = _re_comment.sub("", fnode.get_text_contents())
        ls = [l for l in _re_lang.findall(contents) if l]
        linguas.extend(ls)
    return linguas


#############################################################################

#############################################################################
from SCons.Builder import BuilderBase


#############################################################################
class _POFileBuilder(BuilderBase):
    """ `PO` file builder.
  
    This is multi-target single-source builder. In typical situation the source
    is single `POT` file, e.g. `messages.pot`, and there are multiple `PO`
    targets to be updated from this `POT`. We must run
    `SCons.Builder.BuilderBase._execute()` separatelly for each target to track
    dependencies separatelly for each target file.
    
    **NOTE**: if we call `SCons.Builder.BuilderBase._execute(.., target, ...)`
    with target being list of all targets, all targets would be rebuilt each time
    one of the targets from this list is missing. This would happen, for example,
    when new language `ll` enters `LINGUAS_FILE` (at this moment there is no
    `ll.po` file yet). To avoid this, we override
    `SCons.Builder.BuilerBase._execute()` and call it separatelly for each
    target. Here we also append to the target list the languages read from
    `LINGUAS_FILE`.
    """

    #
    # * The argument for overriding _execute(): We must use environment with
    #  builder overrides applied (see BuilderBase.__init__(). Here it comes for
    #  free.
    # * The argument against using 'emitter': The emitter is called too late
    #  by BuilderBase._execute(). If user calls, for example:
    #
    #    env.POUpdate(LINGUAS_FILE = 'LINGUAS')
    #
    #  the builder throws error, because it is called with target=None,
    #  source=None and is trying to "generate" sources or target list first.
    #  If user calls
    #
    #    env.POUpdate(['foo', 'baz'], LINGUAS_FILE = 'LINGUAS')
    #
    #  the env.BuilderWrapper() calls our builder with target=None,
    #  source=['foo', 'baz']. The BuilderBase._execute() then splits execution
    #  and execute iterativelly (recursion) self._execute(None, source[i]).
    #  After that it calls emitter (which is quite too late). The emitter is
    #  also called in each iteration, what makes things yet worse.
    def __init__(self, env, **kw):
        if not 'suffix' in kw:
            kw['suffix'] = '$POSUFFIX'
        if not 'src_suffix' in kw:
            kw['src_suffix'] = '$POTSUFFIX'
        if not 'src_builder' in kw:
            kw['src_builder'] = '_POTUpdateBuilder'
        if not 'single_source' in kw:
            kw['single_source'] = True
        alias = None
        if 'target_alias' in kw:
            alias = kw['target_alias']
            del kw['target_alias']
        if not 'target_factory' in kw:
            kw['target_factory'] = _POTargetFactory(env, alias=alias).File
        BuilderBase.__init__(self, **kw)

    def _execute(self, env, target, source, *args, **kw):
        """ Execute builder's actions.
        
        Here we append to `target` the languages read from `$LINGUAS_FILE` and 
        apply `SCons.Builder.BuilderBase._execute()` separatelly to each target.
        The arguments and return value are same as for
        `SCons.Builder.BuilderBase._execute()`. 
        """
        import SCons.Util
        import SCons.Node
        linguas_files = None
        if 'LINGUAS_FILE' in env and env['LINGUAS_FILE']:
            linguas_files = env['LINGUAS_FILE']
            # This prevents endless recursion loop (we'll be invoked once for
            # each target appended here, we must not extend the list again).
            env['LINGUAS_FILE'] = None
            linguas = _read_linguas_from_files(env, linguas_files)
            if SCons.Util.is_List(target):
                target.extend(linguas)
            elif target is not None:
                target = [target] + linguas
            else:
                target = linguas
        if not target:
            # Let the SCons.BuilderBase to handle this patologic situation
            return BuilderBase._execute(self, env, target, source, *args, **kw)
        # The rest is ours
        if not SCons.Util.is_List(target):
            target = [target]
        result = []
        for tgt in target:
            r = BuilderBase._execute(self, env, [tgt], source, *args, **kw)
            result.extend(r)
        if linguas_files is not None:
            env['LINGUAS_FILE'] = linguas_files
        return SCons.Node.NodeList(result)


#############################################################################

import SCons.Environment


#############################################################################
def _translate(env, target=None, source=SCons.Environment._null, *args, **kw):
    """ Function for `Translate()` pseudo-builder """
    if target is None: target = []
    pot = env.POTUpdate(None, source, *args, **kw)
    po = env.POUpdate(target, pot, *args, **kw)
    return po


#############################################################################

#############################################################################
class RPaths(object):
    """ Callable object, which returns pathnames relative to SCons current
    working directory.
  
    It seems like `SCons.Node.FS.Base.get_path()` returns absolute paths
    for nodes that are outside of current working directory (`env.fs.getcwd()`).
    Here, we often have `SConscript`, `POT` and `PO` files within `po/`
    directory and source files (e.g. `*.c`) outside of it. When generating `POT`
    template file, references to source files are written to `POT` template, so
    a translator may later quickly jump to appropriate source file and line from
    its `PO` editor (e.g. `poedit`).  Relative paths in  `PO` file are usually
    interpreted by `PO` editor as paths relative to the place, where `PO` file
    lives. The absolute paths would make resultant `POT` file nonportable, as
    the references would be correct only on the machine, where `POT` file was
    recently re-created. For such reason, we need a function, which always
    returns relative paths. This is the purpose of `RPaths` callable object.
  
    The `__call__` method returns paths relative to current working directory, but
    we assume, that *xgettext(1)* is run from the directory, where target file is
    going to be created.
  
    Note, that this may not work for files distributed over several hosts or
    across different drives on windows. We assume here, that single local
    filesystem holds both source files and target `POT` templates.
  
    Intended use of `RPaths` - in `xgettext.py`::
  
      def generate(env):
          from GettextCommon import RPaths
          ...
          sources = '$( ${_concat( "", SOURCES, "", __env__, XgettextRPaths, TARGET, SOURCES)} $)'
          env.Append(
            ...
            XGETTEXTCOM = 'XGETTEXT ... ' + sources,
            ...
            XgettextRPaths = RPaths(env)
          )
    """

    # NOTE: This callable object returns pathnames of dirs/files relative to
    # current working directory. The pathname remains relative also for entries
    # that are outside of current working directory (node, that
    # SCons.Node.FS.File and siblings return absolute path in such case). For
    # simplicity we compute path relative to current working directory, this
    # seems be enough for our purposes (don't need TARGET variable and
    # SCons.Defaults.Variable_Caller stuff).

    def __init__(self, env):
        """ Initialize `RPaths` callable object.
    
          **Arguments**:
    
            - *env* - a `SCons.Environment.Environment` object, defines *current
              working dir*.
        """
        self.env = env

    # FIXME: I'm not sure, how it should be implemented (what the *args are in
    # general, what is **kw).
    def __call__(self, nodes, *args, **kw):
        """ Return nodes' paths (strings) relative to current working directory. 
        
          **Arguments**:
    
            - *nodes* ([`SCons.Node.FS.Base`]) - list of nodes.
            - *args* -  currently unused.
            - *kw* - currently unused.
    
          **Returns**:
    
           - Tuple of strings, which represent paths relative to current working
             directory (for given environment).
        """
        import os
        import SCons.Node.FS
        rpaths = ()
        cwd = self.env.fs.getcwd().get_abspath()
        for node in nodes:
            rpath = None
            if isinstance(node, SCons.Node.FS.Base):
                rpath = os.path.relpath(node.get_abspath(), cwd)
            # FIXME: Other types possible here?
            if rpath is not None:
                rpaths += (rpath,)
        return rpaths


#############################################################################

#############################################################################
def _init_po_files(target, source, env):
    """ Action function for `POInit` builder. """
    nop = lambda target, source, env: 0
    if 'POAUTOINIT' in env:
        autoinit = env['POAUTOINIT']
    else:
        autoinit = False
    # Well, if everything outside works well, this loop should do single
    # iteration. Otherwise we are rebuilding all the targets even, if just
    # one has changed (but is this our fault?).
    for tgt in target:
        if not tgt.exists():
            if autoinit:
                action = SCons.Action.Action('$MSGINITCOM', '$MSGINITCOMSTR')
            else:
                msg = 'File ' + repr(str(tgt)) + ' does not exist. ' \
                      + 'If you are a translator, you can create it through: \n' \
                      + '$MSGINITCOM'
                action = SCons.Action.Action(nop, msg)
            status = action([tgt], source, env)
            if status: return status
    return 0


#############################################################################

#############################################################################
def _detect_xgettext(env):
    """ Detects *xgettext(1)* binary """
    if 'XGETTEXT' in env:
        return env['XGETTEXT']
    xgettext = env.Detect('xgettext');
    if xgettext:
        return xgettext
    raise SCons.Errors.StopError(XgettextNotFound, "Could not detect xgettext")
    return None


#############################################################################
def _xgettext_exists(env):
    return _detect_xgettext(env)


#############################################################################

#############################################################################
def _detect_msginit(env):
    """ Detects *msginit(1)* program. """
    if 'MSGINIT' in env:
        return env['MSGINIT']
    msginit = env.Detect('msginit');
    if msginit:
        return msginit
    raise SCons.Errors.StopError(MsginitNotFound, "Could not detect msginit")
    return None


#############################################################################
def _msginit_exists(env):
    return _detect_msginit(env)


#############################################################################

#############################################################################
def _detect_msgmerge(env):
    """ Detects *msgmerge(1)* program. """
    if 'MSGMERGE' in env:
        return env['MSGMERGE']
    msgmerge = env.Detect('msgmerge');
    if msgmerge:
        return msgmerge
    raise SCons.Errors.StopError(MsgmergeNotFound, "Could not detect msgmerge")
    return None


#############################################################################
def _msgmerge_exists(env):
    return _detect_msgmerge(env)


#############################################################################

#############################################################################
def _detect_msgfmt(env):
    """ Detects *msgmfmt(1)* program. """
    if 'MSGFMT' in env:
        return env['MSGFMT']
    msgfmt = env.Detect('msgfmt');
    if msgfmt:
        return msgfmt
    raise SCons.Errors.StopError(MsgfmtNotFound, "Could not detect msgfmt")
    return None


#############################################################################
def _msgfmt_exists(env):
    return _detect_msgfmt(env)


#############################################################################

#############################################################################
def tool_list(platform, env):
    """ List tools that shall be generated by top-level `gettext` tool """
    return ['xgettext', 'msginit', 'msgmerge', 'msgfmt']

#############################################################################
