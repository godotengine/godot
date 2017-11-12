""" xgettext tool

Tool specific initialization of `xgettext` tool.
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

__revision__ = "src/engine/SCons/Tool/xgettext.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"


#############################################################################
class _CmdRunner(object):
    """ Callable object, which runs shell command storing its stdout and stderr to
    variables. It also provides `strfunction()` method, which shall be used by
    scons Action objects to print command string. """

    def __init__(self, command, commandstr=None):
        self.out = None
        self.err = None
        self.status = None
        self.command = command
        self.commandstr = commandstr

    def __call__(self, target, source, env):
        import SCons.Action
        import subprocess
        import os
        import sys
        kw = {
            'stdin': 'devnull',
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'universal_newlines': True,
            'shell': True
        }
        command = env.subst(self.command, target=target, source=source)
        proc = SCons.Action._subproc(env, command, **kw)
        self.out, self.err = proc.communicate()
        self.status = proc.wait()
        if self.err:
            sys.stderr.write(unicode(self.err))
        return self.status

    def strfunction(self, target, source, env):
        import os
        comstr = self.commandstr
        if env.subst(comstr, target=target, source=source) == "":
            comstr = self.command
        s = env.subst(comstr, target=target, source=source)
        return s


#############################################################################

#############################################################################
def _update_pot_file(target, source, env):
    """ Action function for `POTUpdate` builder """
    import re
    import os
    import SCons.Action
    nop = lambda target, source, env: 0

    # Save scons cwd and os cwd (NOTE: they may be different. After the job, we
    # revert each one to its original state).
    save_cwd = env.fs.getcwd()
    save_os_cwd = os.getcwd()
    chdir = target[0].dir
    chdir_str = repr(chdir.get_abspath())
    # Print chdir message (employ SCons.Action.Action for that. It knows better
    # than me how to to this correctly).
    env.Execute(SCons.Action.Action(nop, "Entering " + chdir_str))
    # Go to target's directory and do our job
    env.fs.chdir(chdir, 1)  # Go into target's directory
    try:
        cmd = _CmdRunner('$XGETTEXTCOM', '$XGETTEXTCOMSTR')
        action = SCons.Action.Action(cmd, strfunction=cmd.strfunction)
        status = action([target[0]], source, env)
    except:
        # Something went wrong.
        env.Execute(SCons.Action.Action(nop, "Leaving " + chdir_str))
        # Revert working dirs to previous state and re-throw exception.
        env.fs.chdir(save_cwd, 0)
        os.chdir(save_os_cwd)
        raise
    # Print chdir message.
    env.Execute(SCons.Action.Action(nop, "Leaving " + chdir_str))
    # Revert working dirs to previous state.
    env.fs.chdir(save_cwd, 0)
    os.chdir(save_os_cwd)
    # If the command was not successfull, return error code.
    if status: return status

    new_content = cmd.out

    if not new_content:
        # When xgettext finds no internationalized messages, no *.pot is created
        # (because we don't want to bother translators with empty POT files).
        needs_update = False
        explain = "no internationalized messages encountered"
    else:
        if target[0].exists():
            # If the file already exists, it's left unaltered unless its messages
            # are outdated (w.r.t. to these recovered by xgettext from sources).
            old_content = target[0].get_text_contents()
            re_cdate = re.compile(r'^"POT-Creation-Date: .*"$[\r\n]?', re.M)
            old_content_nocdate = re.sub(re_cdate, "", old_content)
            new_content_nocdate = re.sub(re_cdate, "", new_content)
            if (old_content_nocdate == new_content_nocdate):
                # Messages are up-to-date
                needs_update = False
                explain = "messages in file found to be up-to-date"
            else:
                # Messages are outdated
                needs_update = True
                explain = "messages in file were outdated"
        else:
            # No POT file found, create new one
            needs_update = True
            explain = "new file"
    if needs_update:
        # Print message employing SCons.Action.Action for that.
        msg = "Writing " + repr(str(target[0])) + " (" + explain + ")"
        env.Execute(SCons.Action.Action(nop, msg))
        f = open(str(target[0]), "w")
        f.write(new_content)
        f.close()
        return 0
    else:
        # Print message employing SCons.Action.Action for that.
        msg = "Not writing " + repr(str(target[0])) + " (" + explain + ")"
        env.Execute(SCons.Action.Action(nop, msg))
        return 0


#############################################################################

#############################################################################
from SCons.Builder import BuilderBase


#############################################################################
class _POTBuilder(BuilderBase):
    def _execute(self, env, target, source, *args):
        if not target:
            if 'POTDOMAIN' in env and env['POTDOMAIN']:
                domain = env['POTDOMAIN']
            else:
                domain = 'messages'
            target = [domain]
        return BuilderBase._execute(self, env, target, source, *args)


#############################################################################

#############################################################################
def _scan_xgettext_from_files(target, source, env, files=None, path=None):
    """ Parses `POTFILES.in`-like file and returns list of extracted file names.
    """
    import re
    import SCons.Util
    import SCons.Node.FS

    if files is None:
        return 0
    if not SCons.Util.is_List(files):
        files = [files]

    if path is None:
        if 'XGETTEXTPATH' in env:
            path = env['XGETTEXTPATH']
        else:
            path = []
    if not SCons.Util.is_List(path):
        path = [path]

    path = SCons.Util.flatten(path)

    dirs = ()
    for p in path:
        if not isinstance(p, SCons.Node.FS.Base):
            if SCons.Util.is_String(p):
                p = env.subst(p, source=source, target=target)
            p = env.arg2nodes(p, env.fs.Dir)
        dirs += tuple(p)
    # cwd is the default search path (when no path is defined by user)
    if not dirs:
        dirs = (env.fs.getcwd(),)

    # Parse 'POTFILE.in' files.
    re_comment = re.compile(r'^#[^\n\r]*$\r?\n?', re.M)
    re_emptyln = re.compile(r'^[ \t\r]*$\r?\n?', re.M)
    re_trailws = re.compile(r'[ \t\r]+$')
    for f in files:
        # Find files in search path $XGETTEXTPATH
        if isinstance(f, SCons.Node.FS.Base) and f.rexists():
            contents = f.get_text_contents()
            contents = re_comment.sub("", contents)
            contents = re_emptyln.sub("", contents)
            contents = re_trailws.sub("", contents)
            depnames = contents.splitlines()
            for depname in depnames:
                depfile = SCons.Node.FS.find_file(depname, dirs)
                if not depfile:
                    depfile = env.arg2nodes(depname, dirs[0].File)
                env.Depends(target, depfile)
    return 0


#############################################################################

#############################################################################
def _pot_update_emitter(target, source, env):
    """ Emitter function for `POTUpdate` builder """
    from SCons.Tool.GettextCommon import _POTargetFactory
    import SCons.Util
    import SCons.Node.FS

    if 'XGETTEXTFROM' in env:
        xfrom = env['XGETTEXTFROM']
    else:
        return target, source
    if not SCons.Util.is_List(xfrom):
        xfrom = [xfrom]

    xfrom = SCons.Util.flatten(xfrom)

    # Prepare list of 'POTFILE.in' files.
    files = []
    for xf in xfrom:
        if not isinstance(xf, SCons.Node.FS.Base):
            if SCons.Util.is_String(xf):
                # Interpolate variables in strings
                xf = env.subst(xf, source=source, target=target)
            xf = env.arg2nodes(xf)
        files.extend(xf)
    if files:
        env.Depends(target, files)
        _scan_xgettext_from_files(target, source, env, files)
    return target, source


#############################################################################

#############################################################################
from SCons.Environment import _null


#############################################################################
def _POTUpdateBuilderWrapper(env, target=None, source=_null, **kw):
    return env._POTUpdateBuilder(target, source, **kw)


#############################################################################

#############################################################################
def _POTUpdateBuilder(env, **kw):
    """ Creates `POTUpdate` builder object """
    import SCons.Action
    from SCons.Tool.GettextCommon import _POTargetFactory
    kw['action'] = SCons.Action.Action(_update_pot_file, None)
    kw['suffix'] = '$POTSUFFIX'
    kw['target_factory'] = _POTargetFactory(env, alias='$POTUPDATE_ALIAS').File
    kw['emitter'] = _pot_update_emitter
    return _POTBuilder(**kw)


#############################################################################

#############################################################################
def generate(env, **kw):
    """ Generate `xgettext` tool """
    import SCons.Util
    from SCons.Tool.GettextCommon import RPaths, _detect_xgettext

    try:
        env['XGETTEXT'] = _detect_xgettext(env)
    except:
        env['XGETTEXT'] = 'xgettext'
    # NOTE: sources="$SOURCES" would work as well. However, we use following
    # construction to convert absolute paths provided by scons onto paths
    # relative to current working dir. Note, that scons expands $SOURCE(S) to
    # absolute paths for sources $SOURCE(s) outside of current subtree (e.g. in
    # "../"). With source=$SOURCE these absolute paths would be written to the
    # resultant *.pot file (and its derived *.po files) as references to lines in
    # source code (e.g. referring lines in *.c files). Such references would be
    # correct (e.g. in poedit) only on machine on which *.pot was generated and
    # would be of no use on other hosts (having a copy of source code located
    # in different place in filesystem).
    sources = '$( ${_concat( "", SOURCES, "", __env__, XgettextRPaths, TARGET' \
              + ', SOURCES)} $)'

    # NOTE: the output from $XGETTEXTCOM command must go to stdout, not to a file.
    # This is required by the POTUpdate builder's action.
    xgettextcom = '$XGETTEXT $XGETTEXTFLAGS $_XGETTEXTPATHFLAGS' \
                  + ' $_XGETTEXTFROMFLAGS -o - ' + sources

    xgettextpathflags = '$( ${_concat( XGETTEXTPATHPREFIX, XGETTEXTPATH' \
                        + ', XGETTEXTPATHSUFFIX, __env__, RDirs, TARGET, SOURCES)} $)'
    xgettextfromflags = '$( ${_concat( XGETTEXTFROMPREFIX, XGETTEXTFROM' \
                        + ', XGETTEXTFROMSUFFIX, __env__, target=TARGET, source=SOURCES)} $)'

    env.SetDefault(
        _XGETTEXTDOMAIN='${TARGET.filebase}',
        XGETTEXTFLAGS=[],
        XGETTEXTCOM=xgettextcom,
        XGETTEXTCOMSTR='',
        XGETTEXTPATH=[],
        XGETTEXTPATHPREFIX='-D',
        XGETTEXTPATHSUFFIX='',
        XGETTEXTFROM=None,
        XGETTEXTFROMPREFIX='-f',
        XGETTEXTFROMSUFFIX='',
        _XGETTEXTPATHFLAGS=xgettextpathflags,
        _XGETTEXTFROMFLAGS=xgettextfromflags,
        POTSUFFIX=['.pot'],
        POTUPDATE_ALIAS='pot-update',
        XgettextRPaths=RPaths(env)
    )
    env.Append(BUILDERS={
        '_POTUpdateBuilder': _POTUpdateBuilder(env)
    })
    env.AddMethod(_POTUpdateBuilderWrapper, 'POTUpdate')
    env.AlwaysBuild(env.Alias('$POTUPDATE_ALIAS'))


#############################################################################

#############################################################################
def exists(env):
    """ Check, whether the tool exists """
    from SCons.Tool.GettextCommon import _xgettext_exists
    try:
        return _xgettext_exists(env)
    except:
        return False

#############################################################################

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
