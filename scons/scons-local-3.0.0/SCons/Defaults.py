"""SCons.Defaults

Builders and other things for the local site.  Here's where we'll
duplicate the functionality of autoconf until we move it into the
installation procedure or use something like qmconf.

The code that reads the registry to find MSVC components was borrowed
from distutils.msvccompiler.

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
from __future__ import division

__revision__ = "src/engine/SCons/Defaults.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"


import os
import errno
import shutil
import stat
import time
import sys

import SCons.Action
import SCons.Builder
import SCons.CacheDir
import SCons.Environment
import SCons.PathList
import SCons.Subst
import SCons.Tool

# A placeholder for a default Environment (for fetching source files
# from source code management systems and the like).  This must be
# initialized later, after the top-level directory is set by the calling
# interface.
_default_env = None

# Lazily instantiate the default environment so the overhead of creating
# it doesn't apply when it's not needed.
def _fetch_DefaultEnvironment(*args, **kw):
    """
    Returns the already-created default construction environment.
    """
    global _default_env
    return _default_env

def DefaultEnvironment(*args, **kw):
    """
    Initial public entry point for creating the default construction
    Environment.

    After creating the environment, we overwrite our name
    (DefaultEnvironment) with the _fetch_DefaultEnvironment() function,
    which more efficiently returns the initialized default construction
    environment without checking for its existence.

    (This function still exists with its _default_check because someone
    else (*cough* Script/__init__.py *cough*) may keep a reference
    to this function.  So we can't use the fully functional idiom of
    having the name originally be a something that *only* creates the
    construction environment and then overwrites the name.)
    """
    global _default_env
    if not _default_env:
        import SCons.Util
        _default_env = SCons.Environment.Environment(*args, **kw)
        if SCons.Util.md5:
            _default_env.Decider('MD5')
        else:
            _default_env.Decider('timestamp-match')
        global DefaultEnvironment
        DefaultEnvironment = _fetch_DefaultEnvironment
        _default_env._CacheDir_path = None
    return _default_env

# Emitters for setting the shared attribute on object files,
# and an action for checking that all of the source files
# going into a shared library are, in fact, shared.
def StaticObjectEmitter(target, source, env):
    for tgt in target:
        tgt.attributes.shared = None
    return (target, source)

def SharedObjectEmitter(target, source, env):
    for tgt in target:
        tgt.attributes.shared = 1
    return (target, source)

def SharedFlagChecker(source, target, env):
    same = env.subst('$STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME')
    if same == '0' or same == '' or same == 'False':
        for src in source:
            try:
                shared = src.attributes.shared
            except AttributeError:
                shared = None
            if not shared:
                raise SCons.Errors.UserError("Source file: %s is static and is not compatible with shared target: %s" % (src, target[0]))

SharedCheck = SCons.Action.Action(SharedFlagChecker, None)

# Some people were using these variable name before we made
# SourceFileScanner part of the public interface.  Don't break their
# SConscript files until we've given them some fair warning and a
# transition period.
CScan = SCons.Tool.CScanner
DScan = SCons.Tool.DScanner
LaTeXScan = SCons.Tool.LaTeXScanner
ObjSourceScan = SCons.Tool.SourceFileScanner
ProgScan = SCons.Tool.ProgramScanner

# These aren't really tool scanners, so they don't quite belong with
# the rest of those in Tool/__init__.py, but I'm not sure where else
# they should go.  Leave them here for now.
import SCons.Scanner.Dir
DirScanner = SCons.Scanner.Dir.DirScanner()
DirEntryScanner = SCons.Scanner.Dir.DirEntryScanner()

# Actions for common languages.
CAction = SCons.Action.Action("$CCCOM", "$CCCOMSTR")
ShCAction = SCons.Action.Action("$SHCCCOM", "$SHCCCOMSTR")
CXXAction = SCons.Action.Action("$CXXCOM", "$CXXCOMSTR")
ShCXXAction = SCons.Action.Action("$SHCXXCOM", "$SHCXXCOMSTR")

DAction = SCons.Action.Action("$DCOM", "$DCOMSTR")
ShDAction = SCons.Action.Action("$SHDCOM", "$SHDCOMSTR")

ASAction = SCons.Action.Action("$ASCOM", "$ASCOMSTR")
ASPPAction = SCons.Action.Action("$ASPPCOM", "$ASPPCOMSTR")

LinkAction = SCons.Action.Action("$LINKCOM", "$LINKCOMSTR")
ShLinkAction = SCons.Action.Action("$SHLINKCOM", "$SHLINKCOMSTR")

LdModuleLinkAction = SCons.Action.Action("$LDMODULECOM", "$LDMODULECOMSTR")

# Common tasks that we allow users to perform in platform-independent
# ways by creating ActionFactory instances.
ActionFactory = SCons.Action.ActionFactory

def get_paths_str(dest):
    # If dest is a list, we need to manually call str() on each element
    if SCons.Util.is_List(dest):
        elem_strs = []
        for element in dest:
            elem_strs.append('"' + str(element) + '"')
        return '[' + ', '.join(elem_strs) + ']'
    else:
        return '"' + str(dest) + '"'

permission_dic = {
    'u':{
        'r':stat.S_IRUSR,
        'w':stat.S_IWUSR,
        'x':stat.S_IXUSR
    },
    'g':{
        'r':stat.S_IRGRP,
        'w':stat.S_IWGRP,
        'x':stat.S_IXGRP
    },
    'o':{
        'r':stat.S_IROTH,
        'w':stat.S_IWOTH,
        'x':stat.S_IXOTH
    }
}

def chmod_func(dest, mode):
    import SCons.Util
    from string import digits
    SCons.Node.FS.invalidate_node_memos(dest)
    if not SCons.Util.is_List(dest):
        dest = [dest]
    if SCons.Util.is_String(mode) and not 0 in [i in digits for i in mode]:
        mode = int(mode, 8)
    if not SCons.Util.is_String(mode):
        for element in dest:
            os.chmod(str(element), mode)
    else:
        mode = str(mode)
        for operation in mode.split(","):
            if "=" in operation:
                operator = "="
            elif "+" in operation:
                operator = "+"
            elif "-" in operation:
                operator = "-"
            else:
                raise SyntaxError("Could not find +, - or =")
            operation_list = operation.split(operator)
            if len(operation_list) is not 2:
                raise SyntaxError("More than one operator found")
            user = operation_list[0].strip().replace("a", "ugo")
            permission = operation_list[1].strip()
            new_perm = 0
            for u in user:
                for p in permission:
                    try:
                        new_perm = new_perm | permission_dic[u][p]
                    except KeyError:
                        raise SyntaxError("Unrecognized user or permission format")
            for element in dest:
                curr_perm = os.stat(str(element)).st_mode
                if operator == "=":
                    os.chmod(str(element), new_perm)
                elif operator == "+":
                    os.chmod(str(element), curr_perm | new_perm)
                elif operator == "-":
                    os.chmod(str(element), curr_perm & ~new_perm)

def chmod_strfunc(dest, mode):
    import SCons.Util
    if not SCons.Util.is_String(mode):
        return 'Chmod(%s, 0%o)' % (get_paths_str(dest), mode)
    else:
        return 'Chmod(%s, "%s")' % (get_paths_str(dest), str(mode))

Chmod = ActionFactory(chmod_func, chmod_strfunc)

def copy_func(dest, src, symlinks=True):
    """
    If symlinks (is true), then a symbolic link will be
    shallow copied and recreated as a symbolic link; otherwise, copying
    a symbolic link will be equivalent to copying the symbolic link's
    final target regardless of symbolic link depth.
    """

    dest = str(dest)
    src = str(src)

    SCons.Node.FS.invalidate_node_memos(dest)
    if SCons.Util.is_List(src) and os.path.isdir(dest):
        for file in src:
            shutil.copy2(file, dest)
        return 0
    elif os.path.islink(src):
        if symlinks:
            return os.symlink(os.readlink(src), dest)
        else:
            return copy_func(dest, os.path.realpath(src))
    elif os.path.isfile(src):
        shutil.copy2(src, dest)
        return 0
    else:
        shutil.copytree(src, dest, symlinks)
        # copytree returns None in python2 and destination string in python3
        # A error is raised in both cases, so we can just return 0 for success
        return 0

Copy = ActionFactory(
    copy_func,
    lambda dest, src, symlinks=True: 'Copy("%s", "%s")' % (dest, src)
)

def delete_func(dest, must_exist=0):
    SCons.Node.FS.invalidate_node_memos(dest)
    if not SCons.Util.is_List(dest):
        dest = [dest]
    for entry in dest:
        entry = str(entry)
        # os.path.exists returns False with broken links that exist
        entry_exists = os.path.exists(entry) or os.path.islink(entry)
        if not entry_exists and not must_exist:
            continue
        # os.path.isdir returns True when entry is a link to a dir
        if os.path.isdir(entry) and not os.path.islink(entry):
            shutil.rmtree(entry, 1)
            continue
        os.unlink(entry)

def delete_strfunc(dest, must_exist=0):
    return 'Delete(%s)' % get_paths_str(dest)

Delete = ActionFactory(delete_func, delete_strfunc)

def mkdir_func(dest):
    SCons.Node.FS.invalidate_node_memos(dest)
    if not SCons.Util.is_List(dest):
        dest = [dest]
    for entry in dest:
        try:
            os.makedirs(str(entry))
        except os.error as e:
            p = str(entry)
            if (e.args[0] == errno.EEXIST or
                    (sys.platform=='win32' and e.args[0]==183)) \
                    and os.path.isdir(str(entry)):
                pass            # not an error if already exists
            else:
                raise

Mkdir = ActionFactory(mkdir_func,
                      lambda dir: 'Mkdir(%s)' % get_paths_str(dir))

def move_func(dest, src):
    SCons.Node.FS.invalidate_node_memos(dest)
    SCons.Node.FS.invalidate_node_memos(src)
    shutil.move(src, dest)

Move = ActionFactory(move_func,
                     lambda dest, src: 'Move("%s", "%s")' % (dest, src),
                     convert=str)

def touch_func(dest):
    SCons.Node.FS.invalidate_node_memos(dest)
    if not SCons.Util.is_List(dest):
        dest = [dest]
    for file in dest:
        file = str(file)
        mtime = int(time.time())
        if os.path.exists(file):
            atime = os.path.getatime(file)
        else:
            open(file, 'w')
            atime = mtime
        os.utime(file, (atime, mtime))

Touch = ActionFactory(touch_func,
                      lambda file: 'Touch(%s)' % get_paths_str(file))

# Internal utility functions

def _concat(prefix, list, suffix, env, f=lambda x: x, target=None, source=None):
    """
    Creates a new list from 'list' by first interpolating each element
    in the list using the 'env' dictionary and then calling f on the
    list, and finally calling _concat_ixes to concatenate 'prefix' and
    'suffix' onto each element of the list.
    """
    if not list:
        return list

    l = f(SCons.PathList.PathList(list).subst_path(env, target, source))
    if l is not None:
        list = l

    return _concat_ixes(prefix, list, suffix, env)

def _concat_ixes(prefix, list, suffix, env):
    """
    Creates a new list from 'list' by concatenating the 'prefix' and
    'suffix' arguments onto each element of the list.  A trailing space
    on 'prefix' or leading space on 'suffix' will cause them to be put
    into separate list elements rather than being concatenated.
    """

    result = []

    # ensure that prefix and suffix are strings
    prefix = str(env.subst(prefix, SCons.Subst.SUBST_RAW))
    suffix = str(env.subst(suffix, SCons.Subst.SUBST_RAW))

    for x in list:
        if isinstance(x, SCons.Node.FS.File):
            result.append(x)
            continue
        x = str(x)
        if x:

            if prefix:
                if prefix[-1] == ' ':
                    result.append(prefix[:-1])
                elif x[:len(prefix)] != prefix:
                    x = prefix + x

            result.append(x)

            if suffix:
                if suffix[0] == ' ':
                    result.append(suffix[1:])
                elif x[-len(suffix):] != suffix:
                    result[-1] = result[-1]+suffix

    return result

def _stripixes(prefix, itms, suffix, stripprefixes, stripsuffixes, env, c=None):
    """
    This is a wrapper around _concat()/_concat_ixes() that checks for
    the existence of prefixes or suffixes on list items and strips them
    where it finds them.  This is used by tools (like the GNU linker)
    that need to turn something like 'libfoo.a' into '-lfoo'.
    """

    if not itms:
        return itms

    if not callable(c):
        env_c = env['_concat']
        if env_c != _concat and callable(env_c):
            # There's a custom _concat() method in the construction
            # environment, and we've allowed people to set that in
            # the past (see test/custom-concat.py), so preserve the
            # backwards compatibility.
            c = env_c
        else:
            c = _concat_ixes

    stripprefixes = list(map(env.subst, SCons.Util.flatten(stripprefixes)))
    stripsuffixes = list(map(env.subst, SCons.Util.flatten(stripsuffixes)))

    stripped = []
    for l in SCons.PathList.PathList(itms).subst_path(env, None, None):
        if isinstance(l, SCons.Node.FS.File):
            stripped.append(l)
            continue

        if not SCons.Util.is_String(l):
            l = str(l)

        for stripprefix in stripprefixes:
            lsp = len(stripprefix)
            if l[:lsp] == stripprefix:
                l = l[lsp:]
                # Do not strip more than one prefix
                break

        for stripsuffix in stripsuffixes:
            lss = len(stripsuffix)
            if l[-lss:] == stripsuffix:
                l = l[:-lss]
                # Do not strip more than one suffix
                break

        stripped.append(l)

    return c(prefix, stripped, suffix, env)

def processDefines(defs):
    """process defines, resolving strings, lists, dictionaries, into a list of
    strings
    """
    if SCons.Util.is_List(defs):
        l = []
        for d in defs:
            if d is None:
                continue
            elif SCons.Util.is_List(d) or isinstance(d, tuple):
                if len(d) >= 2:
                    l.append(str(d[0]) + '=' + str(d[1]))
                else:
                    l.append(str(d[0]))
            elif SCons.Util.is_Dict(d):
                for macro,value in d.items():
                    if value is not None:
                        l.append(str(macro) + '=' + str(value))
                    else:
                        l.append(str(macro))
            elif SCons.Util.is_String(d):
                l.append(str(d))
            else:
                raise SCons.Errors.UserError("DEFINE %s is not a list, dict, string or None."%repr(d))
    elif SCons.Util.is_Dict(defs):
        # The items in a dictionary are stored in random order, but
        # if the order of the command-line options changes from
        # invocation to invocation, then the signature of the command
        # line will change and we'll get random unnecessary rebuilds.
        # Consequently, we have to sort the keys to ensure a
        # consistent order...
        l = []
        for k,v in sorted(defs.items()):
            if v is None:
                l.append(str(k))
            else:
                l.append(str(k) + '=' + str(v))
    else:
        l = [str(defs)]
    return l


def _defines(prefix, defs, suffix, env, c=_concat_ixes):
    """A wrapper around _concat_ixes that turns a list or string
    into a list of C preprocessor command-line definitions.
    """

    return c(prefix, env.subst_path(processDefines(defs)), suffix, env)


class NullCmdGenerator(object):
    """This is a callable class that can be used in place of other
    command generators if you don't want them to do anything.

    The __call__ method for this class simply returns the thing
    you instantiated it with.

    Example usage:
    env["DO_NOTHING"] = NullCmdGenerator
    env["LINKCOM"] = "${DO_NOTHING('$LINK $SOURCES $TARGET')}"
    """

    def __init__(self, cmd):
        self.cmd = cmd

    def __call__(self, target, source, env, for_signature=None):
        return self.cmd


class Variable_Method_Caller(object):
    """A class for finding a construction variable on the stack and
    calling one of its methods.

    We use this to support "construction variables" in our string
    eval()s that actually stand in for methods--specifically, use
    of "RDirs" in call to _concat that should actually execute the
    "TARGET.RDirs" method.  (We used to support this by creating a little
    "build dictionary" that mapped RDirs to the method, but this got in
    the way of Memoizing construction environments, because we had to
    create new environment objects to hold the variables.)
    """
    def __init__(self, variable, method):
        self.variable = variable
        self.method = method
    def __call__(self, *args, **kw):
        try: 1//0
        except ZeroDivisionError:
            # Don't start iterating with the current stack-frame to
            # prevent creating reference cycles (f_back is safe).
            frame = sys.exc_info()[2].tb_frame.f_back
        variable = self.variable
        while frame:
            if variable in frame.f_locals:
                v = frame.f_locals[variable]
                if v:
                    method = getattr(v, self.method)
                    return method(*args, **kw)
            frame = frame.f_back
        return None

# if $version_var is not empty, returns env[flags_var], otherwise returns None
def __libversionflags(env, version_var, flags_var):
    try:
        if env.subst('$'+version_var):
            return env[flags_var]
    except KeyError:
        pass
    return None

ConstructionEnvironment = {
    'BUILDERS'      : {},
    'SCANNERS'      : [ SCons.Tool.SourceFileScanner ],
    'CONFIGUREDIR'  : '#/.sconf_temp',
    'CONFIGURELOG'  : '#/config.log',
    'CPPSUFFIXES'   : SCons.Tool.CSuffixes,
    'DSUFFIXES'     : SCons.Tool.DSuffixes,
    'ENV'           : {},
    'IDLSUFFIXES'   : SCons.Tool.IDLSuffixes,
#    'LATEXSUFFIXES' : SCons.Tool.LaTeXSuffixes, # moved to the TeX tools generate functions
    '_concat'       : _concat,
    '_defines'      : _defines,
    '_stripixes'    : _stripixes,
    '_LIBFLAGS'     : '${_concat(LIBLINKPREFIX, LIBS, LIBLINKSUFFIX, __env__)}',
    '_LIBDIRFLAGS'  : '$( ${_concat(LIBDIRPREFIX, LIBPATH, LIBDIRSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)',
    '_CPPINCFLAGS'  : '$( ${_concat(INCPREFIX, CPPPATH, INCSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)',
    '_CPPDEFFLAGS'  : '${_defines(CPPDEFPREFIX, CPPDEFINES, CPPDEFSUFFIX, __env__)}',

    '__libversionflags'      : __libversionflags,
    '__SHLIBVERSIONFLAGS'    : '${__libversionflags(__env__,"SHLIBVERSION","_SHLIBVERSIONFLAGS")}',
    '__LDMODULEVERSIONFLAGS' : '${__libversionflags(__env__,"LDMODULEVERSION","_LDMODULEVERSIONFLAGS")}',
    '__DSHLIBVERSIONFLAGS'   : '${__libversionflags(__env__,"DSHLIBVERSION","_DSHLIBVERSIONFLAGS")}',

    'TEMPFILE'      : NullCmdGenerator,
    'Dir'           : Variable_Method_Caller('TARGET', 'Dir'),
    'Dirs'          : Variable_Method_Caller('TARGET', 'Dirs'),
    'File'          : Variable_Method_Caller('TARGET', 'File'),
    'RDirs'         : Variable_Method_Caller('TARGET', 'RDirs'),
}

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
