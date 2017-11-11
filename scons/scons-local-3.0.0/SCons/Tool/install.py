"""SCons.Tool.install

Tool-specific initialization for the install tool.

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
from __future__ import print_function

__revision__ = "src/engine/SCons/Tool/install.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os
import re
import shutil
import stat

import SCons.Action
import SCons.Tool
import SCons.Util

#
# We keep track of *all* installed files.
_INSTALLED_FILES = []
_UNIQUE_INSTALLED_FILES = None

class CopytreeError(EnvironmentError):
    pass

# This is a patched version of shutil.copytree from python 2.5.  It
# doesn't fail if the dir exists, which regular copytree does
# (annoyingly).  Note the XXX comment in the docstring.
def scons_copytree(src, dst, symlinks=False):
    """Recursively copy a directory tree using copy2().

    The destination directory must not already exist.
    If exception(s) occur, an CopytreeError is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied.

    XXX Consider this example code rather than the ultimate tool.

    """
    names = os.listdir(src)
    # garyo@genarts.com fix: check for dir before making dirs.
    if not os.path.exists(dst):
        os.makedirs(dst)
    errors = []
    for name in names:
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                scons_copytree(srcname, dstname, symlinks)
            else:
                shutil.copy2(srcname, dstname)
            # XXX What about devices, sockets etc.?
        except (IOError, os.error) as why:
            errors.append((srcname, dstname, str(why)))
        # catch the CopytreeError from the recursive copytree so that we can
        # continue with other files
        except CopytreeError as err:
            errors.extend(err.args[0])
    try:
        shutil.copystat(src, dst)
    except SCons.Util.WinError:
        # can't copy file access times on Windows
        pass
    except OSError as why:
        errors.extend((src, dst, str(why)))
    if errors:
        raise CopytreeError(errors)


#
# Functions doing the actual work of the Install Builder.
#
def copyFunc(dest, source, env):
    """Install a source file or directory into a destination by copying,
    (including copying permission/mode bits)."""

    if os.path.isdir(source):
        if os.path.exists(dest):
            if not os.path.isdir(dest):
                raise SCons.Errors.UserError("cannot overwrite non-directory `%s' with a directory `%s'" % (str(dest), str(source)))
        else:
            parent = os.path.split(dest)[0]
            if not os.path.exists(parent):
                os.makedirs(parent)
        scons_copytree(source, dest)
    else:
        shutil.copy2(source, dest)
        st = os.stat(source)
        os.chmod(dest, stat.S_IMODE(st[stat.ST_MODE]) | stat.S_IWRITE)

    return 0

#
# Functions doing the actual work of the InstallVersionedLib Builder.
#
def copyFuncVersionedLib(dest, source, env):
    """Install a versioned library into a destination by copying,
    (including copying permission/mode bits) and then creating
    required symlinks."""

    if os.path.isdir(source):
        raise SCons.Errors.UserError("cannot install directory `%s' as a version library" % str(source) )
    else:
        # remove the link if it is already there
        try:
            os.remove(dest)
        except:
            pass
        shutil.copy2(source, dest)
        st = os.stat(source)
        os.chmod(dest, stat.S_IMODE(st[stat.ST_MODE]) | stat.S_IWRITE)
        installShlibLinks(dest, source, env)

    return 0

def listShlibLinksToInstall(dest, source, env):
    install_links = []
    source = env.arg2nodes(source)
    dest = env.fs.File(dest)
    install_dir = dest.get_dir()
    for src in source:
        symlinks = getattr(getattr(src,'attributes',None), 'shliblinks', None)
        if symlinks:
            for link, linktgt in symlinks:
                link_base = os.path.basename(link.get_path())
                linktgt_base  = os.path.basename(linktgt.get_path())
                install_link = env.fs.File(link_base, install_dir)
                install_linktgt = env.fs.File(linktgt_base, install_dir)
                install_links.append((install_link, install_linktgt))
    return install_links

def installShlibLinks(dest, source, env):
    """If we are installing a versioned shared library create the required links."""
    Verbose = False
    symlinks = listShlibLinksToInstall(dest, source, env)
    if Verbose:
        print('installShlibLinks: symlinks={:r}'.format(SCons.Tool.StringizeLibSymlinks(symlinks)))
    if symlinks:
        SCons.Tool.CreateLibSymlinks(env, symlinks)
    return

def installFunc(target, source, env):
    """Install a source file into a target using the function specified
    as the INSTALL construction variable."""
    try:
        install = env['INSTALL']
    except KeyError:
        raise SCons.Errors.UserError('Missing INSTALL construction variable.')

    assert len(target)==len(source), \
           "Installing source %s into target %s: target and source lists must have same length."%(list(map(str, source)), list(map(str, target)))
    for t,s in zip(target,source):
        if install(t.get_path(),s.get_path(),env):
            return 1

    return 0

def installFuncVersionedLib(target, source, env):
    """Install a versioned library into a target using the function specified
    as the INSTALLVERSIONEDLIB construction variable."""
    try:
        install = env['INSTALLVERSIONEDLIB']
    except KeyError:
        raise SCons.Errors.UserError('Missing INSTALLVERSIONEDLIB construction variable.')

    assert len(target)==len(source), \
           "Installing source %s into target %s: target and source lists must have same length."%(list(map(str, source)), list(map(str, target)))
    for t,s in zip(target,source):
        if hasattr(t.attributes, 'shlibname'):
            tpath = os.path.join(t.get_dir(), t.attributes.shlibname)
        else:
            tpath = t.get_path()
        if install(tpath,s.get_path(),env):
            return 1

    return 0

def stringFunc(target, source, env):
    installstr = env.get('INSTALLSTR')
    if installstr:
        return env.subst_target_source(installstr, 0, target, source)
    target = str(target[0])
    source = str(source[0])
    if os.path.isdir(source):
        type = 'directory'
    else:
        type = 'file'
    return 'Install %s: "%s" as "%s"' % (type, source, target)

#
# Emitter functions
#
def add_targets_to_INSTALLED_FILES(target, source, env):
    """ An emitter that adds all target files to the list stored in the
    _INSTALLED_FILES global variable. This way all installed files of one
    scons call will be collected.
    """
    global _INSTALLED_FILES, _UNIQUE_INSTALLED_FILES
    _INSTALLED_FILES.extend(target)

    _UNIQUE_INSTALLED_FILES = None
    return (target, source)

def add_versioned_targets_to_INSTALLED_FILES(target, source, env):
    """ An emitter that adds all target files to the list stored in the
    _INSTALLED_FILES global variable. This way all installed files of one
    scons call will be collected.
    """
    global _INSTALLED_FILES, _UNIQUE_INSTALLED_FILES
    Verbose = False
    _INSTALLED_FILES.extend(target)
    if Verbose:
        print("add_versioned_targets_to_INSTALLED_FILES: target={:r}".format(list(map(str, target))))
    symlinks = listShlibLinksToInstall(target[0], source, env)
    if symlinks:
        SCons.Tool.EmitLibSymlinks(env, symlinks, target[0])
    _UNIQUE_INSTALLED_FILES = None
    return (target, source)

class DESTDIR_factory(object):
    """ A node factory, where all files will be relative to the dir supplied
    in the constructor.
    """
    def __init__(self, env, dir):
        self.env = env
        self.dir = env.arg2nodes( dir, env.fs.Dir )[0]

    def Entry(self, name):
        name = SCons.Util.make_path_relative(name)
        return self.dir.Entry(name)

    def Dir(self, name):
        name = SCons.Util.make_path_relative(name)
        return self.dir.Dir(name)

#
# The Builder Definition
#
install_action       = SCons.Action.Action(installFunc, stringFunc)
installas_action     = SCons.Action.Action(installFunc, stringFunc)
installVerLib_action = SCons.Action.Action(installFuncVersionedLib, stringFunc)

BaseInstallBuilder               = None

def InstallBuilderWrapper(env, target=None, source=None, dir=None, **kw):
    if target and dir:
        import SCons.Errors
        raise SCons.Errors.UserError("Both target and dir defined for Install(), only one may be defined.")
    if not dir:
        dir=target

    import SCons.Script
    install_sandbox = SCons.Script.GetOption('install_sandbox')
    if install_sandbox:
        target_factory = DESTDIR_factory(env, install_sandbox)
    else:
        target_factory = env.fs

    try:
        dnodes = env.arg2nodes(dir, target_factory.Dir)
    except TypeError:
        raise SCons.Errors.UserError("Target `%s' of Install() is a file, but should be a directory.  Perhaps you have the Install() arguments backwards?" % str(dir))
    sources = env.arg2nodes(source, env.fs.Entry)
    tgt = []
    for dnode in dnodes:
        for src in sources:
            # Prepend './' so the lookup doesn't interpret an initial
            # '#' on the file name portion as meaning the Node should
            # be relative to the top-level SConstruct directory.
            target = env.fs.Entry('.'+os.sep+src.name, dnode)
            tgt.extend(BaseInstallBuilder(env, target, src, **kw))
    return tgt


def InstallAsBuilderWrapper(env, target=None, source=None, **kw):
    result = []
    for src, tgt in map(lambda x, y: (x, y), source, target):
        result.extend(BaseInstallBuilder(env, tgt, src, **kw))
    return result

BaseVersionedInstallBuilder = None


def InstallVersionedBuilderWrapper(env, target=None, source=None, dir=None, **kw):
    if target and dir:
        import SCons.Errors
        raise SCons.Errors.UserError("Both target and dir defined for Install(), only one may be defined.")
    if not dir:
        dir=target

    import SCons.Script
    install_sandbox = SCons.Script.GetOption('install_sandbox')
    if install_sandbox:
        target_factory = DESTDIR_factory(env, install_sandbox)
    else:
        target_factory = env.fs

    try:
        dnodes = env.arg2nodes(dir, target_factory.Dir)
    except TypeError:
        raise SCons.Errors.UserError("Target `%s' of Install() is a file, but should be a directory.  Perhaps you have the Install() arguments backwards?" % str(dir))
    sources = env.arg2nodes(source, env.fs.Entry)
    tgt = []
    for dnode in dnodes:
        for src in sources:
            # Prepend './' so the lookup doesn't interpret an initial
            # '#' on the file name portion as meaning the Node should
            # be relative to the top-level SConstruct directory.
            target = env.fs.Entry('.'+os.sep+src.name, dnode)
            tgt.extend(BaseVersionedInstallBuilder(env, target, src, **kw))
    return tgt

added = None


def generate(env):

    from SCons.Script import AddOption, GetOption
    global added
    if not added:
        added = 1
        AddOption('--install-sandbox',
                  dest='install_sandbox',
                  type="string",
                  action="store",
                  help='A directory under which all installed files will be placed.')

    global BaseInstallBuilder
    if BaseInstallBuilder is None:
        install_sandbox = GetOption('install_sandbox')
        if install_sandbox:
            target_factory = DESTDIR_factory(env, install_sandbox)
        else:
            target_factory = env.fs

        BaseInstallBuilder = SCons.Builder.Builder(
                              action         = install_action,
                              target_factory = target_factory.Entry,
                              source_factory = env.fs.Entry,
                              multi          = 1,
                              emitter        = [ add_targets_to_INSTALLED_FILES, ],
                              source_scanner = SCons.Scanner.Base( {}, name = 'Install', recursive = False ),
                              name           = 'InstallBuilder')

    global BaseVersionedInstallBuilder
    if BaseVersionedInstallBuilder is None:
        install_sandbox = GetOption('install_sandbox')
        if install_sandbox:
            target_factory = DESTDIR_factory(env, install_sandbox)
        else:
            target_factory = env.fs

        BaseVersionedInstallBuilder = SCons.Builder.Builder(
                                       action         = installVerLib_action,
                                       target_factory = target_factory.Entry,
                                       source_factory = env.fs.Entry,
                                       multi          = 1,
                                       emitter        = [ add_versioned_targets_to_INSTALLED_FILES, ],
                                       name           = 'InstallVersionedBuilder')

    env['BUILDERS']['_InternalInstall'] = InstallBuilderWrapper
    env['BUILDERS']['_InternalInstallAs'] = InstallAsBuilderWrapper
    env['BUILDERS']['_InternalInstallVersionedLib'] = InstallVersionedBuilderWrapper

    # We'd like to initialize this doing something like the following,
    # but there isn't yet support for a ${SOURCE.type} expansion that
    # will print "file" or "directory" depending on what's being
    # installed.  For now we punt by not initializing it, and letting
    # the stringFunc() that we put in the action fall back to the
    # hand-crafted default string if it's not set.
    #
    #try:
    #    env['INSTALLSTR']
    #except KeyError:
    #    env['INSTALLSTR'] = 'Install ${SOURCE.type}: "$SOURCES" as "$TARGETS"'

    try:
        env['INSTALL']
    except KeyError:
        env['INSTALL']    = copyFunc

    try:
        env['INSTALLVERSIONEDLIB']
    except KeyError:
        env['INSTALLVERSIONEDLIB']    = copyFuncVersionedLib

def exists(env):
    return 1

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
