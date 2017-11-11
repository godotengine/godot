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

__revision__ = "src/engine/SCons/Tool/MSCommon/vs.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

__doc__ = """Module to detect Visual Studio and/or Visual C/C++
"""

import os

import SCons.Errors
import SCons.Util

from .common import debug, \
                   get_output, \
                   is_win64, \
                   normalize_env, \
                   parse_output, \
                   read_reg

import SCons.Tool.MSCommon.vc

class VisualStudio(object):
    """
    An abstract base class for trying to find installed versions of
    Visual Studio.
    """
    def __init__(self, version, **kw):
        self.version = version
        kw['vc_version']  = kw.get('vc_version', version)
        kw['sdk_version'] = kw.get('sdk_version', version)
        self.__dict__.update(kw)
        self._cache = {}

    def find_batch_file(self):
        vs_dir = self.get_vs_dir()
        if not vs_dir:
            debug('find_executable():  no vs_dir')
            return None
        batch_file = os.path.join(vs_dir, self.batch_file_path)
        batch_file = os.path.normpath(batch_file)
        if not os.path.isfile(batch_file):
            debug('find_batch_file():  %s not on file system' % batch_file)
            return None
        return batch_file

    def find_vs_dir_by_vc(self):
        SCons.Tool.MSCommon.vc.get_installed_vcs()
        dir = SCons.Tool.MSCommon.vc.find_vc_pdir(self.vc_version)
        if not dir:
            debug('find_vs_dir():  no installed VC %s' % self.vc_version)
            return None
        return dir

    def find_vs_dir_by_reg(self):
        root = 'Software\\'

        if is_win64():
            root = root + 'Wow6432Node\\'
        for key in self.hkeys:
            if key=='use_dir':
                return self.find_vs_dir_by_vc()
            key = root + key
            try:
                comps = read_reg(key)
            except SCons.Util.WinError as e:
                debug('find_vs_dir_by_reg(): no VS registry key {}'.format(repr(key)))
            else:
                debug('find_vs_dir_by_reg(): found VS in registry: {}'.format(comps))
                return comps
        return None

    def find_vs_dir(self):
        """ Can use registry or location of VC to find vs dir
        First try to find by registry, and if that fails find via VC dir
        """


        if True:
            vs_dir=self.find_vs_dir_by_reg()
            return vs_dir
        else:
            return self.find_vs_dir_by_vc()

    def find_executable(self):
        vs_dir = self.get_vs_dir()
        if not vs_dir:
            debug('find_executable():  no vs_dir ({})'.format(vs_dir))
            return None
        executable = os.path.join(vs_dir, self.executable_path)
        executable = os.path.normpath(executable)
        if not os.path.isfile(executable):
            debug('find_executable():  {} not on file system'.format(executable))
            return None
        return executable

    def get_batch_file(self):
        try:
            return self._cache['batch_file']
        except KeyError:
            batch_file = self.find_batch_file()
            self._cache['batch_file'] = batch_file
            return batch_file

    def get_executable(self):
        try:
            debug('get_executable using cache:%s'%self._cache['executable'])
            return self._cache['executable']
        except KeyError:
            executable = self.find_executable()
            self._cache['executable'] = executable
            debug('get_executable not in cache:%s'%executable)
            return executable

    def get_vs_dir(self):
        try:
            return self._cache['vs_dir']
        except KeyError:
            vs_dir = self.find_vs_dir()
            self._cache['vs_dir'] = vs_dir
            return vs_dir

    def get_supported_arch(self):
        try:
            return self._cache['supported_arch']
        except KeyError:
            # RDEVE: for the time being use hardcoded lists
            # supported_arch = self.find_supported_arch()
            self._cache['supported_arch'] = self.supported_arch
            return self.supported_arch

    def reset(self):
        self._cache = {}

# The list of supported Visual Studio versions we know how to detect.
#
# How to look for .bat file ?
#  - VS 2008 Express (x86):
#     * from registry key productdir, gives the full path to vsvarsall.bat. In
#     HKEY_LOCAL_MACHINE):
#         Software\Microsoft\VCEpress\9.0\Setup\VC\productdir
#     * from environmnent variable VS90COMNTOOLS: the path is then ..\..\VC
#     relatively to the path given by the variable.
#
#  - VS 2008 Express (WoW6432: 32 bits on windows x64):
#         Software\Wow6432Node\Microsoft\VCEpress\9.0\Setup\VC\productdir
#
#  - VS 2005 Express (x86):
#     * from registry key productdir, gives the full path to vsvarsall.bat. In
#     HKEY_LOCAL_MACHINE):
#         Software\Microsoft\VCEpress\8.0\Setup\VC\productdir
#     * from environmnent variable VS80COMNTOOLS: the path is then ..\..\VC
#     relatively to the path given by the variable.
#
#  - VS 2005 Express (WoW6432: 32 bits on windows x64): does not seem to have a
#  productdir ?
#
#  - VS 2003 .Net (pro edition ? x86):
#     * from registry key productdir. The path is then ..\Common7\Tools\
#     relatively to the key. The key is in HKEY_LOCAL_MACHINE):
#         Software\Microsoft\VisualStudio\7.1\Setup\VC\productdir
#     * from environmnent variable VS71COMNTOOLS: the path is the full path to
#     vsvars32.bat
#
#  - VS 98 (VS 6):
#     * from registry key productdir. The path is then Bin
#     relatively to the key. The key is in HKEY_LOCAL_MACHINE):
#         Software\Microsoft\VisualStudio\6.0\Setup\VC98\productdir
#
# The first version found in the list is the one used by default if
# there are multiple versions installed.  Barring good reasons to
# the contrary, this means we should list versions from most recent
# to oldest.  Pro versions get listed before Express versions on the
# assumption that, by default, you'd rather use the version you paid
# good money for in preference to whatever Microsoft makes available
# for free.
#
# If you update this list, update _VCVER and _VCVER_TO_PRODUCT_DIR in
# Tool/MSCommon/vc.py, and the MSVC_VERSION documentation in Tool/msvc.xml.

SupportedVSList = [
    # Visual Studio 2017
    VisualStudio('14.1',
                 vc_version='14.1',
                 sdk_version='10.0A',
                 hkeys=[],
                 common_tools_var='VS150COMNTOOLS',
                 executable_path=r'Common7\IDE\devenv.com',
                 batch_file_path=r'VC\Auxiliary\Build\vsvars32.bat',
                 supported_arch=['x86', 'amd64', "arm"],
                 ),

    # Visual Studio 2015
    VisualStudio('14.0',
                 vc_version='14.0',
                 sdk_version='10.0',
                 hkeys=[r'Microsoft\VisualStudio\14.0\Setup\VS\ProductDir'],
                 common_tools_var='VS140COMNTOOLS',
                 executable_path=r'Common7\IDE\devenv.com',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 supported_arch=['x86', 'amd64', "arm"],
    ),

    # Visual C++ 2015 Express Edition (for Desktop)
    VisualStudio('14.0Exp',
                 vc_version='14.0',
                 sdk_version='10.0A',
                 hkeys=[r'Microsoft\VisualStudio\14.0\Setup\VS\ProductDir'],
                 common_tools_var='VS140COMNTOOLS',
                 executable_path=r'Common7\IDE\WDExpress.exe',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 supported_arch=['x86', 'amd64', "arm"],
    ),

    # Visual Studio 2013
    VisualStudio('12.0',
                 vc_version='12.0',
                 sdk_version='8.1A',
                 hkeys=[r'Microsoft\VisualStudio\12.0\Setup\VS\ProductDir'],
                 common_tools_var='VS120COMNTOOLS',
                 executable_path=r'Common7\IDE\devenv.com',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 supported_arch=['x86', 'amd64'],
    ),

    # Visual C++ 2013 Express Edition (for Desktop)
    VisualStudio('12.0Exp',
                 vc_version='12.0',
                 sdk_version='8.1A',
                 hkeys=[r'Microsoft\VisualStudio\12.0\Setup\VS\ProductDir'],
                 common_tools_var='VS120COMNTOOLS',
                 executable_path=r'Common7\IDE\WDExpress.exe',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 supported_arch=['x86', 'amd64'],
    ),

    # Visual Studio 2012
    VisualStudio('11.0',
                 sdk_version='8.0A',
                 hkeys=[r'Microsoft\VisualStudio\11.0\Setup\VS\ProductDir'],
                 common_tools_var='VS110COMNTOOLS',
                 executable_path=r'Common7\IDE\devenv.com',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 supported_arch=['x86', 'amd64'],
    ),

    # Visual C++ 2012 Express Edition (for Desktop)
    VisualStudio('11.0Exp',
                 vc_version='11.0',
                 sdk_version='8.0A',
                 hkeys=[r'Microsoft\VisualStudio\11.0\Setup\VS\ProductDir'],
                 common_tools_var='VS110COMNTOOLS',
                 executable_path=r'Common7\IDE\WDExpress.exe',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 supported_arch=['x86', 'amd64'],
    ),

    # Visual Studio 2010
    VisualStudio('10.0',
                 sdk_version='7.0A',
                 hkeys=[r'Microsoft\VisualStudio\10.0\Setup\VS\ProductDir'],
                 common_tools_var='VS100COMNTOOLS',
                 executable_path=r'Common7\IDE\devenv.com',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 supported_arch=['x86', 'amd64'],
    ),

    # Visual C++ 2010 Express Edition
    VisualStudio('10.0Exp',
                 vc_version='10.0',
                 sdk_version='7.0A',
                 hkeys=[r'Microsoft\VCExpress\10.0\Setup\VS\ProductDir'],
                 common_tools_var='VS100COMNTOOLS',
                 executable_path=r'Common7\IDE\VCExpress.exe',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 supported_arch=['x86'],
    ),

    # Visual Studio 2008
    VisualStudio('9.0',
                 sdk_version='6.0A',
                 hkeys=[r'Microsoft\VisualStudio\9.0\Setup\VS\ProductDir'],
                 common_tools_var='VS90COMNTOOLS',
                 executable_path=r'Common7\IDE\devenv.com',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 supported_arch=['x86', 'amd64'],
    ),

    # Visual C++ 2008 Express Edition
    VisualStudio('9.0Exp',
                 vc_version='9.0',
                 sdk_version='6.0A',
                 hkeys=[r'Microsoft\VCExpress\9.0\Setup\VS\ProductDir'],
                 common_tools_var='VS90COMNTOOLS',
                 executable_path=r'Common7\IDE\VCExpress.exe',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 supported_arch=['x86'],
    ),

    # Visual Studio 2005
    VisualStudio('8.0',
                 sdk_version='6.0A',
                 hkeys=[r'Microsoft\VisualStudio\8.0\Setup\VS\ProductDir'],
                 common_tools_var='VS80COMNTOOLS',
                 executable_path=r'Common7\IDE\devenv.com',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 default_dirname='Microsoft Visual Studio 8',
                 supported_arch=['x86', 'amd64'],
    ),

    # Visual C++ 2005 Express Edition
    VisualStudio('8.0Exp',
                 vc_version='8.0Exp',
                 sdk_version='6.0A',
                 hkeys=[r'Microsoft\VCExpress\8.0\Setup\VS\ProductDir'],
                 common_tools_var='VS80COMNTOOLS',
                 executable_path=r'Common7\IDE\VCExpress.exe',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 default_dirname='Microsoft Visual Studio 8',
                 supported_arch=['x86'],
    ),

    # Visual Studio .NET 2003
    VisualStudio('7.1',
                 sdk_version='6.0',
                 hkeys=[r'Microsoft\VisualStudio\7.1\Setup\VS\ProductDir'],
                 common_tools_var='VS71COMNTOOLS',
                 executable_path=r'Common7\IDE\devenv.com',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 default_dirname='Microsoft Visual Studio .NET 2003',
                 supported_arch=['x86'],
    ),

    # Visual Studio .NET
    VisualStudio('7.0',
                 sdk_version='2003R2',
                 hkeys=[r'Microsoft\VisualStudio\7.0\Setup\VS\ProductDir'],
                 common_tools_var='VS70COMNTOOLS',
                 executable_path=r'IDE\devenv.com',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 default_dirname='Microsoft Visual Studio .NET',
                 supported_arch=['x86'],
    ),

    # Visual Studio 6.0
    VisualStudio('6.0',
                 sdk_version='2003R1',
                 hkeys=[r'Microsoft\VisualStudio\6.0\Setup\Microsoft Visual Studio\ProductDir',
                        'use_dir'],
                 common_tools_var='VS60COMNTOOLS',
                 executable_path=r'Common\MSDev98\Bin\MSDEV.COM',
                 batch_file_path=r'Common7\Tools\vsvars32.bat',
                 default_dirname='Microsoft Visual Studio',
                 supported_arch=['x86'],
    ),
]

SupportedVSMap = {}
for vs in SupportedVSList:
    SupportedVSMap[vs.version] = vs


# Finding installed versions of Visual Studio isn't cheap, because it
# goes not only to the registry but also to the disk to sanity-check
# that there is, in fact, a Visual Studio directory there and that the
# registry entry isn't just stale.  Find this information once, when
# requested, and cache it.

InstalledVSList = None
InstalledVSMap  = None

def get_installed_visual_studios():
    global InstalledVSList
    global InstalledVSMap
    if InstalledVSList is None:
        InstalledVSList = []
        InstalledVSMap = {}
        for vs in SupportedVSList:
            debug('trying to find VS %s' % vs.version)
            if vs.get_executable():
                debug('found VS %s' % vs.version)
                InstalledVSList.append(vs)
                InstalledVSMap[vs.version] = vs
    return InstalledVSList

def reset_installed_visual_studios():
    global InstalledVSList
    global InstalledVSMap
    InstalledVSList = None
    InstalledVSMap  = None
    for vs in SupportedVSList:
        vs.reset()

    # Need to clear installed VC's as well as they are used in finding
    # installed VS's
    SCons.Tool.MSCommon.vc.reset_installed_vcs()


# We may be asked to update multiple construction environments with
# SDK information.  When doing this, we check on-disk for whether
# the SDK has 'mfc' and 'atl' subdirectories.  Since going to disk
# is expensive, cache results by directory.

#SDKEnvironmentUpdates = {}
#
#def set_sdk_by_directory(env, sdk_dir):
#    global SDKEnvironmentUpdates
#    try:
#        env_tuple_list = SDKEnvironmentUpdates[sdk_dir]
#    except KeyError:
#        env_tuple_list = []
#        SDKEnvironmentUpdates[sdk_dir] = env_tuple_list
#
#        include_path = os.path.join(sdk_dir, 'include')
#        mfc_path = os.path.join(include_path, 'mfc')
#        atl_path = os.path.join(include_path, 'atl')
#
#        if os.path.exists(mfc_path):
#            env_tuple_list.append(('INCLUDE', mfc_path))
#        if os.path.exists(atl_path):
#            env_tuple_list.append(('INCLUDE', atl_path))
#        env_tuple_list.append(('INCLUDE', include_path))
#
#        env_tuple_list.append(('LIB', os.path.join(sdk_dir, 'lib')))
#        env_tuple_list.append(('LIBPATH', os.path.join(sdk_dir, 'lib')))
#        env_tuple_list.append(('PATH', os.path.join(sdk_dir, 'bin')))
#
#    for variable, directory in env_tuple_list:
#        env.PrependENVPath(variable, directory)

def msvs_exists():
    return (len(get_installed_visual_studios()) > 0)

def get_vs_by_version(msvs):
    global InstalledVSMap
    global SupportedVSMap

    debug('vs.py:get_vs_by_version()')
    if msvs not in SupportedVSMap:
        msg = "Visual Studio version %s is not supported" % repr(msvs)
        raise SCons.Errors.UserError(msg)
    get_installed_visual_studios()
    vs = InstalledVSMap.get(msvs)
    debug('InstalledVSMap:%s'%InstalledVSMap)
    debug('vs.py:get_vs_by_version: found vs:%s'%vs)
    # Some check like this would let us provide a useful error message
    # if they try to set a Visual Studio version that's not installed.
    # However, we also want to be able to run tests (like the unit
    # tests) on systems that don't, or won't ever, have it installed.
    # It might be worth resurrecting this, with some configurable
    # setting that the tests can use to bypass the check.
    #if not vs:
    #    msg = "Visual Studio version %s is not installed" % repr(msvs)
    #    raise SCons.Errors.UserError, msg
    return vs

def get_default_version(env):
    """Returns the default version string to use for MSVS.

    If no version was requested by the user through the MSVS environment
    variable, query all the available visual studios through
    get_installed_visual_studios, and take the highest one.

    Return
    ------
    version: str
        the default version.
    """
    if 'MSVS' not in env or not SCons.Util.is_Dict(env['MSVS']):
        # get all versions, and remember them for speed later
        versions = [vs.version for vs in get_installed_visual_studios()]
        env['MSVS'] = {'VERSIONS' : versions}
    else:
        versions = env['MSVS'].get('VERSIONS', [])

    if 'MSVS_VERSION' not in env:
        if versions:
            env['MSVS_VERSION'] = versions[0] #use highest version by default
        else:
            debug('get_default_version: WARNING: no installed versions found, '
                  'using first in SupportedVSList (%s)'%SupportedVSList[0].version)
            env['MSVS_VERSION'] = SupportedVSList[0].version

    env['MSVS']['VERSION'] = env['MSVS_VERSION']

    return env['MSVS_VERSION']

def get_default_arch(env):
    """Return the default arch to use for MSVS

    if no version was requested by the user through the MSVS_ARCH environment
    variable, select x86

    Return
    ------
    arch: str
    """
    arch = env.get('MSVS_ARCH', 'x86')

    msvs = InstalledVSMap.get(env['MSVS_VERSION'])

    if not msvs:
        arch = 'x86'
    elif not arch in msvs.get_supported_arch():
        fmt = "Visual Studio version %s does not support architecture %s"
        raise SCons.Errors.UserError(fmt % (env['MSVS_VERSION'], arch))

    return arch

def merge_default_version(env):
    version = get_default_version(env)
    arch = get_default_arch(env)

def msvs_setup_env(env):
    batfilename = msvs.get_batch_file()
    msvs = get_vs_by_version(version)
    if msvs is None:
        return

    # XXX: I think this is broken. This will silently set a bogus tool instead
    # of failing, but there is no other way with the current scons tool
    # framework
    if batfilename is not None:

        vars = ('LIB', 'LIBPATH', 'PATH', 'INCLUDE')

        msvs_list = get_installed_visual_studios()
        vscommonvarnames = [vs.common_tools_var for vs in msvs_list]
        save_ENV = env['ENV']
        nenv = normalize_env(env['ENV'],
                             ['COMSPEC'] + vscommonvarnames,
                             force=True)
        try:
            output = get_output(batfilename, arch, env=nenv)
        finally:
            env['ENV'] = save_ENV
        vars = parse_output(output, vars)

        for k, v in vars.items():
            env.PrependENVPath(k, v, delete_existing=1)

def query_versions():
    """Query the system to get available versions of VS. A version is
    considered when a batfile is found."""
    msvs_list = get_installed_visual_studios()
    versions = [msvs.version for msvs in msvs_list]
    return versions

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
