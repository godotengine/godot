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


__revision__ = "src/engine/SCons/Tool/MSCommon/sdk.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

__doc__ = """Module to detect the Platform/Windows SDK

PSDK 2003 R1 is the earliest version detected.
"""

import os

import SCons.Errors
import SCons.Util

from . import common

debug = common.debug

# SDK Checks. This is of course a mess as everything else on MS platforms. Here
# is what we do to detect the SDK:
#
# For Windows SDK >= 6.0: just look into the registry entries:
#   HKLM\Software\Microsoft\Microsoft SDKs\Windows
# All the keys in there are the available versions.
#
# For Platform SDK before 6.0 (2003 server R1 and R2, etc...), there does not
# seem to be any sane registry key, so the precise location is hardcoded.
#
# For versions below 2003R1, it seems the PSDK is included with Visual Studio?
#
# Also, per the following:
#     http://benjamin.smedbergs.us/blog/tag/atl/
# VC++ Professional comes with the SDK, VC++ Express does not.

# Location of the SDK (checked for 6.1 only)
_CURINSTALLED_SDK_HKEY_ROOT = \
        r"Software\Microsoft\Microsoft SDKs\Windows\CurrentInstallFolder"


class SDKDefinition(object):
    """
    An abstract base class for trying to find installed SDK directories.
    """
    def __init__(self, version, **kw):
        self.version = version
        self.__dict__.update(kw)

    def find_sdk_dir(self):
        """Try to find the MS SDK from the registry.

        Return None if failed or the directory does not exist.
        """
        if not SCons.Util.can_read_reg:
            debug('find_sdk_dir(): can not read registry')
            return None

        hkey = self.HKEY_FMT % self.hkey_data
        debug('find_sdk_dir(): checking registry:{}'.format(hkey))

        try:
            sdk_dir = common.read_reg(hkey)
        except SCons.Util.WinError as e:
            debug('find_sdk_dir(): no SDK registry key {}'.format(repr(hkey)))
            return None

        debug('find_sdk_dir(): Trying SDK Dir: {}'.format(sdk_dir))

        if not os.path.exists(sdk_dir):
            debug('find_sdk_dir():  {} not on file system'.format(sdk_dir))
            return None

        ftc = os.path.join(sdk_dir, self.sanity_check_file)
        if not os.path.exists(ftc):
            debug("find_sdk_dir(): sanity check {} not found".format(ftc))
            return None

        return sdk_dir

    def get_sdk_dir(self):
        """Return the MSSSDK given the version string."""
        try:
            return self._sdk_dir
        except AttributeError:
            sdk_dir = self.find_sdk_dir()
            self._sdk_dir = sdk_dir
            return sdk_dir

    def get_sdk_vc_script(self,host_arch, target_arch):
        """ Return the script to initialize the VC compiler installed by SDK
        """

        if (host_arch == 'amd64' and target_arch == 'x86'):
            # No cross tools needed compiling 32 bits on 64 bit machine
            host_arch=target_arch

        arch_string=target_arch
        if (host_arch != target_arch):
            arch_string='%s_%s'%(host_arch,target_arch)

        debug("sdk.py: get_sdk_vc_script():arch_string:%s host_arch:%s target_arch:%s"%(arch_string,
                                                           host_arch,
                                                           target_arch))
        file=self.vc_setup_scripts.get(arch_string,None)
        debug("sdk.py: get_sdk_vc_script():file:%s"%file)
        return file

class WindowsSDK(SDKDefinition):
    """
    A subclass for trying to find installed Windows SDK directories.
    """
    HKEY_FMT = r'Software\Microsoft\Microsoft SDKs\Windows\v%s\InstallationFolder'
    def __init__(self, *args, **kw):
        SDKDefinition.__init__(self, *args, **kw)
        self.hkey_data = self.version

class PlatformSDK(SDKDefinition):
    """
    A subclass for trying to find installed Platform SDK directories.
    """
    HKEY_FMT = r'Software\Microsoft\MicrosoftSDK\InstalledSDKS\%s\Install Dir'
    def __init__(self, *args, **kw):
        SDKDefinition.__init__(self, *args, **kw)
        self.hkey_data = self.uuid

#
# The list of VC initialization scripts installed by the SDK
# These should be tried if the vcvarsall.bat TARGET_ARCH fails
preSDK61VCSetupScripts = { 'x86'      : r'bin\vcvars32.bat',
                           'amd64'    : r'bin\vcvarsamd64.bat',
                           'x86_amd64': r'bin\vcvarsx86_amd64.bat',
                           'x86_ia64' : r'bin\vcvarsx86_ia64.bat',
                           'ia64'     : r'bin\vcvarsia64.bat'}

SDK61VCSetupScripts = {'x86'      : r'bin\vcvars32.bat',
                       'amd64'    : r'bin\amd64\vcvarsamd64.bat',
                       'x86_amd64': r'bin\x86_amd64\vcvarsx86_amd64.bat',
                       'x86_ia64' : r'bin\x86_ia64\vcvarsx86_ia64.bat',
                       'ia64'     : r'bin\ia64\vcvarsia64.bat'}

SDK70VCSetupScripts =    { 'x86'      : r'bin\vcvars32.bat',
                           'amd64'    : r'bin\vcvars64.bat',
                           'x86_amd64': r'bin\vcvarsx86_amd64.bat',
                           'x86_ia64' : r'bin\vcvarsx86_ia64.bat',
                           'ia64'     : r'bin\vcvarsia64.bat'}

SDK100VCSetupScripts =    {'x86'      : r'bin\vcvars32.bat',
                           'amd64'    : r'bin\vcvars64.bat',
                           'x86_amd64': r'bin\x86_amd64\vcvarsx86_amd64.bat',
                           'x86_arm'  : r'bin\x86_arm\vcvarsx86_arm.bat'}


# The list of support SDKs which we know how to detect.
#
# The first SDK found in the list is the one used by default if there
# are multiple SDKs installed.  Barring good reasons to the contrary,
# this means we should list SDKs from most recent to oldest.
#
# If you update this list, update the documentation in Tool/mssdk.xml.
SupportedSDKList = [
    WindowsSDK('10.0',
               sanity_check_file=r'bin\SetEnv.Cmd',
               include_subdir='include',
               lib_subdir={
                   'x86'       : ['lib'],
                   'x86_64'    : [r'lib\x64'],
                   'ia64'      : [r'lib\ia64'],
               },
               vc_setup_scripts = SDK70VCSetupScripts,
              ),
    WindowsSDK('7.1',
               sanity_check_file=r'bin\SetEnv.Cmd',
               include_subdir='include',
               lib_subdir={
                   'x86'       : ['lib'],
                   'x86_64'    : [r'lib\x64'],
                   'ia64'      : [r'lib\ia64'],
               },
               vc_setup_scripts = SDK70VCSetupScripts,
              ),
    WindowsSDK('7.0A',
               sanity_check_file=r'bin\SetEnv.Cmd',
               include_subdir='include',
               lib_subdir={
                   'x86'       : ['lib'],
                   'x86_64'    : [r'lib\x64'],
                   'ia64'      : [r'lib\ia64'],
               },
               vc_setup_scripts = SDK70VCSetupScripts,
              ),
    WindowsSDK('7.0',
               sanity_check_file=r'bin\SetEnv.Cmd',
               include_subdir='include',
               lib_subdir={
                   'x86'       : ['lib'],
                   'x86_64'    : [r'lib\x64'],
                   'ia64'      : [r'lib\ia64'],
               },
               vc_setup_scripts = SDK70VCSetupScripts,
              ),
    WindowsSDK('6.1',
               sanity_check_file=r'bin\SetEnv.Cmd',
               include_subdir='include',
               lib_subdir={
                   'x86'       : ['lib'],
                   'x86_64'    : [r'lib\x64'],
                   'ia64'      : [r'lib\ia64'],
               },
               vc_setup_scripts = SDK61VCSetupScripts,
              ),

    WindowsSDK('6.0A',
               sanity_check_file=r'include\windows.h',
               include_subdir='include',
               lib_subdir={
                   'x86'       : ['lib'],
                   'x86_64'    : [r'lib\x64'],
                   'ia64'      : [r'lib\ia64'],
               },
               vc_setup_scripts = preSDK61VCSetupScripts,
              ),

    WindowsSDK('6.0',
               sanity_check_file=r'bin\gacutil.exe',
               include_subdir='include',
               lib_subdir='lib',
               vc_setup_scripts = preSDK61VCSetupScripts,
              ),

    PlatformSDK('2003R2',
                sanity_check_file=r'SetEnv.Cmd',
                uuid="D2FF9F89-8AA2-4373-8A31-C838BF4DBBE1",
                vc_setup_scripts = preSDK61VCSetupScripts,
               ),

    PlatformSDK('2003R1',
                sanity_check_file=r'SetEnv.Cmd',
                uuid="8F9E5EF3-A9A5-491B-A889-C58EFFECE8B3",
                vc_setup_scripts = preSDK61VCSetupScripts,
               ),
]

SupportedSDKMap = {}
for sdk in SupportedSDKList:
    SupportedSDKMap[sdk.version] = sdk


# Finding installed SDKs isn't cheap, because it goes not only to the
# registry but also to the disk to sanity-check that there is, in fact,
# an SDK installed there and that the registry entry isn't just stale.
# Find this information once, when requested, and cache it.

InstalledSDKList = None
InstalledSDKMap = None

def get_installed_sdks():
    global InstalledSDKList
    global InstalledSDKMap
    debug('sdk.py:get_installed_sdks()')
    if InstalledSDKList is None:
        InstalledSDKList = []
        InstalledSDKMap = {}
        for sdk in SupportedSDKList:
            debug('MSCommon/sdk.py: trying to find SDK %s' % sdk.version)
            if sdk.get_sdk_dir():
                debug('MSCommon/sdk.py:found SDK %s' % sdk.version)
                InstalledSDKList.append(sdk)
                InstalledSDKMap[sdk.version] = sdk
    return InstalledSDKList


# We may be asked to update multiple construction environments with
# SDK information.  When doing this, we check on-disk for whether
# the SDK has 'mfc' and 'atl' subdirectories.  Since going to disk
# is expensive, cache results by directory.

SDKEnvironmentUpdates = {}

def set_sdk_by_directory(env, sdk_dir):
    global SDKEnvironmentUpdates
    debug('set_sdk_by_directory: Using dir:%s'%sdk_dir)
    try:
        env_tuple_list = SDKEnvironmentUpdates[sdk_dir]
    except KeyError:
        env_tuple_list = []
        SDKEnvironmentUpdates[sdk_dir] = env_tuple_list

        include_path = os.path.join(sdk_dir, 'include')
        mfc_path = os.path.join(include_path, 'mfc')
        atl_path = os.path.join(include_path, 'atl')

        if os.path.exists(mfc_path):
            env_tuple_list.append(('INCLUDE', mfc_path))
        if os.path.exists(atl_path):
            env_tuple_list.append(('INCLUDE', atl_path))
        env_tuple_list.append(('INCLUDE', include_path))

        env_tuple_list.append(('LIB', os.path.join(sdk_dir, 'lib')))
        env_tuple_list.append(('LIBPATH', os.path.join(sdk_dir, 'lib')))
        env_tuple_list.append(('PATH', os.path.join(sdk_dir, 'bin')))

    for variable, directory in env_tuple_list:
        env.PrependENVPath(variable, directory)

def get_sdk_by_version(mssdk):
    if mssdk not in SupportedSDKMap:
        raise SCons.Errors.UserError("SDK version {} is not supported".format(repr(mssdk)))
    get_installed_sdks()
    return InstalledSDKMap.get(mssdk)

def get_default_sdk():
    """Set up the default Platform/Windows SDK."""
    get_installed_sdks()
    if not InstalledSDKList:
        return None
    return InstalledSDKList[0]

def mssdk_setup_env(env):
    debug('sdk.py:mssdk_setup_env()')
    if 'MSSDK_DIR' in env:
        sdk_dir = env['MSSDK_DIR']
        if sdk_dir is None:
            return
        sdk_dir = env.subst(sdk_dir)
        debug('sdk.py:mssdk_setup_env: Using MSSDK_DIR:{}'.format(sdk_dir))
    elif 'MSSDK_VERSION' in env:
        sdk_version = env['MSSDK_VERSION']
        if sdk_version is None:
            msg = "SDK version is specified as None"
            raise SCons.Errors.UserError(msg)
        sdk_version = env.subst(sdk_version)
        mssdk = get_sdk_by_version(sdk_version)
        if mssdk is None:
            msg = "SDK version %s is not installed" % sdk_version
            raise SCons.Errors.UserError(msg)
        sdk_dir = mssdk.get_sdk_dir()
        debug('sdk.py:mssdk_setup_env: Using MSSDK_VERSION:%s'%sdk_dir)
    elif 'MSVS_VERSION' in env:
        msvs_version = env['MSVS_VERSION']
        debug('sdk.py:mssdk_setup_env:Getting MSVS_VERSION from env:%s'%msvs_version)
        if msvs_version is None:
            debug('sdk.py:mssdk_setup_env thinks msvs_version is None')
            return
        msvs_version = env.subst(msvs_version)
        from . import vs
        msvs = vs.get_vs_by_version(msvs_version)
        debug('sdk.py:mssdk_setup_env:msvs is :%s'%msvs)
        if not msvs:
            debug('sdk.py:mssdk_setup_env: no VS version detected, bailingout:%s'%msvs)
            return
        sdk_version = msvs.sdk_version
        debug('sdk.py:msvs.sdk_version is %s'%sdk_version)
        if not sdk_version:
            return
        mssdk = get_sdk_by_version(sdk_version)
        if not mssdk:
            mssdk = get_default_sdk()
            if not mssdk:
                return
        sdk_dir = mssdk.get_sdk_dir()
        debug('sdk.py:mssdk_setup_env: Using MSVS_VERSION:%s'%sdk_dir)
    else:
        mssdk = get_default_sdk()
        if not mssdk:
            return
        sdk_dir = mssdk.get_sdk_dir()
        debug('sdk.py:mssdk_setup_env: not using any env values. sdk_dir:%s'%sdk_dir)

    set_sdk_by_directory(env, sdk_dir)

    #print "No MSVS_VERSION: this is likely to be a bug"

def mssdk_exists(version=None):
    sdks = get_installed_sdks()
    if version is None:
        return len(sdks) > 0
    return version in sdks

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
