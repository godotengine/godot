"""
Common helper functions for working with the Microsoft tool chain.
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

__revision__ = "src/engine/SCons/Tool/MSCommon/common.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import copy
import os
import subprocess
import re

import SCons.Util


LOGFILE = os.environ.get('SCONS_MSCOMMON_DEBUG')
if LOGFILE == '-':
    def debug(message):
        print(message)
elif LOGFILE:
    try:
        import logging
    except ImportError:
        debug = lambda message: open(LOGFILE, 'a').write(message + '\n')
    else:
        logging.basicConfig(filename=LOGFILE, level=logging.DEBUG)
        debug = logging.debug
else:
    debug = lambda x: None


_is_win64 = None

def is_win64():
    """Return true if running on windows 64 bits.

    Works whether python itself runs in 64 bits or 32 bits."""
    # Unfortunately, python does not provide a useful way to determine
    # if the underlying Windows OS is 32-bit or 64-bit.  Worse, whether
    # the Python itself is 32-bit or 64-bit affects what it returns,
    # so nothing in sys.* or os.* help.

    # Apparently the best solution is to use env vars that Windows
    # sets.  If PROCESSOR_ARCHITECTURE is not x86, then the python
    # process is running in 64 bit mode (on a 64-bit OS, 64-bit
    # hardware, obviously).
    # If this python is 32-bit but the OS is 64, Windows will set
    # ProgramW6432 and PROCESSOR_ARCHITEW6432 to non-null.
    # (Checking for HKLM\Software\Wow6432Node in the registry doesn't
    # work, because some 32-bit installers create it.)
    global _is_win64
    if _is_win64 is None:
        # I structured these tests to make it easy to add new ones or
        # add exceptions in the future, because this is a bit fragile.
        _is_win64 = False
        if os.environ.get('PROCESSOR_ARCHITECTURE', 'x86') != 'x86':
            _is_win64 = True
        if os.environ.get('PROCESSOR_ARCHITEW6432'):
            _is_win64 = True
        if os.environ.get('ProgramW6432'):
            _is_win64 = True
    return _is_win64


def read_reg(value, hkroot=SCons.Util.HKEY_LOCAL_MACHINE):
    return SCons.Util.RegGetValue(hkroot, value)[0]

def has_reg(value):
    """Return True if the given key exists in HKEY_LOCAL_MACHINE, False
    otherwise."""
    try:
        SCons.Util.RegOpenKeyEx(SCons.Util.HKEY_LOCAL_MACHINE, value)
        ret = True
    except SCons.Util.WinError:
        ret = False
    return ret

# Functions for fetching environment variable settings from batch files.

def normalize_env(env, keys, force=False):
    """Given a dictionary representing a shell environment, add the variables
    from os.environ needed for the processing of .bat files; the keys are
    controlled by the keys argument.

    It also makes sure the environment values are correctly encoded.

    If force=True, then all of the key values that exist are copied
    into the returned dictionary.  If force=false, values are only
    copied if the key does not already exist in the copied dictionary.

    Note: the environment is copied."""
    normenv = {}
    if env:
        for k in list(env.keys()):
            normenv[k] = copy.deepcopy(env[k])

        for k in keys:
            if k in os.environ and (force or not k in normenv):
                normenv[k] = os.environ[k]

    # This shouldn't be necessary, since the default environment should include system32,
    # but keep this here to be safe, since it's needed to find reg.exe which the MSVC
    # bat scripts use.
    sys32_dir = os.path.join(os.environ.get("SystemRoot",
                                            os.environ.get("windir", r"C:\Windows\system32")),
                             "System32")

    if sys32_dir not in normenv['PATH']:
        normenv['PATH'] = normenv['PATH'] + os.pathsep + sys32_dir

    # Without Wbem in PATH, vcvarsall.bat has a "'wmic' is not recognized"
    # error starting with Visual Studio 2017, although the script still
    # seems to work anyway.
    sys32_wbem_dir = os.path.join(sys32_dir, 'Wbem')
    if sys32_wbem_dir not in normenv['PATH']:
        normenv['PATH'] = normenv['PATH'] + os.pathsep + sys32_wbem_dir

    debug("PATH: %s"%normenv['PATH'])

    return normenv

def get_output(vcbat, args = None, env = None):
    """Parse the output of given bat file, with given args."""

    if env is None:
        # Create a blank environment, for use in launching the tools
        env = SCons.Environment.Environment(tools=[])

    # TODO:  This is a hard-coded list of the variables that (may) need
    # to be imported from os.environ[] for v[sc]*vars*.bat file
    # execution to work.  This list should really be either directly
    # controlled by vc.py, or else derived from the common_tools_var
    # settings in vs.py.
    vs_vc_vars = [
        'COMSPEC',
        # VS100 and VS110: Still set, but modern MSVC setup scripts will
        # discard these if registry has values.  However Intel compiler setup
        # script still requires these as of 2013/2014.
        'VS140COMNTOOLS',
        'VS120COMNTOOLS',
        'VS110COMNTOOLS',
        'VS100COMNTOOLS',
        'VS90COMNTOOLS',
        'VS80COMNTOOLS',
        'VS71COMNTOOLS',
        'VS70COMNTOOLS',
        'VS60COMNTOOLS',
    ]
    env['ENV'] = normalize_env(env['ENV'], vs_vc_vars, force=False)

    if args:
        debug("Calling '%s %s'" % (vcbat, args))
        popen = SCons.Action._subproc(env,
                                      '"%s" %s & set' % (vcbat, args),
                                      stdin='devnull',
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
    else:
        debug("Calling '%s'" % vcbat)
        popen = SCons.Action._subproc(env,
                                      '"%s" & set' % vcbat,
                                      stdin='devnull',
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)

    # Use the .stdout and .stderr attributes directly because the
    # .communicate() method uses the threading module on Windows
    # and won't work under Pythons not built with threading.
    stdout = popen.stdout.read()
    stderr = popen.stderr.read()

    # Extra debug logic, uncomment if necessary
#     debug('get_output():stdout:%s'%stdout)
#     debug('get_output():stderr:%s'%stderr)

    if stderr:
        # TODO: find something better to do with stderr;
        # this at least prevents errors from getting swallowed.
        import sys
        sys.stderr.write(stderr)
    if popen.wait() != 0:
        raise IOError(stderr.decode("mbcs"))

    output = stdout.decode("mbcs")
    return output

def parse_output(output, keep=("INCLUDE", "LIB", "LIBPATH", "PATH")):
    """
    Parse output from running visual c++/studios vcvarsall.bat and running set
    To capture the values listed in keep
    """

    # dkeep is a dict associating key: path_list, where key is one item from
    # keep, and pat_list the associated list of paths
    dkeep = dict([(i, []) for i in keep])

    # rdk will  keep the regex to match the .bat file output line starts
    rdk = {}
    for i in keep:
        rdk[i] = re.compile('%s=(.*)' % i, re.I)

    def add_env(rmatch, key, dkeep=dkeep):
        path_list = rmatch.group(1).split(os.pathsep)
        for path in path_list:
            # Do not add empty paths (when a var ends with ;)
            if path:
                # XXX: For some reason, VC98 .bat file adds "" around the PATH
                # values, and it screws up the environment later, so we strip
                # it.
                path = path.strip('"')
                dkeep[key].append(str(path))

    for line in output.splitlines():
        for k, value in rdk.items():
            match = value.match(line)
            if match:
                add_env(match, k)

    return dkeep

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
