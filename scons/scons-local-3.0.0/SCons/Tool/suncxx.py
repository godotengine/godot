"""SCons.Tool.sunc++

Tool-specific initialization for C++ on SunOS / Solaris.

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

__revision__ = "src/engine/SCons/Tool/suncxx.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons

import os
import re
import subprocess

import SCons.Tool.cxx
cplusplus = SCons.Tool.cxx
#cplusplus = __import__('c++', globals(), locals(), [])

package_info = {}

def get_package_info(package_name, pkginfo, pkgchk):
    try:
        return package_info[package_name]
    except KeyError:
        version = None
        pathname = None
        try:
            sadm_contents = open('/var/sadm/install/contents', 'r').read()
        except EnvironmentError:
            pass
        else:
            sadm_re = re.compile('^(\S*/bin/CC)(=\S*)? %s$' % package_name, re.M)
            sadm_match = sadm_re.search(sadm_contents)
            if sadm_match:
                pathname = os.path.dirname(sadm_match.group(1))

        try:
            p = subprocess.Popen([pkginfo, '-l', package_name],
                                 stdout=subprocess.PIPE,
                                 stderr=open('/dev/null', 'w'))
        except EnvironmentError:
            pass
        else:
            pkginfo_contents = p.communicate()[0]
            version_re = re.compile('^ *VERSION:\s*(.*)$', re.M)
            version_match = version_re.search(pkginfo_contents)
            if version_match:
                version = version_match.group(1)

        if pathname is None:
            try:
                p = subprocess.Popen([pkgchk, '-l', package_name],
                                     stdout=subprocess.PIPE,
                                     stderr=open('/dev/null', 'w'))
            except EnvironmentError:
                pass
            else:
                pkgchk_contents = p.communicate()[0]
                pathname_re = re.compile(r'^Pathname:\s*(.*/bin/CC)$', re.M)
                pathname_match = pathname_re.search(pkgchk_contents)
                if pathname_match:
                    pathname = os.path.dirname(pathname_match.group(1))

        package_info[package_name] = (pathname, version)
        return package_info[package_name]

# use the package installer tool lslpp to figure out where cppc and what
# version of it is installed
def get_cppc(env):
    cxx = env.subst('$CXX')
    if cxx:
        cppcPath = os.path.dirname(cxx)
    else:
        cppcPath = None

    cppcVersion = None

    pkginfo = env.subst('$PKGINFO')
    pkgchk = env.subst('$PKGCHK')

    for package in ['SPROcpl']:
        path, version = get_package_info(package, pkginfo, pkgchk)
        if path and version:
            cppcPath, cppcVersion = path, version
            break

    return (cppcPath, 'CC', 'CC', cppcVersion)

def generate(env):
    """Add Builders and construction variables for SunPRO C++."""
    path, cxx, shcxx, version = get_cppc(env)
    if path:
        cxx = os.path.join(path, cxx)
        shcxx = os.path.join(path, shcxx)

    cplusplus.generate(env)

    env['CXX'] = cxx
    env['SHCXX'] = shcxx
    env['CXXVERSION'] = version
    env['SHCXXFLAGS']   = SCons.Util.CLVar('$CXXFLAGS -KPIC')
    env['SHOBJPREFIX']  = 'so_'
    env['SHOBJSUFFIX']  = '.o'
    
def exists(env):
    path, cxx, shcxx, version = get_cppc(env)
    if path and cxx:
        cppc = os.path.join(path, cxx)
        if os.path.exists(cppc):
            return cppc
    return None

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
