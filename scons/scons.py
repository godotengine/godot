#! /usr/bin/env python
#
# SCons - a Software Constructor
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

from __future__ import print_function

__revision__ = "src/script/scons.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

__version__ = "3.0.0"

__build__ = "rel_3.0.0:4395:8972f6a2f699"

__buildsys__ = "ubuntu-16"

__date__ = "2017/09/18 12:59:24"

__developer__ = "bdbaddog"

import os
import sys


##############################################################################
# BEGIN STANDARD SCons SCRIPT HEADER
#
# This is the cut-and-paste logic so that a self-contained script can
# interoperate correctly with different SCons versions and installation
# locations for the engine.  If you modify anything in this section, you
# should also change other scripts that use this same header.
##############################################################################

# Strip the script directory from sys.path() so on case-insensitive
# (WIN32) systems Python doesn't think that the "scons" script is the
# "SCons" package.  Replace it with our own library directories
# (version-specific first, in case they installed by hand there,
# followed by generic) so we pick up the right version of the build
# engine modules if they're in either directory.


if (3,0,0) < sys.version_info < (3,5,0) or sys.version_info < (2,7,0):
    msg = "scons: *** SCons version %s does not run under Python version %s.\n\
Python < 3.5 is not yet supported.\n"
    sys.stderr.write(msg % (__version__, sys.version.split()[0]))
    sys.exit(1)


script_dir = sys.path[0]

if script_dir in sys.path:
    sys.path.remove(script_dir)

libs = []

if "SCONS_LIB_DIR" in os.environ:
    libs.append(os.environ["SCONS_LIB_DIR"])

# - running from source takes priority (since 2.3.2), excluding SCONS_LIB_DIR settings
script_path = os.path.abspath(os.path.dirname(__file__))
source_path = os.path.join(script_path, '..', 'engine')
libs.append(source_path)

local_version = 'scons-local-' + __version__
local = 'scons-local'
if script_dir:
    local_version = os.path.join(script_dir, local_version)
    local = os.path.join(script_dir, local)
libs.append(os.path.abspath(local_version))
libs.append(os.path.abspath(local))

scons_version = 'scons-%s' % __version__

# preferred order of scons lookup paths
prefs = []


# - running from egg check
try:
    import pkg_resources
except ImportError:
    pass
else:
    # when running from an egg add the egg's directory
    try:
        d = pkg_resources.get_distribution('scons')
    except pkg_resources.DistributionNotFound:
        pass
    else:
        prefs.append(d.location)

if sys.platform == 'win32':
    # sys.prefix is (likely) C:\Python*;
    # check only C:\Python*.
    prefs.append(sys.prefix)
    prefs.append(os.path.join(sys.prefix, 'Lib', 'site-packages'))
else:
    # On other (POSIX) platforms, things are more complicated due to
    # the variety of path names and library locations.  Try to be smart
    # about it.
    if script_dir == 'bin':
        # script_dir is `pwd`/bin;
        # check `pwd`/lib/scons*.
        prefs.append(os.getcwd())
    else:
        if script_dir == '.' or script_dir == '':
            script_dir = os.getcwd()
        head, tail = os.path.split(script_dir)
        if tail == "bin":
            # script_dir is /foo/bin;
            # check /foo/lib/scons*.
            prefs.append(head)

    head, tail = os.path.split(sys.prefix)
    if tail == "usr":
        # sys.prefix is /foo/usr;
        # check /foo/usr/lib/scons* first,
        # then /foo/usr/local/lib/scons*.
        prefs.append(sys.prefix)
        prefs.append(os.path.join(sys.prefix, "local"))
    elif tail == "local":
        h, t = os.path.split(head)
        if t == "usr":
            # sys.prefix is /foo/usr/local;
            # check /foo/usr/local/lib/scons* first,
            # then /foo/usr/lib/scons*.
            prefs.append(sys.prefix)
            prefs.append(head)
        else:
            # sys.prefix is /foo/local;
            # check only /foo/local/lib/scons*.
            prefs.append(sys.prefix)
    else:
        # sys.prefix is /foo (ends in neither /usr or /local);
        # check only /foo/lib/scons*.
        prefs.append(sys.prefix)

    temp = [os.path.join(x, 'lib') for x in prefs]
    temp.extend([os.path.join(x,
                                           'lib',
                                           'python' + sys.version[:3],
                                           'site-packages') for x in prefs])
    prefs = temp

    # Add the parent directory of the current python's library to the
    # preferences.  On SuSE-91/AMD64, for example, this is /usr/lib64,
    # not /usr/lib.
    try:
        libpath = os.__file__
    except AttributeError:
        pass
    else:
        # Split /usr/libfoo/python*/os.py to /usr/libfoo/python*.
        libpath, tail = os.path.split(libpath)
        # Split /usr/libfoo/python* to /usr/libfoo
        libpath, tail = os.path.split(libpath)
        # Check /usr/libfoo/scons*.
        prefs.append(libpath)

# Look first for 'scons-__version__' in all of our preference libs,
# then for 'scons'.
libs.extend([os.path.join(x, scons_version) for x in prefs])
libs.extend([os.path.join(x, 'scons') for x in prefs])

sys.path = libs + sys.path

##############################################################################
# END STANDARD SCons SCRIPT HEADER
##############################################################################

if __name__ == "__main__":
    try:
        import SCons.Script
    except ImportError:
        print("SCons import failed. Unable to find engine files in:")
        for path in libs:
            print("  {}".format(path))
        raise

    # this does all the work, and calls sys.exit
    # with the proper exit status when done.
    SCons.Script.main()

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
