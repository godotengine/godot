"""SCons.Tool.zip

Tool-specific initialization for zip.

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

__revision__ = "src/engine/SCons/Tool/zip.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os.path

import SCons.Builder
import SCons.Defaults
import SCons.Node.FS
import SCons.Util

import zipfile

zipcompression = zipfile.ZIP_DEFLATED
def zip(target, source, env):
    compression = env.get('ZIPCOMPRESSION', 0)
    zf = zipfile.ZipFile(str(target[0]), 'w', compression)
    for s in source:
        if s.isdir():
            for dirpath, dirnames, filenames in os.walk(str(s)):
                for fname in filenames:
                    path = os.path.join(dirpath, fname)
                    if os.path.isfile(path):

                        zf.write(path, os.path.relpath(path, str(env.get('ZIPROOT', ''))))
        else:
            zf.write(str(s), os.path.relpath(str(s), str(env.get('ZIPROOT', ''))))
    zf.close()

zipAction = SCons.Action.Action(zip, varlist=['ZIPCOMPRESSION'])

ZipBuilder = SCons.Builder.Builder(action = SCons.Action.Action('$ZIPCOM', '$ZIPCOMSTR'),
                                   source_factory = SCons.Node.FS.Entry,
                                   source_scanner = SCons.Defaults.DirScanner,
                                   suffix = '$ZIPSUFFIX',
                                   multi = 1)


def generate(env):
    """Add Builders and construction variables for zip to an Environment."""
    try:
        bld = env['BUILDERS']['Zip']
    except KeyError:
        bld = ZipBuilder
        env['BUILDERS']['Zip'] = bld

    env['ZIP']        = 'zip'
    env['ZIPFLAGS']   = SCons.Util.CLVar('')
    env['ZIPCOM']     = zipAction
    env['ZIPCOMPRESSION'] =  zipcompression
    env['ZIPSUFFIX']  = '.zip'
    env['ZIPROOT']    = SCons.Util.CLVar('')

def exists(env):
    return True

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
