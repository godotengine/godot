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

__revision__ = "src/engine/SCons/Tool/MSCommon/__init__.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

__doc__ = """
Common functions for Microsoft Visual Studio and Visual C/C++.
"""

import copy
import os
import re
import subprocess

import SCons.Errors
import SCons.Platform.win32
import SCons.Util

from SCons.Tool.MSCommon.sdk import mssdk_exists, \
                                    mssdk_setup_env

from SCons.Tool.MSCommon.vc import msvc_exists, \
                                   msvc_setup_env, \
                                   msvc_setup_env_once, \
                                   msvc_version_to_maj_min

from SCons.Tool.MSCommon.vs import get_default_version, \
                                   get_vs_by_version, \
                                   merge_default_version, \
                                   msvs_exists, \
                                   query_versions

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
