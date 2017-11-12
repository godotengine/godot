""" msginit tool 

Tool specific initialization of msginit tool.
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

__revision__ = "src/engine/SCons/Tool/msginit.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons.Warnings
import SCons.Builder
import re

#############################################################################
def _optional_no_translator_flag(env):
  """ Return '--no-translator' flag if we run *msginit(1)*  in non-interactive
      mode."""
  import SCons.Util
  if 'POAUTOINIT' in env:
    autoinit = env['POAUTOINIT']
  else:
    autoinit = False
  if autoinit:
    return [SCons.Util.CLVar('--no-translator')]
  else:
    return [SCons.Util.CLVar('')]
#############################################################################

#############################################################################
def _POInitBuilder(env, **kw):
  """ Create builder object for `POInit` builder. """
  import SCons.Action
  from SCons.Tool.GettextCommon import _init_po_files, _POFileBuilder
  action = SCons.Action.Action(_init_po_files, None)
  return _POFileBuilder(env, action=action, target_alias='$POCREATE_ALIAS')
#############################################################################
  
#############################################################################
from SCons.Environment import _null
#############################################################################
def _POInitBuilderWrapper(env, target=None, source=_null, **kw):
  """ Wrapper for _POFileBuilder. We use it to make user's life easier.
  
  This wrapper checks for `$POTDOMAIN` construction variable (or override in
  `**kw`) and treats it appropriatelly. 
  """
  if source is _null:
    if 'POTDOMAIN' in kw:
      domain = kw['POTDOMAIN']
    elif 'POTDOMAIN' in env:
      domain = env['POTDOMAIN']
    else:
      domain = 'messages'
    source = [ domain ] # NOTE: Suffix shall be appended automatically
  return env._POInitBuilder(target, source, **kw)
#############################################################################

#############################################################################
def generate(env,**kw):
  """ Generate the `msginit` tool """
  import SCons.Util
  from SCons.Tool.GettextCommon import _detect_msginit
  try:
    env['MSGINIT'] = _detect_msginit(env)
  except:
    env['MSGINIT'] = 'msginit'
  msginitcom = '$MSGINIT ${_MSGNoTranslator(__env__)} -l ${_MSGINITLOCALE}' \
             + ' $MSGINITFLAGS -i $SOURCE -o $TARGET'
  # NOTE: We set POTSUFFIX here, in case the 'xgettext' is not loaded
  #       (sometimes we really don't need it)
  env.SetDefault(
    POSUFFIX = ['.po'],
    POTSUFFIX = ['.pot'],
    _MSGINITLOCALE = '${TARGET.filebase}',
    _MSGNoTranslator = _optional_no_translator_flag,
    MSGINITCOM = msginitcom,
    MSGINITCOMSTR = '',
    MSGINITFLAGS = [ ],
    POAUTOINIT = False,
    POCREATE_ALIAS = 'po-create'
  )
  env.Append( BUILDERS = { '_POInitBuilder' : _POInitBuilder(env) } )
  env.AddMethod(_POInitBuilderWrapper, 'POInit')
  env.AlwaysBuild(env.Alias('$POCREATE_ALIAS'))
#############################################################################

#############################################################################
def exists(env):
  """ Check if the tool exists """
  from SCons.Tool.GettextCommon import _msginit_exists
  try:
    return  _msginit_exists(env)
  except:
    return False
#############################################################################

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
