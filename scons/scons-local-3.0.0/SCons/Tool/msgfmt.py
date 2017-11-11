""" msgfmt tool """

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

__revision__ = "src/engine/SCons/Tool/msgfmt.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

from SCons.Builder import BuilderBase
#############################################################################
class _MOFileBuilder(BuilderBase):
  """ The builder class for `MO` files.
  
  The reason for this builder to exists and its purpose is quite simillar 
  as for `_POFileBuilder`. This time, we extend list of sources, not targets,
  and call `BuilderBase._execute()` only once (as we assume single-target
  here).
  """

  def _execute(self, env, target, source, *args, **kw):
    # Here we add support for 'LINGUAS_FILE' keyword. Emitter is not suitable
    # in this case, as it is called too late (after multiple sources
    # are handled single_source builder.
    import SCons.Util
    from SCons.Tool.GettextCommon import _read_linguas_from_files
    linguas_files = None
    if 'LINGUAS_FILE' in env and env['LINGUAS_FILE'] is not None:
      linguas_files = env['LINGUAS_FILE']
      # This should prevent from endless recursion. 
      env['LINGUAS_FILE'] = None
      # We read only languages. Suffixes shall be added automatically.
      linguas = _read_linguas_from_files(env, linguas_files)
      if SCons.Util.is_List(source):
        source.extend(linguas)
      elif source is not None:
        source = [source] + linguas
      else:
        source = linguas
    result = BuilderBase._execute(self,env,target,source,*args, **kw)
    if linguas_files is not None:
      env['LINGUAS_FILE'] = linguas_files
    return result
#############################################################################

#############################################################################
def _create_mo_file_builder(env, **kw):
  """ Create builder object for `MOFiles` builder """
  import SCons.Action
  # FIXME: What factory use for source? Ours or their?
  kw['action'] = SCons.Action.Action('$MSGFMTCOM','$MSGFMTCOMSTR')
  kw['suffix'] = '$MOSUFFIX'
  kw['src_suffix'] = '$POSUFFIX'
  kw['src_builder'] = '_POUpdateBuilder'
  kw['single_source'] = True 
  return _MOFileBuilder(**kw)
#############################################################################

#############################################################################
def generate(env,**kw):
  """ Generate `msgfmt` tool """
  import SCons.Util
  from SCons.Tool.GettextCommon import _detect_msgfmt
  try:
    env['MSGFMT'] = _detect_msgfmt(env)
  except:
    env['MSGFMT'] = 'msgfmt'
  env.SetDefault(
    MSGFMTFLAGS = [ SCons.Util.CLVar('-c') ],
    MSGFMTCOM = '$MSGFMT $MSGFMTFLAGS -o $TARGET $SOURCE',
    MSGFMTCOMSTR = '',
    MOSUFFIX = ['.mo'],
    POSUFFIX = ['.po']
  )
  env.Append( BUILDERS = { 'MOFiles'  : _create_mo_file_builder(env) } )
#############################################################################

#############################################################################
def exists(env):
  """ Check if the tool exists """
  from SCons.Tool.GettextCommon import _msgfmt_exists
  try:
    return _msgfmt_exists(env)
  except:
    return False
#############################################################################

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
