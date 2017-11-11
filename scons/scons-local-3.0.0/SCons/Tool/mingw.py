"""SCons.Tool.gcc

Tool-specific initialization for MinGW (http://www.mingw.org/)

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

__revision__ = "src/engine/SCons/Tool/mingw.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os
import os.path

import SCons.Action
import SCons.Builder
import SCons.Defaults
import SCons.Tool
import SCons.Util

# This is what we search for to find mingw:
key_program = 'mingw32-gcc'

def find(env):
    # First search in the SCons path
    path=env.WhereIs(key_program)
    if (path):
        return path
    # then the OS path:
    path=SCons.Util.WhereIs(key_program)
    if (path):
        return path

    # If that doesn't work try default location for mingw
    save_path=env['ENV']['PATH']
    env.AppendENVPath('PATH',r'c:\MinGW\bin')
    path =env.WhereIs(key_program)
    if not path:
        env['ENV']['PATH']=save_path
    return path

def shlib_generator(target, source, env, for_signature):
    cmd = SCons.Util.CLVar(['$SHLINK', '$SHLINKFLAGS']) 

    dll = env.FindIxes(target, 'SHLIBPREFIX', 'SHLIBSUFFIX')
    if dll: cmd.extend(['-o', dll])

    cmd.extend(['$SOURCES', '$_LIBDIRFLAGS', '$_LIBFLAGS'])

    implib = env.FindIxes(target, 'LIBPREFIX', 'LIBSUFFIX')
    if implib: cmd.append('-Wl,--out-implib,'+implib.get_string(for_signature))

    def_target = env.FindIxes(target, 'WINDOWSDEFPREFIX', 'WINDOWSDEFSUFFIX')
    insert_def = env.subst("$WINDOWS_INSERT_DEF")
    if not insert_def in ['', '0', 0] and def_target: \
        cmd.append('-Wl,--output-def,'+def_target.get_string(for_signature))

    return [cmd]

def shlib_emitter(target, source, env):
    dll = env.FindIxes(target, 'SHLIBPREFIX', 'SHLIBSUFFIX')
    no_import_lib = env.get('no_import_lib', 0)

    if not dll:
        raise SCons.Errors.UserError("A shared library should have exactly one target with the suffix: %s Target(s) are:%s" %  \
            (env.subst("$SHLIBSUFFIX"), ",".join([str(t) for t in target])))
    
    if not no_import_lib and \
       not env.FindIxes(target, 'LIBPREFIX', 'LIBSUFFIX'):

        # Create list of target libraries as strings
        targetStrings=env.ReplaceIxes(dll,  
                                      'SHLIBPREFIX', 'SHLIBSUFFIX',
                                      'LIBPREFIX', 'LIBSUFFIX')
        
        # Now add file nodes to target list
        target.append(env.fs.File(targetStrings))

    # Append a def file target if there isn't already a def file target
    # or a def file source or the user has explicitly asked for the target
    # to be emitted.
    def_source = env.FindIxes(source, 'WINDOWSDEFPREFIX', 'WINDOWSDEFSUFFIX')
    def_target = env.FindIxes(target, 'WINDOWSDEFPREFIX', 'WINDOWSDEFSUFFIX')
    skip_def_insert = env.subst("$WINDOWS_INSERT_DEF") in ['', '0', 0]
    if not def_source and not def_target and not skip_def_insert:
        # Create list of target libraries and def files as strings
        targetStrings=env.ReplaceIxes(dll,  
                                      'SHLIBPREFIX', 'SHLIBSUFFIX',
                                      'WINDOWSDEFPREFIX', 'WINDOWSDEFSUFFIX')
        
        # Now add file nodes to target list
        target.append(env.fs.File(targetStrings))

    return (target, source)
                         

shlib_action = SCons.Action.Action(shlib_generator, generator=1)

res_action = SCons.Action.Action('$RCCOM', '$RCCOMSTR')

res_builder = SCons.Builder.Builder(action=res_action, suffix='.o',
                                    source_scanner=SCons.Tool.SourceFileScanner)
SCons.Tool.SourceFileScanner.add_scanner('.rc', SCons.Defaults.CScan)

def generate(env):
    mingw = find(env)
    if mingw:
        dir = os.path.dirname(mingw)
        env.PrependENVPath('PATH', dir )
        

    # Most of mingw is the same as gcc and friends...
    gnu_tools = ['gcc', 'g++', 'gnulink', 'ar', 'gas', 'gfortran', 'm4']
    for tool in gnu_tools:
        SCons.Tool.Tool(tool)(env)

    #... but a few things differ:
    env['CC'] = 'gcc'
    env['SHCCFLAGS'] = SCons.Util.CLVar('$CCFLAGS')
    env['CXX'] = 'g++'
    env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS')
    env['SHLINKFLAGS'] = SCons.Util.CLVar('$LINKFLAGS -shared')
    env['SHLINKCOM']   = shlib_action
    env['LDMODULECOM'] = shlib_action
    env.Append(SHLIBEMITTER = [shlib_emitter])
    env.Append(LDMODULEEMITTER = [shlib_emitter])
    env['AS'] = 'as'

    env['WIN32DEFPREFIX']        = ''
    env['WIN32DEFSUFFIX']        = '.def'
    env['WINDOWSDEFPREFIX']      = '${WIN32DEFPREFIX}'
    env['WINDOWSDEFSUFFIX']      = '${WIN32DEFSUFFIX}'

    env['SHOBJSUFFIX'] = '.o'
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1

    env['RC'] = 'windres'
    env['RCFLAGS'] = SCons.Util.CLVar('')
    env['RCINCFLAGS'] = '$( ${_concat(RCINCPREFIX, CPPPATH, RCINCSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)'
    env['RCINCPREFIX'] = '--include-dir '
    env['RCINCSUFFIX'] = ''
    env['RCCOM'] = '$RC $_CPPDEFFLAGS $RCINCFLAGS ${RCINCPREFIX} ${SOURCE.dir} $RCFLAGS -i $SOURCE -o $TARGET'
    env['BUILDERS']['RES'] = res_builder
    
    # Some setting from the platform also have to be overridden:
    env['OBJSUFFIX'] = '.o'
    env['LIBPREFIX'] = 'lib'
    env['LIBSUFFIX'] = '.a'
    env['PROGSUFFIX'] = '.exe'

def exists(env):
    return find(env)

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
