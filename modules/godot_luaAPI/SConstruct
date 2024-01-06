#!/usr/bin/env python
import os
import sys
import fnmatch

from config import configure

Export('configure')

env = SConscript("external/SConscript")
env.Tool('compilation_db')

if os.name == 'posix':
    if os.environ.get('CXX', None) != None and env['CXX'] != os.environ['CXX']:
        print("Using system CXX: {}".format(os.environ['CXX']))
        env['CXX'] = os.environ['CXX']
    if os.environ.get('CC', None) != None and env['CC'] != os.environ['CC']:
        print("Using system CC: {}".format(os.environ['CC']))
        env['CC'] = os.environ['CC']

env.Append(CPPDEFINES = ['LAPI_GDEXTENSION'])
env.Append(CPPPATH 	  = [Dir('src').abspath, Dir('external').abspath])

env['sources'] = []
Export('env')
SConscript("lua_libraries/SConscript")

sources = Glob('*.cpp')
sources.append(Glob('src/*.cpp'))
sources.append(Glob('src/classes/*.cpp'))
sources.append(env['sources'])

if env["luaapi_luaver"] == 'jit':
    env.Append(CPPDEFINES=['LAPI_LUAJIT'])
elif env["luaapi_luaver"] == '5.1':
    env.Append(CPPDEFINES=['LAPI_51'])

cdb = env.CompilationDatabase('compile_commands.json')
Alias('cdb', cdb)

library = env.SharedLibrary(
    "project/addons/luaAPI/bin/libluaapi{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
    source=sources,
)

env.Default(library)

