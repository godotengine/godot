import os
import sys
import string


def is_active():
    return True


def get_name():
    return "JavaScript"


def can_build():
    return os.environ.has_key("EMSCRIPTEN_ROOT")


def get_opts():

    return [
        ['wasm', 'Compile to WebAssembly', 'no'],
        ['javascript_eval', 'Enable JavaScript eval interface', 'yes'],
    ]


def get_flags():

    return [
        ('tools', 'no'),
        ('module_etc1_enabled', 'no'),
        ('module_mpc_enabled', 'no'),
        ('module_theora_enabled', 'no'),
    ]


def create(env):
    # remove Windows' .exe suffix
    return env.Clone(PROGSUFFIX='')


def escape_sources_backslashes(target, source, env, for_signature):
    return [path.replace('\\','\\\\') for path in env.GetBuildPath(source)]

def escape_target_backslashes(target, source, env, for_signature):
    return env.GetBuildPath(target[0]).replace('\\','\\\\')


def configure(env):
    env['ENV'] = os.environ

    env.Append(CPPPATH=['#platform/javascript'])

    env.PrependENVPath('PATH', os.environ['EMSCRIPTEN_ROOT'])
    env['CC']      = 'emcc'
    env['CXX']     = 'em++'
    env['LINK']    = 'emcc'
    env['RANLIB']  = 'emranlib'
    # Emscripten's ar has issues with duplicate file names, so use cc
    env['AR']      = 'emcc'
    env['ARFLAGS'] = '-o'
    if (os.name == 'nt'):
        # use TempFileMunge on Windows since some commands get too long for
        # cmd.exe even with spawn_fix
        # need to escape backslashes for this
        env['ESCAPED_SOURCES'] = escape_sources_backslashes
        env['ESCAPED_TARGET'] = escape_target_backslashes
        env['ARCOM'] = '${TEMPFILE("%s")}' % env['ARCOM'].replace('$SOURCES', '$ESCAPED_SOURCES').replace('$TARGET', '$ESCAPED_TARGET')

    env['OBJSUFFIX'] = '.bc'
    env['LIBSUFFIX'] = '.bc'

    if (env["target"] == "release"):
        env.Append(CCFLAGS=['-O2'])
    elif (env["target"] == "release_debug"):
        env.Append(CCFLAGS=['-O2', '-DDEBUG_ENABLED'])
    elif (env["target"] == "debug"):
        env.Append(CCFLAGS=['-D_DEBUG', '-Wall', '-O2', '-DDEBUG_ENABLED'])
        #env.Append(CCFLAGS=['-D_DEBUG', '-Wall', '-g4', '-DDEBUG_ENABLED'])
        env.Append(CPPFLAGS=['-DDEBUG_MEMORY_ALLOC'])

    # TODO: Move that to opus module's config
    if("module_opus_enabled" in env and env["module_opus_enabled"] != "no"):
        env.opus_fixed_point = "yes"

    # These flags help keep the file size down
    env.Append(CPPFLAGS=["-fno-exceptions", '-DNO_SAFE_CAST', '-fno-rtti'])
    env.Append(CPPFLAGS=['-DJAVASCRIPT_ENABLED', '-DUNIX_ENABLED', '-DPTHREAD_NO_RENAME', '-DNO_FCNTL', '-DMPC_FIXED_POINT', '-DTYPED_METHOD_BIND', '-DNO_THREADS'])
    env.Append(CPPFLAGS=['-DGLES3_ENABLED'])
    env.Append(CPPFLAGS=['-DGLES_NO_CLIENT_ARRAYS'])

    if env['wasm'] == 'yes':
        env.Append(LINKFLAGS=['-s', 'BINARYEN=1'])
        # Maximum memory size is baked into the WebAssembly binary during
        # compilation, so we need to enable memory growth to allow setting
        # TOTAL_MEMORY at runtime. The value set at runtime must be higher than
        # what is set during compilation, check TOTAL_MEMORY in Emscripten's
        # src/settings.js for the default.
        env.Append(LINKFLAGS=['-s', 'ALLOW_MEMORY_GROWTH=1'])
        env.extra_suffix = '.webassembly' + env.extra_suffix
    else:
        env.Append(CPPFLAGS=['-s', 'ASM_JS=1'])
        env.Append(LINKFLAGS=['-s', 'ASM_JS=1'])
        env.Append(LINKFLAGS=['--separate-asm'])

    if env['javascript_eval'] == 'yes':
        env.Append(CPPFLAGS=['-DJAVASCRIPT_EVAL_ENABLED'])

    env.Append(LINKFLAGS=['-O2'])
    env.Append(LINKFLAGS=['-s', 'USE_WEBGL2=1'])
    # env.Append(LINKFLAGS=['-g4'])

    import methods
