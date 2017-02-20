import os
import sys
import string


def is_active():
    return True


def get_name():
    return "JavaScript"


def can_build():

    import os
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


def configure(env):
    env['ENV'] = os.environ
    env.use_windows_spawn_fix('javascript')

    env.Append(CPPPATH=['#platform/javascript'])

    em_path = os.environ["EMSCRIPTEN_ROOT"]

    env['ENV']['PATH'] = em_path + ":" + env['ENV']['PATH']
    env['CC']     = em_path + '/emcc'
    env['CXX']    = em_path + '/em++'
    env['LINK']   = em_path + '/emcc'
    env['AR']     = em_path + '/emar'
    env['RANLIB'] = em_path + '/emranlib'

    env['OBJSUFFIX'] = '.bc'
    env['LIBSUFFIX'] = '.a'

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
