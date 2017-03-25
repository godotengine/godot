import os
import sys


def is_active():
    return True


def get_name():
    return "Haiku"


def can_build():
    if (os.name != "posix"):
        return False

    if (sys.platform == "darwin"):
        return False

    return True


def get_opts():
    return [
        ('debug_release', 'Add debug symbols to release version', 'no')
    ]


def get_flags():
    return [
    ]


def configure(env):
    is64 = sys.maxsize > 2**32

    if (env["bits"] == "default"):
        if (is64):
            env["bits"] = "64"
        else:
            env["bits"] = "32"

    env.Append(CPPPATH=['#platform/haiku'])

    env["CC"] = "gcc-x86"
    env["CXX"] = "g++-x86"

    if (env["target"] == "release"):
        if (env["debug_release"] == "yes"):
            env.Append(CCFLAGS=['-g2'])
        else:
            env.Append(CCFLAGS=['-O3', '-ffast-math'])
    elif (env["target"] == "release_debug"):
        env.Append(CCFLAGS=['-O2', '-ffast-math', '-DDEBUG_ENABLED'])
    elif (env["target"] == "debug"):
        env.Append(CCFLAGS=['-g2', '-DDEBUG_ENABLED', '-DDEBUG_MEMORY_ENABLED'])

    # env.Append(CCFLAGS=['-DFREETYPE_ENABLED'])
    env.Append(CPPFLAGS=['-DPTHREAD_NO_RENAME'])  # TODO: enable when we have pthread_setname_np
    env.Append(CPPFLAGS=['-DOPENGL_ENABLED', '-DMEDIA_KIT_ENABLED'])
    env.Append(CPPFLAGS=['-DUNIX_ENABLED', '-DGLES2_ENABLED', '-DGLES_OVER_GL'])
    env.Append(LIBS=['be', 'game', 'media', 'network', 'bnetapi', 'z', 'GL'])

    import methods
    env.Append(BUILDERS={'GLSL120': env.Builder(action=methods.build_legacygl_headers, suffix='glsl.h', src_suffix='.glsl')})
    env.Append(BUILDERS={'GLSL': env.Builder(action=methods.build_glsl_headers, suffix='glsl.h', src_suffix='.glsl')})
    env.Append(BUILDERS={'GLSL120GLES': env.Builder(action=methods.build_gles2_headers, suffix='glsl.h', src_suffix='.glsl')})
