import os
import sys


def is_active():
    return True


def get_name():
    return "Haiku"


def can_build():

    if (os.name != "posix" or sys.platform == "darwin"):
        return False

    return True


def get_opts():
    from SCons.Variables import EnumVariable

    return [
        EnumVariable('debug_symbols', 'Add debug symbols to release version', 'yes', ('yes', 'no', 'full')),
    ]


def get_flags():

    return [
    ]


def configure(env):

    ## Build type

    if (env["target"] == "release"):
        env.Prepend(CCFLAGS=['-O3', '-ffast-math'])
        if (env["debug_symbols"] == "yes"):
            env.Prepend(CCFLAGS=['-g1'])
        if (env["debug_symbols"] == "full"):
            env.Prepend(CCFLAGS=['-g2'])

    elif (env["target"] == "release_debug"):
        env.Prepend(CCFLAGS=['-O2', '-ffast-math', '-DDEBUG_ENABLED'])
        if (env["debug_symbols"] == "yes"):
            env.Prepend(CCFLAGS=['-g1'])
        if (env["debug_symbols"] == "full"):
            env.Prepend(CCFLAGS=['-g2'])

    elif (env["target"] == "debug"):
        env.Prepend(CCFLAGS=['-g3', '-DDEBUG_ENABLED', '-DDEBUG_MEMORY_ENABLED'])

    ## Architecture

    is64 = sys.maxsize > 2**32
    if (env["bits"] == "default"):
        env["bits"] = "64" if is64 else "32"

    ## Compiler configuration

    env["CC"] = "gcc-x86"
    env["CXX"] = "g++-x86"

    ## Flags

    env.Append(CPPPATH=['#platform/haiku'])
    env.Append(CPPFLAGS=['-DUNIX_ENABLED', '-DOPENGL_ENABLED', '-DGLES2_ENABLED', '-DGLES_OVER_GL'])
    env.Append(CPPFLAGS=['-DMEDIA_KIT_ENABLED'])
    # env.Append(CCFLAGS=['-DFREETYPE_ENABLED'])
    env.Append(CPPFLAGS=['-DPTHREAD_NO_RENAME'])  # TODO: enable when we have pthread_setname_np
    env.Append(LIBS=['be', 'game', 'media', 'network', 'bnetapi', 'z', 'GL'])
