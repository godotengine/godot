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

    return [
        ('debug_release', 'Add debug symbols to release version', 'no')
    ]


def get_flags():

    return [
    ]


def configure(env):

    ## Build type

    if (env["target"] == "release"):
        if (env["debug_release"] == "yes"):
            env.Prepend(CCFLAGS=['-g2'])
        else:
            env.Prepend(CCFLAGS=['-O3', '-ffast-math'])

    elif (env["target"] == "release_debug"):
        env.Prepend(CCFLAGS=['-O2', '-ffast-math', '-DDEBUG_ENABLED'])

    elif (env["target"] == "debug"):
        env.Prepend(CCFLAGS=['-g2', '-DDEBUG_ENABLED', '-DDEBUG_MEMORY_ENABLED'])

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
