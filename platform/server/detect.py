import os
import sys


def is_active():
    return True


def get_name():
    return "Server"


def can_build():

    # Doesn't build against Godot 3.0 for now, disable to avoid confusing users
    return False

    if (os.name != "posix" or sys.platform == "darwin"):
        return False

    return True


def get_opts():
    from SCons.Variables import BoolVariable
    return [
        BoolVariable('use_llvm', 'Use the LLVM compiler', False),
    ]


def get_flags():

    return [
    ]


def configure(env):

    ## Build type

    if (env["target"] == "release"):
        env.Append(CCFLAGS=['-O2', '-ffast-math', '-fomit-frame-pointer'])

    elif (env["target"] == "release_debug"):
        env.Append(CCFLAGS=['-O2', '-ffast-math', '-DDEBUG_ENABLED'])

    elif (env["target"] == "debug"):
        env.Append(CCFLAGS=['-g2', '-DDEBUG_ENABLED', '-DDEBUG_MEMORY_ENABLED'])

    ## Architecture

    is64 = sys.maxsize > 2**32
    if (env["bits"] == "default"):
        env["bits"] = "64" if is64 else "32"

    ## Compiler configuration

    if env['use_llvm']:
        if ('clang++' not in env['CXX']):
            env["CC"] = "clang"
            env["CXX"] = "clang++"
            env["LD"] = "clang++"
        env.Append(CPPFLAGS=['-DTYPED_METHOD_BIND'])
        env.extra_suffix = ".llvm" + env.extra_suffix

    ## Dependencies

    # FIXME: Check for existence of the libs before parsing their flags with pkg-config

    if not env['builtin_openssl']:
        env.ParseConfig('pkg-config openssl --cflags --libs')

    if not env['builtin_libwebp']:
        env.ParseConfig('pkg-config libwebp --cflags --libs')

    # freetype depends on libpng and zlib, so bundling one of them while keeping others
    # as shared libraries leads to weird issues
    if env['builtin_freetype'] or env['builtin_libpng'] or env['builtin_zlib']:
        env['builtin_freetype'] = True
        env['builtin_libpng'] = True
        env['builtin_zlib'] = True

    if not env['builtin_freetype']:
        env.ParseConfig('pkg-config freetype2 --cflags --libs')

    if not env['builtin_libpng']:
        env.ParseConfig('pkg-config libpng --cflags --libs')

    if not env['builtin_enet']:
        env.ParseConfig('pkg-config libenet --cflags --libs')

    if not env['builtin_squish'] and env['tools']:
        env.ParseConfig('pkg-config libsquish --cflags --libs')

    if not env['builtin_zstd']:
        env.ParseConfig('pkg-config libzstd --cflags --libs')

    # Sound and video libraries
    # Keep the order as it triggers chained dependencies (ogg needed by others, etc.)

    if not env['builtin_libtheora']:
        env['builtin_libogg'] = False  # Needed to link against system libtheora
        env['builtin_libvorbis'] = False  # Needed to link against system libtheora
        env.ParseConfig('pkg-config theora theoradec --cflags --libs')

    if not env['builtin_libvpx']:
        env.ParseConfig('pkg-config vpx --cflags --libs')

    if not env['builtin_libvorbis']:
        env['builtin_libogg'] = False  # Needed to link against system libvorbis
        env.ParseConfig('pkg-config vorbis vorbisfile --cflags --libs')

    if not env['builtin_opus']:
        env['builtin_libogg'] = False  # Needed to link against system opus
        env.ParseConfig('pkg-config opus opusfile --cflags --libs')

    if not env['builtin_libogg']:
        env.ParseConfig('pkg-config ogg --cflags --libs')

    ## Flags

    # Linkflags below this line should typically stay the last ones
    if not env['builtin_zlib']:
        env.ParseConfig('pkg-config zlib --cflags --libs')

    env.Append(CPPPATH=['#platform/server'])
    env.Append(CPPFLAGS=['-DSERVER_ENABLED', '-DUNIX_ENABLED'])
    env.Append(LIBS=['pthread'])
