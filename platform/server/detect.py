import os
import sys


def is_active():
    return True


def get_name():
    return "Server"


def can_build():

    if (os.name != "posix" or sys.platform == "darwin"):
        return False

    return True


def get_opts():

    return [
        ('use_llvm', 'Use the LLVM compiler', 'no'),
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

    if (env["use_llvm"] == "yes"):
        if ('clang++' not in env['CXX']):
            env["CC"] = "clang"
            env["CXX"] = "clang++"
            env["LD"] = "clang++"
        env.Append(CPPFLAGS=['-DTYPED_METHOD_BIND'])
        env.extra_suffix = ".llvm" + env.extra_suffix

    ## Dependencies

    # FIXME: Check for existence of the libs before parsing their flags with pkg-config

    if (env['builtin_openssl'] == 'no'):
        # Currently not compatible with OpenSSL 1.1.0+
        # https://github.com/godotengine/godot/issues/8624
        import subprocess
        openssl_version = subprocess.check_output(['pkg-config', 'openssl', '--modversion']).strip('\n')
        if (openssl_version >= "1.1.0"):
            print("Error: Found system-installed OpenSSL %s, currently only supporting version 1.0.x." % openssl_version)
            print("Aborting.. You can compile with 'builtin_openssl=yes' to use the bundled version.\n")
            sys.exit(255)

        env.ParseConfig('pkg-config openssl --cflags --libs')

    if (env['builtin_libwebp'] == 'no'):
        env.ParseConfig('pkg-config libwebp --cflags --libs')

    # freetype depends on libpng and zlib, so bundling one of them while keeping others
    # as shared libraries leads to weird issues
    if (env['builtin_freetype'] == 'yes' or env['builtin_libpng'] == 'yes' or env['builtin_zlib'] == 'yes'):
        env['builtin_freetype'] = 'yes'
        env['builtin_libpng'] = 'yes'
        env['builtin_zlib'] = 'yes'

    if (env['builtin_freetype'] == 'no'):
        env.ParseConfig('pkg-config freetype2 --cflags --libs')

    if (env['builtin_libpng'] == 'no'):
        env.ParseConfig('pkg-config libpng --cflags --libs')

    if (env['builtin_enet'] == 'no'):
        env.ParseConfig('pkg-config libenet --cflags --libs')

    if (env['builtin_squish'] == 'no' and env["tools"] == "yes"):
        env.ParseConfig('pkg-config libsquish --cflags --libs')

    # Sound and video libraries
    # Keep the order as it triggers chained dependencies (ogg needed by others, etc.)

    if (env['builtin_libtheora'] == 'no'):
        env['builtin_libogg'] = 'no'  # Needed to link against system libtheora
        env['builtin_libvorbis'] = 'no'  # Needed to link against system libtheora
        env.ParseConfig('pkg-config theora theoradec --cflags --libs')

    if (env['builtin_libvpx'] == 'no'):
        env.ParseConfig('pkg-config vpx --cflags --libs')

    if (env['builtin_libvorbis'] == 'no'):
        env['builtin_libogg'] = 'no'  # Needed to link against system libvorbis
        env.ParseConfig('pkg-config vorbis vorbisfile --cflags --libs')

    if (env['builtin_opus'] == 'no'):
        env['builtin_libogg'] = 'no'  # Needed to link against system opus
        env.ParseConfig('pkg-config opus opusfile --cflags --libs')

    if (env['builtin_libogg'] == 'no'):
        env.ParseConfig('pkg-config ogg --cflags --libs')

    ## Flags

    # Linkflags below this line should typically stay the last ones
    if (env['builtin_zlib'] == 'no'):
        env.ParseConfig('pkg-config zlib --cflags --libs')

    env.Append(CPPPATH=['#platform/server'])
    env.Append(CPPFLAGS=['-DSERVER_ENABLED', '-DUNIX_ENABLED'])
    env.Append(LIBS=['pthread'])
