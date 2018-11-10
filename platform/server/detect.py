import os
import platform
import sys


def is_active():
    return True


def get_name():
    return "Server"


def get_program_suffix():
    if (sys.platform == "darwin"):
        return "osx"
    return "x11"


def can_build():

    if (os.name != "posix"):
        return False

    return True


def get_opts():
    from SCons.Variables import BoolVariable
    return [
        BoolVariable('use_llvm', 'Use the LLVM compiler', False),
        BoolVariable('use_static_cpp', 'Link libgcc and libstdc++ statically for better portability', False),
    ]


def get_flags():

    return []


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
        if ('clang++' not in os.path.basename(env['CXX'])):
            env["CC"] = "clang"
            env["CXX"] = "clang++"
            env["LINK"] = "clang++"
        env.Append(CPPFLAGS=['-DTYPED_METHOD_BIND'])
        env.extra_suffix = ".llvm" + env.extra_suffix

    ## Dependencies

    # FIXME: Check for existence of the libs before parsing their flags with pkg-config

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

    if not env['builtin_bullet']:
        # We need at least version 2.88
        import subprocess
        bullet_version = subprocess.check_output(['pkg-config', 'bullet', '--modversion']).strip()
        if bullet_version < "2.88":
            # Abort as system bullet was requested but too old
            print("Bullet: System version {0} does not match minimal requirements ({1}). Aborting.".format(bullet_version, "2.88"))
            sys.exit(255)
        env.ParseConfig('pkg-config bullet --cflags --libs')

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

    if not env['builtin_libwebp']:
        env.ParseConfig('pkg-config libwebp --cflags --libs')

    if not env['builtin_mbedtls']:
        # mbedTLS does not provide a pkgconfig config yet. See https://github.com/ARMmbed/mbedtls/issues/228
        env.Append(LIBS=['mbedtls', 'mbedcrypto', 'mbedx509'])

    if not env['builtin_libwebsockets']:
        env.ParseConfig('pkg-config libwebsockets --cflags --libs')

    if not env['builtin_miniupnpc']:
        # No pkgconfig file so far, hardcode default paths.
        env.Append(CPPPATH=["/usr/include/miniupnpc"])
        env.Append(LIBS=["miniupnpc"])

    # On Linux wchar_t should be 32-bits
    # 16-bit library shouldn't be required due to compiler optimisations
    if not env['builtin_pcre2']:
        env.ParseConfig('pkg-config libpcre2-32 --cflags --libs')

    ## Flags

    # Linkflags below this line should typically stay the last ones
    if not env['builtin_zlib']:
        env.ParseConfig('pkg-config zlib --cflags --libs')

    env.Append(CPPPATH=['#platform/server'])
    env.Append(CPPFLAGS=['-DSERVER_ENABLED', '-DUNIX_ENABLED'])

    if (platform.system() == "Darwin"):
        env.Append(LINKFLAGS=['-framework', 'Cocoa', '-framework', 'Carbon', '-lz', '-framework', 'IOKit'])

    env.Append(LIBS=['pthread'])

    if (platform.system() == "Linux"):
        env.Append(LIBS=['dl'])

    if (platform.system().find("BSD") >= 0):
        env.Append(LIBS=['execinfo'])

    # Link those statically for portability
    if env['use_static_cpp']:
        env.Append(LINKFLAGS=['-static-libgcc', '-static-libstdc++'])
