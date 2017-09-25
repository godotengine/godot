import os
import platform
import sys


def is_active():
    return True


def get_name():
    return "X11"


def can_build():

    if (os.name != "posix" or sys.platform == "darwin"):
        return False

    # Check the minimal dependencies
    x11_error = os.system("pkg-config --version > /dev/null")
    if (x11_error):
        print("pkg-config not found.. x11 disabled.")
        return False

    x11_error = os.system("pkg-config x11 --modversion > /dev/null ")
    if (x11_error):
        print("X11 not found.. x11 disabled.")
        return False

    x11_error = os.system("pkg-config xcursor --modversion > /dev/null ")
    if (x11_error):
        print("xcursor not found.. x11 disabled.")
        return False

    x11_error = os.system("pkg-config xinerama --modversion > /dev/null ")
    if (x11_error):
        print("xinerama not found.. x11 disabled.")
        return False

    x11_error = os.system("pkg-config xrandr --modversion > /dev/null ")
    if (x11_error):
        print("xrandr not found.. x11 disabled.")
        return False

    return True

def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        BoolVariable('use_llvm', 'Use the LLVM compiler', False),
        BoolVariable('use_static_cpp', 'Link stdc++ statically', False),
        BoolVariable('use_sanitizer', 'Use LLVM compiler address sanitizer', False),
        BoolVariable('use_leak_sanitizer', 'Use LLVM compiler memory leaks sanitizer (implies use_sanitizer)', False),
        BoolVariable('use_lto', 'Use link time optimization', False),
        BoolVariable('pulseaudio', 'Detect & use pulseaudio', True),
        BoolVariable('udev', 'Use udev for gamepad connection callbacks', False),
        EnumVariable('debug_symbols', 'Add debug symbols to release version', 'yes', ('yes', 'no', 'full')),
    ]


def get_flags():

    return [
        ('builtin_freetype', False),
        ('builtin_libpng', False),
        ('builtin_openssl', False),
        ('builtin_zlib', False),
    ]


def configure(env):

    ## Build type

    if (env["target"] == "release"):
        # -O3 -ffast-math is identical to -Ofast. We need to split it out so we can selectively disable
        # -ffast-math in code for which it generates wrong results.
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
        env.Append(LINKFLAGS=['-rdynamic'])

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

    # leak sanitizer requires (address) sanitizer
    if env['use_sanitizer'] or env['use_leak_sanitizer']:
        env.Append(CCFLAGS=['-fsanitize=address', '-fno-omit-frame-pointer'])
        env.Append(LINKFLAGS=['-fsanitize=address'])
        env.extra_suffix += "s"
        if env['use_leak_sanitizer']:
            env.Append(CCFLAGS=['-fsanitize=leak'])
            env.Append(LINKFLAGS=['-fsanitize=leak'])

    if env['use_lto']:
        env.Append(CCFLAGS=['-flto'])
        env.Append(LINKFLAGS=['-flto'])
        if not env['use_llvm']:
            env['RANLIB'] = 'gcc-ranlib'
            env['AR'] = 'gcc-ar'

    env.Append(CCFLAGS=['-pipe'])
    env.Append(LINKFLAGS=['-pipe'])

    ## Dependencies

    env.ParseConfig('pkg-config x11 --cflags --libs')
    env.ParseConfig('pkg-config xcursor --cflags --libs')
    env.ParseConfig('pkg-config xinerama --cflags --libs')
    env.ParseConfig('pkg-config xrandr --cflags --libs')

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

    if env['builtin_libtheora']:
        list_of_x86 = ['x86_64', 'x86', 'i386', 'i586']
        if any(platform.machine() in s for s in list_of_x86):
            env["x86_libtheora_opt_gcc"] = True

    # On Linux wchar_t should be 32-bits
    # 16-bit library shouldn't be required due to compiler optimisations
    if not env['builtin_pcre2']:
        env.ParseConfig('pkg-config libpcre2-32 --cflags --libs')

    ## Flags

    if (os.system("pkg-config --exists alsa") == 0): # 0 means found
        print("Enabling ALSA")
        env.Append(CPPFLAGS=["-DALSA_ENABLED"])
        env.ParseConfig('pkg-config alsa --cflags --libs')
    else:
        print("ALSA libraries not found, disabling driver")

    if env['pulseaudio']:
        if (os.system("pkg-config --exists libpulse-simple") == 0): # 0 means found
            print("Enabling PulseAudio")
            env.Append(CPPFLAGS=["-DPULSEAUDIO_ENABLED"])
            env.ParseConfig('pkg-config --cflags --libs libpulse-simple')
        else:
            print("PulseAudio development libraries not found, disabling driver")

    if (platform.system() == "Linux"):
        env.Append(CPPFLAGS=["-DJOYDEV_ENABLED"])

        if env['udev']:
            if (os.system("pkg-config --exists libudev") == 0): # 0 means found
                print("Enabling udev support")
                env.Append(CPPFLAGS=["-DUDEV_ENABLED"])
                env.ParseConfig('pkg-config libudev --cflags --libs')
            else:
                print("libudev development libraries not found, disabling udev support")

    # Linkflags below this line should typically stay the last ones
    if not env['builtin_zlib']:
        env.ParseConfig('pkg-config zlib --cflags --libs')

    env.Append(CPPPATH=['#platform/x11'])
    env.Append(CPPFLAGS=['-DX11_ENABLED', '-DUNIX_ENABLED', '-DOPENGL_ENABLED', '-DGLES2_ENABLED', '-DGLES_OVER_GL'])
    env.Append(LIBS=['GL', 'pthread'])

    if (platform.system() == "Linux"):
        env.Append(LIBS=['dl'])

    ## Cross-compilation

    if (is64 and env["bits"] == "32"):
        env.Append(CPPFLAGS=['-m32'])
        env.Append(LINKFLAGS=['-m32', '-L/usr/lib/i386-linux-gnu'])
    elif (not is64 and env["bits"] == "64"):
        env.Append(CPPFLAGS=['-m64'])
        env.Append(LINKFLAGS=['-m64', '-L/usr/lib/i686-linux-gnu'])

    if env['use_static_cpp']:
        env.Append(LINKFLAGS=['-static-libstdc++'])
