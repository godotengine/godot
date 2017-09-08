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

    return [
        ('use_llvm', 'Use the LLVM compiler', 'no'),
        ('use_static_cpp', 'Link stdc++ statically', 'no'),
        ('use_sanitizer', 'Use LLVM compiler address sanitizer', 'no'),
        ('use_leak_sanitizer', 'Use LLVM compiler memory leaks sanitizer (implies use_sanitizer)', 'no'),
        ('use_lto', 'Use link time optimization', 'no'),
        ('pulseaudio', 'Detect & use pulseaudio', 'yes'),
        ('udev', 'Use udev for gamepad connection callbacks', 'no'),
        ('debug_release', 'Add debug symbols to release version', 'no'),
    ]


def get_flags():

    return [
        ('builtin_freetype', 'no'),
        ('builtin_libpng', 'no'),
        ('builtin_openssl', 'no'),
        ('builtin_zlib', 'no'),
    ]


def configure(env):

    ## Build type

    if (env["target"] == "release"):
        # -O3 -ffast-math is identical to -Ofast. We need to split it out so we can selectively disable
        # -ffast-math in code for which it generates wrong results.
        env.Prepend(CCFLAGS=['-O3', '-ffast-math'])
        if (env["debug_release"] == "yes"):
            env.Prepend(CCFLAGS=['-g2'])

    elif (env["target"] == "release_debug"):
        env.Prepend(CCFLAGS=['-O2', '-ffast-math', '-DDEBUG_ENABLED'])
        if (env["debug_release"] == "yes"):
            env.Prepend(CCFLAGS=['-g2'])

    elif (env["target"] == "debug"):
        env.Prepend(CCFLAGS=['-g2', '-DDEBUG_ENABLED', '-DDEBUG_MEMORY_ENABLED'])
        env.Append(LINKFLAGS=['-rdynamic'])

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

    # leak sanitizer requires (address) sanitizer
    if (env["use_sanitizer"] == "yes" or env["use_leak_sanitizer"] == "yes"):
        env.Append(CCFLAGS=['-fsanitize=address', '-fno-omit-frame-pointer'])
        env.Append(LINKFLAGS=['-fsanitize=address'])
        env.extra_suffix += "s"
        if (env["use_leak_sanitizer"] == "yes"):
            env.Append(CCFLAGS=['-fsanitize=leak'])
            env.Append(LINKFLAGS=['-fsanitize=leak'])

    if (env["use_lto"] == "yes"):
        env.Append(CCFLAGS=['-flto'])
        env.Append(LINKFLAGS=['-flto'])

    env.Append(CCFLAGS=['-pipe'])
    env.Append(LINKFLAGS=['-pipe'])

    ## Dependencies

    env.ParseConfig('pkg-config x11 --cflags --libs')
    env.ParseConfig('pkg-config xcursor --cflags --libs')
    env.ParseConfig('pkg-config xinerama --cflags --libs')
    env.ParseConfig('pkg-config xrandr --cflags --libs')

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

    if (env['builtin_libtheora'] != 'no'):
        list_of_x86 = ['x86_64', 'x86', 'i386', 'i586']
        if any(platform.machine() in s for s in list_of_x86):
            env["x86_libtheora_opt_gcc"] = True

    # On Linux wchar_t should be 32-bits
    # 16-bit library shouldn't be required due to compiler optimisations
    if (env['builtin_pcre2'] == 'no'):
        env.ParseConfig('pkg-config libpcre2-32 --cflags --libs')

    ## Flags

    if (os.system("pkg-config --exists alsa") == 0): # 0 means found
        print("Enabling ALSA")
        env.Append(CPPFLAGS=["-DALSA_ENABLED"])
        env.ParseConfig('pkg-config alsa --cflags --libs')
    else:
        print("ALSA libraries not found, disabling driver")

    if (env["pulseaudio"] == "yes"):
        if (os.system("pkg-config --exists libpulse-simple") == 0): # 0 means found
            print("Enabling PulseAudio")
            env.Append(CPPFLAGS=["-DPULSEAUDIO_ENABLED"])
            env.ParseConfig('pkg-config --cflags --libs libpulse-simple')
        else:
            print("PulseAudio development libraries not found, disabling driver")

    if (platform.system() == "Linux"):
        env.Append(CPPFLAGS=["-DJOYDEV_ENABLED"])

        if (env["udev"] == "yes"):
            if (os.system("pkg-config --exists libudev") == 0): # 0 means found
                print("Enabling udev support")
                env.Append(CPPFLAGS=["-DUDEV_ENABLED"])
                env.ParseConfig('pkg-config libudev --cflags --libs')
            else:
                print("libudev development libraries not found, disabling udev support")

    # Linkflags below this line should typically stay the last ones
    if (env['builtin_zlib'] == 'no'):
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

    if (env["use_static_cpp"] == "yes"):
        env.Append(LINKFLAGS=['-static-libstdc++'])
