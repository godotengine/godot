import os
import platform
import sys


def is_active():
    return True


def get_name():
    return "SailfishOS"


def can_build():

    if (os.name != "posix" or sys.platform == "darwin"):
        return False

    # Check the minimal dependencies
    sdl_error = os.system("pkg-config --version > /dev/null")
    if (sdl_error):
        print("pkg-config not found...")
        return False

    sdl_error = os.system("pkg-config sdl2 --modversion > /dev/null ")
    if (sdl_error):
        print("SDL2 not found. Sailfish build disabled. Install SDL2-devel for all your targets in MerSDK ")
        return False

    ar_error = os.system("pkg-config audioresource --modversion > /dev/null")
    if(ar_error):
        print("libaudioresource-devel not found. Install libaudioresource-devel for all your targets in MerSDK")
        # return False;

    glib_error = os.system("pkg-config glib-2.0 --modversion > /dev/null")
    if(glib_error):
        print("glib2-devel not found. Install glib2-devel for all your targets in MerSDK")
        # return False;

    vpx_error = os.system("pkg-config vpx --modversion > /dev/null")
    if(vpx_error):
        print("libvpx-devel not found. Install libvpx-devel for all your targets in MerSDK")
        return False;

    webp_error = os.system("pkg-config libwebp --modversion > /dev/null")
    if(webp_error):
        print("libwebp-devel not found. Install libwebp-devel for all your targets in MerSDK\n")
        return False;

    return True

def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        BoolVariable('use_llvm', 'Use the LLVM compiler', False),
        BoolVariable('use_static_cpp', 'Link stdc++ statically', False),
        BoolVariable('use_sanitizer', 'Use LLVM compiler address sanitizer', False),
        BoolVariable('use_leak_sanitizer', 'Use LLVM compiler memory leaks sanitizer (implies use_sanitizer)', False),
        BoolVariable('pulseaudio', 'Detect & use pulseaudio', True),
        BoolVariable('udev', 'Use udev for gamepad connection callbacks', False),
        EnumVariable('debug_symbols', 'Add debug symbols to release version', 'no', ('yes', 'no', 'full')),
        BoolVariable('separate_debug_symbols', 'Create a separate file with the debug symbols', False),
        BoolVariable('touch', 'Enable touch events', True),
        BoolVariable('tools', 'Enable editor tools', False),
    ]


def get_flags():

    return [
        ('builtin_freetype', False),
        ('builtin_libpng', False),
        ('builtin_openssl', False),
        ('builtin_zlib', False),
        ('builtin_libvpx', False),
        ('builtin_libwebp', False)
    ]


def configure(env):

    ## Build type

    if (env["target"] == "release"):
        # -O3 -ffast-math is identical to -Ofast. We need to split it out so we can selectively disable
        # -ffast-math in code for which it generates wrong results.
        env.Prepend(CCFLAGS=['-O3', '-ffast-math', '-DGLES_ENABLED'])
        if (env["debug_symbols"] == "yes"):
            env.Prepend(CCFLAGS=['-g1'])
        if (env["debug_symbols"] == "full"):
            env.Prepend(CCFLAGS=['-g2'])

    elif (env["target"] == "release_debug"):
        env.Prepend(CCFLAGS=['-O2', '-ffast-math', '-DDEBUG_ENABLED', '-DGLES_ENABLED'])
        if (env["debug_symbols"] == "yes"):
            env.Prepend(CCFLAGS=['-g1'])
        if (env["debug_symbols"] == "full"):
            env.Prepend(CCFLAGS=['-g2'])

    elif (env["target"] == "debug"):
        env.Prepend(CCFLAGS=['-g3', '-DDEBUG_ENABLED', '-DDEBUG_MEMORY_ENABLED', '-DGLES_ENABLED'])
        env.Append(LINKFLAGS=['-rdynamic'])

    ## Architecture

    is64 = sys.maxsize > 2**32
    if (env["bits"] == "default"):
        env["bits"] = "64" if is64 else "32"

    ## Compiler configuration

    if 'CXX' in env and 'clang' in env['CXX']:
        # Convenience check to enforce the use_llvm overrides when CXX is clang(++)
        env['use_llvm'] = True

    if env['use_llvm']:
        if ('clang++' not in env['CXX']):
            env["CC"] = "clang"
            env["CXX"] = "clang++"
            env["LINK"] = "clang++"
        env.Append(CPPFLAGS=['-DTYPED_METHOD_BIND', '-DGLES_ENABLED'])
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
        if not env['use_llvm'] and env.GetOption("num_jobs") > 1:
            env.Append(LINKFLAGS=['-flto=' + str(env.GetOption("num_jobs"))])
        else:
            env.Append(LINKFLAGS=['-flto'])
        if not env['use_llvm']:
            env['RANLIB'] = 'gcc-ranlib'
            env['AR'] = 'gcc-ar'

    env.Append(CCFLAGS=['-pipe'])
    env.Append(LINKFLAGS=['-pipe'])

    ## Dependencies

    env.ParseConfig('pkg-config sdl2 --cflags --libs')
    env.ParseConfig('pkg-config wayland-client --cflags --libs')
    ar_error = os.system("pkg-config audioresource --modversion > /dev/null")
    if(ar_error):
        env.Prepend(CCFLAGS=['-DDISABLE_LIBAUDIORESOURCE'])
    else:
        env.ParseConfig("pkg-config audioresource --cflags --libs")
        env.ParseConfig("pkg-config glib-2.0 --cflags --libs")

    if (env['touch']):
        env.Append(CPPFLAGS=['-DTOUCH_ENABLED'])

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

    if env['builtin_libtheora']:
        list_of_x86 = ['x86_64', 'x86', 'i386', 'i586']
        if any(platform.machine() in s for s in list_of_x86):
            env["x86_libtheora_opt_gcc"] = True

    # On Linux wchar_t should be 32-bits
    # 16-bit library shouldn't be required due to compiler optimisations
    if not env['builtin_pcre2']:
        env.ParseConfig('pkg-config libpcre2-32 --cflags --libs')

    ## Flags
    # if env['alsa']:
    #     if (os.system("pkg-config --exists alsa") == 0): # 0 means found
    #         print("Enabling ALSA")
    #         env.Append(CPPFLAGS=["-DALSA_ENABLED"])
    #         env.ParseConfig('pkg-config alsa --cflags --libs')
    #     else:
    #         print("ALSA libraries not found, disabling driver")

    if env['pulseaudio']:
        if (os.system("pkg-config --exists libpulse-simple") == 0): # 0 means found
            print("Enabling PulseAudio")
            env.Append(CPPFLAGS=["-DPULSEAUDIO_ENABLED"])
            env.ParseConfig('pkg-config --cflags --libs libpulse-simple')
        else:
            print("PulseAudio development libraries not found, disabling driver")

    if (platform.system() == "Linux"):
        env.Append(CPPFLAGS=["-DJOYDEV_ENABLED", '-DGLES_ENABLED'])

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

    env.Append(CPPPATH=['#platform/sailfish','#core', '#thirdparty/glad', '#platform/sailfish/SDL_src/include', '#platform/sailfish/SDL_src/src'])
    # env.Append(CPPFLAGS=['-DSDL_ENABLED', '-DUNIX_ENABLED', '-DOPENGL_ENABLED', '-DGLES_ENABLED', '-DGLES_OVER_ls -lGL'])
    env.Append(CPPFLAGS=['-DSDL_ENABLED', '-DUNIX_ENABLED', '-DGLES_ENABLED', '-DGLES2_ENABLED', '-Wno-strict-aliasing'])
    env.Append(CPPFLAGS=['-DSAILFISH_FORCE_LANDSCAPE'])
    # include paths for different versions of SailfishSDK width different SDL2 version  
    env.Append(CPPFLAGS=['-I/usr/src/debug/SDL2-2.0.3-1.2.3.jolla.i386/src/','-I/usr/src/debug/SDL2-2.0.3-1.2.1.jolla.i386/include'])
    env.Append(CPPFLAGS=['-I/usr/src/debug/SDL2-2.0.3-1.2.3.jolla.arm/src/','-I/usr/src/debug/SDL2-2.0.3-1.2.1.jolla.arm/include'])
    env.Append(CPPFLAGS=['-I/usr/src/debug/SDL2-2.0.3-1.3.1.jolla.i386/src/','-I/usr/src/debug/SDL2-2.0.3-1.3.1.jolla.i386/include'])
    env.Append(CPPFLAGS=['-I/usr/src/debug/SDL2-2.0.3-1.3.1.jolla.arm/src/','-I/usr/src/debug/SDL2-2.0.3-1.3.1.jolla.arm/include'])
    env.Append(CPPFLAGS=['-I/usr/src/debug/SDL2-2.0.3-1.3.2.jolla.i386/src/','-I/usr/src/debug/SDL2-2.0.3-1.3.2.jolla.i386/include'])
    env.Append(CPPFLAGS=['-I/usr/src/debug/SDL2-2.0.3-1.3.2.jolla.arm/src/','-I/usr/src/debug/SDL2-2.0.3-1.3.2.jolla.arm/include'])
    env.Append(CPPFLAGS=['-I/usr/src/debug/SDL2-2.0.3-1.3.2.jolla.arm/src/','-I/usr/src/debug/SDL2-2.0.3-1.3.2.jolla.arm/include'])
    # env.Append(LIBS=['GL', 'pthread'])
    env.Append(LIBS=['GLESv2', 'EGL', 'pthread'])

    if (platform.system() == "Linux"):
        env.Append(LIBS=['dl'])

    if (platform.system().find("BSD") >= 0):
        env.Append(LIBS=['execinfo'])

    ## Cross-compilation

    if (is64 and env["bits"] == "32"):
        env.Append(CPPFLAGS=['-m32'])
        env.Append(LINKFLAGS=['-m32', '-L/usr/lib/i386-linux-gnu'])
    elif (not is64 and env["bits"] == "64"):
        env.Append(CPPFLAGS=['-m64'])
        env.Append(LINKFLAGS=['-m64', '-L/usr/lib/i686-linux-gnu'])


    if env['use_static_cpp']:
        env.Append(LINKFLAGS=['-static-libstdc++'])
