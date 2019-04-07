import os
import platform
import sys
from methods import get_compiler_version, using_gcc


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
        return False

    x11_error = os.system("pkg-config x11 --modversion > /dev/null ")
    if (x11_error):
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

    x11_error = os.system("pkg-config xrender --modversion > /dev/null ")
    if (x11_error):
        print("xrender not found.. x11 disabled.")
        return False

    x11_error = os.system("pkg-config xi --modversion > /dev/null ")
    if (x11_error):
        print("xi not found.. Aborting.")
        return False

    return True

def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        BoolVariable('use_llvm', 'Use the LLVM compiler', False),
        BoolVariable('use_static_cpp', 'Link libgcc and libstdc++ statically for better portability', False),
        BoolVariable('use_ubsan', 'Use LLVM/GCC compiler undefined behavior sanitizer (UBSAN)', False),
        BoolVariable('use_asan', 'Use LLVM/GCC compiler address sanitizer (ASAN))', False),
        BoolVariable('use_lsan', 'Use LLVM/GCC compiler leak sanitizer (LSAN))', False),
        BoolVariable('pulseaudio', 'Detect & use pulseaudio', True),
        BoolVariable('udev', 'Use udev for gamepad connection callbacks', False),
        EnumVariable('debug_symbols', 'Add debugging symbols to release builds', 'yes', ('yes', 'no', 'full')),
        BoolVariable('separate_debug_symbols', 'Create a separate file containing debugging symbols', False),
        BoolVariable('touch', 'Enable touch events', True),
        BoolVariable('execinfo', 'Use libexecinfo on systems where glibc is not available', False),
    ]


def get_flags():

    return [
        ('builtin_freetype', False),
        ('builtin_libpng', False),
        ('builtin_zlib', False),
    ]


def configure(env):

    ## Build type

    if (env["target"] == "release"):
        if (env["optimize"] == "speed"): #optimize for speed (default)
            env.Prepend(CCFLAGS=['-O3'])
        else: #optimize for size
            env.Prepend(CCFLAGS=['-Os'])

        if (env["debug_symbols"] == "yes"):
            env.Prepend(CCFLAGS=['-g1'])
        if (env["debug_symbols"] == "full"):
            env.Prepend(CCFLAGS=['-g2'])

    elif (env["target"] == "release_debug"):
        if (env["optimize"] == "speed"): #optimize for speed (default)
            env.Prepend(CCFLAGS=['-O2', '-DDEBUG_ENABLED'])
        else: #optimize for size
            env.Prepend(CCFLAGS=['-Os', '-DDEBUG_ENABLED'])

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

    if 'CXX' in env and 'clang' in os.path.basename(env['CXX']):
        # Convenience check to enforce the use_llvm overrides when CXX is clang(++)
        env['use_llvm'] = True

    if env['use_llvm']:
        if ('clang++' not in os.path.basename(env['CXX'])):
            env["CC"] = "clang"
            env["CXX"] = "clang++"
            env["LINK"] = "clang++"
        env.Append(CPPFLAGS=['-DTYPED_METHOD_BIND'])
        env.extra_suffix = ".llvm" + env.extra_suffix


    if env['use_ubsan'] or env['use_asan'] or env['use_lsan']:
        env.extra_suffix += "s"

        if env['use_ubsan']:
            env.Append(CCFLAGS=['-fsanitize=undefined'])
            env.Append(LINKFLAGS=['-fsanitize=undefined'])

        if env['use_asan']:
            env.Append(CCFLAGS=['-fsanitize=address'])
            env.Append(LINKFLAGS=['-fsanitize=address'])

        if env['use_lsan']:
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

    # Check for gcc version >= 6 before adding -no-pie
    if using_gcc(env):
        version = get_compiler_version(env)
        if version != None and version[0] >= '6':
            env.Append(CCFLAGS=['-fpie'])
            env.Append(LINKFLAGS=['-no-pie'])

    ## Dependencies

    env.ParseConfig('pkg-config x11 --cflags --libs')
    env.ParseConfig('pkg-config xcursor --cflags --libs')
    env.ParseConfig('pkg-config xinerama --cflags --libs')
    env.ParseConfig('pkg-config xrandr --cflags --libs')
    env.ParseConfig('pkg-config xrender --cflags --libs')
    env.ParseConfig('pkg-config xi --cflags --libs')

    if (env['touch']):
        env.Append(CPPFLAGS=['-DTOUCH_ENABLED'])

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
        if str(bullet_version) < "2.88":
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
    else:
        list_of_x86 = ['x86_64', 'x86', 'i386', 'i586']
        if any(platform.machine() in s for s in list_of_x86):
            env["x86_libtheora_opt_gcc"] = True

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

    if (os.system("pkg-config --exists alsa") == 0): # 0 means found
        print("Enabling ALSA")
        env.Append(CPPFLAGS=["-DALSA_ENABLED", "-DALSAMIDI_ENABLED"])
	# Don't parse --cflags, we don't need to add /usr/include/alsa to include path
        env.ParseConfig('pkg-config alsa --libs')
    else:
        print("ALSA libraries not found, disabling driver")

    if env['pulseaudio']:
        if (os.system("pkg-config --exists libpulse") == 0): # 0 means found
            print("Enabling PulseAudio")
            env.Append(CPPFLAGS=["-DPULSEAUDIO_ENABLED"])
            env.ParseConfig('pkg-config --cflags --libs libpulse')
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
    env.Append(CPPFLAGS=['-DX11_ENABLED', '-DUNIX_ENABLED', '-DOPENGL_ENABLED', '-DGLES_ENABLED'])
    env.Append(LIBS=['GL', 'pthread'])

    if (platform.system() == "Linux"):
        env.Append(LIBS=['dl'])

    if (platform.system().find("BSD") >= 0):
        env["execinfo"] = True

    if env["execinfo"]:
        env.Append(LIBS=['execinfo'])

    ## Cross-compilation

    if (is64 and env["bits"] == "32"):
        env.Append(CPPFLAGS=['-m32'])
        env.Append(LINKFLAGS=['-m32', '-L/usr/lib/i386-linux-gnu'])
    elif (not is64 and env["bits"] == "64"):
        env.Append(CPPFLAGS=['-m64'])
        env.Append(LINKFLAGS=['-m64', '-L/usr/lib/i686-linux-gnu'])

    # Link those statically for portability
    if env['use_static_cpp']:
        env.Append(LINKFLAGS=['-static-libgcc', '-static-libstdc++'])
