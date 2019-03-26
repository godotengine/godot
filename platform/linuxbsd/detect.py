import os
import platform
import sys


def is_active():
    return True


def get_name():
    return "LinuxBSD"


def can_build():
    if os.name != "posix" or sys.platform == "darwin":
        return False

    # Check the minimal dependencies
    x11_error = os.system("pkg-config --version > /dev/null")
    if x11_error:
        return False

    x11_error = os.system("pkg-config x11 --modversion > /dev/null ")
    if x11_error:
        return False

    x11_error = os.system("pkg-config xcursor --modversion > /dev/null ")
    if x11_error:
        print("xcursor not found.. x11 disabled.")
        return False

    x11_error = os.system("pkg-config xinerama --modversion > /dev/null ")
    if x11_error:
        print("xinerama not found.. x11 disabled.")
        return False

    x11_error = os.system("pkg-config xext --modversion > /dev/null ")
    if x11_error:
        print("xext not found.. x11 disabled.")
        return False

    x11_error = os.system("pkg-config xrandr --modversion > /dev/null ")
    if x11_error:
        print("xrandr not found.. x11 disabled.")
        return False

    x11_error = os.system("pkg-config xrender --modversion > /dev/null ")
    if x11_error:
        print("xrender not found.. x11 disabled.")
        return False

    x11_error = os.system("pkg-config xi --modversion > /dev/null ")
    if x11_error:
        print("xi not found.. Aborting.")
        return False

    return True


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        BoolVariable("use_llvm", "Use the LLVM compiler", False),
        BoolVariable("use_lld", "Use the LLD linker", False),
        BoolVariable("use_thinlto", "Use ThinLTO", False),
        BoolVariable("use_static_cpp", "Link libgcc and libstdc++ statically for better portability", True),
        BoolVariable("use_coverage", "Test Godot coverage", False),
        BoolVariable("use_ubsan", "Use LLVM/GCC compiler undefined behavior sanitizer (UBSAN)", False),
        BoolVariable("use_asan", "Use LLVM/GCC compiler address sanitizer (ASAN)", False),
        BoolVariable("use_lsan", "Use LLVM/GCC compiler leak sanitizer (LSAN)", False),
        BoolVariable("use_tsan", "Use LLVM/GCC compiler thread sanitizer (TSAN)", False),
        BoolVariable("use_msan", "Use LLVM compiler memory sanitizer (MSAN)", False),
        BoolVariable("pulseaudio", "Detect and use PulseAudio", True),
        BoolVariable("udev", "Use udev for gamepad connection callbacks", True),
        BoolVariable("debug_symbols", "Add debugging symbols to release/release_debug builds", True),
        BoolVariable("separate_debug_symbols", "Create a separate file containing debugging symbols", False),
        BoolVariable("touch", "Enable touch events", True),
        BoolVariable("execinfo", "Use libexecinfo on systems where glibc is not available", False),
    ]


def get_flags():
    return []


def configure(env):
    ## Build type

    if env["target"] == "release":
        if env["optimize"] == "speed":  # optimize for speed (default)
            env.Prepend(CCFLAGS=["-O3"])
        elif env["optimize"] == "size":  # optimize for size
            env.Prepend(CCFLAGS=["-Os"])

        if env["debug_symbols"]:
            env.Prepend(CCFLAGS=["-g2"])

    elif env["target"] == "release_debug":
        if env["optimize"] == "speed":  # optimize for speed (default)
            env.Prepend(CCFLAGS=["-O2"])
        elif env["optimize"] == "size":  # optimize for size
            env.Prepend(CCFLAGS=["-Os"])
        env.Prepend(CPPDEFINES=["DEBUG_ENABLED"])

        if env["debug_symbols"]:
            env.Prepend(CCFLAGS=["-g2"])

    elif env["target"] == "debug":
        env.Prepend(CCFLAGS=["-g3"])
        env.Prepend(CPPDEFINES=["DEBUG_ENABLED"])
        env.Append(LINKFLAGS=["-rdynamic"])

    ## Architecture

    is64 = sys.maxsize > 2 ** 32
    if env["bits"] == "default":
        env["bits"] = "64" if is64 else "32"

    ## Compiler configuration

    if "CXX" in env and "clang" in os.path.basename(env["CXX"]):
        # Convenience check to enforce the use_llvm overrides when CXX is clang(++)
        env["use_llvm"] = True

    if env["use_llvm"]:
        if "clang++" not in os.path.basename(env["CXX"]):
            env["CC"] = "clang"
            env["CXX"] = "clang++"
        env.extra_suffix = ".llvm" + env.extra_suffix

    if env["use_lld"]:
        if env["use_llvm"]:
            env.Append(LINKFLAGS=["-fuse-ld=lld"])
            if env["use_thinlto"]:
                # A convenience so you don't need to write use_lto too when using SCons
                env["use_lto"] = True
        else:
            print("Using LLD with GCC is not supported yet, try compiling with 'use_llvm=yes'.")
            sys.exit(255)

    if env["use_coverage"]:
        env.Append(CCFLAGS=["-ftest-coverage", "-fprofile-arcs"])
        env.Append(LINKFLAGS=["-ftest-coverage", "-fprofile-arcs"])

    if env["use_ubsan"] or env["use_asan"] or env["use_lsan"] or env["use_tsan"] or env["use_msan"]:
        env.extra_suffix += "s"

        if env["use_ubsan"]:
            env.Append(
                CCFLAGS=[
                    "-fsanitize=undefined,shift,shift-exponent,integer-divide-by-zero,unreachable,vla-bound,null,return,signed-integer-overflow,bounds,float-divide-by-zero,float-cast-overflow,nonnull-attribute,returns-nonnull-attribute,bool,enum,vptr,pointer-overflow,builtin"
                ]
            )
            env.Append(LINKFLAGS=["-fsanitize=undefined"])
            if env["use_llvm"]:
                env.Append(
                    CCFLAGS=[
                        "-fsanitize=nullability-return,nullability-arg,function,nullability-assign,implicit-integer-sign-change"
                    ]
                )
            else:
                env.Append(CCFLAGS=["-fsanitize=bounds-strict"])

        if env["use_asan"]:
            env.Append(CCFLAGS=["-fsanitize=address,pointer-subtract,pointer-compare"])
            env.Append(LINKFLAGS=["-fsanitize=address"])

        if env["use_lsan"]:
            env.Append(CCFLAGS=["-fsanitize=leak"])
            env.Append(LINKFLAGS=["-fsanitize=leak"])

        if env["use_tsan"]:
            env.Append(CCFLAGS=["-fsanitize=thread"])
            env.Append(LINKFLAGS=["-fsanitize=thread"])

        if env["use_msan"] and env["use_llvm"]:
            env.Append(CCFLAGS=["-fsanitize=memory"])
            env.Append(CCFLAGS=["-fsanitize-memory-track-origins"])
            env.Append(CCFLAGS=["-fsanitize-recover=memory"])
            env.Append(LINKFLAGS=["-fsanitize=memory"])

    if env["use_lto"]:
        if not env["use_llvm"] and env.GetOption("num_jobs") > 1:
            env.Append(CCFLAGS=["-flto"])
            env.Append(LINKFLAGS=["-flto=" + str(env.GetOption("num_jobs"))])
        else:
            if env["use_lld"] and env["use_thinlto"]:
                env.Append(CCFLAGS=["-flto=thin"])
                env.Append(LINKFLAGS=["-flto=thin"])
            else:
                env.Append(CCFLAGS=["-flto"])
                env.Append(LINKFLAGS=["-flto"])

        if not env["use_llvm"]:
            env["RANLIB"] = "gcc-ranlib"
            env["AR"] = "gcc-ar"

    env.Append(CCFLAGS=["-pipe"])
    env.Append(LINKFLAGS=["-pipe"])

    # -fpie and -no-pie is supported on GCC 6+ and Clang 4+, both below our
    # minimal requirements.
    env.Append(CCFLAGS=["-fpie"])
    env.Append(LINKFLAGS=["-no-pie"])

    ## Dependencies

    env.ParseConfig("pkg-config x11 --cflags --libs")
    env.ParseConfig("pkg-config xcursor --cflags --libs")
    env.ParseConfig("pkg-config xinerama --cflags --libs")
    env.ParseConfig("pkg-config xext --cflags --libs")
    env.ParseConfig("pkg-config xrandr --cflags --libs")
    env.ParseConfig("pkg-config xrender --cflags --libs")
    env.ParseConfig("pkg-config xi --cflags --libs")

    if env["touch"]:
        env.Append(CPPDEFINES=["TOUCH_ENABLED"])

    # FIXME: Check for existence of the libs before parsing their flags with pkg-config

    # freetype depends on libpng and zlib, so bundling one of them while keeping others
    # as shared libraries leads to weird issues
    if (
        env["builtin_freetype"]
        or env["builtin_libpng"]
        or env["builtin_zlib"]
        or env["builtin_graphite"]
        or env["builtin_harfbuzz"]
    ):
        env["builtin_freetype"] = True
        env["builtin_libpng"] = True
        env["builtin_zlib"] = True
        env["builtin_graphite"] = True
        env["builtin_harfbuzz"] = True

    if not env["builtin_freetype"]:
        env.ParseConfig("pkg-config freetype2 --cflags --libs")

    if not env["builtin_graphite"]:
        env.ParseConfig("pkg-config graphite2 --cflags --libs")

    if not env["builtin_icu"]:
        env.ParseConfig("pkg-config icu-uc --cflags --libs")

    if not env["builtin_harfbuzz"]:
        env.ParseConfig("pkg-config harfbuzz harfbuzz-icu --cflags --libs")

    if not env["builtin_libpng"]:
        env.ParseConfig("pkg-config libpng16 --cflags --libs")

    if not env["builtin_bullet"]:
        # We need at least version 2.90
        min_bullet_version = "2.90"

        import subprocess

        bullet_version = subprocess.check_output(["pkg-config", "bullet", "--modversion"]).strip()
        if str(bullet_version) < min_bullet_version:
            # Abort as system bullet was requested but too old
            print(
                "Bullet: System version {0} does not match minimal requirements ({1}). Aborting.".format(
                    bullet_version, min_bullet_version
                )
            )
            sys.exit(255)
        env.ParseConfig("pkg-config bullet --cflags --libs")

    if False:  # not env['builtin_assimp']:
        # FIXME: Add min version check
        env.ParseConfig("pkg-config assimp --cflags --libs")

    if not env["builtin_enet"]:
        env.ParseConfig("pkg-config libenet --cflags --libs")

    if not env["builtin_squish"]:
        env.ParseConfig("pkg-config libsquish --cflags --libs")

    if not env["builtin_zstd"]:
        env.ParseConfig("pkg-config libzstd --cflags --libs")

    # Sound and video libraries
    # Keep the order as it triggers chained dependencies (ogg needed by others, etc.)

    if not env["builtin_libtheora"]:
        env["builtin_libogg"] = False  # Needed to link against system libtheora
        env["builtin_libvorbis"] = False  # Needed to link against system libtheora
        env.ParseConfig("pkg-config theora theoradec --cflags --libs")
    else:
        list_of_x86 = ["x86_64", "x86", "i386", "i586"]
        if any(platform.machine() in s for s in list_of_x86):
            env["x86_libtheora_opt_gcc"] = True

    if not env["builtin_libvpx"]:
        env.ParseConfig("pkg-config vpx --cflags --libs")

    if not env["builtin_libvorbis"]:
        env["builtin_libogg"] = False  # Needed to link against system libvorbis
        env.ParseConfig("pkg-config vorbis vorbisfile --cflags --libs")

    if not env["builtin_opus"]:
        env["builtin_libogg"] = False  # Needed to link against system opus
        env.ParseConfig("pkg-config opus opusfile --cflags --libs")

    if not env["builtin_libogg"]:
        env.ParseConfig("pkg-config ogg --cflags --libs")

    if not env["builtin_libwebp"]:
        env.ParseConfig("pkg-config libwebp --cflags --libs")

    if not env["builtin_mbedtls"]:
        # mbedTLS does not provide a pkgconfig config yet. See https://github.com/ARMmbed/mbedtls/issues/228
        env.Append(LIBS=["mbedtls", "mbedcrypto", "mbedx509"])

    if not env["builtin_wslay"]:
        env.ParseConfig("pkg-config libwslay --cflags --libs")

    if not env["builtin_miniupnpc"]:
        # No pkgconfig file so far, hardcode default paths.
        env.Prepend(CPPPATH=["/usr/include/miniupnpc"])
        env.Append(LIBS=["miniupnpc"])

    # On Linux wchar_t should be 32-bits
    # 16-bit library shouldn't be required due to compiler optimisations
    if not env["builtin_pcre2"]:
        env.ParseConfig("pkg-config libpcre2-32 --cflags --libs")

    if not env["builtin_embree"]:
        # No pkgconfig file so far, hardcode expected lib name.
        env.Append(LIBS=["embree3"])

    ## Flags

    if os.system("pkg-config --exists alsa") == 0:  # 0 means found
        print("Enabling ALSA")
        env["alsa"] = True
        env.Append(CPPDEFINES=["ALSA_ENABLED", "ALSAMIDI_ENABLED"])
    else:
        print("ALSA libraries not found, disabling driver")

    if env["pulseaudio"]:
        if os.system("pkg-config --exists libpulse") == 0:  # 0 means found
            print("Enabling PulseAudio")
            env.Append(CPPDEFINES=["PULSEAUDIO_ENABLED"])
            env.ParseConfig("pkg-config --cflags libpulse")
        else:
            print("PulseAudio development libraries not found, disabling driver")

    if platform.system() == "Linux":
        env.Append(CPPDEFINES=["JOYDEV_ENABLED"])
        if env["udev"]:
            if os.system("pkg-config --exists libudev") == 0:  # 0 means found
                print("Enabling udev support")
                env.Append(CPPDEFINES=["UDEV_ENABLED"])
            else:
                print("libudev development libraries not found, disabling udev support")
    else:
        env["udev"] = False  # Linux specific

    # Linkflags below this line should typically stay the last ones
    if not env["builtin_zlib"]:
        env.ParseConfig("pkg-config zlib --cflags --libs")

    env.Prepend(CPPPATH=["#platform/linuxbsd"])
    env.Append(CPPDEFINES=["X11_ENABLED", "UNIX_ENABLED"])
    env.Append(CPPDEFINES=[("_FILE_OFFSET_BITS", 64)])

    env.Append(CPPDEFINES=["VULKAN_ENABLED"])
    if not env["builtin_vulkan"]:
        env.ParseConfig("pkg-config vulkan --cflags --libs")
    if not env["builtin_glslang"]:
        # No pkgconfig file for glslang so far
        env.Append(LIBS=["glslang", "SPIRV"])

    # env.Append(CPPDEFINES=['OPENGL_ENABLED'])
    env.Append(LIBS=["GL"])

    env.Append(LIBS=["pthread"])

    if platform.system() == "Linux":
        env.Append(LIBS=["dl"])

    if platform.system().find("BSD") >= 0:
        env["execinfo"] = True

    if env["execinfo"]:
        env.Append(LIBS=["execinfo"])

    if not env["tools"]:
        import subprocess
        import re

        linker_version_str = subprocess.check_output([env.subst(env["LINK"]), "-Wl,--version"]).decode("utf-8")
        gnu_ld_version = re.search("^GNU ld [^$]*(\d+\.\d+)$", linker_version_str, re.MULTILINE)
        if not gnu_ld_version:
            print(
                "Warning: Creating template binaries enabled for PCK embedding is currently only supported with GNU ld"
            )
        else:
            if float(gnu_ld_version.group(1)) >= 2.30:
                env.Append(LINKFLAGS=["-T", "platform/linuxbsd/pck_embed.ld"])
            else:
                env.Append(LINKFLAGS=["-T", "platform/linuxbsd/pck_embed.legacy.ld"])

    ## Cross-compilation

    if is64 and env["bits"] == "32":
        env.Append(CCFLAGS=["-m32"])
        env.Append(LINKFLAGS=["-m32", "-L/usr/lib/i386-linux-gnu"])
    elif not is64 and env["bits"] == "64":
        env.Append(CCFLAGS=["-m64"])
        env.Append(LINKFLAGS=["-m64", "-L/usr/lib/i686-linux-gnu"])

    # Link those statically for portability
    if env["use_static_cpp"]:
        env.Append(LINKFLAGS=["-static-libgcc", "-static-libstdc++"])
        if env["use_llvm"]:
            env["LINKCOM"] = env["LINKCOM"] + " -l:libatomic.a"

    else:
        if env["use_llvm"]:
            env.Append(LIBS=["atomic"])
