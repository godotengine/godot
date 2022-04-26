import os
import platform
import sys
from methods import get_compiler_version, using_gcc, using_clang


def is_active():
    return True


def get_name():
    return "X11"


def can_build():
    if os.name != "posix" or sys.platform == "darwin":
        return False

    # Check the minimal dependencies
    error = os.system("pkg-config --version > /dev/null")
    if error:
        print("Error: pkg-config not found. Aborting.")
        return False

    x11_error = os.system("pkg-config x11 --exists")
    if x11_error:
        print("Error: libx11-dev library not found. Aborting.")
    xcursor_error = os.system("pkg-config xcursor --exists")
    if xcursor_error:
        print("Error: libxcursor-dev library not found. Aborting.")
    xinerama_error = os.system("pkg-config xinerama --exists")
    if xinerama_error:
        print("Error: libxinerama-dev library not found. Aborting.")
    xext_error = os.system("pkg-config xext --exists")
    if xext_error:
        print("Error: libxext-dev library not found. Aborting.")
    xrandr_error = os.system("pkg-config xrandr --exists")
    if xrandr_error:
        print("Error: libxrandr-dev library not found. Aborting.")
    xrender_error = os.system("pkg-config xrender --exists")
    if xrender_error:
        print("Error: libxrender-dev library not found. Aborting.")
    xi_error = os.system("pkg-config xi --exists")
    if xi_error:
        print("Error: libxi-dev library not found. Aborting.")

    if x11_error or xcursor_error or xinerama_error or xext_error or xrandr_error or xrender_error or xi_error:
        return False

    return True


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        BoolVariable("use_llvm", "Use the LLVM compiler", False),
        BoolVariable("use_lld", "Use the LLD linker", False),
        BoolVariable("use_thinlto", "Use ThinLTO", False),
        BoolVariable("use_static_cpp", "Link libgcc and libstdc++ statically for better portability", True),
        BoolVariable("use_ubsan", "Use LLVM/GCC compiler undefined behavior sanitizer (UBSAN)", False),
        BoolVariable("use_asan", "Use LLVM/GCC compiler address sanitizer (ASAN))", False),
        BoolVariable("use_lsan", "Use LLVM/GCC compiler leak sanitizer (LSAN))", False),
        BoolVariable("use_tsan", "Use LLVM/GCC compiler thread sanitizer (TSAN))", False),
        BoolVariable("use_msan", "Use LLVM/GCC compiler memory sanitizer (MSAN))", False),
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

        if env["debug_symbols"]:
            env.Prepend(CCFLAGS=["-g2"])

    elif env["target"] == "debug":
        env.Prepend(CCFLAGS=["-ggdb"])
        env.Prepend(CCFLAGS=["-g3"])
        env.Append(LINKFLAGS=["-rdynamic"])

    ## Architecture

    is64 = sys.maxsize > 2 ** 32
    if env["bits"] == "default":
        env["bits"] = "64" if is64 else "32"

    machines = {
        "riscv64": "rv64",
        "ppc64le": "ppc64",
        "ppc64": "ppc64",
        "ppcle": "ppc",
        "ppc": "ppc",
    }

    if env["arch"] == "" and platform.machine() in machines:
        env["arch"] = machines[platform.machine()]

    if env["arch"] == "rv64":
        # G = General-purpose extensions, C = Compression extension (very common).
        env.Append(CCFLAGS=["-march=rv64gc"])

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
            print("Using LLD with GCC is not supported yet. Try compiling with 'use_llvm=yes'.")
            sys.exit(255)

    if env["use_ubsan"] or env["use_asan"] or env["use_lsan"] or env["use_tsan"] or env["use_msan"]:
        env.extra_suffix += "s"

        if env["use_ubsan"]:
            env.Append(
                CCFLAGS=[
                    "-fsanitize=undefined,shift,shift-exponent,integer-divide-by-zero,unreachable,vla-bound,null,return,signed-integer-overflow,bounds,float-divide-by-zero,float-cast-overflow,nonnull-attribute,returns-nonnull-attribute,bool,enum,vptr,pointer-overflow,builtin"
                ]
            )

            if env["use_llvm"]:
                env.Append(
                    CCFLAGS=[
                        "-fsanitize=nullability-return,nullability-arg,function,nullability-assign,implicit-integer-sign-change,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation"
                    ]
                )
            else:
                env.Append(CCFLAGS=["-fsanitize=bounds-strict"])
        env.Append(LINKFLAGS=["-fsanitize=undefined"])

        if env["use_asan"]:
            env.Append(CCFLAGS=["-fsanitize=address,pointer-subtract,pointer-compare"])
            env.Append(LINKFLAGS=["-fsanitize=address"])

        if env["use_lsan"]:
            env.Append(CCFLAGS=["-fsanitize=leak"])
            env.Append(LINKFLAGS=["-fsanitize=leak"])

        if env["use_tsan"]:
            env.Append(CCFLAGS=["-fsanitize=thread"])
            env.Append(LINKFLAGS=["-fsanitize=thread"])

        if env["use_msan"]:
            env.Append(CCFLAGS=["-fsanitize=memory"])
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

    # Check for gcc version >= 6 before adding -no-pie
    version = get_compiler_version(env) or [-1, -1]
    if using_gcc(env):
        if version[0] >= 6:
            env.Append(CCFLAGS=["-fpie"])
            env.Append(LINKFLAGS=["-no-pie"])
    # Do the same for clang should be fine with Clang 4 and higher
    if using_clang(env):
        if version[0] >= 4:
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

    # freetype depends on libpng, which depends on zlib.
    # Bundling one of them while keeping others as shared libraries leads to library mismatch issues.
    if not env["builtin_freetype"] and not env["builtin_libpng"] and not env["builtin_zlib"]:
        freetype_error = os.system("pkg-config freetype2 --exists")
        if freetype_error:
            print("Error: libfreetype-dev library not found.")
        libpng_error = os.system("pkg-config libpng16 --exists")
        if libpng_error:
            print("Error: libpng-dev library not found.")
        zlib_error = os.system("pkg-config zlib --exists")
        if zlib_error:
            print("Error: libz-dev library not found.")

        if freetype_error or libpng_error or zlib_error:
            print("Using builtin freetype, libpng and zlib.")
            env["builtin_freetype"] = True
            env["builtin_libpng"] = True
            env["builtin_zlib"] = True
        else:
            env.ParseConfig("pkg-config freetype2 libpng16 zlib --cflags --libs")

    elif not env["builtin_freetype"] or not env["builtin_libpng"] or not env["builtin_zlib"]:
        print("freetype depends on libpng which depends on zlib.")
        print("Bundling one of them while keeping others as shared libraries leads to library")
        print("mismatch issues.")
        print("Using builtin freetype, libpng and zlib.")
        env["builtin_freetype"] = True
        env["builtin_libpng"] = True
        env["builtin_zlib"] = True

    if not env["builtin_bullet"]:
        # We need at least version 2.90
        error = os.system("pkg-config bullet --atleast-version='2.90'")
        if error:
            print("Error: libbullet-dev version 2.90 or higher not found. Using builtin bullet")
            env["builtin_bullet"] = True
        else:
            env.ParseConfig("pkg-config bullet --cflags --libs")

    if not env["builtin_enet"]:
        error = os.system("pkg-config libenet --exists")
        if error:
            print("Error: libenet-dev library not found. Using builtin enet.")
            env["builtin_enet"] = True
        else:
            env.ParseConfig("pkg-config libenet --cflags --libs")

    if not env["builtin_squish"]:
        error = os.system("pkg-config libsquish --exists")
        if error:
            print("Error: libsquish-dev library not found. Using builtin squish.")
            env["builtin_squish"] = True
        else:
            env.ParseConfig("pkg-config libsquish --cflags --libs")

    if not env["builtin_zstd"]:
        error = os.system("pkg-config libzstd --exists")
        if error:
            print("Error: libzstd-dev library not found. Using builtin zstd.")
            env["builtin_zstd"] = True
        else:
            env.ParseConfig("pkg-config libzstd --cflags --libs")

    # Sound and video libraries
    # opus, libtheora and libvorbis depend on libogg.
    # Bundling one of them while keeping others as shared libraries leads to library mismatch issues.
    if (
        not env["builtin_libogg"]
        and not env["builtin_opus"]
        and not env["builtin_libtheora"]
        and not env["builtin_libvorbis"]
    ):
        libogg_error = os.system("pkg-config ogg --exists")
        if libogg_error:
            print("Error: libogg-dev library not found.")
        libopus_error = os.system("pkg-config opus --exists")
        if libopus_error:
            print("Error: libopus-dev library not found.")
        libopusfile_error = os.system("pkg-config opusfile --exists")
        if libopusfile_error:
            print("Error: libopusfile-dev library not found.")
        libtheora_error = os.system("pkg-config theora theoradec --exists")
        if libtheora_error:
            print("Error: libtheora-dev library not found.")
        libvorbis_error = os.system("pkg-config vorbis vorbisfile --exists")
        if libvorbis_error:
            print("Error: libvorbis-dev library not found.")

        if libogg_error or libopus_error or libopusfile_error or libtheora_error or libvorbis_error:
            print("Using builtin libogg, opus, libtheora and libvorbis.")
            env["builtin_libogg"] = True
            env["builtin_opus"] = True
            env["builtin_libtheora"] = True
            env["builtin_libvorbis"] = True
        else:
            env.ParseConfig("pkg-config ogg opus opusfile theora theoradec vorbis vorbisfile --cflags --libs")

    elif (
        not env["builtin_libogg"]
        or not env["builtin_opus"]
        or not env["builtin_libtheora"]
        or not env["builtin_libvorbis"]
    ):
        print("opus, libtheora and libvorbis depend on libogg.")
        print("Bundling one of them while keeping others as shared libraries leads to library")
        print("mismatch issues.")
        print("Using builtin libogg, opus, libtheora and libvorbis.")
        env["builtin_libogg"] = True
        env["builtin_opus"] = True
        env["builtin_libtheora"] = True
        env["builtin_libvorbis"] = True

    if env["builtin_libtheora"]:
        list_of_x86 = ["x86_64", "x86", "i386", "i586"]
        if any(platform.machine() in s for s in list_of_x86):
            env["x86_libtheora_opt_gcc"] = True

    if not env["builtin_libvpx"]:
        error = os.system("pkg-config vpx --exists")
        if error:
            print("Error: libvpx-dev library not found. Using builtin libvpx.")
            env["builtin_libvpx"] = True
        else:
            env.ParseConfig("pkg-config vpx --cflags --libs")

    if not env["builtin_libwebp"]:
        error = os.system("pkg-config libwebp --exists")
        if error:
            print("Error: libwebp-dev library not found. Using builtin libwebp.")
            env["builtin_libwebp"] = True
        else:
            env.ParseConfig("pkg-config libwebp --cflags --libs")

    if not env["builtin_mbedtls"]:
        # mbedTLS does not provide a pkgconfig config yet. See https://github.com/ARMmbed/mbedtls/issues/228
        env.Append(LIBS=["mbedtls", "mbedcrypto", "mbedx509"])

    if not env["builtin_wslay"]:
        error = os.system("pkg-config libwslay --exists")
        if error:
            print("Error: libwslay-dev library not found. Using builtin wslay.")
            env["builtin_wslay"] = True
        else:
            env.ParseConfig("pkg-config libwslay --cflags --libs")

    if not env["builtin_miniupnpc"]:
        # No pkgconfig file so far, hardcode default paths.
        env.Prepend(CPPPATH=["/usr/include/miniupnpc"])
        env.Append(LIBS=["miniupnpc"])

    # On Linux wchar_t should be 32-bits
    # 16-bit library shouldn't be required due to compiler optimisations
    if not env["builtin_pcre2"]:
        error = os.system("pkg-config libpcre2-32 --exists")
        if error:
            print("Error: libpcre2-dev library not found. Using builtin pcre2.")
            env["builtin_pcre2"] = True
        else:
            env.ParseConfig("pkg-config libpcre2-32 --cflags --libs")

    # Embree is only used in tools build on x86_64 and aarch64.
    if env["tools"] and not env["builtin_embree"] and is64:
        # No pkgconfig file so far, hardcode expected lib name.
        env.Append(LIBS=["embree3"])

    ## Flags

    error = os.system("pkg-config alsa --exists")
    if error:
        print("Warning: ALSA libraries not found. Disabling the ALSA audio driver.")
    else:
        env["alsa"] = True
        env.Append(CPPDEFINES=["ALSA_ENABLED", "ALSAMIDI_ENABLED"])

    if env["pulseaudio"]:
        error = os.system("pkg-config libpulse --exists")
        if error:
            print("Warning: libpulse-dev library not found. Disabling the PulseAudio audio driver.")
        else:
            env.Append(CPPDEFINES=["PULSEAUDIO_ENABLED"])
            env.ParseConfig("pkg-config libpulse --cflags")

    if platform.system() == "Linux":
        env.Append(CPPDEFINES=["JOYDEV_ENABLED"])
        if env["udev"]:
            error = os.system("pkg-config libudev --exists")
            if error:
                print("Warning: libudev-dev library not found. Disabling controller hotplugging support.")
            else:
                env.Append(CPPDEFINES=["UDEV_ENABLED"])
    else:
        env["udev"] = False  # Linux specific

    env.Prepend(CPPPATH=["#platform/x11"])
    env.Append(CPPDEFINES=["X11_ENABLED", "UNIX_ENABLED", "OPENGL_ENABLED", "GLES_ENABLED", ("_FILE_OFFSET_BITS", 64)])
    env.Append(LIBS=["GL", "pthread"])

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
                "Warning: Creating template binaries enabled for PCK embedding is currently only supported with GNU ld, not gold or LLD."
            )
        else:
            if float(gnu_ld_version.group(1)) >= 2.30:
                env.Append(LINKFLAGS=["-T", "platform/x11/pck_embed.ld"])
            else:
                env.Append(LINKFLAGS=["-T", "platform/x11/pck_embed.legacy.ld"])

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
        if env["use_llvm"] and platform.system() != "FreeBSD":
            env["LINKCOM"] = env["LINKCOM"] + " -l:libatomic.a"

    else:
        if env["use_llvm"] and platform.system() != "FreeBSD":
            env.Append(LIBS=["atomic"])
