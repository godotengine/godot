import os
import platform
import sys

# This file is mostly based on platform/x11/detect.py.
# If editing this file, make sure to apply relevant changes here too.


def is_active():
    return True


def get_name():
    return "Server"


def get_program_suffix():
    if sys.platform == "darwin":
        return "osx"
    return "x11"


def can_build():
    if os.name != "posix":
        return False

    return True


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        BoolVariable("use_llvm", "Use the LLVM compiler", False),
        BoolVariable("use_static_cpp", "Link libgcc and libstdc++ statically for better portability", True),
        BoolVariable("use_ubsan", "Use LLVM/GCC compiler undefined behavior sanitizer (UBSAN)", False),
        BoolVariable("use_asan", "Use LLVM/GCC compiler address sanitizer (ASAN))", False),
        BoolVariable("use_lsan", "Use LLVM/GCC compiler leak sanitizer (LSAN))", False),
        BoolVariable("use_tsan", "Use LLVM/GCC compiler thread sanitizer (TSAN))", False),
        BoolVariable("debug_symbols", "Add debugging symbols to release/release_debug builds", True),
        BoolVariable("use_msan", "Use LLVM/GCC compiler memory sanitizer (MSAN))", False),
        BoolVariable("separate_debug_symbols", "Create a separate file containing debugging symbols", False),
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
        env.Prepend(CCFLAGS=["-g3"])
        env.Append(LINKFLAGS=["-rdynamic"])

    ## Architecture

    is64 = sys.maxsize > 2 ** 32
    if env["bits"] == "default":
        env["bits"] = "64" if is64 else "32"

    if env["arch"] == "" and platform.machine() == "riscv64":
        env["arch"] = "rv64"

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
        env.Append(LIBS=["atomic"])

    if env["use_ubsan"] or env["use_asan"] or env["use_lsan"] or env["use_tsan"] or env["use_msan"]:
        env.extra_suffix += "s"

        if env["use_ubsan"]:
            env.Append(CCFLAGS=["-fsanitize=undefined"])
            env.Append(LINKFLAGS=["-fsanitize=undefined"])

        if env["use_asan"]:
            env.Append(CCFLAGS=["-fsanitize=address"])
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
        env.Append(CCFLAGS=["-flto"])
        if not env["use_llvm"] and env.GetOption("num_jobs") > 1:
            env.Append(LINKFLAGS=["-flto=" + str(env.GetOption("num_jobs"))])
        else:
            env.Append(LINKFLAGS=["-flto"])
        if not env["use_llvm"]:
            env["RANLIB"] = "gcc-ranlib"
            env["AR"] = "gcc-ar"

    env.Append(CCFLAGS=["-pipe"])
    env.Append(LINKFLAGS=["-pipe"])

    ## Dependencies

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
        env.ParseConfig("pkg-config libenet --cflags --libs")

    if not env["builtin_squish"]:
        env.ParseConfig("pkg-config libsquish --cflags --libs")

    if not env["builtin_zstd"]:
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

    # Embree is only compatible with x86_64. Yet another unreliable hack that will break
    # cross-compilation, this will really need to be handle better. Thankfully only affects
    # people who disable builtin_embree (likely distro packagers).
    if env["tools"] and not env["builtin_embree"] and (is64 and platform.machine() == "x86_64"):
        # No pkgconfig file so far, hardcode expected lib name.
        env.Append(LIBS=["embree3"])

    ## Flags

    env.Prepend(CPPPATH=["#platform/server"])
    env.Append(CPPDEFINES=["SERVER_ENABLED", "UNIX_ENABLED"])

    if platform.system() == "Darwin":
        env.Append(LINKFLAGS=["-framework", "Cocoa", "-framework", "Carbon", "-lz", "-framework", "IOKit"])

    env.Append(LIBS=["pthread"])

    if platform.system() == "Linux":
        env.Append(LIBS=["dl"])

    if platform.system().find("BSD") >= 0:
        env["execinfo"] = True

    if env["execinfo"]:
        env.Append(LIBS=["execinfo"])

    # Link those statically for portability
    if env["use_static_cpp"]:
        env.Append(LINKFLAGS=["-static-libgcc", "-static-libstdc++"])
