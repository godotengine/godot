import os
import platform
import sys
from methods import get_compiler_version, using_gcc
from platform_methods import detect_arch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SCons import Environment


def get_name():
    return "Switch"


def can_build():
    # Check the minimal dependencies
    if "DEVKITPRO" not in os.environ:
        print("Error: DEVKITPRO not defined in environment. Aborting [switch platform]")
        return False

    if not os.path.exists("{}/devkitA64".format(os.environ.get("DEVKITPRO"))):
        print("Error: devkitA64 not found. Aborting [switch platform]")
        return False

    if not os.path.exists("{}/portlibs/switch/bin/aarch64-none-elf-pkg-config".format(os.environ.get("DEVKITPRO"))):
        print("Error: aarch64-none-elf-pkg-config not found. Aborting [switch platform]")
        return False
    
    return True


def get_opts():
    from SCons.Variables import BoolVariable

    return [
        BoolVariable("use_sanitizer", "Use LLVM compiler address sanitizer", False),
        BoolVariable("use_leak_sanitizer", "Use LLVM compiler memory leaks sanitizer (implies use_sanitizer)", False),
        BoolVariable("vulkan", "Enable the vulkan rendering driver", False),
        BoolVariable("opengl3", "Enable the OpenGL/GLES3 rendering driver", True),
        BoolVariable("openxr", "Enable the OpenXR driver", False),
        BoolVariable("use_volk", "Use the volk library to load the Vulkan loader dynamically", False)
    ]


def get_doc_classes():
    return [
        "EditorExportPlatformSwitch",
    ]


def get_doc_path():
    return "doc_classes"


def get_flags():
    return [
        ("arch", "aarch64"),
        ("tools", False),
        ("builtin_enet", True),
        ("builtin_freetype", True),
        ("builtin_libogg", False),
        ("builtin_libpng", False),
        ("builtin_libtheora", False),
        ("builtin_libvorbis", False),
        ("builtin_libwebp", False),
        ("builtin_wslay", False),
        ("builtin_mbedtls", False),
        ("builtin_miniupnpc", False),
        ("builtin_pcre2", False),
        ("builtin_squish", True),
        ("builtin_zlib", False),
        ("builtin_zstd", False),
    ]


def configure(env):
    env["CC"] = "aarch64-none-elf-gcc"
    env["CXX"] = "aarch64-none-elf-g++"
    env["LD"] = "aarch64-none-elf-ld"
    env["RANLIB"] = "aarch64-none-elf-ranlib"
    env["AR"] = "aarch64-none-elf-ar"

    ## Build type

    dkp = os.environ.get("DEVKITPRO", "/opt/devkitpro")
    env["ENV"]["DEVKITPRO"] = dkp
    updated_path = "{}/portlibs/switch/bin:{}/devkitA64/bin:".format(dkp, dkp) + os.environ["PATH"]
    env["ENV"]["PATH"] = updated_path
    os.environ["PATH"] = updated_path  # os environment has to be updated for subprocess calls

    arch = ["-march=armv8-a", "-mtune=cortex-a57", "-mtp=soft", "-fPIE"]
    env.Prepend(CCFLAGS=arch + ["-ffunction-sections"])

    env.Prepend(CPPPATH=["{}/portlibs/switch/include".format(dkp)])
    env.Prepend(CPPFLAGS=["-isystem", "{}/libnx/include".format(dkp)])
    env.Prepend(CPPFLAGS=["-D__SWITCH__", "-DPOSH_COMPILER_GCC", "-DPOSH_OS_HORIZON", '-DPOSH_OS_STRING=\\"horizon\\"'])

    env.Append(LIBPATH=["{}/portlibs/switch/lib".format(dkp), "{}/libnx/lib".format(dkp)])
    env.Prepend(LINKFLAGS=arch + ["-specs={}/libnx/switch.specs".format(dkp)])

    if env["target"].startswith("release"):
        if env["optimize"] == "speed":  # optimize for speed (default)
            env.Prepend(CCFLAGS=["-O3"])
        else:  # optimize for size
            env.Prepend(CCFLAGS=["-Os"])

        if env["target"] == "release_debug":
            env.Prepend(CCFLAGS=["-DDEBUG_ENABLED"])

    elif env["target"] == "debug":
        env.Prepend(CCFLAGS=["-g3", "-DDEBUG_ENABLED", "-DDEBUG_MEMORY_ENABLED"])

    if env["debug_symbols"]:
        if env.dev_build:
            env.Append(CCFLAGS=["-g3"])
        else:
            env.Append(CCFLAGS=["-g2"])

    ## Architecture

    env["bits"] = "64"

    # leak sanitizer requires (address) sanitizer
    if env["use_sanitizer"] or env["use_leak_sanitizer"]:
        env.Append(CCFLAGS=["-fsanitize=address", "-fno-omit-frame-pointer"])
        env.Append(LINKFLAGS=["-fsanitize=address"])
        env.extra_suffix += "s"
        if env["use_leak_sanitizer"]:
            env.Append(CCFLAGS=["-fsanitize=leak"])
            env.Append(LINKFLAGS=["-fsanitize=leak"])

    if env["lto"] != "none":
        env.Append(CCFLAGS=["-flto"])
        if env.GetOption("num_jobs") > 1:
            env.Append(LINKFLAGS=["-flto=" + str(env.GetOption("num_jobs"))])
        else:
            env.Append(LINKFLAGS=["-flto"])

        env["RANLIB"] = "aarch64-none-elf-gcc-ranlib"

    ## Dependencies

    # freetype depends on libpng and zlib, so bundling one of them while keeping others
    # as shared libraries leads to weird issues
    if env["builtin_freetype"] or env["builtin_libpng"] or env["builtin_zlib"]:
        env["builtin_freetype"] = True
        env["builtin_libpng"] = True
        env["builtin_zlib"] = True

    if not env["builtin_freetype"]:
        env.ParseConfig("aarch64-none-elf-pkg-config freetype2 --cflags --libs")

    if not env["builtin_libpng"]:
        env.ParseConfig("aarch64-none-elf-pkg-config libpng --cflags --libs")

    if not env["builtin_enet"]:
        env.ParseConfig("aarch64-none-elf-pkg-config libenet --cflags --libs")

    if not env["builtin_squish"] and env["tools"]:
        env.ParseConfig("aarch64-none-elf-pkg-config libsquish --cflags --libs")

    if not env["builtin_zstd"]:
        env.ParseConfig("aarch64-none-elf-pkg-config libzstd --cflags --libs")

    # Sound and video libraries
    # Keep the order as it triggers chained dependencies (ogg needed by others, etc.)

    if not env["builtin_libtheora"]:
        env["builtin_libogg"] = False  # Needed to link against system libtheora
        env["builtin_libvorbis"] = False  # Needed to link against system libtheora
        env.ParseConfig("aarch64-none-elf-pkg-config theora theoradec --cflags --libs")
    else:
        list_of_x86 = ["x86_64", "x86", "i386", "i586"]
        if any(platform.machine() in s for s in list_of_x86):
            env["x86_libtheora_opt_gcc"] = True

    if not env["builtin_libvorbis"]:
        env["builtin_libogg"] = False  # Needed to link against system libvorbis
        env.ParseConfig("aarch64-none-elf-pkg-config vorbis vorbisfile --cflags --libs")

    if not env["builtin_libogg"]:
        env.ParseConfig("aarch64-none-elf-pkg-config ogg --cflags --libs")

    if not env["builtin_libwebp"]:
        env.ParseConfig("aarch64-none-elf-pkg-config libwebp --cflags --libs")

    if not env["builtin_mbedtls"]:
        # mbedTLS does not provide a pkgconfig config yet. See https://github.com/ARMmbed/mbedtls/issues/228
        env.Append(LIBS=["mbedtls", "mbedx509", "mbedcrypto"])

    if not env["builtin_wslay"]:
        env.ParseConfig("aarch64-none-elf-pkg-config libwslay --cflags --libs")

    if not env["builtin_miniupnpc"]:
        env.ParseConfig("aarch64-none-elf-pkg-config miniupnpc --cflags --libs")

    # On Linux wchar_t should be 32-bits
    # 16-bit library shouldn't be required due to compiler optimisations
    if not env["builtin_pcre2"]:
        env.ParseConfig("aarch64-none-elf-pkg-config libpcre2-32 --cflags --libs")

    # Linkflags below this line should typically stay the last ones
    if not env['builtin_zlib']:
        env.ParseConfig('aarch64-none-elf-pkg-config zlib --cflags --libs')

    ## Flags
    
    env.Append(CPPPATH=["#platform/switch"])
    env.Append(CPPDEFINES=[
        "SWITCH_ENABLED",
        "GLES3_ENABLED",
        "PTHREAD_NO_RENAME",
        "UNIX_SOCKET_UNAVAILABLE"])

    env.Append(LIBS=["EGL", "GLESv2", "glapi", "drm_nouveau", "nx"])
    # -lglad -lEGL -lglapi -ldrm_nouveau
