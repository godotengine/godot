import os
import sys
from methods import detect_darwin_sdk_path


def is_active():
    return True


def get_name():
    return "OSX"


def can_build():
    if sys.platform == "darwin" or ("OSXCROSS_ROOT" in os.environ):
        return True

    return False


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        ("osxcross_sdk", "OSXCross SDK version", "darwin16"),
        ("MACOS_SDK_PATH", "Path to the macOS SDK", ""),
        ("VULKAN_SDK_PATH", "Path to the Vulkan SDK", ""),
        EnumVariable("macports_clang", "Build using Clang from MacPorts", "no", ("no", "5.0", "devel")),
        BoolVariable("debug_symbols", "Add debugging symbols to release/release_debug builds", True),
        BoolVariable("separate_debug_symbols", "Create a separate file containing debugging symbols", False),
        BoolVariable("use_ubsan", "Use LLVM/GCC compiler undefined behavior sanitizer (UBSAN)", False),
        BoolVariable("use_asan", "Use LLVM/GCC compiler address sanitizer (ASAN)", False),
        BoolVariable("use_tsan", "Use LLVM/GCC compiler thread sanitizer (TSAN)", False),
    ]


def get_flags():
    return []


def configure(env):
    ## Build type

    if env["target"] == "release":
        if env["optimize"] == "speed":  # optimize for speed (default)
            env.Prepend(CCFLAGS=["-O3", "-fomit-frame-pointer", "-ftree-vectorize"])
        elif env["optimize"] == "size":  # optimize for size
            env.Prepend(CCFLAGS=["-Os", "-ftree-vectorize"])
        if env["arch"] != "arm64":
            env.Prepend(CCFLAGS=["-msse2"])

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
        env.Prepend(LINKFLAGS=["-Xlinker", "-no_deduplicate"])

    ## Architecture

    # Mac OS X no longer runs on 32-bit since 10.7 which is unsupported since 2014
    # As such, we only support 64-bit
    env["bits"] = "64"

    ## Compiler configuration

    # Save this in environment for use by other modules
    if "OSXCROSS_ROOT" in os.environ:
        env["osxcross"] = True

    if env["arch"] == "arm64":
        print("Building for macOS 10.15+, platform arm64.")
        env.Append(CCFLAGS=["-arch", "arm64", "-mmacosx-version-min=10.15"])
        env.Append(LINKFLAGS=["-arch", "arm64", "-mmacosx-version-min=10.15"])
    else:
        print("Building for macOS 10.12+, platform x86-64.")
        env.Append(CCFLAGS=["-arch", "x86_64", "-mmacosx-version-min=10.12"])
        env.Append(LINKFLAGS=["-arch", "x86_64", "-mmacosx-version-min=10.12"])

    if not "osxcross" in env:  # regular native build
        if env["macports_clang"] != "no":
            mpprefix = os.environ.get("MACPORTS_PREFIX", "/opt/local")
            mpclangver = env["macports_clang"]
            env["CC"] = mpprefix + "/libexec/llvm-" + mpclangver + "/bin/clang"
            env["CXX"] = mpprefix + "/libexec/llvm-" + mpclangver + "/bin/clang++"
            env["AR"] = mpprefix + "/libexec/llvm-" + mpclangver + "/bin/llvm-ar"
            env["RANLIB"] = mpprefix + "/libexec/llvm-" + mpclangver + "/bin/llvm-ranlib"
            env["AS"] = mpprefix + "/libexec/llvm-" + mpclangver + "/bin/llvm-as"
            env.Append(CPPDEFINES=["__MACPORTS__"])  # hack to fix libvpx MM256_BROADCASTSI128_SI256 define
        else:
            env["CC"] = "clang"
            env["CXX"] = "clang++"

        detect_darwin_sdk_path("osx", env)
        env.Append(CCFLAGS=["-isysroot", "$MACOS_SDK_PATH"])
        env.Append(LINKFLAGS=["-isysroot", "$MACOS_SDK_PATH"])

    else:  # osxcross build
        root = os.environ.get("OSXCROSS_ROOT", 0)
        if env["arch"] == "arm64":
            basecmd = root + "/target/bin/arm64-apple-" + env["osxcross_sdk"] + "-"
        else:
            basecmd = root + "/target/bin/x86_64-apple-" + env["osxcross_sdk"] + "-"

        ccache_path = os.environ.get("CCACHE")
        if ccache_path is None:
            env["CC"] = basecmd + "cc"
            env["CXX"] = basecmd + "c++"
        else:
            # there aren't any ccache wrappers available for OS X cross-compile,
            # to enable caching we need to prepend the path to the ccache binary
            env["CC"] = ccache_path + " " + basecmd + "cc"
            env["CXX"] = ccache_path + " " + basecmd + "c++"
        env["AR"] = basecmd + "ar"
        env["RANLIB"] = basecmd + "ranlib"
        env["AS"] = basecmd + "as"
        env.Append(CPPDEFINES=["__MACPORTS__"])  # hack to fix libvpx MM256_BROADCASTSI128_SI256 define

    if env["use_ubsan"] or env["use_asan"] or env["use_tsan"]:
        env.extra_suffix += "s"

        if env["use_ubsan"]:
            env.Append(
                CCFLAGS=[
                    "-fsanitize=undefined,shift,shift-exponent,integer-divide-by-zero,unreachable,vla-bound,null,return,signed-integer-overflow,bounds,float-divide-by-zero,float-cast-overflow,nonnull-attribute,returns-nonnull-attribute,bool,enum,vptr,pointer-overflow,builtin"
                ]
            )
            env.Append(LINKFLAGS=["-fsanitize=undefined"])
            env.Append(CCFLAGS=["-fsanitize=nullability-return,nullability-arg,function,nullability-assign"])

        if env["use_asan"]:
            env.Append(CCFLAGS=["-fsanitize=address,pointer-subtract,pointer-compare"])
            env.Append(LINKFLAGS=["-fsanitize=address"])

        if env["use_tsan"]:
            env.Append(CCFLAGS=["-fsanitize=thread"])
            env.Append(LINKFLAGS=["-fsanitize=thread"])

    ## Dependencies

    if env["builtin_libtheora"]:
        if env["arch"] != "arm64":
            env["x86_libtheora_opt_gcc"] = True

    ## Flags

    env.Prepend(CPPPATH=["#platform/osx"])
    env.Append(CPPDEFINES=["OSX_ENABLED", "UNIX_ENABLED", "APPLE_STYLE_KEYS", "COREAUDIO_ENABLED", "COREMIDI_ENABLED"])
    env.Append(
        LINKFLAGS=[
            "-framework",
            "Cocoa",
            "-framework",
            "Carbon",
            "-framework",
            "AudioUnit",
            "-framework",
            "CoreAudio",
            "-framework",
            "CoreMIDI",
            "-framework",
            "IOKit",
            "-framework",
            "ForceFeedback",
            "-framework",
            "CoreVideo",
            "-framework",
            "AVFoundation",
            "-framework",
            "CoreMedia",
        ]
    )
    env.Append(LIBS=["pthread", "z"])

    if env["vulkan"]:
        env.Append(CPPDEFINES=["VULKAN_ENABLED"])
        env.Append(LINKFLAGS=["-framework", "Metal", "-framework", "QuartzCore", "-framework", "IOSurface"])
        if not env["use_volk"]:
            env.Append(LINKFLAGS=["-L$VULKAN_SDK_PATH/MoltenVK/MoltenVK.xcframework/macos-arm64_x86_64/", "-lMoltenVK"])

    # env.Append(CPPDEFINES=['GLES_ENABLED', 'OPENGL_ENABLED'])
