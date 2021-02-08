import os
import sys
from methods import detect_darwin_sdk_path, get_darwin_sdk_version


def is_active():
    return True


def get_name():
    return "tvOS"


def can_build():
    if sys.platform == "darwin":
        if get_darwin_sdk_version("tvos") < 13.0:
            print("Detected tvOS SDK version older than 13")
            return False
        return True
    elif "OSXCROSS_TVOS" in os.environ:
        return True

    return False


def get_opts():
    from SCons.Variables import BoolVariable

    return [
        (
            "TVOSPATH",
            "Path to tvOS toolchain",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain",
        ),
        ("TVOSSDK", "Path to the tvOS SDK", ""),
        BoolVariable("simulator", "Build for simulator", False),
        ("tvos_triple", "Triple for tvOS toolchain", ""),
    ]


def get_flags():
    return [
        ("tools", False),
    ]


def configure(env):
    ## Build type

    if env["target"].startswith("release"):
        env.Append(CPPDEFINES=["NDEBUG", ("NS_BLOCK_ASSERTIONS", 1)])
        if env["optimize"] == "speed":  # optimize for speed (default)
            env.Append(CCFLAGS=["-O2", "-ftree-vectorize", "-fomit-frame-pointer"])
            env.Append(LINKFLAGS=["-O2"])
        else:  # optimize for size
            env.Append(CCFLAGS=["-Os", "-ftree-vectorize"])
            env.Append(LINKFLAGS=["-Os"])

        if env["target"] == "release_debug":
            env.Append(CPPDEFINES=["DEBUG_ENABLED"])

    elif env["target"] == "debug":
        env.Append(CCFLAGS=["-gdwarf-2", "-O0"])
        env.Append(CPPDEFINES=["_DEBUG", ("DEBUG", 1), "DEBUG_ENABLED"])

    if env["use_lto"]:
        env.Append(CCFLAGS=["-flto=thin"])
        env.Append(LINKFLAGS=["-flto=thin"])

    ## Architecture
    if env["arch"] == "x86":  # i386
        env["bits"] = "32"
    elif env["arch"] == "x86_64":
        env["bits"] = "64"
    else:  # armv64
        env["arch"] = "arm64"
        env["bits"] = "64"

    ## Compiler configuration

    # Save this in environment for use by other modules
    if "OSXCROSS_TVOS" in os.environ:
        env["osxcross"] = True

    env["ENV"]["PATH"] = env["TVOSSDK"] + "/Developer/usr/bin/:" + env["ENV"]["PATH"]

    compiler_path = "$TVOSPATH/usr/bin/${tvos_triple}"
    s_compiler_path = "$TVOSPATH/Developer/usr/bin/"

    ccache_path = os.environ.get("CCACHE")
    if ccache_path is None:
        env["CC"] = compiler_path + "clang"
        env["CXX"] = compiler_path + "clang++"
        env["S_compiler"] = s_compiler_path + "gcc"
    else:
        # there aren't any ccache wrappers available for iOS,
        # to enable caching we need to prepend the path to the ccache binary
        env["CC"] = ccache_path + " " + compiler_path + "clang"
        env["CXX"] = ccache_path + " " + compiler_path + "clang++"
        env["S_compiler"] = ccache_path + " " + s_compiler_path + "gcc"
    env["AR"] = compiler_path + "ar"
    env["RANLIB"] = compiler_path + "ranlib"

    ## Compile flags

    if env["simulator"]:
        detect_darwin_sdk_path("tvossimulator", env)
        env.Append(CCFLAGS=("-isysroot $TVOSSDK -mappletvsimulator-version-min=10.0").split())
        env.Append(LINKFLAGS=["-mappletvsimulator-version-min=10.0"])
        env["LIBSUFFIX"] = ".simulator" + env["LIBSUFFIX"]
    else:
        detect_darwin_sdk_path("tvos", env)
        env.Append(CCFLAGS=("-isysroot $TVOSSDK -mappletvos-version-min=10.0").split())
        env.Append(LINKFLAGS=["-mappletvos-version-min=10.0"])

    if env["arch"] == "x86" or env["arch"] == "x86_64":
        env["ENV"]["MACOSX_DEPLOYMENT_TARGET"] = "10.9"
        arch_flag = "i386" if env["arch"] == "x86" else env["arch"]
        env.Append(
            CCFLAGS=(
                "-arch "
                + arch_flag
                + " -fobjc-arc -fobjc-abi-version=2 -fobjc-legacy-dispatch -fmessage-length=0 -fpascal-strings -fblocks -fasm-blocks"
            ).split()
        )
    elif env["arch"] == "arm64":
        env.Append(
            CCFLAGS="-fobjc-arc -arch arm64 -fmessage-length=0 -fno-strict-aliasing -fdiagnostics-print-source-range-info -fdiagnostics-show-category=id -fdiagnostics-parseable-fixits -fpascal-strings -fblocks -fvisibility=hidden -MMD -MT dependencies".split()
        )
        env.Append(CPPDEFINES=["NEED_LONG_INT"])
        env.Append(CPPDEFINES=["LIBYUV_DISABLE_NEON"])

    # Temp fix for ABS/MAX/MIN macros in tvOS/iOS SDK blocking compilation
    env.Append(CCFLAGS=["-Wno-ambiguous-macro"])

    # tvOS requires Bitcode.
    env.Append(CCFLAGS=["-fembed-bitcode"])
    env.Append(LINKFLAGS=["-bitcode_bundle"])

    ## Link flags

    if env["arch"] == "x86" or env["arch"] == "x86_64":
        arch_flag = "i386" if env["arch"] == "x86" else env["arch"]
        env.Append(
            LINKFLAGS=[
                "-arch",
                arch_flag,
                "-isysroot",
                "$TVOSSDK",
                "-Xlinker",
                "-objc_abi_version",
                "-Xlinker",
                "2",
                "-F$TVOSSDK",
            ]
        )
    if env["arch"] == "arm64":
        env.Append(LINKFLAGS=["-arch", "arm64", "-Wl,-dead_strip"])

    env.Append(
        LINKFLAGS=[
            "-isysroot",
            "$TVOSSDK",
        ]
    )

    env.Prepend(
        CPPPATH=[
            "$TVOSSDK/usr/include",
            "$TVOSSDK/System/Library/Frameworks/OpenGLES.framework/Headers",
            "$TVOSSDK/System/Library/Frameworks/AudioUnit.framework/Headers",
        ]
    )

    env.Prepend(CPPPATH=["#platform/tvos"])
    env.Append(CPPDEFINES=["TVOS_ENABLED", "UNIX_ENABLED", "GLES_ENABLED", "COREAUDIO_ENABLED"])
