import os
import sys
from methods import detect_darwin_sdk_path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SCons import Environment


def get_name():
    return "iOS"


def can_build():
    if sys.platform == "darwin" or ("OSXCROSS_IOS" in os.environ):
        return True

    return False


def get_opts():
    from SCons.Variables import BoolVariable

    return [
        (
            "IOS_TOOLCHAIN_PATH",
            "Path to iOS toolchain",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain",
        ),
        ("IOS_SDK_PATH", "Path to the iOS SDK", ""),
        BoolVariable("ios_simulator", "Build for iOS Simulator", False),
        ("ios_triple", "Triple for ios toolchain", ""),
    ]


def get_doc_classes():
    return [
        "EditorExportPlatformIOS",
    ]


def get_doc_path():
    return "doc_classes"


def get_flags():
    return [
        ("arch", "arm64"),  # Default for convenience.
        ("target", "template_debug"),
        ("use_volk", False),
    ]


def configure(env: "Environment"):
    # Validate arch.
    supported_arches = ["x86_64", "arm64"]
    if env["arch"] not in supported_arches:
        print(
            'Unsupported CPU architecture "%s" for iOS. Supported architectures are: %s.'
            % (env["arch"], ", ".join(supported_arches))
        )
        sys.exit()

    ## LTO

    if env["lto"] == "auto":  # Disable by default as it makes linking in Xcode very slow.
        env["lto"] = "none"

    if env["lto"] != "none":
        if env["lto"] == "thin":
            env.Append(CCFLAGS=["-flto=thin"])
            env.Append(LINKFLAGS=["-flto=thin"])
        else:
            env.Append(CCFLAGS=["-flto"])
            env.Append(LINKFLAGS=["-flto"])

    ## Compiler configuration

    # Save this in environment for use by other modules
    if "OSXCROSS_IOS" in os.environ:
        env["osxcross"] = True

    env["ENV"]["PATH"] = env["IOS_TOOLCHAIN_PATH"] + "/Developer/usr/bin/:" + env["ENV"]["PATH"]

    compiler_path = "$IOS_TOOLCHAIN_PATH/usr/bin/${ios_triple}"

    ccache_path = os.environ.get("CCACHE")
    if ccache_path is None:
        env["CC"] = compiler_path + "clang"
        env["CXX"] = compiler_path + "clang++"
        env["S_compiler"] = compiler_path + "clang"
    else:
        # there aren't any ccache wrappers available for iOS,
        # to enable caching we need to prepend the path to the ccache binary
        env["CC"] = ccache_path + " " + compiler_path + "clang"
        env["CXX"] = ccache_path + " " + compiler_path + "clang++"
        env["S_compiler"] = ccache_path + " " + compiler_path + "clang"
    env["AR"] = compiler_path + "ar"
    env["RANLIB"] = compiler_path + "ranlib"

    ## Compile flags

    if env["ios_simulator"]:
        detect_darwin_sdk_path("iossimulator", env)
        env.Append(ASFLAGS=["-mios-simulator-version-min=12.0"])
        env.Append(CCFLAGS=["-mios-simulator-version-min=12.0"])
        env.Append(CPPDEFINES=["IOS_SIMULATOR"])
        env.extra_suffix = ".simulator" + env.extra_suffix
    else:
        detect_darwin_sdk_path("ios", env)
        env.Append(ASFLAGS=["-miphoneos-version-min=12.0"])
        env.Append(CCFLAGS=["-miphoneos-version-min=12.0"])

    if env["arch"] == "x86_64":
        if not env["ios_simulator"]:
            print("ERROR: Building for iOS with 'arch=x86_64' requires 'ios_simulator=yes'.")
            sys.exit(255)

        env["ENV"]["MACOSX_DEPLOYMENT_TARGET"] = "10.9"
        env.Append(
            CCFLAGS=(
                "-fobjc-arc -arch x86_64"
                " -fobjc-abi-version=2 -fobjc-legacy-dispatch -fmessage-length=0 -fpascal-strings -fblocks"
                " -fasm-blocks -isysroot $IOS_SDK_PATH"
            ).split()
        )
        env.Append(ASFLAGS=["-arch", "x86_64"])
    elif env["arch"] == "arm64":
        env.Append(
            CCFLAGS=(
                "-fobjc-arc -arch arm64 -fmessage-length=0 -fno-strict-aliasing"
                " -fdiagnostics-print-source-range-info -fdiagnostics-show-category=id -fdiagnostics-parseable-fixits"
                " -fpascal-strings -fblocks -fvisibility=hidden -MMD -MT dependencies"
                " -isysroot $IOS_SDK_PATH".split()
            )
        )
        env.Append(ASFLAGS=["-arch", "arm64"])
        env.Append(CPPDEFINES=["NEED_LONG_INT"])

    # Temp fix for ABS/MAX/MIN macros in iOS SDK blocking compilation
    env.Append(CCFLAGS=["-Wno-ambiguous-macro"])

    env.Prepend(
        CPPPATH=[
            "$IOS_SDK_PATH/usr/include",
            "$IOS_SDK_PATH/System/Library/Frameworks/AudioUnit.framework/Headers",
        ]
    )

    env.Prepend(CPPPATH=["#platform/ios"])
    env.Append(CPPDEFINES=["IOS_ENABLED", "UNIX_ENABLED", "COREAUDIO_ENABLED"])

    if env["vulkan"]:
        env.Append(CPPDEFINES=["VULKAN_ENABLED"])

    if env["opengl3"]:
        env.Append(CPPDEFINES=["GLES3_ENABLED", "GLES_SILENCE_DEPRECATION"])
        env.Prepend(
            CPPPATH=[
                "$IOS_SDK_PATH/System/Library/Frameworks/OpenGLES.framework/Headers",
            ]
        )
