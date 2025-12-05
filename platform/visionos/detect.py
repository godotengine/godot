import os
import sys
from typing import TYPE_CHECKING

from methods import detect_darwin_sdk_path, detect_darwin_toolchain_path, print_warning
from platform_methods import validate_arch

if TYPE_CHECKING:
    from SCons.Script.SConscript import SConsEnvironment


def get_name():
    return "visionOS"


def can_build():
    if sys.platform == "darwin" or ("OSXCROSS_VISIONOS" in os.environ):
        return True

    return False


def get_opts():
    from SCons.Variables import BoolVariable

    return [
        ("SWIFT_FRONTEND", "Path to the swift-frontend binary", ""),
        # APPLE_TOOLCHAIN_PATH Example: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain
        ("APPLE_TOOLCHAIN_PATH", "Path to the Apple toolchain", ""),
        (("APPLE_SDK_PATH", "VISIONOS_SDK_PATH"), "Path to the visionOS SDK", ""),
        ("apple_target_triple", "Triple for corresponding target Apple platform toolchain", ""),
        BoolVariable("simulator", "Build for Simulator", False),
        BoolVariable("generate_bundle", "Generate an APP bundle after building visionOS/macOS binaries", False),
    ]


def get_doc_classes():
    return [
        "EditorExportPlatformVisionOS",
    ]


def get_doc_path():
    return "doc_classes"


def get_flags():
    return {
        "arch": "arm64",
        "target": "template_debug",
        "use_volk": False,
        "metal": True,
        "supported": ["metal", "mono"],
        "builtin_pcre2_with_jit": False,
        "vulkan": False,
        "opengl3": False,
    }


def configure(env: "SConsEnvironment"):
    # Validate arch.
    supported_arches = ["x86_64", "arm64"]
    validate_arch(env["arch"], get_name(), supported_arches)
    detect_darwin_toolchain_path(env)

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
    if "OSXCROSS_VISIONOS" in os.environ:
        env["osxcross"] = True

    env.PrependENVPath("PATH", env["APPLE_TOOLCHAIN_PATH"] + "/Developer/usr/bin/")

    compiler_path = "$APPLE_TOOLCHAIN_PATH/usr/bin/${apple_target_triple}"

    ccache_path = os.environ.get("CCACHE")
    if ccache_path is None:
        env["CC"] = compiler_path + "clang"
        env["CXX"] = compiler_path + "clang++"
        env["S_compiler"] = compiler_path + "clang"
    else:
        # there aren't any ccache wrappers available for visionOS,
        # to enable caching we need to prepend the path to the ccache binary
        env["CC"] = ccache_path + " " + compiler_path + "clang"
        env["CXX"] = ccache_path + " " + compiler_path + "clang++"
        env["S_compiler"] = ccache_path + " " + compiler_path + "clang"
    env["AR"] = compiler_path + "ar"
    env["RANLIB"] = compiler_path + "ranlib"

    ## Compile flags

    if env["simulator"]:
        env["APPLE_PLATFORM"] = "visionossimulator"
        env.Append(ASFLAGS=["-mtargetos=xros26.0-simulator"])
        env.Append(CCFLAGS=["-mtargetos=xros26.0-simulator"])
        env.Append(CPPDEFINES=["VISIONOS_SIMULATOR"])
        env.extra_suffix = ".simulator" + env.extra_suffix
    else:
        env["APPLE_PLATFORM"] = "visionos"
        env.Append(ASFLAGS=["-mtargetos=xros26.0"])
        env.Append(CCFLAGS=["-mtargetos=xros26.0"])
    detect_darwin_sdk_path(env["APPLE_PLATFORM"], env)

    if env["arch"] == "arm64":
        env.Append(
            CCFLAGS=(
                "-fobjc-arc -arch arm64 -fmessage-length=0"
                " -fdiagnostics-print-source-range-info -fdiagnostics-show-category=id -fdiagnostics-parseable-fixits"
                " -fpascal-strings -fblocks -fvisibility=hidden -MMD -MT dependencies"
                " -isysroot $APPLE_SDK_PATH".split()
            )
        )
        env.Append(ASFLAGS=["-arch", "arm64"])

    # Temp fix for ABS/MAX/MIN macros in visionOS SDK blocking compilation
    env.Append(CCFLAGS=["-Wno-ambiguous-macro"])

    env.Prepend(
        CPPPATH=[
            "$APPLE_SDK_PATH/usr/include",
            "$APPLE_SDK_PATH/System/Library/Frameworks/AudioUnit.framework/Headers",
        ]
    )

    env.Prepend(CPPPATH=["#platform/visionos"])
    env.Append(CPPDEFINES=["VISIONOS_ENABLED", "APPLE_EMBEDDED_ENABLED", "UNIX_ENABLED", "COREAUDIO_ENABLED"])

    if env["vulkan"]:
        print_warning("The visionOS platform does not support the Vulkan rendering driver")
        env["vulkan"] = False

    if env["metal"] and env["simulator"]:
        print_warning("visionOS Simulator does not support the Metal rendering driver")
        env["metal"] = False

    if env["metal"]:
        env.AppendUnique(CPPDEFINES=["METAL_ENABLED", "RD_ENABLED"])
        env.Prepend(
            CPPPATH=[
                "$APPLE_SDK_PATH/System/Library/Frameworks/Metal.framework/Headers",
                "$APPLE_SDK_PATH/System/Library/Frameworks/MetalFX.framework/Headers",
                "$APPLE_SDK_PATH/System/Library/Frameworks/QuartzCore.framework/Headers",
            ]
        )
        env.Prepend(CPPPATH=["#thirdparty/spirv-cross"])

    if env["opengl3"]:
        print_warning("The visionOS platform does not support the OpenGL rendering driver")
        env["opengl3"] = False
