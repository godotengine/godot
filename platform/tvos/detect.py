import os
import sys
from pathlib import Path

import methods
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SCons import Environment


def is_tvos_platform(platform):
    return platform == "tvos"


def get_name():
    return "tvOS"


def get_tvos_sdk_version(env):
    return methods.get_cmdline_bool("tvos_sdk_version", "")


def can_build():
    if sys.platform == "darwin" or ("OSXCROSS_IOS" in os.environ):
        return True

    return False


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        ("tvos_sdk_version", "tvOS SDK version to build with", ""),
        ("TVOS_TOOLCHAIN_PATH", "Path to the tvOS toolchain", ""),
        ("tvos_flex_version", "tvOS Flex SDK version to use (path without version number or 'use-latest')", "use-latest"),
        ("tvos_min_version", "minimum tvOS version to support (must be lower than tvos_sdk_version)", "12.0"),
        ("tvos_project_only", "tvOS project files generation only", "no"),
        BoolVariable("tvos_exceptions", "Enable C++ exceptions", False),
        BoolVariable("tvos_32_64_arm", "Enable 32/64-bit ARM cross-compiling", False),
        EnumVariable("tvos_arch", "tvOS architecture", "arm64", ("arm64", "x86_64", "universal")),
        BoolVariable("tvos_simulator", "Build for simulator", False),
        BoolVariable("generate_bundle", "Build using xcframework bundle", True),
    ]


def get_flags():
    return [
        ("arch", "arm64"),
        ("use_volk", False),
    ]


def configure(env):
    # Validate arch.
    supported_arches = ["arm64", "x86_64", "universal"]
    if env["tvos_arch"] not in supported_arches:
        print(
            'Unsupported tvOS architecture "%s", only these architectures are supported: %s.'
            % (env["tvos_arch"], ", ".join(supported_arches))
        )
        sys.exit()

    if env["tvos_arch"] == "universal" and not env["tvos_simulator"]:
        print("Building for universal architecture is only for the tvOS simulator.")
        sys.exit()

    if env["tvos_arch"] == "x86_64" and not env["tvos_simulator"]:
        print("Building for x86_64 architecture is only for the tvOS simulator.")
        sys.exit()

    ## Build project only
    if env["tvos_project_only"]:
        env["project_only"] = env["tvos_project_only"]

    ## Environment base
    env.Tool("xcode")

    # Only compile ARM64 for tvOS devices
    if env["tvos_simulator"]:
        sdk_name = "AppleTVSimulator"
        env.Append(ASFLAGS=["-mtvos-simulator-version-min=" + env["tvos_min_version"]])
        env.Append(CCFLAGS=["-mtvos-simulator-version-min=" + env["tvos_min_version"]])
        env.Append(LINKFLAGS=["-mtvos-simulator-version-min=" + env["tvos_min_version"]])
        if env["tvos_arch"] == "universal":
            env.Append(CCFLAGS=["-arch", "arm64", "-arch", "x86_64"])
            env.Append(LINKFLAGS=["-arch", "arm64", "-arch", "x86_64"])
        else:
            env.Append(CCFLAGS=["-arch", env["tvos_arch"]])
            env.Append(LINKFLAGS=["-arch", env["tvos_arch"]])
    else:
        sdk_name = "AppleTVOS"
        env.Append(ASFLAGS=["-mtvos-version-min=" + env["tvos_min_version"]])
        env.Append(CCFLAGS=["-mtvos-version-min=" + env["tvos_min_version"]])
        env.Append(LINKFLAGS=["-mtvos-version-min=" + env["tvos_min_version"]])
        env.Append(CCFLAGS=["-arch", env["tvos_arch"]])
        env.Append(LINKFLAGS=["-arch", env["tvos_arch"]])

    # tvOS SDK version
    sdk_version = env["tvos_sdk_version"]
    if sdk_version == "":
        sdk_version = methods.get_version_from_args(["-sdk", sdk_name], "SDK version", "Xcode not installed, not building for tvOS")
        if sdk_version == None:
            sys.exit()
        env["tvos_sdk_version"] = sdk_version

    # Flash SDK
    flex_ver = env["tvos_flex_version"]
    if flex_ver == "use-latest":
        flex_ver = methods.get_version_from_args(
            ["mxmlc", "-version"],
            "Adobe Flex SDK",
            "Adobe Flex SDK not installed, not building .ipa",
            True,
        )
        if flex_ver is None:
            flex_ver = ""
        else:
            flex_ver = flex_ver.split(" ")[0]

    flex_sdk_path = None
    if len(flex_ver) > 0:
        if os.path.isdir("/Applications/Adobe Flex SDK " + flex_ver):
            flex_sdk_path = "/Applications/Adobe Flex SDK " + flex_ver
        elif os.path.isdir(os.path.expanduser("~/Applications/Adobe Flex SDK " + flex_ver)):
            flex_sdk_path = os.path.expanduser("~/Applications/Adobe Flex SDK " + flex_ver)
        else:
            print("Adobe Flex SDK " + flex_ver + " not found, not building .ipa")

    env["ENV"]["FLEX_VER"] = flex_ver
    if flex_sdk_path is not None:
        env["ENV"]["FLEX_SDK"] = flex_sdk_path
        env["ENV"]["MXMLC_PATH"] = flex_sdk_path + "/bin/mxmlc"

    env["target_triple"] = env["tvos_arch"] + "-apple-tvos"
    env["ENV"]["TVOS_TOOLCHAIN_PATH"] = env["TVOS_TOOLCHAIN_PATH"]
    if "OSXCROSS_IOS" in os.environ:
        root = os.environ.get("OSXCROSS_IOS", 0)
        env.AppendENVPath("PATH", root + "/usr/bin")
        env["ENV"]["OSXCROSS_IOS"] = root
        # Use darwin16.4.0 toolchain to build for ARM64 iOS
        env["tvos_triple"] = env["tvos_arch"] + "-apple-darwin16.4.0"
        env["CC"] = env["tvos_triple"] + "-clang"
        env["CXX"] = env["tvos_triple"] + "-clang++"
        env["AR"] = env["tvos_triple"] + "-ar"
        env["RANLIB"] = env["tvos_triple"] + "-ranlib"
        env["SHLIBSUFFIX"] = ".dylib"
    else:
        env["ENV"]["PATH"] = "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/:" + env["ENV"]["PATH"]

    # XCode version
    xcode_ver = methods.get_version_from_args(["xcodebuild", "-version"], "Xcode", "Xcode not installed")
    if xcode_ver is None:
        sys.exit()
    env["ENV"]["XCODE_VERSION"] = xcode_ver

    # Templated plist
    methods.update_plist_template(env)

    # Format for tvOS device
    if not env["tvos_simulator"]:
        env["platform_type"] = "device"
        env["game_center"] = "on"
        env["store_kit"] = "on"
        env["is_device"] = True
    else:
        env["platform_type"] = "simulator"
        env["game_center"] = "off"
        env["store_kit"] = "off"
        env["is_device"] = False

    # Setup compilation
    cflags = ["-fobjc-arc", "-fmessage-length=0", "-fno-strict-aliasing", "-fdiagnostics-print-source-range-info", "-fdiagnostics-show-category=id", "-fdiagnostics-parseable-fixits", "-fpascal-strings", "-fblocks", "-fvisibility=hidden", "-MMD", "-MT", "dependencies", "-quiet"]
    cflags_c = ["-std=gnu99"]
    cflags_cpp = ["-std=gnu++17"]
    # tvOS uses the ARC memory management model
    env["cxx_std"] = "gnu++17"
    env.Append(CPPPATH=["$TVOS_TOOLCHAIN_PATH/usr/include", "$TVOS_TOOLCHAIN_PATH/usr/include/c++/v1"])
    env.Append(CPPDEFINES=["TVOS_VERSION_" + env["tvos_min_version"].replace(".", "_")])
    env.Append(CCFLAGS=cflags)
    env.Append(CFLAGS=cflags_c)
    env.Append(CXXFLAGS=cflags_cpp)

    ### Configure tvOS SDK, framework, and builtin libraries
    methods.xcapp_configure(env, "tvos", sdk_name, env["tvos_sdk_version"], env["tvos_min_version"])

    ### Link flags
    env.Append(
        LINKFLAGS=[
            "-fobjc-arc",
            "-stdlib=libc++",
            "-fobjc-link-runtime",
            "-fPIC",
            "-isysroot",
            "$TVOS_SDK_PATH",
        ]
    )

    if not env["tvos_exceptions"]:
        env.Append(CCFLAGS=["-fno-exceptions"])

    # Xcode project settings
    env.Prepend(CPPPATH=["#platform/tvos"])
    env.Append(CPPDEFINES=["TVOS_ENABLED", "UNIX_ENABLED", "COREAUDIO_ENABLED", "COREMIDI_ENABLED"])

    ## Disable ARC on cpp files
    flag_dict = env.ParseFlags(env.subst("$CCFLAGS"))
    flags = flag_dict.get("CCFLAGS", [])
    if "-fobjc-arc" in flags:
        flags.remove("-fobjc-arc")
        flag_dict["CCFLAGS"] = flags
        env.MergeFlags(flag_dict)

    ### Disable warning for casting results to larger types due to tvOS targeting ARM64
    env.Append(CCFLAGS=["-Wno-bad-function-cast"])

    env.Append(LINKFLAGS=["-isysroot", "$TVOS_SDK_PATH"])

    env.Append(
        LINKFLAGS=[
            "-framework",
            "Foundation",
            "-framework",
            "TVMLKit",
            "-framework",
            "TVUIKit",
            "-framework",
            "MediaPlayer",
            "-framework",
            "Foundation",
            "-framework",
            "CoreText",
            "-framework",
            "AudioToolbox",
            "-framework",
            "CoreAudio",
            "-framework",
            "AVFoundation",
            "-framework",
            "CoreMedia",
            "-framework",
            "CoreMotion",
            "-framework",
            "CoreVideo",
            "-framework",
            "GameController",
            "-framework",
            "Metal",
            "-framework",
            "QuartzCore",
            "-framework",
            "Security",
            "-framework",
            "UIKit",
            "-framework",
            "TVServices",
            "-framework",
            "SystemConfiguration",
        ]
    )

    if env["vulkan"]:
        env.PrependENVPath("PATH", os.environ["VULKAN_SDK"] + "/bin")
        env.Append(CPPDEFINES=["VULKAN_ENABLED"])
        # Configure ndk-build location
        env.Append(LINKFLAGS=["-framework", "MoltenVK"])
        env.Append(LINKFLAGS=["-L" + os.environ["VULKAN_SDK"] + "/lib"])
    else:
        # OpenGL drivers may need external symbols
        env.Append(CPPDEFINES=["GLES_ENABLED", "GLES_SILENCE_DEPRECATION"])
        env.Append(LINKFLAGS=["-framework", "OpenGLES"])

    env.Append(
        FRAMEWORKPATH=[
            "$TVOS_SDK_PATH/System/Library/Frameworks",
            "$TVOS_SDK_PATH/System/Library/PrivateFrameworks",
        ]
    )

    if not "OSXCROSS_IOS" in os.environ:
        methods.detect_darwin_sdk_path("tvos", env, sdk_name)
    else:
        methods.detect_arch_path("tvos", env, get_name())