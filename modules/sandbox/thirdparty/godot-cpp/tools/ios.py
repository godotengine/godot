import codecs
import os
import subprocess
import sys

import common_compiler_flags
from SCons.Variables import BoolVariable


def has_ios_osxcross():
    return "OSXCROSS_IOS" in os.environ


def options(opts):
    opts.Add(BoolVariable("ios_simulator", "Target iOS Simulator", False))
    opts.Add("ios_min_version", "Target minimum iphoneos/iphonesimulator version", "12.0")
    opts.Add(
        "IOS_TOOLCHAIN_PATH",
        "Path to iOS toolchain",
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain",
    )
    opts.Add("IOS_SDK_PATH", "Path to the iOS SDK", "")

    if has_ios_osxcross():
        opts.Add("ios_triple", "Triple for ios toolchain", "")


def exists(env):
    return sys.platform == "darwin" or has_ios_osxcross()


def generate(env):
    if env["arch"] not in ("universal", "arm64", "x86_64"):
        raise ValueError("Only universal, arm64, and x86_64 are supported on iOS. Exiting.")

    if env["ios_simulator"]:
        sdk_name = "iphonesimulator"
        env.Append(CCFLAGS=["-mios-simulator-version-min=" + env["ios_min_version"]])
        env.Append(LINKFLAGS=["-mios-simulator-version-min=" + env["ios_min_version"]])
    else:
        sdk_name = "iphoneos"
        env.Append(CCFLAGS=["-miphoneos-version-min=" + env["ios_min_version"]])
        env.Append(LINKFLAGS=["-miphoneos-version-min=" + env["ios_min_version"]])

    if sys.platform == "darwin":
        if env["IOS_SDK_PATH"] == "":
            try:
                env["IOS_SDK_PATH"] = codecs.utf_8_decode(
                    subprocess.check_output(["xcrun", "--sdk", sdk_name, "--show-sdk-path"]).strip()
                )[0]
            except (subprocess.CalledProcessError, OSError):
                raise ValueError(
                    "Failed to find SDK path while running xcrun --sdk {} --show-sdk-path.".format(sdk_name)
                )

        compiler_path = env["IOS_TOOLCHAIN_PATH"] + "/usr/bin/"
        env["CC"] = compiler_path + "clang"
        env["CXX"] = compiler_path + "clang++"
        env["AR"] = compiler_path + "ar"
        env["RANLIB"] = compiler_path + "ranlib"
        env["SHLIBSUFFIX"] = ".dylib"
        env["ENV"]["PATH"] = env["IOS_TOOLCHAIN_PATH"] + "/Developer/usr/bin/:" + env["ENV"]["PATH"]

    else:
        # OSXCross
        compiler_path = "$IOS_TOOLCHAIN_PATH/usr/bin/${ios_triple}"
        env["CC"] = compiler_path + "clang"
        env["CXX"] = compiler_path + "clang++"
        env["AR"] = compiler_path + "ar"
        env["RANLIB"] = compiler_path + "ranlib"
        env["SHLIBSUFFIX"] = ".dylib"

        env.Prepend(
            CPPPATH=[
                "$IOS_SDK_PATH/usr/include",
                "$IOS_SDK_PATH/System/Library/Frameworks/AudioUnit.framework/Headers",
            ]
        )

        env.Append(CCFLAGS=["-stdlib=libc++"])

        binpath = os.path.join(env["IOS_TOOLCHAIN_PATH"], "usr", "bin")
        if binpath not in env["ENV"]["PATH"]:
            env.PrependENVPath("PATH", binpath)

    if env["arch"] == "universal":
        if env["ios_simulator"]:
            env.Append(LINKFLAGS=["-arch", "x86_64", "-arch", "arm64"])
            env.Append(CCFLAGS=["-arch", "x86_64", "-arch", "arm64"])
        else:
            env.Append(LINKFLAGS=["-arch", "arm64"])
            env.Append(CCFLAGS=["-arch", "arm64"])
    else:
        env.Append(LINKFLAGS=["-arch", env["arch"]])
        env.Append(CCFLAGS=["-arch", env["arch"]])

    env.Append(CCFLAGS=["-isysroot", env["IOS_SDK_PATH"]])
    env.Append(LINKFLAGS=["-isysroot", env["IOS_SDK_PATH"], "-F" + env["IOS_SDK_PATH"]])

    env.Append(CPPDEFINES=["IOS_ENABLED", "UNIX_ENABLED"])

    # Refer to https://github.com/godotengine/godot/blob/master/platform/ios/detect.py:
    # Disable by default as it makes linking in Xcode very slow.
    if env["lto"] == "auto":
        env["lto"] = "none"

    common_compiler_flags.generate(env)
