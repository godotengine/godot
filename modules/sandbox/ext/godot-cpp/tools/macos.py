import os
import sys

import common_compiler_flags


def has_osxcross():
    return "OSXCROSS_ROOT" in os.environ


def options(opts):
    opts.Add("macos_deployment_target", "macOS deployment target", "default")
    opts.Add("macos_sdk_path", "macOS SDK path", "")
    if has_osxcross():
        opts.Add("osxcross_sdk", "OSXCross SDK version", "darwin16")


def exists(env):
    return sys.platform == "darwin" or has_osxcross()


def generate(env):
    if env["arch"] not in ("universal", "arm64", "x86_64"):
        print("Only universal, arm64, and x86_64 are supported on macOS. Exiting.")
        env.Exit(1)

    if sys.platform == "darwin":
        # Use clang on macOS by default
        env["CXX"] = "clang++"
        env["CC"] = "clang"
    else:
        # OSXCross
        root = os.environ.get("OSXCROSS_ROOT", "")
        if env["arch"] == "arm64":
            basecmd = root + "/target/bin/arm64-apple-" + env["osxcross_sdk"] + "-"
        else:
            basecmd = root + "/target/bin/x86_64-apple-" + env["osxcross_sdk"] + "-"

        env["CC"] = basecmd + "clang"
        env["CXX"] = basecmd + "clang++"
        env["AR"] = basecmd + "ar"
        env["RANLIB"] = basecmd + "ranlib"
        env["AS"] = basecmd + "as"

        binpath = os.path.join(root, "target", "bin")
        if binpath not in env["ENV"]["PATH"]:
            # Add OSXCROSS bin folder to PATH (required for linking).
            env.PrependENVPath("PATH", binpath)

    # Common flags
    if env["arch"] == "universal":
        env.Append(LINKFLAGS=["-arch", "x86_64", "-arch", "arm64"])
        env.Append(CCFLAGS=["-arch", "x86_64", "-arch", "arm64"])
    else:
        env.Append(LINKFLAGS=["-arch", env["arch"]])
        env.Append(CCFLAGS=["-arch", env["arch"]])

    if env["macos_deployment_target"] != "default":
        env.Append(CCFLAGS=["-mmacosx-version-min=" + env["macos_deployment_target"]])
        env.Append(LINKFLAGS=["-mmacosx-version-min=" + env["macos_deployment_target"]])

    if env["macos_sdk_path"]:
        env.Append(CCFLAGS=["-isysroot", env["macos_sdk_path"]])
        env.Append(LINKFLAGS=["-isysroot", env["macos_sdk_path"]])

    env.Append(CPPDEFINES=["MACOS_ENABLED", "UNIX_ENABLED"])

    # Refer to https://github.com/godotengine/godot/blob/master/platform/macos/detect.py
    # LTO benefits for macOS (size, performance) haven't been clearly established yet.
    if env["lto"] == "auto":
        env["lto"] = "none"

    common_compiler_flags.generate(env)
