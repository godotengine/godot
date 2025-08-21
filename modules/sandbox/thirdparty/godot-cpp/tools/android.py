import os
import sys

import common_compiler_flags
import my_spawn


def options(opts):
    opts.Add(
        "android_api_level",
        "Target Android API level",
        "24",
    )
    opts.Add(
        "ndk_version",
        "Fully qualified version of ndk to use for compilation.",
        "28.1.13356709",
    )
    opts.Add(
        "ANDROID_HOME",
        "Path to your Android SDK installation. By default, uses ANDROID_HOME from your defined environment variables.",
        os.environ.get("ANDROID_HOME", os.environ.get("ANDROID_SDK_ROOT")),
    )


def exists(env):
    return get_android_ndk_root(env) is not None


def get_android_ndk_root(env):
    if env["ANDROID_HOME"]:
        return env["ANDROID_HOME"] + "/ndk/" + env["ndk_version"]
    else:
        return os.environ.get("ANDROID_NDK_ROOT")


def generate(env):
    if get_android_ndk_root(env) is None:
        raise ValueError(
            "To build for Android, the path to the NDK must be defined. Please set ANDROID_HOME to the root folder of your Android SDK installation."
        )

    if env["arch"] not in ("arm64", "x86_64", "arm32", "x86_32"):
        print("Only arm64, x86_64, arm32, and x86_32 are supported on Android. Exiting.")
        env.Exit(1)

    if sys.platform == "win32" or sys.platform == "msys":
        my_spawn.configure(env)

    # Validate API level
    if int(env["android_api_level"]) < 24:
        print("WARNING: minimum supported Android target api is 24. Forcing target api 24.")
        env["android_api_level"] = "24"

    # Setup toolchain
    toolchain = get_android_ndk_root(env) + "/toolchains/llvm/prebuilt/"
    if sys.platform == "win32" or sys.platform == "msys":
        toolchain += "windows"
        import platform as pltfm

        if pltfm.machine().endswith("64"):
            toolchain += "-x86_64"
    elif sys.platform.startswith("linux"):
        toolchain += "linux-x86_64"
    elif sys.platform == "darwin":
        toolchain += "darwin-x86_64"
        env.Append(LINKFLAGS=["-shared"])

    if not os.path.exists(toolchain):
        print("ERROR: Could not find NDK toolchain at " + toolchain + ".")
        print("Make sure NDK version " + env["ndk_version"] + " is installed.")
        env.Exit(1)

    env.PrependENVPath("PATH", toolchain + "/bin")  # This does nothing half of the time, but we'll put it here anyways

    # Get architecture info
    arch_info_table = {
        "arm32": {
            "march": "armv7-a",
            "target": "armv7a-linux-androideabi",
            "compiler_path": "armv7a-linux-androideabi",
            "ccflags": ["-mfpu=neon"],
        },
        "arm64": {
            "march": "armv8-a",
            "target": "aarch64-linux-android",
            "compiler_path": "aarch64-linux-android",
            "ccflags": [],
        },
        "x86_32": {
            "march": "i686",
            "target": "i686-linux-android",
            "compiler_path": "i686-linux-android",
            "ccflags": ["-mstackrealign"],
        },
        "x86_64": {
            "march": "x86-64",
            "target": "x86_64-linux-android",
            "compiler_path": "x86_64-linux-android",
            "ccflags": [],
        },
    }
    arch_info = arch_info_table[env["arch"]]

    # Setup tools
    env["CC"] = toolchain + "/bin/clang"
    env["CXX"] = toolchain + "/bin/clang++"
    env["LINK"] = toolchain + "/bin/clang++"
    env["AR"] = toolchain + "/bin/llvm-ar"
    env["AS"] = toolchain + "/bin/llvm-as"
    env["STRIP"] = toolchain + "/bin/llvm-strip"
    env["RANLIB"] = toolchain + "/bin/llvm-ranlib"
    env["SHLIBSUFFIX"] = ".so"

    env.Append(
        CCFLAGS=["--target=" + arch_info["target"] + env["android_api_level"], "-march=" + arch_info["march"], "-fPIC"]
    )
    env.Append(CCFLAGS=arch_info["ccflags"])
    env.Append(LINKFLAGS=["--target=" + arch_info["target"] + env["android_api_level"], "-march=" + arch_info["march"]])

    env.Append(CPPDEFINES=["ANDROID_ENABLED", "UNIX_ENABLED"])

    # Refer to https://github.com/godotengine/godot/blob/master/platform/android/detect.py
    # LTO benefits for Android (size, performance) haven't been clearly established yet.
    if env["lto"] == "auto":
        env["lto"] = "none"

    common_compiler_flags.generate(env)
