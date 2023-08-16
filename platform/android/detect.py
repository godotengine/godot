import os
import sys
import platform
import subprocess


def is_active():
    return True


def get_name():
    return "Android"


def can_build():
    return os.path.exists(get_env_android_sdk_root())


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        ("ANDROID_SDK_ROOT", "Path to the Android SDK", get_env_android_sdk_root()),
        ("ndk_platform", 'Target platform (android-<api>, e.g. "android-19")', "android-19"),
        EnumVariable("android_arch", "Target architecture", "armv7", ("armv7", "arm64v8", "x86", "x86_64")),
        BoolVariable("android_neon", "Enable NEON support (armv7 only)", True),
        BoolVariable("store_release", "Editor build for Google Play Store (for official builds only)", False),
    ]


# Return the ANDROID_SDK_ROOT environment variable.
def get_env_android_sdk_root():
    return os.environ.get("ANDROID_SDK_ROOT", -1)


def get_min_sdk_version(platform):
    return int(platform.split("-")[1])


def get_android_ndk_root(env):
    return env["ANDROID_SDK_ROOT"] + "/ndk/" + get_ndk_version()


# This is kept in sync with the value in 'platform/android/java/app/config.gradle'.
def get_ndk_version():
    return "23.2.8568313"


def get_flags():
    return [
        ("tools", False),
    ]


# Check if Android NDK version is installed
# If not, install it.
def install_ndk_if_needed(env):
    print("Checking for Android NDK...")
    sdk_root = env["ANDROID_SDK_ROOT"]
    if not os.path.exists(get_android_ndk_root(env)):
        extension = ".bat" if os.name == "nt" else ""
        sdkmanager = sdk_root + "/cmdline-tools/latest/bin/sdkmanager" + extension
        if os.path.exists(sdkmanager):
            # Install the Android NDK
            print("Installing Android NDK...")
            ndk_download_args = "ndk;" + get_ndk_version()
            subprocess.check_call([sdkmanager, ndk_download_args])
        else:
            print("Cannot find " + sdkmanager)
            print(
                "Please ensure ANDROID_SDK_ROOT is correct and cmdline-tools are installed, or install NDK version "
                + get_ndk_version()
                + " manually."
            )
            sys.exit()
    env["ANDROID_NDK_ROOT"] = get_android_ndk_root(env)


def configure(env):
    install_ndk_if_needed(env)
    ndk_root = env["ANDROID_NDK_ROOT"]

    # Architecture

    if env["android_arch"] not in ["armv7", "arm64v8", "x86", "x86_64"]:
        env["android_arch"] = "arm64v8"

    neon_text = ""
    if env["android_arch"] == "armv7" and env["android_neon"]:
        neon_text = " (with NEON)"
    print("Building for Android (" + env["android_arch"] + ")" + neon_text)

    if get_min_sdk_version(env["ndk_platform"]) < 21:
        if env["android_arch"] == "x86_64" or env["android_arch"] == "arm64v8":
            print(
                "WARNING: android_arch="
                + env["android_arch"]
                + " is not supported by ndk_platform lower than android-21; setting ndk_platform=android-21"
            )
            env["ndk_platform"] = "android-21"

    if env["android_arch"] == "armv7":
        target_triple = "armv7a-linux-androideabi"
        if env["android_neon"]:
            env.extra_suffix = ".armv7.neon" + env.extra_suffix
        else:
            env.extra_suffix = ".armv7" + env.extra_suffix
    elif env["android_arch"] == "arm64v8":
        target_triple = "aarch64-linux-android"
        env.extra_suffix = ".armv8" + env.extra_suffix
    elif env["android_arch"] == "x86":
        target_triple = "i686-linux-android"
        env.extra_suffix = ".x86" + env.extra_suffix
    elif env["android_arch"] == "x86_64":
        target_triple = "x86_64-linux-android"
        env.extra_suffix = ".x86_64" + env.extra_suffix

    target_option = ["-target", target_triple + str(get_min_sdk_version(env["ndk_platform"]))]
    env.Append(ASFLAGS=[target_option, "-c"])
    env.Append(CCFLAGS=target_option)
    env.Append(LINKFLAGS=target_option)

    # Build type

    if env["target"].startswith("release"):
        if env["optimize"] == "speed":  # optimize for speed (default)
            # `-O2` is more friendly to debuggers than `-O3`, leading to better crash backtraces
            # when using `target=release_debug`.
            opt = "-O3" if env["target"] == "release" else "-O2"
            env.Append(CCFLAGS=[opt])
        elif env["optimize"] == "size":  # optimize for size
            env.Append(CCFLAGS=["-Oz"])
    elif env["target"] == "debug":
        env.Append(LINKFLAGS=["-O0"])
        env.Append(CCFLAGS=["-O0", "-g"])

    # LTO

    if env["lto"] == "auto":  # LTO benefits for Android (size, performance) haven't been clearly established yet.
        env["lto"] = "none"

    if env["lto"] != "none":
        if env["lto"] == "thin":
            env.Append(CCFLAGS=["-flto=thin"])
            env.Append(LINKFLAGS=["-flto=thin"])
        else:
            env.Append(CCFLAGS=["-flto"])
            env.Append(LINKFLAGS=["-flto"])

    # Compiler configuration

    env["SHLIBSUFFIX"] = ".so"

    if env["PLATFORM"] == "win32":
        env.use_windows_spawn_fix()

    if sys.platform.startswith("linux"):
        host_subpath = "linux-x86_64"
    elif sys.platform.startswith("darwin"):
        host_subpath = "darwin-x86_64"
    elif sys.platform.startswith("win"):
        if platform.machine().endswith("64"):
            host_subpath = "windows-x86_64"
        else:
            host_subpath = "windows"

    toolchain_path = ndk_root + "/toolchains/llvm/prebuilt/" + host_subpath
    compiler_path = toolchain_path + "/bin"

    env["CC"] = compiler_path + "/clang"
    env["CXX"] = compiler_path + "/clang++"
    env["AR"] = compiler_path + "/llvm-ar"
    env["RANLIB"] = compiler_path + "/llvm-ranlib"
    env["AS"] = compiler_path + "/clang"

    # Disable rtti on non-tools (template) builds.
    if env["tools"]:
        env.Append(CXXFLAGS=["-frtti"])
    else:
        env.Append(CXXFLAGS=["-fno-rtti"])
        # Don't use dynamic_cast, necessary with no-rtti.
        env.Append(CPPDEFINES=["NO_SAFE_CAST"])

    env.Append(
        CCFLAGS="-fpic -ffunction-sections -funwind-tables -fstack-protector-strong -fvisibility=hidden -fno-strict-aliasing".split()
    )
    env.Append(CPPDEFINES=["NO_STATVFS", "GLES_ENABLED"])

    if get_min_sdk_version(env["ndk_platform"]) >= 24:
        env.Append(CPPDEFINES=[("_FILE_OFFSET_BITS", 64)])

    env["neon_enabled"] = False
    if env["android_arch"] == "x86":
        # The NDK adds this if targeting API < 24, so we can drop it when Godot targets it at least
        env.Append(CCFLAGS=["-mstackrealign"])
    elif env["android_arch"] == "armv7":
        env.Append(CCFLAGS="-march=armv7-a -mfloat-abi=softfp".split())
        env.Append(CPPDEFINES=["__ARM_ARCH_7__", "__ARM_ARCH_7A__"])
        if env["android_neon"]:
            env["neon_enabled"] = True
            env.Append(CPPDEFINES=["__ARM_NEON__"])
        else:
            env.Append(CCFLAGS=["-mfpu=vfpv3-d16"])
    elif env["android_arch"] == "arm64v8":
        env.Append(CCFLAGS=["-mfix-cortex-a53-835769"])
        env.Append(CPPDEFINES=["__ARM_ARCH_8A__"])

    # Link flags

    env.Append(LINKFLAGS="-Wl,--gc-sections -Wl,--no-undefined -Wl,-z,now".split())
    env.Append(LINKFLAGS="-Wl,-soname,libgodot_android.so")

    env.Prepend(CPPPATH=["#platform/android"])
    env.Append(CPPDEFINES=["ANDROID_ENABLED", "UNIX_ENABLED", "NO_FCNTL"])
    env.Append(LIBS=["OpenSLES", "EGL", "GLESv3", "GLESv2", "android", "log", "z", "dl"])
