import os
import sys
import platform
from distutils.version import LooseVersion


def is_active():
    return True


def get_name():
    return "Android"


def can_build():
    return ("ANDROID_SDK_ROOT" in os.environ) or ("ANDROID_HOME" in os.environ)


def get_platform(platform):
    return int(platform.split("-")[1])


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        ("ANDROID_NDK_ROOT", "Path to the Android NDK", get_android_ndk_root()),
        ("ANDROID_SDK_ROOT", "Path to the Android SDK", get_android_sdk_root()),
        ("ndk_platform", 'Target platform (android-<api>, e.g. "android-19")', "android-19"),
        EnumVariable("android_arch", "Target architecture", "armv7", ("armv7", "arm64v8", "x86", "x86_64")),
        BoolVariable("android_neon", "Enable NEON support (armv7 only)", True),
    ]


# Return the ANDROID_SDK_ROOT environment variable.
# While ANDROID_HOME has been deprecated, it's used as a fallback for backward
# compatibility purposes.
def get_android_sdk_root():
    if "ANDROID_SDK_ROOT" in os.environ:
        return os.environ.get("ANDROID_SDK_ROOT", 0)
    else:
        return os.environ.get("ANDROID_HOME", 0)


# Return the ANDROID_NDK_ROOT environment variable.
# We generate one for this build using the ANDROID_SDK_ROOT env
# variable and the project ndk version.
# If the env variable is already defined, we override it with
# our own to match what the project expects.
def get_android_ndk_root():
    return get_android_sdk_root() + "/ndk/" + get_project_ndk_version()


def get_flags():
    return [
        ("tools", False),
    ]


def create(env):
    tools = env["TOOLS"]
    if "mingw" in tools:
        tools.remove("mingw")
    if "applelink" in tools:
        tools.remove("applelink")
        env.Tool("gcc")
    return env.Clone(tools=tools)


# Check if ANDROID_NDK_ROOT is valid.
# If not, install the ndk using ANDROID_SDK_ROOT and sdkmanager.
def install_ndk_if_needed(env):
    print("Checking for Android NDK...")
    env_ndk_version = get_env_ndk_version(env["ANDROID_NDK_ROOT"])
    if env_ndk_version is None:
        # Reinstall the ndk and update ANDROID_NDK_ROOT.
        print("Installing Android NDK...")
        if env["ANDROID_SDK_ROOT"] is None:
            raise Exception("Invalid ANDROID_SDK_ROOT environment variable.")

        import subprocess

        extension = ".bat" if os.name == "nt" else ""
        sdkmanager_path = env["ANDROID_SDK_ROOT"] + "/cmdline-tools/latest/bin/sdkmanager" + extension
        ndk_download_args = "ndk;" + get_project_ndk_version()
        subprocess.check_call([sdkmanager_path, ndk_download_args])

        env["ANDROID_NDK_ROOT"] = env["ANDROID_SDK_ROOT"] + "/ndk/" + get_project_ndk_version()
        print("ANDROID_NDK_ROOT: " + env["ANDROID_NDK_ROOT"])


def configure(env):
    install_ndk_if_needed(env)

    # Workaround for MinGW. See:
    # http://www.scons.org/wiki/LongCmdLinesOnWin32
    if os.name == "nt":

        import subprocess

        def mySubProcess(cmdline, env):
            # print("SPAWNED : " + cmdline)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            proc = subprocess.Popen(
                cmdline,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
                shell=False,
                env=env,
            )
            data, err = proc.communicate()
            rv = proc.wait()
            if rv:
                print("=====")
                print(err)
                print("=====")
            return rv

        def mySpawn(sh, escape, cmd, args, env):

            newargs = " ".join(args[1:])
            cmdline = cmd + " " + newargs

            rv = 0
            if len(cmdline) > 32000 and cmd.endswith("ar"):
                cmdline = cmd + " " + args[1] + " " + args[2] + " "
                for i in range(3, len(args)):
                    rv = mySubProcess(cmdline + args[i], env)
                    if rv:
                        break
            else:
                rv = mySubProcess(cmdline, env)

            return rv

        env["SPAWN"] = mySpawn

    # Architecture

    if env["android_arch"] not in ["armv7", "arm64v8", "x86", "x86_64"]:
        env["android_arch"] = "armv7"

    neon_text = ""
    if env["android_arch"] == "armv7" and env["android_neon"]:
        neon_text = " (with NEON)"
    print("Building for Android (" + env["android_arch"] + ")" + neon_text)

    can_vectorize = True
    if env["android_arch"] == "x86":
        env["ARCH"] = "arch-x86"
        env.extra_suffix = ".x86" + env.extra_suffix
        target_subpath = "x86-4.9"
        abi_subpath = "i686-linux-android"
        arch_subpath = "x86"
        env["x86_libtheora_opt_gcc"] = True
    elif env["android_arch"] == "x86_64":
        if get_platform(env["ndk_platform"]) < 21:
            print(
                "WARNING: android_arch=x86_64 is not supported by ndk_platform lower than android-21; setting ndk_platform=android-21"
            )
            env["ndk_platform"] = "android-21"
        env["ARCH"] = "arch-x86_64"
        env.extra_suffix = ".x86_64" + env.extra_suffix
        target_subpath = "x86_64-4.9"
        abi_subpath = "x86_64-linux-android"
        arch_subpath = "x86_64"
        env["x86_libtheora_opt_gcc"] = True
    elif env["android_arch"] == "armv7":
        env["ARCH"] = "arch-arm"
        target_subpath = "arm-linux-androideabi-4.9"
        abi_subpath = "arm-linux-androideabi"
        arch_subpath = "armeabi-v7a"
        if env["android_neon"]:
            env.extra_suffix = ".armv7.neon" + env.extra_suffix
        else:
            env.extra_suffix = ".armv7" + env.extra_suffix
    elif env["android_arch"] == "arm64v8":
        if get_platform(env["ndk_platform"]) < 21:
            print(
                "WARNING: android_arch=arm64v8 is not supported by ndk_platform lower than android-21; setting ndk_platform=android-21"
            )
            env["ndk_platform"] = "android-21"
        env["ARCH"] = "arch-arm64"
        target_subpath = "aarch64-linux-android-4.9"
        abi_subpath = "aarch64-linux-android"
        arch_subpath = "arm64-v8a"
        env.extra_suffix = ".armv8" + env.extra_suffix

    # Build type

    if env["target"].startswith("release"):
        if env["optimize"] == "speed":  # optimize for speed (default)
            env.Append(LINKFLAGS=["-O2"])
            env.Append(CCFLAGS=["-O2", "-fomit-frame-pointer"])
        elif env["optimize"] == "size":  # optimize for size
            env.Append(CCFLAGS=["-Os"])
            env.Append(LINKFLAGS=["-Os"])

        env.Append(CPPDEFINES=["NDEBUG"])
        if can_vectorize:
            env.Append(CCFLAGS=["-ftree-vectorize"])
    elif env["target"] == "debug":
        env.Append(LINKFLAGS=["-O0"])
        env.Append(CCFLAGS=["-O0", "-g", "-fno-limit-debug-info"])
        env.Append(CPPDEFINES=["_DEBUG"])
        env.Append(CPPFLAGS=["-UNDEBUG"])

    # Compiler configuration

    env["SHLIBSUFFIX"] = ".so"

    if env["PLATFORM"] == "win32":
        env.Tool("gcc")
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

    compiler_path = env["ANDROID_NDK_ROOT"] + "/toolchains/llvm/prebuilt/" + host_subpath + "/bin"
    gcc_toolchain_path = env["ANDROID_NDK_ROOT"] + "/toolchains/" + target_subpath + "/prebuilt/" + host_subpath
    tools_path = gcc_toolchain_path + "/" + abi_subpath + "/bin"

    # For Clang to find NDK tools in preference of those system-wide
    env.PrependENVPath("PATH", tools_path)

    ccache_path = os.environ.get("CCACHE")
    if ccache_path is None:
        env["CC"] = compiler_path + "/clang"
        env["CXX"] = compiler_path + "/clang++"
    else:
        # there aren't any ccache wrappers available for Android,
        # to enable caching we need to prepend the path to the ccache binary
        env["CC"] = ccache_path + " " + compiler_path + "/clang"
        env["CXX"] = ccache_path + " " + compiler_path + "/clang++"
    env["AR"] = tools_path + "/ar"
    env["RANLIB"] = tools_path + "/ranlib"
    env["AS"] = tools_path + "/as"

    common_opts = ["-gcc-toolchain", gcc_toolchain_path]

    # Compile flags

    env.Append(CPPFLAGS=["-isystem", env["ANDROID_NDK_ROOT"] + "/sources/cxx-stl/llvm-libc++/include"])
    env.Append(CPPFLAGS=["-isystem", env["ANDROID_NDK_ROOT"] + "/sources/cxx-stl/llvm-libc++abi/include"])

    # Disable exceptions and rtti on non-tools (template) builds
    if env["tools"]:
        env.Append(CXXFLAGS=["-frtti"])
    else:
        env.Append(CXXFLAGS=["-fno-rtti", "-fno-exceptions"])
        # Don't use dynamic_cast, necessary with no-rtti.
        env.Append(CPPDEFINES=["NO_SAFE_CAST"])

    lib_sysroot = env["ANDROID_NDK_ROOT"] + "/platforms/" + env["ndk_platform"] + "/" + env["ARCH"]

    # Using NDK unified headers (NDK r15+)
    sysroot = env["ANDROID_NDK_ROOT"] + "/sysroot"
    env.Append(CPPFLAGS=["--sysroot=" + sysroot])
    env.Append(CPPFLAGS=["-isystem", sysroot + "/usr/include/" + abi_subpath])
    env.Append(CPPFLAGS=["-isystem", env["ANDROID_NDK_ROOT"] + "/sources/android/support/include"])
    # For unified headers this define has to be set manually
    env.Append(CPPDEFINES=[("__ANDROID_API__", str(get_platform(env["ndk_platform"])))])

    env.Append(
        CCFLAGS="-fpic -ffunction-sections -funwind-tables -fstack-protector-strong -fvisibility=hidden -fno-strict-aliasing".split()
    )
    env.Append(CPPDEFINES=["NO_STATVFS", "GLES_ENABLED"])

    if get_platform(env["ndk_platform"]) >= 24:
        env.Append(CPPDEFINES=[("_FILE_OFFSET_BITS", 64)])

    env["neon_enabled"] = False
    if env["android_arch"] == "x86":
        target_opts = ["-target", "i686-none-linux-android"]
        # The NDK adds this if targeting API < 21, so we can drop it when Godot targets it at least
        env.Append(CCFLAGS=["-mstackrealign"])

    elif env["android_arch"] == "x86_64":
        target_opts = ["-target", "x86_64-none-linux-android"]

    elif env["android_arch"] == "armv7":
        target_opts = ["-target", "armv7-none-linux-androideabi"]
        env.Append(CCFLAGS="-march=armv7-a -mfloat-abi=softfp".split())
        env.Append(CPPDEFINES=["__ARM_ARCH_7__", "__ARM_ARCH_7A__"])
        if env["android_neon"]:
            env["neon_enabled"] = True
            env.Append(CCFLAGS=["-mfpu=neon"])
            env.Append(CPPDEFINES=["__ARM_NEON__"])
        else:
            env.Append(CCFLAGS=["-mfpu=vfpv3-d16"])

    elif env["android_arch"] == "arm64v8":
        target_opts = ["-target", "aarch64-none-linux-android"]
        env.Append(CCFLAGS=["-mfix-cortex-a53-835769"])
        env.Append(CPPDEFINES=["__ARM_ARCH_8A__"])

    env.Append(CCFLAGS=target_opts)
    env.Append(CCFLAGS=common_opts)

    # Link flags

    ndk_version = get_env_ndk_version(env["ANDROID_NDK_ROOT"])
    if ndk_version != None and LooseVersion(ndk_version) >= LooseVersion("17.1.4828580"):
        env.Append(LINKFLAGS=["-Wl,--exclude-libs,libgcc.a", "-Wl,--exclude-libs,libatomic.a", "-nostdlib++"])
    else:
        env.Append(
            LINKFLAGS=[
                env["ANDROID_NDK_ROOT"] + "/sources/cxx-stl/llvm-libc++/libs/" + arch_subpath + "/libandroid_support.a"
            ]
        )
    env.Append(LINKFLAGS=["-shared", "--sysroot=" + lib_sysroot, "-Wl,--warn-shared-textrel"])
    env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"] + "/sources/cxx-stl/llvm-libc++/libs/" + arch_subpath + "/"])
    env.Append(
        LINKFLAGS=[env["ANDROID_NDK_ROOT"] + "/sources/cxx-stl/llvm-libc++/libs/" + arch_subpath + "/libc++_shared.so"]
    )

    if env["android_arch"] == "armv7":
        env.Append(LINKFLAGS="-Wl,--fix-cortex-a8".split())
    env.Append(LINKFLAGS="-Wl,--no-undefined -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now".split())
    env.Append(LINKFLAGS="-Wl,-soname,libgodot_android.so -Wl,--gc-sections".split())

    env.Append(LINKFLAGS=target_opts)
    env.Append(LINKFLAGS=common_opts)

    env.Append(
        LIBPATH=[
            env["ANDROID_NDK_ROOT"]
            + "/toolchains/"
            + target_subpath
            + "/prebuilt/"
            + host_subpath
            + "/lib/gcc/"
            + abi_subpath
            + "/4.9.x"
        ]
    )
    env.Append(
        LIBPATH=[
            env["ANDROID_NDK_ROOT"]
            + "/toolchains/"
            + target_subpath
            + "/prebuilt/"
            + host_subpath
            + "/"
            + abi_subpath
            + "/lib"
        ]
    )

    env.Prepend(CPPPATH=["#platform/android"])
    env.Append(CPPDEFINES=["ANDROID_ENABLED", "UNIX_ENABLED", "NO_FCNTL"])
    env.Append(LIBS=["OpenSLES", "EGL", "GLESv3", "GLESv2", "android", "log", "z", "dl"])


# Return the project NDK version.
# This is kept in sync with the value in 'platform/android/java/app/config.gradle'.
def get_project_ndk_version():
    return "21.4.7075529"


# Return NDK version string in source.properties (adapted from the Chromium project).
def get_env_ndk_version(path):
    if path is None:
        return None
    prop_file_path = os.path.join(path, "source.properties")
    try:
        with open(prop_file_path) as prop_file:
            for line in prop_file:
                key_value = list(map(lambda x: x.strip(), line.split("=")))
                if key_value[0] == "Pkg.Revision":
                    return key_value[1]
    except Exception:
        print("Could not read source prop file '%s'" % prop_file_path)
    return None
