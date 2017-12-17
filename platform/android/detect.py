import os
import sys
import string
import platform
from distutils.version import LooseVersion


def is_active():
    return True


def get_name():
    return "Android"


def can_build():

    return ("ANDROID_NDK_ROOT" in os.environ)


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        ('ANDROID_NDK_ROOT', 'Path to the Android NDK', os.environ.get("ANDROID_NDK_ROOT", 0)),
        ('ndk_platform', 'Target platform (android-<api>, e.g. "android-18")', "android-18"),
        EnumVariable('android_arch', 'Target architecture', "armv7", ('armv7', 'armv6', 'arm64v8', 'x86')),
        BoolVariable('android_neon', 'Enable NEON support (armv7 only)', True),
        BoolVariable('android_stl', 'Enable Android STL support (for modules)', True)
    ]


def get_flags():

    return [
        ('tools', False),
    ]


def create(env):

    tools = env['TOOLS']
    if "mingw" in tools:
        tools.remove('mingw')
    if "applelink" in tools:
        tools.remove("applelink")
        env.Tool('gcc')
    return env.Clone(tools=tools)


def configure(env):

    # Workaround for MinGW. See:
    # http://www.scons.org/wiki/LongCmdLinesOnWin32
    if (os.name == "nt"):

        import subprocess

        def mySubProcess(cmdline, env):
            # print("SPAWNED : " + cmdline)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            proc = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, startupinfo=startupinfo, shell=False, env=env)
            data, err = proc.communicate()
            rv = proc.wait()
            if rv:
                print("=====")
                print(err)
                print("=====")
            return rv

        def mySpawn(sh, escape, cmd, args, env):

            newargs = ' '.join(args[1:])
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

        env['SPAWN'] = mySpawn

    ## Architecture

    if env['android_arch'] not in ['armv7', 'armv6', 'arm64v8', 'x86']:
        env['android_arch'] = 'armv7'

    neon_text = ""
    if env["android_arch"] == "armv7" and env['android_neon']:
        neon_text = " (with NEON)"
    print("Building for Android (" + env['android_arch'] + ")" + neon_text)

    can_vectorize = True
    if env['android_arch'] == 'x86':
        env['ARCH'] = 'arch-x86'
        env.extra_suffix = ".x86" + env.extra_suffix
        target_subpath = "x86-4.9"
        abi_subpath = "i686-linux-android"
        arch_subpath = "x86"
        env["x86_libtheora_opt_gcc"] = True
    elif env['android_arch'] == 'armv6':
        env['ARCH'] = 'arch-arm'
        env.extra_suffix = ".armv6" + env.extra_suffix
        target_subpath = "arm-linux-androideabi-4.9"
        abi_subpath = "arm-linux-androideabi"
        arch_subpath = "armeabi"
        can_vectorize = False
    elif env["android_arch"] == "armv7":
        env['ARCH'] = 'arch-arm'
        target_subpath = "arm-linux-androideabi-4.9"
        abi_subpath = "arm-linux-androideabi"
        arch_subpath = "armeabi-v7a"
        if env['android_neon']:
            env.extra_suffix = ".armv7.neon" + env.extra_suffix
        else:
            env.extra_suffix = ".armv7" + env.extra_suffix
    elif env["android_arch"] == "arm64v8":
        env['ARCH'] = 'arch-arm64'
        target_subpath = "aarch64-linux-android-4.9"
        abi_subpath = "aarch64-linux-android"
        arch_subpath = "arm64-v8a"
        env.extra_suffix = ".armv8" + env.extra_suffix

    ## Build type

    if (env["target"].startswith("release")):
        env.Append(LINKFLAGS=['-O2'])
        env.Append(CPPFLAGS=['-O2', '-DNDEBUG', '-ffast-math', '-funsafe-math-optimizations', '-fomit-frame-pointer'])
        if (can_vectorize):
            env.Append(CPPFLAGS=['-ftree-vectorize'])
        if (env["target"] == "release_debug"):
            env.Append(CPPFLAGS=['-DDEBUG_ENABLED'])
    elif (env["target"] == "debug"):
        env.Append(LINKFLAGS=['-O0'])
        env.Append(CPPFLAGS=['-O0', '-D_DEBUG', '-UNDEBUG', '-DDEBUG_ENABLED',
                             '-DDEBUG_MEMORY_ENABLED', '-g', '-fno-limit-debug-info'])

    ## Compiler configuration

    env['SHLIBSUFFIX'] = '.so'

    if env['PLATFORM'] == 'win32':
        env.Tool('gcc')
        env.use_windows_spawn_fix()

    mt_link = True
    if (sys.platform.startswith("linux")):
        host_subpath = "linux-x86_64"
    elif (sys.platform.startswith("darwin")):
        host_subpath = "darwin-x86_64"
    elif (sys.platform.startswith('win')):
        if (platform.machine().endswith('64')):
            host_subpath = "windows-x86_64"
            if env["android_arch"] == "arm64v8":
                mt_link = False
        else:
            mt_link = False
            host_subpath = "windows"

    compiler_path = env["ANDROID_NDK_ROOT"] + "/toolchains/llvm/prebuilt/" + host_subpath + "/bin"
    gcc_toolchain_path = env["ANDROID_NDK_ROOT"] + "/toolchains/" + target_subpath + "/prebuilt/" + host_subpath
    tools_path = gcc_toolchain_path + "/" + abi_subpath + "/bin"

    # For Clang to find NDK tools in preference of those system-wide
    env.PrependENVPath('PATH', tools_path)

    ccache_path = os.environ.get("CCACHE")
    if ccache_path == None:
        env['CC'] = compiler_path + '/clang'
        env['CXX'] = compiler_path + '/clang++'
    else:
        # there aren't any ccache wrappers available for Android,
        # to enable caching we need to prepend the path to the ccache binary
        env['CC'] = ccache_path + ' ' + compiler_path + '/clang'
        env['CXX'] = ccache_path + ' ' + compiler_path + '/clang++'
    env['AR'] = tools_path + "/ar"
    env['RANLIB'] = tools_path + "/ranlib"
    env['AS'] = tools_path + "/as"

    common_opts = ['-fno-integrated-as', '-gcc-toolchain', gcc_toolchain_path]

    lib_sysroot = env["ANDROID_NDK_ROOT"] + "/platforms/" + env['ndk_platform'] + "/" + env['ARCH']

    ## Compile flags

    ndk_version = get_ndk_version(env["ANDROID_NDK_ROOT"])
    if ndk_version != None and LooseVersion(ndk_version) >= LooseVersion("15.0.4075724"):
        print("Using NDK unified headers")
        sysroot = env["ANDROID_NDK_ROOT"] + "/sysroot"
        env.Append(CPPFLAGS=["-isystem", sysroot + "/usr/include"])
        env.Append(CPPFLAGS=["-isystem", sysroot + "/usr/include/" + abi_subpath])
        # For unified headers this define has to be set manually
        env.Append(CPPFLAGS=["-D__ANDROID_API__=" + str(int(env['ndk_platform'].split("-")[1]))])
    else:
        print("Using NDK deprecated headers")
        env.Append(CPPFLAGS=["-isystem", lib_sysroot + "/usr/include"])

    env.Append(CPPFLAGS='-fpic -ffunction-sections -funwind-tables -fstack-protector-strong -fvisibility=hidden -fno-strict-aliasing'.split())
    env.Append(CPPFLAGS='-DNO_STATVFS -DGLES_ENABLED'.split())

    env['neon_enabled'] = False
    if env['android_arch'] == 'x86':
        target_opts = ['-target', 'i686-none-linux-android']
        # The NDK adds this if targeting API < 21, so we can drop it when Godot targets it at least
        env.Append(CPPFLAGS=['-mstackrealign'])

    elif env["android_arch"] == "armv6":
        target_opts = ['-target', 'armv6-none-linux-androideabi']
        env.Append(CPPFLAGS='-D__ARM_ARCH_6__ -march=armv6 -mfpu=vfp -mfloat-abi=softfp'.split())

    elif env["android_arch"] == "armv7":
        target_opts = ['-target', 'armv7-none-linux-androideabi']
        env.Append(CPPFLAGS='-D__ARM_ARCH_7__ -D__ARM_ARCH_7A__ -march=armv7-a -mfloat-abi=softfp'.split())
        if env['android_neon']:
            env['neon_enabled'] = True
            env.Append(CPPFLAGS=['-mfpu=neon', '-D__ARM_NEON__'])
        else:
            env.Append(CPPFLAGS=['-mfpu=vfpv3-d16'])

    elif env["android_arch"] == "arm64v8":
        target_opts = ['-target', 'aarch64-none-linux-android']
        env.Append(CPPFLAGS=['-D__ARM_ARCH_8A__'])
        env.Append(CPPFLAGS=['-mfix-cortex-a53-835769'])

    env.Append(CPPFLAGS=target_opts)
    env.Append(CPPFLAGS=common_opts)

    if env['android_stl']:
        env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"] + "/sources/cxx-stl/gnu-libstdc++/4.9/include"])
        env.Append(CPPPATH=[env["ANDROID_NDK_ROOT"] + "/sources/cxx-stl/gnu-libstdc++/4.9/libs/" + arch_subpath + "/include"])
        env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"] + "/sources/cxx-stl/gnu-libstdc++/4.9/libs/" + arch_subpath])
        env.Append(LIBS=["gnustl_static"])
    else:
        env.Append(CXXFLAGS=['-fno-rtti', '-fno-exceptions', '-DNO_SAFE_CAST'])

    ## Link flags

    env['LINKFLAGS'] = ['-shared', '--sysroot=' + lib_sysroot, '-Wl,--warn-shared-textrel']
    if env["android_arch"] == "armv7":
        env.Append(LINKFLAGS='-Wl,--fix-cortex-a8'.split())
    env.Append(LINKFLAGS='-Wl,--no-undefined -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now'.split())
    env.Append(LINKFLAGS='-Wl,-soname,libgodot_android.so -Wl,--gc-sections'.split())
    if mt_link:
        env.Append(LINKFLAGS=['-Wl,--threads'])
    env.Append(LINKFLAGS=target_opts)
    env.Append(LINKFLAGS=common_opts)

    env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"] + '/toolchains/arm-linux-androideabi-4.9/prebuilt/' +
                        host_subpath + '/lib/gcc/' + abi_subpath + '/4.9.x'])
    env.Append(LIBPATH=[env["ANDROID_NDK_ROOT"] +
                        '/toolchains/arm-linux-androideabi-4.9/prebuilt/' + host_subpath + '/' + abi_subpath + '/lib'])

    env.Append(CPPPATH=['#platform/android'])
    env.Append(CPPFLAGS=['-DANDROID_ENABLED', '-DUNIX_ENABLED', '-DNO_FCNTL', '-DMPC_FIXED_POINT'])
    env.Append(LIBS=['OpenSLES', 'EGL', 'GLESv3', 'android', 'log', 'z', 'dl'])

    # TODO: Move that to opus module's config
    if 'module_opus_enabled' in env and env['module_opus_enabled']:
        if (env["android_arch"] == "armv6" or env["android_arch"] == "armv7"):
            env.Append(CFLAGS=["-DOPUS_ARM_OPT"])
        env.opus_fixed_point = "yes"

# Return NDK version string in source.properties (adapted from the Chromium project).
def get_ndk_version(path):
    if path == None:
        return None
    prop_file_path = os.path.join(path, "source.properties")
    try:
        with open(prop_file_path) as prop_file:
            for line in prop_file:
                key_value = list(map(lambda x: x.strip(), line.split("=")))
                if key_value[0] == "Pkg.Revision":
                    return key_value[1]
    except:
        print("Could not read source prop file '%s'" % prop_file_path)
    return None
