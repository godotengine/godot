import methods
import os
import sys


def is_active():
    return True


def get_name():
    return "Windows"


def can_build():

    if (os.name == "nt"):
        # Building natively on Windows
        if (os.getenv("VCINSTALLDIR")): # MSVC
            return True

        print("MSVC not detected (no VCINSTALLDIR environment variable), attempting MinGW.")
        mingw32 = ""
        mingw64 = ""
        if (os.getenv("MINGW32_PREFIX")):
            mingw32 = os.getenv("MINGW32_PREFIX")
        if (os.getenv("MINGW64_PREFIX")):
            mingw64 = os.getenv("MINGW64_PREFIX")

        test = "gcc --version > NUL 2>&1"
        if (os.system(test) == 0 or os.system(mingw32 + test) == 0 or os.system(mingw64 + test) == 0):
            return True

    if (os.name == "posix"):
        # Cross-compiling with MinGW-w64 (old MinGW32 is not supported)
        mingw32 = "i686-w64-mingw32-"
        mingw64 = "x86_64-w64-mingw32-"

        if (os.getenv("MINGW32_PREFIX")):
            mingw32 = os.getenv("MINGW32_PREFIX")
        if (os.getenv("MINGW64_PREFIX")):
            mingw64 = os.getenv("MINGW64_PREFIX")

        test = "gcc --version > /dev/null 2>&1"
        if (os.system(mingw64 + test) == 0 or os.system(mingw32 + test) == 0):
            return True

    print("Could not detect MinGW. Ensure its binaries are in your PATH or that MINGW32_PREFIX or MINGW64_PREFIX are properly defined.")
    return False


def get_opts():

    mingw32 = ""
    mingw64 = ""
    if (os.name == "posix"):
        mingw32 = "i686-w64-mingw32-"
        mingw64 = "x86_64-w64-mingw32-"

    if (os.getenv("MINGW32_PREFIX")):
        mingw32 = os.getenv("MINGW32_PREFIX")
    if (os.getenv("MINGW64_PREFIX")):
        mingw64 = os.getenv("MINGW64_PREFIX")

    return [
        ('mingw_prefix_32', 'MinGW prefix (Win32)', mingw32),
        ('mingw_prefix_64', 'MinGW prefix (Win64)', mingw64),
    ]


def get_flags():

    return [
    ]


def build_res_file(target, source, env):

    if (env["bits"] == "32"):
        cmdbase = env['mingw_prefix_32']
    else:
        cmdbase = env['mingw_prefix_64']
    cmdbase = cmdbase + 'windres --include-dir . '
    import subprocess
    for x in range(len(source)):
        cmd = cmdbase + '-i ' + str(source[x]) + ' -o ' + str(target[x])
        try:
            out = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE).communicate()
            if len(out[1]):
                return 1
        except:
            return 1
    return 0


def configure(env):

    env.Append(CPPPATH=['#platform/windows'])

    # Targeted Windows version: 7 (and later), minimum supported version
    # XP support dropped after EOL due to missing API for IPv6 and other issues
    # Vista support dropped after EOL due to GH-10243
    winver = "0x0601"

    if (os.name == "nt" and os.getenv("VCINSTALLDIR")): # MSVC

        env['ENV']['TMP'] = os.environ['TMP']

        ## Build type

        if (env["target"] == "release"):
            env.Append(CCFLAGS=['/O2'])
            env.Append(LINKFLAGS=['/SUBSYSTEM:WINDOWS'])
            env.Append(LINKFLAGS=['/ENTRY:mainCRTStartup'])

        elif (env["target"] == "release_debug"):
            env.Append(CCFLAGS=['/O2', '/DDEBUG_ENABLED'])
            env.Append(LINKFLAGS=['/SUBSYSTEM:CONSOLE'])

        elif (env["target"] == "debug_release"):
            env.Append(CCFLAGS=['/Z7', '/Od'])
            env.Append(LINKFLAGS=['/DEBUG'])
            env.Append(LINKFLAGS=['/SUBSYSTEM:WINDOWS'])
            env.Append(LINKFLAGS=['/ENTRY:mainCRTStartup'])

        elif (env["target"] == "debug"):
            env.Append(CCFLAGS=['/Z7', '/DDEBUG_ENABLED', '/DDEBUG_MEMORY_ENABLED', '/DD3D_DEBUG_INFO', '/Od', '/EHsc'])
            env.Append(LINKFLAGS=['/SUBSYSTEM:CONSOLE'])
            env.Append(LINKFLAGS=['/DEBUG'])

        ## Architecture

        # Note: this detection/override code from here onward should be here instead of in SConstruct because it's platform and compiler specific (MSVC/Windows)
        if (env["bits"] != "default"):
            print("Error: bits argument is disabled for MSVC")
            print("""
                Bits argument is not supported for MSVC compilation. Architecture depends on the Native/Cross Compile Tools Prompt/Developer Console
                (or Visual Studio settings) that is being used to run SCons. As a consequence, bits argument is disabled. Run scons again without bits
                argument (example: scons p=windows) and SCons will attempt to detect what MSVC compiler will be executed and inform you.
                """)
            sys.exit()

        # Forcing bits argument because MSVC does not have a flag to set this through SCons... it's different compilers (cl.exe's) called from the proper command prompt
        # that decide the architecture that is build for. Scons can only detect the os.getenviron (because vsvarsall.bat sets a lot of stuff for cl.exe to work with)
        env["bits"] = "32"
        env["x86_libtheora_opt_vc"] = True

        ## Compiler configuration

        env['ENV'] = os.environ
        # This detection function needs the tools env (that is env['ENV'], not SCons's env), and that is why it's this far bellow in the code
        compiler_version_str = methods.detect_visual_c_compiler_version(env['ENV'])

        print("Detected MSVC compiler: " + compiler_version_str)
        # If building for 64bit architecture, disable assembly optimisations for 32 bit builds (theora as of writting)... vc compiler for 64bit can not compile _asm
        if(compiler_version_str == "amd64" or compiler_version_str == "x86_amd64"):
            env["bits"] = "64"
            env["x86_libtheora_opt_vc"] = False
            print("Compiled program architecture will be a 64 bit executable (forcing bits=64).")
        elif (compiler_version_str == "x86" or compiler_version_str == "amd64_x86"):
            print("Compiled program architecture will be a 32 bit executable. (forcing bits=32).")
        else:
            print("Failed to detect MSVC compiler architecture version... Defaulting to 32bit executable settings (forcing bits=32). Compilation attempt will continue, but SCons can not detect for what architecture this build is compiled for. You should check your settings/compilation setup.")

        ## Compile flags

        env.Append(CCFLAGS=['/MT', '/Gd', '/GR', '/nologo'])
        env.Append(CXXFLAGS=['/TP'])
        env.Append(CPPFLAGS=['/DMSVC', '/GR', ])
        env.Append(CCFLAGS=['/I' + os.getenv("WindowsSdkDir") + "/Include"])

        env.Append(CCFLAGS=['/DWINDOWS_ENABLED'])
        env.Append(CCFLAGS=['/DOPENGL_ENABLED'])
        env.Append(CCFLAGS=['/DRTAUDIO_ENABLED'])
        env.Append(CCFLAGS=['/DWASAPI_ENABLED'])
        env.Append(CCFLAGS=['/DTYPED_METHOD_BIND'])
        env.Append(CCFLAGS=['/DWIN32'])
        env.Append(CCFLAGS=['/DWINVER=%s' % winver, '/D_WIN32_WINNT=%s' % winver])
        if env["bits"] == "64":
            env.Append(CCFLAGS=['/D_WIN64'])

        LIBS = ['winmm', 'opengl32', 'dsound', 'kernel32', 'ole32', 'oleaut32', 'user32', 'gdi32', 'IPHLPAPI', 'Shlwapi', 'wsock32', 'Ws2_32', 'shell32', 'advapi32', 'dinput8', 'dxguid']
        env.Append(LINKFLAGS=[p + env["LIBSUFFIX"] for p in LIBS])

        env.Append(LIBPATH=[os.getenv("WindowsSdkDir") + "/Lib"])

        if (os.getenv("VCINSTALLDIR")):
            VC_PATH = os.getenv("VCINSTALLDIR")
        else:
            VC_PATH = ""

        env.Append(CCFLAGS=["/I" + p for p in os.getenv("INCLUDE").split(";")])
        env.Append(LIBPATH=[p for p in os.getenv("LIB").split(";")])

        # Incremental linking fix
        env['BUILDERS']['ProgramOriginal'] = env['BUILDERS']['Program']
        env['BUILDERS']['Program'] = methods.precious_program

    else: # MinGW

        # Workaround for MinGW. See:
        # http://www.scons.org/wiki/LongCmdLinesOnWin32
        env.use_windows_spawn_fix()

        ## Build type

        if (env["target"] == "release"):
            env.Append(CCFLAGS=['-msse2'])

            if (env["bits"] == "64"):
                env.Append(CCFLAGS=['-O3'])
            else:
                env.Append(CCFLAGS=['-O2'])

            env.Append(LINKFLAGS=['-Wl,--subsystem,windows'])

        elif (env["target"] == "release_debug"):
            env.Append(CCFLAGS=['-O2', '-DDEBUG_ENABLED'])

        elif (env["target"] == "debug"):
            env.Append(CCFLAGS=['-g', '-DDEBUG_ENABLED', '-DDEBUG_MEMORY_ENABLED'])

        ## Compiler configuration

        if (os.name == "nt"):
            env['ENV']['TMP'] = os.environ['TMP']  # way to go scons, you can be so stupid sometimes
        else:
            env["PROGSUFFIX"] = env["PROGSUFFIX"] + ".exe"  # for linux cross-compilation

        if (env["bits"] == "default"):
            if (os.name == "nt"):
                env["bits"] = "64" if "PROGRAMFILES(X86)" in os.environ else "32"
            else: # default to 64-bit on Linux
                env["bits"] = "64"

        mingw_prefix = ""

        if (env["bits"] == "32"):
            env.Append(LINKFLAGS=['-static'])
            env.Append(LINKFLAGS=['-static-libgcc'])
            env.Append(LINKFLAGS=['-static-libstdc++'])
            mingw_prefix = env["mingw_prefix_32"]
        else:
            env.Append(LINKFLAGS=['-static'])
            mingw_prefix = env["mingw_prefix_64"]

        env["CC"] = mingw_prefix + "gcc"
        env['AS'] = mingw_prefix + "as"
        env['CXX'] = mingw_prefix + "g++"
        env['AR'] = mingw_prefix + "ar"
        env['RANLIB'] = mingw_prefix + "ranlib"
        env['LD'] = mingw_prefix + "g++"
        env["x86_libtheora_opt_gcc"] = True

        ## Compile flags

        env.Append(CCFLAGS=['-DWINDOWS_ENABLED', '-mwindows'])
        env.Append(CCFLAGS=['-DOPENGL_ENABLED'])
        env.Append(CCFLAGS=['-DRTAUDIO_ENABLED'])
        env.Append(CCFLAGS=['-DWASAPI_ENABLED'])
        env.Append(CCFLAGS=['-DWINVER=%s' % winver, '-D_WIN32_WINNT=%s' % winver])
        env.Append(LIBS=['mingw32', 'opengl32', 'dsound', 'ole32', 'd3d9', 'winmm', 'gdi32', 'iphlpapi', 'shlwapi', 'wsock32', 'ws2_32', 'kernel32', 'oleaut32', 'dinput8', 'dxguid', 'ksuser'])

        env.Append(CPPFLAGS=['-DMINGW_ENABLED'])

        # resrc
        env.Append(BUILDERS={'RES': env.Builder(action=build_res_file, suffix='.o', src_suffix='.rc')})
