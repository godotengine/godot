import methods
import os
import sys


def is_active():
    return True


def get_name():
    return "Windows"


def can_build():

    if (os.name == "nt"):
        # building natively on windows!
        if (os.getenv("VCINSTALLDIR")):
            return True
        else:
            print("\nMSVC not detected, attempting MinGW.")
            mingw32 = ""
            mingw64 = ""
            if (os.getenv("MINGW32_PREFIX")):
                mingw32 = os.getenv("MINGW32_PREFIX")
            if (os.getenv("MINGW64_PREFIX")):
                mingw64 = os.getenv("MINGW64_PREFIX")

            test = "gcc --version > NUL 2>&1"
            if os.system(test) != 0 and os.system(mingw32 + test) != 0 and os.system(mingw64 + test) != 0:
                print("- could not detect gcc.")
                print("Please, make sure a path to a MinGW /bin directory is accessible into the environment PATH.\n")
                return False
            else:
                print("- gcc detected.")

            return True

    if (os.name == "posix"):

        mingw = "i586-mingw32msvc-"
        mingw64 = "x86_64-w64-mingw32-"
        mingw32 = "i686-w64-mingw32-"

        if (os.getenv("MINGW32_PREFIX")):
            mingw32 = os.getenv("MINGW32_PREFIX")
            mingw = mingw32
        if (os.getenv("MINGW64_PREFIX")):
            mingw64 = os.getenv("MINGW64_PREFIX")

        test = "gcc --version &>/dev/null"
        if (os.system(mingw + test) == 0 or os.system(mingw64 + test) == 0 or os.system(mingw32 + test) == 0):
            return True

    return False


def get_opts():

    mingw = ""
    mingw32 = ""
    mingw64 = ""
    if (os.name == "posix"):
        mingw = "i586-mingw32msvc-"
        mingw32 = "i686-w64-mingw32-"
        mingw64 = "x86_64-w64-mingw32-"

        if os.system(mingw32 + "gcc --version &>/dev/null") != 0:
            mingw32 = mingw

    if (os.getenv("MINGW32_PREFIX")):
        mingw32 = os.getenv("MINGW32_PREFIX")
        mingw = mingw32
    if (os.getenv("MINGW64_PREFIX")):
        mingw64 = os.getenv("MINGW64_PREFIX")

    return [
        ('mingw_prefix', 'MinGW Prefix', mingw32),
        ('mingw_prefix_64', 'MinGW Prefix 64 bits', mingw64),
    ]


def get_flags():

    return [
    ]


def build_res_file(target, source, env):

    cmdbase = ""
    if (env["bits"] == "32"):
        cmdbase = env['mingw_prefix']
    else:
        cmdbase = env['mingw_prefix_64']
    CPPPATH = env['CPPPATH']
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

    # Targeted Windows version: Vista (and later)
    winver = "0x0600" # Windows Vista is the minimum target for windows builds

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
            env.Append(CCFLAGS=['/Z7', '/DDEBUG_ENABLED', '/DDEBUG_MEMORY_ENABLED', '/DD3D_DEBUG_INFO', '/Od'])
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

        mingw_prefix = ""

        if (env["bits"] == "default"):
            env["bits"] = "64" if "PROGRAMFILES(X86)" in os.environ else "32"

        if (env["bits"] == "32"):
            env.Append(LINKFLAGS=['-static'])
            env.Append(LINKFLAGS=['-static-libgcc'])
            env.Append(LINKFLAGS=['-static-libstdc++'])
            mingw_prefix = env["mingw_prefix"]
        else:
            env.Append(LINKFLAGS=['-static'])
            mingw_prefix = env["mingw_prefix_64"]

        nulstr = ""

        if (os.name == "posix"):
            nulstr = ">/dev/null"
        else:
            nulstr = ">nul"

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
        env.Append(CCFLAGS=['-DWINVER=%s' % winver, '-D_WIN32_WINNT=%s' % winver])
        env.Append(LIBS=['mingw32', 'opengl32', 'dsound', 'ole32', 'd3d9', 'winmm', 'gdi32', 'iphlpapi', 'shlwapi', 'wsock32', 'ws2_32', 'kernel32', 'oleaut32', 'dinput8', 'dxguid'])

        env.Append(CPPFLAGS=['-DMINGW_ENABLED'])

        # resrc
        env.Append(BUILDERS={'RES': env.Builder(action=build_res_file, suffix='.o', src_suffix='.rc')})
