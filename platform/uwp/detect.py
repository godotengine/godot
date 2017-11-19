import methods
import os
import string
import sys


def is_active():
    return True


def get_name():
    return "UWP"


def can_build():
    if (os.name == "nt"):
        # building natively on windows!
        if (os.getenv("VSINSTALLDIR")):

            if (os.getenv("ANGLE_SRC_PATH") == None):
                return False

            return True
    return False


def get_opts():

    return [
    ]


def get_flags():

    return [
        ('tools', False),
        ('xaudio2', True),
    ]


def configure(env):

    if (env["bits"] != "default"):
        print("Error: bits argument is disabled for MSVC")
        print("""
            Bits argument is not supported for MSVC compilation. Architecture depends on the Native/Cross Compile Tools Prompt/Developer Console
            (or Visual Studio settings) that is being used to run SCons. As a consequence, bits argument is disabled. Run scons again without bits
            argument (example: scons p=uwp) and SCons will attempt to detect what MSVC compiler will be executed and inform you.
            """)
        sys.exit()

    ## Build type

    if (env["target"] == "release"):
        env.Append(CPPFLAGS=['/O2', '/GL'])
        env.Append(CPPFLAGS=['/MD'])
        env.Append(LINKFLAGS=['/SUBSYSTEM:WINDOWS', '/LTCG'])

    elif (env["target"] == "release_debug"):
        env.Append(CCFLAGS=['/O2', '/Zi', '/DDEBUG_ENABLED'])
        env.Append(CPPFLAGS=['/MD'])
        env.Append(LINKFLAGS=['/SUBSYSTEM:CONSOLE'])

    elif (env["target"] == "debug"):
        env.Append(CCFLAGS=['/Zi', '/DDEBUG_ENABLED', '/DDEBUG_MEMORY_ENABLED'])
        env.Append(CPPFLAGS=['/MDd'])
        env.Append(LINKFLAGS=['/SUBSYSTEM:CONSOLE'])
        env.Append(LINKFLAGS=['/DEBUG'])

    ## Compiler configuration

    env['ENV'] = os.environ
    vc_base_path = os.environ['VCTOOLSINSTALLDIR'] if "VCTOOLSINSTALLDIR" in os.environ else os.environ['VCINSTALLDIR']

    # ANGLE
    angle_root = os.getenv("ANGLE_SRC_PATH")
    env.Append(CPPPATH=[angle_root + '/include'])
    jobs = str(env.GetOption("num_jobs"))
    angle_build_cmd = "msbuild.exe " + angle_root + "/winrt/10/src/angle.sln /nologo /v:m /m:" + jobs + " /p:Configuration=Release /p:Platform="

    if os.path.isfile(str(os.getenv("ANGLE_SRC_PATH")) + "/winrt/10/src/angle.sln"):
        env["build_angle"] = True

    ## Architecture

    arch = ""
    if str(os.getenv('Platform')).lower() == "arm":

        print("Compiled program architecture will be an ARM executable. (forcing bits=32).")

        arch = "arm"
        env["bits"] = "32"
        env.Append(LINKFLAGS=['/MACHINE:ARM'])
        env.Append(LIBPATH=[vc_base_path + 'lib/store/arm'])

        angle_build_cmd += "ARM"

        env.Append(LIBPATH=[angle_root + '/winrt/10/src/Release_ARM/lib'])

    else:
        compiler_version_str = methods.detect_visual_c_compiler_version(env['ENV'])

        if(compiler_version_str == "amd64" or compiler_version_str == "x86_amd64"):
            env["bits"] = "64"
            print("Compiled program architecture will be a x64 executable (forcing bits=64).")
        elif (compiler_version_str == "x86" or compiler_version_str == "amd64_x86"):
            env["bits"] = "32"
            print("Compiled program architecture will be a x86 executable. (forcing bits=32).")
        else:
            print("Failed to detect MSVC compiler architecture version... Defaulting to 32bit executable settings (forcing bits=32). Compilation attempt will continue, but SCons can not detect for what architecture this build is compiled for. You should check your settings/compilation setup.")
            env["bits"] = "32"

        if (env["bits"] == "32"):
            arch = "x86"

            angle_build_cmd += "Win32"

            env.Append(LINKFLAGS=['/MACHINE:X86'])
            env.Append(LIBPATH=[vc_base_path + 'lib/store'])
            env.Append(LIBPATH=[angle_root + '/winrt/10/src/Release_Win32/lib'])

        else:
            arch = "x64"

            angle_build_cmd += "x64"

            env.Append(LINKFLAGS=['/MACHINE:X64'])
            env.Append(LIBPATH=[os.environ['VCINSTALLDIR'] + 'lib/store/amd64'])
            env.Append(LIBPATH=[angle_root + '/winrt/10/src/Release_x64/lib'])

    env["PROGSUFFIX"] = "." + arch + env["PROGSUFFIX"]
    env["OBJSUFFIX"] = "." + arch + env["OBJSUFFIX"]
    env["LIBSUFFIX"] = "." + arch + env["LIBSUFFIX"]

    ## Compile flags

    env.Append(CPPPATH=['#platform/uwp', '#drivers/windows'])
    env.Append(CCFLAGS=['/DUWP_ENABLED', '/DWINDOWS_ENABLED', '/DTYPED_METHOD_BIND'])
    env.Append(CCFLAGS=['/DGLES_ENABLED', '/DGL_GLEXT_PROTOTYPES', '/DEGL_EGLEXT_PROTOTYPES', '/DANGLE_ENABLED'])
    winver = "0x0602" # Windows 8 is the minimum target for UWP build
    env.Append(CCFLAGS=['/DWINVER=%s' % winver, '/D_WIN32_WINNT=%s' % winver])

    env.Append(CPPFLAGS=['/D', '__WRL_NO_DEFAULT_LIB__', '/D', 'WIN32', '/DPNG_ABORT=abort'])

    env.Append(CPPFLAGS=['/AI', vc_base_path + 'lib/store/references'])
    env.Append(CPPFLAGS=['/AI', vc_base_path + 'lib/x86/store/references'])

    env.Append(CCFLAGS='/FS /MP /GS /wd"4453" /wd"28204" /wd"4291" /Zc:wchar_t /Gm- /fp:precise /D "_UNICODE" /D "UNICODE" /D "WINAPI_FAMILY=WINAPI_FAMILY_APP" /errorReport:prompt /WX- /Zc:forScope /Gd /EHsc /nologo'.split())
    env.Append(CXXFLAGS='/ZW /FS'.split())
    env.Append(CCFLAGS=['/AI', vc_base_path + '\\vcpackages', '/AI', os.environ['WINDOWSSDKDIR'] + '\\References\\CommonConfiguration\\Neutral'])

    ## Link flags

    env.Append(LINKFLAGS=['/MANIFEST:NO', '/NXCOMPAT', '/DYNAMICBASE', '/WINMD', '/APPCONTAINER', '/ERRORREPORT:PROMPT', '/NOLOGO', '/TLBID:1', '/NODEFAULTLIB:"kernel32.lib"', '/NODEFAULTLIB:"ole32.lib"'])

    LIBS = [
        'WindowsApp',
        'mincore',
        'ws2_32',
        'libANGLE',
        'libEGL',
        'libGLESv2',
    ]
    env.Append(LINKFLAGS=[p + ".lib" for p in LIBS])

    # Incremental linking fix
    env['BUILDERS']['ProgramOriginal'] = env['BUILDERS']['Program']
    env['BUILDERS']['Program'] = methods.precious_program

    env.Append(BUILDERS={'ANGLE': env.Builder(action=angle_build_cmd)})
