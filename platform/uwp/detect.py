import methods
import os
import sys
from platform_methods import detect_arch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SCons import Environment


def is_active():
    return True


def get_name():
    return "UWP"


def can_build():
    if os.name == "nt":
        # building natively on windows!
        if os.getenv("VSINSTALLDIR"):

            if os.getenv("ANGLE_SRC_PATH") is None:
                return False

            return True
    return False


def get_opts():
    return [
        ("msvc_version", "MSVC version to use (ignored if the VCINSTALLDIR environment variable is set)", None),
    ]


def get_flags():
    return [
        ("arch", detect_arch()),
        ("tools", False),
        ("xaudio2", True),
        ("builtin_pcre2_with_jit", False),
    ]


def configure(env: "Environment"):
    # Validate arch.
    supported_arches = ["x86_32", "x86_64", "arm32"]
    if env["arch"] not in supported_arches:
        print(
            'Unsupported CPU architecture "%s" for UWP. Supported architectures are: %s.'
            % (env["arch"], ", ".join(supported_arches))
        )
        sys.exit()

    env.msvc = True

    ## Build type

    if env["target"] == "release":
        env.Append(CCFLAGS=["/MD"])
        env.Append(LINKFLAGS=["/SUBSYSTEM:WINDOWS"])
        if env["optimize"] != "none":
            env.Append(CCFLAGS=["/O2", "/GL"])
            env.Append(LINKFLAGS=["/LTCG"])

    elif env["target"] == "release_debug":
        env.Append(CCFLAGS=["/MD"])
        env.Append(LINKFLAGS=["/SUBSYSTEM:CONSOLE"])
        env.AppendUnique(CPPDEFINES=["WINDOWS_SUBSYSTEM_CONSOLE"])
        if env["optimize"] != "none":
            env.Append(CCFLAGS=["/O2", "/Zi"])

    elif env["target"] == "debug":
        env.Append(CCFLAGS=["/Zi"])
        env.Append(CCFLAGS=["/MDd"])
        env.Append(LINKFLAGS=["/SUBSYSTEM:CONSOLE"])
        env.AppendUnique(CPPDEFINES=["WINDOWS_SUBSYSTEM_CONSOLE"])
        env.Append(LINKFLAGS=["/DEBUG"])

    ## Compiler configuration

    env["ENV"] = os.environ
    vc_base_path = os.environ["VCTOOLSINSTALLDIR"] if "VCTOOLSINSTALLDIR" in os.environ else os.environ["VCINSTALLDIR"]

    # Force to use Unicode encoding
    env.AppendUnique(CCFLAGS=["/utf-8"])

    # ANGLE
    angle_root = os.environ["ANGLE_SRC_PATH"]
    env.Prepend(CPPPATH=[angle_root + "/include"])
    jobs = str(env.GetOption("num_jobs"))
    angle_build_cmd = (
        "msbuild.exe "
        + angle_root
        + "/winrt/10/src/angle.sln /nologo /v:m /m:"
        + jobs
        + " /p:Configuration=Release /p:Platform="
    )

    if os.path.isfile(f"{angle_root}/winrt/10/src/angle.sln"):
        env["build_angle"] = True

    ## Architecture

    arch = ""
    if str(os.getenv("Platform")).lower() == "arm":
        print("Compiled program architecture will be an ARM executable (forcing arch=arm32).")

        arch = "arm"
        env["arch"] = "arm32"
        env.Append(LINKFLAGS=["/MACHINE:ARM"])
        env.Append(LIBPATH=[vc_base_path + "lib/store/arm"])

        angle_build_cmd += "ARM"

        env.Append(LIBPATH=[angle_root + "/winrt/10/src/Release_ARM/lib"])

    else:
        compiler_version_str = methods.detect_visual_c_compiler_version(env["ENV"])

        if compiler_version_str == "amd64" or compiler_version_str == "x86_amd64":
            env["arch"] = "x86_64"
            print("Compiled program architecture will be a x64 executable (forcing arch=x86_64).")
        elif compiler_version_str == "x86" or compiler_version_str == "amd64_x86":
            env["arch"] = "x86_32"
            print("Compiled program architecture will be a x86 executable (forcing arch=x86_32).")
        else:
            print(
                "Failed to detect MSVC compiler architecture version... Defaulting to x86 32-bit executable settings"
                " (forcing arch=x86_32). Compilation attempt will continue, but SCons can not detect for what architecture"
                " this build is compiled for. You should check your settings/compilation setup."
            )
            env["arch"] = "x86_32"

        if env["arch"] == "x86_32":
            arch = "x86"

            angle_build_cmd += "Win32"

            env.Append(LINKFLAGS=["/MACHINE:X86"])
            env.Append(LIBPATH=[vc_base_path + "lib/store"])
            env.Append(LIBPATH=[angle_root + "/winrt/10/src/Release_Win32/lib"])

        else:
            arch = "x64"

            angle_build_cmd += "x64"

            env.Append(LINKFLAGS=["/MACHINE:X64"])
            env.Append(LIBPATH=[os.environ["VCINSTALLDIR"] + "lib/store/amd64"])
            env.Append(LIBPATH=[angle_root + "/winrt/10/src/Release_x64/lib"])

    env["PROGSUFFIX"] = "." + arch + env["PROGSUFFIX"]
    env["OBJSUFFIX"] = "." + arch + env["OBJSUFFIX"]
    env["LIBSUFFIX"] = "." + arch + env["LIBSUFFIX"]

    ## Compile flags

    env.Prepend(CPPPATH=["#platform/uwp", "#drivers/windows"])
    env.Append(CPPDEFINES=["UWP_ENABLED", "WINDOWS_ENABLED", "TYPED_METHOD_BIND"])
    env.Append(CPPDEFINES=["GLES_ENABLED", "GL_GLEXT_PROTOTYPES", "EGL_EGLEXT_PROTOTYPES", "ANGLE_ENABLED"])
    winver = "0x0602"  # Windows 8 is the minimum target for UWP build
    env.Append(CPPDEFINES=[("WINVER", winver), ("_WIN32_WINNT", winver), "WIN32"])

    env.Append(CPPDEFINES=["__WRL_NO_DEFAULT_LIB__", ("PNG_ABORT", "abort")])

    env.Append(CPPFLAGS=["/AI", vc_base_path + "lib/store/references"])
    env.Append(CPPFLAGS=["/AI", vc_base_path + "lib/x86/store/references"])

    env.Append(
        CCFLAGS=(
            '/FS /MP /GS /wd"4453" /wd"28204" /wd"4291" /Zc:wchar_t /Gm- /fp:precise /errorReport:prompt /WX-'
            " /Zc:forScope /Gd /EHsc /nologo".split()
        )
    )
    env.Append(CPPDEFINES=["_UNICODE", "UNICODE", ("WINAPI_FAMILY", "WINAPI_FAMILY_APP")])
    env.Append(CXXFLAGS=["/ZW"])
    env.Append(
        CCFLAGS=[
            "/AI",
            vc_base_path + "\\vcpackages",
            "/AI",
            os.environ["WINDOWSSDKDIR"] + "\\References\\CommonConfiguration\\Neutral",
        ]
    )

    ## Link flags

    env.Append(
        LINKFLAGS=[
            "/MANIFEST:NO",
            "/NXCOMPAT",
            "/DYNAMICBASE",
            "/WINMD",
            "/APPCONTAINER",
            "/ERRORREPORT:PROMPT",
            "/NOLOGO",
            "/TLBID:1",
            '/NODEFAULTLIB:"kernel32.lib"',
            '/NODEFAULTLIB:"ole32.lib"',
        ]
    )

    LIBS = [
        "WindowsApp",
        "mincore",
        "ws2_32",
        "libANGLE",
        "libEGL",
        "libGLESv2",
        "bcrypt",
    ]
    env.Append(LINKFLAGS=[p + ".lib" for p in LIBS])

    # Incremental linking fix
    env["BUILDERS"]["ProgramOriginal"] = env["BUILDERS"]["Program"]
    env["BUILDERS"]["Program"] = methods.precious_program

    env.Append(BUILDERS={"ANGLE": env.Builder(action=angle_build_cmd)})
