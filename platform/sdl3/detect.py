import os
import sys
from typing import TYPE_CHECKING

from methods import print_error
from platform_methods import detect_arch, validate_arch

if TYPE_CHECKING:
    from SCons.Script.SConscript import SConsEnvironment


def get_name():
    return "SDL3"


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        EnumVariable(
            "sdl3_host",
            "How the in-tree SDL3 is built",
            "windows" if os.name == "nt" else "linux",
            ["windows", "linux", "external"],
            ignorecase=2,
        ),
        BoolVariable("use_llvm", "Use the LLVM compiler", False),
        BoolVariable("use_mingw", "Use MinGW-w64 even if MSVC is installed.", False),
        ("mingw_prefix", "MinGW-w64 install prefix (empty: tools are on PATH)", os.getenv("MINGW_PREFIX", "")),
        BoolVariable("use_static_cpp", "Link libgcc and libstdc++ statically for better portability", True),
    ]


def get_flags():
    return {
        "arch": detect_arch(),
        # 2dog only ever builds libgodot for this platform.
        "supported": ["library", "mono", "template"],
        # Vulkan always goes through volk. Console targets feed volk the entry
        # points SDL hands back instead of a loader.
        "use_volk": True,
        "d3d12": False,
        "metal": False,
        "angle": False,
    }


def get_tools(env: "SConsEnvironment"):
    # MSVC on a Windows host, MinGW-w64 otherwise (including cross-compiling from
    # Linux). Console builds supply their own toolchain through env["TOOLS"].
    if env.get("sdl3_host") in ("external", "linux"):
        return ["default"]
    if os.name != "nt" or env.get("use_mingw"):
        return ["mingw"]
    from SCons.Tool.MSCommon import msvc_exists

    if not msvc_exists():
        return ["mingw"]
    return ["msvc", "mslink", "mslib", "default"]


def _mingw_tool(env, tool):
    prefix = env["mingw_prefix"]
    bin_dir = (os.path.normpath(os.path.join(prefix, "bin")) + os.sep) if prefix else ""
    triple = {"x86_64": "x86_64-w64-mingw32-", "arm64": "aarch64-w64-mingw32-"}[env["arch"]]
    return str(env.Detect([bin_dir + triple + tool, bin_dir + tool]) or bin_dir + triple + tool)


def can_build():
    # Opt-in only: an ordinary build of the fork must not pick this up as a
    # candidate platform. Matches `scons platform=sdl3 ...` and platform=sdl3 in
    # SCONSFLAGS.
    if "platform" in sys.argv:
        return sys.argv[sys.argv.index("platform") + 1] == "sdl3"
    if "platform=sdl3" in sys.argv:
        return True
    return "platform=sdl3" in os.getenv("SCONSFLAGS", "")


def configure(env: "SConsEnvironment"):
    supported_arches = ["x86_64", "arm64"]
    validate_arch(env["arch"], get_name(), supported_arches)

    # The in-tree SDL3 is the full library. The joypad driver must not also build
    # the reduced copy in thirdparty/sdl: both define the same SDL symbols.
    env["builtin_sdl"] = False
    env["sdl"] = True
    env["sdl3"] = True

    if not os.path.isdir("thirdparty/sdl3-full/include"):
        print_error("thirdparty/sdl3-full is missing. Run: git submodule update --init --recursive")
        sys.exit(255)

    # <windows.h> pulls winsock.h, which then clashes with winsock2.h in every
    # translation unit (main/performance.cpp, SDL, core/io). Godot's Windows
    # platform gets this from its own build setup; here it must be global.
    env.AppendUnique(CPPDEFINES=["WIN32_LEAN_AND_MEAN", "_CRT_SECURE_NO_WARNINGS"])

    env.Prepend(CPPPATH=["#thirdparty/sdl3-full/include", "#thirdparty/sdl3-full/include/build_config", "#platform/sdl3"])
    # SDL_MAIN_HANDLED: the engine owns main(); SDL_main.h would redefine it.
    env.Append(CPPDEFINES=["SDL3_ENABLED", "SDL_ENABLED", "SDL_MAIN_HANDLED"])

    if env["sdl3_host"] == "windows":
        # One flag for modules that branch on "is this a Windows build".
        env["windows_like"] = True
        env.msvc = "mingw" not in env["TOOLS"]
        if env.msvc:
            env["TARGET_ARCH"] = {"x86_32": "x86", "arm32": "arm"}.get(env["arch"], env["arch"])
            env.AppendUnique(CCFLAGS=["/MD", "/fp:strict", "/Gd", "/GR", "/nologo", "/utf-8", "/bigobj", "/EHsc"])
            env.AppendUnique(LINKFLAGS=["/INCREMENTAL:NO"])
            # Nothing calls this export, so without SCU its archive member is never pulled in
            # and the .drectve /EXPORT is dropped.
            env.AppendUnique(LINKFLAGS=["/INCLUDE:libgodot_set_error_callback"])
            # SDL's Win32 backend calls into the Win32 API directly. These go in
            # LINKFLAGS rather than LIBS because LIBS names get the engine's
            # arch/target suffix appended and then fail to resolve.
            env.AppendUnique(
                LINKFLAGS=[lib + env["LIBSUFFIX"] for lib in ["user32", "gdi32", "ole32", "oleaut32", "shell32", "advapi32", "bcrypt", "winmm", "imm32", "setupapi", "cfgmgr32", "version", "uuid", "iphlpapi", "ws2_32", "ntdll"]]
            )
        else:
            env["CC"] = _mingw_tool(env, "clang" if env["use_llvm"] else "gcc")
            env["CXX"] = _mingw_tool(env, "clang++" if env["use_llvm"] else "g++")
            env["AS"] = env["CC"]
            env["AR"] = _mingw_tool(env, "ar")
            env["RANLIB"] = _mingw_tool(env, "ranlib")
            env["RC"] = _mingw_tool(env, "windres")
            env.Append(ASFLAGS=["-c"])
            # Big objects: SCU and SDL translation units overflow the default COFF section count.
            env.AppendUnique(CCFLAGS=["-Wa,-mbig-obj", "-ffp-contract=off"])
            env.Append(LINKFLAGS=["-Wl,--subsystem,console"])
            # SCons' mingw tool leaves the host's .so naming when cross-compiling.
            env["SHLIBPREFIX"] = ""
            env["SHLIBSUFFIX"] = ".dll"
            # core/SCsub only adds zstd's assembly Huffman loop for the stock platform names.
            env.AppendUnique(CPPDEFINES=["ZSTD_DISABLE_ASM"])
            if env["use_static_cpp"]:
                env.Append(LINKFLAGS=["-static"])
            # Same reason as the MSVC /INCLUDE below; x86_64 and arm64 PE have no symbol prefix.
            env.Append(LINKFLAGS=["-Wl,-u,libgodot_set_error_callback"])
            env.Append(
                LIBS=["user32", "gdi32", "ole32", "oleaut32", "shell32", "advapi32", "bcrypt", "winmm", "imm32", "setupapi", "cfgmgr32", "version", "uuid", "iphlpapi", "ws2_32", "ntdll", "dxguid"]
            )
        # Same Windows baseline as platform/windows: without WINVER and the
        # lean/no-minmax defines, <windows.h> and <winsock2.h> clash.
        env.AppendUnique(
            CPPDEFINES=[
                "WINDOWS_ENABLED",
                "TYPED_METHOD_BIND",
                "WIN32",
                "_WIN32",
                "NOMINMAX",
                "SDL_PLATFORM_WIN32",
                ("WINVER", "0x0A00"),
                ("_WIN32_WINNT", "0x0A00"),
            ]
        )
        if env["arch"] == "x86_64":
            env.AppendUnique(CPPDEFINES=["_WIN64"])
    elif env["sdl3_host"] == "linux":
        env.msvc = False
        env.AppendUnique(CPPDEFINES=["UNIX_ENABLED", "LINUXBSD_ENABLED"])
        env.Append(CCFLAGS=["-pthread", "-fPIC"])
        # Every object is built -fPIC already; without this SCons rejects the SDL objects, which reach the
        # link as pre-built static Objects instead of source files like platform/linuxbsd passes.
        env["STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME"] = 1
        # core/SCsub only adds zstd's assembly Huffman loop for the stock platform names.
        env.AppendUnique(CPPDEFINES=["ZSTD_DISABLE_ASM"])
        env.Append(LINKFLAGS=["-pthread", "-Wl,-u,libgodot_set_error_callback"])
        # Same reason as platform/linuxbsd/detect.py: keep unique symbols out of the .so so
        # dlclose() really unloads it between engine lifetimes.
        env.Append(CCFLAGS=["-fno-gnu-unique"])
        if env["use_static_cpp"]:
            env.Append(LINKFLAGS=["-static-libgcc", "-static-libstdc++"])
        env.Append(LIBS=["dl", "m", "rt"])
    else:
        env.AppendUnique(CPPDEFINES=["UNIX_ENABLED"])

    env.AppendUnique(CPPDEFINES=["_FILE_OFFSET_BITS=64"])

    # Rendering backends this platform builds. D3D12 and Metal stay out of scope.
    if env["vulkan"]:
        env.AppendUnique(CPPDEFINES=["VULKAN_ENABLED", "RD_ENABLED"])
    if env["opengl3"]:
        env.AppendUnique(CPPDEFINES=["GLES3_ENABLED"])
