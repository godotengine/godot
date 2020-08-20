import methods
import os

# To match other platforms
STACK_SIZE = 8388608


def is_active():
    return True


def get_name():
    return "Windows"


def can_build():
    if os.name == "nt":
        # Building natively on Windows
        # If VCINSTALLDIR is set in the OS environ, use traditional Godot logic to set up MSVC
        if os.getenv("VCINSTALLDIR"):  # MSVC, manual setup
            return True

        # Otherwise, let SCons find MSVC if installed, or else Mingw.
        # Since we're just returning True here, if there's no compiler
        # installed, we'll get errors when it tries to build with the
        # null compiler.
        return True

    if os.name == "posix":
        # Cross-compiling with MinGW-w64 (old MinGW32 is not supported)
        mingw32 = "i686-w64-mingw32-"
        mingw64 = "x86_64-w64-mingw32-"

        if os.getenv("MINGW32_PREFIX"):
            mingw32 = os.getenv("MINGW32_PREFIX")
        if os.getenv("MINGW64_PREFIX"):
            mingw64 = os.getenv("MINGW64_PREFIX")

        test = "gcc --version > /dev/null 2>&1"
        if os.system(mingw64 + test) == 0 or os.system(mingw32 + test) == 0:
            return True

    return False


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    mingw32 = ""
    mingw64 = ""
    if os.name == "posix":
        mingw32 = "i686-w64-mingw32-"
        mingw64 = "x86_64-w64-mingw32-"

    if os.getenv("MINGW32_PREFIX"):
        mingw32 = os.getenv("MINGW32_PREFIX")
    if os.getenv("MINGW64_PREFIX"):
        mingw64 = os.getenv("MINGW64_PREFIX")

    return [
        ("mingw_prefix_32", "MinGW prefix (Win32)", mingw32),
        ("mingw_prefix_64", "MinGW prefix (Win64)", mingw64),
        # Targeted Windows version: 7 (and later), minimum supported version
        # XP support dropped after EOL due to missing API for IPv6 and other issues
        # Vista support dropped after EOL due to GH-10243
        ("target_win_version", "Targeted Windows version, >= 0x0601 (Windows 7)", "0x0601"),
        EnumVariable("debug_symbols", "Add debugging symbols to release/release_debug builds", "yes", ("yes", "no")),
        BoolVariable("separate_debug_symbols", "Create a separate file containing debugging symbols", False),
        ("msvc_version", "MSVC version to use. Ignored if VCINSTALLDIR is set in shell env.", None),
        BoolVariable("use_mingw", "Use the Mingw compiler, even if MSVC is installed. Only used on Windows.", False),
        BoolVariable("use_llvm", "Use the LLVM compiler", False),
        BoolVariable("use_thinlto", "Use ThinLTO", False),
    ]


def get_flags():

    return []


def build_res_file(target, source, env):

    if env["bits"] == "32":
        cmdbase = env["mingw_prefix_32"]
    else:
        cmdbase = env["mingw_prefix_64"]
    cmdbase = cmdbase + "windres --include-dir . "
    import subprocess

    for x in range(len(source)):
        cmd = cmdbase + "-i " + str(source[x]) + " -o " + str(target[x])
        try:
            out = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE).communicate()
            if len(out[1]):
                return 1
        except:
            return 1
    return 0


def setup_msvc_manual(env):
    """Set up env to use MSVC manually, using VCINSTALLDIR"""
    if env["bits"] != "default":
        print(
            """
            Bits argument is not supported for MSVC compilation. Architecture depends on the Native/Cross Compile Tools Prompt/Developer Console
            (or Visual Studio settings) that is being used to run SCons. As a consequence, bits argument is disabled. Run scons again without bits
            argument (example: scons p=windows) and SCons will attempt to detect what MSVC compiler will be executed and inform you.
            """
        )
        raise SCons.Errors.UserError("Bits argument should not be used when using VCINSTALLDIR")

    # Force bits arg
    # (Actually msys2 mingw can support 64-bit, we could detect that)
    env["bits"] = "32"
    env["x86_libtheora_opt_vc"] = True

    # find compiler manually
    compiler_version_str = methods.detect_visual_c_compiler_version(env["ENV"])
    print("Found MSVC compiler: " + compiler_version_str)

    # If building for 64bit architecture, disable assembly optimisations for 32 bit builds (theora as of writing)... vc compiler for 64bit can not compile _asm
    if compiler_version_str == "amd64" or compiler_version_str == "x86_amd64":
        env["bits"] = "64"
        env["x86_libtheora_opt_vc"] = False
        print("Compiled program architecture will be a 64 bit executable (forcing bits=64).")
    elif compiler_version_str == "x86" or compiler_version_str == "amd64_x86":
        print("Compiled program architecture will be a 32 bit executable. (forcing bits=32).")
    else:
        print(
            "Failed to manually detect MSVC compiler architecture version... Defaulting to 32bit executable settings (forcing bits=32). Compilation attempt will continue, but SCons can not detect for what architecture this build is compiled for. You should check your settings/compilation setup, or avoid setting VCINSTALLDIR."
        )


def setup_msvc_auto(env):
    """Set up MSVC using SCons's auto-detection logic"""

    # If MSVC_VERSION is set by SCons, we know MSVC is installed.
    # But we may want a different version or target arch.

    # The env may have already been set up with default MSVC tools, so
    # reset a few things so we can set it up with the tools we want.
    # (Ideally we'd decide on the tool config before configuring any
    # environment, and just set the env up once, but this function runs
    # on an existing env so this is the simplest way.)
    env["MSVC_SETUP_RUN"] = False  # Need to set this to re-run the tool
    env["MSVS_VERSION"] = None
    env["MSVC_VERSION"] = None
    env["TARGET_ARCH"] = None
    if env["bits"] != "default":
        env["TARGET_ARCH"] = {"32": "x86", "64": "x86_64"}[env["bits"]]
    if env.has_key("msvc_version"):
        env["MSVC_VERSION"] = env["msvc_version"]
    env.Tool("msvc")
    env.Tool("mssdk")  # we want the MS SDK
    # Note: actual compiler version can be found in env['MSVC_VERSION'], e.g. "14.1" for VS2015
    # Get actual target arch into bits (it may be "default" at this point):
    if env["TARGET_ARCH"] in ("amd64", "x86_64"):
        env["bits"] = "64"
    else:
        env["bits"] = "32"
    print("Found MSVC version %s, arch %s, bits=%s" % (env["MSVC_VERSION"], env["TARGET_ARCH"], env["bits"]))
    if env["TARGET_ARCH"] in ("amd64", "x86_64"):
        env["x86_libtheora_opt_vc"] = False


def setup_mingw(env):
    """Set up env for use with mingw"""
    # Nothing to do here
    print("Using MinGW")
    pass


def configure_msvc(env, manual_msvc_config):
    """Configure env to work with MSVC"""

    # Build type

    if env["target"] == "release":
        if env["debug_symbols"] != "full":
            if env["optimize"] == "speed":  # optimize for speed (default)
                env.Append(CCFLAGS=["/O2"])
            else:  # optimize for size
                env.Append(CCFLAGS=["/O1"])
        env.Append(LINKFLAGS=["/SUBSYSTEM:WINDOWS"])
        env.Append(LINKFLAGS=["/ENTRY:mainCRTStartup"])
        env.Append(LINKFLAGS=["/OPT:REF"])

    elif env["target"] == "release_debug":
        if env["debug_symbols"] != "full":
            if env["optimize"] == "speed":  # optimize for speed (default)
                env.Append(CCFLAGS=["/O2"])
            else:  # optimize for size
                env.Append(CCFLAGS=["/O1"])
        env.AppendUnique(CPPDEFINES=["DEBUG_ENABLED"])
        env.Append(LINKFLAGS=["/SUBSYSTEM:CONSOLE"])
        env.Append(LINKFLAGS=["/OPT:REF"])

    elif env["target"] == "debug":
        env.AppendUnique(CCFLAGS=["/Z7", "/Od", "/EHsc"])
        env.AppendUnique(CPPDEFINES=["DEBUG_ENABLED"])
        env.Append(LINKFLAGS=["/SUBSYSTEM:CONSOLE"])
        env.Append(LINKFLAGS=["/DEBUG"])

    if env["debug_symbols"] == "yes":
        env.AppendUnique(CCFLAGS=["/Z7"])
        env.AppendUnique(LINKFLAGS=["/DEBUG"])

    ## Compile/link flags

    env.AppendUnique(CCFLAGS=["/MT", "/Gd", "/GR", "/nologo"])
    # Force to use Unicode encoding
    env.AppendUnique(CCFLAGS=["/utf-8"])
    env.AppendUnique(CXXFLAGS=["/TP"])  # assume all sources are C++
    if manual_msvc_config:  # should be automatic if SCons found it
        if os.getenv("WindowsSdkDir") is not None:
            env.Prepend(CPPPATH=[os.getenv("WindowsSdkDir") + "/Include"])
        else:
            print("Missing environment variable: WindowsSdkDir")

    env.AppendUnique(
        CPPDEFINES=[
            "WINDOWS_ENABLED",
            "OPENGL_ENABLED",
            "WASAPI_ENABLED",
            "WINMIDI_ENABLED",
            "TYPED_METHOD_BIND",
            "WIN32",
            "MSVC",
            "WINVER=%s" % env["target_win_version"],
            "_WIN32_WINNT=%s" % env["target_win_version"],
        ]
    )
    env.AppendUnique(CPPDEFINES=["NOMINMAX"])  # disable bogus min/max WinDef.h macros
    if env["bits"] == "64":
        env.AppendUnique(CPPDEFINES=["_WIN64"])

    ## Libs

    LIBS = [
        "winmm",
        "opengl32",
        "dsound",
        "kernel32",
        "ole32",
        "oleaut32",
        "user32",
        "gdi32",
        "IPHLPAPI",
        "Shlwapi",
        "wsock32",
        "Ws2_32",
        "shell32",
        "advapi32",
        "dinput8",
        "dxguid",
        "imm32",
        "bcrypt",
        "Avrt",
        "dwmapi",
    ]
    env.Append(LINKFLAGS=[p + env["LIBSUFFIX"] for p in LIBS])

    if manual_msvc_config:
        if os.getenv("WindowsSdkDir") is not None:
            env.Append(LIBPATH=[os.getenv("WindowsSdkDir") + "/Lib"])
        else:
            print("Missing environment variable: WindowsSdkDir")

    ## LTO

    if env["use_lto"]:
        env.AppendUnique(CCFLAGS=["/GL"])
        env.AppendUnique(ARFLAGS=["/LTCG"])
        if env["progress"]:
            env.AppendUnique(LINKFLAGS=["/LTCG:STATUS"])
        else:
            env.AppendUnique(LINKFLAGS=["/LTCG"])

    if manual_msvc_config:
        env.Prepend(CPPPATH=[p for p in os.getenv("INCLUDE").split(";")])
        env.Append(LIBPATH=[p for p in os.getenv("LIB").split(";")])

    # Incremental linking fix
    env["BUILDERS"]["ProgramOriginal"] = env["BUILDERS"]["Program"]
    env["BUILDERS"]["Program"] = methods.precious_program

    env.AppendUnique(LINKFLAGS=["/STACK:" + str(STACK_SIZE)])


def configure_mingw(env):
    # Workaround for MinGW. See:
    # http://www.scons.org/wiki/LongCmdLinesOnWin32
    env.use_windows_spawn_fix()

    ## Build type

    if env["target"] == "release":
        env.Append(CCFLAGS=["-msse2"])

        if env["optimize"] == "speed":  # optimize for speed (default)
            if env["bits"] == "64":
                env.Append(CCFLAGS=["-O3"])
            else:
                env.Append(CCFLAGS=["-O2"])
        else:  # optimize for size
            env.Prepend(CCFLAGS=["-Os"])

        env.Append(LINKFLAGS=["-Wl,--subsystem,windows"])

        if env["debug_symbols"] == "yes":
            env.Prepend(CCFLAGS=["-g2"])

    elif env["target"] == "release_debug":
        env.Append(CCFLAGS=["-O2"])
        env.Append(CPPDEFINES=["DEBUG_ENABLED"])
        if env["debug_symbols"] == "yes":
            env.Prepend(CCFLAGS=["-g2"])
        if env["optimize"] == "speed":  # optimize for speed (default)
            env.Append(CCFLAGS=["-O2"])
        else:  # optimize for size
            env.Prepend(CCFLAGS=["-Os"])

    elif env["target"] == "debug":
        env.Append(CCFLAGS=["-g3"])
        env.Append(CPPDEFINES=["DEBUG_ENABLED"])

    ## Compiler configuration

    if os.name == "nt":
        # Force splitting libmodules.a in multiple chunks to work around
        # issues reaching the linker command line size limit, which also
        # seem to induce huge slowdown for 'ar' (GH-30892).
        env["split_libmodules"] = True
    else:
        env["PROGSUFFIX"] = env["PROGSUFFIX"] + ".exe"  # for linux cross-compilation

    if env["bits"] == "default":
        if os.name == "nt":
            env["bits"] = "64" if "PROGRAMFILES(X86)" in os.environ else "32"
        else:  # default to 64-bit on Linux
            env["bits"] = "64"

    mingw_prefix = ""

    if env["bits"] == "32":
        env.Append(LINKFLAGS=["-static"])
        env.Append(LINKFLAGS=["-static-libgcc"])
        env.Append(LINKFLAGS=["-static-libstdc++"])
        mingw_prefix = env["mingw_prefix_32"]
    else:
        env.Append(LINKFLAGS=["-static"])
        mingw_prefix = env["mingw_prefix_64"]

    if env["use_llvm"]:
        env["CC"] = mingw_prefix + "clang"
        env["AS"] = mingw_prefix + "as"
        env["CXX"] = mingw_prefix + "clang++"
        env["AR"] = mingw_prefix + "ar"
        env["RANLIB"] = mingw_prefix + "ranlib"
        env["LINK"] = mingw_prefix + "clang++"
    else:
        env["CC"] = mingw_prefix + "gcc"
        env["AS"] = mingw_prefix + "as"
        env["CXX"] = mingw_prefix + "g++"
        env["AR"] = mingw_prefix + "gcc-ar"
        env["RANLIB"] = mingw_prefix + "gcc-ranlib"
        env["LINK"] = mingw_prefix + "g++"
    env["x86_libtheora_opt_gcc"] = True

    if env["use_lto"]:
        if not env["use_llvm"] and env.GetOption("num_jobs") > 1:
            env.Append(CCFLAGS=["-flto"])
            env.Append(LINKFLAGS=["-flto=" + str(env.GetOption("num_jobs"))])
        else:
            if env["use_thinlto"]:
                env.Append(CCFLAGS=["-flto=thin"])
                env.Append(LINKFLAGS=["-flto=thin"])
            else:
                env.Append(CCFLAGS=["-flto"])
                env.Append(LINKFLAGS=["-flto"])

    env.Append(LINKFLAGS=["-Wl,--stack," + str(STACK_SIZE)])

    ## Compile flags

    env.Append(CCFLAGS=["-mwindows"])
    env.Append(CPPDEFINES=["WINDOWS_ENABLED", "OPENGL_ENABLED", "WASAPI_ENABLED", "WINMIDI_ENABLED"])
    env.Append(CPPDEFINES=[("WINVER", env["target_win_version"]), ("_WIN32_WINNT", env["target_win_version"])])
    env.Append(
        LIBS=[
            "mingw32",
            "opengl32",
            "dsound",
            "ole32",
            "d3d9",
            "winmm",
            "gdi32",
            "iphlpapi",
            "shlwapi",
            "wsock32",
            "ws2_32",
            "kernel32",
            "oleaut32",
            "dinput8",
            "dxguid",
            "ksuser",
            "imm32",
            "bcrypt",
            "avrt",
            "uuid",
            "dwmapi",
        ]
    )

    env.Append(CPPDEFINES=["MINGW_ENABLED", ("MINGW_HAS_SECURE_API", 1)])

    # resrc
    env.Append(BUILDERS={"RES": env.Builder(action=build_res_file, suffix=".o", src_suffix=".rc")})


def configure(env):
    # At this point the env has been set up with basic tools/compilers.
    env.Prepend(CPPPATH=["#platform/windows"])

    print("Configuring for Windows: target=%s, bits=%s" % (env["target"], env["bits"]))

    if os.name == "nt":
        env["ENV"] = os.environ  # this makes build less repeatable, but simplifies some things
        env["ENV"]["TMP"] = os.environ["TMP"]

    # First figure out which compiler, version, and target arch we're using
    if os.getenv("VCINSTALLDIR") and not env["use_mingw"]:
        # Manual setup of MSVC
        setup_msvc_manual(env)
        env.msvc = True
        manual_msvc_config = True
    elif env.get("MSVC_VERSION", "") and not env["use_mingw"]:
        setup_msvc_auto(env)
        env.msvc = True
        manual_msvc_config = False
    else:
        setup_mingw(env)
        env.msvc = False

    # Now set compiler/linker flags
    if env.msvc:
        configure_msvc(env, manual_msvc_config)

    else:  # MinGW
        configure_mingw(env)
