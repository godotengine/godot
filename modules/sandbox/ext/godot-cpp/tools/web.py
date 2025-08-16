import common_compiler_flags
from SCons.Util import WhereIs


def exists(env):
    return WhereIs("emcc") is not None


def generate(env):
    if env["arch"] not in ("wasm32"):
        print("Only wasm32 supported on web. Exiting.")
        env.Exit(1)

    # Emscripten toolchain
    env["CC"] = "emcc"
    env["CXX"] = "em++"
    env["AR"] = "emar"
    env["RANLIB"] = "emranlib"

    # Use TempFileMunge since some AR invocations are too long for cmd.exe.
    # Use POSIX-style paths, required with TempFileMunge.
    env["ARCOM_POSIX"] = env["ARCOM"].replace("$TARGET", "$TARGET.posix").replace("$SOURCES", "$SOURCES.posix")
    env["ARCOM"] = "${TEMPFILE(ARCOM_POSIX)}"

    # All intermediate files are just object files.
    env["OBJSUFFIX"] = ".o"
    env["SHOBJSUFFIX"] = ".o"

    # Static libraries clang-style.
    env["LIBPREFIX"] = "lib"
    env["LIBSUFFIX"] = ".a"

    # Shared library as wasm.
    env["SHLIBSUFFIX"] = ".wasm"

    # Thread support (via SharedArrayBuffer).
    if env["threads"]:
        env.Append(CCFLAGS=["-sUSE_PTHREADS=1"])
        env.Append(LINKFLAGS=["-sUSE_PTHREADS=1"])

    # Build as side module (shared library).
    env.Append(CCFLAGS=["-sSIDE_MODULE=1"])
    env.Append(LINKFLAGS=["-sSIDE_MODULE=1"])

    # Enable WebAssembly BigInt <-> i64 conversion.
    # This must match the flag used to build Godot (true in official builds since 4.3)
    env.Append(LINKFLAGS=["-sWASM_BIGINT"])

    # Force wasm longjmp mode.
    env.Append(CCFLAGS=["-sSUPPORT_LONGJMP='wasm'"])
    env.Append(LINKFLAGS=["-sSUPPORT_LONGJMP='wasm'"])

    env.Append(CPPDEFINES=["WEB_ENABLED", "UNIX_ENABLED"])

    # Refer to https://github.com/godotengine/godot/blob/master/platform/web/detect.py
    if env["lto"] == "auto":
        env["lto"] = "full"

    common_compiler_flags.generate(env)
