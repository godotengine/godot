import os
import sys

from emscripten_helpers import (
    run_closure_compiler,
    create_engine_file,
    add_js_libraries,
    add_js_pre,
    add_js_externs,
    create_template_zip,
)
from methods import get_compiler_version
from SCons.Util import WhereIs


def is_active():
    return True


def get_name():
    return "JavaScript"


def can_build():
    return WhereIs("emcc") is not None


def get_opts():
    from SCons.Variables import BoolVariable

    return [
        ("initial_memory", "Initial WASM memory (in MiB)", 32),
        # Matches default values from before Emscripten 3.1.27. New defaults are too low for Godot.
        ("stack_size", "WASM stack size (in KiB)", 5120),
        ("default_pthread_stack_size", "WASM pthread default stack size (in KiB)", 2048),
        BoolVariable("use_assertions", "Use Emscripten runtime assertions", False),
        BoolVariable("use_ubsan", "Use Emscripten undefined behavior sanitizer (UBSAN)", False),
        BoolVariable("use_asan", "Use Emscripten address sanitizer (ASAN)", False),
        BoolVariable("use_lsan", "Use Emscripten leak sanitizer (LSAN)", False),
        BoolVariable("use_safe_heap", "Use Emscripten SAFE_HEAP sanitizer", False),
        # eval() can be a security concern, so it can be disabled.
        BoolVariable("javascript_eval", "Enable JavaScript eval interface", True),
        BoolVariable("threads_enabled", "Enable WebAssembly Threads support (limited browser support)", False),
        BoolVariable("gdnative_enabled", "Enable WebAssembly GDNative support (produces bigger binaries)", False),
        BoolVariable("use_closure_compiler", "Use closure compiler to minimize JavaScript code", False),
    ]


def get_flags():
    return [
        ("tools", False),
        ("builtin_pcre2_with_jit", False),
    ]


def configure(env):
    try:
        env["initial_memory"] = int(env["initial_memory"])
    except Exception:
        print("Initial memory must be a valid integer")
        sys.exit(255)

    ## Build type
    if env["target"].startswith("release"):
        # Use -Os to prioritize optimizing for reduced file size. This is
        # particularly valuable for the web platform because it directly
        # decreases download time.
        # -Os reduces file size by around 5 MiB over -O3. -Oz only saves about
        # 100 KiB over -Os, which does not justify the negative impact on
        # run-time performance.
        if env["optimize"] != "none":
            env.Append(CCFLAGS=["-Os"])
            env.Append(LINKFLAGS=["-Os"])

        if env["target"] == "release_debug":
            # Retain function names for backtraces at the cost of file size.
            env.Append(LINKFLAGS=["--profiling-funcs"])
    else:  # "debug"
        env.Append(CCFLAGS=["-O1", "-g"])
        env.Append(LINKFLAGS=["-O1", "-g"])
        env["use_assertions"] = True

    if env["use_assertions"]:
        env.Append(LINKFLAGS=["-s", "ASSERTIONS=1"])

    if env["tools"]:
        if not env["threads_enabled"]:
            print('Note: Forcing "threads_enabled=yes" as it is required for the web editor.')
            env["threads_enabled"] = "yes"
        if env["initial_memory"] < 64:
            print('Note: Forcing "initial_memory=64" as it is required for the web editor.')
            env["initial_memory"] = 64
    else:
        # Disable rtti on non-tools (template) builds.
        # This helps keep the file size down.
        env.Append(CCFLAGS=["-fno-rtti"])
        # Don't use dynamic_cast, necessary with no-rtti.
        env.Append(CPPDEFINES=["NO_SAFE_CAST"])

    env.Append(LINKFLAGS=["-s", "INITIAL_MEMORY=%sMB" % env["initial_memory"]])

    ## Copy env variables.
    env["ENV"] = os.environ

    # LTO

    if env["lto"] == "auto":  # Full LTO for production.
        env["lto"] = "full"

    if env["lto"] != "none":
        if env["lto"] == "thin":
            env.Append(CCFLAGS=["-flto=thin"])
            env.Append(LINKFLAGS=["-flto=thin"])
        else:
            env.Append(CCFLAGS=["-flto"])
            env.Append(LINKFLAGS=["-flto"])

    # Sanitizers
    if env["use_ubsan"]:
        env.Append(CCFLAGS=["-fsanitize=undefined"])
        env.Append(LINKFLAGS=["-fsanitize=undefined"])
    if env["use_asan"]:
        env.Append(CCFLAGS=["-fsanitize=address"])
        env.Append(LINKFLAGS=["-fsanitize=address"])
    if env["use_lsan"]:
        env.Append(CCFLAGS=["-fsanitize=leak"])
        env.Append(LINKFLAGS=["-fsanitize=leak"])
    if env["use_safe_heap"]:
        env.Append(LINKFLAGS=["-s", "SAFE_HEAP=1"])

    # Closure compiler
    if env["use_closure_compiler"]:
        # For emscripten support code.
        env.Append(LINKFLAGS=["--closure", "1"])
        # Register builder for our Engine files
        jscc = env.Builder(generator=run_closure_compiler, suffix=".cc.js", src_suffix=".js")
        env.Append(BUILDERS={"BuildJS": jscc})

    # Add helper method for adding libraries, externs, pre-js.
    env["JS_LIBS"] = []
    env["JS_PRE"] = []
    env["JS_EXTERNS"] = []
    env.AddMethod(add_js_libraries, "AddJSLibraries")
    env.AddMethod(add_js_pre, "AddJSPre")
    env.AddMethod(add_js_externs, "AddJSExterns")

    # Add method that joins/compiles our Engine files.
    env.AddMethod(create_engine_file, "CreateEngineFile")

    # Add method for creating the final zip file
    env.AddMethod(create_template_zip, "CreateTemplateZip")

    # Closure compiler extern and support for ecmascript specs (const, let, etc).
    env["ENV"]["EMCC_CLOSURE_ARGS"] = "--language_in ECMASCRIPT_2021"

    env["CC"] = "emcc"
    env["CXX"] = "em++"

    env["AR"] = "emar"
    env["RANLIB"] = "emranlib"

    # Use TempFileMunge since some AR invocations are too long for cmd.exe.
    # Use POSIX-style paths, required with TempFileMunge.
    env["ARCOM_POSIX"] = env["ARCOM"].replace("$TARGET", "$TARGET.posix").replace("$SOURCES", "$SOURCES.posix")
    env["ARCOM"] = "${TEMPFILE(ARCOM_POSIX)}"

    # All intermediate files are just object files.
    env["OBJPREFIX"] = ""
    env["OBJSUFFIX"] = ".o"
    env["PROGPREFIX"] = ""
    # Program() output consists of multiple files, so specify suffixes manually at builder.
    env["PROGSUFFIX"] = ""
    env["LIBPREFIX"] = "lib"
    env["LIBSUFFIX"] = ".a"
    env["LIBPREFIXES"] = ["$LIBPREFIX"]
    env["LIBSUFFIXES"] = ["$LIBSUFFIX"]

    # Get version info for checks below.
    cc_semver = tuple(get_compiler_version(env) or (3, 1, 39))

    env.Prepend(CPPPATH=["#platform/javascript"])
    env.Append(CPPDEFINES=["JAVASCRIPT_ENABLED", "UNIX_ENABLED"])

    if env["javascript_eval"]:
        env.Append(CPPDEFINES=["JAVASCRIPT_EVAL_ENABLED"])

    stack_size_opt = "STACK_SIZE" if cc_semver >= (3, 1, 25) else "TOTAL_STACK"
    env.Append(LINKFLAGS=["-s", "%s=%sKB" % (stack_size_opt, env["stack_size"])])

    # Thread support (via SharedArrayBuffer).
    if env["threads_enabled"]:
        stack_size_opt = "STACK_SIZE" if cc_semver >= (3, 1, 25) else "TOTAL_STACK"
        env.Append(LINKFLAGS=["-s", "%s=%sKB" % (stack_size_opt, env["stack_size"])])

        env.Append(CPPDEFINES=["PTHREAD_NO_RENAME"])
        env.Append(CCFLAGS=["-s", "USE_PTHREADS=1"])
        env.Append(LINKFLAGS=["-s", "USE_PTHREADS=1"])
        env.Append(LINKFLAGS=["-s", "DEFAULT_PTHREAD_STACK_SIZE=%sKB" % env["default_pthread_stack_size"]])
        env.Append(LINKFLAGS=["-s", "PTHREAD_POOL_SIZE=8"])
        env.Append(LINKFLAGS=["-s", "WASM_MEM_MAX=2048MB"])
        env.extra_suffix = ".threads" + env.extra_suffix
    else:
        env.Append(CPPDEFINES=["NO_THREADS"])

    if env["lto"] != "none":
        # Workaround https://github.com/emscripten-core/emscripten/issues/19781.
        if cc_semver >= (3, 1, 42) and cc_semver < (3, 1, 46):
            env.Append(LINKFLAGS=["-Wl,-u,scalbnf"])
        # Workaround https://github.com/emscripten-core/emscripten/issues/16836.
        if cc_semver >= (3, 1, 47):
            env.Append(LINKFLAGS=["-Wl,-u,_emscripten_run_callback_on_thread"])

    if env["gdnative_enabled"]:
        if cc_semver < (2, 0, 10):
            print("GDNative support requires emscripten >= 2.0.10, detected: %s.%s.%s" % cc_semver)
            sys.exit(255)
        if env["threads_enabled"] and cc_semver < (3, 1, 14):
            print("Threads and GDNative requires emscripten => 3.1.14, detected: %s.%s.%s" % cc_semver)
            sys.exit(255)
        env.Append(CCFLAGS=["-s", "RELOCATABLE=1"])
        env.Append(LINKFLAGS=["-s", "RELOCATABLE=1"])
        # Weak symbols are broken upstream: https://github.com/emscripten-core/emscripten/issues/12819
        env.Append(CPPDEFINES=["ZSTD_HAVE_WEAK_SYMBOLS=0"])
        env.extra_suffix = ".gdnative" + env.extra_suffix

    # WASM_BIGINT is needed since emscripten ≥ 3.1.41
    if cc_semver >= (3, 1, 41):
        env.Append(LINKFLAGS=["-s", "WASM_BIGINT"])

    # Reduce code size by generating less support code (e.g. skip NodeJS support).
    env.Append(LINKFLAGS=["-s", "ENVIRONMENT=web,worker"])

    # Wrap the JavaScript support code around a closure named Godot.
    env.Append(LINKFLAGS=["-s", "MODULARIZE=1", "-s", "EXPORT_NAME='Godot'"])

    # Allow increasing memory buffer size during runtime. This is efficient
    # when using WebAssembly (in comparison to asm.js) and works well for
    # us since we don't know requirements at compile-time.
    env.Append(LINKFLAGS=["-s", "ALLOW_MEMORY_GROWTH=1"])

    # This setting just makes WebGL 2 APIs available, it does NOT disable WebGL 1.
    env.Append(LINKFLAGS=["-s", "USE_WEBGL2=1"])
    # Breaking change since emscripten 3.1.51
    # https://github.com/emscripten-core/emscripten/blob/main/ChangeLog.md#3151---121323
    if cc_semver >= (3, 1, 51):
        # Enables the use of *glGetProcAddress()
        env.Append(LINKFLAGS=["-s", "GL_ENABLE_GET_PROC_ADDRESS=1"])

    # Do not call main immediately when the support code is ready.
    env.Append(LINKFLAGS=["-s", "INVOKE_RUN=0"])

    # Allow use to take control of swapping WebGL buffers.
    env.Append(LINKFLAGS=["-s", "OFFSCREEN_FRAMEBUFFER=1"])

    # callMain for manual start, cwrap for the mono version.
    env.Append(LINKFLAGS=["-s", "EXPORTED_RUNTIME_METHODS=['callMain','cwrap']"])

    # Add code that allow exiting runtime.
    env.Append(LINKFLAGS=["-s", "EXIT_RUNTIME=1"])
