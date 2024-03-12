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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SCons import Environment


def get_name():
    return "Web"


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
        BoolVariable(
            "dlink_enabled", "Enable WebAssembly dynamic linking (GDExtension support). Produces bigger binaries", False
        ),
        BoolVariable("use_closure_compiler", "Use closure compiler to minimize JavaScript code", False),
        BoolVariable(
            "proxy_to_pthread",
            "Use Emscripten PROXY_TO_PTHREAD option to run the main application code to a separate thread",
            False,
        ),
    ]


def get_doc_classes():
    return [
        "EditorExportPlatformWeb",
    ]


def get_doc_path():
    return "doc_classes"


def get_flags():
    return [
        ("arch", "wasm32"),
        ("target", "template_debug"),
        ("builtin_pcre2_with_jit", False),
        ("vulkan", False),
        # Embree is heavy and requires too much memory (GH-70621).
        ("module_raycast_enabled", False),
        # Use -Os to prioritize optimizing for reduced file size. This is
        # particularly valuable for the web platform because it directly
        # decreases download time.
        # -Os reduces file size by around 5 MiB over -O3. -Oz only saves about
        # 100 KiB over -Os, which does not justify the negative impact on
        # run-time performance.
        ("optimize", "size"),
    ]


def configure(env: "Environment"):
    # Validate arch.
    supported_arches = ["wasm32"]
    if env["arch"] not in supported_arches:
        print(
            'Unsupported CPU architecture "%s" for iOS. Supported architectures are: %s.'
            % (env["arch"], ", ".join(supported_arches))
        )
        sys.exit()

    try:
        env["initial_memory"] = int(env["initial_memory"])
    except Exception:
        print("Initial memory must be a valid integer")
        sys.exit(255)

    ## Build type

    if env.debug_features:
        # Retain function names for backtraces at the cost of file size.
        env.Append(LINKFLAGS=["--profiling-funcs"])
    else:
        env["use_assertions"] = True

    if env["use_assertions"]:
        env.Append(LINKFLAGS=["-s", "ASSERTIONS=1"])

    if env.editor_build and env["initial_memory"] < 64:
        print('Note: Forcing "initial_memory=64" as it is required for the web editor.')
        env["initial_memory"] = 64

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
    cc_version = get_compiler_version(env)
    cc_semver = (cc_version["major"], cc_version["minor"], cc_version["patch"])

    env.Prepend(CPPPATH=["#platform/web"])
    env.Append(CPPDEFINES=["WEB_ENABLED", "UNIX_ENABLED"])

    if env["opengl3"]:
        env.AppendUnique(CPPDEFINES=["GLES3_ENABLED"])
        # This setting just makes WebGL 2 APIs available, it does NOT disable WebGL 1.
        env.Append(LINKFLAGS=["-s", "USE_WEBGL2=1"])
        # Allow use to take control of swapping WebGL buffers.
        env.Append(LINKFLAGS=["-s", "OFFSCREEN_FRAMEBUFFER=1"])
        # Breaking change since emscripten 3.1.51
        # https://github.com/emscripten-core/emscripten/blob/main/ChangeLog.md#3151---121323
        if cc_semver >= (3, 1, 51):
            # Enables the use of *glGetProcAddress()
            env.Append(LINKFLAGS=["-s", "GL_ENABLE_GET_PROC_ADDRESS=1"])

    if env["javascript_eval"]:
        env.Append(CPPDEFINES=["JAVASCRIPT_EVAL_ENABLED"])

    stack_size_opt = "STACK_SIZE" if cc_semver >= (3, 1, 25) else "TOTAL_STACK"
    env.Append(LINKFLAGS=["-s", "%s=%sKB" % (stack_size_opt, env["stack_size"])])

    # Thread support (via SharedArrayBuffer).
    env.Append(CPPDEFINES=["PTHREAD_NO_RENAME"])
    env.Append(CCFLAGS=["-s", "USE_PTHREADS=1"])
    env.Append(LINKFLAGS=["-s", "USE_PTHREADS=1"])
    env.Append(LINKFLAGS=["-s", "DEFAULT_PTHREAD_STACK_SIZE=%sKB" % env["default_pthread_stack_size"]])
    env.Append(LINKFLAGS=["-s", "PTHREAD_POOL_SIZE=8"])
    env.Append(LINKFLAGS=["-s", "WASM_MEM_MAX=2048MB"])

    if env["lto"] != "none":
        # Workaround https://github.com/emscripten-core/emscripten/issues/19781.
        if cc_semver >= (3, 1, 42) and cc_semver < (3, 1, 46):
            env.Append(LINKFLAGS=["-Wl,-u,scalbnf"])
        # Workaround https://github.com/emscripten-core/emscripten/issues/16836.
        if cc_semver >= (3, 1, 47):
            env.Append(LINKFLAGS=["-Wl,-u,_emscripten_run_callback_on_thread"])

    if env["dlink_enabled"]:
        if env["proxy_to_pthread"]:
            print("GDExtension support requires proxy_to_pthread=no, disabling")
            env["proxy_to_pthread"] = False

        if cc_semver < (3, 1, 14):
            print("GDExtension support requires emscripten >= 3.1.14, detected: %s.%s.%s" % cc_semver)
            sys.exit(255)

        env.Append(CCFLAGS=["-s", "SIDE_MODULE=2"])
        env.Append(LINKFLAGS=["-s", "SIDE_MODULE=2"])
        env.Append(CCFLAGS=["-fvisibility=hidden"])
        env.Append(LINKFLAGS=["-fvisibility=hidden"])
        env.extra_suffix = ".dlink" + env.extra_suffix

    # WASM_BIGINT is needed since emscripten ≥ 3.1.41
    needs_wasm_bigint = cc_semver >= (3, 1, 41)

    # Run the main application in a web worker
    if env["proxy_to_pthread"]:
        env.Append(LINKFLAGS=["-s", "PROXY_TO_PTHREAD=1"])
        env.Append(CPPDEFINES=["PROXY_TO_PTHREAD_ENABLED"])
        env.Append(LINKFLAGS=["-s", "EXPORTED_RUNTIME_METHODS=['_emscripten_proxy_main']"])
        # https://github.com/emscripten-core/emscripten/issues/18034#issuecomment-1277561925
        env.Append(LINKFLAGS=["-s", "TEXTDECODER=0"])
        # BigInt support to pass object pointers between contexts
        needs_wasm_bigint = True

    if needs_wasm_bigint:
        env.Append(LINKFLAGS=["-s", "WASM_BIGINT"])

    # Reduce code size by generating less support code (e.g. skip NodeJS support).
    env.Append(LINKFLAGS=["-s", "ENVIRONMENT=web,worker"])

    # Wrap the JavaScript support code around a closure named Godot.
    env.Append(LINKFLAGS=["-s", "MODULARIZE=1", "-s", "EXPORT_NAME='Godot'"])

    # Allow increasing memory buffer size during runtime. This is efficient
    # when using WebAssembly (in comparison to asm.js) and works well for
    # us since we don't know requirements at compile-time.
    env.Append(LINKFLAGS=["-s", "ALLOW_MEMORY_GROWTH=1"])

    # Do not call main immediately when the support code is ready.
    env.Append(LINKFLAGS=["-s", "INVOKE_RUN=0"])

    # callMain for manual start, cwrap for the mono version.
    env.Append(LINKFLAGS=["-s", "EXPORTED_RUNTIME_METHODS=['callMain','cwrap']"])

    # Add code that allow exiting runtime.
    env.Append(LINKFLAGS=["-s", "EXIT_RUNTIME=1"])

    # This workaround creates a closure that prevents the garbage collector from freeing the WebGL context.
    # We also only use WebGL2, and changing context version is not widely supported anyway.
    env.Append(LINKFLAGS=["-s", "GL_WORKAROUND_SAFARI_GETCONTEXT_BUG=0"])
