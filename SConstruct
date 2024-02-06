#!/usr/bin/env python

EnsureSConsVersion(3, 0, 0)
EnsurePythonVersion(3, 6)

# System
import atexit
import glob
import os
import pickle
import sys
import time
from types import ModuleType
from collections import OrderedDict
from importlib.util import spec_from_file_location, module_from_spec

# Explicitly resolve the helper modules, this is done to avoid clash with
# modules of the same name that might be randomly added (e.g. someone adding
# an `editor.py` file at the root of the module creates a clash with the editor
# folder when doing `import editor.template_builder`)


def _helper_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    # Ensure the module's parents are in loaded to avoid loading the wrong parent
    # when doing "import foo.bar" while only "foo.bar" as declared as helper module
    child_module = module
    parent_name = name
    while True:
        try:
            parent_name, child_name = parent_name.rsplit(".", 1)
        except ValueError:
            break
        try:
            parent_module = sys.modules[parent_name]
        except KeyError:
            parent_module = ModuleType(parent_name)
            sys.modules[parent_name] = parent_module
        setattr(parent_module, child_name, child_module)


_helper_module("gles3_builders", "gles3_builders.py")
_helper_module("glsl_builders", "glsl_builders.py")
_helper_module("methods", "methods.py")
_helper_module("platform_methods", "platform_methods.py")
_helper_module("version", "version.py")
_helper_module("core.core_builders", "core/core_builders.py")
_helper_module("main.main_builders", "main/main_builders.py")
_helper_module("modules.modules_builders", "modules/modules_builders.py")

# Local
import methods
import glsl_builders
import gles3_builders
import scu_builders
from platform_methods import architectures, architecture_aliases, generate_export_icons

if ARGUMENTS.get("target", "editor") == "editor":
    _helper_module("editor.editor_builders", "editor/editor_builders.py")
    _helper_module("editor.template_builders", "editor/template_builders.py")

# Scan possible build platforms

platform_list = []  # list of platforms
platform_opts = {}  # options for each platform
platform_flags = {}  # flags for each platform
platform_doc_class_path = {}
platform_exporters = []
platform_apis = []

time_at_start = time.time()

for x in sorted(glob.glob("platform/*")):
    if not os.path.isdir(x) or not os.path.exists(x + "/detect.py"):
        continue
    tmppath = "./" + x

    sys.path.insert(0, tmppath)
    import detect

    # Get doc classes paths (if present)
    try:
        doc_classes = detect.get_doc_classes()
        doc_path = detect.get_doc_path()
        for c in doc_classes:
            platform_doc_class_path[c] = x.replace("\\", "/") + "/" + doc_path
    except Exception:
        pass

    platform_name = x[9:]

    if os.path.exists(x + "/export/export.cpp"):
        platform_exporters.append(platform_name)
        generate_export_icons(x, platform_name)
    if os.path.exists(x + "/api/api.cpp"):
        platform_apis.append(platform_name)
    if detect.can_build():
        x = x.replace("platform/", "")  # rest of world
        x = x.replace("platform\\", "")  # win32
        platform_list += [x]
        platform_opts[x] = detect.get_opts()
        platform_flags[x] = detect.get_flags()
    sys.path.remove(tmppath)
    sys.modules.pop("detect")

custom_tools = ["default"]

platform_arg = ARGUMENTS.get("platform", ARGUMENTS.get("p", False))

if platform_arg == "android":
    custom_tools = ["clang", "clang++", "as", "ar", "link"]
elif platform_arg == "web":
    # Use generic POSIX build toolchain for Emscripten.
    custom_tools = ["cc", "c++", "ar", "link", "textfile", "zip"]
elif os.name == "nt" and methods.get_cmdline_bool("use_mingw", False):
    custom_tools = ["mingw"]

# We let SCons build its default ENV as it includes OS-specific things which we don't
# want to have to pull in manually.
# Then we prepend PATH to make it take precedence, while preserving SCons' own entries.
env_base = Environment(tools=custom_tools)
env_base.PrependENVPath("PATH", os.getenv("PATH"))
env_base.PrependENVPath("PKG_CONFIG_PATH", os.getenv("PKG_CONFIG_PATH"))
if "TERM" in os.environ:  # Used for colored output.
    env_base["ENV"]["TERM"] = os.environ["TERM"]

env_base.disabled_modules = []
env_base.module_version_string = ""
env_base.msvc = False

env_base.__class__.disable_module = methods.disable_module

env_base.__class__.add_module_version_string = methods.add_module_version_string

env_base.__class__.add_source_files = methods.add_source_files
env_base.__class__.use_windows_spawn_fix = methods.use_windows_spawn_fix

env_base.__class__.add_shared_library = methods.add_shared_library
env_base.__class__.add_library = methods.add_library
env_base.__class__.add_program = methods.add_program
env_base.__class__.CommandNoCache = methods.CommandNoCache
env_base.__class__.Run = methods.Run
env_base.__class__.disable_warnings = methods.disable_warnings
env_base.__class__.force_optimization_on_debug = methods.force_optimization_on_debug
env_base.__class__.module_add_dependencies = methods.module_add_dependencies
env_base.__class__.module_check_dependencies = methods.module_check_dependencies

env_base["x86_libtheora_opt_gcc"] = False
env_base["x86_libtheora_opt_vc"] = False

# avoid issues when building with different versions of python out of the same directory
env_base.SConsignFile(File("#.sconsign{0}.dblite".format(pickle.HIGHEST_PROTOCOL)).abspath)

# Build options

customs = ["custom.py"]

profile = ARGUMENTS.get("profile", "")
if profile:
    if os.path.isfile(profile):
        customs.append(profile)
    elif os.path.isfile(profile + ".py"):
        customs.append(profile + ".py")

opts = Variables(customs, ARGUMENTS)

# Target build options
opts.Add("platform", "Target platform (%s)" % ("|".join(platform_list),), "")
opts.Add("p", "Platform (alias for 'platform')", "")
opts.Add(EnumVariable("target", "Compilation target", "editor", ("editor", "template_release", "template_debug")))
opts.Add(EnumVariable("arch", "CPU architecture", "auto", ["auto"] + architectures, architecture_aliases))
opts.Add(BoolVariable("dev_build", "Developer build with dev-only debugging code (DEV_ENABLED)", False))
opts.Add(
    EnumVariable(
        "optimize", "Optimization level", "speed_trace", ("none", "custom", "debug", "speed", "speed_trace", "size")
    )
)
opts.Add(BoolVariable("debug_symbols", "Build with debugging symbols", False))
opts.Add(BoolVariable("separate_debug_symbols", "Extract debugging symbols to a separate file", False))
opts.Add(EnumVariable("lto", "Link-time optimization (production builds)", "none", ("none", "auto", "thin", "full")))
opts.Add(BoolVariable("production", "Set defaults to build Godot for use in production", False))

# Components
opts.Add(BoolVariable("deprecated", "Enable compatibility code for deprecated and removed features", True))
opts.Add(EnumVariable("precision", "Set the floating-point precision level", "single", ("single", "double")))
opts.Add(BoolVariable("minizip", "Enable ZIP archive support using minizip", True))
opts.Add(BoolVariable("brotli", "Enable Brotli for decompresson and WOFF2 fonts support", True))
opts.Add(BoolVariable("xaudio2", "Enable the XAudio2 audio driver", False))
opts.Add(BoolVariable("vulkan", "Enable the vulkan rendering driver", True))
opts.Add(BoolVariable("opengl3", "Enable the OpenGL/GLES3 rendering driver", True))
opts.Add(BoolVariable("openxr", "Enable the OpenXR driver", True))
opts.Add(BoolVariable("use_volk", "Use the volk library to load the Vulkan loader dynamically", True))
opts.Add(BoolVariable("disable_exceptions", "Force disabling exception handling code", True))
opts.Add("custom_modules", "A list of comma-separated directory paths containing custom modules to build.", "")
opts.Add(BoolVariable("custom_modules_recursive", "Detect custom modules recursively for each specified path.", True))

# Advanced options
opts.Add(BoolVariable("dev_mode", "Alias for dev options: verbose=yes warnings=extra werror=yes tests=yes", False))
opts.Add(BoolVariable("tests", "Build the unit tests", False))
opts.Add(BoolVariable("fast_unsafe", "Enable unsafe options for faster rebuilds", False))
opts.Add(BoolVariable("compiledb", "Generate compilation DB (`compile_commands.json`) for external tools", False))
opts.Add(BoolVariable("verbose", "Enable verbose output for the compilation", False))
opts.Add(BoolVariable("progress", "Show a progress indicator during compilation", True))
opts.Add(EnumVariable("warnings", "Level of compilation warnings", "all", ("extra", "all", "moderate", "no")))
opts.Add(BoolVariable("werror", "Treat compiler warnings as errors", False))
opts.Add("extra_suffix", "Custom extra suffix added to the base filename of all generated binary files", "")
opts.Add("object_prefix", "Custom prefix added to the base filename of all generated object files", "")
opts.Add(BoolVariable("vsproj", "Generate a Visual Studio solution", False))
opts.Add("vsproj_name", "Name of the Visual Studio solution", "godot")
opts.Add(BoolVariable("disable_3d", "Disable 3D nodes for a smaller executable", False))
opts.Add(BoolVariable("disable_advanced_gui", "Disable advanced GUI nodes and behaviors", False))
opts.Add("build_profile", "Path to a file containing a feature build profile", "")
opts.Add(BoolVariable("modules_enabled_by_default", "If no, disable all modules except ones explicitly enabled", True))
opts.Add(BoolVariable("no_editor_splash", "Don't use the custom splash screen for the editor", True))
opts.Add(
    "system_certs_path",
    "Use this path as TLS certificates default for editor and Linux/BSD export templates (for package maintainers)",
    "",
)
opts.Add(BoolVariable("use_precise_math_checks", "Math checks use very precise epsilon (debug option)", False))
opts.Add(BoolVariable("scu_build", "Use single compilation unit build", False))
opts.Add("scu_limit", "Max includes per SCU file when using scu_build (determines RAM use)", "0")

# Thirdparty libraries
opts.Add(BoolVariable("builtin_brotli", "Use the built-in Brotli library", True))
opts.Add(BoolVariable("builtin_certs", "Use the built-in SSL certificates bundles", True))
opts.Add(BoolVariable("builtin_clipper2", "Use the built-in Clipper2 library", True))
opts.Add(BoolVariable("builtin_embree", "Use the built-in Embree library", True))
opts.Add(BoolVariable("builtin_enet", "Use the built-in ENet library", True))
opts.Add(BoolVariable("builtin_freetype", "Use the built-in FreeType library", True))
opts.Add(BoolVariable("builtin_msdfgen", "Use the built-in MSDFgen library", True))
opts.Add(BoolVariable("builtin_glslang", "Use the built-in glslang library", True))
opts.Add(BoolVariable("builtin_graphite", "Use the built-in Graphite library", True))
opts.Add(BoolVariable("builtin_harfbuzz", "Use the built-in HarfBuzz library", True))
opts.Add(BoolVariable("builtin_icu4c", "Use the built-in ICU library", True))
opts.Add(BoolVariable("builtin_libogg", "Use the built-in libogg library", True))
opts.Add(BoolVariable("builtin_libpng", "Use the built-in libpng library", True))
opts.Add(BoolVariable("builtin_libtheora", "Use the built-in libtheora library", True))
opts.Add(BoolVariable("builtin_libvorbis", "Use the built-in libvorbis library", True))
opts.Add(BoolVariable("builtin_libwebp", "Use the built-in libwebp library", True))
opts.Add(BoolVariable("builtin_wslay", "Use the built-in wslay library", True))
opts.Add(BoolVariable("builtin_mbedtls", "Use the built-in mbedTLS library", True))
opts.Add(BoolVariable("builtin_miniupnpc", "Use the built-in miniupnpc library", True))
opts.Add(BoolVariable("builtin_openxr", "Use the built-in OpenXR library", True))
opts.Add(BoolVariable("builtin_pcre2", "Use the built-in PCRE2 library", True))
opts.Add(BoolVariable("builtin_pcre2_with_jit", "Use JIT compiler for the built-in PCRE2 library", True))
opts.Add(BoolVariable("builtin_recastnavigation", "Use the built-in Recast navigation library", True))
opts.Add(BoolVariable("builtin_rvo2_2d", "Use the built-in RVO2 2D library", True))
opts.Add(BoolVariable("builtin_rvo2_3d", "Use the built-in RVO2 3D library", True))
opts.Add(BoolVariable("builtin_squish", "Use the built-in squish library", True))
opts.Add(BoolVariable("builtin_xatlas", "Use the built-in xatlas library", True))
opts.Add(BoolVariable("builtin_zlib", "Use the built-in zlib library", True))
opts.Add(BoolVariable("builtin_zstd", "Use the built-in Zstd library", True))

# Compilation environment setup
opts.Add("CXX", "C++ compiler")
opts.Add("CC", "C compiler")
opts.Add("LINK", "Linker")
opts.Add("CCFLAGS", "Custom flags for both the C and C++ compilers")
opts.Add("CFLAGS", "Custom flags for the C compiler")
opts.Add("CXXFLAGS", "Custom flags for the C++ compiler")
opts.Add("LINKFLAGS", "Custom flags for the linker")

# Update the environment to have all above options defined
# in following code (especially platform and custom_modules).
opts.Update(env_base)

# Platform selection: validate input, and add options.

selected_platform = ""

if env_base["platform"] != "":
    selected_platform = env_base["platform"]
elif env_base["p"] != "":
    selected_platform = env_base["p"]
else:
    # Missing `platform` argument, try to detect platform automatically
    if (
        sys.platform.startswith("linux")
        or sys.platform.startswith("dragonfly")
        or sys.platform.startswith("freebsd")
        or sys.platform.startswith("netbsd")
        or sys.platform.startswith("openbsd")
    ):
        selected_platform = "linuxbsd"
    elif sys.platform == "darwin":
        selected_platform = "macos"
    elif sys.platform == "win32":
        selected_platform = "windows"
    else:
        print("Could not detect platform automatically. Supported platforms:")
        for x in platform_list:
            print("\t" + x)
        print("\nPlease run SCons again and select a valid platform: platform=<string>")

    if selected_platform != "":
        print("Automatically detected platform: " + selected_platform)

if selected_platform in ["macos", "osx"]:
    if selected_platform == "osx":
        # Deprecated alias kept for compatibility.
        print('Platform "osx" has been renamed to "macos" in Godot 4. Building for platform "macos".')
    # Alias for convenience.
    selected_platform = "macos"

if selected_platform in ["ios", "iphone"]:
    if selected_platform == "iphone":
        # Deprecated alias kept for compatibility.
        print('Platform "iphone" has been renamed to "ios" in Godot 4. Building for platform "ios".')
    # Alias for convenience.
    selected_platform = "ios"

if selected_platform in ["linux", "bsd", "x11"]:
    if selected_platform == "x11":
        # Deprecated alias kept for compatibility.
        print('Platform "x11" has been renamed to "linuxbsd" in Godot 4. Building for platform "linuxbsd".')
    # Alias for convenience.
    selected_platform = "linuxbsd"

# Make sure to update this to the found, valid platform as it's used through the buildsystem as the reference.
# It should always be re-set after calling `opts.Update()` otherwise it uses the original input value.
env_base["platform"] = selected_platform

# Add platform-specific options.
if selected_platform in platform_opts:
    for opt in platform_opts[selected_platform]:
        opts.Add(opt)

# Update the environment to take platform-specific options into account.
opts.Update(env_base)
env_base["platform"] = selected_platform  # Must always be re-set after calling opts.Update().

# Detect modules.
modules_detected = OrderedDict()
module_search_paths = ["modules"]  # Built-in path.

if env_base["custom_modules"]:
    paths = env_base["custom_modules"].split(",")
    for p in paths:
        try:
            module_search_paths.append(methods.convert_custom_modules_path(p))
        except ValueError as e:
            print(e)
            Exit(255)

for path in module_search_paths:
    if path == "modules":
        # Built-in modules don't have nested modules,
        # so save the time it takes to parse directories.
        modules = methods.detect_modules(path, recursive=False)
    else:  # Custom.
        modules = methods.detect_modules(path, env_base["custom_modules_recursive"])
        # Provide default include path for both the custom module search `path`
        # and the base directory containing custom modules, as it may be different
        # from the built-in "modules" name (e.g. "custom_modules/summator/summator.h"),
        # so it can be referenced simply as `#include "summator/summator.h"`
        # independently of where a module is located on user's filesystem.
        env_base.Prepend(CPPPATH=[path, os.path.dirname(path)])
    # Note: custom modules can override built-in ones.
    modules_detected.update(modules)

# Add module options.
for name, path in modules_detected.items():
    sys.path.insert(0, path)
    import config

    if env_base["modules_enabled_by_default"]:
        enabled = True
        try:
            enabled = config.is_enabled()
        except AttributeError:
            pass
    else:
        enabled = False

    opts.Add(BoolVariable("module_" + name + "_enabled", "Enable module '%s'" % (name,), enabled))

    # Add module-specific options.
    try:
        for opt in config.get_opts(selected_platform):
            opts.Add(opt)
    except AttributeError:
        pass

    sys.path.remove(path)
    sys.modules.pop("config")

methods.write_modules(modules_detected)

# Update the environment again after all the module options are added.
opts.Update(env_base)
env_base["platform"] = selected_platform  # Must always be re-set after calling opts.Update().
Help(opts.GenerateHelpText(env_base))

# add default include paths

env_base.Prepend(CPPPATH=["#"])

# configure ENV for platform
env_base.platform_exporters = platform_exporters
env_base.platform_apis = platform_apis

# Configuration of build targets:
# - Editor or template
# - Debug features (DEBUG_ENABLED code)
# - Dev only code (DEV_ENABLED code)
# - Optimization level
# - Debug symbols for crash traces / debuggers

env_base.editor_build = env_base["target"] == "editor"
env_base.dev_build = env_base["dev_build"]
env_base.debug_features = env_base["target"] in ["editor", "template_debug"]

if env_base.dev_build:
    opt_level = "none"
elif env_base.debug_features:
    opt_level = "speed_trace"
else:  # Release
    opt_level = "speed"

env_base["optimize"] = ARGUMENTS.get("optimize", opt_level)
env_base["debug_symbols"] = methods.get_cmdline_bool("debug_symbols", env_base.dev_build)

if env_base.editor_build:
    env_base.Append(CPPDEFINES=["TOOLS_ENABLED"])

if env_base.debug_features:
    # DEBUG_ENABLED enables debugging *features* and debug-only code, which is intended
    # to give *users* extra debugging information for their game development.
    env_base.Append(CPPDEFINES=["DEBUG_ENABLED"])

if env_base.dev_build:
    # DEV_ENABLED enables *engine developer* code which should only be compiled for those
    # working on the engine itself.
    env_base.Append(CPPDEFINES=["DEV_ENABLED"])
else:
    # Disable assert() for production targets (only used in thirdparty code).
    env_base.Append(CPPDEFINES=["NDEBUG"])

# SCons speed optimization controlled by the `fast_unsafe` option, which provide
# more than 10 s speed up for incremental rebuilds.
# Unsafe as they reduce the certainty of rebuilding all changed files, so it's
# enabled by default for `debug` builds, and can be overridden from command line.
# Ref: https://github.com/SCons/scons/wiki/GoFastButton
if methods.get_cmdline_bool("fast_unsafe", env_base.dev_build):
    # Renamed to `content-timestamp` in SCons >= 4.2, keeping MD5 for compat.
    env_base.Decider("MD5-timestamp")
    env_base.SetOption("implicit_cache", 1)
    env_base.SetOption("max_drift", 60)

if env_base["use_precise_math_checks"]:
    env_base.Append(CPPDEFINES=["PRECISE_MATH_CHECKS"])

if not env_base.File("#main/splash_editor.png").exists():
    # Force disabling editor splash if missing.
    env_base["no_editor_splash"] = True
if env_base["no_editor_splash"]:
    env_base.Append(CPPDEFINES=["NO_EDITOR_SPLASH"])

if not env_base["deprecated"]:
    env_base.Append(CPPDEFINES=["DISABLE_DEPRECATED"])

if env_base["precision"] == "double":
    env_base.Append(CPPDEFINES=["REAL_T_IS_DOUBLE"])

if selected_platform in platform_list:
    tmppath = "./platform/" + selected_platform
    sys.path.insert(0, tmppath)
    import detect

    env = env_base.Clone()

    # Default num_jobs to local cpu count if not user specified.
    # SCons has a peculiarity where user-specified options won't be overridden
    # by SetOption, so we can rely on this to know if we should use our default.
    initial_num_jobs = env.GetOption("num_jobs")
    altered_num_jobs = initial_num_jobs + 1
    env.SetOption("num_jobs", altered_num_jobs)
    if env.GetOption("num_jobs") == altered_num_jobs:
        cpu_count = os.cpu_count()
        if cpu_count is None:
            print("Couldn't auto-detect CPU count to configure build parallelism. Specify it with the -j argument.")
        else:
            safer_cpu_count = cpu_count if cpu_count <= 4 else cpu_count - 1
            print(
                "Auto-detected %d CPU cores available for build parallelism. Using %d cores by default. You can override it with the -j argument."
                % (cpu_count, safer_cpu_count)
            )
            env.SetOption("num_jobs", safer_cpu_count)

    env.extra_suffix = ""

    if env["extra_suffix"] != "":
        env.extra_suffix += "." + env["extra_suffix"]

    # Environment flags
    CCFLAGS = env.get("CCFLAGS", "")
    env["CCFLAGS"] = ""
    env.Append(CCFLAGS=str(CCFLAGS).split())

    CFLAGS = env.get("CFLAGS", "")
    env["CFLAGS"] = ""
    env.Append(CFLAGS=str(CFLAGS).split())

    CXXFLAGS = env.get("CXXFLAGS", "")
    env["CXXFLAGS"] = ""
    env.Append(CXXFLAGS=str(CXXFLAGS).split())

    LINKFLAGS = env.get("LINKFLAGS", "")
    env["LINKFLAGS"] = ""
    env.Append(LINKFLAGS=str(LINKFLAGS).split())

    # Feature build profile
    disabled_classes = []
    if env["build_profile"] != "":
        print("Using feature build profile: " + env["build_profile"])
        import json

        try:
            ft = json.load(open(env["build_profile"]))
            if "disabled_classes" in ft:
                disabled_classes = ft["disabled_classes"]
            if "disabled_build_options" in ft:
                dbo = ft["disabled_build_options"]
                for c in dbo:
                    env[c] = dbo[c]
        except:
            print("Error opening feature build profile: " + env["build_profile"])
            Exit(255)
    methods.write_disabled_classes(disabled_classes)

    # Platform specific flags.
    # These can sometimes override default options.
    flag_list = platform_flags[selected_platform]
    for f in flag_list:
        if not (f[0] in ARGUMENTS) or ARGUMENTS[f[0]] == "auto":  # Allow command line to override platform flags
            env[f[0]] = f[1]

    # 'dev_mode' and 'production' are aliases to set default options if they haven't been
    # set manually by the user.
    # These need to be checked *after* platform specific flags so that different
    # default values can be set (e.g. to keep LTO off for `production` on some platforms).
    if env["dev_mode"]:
        env["verbose"] = methods.get_cmdline_bool("verbose", True)
        env["warnings"] = ARGUMENTS.get("warnings", "extra")
        env["werror"] = methods.get_cmdline_bool("werror", True)
        env["tests"] = methods.get_cmdline_bool("tests", True)
    if env["production"]:
        env["use_static_cpp"] = methods.get_cmdline_bool("use_static_cpp", True)
        env["debug_symbols"] = methods.get_cmdline_bool("debug_symbols", False)
        # LTO "auto" means we handle the preferred option in each platform detect.py.
        env["lto"] = ARGUMENTS.get("lto", "auto")

    # Run SCU file generation script if in a SCU build.
    if env["scu_build"]:
        max_includes_per_scu = 8
        if env_base.dev_build == True:
            max_includes_per_scu = 1024

        read_scu_limit = int(env["scu_limit"])
        read_scu_limit = max(0, min(read_scu_limit, 1024))
        if read_scu_limit != 0:
            max_includes_per_scu = read_scu_limit

        methods.set_scu_folders(scu_builders.generate_scu_files(max_includes_per_scu))

    # Must happen after the flags' definition, as configure is when most flags
    # are actually handled to change compile options, etc.
    detect.configure(env)

    print(f'Building for platform "{selected_platform}", architecture "{env["arch"]}", target "{env["target"]}".')
    if env.dev_build:
        print("NOTE: Developer build, with debug optimization level and debug symbols (unless overridden).")

    # Set optimize and debug_symbols flags.
    # "custom" means do nothing and let users set their own optimization flags.
    # Needs to happen after configure to have `env.msvc` defined.
    if env.msvc:
        if env["debug_symbols"]:
            env.Append(CCFLAGS=["/Zi", "/FS"])
            env.Append(LINKFLAGS=["/DEBUG:FULL"])
        else:
            env.Append(LINKFLAGS=["/DEBUG:NONE"])

        if env["optimize"] == "speed":
            env.Append(CCFLAGS=["/O2"])
            env.Append(LINKFLAGS=["/OPT:REF"])
        elif env["optimize"] == "speed_trace":
            env.Append(CCFLAGS=["/O2"])
            env.Append(LINKFLAGS=["/OPT:REF", "/OPT:NOICF"])
        elif env["optimize"] == "size":
            env.Append(CCFLAGS=["/O1"])
            env.Append(LINKFLAGS=["/OPT:REF"])
        elif env["optimize"] == "debug" or env["optimize"] == "none":
            env.Append(CCFLAGS=["/Od"])
    else:
        if env["debug_symbols"]:
            # Adding dwarf-4 explicitly makes stacktraces work with clang builds,
            # otherwise addr2line doesn't understand them
            env.Append(CCFLAGS=["-gdwarf-4"])
            if env.dev_build:
                env.Append(CCFLAGS=["-g3"])
            else:
                env.Append(CCFLAGS=["-g2"])
        else:
            if methods.using_clang(env) and not methods.is_vanilla_clang(env):
                # Apple Clang, its linker doesn't like -s.
                env.Append(LINKFLAGS=["-Wl,-S", "-Wl,-x", "-Wl,-dead_strip"])
            else:
                env.Append(LINKFLAGS=["-s"])

        if env["optimize"] == "speed":
            env.Append(CCFLAGS=["-O3"])
        # `-O2` is friendlier to debuggers than `-O3`, leading to better crash backtraces.
        elif env["optimize"] == "speed_trace":
            env.Append(CCFLAGS=["-O2"])
        elif env["optimize"] == "size":
            env.Append(CCFLAGS=["-Os"])
        elif env["optimize"] == "debug":
            env.Append(CCFLAGS=["-Og"])
        elif env["optimize"] == "none":
            env.Append(CCFLAGS=["-O0"])

    # Needs to happen after configure to handle "auto".
    if env["lto"] != "none":
        print("Using LTO: " + env["lto"])

    # Set our C and C++ standard requirements.
    # C++17 is required as we need guaranteed copy elision as per GH-36436.
    # Prepending to make it possible to override.
    # This needs to come after `configure`, otherwise we don't have env.msvc.
    if not env.msvc:
        # Specifying GNU extensions support explicitly, which are supported by
        # both GCC and Clang. Both currently default to gnu11 and gnu++14.
        env.Prepend(CFLAGS=["-std=gnu11"])
        env.Prepend(CXXFLAGS=["-std=gnu++17"])
    else:
        # MSVC doesn't have clear C standard support, /std only covers C++.
        # We apply it to CCFLAGS (both C and C++ code) in case it impacts C features.
        env.Prepend(CCFLAGS=["/std:c++17"])

    # Enforce our minimal compiler version requirements
    cc_version = methods.get_compiler_version(env) or {
        "major": None,
        "minor": None,
        "patch": None,
        "metadata1": None,
        "metadata2": None,
        "date": None,
    }
    cc_version_major = int(cc_version["major"] or -1)
    cc_version_minor = int(cc_version["minor"] or -1)
    cc_version_metadata1 = cc_version["metadata1"] or ""

    if methods.using_gcc(env):
        if cc_version_major == -1:
            print(
                "Couldn't detect compiler version, skipping version checks. "
                "Build may fail if the compiler doesn't support C++17 fully."
            )
        # GCC 8 before 8.4 has a regression in the support of guaranteed copy elision
        # which causes a build failure: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=86521
        elif cc_version_major == 8 and cc_version_minor < 4:
            print(
                "Detected GCC 8 version < 8.4, which is not supported due to a "
                "regression in its C++17 guaranteed copy elision support. Use a "
                'newer GCC version, or Clang 6 or later by passing "use_llvm=yes" '
                "to the SCons command line."
            )
            Exit(255)
        elif cc_version_major < 7:
            print(
                "Detected GCC version older than 7, which does not fully support "
                "C++17. Supported versions are GCC 7, 9 and later. Use a newer GCC "
                'version, or Clang 6 or later by passing "use_llvm=yes" to the '
                "SCons command line."
            )
            Exit(255)
        elif cc_version_metadata1 == "win32":
            print(
                "Detected mingw version is not using posix threads. Only posix "
                "version of mingw is supported. "
                'Use "update-alternatives --config x86_64-w64-mingw32-g++" '
                "to switch to posix threads."
            )
            Exit(255)
    elif methods.using_clang(env):
        if cc_version_major == -1:
            print(
                "Couldn't detect compiler version, skipping version checks. "
                "Build may fail if the compiler doesn't support C++17 fully."
            )
        # Apple LLVM versions differ from upstream LLVM version \o/, compare
        # in https://en.wikipedia.org/wiki/Xcode#Toolchain_versions
        elif env["platform"] == "macos" or env["platform"] == "ios":
            vanilla = methods.is_vanilla_clang(env)
            if vanilla and cc_version_major < 6:
                print(
                    "Detected Clang version older than 6, which does not fully support "
                    "C++17. Supported versions are Clang 6 and later."
                )
                Exit(255)
            elif not vanilla and cc_version_major < 10:
                print(
                    "Detected Apple Clang version older than 10, which does not fully "
                    "support C++17. Supported versions are Apple Clang 10 and later."
                )
                Exit(255)
        elif cc_version_major < 6:
            print(
                "Detected Clang version older than 6, which does not fully support "
                "C++17. Supported versions are Clang 6 and later."
            )
            Exit(255)

    # Disable exception handling. Godot doesn't use exceptions anywhere, and this
    # saves around 20% of binary size and very significant build time (GH-80513).
    if env["disable_exceptions"]:
        if env.msvc:
            env.Append(CPPDEFINES=[("_HAS_EXCEPTIONS", 0)])
        else:
            env.Append(CXXFLAGS=["-fno-exceptions"])
    elif env.msvc:
        env.Append(CXXFLAGS=["/EHsc"])

    # Configure compiler warnings
    if env.msvc:  # MSVC
        if env["warnings"] == "no":
            env.Append(CCFLAGS=["/w"])
        else:
            if env["warnings"] == "extra":
                env.Append(CCFLAGS=["/W4"])
            elif env["warnings"] == "all":
                # C4458 is like -Wshadow. Part of /W4 but let's apply it for the default /W3 too.
                env.Append(CCFLAGS=["/W3", "/w34458"])
            elif env["warnings"] == "moderate":
                env.Append(CCFLAGS=["/W2"])
            # Disable warnings which we don't plan to fix.

            env.Append(
                CCFLAGS=[
                    "/wd4100",  # C4100 (unreferenced formal parameter): Doesn't play nice with polymorphism.
                    "/wd4127",  # C4127 (conditional expression is constant)
                    "/wd4201",  # C4201 (non-standard nameless struct/union): Only relevant for C89.
                    "/wd4244",  # C4244 C4245 C4267 (narrowing conversions): Unavoidable at this scale.
                    "/wd4245",
                    "/wd4267",
                    "/wd4305",  # C4305 (truncation): double to float or real_t, too hard to avoid.
                    "/wd4514",  # C4514 (unreferenced inline function has been removed)
                    "/wd4714",  # C4714 (function marked as __forceinline not inlined)
                    "/wd4820",  # C4820 (padding added after construct)
                ]
            )

        if env["werror"]:
            env.Append(CCFLAGS=["/WX"])
            env.Append(LINKFLAGS=["/WX"])
    else:  # GCC, Clang
        common_warnings = []

        if methods.using_gcc(env):
            common_warnings += ["-Wshadow", "-Wno-misleading-indentation"]
            if cc_version_major == 7:  # Bogus warning fixed in 8+.
                common_warnings += ["-Wno-strict-overflow"]
            if cc_version_major < 11:
                # Regression in GCC 9/10, spams so much in our variadic templates
                # that we need to outright disable it.
                common_warnings += ["-Wno-type-limits"]
            if cc_version_major >= 12:  # False positives in our error macros, see GH-58747.
                common_warnings += ["-Wno-return-type"]
        elif methods.using_clang(env) or methods.using_emcc(env):
            common_warnings += ["-Wshadow-field-in-constructor", "-Wshadow-uncaptured-local"]
            # We often implement `operator<` for structs of pointers as a requirement
            # for putting them in `Set` or `Map`. We don't mind about unreliable ordering.
            common_warnings += ["-Wno-ordered-compare-function-pointers"]

        if env["warnings"] == "extra":
            env.Append(CCFLAGS=["-Wall", "-Wextra", "-Wwrite-strings", "-Wno-unused-parameter"] + common_warnings)
            env.Append(CXXFLAGS=["-Wctor-dtor-privacy", "-Wnon-virtual-dtor"])
            if methods.using_gcc(env):
                env.Append(
                    CCFLAGS=[
                        "-Walloc-zero",
                        "-Wduplicated-branches",
                        "-Wduplicated-cond",
                        "-Wstringop-overflow=4",
                    ]
                )
                env.Append(CXXFLAGS=["-Wplacement-new=1"])
                # Need to fix a warning with AudioServer lambdas before enabling.
                # if cc_version_major != 9:  # GCC 9 had a regression (GH-36325).
                #    env.Append(CXXFLAGS=["-Wnoexcept"])
                if cc_version_major >= 9:
                    env.Append(CCFLAGS=["-Wattribute-alias=2"])
                if cc_version_major >= 11:  # Broke on MethodBind templates before GCC 11.
                    env.Append(CCFLAGS=["-Wlogical-op"])
            elif methods.using_clang(env) or methods.using_emcc(env):
                env.Append(CCFLAGS=["-Wimplicit-fallthrough"])
        elif env["warnings"] == "all":
            env.Append(CCFLAGS=["-Wall"] + common_warnings)
        elif env["warnings"] == "moderate":
            env.Append(CCFLAGS=["-Wall", "-Wno-unused"] + common_warnings)
        else:  # 'no'
            env.Append(CCFLAGS=["-w"])

        if env["werror"]:
            env.Append(CCFLAGS=["-Werror"])

    if hasattr(detect, "get_program_suffix"):
        suffix = "." + detect.get_program_suffix()
    else:
        suffix = "." + selected_platform

    suffix += "." + env["target"]
    if env.dev_build:
        suffix += ".dev"

    if env_base["precision"] == "double":
        suffix += ".double"

    suffix += "." + env["arch"]
    suffix += env.extra_suffix

    sys.path.remove(tmppath)
    sys.modules.pop("detect")

    modules_enabled = OrderedDict()
    env.module_dependencies = {}
    env.module_icons_paths = []
    env.doc_class_path = platform_doc_class_path

    for name, path in modules_detected.items():
        if not env["module_" + name + "_enabled"]:
            continue
        sys.path.insert(0, path)
        env.current_module = name
        import config

        if config.can_build(env, selected_platform):
            # Disable it if a required dependency is missing.
            if not env.module_check_dependencies(name):
                continue

            config.configure(env)
            # Get doc classes paths (if present)
            try:
                doc_classes = config.get_doc_classes()
                doc_path = config.get_doc_path()
                for c in doc_classes:
                    env.doc_class_path[c] = path + "/" + doc_path
            except Exception:
                pass
            # Get icon paths (if present)
            try:
                icons_path = config.get_icons_path()
                env.module_icons_paths.append(path + "/" + icons_path)
            except Exception:
                # Default path for module icons
                env.module_icons_paths.append(path + "/" + "icons")
            modules_enabled[name] = path

        sys.path.remove(path)
        sys.modules.pop("config")

    env.module_list = modules_enabled
    methods.sort_module_list(env)

    if env.editor_build:
        # Add editor-specific dependencies to the dependency graph.
        env.module_add_dependencies("editor", ["freetype", "svg"])

        # And check if they are met.
        if not env.module_check_dependencies("editor"):
            print("Not all modules required by editor builds are enabled.")
            Exit(255)

    methods.generate_version_header(env.module_version_string)

    env["PROGSUFFIX_WRAP"] = suffix + env.module_version_string + ".console" + env["PROGSUFFIX"]
    env["PROGSUFFIX"] = suffix + env.module_version_string + env["PROGSUFFIX"]
    env["OBJSUFFIX"] = suffix + env["OBJSUFFIX"]
    # (SH)LIBSUFFIX will be used for our own built libraries
    # LIBSUFFIXES contains LIBSUFFIX and SHLIBSUFFIX by default,
    # so we need to append the default suffixes to keep the ability
    # to link against thirdparty libraries (.a, .so, .lib, etc.).
    if os.name == "nt":
        # On Windows, only static libraries and import libraries can be
        # statically linked - both using .lib extension
        env["LIBSUFFIXES"] += [env["LIBSUFFIX"]]
    else:
        env["LIBSUFFIXES"] += [env["LIBSUFFIX"], env["SHLIBSUFFIX"]]
    env["LIBSUFFIX"] = suffix + env["LIBSUFFIX"]
    env["SHLIBSUFFIX"] = suffix + env["SHLIBSUFFIX"]

    env["OBJPREFIX"] = env["object_prefix"]
    env["SHOBJPREFIX"] = env["object_prefix"]

    if env["disable_3d"]:
        if env.editor_build:
            print("Build option 'disable_3d=yes' cannot be used for editor builds, only for export template builds.")
            Exit(255)
        else:
            env.Append(CPPDEFINES=["_3D_DISABLED"])
    if env["disable_advanced_gui"]:
        if env.editor_build:
            print(
                "Build option 'disable_advanced_gui=yes' cannot be used for editor builds, "
                "only for export template builds."
            )
            Exit(255)
        else:
            env.Append(CPPDEFINES=["ADVANCED_GUI_DISABLED"])
    if env["minizip"]:
        env.Append(CPPDEFINES=["MINIZIP_ENABLED"])
    if env["brotli"]:
        env.Append(CPPDEFINES=["BROTLI_ENABLED"])

    if not env["verbose"]:
        methods.no_verbose(sys, env)

    GLSL_BUILDERS = {
        "RD_GLSL": env.Builder(
            action=env.Run(glsl_builders.build_rd_headers, 'Building RD_GLSL header: "$TARGET"'),
            suffix="glsl.gen.h",
            src_suffix=".glsl",
        ),
        "GLSL_HEADER": env.Builder(
            action=env.Run(glsl_builders.build_raw_headers, 'Building GLSL header: "$TARGET"'),
            suffix="glsl.gen.h",
            src_suffix=".glsl",
        ),
        "GLES3_GLSL": env.Builder(
            action=env.Run(gles3_builders.build_gles3_headers, 'Building GLES3 GLSL header: "$TARGET"'),
            suffix="glsl.gen.h",
            src_suffix=".glsl",
        ),
    }
    env.Append(BUILDERS=GLSL_BUILDERS)

    scons_cache_path = os.environ.get("SCONS_CACHE")
    if scons_cache_path != None:
        CacheDir(scons_cache_path)
        print("Scons cache enabled... (path: '" + scons_cache_path + "')")

    if env["vsproj"]:
        env.vs_incs = []
        env.vs_srcs = []

    if env["compiledb"]:
        # Generating the compilation DB (`compile_commands.json`) requires SCons 4.0.0 or later.
        from SCons import __version__ as scons_raw_version

        scons_ver = env._get_major_minor_revision(scons_raw_version)

        if scons_ver < (4, 0, 0):
            print("The `compiledb=yes` option requires SCons 4.0 or later, but your version is %s." % scons_raw_version)
            Exit(255)

        env.Tool("compilation_db")
        env.Alias("compiledb", env.CompilationDatabase())

    Export("env")

    # Build subdirs, the build order is dependent on link order.
    SConscript("core/SCsub")
    SConscript("servers/SCsub")
    SConscript("scene/SCsub")
    if env.editor_build:
        SConscript("editor/SCsub")
    SConscript("drivers/SCsub")

    SConscript("platform/SCsub")
    SConscript("modules/SCsub")
    if env["tests"]:
        SConscript("tests/SCsub")
    SConscript("main/SCsub")

    SConscript("platform/" + selected_platform + "/SCsub")  # Build selected platform.

    # Microsoft Visual Studio Project Generation
    if env["vsproj"]:
        if os.name != "nt":
            print("Error: The `vsproj` option is only usable on Windows with Visual Studio.")
            Exit(255)
        env["CPPPATH"] = [Dir(path) for path in env["CPPPATH"]]
        methods.generate_vs_project(env, ARGUMENTS, env["vsproj_name"])
        methods.generate_cpp_hint_file("cpp.hint")

    # Check for the existence of headers
    conf = Configure(env)
    if "check_c_headers" in env:
        headers = env["check_c_headers"]
        for header in headers:
            if conf.CheckCHeader(header):
                env.AppendUnique(CPPDEFINES=[headers[header]])

elif selected_platform != "":
    if selected_platform == "list":
        print("The following platforms are available:\n")
    else:
        print('Invalid target platform "' + selected_platform + '".')
        print("The following platforms were detected:\n")

    for x in platform_list:
        print("\t" + x)

    print("\nPlease run SCons again and select a valid platform: platform=<string>")

    if selected_platform == "list":
        # Exit early to suppress the rest of the built-in SCons messages
        Exit()
    else:
        Exit(255)

# The following only makes sense when the 'env' is defined, and assumes it is.
if "env" in locals():
    # FIXME: This method mixes both cosmetic progress stuff and cache handling...
    methods.show_progress(env)
    # TODO: replace this with `env.Dump(format="json")`
    # once we start requiring SCons 4.0 as min version.
    methods.dump(env)


def print_elapsed_time():
    elapsed_time_sec = round(time.time() - time_at_start, 3)
    time_ms = round((elapsed_time_sec % 1) * 1000)
    print("[Time elapsed: {}.{:03}]".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time_sec)), time_ms))


atexit.register(print_elapsed_time)
