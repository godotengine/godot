import os
import re
import glob
import subprocess
from collections import OrderedDict

# We need to define our own `Action` method to control the verbosity of output
# and whenever we need to run those commands in a subprocess on some platforms.
from SCons import Node
from SCons.Script import Action
from SCons.Script import ARGUMENTS
from SCons.Script import Glob
from SCons.Variables.BoolVariable import _text2bool
from platform_methods import run_in_subprocess


def add_source_files(self, sources, files, warn_duplicates=True):
    # Convert string to list of absolute paths (including expanding wildcard)
    if isinstance(files, (str, bytes)):
        # Keep SCons project-absolute path as they are (no wildcard support)
        if files.startswith("#"):
            if "*" in files:
                print("ERROR: Wildcards can't be expanded in SCons project-absolute path: '{}'".format(files))
                return
            files = [files]
        else:
            dir_path = self.Dir(".").abspath
            files = sorted(glob.glob(dir_path + "/" + files))

    # Add each path as compiled Object following environment (self) configuration
    for path in files:
        obj = self.Object(path)
        if obj in sources:
            if warn_duplicates:
                print('WARNING: Object "{}" already included in environment sources.'.format(obj))
            else:
                continue
        sources.append(obj)


def disable_warnings(self):
    # 'self' is the environment
    if self.msvc:
        # We have to remove existing warning level defines before appending /w,
        # otherwise we get: "warning D9025 : overriding '/W3' with '/w'"
        warn_flags = ["/Wall", "/W4", "/W3", "/W2", "/W1", "/WX"]
        self.Append(CCFLAGS=["/w"])
        self.Append(CFLAGS=["/w"])
        self.Append(CXXFLAGS=["/w"])
        self["CCFLAGS"] = [x for x in self["CCFLAGS"] if not x in warn_flags]
        self["CFLAGS"] = [x for x in self["CFLAGS"] if not x in warn_flags]
        self["CXXFLAGS"] = [x for x in self["CXXFLAGS"] if not x in warn_flags]
    else:
        self.Append(CCFLAGS=["-w"])
        self.Append(CFLAGS=["-w"])
        self.Append(CXXFLAGS=["-w"])


def add_module_version_string(self, s):
    self.module_version_string += "." + s


def update_version(module_version_string=""):

    build_name = "custom_build"
    if os.getenv("BUILD_NAME") != None:
        build_name = os.getenv("BUILD_NAME")
        print("Using custom build name: " + build_name)

    import version

    # NOTE: It is safe to generate this file here, since this is still executed serially
    f = open("core/version_generated.gen.h", "w")
    f.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    f.write("#ifndef VERSION_GENERATED_GEN_H\n")
    f.write("#define VERSION_GENERATED_GEN_H\n")
    f.write('#define VERSION_SHORT_NAME "' + str(version.short_name) + '"\n')
    f.write('#define VERSION_NAME "' + str(version.name) + '"\n')
    f.write("#define VERSION_MAJOR " + str(version.major) + "\n")
    f.write("#define VERSION_MINOR " + str(version.minor) + "\n")
    f.write("#define VERSION_PATCH " + str(version.patch) + "\n")
    f.write('#define VERSION_STATUS "' + str(version.status) + '"\n')
    f.write('#define VERSION_BUILD "' + str(build_name) + '"\n')
    f.write('#define VERSION_MODULE_CONFIG "' + str(version.module_config) + module_version_string + '"\n')
    f.write("#define VERSION_YEAR " + str(version.year) + "\n")
    f.write('#define VERSION_WEBSITE "' + str(version.website) + '"\n')
    f.write("#endif // VERSION_GENERATED_GEN_H\n")
    f.close()

    # NOTE: It is safe to generate this file here, since this is still executed serially
    fhash = open("core/version_hash.gen.h", "w")
    fhash.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    fhash.write("#ifndef VERSION_HASH_GEN_H\n")
    fhash.write("#define VERSION_HASH_GEN_H\n")
    githash = ""
    gitfolder = ".git"

    if os.path.isfile(".git"):
        module_folder = open(".git", "r").readline().strip()
        if module_folder.startswith("gitdir: "):
            gitfolder = module_folder[8:]

    if os.path.isfile(os.path.join(gitfolder, "HEAD")):
        head = open(os.path.join(gitfolder, "HEAD"), "r", encoding="utf8").readline().strip()
        if head.startswith("ref: "):
            head = os.path.join(gitfolder, head[5:])
            if os.path.isfile(head):
                githash = open(head, "r").readline().strip()
        else:
            githash = head

    fhash.write('#define VERSION_HASH "' + githash + '"\n')
    fhash.write("#endif // VERSION_HASH_GEN_H\n")
    fhash.close()


def parse_cg_file(fname, uniforms, sizes, conditionals):

    fs = open(fname, "r")
    line = fs.readline()

    while line:

        if re.match(r"^\s*uniform", line):

            res = re.match(r"uniform ([\d\w]*) ([\d\w]*)")
            type = res.groups(1)
            name = res.groups(2)

            uniforms.append(name)

            if type.find("texobj") != -1:
                sizes.append(1)
            else:
                t = re.match(r"float(\d)x(\d)", type)
                if t:
                    sizes.append(int(t.groups(1)) * int(t.groups(2)))
                else:
                    t = re.match(r"float(\d)", type)
                    sizes.append(int(t.groups(1)))

            if line.find("[branch]") != -1:
                conditionals.append(name)

        line = fs.readline()

    fs.close()


def get_cmdline_bool(option, default):
    """We use `ARGUMENTS.get()` to check if options were manually overridden on the command line,
    and SCons' _text2bool helper to convert them to booleans, otherwise they're handled as strings.
    """
    cmdline_val = ARGUMENTS.get(option)
    if cmdline_val is not None:
        return _text2bool(cmdline_val)
    else:
        return default


def detect_modules(search_path, recursive=False):
    """Detects and collects a list of C++ modules at specified path

    `search_path` - a directory path containing modules. The path may point to
    a single module, which may have other nested modules. A module must have
    "register_types.h", "SCsub", "config.py" files created to be detected.

    `recursive` - if `True`, then all subdirectories are searched for modules as
    specified by the `search_path`, otherwise collects all modules under the
    `search_path` directory. If the `search_path` is a module, it is collected
    in all cases.

    Returns an `OrderedDict` with module names as keys, and directory paths as
    values. If a path is relative, then it is a built-in module. If a path is
    absolute, then it is a custom module collected outside of the engine source.
    """
    modules = OrderedDict()

    def add_module(path):
        module_name = os.path.basename(path)
        module_path = path.replace("\\", "/")  # win32
        modules[module_name] = module_path

    def is_engine(path):
        # Prevent recursively detecting modules in self and other
        # Godot sources when using `custom_modules` build option.
        version_path = os.path.join(path, "version.py")
        if os.path.exists(version_path):
            with open(version_path) as f:
                if 'short_name = "godot"' in f.read():
                    return True
        return False

    def get_files(path):
        files = glob.glob(os.path.join(path, "*"))
        # Sort so that `register_module_types` does not change that often,
        # and plugins are registered in alphabetic order as well.
        files.sort()
        return files

    if not recursive:
        if is_module(search_path):
            add_module(search_path)
        for path in get_files(search_path):
            if is_engine(path):
                continue
            if is_module(path):
                add_module(path)
    else:
        to_search = [search_path]
        while to_search:
            path = to_search.pop()
            if is_module(path):
                add_module(path)
            for child in get_files(path):
                if not os.path.isdir(child):
                    continue
                if is_engine(child):
                    continue
                to_search.insert(0, child)
    return modules


def is_module(path):
    if not os.path.isdir(path):
        return False
    must_exist = ["register_types.h", "SCsub", "config.py"]
    for f in must_exist:
        if not os.path.exists(os.path.join(path, f)):
            return False
    return True


def write_modules(modules):
    includes_cpp = ""
    preregister_cpp = ""
    register_cpp = ""
    unregister_cpp = ""

    for name, path in modules.items():
        try:
            with open(os.path.join(path, "register_types.h")):
                includes_cpp += '#include "' + path + '/register_types.h"\n'
                preregister_cpp += "#ifdef MODULE_" + name.upper() + "_ENABLED\n"
                preregister_cpp += "#ifdef MODULE_" + name.upper() + "_HAS_PREREGISTER\n"
                preregister_cpp += "\tpreregister_" + name + "_types();\n"
                preregister_cpp += "#endif\n"
                preregister_cpp += "#endif\n"
                register_cpp += "#ifdef MODULE_" + name.upper() + "_ENABLED\n"
                register_cpp += "\tregister_" + name + "_types();\n"
                register_cpp += "#endif\n"
                unregister_cpp += "#ifdef MODULE_" + name.upper() + "_ENABLED\n"
                unregister_cpp += "\tunregister_" + name + "_types();\n"
                unregister_cpp += "#endif\n"
        except OSError:
            pass

    modules_cpp = """// register_module_types.gen.cpp
/* THIS FILE IS GENERATED DO NOT EDIT */
#include "register_module_types.h"

#include "modules/modules_enabled.gen.h"

%s

void preregister_module_types() {
%s
}

void register_module_types() {
%s
}

void unregister_module_types() {
%s
}
""" % (
        includes_cpp,
        preregister_cpp,
        register_cpp,
        unregister_cpp,
    )

    # NOTE: It is safe to generate this file here, since this is still executed serially
    with open("modules/register_module_types.gen.cpp", "w") as f:
        f.write(modules_cpp)


def convert_custom_modules_path(path):
    if not path:
        return path
    path = os.path.realpath(os.path.expanduser(os.path.expandvars(path)))
    err_msg = "Build option 'custom_modules' must %s"
    if not os.path.isdir(path):
        raise ValueError(err_msg % "point to an existing directory.")
    if path == os.path.realpath("modules"):
        raise ValueError(err_msg % "be a directory other than built-in `modules` directory.")
    return path


def disable_module(self):
    self.disabled_modules.append(self.current_module)


def module_check_dependencies(self, module, dependencies):
    """
    Checks if module dependencies are enabled for a given module,
    and prints a warning if they aren't.
    Meant to be used in module `can_build` methods.
    Returns a boolean (True if dependencies are satisfied).
    """
    missing_deps = []
    for dep in dependencies:
        opt = "module_{}_enabled".format(dep)
        if not opt in self or not self[opt]:
            missing_deps.append(dep)

    if missing_deps != []:
        print(
            "Disabling '{}' module as the following dependencies are not satisfied: {}".format(
                module, ", ".join(missing_deps)
            )
        )
        return False
    else:
        return True


def use_windows_spawn_fix(self, platform=None):

    if os.name != "nt":
        return  # not needed, only for windows

    # On Windows, due to the limited command line length, when creating a static library
    # from a very high number of objects SCons will invoke "ar" once per object file;
    # that makes object files with same names to be overwritten so the last wins and
    # the library loses symbols defined by overwritten objects.
    # By enabling quick append instead of the default mode (replacing), libraries will
    # got built correctly regardless the invocation strategy.
    # Furthermore, since SCons will rebuild the library from scratch when an object file
    # changes, no multiple versions of the same object file will be present.
    self.Replace(ARFLAGS="q")

    def mySubProcess(cmdline, env):

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        proc = subprocess.Popen(
            cmdline,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            shell=False,
            env=env,
        )
        _, err = proc.communicate()
        rv = proc.wait()
        if rv:
            print("=====")
            print(err)
            print("=====")
        return rv

    def mySpawn(sh, escape, cmd, args, env):

        newargs = " ".join(args[1:])
        cmdline = cmd + " " + newargs

        rv = 0
        env = {str(key): str(value) for key, value in iter(env.items())}
        if len(cmdline) > 32000 and cmd.endswith("ar"):
            cmdline = cmd + " " + args[1] + " " + args[2] + " "
            for i in range(3, len(args)):
                rv = mySubProcess(cmdline + args[i], env)
                if rv:
                    break
        else:
            rv = mySubProcess(cmdline, env)

        return rv

    self["SPAWN"] = mySpawn


def save_active_platforms(apnames, ap):

    for x in ap:
        names = ["logo"]
        if os.path.isfile(x + "/run_icon.png"):
            names.append("run_icon")

        for name in names:
            pngf = open(x + "/" + name + ".png", "rb")
            b = pngf.read(1)
            str = " /* AUTOGENERATED FILE, DO NOT EDIT */ \n"
            str += " static const unsigned char _" + x[9:] + "_" + name + "[]={"
            while len(b) == 1:
                str += hex(ord(b))
                b = pngf.read(1)
                if len(b) == 1:
                    str += ","

            str += "};\n"

            pngf.close()

            # NOTE: It is safe to generate this file here, since this is still executed serially
            wf = x + "/" + name + ".gen.h"
            with open(wf, "w") as pngw:
                pngw.write(str)


def no_verbose(sys, env):

    colors = {}

    # Colors are disabled in non-TTY environments such as pipes. This means
    # that if output is redirected to a file, it will not contain color codes
    if sys.stdout.isatty():
        colors["cyan"] = "\033[96m"
        colors["purple"] = "\033[95m"
        colors["blue"] = "\033[94m"
        colors["green"] = "\033[92m"
        colors["yellow"] = "\033[93m"
        colors["red"] = "\033[91m"
        colors["end"] = "\033[0m"
    else:
        colors["cyan"] = ""
        colors["purple"] = ""
        colors["blue"] = ""
        colors["green"] = ""
        colors["yellow"] = ""
        colors["red"] = ""
        colors["end"] = ""

    compile_source_message = "{}Compiling {}==> {}$SOURCE{}".format(
        colors["blue"], colors["purple"], colors["yellow"], colors["end"]
    )
    java_compile_source_message = "{}Compiling {}==> {}$SOURCE{}".format(
        colors["blue"], colors["purple"], colors["yellow"], colors["end"]
    )
    compile_shared_source_message = "{}Compiling shared {}==> {}$SOURCE{}".format(
        colors["blue"], colors["purple"], colors["yellow"], colors["end"]
    )
    link_program_message = "{}Linking Program        {}==> {}$TARGET{}".format(
        colors["red"], colors["purple"], colors["yellow"], colors["end"]
    )
    link_library_message = "{}Linking Static Library {}==> {}$TARGET{}".format(
        colors["red"], colors["purple"], colors["yellow"], colors["end"]
    )
    ranlib_library_message = "{}Ranlib Library         {}==> {}$TARGET{}".format(
        colors["red"], colors["purple"], colors["yellow"], colors["end"]
    )
    link_shared_library_message = "{}Linking Shared Library {}==> {}$TARGET{}".format(
        colors["red"], colors["purple"], colors["yellow"], colors["end"]
    )
    java_library_message = "{}Creating Java Archive  {}==> {}$TARGET{}".format(
        colors["red"], colors["purple"], colors["yellow"], colors["end"]
    )

    env.Append(CXXCOMSTR=[compile_source_message])
    env.Append(CCCOMSTR=[compile_source_message])
    env.Append(SHCCCOMSTR=[compile_shared_source_message])
    env.Append(SHCXXCOMSTR=[compile_shared_source_message])
    env.Append(ARCOMSTR=[link_library_message])
    env.Append(RANLIBCOMSTR=[ranlib_library_message])
    env.Append(SHLINKCOMSTR=[link_shared_library_message])
    env.Append(LINKCOMSTR=[link_program_message])
    env.Append(JARCOMSTR=[java_library_message])
    env.Append(JAVACCOMSTR=[java_compile_source_message])


def detect_visual_c_compiler_version(tools_env):
    # tools_env is the variable scons uses to call tools that execute tasks, SCons's env['ENV'] that executes tasks...
    # (see the SCons documentation for more information on what it does)...
    # in order for this function to be well encapsulated i choose to force it to receive SCons's TOOLS env (env['ENV']
    # and not scons setup environment (env)... so make sure you call the right environment on it or it will fail to detect
    # the proper vc version that will be called

    # There is no flag to give to visual c compilers to set the architecture, i.e. scons bits argument (32,64,ARM etc)
    # There are many different cl.exe files that are run, and each one compiles & links to a different architecture
    # As far as I know, the only way to figure out what compiler will be run when Scons calls cl.exe via Program()
    # is to check the PATH variable and figure out which one will be called first. Code below does that and returns:
    # the following string values:

    # ""              Compiler not detected
    # "amd64"         Native 64 bit compiler
    # "amd64_x86"     64 bit Cross Compiler for 32 bit
    # "x86"           Native 32 bit compiler
    # "x86_amd64"     32 bit Cross Compiler for 64 bit

    # There are other architectures, but Godot does not support them currently, so this function does not detect arm/amd64_arm
    # and similar architectures/compilers

    # Set chosen compiler to "not detected"
    vc_chosen_compiler_index = -1
    vc_chosen_compiler_str = ""

    # Start with Pre VS 2017 checks which uses VCINSTALLDIR:
    if "VCINSTALLDIR" in tools_env:
        # print("Checking VCINSTALLDIR")

        # find() works with -1 so big ifs below are needed... the simplest solution, in fact
        # First test if amd64 and amd64_x86 compilers are present in the path
        vc_amd64_compiler_detection_index = tools_env["PATH"].find(tools_env["VCINSTALLDIR"] + "BIN\\amd64;")
        if vc_amd64_compiler_detection_index > -1:
            vc_chosen_compiler_index = vc_amd64_compiler_detection_index
            vc_chosen_compiler_str = "amd64"

        vc_amd64_x86_compiler_detection_index = tools_env["PATH"].find(tools_env["VCINSTALLDIR"] + "BIN\\amd64_x86;")
        if vc_amd64_x86_compiler_detection_index > -1 and (
            vc_chosen_compiler_index == -1 or vc_chosen_compiler_index > vc_amd64_x86_compiler_detection_index
        ):
            vc_chosen_compiler_index = vc_amd64_x86_compiler_detection_index
            vc_chosen_compiler_str = "amd64_x86"

        # Now check the 32 bit compilers
        vc_x86_compiler_detection_index = tools_env["PATH"].find(tools_env["VCINSTALLDIR"] + "BIN;")
        if vc_x86_compiler_detection_index > -1 and (
            vc_chosen_compiler_index == -1 or vc_chosen_compiler_index > vc_x86_compiler_detection_index
        ):
            vc_chosen_compiler_index = vc_x86_compiler_detection_index
            vc_chosen_compiler_str = "x86"

        vc_x86_amd64_compiler_detection_index = tools_env["PATH"].find(tools_env["VCINSTALLDIR"] + "BIN\\x86_amd64;")
        if vc_x86_amd64_compiler_detection_index > -1 and (
            vc_chosen_compiler_index == -1 or vc_chosen_compiler_index > vc_x86_amd64_compiler_detection_index
        ):
            vc_chosen_compiler_index = vc_x86_amd64_compiler_detection_index
            vc_chosen_compiler_str = "x86_amd64"

    # and for VS 2017 and newer we check VCTOOLSINSTALLDIR:
    if "VCTOOLSINSTALLDIR" in tools_env:

        # Newer versions have a different path available
        vc_amd64_compiler_detection_index = (
            tools_env["PATH"].upper().find(tools_env["VCTOOLSINSTALLDIR"].upper() + "BIN\\HOSTX64\\X64;")
        )
        if vc_amd64_compiler_detection_index > -1:
            vc_chosen_compiler_index = vc_amd64_compiler_detection_index
            vc_chosen_compiler_str = "amd64"

        vc_amd64_x86_compiler_detection_index = (
            tools_env["PATH"].upper().find(tools_env["VCTOOLSINSTALLDIR"].upper() + "BIN\\HOSTX64\\X86;")
        )
        if vc_amd64_x86_compiler_detection_index > -1 and (
            vc_chosen_compiler_index == -1 or vc_chosen_compiler_index > vc_amd64_x86_compiler_detection_index
        ):
            vc_chosen_compiler_index = vc_amd64_x86_compiler_detection_index
            vc_chosen_compiler_str = "amd64_x86"

        vc_x86_compiler_detection_index = (
            tools_env["PATH"].upper().find(tools_env["VCTOOLSINSTALLDIR"].upper() + "BIN\\HOSTX86\\X86;")
        )
        if vc_x86_compiler_detection_index > -1 and (
            vc_chosen_compiler_index == -1 or vc_chosen_compiler_index > vc_x86_compiler_detection_index
        ):
            vc_chosen_compiler_index = vc_x86_compiler_detection_index
            vc_chosen_compiler_str = "x86"

        vc_x86_amd64_compiler_detection_index = (
            tools_env["PATH"].upper().find(tools_env["VCTOOLSINSTALLDIR"].upper() + "BIN\\HOSTX86\\X64;")
        )
        if vc_x86_amd64_compiler_detection_index > -1 and (
            vc_chosen_compiler_index == -1 or vc_chosen_compiler_index > vc_x86_amd64_compiler_detection_index
        ):
            vc_chosen_compiler_index = vc_x86_amd64_compiler_detection_index
            vc_chosen_compiler_str = "x86_amd64"

    return vc_chosen_compiler_str


def find_visual_c_batch_file(env):
    from SCons.Tool.MSCommon.vc import get_default_version, get_host_target, find_batch_file

    version = get_default_version(env)
    (host_platform, target_platform, _) = get_host_target(env)
    return find_batch_file(env, version, host_platform, target_platform)[0]


def generate_cpp_hint_file(filename):
    if os.path.isfile(filename):
        # Don't overwrite an existing hint file since the user may have customized it.
        pass
    else:
        try:
            with open(filename, "w") as fd:
                fd.write("#define GDCLASS(m_class, m_inherits)\n")
        except OSError:
            print("Could not write cpp.hint file.")


def glob_recursive(pattern, node="."):
    results = []
    for f in Glob(str(node) + "/*", source=True):
        if type(f) is Node.FS.Dir:
            results += glob_recursive(pattern, f)
    results += Glob(str(node) + "/" + pattern, source=True)
    return results


def add_to_vs_project(env, sources):
    for x in sources:
        if type(x) == type(""):
            fname = env.File(x).path
        else:
            fname = env.File(x)[0].path
        pieces = fname.split(".")
        if len(pieces) > 0:
            basename = pieces[0]
            basename = basename.replace("\\\\", "/")
            if os.path.isfile(basename + ".h"):
                env.vs_incs += [basename + ".h"]
            elif os.path.isfile(basename + ".hpp"):
                env.vs_incs += [basename + ".hpp"]
            if os.path.isfile(basename + ".c"):
                env.vs_srcs += [basename + ".c"]
            elif os.path.isfile(basename + ".cpp"):
                env.vs_srcs += [basename + ".cpp"]


def generate_vs_project(env, num_jobs):
    batch_file = find_visual_c_batch_file(env)
    if batch_file:

        def build_commandline(commands):
            common_build_prefix = [
                'cmd /V /C set "plat=$(PlatformTarget)"',
                '(if "$(PlatformTarget)"=="x64" (set "plat=x86_amd64"))',
                'set "tools=%s"' % env["tools"],
                '(if "$(Configuration)"=="release" (set "tools=no"))',
                'call "' + batch_file + '" !plat!',
            ]

            # Windows allows us to have spaces in paths, so we need
            # to double quote off the directory. However, the path ends
            # in a backslash, so we need to remove this, lest it escape the
            # last double quote off, confusing MSBuild
            common_build_postfix = [
                "--directory=\"$(ProjectDir.TrimEnd('\\'))\"",
                "platform=windows",
                "target=$(Configuration)",
                "progress=no",
                "tools=!tools!",
                "-j%s" % num_jobs,
            ]

            if env["tests"]:
                common_build_postfix.append("tests=yes")

            if env["custom_modules"]:
                common_build_postfix.append("custom_modules=%s" % env["custom_modules"])

            result = " ^& ".join(common_build_prefix + [" ".join([commands] + common_build_postfix)])
            return result

        add_to_vs_project(env, env.core_sources)
        add_to_vs_project(env, env.drivers_sources)
        add_to_vs_project(env, env.main_sources)
        add_to_vs_project(env, env.modules_sources)
        add_to_vs_project(env, env.scene_sources)
        add_to_vs_project(env, env.servers_sources)
        if env["tests"]:
            add_to_vs_project(env, env.tests_sources)
        add_to_vs_project(env, env.editor_sources)

        for header in glob_recursive("**/*.h"):
            env.vs_incs.append(str(header))

        env["MSVSBUILDCOM"] = build_commandline("scons")
        env["MSVSREBUILDCOM"] = build_commandline("scons vsproj=yes")
        env["MSVSCLEANCOM"] = build_commandline("scons --clean")

        # This version information (Win32, x64, Debug, Release, Release_Debug seems to be
        # required for Visual Studio to understand that it needs to generate an NMAKE
        # project. Do not modify without knowing what you are doing.
        debug_variants = ["debug|Win32"] + ["debug|x64"]
        release_variants = ["release|Win32"] + ["release|x64"]
        release_debug_variants = ["release_debug|Win32"] + ["release_debug|x64"]
        variants = debug_variants + release_variants + release_debug_variants
        debug_targets = ["bin\\godot.windows.tools.32.exe"] + ["bin\\godot.windows.tools.64.exe"]
        release_targets = ["bin\\godot.windows.opt.32.exe"] + ["bin\\godot.windows.opt.64.exe"]
        release_debug_targets = ["bin\\godot.windows.opt.tools.32.exe"] + ["bin\\godot.windows.opt.tools.64.exe"]
        targets = debug_targets + release_targets + release_debug_targets
        if not env.get("MSVS"):
            env["MSVS"]["PROJECTSUFFIX"] = ".vcxproj"
            env["MSVS"]["SOLUTIONSUFFIX"] = ".sln"
        env.MSVSProject(
            target=["#godot" + env["MSVSPROJECTSUFFIX"]],
            incs=env.vs_incs,
            srcs=env.vs_srcs,
            runfile=targets,
            buildtarget=targets,
            auto_build_solution=1,
            variant=variants,
        )
    else:
        print("Could not locate Visual Studio batch file to set up the build environment. Not generating VS project.")


def precious_program(env, program, sources, **args):
    program = env.ProgramOriginal(program, sources, **args)
    env.Precious(program)
    return program


def add_shared_library(env, name, sources, **args):
    library = env.SharedLibrary(name, sources, **args)
    env.NoCache(library)
    return library


def add_library(env, name, sources, **args):
    library = env.Library(name, sources, **args)
    env.NoCache(library)
    return library


def add_program(env, name, sources, **args):
    program = env.Program(name, sources, **args)
    env.NoCache(program)
    return program


def CommandNoCache(env, target, sources, command, **args):
    result = env.Command(target, sources, command, **args)
    env.NoCache(result)
    return result


def Run(env, function, short_message, subprocess=True):
    output_print = short_message if not env["verbose"] else ""
    if not subprocess:
        return Action(function, output_print)
    else:
        return Action(run_in_subprocess(function), output_print)


def detect_darwin_sdk_path(platform, env):
    sdk_name = ""
    if platform == "osx":
        sdk_name = "macosx"
        var_name = "MACOS_SDK_PATH"
    elif platform == "iphone":
        sdk_name = "iphoneos"
        var_name = "IPHONESDK"
    elif platform == "iphonesimulator":
        sdk_name = "iphonesimulator"
        var_name = "IPHONESDK"
    else:
        raise Exception("Invalid platform argument passed to detect_darwin_sdk_path")

    if not env[var_name]:
        try:
            sdk_path = subprocess.check_output(["xcrun", "--sdk", sdk_name, "--show-sdk-path"]).strip().decode("utf-8")
            if sdk_path:
                env[var_name] = sdk_path
        except (subprocess.CalledProcessError, OSError):
            print("Failed to find SDK path while running xcrun --sdk {} --show-sdk-path.".format(sdk_name))
            raise


def is_vanilla_clang(env):
    if not using_clang(env):
        return False
    try:
        version = subprocess.check_output([env.subst(env["CXX"]), "--version"]).strip().decode("utf-8")
    except (subprocess.CalledProcessError, OSError):
        print("Couldn't parse CXX environment variable to infer compiler version.")
        return False
    return not version.startswith("Apple")


def get_compiler_version(env):
    """
    Returns an array of version numbers as ints: [major, minor, patch].
    The return array should have at least two values (major, minor).
    """
    if not env.msvc:
        # Not using -dumpversion as some GCC distros only return major, and
        # Clang used to return hardcoded 4.2.1: # https://reviews.llvm.org/D56803
        try:
            version = subprocess.check_output([env.subst(env["CXX"]), "--version"]).strip().decode("utf-8")
        except (subprocess.CalledProcessError, OSError):
            print("Couldn't parse CXX environment variable to infer compiler version.")
            return None
    else:  # TODO: Implement for MSVC
        return None
    match = re.search("[0-9]+\.[0-9.]+", version)
    if match is not None:
        return list(map(int, match.group().split(".")))
    else:
        return None


def using_gcc(env):
    return "gcc" in os.path.basename(env["CC"])


def using_clang(env):
    return "clang" in os.path.basename(env["CC"])


def show_progress(env):
    import sys
    from SCons.Script import Progress, Command, AlwaysBuild

    screen = sys.stdout
    # Progress reporting is not available in non-TTY environments since it
    # messes with the output (for example, when writing to a file)
    show_progress = env["progress"] and sys.stdout.isatty()
    node_count = 0
    node_count_max = 0
    node_count_interval = 1
    node_count_fname = str(env.Dir("#")) + "/.scons_node_count"

    import time, math

    class cache_progress:
        # The default is 1 GB cache and 12 hours half life
        def __init__(self, path=None, limit=1073741824, half_life=43200):
            self.path = path
            self.limit = limit
            self.exponent_scale = math.log(2) / half_life
            if env["verbose"] and path != None:
                screen.write(
                    "Current cache limit is {} (used: {})\n".format(
                        self.convert_size(limit), self.convert_size(self.get_size(path))
                    )
                )
            self.delete(self.file_list())

        def __call__(self, node, *args, **kw):
            nonlocal node_count, node_count_max, node_count_interval, node_count_fname, show_progress
            if show_progress:
                # Print the progress percentage
                node_count += node_count_interval
                if node_count_max > 0 and node_count <= node_count_max:
                    screen.write("\r[%3d%%] " % (node_count * 100 / node_count_max))
                    screen.flush()
                elif node_count_max > 0 and node_count > node_count_max:
                    screen.write("\r[100%] ")
                    screen.flush()
                else:
                    screen.write("\r[Initial build] ")
                    screen.flush()

        def delete(self, files):
            if len(files) == 0:
                return
            if env["verbose"]:
                # Utter something
                screen.write("\rPurging %d %s from cache...\n" % (len(files), len(files) > 1 and "files" or "file"))
            [os.remove(f) for f in files]

        def file_list(self):
            if self.path is None:
                # Nothing to do
                return []
            # Gather a list of (filename, (size, atime)) within the
            # cache directory
            file_stat = [(x, os.stat(x)[6:8]) for x in glob.glob(os.path.join(self.path, "*", "*"))]
            if file_stat == []:
                # Nothing to do
                return []
            # Weight the cache files by size (assumed to be roughly
            # proportional to the recompilation time) times an exponential
            # decay since the ctime, and return a list with the entries
            # (filename, size, weight).
            current_time = time.time()
            file_stat = [(x[0], x[1][0], (current_time - x[1][1])) for x in file_stat]
            # Sort by the most recently accessed files (most sensible to keep) first
            file_stat.sort(key=lambda x: x[2])
            # Search for the first entry where the storage limit is
            # reached
            sum, mark = 0, None
            for i, x in enumerate(file_stat):
                sum += x[1]
                if sum > self.limit:
                    mark = i
                    break
            if mark is None:
                return []
            else:
                return [x[0] for x in file_stat[mark:]]

        def convert_size(self, size_bytes):
            if size_bytes == 0:
                return "0 bytes"
            size_name = ("bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return "%s %s" % (int(s) if i == 0 else s, size_name[i])

        def get_size(self, start_path="."):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(start_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            return total_size

    def progress_finish(target, source, env):
        nonlocal node_count, progressor
        with open(node_count_fname, "w") as f:
            f.write("%d\n" % node_count)
        progressor.delete(progressor.file_list())

    try:
        with open(node_count_fname) as f:
            node_count_max = int(f.readline())
    except Exception:
        pass

    cache_directory = os.environ.get("SCONS_CACHE")
    # Simple cache pruning, attached to SCons' progress callback. Trim the
    # cache directory to a size not larger than cache_limit.
    cache_limit = float(os.getenv("SCONS_CACHE_LIMIT", 1024)) * 1024 * 1024
    progressor = cache_progress(cache_directory, cache_limit)
    Progress(progressor, interval=node_count_interval)

    progress_finish_command = Command("progress_finish", [], progress_finish)
    AlwaysBuild(progress_finish_command)


def dump(env):
    # Dumps latest build information for debugging purposes and external tools.
    from json import dump

    def non_serializable(obj):
        return "<<non-serializable: %s>>" % (type(obj).__qualname__)

    with open(".scons_env.json", "w") as f:
        dump(env.Dictionary(), f, indent=4, default=non_serializable)
