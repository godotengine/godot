import os
import os.path
import sys
import re
import glob
import string
import subprocess
from compat import iteritems, isbasestring, decode_utf8


def add_source_files(self, sources, filetype, lib_env=None, shared=False):

    if isbasestring(filetype):
        dir_path = self.Dir('.').abspath
        filetype = sorted(glob.glob(dir_path + "/" + filetype))

    for path in filetype:
        sources.append(self.Object(path))


def disable_warnings(self):
    # 'self' is the environment
    if self.msvc:
        # We have to remove existing warning level defines before appending /w,
        # otherwise we get: "warning D9025 : overriding '/W3' with '/w'"
        warn_flags = ['/Wall', '/W4', '/W3', '/W2', '/W1', '/WX']
        self.Append(CCFLAGS=['/w'])
        self.Append(CFLAGS=['/w'])
        self.Append(CXXFLAGS=['/w'])
        self['CCFLAGS'] = [x for x in self['CCFLAGS'] if not x in warn_flags]
        self['CFLAGS'] = [x for x in self['CFLAGS'] if not x in warn_flags]
        self['CXXFLAGS'] = [x for x in self['CXXFLAGS'] if not x in warn_flags]
    else:
        self.Append(CCFLAGS=['-w'])
        self.Append(CFLAGS=['-w'])
        self.Append(CXXFLAGS=['-w'])


def add_module_version_string(self,s):
    self.module_version_string += "." + s


def update_version(module_version_string=""):

    build_name = "custom_build"
    if os.getenv("BUILD_NAME") != None:
        build_name = os.getenv("BUILD_NAME")
        print("Using custom build name: " + build_name)

    import version

    # NOTE: It is safe to generate this file here, since this is still executed serially
    f = open("core/version_generated.gen.h", "w")
    f.write("#define VERSION_SHORT_NAME \"" + str(version.short_name) + "\"\n")
    f.write("#define VERSION_NAME \"" + str(version.name) + "\"\n")
    f.write("#define VERSION_MAJOR " + str(version.major) + "\n")
    f.write("#define VERSION_MINOR " + str(version.minor) + "\n")
    if hasattr(version, 'patch'):
        f.write("#define VERSION_PATCH " + str(version.patch) + "\n")
    f.write("#define VERSION_STATUS \"" + str(version.status) + "\"\n")
    f.write("#define VERSION_BUILD \"" + str(build_name) + "\"\n")
    f.write("#define VERSION_MODULE_CONFIG \"" + str(version.module_config) + module_version_string + "\"\n")
    f.write("#define VERSION_YEAR " + str(version.year) + "\n")
    f.write("#define VERSION_WEBSITE \"" + str(version.website) + "\"\n")
    f.close()

    # NOTE: It is safe to generate this file here, since this is still executed serially
    fhash = open("core/version_hash.gen.h", "w")
    githash = ""
    gitfolder = ".git"

    if os.path.isfile(".git"):
        module_folder = open(".git", "r").readline().strip()
        if module_folder.startswith("gitdir: "):
            gitfolder = module_folder[8:]

    if os.path.isfile(os.path.join(gitfolder, "HEAD")):
        head = open(os.path.join(gitfolder, "HEAD"), "r").readline().strip()
        if head.startswith("ref: "):
            head = os.path.join(gitfolder, head[5:])
            if os.path.isfile(head):
                githash = open(head, "r").readline().strip()
        else:
            githash = head

    fhash.write("#define VERSION_HASH \"" + githash + "\"")
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


def detect_modules():

    module_list = []
    includes_cpp = ""
    register_cpp = ""
    unregister_cpp = ""

    files = glob.glob("modules/*")
    files.sort()  # so register_module_types does not change that often, and also plugins are registered in alphabetic order
    for x in files:
        if not os.path.isdir(x):
            continue
        if not os.path.exists(x + "/config.py"):
            continue
        x = x.replace("modules/", "")  # rest of world
        x = x.replace("modules\\", "")  # win32
        module_list.append(x)
        try:
            with open("modules/" + x + "/register_types.h"):
                includes_cpp += '#include "modules/' + x + '/register_types.h"\n'
                register_cpp += '#ifdef MODULE_' + x.upper() + '_ENABLED\n'
                register_cpp += '\tregister_' + x + '_types();\n'
                register_cpp += '#endif\n'
                unregister_cpp += '#ifdef MODULE_' + x.upper() + '_ENABLED\n'
                unregister_cpp += '\tunregister_' + x + '_types();\n'
                unregister_cpp += '#endif\n'
        except IOError:
            pass

    modules_cpp = """
// modules.cpp - THIS FILE IS GENERATED, DO NOT EDIT!!!!!!!
#include "register_module_types.h"

""" + includes_cpp + """

void register_module_types() {
""" + register_cpp + """
}

void unregister_module_types() {
""" + unregister_cpp + """
}
"""

    # NOTE: It is safe to generate this file here, since this is still executed serially
    with open("modules/register_module_types.gen.cpp", "w") as f:
        f.write(modules_cpp)

    return module_list


def win32_spawn(sh, escape, cmd, args, env):
    import subprocess
    newargs = ' '.join(args[1:])
    cmdline = cmd + " " + newargs
    startupinfo = subprocess.STARTUPINFO()
    for e in env:
        if type(env[e]) != type(""):
            env[e] = str(env[e])
    proc = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, startupinfo=startupinfo, shell=False, env=env)
    _, err = proc.communicate()
    rv = proc.wait()
    if rv:
        print("=====")
        print(err)
        print("=====")
    return rv

"""
def win32_spawn(sh, escape, cmd, args, spawnenv):
	import win32file
	import win32event
	import win32process
	import win32security
	for var in spawnenv:
		spawnenv[var] = spawnenv[var].encode('ascii', 'replace')

	sAttrs = win32security.SECURITY_ATTRIBUTES()
	StartupInfo = win32process.STARTUPINFO()
	newargs = ' '.join(map(escape, args[1:]))
	cmdline = cmd + " " + newargs

	# check for any special operating system commands
	if cmd == 'del':
		for arg in args[1:]:
			win32file.DeleteFile(arg)
		exit_code = 0
	else:
		# otherwise execute the command.
		hProcess, hThread, dwPid, dwTid = win32process.CreateProcess(None, cmdline, None, None, 1, 0, spawnenv, None, StartupInfo)
		win32event.WaitForSingleObject(hProcess, win32event.INFINITE)
		exit_code = win32process.GetExitCodeProcess(hProcess)
		win32file.CloseHandle(hProcess);
		win32file.CloseHandle(hThread);
	return exit_code
"""

def disable_module(self):
    self.disabled_modules.append(self.current_module)

def use_windows_spawn_fix(self, platform=None):

    if (os.name != "nt"):
        return  # not needed, only for windows

    # On Windows, due to the limited command line length, when creating a static library
    # from a very high number of objects SCons will invoke "ar" once per object file;
    # that makes object files with same names to be overwritten so the last wins and
    # the library looses symbols defined by overwritten objects.
    # By enabling quick append instead of the default mode (replacing), libraries will
    # got built correctly regardless the invocation strategy.
    # Furthermore, since SCons will rebuild the library from scratch when an object file
    # changes, no multiple versions of the same object file will be present.
    self.Replace(ARFLAGS='q')

    def mySubProcess(cmdline, env):

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        proc = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, startupinfo=startupinfo, shell=False, env=env)
        _, err = proc.communicate()
        rv = proc.wait()
        if rv:
            print("=====")
            print(err)
            print("=====")
        return rv

    def mySpawn(sh, escape, cmd, args, env):

        newargs = ' '.join(args[1:])
        cmdline = cmd + " " + newargs

        rv = 0
        env = {str(key): str(value) for key, value in iteritems(env)}
        if len(cmdline) > 32000 and cmd.endswith("ar"):
            cmdline = cmd + " " + args[1] + " " + args[2] + " "
            for i in range(3, len(args)):
                rv = mySubProcess(cmdline + args[i], env)
                if rv:
                    break
        else:
            rv = mySubProcess(cmdline, env)

        return rv

    self['SPAWN'] = mySpawn


def split_lib(self, libname, src_list = None, env_lib = None):
    env = self

    num = 0
    cur_base = ""
    max_src = 64
    list = []
    lib_list = []

    if src_list is None:
        src_list = getattr(env, libname + "_sources")

    if type(env_lib) == type(None):
        env_lib = env

    for f in src_list:
        fname = ""
        if type(f) == type(""):
            fname = env.File(f).path
        else:
            fname = env.File(f)[0].path
        fname = fname.replace("\\", "/")
        base = string.join(fname.split("/")[:2], "/")
        if base != cur_base and len(list) > max_src:
            if num > 0:
                lib = env_lib.add_library(libname + str(num), list)
                lib_list.append(lib)
                list = []
            num = num + 1
        cur_base = base
        list.append(f)

    lib = env_lib.add_library(libname + str(num), list)
    lib_list.append(lib)

    if len(lib_list) > 0:
        if os.name == 'posix' and sys.platform == 'msys':
            env.Replace(ARFLAGS=['rcsT'])
            lib = env_lib.add_library(libname + "_collated", lib_list)
            lib_list = [lib]

    lib_base = []
    env_lib.add_source_files(lib_base, "*.cpp")
    lib = env_lib.add_library(libname, lib_base)
    lib_list.insert(0, lib)

    env.Prepend(LIBS=lib_list)


def save_active_platforms(apnames, ap):

    for x in ap:
        names = ['logo']
        if os.path.isfile(x + "/run_icon.png"):
            names.append('run_icon')

        for name in names:
            pngf = open(x + "/" + name + ".png", "rb")
            b = pngf.read(1)
            str = " /* AUTOGENERATED FILE, DO NOT EDIT */ \n"
            str += " static const unsigned char _" + x[9:] + "_" + name + "[]={"
            while len(b) == 1:
                str += hex(ord(b))
                b = pngf.read(1)
                if (len(b) == 1):
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
        colors['cyan'] = '\033[96m'
        colors['purple'] = '\033[95m'
        colors['blue'] = '\033[94m'
        colors['green'] = '\033[92m'
        colors['yellow'] = '\033[93m'
        colors['red'] = '\033[91m'
        colors['end'] = '\033[0m'
    else:
        colors['cyan'] = ''
        colors['purple'] = ''
        colors['blue'] = ''
        colors['green'] = ''
        colors['yellow'] = ''
        colors['red'] = ''
        colors['end'] = ''

    compile_source_message = '%sCompiling %s==> %s$SOURCE%s' % (colors['blue'], colors['purple'], colors['yellow'], colors['end'])
    java_compile_source_message = '%sCompiling %s==> %s$SOURCE%s' % (colors['blue'], colors['purple'], colors['yellow'], colors['end'])
    compile_shared_source_message = '%sCompiling shared %s==> %s$SOURCE%s' % (colors['blue'], colors['purple'], colors['yellow'], colors['end'])
    link_program_message = '%sLinking Program        %s==> %s$TARGET%s' % (colors['red'], colors['purple'], colors['yellow'], colors['end'])
    link_library_message = '%sLinking Static Library %s==> %s$TARGET%s' % (colors['red'], colors['purple'], colors['yellow'], colors['end'])
    ranlib_library_message = '%sRanlib Library         %s==> %s$TARGET%s' % (colors['red'], colors['purple'], colors['yellow'], colors['end'])
    link_shared_library_message = '%sLinking Shared Library %s==> %s$TARGET%s' % (colors['red'], colors['purple'], colors['yellow'], colors['end'])
    java_library_message = '%sCreating Java Archive  %s==> %s$TARGET%s' % (colors['red'], colors['purple'], colors['yellow'], colors['end'])

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

    # There is no flag to give to visual c compilers to set the architecture, ie scons bits argument (32,64,ARM etc)
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
    if 'VCINSTALLDIR' in tools_env:
        # print("Checking VCINSTALLDIR")

        # find() works with -1 so big ifs below are needed... the simplest solution, in fact
        # First test if amd64 and amd64_x86 compilers are present in the path
        vc_amd64_compiler_detection_index = tools_env["PATH"].find(tools_env["VCINSTALLDIR"] + "BIN\\amd64;")
        if(vc_amd64_compiler_detection_index > -1):
            vc_chosen_compiler_index = vc_amd64_compiler_detection_index
            vc_chosen_compiler_str = "amd64"

        vc_amd64_x86_compiler_detection_index = tools_env["PATH"].find(tools_env["VCINSTALLDIR"] + "BIN\\amd64_x86;")
        if(vc_amd64_x86_compiler_detection_index > -1
           and (vc_chosen_compiler_index == -1
                or vc_chosen_compiler_index > vc_amd64_x86_compiler_detection_index)):
            vc_chosen_compiler_index = vc_amd64_x86_compiler_detection_index
            vc_chosen_compiler_str = "amd64_x86"

        # Now check the 32 bit compilers
        vc_x86_compiler_detection_index = tools_env["PATH"].find(tools_env["VCINSTALLDIR"] + "BIN;")
        if(vc_x86_compiler_detection_index > -1
           and (vc_chosen_compiler_index == -1
                or vc_chosen_compiler_index > vc_x86_compiler_detection_index)):
            vc_chosen_compiler_index = vc_x86_compiler_detection_index
            vc_chosen_compiler_str = "x86"

        vc_x86_amd64_compiler_detection_index = tools_env["PATH"].find(tools_env['VCINSTALLDIR'] + "BIN\\x86_amd64;")
        if(vc_x86_amd64_compiler_detection_index > -1
           and (vc_chosen_compiler_index == -1
                or vc_chosen_compiler_index > vc_x86_amd64_compiler_detection_index)):
            vc_chosen_compiler_index = vc_x86_amd64_compiler_detection_index
            vc_chosen_compiler_str = "x86_amd64"

    # and for VS 2017 and newer we check VCTOOLSINSTALLDIR:
    if 'VCTOOLSINSTALLDIR' in tools_env:

        # Newer versions have a different path available
        vc_amd64_compiler_detection_index = tools_env["PATH"].upper().find(tools_env['VCTOOLSINSTALLDIR'].upper() + "BIN\\HOSTX64\\X64;")
        if(vc_amd64_compiler_detection_index > -1):
            vc_chosen_compiler_index = vc_amd64_compiler_detection_index
            vc_chosen_compiler_str = "amd64"

        vc_amd64_x86_compiler_detection_index = tools_env["PATH"].upper().find(tools_env['VCTOOLSINSTALLDIR'].upper() + "BIN\\HOSTX64\\X86;")
        if(vc_amd64_x86_compiler_detection_index > -1
           and (vc_chosen_compiler_index == -1
                or vc_chosen_compiler_index > vc_amd64_x86_compiler_detection_index)):
            vc_chosen_compiler_index = vc_amd64_x86_compiler_detection_index
            vc_chosen_compiler_str = "amd64_x86"

        vc_x86_compiler_detection_index = tools_env["PATH"].upper().find(tools_env['VCTOOLSINSTALLDIR'].upper() + "BIN\\HOSTX86\\X86;")
        if(vc_x86_compiler_detection_index > -1
           and (vc_chosen_compiler_index == -1
                or vc_chosen_compiler_index > vc_x86_compiler_detection_index)):
            vc_chosen_compiler_index = vc_x86_compiler_detection_index
            vc_chosen_compiler_str = "x86"

        vc_x86_amd64_compiler_detection_index = tools_env["PATH"].upper().find(tools_env['VCTOOLSINSTALLDIR'].upper() + "BIN\\HOSTX86\\X64;")
        if(vc_x86_amd64_compiler_detection_index > -1
           and (vc_chosen_compiler_index == -1
                or vc_chosen_compiler_index > vc_x86_amd64_compiler_detection_index)):
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
        except IOError:
            print("Could not write cpp.hint file.")

def generate_vs_project(env, num_jobs):
    batch_file = find_visual_c_batch_file(env)
    if batch_file:
        def build_commandline(commands):
            common_build_prefix = ['cmd /V /C set "plat=$(PlatformTarget)"',
                                    '(if "$(PlatformTarget)"=="x64" (set "plat=x86_amd64"))',
                                    'set "tools=yes"',
                                    '(if "$(Configuration)"=="release" (set "tools=no"))',
                                    'call "' + batch_file + '" !plat!']

            result = " ^& ".join(common_build_prefix + [commands])
            return result

        env.AddToVSProject(env.core_sources)
        env.AddToVSProject(env.main_sources)
        env.AddToVSProject(env.modules_sources)
        env.AddToVSProject(env.scene_sources)
        env.AddToVSProject(env.servers_sources)
        env.AddToVSProject(env.editor_sources)

        # windows allows us to have spaces in paths, so we need
        # to double quote off the directory. However, the path ends
        # in a backslash, so we need to remove this, lest it escape the
        # last double quote off, confusing MSBuild
        env['MSVSBUILDCOM'] = build_commandline('scons --directory="$(ProjectDir.TrimEnd(\'\\\'))" platform=windows progress=no target=$(Configuration) tools=!tools! -j' + str(num_jobs))
        env['MSVSREBUILDCOM'] = build_commandline('scons --directory="$(ProjectDir.TrimEnd(\'\\\'))" platform=windows progress=no target=$(Configuration) tools=!tools! vsproj=yes -j' + str(num_jobs))
        env['MSVSCLEANCOM'] = build_commandline('scons --directory="$(ProjectDir.TrimEnd(\'\\\'))" --clean platform=windows progress=no target=$(Configuration) tools=!tools! -j' + str(num_jobs))

        # This version information (Win32, x64, Debug, Release, Release_Debug seems to be
        # required for Visual Studio to understand that it needs to generate an NMAKE
        # project. Do not modify without knowing what you are doing.
        debug_variants = ['debug|Win32'] + ['debug|x64']
        release_variants = ['release|Win32'] + ['release|x64']
        release_debug_variants = ['release_debug|Win32'] + ['release_debug|x64']
        variants = debug_variants + release_variants + release_debug_variants
        debug_targets = ['bin\\godot.windows.tools.32.exe'] + ['bin\\godot.windows.tools.64.exe']
        release_targets = ['bin\\godot.windows.opt.32.exe'] + ['bin\\godot.windows.opt.64.exe']
        release_debug_targets = ['bin\\godot.windows.opt.tools.32.exe'] + ['bin\\godot.windows.opt.tools.64.exe']
        targets = debug_targets + release_targets + release_debug_targets
        if not env.get('MSVS'):
            env['MSVS']['PROJECTSUFFIX'] = '.vcxproj'
            env['MSVS']['SOLUTIONSUFFIX'] = '.sln'
        env.MSVSProject(
            target=['#godot' + env['MSVSPROJECTSUFFIX']],
            incs=env.vs_incs,
            srcs=env.vs_srcs,
            runfile=targets,
            buildtarget=targets,
            auto_build_solution=1,
            variant=variants)
    else:
        print("Could not locate Visual Studio batch file for setting up the build environment. Not generating VS project.")

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

def detect_darwin_sdk_path(platform, env):
    sdk_name = ''
    if platform == 'osx':
        sdk_name = 'macosx'
        var_name = 'MACOS_SDK_PATH'
    elif platform == 'iphone':
        sdk_name = 'iphoneos'
        var_name = 'IPHONESDK'
    elif platform == 'iphonesimulator':
        sdk_name = 'iphonesimulator'
        var_name = 'IPHONESDK'
    else:
        raise Exception("Invalid platform argument passed to detect_darwin_sdk_path")

    if not env[var_name]:
        try:
            sdk_path = decode_utf8(subprocess.check_output(['xcrun', '--sdk', sdk_name, '--show-sdk-path']).strip())
            if sdk_path:
                env[var_name] = sdk_path
        except (subprocess.CalledProcessError, OSError):
            print("Failed to find SDK path while running xcrun --sdk {} --show-sdk-path.".format(sdk_name))
            raise

def get_compiler_version(env):
    version = decode_utf8(subprocess.check_output([env['CXX'], '--version']).strip())
    match = re.search('[0-9][0-9.]*', version)
    if match is not None:
        return match.group().split('.')
    else:
        return None

def using_gcc(env):
    return 'gcc' in os.path.basename(env["CC"])

def using_clang(env):
    return 'clang' in os.path.basename(env["CC"])
