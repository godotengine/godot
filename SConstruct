
#!/usr/bin/env python

EnsureSConsVersion(0, 14)


import string
import os
import os.path
import glob
import sys
import methods

methods.update_version()

# scan possible build platforms

platform_list = []  # list of platforms
platform_opts = {}  # options for each platform
platform_flags = {}  # flags for each platform

active_platforms = []
active_platform_ids = []
platform_exporters = []
global_defaults = []

for x in glob.glob("platform/*"):
    if (not os.path.isdir(x) or not os.path.exists(x + "/detect.py")):
        continue
    tmppath = "./" + x

    sys.path.append(tmppath)
    import detect

    if (os.path.exists(x + "/export/export.cpp")):
        platform_exporters.append(x[9:])
    if (os.path.exists(x + "/globals/global_defaults.cpp")):
        global_defaults.append(x[9:])
    if (detect.is_active()):
        active_platforms.append(detect.get_name())
        active_platform_ids.append(x)
    if (detect.can_build()):
        x = x.replace("platform/", "")  # rest of world
        x = x.replace("platform\\", "")  # win32
        platform_list += [x]
        platform_opts[x] = detect.get_opts()
        platform_flags[x] = detect.get_flags()
    sys.path.remove(tmppath)
    sys.modules.pop('detect')

module_list = methods.detect_modules()


# print "Detected Platforms: "+str(platform_list)

methods.save_active_platforms(active_platforms, active_platform_ids)

custom_tools = ['default']

platform_arg = ARGUMENTS.get("platform", ARGUMENTS.get("p", False))

if (os.name == "posix"):
    pass
elif (os.name == "nt"):
    if (os.getenv("VCINSTALLDIR") == None or platform_arg == "android" or platform_arg == "javascript"):
        custom_tools = ['mingw']

env_base = Environment(tools=custom_tools)
if 'TERM' in os.environ:
    env_base['ENV']['TERM'] = os.environ['TERM']
env_base.AppendENVPath('PATH', os.getenv('PATH'))
env_base.AppendENVPath('PKG_CONFIG_PATH', os.getenv('PKG_CONFIG_PATH'))
env_base.global_defaults = global_defaults
env_base.android_maven_repos = []
env_base.android_dependencies = []
env_base.android_gradle_plugins = []
env_base.android_gradle_classpath = []
env_base.android_java_dirs = []
env_base.android_res_dirs = []
env_base.android_aidl_dirs = []
env_base.android_jni_dirs = []
env_base.android_default_config = []
env_base.android_manifest_chunk = ""
env_base.android_permission_chunk = ""
env_base.android_appattributes_chunk = ""
env_base.disabled_modules = []
env_base.use_ptrcall = False
env_base.split_drivers = False

# To decide whether to rebuild a file, use the MD5 sum only if the timestamp has changed.
# http://scons.org/doc/production/HTML/scons-user/ch06.html#idm139837621851792
env_base.Decider('MD5-timestamp')
# Use cached implicit dependencies by default. Can be overridden by specifying `--implicit-deps-changed` in the command line.
# http://scons.org/doc/production/HTML/scons-user/ch06s04.html
env_base.SetOption('implicit_cache', 1)


env_base.__class__.android_add_maven_repository = methods.android_add_maven_repository
env_base.__class__.android_add_dependency = methods.android_add_dependency
env_base.__class__.android_add_java_dir = methods.android_add_java_dir
env_base.__class__.android_add_res_dir = methods.android_add_res_dir
env_base.__class__.android_add_aidl_dir = methods.android_add_aidl_dir
env_base.__class__.android_add_jni_dir = methods.android_add_jni_dir
env_base.__class__.android_add_default_config = methods.android_add_default_config
env_base.__class__.android_add_to_manifest = methods.android_add_to_manifest
env_base.__class__.android_add_to_permissions = methods.android_add_to_permissions
env_base.__class__.android_add_to_attributes = methods.android_add_to_attributes
env_base.__class__.android_add_gradle_plugin = methods.android_add_gradle_plugin
env_base.__class__.android_add_gradle_classpath = methods.android_add_gradle_classpath
env_base.__class__.disable_module = methods.disable_module

env_base.__class__.add_source_files = methods.add_source_files
env_base.__class__.use_windows_spawn_fix = methods.use_windows_spawn_fix
env_base.__class__.split_lib = methods.split_lib

env_base["x86_libtheora_opt_gcc"] = False
env_base["x86_libtheora_opt_vc"] = False

# Build options

customs = ['custom.py']

profile = ARGUMENTS.get("profile", False)
if profile:
    import os.path
    if os.path.isfile(profile):
        customs.append(profile)
    elif os.path.isfile(profile + ".py"):
        customs.append(profile + ".py")

opts = Variables(customs, ARGUMENTS)

# Target build options
opts.Add('arch', "Platform-dependent architecture (arm/arm64/x86/x64/mips/etc)", '')
opts.Add('bits', "Target platform bits (default/32/64/fat)", 'default')
opts.Add('p', "Platform (alias for 'platform')", '')
opts.Add('platform', "Target platform: any in " + str(platform_list), '')
opts.Add('target', "Compilation target (debug/release_debug/release)", 'debug')
opts.Add('tools', "Build the tools a.k.a. the Godot editor (yes/no)", 'yes')

# Components
opts.Add('deprecated', "Enable deprecated features (yes/no)", 'yes')
opts.Add('gdscript', "Build GDSCript support (yes/no)", 'yes')
opts.Add('minizip', "Build minizip archive support (yes/no)", 'yes')
opts.Add('xaudio2', "XAudio2 audio driver (yes/no)", 'no')
opts.Add('xml', "XML format support for resources (yes/no)", 'yes')

# Advanced options
opts.Add('disable_3d', "Disable 3D nodes for smaller executable (yes/no)", 'no')
opts.Add('disable_advanced_gui', "Disable advance 3D gui nodes and behaviors (yes/no)", 'no')
opts.Add('extra_suffix', "Custom extra suffix added to the base filename of all generated binary files", '')
opts.Add('unix_global_settings_path', "UNIX-specific path to system-wide settings. Currently only used for templates", '')
opts.Add('verbose', "Enable verbose output for the compilation (yes/no)", 'no')
opts.Add('vsproj', "Generate Visual Studio Project. (yes/no)", 'no')
opts.Add('warnings', "Set the level of warnings emitted during compilation (extra/all/moderate/no)", 'no')
opts.Add('progress', "Show a progress indicator during build (yes/no)", 'yes')
opts.Add('dev', "If yes, alias for verbose=yes warnings=all (yes/no)", 'no')

# Thirdparty libraries
opts.Add('builtin_enet', "Use the builtin enet library (yes/no)", 'yes')
opts.Add('builtin_freetype', "Use the builtin freetype library (yes/no)", 'yes')
opts.Add('builtin_libogg', "Use the builtin libogg library (yes/no)", 'yes')
opts.Add('builtin_libpng', "Use the builtin libpng library (yes/no)", 'yes')
opts.Add('builtin_libtheora', "Use the builtin libtheora library (yes/no)", 'yes')
opts.Add('builtin_libvorbis', "Use the builtin libvorbis library (yes/no)", 'yes')
opts.Add('builtin_libvpx', "Use the builtin libvpx library (yes/no)", 'yes')
opts.Add('builtin_libwebp', "Use the builtin libwebp library (yes/no)", 'yes')
opts.Add('builtin_openssl', "Use the builtin openssl library (yes/no)", 'yes')
opts.Add('builtin_opus', "Use the builtin opus library (yes/no)", 'yes')
opts.Add('builtin_pcre2', "Use the builtin pcre2 library (yes/no)", 'yes')
opts.Add('builtin_squish', "Use the builtin squish library (yes/no)", 'yes')
opts.Add('builtin_zlib', "Use the builtin zlib library (yes/no)", 'yes')

# Environment setup
opts.Add("CXX", "C++ compiler")
opts.Add("CC", "C compiler")
opts.Add("CCFLAGS", "Custom flags for the C and C++ compilers")
opts.Add("CFLAGS", "Custom flags for the C compiler")
opts.Add("LINKFLAGS", "Custom flags for the linker")


# add platform specific options

for k in platform_opts.keys():
    opt_list = platform_opts[k]
    for o in opt_list:
        opts.Add(o[0], o[1], o[2])

for x in module_list:
    opts.Add('module_' + x + '_enabled', "Enable module '" + x + "' (yes/no)", "yes")

opts.Update(env_base)  # update environment
Help(opts.GenerateHelpText(env_base))  # generate help

# add default include paths

env_base.Append(CPPPATH=['#core', '#core/math', '#editor', '#drivers', '#'])

# configure ENV for platform
env_base.platform_exporters = platform_exporters

"""
sys.path.append("./platform/"+env_base["platform"])
import detect
detect.configure(env_base)
sys.path.remove("./platform/"+env_base["platform"])
sys.modules.pop('detect')
"""

if (env_base['target'] == 'debug'):
    env_base.Append(CPPFLAGS=['-DDEBUG_MEMORY_ALLOC'])
    env_base.Append(CPPFLAGS=['-DSCI_NAMESPACE'])

if (env_base['deprecated'] == 'no'):
    env_base.Append(CPPFLAGS=['-DDISABLE_DEPRECATED'])

env_base.platforms = {}


selected_platform = ""

if env_base['platform'] != "":
    selected_platform = env_base['platform']
elif env_base['p'] != "":
    selected_platform = env_base['p']
    env_base["platform"] = selected_platform


if selected_platform in platform_list:

    sys.path.append("./platform/" + selected_platform)
    import detect
    if "create" in dir(detect):
        env = detect.create(env_base)
    else:
        env = env_base.Clone()
    
    if (env["dev"] == "yes"):
        env["warnings"] = "all"
        env["verbose"] = "yes"

    if env['vsproj'] == "yes":
        env.vs_incs = []
        env.vs_srcs = []

        def AddToVSProject(sources):
            for x in sources:
                if type(x) == type(""):
                    fname = env.File(x).path
                else:
                    fname = env.File(x)[0].path
                pieces = fname.split(".")
                if len(pieces) > 0:
                    basename = pieces[0]
                    basename = basename.replace('\\\\', '/')
                    env.vs_srcs = env.vs_srcs + [basename + ".cpp"]
                    env.vs_incs = env.vs_incs + [basename + ".h"]
                    # print basename
        env.AddToVSProject = AddToVSProject

    env.extra_suffix = ""

    if env["extra_suffix"] != '':
        env.extra_suffix += '.' + env["extra_suffix"]

    CCFLAGS = env.get('CCFLAGS', '')
    env['CCFLAGS'] = ''

    env.Append(CCFLAGS=str(CCFLAGS).split())

    CFLAGS = env.get('CFLAGS', '')
    env['CFLAGS'] = ''

    env.Append(CFLAGS=str(CFLAGS).split())

    LINKFLAGS = env.get('LINKFLAGS', '')
    env['LINKFLAGS'] = ''

    env.Append(LINKFLAGS=str(LINKFLAGS).split())

    flag_list = platform_flags[selected_platform]
    for f in flag_list:
        if not (f[0] in ARGUMENTS):  # allow command line to override platform flags
            env[f[0]] = f[1]

    # must happen after the flags, so when flags are used by configure, stuff happens (ie, ssl on x11)
    detect.configure(env)

    if (env["warnings"] == 'yes'):
        print("WARNING: warnings=yes is deprecated; assuming warnings=all")

    if (os.name == "nt" and os.getenv("VCINSTALLDIR") and (platform_arg == "windows" or platform_arg == "uwp")): # MSVC, needs to stand out of course
        disable_nonessential_warnings = ['/wd4267', '/wd4244', '/wd4305', '/wd4800'] # Truncations, narrowing conversions...
        if (env["warnings"] == 'extra'):
            env.Append(CCFLAGS=['/Wall']) # Implies /W4
        elif (env["warnings"] == 'all' or env["warnings"] == 'yes'):
            env.Append(CCFLAGS=['/W3'] + disable_nonessential_warnings)
        elif (env["warnings"] == 'moderate'):
            # C4244 shouldn't be needed here being a level-3 warning, but it is
            env.Append(CCFLAGS=['/W2'] + disable_nonessential_warnings)
        else: # 'no'
            env.Append(CCFLAGS=['/w'])
    else: # Rest of the world
        if (env["warnings"] == 'extra'):
            env.Append(CCFLAGS=['-Wall', '-Wextra'])
        elif (env["warnings"] == 'all' or env["warnings"] == 'yes'):
            env.Append(CCFLAGS=['-Wall'])
        elif (env["warnings"] == 'moderate'):
            env.Append(CCFLAGS=['-Wall', '-Wno-unused'])
        else: # 'no'
            env.Append(CCFLAGS=['-w'])

    #env['platform_libsuffix'] = env['LIBSUFFIX']

    suffix = "." + selected_platform

    if (env["target"] == "release"):
        if (env["tools"] == "yes"):
            print("Tools can only be built with targets 'debug' and 'release_debug'.")
            sys.exit(255)
        suffix += ".opt"
        env.Append(CCFLAGS=['-DNDEBUG'])

    elif (env["target"] == "release_debug"):
        if (env["tools"] == "yes"):
            suffix += ".opt.tools"
        else:
            suffix += ".opt.debug"
    else:
        if (env["tools"] == "yes"):
            suffix += ".tools"
        else:
            suffix += ".debug"

    if env["arch"] != "":
        suffix += "." + env["arch"]
    elif (env["bits"] == "32"):
        suffix += ".32"
    elif (env["bits"] == "64"):
        suffix += ".64"
    elif (env["bits"] == "fat"):
        suffix += ".fat"

    suffix += env.extra_suffix

    env["PROGSUFFIX"] = suffix + env["PROGSUFFIX"]
    env["OBJSUFFIX"] = suffix + env["OBJSUFFIX"]
    env["LIBSUFFIX"] = suffix + env["LIBSUFFIX"]
    env["SHLIBSUFFIX"] = suffix + env["SHLIBSUFFIX"]

    sys.path.remove("./platform/" + selected_platform)
    sys.modules.pop('detect')

    env.module_list = []
    env.doc_class_path={}

    for x in module_list:
        if env['module_' + x + '_enabled'] != "yes":
            continue
        tmppath = "./modules/" + x
        sys.path.append(tmppath)
        env.current_module = x
        import config
        if (config.can_build(selected_platform)):
            config.configure(env)
            env.module_list.append(x)
	    try:
		 doc_classes = config.get_doc_classes()
		 doc_path = config.get_doc_path()
		 for c in doc_classes:
		     env.doc_class_path[c]="modules/"+x+"/"+doc_path
	    except:
		pass


        sys.path.remove(tmppath)
        sys.modules.pop('config')

    if (env.use_ptrcall):
        env.Append(CPPFLAGS=['-DPTRCALL_ENABLED'])

    # to test 64 bits compiltion
    # env.Append(CPPFLAGS=['-m64'])

    if (env['tools'] == 'yes'):
        env.Append(CPPFLAGS=['-DTOOLS_ENABLED'])
    if (env['disable_3d'] == 'yes'):
        env.Append(CPPFLAGS=['-D_3D_DISABLED'])
    if (env['gdscript'] == 'yes'):
        env.Append(CPPFLAGS=['-DGDSCRIPT_ENABLED'])
    if (env['disable_advanced_gui'] == 'yes'):
        env.Append(CPPFLAGS=['-DADVANCED_GUI_DISABLED'])

    if (env['minizip'] == 'yes'):
        env.Append(CPPFLAGS=['-DMINIZIP_ENABLED'])

    if (env['xml'] == 'yes'):
        env.Append(CPPFLAGS=['-DXML_ENABLED'])

    if (env['verbose'] == 'no'):
        methods.no_verbose(sys, env)

    if (True): # FIXME: detect GLES3
        env.Append( BUILDERS = { 'GLES3_GLSL' : env.Builder(action = methods.build_gles3_headers, suffix = 'glsl.gen.h',src_suffix = '.glsl') } )

    Export('env')

    # build subdirs, the build order is dependent on link order.

    SConscript("core/SCsub")
    SConscript("servers/SCsub")
    SConscript("scene/SCsub")
    SConscript("editor/SCsub")
    SConscript("drivers/SCsub")

    SConscript("modules/SCsub")
    SConscript("main/SCsub")

    SConscript("platform/" + selected_platform + "/SCsub")  # build selected platform

    # Microsoft Visual Studio Project Generation
    if (env['vsproj']) == "yes":
        methods.generate_vs_project(env, GetOption("num_jobs"))

else:

    print("No valid target platform selected.")
    print("The following were detected:")
    for x in platform_list:
        print("\t" + x)
    print("\nPlease run scons again with argument: platform=<string>")


screen = sys.stdout
node_count = 0
node_count_max = 0
node_count_interval = 1
if ('env' in locals()):
    node_count_fname = str(env.Dir('#')) + '/.scons_node_count'

def progress_function(node):
    global node_count, node_count_max, node_count_interval, node_count_fname
    node_count += node_count_interval
    if (node_count_max > 0 and node_count <= node_count_max):
        screen.write('\r[%3d%%] ' % (node_count * 100 / node_count_max))
        screen.flush()
    elif (node_count_max > 0 and node_count > node_count_max):
        screen.write('\r[100%] ')
        screen.flush()
    else:
        screen.write('\r[Initial build] ')
        screen.flush()

def progress_finish(target, source, env):
    global node_count
    with open(node_count_fname, 'w') as f:
        f.write('%d\n' % node_count)

if ('env' in locals() and env["progress"] == "yes"):
    try:
        with open(node_count_fname) as f:
            node_count_max = int(f.readline())
    except:
        pass
    Progress(progress_function, interval = node_count_interval)
    progress_finish_command = Command('progress_finish', [], progress_finish)
    AlwaysBuild(progress_finish_command)
