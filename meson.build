project(
    'godot',
    ['cpp', 'c'],
    version: '4.0.0',
    license: 'MIT',
    meson_version: '>= 0.56',
    default_options: [
        'c_std=c11',
        'cpp_std=c++17',
        'warning_level=0',
        'werror=false',
        'default_library=static',
        'buildtype=debugoptimized'
        ])


message(get_option('buildtype'))

# The list of modules that will not be included in the build

################################################################################
# Determine Platform
################################################################################

# First, let us determine what platform we are from a cross or native file
# if it has been specified. Otherwise, fallback to the host machine.
PLATFORM = meson.get_external_property('platform', 'none')
if PLATFORM == 'none'
    PLATFORM = 'linuxbsd'
    if host_machine.system() == 'windows'
        PLATFORM = 'windows'
    elif host_machine.system() == 'darwin'
        PLATFORM = 'osx'
    endif
endif

################################################################################
# Check Options
################################################################################

# TODO - temporary way to do proper debug without /ZI on msvc (there is a PR for removing ZI already in meson)
IS_DEBUG_BUILD = get_option('debug') or get_option('buildtype') == 'plain'


# This is one of the very few, if only, times we will add defines to the entire project
# WINVER is required for quite a few thirdparty libs when building, especially when using mingw
if PLATFORM == 'windows'
    add_project_arguments(
        '-DWINVER=' + get_option('windows_version'),
        '-D_WIN32_WINNT=' + get_option('windows_version'),
        language: ['c', 'cpp']
    )
endif

if get_option('tools') and not IS_DEBUG_BUILD # get_option('debug') - TODO change when ZI issue fixed
    error('tools can only be built with debug symbols')
endif

if get_option('disable_3d') and get_option('tools')
    error('disable_3d can only be enabled when tools are disabled')
endif

if get_option('disable_advanced_gui') and get_option('tools')
    error('disable_advanced_gui can only be enabled when tools are disabled')
endif

if get_option('builtin_freetype') or get_option('builtin_libpng') or get_option('builtin_zlib')
    if not get_option('builtin_freetype') or not get_option('builtin_libpng') or not get_option('builtin_zlib')
        error('builtin_freetype, builtin_libpng, and builtin_zlib must all be on if one is on.')
    endif
endif

if not get_option('builtin_libtheora')
    # Needed to link against system libtheora
    if get_option('builtin_libogg')
        error('When builtin_libtheora is off, builtin_libogg must be off.')
    endif
    if get_option('builtin_libvorbis')
        error('When builtin_libtheora is off, builtin_libvorbis must be off.')
    endif
endif

################################################################################
# Load Scripts
################################################################################

subdir('scripts')

################################################################################
# Figure out what modules we are building
################################################################################

MODULE_SEARCH_PATHS = ['modules/']
MODULES_DB_FILE = meson.project_build_root() / 'modules_db.json'

# Create the modules_db file
_module_script_args = [
    'create_db', MODULES_DB_FILE, 
    '--platform', PLATFORM,
    '--cpu_family', host_machine.cpu_family()
]

if get_option('tools')
    _module_script_args += '--tools_enabled'
endif

foreach msp : MODULE_SEARCH_PATHS
    _module_script_args += ['--module_search_path', msp]
endforeach

foreach dm : get_option('disabled_modules')
    _module_script_args += ['--module_disabled', dm]
endforeach

if get_option('disable_all_modules')
    _module_script_args += ['--disable_all_modules']
endif

_module_output = run_command(SCRIPT_MODULE_DB, _module_script_args)
if _module_output.returncode() != 0 or _module_output.stderr() != ''
    error(_module_output.stderr())
endif

# Get the list of enabled modules
_module_enabled_output = run_command(SCRIPT_MODULE_DB, ['get_enabled_modules', MODULES_DB_FILE])
if _module_enabled_output.returncode() != 0
    error(_module_enabled_output.stderr())
endif

MODULES_ENABLED = _module_enabled_output.stdout().strip().split(',')
message('Modules Enabled - ', MODULES_ENABLED)

# Get the list of disabled modules
_module_disabled_output = run_command(SCRIPT_MODULE_DB, ['get_disabled_modules', MODULES_DB_FILE])
if _module_disabled_output.returncode() != 0
    error(_module_disabled_output.stderr())
endif

MODULES_DISABLED = _module_disabled_output.stdout().strip().split(',')
message('Modules Disabled - ', MODULES_DISABLED)

_module_get_build_paths = run_command(SCRIPT_MODULE_DB, ['get_module_build_paths', MODULES_DB_FILE])
MODULE_BUILD_PATHS = _module_get_build_paths.stdout().strip().split(',')

#_module_list = run_command(SCRIPT_MODULE_DB, ['print_modules', MODULES_DB_FILE])
#message(_module_list.stdout())

################################################################################
# COMPILER FLAGS - DEFINED BEFORE ANY TARGETS
################################################################################


# MSVC specific compiler settings
_cpp_compiler_id = meson.get_compiler('cpp').get_id()

CPP_ARGS = []

if get_option('use_lto')
    # Cant use GL and LTGC when ZI is forced by meson debug builds
    if _cpp_compiler_id == 'msvc' and get_option('buildtype') == 'debug' # remove the debug check when ZI is removed from debug
        add_project_arguments('/GL', language: ['c', 'cpp'])
        add_project_link_arguments('/LTGC', language: ['c', 'cpp'])

    else
        message('use_lto option is on, but has only been setup for MSVC so far')
    endif

endif

# do we want to show our #warnings ?
if not get_option('godot_warnings')
    if _cpp_compiler_id == 'gcc'
        CPP_ARGS += '-Wno-cpp'
    endif
endif

if _cpp_compiler_id == 'msvc'
     msvc_nonessential_warnings = ['/wd4267', '/wd4244', '/wd4305', '/wd4018', '/wd4800']
     CPP_ARGS += msvc_nonessential_warnings
endif

if _cpp_compiler_id == 'clang' or _cpp_compiler_id == 'clang-cl'
    CPP_ARGS += ['-Wdeprecated-declarations']
endif


if IS_DEBUG_BUILD # TODO # get_option(debug) # using IS_DEBUG_BUILD to create a fast debug build under MSVC without ZI
    CPP_ARGS += [
        '-DDEBUG_ENABLED',
        #'-DDEBUG_MEMORY_ALLOC', # This is unused anywhere...
        '-DDISABLE_FORCED_INLINE',
    ]

    # only, only, only add this in if its not msvc. you will break msvc's linking if you set this manually
    # add this in at the PROJECT LEVEL. We have to pass this through to our thirdparties
    if _cpp_compiler_id != 'msvc'
        add_project_arguments('-D_DEBUG', language: ['c', 'cpp'])
    endif

    # TODO remove when ZI issue is gone
    if _cpp_compiler_id == 'msvc' and get_option('buildtype') == 'plain'
        add_project_arguments('/Zi', language: ['c', 'cpp'])
    endif


    # DEBUG is only used by thirdparties, and most use it in conjuction with _DEBUG.
    # If it does need to be added, it has to be added as a project argument
    # CPP_ARGS += '-DDEBUG'
    
else
    # NDEBUG is only used by thirdparties. Im going to not add it in.
    # If it does need to be added, it has to be added as a project argument
    # CPP_ARGS += '-DNDEBUG'
endif

if get_option('no_editor_splash')
    CPP_ARGS += '-DNO_EDITOR_SPLASH'
endif

if get_option('use_precise_math_checks')
    CPP_ARGS += '-DPRECISE_MATH_CHECKS'
endif

if not get_option('deprecated')
    CPP_ARGS += '-DDISABLE_DEPRECATED'
endif

if get_option('tools')
    CPP_ARGS += '-DTOOLS_ENABLED'

    if get_option('disable_3d')
        CPP_ARGS += '-D_3D_DISABLED'
    endif

    if get_option('disable_advanced_gui')
        CPP_ARGS += '-DADVANCED_GUI_DISABLED'
    endif

    if 'minizip' in MODULES_ENABLED
        CPP_ARGS += '-DMINIZIP_ENABLED'
    endif
endif



# global constants
# To match other platforms
STACK_SIZE = 8388608

################################################################################
# FINALLY - GET OUR SOURCES
################################################################################

DIR_PLATFORM = 'platform' / PLATFORM

MAIN_SOURCES = []

INCDIRS = include_directories('.', 'thirdparty', DIR_PLATFORM)
SOURCES = []
HEADERS = []
DEPENDENCIES = []
LINK_ARGS = []

# thirdparty first
subdir('thirdparty')

# get all of godot except main
subdir('platform')
subdir('core')
subdir('scene')
subdir('drivers')
subdir('servers')
subdir('modules')
subdir('editor')

# generate our splashes
subdir('splash')

_godot_all = library('godot_all', SOURCES + HEADERS, include_directories: INCDIRS, dependencies: DEPENDENCIES, cpp_args: CPP_ARGS, link_args: LINK_ARGS)

DEP_GODOT_ALL = declare_dependency(link_with: _godot_all, include_directories: INCDIRS, compile_args: CPP_ARGS, link_args: LINK_ARGS)

# build up our module list
MODULE_DEPENDENCIES = []
foreach mp : MODULE_BUILD_PATHS
    subdir(mp)
endforeach

# finally, get the sources for main separately
subdir('main')

# assemble our exe
EXEC_GODOT = executable('godot', MAIN_SOURCES, dependencies: [DEP_GODOT_ALL, MODULE_DEPENDENCIES, DEPENDENCIES])