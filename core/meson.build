DEPENDENCIES += [DEP_MBEDTLS, DEP_ZLIB, DEP_ZSTD, DEP_MINIZIP, DEP_MISC]

################################################################################
# MISC LIB FOR CORE - BEGIN
################################################################################

# random misc sources, we are going to build these as a small static lib
_core_misc_srcs = files([
    # c sources
    '../thirdparty/misc/fastlz.c',
    '../thirdparty/misc/r128.c',
    '../thirdparty/misc/smaz.c',
    # cpp sources
    '../thirdparty/misc/pcg.cpp',
    '../thirdparty/misc/polypartition.cpp',
    '../thirdparty/misc/clipper.cpp',
    '../thirdparty/misc/smolv.cpp',
])

_core_misc_c_args = []
_core_misc_cpp_args = []

if meson.get_compiler('c').get_id() == 'msvc'
    _core_misc_c_args += '/wd4090'
elif meson.get_compiler('c').get_id() == 'gcc'
    _core_misc_c_args += [
        '-Wno-discarded-qualifiers',
        '-Wno-incompatible-pointer-types',
    ]
endif

if meson.get_compiler('cpp').get_id() == 'msvc'
    _core_misc_cpp_args += '/wd4003'
endif

_lib_core_misc = static_library('builtin_core_misc', _core_misc_srcs, c_args: _core_misc_c_args, cpp_args: _core_misc_cpp_args, include_directories: INCDIRS)

################################################################################
# MISC LIB FOR CORE - END
################################################################################

SOURCES += files(run_command(GLOB, '*.cpp').stdout().split('\n'))

# generated version file
version_arr = meson.project_version().split('.')
version_cdata = configuration_data()
version_cdata.set_quoted('VERSION_SHORT_NAME', 'Godot')
version_cdata.set_quoted('VERSION_NAME', 'Godot Engine')
version_cdata.set('VERSION_MAJOR', version_arr[0])
version_cdata.set('VERSION_MINOR', version_arr[1])
version_cdata.set('VERSION_PATCH', version_arr[2])
version_cdata.set_quoted('VERSION_STATUS', 'dev')
version_cdata.set_quoted('VERSION_BUILD', 'custom_build')
version_cdata.set_quoted('VERSION_MODULE_CONFIG', '')
version_cdata.set('VERSION_YEAR', 2020)
version_cdata.set_quoted('VERSION_WEBSITE', 'https://godotengine.org')
configure_file(output: 'version_generated.gen.h',
    configuration: version_cdata)
version_hash_data = configuration_data()
version_hash_data.set_quoted('VERSION_HASH', '')
configure_file(output: 'version_hash.gen.h',
    configuration: version_hash_data)

# generated input mappings
_encryption_key_target = custom_target(
    'encryption_key',
    output: ['script_encryption_key.gen.cpp'],
    command: [SCRIPT_ENCRYPTION_KEY, '@OUTPUT@'],
    env: SCRIPTS_ENV,
    build_by_default: true
)
SOURCES += [_encryption_key_target]

# generated authors file
_authors_target = custom_target(
    'authors',
    output: ['authors.gen.h'],
    input: ['../AUTHORS.md'],
    env: SCRIPTS_ENV,
    command: [SCRIPT_COMPAT, 'make_authors_header', SCRIPT_CORE_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_authors_target]

# generated donors file
_donors_target = custom_target(
    'donors',
    output: ['donors.gen.h'],
    input: ['../DONORS.md'],
    env: SCRIPTS_ENV,
    command: [SCRIPT_COMPAT, 'make_donors_header', SCRIPT_CORE_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_donors_target]

# generated license file
_license_target = custom_target(
    'license',
    output: ['license.gen.h'],
    input: ['../COPYRIGHT.txt', '../LICENSE.txt'],
    env: SCRIPTS_ENV,
    command: [SCRIPT_COMPAT, 'make_license_header', SCRIPT_CORE_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_license_target]

# Do any subdir's here (for any custom target they might have...)
subdir('config')
subdir('crypto')
subdir('debugger')
subdir('error')
subdir('extension')
subdir('input')
subdir('io')
subdir('multiplayer')
subdir('math')
subdir('object')
subdir('os')
subdir('string')
subdir('templates')
subdir('variant')

cdata = configuration_data()
configure_file(output: 'disabled_classes.gen.h',
    configuration: cdata)
    DEP_CORE_MISC = declare_dependency(link_with: _lib_core_misc)
    DEPENDENCIES += [DEP_CORE_MISC]
