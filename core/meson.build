DEPENDENCIES += [DEP_MBEDTLS, DEP_ZLIB, DEP_ZSTD, DEP_MINIZIP]

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
    '../thirdparty/misc/triangulator.cpp',
    '../thirdparty/misc/clipper.cpp',
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
DEP_CORE_MISC = declare_dependency(link_with: _lib_core_misc)
DEPENDENCIES += [DEP_CORE_MISC]

################################################################################
# MISC LIB FOR CORE - END
################################################################################

SOURCES += files([
    'core_bind.cpp',
    'core_constants.cpp',
    'core_string_names.cpp',
    'doc_data.cpp',
    'register_core_types.cpp',
])

# generated version file
_version_target = custom_target(
    'version',
    output: ['version_generated.gen.h', 'version_hash.gen.h'],
    input: ['../version.json'],
    command: [SCRIPT_VERSION, '@INPUT@', '@OUTPUT@', MODULES_ENABLED],
    build_by_default: true
)
SOURCES += [_version_target]

# generated input mappings
_encryption_key_target = custom_target(
    'encryption_key',
    output: ['script_encryption_key.gen.cpp'],
    command: [SCRIPT_ENCRYPTION_KEY, '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_encryption_key_target]

# generated authors file
_authors_target = custom_target(
    'authors',
    output: ['authors.gen.h'],
    input: ['../AUTHORS.md'],
    command: [SCRIPT_AUTHORS, '@INPUT@', '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_authors_target]

# generated donors file
_donors_target = custom_target(
    'donors',
    output: ['donors.gen.h'],
    input: ['../DONORS.md'],
    command: [SCRIPT_DONORS, '@INPUT@', '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_donors_target]

# generated license file
_license_target = custom_target(
    'license',
    output: ['license.gen.h'],
    input: ['../COPYRIGHT.txt', '../LICENSE.txt'],
    command: [SCRIPT_LICENSE, '@INPUT@', '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_license_target]

# Do any subdir's here (for any custom target they might have...)
subdir('config')
subdir('crypto')
subdir('debugger')
subdir('error')
subdir('input')
subdir('io')
subdir('math')
subdir('object')
subdir('os')
subdir('string')
subdir('templates')
subdir('variant')