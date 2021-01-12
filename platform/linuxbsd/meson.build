# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
_target_linuxbsd_logo = custom_target(
    'linuxbsd_logo',
    output: ['logo.gen.h'],
    input: ['logo.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'linuxbsd'],
    build_by_default: true
)
SOURCES += [_target_linuxbsd_logo]


################################################################################
# Host Machine Specifics
################################################################################
if PLATFORM == 'linuxbsd'

    # add the godot linuxbsd main to main sources
    MAIN_SOURCES += files('godot_linuxbsd.cpp')

    # the src files for the linux *platform*
    _linuxbsd_srcs = files([
        'crash_handler_linuxbsd.cpp',
        'os_linuxbsd.cpp',
        'joypad_linux.cpp',
        'context_gl_x11.cpp',
        'detect_prime_x11.cpp',
        'display_server_x11.cpp',
        'vulkan_context_x11.cpp',
        'key_mapping_x11.cpp',
    ])

    # get all of our dependencies
    _cpp = meson.get_compiler('cpp')
    _linuxbsd_deps = [
        _cpp.find_library('dl'),
        dependency('threads', method: 'auto'),
        
        dependency('x11', method: 'pkg-config'),
        dependency('xcursor', method: 'pkg-config'),
        dependency('xinerama', method: 'pkg-config'),
        dependency('xext', method: 'pkg-config'),
        dependency('xrandr', method: 'pkg-config'),
        dependency('xrender', method: 'pkg-config'),
        dependency('xi', method: 'pkg-config'),
        #_cpp.find_library('gl'),
        #_cpp.find_library('X11')
    ]

    CPP_ARGS += [
        '-DUNIX_ENABLED',
        '-DVULKAN_ENABLED',
        '-DX11_ENABLED',
    ]

    DEPENDENCIES += _linuxbsd_deps
    SOURCES += _linuxbsd_srcs
endif