# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
linuxbsd_logo = custom_target(
    'linuxbsd_logo',
    output: ['logo_svg.gen.h'],
    input: ['logo.svg'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'linuxbsd'],
    build_by_default: true
)

linuxbsd_run_icon = custom_target(
    'android_run_icon',
    output: ['run_icon_svg.gen.h'],
    input: ['run_icon.svg'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'linuxbsd'],
    build_by_default: true
)

SOURCES += [linuxbsd_logo, linuxbsd_run_icon]


################################################################################
# Host Machine Specifics
################################################################################
if PLATFORM == 'linuxbsd'
    subdir('x11')

    # add the godot linuxbsd main to main sources
    MAIN_SOURCES += files('godot_linuxbsd.cpp')

    # the src files for the linux *platform*
    _linuxbsd_srcs += files(
        'crash_handler_linuxbsd.cpp',
        'os_linuxbsd.cpp',
        'joypad_linux.cpp',
        'xkbcommon-so_wrap.c',
    )

    # get all of our dependencies
    _cpp = meson.get_compiler('cpp')
    _linuxbsd_deps = [
        _cpp.find_library('dl'),
        dependency('threads'),

        dependency('x11'),
        dependency('xcursor'),
        dependency('xinerama'),
        dependency('xext'),
        dependency('xrandr'),
        dependency('xrender'),
        dependency('xi'),
        #dependency('vulkan')
        #_cpp.find_library('vulkan'),
    ]

    CPP_ARGS += [
        '-DUNIX_ENABLED',
        '-DVULKAN_ENABLED',
        '-DX11_ENABLED',
    ]

    DEPENDENCIES += _linuxbsd_deps
    SOURCES += _linuxbsd_srcs
endif
