# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
linuxbsd_logo = custom_target(
    'linuxbsd_logo',
    output: ['logo_svg.gen.h'],
    input: ['export/logo.svg'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'linuxbsd'],
    build_by_default: true
)

linuxbsd_run_icon = custom_target(
    'android_run_icon',
    output: ['run_icon_svg.gen.h'],
    input: ['export/run_icon.svg'],
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
        'freedesktop_portal_desktop.cpp',
        'freedesktop_screensaver.cpp',
    )

    if get_option('use_sowrap')
        _linuxbsd_srcs += files('xkbcommon-so_wrap.c')
        CPP_ARGS += '-DSOWRAP_ENABLED'
    endif

    _linuxbsd_deps = []

    _WRAPPED_LIBS = [
        [
            'speechd', 'speech-dispatcher', '-DSPEECHD_ENABLED',
            'speechd-so_wrap.c', 'tts_linux.cpp'
        ], [
            'fontconfig', 'fontconfig', '-DFONTCONFIG_ENABLED',
            'fontconfig-so_wrap.c',
        ], [
            'udev', 'libudev', '-DUDEV_ENABLED', 'libudev-so_wrap.c'
        ], [
            'dbus', 'dbus-1', '-DDBUS_ENABLED', 'dbus-so_wrap.c'
        ], [
            'alsa', 'alsa', ['-DALSA_ENABLED', '-DALSAMIDI_ENABLED'],
            '../../drivers/alsa/asound-so_wrap.c'
        ], [
            'pulseaudio', 'libpulse', '-DPULSEAUDIO_ENABLED',
            '../../drivers/pulseaudio/pulse-so_wrap.c'
        ],
    ]

    foreach wl : _WRAPPED_LIBS
        if get_option(wl[0])
            if not get_option('use_sowrap')
                d = dependency(wl[1])
                _linuxbsd_deps += d
                if d.found()
                    _linuxbsd_srcs += files(wl.get(4, []))
                    CPP_ARGS += wl[2]
                endif
            else
                _linuxbsd_srcs += files(wl.get(3, []))
                _linuxbsd_srcs += files(wl.get(4, []))
                CPP_ARGS += wl[2]
            endif
        endif
    endforeach

    # get all of our dependencies
    _cpp = meson.get_compiler('cpp')
    _linuxbsd_deps += [
        _cpp.find_library('dl'),
        dependency('threads'),

        dependency('x11'),
        dependency('xcursor'),
        dependency('xinerama'),
        dependency('xext'),
        dependency('xrandr'),
        dependency('xrender'),
        dependency('xi'),
        DEP_GLAD,
        DEP_LINUXBSD_HEADERS,
    ]

    CPP_ARGS += [
        '-DUNIX_ENABLED',
        '-DX11_ENABLED',
    ]

    if get_option('vulkan')
        CPP_ARGS += ['-DVULKAN_ENABLED']
        if not get_option('use_volk')
            _linuxbsd_deps += dependency('vulkan')
        endif
    endif

    if get_option('opengl3')
        CPP_ARGS += ['-DGLES3_ENABLED']
    endif

    DEPENDENCIES += _linuxbsd_deps
    SOURCES += _linuxbsd_srcs
endif
