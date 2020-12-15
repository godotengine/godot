# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
_target_windows_logo = custom_target(
    'windows_logo',
    output: ['logo.gen.h'],
    input: ['logo.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'windows'],
    build_by_default: true
)
SOURCES += [_target_windows_logo]

################################################################################
# Host Machine Specifics
################################################################################
if PLATFORM == 'windows'
    # add the windows main to main sources
    MAIN_SOURCES += files('godot_windows.cpp')

    # the src files for the windows *platform* library
    _windows_srcs = files([
        'context_gl_windows.cpp',
        'crash_handler_windows.cpp',
        'display_server_windows.cpp',
        'joypad_windows.cpp',
        'key_mapping_windows.cpp',
        'os_windows.cpp',
        'vulkan_context_win.cpp',
        'windows_terminal_logger.cpp'
    ])

    # get all of our dependencies
    _cpp = meson.get_compiler('cpp')

    # common deps between msvc and mingw
    # Only need to define those that are not part of Meson's defaultlibs,
    # which has things like kernel32 et al.
    _windows_deps = [
        _cpp.find_library('avrt'),
        _cpp.find_library('bcrypt'),
        _cpp.find_library('dinput8'),
        _cpp.find_library('dsound'),
        _cpp.find_library('dwmapi'),
        _cpp.find_library('dxguid'),
        _cpp.find_library('imm32'),
        _cpp.find_library('iphlpapi'),
        _cpp.find_library('shlwapi'),
        _cpp.find_library('winmm'),
        _cpp.find_library('ws2_32'),
        _cpp.find_library('wsock32'),
        _cpp.find_library('opengl32'),
        _cpp.find_library('ksuser'),
    ]

    CPP_ARGS += [
        '-DVULKAN_ENABLED',
        '-DWINDOWS_ENABLED',
        '-DWASAPI_ENABLED',
        '-DWINMIDI_ENABLED',
        '-DWIN32',
    ]

    if get_option('windows_subsystem') == 'console'
        CPP_ARGS += '-DWINDOWS_SUBSYSTEM_CONSOLE'
    endif

    # TODO: Check if this is needed, msvc will define this, I assume we also define it because of mingw/clang-cl.
    #       But I would assume it does already.
    if host_machine.cpu_family() == 'x86_64'
        CPP_ARGS += '-D_WIN64'
    endif

    # Native windows compile
    if _cpp.get_id() == 'msvc'

        CPP_ARGS += [
            '-DMSVC',
            '-DNOMINMAX',
            '-DTYPED_METHOD_BIND',
            #'-DWIN32', #?
        ]

        # TODO: stack size is only used on windows platform...? (1/2)
        LINK_ARGS += [
            '/STACK:' + STACK_SIZE.to_string()
        ]
    # Cross compile with mingw
    elif _cpp.get_id() == 'gcc'
        _windows_deps += [
            _cpp.find_library('d3d9'), # TODO - not sure why mingw32 needs d3d9 ?
            _cpp.find_library('mingw32'),
        ]

        CPP_ARGS += [
            '-mwindows',
            '-DMINGW_ENABLED',
            '-DMINGW_HAS_SECURE_API=1',
        ]

        # TODO: stack size is only used on windows platform...? (2/2)
        LINK_ARGS += [
            '-Wl,--stack,' + STACK_SIZE.to_string(),
            '-static'
        ]
    endif

    DEPENDENCIES += _windows_deps
    SOURCES += _windows_srcs

endif
