# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
_target_javascript_logo = custom_target(
    'javascript_logo',
    output: ['logo.gen.h'],
    input: ['logo.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'javascript'],
    build_by_default: true
)
SOURCES += [_target_javascript_logo]

_target_javascript_run_icon = custom_target(
    'javascript_run_icon',
    output: ['run_icon.gen.h'],
    input: ['run_icon.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'javascript'],
    build_by_default: true
)
SOURCES += [_target_javascript_run_icon]


################################################################################
# Host Machine Specifics
################################################################################
if PLATFORM == 'javascript'
dir_base = meson.current_source_dir()
    _js_deps = [
        cpp.find_library('library_godot_audio.js', dirs: [meson.current_source_dir() / 'js/libs']),
        cpp.find_library('library_godot_audio.js', dirs: [meson.current_source_dir() / 'js/libs']),
        cpp.find_library('library_godot_display.js', dirs: [meson.current_source_dir() / 'js/libs']),
        cpp.find_library('library_godot_fetch.js', dirs: [meson.current_source_dir() / 'js/libs']),
        cpp.find_library('library_godot_os.js', dirs: [meson.current_source_dir() / 'js/libs']),
        cpp.find_library('library_godot_runtime.js', dirs: [meson.current_source_dir() / 'js/libs']),
        cpp.find_library('library_godot_javascript_singleton.js', dirs: [meson.current_source_dir() / 'js/libs'])
    ]

   # add the javascript main to main sources
   MAIN_SOURCES += files('javascript_runtime.cpp')

    # the src files for the javascript *platform* library
    _javascript_srcs = files([
        'audio_driver_javascript.cpp',
        'display_server_javascript.cpp',
        'http_client_javascript.cpp',
        'javascript_singleton.cpp',
        'javascript_main.cpp',
        'os_javascript.cpp',
        'api/javascript_tools_editor_plugin.cpp',
    ])

    DEPENDENCIES += _js_deps

    # Explicit IDBFS dependency
    DEPENDENCIES += declare_dependency(link_args: '-lidbfs.js')

    CPP_ARGS += [
        '-DJAVASCRIPT_ENABLED',
        '-DUNIX_ENABLED',
        '-DJAVASCRIPT_EVAL_ENABLED', # TODO
        '-DPTHREAD_NO_RENAME'
    ]

    SOURCES += _javascript_srcs
endif
