# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
macos_logo = custom_target(
                     'macos_logo',
    output: ['logo_svg.gen.h'],
    input: ['export/logo.svg'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'macos'],
    build_by_default: true
)

macos_run_icon = custom_target(
    'android_run_icon',
    output: ['run_icon_svg.gen.h'],
    input: ['export/run_icon.svg'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'macos'],
    build_by_default: true
)

SOURCES += [macos_logo, macos_run_icon]


################################################################################
# Host Machine Specifics
################################################################################
if PLATFORM == 'macos'
    MAIN_SOURCES += files('godot_main_macos.mm')

    macos_srcs = files('joypad_macos.cpp',
                      #'context_gl_macos.mm',
                      'crash_handler_macos.mm',
                      'dir_access_macos.mm',
                      'display_server_macos.mm',
                      'os_macos.mm',
                      'vulkan_context_macos.mm',
                      'godot_application.mm',
                      'godot_application_delegate.mm',
                      'godot_button_view.mm',
                      'godot_content_view.mm',
                      'godot_main_macos.mm',
                      'godot_menu_delegate.mm',
                      'godot_menu_item.mm',
                      'godot_window.mm',
                      'godot_window_delegate.mm',
                      'key_mapping_macos.mm',
                      'macos_terminal_logger.mm',
                      'tts_macos.mm',
                      )
    CPP_ARGS += ['-mmacosx-version-min=10.12', '-DUNIX_ENABLED', '-DVULKAN_ENABLED', '-DMACOS_ENABLED', '-DAPPLE_STYLE_KEYS', '-DCOREAUDIO_ENABLED', '-DCOREMIDI_ENABLED']
    OBJCPP_ARGS += ['-mmacosx-version-min=10.12', '-DUNIX_ENABLED', '-DVULKAN_ENABLED', '-DMACOS_ENABLED', '-DAPPLE_STYLE_KEYS', '-DCOREAUDIO_ENABLED', '-DCOREMIDI_ENABLED', '-fobjc-arc']
    SOURCES += macos_srcs
    DEPENDENCIES += dependency('appleframeworks', modules: ['AppKit',
                                                            'IOKit',
                                                            'CoreAudio',
                                                            'CoreMidi',
                                                            'CoreVideo',
                                                            'CoreMedia',
                                                            'QuartzCore',
                                                            'CoreServices',
                                                            'Carbon',
                                                            'GameController',
                                                            'CoreHaptics',
                                                            'ForceFeedback',
                                                            'AVFoundation',
                                                            'AudioToolbox',
                                                            'foundation',
                                                            'security',
                                                            ])
endif
