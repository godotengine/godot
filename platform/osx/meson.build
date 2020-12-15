# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
_target_osx_logo = custom_target(
                     'osx_logo',
    output: ['logo.gen.h'],
    input: ['logo.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'osx'],
    build_by_default: true
)
SOURCES += [_target_osx_logo]


################################################################################
# Host Machine Specifics
################################################################################
if PLATFORM == 'osx'
    MAIN_SOURCES += files('godot_main_osx.mm')

    _osx_srcs = files('joypad_osx.cpp',
                      #'context_gl_osx.mm',
                      'crash_handler_osx.mm',
                      'dir_access_osx.mm',
                      'display_server_osx.mm',
                      'godot_main_osx.mm',
                      'os_osx.mm',
                      'vulkan_context_osx.mm',
                      )
    CPP_ARGS += ['-DUNIX_ENABLED', '-DVULKAN_ENABLED', '-DOSX_ENABLED', '-DAPPLE_STYLE_KEYS', '-DCOREAUDIO_ENABLED', '-DCOREMIDI_ENABLED']
    OBJCPP_ARGS += ['-DUNIX_ENABLED', '-DVULKAN_ENABLED', '-DOSX_ENABLED', '-DAPPLE_STYLE_KEYS', '-DCOREAUDIO_ENABLED', '-DCOREMIDI_ENABLED']
    SOURCES += _osx_srcs
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
                                                            'foundation'])
endif
