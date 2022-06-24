# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
_target_ios_logo = custom_target(
    'ios_logo',
    output: ['logo.gen.h'],
    input: ['logo.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'ios'],
    build_by_default: true
)
SOURCES += [_target_ios_logo]


################################################################################
# Host Machine Specifics
################################################################################
if PLATFORM == 'ios'
    MAIN_SOURCES += files('godot_ios.mm')
    ios_deps = dependency('AppleFrameworks', modules: ['UIKit',
                                                       #
                                                       ])
    # All Vulkan files commented out.
    ios_sources = files(
        'godot_ios.mm',
        'os_ios.mm',
        'main.m',
        'app_delegate.mm',
        'view_controller.mm',
        'ios.mm',
        'vulkan_context_ios.mm',
        'display_server_ios.mm',
        #'joypad_ios.mm',
        'godot_view.mm',
        'display_layer.mm',
        'godot_app_delegate.m',
        'godot_view_renderer.mm',
        'godot_view_gesture_recognizer.mm',
        'device_metrics.m',
        'keyboard_input_view.mm',
    )

    SOURCES += ios_sources
    CPP_ARGS += ['-DIOS_ENABLED', '-DUNIX_ENABLED', '-DCOREAUDIO_ENABLED', '-DVULKAN_ENABLED']
    OBJCPP_ARGS += ['-fobjc-arc', '-DIOS_ENABLED', '-DUNIX_ENABLED', '-DCOREAUDIO_ENABLED', '-DVULKAN_ENABLED']
    DEPENDENCIES += ios_deps
endif
