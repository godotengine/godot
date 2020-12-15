# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
_target_iphone_logo = custom_target(
    'iphone_logo',
    output: ['logo.gen.h'],
    input: ['logo.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'iphone'],
    build_by_default: true
)
SOURCES += [_target_iphone_logo]


################################################################################
# Host Machine Specifics
################################################################################
if PLATFORM == 'iphone'
    MAIN_SOURCES += files('godot_iphone.mm')
    ios_deps = dependency('AppleFrameworks', modules: ['UIKit',
                                                       #
                                                       ])
    # All Vulkan files commented out.
    iphone_sources = files(
        'godot_iphone.mm',
        'os_iphone.mm',
        'main.m',
        'app_delegate.mm',
        'view_controller.mm',
        'ios.mm',
        'vulkan_context_iphone.mm',
        'display_server_iphone.mm',
        #'joypad_iphone.mm',
        'godot_view.mm',
        'display_layer.mm',
        'godot_app_delegate.m',
        'godot_view_renderer.mm',
        'godot_view_gesture_recognizer.mm',
        'device_metrics.m',
        'keyboard_input_view.mm',
    )

    SOURCES += iphone_sources
    CPP_ARGS += ['-DIPHONE_ENABLED', '-DUNIX_ENABLED', '-DCOREAUDIO_ENABLED', '-DVULKAN_ENABLED']
    OBJCPP_ARGS += ['-fobjc-arc', '-DIPHONE_ENABLED', '-DUNIX_ENABLED', '-DCOREAUDIO_ENABLED', '-DVULKAN_ENABLED']
    DEPENDENCIES += ios_deps
endif
