# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
_target_android_logo = custom_target(
    'android_logo',
    output: ['logo.gen.h'],
    input: ['logo.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'android'],
    build_by_default: true
)
SOURCES += [_target_android_logo]

_target_android_run_icon = custom_target(
    'android_run_icon',
    output: ['run_icon.gen.h'],
    input: ['run_icon.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'android'],
    build_by_default: true
)
SOURCES += [_target_android_run_icon]


################################################################################
# Host Machine Specifics
################################################################################
if PLATFORM == 'android'
    error('android not implemented yet')
endif