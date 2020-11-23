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
    error('iphone not implemented yet')
endif