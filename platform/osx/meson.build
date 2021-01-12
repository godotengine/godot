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
    error('osx not implemented yet')
endif