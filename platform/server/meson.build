# The libraries and defines added in this file are the *requirements* that are
# needed to build on the selected platform. Do not add debug/release/optional
# compilation flags here.

################################################################################
# Commands to always run
################################################################################
_target_server_logo = custom_target(
    'server_logo',
    output: ['logo.gen.h'],
    input: ['logo.png'],
    command: [SCRIPT_PLATFORM_LOGO, '@INPUT@', '@OUTPUT@', 'server'],
    build_by_default: true
)
SOURCES += [_target_server_logo]


################################################################################
# Host Machine Specifics
################################################################################
if PLATFORM == 'server'
    error('server not implemented yet')
endif