
_platform_apis = [
    'android',
    'javascript'
]

_register_platform_apis_target = custom_target(
    'register_platform_apis',
    output: ['register_platform_apis.gen.cpp'],
    command: [SCRIPT_REGISTER_PLATFORM_APIS, _platform_apis, '@OUTPUT@'],
    build_by_default: true
)
SOURCES += [_register_platform_apis_target]

# platform apis
SOURCES += files([
    'android/api/api.cpp',
    'javascript/api/api.cpp'
])

subdir('android')
subdir('iphone')
subdir('javascript')
subdir('linuxbsd')
subdir('osx')
subdir('server')
subdir('uwp')
subdir('windows')