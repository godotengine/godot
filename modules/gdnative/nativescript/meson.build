module_gdnative_srcs += files([
    'api_generator.cpp',
    'godot_nativescript.cpp',
    'nativescript.cpp',
    'register_types.cpp'
])

if PLATFORM == 'linuxbsd' or PLATFORM == 'iphone'
    module_gdnative_link_args += ['-rdynamic']
endif