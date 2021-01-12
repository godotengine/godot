if get_option('builtin_enet')
    # builtin enet uses files directly from godot source.... and platform folder....
    _enet_incdirs = include_directories('.', '..' / '..', '..' / '..' / DIR_PLATFORM)
    _enet_incdirs_pub = include_directories('.')

    _enet_srcs = files([
        'godot.cpp',
        'callbacks.c',
        'compress.c',
        'host.c',
        'list.c',
        'packet.c',
        'peer.c',
        'protocol.c',
    ])

    _enet_compile_dep_defines = ['-DGODOT_ENET']

    if PLATFORM == 'windows'
        _enet_compile_dep_defines += '-DGODOT_ENET_WINDOWS_ENABLED'
    else
        _enet_compile_dep_defines += '-DGODOT_ENET_UNIX_ENABLED'
    endif

    _lib_enet = static_library('builtin_enet', _enet_srcs,
        include_directories: _enet_incdirs,
        cpp_args: _enet_compile_dep_defines,
        c_args: _enet_compile_dep_defines,
        build_by_default: false
    )

    DEP_ENET = declare_dependency(link_with: _lib_enet, compile_args: _enet_compile_dep_defines, include_directories: _enet_incdirs_pub)
else
    DEP_ENET = dependency('libenet')
endif