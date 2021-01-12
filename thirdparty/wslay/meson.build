if get_option('builtin_wslay')
    _builtin_wslay_srcs = files([
        'wslay_event.c',
        'wslay_frame.c',
        'wslay_net.c',
        'wslay_queue.c',
        'wslay_stack.c',
    ])

    _builtin_wslay_incdirs = include_directories('includes')

    _builtin_wslay_compile_args = ['-DHAVE_CONFIG_H']

    if PLATFORM == 'windows' or PLATFORM == 'uwp'
        _builtin_wslay_compile_args += '-DHAVE_WINSOCK2_H'
    else
        _builtin_wslay_compile_args += '-DHAVE_NETINET_IN_H'
    endif

    lib_builtin_wslay = static_library('builtin_wslay', _builtin_wslay_srcs,
        include_directories: _builtin_wslay_incdirs,
        c_args: _builtin_wslay_compile_args,
        build_by_default: false
    )

    DEP_WSLAY = declare_dependency(link_with: lib_builtin_wslay,
        include_directories: _builtin_wslay_incdirs,
        compile_args: _builtin_wslay_compile_args
    )
else
    DEP_WSLAY = dependency('libwslay')
endif