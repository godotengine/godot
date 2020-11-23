if get_option('builtin_miniupnpc')
    _builtin_miniupnpc_srcs = files([
        'miniupnpc/miniupnpc.c',
        'miniupnpc/upnpcommands.c',
        'miniupnpc/miniwget.c',
        'miniupnpc/upnpdev.c',
        'miniupnpc/igd_desc_parse.c',
        'miniupnpc/minissdpc.c',
        'miniupnpc/minisoap.c',
        'miniupnpc/minixml.c',
        'miniupnpc/connecthostport.c',
        'miniupnpc/receivedata.c',
        'miniupnpc/portlistingparse.c',
        'miniupnpc/upnpreplyparse.c',
    ])

    _builtin_miniupnpc_c_args = ['-DMINIUPNPC_SET_SOCKET_TIMEOUT']
    _builtin_miniupnpc_dep_compile_args = ['-DMINIUPNP_STATICLIB']

    if meson.get_compiler('c').get_id() == 'msvc'
        _builtin_miniupnpc_c_args += '/wd4133'
    elif meson.get_compiler('c').get_id() == 'gcc'
        _builtin_miniupnpc_c_args += '-std=gnu99'
    endif

    lib_builtin_miniupnpc = static_library('builtin_miniupnpc', _builtin_miniupnpc_srcs,
        c_args: [_builtin_miniupnpc_c_args, _builtin_miniupnpc_dep_compile_args],
        build_by_default: false
    )

    DEP_MINIUPNPC = declare_dependency(link_with: lib_builtin_miniupnpc,
        compile_args: _builtin_miniupnpc_dep_compile_args,
        include_directories: include_directories('.')
    )
else
    _dep_miniupnpc = dependency('miniupnpc', required: false)
    if _dep_miniupnpc.found()
        DEP_MINIUPNPC = _dep_miniupnpc
    else
        DEP_MINIUPNPC = declare_dependency(
            link_with: meson.get_compiler('cpp').find_library('miniupnpc'),
            include_directories: include_directories('/usr/include/miniupnpc')
        )
    endif
endif