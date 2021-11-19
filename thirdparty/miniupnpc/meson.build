if get_option('builtin_miniupnpc')
    _builtin_miniupnpc_srcs = files(
        'src/addr_is_reserved.c',
        'src/miniupnpc.c',
        'src/upnpcommands.c',
        'src/miniwget.c',
        'src/upnpdev.c',
        'src/igd_desc_parse.c',
        'src/minissdpc.c',
        'src/minisoap.c',
        'src/minixml.c',
        'src/connecthostport.c',
        'src/receivedata.c',
        'src/portlistingparse.c',
        'src/upnpreplyparse.c',
    )

    _builtin_miniupnpc_c_args = ['-DMINIUPNPC_SET_SOCKET_TIMEOUT']
    _builtin_miniupnpc_dep_compile_args = ['-DMINIUPNP_STATICLIB']

    if meson.get_compiler('c').get_id() == 'msvc'
        _builtin_miniupnpc_c_args += '/wd4133'
    elif meson.get_compiler('c').get_id() == 'gcc'
        _builtin_miniupnpc_c_args += '-std=gnu99'
    elif meson.get_compiler('c').get_id() == 'emscripten'
        _builtin_miniupnpc_c_args += '-std=gnu99'
    endif

    lib_builtin_miniupnpc = static_library('builtin_miniupnpc',
        _builtin_miniupnpc_srcs,
#        _version_target,
        c_args: [_builtin_miniupnpc_c_args, _builtin_miniupnpc_dep_compile_args],
        include_directories: ['include', '../..'],
        build_by_default: false
    )

    DEP_MINIUPNPC = declare_dependency(link_with: lib_builtin_miniupnpc,
        compile_args: _builtin_miniupnpc_dep_compile_args,
        include_directories: include_directories('include')
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
