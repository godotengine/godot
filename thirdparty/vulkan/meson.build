if get_option('builtin_vulkan')
    _vulkan_lib_incdirs = include_directories('.', 'include')

    _vulkan_deps = []
    _vulkan_link_args = []
    # CPP flags that we need to link against in godot
    _vulkan_cpp_dep_args = []
    # CPP flags for building the vulkan library
    _vulkan_cpp_args = ['-DVULKAN_NON_CMAKE_BUILD']

    _vulkan_srcs = files(
        'vk_mem_alloc.cpp',
    )

    if PLATFORM == 'windows'
        _vulkan_deps += meson.get_compiler('cpp').find_library('cfgmgr32')
        _vulkan_cpp_dep_args += ['-DVK_USE_PLATFORM_WIN32_KHR', '-DAPI_NAME="Vulkan"']
        _vulkan_cpp_args += ['-DWIN32_LEAN_AND_MEAN']

        if _cpp_compiler_id != 'msvc'
            _vulkan_cpp_args += ['-DCM_GETIDLIST_FILTER_CLASS=0x00000200', '-DCM_GETIDLIST_FILTER_PRESENT=0x00000100']
        endif
    elif PLATFORM == 'osx'
        _vulkan_cpp_dep_args += ['-DVK_USE_PLATFORM_MACOS_MVK']
        _vulkan_cpp_args += [
            '-DSYSCONFDIR="/etc"',
            '-DFALLBACK_DATA_DIRS="/usr/local/share:/usr/share"',
            '-DFALLBACK_CONFIG_DIRS="/etc/xdg"'
        ]
    elif PLATFORM == 'iphone'
        _vulkan_cpp_dep_args += ['-DVK_USE_PLATFORM_IOS_MVK']
        _vulkan_cpp_args += [
            '-DSYSCONFDIR="/etc"',
            '-DFALLBACK_DATA_DIRS="/usr/local/share:/usr/share"',
            '-DFALLBACK_CONFIG_DIRS="/etc/xdg"'
        ]
    elif PLATFORM == 'linuxbsd'
        _vulkan_cpp_dep_args += ['-DVK_USE_PLATFORM_XLIB_KHR']
        _vulkan_cpp_args += [
            '-DSYSCONFDIR="/etc"',
            '-DFALLBACK_DATA_DIRS="/usr/local/share:/usr/share"',
            '-DFALLBACK_CONFIG_DIRS="/etc/xdg"',
            '-DHAVE_SECURE_GETENV',
            '-D_DEFAULT_SOURCE',
        ]
    elif PLATFORM == 'android'
        _vulkan_srcs = []
    endif

    if _cpp_compiler_id == 'gcc'
        _vulkan_cpp_args += '-Wno-stringop-overflow'
    endif


    #if PLATFORM == 'windows'
    #    _vulkan_srcs += files('loader/dirent_on_windows.c')
    #endif

    _vulkan_incdirs = include_directories('include', '.')

    # force cpp language, even though its c files (for pragma once)
    _vulkan_lib = static_library('vulkan', _vulkan_srcs,
        include_directories: _vulkan_lib_incdirs, dependencies: _vulkan_deps,
        link_args: _vulkan_link_args,
        cpp_args: [_vulkan_cpp_args, _vulkan_cpp_dep_args],
        c_args: [_vulkan_cpp_args, _vulkan_cpp_dep_args],
        build_by_default: false)

    DEP_VULKAN = declare_dependency(link_with: _vulkan_lib, include_directories: _vulkan_incdirs, compile_args: _vulkan_cpp_dep_args)
else
    DEP_VULKAN = dependency('vulkan')
endif
