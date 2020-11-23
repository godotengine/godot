if PLATFORM == 'android'
    # TODO
    error('android Vulkan library not implemented')

elif PLATFORM == 'iphone'
    # TODO
    error('iphone Vulkan library not implemented')

elif get_option('builtin_vulkan')
    _vulkan_lib_incdirs = include_directories('.', 'include', 'loader')

    _vulkan_deps = []
    _vulkan_link_args = []
    # CPP flags that we need to link against in godot
    _vulkan_cpp_dep_args = []
    # CPP flags for building the vulkan library
    _vulkan_cpp_args = ['-DVULKAN_NON_CMAKE_BUILD']

    _cpp_compiler = meson.get_compiler('cpp').get_id()

    if PLATFORM == 'windows'
        _vulkan_deps += meson.get_compiler('cpp').find_library('cfgmgr32')
        _vulkan_cpp_dep_args += ['-DVK_USE_PLATFORM_WIN32_KHR', '-DAPI_NAME="Vulkan"']
        _vulkan_cpp_args += ['-DWIN32_LEAN_AND_MEAN']
        
        if _cpp_compiler != 'msvc'
            _vulkan_cpp_args += ['-DCM_GETIDLIST_FILTER_CLASS=0x00000200', '-DCM_GETIDLIST_FILTER_PRESENT=0x00000100']
        endif
    elif PLATFORM == 'osx'
        _vulkan_cpp_dep_args += ['-DVK_USE_PLATFORM_MACOS_MVK']
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
    endif

    
    if _cpp_compiler_id == 'gcc'
        _vulkan_cpp_args += '-Wno-stringop-overflow'
    endif

    _vulkan_srcs = [
        'vk_mem_alloc.cpp',
        #'loader/asm_offset.c',
        'loader/cJSON.c',
        'loader/debug_utils.c',
        'loader/dev_ext_trampoline.c',
        'loader/extension_manual.c',
        'loader/loader.c',
        'loader/murmurhash.c',
        'loader/phys_dev_ext.c',
        'loader/trampoline.c',
        'loader/unknown_ext_chain.c',
        # 'loader/vk_loader_extensions.c', # do not include this file, its included in a header directly
        'loader/wsi.c'
    ]

    if PLATFORM == 'windows'
        _vulkan_srcs += files(['loader/dirent_on_windows.c'])
    endif

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
