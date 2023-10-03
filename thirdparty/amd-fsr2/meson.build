if cpp.get_id() == 'msvc'
    fsr_vis_args = []
else
    fsr_vis_args = ['-DFFX_GCC']
endif

lib_fsr2 = static_library('amdfsr2',
    'ffx_assert.cpp',
    'ffx_fsr2.cpp',
    cpp_args: fsr_vis_args
    #include_directories: 'include',
)

fsr2_dep = declare_dependency(link_with: lib_fsr2,
                              #include_directories: 'include',
                              compile_args: fsr_vis_args,
                              )

DEPENDENCIES += fsr2_dep
