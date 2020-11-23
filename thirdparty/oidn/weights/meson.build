_oidn_resource_to_cpp = find_program('resource_to_cpp.py')

_oidn_lightmap_target = custom_target(
    'oidn_lightmap',
    output: ['rtlightmap_hdr.gen.cpp'],
    input: ['rtlightmap_hdr.tza'],
    command: [_oidn_resource_to_cpp, '@INPUT@', '@OUTPUT@'],
    build_by_default: false
)

oidn_srcs += [_oidn_lightmap_target]