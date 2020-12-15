_oidn_resource_to_cpp = find_program(MESON_SOURCE_ROOT / 'modules' / 'denoise' / 'resource_to_cpp.py').full_path()

_oidn_lightmap_target = custom_target(
    'oidn_lightmap',
    output: ['rtlightmap_hdr.gen.cpp'],
    input: ['rtlightmap_hdr.tza'],
    env: SCRIPTS_ENV,
    command: [SCRIPT_COMPAT, 'tza_to_cpp', _oidn_resource_to_cpp, '-i', '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default: false
)

oidn_srcs += [_oidn_lightmap_target]
