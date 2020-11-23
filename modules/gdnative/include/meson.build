# generated api structs
_gdnative_api_structs_header_target = custom_target(
    'gndative_api_structs_header',
    input: ['../gdnative_api.json'],
    output: [ 'gdnative_api_struct.gen.h'],
    command: [SCRIPT_GDNATIVE_API_GEN, '@INPUT@', 'gen_header', '@OUTPUT@'],
    build_by_default: true
)
module_gdnative_srcs += [_gdnative_api_structs_header_target]