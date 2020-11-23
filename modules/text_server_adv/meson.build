module_tsa_srcs = files([
    'bitmap_font_adv.cpp',
    'dynamic_font_adv.cpp',
    'register_types.cpp',
    'script_iterator.cpp',
    'text_server_adv.cpp'
])

subdir('icu_data')

_module_tsa_compile_args = []
if get_option('tools')
    _module_tsa_compile_args += '-DICU_STATIC_DATA'
endif

lib_module_tsa = library('module_text_server_adv', module_tsa_srcs,
    dependencies: [DEP_GODOT_ALL, DEP_HARFBUZZ, DEP_ICU, DEP_FREETYPE, DEP_GRAPHITE],
    cpp_args: _module_tsa_compile_args
)

DEP_MODULE_TEXT_SERVER_ADV = declare_dependency(link_with: lib_module_tsa)

MODULE_DEPENDENCIES += DEP_MODULE_TEXT_SERVER_ADV