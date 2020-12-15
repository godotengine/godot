module_tsa_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))

subdir('icu_data')

_module_tsa_compile_args = ['-DMODULE_FREETYPE_ENABLED']
if get_option('tools')
    _module_tsa_compile_args += '-DICU_STATIC_DATA'
endif

lib_module_tsa = library('module_text_server_adv', module_tsa_srcs,
    dependencies: [DEP_GODOT_ALL, DEP_HARFBUZZ, DEP_ICU, DEP_FREETYPE, DEP_GRAPHITE, DEP_MSDF],
    cpp_args: _module_tsa_compile_args
)

DEP_MODULE_TEXT_SERVER_ADV = declare_dependency(link_with: lib_module_tsa)

MODULE_DEPENDENCIES += DEP_MODULE_TEXT_SERVER_ADV
