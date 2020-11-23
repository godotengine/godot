_module_tsf_srcs = files([
    'bitmap_font_fb.cpp',
    'dynamic_font_fb.cpp',
    'register_types.cpp',
    'text_server_fb.cpp',
])

lib_module_tsf = library('module_text_server_fb', _module_tsf_srcs,
    dependencies: [DEP_GODOT_ALL, DEP_FREETYPE]
)

DEP_MODULE_TEXT_SERVER_FB = declare_dependency(link_with: lib_module_tsf)

MODULE_DEPENDENCIES += DEP_MODULE_TEXT_SERVER_FB