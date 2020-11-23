_module_visual_script_srcs = files([
    'register_types.cpp',
    'visual_script.cpp',
    'visual_script_builtin_funcs.cpp',
    'visual_script_editor.cpp',
    'visual_script_expression.cpp',
    'visual_script_flow_control.cpp',
    'visual_script_func_nodes.cpp',
    'visual_script_nodes.cpp',
    'visual_script_property_selector.cpp',
    'visual_script_yield_nodes.cpp',
])

lib_module_visual_script = library('module_visual_script', _module_visual_script_srcs, dependencies: [DEP_GODOT_ALL])

DEP_MODULE_VISUAL_SCRIPT = declare_dependency(link_with: lib_module_visual_script)

MODULE_DEPENDENCIES += DEP_MODULE_VISUAL_SCRIPT