_module_visual_script_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))
lib_module_visual_script = library('module_visual_script',
    _module_visual_script_srcs,
    gdivirtuals_gen,
    include_directories: '../..',
    dependencies: [DEP_GODOT_ALL])

DEP_MODULE_VISUAL_SCRIPT = declare_dependency(link_with: lib_module_visual_script)

MODULE_DEPENDENCIES += DEP_MODULE_VISUAL_SCRIPT
