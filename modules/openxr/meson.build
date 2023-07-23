openxr_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
openxr_srcs += files(run_command(GLOB, 'editor/*.cpp', check: true).stdout().split('\n'))
openxr_srcs += files(run_command(GLOB, 'scene/*.cpp', check: true).stdout().split('\n'))
openxr_srcs += files(run_command(GLOB, 'action_map/*.cpp', check: true).stdout().split('\n'))
openxr_srcs += files(
     'extensions/openxr_composition_layer_depth_extension.cpp',
     'extensions/openxr_fb_display_refresh_rate_extension.cpp',
     'extensions/openxr_fb_passthrough_extension_wrapper.cpp',
     'extensions/openxr_extension_wrapper_extension.cpp',
     'extensions/openxr_hand_tracking_extension.cpp',
     'extensions/openxr_htc_controller_extension.cpp',
     'extensions/openxr_htc_vive_tracker_extension.cpp',
     'extensions/openxr_huawei_controller_extension.cpp',
     'extensions/openxr_ml2_controller_extension.cpp',
     'extensions/openxr_opengl_extension.cpp',
     'extensions/openxr_palm_pose_extension.cpp',
     'extensions/openxr_pico_controller_extension.cpp',
     'extensions/openxr_vulkan_extension.cpp',
     'extensions/openxr_wmr_controller_extension.cpp',
)

lib_module_openxr = library('module_openxr',
    openxr_srcs,
    SHADER_HEADERS,
    cpp_args: ['-DNOMINMAX'],
    include_directories: ['extensions'],
    dependencies: [godotcore_dep, DEP_OPENXR, DEP_VULKAN]
)

DEP_MODULE_OPENXR = declare_dependency(link_with: lib_module_openxr)

MODULE_DEPENDENCIES += DEP_MODULE_OPENXR
MODULES_ENABLED += 'openxr'
