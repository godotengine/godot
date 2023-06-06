noise_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
noise_srcs += files(run_command(GLOB, 'editor/*.cpp', check: true).stdout().split('\n'))

lib_module_noise = library('module_noise',
    noise_srcs,
    dependencies: [godotcore_dep]
)

DEP_MODULE_NOISE = declare_dependency(link_with: lib_module_noise)

MODULE_DEPENDENCIES += DEP_MODULE_NOISE
MODULES_ENABLED += 'noise'
