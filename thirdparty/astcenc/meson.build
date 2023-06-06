astcenc_lib = static_library('astcenc',
                          files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n')),
                          cpp_args: [],
                          build_by_default: false)

DEP_ASTCENC = declare_dependency(link_with: astcenc_lib, include_directories: '.')
