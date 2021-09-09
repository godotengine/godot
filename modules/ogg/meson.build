ogg_sources = files(run_command(GLOB, '*.cpp').stdout().split('\n'))

ogg_lib = library('ogg',
                  ogg_sources,
                  gdivirtuals_gen,
                  include_directories: [INCDIRS],
                  cpp_args: CPP_ARGS,
                  dependencies: DEP_LIBOGG)

DEP_MODULE_OGG = declare_dependency(link_with: ogg_lib)

MODULE_DEPENDENCIES += DEP_MODULE_OGG
