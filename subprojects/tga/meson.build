project('tgamodule', 'cpp')

godotcore_dep = dependency('godotcore')

subdir('modules/tga')

module_dep = declare_dependency(link_with: lib_module_tga,
  include_directories: '.',
  variables: {'module_names': 'tga'})
