_regex_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_lib_module_regex = library('module_regex', _regex_srcs, dependencies: [DEP_GODOT_ALL, DEP_PCRE2])

DEP_MODULE_REGEX = declare_dependency(link_with: _lib_module_regex)

MODULE_DEPENDENCIES += [DEP_MODULE_REGEX]
