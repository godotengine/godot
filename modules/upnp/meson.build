_module_upnp_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))
lib_module_upnp = library('module_upnp', _module_upnp_srcs, dependencies: [DEP_GODOT_ALL, DEP_MINIUPNPC])

DEP_MODULE_UPNP = declare_dependency(link_with: lib_module_upnp)

MODULE_DEPENDENCIES += DEP_MODULE_UPNP
