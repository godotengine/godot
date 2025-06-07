#ifdef WEB_ENABLED
	// For web platform, we need to set up the Mono runtime differently
	// The runtime is statically linked in this case
	print_verbose("Mono: Static linking for web platform.");
	runtime_initialized = true;
	_domain = nullptr;

	mono_jit_init("godot");
	
	_load_assemblies(p_resource_dir_path);

	return true;
#endif 