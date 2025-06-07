#ifdef WEB_ENABLED
	mono_user_dir = "user://";
	api_assemblies_dir = "res://.godot/mono/publish/wasm";
#else
	mono_user_dir = _get_mono_user_dir();
#endif 