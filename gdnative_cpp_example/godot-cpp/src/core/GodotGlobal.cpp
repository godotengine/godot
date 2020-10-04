#include "GodotGlobal.hpp"

#include "String.hpp"

#include "Wrapped.hpp"

static GDCALLINGCONV void *wrapper_create(void *data, const void *type_tag, godot_object *instance) {
	godot::_Wrapped *wrapper_memory = (godot::_Wrapped *)godot::api->godot_alloc(sizeof(godot::_Wrapped));

	if (!wrapper_memory)
		return NULL;
	wrapper_memory->_owner = instance;
	wrapper_memory->_type_tag = (size_t)type_tag;

	return (void *)wrapper_memory;
}

static GDCALLINGCONV void wrapper_destroy(void *data, void *wrapper) {
	if (wrapper)
		godot::api->godot_free(wrapper);
}

namespace godot {

void *_RegisterState::nativescript_handle;
int _RegisterState::language_index;

const godot_gdnative_core_api_struct *api = nullptr;
const godot_gdnative_core_1_1_api_struct *core_1_1_api = nullptr;
const godot_gdnative_core_1_2_api_struct *core_1_2_api = nullptr;

const godot_gdnative_ext_nativescript_api_struct *nativescript_api = nullptr;
const godot_gdnative_ext_nativescript_1_1_api_struct *nativescript_1_1_api = nullptr;
const godot_gdnative_ext_pluginscript_api_struct *pluginscript_api = nullptr;
const godot_gdnative_ext_android_api_struct *android_api = nullptr;
const godot_gdnative_ext_arvr_api_struct *arvr_api = nullptr;
const godot_gdnative_ext_videodecoder_api_struct *videodecoder_api = nullptr;
const godot_gdnative_ext_net_api_struct *net_api = nullptr;
const godot_gdnative_ext_net_3_2_api_struct *net_3_2_api = nullptr;

const void *gdnlib = NULL;

void Godot::print(const String &message) {
	godot::api->godot_print((godot_string *)&message);
}

void Godot::print_warning(const String &description, const String &function, const String &file, int line) {
	int len;

	char *c_desc = description.alloc_c_string();
	char *c_func = function.alloc_c_string();
	char *c_file = file.alloc_c_string();

	if (c_desc != nullptr && c_func != nullptr && c_file != nullptr) {
		godot::api->godot_print_warning(c_desc, c_func, c_file, line);
	};

	if (c_desc != nullptr) godot::api->godot_free(c_desc);
	if (c_func != nullptr) godot::api->godot_free(c_func);
	if (c_file != nullptr) godot::api->godot_free(c_file);
}

void Godot::print_error(const String &description, const String &function, const String &file, int line) {
	int len;

	char *c_desc = description.alloc_c_string();
	char *c_func = function.alloc_c_string();
	char *c_file = file.alloc_c_string();

	if (c_desc != nullptr && c_func != nullptr && c_file != nullptr) {
		godot::api->godot_print_error(c_desc, c_func, c_file, line);
	};

	if (c_desc != nullptr) godot::api->godot_free(c_desc);
	if (c_func != nullptr) godot::api->godot_free(c_func);
	if (c_file != nullptr) godot::api->godot_free(c_file);
}

void ___register_types();
void ___init_method_bindings();

void Godot::gdnative_init(godot_gdnative_init_options *options) {
	godot::api = options->api_struct;
	godot::gdnlib = options->gd_native_library;

	const godot_gdnative_api_struct *core_extension = godot::api->next;

	while (core_extension) {
		if (core_extension->version.major == 1 && core_extension->version.minor == 1) {
			godot::core_1_1_api = (const godot_gdnative_core_1_1_api_struct *)core_extension;
		} else if (core_extension->version.major == 1 && core_extension->version.minor == 2) {
			godot::core_1_2_api = (const godot_gdnative_core_1_2_api_struct *)core_extension;
		}
		core_extension = core_extension->next;
	}

	// now find our extensions
	for (int i = 0; i < godot::api->num_extensions; i++) {
		switch (godot::api->extensions[i]->type) {
			case GDNATIVE_EXT_NATIVESCRIPT: {
				godot::nativescript_api = (const godot_gdnative_ext_nativescript_api_struct *)godot::api->extensions[i];

				const godot_gdnative_api_struct *extension = godot::nativescript_api->next;

				while (extension) {
					if (extension->version.major == 1 && extension->version.minor == 1) {
						godot::nativescript_1_1_api = (const godot_gdnative_ext_nativescript_1_1_api_struct *)extension;
					}

					extension = extension->next;
				}
			} break;
			case GDNATIVE_EXT_PLUGINSCRIPT: {
				godot::pluginscript_api = (const godot_gdnative_ext_pluginscript_api_struct *)godot::api->extensions[i];
			} break;
			case GDNATIVE_EXT_ANDROID: {
				godot::android_api = (const godot_gdnative_ext_android_api_struct *)godot::api->extensions[i];
			} break;
			case GDNATIVE_EXT_ARVR: {
				godot::arvr_api = (const godot_gdnative_ext_arvr_api_struct *)godot::api->extensions[i];
			} break;
			case GDNATIVE_EXT_VIDEODECODER: {
				godot::videodecoder_api = (const godot_gdnative_ext_videodecoder_api_struct *)godot::api->extensions[i];
			} break;
			case GDNATIVE_EXT_NET: {
				godot::net_api = (const godot_gdnative_ext_net_api_struct *)godot::api->extensions[i];

				const godot_gdnative_api_struct *extension = godot::net_api->next;

				while (extension) {
					if (extension->version.major == 3 && extension->version.minor == 2) {
						godot::net_3_2_api = (const godot_gdnative_ext_net_3_2_api_struct *)extension;
					}

					extension = extension->next;
				}
			} break;

			default: break;
		}
	}

	// Initialize the `language_index` here since `__register_types()` makes use of it.
	godot_instance_binding_functions binding_funcs = {};
	binding_funcs.alloc_instance_binding_data = wrapper_create;
	binding_funcs.free_instance_binding_data = wrapper_destroy;

	godot::_RegisterState::language_index = godot::nativescript_1_1_api->godot_nativescript_register_instance_binding_data_functions(binding_funcs);

	// register these now
	___register_types();
	___init_method_bindings();
}

void Godot::gdnative_terminate(godot_gdnative_terminate_options *options) {
	// reserved for future use.
}

void Godot::gdnative_profiling_add_data(const char *p_signature, uint64_t p_time) {
	godot::nativescript_1_1_api->godot_nativescript_profiling_add_data(p_signature, p_time);
}

void Godot::nativescript_init(void *handle) {
	godot::_RegisterState::nativescript_handle = handle;
}

void Godot::nativescript_terminate(void *handle) {
	godot::nativescript_1_1_api->godot_nativescript_unregister_instance_binding_data_functions(godot::_RegisterState::language_index);
}

} // namespace godot
