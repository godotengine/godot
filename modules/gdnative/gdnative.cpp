/*************************************************************************/
/*  gdnative.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "gdnative.h"

#include "global_constants.h"
#include "io/file_access_encrypted.h"
#include "os/file_access.h"
#include "os/os.h"
#include "project_settings.h"

#include "scene/main/scene_tree.h"

const String init_symbol = "godot_gdnative_init";
const String terminate_symbol = "godot_gdnative_terminate";

#define GDAPI_FUNC(name, ret_type, ...) .name = name,
#define GDAPI_FUNC_VOID(name, ...) .name = name,

const godot_gdnative_api_struct api_struct = {
	GODOT_GDNATIVE_API_FUNCTIONS
};

#undef GDAPI_FUNC
#undef GDAPI_FUNC_VOID

String GDNativeLibrary::platform_names[NUM_PLATFORMS + 1] = {
	"X11_32bit",
	"X11_64bit",
	"Windows_32bit",
	"Windows_64bit",
	"OSX",

	"Android",

	"iOS_32bit",
	"iOS_64bit",

	"WebAssembly",

	""
};
String GDNativeLibrary::platform_lib_ext[NUM_PLATFORMS + 1] = {
	"so",
	"so",
	"dll",
	"dll",
	"dylib",

	"so",

	"dylib",
	"dylib",

	"wasm",

	""
};

GDNativeLibrary::Platform GDNativeLibrary::current_platform =
#if defined(X11_ENABLED)
		(sizeof(void *) == 8 ? X11_64BIT : X11_32BIT);
#elif defined(WINDOWS_ENABLED)
		(sizeof(void *) == 8 ? WINDOWS_64BIT : WINDOWS_32BIT);
#elif defined(OSX_ENABLED)
		OSX;
#elif defined(IPHONE_ENABLED)
		(sizeof(void *) == 8 ? IOS_64BIT : IOS_32BIT);
#elif defined(ANDROID_ENABLED)
		ANDROID;
#elif defined(JAVASCRIPT_ENABLED)
		WASM;
#else
		NUM_PLATFORMS;
#endif

GDNativeLibrary::GDNativeLibrary()
	: library_paths(), singleton_gdnative(false) {
}

GDNativeLibrary::~GDNativeLibrary() {
}

void GDNativeLibrary::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_library_path", "platform", "path"), &GDNativeLibrary::set_library_path);
	ClassDB::bind_method(D_METHOD("get_library_path", "platform"), &GDNativeLibrary::get_library_path);

	ClassDB::bind_method(D_METHOD("is_singleton_gdnative"), &GDNativeLibrary::is_singleton_gdnative);
	ClassDB::bind_method(D_METHOD("set_singleton_gdnative", "singleton"), &GDNativeLibrary::set_singleton_gdnative);

	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "singleton_gdnative"), "set_singleton_gdnative", "is_singleton_gdnative");
}

bool GDNativeLibrary::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	if (name.begins_with("platform/")) {
		set_library_path(name.get_slice("/", 1), p_value);
		return true;
	}
	return false;
}

bool GDNativeLibrary::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	if (name.begins_with("platform/")) {
		r_ret = get_library_path(name.get_slice("/", 1));
		return true;
	}
	return false;
}

void GDNativeLibrary::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < NUM_PLATFORMS; i++) {
		p_list->push_back(PropertyInfo(Variant::STRING,
				"platform/" + platform_names[i],
				PROPERTY_HINT_FILE,
				"*." + platform_lib_ext[i]));
	}
}

void GDNativeLibrary::set_library_path(StringName p_platform, String p_path) {
	int i;
	for (i = 0; i <= NUM_PLATFORMS; i++) {
		if (i == NUM_PLATFORMS) break;
		if (platform_names[i] == p_platform) {
			break;
		}
	}

	if (i == NUM_PLATFORMS) {
		ERR_EXPLAIN(String("No such platform: ") + p_platform);
		ERR_FAIL();
	}

	library_paths[i] = p_path;
}

String GDNativeLibrary::get_library_path(StringName p_platform) const {
	int i;
	for (i = 0; i <= NUM_PLATFORMS; i++) {
		if (i == NUM_PLATFORMS) break;
		if (platform_names[i] == p_platform) {
			break;
		}
	}

	if (i == NUM_PLATFORMS) {
		ERR_EXPLAIN(String("No such platform: ") + p_platform);
		ERR_FAIL_V("");
	}

	return library_paths[i];
}

String GDNativeLibrary::get_active_library_path() const {
	if (GDNativeLibrary::current_platform != NUM_PLATFORMS) {
		return library_paths[GDNativeLibrary::current_platform];
	}
	return "";
}

GDNative::GDNative() {
	native_handle = NULL;
}

GDNative::~GDNative() {
}

extern "C" void _api_anchor();

void GDNative::_compile_dummy_for_api() {
	_api_anchor();
}

void GDNative::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_library", "library"), &GDNative::set_library);
	ClassDB::bind_method(D_METHOD("get_library"), &GDNative::get_library);

	ClassDB::bind_method(D_METHOD("initialize"), &GDNative::initialize);
	ClassDB::bind_method(D_METHOD("terminate"), &GDNative::terminate);

	// TODO(karroffel): get_native_(raw_)call_types binding?

	// TODO(karroffel): make this a varargs function?
	ClassDB::bind_method(D_METHOD("call_native", "procedure_name", "arguments"), &GDNative::call_native);

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "library", PROPERTY_HINT_RESOURCE_TYPE, "GDNativeLibrary"), "set_library", "get_library");
}

void GDNative::set_library(Ref<GDNativeLibrary> p_library) {
	ERR_EXPLAIN("Tried to change library of GDNative when it is already set");
	ERR_FAIL_COND(library.is_valid());
	library = p_library;
}

Ref<GDNativeLibrary> GDNative::get_library() {
	return library;
}

bool GDNative::initialize() {
	if (library.is_null()) {
		ERR_PRINT("No library set, can't initialize GDNative object");
		return false;
	}

	String lib_path = library->get_active_library_path();
	if (lib_path.empty()) {
		ERR_PRINT("No library set for this platform");
		return false;
	}

	String path = ProjectSettings::get_singleton()->globalize_path(lib_path);
	Error err = OS::get_singleton()->open_dynamic_library(path, native_handle);
	if (err != OK) {
		return false;
	}

	void *library_init;
	err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			native_handle,
			init_symbol,
			library_init);

	if (err || !library_init) {
		OS::get_singleton()->close_dynamic_library(native_handle);
		native_handle = NULL;
		ERR_PRINT("Failed to obtain godot_gdnative_init symbol");
		return false;
	}

	godot_gdnative_init_fn library_init_fpointer;
	library_init_fpointer = (godot_gdnative_init_fn)library_init;

	godot_gdnative_init_options options;

	options.api_struct = &api_struct;
	options.in_editor = Engine::get_singleton()->is_editor_hint();
	options.core_api_hash = ClassDB::get_api_hash(ClassDB::API_CORE);
	options.editor_api_hash = ClassDB::get_api_hash(ClassDB::API_EDITOR);
	options.no_api_hash = ClassDB::get_api_hash(ClassDB::API_NONE);
	options.gd_native_library = (godot_object *)(get_library().ptr());

	library_init_fpointer(&options);

	return true;
}

bool GDNative::terminate() {

	if (native_handle == NULL) {
		ERR_PRINT("No valid library handle, can't terminate GDNative object");
		return false;
	}

	void *library_terminate;
	Error error = OS::get_singleton()->get_dynamic_library_symbol_handle(
			native_handle,
			terminate_symbol,
			library_terminate);
	if (error) {
		OS::get_singleton()->close_dynamic_library(native_handle);
		native_handle = NULL;
		return true;
	}

	godot_gdnative_terminate_fn library_terminate_pointer;
	library_terminate_pointer = (godot_gdnative_terminate_fn)library_terminate;

	// TODO(karroffel): remove this? Should be part of NativeScript, not
	// GDNative IMO
	godot_gdnative_terminate_options options;
	options.in_editor = Engine::get_singleton()->is_editor_hint();

	library_terminate_pointer(&options);

	// GDNativeScriptLanguage::get_singleton()->initialized_libraries.erase(p_native_lib->path);

	OS::get_singleton()->close_dynamic_library(native_handle);
	native_handle = NULL;

	return true;
}

bool GDNative::is_initialized() {
	return (native_handle != NULL);
}

void GDNativeCallRegistry::register_native_call_type(StringName p_call_type, native_call_cb p_callback) {
	native_calls.insert(p_call_type, p_callback);
}

void GDNativeCallRegistry::register_native_raw_call_type(StringName p_raw_call_type, native_raw_call_cb p_callback) {
	native_raw_calls.insert(p_raw_call_type, p_callback);
}

Vector<StringName> GDNativeCallRegistry::get_native_call_types() {
	Vector<StringName> call_types;
	call_types.resize(native_calls.size());

	size_t idx = 0;
	for (Map<StringName, native_call_cb>::Element *E = native_calls.front(); E; E = E->next(), idx++) {
		call_types[idx] = E->key();
	}

	return call_types;
}

Vector<StringName> GDNativeCallRegistry::get_native_raw_call_types() {
	Vector<StringName> call_types;
	call_types.resize(native_raw_calls.size());

	size_t idx = 0;
	for (Map<StringName, native_raw_call_cb>::Element *E = native_raw_calls.front(); E; E = E->next(), idx++) {
		call_types[idx] = E->key();
	}

	return call_types;
}

Variant GDNative::call_native(StringName p_native_call_type, StringName p_procedure_name, Array p_arguments) {

	Map<StringName, native_call_cb>::Element *E = GDNativeCallRegistry::singleton->native_calls.find(p_native_call_type);
	if (!E) {
		ERR_PRINT((String("No handler for native call type \"" + p_native_call_type) + "\" found").utf8().get_data());
		return Variant();
	}

	String procedure_name = p_procedure_name;
	godot_variant result = E->get()(native_handle, (godot_string *)&procedure_name, (godot_array *)&p_arguments);

	return *(Variant *)&result;
}

void GDNative::call_native_raw(StringName p_raw_call_type, StringName p_procedure_name, void *data, int num_args, void **args, void *r_return) {

	Map<StringName, native_raw_call_cb>::Element *E = GDNativeCallRegistry::singleton->native_raw_calls.find(p_raw_call_type);
	if (!E) {
		ERR_PRINT((String("No handler for native raw call type \"" + p_raw_call_type) + "\" found").utf8().get_data());
		return;
	}

	String procedure_name = p_procedure_name;
	E->get()(native_handle, (godot_string *)&procedure_name, data, num_args, args, r_return);
}
