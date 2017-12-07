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

const String init_symbol = "gdnative_init";
const String terminate_symbol = "gdnative_terminate";
const String default_symbol_prefix = "godot_";

// Defined in gdnative_api_struct.gen.cpp
extern const godot_gdnative_core_api_struct api_struct;

Map<String, Vector<Ref<GDNative> > > *GDNativeLibrary::loaded_libraries = NULL;

GDNativeLibrary::GDNativeLibrary() {
	config_file.instance();

	symbol_prefix = default_symbol_prefix;

	if (GDNativeLibrary::loaded_libraries == NULL) {
		GDNativeLibrary::loaded_libraries = memnew((Map<String, Vector<Ref<GDNative> > >));
	}
}

GDNativeLibrary::~GDNativeLibrary() {
}

void GDNativeLibrary::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_config_file"), &GDNativeLibrary::get_config_file);

	ClassDB::bind_method(D_METHOD("get_current_library_path"), &GDNativeLibrary::get_current_library_path);
	ClassDB::bind_method(D_METHOD("get_current_dependencies"), &GDNativeLibrary::get_current_dependencies);

	ClassDB::bind_method(D_METHOD("should_load_once"), &GDNativeLibrary::should_load_once);
	ClassDB::bind_method(D_METHOD("is_singleton"), &GDNativeLibrary::is_singleton);
	ClassDB::bind_method(D_METHOD("get_symbol_prefix"), &GDNativeLibrary::get_symbol_prefix);

	ClassDB::bind_method(D_METHOD("set_load_once", "load_once"), &GDNativeLibrary::set_load_once);
	ClassDB::bind_method(D_METHOD("set_singleton", "singleton"), &GDNativeLibrary::set_singleton);
	ClassDB::bind_method(D_METHOD("set_symbol_prefix", "symbol_prefix"), &GDNativeLibrary::set_symbol_prefix);

	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "load_once"), "set_load_once", "should_load_once");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "singleton"), "set_singleton", "is_singleton");
	ADD_PROPERTYNZ(PropertyInfo(Variant::STRING, "symbol_prefix"), "set_symbol_prefix", "get_symbol_prefix");
}

GDNative::GDNative() {
	native_handle = NULL;
	initialized = false;
}

GDNative::~GDNative() {
}

void GDNative::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_library", "library"), &GDNative::set_library);
	ClassDB::bind_method(D_METHOD("get_library"), &GDNative::get_library);

	ClassDB::bind_method(D_METHOD("initialize"), &GDNative::initialize);
	ClassDB::bind_method(D_METHOD("terminate"), &GDNative::terminate);

	ClassDB::bind_method(D_METHOD("call_native", "calling_type", "procedure_name", "arguments"), &GDNative::call_native);

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

extern "C" void _gdnative_report_version_mismatch(const godot_object *p_library, const char *p_ext, godot_gdnative_api_version p_want, godot_gdnative_api_version p_have);
extern "C" void _gdnative_report_loading_error(const godot_object *p_library, const char *p_what);

bool GDNative::initialize() {
	if (library.is_null()) {
		ERR_PRINT("No library set, can't initialize GDNative object");
		return false;
	}

	String lib_path = library->get_current_library_path();
	if (lib_path.empty()) {
		ERR_PRINT("No library set for this platform");
		return false;
	}
#ifdef IPHONE_ENABLED
	// on iOS we use static linking
	String path = "";
#elif defined(ANDROID_ENABLED)
	// On Android dynamic libraries are located separately from resource assets,
	// we should pass library name to dlopen(). The library name is flattened
	// during export.
	String path = lib_path.get_file();
#else
	String path = ProjectSettings::get_singleton()->globalize_path(lib_path);
#endif

	if (library->should_load_once()) {
		if (GDNativeLibrary::loaded_libraries->has(lib_path)) {
			// already loaded. Don't load again.
			// copy some of the stuff instead
			this->native_handle = (*GDNativeLibrary::loaded_libraries)[lib_path][0]->native_handle;
			initialized = true;
			return true;
		}
	}

	Error err = OS::get_singleton()->open_dynamic_library(path, native_handle, true);
	if (err != OK) {
		return false;
	}

	void *library_init;

	// we cheat here a little bit. you saw nothing
	initialized = true;

	err = get_symbol(library->get_symbol_prefix() + init_symbol, library_init, false);

	initialized = false;

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
	options.report_version_mismatch = &_gdnative_report_version_mismatch;
	options.report_loading_error = &_gdnative_report_loading_error;
	options.gd_native_library = (godot_object *)(get_library().ptr());
	options.active_library_path = (godot_string *)&path;

	library_init_fpointer(&options);

	initialized = true;

	if (library->should_load_once() && !GDNativeLibrary::loaded_libraries->has(lib_path)) {
		Vector<Ref<GDNative> > gdnatives;
		gdnatives.resize(1);
		gdnatives[0] = Ref<GDNative>(this);
		GDNativeLibrary::loaded_libraries->insert(lib_path, gdnatives);
	}

	return true;
}

bool GDNative::terminate() {

	if (!initialized) {
		ERR_PRINT("No valid library handle, can't terminate GDNative object");
		return false;
	}

	if (library->should_load_once()) {
		Vector<Ref<GDNative> > *gdnatives = &(*GDNativeLibrary::loaded_libraries)[library->get_current_library_path()];
		if (gdnatives->size() > 1) {
			// there are other GDNative's still using this library, so we actually don't terminte
			gdnatives->erase(Ref<GDNative>(this));
			initialized = false;
			return true;
		} else if (gdnatives->size() == 1) {
			// we're the last one, terminate!
			gdnatives->clear();
			// wew this looks scary, but all it does is remove the entry completely
			GDNativeLibrary::loaded_libraries->erase(GDNativeLibrary::loaded_libraries->find(library->get_current_library_path()));
		}
	}

	void *library_terminate;
	Error error = get_symbol(library->get_symbol_prefix() + terminate_symbol, library_terminate);
	if (error || !library_terminate) {
		OS::get_singleton()->close_dynamic_library(native_handle);
		native_handle = NULL;
		return true;
	}

	godot_gdnative_terminate_fn library_terminate_pointer;
	library_terminate_pointer = (godot_gdnative_terminate_fn)library_terminate;

	godot_gdnative_terminate_options options;
	options.in_editor = Engine::get_singleton()->is_editor_hint();

	library_terminate_pointer(&options);

	initialized = false;

	// GDNativeScriptLanguage::get_singleton()->initialized_libraries.erase(p_native_lib->path);

	OS::get_singleton()->close_dynamic_library(native_handle);
	native_handle = NULL;

	return true;
}

bool GDNative::is_initialized() {
	return initialized;
}

void GDNativeCallRegistry::register_native_call_type(StringName p_call_type, native_call_cb p_callback) {
	native_calls.insert(p_call_type, p_callback);
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

Variant GDNative::call_native(StringName p_native_call_type, StringName p_procedure_name, Array p_arguments) {

	Map<StringName, native_call_cb>::Element *E = GDNativeCallRegistry::singleton->native_calls.find(p_native_call_type);
	if (!E) {
		ERR_PRINT((String("No handler for native call type \"" + p_native_call_type) + "\" found").utf8().get_data());
		return Variant();
	}

	void *procedure_handle;

	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			native_handle,
			p_procedure_name,
			procedure_handle);

	if (err != OK || procedure_handle == NULL) {
		return Variant();
	}

	godot_variant result = E->get()(procedure_handle, (godot_array *)&p_arguments);

	return *(Variant *)&result;
}

Error GDNative::get_symbol(StringName p_procedure_name, void *&r_handle, bool p_optional) {

	if (!initialized) {
		ERR_PRINT("No valid library handle, can't get symbol from GDNative object");
		return ERR_CANT_OPEN;
	}

	Error result = OS::get_singleton()->get_dynamic_library_symbol_handle(
			native_handle,
			p_procedure_name,
			r_handle,
			p_optional);

	return result;
}

RES GDNativeLibraryResourceLoader::load(const String &p_path, const String &p_original_path, Error *r_error) {
	Ref<GDNativeLibrary> lib;
	lib.instance();

	Ref<ConfigFile> config = lib->get_config_file();

	Error err = config->load(p_path);

	if (r_error) {
		*r_error = err;
	}

	lib->set_singleton(config->get_value("general", "singleton", false));
	lib->set_load_once(config->get_value("general", "load_once", true));
	lib->set_symbol_prefix(config->get_value("general", "symbol_prefix", default_symbol_prefix));

	String entry_lib_path;
	{

		List<String> entry_keys;
		config->get_section_keys("entry", &entry_keys);

		for (List<String>::Element *E = entry_keys.front(); E; E = E->next()) {
			String key = E->get();

			Vector<String> tags = key.split(".");

			bool skip = false;
			for (int i = 0; i < tags.size(); i++) {
				bool has_feature = OS::get_singleton()->has_feature(tags[i]);

				if (!has_feature) {
					skip = true;
					break;
				}
			}

			if (skip) {
				continue;
			}

			entry_lib_path = config->get_value("entry", key);
			break;
		}
	}

	Vector<String> dependency_paths;
	{

		List<String> dependency_keys;
		config->get_section_keys("dependencies", &dependency_keys);

		for (List<String>::Element *E = dependency_keys.front(); E; E = E->next()) {
			String key = E->get();

			Vector<String> tags = key.split(".");

			bool skip = false;
			for (int i = 0; i < tags.size(); i++) {
				bool has_feature = OS::get_singleton()->has_feature(tags[i]);

				if (!has_feature) {
					skip = true;
					break;
				}
			}

			if (skip) {
				continue;
			}

			dependency_paths = config->get_value("dependencies", key);
			break;
		}
	}

	lib->current_library_path = entry_lib_path;
	lib->current_dependencies = dependency_paths;

	return lib;
}

void GDNativeLibraryResourceLoader::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gdnlib");
}

bool GDNativeLibraryResourceLoader::handles_type(const String &p_type) const {
	return p_type == "GDNativeLibrary";
}

String GDNativeLibraryResourceLoader::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "gdnlib")
		return "GDNativeLibrary";
	return "";
}

Error GDNativeLibraryResourceSaver::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {

	Ref<GDNativeLibrary> lib = p_resource;

	if (lib.is_null()) {
		return ERR_INVALID_DATA;
	}

	Ref<ConfigFile> config = lib->get_config_file();

	config->set_value("general", "singleton", lib->is_singleton());
	config->set_value("general", "load_once", lib->should_load_once());
	config->set_value("general", "symbol_prefix", lib->get_symbol_prefix());

	return config->save(p_path);
}

bool GDNativeLibraryResourceSaver::recognize(const RES &p_resource) const {
	return Object::cast_to<GDNativeLibrary>(*p_resource) != NULL;
}

void GDNativeLibraryResourceSaver::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<GDNativeLibrary>(*p_resource) != NULL) {
		p_extensions->push_back("gdnlib");
	}
}
