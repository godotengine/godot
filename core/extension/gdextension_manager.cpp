/**************************************************************************/
/*  gdextension_manager.cpp                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "gdextension_manager.h"

#include "core/extension/gdextension_compat_hashes.h"
#include "core/io/file_access.h"
#include "core/object/script_language.h"

GDExtensionManager::LoadStatus GDExtensionManager::_load_extension_internal(const Ref<GDExtension> &p_extension) {
	if (level >= 0) { // Already initialized up to some level.
		int32_t minimum_level = p_extension->get_minimum_library_initialization_level();
		if (minimum_level < MIN(level, GDExtension::INITIALIZATION_LEVEL_SCENE)) {
			return LOAD_STATUS_NEEDS_RESTART;
		}
		// Initialize up to current level.
		for (int32_t i = minimum_level; i <= level; i++) {
			p_extension->initialize_library(GDExtension::InitializationLevel(i));
		}
	}

	for (const KeyValue<String, String> &kv : p_extension->class_icon_paths) {
		gdextension_class_icon_paths[kv.key] = kv.value;
	}

	return LOAD_STATUS_OK;
}

GDExtensionManager::LoadStatus GDExtensionManager::_unload_extension_internal(const Ref<GDExtension> &p_extension) {
	if (level >= 0) { // Already initialized up to some level.
		// Deinitialize down from current level.
		for (int32_t i = level; i >= GDExtension::INITIALIZATION_LEVEL_CORE; i--) {
			p_extension->deinitialize_library(GDExtension::InitializationLevel(i));
		}
	}

	for (const KeyValue<String, String> &kv : p_extension->class_icon_paths) {
		gdextension_class_icon_paths.erase(kv.key);
	}

	return LOAD_STATUS_OK;
}

GDExtensionManager::LoadStatus GDExtensionManager::load_extension(const String &p_path) {
	if (gdextension_map.has(p_path)) {
		return LOAD_STATUS_ALREADY_LOADED;
	}
	Ref<GDExtension> extension = ResourceLoader::load(p_path);
	if (extension.is_null()) {
		return LOAD_STATUS_FAILED;
	}

	LoadStatus status = _load_extension_internal(extension);
	if (status != LOAD_STATUS_OK) {
		return status;
	}

	gdextension_map[p_path] = extension;
	return LOAD_STATUS_OK;
}

GDExtensionManager::LoadStatus GDExtensionManager::reload_extension(const String &p_path) {
#ifndef TOOLS_ENABLED
	ERR_FAIL_V_MSG(LOAD_STATUS_FAILED, "GDExtensions can only be reloaded in an editor build.");
#else
	ERR_FAIL_COND_V_MSG(!Engine::get_singleton()->is_extension_reloading_enabled(), LOAD_STATUS_FAILED, "GDExtension reloading is disabled.");

	if (!gdextension_map.has(p_path)) {
		return LOAD_STATUS_NOT_LOADED;
	}

	Ref<GDExtension> extension = gdextension_map[p_path];
	ERR_FAIL_COND_V_MSG(!extension->is_reloadable(), LOAD_STATUS_FAILED, vformat("This GDExtension is not marked as 'reloadable' or doesn't support reloading: %s.", p_path));

	LoadStatus status;

	extension->prepare_reload();

	// Unload library if it's open. It may not be open if the developer made a
	// change that broke loading in a previous hot-reload attempt.
	if (extension->is_library_open()) {
		status = _unload_extension_internal(extension);
		if (status != LOAD_STATUS_OK) {
			// We need to clear these no matter what.
			extension->clear_instance_bindings();
			return status;
		}

		extension->clear_instance_bindings();
		extension->close_library();
	}

	Error err = GDExtensionResourceLoader::load_gdextension_resource(p_path, extension);
	if (err != OK) {
		return LOAD_STATUS_FAILED;
	}

	status = _load_extension_internal(extension);
	if (status != LOAD_STATUS_OK) {
		return status;
	}

	extension->finish_reload();

	return LOAD_STATUS_OK;
#endif
}

GDExtensionManager::LoadStatus GDExtensionManager::unload_extension(const String &p_path) {
	if (!gdextension_map.has(p_path)) {
		return LOAD_STATUS_NOT_LOADED;
	}

	Ref<GDExtension> extension = gdextension_map[p_path];

	LoadStatus status = _unload_extension_internal(extension);
	if (status != LOAD_STATUS_OK) {
		return status;
	}

	gdextension_map.erase(p_path);
	return LOAD_STATUS_OK;
}

bool GDExtensionManager::is_extension_loaded(const String &p_path) const {
	return gdextension_map.has(p_path);
}

Vector<String> GDExtensionManager::get_loaded_extensions() const {
	Vector<String> ret;
	for (const KeyValue<String, Ref<GDExtension>> &E : gdextension_map) {
		ret.push_back(E.key);
	}
	return ret;
}
Ref<GDExtension> GDExtensionManager::get_extension(const String &p_path) {
	HashMap<String, Ref<GDExtension>>::Iterator E = gdextension_map.find(p_path);
	ERR_FAIL_COND_V(!E, Ref<GDExtension>());
	return E->value;
}

bool GDExtensionManager::class_has_icon_path(const String &p_class) const {
	// TODO: Check that the icon belongs to a registered class somehow.
	return gdextension_class_icon_paths.has(p_class);
}

String GDExtensionManager::class_get_icon_path(const String &p_class) const {
	// TODO: Check that the icon belongs to a registered class somehow.
	if (gdextension_class_icon_paths.has(p_class)) {
		return gdextension_class_icon_paths[p_class];
	}
	return "";
}

void GDExtensionManager::initialize_extensions(GDExtension::InitializationLevel p_level) {
	ERR_FAIL_COND(int32_t(p_level) - 1 != level);
	for (KeyValue<String, Ref<GDExtension>> &E : gdextension_map) {
		E.value->initialize_library(p_level);
	}
	level = p_level;
}

void GDExtensionManager::deinitialize_extensions(GDExtension::InitializationLevel p_level) {
	ERR_FAIL_COND(int32_t(p_level) != level);
	for (KeyValue<String, Ref<GDExtension>> &E : gdextension_map) {
		E.value->deinitialize_library(p_level);
	}
	level = int32_t(p_level) - 1;
}

#ifdef TOOLS_ENABLED
void GDExtensionManager::track_instance_binding(void *p_token, Object *p_object) {
	for (KeyValue<String, Ref<GDExtension>> &E : gdextension_map) {
		if (E.value.ptr() == p_token) {
			if (E.value->is_reloadable()) {
				E.value->track_instance_binding(p_object);
				return;
			}
		}
	}
}

void GDExtensionManager::untrack_instance_binding(void *p_token, Object *p_object) {
	for (KeyValue<String, Ref<GDExtension>> &E : gdextension_map) {
		if (E.value.ptr() == p_token) {
			if (E.value->is_reloadable()) {
				E.value->untrack_instance_binding(p_object);
				return;
			}
		}
	}
}

void GDExtensionManager::_reload_all_scripts() {
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->reload_all_scripts();
	}
}
#endif // TOOLS_ENABLED

void GDExtensionManager::load_extensions() {
	Ref<FileAccess> f = FileAccess::open(GDExtension::get_extension_list_config_file(), FileAccess::READ);
	while (f.is_valid() && !f->eof_reached()) {
		String s = f->get_line().strip_edges();
		if (!s.is_empty()) {
			LoadStatus err = load_extension(s);
			ERR_CONTINUE_MSG(err == LOAD_STATUS_FAILED, "Error loading extension: " + s);
		}
	}

	OS::get_singleton()->load_platform_gdextensions();
}

void GDExtensionManager::reload_extensions() {
#ifdef TOOLS_ENABLED
	bool reloaded = false;
	for (const KeyValue<String, Ref<GDExtension>> &E : gdextension_map) {
		if (!E.value->is_reloadable()) {
			continue;
		}

		if (E.value->has_library_changed()) {
			reloaded = true;
			reload_extension(E.value->get_path());
		}
	}

	if (reloaded) {
		emit_signal("extensions_reloaded");

		// Reload all scripts to clear out old references.
		callable_mp_static(&GDExtensionManager::_reload_all_scripts).call_deferred();
	}
#endif
}

GDExtensionManager *GDExtensionManager::get_singleton() {
	return singleton;
}

void GDExtensionManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_extension", "path"), &GDExtensionManager::load_extension);
	ClassDB::bind_method(D_METHOD("reload_extension", "path"), &GDExtensionManager::reload_extension);
	ClassDB::bind_method(D_METHOD("unload_extension", "path"), &GDExtensionManager::unload_extension);
	ClassDB::bind_method(D_METHOD("is_extension_loaded", "path"), &GDExtensionManager::is_extension_loaded);

	ClassDB::bind_method(D_METHOD("get_loaded_extensions"), &GDExtensionManager::get_loaded_extensions);
	ClassDB::bind_method(D_METHOD("get_extension", "path"), &GDExtensionManager::get_extension);

	BIND_ENUM_CONSTANT(LOAD_STATUS_OK);
	BIND_ENUM_CONSTANT(LOAD_STATUS_FAILED);
	BIND_ENUM_CONSTANT(LOAD_STATUS_ALREADY_LOADED);
	BIND_ENUM_CONSTANT(LOAD_STATUS_NOT_LOADED);
	BIND_ENUM_CONSTANT(LOAD_STATUS_NEEDS_RESTART);

	ADD_SIGNAL(MethodInfo("extensions_reloaded"));
}

GDExtensionManager *GDExtensionManager::singleton = nullptr;

GDExtensionManager::GDExtensionManager() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;

#ifndef DISABLE_DEPRECATED
	GDExtensionCompatHashes::initialize();
#endif
}

GDExtensionManager::~GDExtensionManager() {
#ifndef DISABLE_DEPRECATED
	GDExtensionCompatHashes::finalize();
#endif
}
