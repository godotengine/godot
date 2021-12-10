/*************************************************************************/
/*  native_extension_manager.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "native_extension_manager.h"
#include "core/io/file_access.h"

NativeExtensionManager::LoadStatus NativeExtensionManager::load_extension(const String &p_path) {
	if (native_extension_map.has(p_path)) {
		return LOAD_STATUS_ALREADY_LOADED;
	}
	Ref<NativeExtension> extension = ResourceLoader::load(p_path);
	if (extension.is_null()) {
		return LOAD_STATUS_FAILED;
	}

	if (level >= 0) { //already initialized up to some level
		int32_t minimum_level = extension->get_minimum_library_initialization_level();
		if (minimum_level < MIN(level, NativeExtension::INITIALIZATION_LEVEL_SCENE)) {
			return LOAD_STATUS_NEEDS_RESTART;
		}
		//initialize up to current level
		for (int32_t i = minimum_level; i < level; i++) {
			extension->initialize_library(NativeExtension::InitializationLevel(level));
		}
	}
	native_extension_map[p_path] = extension;
	return LOAD_STATUS_OK;
}

NativeExtensionManager::LoadStatus NativeExtensionManager::reload_extension(const String &p_path) {
	return LOAD_STATUS_OK; //TODO
}
NativeExtensionManager::LoadStatus NativeExtensionManager::unload_extension(const String &p_path) {
	if (!native_extension_map.has(p_path)) {
		return LOAD_STATUS_NOT_LOADED;
	}

	Ref<NativeExtension> extension = native_extension_map[p_path];

	if (level >= 0) { //already initialized up to some level
		int32_t minimum_level = extension->get_minimum_library_initialization_level();
		if (minimum_level < MIN(level, NativeExtension::INITIALIZATION_LEVEL_SCENE)) {
			return LOAD_STATUS_NEEDS_RESTART;
		}
		//initialize up to current level
		for (int32_t i = level; i >= minimum_level; i--) {
			extension->deinitialize_library(NativeExtension::InitializationLevel(level));
		}
	}
	native_extension_map.erase(p_path);
	return LOAD_STATUS_OK;
}

bool NativeExtensionManager::is_extension_loaded(const String &p_path) const {
	return native_extension_map.has(p_path);
}

Vector<String> NativeExtensionManager::get_loaded_extensions() const {
	Vector<String> ret;
	for (const KeyValue<String, Ref<NativeExtension>> &E : native_extension_map) {
		ret.push_back(E.key);
	}
	return ret;
}
Ref<NativeExtension> NativeExtensionManager::get_extension(const String &p_path) {
	Map<String, Ref<NativeExtension>>::Element *E = native_extension_map.find(p_path);
	ERR_FAIL_COND_V(!E, Ref<NativeExtension>());
	return E->get();
}

void NativeExtensionManager::initialize_extensions(NativeExtension::InitializationLevel p_level) {
	ERR_FAIL_COND(int32_t(p_level) - 1 != level);
	for (KeyValue<String, Ref<NativeExtension>> &E : native_extension_map) {
		E.value->initialize_library(p_level);
	}
	level = p_level;
}

void NativeExtensionManager::deinitialize_extensions(NativeExtension::InitializationLevel p_level) {
	ERR_FAIL_COND(int32_t(p_level) != level);
	for (KeyValue<String, Ref<NativeExtension>> &E : native_extension_map) {
		E.value->deinitialize_library(p_level);
	}
	level = int32_t(p_level) - 1;
}

void NativeExtensionManager::load_extensions() {
	FileAccessRef f = FileAccess::open(NativeExtension::get_extension_list_config_file(), FileAccess::READ);
	while (f && !f->eof_reached()) {
		String s = f->get_line().strip_edges();
		if (!s.is_empty()) {
			LoadStatus err = load_extension(s);
			ERR_CONTINUE_MSG(err == LOAD_STATUS_FAILED, "Error loading extension: " + s);
		}
	}
}

NativeExtensionManager *NativeExtensionManager::get_singleton() {
	return singleton;
}
void NativeExtensionManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_extension", "path"), &NativeExtensionManager::load_extension);
	ClassDB::bind_method(D_METHOD("reload_extension", "path"), &NativeExtensionManager::reload_extension);
	ClassDB::bind_method(D_METHOD("unload_extension", "path"), &NativeExtensionManager::unload_extension);
	ClassDB::bind_method(D_METHOD("is_extension_loaded", "path"), &NativeExtensionManager::is_extension_loaded);

	ClassDB::bind_method(D_METHOD("get_loaded_extensions"), &NativeExtensionManager::get_loaded_extensions);
	ClassDB::bind_method(D_METHOD("get_extension", "path"), &NativeExtensionManager::get_extension);

	BIND_ENUM_CONSTANT(LOAD_STATUS_OK);
	BIND_ENUM_CONSTANT(LOAD_STATUS_FAILED);
	BIND_ENUM_CONSTANT(LOAD_STATUS_ALREADY_LOADED);
	BIND_ENUM_CONSTANT(LOAD_STATUS_NOT_LOADED);
	BIND_ENUM_CONSTANT(LOAD_STATUS_NEEDS_RESTART);
}

NativeExtensionManager *NativeExtensionManager::singleton = nullptr;

NativeExtensionManager::NativeExtensionManager() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}
