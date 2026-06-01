/**************************************************************************/
/*  gdextension_resource_format.cpp                                       */
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

#include "gdextension_resource_format.h"

#include "core/extension/gdextension_manager.h"
#include "core/object/class_db.h"

Ref<Resource> GDExtensionResourceLoader::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	// We can't have two GDExtension resource object representing the same library, because
	// loading (or unloading) a GDExtension affects global data. So, we need reuse the same
	// object if one has already been loaded (even if caching is disabled at the resource
	// loader level).
	GDExtensionManager *manager = GDExtensionManager::get_singleton();
	if (manager->is_extension_loaded(p_path)) {
		return manager->get_extension(p_path);
	}

	Ref<GDExtension> lib;
	Error err = load_gdextension_resource(p_path, lib);
	if (err != OK && r_error) {
		// Errors already logged in load_gdextension_resource().
		*r_error = err;
	}
	return lib;
}

void GDExtensionResourceLoader::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gdextension");
}

bool GDExtensionResourceLoader::handles_type(const String &p_type) const {
	return p_type == "GDExtension";
}

String GDExtensionResourceLoader::get_resource_type(const String &p_path) const {
	if (p_path.has_extension("gdextension")) {
		return "GDExtension";
	}
	return "";
}

Error GDExtensionResourceLoader::load_gdextension_resource(const String &p_path, Ref<GDExtension> &p_extension) {
	ERR_FAIL_COND_V_MSG(p_extension.is_valid() && p_extension->is_library_open(), ERR_ALREADY_IN_USE, "Cannot load GDExtension resource into already opened library.");

	GDExtensionManager *extension_manager = GDExtensionManager::get_singleton();

	GDExtensionManager::LoadStatus status = extension_manager->load_extension(p_path);
	if (status != GDExtensionManager::LOAD_STATUS_OK && status != GDExtensionManager::LOAD_STATUS_ALREADY_LOADED) {
		// Errors already logged in load_extension().
		return FAILED;
	}

	p_extension = extension_manager->get_extension(p_path);
	return OK;
}

#ifdef TOOLS_ENABLED
void GDExtensionResourceLoader::get_classes_used(const String &p_path, HashSet<StringName> *r_classes) {
	Ref<GDExtension> gdext = ResourceLoader::load(p_path);
	if (gdext.is_null()) {
		return;
	}

	for (const StringName class_name : gdext->get_classes_used()) {
		if (ClassDB::class_exists(class_name)) {
			r_classes->insert(class_name);
		}
	}
}
#endif // TOOLS_ENABLED
