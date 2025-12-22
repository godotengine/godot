/**************************************************************************/
/*  extension_source_code_manager.cpp                                     */
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

#include "extension_source_code_manager.h"

int ExtensionSourceCodeManager::get_plugin_count() const {
	return plugins.size();
}

const Ref<EditorExtensionSourceCodePlugin> &ExtensionSourceCodeManager::get_plugin_at_index(int p_index) const {
	return plugins[p_index];
}

void ExtensionSourceCodeManager::add_plugin(const Ref<EditorExtensionSourceCodePlugin> &p_plugin) {
	ERR_FAIL_COND_MSG(plugins.find(p_plugin) != -1, "Plugin already registered.");
	plugins.push_back(p_plugin);
}

void ExtensionSourceCodeManager::remove_plugin(const Ref<EditorExtensionSourceCodePlugin> &p_plugin) {
	int idx = plugins.find(p_plugin);
	ERR_FAIL_COND_MSG(idx == -1, "Plugin not registered.");
	plugins.remove_at(idx);
}

const Ref<EditorExtensionSourceCodePlugin> ExtensionSourceCodeManager::get_plugin_for_object(const Object *p_object) const {
	ERR_FAIL_NULL_V(p_object, nullptr);

	const GDExtension *library = p_object->get_extension_library();
	if (library == nullptr) {
		// Only objects that are GDExtension classes can be handled by source code plugins.
		return nullptr;
	}

	for (const Ref<EditorExtensionSourceCodePlugin> &plugin : plugins) {
		if (plugin->can_handle_object(p_object)) {
			return plugin;
		}
	}

	return nullptr;
}

const Ref<EditorExtensionSourceCodePlugin> ExtensionSourceCodeManager::get_plugin_for_file(const String &p_source_path) const {
	ERR_FAIL_COND_V(p_source_path.is_empty(), nullptr);

	for (const Ref<EditorExtensionSourceCodePlugin> &plugin : plugins) {
		StringName class_name = plugin->get_class_name_from_source_path(p_source_path);
		if (!class_name.is_empty()) {
			return plugin;
		}
	}

	return nullptr;
}

bool ExtensionSourceCodeManager::has_plugins_that_can_create_class_source() {
	for (const Ref<EditorExtensionSourceCodePlugin> &plugin : plugins) {
		if (plugin->can_create_class_source()) {
			return true;
		}
	}
	return false;
}

void ExtensionSourceCodeManager::create() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = memnew(ExtensionSourceCodeManager);
}

void ExtensionSourceCodeManager::cleanup() {
	ERR_FAIL_NULL(singleton);
	memdelete(singleton);
	singleton = nullptr;
}
