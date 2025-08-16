/**************************************************************************/
/*  editor_plugin_registration.cpp                                        */
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

#include <godot_cpp/classes/editor_plugin_registration.hpp>

#include <godot_cpp/variant/variant.hpp>

namespace godot {

Vector<StringName> EditorPlugins::plugin_classes;

void EditorPlugins::add_plugin_class(const StringName &p_class_name) {
	ERR_FAIL_COND_MSG(plugin_classes.find(p_class_name) != -1, vformat("Editor plugin already registered: %s", p_class_name));
	plugin_classes.push_back(p_class_name);
	internal::gdextension_interface_editor_add_plugin(p_class_name._native_ptr());
}

void EditorPlugins::remove_plugin_class(const StringName &p_class_name) {
	int index = plugin_classes.find(p_class_name);
	ERR_FAIL_COND_MSG(index == -1, vformat("Editor plugin is not registered: %s", p_class_name));
	plugin_classes.remove_at(index);
	internal::gdextension_interface_editor_remove_plugin(p_class_name._native_ptr());
}

void EditorPlugins::deinitialize(GDExtensionInitializationLevel p_level) {
	if (p_level == GDEXTENSION_INITIALIZATION_EDITOR) {
		for (const StringName &class_name : plugin_classes) {
			internal::gdextension_interface_editor_remove_plugin(class_name._native_ptr());
		}
		plugin_classes.clear();
	}
}

} // namespace godot
