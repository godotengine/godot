/**************************************************************************/
/*  display_if_editor_plugin.cpp                                          */
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

#include "display_if_editor_plugin.h"

#include "editor_properties.h"

bool EditorInspectorDisplayIfPlugin::can_handle(Object *p_object) {
	return true;
}

bool EditorInspectorDisplayIfPlugin::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
	if (!p_hint_text.contains("condition:") || !p_usage.has_flag(PROPERTY_USAGE_EDITOR)) {
		return false;
	}

	Vector<String> slices = p_hint_text.split(",");
	String property_name;

	for (String slice : slices) {
		if (slice.contains("condition:")) {
			property_name = slice.lstrip("condition:");
		}
	}

	Variant value = p_object->get(property_name);

	ERR_FAIL_COND_V_MSG(value.get_type() == Variant::NIL, false, vformat(R"(Unable to obtain property "%s" in "@display_if".)", property_name));
	ERR_FAIL_COND_V_MSG(value.get_type() != Variant::BOOL, false, vformat(R"(The value of property "%s" in "@display_if" is %s, but bool was expected.)", p_hint_text, Variant::get_type_name(value.get_type())));

	const bool condition = value;

	if (!condition) {
		EditorProperty *editor = EditorInspectorDefaultPlugin::get_editor_for_property(p_object, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide);
		editor->hide();
		add_custom_control(editor);
		return true;
	}

	return false;
}

DisplayIfEditorPlugin::DisplayIfEditorPlugin() {
	Ref<EditorInspectorDisplayIfPlugin> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
