/**************************************************************************/
/*  editor_inspector_plugin.cpp                                           */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/editor_inspector_plugin.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

namespace godot {

void EditorInspectorPlugin::add_custom_control(Control *p_control) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInspectorPlugin::get_class_static()._native_ptr(), StringName("add_custom_control")._native_ptr(), 1496901182);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_control != nullptr ? &p_control->_owner : nullptr));
}

void EditorInspectorPlugin::add_property_editor(const String &p_property, Control *p_editor, bool p_add_to_end, const String &p_label) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInspectorPlugin::get_class_static()._native_ptr(), StringName("add_property_editor")._native_ptr(), 2042698479);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_add_to_end_encoded;
	PtrToArg<bool>::encode(p_add_to_end, &p_add_to_end_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_property, (p_editor != nullptr ? &p_editor->_owner : nullptr), &p_add_to_end_encoded, &p_label);
}

void EditorInspectorPlugin::add_property_editor_for_multiple_properties(const String &p_label, const PackedStringArray &p_properties, Control *p_editor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorInspectorPlugin::get_class_static()._native_ptr(), StringName("add_property_editor_for_multiple_properties")._native_ptr(), 788598683);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label, &p_properties, (p_editor != nullptr ? &p_editor->_owner : nullptr));
}

bool EditorInspectorPlugin::_can_handle(Object *p_object) const {
	return false;
}

void EditorInspectorPlugin::_parse_begin(Object *p_object) {}

void EditorInspectorPlugin::_parse_category(Object *p_object, const String &p_category) {}

void EditorInspectorPlugin::_parse_group(Object *p_object, const String &p_group) {}

bool EditorInspectorPlugin::_parse_property(Object *p_object, Variant::Type p_type, const String &p_name, PropertyHint p_hint_type, const String &p_hint_string, BitField<PropertyUsageFlags> p_usage_flags, bool p_wide) {
	return false;
}

void EditorInspectorPlugin::_parse_end(Object *p_object) {}

} // namespace godot
