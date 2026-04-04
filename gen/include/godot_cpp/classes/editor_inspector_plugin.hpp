/**************************************************************************/
/*  editor_inspector_plugin.hpp                                           */
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

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Control;
class Object;
class PackedStringArray;

class EditorInspectorPlugin : public RefCounted {
	GDEXTENSION_CLASS(EditorInspectorPlugin, RefCounted)

public:
	void add_custom_control(Control *p_control);
	void add_property_editor(const String &p_property, Control *p_editor, bool p_add_to_end = false, const String &p_label = String());
	void add_property_editor_for_multiple_properties(const String &p_label, const PackedStringArray &p_properties, Control *p_editor);
	virtual bool _can_handle(Object *p_object) const;
	virtual void _parse_begin(Object *p_object);
	virtual void _parse_category(Object *p_object, const String &p_category);
	virtual void _parse_group(Object *p_object, const String &p_group);
	virtual bool _parse_property(Object *p_object, Variant::Type p_type, const String &p_name, PropertyHint p_hint_type, const String &p_hint_string, BitField<PropertyUsageFlags> p_usage_flags, bool p_wide);
	virtual void _parse_end(Object *p_object);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_can_handle), decltype(&T::_can_handle)>) {
			BIND_VIRTUAL_METHOD(T, _can_handle, 397768994);
		}
		if constexpr (!std::is_same_v<decltype(&B::_parse_begin), decltype(&T::_parse_begin)>) {
			BIND_VIRTUAL_METHOD(T, _parse_begin, 3975164845);
		}
		if constexpr (!std::is_same_v<decltype(&B::_parse_category), decltype(&T::_parse_category)>) {
			BIND_VIRTUAL_METHOD(T, _parse_category, 357144787);
		}
		if constexpr (!std::is_same_v<decltype(&B::_parse_group), decltype(&T::_parse_group)>) {
			BIND_VIRTUAL_METHOD(T, _parse_group, 357144787);
		}
		if constexpr (!std::is_same_v<decltype(&B::_parse_property), decltype(&T::_parse_property)>) {
			BIND_VIRTUAL_METHOD(T, _parse_property, 1087679910);
		}
		if constexpr (!std::is_same_v<decltype(&B::_parse_end), decltype(&T::_parse_end)>) {
			BIND_VIRTUAL_METHOD(T, _parse_end, 3975164845);
		}
	}

public:
};

} // namespace godot

