/**************************************************************************/
/*  editor_inspector_plugin.h                                             */
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

#pragma once

#include "core/object/gdvirtual.gen.h"
#include "core/object/ref_counted.h"

class Control;
class EditorInspector;
class EditorProperty;

class EditorInspectorPlugin : public RefCounted {
	GDCLASS(EditorInspectorPlugin, RefCounted);

public:
	friend class EditorInspector;
	struct AddedEditor {
		Control *property_editor = nullptr;
		Vector<String> properties;
		String label;
		bool add_to_end = false;
	};

	List<AddedEditor> added_editors;

protected:
	static void _bind_methods();

	GDVIRTUAL1RC(bool, _can_handle, Object *)
	GDVIRTUAL1(_parse_begin, Object *)
	GDVIRTUAL2(_parse_category, Object *, String)
	GDVIRTUAL2(_parse_group, Object *, String)
	GDVIRTUAL7R(bool, _parse_property, Object *, Variant::Type, String, PropertyHint, String, BitField<PropertyUsageFlags>, bool)
	GDVIRTUAL1(_parse_end, Object *)

#ifndef DISABLE_DEPRECATED
	void _add_property_editor_bind_compat_92322(const String &p_for_property, Control *p_prop, bool p_add_to_end);
	static void _bind_compatibility_methods();
#endif // DISABLE_DEPRECATED
public:
	void add_custom_control(Control *control);
	void add_property_editor(const String &p_for_property, Control *p_prop, bool p_add_to_end = false, const String &p_label = String());
	void add_property_editor_for_multiple_properties(const String &p_label, const Vector<String> &p_properties, Control *p_prop);

	virtual bool can_handle(Object *p_object);
	virtual void parse_begin(Object *p_object);
	virtual void parse_category(Object *p_object, const String &p_category);
	virtual void parse_group(Object *p_object, const String &p_group);
	virtual bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide = false);
	virtual void parse_end(Object *p_object);
};

class EditorInspectorDefaultPlugin : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorDefaultPlugin, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide = false) override;

	static EditorProperty *get_editor_for_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide = false);
};
