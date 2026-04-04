/**************************************************************************/
/*  editor_property.hpp                                                   */
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

#include <godot_cpp/classes/container.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Control;
class Object;
class Variant;

class EditorProperty : public Container {
	GDEXTENSION_CLASS(EditorProperty, Container)

public:
	void set_label(const String &p_text);
	String get_label() const;
	void set_read_only(bool p_read_only);
	bool is_read_only() const;
	void set_draw_label(bool p_draw_label);
	bool is_draw_label() const;
	void set_draw_background(bool p_draw_background);
	bool is_draw_background() const;
	void set_checkable(bool p_checkable);
	bool is_checkable() const;
	void set_checked(bool p_checked);
	bool is_checked() const;
	void set_draw_warning(bool p_draw_warning);
	bool is_draw_warning() const;
	void set_keying(bool p_keying);
	bool is_keying() const;
	void set_deletable(bool p_deletable);
	bool is_deletable() const;
	StringName get_edited_property() const;
	Object *get_edited_object();
	void update_property();
	void add_focusable(Control *p_control);
	void set_bottom_editor(Control *p_editor);
	void set_selectable(bool p_selectable);
	bool is_selectable() const;
	void set_use_folding(bool p_use_folding);
	bool is_using_folding() const;
	void set_name_split_ratio(float p_ratio);
	float get_name_split_ratio() const;
	void deselect();
	bool is_selected() const;
	void select(int32_t p_focusable = -1);
	void set_object_and_property(Object *p_object, const StringName &p_property);
	void set_label_reference(Control *p_control);
	void emit_changed(const StringName &p_property, const Variant &p_value, const StringName &p_field = StringName(), bool p_changing = false);
	virtual void _update_property();
	virtual void _set_read_only(bool p_read_only);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Container::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_update_property), decltype(&T::_update_property)>) {
			BIND_VIRTUAL_METHOD(T, _update_property, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_read_only), decltype(&T::_set_read_only)>) {
			BIND_VIRTUAL_METHOD(T, _set_read_only, 2586408642);
		}
	}

public:
};

} // namespace godot

