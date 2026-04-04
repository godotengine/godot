/**************************************************************************/
/*  option_button.hpp                                                     */
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

#include <godot_cpp/classes/button.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PopupMenu;
class Texture2D;

class OptionButton : public Button {
	GDEXTENSION_CLASS(OptionButton, Button)

public:
	void add_item(const String &p_label, int32_t p_id = -1);
	void add_icon_item(const Ref<Texture2D> &p_texture, const String &p_label, int32_t p_id = -1);
	void set_item_text(int32_t p_idx, const String &p_text);
	void set_item_icon(int32_t p_idx, const Ref<Texture2D> &p_texture);
	void set_item_disabled(int32_t p_idx, bool p_disabled);
	void set_item_id(int32_t p_idx, int32_t p_id);
	void set_item_metadata(int32_t p_idx, const Variant &p_metadata);
	void set_item_tooltip(int32_t p_idx, const String &p_tooltip);
	void set_item_auto_translate_mode(int32_t p_idx, Node::AutoTranslateMode p_mode);
	String get_item_text(int32_t p_idx) const;
	Ref<Texture2D> get_item_icon(int32_t p_idx) const;
	int32_t get_item_id(int32_t p_idx) const;
	int32_t get_item_index(int32_t p_id) const;
	Variant get_item_metadata(int32_t p_idx) const;
	String get_item_tooltip(int32_t p_idx) const;
	Node::AutoTranslateMode get_item_auto_translate_mode(int32_t p_idx) const;
	bool is_item_disabled(int32_t p_idx) const;
	bool is_item_separator(int32_t p_idx) const;
	void add_separator(const String &p_text = String());
	void clear();
	void select(int32_t p_idx);
	int32_t get_selected() const;
	int32_t get_selected_id() const;
	Variant get_selected_metadata() const;
	void remove_item(int32_t p_idx);
	PopupMenu *get_popup() const;
	void show_popup();
	void set_item_count(int32_t p_count);
	int32_t get_item_count() const;
	bool has_selectable_items() const;
	int32_t get_selectable_item(bool p_from_last = false) const;
	void set_fit_to_longest_item(bool p_fit);
	bool is_fit_to_longest_item() const;
	void set_allow_reselect(bool p_allow);
	bool get_allow_reselect() const;
	void set_disable_shortcuts(bool p_disabled);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Button::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

