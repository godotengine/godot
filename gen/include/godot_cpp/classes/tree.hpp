/**************************************************************************/
/*  tree.hpp                                                              */
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

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/tree_item.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Tree : public Control {
	GDEXTENSION_CLASS(Tree, Control)

public:
	enum SelectMode {
		SELECT_SINGLE = 0,
		SELECT_ROW = 1,
		SELECT_MULTI = 2,
	};

	enum DropModeFlags {
		DROP_MODE_DISABLED = 0,
		DROP_MODE_ON_ITEM = 1,
		DROP_MODE_INBETWEEN = 2,
	};

	enum ScrollHintMode {
		SCROLL_HINT_MODE_DISABLED = 0,
		SCROLL_HINT_MODE_BOTH = 1,
		SCROLL_HINT_MODE_TOP = 2,
		SCROLL_HINT_MODE_BOTTOM = 3,
	};

	void clear();
	TreeItem *create_item(TreeItem *p_parent = nullptr, int32_t p_index = -1);
	TreeItem *get_root() const;
	void set_column_custom_minimum_width(int32_t p_column, int32_t p_min_width);
	void set_column_expand(int32_t p_column, bool p_expand);
	void set_column_expand_ratio(int32_t p_column, int32_t p_ratio);
	void set_column_clip_content(int32_t p_column, bool p_enable);
	bool is_column_expanding(int32_t p_column) const;
	bool is_column_clipping_content(int32_t p_column) const;
	int32_t get_column_expand_ratio(int32_t p_column) const;
	int32_t get_column_width(int32_t p_column) const;
	void set_hide_root(bool p_enable);
	bool is_root_hidden() const;
	TreeItem *get_next_selected(TreeItem *p_from);
	TreeItem *get_selected() const;
	void set_selected(TreeItem *p_item, int32_t p_column);
	int32_t get_selected_column() const;
	int32_t get_pressed_button() const;
	void set_select_mode(Tree::SelectMode p_mode);
	Tree::SelectMode get_select_mode() const;
	void deselect_all();
	void set_columns(int32_t p_amount);
	int32_t get_columns() const;
	TreeItem *get_edited() const;
	int32_t get_edited_column() const;
	bool edit_selected(bool p_force_edit = false);
	Rect2 get_custom_popup_rect() const;
	Rect2 get_item_area_rect(TreeItem *p_item, int32_t p_column = -1, int32_t p_button_index = -1) const;
	TreeItem *get_item_at_position(const Vector2 &p_position) const;
	int32_t get_column_at_position(const Vector2 &p_position) const;
	int32_t get_drop_section_at_position(const Vector2 &p_position) const;
	int32_t get_button_id_at_position(const Vector2 &p_position) const;
	void ensure_cursor_is_visible();
	void set_column_titles_visible(bool p_visible);
	bool are_column_titles_visible() const;
	void set_column_title(int32_t p_column, const String &p_title);
	String get_column_title(int32_t p_column) const;
	void set_column_title_tooltip_text(int32_t p_column, const String &p_tooltip_text);
	String get_column_title_tooltip_text(int32_t p_column) const;
	void set_column_title_alignment(int32_t p_column, HorizontalAlignment p_title_alignment);
	HorizontalAlignment get_column_title_alignment(int32_t p_column) const;
	void set_column_title_direction(int32_t p_column, Control::TextDirection p_direction);
	Control::TextDirection get_column_title_direction(int32_t p_column) const;
	void set_column_title_language(int32_t p_column, const String &p_language);
	String get_column_title_language(int32_t p_column) const;
	Vector2 get_scroll() const;
	void scroll_to_item(TreeItem *p_item, bool p_center_on_item = false);
	void set_h_scroll_enabled(bool p_h_scroll);
	bool is_h_scroll_enabled() const;
	void set_v_scroll_enabled(bool p_h_scroll);
	bool is_v_scroll_enabled() const;
	void set_scroll_hint_mode(Tree::ScrollHintMode p_scroll_hint_mode);
	Tree::ScrollHintMode get_scroll_hint_mode() const;
	void set_tile_scroll_hint(bool p_tile_scroll_hint);
	bool is_scroll_hint_tiled();
	void set_hide_folding(bool p_hide);
	bool is_folding_hidden() const;
	void set_enable_recursive_folding(bool p_enable);
	bool is_recursive_folding_enabled() const;
	void set_enable_drag_unfolding(bool p_enable);
	bool is_drag_unfolding_enabled() const;
	void set_drop_mode_flags(int32_t p_flags);
	int32_t get_drop_mode_flags() const;
	void set_allow_rmb_select(bool p_allow);
	bool get_allow_rmb_select() const;
	void set_allow_reselect(bool p_allow);
	bool get_allow_reselect() const;
	void set_allow_search(bool p_allow);
	bool get_allow_search() const;
	void set_auto_tooltip(bool p_enable);
	bool is_auto_tooltip_enabled() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Control::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Tree::SelectMode);
VARIANT_ENUM_CAST(Tree::DropModeFlags);
VARIANT_ENUM_CAST(Tree::ScrollHintMode);

