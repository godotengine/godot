/**************************************************************************/
/*  item_list.hpp                                                         */
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
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class HScrollBar;
class VScrollBar;
struct Vector2;

class ItemList : public Control {
	GDEXTENSION_CLASS(ItemList, Control)

public:
	enum IconMode {
		ICON_MODE_TOP = 0,
		ICON_MODE_LEFT = 1,
	};

	enum SelectMode {
		SELECT_SINGLE = 0,
		SELECT_MULTI = 1,
		SELECT_TOGGLE = 2,
	};

	enum ScrollHintMode {
		SCROLL_HINT_MODE_DISABLED = 0,
		SCROLL_HINT_MODE_BOTH = 1,
		SCROLL_HINT_MODE_TOP = 2,
		SCROLL_HINT_MODE_BOTTOM = 3,
	};

	int32_t add_item(const String &p_text, const Ref<Texture2D> &p_icon = nullptr, bool p_selectable = true);
	int32_t add_icon_item(const Ref<Texture2D> &p_icon, bool p_selectable = true);
	void set_item_text(int32_t p_idx, const String &p_text);
	String get_item_text(int32_t p_idx) const;
	void set_item_icon(int32_t p_idx, const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_item_icon(int32_t p_idx) const;
	void set_item_text_direction(int32_t p_idx, Control::TextDirection p_direction);
	Control::TextDirection get_item_text_direction(int32_t p_idx) const;
	void set_item_language(int32_t p_idx, const String &p_language);
	String get_item_language(int32_t p_idx) const;
	void set_item_auto_translate_mode(int32_t p_idx, Node::AutoTranslateMode p_mode);
	Node::AutoTranslateMode get_item_auto_translate_mode(int32_t p_idx) const;
	void set_item_icon_transposed(int32_t p_idx, bool p_transposed);
	bool is_item_icon_transposed(int32_t p_idx) const;
	void set_item_icon_region(int32_t p_idx, const Rect2 &p_rect);
	Rect2 get_item_icon_region(int32_t p_idx) const;
	void set_item_icon_modulate(int32_t p_idx, const Color &p_modulate);
	Color get_item_icon_modulate(int32_t p_idx) const;
	void set_item_selectable(int32_t p_idx, bool p_selectable);
	bool is_item_selectable(int32_t p_idx) const;
	void set_item_disabled(int32_t p_idx, bool p_disabled);
	bool is_item_disabled(int32_t p_idx) const;
	void set_item_metadata(int32_t p_idx, const Variant &p_metadata);
	Variant get_item_metadata(int32_t p_idx) const;
	void set_item_custom_bg_color(int32_t p_idx, const Color &p_custom_bg_color);
	Color get_item_custom_bg_color(int32_t p_idx) const;
	void set_item_custom_fg_color(int32_t p_idx, const Color &p_custom_fg_color);
	Color get_item_custom_fg_color(int32_t p_idx) const;
	Rect2 get_item_rect(int32_t p_idx, bool p_expand = true) const;
	void set_item_tooltip_enabled(int32_t p_idx, bool p_enable);
	bool is_item_tooltip_enabled(int32_t p_idx) const;
	void set_item_tooltip(int32_t p_idx, const String &p_tooltip);
	String get_item_tooltip(int32_t p_idx) const;
	void select(int32_t p_idx, bool p_single = true);
	void deselect(int32_t p_idx);
	void deselect_all();
	bool is_selected(int32_t p_idx) const;
	PackedInt32Array get_selected_items();
	void move_item(int32_t p_from_idx, int32_t p_to_idx);
	void set_item_count(int32_t p_count);
	int32_t get_item_count() const;
	void remove_item(int32_t p_idx);
	void clear();
	void sort_items_by_text();
	void set_fixed_column_width(int32_t p_width);
	int32_t get_fixed_column_width() const;
	void set_same_column_width(bool p_enable);
	bool is_same_column_width() const;
	void set_max_text_lines(int32_t p_lines);
	int32_t get_max_text_lines() const;
	void set_max_columns(int32_t p_amount);
	int32_t get_max_columns() const;
	void set_select_mode(ItemList::SelectMode p_mode);
	ItemList::SelectMode get_select_mode() const;
	void set_icon_mode(ItemList::IconMode p_mode);
	ItemList::IconMode get_icon_mode() const;
	void set_fixed_icon_size(const Vector2i &p_size);
	Vector2i get_fixed_icon_size() const;
	void set_icon_scale(float p_scale);
	float get_icon_scale() const;
	void set_allow_rmb_select(bool p_allow);
	bool get_allow_rmb_select() const;
	void set_allow_reselect(bool p_allow);
	bool get_allow_reselect() const;
	void set_allow_search(bool p_allow);
	bool get_allow_search() const;
	void set_auto_width(bool p_enable);
	bool has_auto_width() const;
	void set_auto_height(bool p_enable);
	bool has_auto_height() const;
	bool is_anything_selected();
	int32_t get_item_at_position(const Vector2 &p_position, bool p_exact = false) const;
	void ensure_current_is_visible();
	VScrollBar *get_v_scroll_bar();
	HScrollBar *get_h_scroll_bar();
	void set_scroll_hint_mode(ItemList::ScrollHintMode p_scroll_hint_mode);
	ItemList::ScrollHintMode get_scroll_hint_mode() const;
	void set_tile_scroll_hint(bool p_tile_scroll_hint);
	bool is_scroll_hint_tiled();
	void set_text_overrun_behavior(TextServer::OverrunBehavior p_overrun_behavior);
	TextServer::OverrunBehavior get_text_overrun_behavior() const;
	void set_wraparound_items(bool p_enable);
	bool has_wraparound_items() const;
	void force_update_list_size();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Control::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(ItemList::IconMode);
VARIANT_ENUM_CAST(ItemList::SelectMode);
VARIANT_ENUM_CAST(ItemList::ScrollHintMode);

