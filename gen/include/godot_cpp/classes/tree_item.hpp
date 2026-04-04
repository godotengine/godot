/**************************************************************************/
/*  tree_item.hpp                                                         */
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
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Font;
class StringName;
class StyleBox;
class Texture2D;
class Tree;

class TreeItem : public Object {
	GDEXTENSION_CLASS(TreeItem, Object)

public:
	enum TreeCellMode {
		CELL_MODE_STRING = 0,
		CELL_MODE_CHECK = 1,
		CELL_MODE_RANGE = 2,
		CELL_MODE_ICON = 3,
		CELL_MODE_CUSTOM = 4,
	};

	void set_cell_mode(int32_t p_column, TreeItem::TreeCellMode p_mode);
	TreeItem::TreeCellMode get_cell_mode(int32_t p_column) const;
	void set_auto_translate_mode(int32_t p_column, Node::AutoTranslateMode p_mode);
	Node::AutoTranslateMode get_auto_translate_mode(int32_t p_column) const;
	void set_edit_multiline(int32_t p_column, bool p_multiline);
	bool is_edit_multiline(int32_t p_column) const;
	void set_checked(int32_t p_column, bool p_checked);
	void set_indeterminate(int32_t p_column, bool p_indeterminate);
	bool is_checked(int32_t p_column) const;
	bool is_indeterminate(int32_t p_column) const;
	void propagate_check(int32_t p_column, bool p_emit_signal = true);
	void set_text(int32_t p_column, const String &p_text);
	String get_text(int32_t p_column) const;
	void set_description(int32_t p_column, const String &p_description);
	String get_description(int32_t p_column) const;
	void set_text_direction(int32_t p_column, Control::TextDirection p_direction);
	Control::TextDirection get_text_direction(int32_t p_column) const;
	void set_autowrap_mode(int32_t p_column, TextServer::AutowrapMode p_autowrap_mode);
	TextServer::AutowrapMode get_autowrap_mode(int32_t p_column) const;
	void set_text_overrun_behavior(int32_t p_column, TextServer::OverrunBehavior p_overrun_behavior);
	TextServer::OverrunBehavior get_text_overrun_behavior(int32_t p_column) const;
	void set_structured_text_bidi_override(int32_t p_column, TextServer::StructuredTextParser p_parser);
	TextServer::StructuredTextParser get_structured_text_bidi_override(int32_t p_column) const;
	void set_structured_text_bidi_override_options(int32_t p_column, const Array &p_args);
	Array get_structured_text_bidi_override_options(int32_t p_column) const;
	void set_language(int32_t p_column, const String &p_language);
	String get_language(int32_t p_column) const;
	void set_suffix(int32_t p_column, const String &p_text);
	String get_suffix(int32_t p_column) const;
	void set_icon(int32_t p_column, const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_icon(int32_t p_column) const;
	void set_icon_overlay(int32_t p_column, const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_icon_overlay(int32_t p_column) const;
	void set_icon_region(int32_t p_column, const Rect2 &p_region);
	Rect2 get_icon_region(int32_t p_column) const;
	void set_icon_max_width(int32_t p_column, int32_t p_width);
	int32_t get_icon_max_width(int32_t p_column) const;
	void set_icon_modulate(int32_t p_column, const Color &p_modulate);
	Color get_icon_modulate(int32_t p_column) const;
	void set_range(int32_t p_column, double p_value);
	double get_range(int32_t p_column) const;
	void set_range_config(int32_t p_column, double p_min, double p_max, double p_step, bool p_expr = false);
	Dictionary get_range_config(int32_t p_column);
	void set_metadata(int32_t p_column, const Variant &p_meta);
	Variant get_metadata(int32_t p_column) const;
	void set_custom_draw(int32_t p_column, Object *p_object, const StringName &p_callback);
	void set_custom_draw_callback(int32_t p_column, const Callable &p_callback);
	Callable get_custom_draw_callback(int32_t p_column) const;
	void set_custom_stylebox(int32_t p_column, const Ref<StyleBox> &p_stylebox);
	Ref<StyleBox> get_custom_stylebox(int32_t p_column) const;
	void set_collapsed(bool p_enable);
	bool is_collapsed();
	void set_collapsed_recursive(bool p_enable);
	bool is_any_collapsed(bool p_only_visible = false);
	void set_visible(bool p_enable);
	bool is_visible();
	bool is_visible_in_tree() const;
	void uncollapse_tree();
	void set_custom_minimum_height(int32_t p_height);
	int32_t get_custom_minimum_height() const;
	void set_selectable(int32_t p_column, bool p_selectable);
	bool is_selectable(int32_t p_column) const;
	bool is_selected(int32_t p_column);
	void select(int32_t p_column);
	void deselect(int32_t p_column);
	void set_editable(int32_t p_column, bool p_enabled);
	bool is_editable(int32_t p_column);
	void set_custom_color(int32_t p_column, const Color &p_color);
	Color get_custom_color(int32_t p_column) const;
	void clear_custom_color(int32_t p_column);
	void set_custom_font(int32_t p_column, const Ref<Font> &p_font);
	Ref<Font> get_custom_font(int32_t p_column) const;
	void set_custom_font_size(int32_t p_column, int32_t p_font_size);
	int32_t get_custom_font_size(int32_t p_column) const;
	void set_custom_bg_color(int32_t p_column, const Color &p_color, bool p_just_outline = false);
	void clear_custom_bg_color(int32_t p_column);
	Color get_custom_bg_color(int32_t p_column) const;
	void set_custom_as_button(int32_t p_column, bool p_enable);
	bool is_custom_set_as_button(int32_t p_column) const;
	void clear_buttons();
	void add_button(int32_t p_column, const Ref<Texture2D> &p_button, int32_t p_id = -1, bool p_disabled = false, const String &p_tooltip_text = String(), const String &p_description = String());
	int32_t get_button_count(int32_t p_column) const;
	String get_button_tooltip_text(int32_t p_column, int32_t p_button_index) const;
	int32_t get_button_id(int32_t p_column, int32_t p_button_index) const;
	int32_t get_button_by_id(int32_t p_column, int32_t p_id) const;
	Color get_button_color(int32_t p_column, int32_t p_id) const;
	Ref<Texture2D> get_button(int32_t p_column, int32_t p_button_index) const;
	void set_button_tooltip_text(int32_t p_column, int32_t p_button_index, const String &p_tooltip);
	void set_button(int32_t p_column, int32_t p_button_index, const Ref<Texture2D> &p_button);
	void erase_button(int32_t p_column, int32_t p_button_index);
	void set_button_description(int32_t p_column, int32_t p_button_index, const String &p_description);
	void set_button_disabled(int32_t p_column, int32_t p_button_index, bool p_disabled);
	void set_button_color(int32_t p_column, int32_t p_button_index, const Color &p_color);
	bool is_button_disabled(int32_t p_column, int32_t p_button_index) const;
	void set_tooltip_text(int32_t p_column, const String &p_tooltip);
	String get_tooltip_text(int32_t p_column) const;
	void set_text_alignment(int32_t p_column, HorizontalAlignment p_text_alignment);
	HorizontalAlignment get_text_alignment(int32_t p_column) const;
	void set_expand_right(int32_t p_column, bool p_enable);
	bool get_expand_right(int32_t p_column) const;
	void set_disable_folding(bool p_disable);
	bool is_folding_disabled() const;
	TreeItem *create_child(int32_t p_index = -1);
	void add_child(TreeItem *p_child);
	void remove_child(TreeItem *p_child);
	Tree *get_tree() const;
	TreeItem *get_next() const;
	TreeItem *get_prev();
	TreeItem *get_parent() const;
	TreeItem *get_first_child() const;
	TreeItem *get_next_in_tree(bool p_wrap = false);
	TreeItem *get_prev_in_tree(bool p_wrap = false);
	TreeItem *get_next_visible(bool p_wrap = false);
	TreeItem *get_prev_visible(bool p_wrap = false);
	TreeItem *get_child(int32_t p_index);
	int32_t get_child_count();
	TypedArray<TreeItem> get_children();
	int32_t get_index();
	void move_before(TreeItem *p_item);
	void move_after(TreeItem *p_item);

private:
	void call_recursive_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	void call_recursive(const StringName &p_method, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ Variant(p_method), Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		call_recursive_internal(call_args.data(), variant_args.size());
	}

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TreeItem::TreeCellMode);

