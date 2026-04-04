/**************************************************************************/
/*  theme.hpp                                                             */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Font;
class String;
class StyleBox;
class Texture2D;

class Theme : public Resource {
	GDEXTENSION_CLASS(Theme, Resource)

public:
	enum DataType {
		DATA_TYPE_COLOR = 0,
		DATA_TYPE_CONSTANT = 1,
		DATA_TYPE_FONT = 2,
		DATA_TYPE_FONT_SIZE = 3,
		DATA_TYPE_ICON = 4,
		DATA_TYPE_STYLEBOX = 5,
		DATA_TYPE_MAX = 6,
	};

	void set_icon(const StringName &p_name, const StringName &p_theme_type, const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_icon(const StringName &p_name, const StringName &p_theme_type) const;
	bool has_icon(const StringName &p_name, const StringName &p_theme_type) const;
	void rename_icon(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type);
	void clear_icon(const StringName &p_name, const StringName &p_theme_type);
	PackedStringArray get_icon_list(const String &p_theme_type) const;
	PackedStringArray get_icon_type_list() const;
	void set_stylebox(const StringName &p_name, const StringName &p_theme_type, const Ref<StyleBox> &p_texture);
	Ref<StyleBox> get_stylebox(const StringName &p_name, const StringName &p_theme_type) const;
	bool has_stylebox(const StringName &p_name, const StringName &p_theme_type) const;
	void rename_stylebox(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type);
	void clear_stylebox(const StringName &p_name, const StringName &p_theme_type);
	PackedStringArray get_stylebox_list(const String &p_theme_type) const;
	PackedStringArray get_stylebox_type_list() const;
	void set_font(const StringName &p_name, const StringName &p_theme_type, const Ref<Font> &p_font);
	Ref<Font> get_font(const StringName &p_name, const StringName &p_theme_type) const;
	bool has_font(const StringName &p_name, const StringName &p_theme_type) const;
	void rename_font(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type);
	void clear_font(const StringName &p_name, const StringName &p_theme_type);
	PackedStringArray get_font_list(const String &p_theme_type) const;
	PackedStringArray get_font_type_list() const;
	void set_font_size(const StringName &p_name, const StringName &p_theme_type, int32_t p_font_size);
	int32_t get_font_size(const StringName &p_name, const StringName &p_theme_type) const;
	bool has_font_size(const StringName &p_name, const StringName &p_theme_type) const;
	void rename_font_size(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type);
	void clear_font_size(const StringName &p_name, const StringName &p_theme_type);
	PackedStringArray get_font_size_list(const String &p_theme_type) const;
	PackedStringArray get_font_size_type_list() const;
	void set_color(const StringName &p_name, const StringName &p_theme_type, const Color &p_color);
	Color get_color(const StringName &p_name, const StringName &p_theme_type) const;
	bool has_color(const StringName &p_name, const StringName &p_theme_type) const;
	void rename_color(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type);
	void clear_color(const StringName &p_name, const StringName &p_theme_type);
	PackedStringArray get_color_list(const String &p_theme_type) const;
	PackedStringArray get_color_type_list() const;
	void set_constant(const StringName &p_name, const StringName &p_theme_type, int32_t p_constant);
	int32_t get_constant(const StringName &p_name, const StringName &p_theme_type) const;
	bool has_constant(const StringName &p_name, const StringName &p_theme_type) const;
	void rename_constant(const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type);
	void clear_constant(const StringName &p_name, const StringName &p_theme_type);
	PackedStringArray get_constant_list(const String &p_theme_type) const;
	PackedStringArray get_constant_type_list() const;
	void set_default_base_scale(float p_base_scale);
	float get_default_base_scale() const;
	bool has_default_base_scale() const;
	void set_default_font(const Ref<Font> &p_font);
	Ref<Font> get_default_font() const;
	bool has_default_font() const;
	void set_default_font_size(int32_t p_font_size);
	int32_t get_default_font_size() const;
	bool has_default_font_size() const;
	void set_theme_item(Theme::DataType p_data_type, const StringName &p_name, const StringName &p_theme_type, const Variant &p_value);
	Variant get_theme_item(Theme::DataType p_data_type, const StringName &p_name, const StringName &p_theme_type) const;
	bool has_theme_item(Theme::DataType p_data_type, const StringName &p_name, const StringName &p_theme_type) const;
	void rename_theme_item(Theme::DataType p_data_type, const StringName &p_old_name, const StringName &p_name, const StringName &p_theme_type);
	void clear_theme_item(Theme::DataType p_data_type, const StringName &p_name, const StringName &p_theme_type);
	PackedStringArray get_theme_item_list(Theme::DataType p_data_type, const String &p_theme_type) const;
	PackedStringArray get_theme_item_type_list(Theme::DataType p_data_type) const;
	void set_type_variation(const StringName &p_theme_type, const StringName &p_base_type);
	bool is_type_variation(const StringName &p_theme_type, const StringName &p_base_type) const;
	void clear_type_variation(const StringName &p_theme_type);
	StringName get_type_variation_base(const StringName &p_theme_type) const;
	PackedStringArray get_type_variation_list(const StringName &p_base_type) const;
	void add_type(const StringName &p_theme_type);
	void remove_type(const StringName &p_theme_type);
	void rename_type(const StringName &p_old_theme_type, const StringName &p_theme_type);
	PackedStringArray get_type_list() const;
	void merge_with(const Ref<Theme> &p_other);
	void clear();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Theme::DataType);

