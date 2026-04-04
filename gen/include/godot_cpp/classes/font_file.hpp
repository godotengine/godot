/**************************************************************************/
/*  font_file.hpp                                                         */
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

#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Image;
class String;

class FontFile : public Font {
	GDEXTENSION_CLASS(FontFile, Font)

public:
	Error load_bitmap_font(const String &p_path);
	Error load_dynamic_font(const String &p_path);
	void set_data(const PackedByteArray &p_data);
	PackedByteArray get_data() const;
	void set_font_name(const String &p_name);
	void set_font_style_name(const String &p_name);
	void set_font_style(BitField<TextServer::FontStyle> p_style);
	void set_font_weight(int32_t p_weight);
	void set_font_stretch(int32_t p_stretch);
	void set_antialiasing(TextServer::FontAntialiasing p_antialiasing);
	TextServer::FontAntialiasing get_antialiasing() const;
	void set_disable_embedded_bitmaps(bool p_disable_embedded_bitmaps);
	bool get_disable_embedded_bitmaps() const;
	void set_generate_mipmaps(bool p_generate_mipmaps);
	bool get_generate_mipmaps() const;
	void set_multichannel_signed_distance_field(bool p_msdf);
	bool is_multichannel_signed_distance_field() const;
	void set_msdf_pixel_range(int32_t p_msdf_pixel_range);
	int32_t get_msdf_pixel_range() const;
	void set_msdf_size(int32_t p_msdf_size);
	int32_t get_msdf_size() const;
	void set_fixed_size(int32_t p_fixed_size);
	int32_t get_fixed_size() const;
	void set_fixed_size_scale_mode(TextServer::FixedSizeScaleMode p_fixed_size_scale_mode);
	TextServer::FixedSizeScaleMode get_fixed_size_scale_mode() const;
	void set_allow_system_fallback(bool p_allow_system_fallback);
	bool is_allow_system_fallback() const;
	void set_force_autohinter(bool p_force_autohinter);
	bool is_force_autohinter() const;
	void set_modulate_color_glyphs(bool p_modulate);
	bool is_modulate_color_glyphs() const;
	void set_hinting(TextServer::Hinting p_hinting);
	TextServer::Hinting get_hinting() const;
	void set_subpixel_positioning(TextServer::SubpixelPositioning p_subpixel_positioning);
	TextServer::SubpixelPositioning get_subpixel_positioning() const;
	void set_keep_rounding_remainders(bool p_keep_rounding_remainders);
	bool get_keep_rounding_remainders() const;
	void set_oversampling(float p_oversampling);
	float get_oversampling() const;
	int32_t get_cache_count() const;
	void clear_cache();
	void remove_cache(int32_t p_cache_index);
	TypedArray<Vector2i> get_size_cache_list(int32_t p_cache_index) const;
	void clear_size_cache(int32_t p_cache_index);
	void remove_size_cache(int32_t p_cache_index, const Vector2i &p_size);
	void set_variation_coordinates(int32_t p_cache_index, const Dictionary &p_variation_coordinates);
	Dictionary get_variation_coordinates(int32_t p_cache_index) const;
	void set_embolden(int32_t p_cache_index, float p_strength);
	float get_embolden(int32_t p_cache_index) const;
	void set_transform(int32_t p_cache_index, const Transform2D &p_transform);
	Transform2D get_transform(int32_t p_cache_index) const;
	void set_extra_spacing(int32_t p_cache_index, TextServer::SpacingType p_spacing, int64_t p_value);
	int64_t get_extra_spacing(int32_t p_cache_index, TextServer::SpacingType p_spacing) const;
	void set_extra_baseline_offset(int32_t p_cache_index, float p_baseline_offset);
	float get_extra_baseline_offset(int32_t p_cache_index) const;
	void set_face_index(int32_t p_cache_index, int64_t p_face_index);
	int64_t get_face_index(int32_t p_cache_index) const;
	void set_cache_ascent(int32_t p_cache_index, int32_t p_size, float p_ascent);
	float get_cache_ascent(int32_t p_cache_index, int32_t p_size) const;
	void set_cache_descent(int32_t p_cache_index, int32_t p_size, float p_descent);
	float get_cache_descent(int32_t p_cache_index, int32_t p_size) const;
	void set_cache_underline_position(int32_t p_cache_index, int32_t p_size, float p_underline_position);
	float get_cache_underline_position(int32_t p_cache_index, int32_t p_size) const;
	void set_cache_underline_thickness(int32_t p_cache_index, int32_t p_size, float p_underline_thickness);
	float get_cache_underline_thickness(int32_t p_cache_index, int32_t p_size) const;
	void set_cache_scale(int32_t p_cache_index, int32_t p_size, float p_scale);
	float get_cache_scale(int32_t p_cache_index, int32_t p_size) const;
	int32_t get_texture_count(int32_t p_cache_index, const Vector2i &p_size) const;
	void clear_textures(int32_t p_cache_index, const Vector2i &p_size);
	void remove_texture(int32_t p_cache_index, const Vector2i &p_size, int32_t p_texture_index);
	void set_texture_image(int32_t p_cache_index, const Vector2i &p_size, int32_t p_texture_index, const Ref<Image> &p_image);
	Ref<Image> get_texture_image(int32_t p_cache_index, const Vector2i &p_size, int32_t p_texture_index) const;
	void set_texture_offsets(int32_t p_cache_index, const Vector2i &p_size, int32_t p_texture_index, const PackedInt32Array &p_offset);
	PackedInt32Array get_texture_offsets(int32_t p_cache_index, const Vector2i &p_size, int32_t p_texture_index) const;
	PackedInt32Array get_glyph_list(int32_t p_cache_index, const Vector2i &p_size) const;
	void clear_glyphs(int32_t p_cache_index, const Vector2i &p_size);
	void remove_glyph(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph);
	void set_glyph_advance(int32_t p_cache_index, int32_t p_size, int32_t p_glyph, const Vector2 &p_advance);
	Vector2 get_glyph_advance(int32_t p_cache_index, int32_t p_size, int32_t p_glyph) const;
	void set_glyph_offset(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset);
	Vector2 get_glyph_offset(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph) const;
	void set_glyph_size(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size);
	Vector2 get_glyph_size(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph) const;
	void set_glyph_uv_rect(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect);
	Rect2 get_glyph_uv_rect(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph) const;
	void set_glyph_texture_idx(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph, int32_t p_texture_idx);
	int32_t get_glyph_texture_idx(int32_t p_cache_index, const Vector2i &p_size, int32_t p_glyph) const;
	TypedArray<Vector2i> get_kerning_list(int32_t p_cache_index, int32_t p_size) const;
	void clear_kerning_map(int32_t p_cache_index, int32_t p_size);
	void remove_kerning(int32_t p_cache_index, int32_t p_size, const Vector2i &p_glyph_pair);
	void set_kerning(int32_t p_cache_index, int32_t p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning);
	Vector2 get_kerning(int32_t p_cache_index, int32_t p_size, const Vector2i &p_glyph_pair) const;
	void render_range(int32_t p_cache_index, const Vector2i &p_size, char32_t p_start, char32_t p_end);
	void render_glyph(int32_t p_cache_index, const Vector2i &p_size, int32_t p_index);
	void set_language_support_override(const String &p_language, bool p_supported);
	bool get_language_support_override(const String &p_language) const;
	void remove_language_support_override(const String &p_language);
	PackedStringArray get_language_support_overrides() const;
	void set_script_support_override(const String &p_script, bool p_supported);
	bool get_script_support_override(const String &p_script) const;
	void remove_script_support_override(const String &p_script);
	PackedStringArray get_script_support_overrides() const;
	void set_opentype_feature_overrides(const Dictionary &p_overrides);
	Dictionary get_opentype_feature_overrides() const;
	int32_t get_glyph_index(int32_t p_size, char32_t p_char, char32_t p_variation_selector) const;
	char32_t get_char_from_glyph_index(int32_t p_size, int32_t p_glyph_index) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Font::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

