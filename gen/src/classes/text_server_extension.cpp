/**************************************************************************/
/*  text_server_extension.cpp                                             */
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

#include <godot_cpp/classes/text_server_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

namespace godot {

bool TextServerExtension::_has_feature(TextServer::Feature p_feature) const {
	return false;
}

String TextServerExtension::_get_name() const {
	return String();
}

int64_t TextServerExtension::_get_features() const {
	return 0;
}

void TextServerExtension::_free_rid(const RID &p_rid) {}

bool TextServerExtension::_has(const RID &p_rid) {
	return false;
}

bool TextServerExtension::_load_support_data(const String &p_filename) {
	return false;
}

String TextServerExtension::_get_support_data_filename() const {
	return String();
}

String TextServerExtension::_get_support_data_info() const {
	return String();
}

bool TextServerExtension::_save_support_data(const String &p_filename) const {
	return false;
}

PackedByteArray TextServerExtension::_get_support_data() const {
	return PackedByteArray();
}

bool TextServerExtension::_is_locale_using_support_data(const String &p_locale) const {
	return false;
}

bool TextServerExtension::_is_locale_right_to_left(const String &p_locale) const {
	return false;
}

int64_t TextServerExtension::_name_to_tag(const String &p_name) const {
	return 0;
}

String TextServerExtension::_tag_to_name(int64_t p_tag) const {
	return String();
}

RID TextServerExtension::_create_font() {
	return RID();
}

RID TextServerExtension::_create_font_linked_variation(const RID &p_font_rid) {
	return RID();
}

void TextServerExtension::_font_set_data(const RID &p_font_rid, const PackedByteArray &p_data) {}

void TextServerExtension::_font_set_data_ptr(const RID &p_font_rid, const uint8_t *p_data_ptr, int64_t p_data_size) {}

void TextServerExtension::_font_set_face_index(const RID &p_font_rid, int64_t p_face_index) {}

int64_t TextServerExtension::_font_get_face_index(const RID &p_font_rid) const {
	return 0;
}

int64_t TextServerExtension::_font_get_face_count(const RID &p_font_rid) const {
	return 0;
}

void TextServerExtension::_font_set_style(const RID &p_font_rid, BitField<TextServer::FontStyle> p_style) {}

BitField<TextServer::FontStyle> TextServerExtension::_font_get_style(const RID &p_font_rid) const {
	return BitField<TextServer::FontStyle>(0);
}

void TextServerExtension::_font_set_name(const RID &p_font_rid, const String &p_name) {}

String TextServerExtension::_font_get_name(const RID &p_font_rid) const {
	return String();
}

Dictionary TextServerExtension::_font_get_ot_name_strings(const RID &p_font_rid) const {
	return Dictionary();
}

void TextServerExtension::_font_set_style_name(const RID &p_font_rid, const String &p_name_style) {}

String TextServerExtension::_font_get_style_name(const RID &p_font_rid) const {
	return String();
}

void TextServerExtension::_font_set_weight(const RID &p_font_rid, int64_t p_weight) {}

int64_t TextServerExtension::_font_get_weight(const RID &p_font_rid) const {
	return 0;
}

void TextServerExtension::_font_set_stretch(const RID &p_font_rid, int64_t p_stretch) {}

int64_t TextServerExtension::_font_get_stretch(const RID &p_font_rid) const {
	return 0;
}

void TextServerExtension::_font_set_antialiasing(const RID &p_font_rid, TextServer::FontAntialiasing p_antialiasing) {}

TextServer::FontAntialiasing TextServerExtension::_font_get_antialiasing(const RID &p_font_rid) const {
	return TextServer::FontAntialiasing(0);
}

void TextServerExtension::_font_set_disable_embedded_bitmaps(const RID &p_font_rid, bool p_disable_embedded_bitmaps) {}

bool TextServerExtension::_font_get_disable_embedded_bitmaps(const RID &p_font_rid) const {
	return false;
}

void TextServerExtension::_font_set_generate_mipmaps(const RID &p_font_rid, bool p_generate_mipmaps) {}

bool TextServerExtension::_font_get_generate_mipmaps(const RID &p_font_rid) const {
	return false;
}

void TextServerExtension::_font_set_multichannel_signed_distance_field(const RID &p_font_rid, bool p_msdf) {}

bool TextServerExtension::_font_is_multichannel_signed_distance_field(const RID &p_font_rid) const {
	return false;
}

void TextServerExtension::_font_set_msdf_pixel_range(const RID &p_font_rid, int64_t p_msdf_pixel_range) {}

int64_t TextServerExtension::_font_get_msdf_pixel_range(const RID &p_font_rid) const {
	return 0;
}

void TextServerExtension::_font_set_msdf_size(const RID &p_font_rid, int64_t p_msdf_size) {}

int64_t TextServerExtension::_font_get_msdf_size(const RID &p_font_rid) const {
	return 0;
}

void TextServerExtension::_font_set_fixed_size(const RID &p_font_rid, int64_t p_fixed_size) {}

int64_t TextServerExtension::_font_get_fixed_size(const RID &p_font_rid) const {
	return 0;
}

void TextServerExtension::_font_set_fixed_size_scale_mode(const RID &p_font_rid, TextServer::FixedSizeScaleMode p_fixed_size_scale_mode) {}

TextServer::FixedSizeScaleMode TextServerExtension::_font_get_fixed_size_scale_mode(const RID &p_font_rid) const {
	return TextServer::FixedSizeScaleMode(0);
}

void TextServerExtension::_font_set_allow_system_fallback(const RID &p_font_rid, bool p_allow_system_fallback) {}

bool TextServerExtension::_font_is_allow_system_fallback(const RID &p_font_rid) const {
	return false;
}

void TextServerExtension::_font_clear_system_fallback_cache() {}

void TextServerExtension::_font_set_force_autohinter(const RID &p_font_rid, bool p_force_autohinter) {}

bool TextServerExtension::_font_is_force_autohinter(const RID &p_font_rid) const {
	return false;
}

void TextServerExtension::_font_set_modulate_color_glyphs(const RID &p_font_rid, bool p_modulate) {}

bool TextServerExtension::_font_is_modulate_color_glyphs(const RID &p_font_rid) const {
	return false;
}

void TextServerExtension::_font_set_hinting(const RID &p_font_rid, TextServer::Hinting p_hinting) {}

TextServer::Hinting TextServerExtension::_font_get_hinting(const RID &p_font_rid) const {
	return TextServer::Hinting(0);
}

void TextServerExtension::_font_set_subpixel_positioning(const RID &p_font_rid, TextServer::SubpixelPositioning p_subpixel_positioning) {}

TextServer::SubpixelPositioning TextServerExtension::_font_get_subpixel_positioning(const RID &p_font_rid) const {
	return TextServer::SubpixelPositioning(0);
}

void TextServerExtension::_font_set_keep_rounding_remainders(const RID &p_font_rid, bool p_keep_rounding_remainders) {}

bool TextServerExtension::_font_get_keep_rounding_remainders(const RID &p_font_rid) const {
	return false;
}

void TextServerExtension::_font_set_embolden(const RID &p_font_rid, double p_strength) {}

double TextServerExtension::_font_get_embolden(const RID &p_font_rid) const {
	return 0.0;
}

void TextServerExtension::_font_set_spacing(const RID &p_font_rid, TextServer::SpacingType p_spacing, int64_t p_value) {}

int64_t TextServerExtension::_font_get_spacing(const RID &p_font_rid, TextServer::SpacingType p_spacing) const {
	return 0;
}

void TextServerExtension::_font_set_baseline_offset(const RID &p_font_rid, double p_baseline_offset) {}

double TextServerExtension::_font_get_baseline_offset(const RID &p_font_rid) const {
	return 0.0;
}

void TextServerExtension::_font_set_transform(const RID &p_font_rid, const Transform2D &p_transform) {}

Transform2D TextServerExtension::_font_get_transform(const RID &p_font_rid) const {
	return Transform2D();
}

void TextServerExtension::_font_set_variation_coordinates(const RID &p_font_rid, const Dictionary &p_variation_coordinates) {}

Dictionary TextServerExtension::_font_get_variation_coordinates(const RID &p_font_rid) const {
	return Dictionary();
}

void TextServerExtension::_font_set_oversampling(const RID &p_font_rid, double p_oversampling) {}

double TextServerExtension::_font_get_oversampling(const RID &p_font_rid) const {
	return 0.0;
}

TypedArray<Vector2i> TextServerExtension::_font_get_size_cache_list(const RID &p_font_rid) const {
	return TypedArray<Vector2i>();
}

void TextServerExtension::_font_clear_size_cache(const RID &p_font_rid) {}

void TextServerExtension::_font_remove_size_cache(const RID &p_font_rid, const Vector2i &p_size) {}

TypedArray<Dictionary> TextServerExtension::_font_get_size_cache_info(const RID &p_font_rid) const {
	return TypedArray<Dictionary>();
}

void TextServerExtension::_font_set_ascent(const RID &p_font_rid, int64_t p_size, double p_ascent) {}

double TextServerExtension::_font_get_ascent(const RID &p_font_rid, int64_t p_size) const {
	return 0.0;
}

void TextServerExtension::_font_set_descent(const RID &p_font_rid, int64_t p_size, double p_descent) {}

double TextServerExtension::_font_get_descent(const RID &p_font_rid, int64_t p_size) const {
	return 0.0;
}

void TextServerExtension::_font_set_underline_position(const RID &p_font_rid, int64_t p_size, double p_underline_position) {}

double TextServerExtension::_font_get_underline_position(const RID &p_font_rid, int64_t p_size) const {
	return 0.0;
}

void TextServerExtension::_font_set_underline_thickness(const RID &p_font_rid, int64_t p_size, double p_underline_thickness) {}

double TextServerExtension::_font_get_underline_thickness(const RID &p_font_rid, int64_t p_size) const {
	return 0.0;
}

void TextServerExtension::_font_set_scale(const RID &p_font_rid, int64_t p_size, double p_scale) {}

double TextServerExtension::_font_get_scale(const RID &p_font_rid, int64_t p_size) const {
	return 0.0;
}

int64_t TextServerExtension::_font_get_texture_count(const RID &p_font_rid, const Vector2i &p_size) const {
	return 0;
}

void TextServerExtension::_font_clear_textures(const RID &p_font_rid, const Vector2i &p_size) {}

void TextServerExtension::_font_remove_texture(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) {}

void TextServerExtension::_font_set_texture_image(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index, const Ref<Image> &p_image) {}

Ref<Image> TextServerExtension::_font_get_texture_image(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) const {
	return Ref<Image>();
}

void TextServerExtension::_font_set_texture_offsets(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index, const PackedInt32Array &p_offset) {}

PackedInt32Array TextServerExtension::_font_get_texture_offsets(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) const {
	return PackedInt32Array();
}

PackedInt32Array TextServerExtension::_font_get_glyph_list(const RID &p_font_rid, const Vector2i &p_size) const {
	return PackedInt32Array();
}

void TextServerExtension::_font_clear_glyphs(const RID &p_font_rid, const Vector2i &p_size) {}

void TextServerExtension::_font_remove_glyph(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) {}

Vector2 TextServerExtension::_font_get_glyph_advance(const RID &p_font_rid, int64_t p_size, int64_t p_glyph) const {
	return Vector2();
}

void TextServerExtension::_font_set_glyph_advance(const RID &p_font_rid, int64_t p_size, int64_t p_glyph, const Vector2 &p_advance) {}

Vector2 TextServerExtension::_font_get_glyph_offset(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	return Vector2();
}

void TextServerExtension::_font_set_glyph_offset(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Vector2 &p_offset) {}

Vector2 TextServerExtension::_font_get_glyph_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	return Vector2();
}

void TextServerExtension::_font_set_glyph_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Vector2 &p_gl_size) {}

Rect2 TextServerExtension::_font_get_glyph_uv_rect(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	return Rect2();
}

void TextServerExtension::_font_set_glyph_uv_rect(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Rect2 &p_uv_rect) {}

int64_t TextServerExtension::_font_get_glyph_texture_idx(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	return 0;
}

void TextServerExtension::_font_set_glyph_texture_idx(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, int64_t p_texture_idx) {}

RID TextServerExtension::_font_get_glyph_texture_rid(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	return RID();
}

Vector2 TextServerExtension::_font_get_glyph_texture_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const {
	return Vector2();
}

Dictionary TextServerExtension::_font_get_glyph_contours(const RID &p_font_rid, int64_t p_size, int64_t p_index) const {
	return Dictionary();
}

TypedArray<Vector2i> TextServerExtension::_font_get_kerning_list(const RID &p_font_rid, int64_t p_size) const {
	return TypedArray<Vector2i>();
}

void TextServerExtension::_font_clear_kerning_map(const RID &p_font_rid, int64_t p_size) {}

void TextServerExtension::_font_remove_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair) {}

void TextServerExtension::_font_set_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) {}

Vector2 TextServerExtension::_font_get_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair) const {
	return Vector2();
}

int64_t TextServerExtension::_font_get_glyph_index(const RID &p_font_rid, int64_t p_size, int64_t p_char, int64_t p_variation_selector) const {
	return 0;
}

int64_t TextServerExtension::_font_get_char_from_glyph_index(const RID &p_font_rid, int64_t p_size, int64_t p_glyph_index) const {
	return 0;
}

bool TextServerExtension::_font_has_char(const RID &p_font_rid, int64_t p_char) const {
	return false;
}

String TextServerExtension::_font_get_supported_chars(const RID &p_font_rid) const {
	return String();
}

PackedInt32Array TextServerExtension::_font_get_supported_glyphs(const RID &p_font_rid) const {
	return PackedInt32Array();
}

void TextServerExtension::_font_render_range(const RID &p_font_rid, const Vector2i &p_size, int64_t p_start, int64_t p_end) {}

void TextServerExtension::_font_render_glyph(const RID &p_font_rid, const Vector2i &p_size, int64_t p_index) {}

void TextServerExtension::_font_draw_glyph(const RID &p_font_rid, const RID &p_canvas, int64_t p_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color, float p_oversampling) const {}

void TextServerExtension::_font_draw_glyph_outline(const RID &p_font_rid, const RID &p_canvas, int64_t p_size, int64_t p_outline_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color, float p_oversampling) const {}

bool TextServerExtension::_font_is_language_supported(const RID &p_font_rid, const String &p_language) const {
	return false;
}

void TextServerExtension::_font_set_language_support_override(const RID &p_font_rid, const String &p_language, bool p_supported) {}

bool TextServerExtension::_font_get_language_support_override(const RID &p_font_rid, const String &p_language) {
	return false;
}

void TextServerExtension::_font_remove_language_support_override(const RID &p_font_rid, const String &p_language) {}

PackedStringArray TextServerExtension::_font_get_language_support_overrides(const RID &p_font_rid) {
	return PackedStringArray();
}

bool TextServerExtension::_font_is_script_supported(const RID &p_font_rid, const String &p_script) const {
	return false;
}

void TextServerExtension::_font_set_script_support_override(const RID &p_font_rid, const String &p_script, bool p_supported) {}

bool TextServerExtension::_font_get_script_support_override(const RID &p_font_rid, const String &p_script) {
	return false;
}

void TextServerExtension::_font_remove_script_support_override(const RID &p_font_rid, const String &p_script) {}

PackedStringArray TextServerExtension::_font_get_script_support_overrides(const RID &p_font_rid) {
	return PackedStringArray();
}

void TextServerExtension::_font_set_opentype_feature_overrides(const RID &p_font_rid, const Dictionary &p_overrides) {}

Dictionary TextServerExtension::_font_get_opentype_feature_overrides(const RID &p_font_rid) const {
	return Dictionary();
}

Dictionary TextServerExtension::_font_supported_feature_list(const RID &p_font_rid) const {
	return Dictionary();
}

Dictionary TextServerExtension::_font_supported_variation_list(const RID &p_font_rid) const {
	return Dictionary();
}

double TextServerExtension::_font_get_global_oversampling() const {
	return 0.0;
}

void TextServerExtension::_font_set_global_oversampling(double p_oversampling) {}

void TextServerExtension::_reference_oversampling_level(double p_oversampling) {}

void TextServerExtension::_unreference_oversampling_level(double p_oversampling) {}

Vector2 TextServerExtension::_get_hex_code_box_size(int64_t p_size, int64_t p_index) const {
	return Vector2();
}

void TextServerExtension::_draw_hex_code_box(const RID &p_canvas, int64_t p_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color) const {}

RID TextServerExtension::_create_shaped_text(TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	return RID();
}

void TextServerExtension::_shaped_text_clear(const RID &p_shaped) {}

RID TextServerExtension::_shaped_text_duplicate(const RID &p_shaped) {
	return RID();
}

void TextServerExtension::_shaped_text_set_direction(const RID &p_shaped, TextServer::Direction p_direction) {}

TextServer::Direction TextServerExtension::_shaped_text_get_direction(const RID &p_shaped) const {
	return TextServer::Direction(0);
}

TextServer::Direction TextServerExtension::_shaped_text_get_inferred_direction(const RID &p_shaped) const {
	return TextServer::Direction(0);
}

void TextServerExtension::_shaped_text_set_bidi_override(const RID &p_shaped, const Array &p_override) {}

void TextServerExtension::_shaped_text_set_custom_punctuation(const RID &p_shaped, const String &p_punct) {}

String TextServerExtension::_shaped_text_get_custom_punctuation(const RID &p_shaped) const {
	return String();
}

void TextServerExtension::_shaped_text_set_custom_ellipsis(const RID &p_shaped, int64_t p_char) {}

int64_t TextServerExtension::_shaped_text_get_custom_ellipsis(const RID &p_shaped) const {
	return 0;
}

void TextServerExtension::_shaped_text_set_orientation(const RID &p_shaped, TextServer::Orientation p_orientation) {}

TextServer::Orientation TextServerExtension::_shaped_text_get_orientation(const RID &p_shaped) const {
	return TextServer::Orientation(0);
}

void TextServerExtension::_shaped_text_set_preserve_invalid(const RID &p_shaped, bool p_enabled) {}

bool TextServerExtension::_shaped_text_get_preserve_invalid(const RID &p_shaped) const {
	return false;
}

void TextServerExtension::_shaped_text_set_preserve_control(const RID &p_shaped, bool p_enabled) {}

bool TextServerExtension::_shaped_text_get_preserve_control(const RID &p_shaped) const {
	return false;
}

void TextServerExtension::_shaped_text_set_spacing(const RID &p_shaped, TextServer::SpacingType p_spacing, int64_t p_value) {}

int64_t TextServerExtension::_shaped_text_get_spacing(const RID &p_shaped, TextServer::SpacingType p_spacing) const {
	return 0;
}

bool TextServerExtension::_shaped_text_add_string(const RID &p_shaped, const String &p_text, const TypedArray<RID> &p_fonts, int64_t p_size, const Dictionary &p_opentype_features, const String &p_language, const Variant &p_meta) {
	return false;
}

bool TextServerExtension::_shaped_text_add_object(const RID &p_shaped, const Variant &p_key, const Vector2 &p_size, InlineAlignment p_inline_align, int64_t p_length, double p_baseline) {
	return false;
}

bool TextServerExtension::_shaped_text_resize_object(const RID &p_shaped, const Variant &p_key, const Vector2 &p_size, InlineAlignment p_inline_align, double p_baseline) {
	return false;
}

bool TextServerExtension::_shaped_text_has_object(const RID &p_shaped, const Variant &p_key) const {
	return false;
}

String TextServerExtension::_shaped_get_text(const RID &p_shaped) const {
	return String();
}

int64_t TextServerExtension::_shaped_get_span_count(const RID &p_shaped) const {
	return 0;
}

Variant TextServerExtension::_shaped_get_span_meta(const RID &p_shaped, int64_t p_index) const {
	return Variant();
}

Variant TextServerExtension::_shaped_get_span_embedded_object(const RID &p_shaped, int64_t p_index) const {
	return Variant();
}

String TextServerExtension::_shaped_get_span_text(const RID &p_shaped, int64_t p_index) const {
	return String();
}

Variant TextServerExtension::_shaped_get_span_object(const RID &p_shaped, int64_t p_index) const {
	return Variant();
}

void TextServerExtension::_shaped_set_span_update_font(const RID &p_shaped, int64_t p_index, const TypedArray<RID> &p_fonts, int64_t p_size, const Dictionary &p_opentype_features) {}

int64_t TextServerExtension::_shaped_get_run_count(const RID &p_shaped) const {
	return 0;
}

String TextServerExtension::_shaped_get_run_text(const RID &p_shaped, int64_t p_index) const {
	return String();
}

Vector2i TextServerExtension::_shaped_get_run_range(const RID &p_shaped, int64_t p_index) const {
	return Vector2i();
}

RID TextServerExtension::_shaped_get_run_font_rid(const RID &p_shaped, int64_t p_index) const {
	return RID();
}

int32_t TextServerExtension::_shaped_get_run_font_size(const RID &p_shaped, int64_t p_index) const {
	return 0;
}

String TextServerExtension::_shaped_get_run_language(const RID &p_shaped, int64_t p_index) const {
	return String();
}

TextServer::Direction TextServerExtension::_shaped_get_run_direction(const RID &p_shaped, int64_t p_index) const {
	return TextServer::Direction(0);
}

Variant TextServerExtension::_shaped_get_run_object(const RID &p_shaped, int64_t p_index) const {
	return Variant();
}

RID TextServerExtension::_shaped_text_substr(const RID &p_shaped, int64_t p_start, int64_t p_length) const {
	return RID();
}

RID TextServerExtension::_shaped_text_get_parent(const RID &p_shaped) const {
	return RID();
}

double TextServerExtension::_shaped_text_fit_to_width(const RID &p_shaped, double p_width, BitField<TextServer::JustificationFlag> p_justification_flags) {
	return 0.0;
}

double TextServerExtension::_shaped_text_tab_align(const RID &p_shaped, const PackedFloat32Array &p_tab_stops) {
	return 0.0;
}

bool TextServerExtension::_shaped_text_shape(const RID &p_shaped) {
	return false;
}

bool TextServerExtension::_shaped_text_update_breaks(const RID &p_shaped) {
	return false;
}

bool TextServerExtension::_shaped_text_update_justification_ops(const RID &p_shaped) {
	return false;
}

bool TextServerExtension::_shaped_text_is_ready(const RID &p_shaped) const {
	return false;
}

const Glyph *TextServerExtension::_shaped_text_get_glyphs(const RID &p_shaped) const {
	return nullptr;
}

const Glyph *TextServerExtension::_shaped_text_sort_logical(const RID &p_shaped) {
	return nullptr;
}

int64_t TextServerExtension::_shaped_text_get_glyph_count(const RID &p_shaped) const {
	return 0;
}

Vector2i TextServerExtension::_shaped_text_get_range(const RID &p_shaped) const {
	return Vector2i();
}

PackedInt32Array TextServerExtension::_shaped_text_get_line_breaks_adv(const RID &p_shaped, const PackedFloat32Array &p_width, int64_t p_start, bool p_once, BitField<TextServer::LineBreakFlag> p_break_flags) const {
	return PackedInt32Array();
}

PackedInt32Array TextServerExtension::_shaped_text_get_line_breaks(const RID &p_shaped, double p_width, int64_t p_start, BitField<TextServer::LineBreakFlag> p_break_flags) const {
	return PackedInt32Array();
}

PackedInt32Array TextServerExtension::_shaped_text_get_word_breaks(const RID &p_shaped, BitField<TextServer::GraphemeFlag> p_grapheme_flags, BitField<TextServer::GraphemeFlag> p_skip_grapheme_flags) const {
	return PackedInt32Array();
}

int64_t TextServerExtension::_shaped_text_get_trim_pos(const RID &p_shaped) const {
	return 0;
}

int64_t TextServerExtension::_shaped_text_get_ellipsis_pos(const RID &p_shaped) const {
	return 0;
}

int64_t TextServerExtension::_shaped_text_get_ellipsis_glyph_count(const RID &p_shaped) const {
	return 0;
}

const Glyph *TextServerExtension::_shaped_text_get_ellipsis_glyphs(const RID &p_shaped) const {
	return nullptr;
}

void TextServerExtension::_shaped_text_overrun_trim_to_width(const RID &p_shaped, double p_width, BitField<TextServer::TextOverrunFlag> p_trim_flags) {}

Array TextServerExtension::_shaped_text_get_objects(const RID &p_shaped) const {
	return Array();
}

Rect2 TextServerExtension::_shaped_text_get_object_rect(const RID &p_shaped, const Variant &p_key) const {
	return Rect2();
}

Vector2i TextServerExtension::_shaped_text_get_object_range(const RID &p_shaped, const Variant &p_key) const {
	return Vector2i();
}

int64_t TextServerExtension::_shaped_text_get_object_glyph(const RID &p_shaped, const Variant &p_key) const {
	return 0;
}

Vector2 TextServerExtension::_shaped_text_get_size(const RID &p_shaped) const {
	return Vector2();
}

double TextServerExtension::_shaped_text_get_ascent(const RID &p_shaped) const {
	return 0.0;
}

double TextServerExtension::_shaped_text_get_descent(const RID &p_shaped) const {
	return 0.0;
}

double TextServerExtension::_shaped_text_get_width(const RID &p_shaped) const {
	return 0.0;
}

double TextServerExtension::_shaped_text_get_underline_position(const RID &p_shaped) const {
	return 0.0;
}

double TextServerExtension::_shaped_text_get_underline_thickness(const RID &p_shaped) const {
	return 0.0;
}

int64_t TextServerExtension::_shaped_text_get_dominant_direction_in_range(const RID &p_shaped, int64_t p_start, int64_t p_end) const {
	return 0;
}

void TextServerExtension::_shaped_text_get_carets(const RID &p_shaped, int64_t p_position, CaretInfo *p_caret) const {}

PackedVector2Array TextServerExtension::_shaped_text_get_selection(const RID &p_shaped, int64_t p_start, int64_t p_end) const {
	return PackedVector2Array();
}

int64_t TextServerExtension::_shaped_text_hit_test_grapheme(const RID &p_shaped, double p_coord) const {
	return 0;
}

int64_t TextServerExtension::_shaped_text_hit_test_position(const RID &p_shaped, double p_coord) const {
	return 0;
}

void TextServerExtension::_shaped_text_draw(const RID &p_shaped, const RID &p_canvas, const Vector2 &p_pos, double p_clip_l, double p_clip_r, const Color &p_color, float p_oversampling) const {}

void TextServerExtension::_shaped_text_draw_outline(const RID &p_shaped, const RID &p_canvas, const Vector2 &p_pos, double p_clip_l, double p_clip_r, int64_t p_outline_size, const Color &p_color, float p_oversampling) const {}

Vector2 TextServerExtension::_shaped_text_get_grapheme_bounds(const RID &p_shaped, int64_t p_pos) const {
	return Vector2();
}

int64_t TextServerExtension::_shaped_text_next_grapheme_pos(const RID &p_shaped, int64_t p_pos) const {
	return 0;
}

int64_t TextServerExtension::_shaped_text_prev_grapheme_pos(const RID &p_shaped, int64_t p_pos) const {
	return 0;
}

PackedInt32Array TextServerExtension::_shaped_text_get_character_breaks(const RID &p_shaped) const {
	return PackedInt32Array();
}

int64_t TextServerExtension::_shaped_text_next_character_pos(const RID &p_shaped, int64_t p_pos) const {
	return 0;
}

int64_t TextServerExtension::_shaped_text_prev_character_pos(const RID &p_shaped, int64_t p_pos) const {
	return 0;
}

int64_t TextServerExtension::_shaped_text_closest_character_pos(const RID &p_shaped, int64_t p_pos) const {
	return 0;
}

String TextServerExtension::_format_number(const String &p_number, const String &p_language) const {
	return String();
}

String TextServerExtension::_parse_number(const String &p_number, const String &p_language) const {
	return String();
}

String TextServerExtension::_percent_sign(const String &p_language) const {
	return String();
}

String TextServerExtension::_strip_diacritics(const String &p_string) const {
	return String();
}

bool TextServerExtension::_is_valid_identifier(const String &p_string) const {
	return false;
}

bool TextServerExtension::_is_valid_letter(uint64_t p_unicode) const {
	return false;
}

PackedInt32Array TextServerExtension::_string_get_word_breaks(const String &p_string, const String &p_language, int64_t p_chars_per_line) const {
	return PackedInt32Array();
}

PackedInt32Array TextServerExtension::_string_get_character_breaks(const String &p_string, const String &p_language) const {
	return PackedInt32Array();
}

int64_t TextServerExtension::_is_confusable(const String &p_string, const PackedStringArray &p_dict) const {
	return 0;
}

bool TextServerExtension::_spoof_check(const String &p_string) const {
	return false;
}

String TextServerExtension::_string_to_upper(const String &p_string, const String &p_language) const {
	return String();
}

String TextServerExtension::_string_to_lower(const String &p_string, const String &p_language) const {
	return String();
}

String TextServerExtension::_string_to_title(const String &p_string, const String &p_language) const {
	return String();
}

TypedArray<Vector3i> TextServerExtension::_parse_structured_text(TextServer::StructuredTextParser p_parser_type, const Array &p_args, const String &p_text) const {
	return TypedArray<Vector3i>();
}

void TextServerExtension::_cleanup() {}

} // namespace godot
