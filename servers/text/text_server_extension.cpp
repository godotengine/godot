/*************************************************************************/
/*  text_server_extension.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "text_server_extension.h"

void TextServerExtension::_bind_methods() {
	GDVIRTUAL_BIND(_has_feature, "feature");
	GDVIRTUAL_BIND(_get_name);
	GDVIRTUAL_BIND(_get_features);

	GDVIRTUAL_BIND(_free, "rid");
	GDVIRTUAL_BIND(_has, "rid");
	GDVIRTUAL_BIND(_load_support_data, "filename");

	GDVIRTUAL_BIND(_get_support_data_filename);
	GDVIRTUAL_BIND(_get_support_data_info);
	GDVIRTUAL_BIND(_save_support_data, "filename");

	GDVIRTUAL_BIND(_is_locale_right_to_left, "locale");

	GDVIRTUAL_BIND(_name_to_tag, "name");
	GDVIRTUAL_BIND(_tag_to_name, "tag");

	/* Font interface */

	GDVIRTUAL_BIND(_create_font);

	GDVIRTUAL_BIND(_font_set_data, "font_rid", "data");
	GDVIRTUAL_BIND(_font_set_data_ptr, "font_rid", "data_ptr", "data_size");

	GDVIRTUAL_BIND(_font_set_style, "font_rid", "style");
	GDVIRTUAL_BIND(_font_get_style, "font_rid");

	GDVIRTUAL_BIND(_font_set_name, "font_rid", "name");
	GDVIRTUAL_BIND(_font_get_name, "font_rid");

	GDVIRTUAL_BIND(_font_set_style_name, "font_rid", "name_style");
	GDVIRTUAL_BIND(_font_get_style_name, "font_rid");

	GDVIRTUAL_BIND(_font_set_antialiased, "font_rid", "antialiased");
	GDVIRTUAL_BIND(_font_is_antialiased, "font_rid");

	GDVIRTUAL_BIND(_font_set_multichannel_signed_distance_field, "font_rid", "msdf");
	GDVIRTUAL_BIND(_font_is_multichannel_signed_distance_field, "font_rid");

	GDVIRTUAL_BIND(_font_set_msdf_pixel_range, "font_rid", "msdf_pixel_range");
	GDVIRTUAL_BIND(_font_get_msdf_pixel_range, "font_rid");

	GDVIRTUAL_BIND(_font_set_msdf_size, "font_rid", "msdf_size");
	GDVIRTUAL_BIND(_font_get_msdf_size, "font_rid");

	GDVIRTUAL_BIND(_font_set_fixed_size, "font_rid", "fixed_size");
	GDVIRTUAL_BIND(_font_get_fixed_size, "font_rid");

	GDVIRTUAL_BIND(_font_set_force_autohinter, "font_rid", "force_autohinter");
	GDVIRTUAL_BIND(_font_is_force_autohinter, "font_rid");

	GDVIRTUAL_BIND(_font_set_hinting, "font_rid", "hinting");
	GDVIRTUAL_BIND(_font_get_hinting, "font_rid");

	GDVIRTUAL_BIND(_font_set_variation_coordinates, "font_rid", "variation_coordinates");
	GDVIRTUAL_BIND(_font_get_variation_coordinates, "font_rid");

	GDVIRTUAL_BIND(_font_set_oversampling, "font_rid", "oversampling");
	GDVIRTUAL_BIND(_font_get_oversampling, "font_rid");

	GDVIRTUAL_BIND(_font_get_size_cache_list, "font_rid");
	GDVIRTUAL_BIND(_font_clear_size_cache, "font_rid");
	GDVIRTUAL_BIND(_font_remove_size_cache, "font_rid", "size");

	GDVIRTUAL_BIND(_font_set_ascent, "font_rid", "size", "ascent");
	GDVIRTUAL_BIND(_font_get_ascent, "font_rid", "size");

	GDVIRTUAL_BIND(_font_set_descent, "font_rid", "size", "descent");
	GDVIRTUAL_BIND(_font_get_descent, "font_rid", "size");

	GDVIRTUAL_BIND(_font_set_underline_position, "font_rid", "size", "underline_position");
	GDVIRTUAL_BIND(_font_get_underline_position, "font_rid", "size");

	GDVIRTUAL_BIND(_font_set_underline_thickness, "font_rid", "size", "underline_thickness");
	GDVIRTUAL_BIND(_font_get_underline_thickness, "font_rid", "size");

	GDVIRTUAL_BIND(_font_set_scale, "font_rid", "size", "scale");
	GDVIRTUAL_BIND(_font_get_scale, "font_rid", "size");

	GDVIRTUAL_BIND(_font_set_spacing, "font_rid", "size", "spacing", "value");
	GDVIRTUAL_BIND(_font_get_spacing, "font_rid", "size", "spacing");

	GDVIRTUAL_BIND(_font_get_texture_count, "font_rid", "size");
	GDVIRTUAL_BIND(_font_clear_textures, "font_rid", "size");
	GDVIRTUAL_BIND(_font_remove_texture, "font_rid", "size", "texture_index");

	GDVIRTUAL_BIND(_font_set_texture_image, "font_rid", "size", "texture_index", "image");
	GDVIRTUAL_BIND(_font_get_texture_image, "font_rid", "size", "texture_index");

	GDVIRTUAL_BIND(_font_set_texture_offsets, "font_rid", "size", "texture_index", "offset");
	GDVIRTUAL_BIND(_font_get_texture_offsets, "font_rid", "size", "texture_index");

	GDVIRTUAL_BIND(_font_get_glyph_list, "font_rid", "size");
	GDVIRTUAL_BIND(_font_clear_glyphs, "font_rid", "size");
	GDVIRTUAL_BIND(_font_remove_glyph, "font_rid", "size", "glyph");

	GDVIRTUAL_BIND(_font_get_glyph_advance, "font_rid", "size", "glyph");
	GDVIRTUAL_BIND(_font_set_glyph_advance, "font_rid", "size", "glyph", "advance");

	GDVIRTUAL_BIND(_font_get_glyph_offset, "font_rid", "size", "glyph");
	GDVIRTUAL_BIND(_font_set_glyph_offset, "font_rid", "size", "glyph", "offset");

	GDVIRTUAL_BIND(_font_get_glyph_size, "font_rid", "size", "glyph");
	GDVIRTUAL_BIND(_font_set_glyph_size, "font_rid", "size", "glyph", "gl_size");

	GDVIRTUAL_BIND(_font_get_glyph_uv_rect, "font_rid", "size", "glyph");
	GDVIRTUAL_BIND(_font_set_glyph_uv_rect, "font_rid", "size", "glyph", "uv_rect");

	GDVIRTUAL_BIND(_font_get_glyph_texture_idx, "font_rid", "size", "glyph");
	GDVIRTUAL_BIND(_font_set_glyph_texture_idx, "font_rid", "size", "glyph", "texture_idx");

	GDVIRTUAL_BIND(_font_get_glyph_contours, "font_rid", "size", "index");

	GDVIRTUAL_BIND(_font_get_kerning_list, "font_rid", "size");
	GDVIRTUAL_BIND(_font_clear_kerning_map, "font_rid", "size");
	GDVIRTUAL_BIND(_font_remove_kerning, "font_rid", "size", "glyph_pair");

	GDVIRTUAL_BIND(_font_set_kerning, "font_rid", "size", "glyph_pair", "kerning");
	GDVIRTUAL_BIND(_font_get_kerning, "font_rid", "size", "glyph_pair");

	GDVIRTUAL_BIND(_font_get_glyph_index, "font_rid", "size", "char", "variation_selector");

	GDVIRTUAL_BIND(_font_has_char, "font_rid", "char");
	GDVIRTUAL_BIND(_font_get_supported_chars, "font_rid");

	GDVIRTUAL_BIND(_font_render_range, "font_rid", "size", "start", "end");
	GDVIRTUAL_BIND(_font_render_glyph, "font_rid", "size", "index");

	GDVIRTUAL_BIND(_font_draw_glyph, "font_rid", "canvas", "size", "pos", "index", "color");
	GDVIRTUAL_BIND(_font_draw_glyph_outline, "font_rid", "canvas", "size", "outline_size", "pos", "index", "color");

	GDVIRTUAL_BIND(_font_is_language_supported, "font_rid", "language");
	GDVIRTUAL_BIND(_font_set_language_support_override, "font_rid", "language", "supported");
	GDVIRTUAL_BIND(_font_get_language_support_override, "font_rid", "language");
	GDVIRTUAL_BIND(_font_remove_language_support_override, "font_rid", "language");
	GDVIRTUAL_BIND(_font_get_language_support_overrides, "font_rid");

	GDVIRTUAL_BIND(_font_is_script_supported, "font_rid", "script");
	GDVIRTUAL_BIND(_font_set_script_support_override, "font_rid", "script", "supported");
	GDVIRTUAL_BIND(_font_get_script_support_override, "font_rid", "script");
	GDVIRTUAL_BIND(_font_remove_script_support_override, "font_rid", "script");
	GDVIRTUAL_BIND(_font_get_script_support_overrides, "font_rid");

	GDVIRTUAL_BIND(_font_supported_feature_list, "font_rid");
	GDVIRTUAL_BIND(_font_supported_variation_list, "font_rid");

	GDVIRTUAL_BIND(_font_get_global_oversampling);
	GDVIRTUAL_BIND(_font_set_global_oversampling, "oversampling");

	GDVIRTUAL_BIND(_get_hex_code_box_size, "size", "index");
	GDVIRTUAL_BIND(_draw_hex_code_box, "canvas", "size", "pos", "index", "color");

	/* Shaped text buffer interface */

	GDVIRTUAL_BIND(_create_shaped_text, "direction", "orientation");

	GDVIRTUAL_BIND(_shaped_text_clear, "shaped");

	GDVIRTUAL_BIND(_shaped_text_set_direction, "shaped", "direction");
	GDVIRTUAL_BIND(_shaped_text_get_direction, "shaped");

	GDVIRTUAL_BIND(_shaped_text_set_bidi_override, "shaped", "override");

	GDVIRTUAL_BIND(_shaped_text_set_custom_punctuation, "shaped", "punct");
	GDVIRTUAL_BIND(_shaped_text_get_custom_punctuation, "shaped");

	GDVIRTUAL_BIND(_shaped_text_set_orientation, "shaped", "orientation");
	GDVIRTUAL_BIND(_shaped_text_get_orientation, "shaped");

	GDVIRTUAL_BIND(_shaped_text_set_preserve_invalid, "shaped", "enabled");
	GDVIRTUAL_BIND(_shaped_text_get_preserve_invalid, "shaped");

	GDVIRTUAL_BIND(_shaped_text_set_preserve_control, "shaped", "enabled");
	GDVIRTUAL_BIND(_shaped_text_get_preserve_control, "shaped");

	GDVIRTUAL_BIND(_shaped_text_add_string, "shaped", "text", "fonts", "size", "opentype_features", "language");
	GDVIRTUAL_BIND(_shaped_text_add_object, "shaped", "key", "size", "inline_align", "length");
	GDVIRTUAL_BIND(_shaped_text_resize_object, "shaped", "key", "size", "inline_align");

	GDVIRTUAL_BIND(_shaped_text_substr, "shaped", "start", "length");
	GDVIRTUAL_BIND(_shaped_text_get_parent, "shaped");

	GDVIRTUAL_BIND(_shaped_text_fit_to_width, "shaped", "width", "jst_flags");
	GDVIRTUAL_BIND(_shaped_text_tab_align, "shaped", "tab_stops");

	GDVIRTUAL_BIND(_shaped_text_shape, "shaped");
	GDVIRTUAL_BIND(_shaped_text_update_breaks, "shaped");
	GDVIRTUAL_BIND(_shaped_text_update_justification_ops, "shaped");

	GDVIRTUAL_BIND(_shaped_text_is_ready, "shaped");

	GDVIRTUAL_BIND(_shaped_text_get_glyphs, "shaped", "r_glyphs");
	GDVIRTUAL_BIND(_shaped_text_sort_logical, "shaped", "r_glyphs");
	GDVIRTUAL_BIND(_shaped_text_get_glyph_count, "shaped");

	GDVIRTUAL_BIND(_shaped_text_get_range, "shaped");

	GDVIRTUAL_BIND(_shaped_text_get_line_breaks_adv, "shaped", "width", "start", "once", "break_flags");
	GDVIRTUAL_BIND(_shaped_text_get_line_breaks, "shaped", "width", "start", "break_flags");
	GDVIRTUAL_BIND(_shaped_text_get_word_breaks, "shaped", "grapheme_flags");

	GDVIRTUAL_BIND(_shaped_text_get_trim_pos, "shaped");
	GDVIRTUAL_BIND(_shaped_text_get_ellipsis_pos, "shaped");
	GDVIRTUAL_BIND(_shaped_text_get_ellipsis_glyph_count, "shaped");
	GDVIRTUAL_BIND(_shaped_text_get_ellipsis_glyphs, "shaped", "r_glyphs");

	GDVIRTUAL_BIND(_shaped_text_overrun_trim_to_width, "shaped", "width", "trim_flags");

	GDVIRTUAL_BIND(_shaped_text_get_objects, "shaped");
	GDVIRTUAL_BIND(_shaped_text_get_object_rect, "shaped", "key");

	GDVIRTUAL_BIND(_shaped_text_get_size, "shaped");
	GDVIRTUAL_BIND(_shaped_text_get_ascent, "shaped");
	GDVIRTUAL_BIND(_shaped_text_get_descent, "shaped");
	GDVIRTUAL_BIND(_shaped_text_get_width, "shaped");
	GDVIRTUAL_BIND(_shaped_text_get_underline_position, "shaped");
	GDVIRTUAL_BIND(_shaped_text_get_underline_thickness, "shaped");

	GDVIRTUAL_BIND(_shaped_text_get_dominant_direction_in_range, "shaped", "start", "end");

	GDVIRTUAL_BIND(_shaped_text_get_carets, "shaped", "position", "caret");
	GDVIRTUAL_BIND(_shaped_text_get_selection, "shaped", "start", "end");

	GDVIRTUAL_BIND(_shaped_text_hit_test_grapheme, "shaped", "coord");
	GDVIRTUAL_BIND(_shaped_text_hit_test_position, "shaped", "coord");

	GDVIRTUAL_BIND(_shaped_text_draw, "shaped", "canvas", "pos", "clip_l", "clip_r", "color");
	GDVIRTUAL_BIND(_shaped_text_draw_outline, "shaped", "canvas", "pos", "clip_l", "clip_r", "outline_size", "color");

	GDVIRTUAL_BIND(_shaped_text_get_grapheme_bounds, "shaped", "pos");
	GDVIRTUAL_BIND(_shaped_text_next_grapheme_pos, "shaped", "pos");
	GDVIRTUAL_BIND(_shaped_text_prev_grapheme_pos, "shaped", "pos");

	GDVIRTUAL_BIND(_format_number, "string", "language");
	GDVIRTUAL_BIND(_parse_number, "string", "language");
	GDVIRTUAL_BIND(_percent_sign, "language");
}

bool TextServerExtension::has_feature(Feature p_feature) const {
	bool ret;
	if (GDVIRTUAL_CALL(_has_feature, p_feature, ret)) {
		return ret;
	}
	return false;
}

String TextServerExtension::get_name() const {
	String ret;
	if (GDVIRTUAL_CALL(_get_name, ret)) {
		return ret;
	}
	return "Unknown";
}

uint32_t TextServerExtension::get_features() const {
	uint32_t ret;
	if (GDVIRTUAL_CALL(_get_features, ret)) {
		return ret;
	}
	return 0;
}

void TextServerExtension::free(RID p_rid) {
	GDVIRTUAL_CALL(_free, p_rid);
}

bool TextServerExtension::has(RID p_rid) {
	bool ret;
	if (GDVIRTUAL_CALL(_has, p_rid, ret)) {
		return ret;
	}
	return false;
}

bool TextServerExtension::load_support_data(const String &p_filename) {
	bool ret;
	if (GDVIRTUAL_CALL(_load_support_data, p_filename, ret)) {
		return ret;
	}
	return false;
}

String TextServerExtension::get_support_data_filename() const {
	String ret;
	if (GDVIRTUAL_CALL(_get_support_data_filename, ret)) {
		return ret;
	}
	return String();
}

String TextServerExtension::get_support_data_info() const {
	String ret;
	if (GDVIRTUAL_CALL(_get_support_data_info, ret)) {
		return ret;
	}
	return String();
}

bool TextServerExtension::save_support_data(const String &p_filename) const {
	bool ret;
	if (GDVIRTUAL_CALL(_save_support_data, p_filename, ret)) {
		return ret;
	}
	return false;
}

bool TextServerExtension::is_locale_right_to_left(const String &p_locale) const {
	bool ret;
	if (GDVIRTUAL_CALL(_is_locale_right_to_left, p_locale, ret)) {
		return ret;
	}
	return false;
}

int32_t TextServerExtension::name_to_tag(const String &p_name) const {
	int32_t ret;
	if (GDVIRTUAL_CALL(_name_to_tag, p_name, ret)) {
		return ret;
	}
	return 0;
}

String TextServerExtension::tag_to_name(int32_t p_tag) const {
	String ret;
	if (GDVIRTUAL_CALL(_tag_to_name, p_tag, ret)) {
		return ret;
	}
	return "";
}

/*************************************************************************/
/* Font                                                                  */
/*************************************************************************/

RID TextServerExtension::create_font() {
	RID ret;
	if (GDVIRTUAL_CALL(_create_font, ret)) {
		return ret;
	}
	return RID();
}

void TextServerExtension::font_set_data(RID p_font_rid, const PackedByteArray &p_data) {
	GDVIRTUAL_CALL(_font_set_data, p_font_rid, p_data);
}

void TextServerExtension::font_set_data_ptr(RID p_font_rid, const uint8_t *p_data_ptr, size_t p_data_size) {
	GDVIRTUAL_CALL(_font_set_data_ptr, p_font_rid, p_data_ptr, p_data_size);
}

void TextServerExtension::font_set_style(RID p_font_rid, uint32_t /*FontStyle*/ p_style) {
	GDVIRTUAL_CALL(_font_set_style, p_font_rid, p_style);
}

uint32_t /*FontStyle*/ TextServerExtension::font_get_style(RID p_font_rid) const {
	uint32_t ret;
	if (GDVIRTUAL_CALL(_font_get_style, p_font_rid, ret)) {
		return ret;
	}
	return 0;
}

void TextServerExtension::font_set_style_name(RID p_font_rid, const String &p_name) {
	GDVIRTUAL_CALL(_font_set_style_name, p_font_rid, p_name);
}

String TextServerExtension::font_get_style_name(RID p_font_rid) const {
	String ret;
	if (GDVIRTUAL_CALL(_font_get_style_name, p_font_rid, ret)) {
		return ret;
	}
	return String();
}

void TextServerExtension::font_set_name(RID p_font_rid, const String &p_name) {
	GDVIRTUAL_CALL(_font_set_name, p_font_rid, p_name);
}

String TextServerExtension::font_get_name(RID p_font_rid) const {
	String ret;
	if (GDVIRTUAL_CALL(_font_get_name, p_font_rid, ret)) {
		return ret;
	}
	return String();
}

void TextServerExtension::font_set_antialiased(RID p_font_rid, bool p_antialiased) {
	GDVIRTUAL_CALL(_font_set_antialiased, p_font_rid, p_antialiased);
}

bool TextServerExtension::font_is_antialiased(RID p_font_rid) const {
	bool ret;
	if (GDVIRTUAL_CALL(_font_is_antialiased, p_font_rid, ret)) {
		return ret;
	}
	return false;
}

void TextServerExtension::font_set_multichannel_signed_distance_field(RID p_font_rid, bool p_msdf) {
	GDVIRTUAL_CALL(_font_set_multichannel_signed_distance_field, p_font_rid, p_msdf);
}

bool TextServerExtension::font_is_multichannel_signed_distance_field(RID p_font_rid) const {
	bool ret;
	if (GDVIRTUAL_CALL(_font_is_multichannel_signed_distance_field, p_font_rid, ret)) {
		return ret;
	}
	return false;
}

void TextServerExtension::font_set_msdf_pixel_range(RID p_font_rid, int p_msdf_pixel_range) {
	GDVIRTUAL_CALL(_font_set_msdf_pixel_range, p_font_rid, p_msdf_pixel_range);
}

int TextServerExtension::font_get_msdf_pixel_range(RID p_font_rid) const {
	int ret;
	if (GDVIRTUAL_CALL(_font_get_msdf_pixel_range, p_font_rid, ret)) {
		return ret;
	}
	return 0;
}

void TextServerExtension::font_set_msdf_size(RID p_font_rid, int p_msdf_size) {
	GDVIRTUAL_CALL(_font_set_msdf_size, p_font_rid, p_msdf_size);
}

int TextServerExtension::font_get_msdf_size(RID p_font_rid) const {
	int ret;
	if (GDVIRTUAL_CALL(_font_get_msdf_size, p_font_rid, ret)) {
		return ret;
	}
	return 0;
}

void TextServerExtension::font_set_fixed_size(RID p_font_rid, int p_fixed_size) {
	GDVIRTUAL_CALL(_font_set_fixed_size, p_font_rid, p_fixed_size);
}

int TextServerExtension::font_get_fixed_size(RID p_font_rid) const {
	int ret;
	if (GDVIRTUAL_CALL(_font_get_fixed_size, p_font_rid, ret)) {
		return ret;
	}
	return 0;
}

void TextServerExtension::font_set_force_autohinter(RID p_font_rid, bool p_force_autohinter) {
	GDVIRTUAL_CALL(_font_set_force_autohinter, p_font_rid, p_force_autohinter);
}

bool TextServerExtension::font_is_force_autohinter(RID p_font_rid) const {
	bool ret;
	if (GDVIRTUAL_CALL(_font_is_force_autohinter, p_font_rid, ret)) {
		return ret;
	}
	return false;
}

void TextServerExtension::font_set_hinting(RID p_font_rid, TextServer::Hinting p_hinting) {
	GDVIRTUAL_CALL(_font_set_hinting, p_font_rid, p_hinting);
}

TextServer::Hinting TextServerExtension::font_get_hinting(RID p_font_rid) const {
	int ret;
	if (GDVIRTUAL_CALL(_font_get_hinting, p_font_rid, ret)) {
		return (TextServer::Hinting)ret;
	}
	return TextServer::Hinting::HINTING_NONE;
}

void TextServerExtension::font_set_variation_coordinates(RID p_font_rid, const Dictionary &p_variation_coordinates) {
	GDVIRTUAL_CALL(_font_set_variation_coordinates, p_font_rid, p_variation_coordinates);
}

Dictionary TextServerExtension::font_get_variation_coordinates(RID p_font_rid) const {
	Dictionary ret;
	if (GDVIRTUAL_CALL(_font_get_variation_coordinates, p_font_rid, ret)) {
		return ret;
	}
	return Dictionary();
}

void TextServerExtension::font_set_oversampling(RID p_font_rid, float p_oversampling) {
	GDVIRTUAL_CALL(_font_set_oversampling, p_font_rid, p_oversampling);
}

float TextServerExtension::font_get_oversampling(RID p_font_rid) const {
	float ret;
	if (GDVIRTUAL_CALL(_font_get_oversampling, p_font_rid, ret)) {
		return ret;
	}
	return 0.f;
}

Array TextServerExtension::font_get_size_cache_list(RID p_font_rid) const {
	Array ret;
	if (GDVIRTUAL_CALL(_font_get_size_cache_list, p_font_rid, ret)) {
		return ret;
	}
	return Array();
}

void TextServerExtension::font_clear_size_cache(RID p_font_rid) {
	GDVIRTUAL_CALL(_font_clear_size_cache, p_font_rid);
}

void TextServerExtension::font_remove_size_cache(RID p_font_rid, const Vector2i &p_size) {
	GDVIRTUAL_CALL(_font_remove_size_cache, p_font_rid, p_size);
}

void TextServerExtension::font_set_ascent(RID p_font_rid, int p_size, float p_ascent) {
	GDVIRTUAL_CALL(_font_set_ascent, p_font_rid, p_size, p_ascent);
}

float TextServerExtension::font_get_ascent(RID p_font_rid, int p_size) const {
	float ret;
	if (GDVIRTUAL_CALL(_font_get_ascent, p_font_rid, p_size, ret)) {
		return ret;
	}
	return 0.f;
}

void TextServerExtension::font_set_descent(RID p_font_rid, int p_size, float p_descent) {
	GDVIRTUAL_CALL(_font_set_descent, p_font_rid, p_size, p_descent);
}

float TextServerExtension::font_get_descent(RID p_font_rid, int p_size) const {
	float ret;
	if (GDVIRTUAL_CALL(_font_get_descent, p_font_rid, p_size, ret)) {
		return ret;
	}
	return 0.f;
}

void TextServerExtension::font_set_underline_position(RID p_font_rid, int p_size, float p_underline_position) {
	GDVIRTUAL_CALL(_font_set_underline_position, p_font_rid, p_size, p_underline_position);
}

float TextServerExtension::font_get_underline_position(RID p_font_rid, int p_size) const {
	float ret;
	if (GDVIRTUAL_CALL(_font_get_underline_position, p_font_rid, p_size, ret)) {
		return ret;
	}
	return 0.f;
}

void TextServerExtension::font_set_underline_thickness(RID p_font_rid, int p_size, float p_underline_thickness) {
	GDVIRTUAL_CALL(_font_set_underline_thickness, p_font_rid, p_size, p_underline_thickness);
}

float TextServerExtension::font_get_underline_thickness(RID p_font_rid, int p_size) const {
	float ret;
	if (GDVIRTUAL_CALL(_font_get_underline_thickness, p_font_rid, p_size, ret)) {
		return ret;
	}
	return 0.f;
}

void TextServerExtension::font_set_scale(RID p_font_rid, int p_size, float p_scale) {
	GDVIRTUAL_CALL(_font_set_scale, p_font_rid, p_size, p_scale);
}

float TextServerExtension::font_get_scale(RID p_font_rid, int p_size) const {
	float ret;
	if (GDVIRTUAL_CALL(_font_get_scale, p_font_rid, p_size, ret)) {
		return ret;
	}
	return 0.f;
}

void TextServerExtension::font_set_spacing(RID p_font_rid, int p_size, TextServer::SpacingType p_spacing, int p_value) {
	GDVIRTUAL_CALL(_font_set_spacing, p_font_rid, p_size, p_spacing, p_value);
}

int TextServerExtension::font_get_spacing(RID p_font_rid, int p_size, TextServer::SpacingType p_spacing) const {
	int ret;
	if (GDVIRTUAL_CALL(_font_get_spacing, p_font_rid, p_size, p_spacing, ret)) {
		return ret;
	}
	return 0;
}

int TextServerExtension::font_get_texture_count(RID p_font_rid, const Vector2i &p_size) const {
	int ret;
	if (GDVIRTUAL_CALL(_font_get_texture_count, p_font_rid, p_size, ret)) {
		return ret;
	}
	return 0;
}

void TextServerExtension::font_clear_textures(RID p_font_rid, const Vector2i &p_size) {
	GDVIRTUAL_CALL(_font_clear_textures, p_font_rid, p_size);
}

void TextServerExtension::font_remove_texture(RID p_font_rid, const Vector2i &p_size, int p_texture_index) {
	GDVIRTUAL_CALL(_font_remove_texture, p_font_rid, p_size, p_texture_index);
}

void TextServerExtension::font_set_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const Ref<Image> &p_image) {
	GDVIRTUAL_CALL(_font_set_texture_image, p_font_rid, p_size, p_texture_index, p_image);
}

Ref<Image> TextServerExtension::font_get_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const {
	Ref<Image> ret;
	if (GDVIRTUAL_CALL(_font_get_texture_image, p_font_rid, p_size, p_texture_index, ret)) {
		return ret;
	}
	return Ref<Image>();
}

void TextServerExtension::font_set_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const PackedInt32Array &p_offset) {
	GDVIRTUAL_CALL(_font_set_texture_offsets, p_font_rid, p_size, p_texture_index, p_offset);
}

PackedInt32Array TextServerExtension::font_get_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const {
	PackedInt32Array ret;
	if (GDVIRTUAL_CALL(_font_get_texture_offsets, p_font_rid, p_size, p_texture_index, ret)) {
		return ret;
	}
	return PackedInt32Array();
}

Array TextServerExtension::font_get_glyph_list(RID p_font_rid, const Vector2i &p_size) const {
	Array ret;
	if (GDVIRTUAL_CALL(_font_get_glyph_list, p_font_rid, p_size, ret)) {
		return ret;
	}
	return Array();
}

void TextServerExtension::font_clear_glyphs(RID p_font_rid, const Vector2i &p_size) {
	GDVIRTUAL_CALL(_font_clear_glyphs, p_font_rid, p_size);
}

void TextServerExtension::font_remove_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) {
	GDVIRTUAL_CALL(_font_remove_glyph, p_font_rid, p_size, p_glyph);
}

Vector2 TextServerExtension::font_get_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph) const {
	Vector2 ret;
	if (GDVIRTUAL_CALL(_font_get_glyph_advance, p_font_rid, p_size, p_glyph, ret)) {
		return ret;
	}
	return Vector2();
}

void TextServerExtension::font_set_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph, const Vector2 &p_advance) {
	GDVIRTUAL_CALL(_font_set_glyph_advance, p_font_rid, p_size, p_glyph, p_advance);
}

Vector2 TextServerExtension::font_get_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	Vector2 ret;
	if (GDVIRTUAL_CALL(_font_get_glyph_offset, p_font_rid, p_size, p_glyph, ret)) {
		return ret;
	}
	return Vector2();
}

void TextServerExtension::font_set_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset) {
	GDVIRTUAL_CALL(_font_set_glyph_offset, p_font_rid, p_size, p_glyph, p_offset);
}

Vector2 TextServerExtension::font_get_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	Vector2 ret;
	if (GDVIRTUAL_CALL(_font_get_glyph_size, p_font_rid, p_size, p_glyph, ret)) {
		return ret;
	}
	return Vector2();
}

void TextServerExtension::font_set_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size) {
	GDVIRTUAL_CALL(_font_set_glyph_size, p_font_rid, p_size, p_glyph, p_gl_size);
}

Rect2 TextServerExtension::font_get_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	Rect2 ret;
	if (GDVIRTUAL_CALL(_font_get_glyph_uv_rect, p_font_rid, p_size, p_glyph, ret)) {
		return ret;
	}
	return Rect2();
}

void TextServerExtension::font_set_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect) {
	GDVIRTUAL_CALL(_font_set_glyph_uv_rect, p_font_rid, p_size, p_glyph, p_uv_rect);
}

int TextServerExtension::font_get_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	int ret;
	if (GDVIRTUAL_CALL(_font_get_glyph_texture_idx, p_font_rid, p_size, p_glyph, ret)) {
		return ret;
	}
	return 0;
}

void TextServerExtension::font_set_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, int p_texture_idx) {
	GDVIRTUAL_CALL(_font_set_glyph_texture_idx, p_font_rid, p_size, p_glyph, p_texture_idx);
}

Dictionary TextServerExtension::font_get_glyph_contours(RID p_font_rid, int p_size, int32_t p_index) const {
	Dictionary ret;
	if (GDVIRTUAL_CALL(_font_get_glyph_contours, p_font_rid, p_size, p_index, ret)) {
		return ret;
	}
	return Dictionary();
}

Array TextServerExtension::font_get_kerning_list(RID p_font_rid, int p_size) const {
	Array ret;
	if (GDVIRTUAL_CALL(_font_get_kerning_list, p_font_rid, p_size, ret)) {
		return ret;
	}
	return Array();
}

void TextServerExtension::font_clear_kerning_map(RID p_font_rid, int p_size) {
	GDVIRTUAL_CALL(_font_clear_kerning_map, p_font_rid, p_size);
}

void TextServerExtension::font_remove_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) {
	GDVIRTUAL_CALL(_font_remove_kerning, p_font_rid, p_size, p_glyph_pair);
}

void TextServerExtension::font_set_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) {
	GDVIRTUAL_CALL(_font_set_kerning, p_font_rid, p_size, p_glyph_pair, p_kerning);
}

Vector2 TextServerExtension::font_get_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) const {
	Vector2 ret;
	if (GDVIRTUAL_CALL(_font_get_kerning, p_font_rid, p_size, p_glyph_pair, ret)) {
		return ret;
	}
	return Vector2();
}

int32_t TextServerExtension::font_get_glyph_index(RID p_font_rid, int p_size, char32_t p_char, char32_t p_variation_selector) const {
	int32_t ret;
	if (GDVIRTUAL_CALL(_font_get_glyph_index, p_font_rid, p_size, p_char, p_variation_selector, ret)) {
		return ret;
	}
	return 0;
}

bool TextServerExtension::font_has_char(RID p_font_rid, char32_t p_char) const {
	bool ret;
	if (GDVIRTUAL_CALL(_font_has_char, p_font_rid, p_char, ret)) {
		return ret;
	}
	return false;
}

String TextServerExtension::font_get_supported_chars(RID p_font_rid) const {
	String ret;
	if (GDVIRTUAL_CALL(_font_get_supported_chars, p_font_rid, ret)) {
		return ret;
	}
	return String();
}

void TextServerExtension::font_render_range(RID p_font_rid, const Vector2i &p_size, char32_t p_start, char32_t p_end) {
	GDVIRTUAL_CALL(_font_render_range, p_font_rid, p_size, p_start, p_end);
}

void TextServerExtension::font_render_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_index) {
	GDVIRTUAL_CALL(_font_render_glyph, p_font_rid, p_size, p_index);
}

void TextServerExtension::font_draw_glyph(RID p_font_rid, RID p_canvas, int p_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color) const {
	GDVIRTUAL_CALL(_font_draw_glyph, p_font_rid, p_canvas, p_size, p_pos, p_index, p_color);
}

void TextServerExtension::font_draw_glyph_outline(RID p_font_rid, RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color) const {
	GDVIRTUAL_CALL(_font_draw_glyph_outline, p_font_rid, p_canvas, p_size, p_outline_size, p_pos, p_index, p_color);
}

bool TextServerExtension::font_is_language_supported(RID p_font_rid, const String &p_language) const {
	bool ret;
	if (GDVIRTUAL_CALL(_font_is_language_supported, p_font_rid, p_language, ret)) {
		return ret;
	}
	return false;
}

void TextServerExtension::font_set_language_support_override(RID p_font_rid, const String &p_language, bool p_supported) {
	GDVIRTUAL_CALL(_font_set_language_support_override, p_font_rid, p_language, p_supported);
}

bool TextServerExtension::font_get_language_support_override(RID p_font_rid, const String &p_language) {
	bool ret;
	if (GDVIRTUAL_CALL(_font_get_language_support_override, p_font_rid, p_language, ret)) {
		return ret;
	}
	return false;
}

void TextServerExtension::font_remove_language_support_override(RID p_font_rid, const String &p_language) {
	GDVIRTUAL_CALL(_font_remove_language_support_override, p_font_rid, p_language);
}

Vector<String> TextServerExtension::font_get_language_support_overrides(RID p_font_rid) {
	Vector<String> ret;
	if (GDVIRTUAL_CALL(_font_get_language_support_overrides, p_font_rid, ret)) {
		return ret;
	}
	return Vector<String>();
}

bool TextServerExtension::font_is_script_supported(RID p_font_rid, const String &p_script) const {
	bool ret;
	if (GDVIRTUAL_CALL(_font_is_script_supported, p_font_rid, p_script, ret)) {
		return ret;
	}
	return false;
}

void TextServerExtension::font_set_script_support_override(RID p_font_rid, const String &p_script, bool p_supported) {
	GDVIRTUAL_CALL(_font_set_script_support_override, p_font_rid, p_script, p_supported);
}

bool TextServerExtension::font_get_script_support_override(RID p_font_rid, const String &p_script) {
	bool ret;
	if (GDVIRTUAL_CALL(_font_get_script_support_override, p_font_rid, p_script, ret)) {
		return ret;
	}
	return false;
}

void TextServerExtension::font_remove_script_support_override(RID p_font_rid, const String &p_script) {
	GDVIRTUAL_CALL(_font_remove_script_support_override, p_font_rid, p_script);
}

Vector<String> TextServerExtension::font_get_script_support_overrides(RID p_font_rid) {
	Vector<String> ret;
	if (GDVIRTUAL_CALL(_font_get_script_support_overrides, p_font_rid, ret)) {
		return ret;
	}
	return Vector<String>();
}

Dictionary TextServerExtension::font_supported_feature_list(RID p_font_rid) const {
	Dictionary ret;
	if (GDVIRTUAL_CALL(_font_supported_feature_list, p_font_rid, ret)) {
		return ret;
	}
	return Dictionary();
}

Dictionary TextServerExtension::font_supported_variation_list(RID p_font_rid) const {
	Dictionary ret;
	if (GDVIRTUAL_CALL(_font_supported_variation_list, p_font_rid, ret)) {
		return ret;
	}
	return Dictionary();
}

float TextServerExtension::font_get_global_oversampling() const {
	float ret;
	if (GDVIRTUAL_CALL(_font_get_global_oversampling, ret)) {
		return ret;
	}
	return 0.f;
}

void TextServerExtension::font_set_global_oversampling(float p_oversampling) {
	GDVIRTUAL_CALL(_font_set_global_oversampling, p_oversampling);
}

Vector2 TextServerExtension::get_hex_code_box_size(int p_size, char32_t p_index) const {
	Vector2 ret;
	if (GDVIRTUAL_CALL(_get_hex_code_box_size, p_size, p_index, ret)) {
		return ret;
	}
	return TextServer::get_hex_code_box_size(p_size, p_index);
}

void TextServerExtension::draw_hex_code_box(RID p_canvas, int p_size, const Vector2 &p_pos, char32_t p_index, const Color &p_color) const {
	if (!GDVIRTUAL_CALL(_draw_hex_code_box, p_canvas, p_size, p_pos, p_index, p_color)) {
		TextServer::draw_hex_code_box(p_canvas, p_size, p_pos, p_index, p_color);
	}
}

/*************************************************************************/
/* Shaped text buffer interface                                          */
/*************************************************************************/

RID TextServerExtension::create_shaped_text(TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	RID ret;
	if (GDVIRTUAL_CALL(_create_shaped_text, p_direction, p_orientation, ret)) {
		return ret;
	}
	return RID();
}

void TextServerExtension::shaped_text_clear(RID p_shaped) {
	GDVIRTUAL_CALL(_shaped_text_clear, p_shaped);
}

void TextServerExtension::shaped_text_set_direction(RID p_shaped, TextServer::Direction p_direction) {
	GDVIRTUAL_CALL(_shaped_text_set_direction, p_shaped, p_direction);
}

TextServer::Direction TextServerExtension::shaped_text_get_direction(RID p_shaped) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_direction, p_shaped, ret)) {
		return (TextServer::Direction)ret;
	}
	return TextServer::Direction::DIRECTION_AUTO;
}

void TextServerExtension::shaped_text_set_orientation(RID p_shaped, TextServer::Orientation p_orientation) {
	GDVIRTUAL_CALL(_shaped_text_set_orientation, p_shaped, p_orientation);
}

TextServer::Orientation TextServerExtension::shaped_text_get_orientation(RID p_shaped) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_orientation, p_shaped, ret)) {
		return (TextServer::Orientation)ret;
	}
	return TextServer::Orientation::ORIENTATION_HORIZONTAL;
}

void TextServerExtension::shaped_text_set_bidi_override(RID p_shaped, const Array &p_override) {
	GDVIRTUAL_CALL(_shaped_text_set_bidi_override, p_shaped, p_override);
}

void TextServerExtension::shaped_text_set_custom_punctuation(RID p_shaped, const String &p_punct) {
	GDVIRTUAL_CALL(_shaped_text_set_custom_punctuation, p_shaped, p_punct);
}

String TextServerExtension::shaped_text_get_custom_punctuation(RID p_shaped) const {
	String ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_custom_punctuation, p_shaped, ret)) {
		return ret;
	}
	return String();
}

void TextServerExtension::shaped_text_set_preserve_invalid(RID p_shaped, bool p_enabled) {
	GDVIRTUAL_CALL(_shaped_text_set_preserve_invalid, p_shaped, p_enabled);
}

bool TextServerExtension::shaped_text_get_preserve_invalid(RID p_shaped) const {
	bool ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_preserve_invalid, p_shaped, ret)) {
		return ret;
	}
	return false;
}

void TextServerExtension::shaped_text_set_preserve_control(RID p_shaped, bool p_enabled) {
	GDVIRTUAL_CALL(_shaped_text_set_preserve_control, p_shaped, p_enabled);
}

bool TextServerExtension::shaped_text_get_preserve_control(RID p_shaped) const {
	bool ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_preserve_control, p_shaped, ret)) {
		return ret;
	}
	return false;
}

bool TextServerExtension::shaped_text_add_string(RID p_shaped, const String &p_text, const Vector<RID> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language) {
	bool ret;
	Array fonts;
	for (int i = 0; i < p_fonts.size(); i++) {
		fonts.push_back(p_fonts[i]);
	}
	if (GDVIRTUAL_CALL(_shaped_text_add_string, p_shaped, p_text, fonts, p_size, p_opentype_features, p_language, ret)) {
		return ret;
	}
	return false;
}

bool TextServerExtension::shaped_text_add_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align, int p_length) {
	bool ret;
	if (GDVIRTUAL_CALL(_shaped_text_add_object, p_shaped, p_key, p_size, p_inline_align, p_length, ret)) {
		return ret;
	}
	return false;
}

bool TextServerExtension::shaped_text_resize_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align) {
	bool ret;
	if (GDVIRTUAL_CALL(_shaped_text_resize_object, p_shaped, p_key, p_size, p_inline_align, ret)) {
		return ret;
	}
	return false;
}

RID TextServerExtension::shaped_text_substr(RID p_shaped, int p_start, int p_length) const {
	RID ret;
	if (GDVIRTUAL_CALL(_shaped_text_substr, p_shaped, p_start, p_length, ret)) {
		return ret;
	}
	return RID();
}

RID TextServerExtension::shaped_text_get_parent(RID p_shaped) const {
	RID ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_parent, p_shaped, ret)) {
		return ret;
	}
	return RID();
}

float TextServerExtension::shaped_text_fit_to_width(RID p_shaped, float p_width, uint16_t p_jst_flags) {
	float ret;
	if (GDVIRTUAL_CALL(_shaped_text_fit_to_width, p_shaped, p_width, p_jst_flags, ret)) {
		return ret;
	}
	return 0.f;
}

float TextServerExtension::shaped_text_tab_align(RID p_shaped, const PackedFloat32Array &p_tab_stops) {
	float ret;
	if (GDVIRTUAL_CALL(_shaped_text_tab_align, p_shaped, p_tab_stops, ret)) {
		return ret;
	}
	return 0.f;
}

bool TextServerExtension::shaped_text_shape(RID p_shaped) {
	bool ret;
	if (GDVIRTUAL_CALL(_shaped_text_shape, p_shaped, ret)) {
		return ret;
	}
	return false;
}

bool TextServerExtension::shaped_text_update_breaks(RID p_shaped) {
	bool ret;
	if (GDVIRTUAL_CALL(_shaped_text_update_breaks, p_shaped, ret)) {
		return ret;
	}
	return false;
}

bool TextServerExtension::shaped_text_update_justification_ops(RID p_shaped) {
	bool ret;
	if (GDVIRTUAL_CALL(_shaped_text_update_justification_ops, p_shaped, ret)) {
		return ret;
	}
	return false;
}

bool TextServerExtension::shaped_text_is_ready(RID p_shaped) const {
	bool ret;
	if (GDVIRTUAL_CALL(_shaped_text_is_ready, p_shaped, ret)) {
		return ret;
	}
	return false;
}

const Glyph *TextServerExtension::shaped_text_get_glyphs(RID p_shaped) const {
	const Glyph *ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_glyphs, p_shaped, &ret)) {
		return ret;
	}
	return nullptr;
}

const Glyph *TextServerExtension::shaped_text_sort_logical(RID p_shaped) {
	const Glyph *ret;
	if (GDVIRTUAL_CALL(_shaped_text_sort_logical, p_shaped, &ret)) {
		return ret;
	}
	return nullptr;
}

int TextServerExtension::shaped_text_get_glyph_count(RID p_shaped) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_glyph_count, p_shaped, ret)) {
		return ret;
	}
	return 0;
}

Vector2i TextServerExtension::shaped_text_get_range(RID p_shaped) const {
	Vector2i ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_range, p_shaped, ret)) {
		return ret;
	}
	return Vector2i();
}

PackedInt32Array TextServerExtension::shaped_text_get_line_breaks_adv(RID p_shaped, const PackedFloat32Array &p_width, int p_start, bool p_once, uint16_t p_break_flags) const {
	PackedInt32Array ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_line_breaks_adv, p_shaped, p_width, p_start, p_once, p_break_flags, ret)) {
		return ret;
	}
	return TextServer::shaped_text_get_line_breaks_adv(p_shaped, p_width, p_start, p_once, p_break_flags);
}

PackedInt32Array TextServerExtension::shaped_text_get_line_breaks(RID p_shaped, float p_width, int p_start, uint16_t p_break_flags) const {
	PackedInt32Array ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_line_breaks, p_shaped, p_width, p_start, p_break_flags, ret)) {
		return ret;
	}
	return TextServer::shaped_text_get_line_breaks(p_shaped, p_width, p_start, p_break_flags);
}

PackedInt32Array TextServerExtension::shaped_text_get_word_breaks(RID p_shaped, int p_grapheme_flags) const {
	PackedInt32Array ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_word_breaks, p_shaped, p_grapheme_flags, ret)) {
		return ret;
	}
	return TextServer::shaped_text_get_word_breaks(p_shaped, p_grapheme_flags);
}

int TextServerExtension::shaped_text_get_trim_pos(RID p_shaped) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_trim_pos, p_shaped, ret)) {
		return ret;
	}
	return -1;
}

int TextServerExtension::shaped_text_get_ellipsis_pos(RID p_shaped) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_ellipsis_pos, p_shaped, ret)) {
		return ret;
	}
	return -1;
}

const Glyph *TextServerExtension::shaped_text_get_ellipsis_glyphs(RID p_shaped) const {
	const Glyph *ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_ellipsis_glyphs, p_shaped, &ret)) {
		return ret;
	}
	return nullptr;
}

int TextServerExtension::shaped_text_get_ellipsis_glyph_count(RID p_shaped) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_ellipsis_glyph_count, p_shaped, ret)) {
		return ret;
	}
	return -1;
}

void TextServerExtension::shaped_text_overrun_trim_to_width(RID p_shaped_line, float p_width, uint16_t p_trim_flags) {
	GDVIRTUAL_CALL(_shaped_text_overrun_trim_to_width, p_shaped_line, p_width, p_trim_flags);
}

Array TextServerExtension::shaped_text_get_objects(RID p_shaped) const {
	Array ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_objects, p_shaped, ret)) {
		return ret;
	}
	return Array();
}

Rect2 TextServerExtension::shaped_text_get_object_rect(RID p_shaped, Variant p_key) const {
	Rect2 ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_object_rect, p_shaped, p_key, ret)) {
		return ret;
	}
	return Rect2();
}

Size2 TextServerExtension::shaped_text_get_size(RID p_shaped) const {
	Size2 ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_size, p_shaped, ret)) {
		return ret;
	}
	return Size2();
}

float TextServerExtension::shaped_text_get_ascent(RID p_shaped) const {
	float ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_ascent, p_shaped, ret)) {
		return ret;
	}
	return 0.f;
}

float TextServerExtension::shaped_text_get_descent(RID p_shaped) const {
	float ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_descent, p_shaped, ret)) {
		return ret;
	}
	return 0.f;
}

float TextServerExtension::shaped_text_get_width(RID p_shaped) const {
	float ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_width, p_shaped, ret)) {
		return ret;
	}
	return 0.f;
}

float TextServerExtension::shaped_text_get_underline_position(RID p_shaped) const {
	float ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_underline_position, p_shaped, ret)) {
		return ret;
	}
	return 0.f;
}

float TextServerExtension::shaped_text_get_underline_thickness(RID p_shaped) const {
	float ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_underline_thickness, p_shaped, ret)) {
		return ret;
	}
	return 0.f;
}

TextServer::Direction TextServerExtension::shaped_text_get_dominant_direction_in_range(RID p_shaped, int p_start, int p_end) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_dominant_direction_in_range, p_shaped, p_start, p_end, ret)) {
		return (TextServer::Direction)ret;
	}
	return TextServer::shaped_text_get_dominant_direction_in_range(p_shaped, p_start, p_end);
}

CaretInfo TextServerExtension::shaped_text_get_carets(RID p_shaped, int p_position) const {
	CaretInfo ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_carets, p_shaped, p_position, &ret)) {
		return ret;
	}
	return TextServer::shaped_text_get_carets(p_shaped, p_position);
}

Vector<Vector2> TextServerExtension::shaped_text_get_selection(RID p_shaped, int p_start, int p_end) const {
	Vector<Vector2> ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_selection, p_shaped, p_start, p_end, ret)) {
		return ret;
	}
	return TextServer::shaped_text_get_selection(p_shaped, p_start, p_end);
}

int TextServerExtension::shaped_text_hit_test_grapheme(RID p_shaped, float p_coords) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_hit_test_grapheme, p_shaped, p_coords, ret)) {
		return ret;
	}
	return TextServer::shaped_text_hit_test_grapheme(p_shaped, p_coords);
}

int TextServerExtension::shaped_text_hit_test_position(RID p_shaped, float p_coords) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_hit_test_position, p_shaped, p_coords, ret)) {
		return ret;
	}
	return TextServer::shaped_text_hit_test_position(p_shaped, p_coords);
}

void TextServerExtension::shaped_text_draw(RID p_shaped, RID p_canvas, const Vector2 &p_pos, float p_clip_l, float p_clip_r, const Color &p_color) const {
	if (GDVIRTUAL_CALL(_shaped_text_draw, p_shaped, p_canvas, p_pos, p_clip_l, p_clip_r, p_color)) {
		return;
	}
	TextServer::shaped_text_draw(p_shaped, p_canvas, p_pos, p_clip_l, p_clip_r, p_color);
}

void TextServerExtension::shaped_text_draw_outline(RID p_shaped, RID p_canvas, const Vector2 &p_pos, float p_clip_l, float p_clip_r, int p_outline_size, const Color &p_color) const {
	if (GDVIRTUAL_CALL(_shaped_text_draw_outline, p_shaped, p_canvas, p_pos, p_clip_l, p_clip_r, p_outline_size, p_color)) {
		return;
	}
	shaped_text_draw_outline(p_shaped, p_canvas, p_pos, p_clip_l, p_clip_r, p_outline_size, p_color);
}

Vector2 TextServerExtension::shaped_text_get_grapheme_bounds(RID p_shaped, int p_pos) const {
	Vector2 ret;
	if (GDVIRTUAL_CALL(_shaped_text_get_grapheme_bounds, p_shaped, p_pos, ret)) {
		return ret;
	}
	return TextServer::shaped_text_get_grapheme_bounds(p_shaped, p_pos);
}

int TextServerExtension::shaped_text_next_grapheme_pos(RID p_shaped, int p_pos) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_next_grapheme_pos, p_shaped, p_pos, ret)) {
		return ret;
	}
	return TextServer::shaped_text_next_grapheme_pos(p_shaped, p_pos);
}

int TextServerExtension::shaped_text_prev_grapheme_pos(RID p_shaped, int p_pos) const {
	int ret;
	if (GDVIRTUAL_CALL(_shaped_text_prev_grapheme_pos, p_shaped, p_pos, ret)) {
		return ret;
	}
	return TextServer::shaped_text_prev_grapheme_pos(p_shaped, p_pos);
}

String TextServerExtension::format_number(const String &p_string, const String &p_language) const {
	String ret;
	if (GDVIRTUAL_CALL(_format_number, p_string, p_language, ret)) {
		return ret;
	}
	return TextServer::format_number(p_string, p_language);
}

String TextServerExtension::parse_number(const String &p_string, const String &p_language) const {
	String ret;
	if (GDVIRTUAL_CALL(_parse_number, p_string, p_language, ret)) {
		return ret;
	}
	return TextServer::parse_number(p_string, p_language);
}

String TextServerExtension::percent_sign(const String &p_language) const {
	String ret;
	if (GDVIRTUAL_CALL(_percent_sign, p_language, ret)) {
		return ret;
	}
	return TextServer::percent_sign(p_language);
}

TextServerExtension::TextServerExtension() {
	//NOP
}

TextServerExtension::~TextServerExtension() {
	//NOP
}
