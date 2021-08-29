/*************************************************************************/
/*  text_server_gdnative.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "text_server_gdnative.h"

bool TextServerGDNative::has_feature(Feature p_feature) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->has_feature(data, (godot_int)p_feature);
}

String TextServerGDNative::get_name() const {
	ERR_FAIL_COND_V(interface == nullptr, String());
	godot_string result = interface->get_name(data);
	String name = *(String *)&result;
	godot_string_destroy(&result);
	return name;
}

void TextServerGDNative::free(RID p_rid) {
	ERR_FAIL_COND(interface == nullptr);
	interface->free(data, (godot_rid *)&p_rid);
}

bool TextServerGDNative::has(RID p_rid) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->has(data, (godot_rid *)&p_rid);
}

bool TextServerGDNative::load_support_data(const String &p_filename) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->load_support_data(data, (godot_string *)&p_filename);
}

#ifdef TOOLS_ENABLED

String TextServerGDNative::get_support_data_filename() {
	ERR_FAIL_COND_V(interface == nullptr, String());
	godot_string result = interface->get_support_data_filename(data);
	String name = *(String *)&result;
	godot_string_destroy(&result);
	return name;
}

String TextServerGDNative::get_support_data_info() {
	ERR_FAIL_COND_V(interface == nullptr, String());
	godot_string result = interface->get_support_data_info(data);
	String info = *(String *)&result;
	godot_string_destroy(&result);
	return info;
}

bool TextServerGDNative::save_support_data(const String &p_filename) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->save_support_data(data, (godot_string *)&p_filename);
}

#endif

bool TextServerGDNative::is_locale_right_to_left(const String &p_locale) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->is_locale_right_to_left(data, (godot_string *)&p_locale);
}

int32_t TextServerGDNative::name_to_tag(const String &p_name) const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->name_to_tag(data, (godot_string *)&p_name);
}

String TextServerGDNative::tag_to_name(int32_t p_tag) const {
	ERR_FAIL_COND_V(interface == nullptr, String());
	godot_string result = interface->tag_to_name(data, p_tag);
	String name = *(String *)&result;
	godot_string_destroy(&result);
	return name;
}

/*************************************************************************/
/* Font                                                                  */
/*************************************************************************/

RID TextServerGDNative::create_font() {
	ERR_FAIL_COND_V(interface == nullptr, RID());
	godot_rid result = interface->create_font(data);
	RID rid = *(RID *)&result;
	return rid;
}

void TextServerGDNative::font_set_data(RID p_font_rid, const PackedByteArray &p_data) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_data(data, (godot_rid *)&p_font_rid, (const godot_packed_byte_array *)&p_data);
}

void TextServerGDNative::font_set_data_ptr(RID p_font_rid, const uint8_t *p_data_ptr, size_t p_data_size) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_data_ptr(data, (godot_rid *)&p_font_rid, p_data_ptr, p_data_size);
}

void TextServerGDNative::font_set_antialiased(RID p_font_rid, bool p_antialiased) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_antialiased(data, (godot_rid *)&p_font_rid, p_antialiased);
}

bool TextServerGDNative::font_is_antialiased(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->font_is_antialiased(data, (godot_rid *)&p_font_rid);
}

void TextServerGDNative::font_set_multichannel_signed_distance_field(RID p_font_rid, bool p_msdf) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_multichannel_signed_distance_field(data, (godot_rid *)&p_font_rid, p_msdf);
}

bool TextServerGDNative::font_is_multichannel_signed_distance_field(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->font_is_multichannel_signed_distance_field(data, (godot_rid *)&p_font_rid);
}

void TextServerGDNative::font_set_msdf_pixel_range(RID p_font_rid, int p_msdf_pixel_range) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_msdf_pixel_range(data, (godot_rid *)&p_font_rid, p_msdf_pixel_range);
}

int TextServerGDNative::font_get_msdf_pixel_range(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->font_get_msdf_pixel_range(data, (godot_rid *)&p_font_rid);
}

void TextServerGDNative::font_set_msdf_size(RID p_font_rid, int p_msdf_size) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_msdf_size(data, (godot_rid *)&p_font_rid, p_msdf_size);
}

int TextServerGDNative::font_get_msdf_size(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->font_get_msdf_size(data, (godot_rid *)&p_font_rid);
}

void TextServerGDNative::font_set_fixed_size(RID p_font_rid, int p_fixed_size) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_fixed_size(data, (godot_rid *)&p_font_rid, p_fixed_size);
}

int TextServerGDNative::font_get_fixed_size(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->font_get_fixed_size(data, (godot_rid *)&p_font_rid);
}

void TextServerGDNative::font_set_force_autohinter(RID p_font_rid, bool p_force_autohinter) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_force_autohinter(data, (godot_rid *)&p_font_rid, p_force_autohinter);
}

bool TextServerGDNative::font_is_force_autohinter(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->font_is_force_autohinter(data, (godot_rid *)&p_font_rid);
}

void TextServerGDNative::font_set_hinting(RID p_font_rid, TextServer::Hinting p_hinting) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_hinting(data, (godot_rid *)&p_font_rid, (godot_int)p_hinting);
}

TextServer::Hinting TextServerGDNative::font_get_hinting(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, TextServer::HINTING_NONE);
	return (TextServer::Hinting)interface->font_get_hinting(data, (godot_rid *)&p_font_rid);
}

void TextServerGDNative::font_set_variation_coordinates(RID p_font_rid, const Dictionary &p_variation_coordinates) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_variation_coordinates(data, (godot_rid *)&p_font_rid, (const godot_dictionary *)&p_variation_coordinates);
}

Dictionary TextServerGDNative::font_get_variation_coordinates(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, Dictionary());
	godot_dictionary result = interface->font_get_variation_coordinates(data, (godot_rid *)&p_font_rid);
	Dictionary dict = *(Dictionary *)&result;
	godot_dictionary_destroy(&result);
	return dict;
}

void TextServerGDNative::font_set_oversampling(RID p_font_rid, real_t p_oversampling) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_oversampling(data, (godot_rid *)&p_font_rid, p_oversampling);
}

real_t TextServerGDNative::font_get_oversampling(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.0f);
	return interface->font_get_oversampling(data, (godot_rid *)&p_font_rid);
}

Array TextServerGDNative::font_get_size_cache_list(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, Array());
	godot_array result = interface->font_get_size_cache_list(data, (godot_rid *)&p_font_rid);
	Array list = *(Array *)&result;
	godot_array_destroy(&result);
	return list;
}

void TextServerGDNative::font_clear_size_cache(RID p_font_rid) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_clear_size_cache(data, (godot_rid *)&p_font_rid);
}

void TextServerGDNative::font_remove_size_cache(RID p_font_rid, const Vector2i &p_size) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_remove_size_cache(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size);
}

void TextServerGDNative::font_set_ascent(RID p_font_rid, int p_size, real_t p_ascent) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_ascent(data, (godot_rid *)&p_font_rid, p_size, p_ascent);
}

real_t TextServerGDNative::font_get_ascent(RID p_font_rid, int p_size) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.0f);
	return interface->font_get_ascent(data, (godot_rid *)&p_font_rid, p_size);
}

void TextServerGDNative::font_set_descent(RID p_font_rid, int p_size, real_t p_descent) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_descent(data, (godot_rid *)&p_font_rid, p_size, p_descent);
}

real_t TextServerGDNative::font_get_descent(RID p_font_rid, int p_size) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.0f);
	return interface->font_get_descent(data, (godot_rid *)&p_font_rid, p_size);
}

void TextServerGDNative::font_set_underline_position(RID p_font_rid, int p_size, real_t p_underline_position) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_underline_position(data, (godot_rid *)&p_font_rid, p_size, p_underline_position);
}

real_t TextServerGDNative::font_get_underline_position(RID p_font_rid, int p_size) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.0f);
	return interface->font_get_underline_position(data, (godot_rid *)&p_font_rid, p_size);
}

void TextServerGDNative::font_set_underline_thickness(RID p_font_rid, int p_size, real_t p_underline_thickness) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_underline_thickness(data, (godot_rid *)&p_font_rid, p_size, p_underline_thickness);
}

real_t TextServerGDNative::font_get_underline_thickness(RID p_font_rid, int p_size) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.0f);
	return interface->font_get_underline_thickness(data, (godot_rid *)&p_font_rid, p_size);
}

void TextServerGDNative::font_set_scale(RID p_font_rid, int p_size, real_t p_scale) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_scale(data, (godot_rid *)&p_font_rid, p_size, p_scale);
}

real_t TextServerGDNative::font_get_scale(RID p_font_rid, int p_size) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.0f);
	return interface->font_get_scale(data, (godot_rid *)&p_font_rid, p_size);
}

void TextServerGDNative::font_set_spacing(RID p_font_rid, int p_size, TextServer::SpacingType p_spacing, int p_value) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_spacing(data, (godot_rid *)&p_font_rid, p_size, (godot_int)p_spacing, p_value);
}

int TextServerGDNative::font_get_spacing(RID p_font_rid, int p_size, TextServer::SpacingType p_spacing) const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->font_get_spacing(data, (godot_rid *)&p_font_rid, p_size, (godot_int)p_spacing);
}

int TextServerGDNative::font_get_texture_count(RID p_font_rid, const Vector2i &p_size) const {
	ERR_FAIL_COND_V(interface == nullptr, -1);
	return interface->font_get_texture_count(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size);
}

void TextServerGDNative::font_clear_textures(RID p_font_rid, const Vector2i &p_size) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_clear_textures(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size);
}

void TextServerGDNative::font_remove_texture(RID p_font_rid, const Vector2i &p_size, int p_texture_index) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_remove_texture(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_texture_index);
}

void TextServerGDNative::font_set_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const Ref<Image> &p_image) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_texture_image(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_texture_index, (const godot_object *)p_image.ptr());
}

Ref<Image> TextServerGDNative::font_get_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const {
	ERR_FAIL_COND_V(interface == nullptr, Ref<Image>());
	godot_object *result = interface->font_get_texture_image(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_texture_index);
	return Ref<Image>((Image *)result);
}

void TextServerGDNative::font_set_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const PackedInt32Array &p_offset) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_texture_offsets(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_texture_index, (const godot_packed_int32_array *)&p_offset);
}

PackedInt32Array TextServerGDNative::font_get_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const {
	ERR_FAIL_COND_V(interface == nullptr, PackedInt32Array());
	godot_packed_int32_array result = interface->font_get_texture_offsets(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_texture_index);
	PackedInt32Array offset = *(PackedInt32Array *)&result;
	godot_packed_int32_array_destroy(&result);
	return offset;
}

Array TextServerGDNative::font_get_glyph_list(RID p_font_rid, const Vector2i &p_size) const {
	ERR_FAIL_COND_V(interface == nullptr, Array());
	godot_array result = interface->font_get_glyph_list(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size);
	Array list = *(Array *)&result;
	godot_array_destroy(&result);
	return list;
}

void TextServerGDNative::font_clear_glyphs(RID p_font_rid, const Vector2i &p_size) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_clear_glyphs(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size);
}

void TextServerGDNative::font_remove_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_remove_glyph(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_glyph);
}

Vector2 TextServerGDNative::font_get_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(interface == nullptr, Vector2());
	godot_vector2 result = interface->font_get_glyph_advance(data, (godot_rid *)&p_font_rid, p_size, p_glyph);
	Vector2 adv = *(Vector2 *)&result;
	return adv;
}

void TextServerGDNative::font_set_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph, const Vector2 &p_advance) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_glyph_advance(data, (godot_rid *)&p_font_rid, p_size, p_glyph, (const godot_vector2 *)&p_advance);
}

Vector2 TextServerGDNative::font_get_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(interface == nullptr, Vector2());
	godot_vector2 result = interface->font_get_glyph_offset(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_glyph);
	Vector2 off = *(Vector2 *)&result;
	return off;
}

void TextServerGDNative::font_set_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_glyph_offset(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_glyph, (const godot_vector2 *)&p_offset);
}

Vector2 TextServerGDNative::font_get_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(interface == nullptr, Vector2());
	godot_vector2 result = interface->font_get_glyph_size(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_glyph);
	Vector2 sz = *(Vector2 *)&result;
	return sz;
}

void TextServerGDNative::font_set_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_glyph_size(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_glyph, (const godot_vector2 *)&p_gl_size);
}

Rect2 TextServerGDNative::font_get_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(interface == nullptr, Rect2());
	godot_rect2 result = interface->font_get_glyph_uv_rect(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_glyph);
	Rect2 uv = *(Rect2 *)&result;
	return uv;
}

void TextServerGDNative::font_set_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_glyph_uv_rect(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_glyph, (const godot_rect2 *)&p_uv_rect);
}

int TextServerGDNative::font_get_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(interface == nullptr, -1);
	return interface->font_get_glyph_texture_idx(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_glyph);
}

void TextServerGDNative::font_set_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, int p_texture_idx) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_glyph_texture_idx(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_glyph, p_texture_idx);
}

bool TextServerGDNative::font_get_glyph_contours(RID p_font_rid, int p_size, int32_t p_index, Vector<Vector3> &r_points, Vector<int32_t> &r_contours, bool &r_orientation) const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->font_get_glyph_contours(data, (godot_rid *)&p_font_rid, p_size, p_index, (godot_packed_vector3_array *)&r_points, (godot_packed_int32_array *)&r_contours, &r_orientation);
}

Array TextServerGDNative::font_get_kerning_list(RID p_font_rid, int p_size) const {
	ERR_FAIL_COND_V(interface == nullptr, Array());
	godot_array result = interface->font_get_kerning_list(data, (godot_rid *)&p_font_rid, p_size);
	Array list = *(Array *)&result;
	godot_array_destroy(&result);
	return list;
}

void TextServerGDNative::font_clear_kerning_map(RID p_font_rid, int p_size) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_clear_kerning_map(data, (godot_rid *)&p_font_rid, p_size);
}

void TextServerGDNative::font_remove_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_remove_kerning(data, (godot_rid *)&p_font_rid, p_size, (const godot_vector2i *)&p_glyph_pair);
}

void TextServerGDNative::font_set_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_kerning(data, (godot_rid *)&p_font_rid, p_size, (const godot_vector2i *)&p_glyph_pair, (const godot_vector2 *)&p_kerning);
}

Vector2 TextServerGDNative::font_get_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) const {
	ERR_FAIL_COND_V(interface == nullptr, Vector2());
	godot_vector2 result = interface->font_get_kerning(data, (godot_rid *)&p_font_rid, p_size, (const godot_vector2i *)&p_glyph_pair);
	Vector2 kern = *(Vector2 *)&result;
	return kern;
}

int32_t TextServerGDNative::font_get_glyph_index(RID p_font_rid, int p_size, char32_t p_char, char32_t p_variation_selector) const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->font_get_glyph_index(data, (godot_rid *)&p_font_rid, p_size, p_char, p_variation_selector);
}

bool TextServerGDNative::font_has_char(RID p_font_rid, char32_t p_char) const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->font_has_char(data, (godot_rid *)&p_font_rid, p_char);
}

String TextServerGDNative::font_get_supported_chars(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, String());
	godot_string result = interface->font_get_supported_chars(data, (godot_rid *)&p_font_rid);
	String chars = *(String *)&result;
	godot_string_destroy(&result);
	return chars;
}

void TextServerGDNative::font_render_range(RID p_font_rid, const Vector2i &p_size, char32_t p_start, char32_t p_end) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_render_range(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_start, p_end);
}

void TextServerGDNative::font_render_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_index) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_render_glyph(data, (godot_rid *)&p_font_rid, (const godot_vector2i *)&p_size, p_index);
}

void TextServerGDNative::font_draw_glyph(RID p_font_rid, RID p_canvas, int p_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color) const {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_draw_glyph(data, (godot_rid *)&p_font_rid, (godot_rid *)&p_canvas, p_size, (const godot_vector2 *)&p_pos, p_index, (const godot_color *)&p_color);
}

void TextServerGDNative::font_draw_glyph_outline(RID p_font_rid, RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color) const {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_draw_glyph_outline(data, (godot_rid *)&p_font_rid, (godot_rid *)&p_canvas, p_size, p_outline_size, (const godot_vector2 *)&p_pos, p_index, (const godot_color *)&p_color);
}

bool TextServerGDNative::font_is_language_supported(RID p_font_rid, const String &p_language) const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->font_is_language_supported(data, (godot_rid *)&p_font_rid, (const godot_string *)&p_language);
}

void TextServerGDNative::font_set_language_support_override(RID p_font_rid, const String &p_language, bool p_supported) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_language_support_override(data, (godot_rid *)&p_font_rid, (const godot_string *)&p_language, p_supported);
}

bool TextServerGDNative::font_get_language_support_override(RID p_font_rid, const String &p_language) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->font_get_language_support_override(data, (godot_rid *)&p_font_rid, (const godot_string *)&p_language);
}

void TextServerGDNative::font_remove_language_support_override(RID p_font_rid, const String &p_language) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_remove_language_support_override(data, (godot_rid *)&p_font_rid, (const godot_string *)&p_language);
}

Vector<String> TextServerGDNative::font_get_language_support_overrides(RID p_font_rid) {
	ERR_FAIL_COND_V(interface == nullptr, PackedStringArray());
	godot_packed_string_array result = interface->font_get_language_support_overrides(data, (godot_rid *)&p_font_rid);
	PackedStringArray list = *(PackedStringArray *)&result;
	godot_packed_string_array_destroy(&result);
	return list;
}

bool TextServerGDNative::font_is_script_supported(RID p_font_rid, const String &p_script) const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->font_is_script_supported(data, (godot_rid *)&p_font_rid, (const godot_string *)&p_script);
}

void TextServerGDNative::font_set_script_support_override(RID p_font_rid, const String &p_script, bool p_supported) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_script_support_override(data, (godot_rid *)&p_font_rid, (const godot_string *)&p_script, p_supported);
}

bool TextServerGDNative::font_get_script_support_override(RID p_font_rid, const String &p_script) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->font_get_script_support_override(data, (godot_rid *)&p_font_rid, (const godot_string *)&p_script);
}

void TextServerGDNative::font_remove_script_support_override(RID p_font_rid, const String &p_script) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_remove_script_support_override(data, (godot_rid *)&p_font_rid, (const godot_string *)&p_script);
}

Vector<String> TextServerGDNative::font_get_script_support_overrides(RID p_font_rid) {
	ERR_FAIL_COND_V(interface == nullptr, PackedStringArray());
	godot_packed_string_array result = interface->font_get_script_support_overrides(data, (godot_rid *)&p_font_rid);
	PackedStringArray list = *(PackedStringArray *)&result;
	godot_packed_string_array_destroy(&result);
	return list;
}

Dictionary TextServerGDNative::font_supported_feature_list(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, Dictionary());
	godot_dictionary result = interface->font_supported_feature_list(data, (godot_rid *)&p_font_rid);
	Dictionary dict = *(Dictionary *)&result;
	godot_dictionary_destroy(&result);
	return dict;
}

Dictionary TextServerGDNative::font_supported_variation_list(RID p_font_rid) const {
	ERR_FAIL_COND_V(interface == nullptr, Dictionary());
	godot_dictionary result = interface->font_supported_variation_list(data, (godot_rid *)&p_font_rid);
	Dictionary dict = *(Dictionary *)&result;
	godot_dictionary_destroy(&result);
	return dict;
}

real_t TextServerGDNative::font_get_global_oversampling() const {
	ERR_FAIL_COND_V(interface == nullptr, 0.0f);
	return interface->font_get_global_oversampling(data);
}

void TextServerGDNative::font_set_global_oversampling(real_t p_oversampling) {
	ERR_FAIL_COND(interface == nullptr);
	interface->font_set_global_oversampling(data, p_oversampling);
}

/*************************************************************************/
/* Shaped text buffer interface                                          */
/*************************************************************************/

RID TextServerGDNative::create_shaped_text(TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	ERR_FAIL_COND_V(interface == nullptr, RID());
	godot_rid result = interface->create_shaped_text(data, (godot_int)p_direction, (godot_int)p_orientation);
	RID rid = *(RID *)&result;
	return rid;
}

void TextServerGDNative::shaped_text_clear(RID p_shaped) {
	ERR_FAIL_COND(interface == nullptr);
	interface->shaped_text_clear(data, (godot_rid *)&p_shaped);
}

void TextServerGDNative::shaped_text_set_direction(RID p_shaped, TextServer::Direction p_direction) {
	ERR_FAIL_COND(interface == nullptr);
	interface->shaped_text_set_direction(data, (godot_rid *)&p_shaped, (godot_int)p_direction);
}

TextServer::Direction TextServerGDNative::shaped_text_get_direction(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, TextServer::DIRECTION_LTR);
	return (TextServer::Direction)interface->shaped_text_get_direction(data, (godot_rid *)&p_shaped);
}

void TextServerGDNative::shaped_text_set_orientation(RID p_shaped, TextServer::Orientation p_orientation) {
	ERR_FAIL_COND(interface == nullptr);
	interface->shaped_text_set_orientation(data, (godot_rid *)&p_shaped, (godot_int)p_orientation);
}

TextServer::Orientation TextServerGDNative::shaped_text_get_orientation(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, TextServer::ORIENTATION_HORIZONTAL);
	return (TextServer::Orientation)interface->shaped_text_get_orientation(data, (godot_rid *)&p_shaped);
}

void TextServerGDNative::shaped_text_set_bidi_override(RID p_shaped, const Vector<Vector2i> &p_override) {
	ERR_FAIL_COND(interface == nullptr);
	interface->shaped_text_set_bidi_override(data, (godot_rid *)&p_shaped, (const godot_packed_vector2i_array *)&p_override);
}

void TextServerGDNative::shaped_text_set_preserve_invalid(RID p_shaped, bool p_enabled) {
	ERR_FAIL_COND(interface == nullptr);
	interface->shaped_text_set_preserve_invalid(data, (godot_rid *)&p_shaped, p_enabled);
}

bool TextServerGDNative::shaped_text_get_preserve_invalid(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return (TextServer::Orientation)interface->shaped_text_get_preserve_invalid(data, (godot_rid *)&p_shaped);
}

void TextServerGDNative::shaped_text_set_preserve_control(RID p_shaped, bool p_enabled) {
	ERR_FAIL_COND(interface == nullptr);
	interface->shaped_text_set_preserve_control(data, (godot_rid *)&p_shaped, p_enabled);
}

bool TextServerGDNative::shaped_text_get_preserve_control(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return (TextServer::Orientation)interface->shaped_text_get_preserve_control(data, (godot_rid *)&p_shaped);
}

bool TextServerGDNative::shaped_text_add_string(RID p_shaped, const String &p_text, const Vector<RID> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->shaped_text_add_string(data, (godot_rid *)&p_shaped, (const godot_string *)&p_text, (const godot_rid **)p_fonts.ptr(), p_size, (const godot_dictionary *)&p_opentype_features, (const godot_string *)&p_language);
}

bool TextServerGDNative::shaped_text_add_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlign p_inline_align, int p_length) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->shaped_text_add_object(data, (godot_rid *)&p_shaped, (const godot_variant *)&p_key, (const godot_vector2 *)&p_size, (godot_int)p_inline_align, p_length);
}

bool TextServerGDNative::shaped_text_resize_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlign p_inline_align) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->shaped_text_resize_object(data, (godot_rid *)&p_shaped, (const godot_variant *)&p_key, (const godot_vector2 *)&p_size, (godot_int)p_inline_align);
}

RID TextServerGDNative::shaped_text_substr(RID p_shaped, int p_start, int p_length) const {
	ERR_FAIL_COND_V(interface == nullptr, RID());
	godot_rid result = interface->shaped_text_substr(data, (godot_rid *)&p_shaped, (godot_int)p_start, (godot_int)p_length);
	RID rid = *(RID *)&result;
	return rid;
}

RID TextServerGDNative::shaped_text_get_parent(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, RID());
	godot_rid result = interface->shaped_text_get_parent(data, (godot_rid *)&p_shaped);
	RID rid = *(RID *)&result;
	return rid;
}

real_t TextServerGDNative::shaped_text_fit_to_width(RID p_shaped, real_t p_width, uint8_t p_jst_flags) {
	ERR_FAIL_COND_V(interface == nullptr, 0.f);
	return interface->shaped_text_fit_to_width(data, (godot_rid *)&p_shaped, p_width, p_jst_flags);
}

real_t TextServerGDNative::shaped_text_tab_align(RID p_shaped, const Vector<real_t> &p_tab_stops) {
	ERR_FAIL_COND_V(interface == nullptr, 0.f);
	return interface->shaped_text_tab_align(data, (godot_rid *)&p_shaped, (godot_packed_float32_array *)&p_tab_stops);
}

bool TextServerGDNative::shaped_text_shape(RID p_shaped) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->shaped_text_shape(data, (godot_rid *)&p_shaped);
}

bool TextServerGDNative::shaped_text_update_breaks(RID p_shaped) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->shaped_text_update_breaks(data, (godot_rid *)&p_shaped);
}

bool TextServerGDNative::shaped_text_update_justification_ops(RID p_shaped) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->shaped_text_update_justification_ops(data, (godot_rid *)&p_shaped);
}

void TextServerGDNative::shaped_text_overrun_trim_to_width(RID p_shaped_line, real_t p_width, uint8_t p_trim_flags) {
	ERR_FAIL_COND(interface == nullptr);
	interface->shaped_text_overrun_trim_to_width(data, (godot_rid *)&p_shaped_line, p_width, p_trim_flags);
};

bool TextServerGDNative::shaped_text_is_ready(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->shaped_text_is_ready(data, (godot_rid *)&p_shaped);
}

Vector<TextServer::Glyph> TextServerGDNative::shaped_text_get_glyphs(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, Vector<TextServer::Glyph>());
	godot_packed_glyph_array result = interface->shaped_text_get_glyphs(data, (godot_rid *)&p_shaped);
	Vector<TextServer::Glyph> glyphs = *(Vector<TextServer::Glyph> *)&result;
	godot_packed_glyph_array_destroy(&result);
	return glyphs;
}

Vector2i TextServerGDNative::shaped_text_get_range(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, Vector2i());
	godot_vector2i result = interface->shaped_text_get_range(data, (godot_rid *)&p_shaped);
	Vector2i range = *(Vector2i *)&result;
	return range;
}

Vector<TextServer::Glyph> TextServerGDNative::shaped_text_sort_logical(RID p_shaped) {
	ERR_FAIL_COND_V(interface == nullptr, Vector<TextServer::Glyph>());
	godot_packed_glyph_array result = interface->shaped_text_sort_logical(data, (godot_rid *)&p_shaped);
	Vector<TextServer::Glyph> glyphs = *(Vector<TextServer::Glyph> *)&result;
	godot_packed_glyph_array_destroy(&result);
	return glyphs;
}

Vector<Vector2i> TextServerGDNative::shaped_text_get_line_breaks_adv(RID p_shaped, const Vector<real_t> &p_width, int p_start, bool p_once, uint8_t p_break_flags) const {
	ERR_FAIL_COND_V(interface == nullptr, Vector<Vector2i>());
	if (interface->shaped_text_get_line_breaks_adv != nullptr) {
		godot_packed_vector2i_array result = interface->shaped_text_get_line_breaks_adv(data, (godot_rid *)&p_shaped, (godot_packed_float32_array *)&p_width, p_start, p_once, p_break_flags);
		Vector<Vector2i> breaks = *(Vector<Vector2i> *)&result;
		godot_packed_vector2i_array_destroy(&result);
		return breaks;
	} else {
		return TextServer::shaped_text_get_line_breaks_adv(p_shaped, p_width, p_break_flags);
	}
}

Vector<Vector2i> TextServerGDNative::shaped_text_get_line_breaks(RID p_shaped, real_t p_width, int p_start, uint8_t p_break_flags) const {
	ERR_FAIL_COND_V(interface == nullptr, Vector<Vector2i>());
	if (interface->shaped_text_get_line_breaks != nullptr) {
		godot_packed_vector2i_array result = interface->shaped_text_get_line_breaks(data, (godot_rid *)&p_shaped, p_width, p_start, p_break_flags);
		Vector<Vector2i> breaks = *(Vector<Vector2i> *)&result;
		godot_packed_vector2i_array_destroy(&result);
		return breaks;
	} else {
		return TextServer::shaped_text_get_line_breaks(p_shaped, p_width, p_break_flags);
	}
}

Vector<Vector2i> TextServerGDNative::shaped_text_get_word_breaks(RID p_shaped, int p_grapheme_flags) const {
	ERR_FAIL_COND_V(interface == nullptr, Vector<Vector2i>());
	if (interface->shaped_text_get_word_breaks != nullptr) {
		godot_packed_vector2i_array result = interface->shaped_text_get_word_breaks(data, (godot_rid *)&p_shaped, p_grapheme_flags);
		Vector<Vector2i> breaks = *(Vector<Vector2i> *)&result;
		godot_packed_vector2i_array_destroy(&result);
		return breaks;
	} else {
		return TextServer::shaped_text_get_word_breaks(p_shaped, p_grapheme_flags);
	}
}

Array TextServerGDNative::shaped_text_get_objects(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, Array());
	godot_array result = interface->shaped_text_get_objects(data, (godot_rid *)&p_shaped);
	Array rect = *(Array *)&result;
	return rect;
}

Rect2 TextServerGDNative::shaped_text_get_object_rect(RID p_shaped, Variant p_key) const {
	ERR_FAIL_COND_V(interface == nullptr, Rect2());
	godot_rect2 result = interface->shaped_text_get_object_rect(data, (godot_rid *)&p_shaped, (const godot_variant *)&p_key);
	Rect2 rect = *(Rect2 *)&result;
	return rect;
}

Size2 TextServerGDNative::shaped_text_get_size(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, Size2());
	godot_vector2 result = interface->shaped_text_get_size(data, (godot_rid *)&p_shaped);
	Size2 size = *(Size2 *)&result;
	return size;
}

real_t TextServerGDNative::shaped_text_get_ascent(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.f);
	return interface->shaped_text_get_ascent(data, (godot_rid *)&p_shaped);
}

real_t TextServerGDNative::shaped_text_get_descent(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.f);
	return interface->shaped_text_get_descent(data, (godot_rid *)&p_shaped);
}

real_t TextServerGDNative::shaped_text_get_width(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.f);
	return interface->shaped_text_get_width(data, (godot_rid *)&p_shaped);
}

real_t TextServerGDNative::shaped_text_get_underline_position(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.f);
	return interface->shaped_text_get_underline_position(data, (godot_rid *)&p_shaped);
}

real_t TextServerGDNative::shaped_text_get_underline_thickness(RID p_shaped) const {
	ERR_FAIL_COND_V(interface == nullptr, 0.f);
	return interface->shaped_text_get_underline_thickness(data, (godot_rid *)&p_shaped);
}

String TextServerGDNative::format_number(const String &p_string, const String &p_language) const {
	ERR_FAIL_COND_V(interface == nullptr, String());
	godot_string result = interface->format_number(data, (const godot_string *)&p_string, (const godot_string *)&p_language);
	if (interface->format_number == nullptr) {
		return p_string;
	}
	String ret = *(String *)&result;
	godot_string_destroy(&result);
	return ret;
}

String TextServerGDNative::parse_number(const String &p_string, const String &p_language) const {
	ERR_FAIL_COND_V(interface == nullptr, String());
	if (interface->parse_number == nullptr) {
		return p_string;
	}
	godot_string result = interface->parse_number(data, (const godot_string *)&p_string, (const godot_string *)&p_language);
	String ret = *(String *)&result;
	godot_string_destroy(&result);
	return ret;
}

String TextServerGDNative::percent_sign(const String &p_language) const {
	ERR_FAIL_COND_V(interface == nullptr, String());
	if (interface->percent_sign == nullptr) {
		return "%";
	}
	godot_string result = interface->percent_sign(data, (const godot_string *)&p_language);
	String ret = *(String *)&result;
	godot_string_destroy(&result);
	return ret;
}

TextServer *TextServerGDNative::create_func(Error &r_error, void *p_user_data) {
	const godot_text_interface_gdnative *interface = (const godot_text_interface_gdnative *)p_user_data;
	r_error = OK;

	TextServerGDNative *server = memnew(TextServerGDNative());
	server->interface = interface;
	server->data = interface->constructor((godot_object *)server);

	return server;
}

TextServerGDNative::TextServerGDNative() {
	data = nullptr;
	interface = nullptr;
}

TextServerGDNative::~TextServerGDNative() {
	if (interface != nullptr) {
		interface->destructor(data);
		data = nullptr;
		interface = nullptr;
	}
}

/*************************************************************************/
/* GDNative functions                                                    */
/*************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

static_assert(sizeof(godot_glyph) == sizeof(TextServer::Glyph), "Glyph size mismatch");
static_assert(sizeof(godot_packed_glyph_array) == sizeof(Vector<TextServer::Glyph>), "Vector<Glyph> size mismatch");

void GDAPI godot_text_register_interface(const godot_text_interface_gdnative *p_interface, const godot_string *p_name, uint32_t p_features) {
	ERR_FAIL_COND(p_interface->version.major != 1);
	String name = *(String *)p_name;
	TextServerManager::register_create_function(name + "(GDNative)", p_features, TextServerGDNative::create_func, (void *)p_interface);
}

// Glyph

void GDAPI godot_glyph_new(godot_glyph *r_dest) {
	TextServer::Glyph *dest = (TextServer::Glyph *)r_dest;
	*dest = TextServer::Glyph();
}

godot_vector2i GDAPI godot_glyph_get_range(const godot_glyph *p_self) {
	godot_vector2i dest;
	Vector2i *d = (Vector2i *)&dest;
	const TextServer::Glyph *self = (const TextServer::Glyph *)p_self;
	d->x = self->start;
	d->y = self->end;
	return dest;
}

void GDAPI godot_glyph_set_range(godot_glyph *p_self, const godot_vector2i *p_range) {
	TextServer::Glyph *self = (TextServer::Glyph *)p_self;
	const Vector2i *range = (const Vector2i *)p_range;
	self->start = range->x;
	self->end = range->y;
}

godot_int GDAPI godot_glyph_get_count(const godot_glyph *p_self) {
	const TextServer::Glyph *self = (const TextServer::Glyph *)p_self;
	return self->count;
}

void GDAPI godot_glyph_set_count(godot_glyph *p_self, godot_int p_count) {
	TextServer::Glyph *self = (TextServer::Glyph *)p_self;
	self->count = p_count;
}

godot_int GDAPI godot_glyph_get_repeat(const godot_glyph *p_self) {
	const TextServer::Glyph *self = (const TextServer::Glyph *)p_self;
	return self->repeat;
}

void GDAPI godot_glyph_set_repeat(godot_glyph *p_self, godot_int p_repeat) {
	TextServer::Glyph *self = (TextServer::Glyph *)p_self;
	self->repeat = p_repeat;
}

godot_int GDAPI godot_glyph_get_flags(const godot_glyph *p_self) {
	const TextServer::Glyph *self = (const TextServer::Glyph *)p_self;
	return self->flags;
}

void GDAPI godot_glyph_set_flags(godot_glyph *p_self, godot_int p_flags) {
	TextServer::Glyph *self = (TextServer::Glyph *)p_self;
	self->flags = p_flags;
}

godot_vector2 GDAPI godot_glyph_get_offset(const godot_glyph *p_self) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const TextServer::Glyph *self = (const TextServer::Glyph *)p_self;
	d->x = self->x_off;
	d->y = self->y_off;
	return dest;
}

void GDAPI godot_glyph_set_offset(godot_glyph *p_self, const godot_vector2 *p_offset) {
	TextServer::Glyph *self = (TextServer::Glyph *)p_self;
	const Vector2 *offset = (const Vector2 *)p_offset;
	self->x_off = offset->x;
	self->y_off = offset->y;
}

godot_real_t GDAPI godot_glyph_get_advance(const godot_glyph *p_self) {
	const TextServer::Glyph *self = (const TextServer::Glyph *)p_self;
	return self->advance;
}

void GDAPI godot_glyph_set_advance(godot_glyph *p_self, godot_real_t p_advance) {
	TextServer::Glyph *self = (TextServer::Glyph *)p_self;
	self->advance = p_advance;
}

godot_rid GDAPI godot_glyph_get_font(const godot_glyph *p_self) {
	godot_rid dest;
	RID *d = (RID *)&dest;
	const TextServer::Glyph *self = (const TextServer::Glyph *)p_self;
	*d = self->font_rid;
	return dest;
}

void GDAPI godot_glyph_set_font(godot_glyph *p_self, godot_rid *p_font) {
	TextServer::Glyph *self = (TextServer::Glyph *)p_self;
	const RID *font = (const RID *)p_font;
	self->font_rid = *font;
}

godot_int GDAPI godot_glyph_get_font_size(const godot_glyph *p_self) {
	const TextServer::Glyph *self = (const TextServer::Glyph *)p_self;
	return self->font_size;
}

void GDAPI godot_glyph_set_font_size(godot_glyph *p_self, godot_int p_size) {
	TextServer::Glyph *self = (TextServer::Glyph *)p_self;
	self->font_size = p_size;
}

godot_int GDAPI godot_glyph_get_index(const godot_glyph *p_self) {
	const TextServer::Glyph *self = (const TextServer::Glyph *)p_self;
	return self->index;
}

void GDAPI godot_glyph_set_index(godot_glyph *p_self, godot_int p_index) {
	TextServer::Glyph *self = (TextServer::Glyph *)p_self;
	self->index = p_index;
}

// GlyphArray

void GDAPI godot_packed_glyph_array_new(godot_packed_glyph_array *r_dest) {
	Vector<TextServer::Glyph> *dest = (Vector<TextServer::Glyph> *)r_dest;
	memnew_placement(dest, Vector<TextServer::Glyph>);
}

void GDAPI godot_packed_glyph_array_new_copy(godot_packed_glyph_array *r_dest, const godot_packed_glyph_array *p_src) {
	Vector<TextServer::Glyph> *dest = (Vector<TextServer::Glyph> *)r_dest;
	const Vector<TextServer::Glyph> *src = (const Vector<TextServer::Glyph> *)p_src;
	memnew_placement(dest, Vector<TextServer::Glyph>(*src));
}

const godot_glyph GDAPI *godot_packed_glyph_array_ptr(const godot_packed_glyph_array *p_self) {
	const Vector<TextServer::Glyph> *self = (const Vector<TextServer::Glyph> *)p_self;
	return (const godot_glyph *)self->ptr();
}

godot_glyph GDAPI *godot_packed_glyph_array_ptrw(godot_packed_glyph_array *p_self) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	return (godot_glyph *)self->ptrw();
}

void GDAPI godot_packed_glyph_array_append(godot_packed_glyph_array *p_self, const godot_glyph *p_data) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	TextServer::Glyph &s = *(TextServer::Glyph *)p_data;
	self->push_back(s);
}

void GDAPI godot_packed_glyph_array_append_array(godot_packed_glyph_array *p_self, const godot_packed_glyph_array *p_array) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	Vector<TextServer::Glyph> *array = (Vector<TextServer::Glyph> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_packed_glyph_array_insert(godot_packed_glyph_array *p_self, const godot_int p_idx, const godot_glyph *p_data) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	TextServer::Glyph &s = *(TextServer::Glyph *)p_data;
	return (godot_error)self->insert(p_idx, s);
}

godot_bool GDAPI godot_packed_glyph_array_has(godot_packed_glyph_array *p_self, const godot_glyph *p_value) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	TextServer::Glyph &v = *(TextServer::Glyph *)p_value;
	return (godot_bool)self->has(v);
}

void GDAPI godot_packed_glyph_array_sort(godot_packed_glyph_array *p_self) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	self->sort();
}

void GDAPI godot_packed_glyph_array_reverse(godot_packed_glyph_array *p_self) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	self->reverse();
}

void GDAPI godot_packed_glyph_array_push_back(godot_packed_glyph_array *p_self, const godot_glyph *p_data) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	TextServer::Glyph &s = *(TextServer::Glyph *)p_data;
	self->push_back(s);
}

void GDAPI godot_packed_glyph_array_remove(godot_packed_glyph_array *p_self, const godot_int p_idx) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_packed_glyph_array_resize(godot_packed_glyph_array *p_self, const godot_int p_size) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	self->resize(p_size);
}

void GDAPI godot_packed_glyph_array_set(godot_packed_glyph_array *p_self, const godot_int p_idx, const godot_glyph *p_data) {
	Vector<TextServer::Glyph> *self = (Vector<TextServer::Glyph> *)p_self;
	TextServer::Glyph &s = *(TextServer::Glyph *)p_data;
	self->set(p_idx, s);
}

godot_glyph GDAPI godot_packed_glyph_array_get(const godot_packed_glyph_array *p_self, const godot_int p_idx) {
	const Vector<TextServer::Glyph> *self = (const Vector<TextServer::Glyph> *)p_self;
	godot_glyph v;
	TextServer::Glyph *s = (TextServer::Glyph *)&v;
	*s = self->get(p_idx);
	return v;
}

godot_int GDAPI godot_packed_glyph_array_size(const godot_packed_glyph_array *p_self) {
	const Vector<TextServer::Glyph> *self = (const Vector<TextServer::Glyph> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_packed_glyph_array_is_empty(const godot_packed_glyph_array *p_self) {
	const Vector<TextServer::Glyph> *self = (const Vector<TextServer::Glyph> *)p_self;
	return self->is_empty();
}

void GDAPI godot_packed_glyph_array_destroy(godot_packed_glyph_array *p_self) {
	((Vector<TextServer::Glyph> *)p_self)->~Vector();
}

#ifdef __cplusplus
}
#endif
