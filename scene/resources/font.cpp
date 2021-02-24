/*************************************************************************/
/*  font.cpp                                                             */
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

#include "font.h"

#include "core/io/resource_loader.h"
#include "core/string/translation.h"
#include "core/templates/hashfuncs.h"
#include "scene/resources/text_line.h"
#include "scene/resources/text_paragraph.h"

void FontData::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_resource", "filename", "base_size"), &FontData::load_resource, DEFVAL(16));
	ClassDB::bind_method(D_METHOD("load_memory", "data", "type", "base_size"), &FontData::_load_memory, DEFVAL(16));
	ClassDB::bind_method(D_METHOD("new_bitmap", "height", "ascent", "base_size"), &FontData::new_bitmap);

	ClassDB::bind_method(D_METHOD("bitmap_add_texture", "texture"), &FontData::bitmap_add_texture);
	ClassDB::bind_method(D_METHOD("bitmap_add_char", "char", "texture_idx", "rect", "align", "advance"), &FontData::bitmap_add_char);
	ClassDB::bind_method(D_METHOD("bitmap_add_kerning_pair", "A", "B", "kerning"), &FontData::bitmap_add_kerning_pair);

	ClassDB::bind_method(D_METHOD("set_data_path", "path"), &FontData::set_data_path);
	ClassDB::bind_method(D_METHOD("get_data_path"), &FontData::get_data_path);

	ClassDB::bind_method(D_METHOD("get_height", "size"), &FontData::get_height);
	ClassDB::bind_method(D_METHOD("get_ascent", "size"), &FontData::get_ascent);
	ClassDB::bind_method(D_METHOD("get_descent", "size"), &FontData::get_descent);

	ClassDB::bind_method(D_METHOD("get_underline_position", "size"), &FontData::get_underline_position);
	ClassDB::bind_method(D_METHOD("get_underline_thickness", "size"), &FontData::get_underline_thickness);

	ClassDB::bind_method(D_METHOD("get_spacing", "type"), &FontData::get_spacing);
	ClassDB::bind_method(D_METHOD("set_spacing", "type", "value"), &FontData::set_spacing);

	ClassDB::bind_method(D_METHOD("set_antialiased", "antialiased"), &FontData::set_antialiased);
	ClassDB::bind_method(D_METHOD("get_antialiased"), &FontData::get_antialiased);

	ClassDB::bind_method(D_METHOD("get_variation_list"), &FontData::get_variation_list);

	ClassDB::bind_method(D_METHOD("set_variation", "tag", "value"), &FontData::set_variation);
	ClassDB::bind_method(D_METHOD("get_variation", "tag"), &FontData::get_variation);

	ClassDB::bind_method(D_METHOD("set_hinting", "hinting"), &FontData::set_hinting);
	ClassDB::bind_method(D_METHOD("get_hinting"), &FontData::get_hinting);

	ClassDB::bind_method(D_METHOD("set_force_autohinter", "enabled"), &FontData::set_force_autohinter);
	ClassDB::bind_method(D_METHOD("get_force_autohinter"), &FontData::get_force_autohinter);

	ClassDB::bind_method(D_METHOD("set_distance_field_hint", "distance_field"), &FontData::set_distance_field_hint);
	ClassDB::bind_method(D_METHOD("get_distance_field_hint"), &FontData::get_distance_field_hint);

	ClassDB::bind_method(D_METHOD("has_char", "char"), &FontData::has_char);
	ClassDB::bind_method(D_METHOD("get_supported_chars"), &FontData::get_supported_chars);

	ClassDB::bind_method(D_METHOD("get_glyph_advance", "index", "size"), &FontData::get_glyph_advance);
	ClassDB::bind_method(D_METHOD("get_glyph_kerning", "index_a", "index_b", "size"), &FontData::get_glyph_kerning);

	ClassDB::bind_method(D_METHOD("get_base_size"), &FontData::get_base_size);

	ClassDB::bind_method(D_METHOD("has_outline"), &FontData::has_outline);

	ClassDB::bind_method(D_METHOD("is_language_supported", "language"), &FontData::is_language_supported);
	ClassDB::bind_method(D_METHOD("set_language_support_override", "language", "supported"), &FontData::set_language_support_override);
	ClassDB::bind_method(D_METHOD("get_language_support_override", "language"), &FontData::get_language_support_override);
	ClassDB::bind_method(D_METHOD("remove_language_support_override", "language"), &FontData::remove_language_support_override);
	ClassDB::bind_method(D_METHOD("get_language_support_overrides"), &FontData::get_language_support_overrides);

	ClassDB::bind_method(D_METHOD("is_script_supported", "script"), &FontData::is_script_supported);
	ClassDB::bind_method(D_METHOD("set_script_support_override", "script", "supported"), &FontData::set_script_support_override);
	ClassDB::bind_method(D_METHOD("get_script_support_override", "script"), &FontData::get_script_support_override);
	ClassDB::bind_method(D_METHOD("remove_script_support_override", "script"), &FontData::remove_script_support_override);
	ClassDB::bind_method(D_METHOD("get_script_support_overrides"), &FontData::get_script_support_overrides);

	ClassDB::bind_method(D_METHOD("get_glyph_index", "char", "variation_selector"), &FontData::get_glyph_index, DEFVAL(0x0000));
	ClassDB::bind_method(D_METHOD("draw_glyph", "canvas", "size", "pos", "index", "color"), &FontData::draw_glyph, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_glyph_outline", "canvas", "size", "outline_size", "pos", "index", "color"), &FontData::draw_glyph_outline, DEFVAL(Color(1, 1, 1)));

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "data_path", PROPERTY_HINT_FILE, "*.ttf,*.otf,*.woff,*.fnt,*.font"), "set_data_path", "get_data_path");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "antialiased"), "set_antialiased", "get_antialiased");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "force_autohinter"), "set_force_autohinter", "get_force_autohinter");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "distance_field_hint"), "set_distance_field_hint", "get_distance_field_hint");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "hinting", PROPERTY_HINT_ENUM, "None,Light,Normal"), "set_hinting", "get_hinting");

	ADD_GROUP("Extra Spacing", "extra_spacing");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "extra_spacing_glyph"), "set_spacing", "get_spacing", SPACING_GLYPH);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "extra_spacing_space"), "set_spacing", "get_spacing", SPACING_SPACE);

	BIND_ENUM_CONSTANT(SPACING_GLYPH);
	BIND_ENUM_CONSTANT(SPACING_SPACE);
}

bool FontData::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
	if (str.begins_with("language_support_override/")) {
		String lang = str.get_slicec('/', 1);
		if (lang == "_new") {
			return false;
		}
		set_language_support_override(lang, p_value);
		return true;
	}
	if (str.begins_with("script_support_override/")) {
		String scr = str.get_slicec('/', 1);
		if (scr == "_new") {
			return false;
		}
		set_script_support_override(scr, p_value);
		return true;
	}
	if (str.begins_with("variation/")) {
		String name = str.get_slicec('/', 1);
		set_variation(name, p_value);
		return true;
	}

	return false;
}

bool FontData::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;
	if (str.begins_with("language_support_override/")) {
		String lang = str.get_slicec('/', 1);
		if (lang == "_new") {
			return true;
		}
		r_ret = get_language_support_override(lang);
		return true;
	}
	if (str.begins_with("script_support_override/")) {
		String scr = str.get_slicec('/', 1);
		if (scr == "_new") {
			return true;
		}
		r_ret = get_script_support_override(scr);
		return true;
	}
	if (str.begins_with("variation/")) {
		String name = str.get_slicec('/', 1);

		r_ret = get_variation(name);
		return true;
	}

	return false;
}

void FontData::_get_property_list(List<PropertyInfo> *p_list) const {
	Vector<String> lang_over = get_language_support_overrides();
	for (int i = 0; i < lang_over.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "language_support_override/" + lang_over[i], PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_STORAGE));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "language_support_override/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));

	Vector<String> scr_over = get_script_support_overrides();
	for (int i = 0; i < scr_over.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "script_support_override/" + scr_over[i], PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_STORAGE));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "script_support_override/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));

	Dictionary variations = get_variation_list();
	for (const Variant *ftr = variations.next(nullptr); ftr != nullptr; ftr = variations.next(ftr)) {
		Vector3i v = variations[*ftr];
		p_list->push_back(PropertyInfo(Variant::FLOAT, "variation/" + TS->tag_to_name(*ftr), PROPERTY_HINT_RANGE, itos(v.x) + "," + itos(v.y), PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_STORAGE));
	}
}

void FontData::reset_state() {
	if (rid != RID()) {
		TS->free(rid);
	}
	base_size = 16;
	path = String();
}

RID FontData::get_rid() const {
	return rid;
}

void FontData::load_resource(const String &p_filename, int p_base_size) {
	if (rid != RID()) {
		TS->free(rid);
	}
	rid = TS->create_font_resource(p_filename, p_base_size);
	path = p_filename;
	base_size = TS->font_get_base_size(rid);
	emit_changed();
}

void FontData::_load_memory(const PackedByteArray &p_data, const String &p_type, int p_base_size) {
	if (rid != RID()) {
		TS->free(rid);
	}
	rid = TS->create_font_memory(p_data.ptr(), p_data.size(), p_type, p_base_size);
	path = TTR("(Memory: " + p_type.to_upper() + " @ 0x" + String::num_int64((uint64_t)p_data.ptr(), 16, true) + ")");
	base_size = TS->font_get_base_size(rid);
	emit_changed();
}

void FontData::load_memory(const uint8_t *p_data, size_t p_size, const String &p_type, int p_base_size) {
	if (rid != RID()) {
		TS->free(rid);
	}
	rid = TS->create_font_memory(p_data, p_size, p_type, p_base_size);
	path = TTR("(Memory: " + p_type.to_upper() + " @ 0x" + String::num_int64((uint64_t)p_data, 16, true) + ")");
	base_size = TS->font_get_base_size(rid);
	emit_changed();
}

void FontData::new_bitmap(float p_height, float p_ascent, int p_base_size) {
	if (rid != RID()) {
		TS->free(rid);
	}
	rid = TS->create_font_bitmap(p_height, p_ascent, p_base_size);
	path = TTR("(Bitmap: " + String::num_int64(rid.get_id(), 16, true) + ")");
	base_size = TS->font_get_base_size(rid);
	emit_changed();
}

void FontData::bitmap_add_texture(const Ref<Texture> &p_texture) {
	if (rid != RID()) {
		TS->font_bitmap_add_texture(rid, p_texture);
	}
}

void FontData::bitmap_add_char(char32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance) {
	if (rid != RID()) {
		TS->font_bitmap_add_char(rid, p_char, p_texture_idx, p_rect, p_align, p_advance);
	}
}

void FontData::bitmap_add_kerning_pair(char32_t p_A, char32_t p_B, int p_kerning) {
	if (rid != RID()) {
		TS->font_bitmap_add_kerning_pair(rid, p_A, p_B, p_kerning);
	}
}

void FontData::set_data_path(const String &p_path) {
	load_resource(p_path, base_size);
}

String FontData::get_data_path() const {
	return path;
}

float FontData::get_height(int p_size) const {
	if (rid == RID()) {
		return 0.f; // Do not raise errors in getters, to prevent editor from spamming errors on incomplete (without data_path set) fonts.
	}
	return TS->font_get_height(rid, (p_size < 0) ? base_size : p_size);
}

float FontData::get_ascent(int p_size) const {
	if (rid == RID()) {
		return 0.f;
	}
	return TS->font_get_ascent(rid, (p_size < 0) ? base_size : p_size);
}

float FontData::get_descent(int p_size) const {
	if (rid == RID()) {
		return 0.f;
	}
	return TS->font_get_descent(rid, (p_size < 0) ? base_size : p_size);
}

float FontData::get_underline_position(int p_size) const {
	if (rid == RID()) {
		return 0.f;
	}
	return TS->font_get_underline_position(rid, (p_size < 0) ? base_size : p_size);
}

Dictionary FontData::get_feature_list() const {
	if (rid == RID()) {
		return Dictionary();
	}
	return TS->font_get_feature_list(rid);
}

float FontData::get_underline_thickness(int p_size) const {
	if (rid == RID()) {
		return 0.f;
	}
	return TS->font_get_underline_thickness(rid, (p_size < 0) ? base_size : p_size);
}

Dictionary FontData::get_variation_list() const {
	if (rid == RID()) {
		return Dictionary();
	}
	return TS->font_get_variation_list(rid);
}

void FontData::set_variation(const String &p_name, double p_value) {
	ERR_FAIL_COND(rid == RID());
	TS->font_set_variation(rid, p_name, p_value);
	emit_changed();
}

double FontData::get_variation(const String &p_name) const {
	if (rid == RID()) {
		return 0;
	}
	return TS->font_get_variation(rid, p_name);
}

int FontData::get_spacing(int p_type) const {
	if (rid == RID()) {
		return 0;
	}
	if (p_type == SPACING_GLYPH) {
		return TS->font_get_spacing_glyph(rid);
	} else {
		return TS->font_get_spacing_space(rid);
	}
}

void FontData::set_spacing(int p_type, int p_value) {
	ERR_FAIL_COND(rid == RID());
	if (p_type == SPACING_GLYPH) {
		TS->font_set_spacing_glyph(rid, p_value);
	} else {
		TS->font_set_spacing_space(rid, p_value);
	}
	emit_changed();
}

void FontData::set_antialiased(bool p_antialiased) {
	ERR_FAIL_COND(rid == RID());
	TS->font_set_antialiased(rid, p_antialiased);
	emit_changed();
}

bool FontData::get_antialiased() const {
	if (rid == RID()) {
		return false;
	}
	return TS->font_get_antialiased(rid);
}

void FontData::set_distance_field_hint(bool p_distance_field) {
	ERR_FAIL_COND(rid == RID());
	TS->font_set_distance_field_hint(rid, p_distance_field);
	emit_changed();
}

bool FontData::get_distance_field_hint() const {
	if (rid == RID()) {
		return false;
	}
	return TS->font_get_distance_field_hint(rid);
}

void FontData::set_hinting(TextServer::Hinting p_hinting) {
	ERR_FAIL_COND(rid == RID());
	TS->font_set_hinting(rid, p_hinting);
	emit_changed();
}

TextServer::Hinting FontData::get_hinting() const {
	if (rid == RID()) {
		return TextServer::HINTING_NONE;
	}
	return TS->font_get_hinting(rid);
}

void FontData::set_force_autohinter(bool p_enabeld) {
	ERR_FAIL_COND(rid == RID());
	TS->font_set_force_autohinter(rid, p_enabeld);
	emit_changed();
}

bool FontData::get_force_autohinter() const {
	if (rid == RID()) {
		return false;
	}
	return TS->font_get_force_autohinter(rid);
}

bool FontData::has_char(char32_t p_char) const {
	if (rid == RID()) {
		return false;
	}
	return TS->font_has_char(rid, p_char);
}

String FontData::get_supported_chars() const {
	ERR_FAIL_COND_V(rid == RID(), String());
	return TS->font_get_supported_chars(rid);
}

Vector2 FontData::get_glyph_advance(uint32_t p_index, int p_size) const {
	ERR_FAIL_COND_V(rid == RID(), Vector2());
	return TS->font_get_glyph_advance(rid, p_index, (p_size < 0) ? base_size : p_size);
}

Vector2 FontData::get_glyph_kerning(uint32_t p_index_a, uint32_t p_index_b, int p_size) const {
	ERR_FAIL_COND_V(rid == RID(), Vector2());
	return TS->font_get_glyph_kerning(rid, p_index_a, p_index_b, (p_size < 0) ? base_size : p_size);
}

bool FontData::has_outline() const {
	if (rid == RID()) {
		return false;
	}
	return TS->font_has_outline(rid);
}

float FontData::get_base_size() const {
	return base_size;
}

bool FontData::is_language_supported(const String &p_language) const {
	if (rid == RID()) {
		return false;
	}
	return TS->font_is_language_supported(rid, p_language);
}

void FontData::set_language_support_override(const String &p_language, bool p_supported) {
	ERR_FAIL_COND(rid == RID());
	TS->font_set_language_support_override(rid, p_language, p_supported);
	emit_changed();
}

bool FontData::get_language_support_override(const String &p_language) const {
	if (rid == RID()) {
		return false;
	}
	return TS->font_get_language_support_override(rid, p_language);
}

void FontData::remove_language_support_override(const String &p_language) {
	ERR_FAIL_COND(rid == RID());
	TS->font_remove_language_support_override(rid, p_language);
	emit_changed();
}

Vector<String> FontData::get_language_support_overrides() const {
	if (rid == RID()) {
		return Vector<String>();
	}
	return TS->font_get_language_support_overrides(rid);
}

bool FontData::is_script_supported(const String &p_script) const {
	if (rid == RID()) {
		return false;
	}
	return TS->font_is_script_supported(rid, p_script);
}

void FontData::set_script_support_override(const String &p_script, bool p_supported) {
	ERR_FAIL_COND(rid == RID());
	TS->font_set_script_support_override(rid, p_script, p_supported);
	emit_changed();
}

bool FontData::get_script_support_override(const String &p_script) const {
	if (rid == RID()) {
		return false;
	}
	return TS->font_get_script_support_override(rid, p_script);
}

void FontData::remove_script_support_override(const String &p_script) {
	ERR_FAIL_COND(rid == RID());
	TS->font_remove_script_support_override(rid, p_script);
	emit_changed();
}

Vector<String> FontData::get_script_support_overrides() const {
	if (rid == RID()) {
		return Vector<String>();
	}
	return TS->font_get_script_support_overrides(rid);
}

uint32_t FontData::get_glyph_index(char32_t p_char, char32_t p_variation_selector) const {
	ERR_FAIL_COND_V(rid == RID(), 0);
	return TS->font_get_glyph_index(rid, p_char, p_variation_selector);
}

Vector2 FontData::draw_glyph(RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	ERR_FAIL_COND_V(rid == RID(), Vector2());
	return TS->font_draw_glyph(rid, p_canvas, (p_size <= 0) ? base_size : p_size, p_pos, p_index, p_color);
}

Vector2 FontData::draw_glyph_outline(RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	ERR_FAIL_COND_V(rid == RID(), Vector2());
	return TS->font_draw_glyph_outline(rid, p_canvas, (p_size <= 0) ? base_size : p_size, p_outline_size, p_pos, p_index, p_color);
}

FontData::FontData() {}

FontData::FontData(const String &p_filename, int p_base_size) {
	load_resource(p_filename, p_base_size);
}

FontData::FontData(const PackedByteArray &p_data, const String &p_type, int p_base_size) {
	_load_memory(p_data, p_type, p_base_size);
}

FontData::~FontData() {
	if (rid != RID()) {
		TS->free(rid);
	}
}

/*************************************************************************/

void Font::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_data", "data"), &Font::add_data);
	ClassDB::bind_method(D_METHOD("set_data", "idx", "data"), &Font::set_data);
	ClassDB::bind_method(D_METHOD("get_data_count"), &Font::get_data_count);
	ClassDB::bind_method(D_METHOD("get_data", "idx"), &Font::get_data);
	ClassDB::bind_method(D_METHOD("remove_data", "idx"), &Font::remove_data);

	ClassDB::bind_method(D_METHOD("get_height", "size"), &Font::get_height, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("get_ascent", "size"), &Font::get_ascent, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("get_descent", "size"), &Font::get_descent, DEFVAL(-1));

	ClassDB::bind_method(D_METHOD("get_underline_position", "size"), &Font::get_underline_position, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("get_underline_thickness", "size"), &Font::get_underline_thickness, DEFVAL(-1));

	ClassDB::bind_method(D_METHOD("get_spacing", "type"), &Font::get_spacing);
	ClassDB::bind_method(D_METHOD("set_spacing", "type", "value"), &Font::set_spacing);

	ClassDB::bind_method(D_METHOD("get_string_size", "text", "size"), &Font::get_string_size, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("get_multiline_string_size", "text", "width", "size", "flags"), &Font::get_multiline_string_size, DEFVAL(-1), DEFVAL(-1), DEFVAL(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND));

	ClassDB::bind_method(D_METHOD("draw_string", "canvas_item", "pos", "text", "align", "width", "size", "modulate", "outline_size", "outline_modulate", "flags"), &Font::draw_string, DEFVAL(HALIGN_LEFT), DEFVAL(-1), DEFVAL(-1), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(1, 1, 1, 0)), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND));
	ClassDB::bind_method(D_METHOD("draw_multiline_string", "canvas_item", "pos", "text", "align", "width", "max_lines", "size", "modulate", "outline_size", "outline_modulate", "flags"), &Font::draw_multiline_string, DEFVAL(HALIGN_LEFT), DEFVAL(-1), DEFVAL(-1), DEFVAL(-1), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(1, 1, 1, 0)), DEFVAL(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND));

	ClassDB::bind_method(D_METHOD("has_char", "char"), &Font::has_char);
	ClassDB::bind_method(D_METHOD("get_supported_chars"), &Font::get_supported_chars);

	ClassDB::bind_method(D_METHOD("get_char_size", "char", "next", "size"), &Font::get_char_size, DEFVAL(0), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("draw_char", "canvas_item", "pos", "char", "next", "size", "modulate", "outline_size", "outline_modulate"), &Font::draw_char, DEFVAL(0), DEFVAL(-1), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(1, 1, 1, 0)));

	ClassDB::bind_method(D_METHOD("update_changes"), &Font::update_changes);

	ADD_GROUP("Extra Spacing", "extra_spacing");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "extra_spacing_top"), "set_spacing", "get_spacing", SPACING_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "extra_spacing_bottom"), "set_spacing", "get_spacing", SPACING_BOTTOM);

	BIND_ENUM_CONSTANT(SPACING_TOP);
	BIND_ENUM_CONSTANT(SPACING_BOTTOM);
}

void Font::_data_changed() {
	cache.clear();
	cache_wrap.clear();

	emit_changed();
	notify_property_list_changed();
}

bool Font::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
#ifndef DISABLE_DEPRECATED
	if (str == "font_data") { // Compatibility, DynamicFont main data
		Ref<FontData> fd = p_value;
		if (fd.is_valid()) {
			add_data(fd);
			return true;
		}
		return false;
	} else if (str.begins_with("fallback/")) { // Compatibility, DynamicFont fallback data
		Ref<FontData> fd = p_value;
		if (fd.is_valid()) {
			add_data(fd);
			return true;
		}
		return false;
	} else if (str == "fallback") { // Compatibility, BitmapFont fallback
		Ref<Font> f = p_value;
		if (f.is_valid()) {
			for (int i = 0; i < f->get_data_count(); i++) {
				add_data(f->get_data(i));
			}
			return true;
		}
		return false;
	}
#endif /* DISABLE_DEPRECATED */
	if (str.begins_with("data/")) {
		int idx = str.get_slicec('/', 1).to_int();
		Ref<FontData> fd = p_value;

		if (fd.is_valid()) {
			if (idx == data.size()) {
				add_data(fd);
				return true;
			} else if (idx >= 0 && idx < data.size()) {
				set_data(idx, fd);
				return true;
			} else {
				return false;
			}
		} else if (idx >= 0 && idx < data.size()) {
			remove_data(idx);
			return true;
		}
	}

	return false;
}

bool Font::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;
	if (str.begins_with("data/")) {
		int idx = str.get_slicec('/', 1).to_int();

		if (idx == data.size()) {
			r_ret = Ref<FontData>();
			return true;
		} else if (idx >= 0 && idx < data.size()) {
			r_ret = get_data(idx);
			return true;
		}
	}

	return false;
}

void Font::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < data.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "data/" + itos(i), PROPERTY_HINT_RESOURCE_TYPE, "FontData"));
	}

	p_list->push_back(PropertyInfo(Variant::OBJECT, "data/" + itos(data.size()), PROPERTY_HINT_RESOURCE_TYPE, "FontData"));
}

void Font::reset_state() {
	spacing_top = 0;
	spacing_bottom = 0;
	cache.clear();
	cache_wrap.clear();
	data.clear();
}

void Font::add_data(const Ref<FontData> &p_data) {
	ERR_FAIL_COND(p_data.is_null());
	data.push_back(p_data);

	if (data[data.size() - 1].is_valid()) {
		data.write[data.size() - 1]->connect("changed", callable_mp(this, &Font::_data_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}

	cache.clear();
	cache_wrap.clear();

	emit_changed();
	notify_property_list_changed();
}

void Font::set_data(int p_idx, const Ref<FontData> &p_data) {
	ERR_FAIL_COND(p_data.is_null());
	ERR_FAIL_INDEX(p_idx, data.size());

	if (data[p_idx].is_valid()) {
		data.write[p_idx]->disconnect("changed", callable_mp(this, &Font::_data_changed));
	}

	data.write[p_idx] = p_data;

	if (data[p_idx].is_valid()) {
		data.write[p_idx]->connect("changed", callable_mp(this, &Font::_data_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}

	cache.clear();
	cache_wrap.clear();

	emit_changed();
	notify_property_list_changed();
}

int Font::get_data_count() const {
	return data.size();
}

Ref<FontData> Font::get_data(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, data.size(), Ref<FontData>());
	return data[p_idx];
}

void Font::remove_data(int p_idx) {
	ERR_FAIL_INDEX(p_idx, data.size());

	if (data[p_idx].is_valid()) {
		data.write[p_idx]->disconnect("changed", callable_mp(this, &Font::_data_changed));
	}

	data.remove(p_idx);

	cache.clear();
	cache_wrap.clear();

	emit_changed();
	notify_property_list_changed();
}

Dictionary Font::get_feature_list() const {
	Dictionary out;
	for (int i = 0; i < data.size(); i++) {
		Dictionary data_ftrs = data[i]->get_feature_list();
		for (const Variant *ftr = data_ftrs.next(nullptr); ftr != nullptr; ftr = data_ftrs.next(ftr)) {
			out[*ftr] = data_ftrs[*ftr];
		}
	}
	return out;
}

float Font::get_height(int p_size) const {
	float ret = 0.f;
	for (int i = 0; i < data.size(); i++) {
		ret = MAX(ret, data[i]->get_height(p_size));
	}
	return ret + spacing_top + spacing_bottom;
}

float Font::get_ascent(int p_size) const {
	float ret = 0.f;
	for (int i = 0; i < data.size(); i++) {
		ret = MAX(ret, data[i]->get_ascent(p_size));
	}
	return ret + spacing_top;
}

float Font::get_descent(int p_size) const {
	float ret = 0.f;
	for (int i = 0; i < data.size(); i++) {
		ret = MAX(ret, data[i]->get_descent(p_size));
	}
	return ret + spacing_bottom;
}

float Font::get_underline_position(int p_size) const {
	float ret = 0.f;
	for (int i = 0; i < data.size(); i++) {
		ret = MAX(ret, data[i]->get_underline_position(p_size));
	}
	return ret;
}

float Font::get_underline_thickness(int p_size) const {
	float ret = 0.f;
	for (int i = 0; i < data.size(); i++) {
		ret = MAX(ret, data[i]->get_underline_thickness(p_size));
	}
	return ret;
}

int Font::get_spacing(int p_type) const {
	if (p_type == SPACING_TOP) {
		return spacing_top;
	} else if (p_type == SPACING_BOTTOM) {
		return spacing_bottom;
	}

	return 0;
}

void Font::set_spacing(int p_type, int p_value) {
	if (p_type == SPACING_TOP) {
		spacing_top = p_value;
	} else if (p_type == SPACING_BOTTOM) {
		spacing_bottom = p_value;
	}

	emit_changed();
	notify_property_list_changed();
}

// Drawing string and string sizes, cached.

Size2 Font::get_string_size(const String &p_text, int p_size) const {
	ERR_FAIL_COND_V(data.is_empty(), Size2());

	uint64_t hash = p_text.hash64();
	hash = hash_djb2_one_64(p_size, hash);

	Ref<TextLine> buffer;
	if (cache.has(hash)) {
		buffer = cache.get(hash);
	} else {
		buffer.instance();
		int size = p_size <= 0 ? data[0]->get_base_size() : p_size;
		buffer->add_string(p_text, Ref<Font>(this), size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		cache.insert(hash, buffer);
	}
	if (buffer->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
		return buffer->get_size() + Vector2(0, spacing_top + spacing_bottom);
	} else {
		return buffer->get_size() + Vector2(spacing_top + spacing_bottom, 0);
	}
}

Size2 Font::get_multiline_string_size(const String &p_text, float p_width, int p_size, uint8_t p_flags) const {
	ERR_FAIL_COND_V(data.is_empty(), Size2());

	uint64_t hash = p_text.hash64();
	hash = hash_djb2_one_64(p_size, hash);

	uint64_t wrp_hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
	wrp_hash = hash_djb2_one_64(p_flags, wrp_hash);

	Ref<TextParagraph> lines_buffer;
	if (cache_wrap.has(wrp_hash)) {
		lines_buffer = cache_wrap.get(wrp_hash);
	} else {
		lines_buffer.instance();
		int size = p_size <= 0 ? data[0]->get_base_size() : p_size;
		lines_buffer->add_string(p_text, Ref<Font>(this), size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		lines_buffer->set_width(p_width);
		lines_buffer->set_flags(p_flags);
		cache_wrap.insert(wrp_hash, lines_buffer);
	}

	Size2 ret;
	for (int i = 0; i < lines_buffer->get_line_count(); i++) {
		Size2 line_size = lines_buffer->get_line_size(i);
		if (lines_buffer->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
			ret.x = MAX(ret.x, line_size.x);
			ret.y += line_size.y + spacing_top + spacing_bottom;
		} else {
			ret.y = MAX(ret.y, line_size.y);
			ret.x += line_size.x + spacing_top + spacing_bottom;
		}
	}
	return ret;
}

void Font::draw_string(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HAlign p_align, float p_width, int p_size, const Color &p_modulate, int p_outline_size, const Color &p_outline_modulate, uint8_t p_flags) const {
	ERR_FAIL_COND(data.is_empty());

	uint64_t hash = p_text.hash64();
	hash = hash_djb2_one_64(p_size, hash);

	Ref<TextLine> buffer;
	if (cache.has(hash)) {
		buffer = cache.get(hash);
	} else {
		buffer.instance();
		int size = p_size <= 0 ? data[0]->get_base_size() : p_size;
		buffer->add_string(p_text, Ref<Font>(this), size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		cache.insert(hash, buffer);
	}

	Vector2 ofs = p_pos;
	if (buffer->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += spacing_top - buffer->get_line_ascent();
	} else {
		ofs.x += spacing_top - buffer->get_line_ascent();
	}

	buffer->set_width(p_width);
	buffer->set_align(p_align);

	if (p_outline_size > 0 && p_outline_modulate.a != 0.0f) {
		buffer->draw_outline(p_canvas_item, ofs, p_outline_size, p_outline_modulate);
	}
	buffer->draw(p_canvas_item, ofs, p_modulate);
}

void Font::draw_multiline_string(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HAlign p_align, float p_width, int p_max_lines, int p_size, const Color &p_modulate, int p_outline_size, const Color &p_outline_modulate, uint8_t p_flags) const {
	ERR_FAIL_COND(data.is_empty());

	uint64_t hash = p_text.hash64();
	hash = hash_djb2_one_64(p_size, hash);

	uint64_t wrp_hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
	wrp_hash = hash_djb2_one_64(p_flags, wrp_hash);

	Ref<TextParagraph> lines_buffer;
	if (cache_wrap.has(wrp_hash)) {
		lines_buffer = cache_wrap.get(wrp_hash);
	} else {
		lines_buffer.instance();
		int size = p_size <= 0 ? data[0]->get_base_size() : p_size;
		lines_buffer->add_string(p_text, Ref<Font>(this), size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		lines_buffer->set_width(p_width);
		lines_buffer->set_flags(p_flags);
		cache_wrap.insert(wrp_hash, lines_buffer);
	}

	lines_buffer->set_align(p_align);

	Vector2 lofs = p_pos;
	for (int i = 0; i < lines_buffer->get_line_count(); i++) {
		if (lines_buffer->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
			lofs.y += spacing_top;
			if (i == 0) {
				lofs.y -= lines_buffer->get_line_ascent(0);
			}
		} else {
			lofs.x += spacing_top;
			if (i == 0) {
				lofs.x -= lines_buffer->get_line_ascent(0);
			}
		}
		if (p_width > 0) {
			lines_buffer->set_align(p_align);
		}

		if (p_outline_size > 0 && p_outline_modulate.a != 0.0f) {
			lines_buffer->draw_line_outline(p_canvas_item, lofs, i, p_outline_size, p_outline_modulate);
		}
		lines_buffer->draw_line(p_canvas_item, lofs, i, p_modulate);

		Size2 line_size = lines_buffer->get_line_size(i);
		if (lines_buffer->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
			lofs.y += line_size.y + spacing_bottom;
		} else {
			lofs.x += line_size.x + spacing_bottom;
		}

		if ((p_max_lines > 0) && (i >= p_max_lines)) {
			return;
		}
	}
}

bool Font::has_char(char32_t p_char) const {
	for (int i = 0; i < data.size(); i++) {
		if (data[i]->has_char(p_char)) {
			return true;
		}
	}
	return false;
}

String Font::get_supported_chars() const {
	String chars;
	for (int i = 0; i < data.size(); i++) {
		String data_chars = data[i]->get_supported_chars();
		for (int j = 0; j < data_chars.length(); j++) {
			if (chars.find_char(data_chars[j]) == -1) {
				chars += data_chars[j];
			}
		}
	}
	return chars;
}

Size2 Font::get_char_size(char32_t p_char, char32_t p_next, int p_size) const {
	for (int i = 0; i < data.size(); i++) {
		if (data[i]->has_char(p_char)) {
			int size = p_size <= 0 ? data[i]->get_base_size() : p_size;
			uint32_t glyph_a = data[i]->get_glyph_index(p_char);
			Size2 ret = Size2(data[i]->get_glyph_advance(glyph_a, size).x, data[i]->get_height(size));
			if ((p_next != 0) && data[i]->has_char(p_next)) {
				uint32_t glyph_b = data[i]->get_glyph_index(p_next);
				ret.x -= data[i]->get_glyph_kerning(glyph_a, glyph_b, size).x;
			}
			return ret;
		}
	}
	return Size2();
}

float Font::draw_char(RID p_canvas_item, const Point2 &p_pos, char32_t p_char, char32_t p_next, int p_size, const Color &p_modulate, int p_outline_size, const Color &p_outline_modulate) const {
	for (int i = 0; i < data.size(); i++) {
		if (data[i]->has_char(p_char)) {
			int size = p_size <= 0 ? data[i]->get_base_size() : p_size;
			uint32_t glyph_a = data[i]->get_glyph_index(p_char);
			float ret = data[i]->get_glyph_advance(glyph_a, size).x;
			if ((p_next != 0) && data[i]->has_char(p_next)) {
				uint32_t glyph_b = data[i]->get_glyph_index(p_next);
				ret -= data[i]->get_glyph_kerning(glyph_a, glyph_b, size).x;
			}
			if (p_outline_size > 0 && p_outline_modulate.a != 0.0f) {
				data[i]->draw_glyph_outline(p_canvas_item, size, p_outline_size, p_pos, glyph_a, p_outline_modulate);
			}
			data[i]->draw_glyph(p_canvas_item, size, p_pos, glyph_a, p_modulate);
			return ret;
		}
	}
	return 0;
}

Vector<RID> Font::get_rids() const {
	Vector<RID> ret;
	for (int i = 0; i < data.size(); i++) {
		RID rid = data[i]->get_rid();
		if (rid != RID()) {
			ret.push_back(rid);
		}
	}
	return ret;
}

void Font::update_changes() {
	emit_changed();
}

Font::Font() {
	cache.set_capacity(128);
	cache_wrap.set_capacity(32);
}

Font::~Font() {
	cache.clear();
	cache_wrap.clear();
}

/*************************************************************************/

RES ResourceFormatLoaderFont::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	Ref<FontData> dfont;
	dfont.instance();
	dfont->load_resource(p_path);

	if (r_error) {
		*r_error = OK;
	}

	return dfont;
}

void ResourceFormatLoaderFont::get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const {
#ifndef DISABLE_DEPRECATED
	if (p_type == "DynamicFontData") {
		p_extensions->push_back("ttf");
		p_extensions->push_back("otf");
		p_extensions->push_back("woff");
		return;
	}
	if (p_type == "BitmapFont") { // BitmapFont (*.font, *fnt) is handled by ResourceFormatLoaderCompatFont
		return;
	}
#endif /* DISABLE_DEPRECATED */
	if (p_type == "" || handles_type(p_type)) {
		get_recognized_extensions(p_extensions);
	}
}

void ResourceFormatLoaderFont::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("ttf");
	p_extensions->push_back("otf");
	p_extensions->push_back("woff");
	p_extensions->push_back("font");
	p_extensions->push_back("fnt");
}

bool ResourceFormatLoaderFont::handles_type(const String &p_type) const {
	return (p_type == "FontData");
}

String ResourceFormatLoaderFont::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "ttf" || el == "otf" || el == "woff" || el == "font" || el == "fnt") {
		return "FontData";
	}
	return "";
}

#ifndef DISABLE_DEPRECATED

RES ResourceFormatLoaderCompatFont::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	Ref<FontData> dfont;
	dfont.instance();
	dfont->load_resource(p_path);

	Ref<Font> font;
	font.instance();
	font->add_data(dfont);

	if (r_error) {
		*r_error = OK;
	}

	return font;
}

void ResourceFormatLoaderCompatFont::get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const {
	if (p_type == "BitmapFont") {
		p_extensions->push_back("font");
		p_extensions->push_back("fnt");
	}
}

void ResourceFormatLoaderCompatFont::get_recognized_extensions(List<String> *p_extensions) const {
}

bool ResourceFormatLoaderCompatFont::handles_type(const String &p_type) const {
	return (p_type == "Font");
}

String ResourceFormatLoaderCompatFont::get_resource_type(const String &p_path) const {
	return "";
}

#endif /* DISABLE_DEPRECATED */
