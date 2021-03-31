/*************************************************************************/
/*  text_server_fb.cpp                                                   */
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

#include "text_server_fb.h"

#include "bitmap_font_fb.h"
#include "dynamic_font_fb.h"

_FORCE_INLINE_ bool is_control(char32_t p_char) {
	return (p_char <= 0x001f) || (p_char >= 0x007f && p_char <= 0x009F);
}

_FORCE_INLINE_ bool is_whitespace(char32_t p_char) {
	return (p_char == 0x0020) || (p_char == 0x00A0) || (p_char == 0x1680) || (p_char >= 0x2000 && p_char <= 0x200a) || (p_char == 0x202f) || (p_char == 0x205f) || (p_char == 0x3000) || (p_char == 0x2028) || (p_char == 0x2029) || (p_char >= 0x0009 && p_char <= 0x000d) || (p_char == 0x0085);
}

_FORCE_INLINE_ bool is_linebreak(char32_t p_char) {
	return (p_char >= 0x000a && p_char <= 0x000d) || (p_char == 0x0085) || (p_char == 0x2028) || (p_char == 0x2029);
}

_FORCE_INLINE_ bool is_punct(char32_t p_char) {
	return (p_char >= 0x0020 && p_char <= 0x002F) || (p_char >= 0x003A && p_char <= 0x0040) || (p_char >= 0x005B && p_char <= 0x0060) || (p_char >= 0x007B && p_char <= 0x007E) || (p_char >= 0x2000 && p_char <= 0x206F) || (p_char >= 0x3000 && p_char <= 0x303F);
}

/*************************************************************************/

String TextServerFallback::interface_name = "Fallback";
uint32_t TextServerFallback::interface_features = 0; // Nothing is supported.

bool TextServerFallback::has_feature(Feature p_feature) {
	return (interface_features & p_feature) == p_feature;
}

String TextServerFallback::get_name() const {
	return interface_name;
}

void TextServerFallback::free(RID p_rid) {
	_THREAD_SAFE_METHOD_
	if (font_owner.owns(p_rid)) {
		FontDataFallback *fd = font_owner.getornull(p_rid);
		font_owner.free(p_rid);
		memdelete(fd);
	} else if (shaped_owner.owns(p_rid)) {
		ShapedTextData *sd = shaped_owner.getornull(p_rid);
		shaped_owner.free(p_rid);
		memdelete(sd);
	}
}

bool TextServerFallback::has(RID p_rid) {
	_THREAD_SAFE_METHOD_
	return font_owner.owns(p_rid) || shaped_owner.owns(p_rid);
}

bool TextServerFallback::load_support_data(const String &p_filename) {
	return false; // No extra data used.
}

#ifdef TOOLS_ENABLED

bool TextServerFallback::save_support_data(const String &p_filename) {
	return false; // No extra data used.
}

#endif

bool TextServerFallback::is_locale_right_to_left(const String &p_locale) {
	return false; // No RTL support.
}

/*************************************************************************/
/* Font interface                                                        */
/*************************************************************************/

RID TextServerFallback::create_font_system(const String &p_name, int p_base_size) {
	ERR_FAIL_V_MSG(RID(), "System fonts are not supported by this text server.");
}

RID TextServerFallback::create_font_resource(const String &p_filename, int p_base_size) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = nullptr;
	if (p_filename.get_extension() == "fnt" || p_filename.get_extension() == "font") {
		fd = memnew(BitmapFontDataFallback);
#ifdef MODULE_FREETYPE_ENABLED
	} else if (p_filename.get_extension() == "ttf" || p_filename.get_extension() == "otf" || p_filename.get_extension() == "woff") {
		fd = memnew(DynamicFontDataFallback);
#endif
	} else {
		return RID();
	}

	Error err = fd->load_from_file(p_filename, p_base_size);
	if (err != OK) {
		memdelete(fd);
		return RID();
	}

	return font_owner.make_rid(fd);
}

RID TextServerFallback::create_font_memory(const uint8_t *p_data, size_t p_size, const String &p_type, int p_base_size) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = nullptr;
	if (p_type == "fnt" || p_type == "font") {
		fd = memnew(BitmapFontDataFallback);
#ifdef MODULE_FREETYPE_ENABLED
	} else if (p_type == "ttf" || p_type == "otf" || p_type == "woff") {
		fd = memnew(DynamicFontDataFallback);
#endif
	} else {
		return RID();
	}

	Error err = fd->load_from_memory(p_data, p_size, p_base_size);
	if (err != OK) {
		memdelete(fd);
		return RID();
	}

	return font_owner.make_rid(fd);
}

RID TextServerFallback::create_font_bitmap(float p_height, float p_ascent, int p_base_size) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = memnew(BitmapFontDataFallback);
	Error err = fd->bitmap_new(p_height, p_ascent, p_base_size);
	if (err != OK) {
		memdelete(fd);
		return RID();
	}

	return font_owner.make_rid(fd);
}

void TextServerFallback::font_bitmap_add_texture(RID p_font, const Ref<Texture> &p_texture) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->bitmap_add_texture(p_texture);
}

void TextServerFallback::font_bitmap_add_char(RID p_font, char32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->bitmap_add_char(p_char, p_texture_idx, p_rect, p_align, p_advance);
}

void TextServerFallback::font_bitmap_add_kerning_pair(RID p_font, char32_t p_A, char32_t p_B, int p_kerning) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->bitmap_add_kerning_pair(p_A, p_B, p_kerning);
}

float TextServerFallback::font_get_height(RID p_font, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_height(p_size);
}

float TextServerFallback::font_get_ascent(RID p_font, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_ascent(p_size);
}

float TextServerFallback::font_get_descent(RID p_font, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_descent(p_size);
}

float TextServerFallback::font_get_underline_position(RID p_font, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_underline_position(p_size);
}

float TextServerFallback::font_get_underline_thickness(RID p_font, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_underline_thickness(p_size);
}

int TextServerFallback::font_get_spacing_space(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0);
	return fd->get_spacing_space();
}

void TextServerFallback::font_set_spacing_space(RID p_font, int p_value) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_spacing_space(p_value);
}

int TextServerFallback::font_get_spacing_glyph(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0);
	return fd->get_spacing_glyph();
}

void TextServerFallback::font_set_spacing_glyph(RID p_font, int p_value) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_spacing_glyph(p_value);
}

void TextServerFallback::font_set_antialiased(RID p_font, bool p_antialiased) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_antialiased(p_antialiased);
}

bool TextServerFallback::font_get_antialiased(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->get_antialiased();
}

void TextServerFallback::font_set_distance_field_hint(RID p_font, bool p_distance_field) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_distance_field_hint(p_distance_field);
}

bool TextServerFallback::font_get_distance_field_hint(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->get_distance_field_hint();
}

void TextServerFallback::font_set_hinting(RID p_font, TextServer::Hinting p_hinting) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_hinting(p_hinting);
}

TextServer::Hinting TextServerFallback::font_get_hinting(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, TextServer::HINTING_NONE);
	return fd->get_hinting();
}

void TextServerFallback::font_set_force_autohinter(RID p_font, bool p_enabeld) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_force_autohinter(p_enabeld);
}

bool TextServerFallback::font_get_force_autohinter(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->get_force_autohinter();
}

bool TextServerFallback::font_has_char(RID p_font, char32_t p_char) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->has_char(p_char);
}

String TextServerFallback::font_get_supported_chars(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, String());
	return fd->get_supported_chars();
}

bool TextServerFallback::font_has_outline(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->has_outline();
}

float TextServerFallback::font_get_base_size(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_base_size();
}

bool TextServerFallback::font_is_language_supported(RID p_font, const String &p_language) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	if (fd->lang_support_overrides.has(p_language)) {
		return fd->lang_support_overrides[p_language];
	} else {
		Vector<String> tags = p_language.replace("-", "_").split("_");
		if (tags.size() > 0) {
			if (fd->lang_support_overrides.has(tags[0])) {
				return fd->lang_support_overrides[tags[0]];
			}
		}
		return false;
	}
}

void TextServerFallback::font_set_language_support_override(RID p_font, const String &p_language, bool p_supported) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->lang_support_overrides[p_language] = p_supported;
}

bool TextServerFallback::font_get_language_support_override(RID p_font, const String &p_language) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->lang_support_overrides[p_language];
}

void TextServerFallback::font_remove_language_support_override(RID p_font, const String &p_language) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->lang_support_overrides.erase(p_language);
}

Vector<String> TextServerFallback::font_get_language_support_overrides(RID p_font) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector<String>());
	Vector<String> ret;
	for (Map<String, bool>::Element *E = fd->lang_support_overrides.front(); E; E = E->next()) {
		ret.push_back(E->key());
	}
	return ret;
}

bool TextServerFallback::font_is_script_supported(RID p_font, const String &p_script) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	if (fd->script_support_overrides.has(p_script)) {
		return fd->script_support_overrides[p_script];
	} else {
		return true;
	}
}

void TextServerFallback::font_set_script_support_override(RID p_font, const String &p_script, bool p_supported) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->script_support_overrides[p_script] = p_supported;
}

bool TextServerFallback::font_get_script_support_override(RID p_font, const String &p_script) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->script_support_overrides[p_script];
}

void TextServerFallback::font_remove_script_support_override(RID p_font, const String &p_script) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->script_support_overrides.erase(p_script);
}

Vector<String> TextServerFallback::font_get_script_support_overrides(RID p_font) {
	_THREAD_SAFE_METHOD_
	FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector<String>());
	Vector<String> ret;
	for (Map<String, bool>::Element *E = fd->script_support_overrides.front(); E; E = E->next()) {
		ret.push_back(E->key());
	}
	return ret;
}

uint32_t TextServerFallback::font_get_glyph_index(RID p_font, char32_t p_char, char32_t p_variation_selector) const {
	return (uint32_t)p_char;
}

Vector2 TextServerFallback::font_get_glyph_advance(RID p_font, uint32_t p_index, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector2());
	return fd->get_advance(p_index, p_size);
}

Vector2 TextServerFallback::font_get_glyph_kerning(RID p_font, uint32_t p_index_a, uint32_t p_index_b, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector2());
	return fd->get_kerning(p_index_a, p_index_b, p_size);
}

Vector2 TextServerFallback::font_draw_glyph(RID p_font, RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector2());
	return fd->draw_glyph(p_canvas, p_size, p_pos, p_index, p_color);
}

Vector2 TextServerFallback::font_draw_glyph_outline(RID p_font, RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector2());
	return fd->draw_glyph_outline(p_canvas, p_size, p_outline_size, p_pos, p_index, p_color);
}

bool TextServerFallback::font_get_glyph_contours(RID p_font, int p_size, uint32_t p_index, Vector<Vector3> &r_points, Vector<int32_t> &r_contours, bool &r_orientation) const {
	_THREAD_SAFE_METHOD_
	const FontDataFallback *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->get_glyph_contours(p_size, p_index, r_points, r_contours, r_orientation);
}

float TextServerFallback::font_get_oversampling() const {
	return oversampling;
}

void TextServerFallback::font_set_oversampling(float p_oversampling) {
	_THREAD_SAFE_METHOD_
	if (oversampling != p_oversampling) {
		oversampling = p_oversampling;
		List<RID> fonts;
		font_owner.get_owned_list(&fonts);
		for (List<RID>::Element *E = fonts.front(); E; E = E->next()) {
			font_owner.getornull(E->get())->clear_cache();
		}
	}
}

Vector<String> TextServerFallback::get_system_fonts() const {
	return Vector<String>();
}

/*************************************************************************/
/* Shaped text buffer interface                                          */
/*************************************************************************/

void TextServerFallback::invalidate(ShapedTextData *p_shaped) {
	p_shaped->valid = false;
	p_shaped->sort_valid = false;
	p_shaped->line_breaks_valid = false;
	p_shaped->justification_ops_valid = false;
	p_shaped->ascent = 0.f;
	p_shaped->descent = 0.f;
	p_shaped->width = 0.f;
	p_shaped->upos = 0.f;
	p_shaped->uthk = 0.f;
	p_shaped->glyphs.clear();
	p_shaped->glyphs_logical.clear();
}

void TextServerFallback::full_copy(ShapedTextData *p_shaped) {
	ShapedTextData *parent = shaped_owner.getornull(p_shaped->parent);

	for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = parent->objects.front(); E; E = E->next()) {
		if (E->get().pos >= p_shaped->start && E->get().pos < p_shaped->end) {
			p_shaped->objects[E->key()] = E->get();
		}
	}

	for (int k = 0; k < parent->spans.size(); k++) {
		ShapedTextData::Span span = parent->spans[k];
		if (span.start >= p_shaped->end || span.end <= p_shaped->start) {
			continue;
		}
		span.start = MAX(p_shaped->start, span.start);
		span.end = MIN(p_shaped->end, span.end);
		p_shaped->spans.push_back(span);
	}

	p_shaped->parent = RID();
}

RID TextServerFallback::create_shaped_text(TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = memnew(ShapedTextData);
	sd->direction = p_direction;
	sd->orientation = p_orientation;

	return shaped_owner.make_rid(sd);
}

void TextServerFallback::shaped_text_clear(RID p_shaped) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND(!sd);

	sd->parent = RID();
	sd->start = 0;
	sd->end = 0;
	sd->text = String();
	sd->spans.clear();
	sd->objects.clear();
	invalidate(sd);
}

void TextServerFallback::shaped_text_set_direction(RID p_shaped, TextServer::Direction p_direction) {
	if (p_direction == DIRECTION_RTL) {
		ERR_PRINT_ONCE("Right-to-left layout is not supported by this text server.");
	}
}

TextServer::Direction TextServerFallback::shaped_text_get_direction(RID p_shaped) const {
	return TextServer::DIRECTION_LTR;
}

void TextServerFallback::shaped_text_set_orientation(RID p_shaped, TextServer::Orientation p_orientation) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND(!sd);
	if (sd->orientation != p_orientation) {
		if (sd->parent != RID()) {
			full_copy(sd);
		}
		sd->orientation = p_orientation;
		invalidate(sd);
	}
}

void TextServerFallback::shaped_text_set_bidi_override(RID p_shaped, const Vector<Vector2i> &p_override) {
	//No BiDi support, ignore.
}

TextServer::Orientation TextServerFallback::shaped_text_get_orientation(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, TextServer::ORIENTATION_HORIZONTAL);
	return sd->orientation;
}

void TextServerFallback::shaped_text_set_preserve_invalid(RID p_shaped, bool p_enabled) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND(!sd);
	if (sd->preserve_invalid != p_enabled) {
		if (sd->parent != RID()) {
			full_copy(sd);
		}
		sd->preserve_invalid = p_enabled;
		invalidate(sd);
	}
}

bool TextServerFallback::shaped_text_get_preserve_invalid(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	return sd->preserve_invalid;
}

void TextServerFallback::shaped_text_set_preserve_control(RID p_shaped, bool p_enabled) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND(!sd);
	if (sd->preserve_control != p_enabled) {
		if (sd->parent != RID()) {
			full_copy(sd);
		}
		sd->preserve_control = p_enabled;
		invalidate(sd);
	}
}

bool TextServerFallback::shaped_text_get_preserve_control(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	return sd->preserve_control;
}

bool TextServerFallback::shaped_text_add_string(RID p_shaped, const String &p_text, const Vector<RID> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	ERR_FAIL_COND_V(p_size <= 0, false);

	if (p_text.is_empty()) {
		return true;
	}

	if (sd->parent != RID()) {
		full_copy(sd);
	}

	ShapedTextData::Span span;
	span.start = sd->text.length();
	span.end = span.start + p_text.length();
	// Pre-sort fonts, push fonts with the language support first.
	for (int i = 0; i < p_fonts.size(); i++) {
		if (font_is_language_supported(p_fonts[i], p_language)) {
			span.fonts.push_back(p_fonts[i]);
		}
	}
	// Push the rest valid fonts.
	for (int i = 0; i < p_fonts.size(); i++) {
		if (!font_is_language_supported(p_fonts[i], p_language)) {
			span.fonts.push_back(p_fonts[i]);
		}
	}
	ERR_FAIL_COND_V(span.fonts.is_empty(), false);
	span.font_size = p_size;
	span.language = p_language;

	sd->spans.push_back(span);
	sd->text += p_text;
	sd->end += p_text.length();
	invalidate(sd);

	return true;
}

bool TextServerFallback::shaped_text_add_object(RID p_shaped, Variant p_key, const Size2 &p_size, VAlign p_inline_align, int p_length) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	ERR_FAIL_COND_V(p_key == Variant(), false);
	ERR_FAIL_COND_V(sd->objects.has(p_key), false);

	if (sd->parent != RID()) {
		full_copy(sd);
	}

	ShapedTextData::Span span;
	span.start = sd->text.length();
	span.end = span.start + p_length;
	span.embedded_key = p_key;

	ShapedTextData::EmbeddedObject obj;
	obj.inline_align = p_inline_align;
	obj.rect.size = p_size;
	obj.pos = span.start;

	sd->spans.push_back(span);
	sd->text += String::chr(0xfffc).repeat(p_length);
	sd->end += p_length;
	sd->objects[p_key] = obj;
	invalidate(sd);

	return true;
}

bool TextServerFallback::shaped_text_resize_object(RID p_shaped, Variant p_key, const Size2 &p_size, VAlign p_inline_align) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	ERR_FAIL_COND_V(!sd->objects.has(p_key), false);
	sd->objects[p_key].rect.size = p_size;
	sd->objects[p_key].inline_align = p_inline_align;
	if (sd->valid) {
		// Recalc string metrics.
		sd->ascent = 0;
		sd->descent = 0;
		sd->width = 0;
		sd->upos = 0;
		sd->uthk = 0;
		int sd_size = sd->glyphs.size();
		const FontDataFallback *fd = nullptr;
		RID prev_rid = RID();

		for (int i = 0; i < sd_size; i++) {
			Glyph gl = sd->glyphs[i];
			Variant key;
			if (gl.count == 1) {
				for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = sd->objects.front(); E; E = E->next()) {
					if (E->get().pos == gl.start) {
						key = E->key();
						break;
					}
				}
			}
			if (key != Variant()) {
				if (sd->orientation == ORIENTATION_HORIZONTAL) {
					sd->objects[key].rect.position.x = sd->width;
					sd->width += sd->objects[key].rect.size.x;
					switch (sd->objects[key].inline_align) {
						case VALIGN_TOP: {
							sd->ascent = MAX(sd->ascent, sd->objects[key].rect.size.y);
						} break;
						case VALIGN_CENTER: {
							sd->ascent = MAX(sd->ascent, Math::round(sd->objects[key].rect.size.y / 2));
							sd->descent = MAX(sd->descent, Math::round(sd->objects[key].rect.size.y / 2));
						} break;
						case VALIGN_BOTTOM: {
							sd->descent = MAX(sd->descent, sd->objects[key].rect.size.y);
						} break;
					}
					sd->glyphs.write[i].advance = sd->objects[key].rect.size.x;
				} else {
					sd->objects[key].rect.position.y = sd->width;
					sd->width += sd->objects[key].rect.size.y;
					switch (sd->objects[key].inline_align) {
						case VALIGN_TOP: {
							sd->ascent = MAX(sd->ascent, sd->objects[key].rect.size.x);
						} break;
						case VALIGN_CENTER: {
							sd->ascent = MAX(sd->ascent, Math::round(sd->objects[key].rect.size.x / 2));
							sd->descent = MAX(sd->descent, Math::round(sd->objects[key].rect.size.x / 2));
						} break;
						case VALIGN_BOTTOM: {
							sd->descent = MAX(sd->descent, sd->objects[key].rect.size.x);
						} break;
					}
					sd->glyphs.write[i].advance = sd->objects[key].rect.size.y;
				}
			} else {
				if (prev_rid != gl.font_rid) {
					fd = font_owner.getornull(gl.font_rid);
					prev_rid = gl.font_rid;
				}
				if (fd != nullptr) {
					if (sd->orientation == ORIENTATION_HORIZONTAL) {
						sd->ascent = MAX(sd->ascent, fd->get_ascent(gl.font_size));
						sd->descent = MAX(sd->descent, fd->get_descent(gl.font_size));
					} else {
						sd->ascent = MAX(sd->ascent, Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5));
						sd->descent = MAX(sd->descent, Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5));
					}
					sd->upos = MAX(sd->upos, font_get_underline_position(gl.font_rid, gl.font_size));
					sd->uthk = MAX(sd->uthk, font_get_underline_thickness(gl.font_rid, gl.font_size));
				} else if (sd->preserve_invalid || (sd->preserve_control && is_control(gl.index))) {
					// Glyph not found, replace with hex code box.
					if (sd->orientation == ORIENTATION_HORIZONTAL) {
						sd->ascent = MAX(sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).y * 0.75f));
						sd->descent = MAX(sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).y * 0.25f));
					} else {
						sd->ascent = MAX(sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
						sd->descent = MAX(sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
					}
				}
				sd->width += gl.advance * gl.repeat;
			}
		}

		// Align embedded objects to baseline.
		for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = sd->objects.front(); E; E = E->next()) {
			if ((E->get().pos >= sd->start) && (E->get().pos < sd->end)) {
				if (sd->orientation == ORIENTATION_HORIZONTAL) {
					switch (E->get().inline_align) {
						case VALIGN_TOP: {
							E->get().rect.position.y = -sd->ascent;
						} break;
						case VALIGN_CENTER: {
							E->get().rect.position.y = -(E->get().rect.size.y / 2);
						} break;
						case VALIGN_BOTTOM: {
							E->get().rect.position.y = sd->descent - E->get().rect.size.y;
						} break;
					}
				} else {
					switch (E->get().inline_align) {
						case VALIGN_TOP: {
							E->get().rect.position.x = -sd->ascent;
						} break;
						case VALIGN_CENTER: {
							E->get().rect.position.x = -(E->get().rect.size.x / 2);
						} break;
						case VALIGN_BOTTOM: {
							E->get().rect.position.x = sd->descent - E->get().rect.size.x;
						} break;
					}
				}
			}
		}
	}
	return true;
}

RID TextServerFallback::shaped_text_substr(RID p_shaped, int p_start, int p_length) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, RID());
	if (sd->parent != RID()) {
		return shaped_text_substr(sd->parent, p_start, p_length);
	}
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	ERR_FAIL_COND_V(p_start < 0 || p_length < 0, RID());
	ERR_FAIL_COND_V(sd->start > p_start || sd->end < p_start, RID());
	ERR_FAIL_COND_V(sd->end < p_start + p_length, RID());

	ShapedTextData *new_sd = memnew(ShapedTextData);
	new_sd->parent = p_shaped;
	new_sd->start = p_start;
	new_sd->end = p_start + p_length;

	new_sd->orientation = sd->orientation;
	new_sd->direction = sd->direction;
	new_sd->para_direction = sd->para_direction;
	new_sd->line_breaks_valid = sd->line_breaks_valid;
	new_sd->justification_ops_valid = sd->justification_ops_valid;
	new_sd->sort_valid = false;
	new_sd->upos = sd->upos;
	new_sd->uthk = sd->uthk;

	if (p_length > 0) {
		new_sd->text = sd->text.substr(p_start, p_length);
		int sd_size = sd->glyphs.size();
		const Glyph *sd_glyphs = sd->glyphs.ptr();

		for (int i = 0; i < sd_size; i++) {
			if ((sd_glyphs[i].start >= new_sd->start) && (sd_glyphs[i].end <= new_sd->end)) {
				Glyph gl = sd_glyphs[i];
				Variant key;
				bool find_embedded = false;
				if (gl.count == 1) {
					for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = sd->objects.front(); E; E = E->next()) {
						if (E->get().pos == gl.start) {
							find_embedded = true;
							key = E->key();
							new_sd->objects[key] = E->get();
							break;
						}
					}
				}
				if (find_embedded) {
					if (new_sd->orientation == ORIENTATION_HORIZONTAL) {
						new_sd->objects[key].rect.position.x = new_sd->width;
						new_sd->width += new_sd->objects[key].rect.size.x;
						switch (new_sd->objects[key].inline_align) {
							case VALIGN_TOP: {
								new_sd->ascent = MAX(new_sd->ascent, new_sd->objects[key].rect.size.y);
							} break;
							case VALIGN_CENTER: {
								new_sd->ascent = MAX(new_sd->ascent, Math::round(new_sd->objects[key].rect.size.y / 2));
								new_sd->descent = MAX(new_sd->descent, Math::round(new_sd->objects[key].rect.size.y / 2));
							} break;
							case VALIGN_BOTTOM: {
								new_sd->descent = MAX(new_sd->descent, new_sd->objects[key].rect.size.y);
							} break;
						}
					} else {
						new_sd->objects[key].rect.position.y = new_sd->width;
						new_sd->width += new_sd->objects[key].rect.size.y;
						switch (new_sd->objects[key].inline_align) {
							case VALIGN_TOP: {
								new_sd->ascent = MAX(new_sd->ascent, new_sd->objects[key].rect.size.x);
							} break;
							case VALIGN_CENTER: {
								new_sd->ascent = MAX(new_sd->ascent, Math::round(new_sd->objects[key].rect.size.x / 2));
								new_sd->descent = MAX(new_sd->descent, Math::round(new_sd->objects[key].rect.size.x / 2));
							} break;
							case VALIGN_BOTTOM: {
								new_sd->descent = MAX(new_sd->descent, new_sd->objects[key].rect.size.x);
							} break;
						}
					}
				} else {
					const FontDataFallback *fd = font_owner.getornull(gl.font_rid);
					if (fd != nullptr) {
						if (new_sd->orientation == ORIENTATION_HORIZONTAL) {
							new_sd->ascent = MAX(new_sd->ascent, fd->get_ascent(gl.font_size));
							new_sd->descent = MAX(new_sd->descent, fd->get_descent(gl.font_size));
						} else {
							new_sd->ascent = MAX(new_sd->ascent, Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5));
							new_sd->descent = MAX(new_sd->descent, Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5));
						}
					} else if (new_sd->preserve_invalid || (new_sd->preserve_control && is_control(gl.index))) {
						// Glyph not found, replace with hex code box.
						if (new_sd->orientation == ORIENTATION_HORIZONTAL) {
							new_sd->ascent = MAX(new_sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).y * 0.75f));
							new_sd->descent = MAX(new_sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).y * 0.25f));
						} else {
							new_sd->ascent = MAX(new_sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
							new_sd->descent = MAX(new_sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
						}
					}
					new_sd->width += gl.advance * gl.repeat;
				}
				new_sd->glyphs.push_back(gl);
			}
		}

		for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = new_sd->objects.front(); E; E = E->next()) {
			if ((E->get().pos >= new_sd->start) && (E->get().pos < new_sd->end)) {
				if (sd->orientation == ORIENTATION_HORIZONTAL) {
					switch (E->get().inline_align) {
						case VALIGN_TOP: {
							E->get().rect.position.y = -new_sd->ascent;
						} break;
						case VALIGN_CENTER: {
							E->get().rect.position.y = -(E->get().rect.size.y / 2);
						} break;
						case VALIGN_BOTTOM: {
							E->get().rect.position.y = new_sd->descent - E->get().rect.size.y;
						} break;
					}
				} else {
					switch (E->get().inline_align) {
						case VALIGN_TOP: {
							E->get().rect.position.x = -new_sd->ascent;
						} break;
						case VALIGN_CENTER: {
							E->get().rect.position.x = -(E->get().rect.size.x / 2);
						} break;
						case VALIGN_BOTTOM: {
							E->get().rect.position.x = new_sd->descent - E->get().rect.size.x;
						} break;
					}
				}
			}
		}
	}
	new_sd->valid = true;

	return shaped_owner.make_rid(new_sd);
}

RID TextServerFallback::shaped_text_get_parent(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, RID());
	return sd->parent;
}

float TextServerFallback::shaped_text_fit_to_width(RID p_shaped, float p_width, uint8_t /*JustificationFlag*/ p_jst_flags) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	if (!sd->justification_ops_valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_update_justification_ops(p_shaped);
	}

	int start_pos = 0;
	int end_pos = sd->glyphs.size() - 1;

	if ((p_jst_flags & JUSTIFICATION_AFTER_LAST_TAB) == JUSTIFICATION_AFTER_LAST_TAB) {
		int start, end, delta;
		if (sd->para_direction == DIRECTION_LTR) {
			start = sd->glyphs.size() - 1;
			end = -1;
			delta = -1;
		} else {
			start = 0;
			end = sd->glyphs.size();
			delta = +1;
		}

		for (int i = start; i != end; i += delta) {
			if ((sd->glyphs[i].flags & GRAPHEME_IS_TAB) == GRAPHEME_IS_TAB) {
				if (sd->para_direction == DIRECTION_LTR) {
					start_pos = i;
					break;
				} else {
					end_pos = i;
					break;
				}
			}
		}
	}

	if ((p_jst_flags & JUSTIFICATION_TRIM_EDGE_SPACES) == JUSTIFICATION_TRIM_EDGE_SPACES) {
		while ((start_pos < end_pos) && ((sd->glyphs[start_pos].flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE || (sd->glyphs[start_pos].flags & GRAPHEME_IS_BREAK_HARD) == GRAPHEME_IS_BREAK_HARD || (sd->glyphs[start_pos].flags & GRAPHEME_IS_BREAK_SOFT) == GRAPHEME_IS_BREAK_SOFT)) {
			sd->width -= sd->glyphs[start_pos].advance * sd->glyphs[start_pos].repeat;
			sd->glyphs.write[start_pos].advance = 0;
			start_pos += sd->glyphs[start_pos].count;
		}
		while ((start_pos < end_pos) && ((sd->glyphs[end_pos].flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE || (sd->glyphs[end_pos].flags & GRAPHEME_IS_BREAK_HARD) == GRAPHEME_IS_BREAK_HARD || (sd->glyphs[end_pos].flags & GRAPHEME_IS_BREAK_SOFT) == GRAPHEME_IS_BREAK_SOFT)) {
			sd->width -= sd->glyphs[end_pos].advance * sd->glyphs[end_pos].repeat;
			sd->glyphs.write[end_pos].advance = 0;
			end_pos -= sd->glyphs[end_pos].count;
		}
	}

	int space_count = 0;
	for (int i = start_pos; i <= end_pos; i++) {
		const Glyph &gl = sd->glyphs[i];
		if (gl.count > 0) {
			if ((gl.flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE) {
				space_count++;
			}
		}
	}

	if ((space_count > 0) && ((p_jst_flags & JUSTIFICATION_WORD_BOUND) == JUSTIFICATION_WORD_BOUND)) {
		float delta_width_per_space = (p_width - sd->width) / space_count;
		for (int i = start_pos; i <= end_pos; i++) {
			Glyph &gl = sd->glyphs.write[i];
			if (gl.count > 0) {
				if ((gl.flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE) {
					float old_adv = gl.advance;
					gl.advance = Math::round(MAX(gl.advance + delta_width_per_space, 0.05 * gl.font_size));
					sd->width += (gl.advance - old_adv);
				}
			}
		}
	}

	return sd->width;
}

float TextServerFallback::shaped_text_tab_align(RID p_shaped, const Vector<float> &p_tab_stops) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	if (!sd->line_breaks_valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_update_breaks(p_shaped);
	}

	int tab_index = 0;
	float off = 0.f;

	int start, end, delta;
	if (sd->para_direction == DIRECTION_LTR) {
		start = 0;
		end = sd->glyphs.size();
		delta = +1;
	} else {
		start = sd->glyphs.size() - 1;
		end = -1;
		delta = -1;
	}

	Glyph *gl = sd->glyphs.ptrw();

	for (int i = start; i != end; i += delta) {
		if ((gl[i].flags & GRAPHEME_IS_TAB) == GRAPHEME_IS_TAB) {
			float tab_off = 0.f;
			while (tab_off <= off) {
				tab_off += p_tab_stops[tab_index];
				tab_index++;
				if (tab_index >= p_tab_stops.size()) {
					tab_index = 0;
				}
			}
			float old_adv = gl[i].advance;
			gl[i].advance = tab_off - off;
			sd->width += gl[i].advance - old_adv;
			off = 0;
			continue;
		}
		off += gl[i].advance * gl[i].repeat;
	}

	return 0.f;
}

bool TextServerFallback::shaped_text_update_breaks(RID p_shaped) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	if (!sd->valid) {
		shaped_text_shape(p_shaped);
	}

	if (sd->line_breaks_valid) {
		return true; // Nothing to do.
	}

	int sd_size = sd->glyphs.size();
	for (int i = 0; i < sd_size; i++) {
		if (sd->glyphs[i].count > 0) {
			char32_t c = sd->text[sd->glyphs[i].start];
			if (is_punct(c)) {
				sd->glyphs.write[i].flags |= GRAPHEME_IS_PUNCTUATION;
			}
			if (is_whitespace(c) && !is_linebreak(c)) {
				sd->glyphs.write[i].flags |= GRAPHEME_IS_SPACE;
				sd->glyphs.write[i].flags |= GRAPHEME_IS_BREAK_SOFT;
			}
			if (is_linebreak(c)) {
				sd->glyphs.write[i].flags |= GRAPHEME_IS_BREAK_HARD;
			}
			if (c == 0x0009 || c == 0x000b) {
				sd->glyphs.write[i].flags |= GRAPHEME_IS_TAB;
			}

			i += (sd->glyphs[i].count - 1);
		}
	}
	sd->line_breaks_valid = true;
	return sd->line_breaks_valid;
}

bool TextServerFallback::shaped_text_update_justification_ops(RID p_shaped) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	if (!sd->valid) {
		shaped_text_shape(p_shaped);
	}
	if (!sd->line_breaks_valid) {
		shaped_text_update_breaks(p_shaped);
	}

	sd->justification_ops_valid = true; // Not supported by fallback server.
	return true;
}

bool TextServerFallback::shaped_text_shape(RID p_shaped) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	if (sd->valid) {
		return true;
	}

	if (sd->parent != RID()) {
		full_copy(sd);
	}

	// Cleanup.
	sd->justification_ops_valid = false;
	sd->line_breaks_valid = false;
	sd->ascent = 0.f;
	sd->descent = 0.f;
	sd->width = 0.f;
	sd->glyphs.clear();

	if (sd->text.length() == 0) {
		sd->valid = true;
		return true;
	}

	// "Shape" string.
	for (int i = 0; i < sd->spans.size(); i++) {
		const ShapedTextData::Span &span = sd->spans[i];
		if (span.embedded_key != Variant()) {
			// Embedded object.
			if (sd->orientation == ORIENTATION_HORIZONTAL) {
				sd->objects[span.embedded_key].rect.position.x = sd->width;
				sd->width += sd->objects[span.embedded_key].rect.size.x;
				switch (sd->objects[span.embedded_key].inline_align) {
					case VALIGN_TOP: {
						sd->ascent = MAX(sd->ascent, sd->objects[span.embedded_key].rect.size.y);
					} break;
					case VALIGN_CENTER: {
						sd->ascent = MAX(sd->ascent, Math::round(sd->objects[span.embedded_key].rect.size.y / 2));
						sd->descent = MAX(sd->descent, Math::round(sd->objects[span.embedded_key].rect.size.y / 2));
					} break;
					case VALIGN_BOTTOM: {
						sd->descent = MAX(sd->descent, sd->objects[span.embedded_key].rect.size.y);
					} break;
				}
			} else {
				sd->objects[span.embedded_key].rect.position.y = sd->width;
				sd->width += sd->objects[span.embedded_key].rect.size.y;
				switch (sd->objects[span.embedded_key].inline_align) {
					case VALIGN_TOP: {
						sd->ascent = MAX(sd->ascent, sd->objects[span.embedded_key].rect.size.x);
					} break;
					case VALIGN_CENTER: {
						sd->ascent = MAX(sd->ascent, Math::round(sd->objects[span.embedded_key].rect.size.x / 2));
						sd->descent = MAX(sd->descent, Math::round(sd->objects[span.embedded_key].rect.size.x / 2));
					} break;
					case VALIGN_BOTTOM: {
						sd->descent = MAX(sd->descent, sd->objects[span.embedded_key].rect.size.x);
					} break;
				}
			}
			Glyph gl;
			gl.start = span.start;
			gl.end = span.end;
			gl.count = 1;
			gl.index = 0;
			gl.flags = GRAPHEME_IS_VALID | GRAPHEME_IS_VIRTUAL;
			if (sd->orientation == ORIENTATION_HORIZONTAL) {
				gl.advance = sd->objects[span.embedded_key].rect.size.x;
			} else {
				gl.advance = sd->objects[span.embedded_key].rect.size.y;
			}
			sd->glyphs.push_back(gl);
		} else {
			// Text span.
			for (int j = span.start; j < span.end; j++) {
				const FontDataFallback *fd = nullptr;

				Glyph gl;
				gl.start = j;
				gl.end = j + 1;
				gl.count = 1;
				gl.font_size = span.font_size;
				gl.index = (uint32_t)sd->text[j]; // Use codepoint.
				if (gl.index == 0x0009 || gl.index == 0x000b) {
					gl.index = 0x0020;
				}
				if (!sd->preserve_control && is_control(gl.index)) {
					gl.index = 0x0020;
				}
				// Select first font which has character (font are already sorted by span language).
				for (int k = 0; k < span.fonts.size(); k++) {
					fd = font_owner.getornull(span.fonts[k]);
					if (fd != nullptr && fd->has_char(gl.index)) {
						gl.font_rid = span.fonts[k];
						break;
					}
				}

				if (gl.font_rid != RID()) {
					if (sd->text[j] != 0 && !is_linebreak(sd->text[j])) {
						if (sd->orientation == ORIENTATION_HORIZONTAL) {
							gl.advance = fd->get_advance(gl.index, gl.font_size).x;
							gl.x_off = 0;
							gl.y_off = 0;
							sd->ascent = MAX(sd->ascent, fd->get_ascent(gl.font_size));
							sd->descent = MAX(sd->descent, fd->get_descent(gl.font_size));
						} else {
							gl.advance = fd->get_advance(gl.index, gl.font_size).y;
							gl.x_off = -Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5);
							gl.y_off = fd->get_ascent(gl.font_size);
							sd->ascent = MAX(sd->ascent, Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5));
							sd->descent = MAX(sd->descent, Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5));
						}
					}
					if (fd->get_spacing_space() && is_whitespace(sd->text[j])) {
						gl.advance += fd->get_spacing_space();
					} else {
						gl.advance += fd->get_spacing_glyph();
					}
					sd->upos = MAX(sd->upos, fd->get_underline_position(gl.font_size));
					sd->uthk = MAX(sd->uthk, fd->get_underline_thickness(gl.font_size));

					// Add kerning to previous glyph.
					if (sd->glyphs.size() > 0) {
						Glyph &prev_gl = sd->glyphs.write[sd->glyphs.size() - 1];
						if (prev_gl.font_rid == gl.font_rid && prev_gl.font_size == gl.font_size) {
							if (sd->orientation == ORIENTATION_HORIZONTAL) {
								prev_gl.advance += fd->get_kerning(prev_gl.index, gl.index, gl.font_size).x;
							} else {
								prev_gl.advance += fd->get_kerning(prev_gl.index, gl.index, gl.font_size).y;
							}
						}
					}
				} else if (sd->preserve_invalid || (sd->preserve_control && is_control(gl.index))) {
					// Glyph not found, replace with hex code box.
					if (sd->orientation == ORIENTATION_HORIZONTAL) {
						gl.advance = get_hex_code_box_size(gl.font_size, gl.index).x;
						sd->ascent = MAX(sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).y * 0.75f));
						sd->descent = MAX(sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).y * 0.25f));
					} else {
						gl.advance = get_hex_code_box_size(gl.font_size, gl.index).y;
						sd->ascent = MAX(sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
						sd->descent = MAX(sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
					}
				}
				sd->width += gl.advance;
				sd->glyphs.push_back(gl);
			}
		}
	}

	// Align embedded objects to baseline.
	for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = sd->objects.front(); E; E = E->next()) {
		if (sd->orientation == ORIENTATION_HORIZONTAL) {
			switch (E->get().inline_align) {
				case VALIGN_TOP: {
					E->get().rect.position.y = -sd->ascent;
				} break;
				case VALIGN_CENTER: {
					E->get().rect.position.y = -(E->get().rect.size.y / 2);
				} break;
				case VALIGN_BOTTOM: {
					E->get().rect.position.y = sd->descent - E->get().rect.size.y;
				} break;
			}
		} else {
			switch (E->get().inline_align) {
				case VALIGN_TOP: {
					E->get().rect.position.x = -sd->ascent;
				} break;
				case VALIGN_CENTER: {
					E->get().rect.position.x = -(E->get().rect.size.x / 2);
				} break;
				case VALIGN_BOTTOM: {
					E->get().rect.position.x = sd->descent - E->get().rect.size.x;
				} break;
			}
		}
	}

	sd->valid = true;
	return sd->valid;
}

bool TextServerFallback::shaped_text_is_ready(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	return sd->valid;
}

Vector<TextServer::Glyph> TextServerFallback::shaped_text_get_glyphs(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, Vector<TextServer::Glyph>());
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->glyphs;
}

Vector2i TextServerFallback::shaped_text_get_range(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, Vector2i());
	return Vector2(sd->start, sd->end);
}

Vector<TextServer::Glyph> TextServerFallback::shaped_text_sort_logical(RID p_shaped) {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, Vector<TextServer::Glyph>());
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}

	return sd->glyphs; // Already in the logical order, return as is.
}

Array TextServerFallback::shaped_text_get_objects(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	Array ret;
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, ret);
	for (const Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = sd->objects.front(); E; E = E->next()) {
		ret.push_back(E->key());
	}

	return ret;
}

Rect2 TextServerFallback::shaped_text_get_object_rect(RID p_shaped, Variant p_key) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, Rect2());
	ERR_FAIL_COND_V(!sd->objects.has(p_key), Rect2());
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->objects[p_key].rect;
}

Size2 TextServerFallback::shaped_text_get_size(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, Size2());
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	if (sd->orientation == TextServer::ORIENTATION_HORIZONTAL) {
		return Size2(sd->width, sd->ascent + sd->descent);
	} else {
		return Size2(sd->ascent + sd->descent, sd->width);
	}
}

float TextServerFallback::shaped_text_get_ascent(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->ascent;
}

float TextServerFallback::shaped_text_get_descent(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->descent;
}

float TextServerFallback::shaped_text_get_width(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->width;
}

float TextServerFallback::shaped_text_get_underline_position(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}

	return sd->upos;
}

float TextServerFallback::shaped_text_get_underline_thickness(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}

	return sd->uthk;
}

TextServer *TextServerFallback::create_func(Error &r_error, void *p_user_data) {
	r_error = OK;
	return memnew(TextServerFallback());
}

void TextServerFallback::register_server() {
	TextServerManager::register_create_function(interface_name, interface_features, create_func, nullptr);
}
