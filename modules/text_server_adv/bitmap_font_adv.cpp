/*************************************************************************/
/*  bitmap_font_adv.cpp                                                  */
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

#include "bitmap_font_adv.h"

/*************************************************************************/
/*  hb_bmp_font_t HarfBuzz Bitmap font interface                         */
/*************************************************************************/

struct hb_bmp_font_t {
	BitmapFontDataAdvanced *face = nullptr;
	float font_size = 0.0;
	bool unref = false; /* Whether to destroy bm_face when done. */
};

static hb_bmp_font_t *_hb_bmp_font_create(BitmapFontDataAdvanced *p_face, float p_font_size, bool p_unref) {
	hb_bmp_font_t *bm_font = reinterpret_cast<hb_bmp_font_t *>(calloc(1, sizeof(hb_bmp_font_t)));

	if (!bm_font) {
		return nullptr;
	}

	bm_font->face = p_face;
	bm_font->font_size = p_font_size;
	bm_font->unref = p_unref;

	return bm_font;
}

static void _hb_bmp_font_destroy(void *data) {
	hb_bmp_font_t *bm_font = reinterpret_cast<hb_bmp_font_t *>(data);
	free(bm_font);
}

static hb_bool_t hb_bmp_get_nominal_glyph(hb_font_t *font, void *font_data, hb_codepoint_t unicode, hb_codepoint_t *glyph, void *user_data) {
	const hb_bmp_font_t *bm_font = reinterpret_cast<const hb_bmp_font_t *>(font_data);

	if (!bm_font->face) {
		return false;
	}

	if (!bm_font->face->has_char(unicode)) {
		if (bm_font->face->has_char(0xF000u + unicode)) {
			*glyph = 0xF000u + unicode;
			return true;
		} else {
			return false;
		}
	}

	*glyph = unicode;
	return true;
}

static hb_position_t hb_bmp_get_glyph_h_advance(hb_font_t *font, void *font_data, hb_codepoint_t glyph, void *user_data) {
	const hb_bmp_font_t *bm_font = reinterpret_cast<const hb_bmp_font_t *>(font_data);

	if (!bm_font->face) {
		return 0;
	}

	if (!bm_font->face->has_char(glyph)) {
		return 0;
	}

	return bm_font->face->get_advance(glyph, bm_font->font_size).x * 64;
}

static hb_position_t hb_bmp_get_glyph_v_advance(hb_font_t *font, void *font_data, hb_codepoint_t glyph, void *user_data) {
	const hb_bmp_font_t *bm_font = reinterpret_cast<const hb_bmp_font_t *>(font_data);

	if (!bm_font->face) {
		return 0;
	}

	if (!bm_font->face->has_char(glyph)) {
		return 0;
	}

	return -bm_font->face->get_advance(glyph, bm_font->font_size).y * 64;
}

static hb_position_t hb_bmp_get_glyph_h_kerning(hb_font_t *font, void *font_data, hb_codepoint_t left_glyph, hb_codepoint_t right_glyph, void *user_data) {
	const hb_bmp_font_t *bm_font = reinterpret_cast<const hb_bmp_font_t *>(font_data);

	if (!bm_font->face) {
		return 0;
	}

	if (!bm_font->face->has_char(left_glyph)) {
		return 0;
	}

	if (!bm_font->face->has_char(right_glyph)) {
		return 0;
	}

	return bm_font->face->get_kerning(left_glyph, right_glyph, bm_font->font_size).x * 64;
}

static hb_bool_t hb_bmp_get_glyph_v_origin(hb_font_t *font, void *font_data, hb_codepoint_t glyph, hb_position_t *x, hb_position_t *y, void *user_data) {
	const hb_bmp_font_t *bm_font = reinterpret_cast<const hb_bmp_font_t *>(font_data);

	if (!bm_font->face) {
		return false;
	}

	if (!bm_font->face->has_char(glyph)) {
		return false;
	}

	*x = bm_font->face->get_advance(glyph, bm_font->font_size).x * 32;
	*y = bm_font->face->get_ascent(bm_font->font_size) * 64;

	return true;
}

static hb_bool_t hb_bmp_get_glyph_extents(hb_font_t *font, void *font_data, hb_codepoint_t glyph, hb_glyph_extents_t *extents, void *user_data) {
	const hb_bmp_font_t *bm_font = reinterpret_cast<const hb_bmp_font_t *>(font_data);

	if (!bm_font->face) {
		return false;
	}

	if (!bm_font->face->has_char(glyph)) {
		return false;
	}

	extents->x_bearing = 0;
	extents->y_bearing = 0;
	extents->width = bm_font->face->get_size(glyph, bm_font->font_size).x * 64;
	extents->height = bm_font->face->get_size(glyph, bm_font->font_size).y * 64;

	return true;
}

static hb_bool_t hb_bmp_get_font_h_extents(hb_font_t *font, void *font_data, hb_font_extents_t *metrics, void *user_data) {
	const hb_bmp_font_t *bm_font = reinterpret_cast<const hb_bmp_font_t *>(font_data);

	if (!bm_font->face) {
		return false;
	}

	metrics->ascender = bm_font->face->get_ascent(bm_font->font_size);
	metrics->descender = bm_font->face->get_descent(bm_font->font_size);
	metrics->line_gap = 0;

	return true;
}

static hb_font_funcs_t *funcs = nullptr;
void hb_bmp_create_font_funcs() {
	funcs = hb_font_funcs_create();

	hb_font_funcs_set_font_h_extents_func(funcs, hb_bmp_get_font_h_extents, nullptr, nullptr);
	//hb_font_funcs_set_font_v_extents_func (funcs, hb_bmp_get_font_v_extents, nullptr, nullptr);
	hb_font_funcs_set_nominal_glyph_func(funcs, hb_bmp_get_nominal_glyph, nullptr, nullptr);
	//hb_font_funcs_set_variation_glyph_func (funcs, hb_bmp_get_variation_glyph, nullptr, nullptr);
	hb_font_funcs_set_glyph_h_advance_func(funcs, hb_bmp_get_glyph_h_advance, nullptr, nullptr);
	hb_font_funcs_set_glyph_v_advance_func(funcs, hb_bmp_get_glyph_v_advance, nullptr, nullptr);
	//hb_font_funcs_set_glyph_h_origin_func(funcs, hb_bmp_get_glyph_h_origin, nullptr, nullptr);
	hb_font_funcs_set_glyph_v_origin_func(funcs, hb_bmp_get_glyph_v_origin, nullptr, nullptr);
	hb_font_funcs_set_glyph_h_kerning_func(funcs, hb_bmp_get_glyph_h_kerning, nullptr, nullptr);
	//hb_font_funcs_set_glyph_v_kerning_func (funcs, hb_bmp_get_glyph_v_kerning, nullptr, nullptr);
	hb_font_funcs_set_glyph_extents_func(funcs, hb_bmp_get_glyph_extents, nullptr, nullptr);
	//hb_font_funcs_set_glyph_contour_point_func (funcs, hb_bmp_get_glyph_contour_point, nullptr, nullptr);
	//hb_font_funcs_set_glyph_name_func (funcs, hb_bmp_get_glyph_name, nullptr, nullptr);
	//hb_font_funcs_set_glyph_from_name_func (funcs, hb_bmp_get_glyph_from_name, nullptr, nullptr);

	hb_font_funcs_make_immutable(funcs);
}

void hb_bmp_free_font_funcs() {
	if (funcs != nullptr) {
		hb_font_funcs_destroy(funcs);
		funcs = nullptr;
	}
}

static void _hb_bmp_font_set_funcs(hb_font_t *p_font, BitmapFontDataAdvanced *p_face, int p_size, bool p_unref) {
	hb_font_set_funcs(p_font, funcs, _hb_bmp_font_create(p_face, p_size, p_unref), _hb_bmp_font_destroy);
}

hb_font_t *hb_bmp_font_create(BitmapFontDataAdvanced *p_face, int p_size, hb_destroy_func_t p_destroy) {
	hb_font_t *font;
	hb_face_t *face = hb_face_create(nullptr, 0);

	font = hb_font_create(face);
	hb_face_destroy(face);
	_hb_bmp_font_set_funcs(font, p_face, p_size, false);
	return font;
}

/*************************************************************************/
/*  BitmapFontDataAdvanced                                               */
/*************************************************************************/

Error BitmapFontDataAdvanced::load_from_file(const String &p_filename, int p_base_size) {
	_THREAD_SAFE_METHOD_
	//fnt format used by angelcode bmfont
	//http://www.angelcode.com/products/bmfont/

	FileAccess *f = FileAccess::open(p_filename, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(!f, ERR_FILE_NOT_FOUND, "Can't open font: " + p_filename + ".");

	while (true) {
		String line = f->get_line();

		int delimiter = line.find(" ");
		String type = line.substr(0, delimiter);
		int pos = delimiter + 1;
		Map<String, String> keys;

		while (pos < line.size() && line[pos] == ' ') {
			pos++;
		}

		while (pos < line.size()) {
			int eq = line.find("=", pos);
			if (eq == -1) {
				break;
			}
			String key = line.substr(pos, eq - pos);
			int end = -1;
			String value;
			if (line[eq + 1] == '"') {
				end = line.find("\"", eq + 2);
				if (end == -1) {
					break;
				}
				value = line.substr(eq + 2, end - 1 - eq - 1);
				pos = end + 1;
			} else {
				end = line.find(" ", eq + 1);
				if (end == -1) {
					end = line.size();
				}
				value = line.substr(eq + 1, end - eq);
				pos = end;
			}

			while (pos < line.size() && line[pos] == ' ') {
				pos++;
			}

			keys[key] = value;
		}

		if (type == "info") {
			if (keys.has("size")) {
				base_size = keys["size"].to_int();
			}
		} else if (type == "common") {
			if (keys.has("lineHeight")) {
				height = keys["lineHeight"].to_int();
			}
			if (keys.has("base")) {
				ascent = keys["base"].to_int();
			}
		} else if (type == "page") {
			if (keys.has("file")) {
				String base_dir = p_filename.get_base_dir();
				String file = base_dir.plus_file(keys["file"]);
				if (RenderingServer::get_singleton() != nullptr) {
					Ref<Texture2D> tex = ResourceLoader::load(file);
					if (tex.is_null()) {
						ERR_PRINT("Can't load font texture!");
					} else {
						ERR_FAIL_COND_V_MSG(tex.is_null(), ERR_FILE_CANT_READ, "It's not a reference to a valid Texture object.");
						textures.push_back(tex);
					}
				}
			}
		} else if (type == "char") {
			Character c;
			char32_t idx = 0;
			if (keys.has("id")) {
				idx = keys["id"].to_int();
			}
			if (keys.has("x")) {
				c.rect.position.x = keys["x"].to_int();
			}
			if (keys.has("y")) {
				c.rect.position.y = keys["y"].to_int();
			}
			if (keys.has("width")) {
				c.rect.size.width = keys["width"].to_int();
			}
			if (keys.has("height")) {
				c.rect.size.height = keys["height"].to_int();
			}
			if (keys.has("xoffset")) {
				c.align.x = keys["xoffset"].to_int();
			}
			if (keys.has("yoffset")) {
				c.align.y = keys["yoffset"].to_int();
			}
			if (keys.has("page")) {
				c.texture_idx = keys["page"].to_int();
			}
			if (keys.has("xadvance")) {
				c.advance.x = keys["xadvance"].to_int();
			}
			if (keys.has("yadvance")) {
				c.advance.x = keys["yadvance"].to_int();
			}
			if (c.advance.x < 0) {
				c.advance.x = c.rect.size.width + 1;
			}
			if (c.advance.y < 0) {
				c.advance.y = c.rect.size.height + 1;
			}
			char_map[idx] = c;
		} else if (type == "kerning") {
			KerningPairKey kpk;
			float k = 0.0;
			if (keys.has("first")) {
				kpk.A = keys["first"].to_int();
			}
			if (keys.has("second")) {
				kpk.B = keys["second"].to_int();
			}
			if (keys.has("amount")) {
				k = keys["amount"].to_int();
			}
			kerning_map[kpk] = k;
		}

		if (f->eof_reached()) {
			break;
		}
	}
	if (base_size == 0) {
		base_size = height;
	}

	if (hb_handle) {
		hb_font_destroy(hb_handle);
	}
	hb_handle = hb_bmp_font_create(this, base_size, nullptr);
	valid = true;

	memdelete(f);
	return OK;
}

Error BitmapFontDataAdvanced::bitmap_new(float p_height, float p_ascent, int p_base_size) {
	height = p_height;
	ascent = p_ascent;

	base_size = p_base_size;
	if (base_size == 0) {
		base_size = height;
	}

	char_map.clear();
	textures.clear();
	kerning_map.clear();
	if (hb_handle) {
		hb_font_destroy(hb_handle);
	}
	hb_handle = hb_bmp_font_create(this, base_size, nullptr);
	valid = true;

	return OK;
}

void BitmapFontDataAdvanced::bitmap_add_texture(const Ref<Texture> &p_texture) {
	ERR_FAIL_COND(!valid);
	ERR_FAIL_COND_MSG(p_texture.is_null(), "It's not a reference to a valid Texture object.");

	textures.push_back(p_texture);
}

void BitmapFontDataAdvanced::bitmap_add_char(char32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance) {
	ERR_FAIL_COND(!valid);

	Character chr;
	chr.rect = p_rect;
	chr.texture_idx = p_texture_idx;
	if (p_advance < 0) {
		chr.advance.x = chr.rect.size.x;
	} else {
		chr.advance.x = p_advance;
	}
	chr.align = p_align;
	char_map[p_char] = chr;
}

void BitmapFontDataAdvanced::bitmap_add_kerning_pair(char32_t p_A, char32_t p_B, int p_kerning) {
	ERR_FAIL_COND(!valid);

	KerningPairKey kpk;
	kpk.A = p_A;
	kpk.B = p_B;

	if (p_kerning == 0 && kerning_map.has(kpk)) {
		kerning_map.erase(kpk);
	} else {
		kerning_map[kpk] = p_kerning;
	}
}

float BitmapFontDataAdvanced::get_height(int p_size) const {
	ERR_FAIL_COND_V(!valid, 0.f);
	return height * (float(p_size) / float(base_size));
}

float BitmapFontDataAdvanced::get_ascent(int p_size) const {
	ERR_FAIL_COND_V(!valid, 0.f);
	return ascent * (float(p_size) / float(base_size));
}

float BitmapFontDataAdvanced::get_descent(int p_size) const {
	ERR_FAIL_COND_V(!valid, 0.f);
	return (height - ascent) * (float(p_size) / float(base_size));
}

float BitmapFontDataAdvanced::get_underline_position(int p_size) const {
	ERR_FAIL_COND_V(!valid, 0.f);
	return 2 * (float(p_size) / float(base_size));
}

float BitmapFontDataAdvanced::get_underline_thickness(int p_size) const {
	ERR_FAIL_COND_V(!valid, 0.f);
	return 1 * (float(p_size) / float(base_size));
}

void BitmapFontDataAdvanced::set_distance_field_hint(bool p_distance_field) {
	distance_field_hint = p_distance_field;
}

bool BitmapFontDataAdvanced::get_distance_field_hint() const {
	return distance_field_hint;
}

float BitmapFontDataAdvanced::get_base_size() const {
	return base_size;
}

hb_font_t *BitmapFontDataAdvanced::get_hb_handle(int p_size) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, nullptr);
	return hb_handle;
}

bool BitmapFontDataAdvanced::has_char(char32_t p_char) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, false);
	return char_map.has(p_char);
}

String BitmapFontDataAdvanced::get_supported_chars() const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, String());
	String chars;
	const uint32_t *k = nullptr;
	while ((k = char_map.next(k))) {
		chars += char32_t(*k);
	}
	return chars;
}

Vector2 BitmapFontDataAdvanced::get_advance(uint32_t p_char, int p_size) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, Vector2());
	const Character *c = char_map.getptr(p_char);
	ERR_FAIL_COND_V(c == nullptr, Vector2());

	return c->advance * (float(p_size) / float(base_size));
}

Vector2 BitmapFontDataAdvanced::get_align(uint32_t p_char, int p_size) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, Vector2());
	const Character *c = char_map.getptr(p_char);
	ERR_FAIL_COND_V(c == nullptr, Vector2());

	return c->align * (float(p_size) / float(base_size));
}

Vector2 BitmapFontDataAdvanced::get_size(uint32_t p_char, int p_size) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, Vector2());
	const Character *c = char_map.getptr(p_char);
	ERR_FAIL_COND_V(c == nullptr, Vector2());

	return c->rect.size * (float(p_size) / float(base_size));
}

float BitmapFontDataAdvanced::get_font_scale(int p_size) const {
	return float(p_size) / float(base_size);
}

Vector2 BitmapFontDataAdvanced::get_kerning(uint32_t p_char, uint32_t p_next, int p_size) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, Vector2());
	KerningPairKey kpk;
	kpk.A = p_char;
	kpk.B = p_next;

	const Map<KerningPairKey, int>::Element *E = kerning_map.find(kpk);
	if (E) {
		return Vector2(-E->get() * (float(p_size) / float(base_size)), 0.f);
	} else {
		return Vector2();
	}
}

Vector2 BitmapFontDataAdvanced::draw_glyph(RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_
	if (p_index == 0) {
		return Vector2();
	}
	ERR_FAIL_COND_V(!valid, Vector2());
	const Character *c = char_map.getptr(p_index);

	ERR_FAIL_COND_V(c == nullptr, Vector2());
	ERR_FAIL_COND_V(c->texture_idx < -1 || c->texture_idx >= textures.size(), Vector2());
	if (c->texture_idx != -1) {
		Point2i cpos = p_pos;
		cpos += (c->align + Vector2(0, -ascent)) * (float(p_size) / float(base_size));
		Size2i csize = c->rect.size * (float(p_size) / float(base_size));
		if (RenderingServer::get_singleton() != nullptr) {
			//if (distance_field_hint) { // Not implemented.
			//	RenderingServer::get_singleton()->canvas_item_set_distance_field_mode(p_canvas, true);
			//}
			RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, Rect2(cpos, csize), textures[c->texture_idx]->get_rid(), c->rect, p_color, false, false);
			//if (distance_field_hint) {
			//	RenderingServer::get_singleton()->canvas_item_set_distance_field_mode(p_canvas, false);
			//}
		}
	}

	return c->advance * (float(p_size) / float(base_size));
}

Vector2 BitmapFontDataAdvanced::draw_glyph_outline(RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_
	if (p_index == 0) {
		return Vector2();
	}
	ERR_FAIL_COND_V(!valid, Vector2());
	const Character *c = char_map.getptr(p_index);

	ERR_FAIL_COND_V(c == nullptr, Vector2());
	ERR_FAIL_COND_V(c->texture_idx < -1 || c->texture_idx >= textures.size(), Vector2());

	// Not supported, return advance for compatibility.

	return c->advance * (float(p_size) / float(base_size));
}

BitmapFontDataAdvanced::~BitmapFontDataAdvanced() {
	if (hb_handle) {
		hb_font_destroy(hb_handle);
	}
}
