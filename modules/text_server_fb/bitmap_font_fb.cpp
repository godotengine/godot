/*************************************************************************/
/*  bitmap_font_fb.cpp                                                   */
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

#include "bitmap_font_fb.h"

Error BitmapFontDataFallback::load_from_file(const String &p_filename, int p_base_size) {
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
				c.advance.y = keys["yadvance"].to_int();
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

	valid = true;

	memdelete(f);
	return OK;
}

Error BitmapFontDataFallback::bitmap_new(float p_height, float p_ascent, int p_base_size) {
	height = p_height;
	ascent = p_ascent;

	base_size = p_base_size;
	if (base_size == 0) {
		base_size = height;
	}

	char_map.clear();
	textures.clear();
	kerning_map.clear();

	valid = true;

	return OK;
}

void BitmapFontDataFallback::bitmap_add_texture(const Ref<Texture> &p_texture) {
	ERR_FAIL_COND(!valid);
	ERR_FAIL_COND_MSG(p_texture.is_null(), "It's not a reference to a valid Texture object.");

	textures.push_back(p_texture);
}

void BitmapFontDataFallback::bitmap_add_char(char32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance) {
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

void BitmapFontDataFallback::bitmap_add_kerning_pair(char32_t p_A, char32_t p_B, int p_kerning) {
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

float BitmapFontDataFallback::get_height(int p_size) const {
	ERR_FAIL_COND_V(!valid, 0.f);
	return height * (float(p_size) / float(base_size));
}

float BitmapFontDataFallback::get_ascent(int p_size) const {
	ERR_FAIL_COND_V(!valid, 0.f);
	return ascent * (float(p_size) / float(base_size));
}

float BitmapFontDataFallback::get_descent(int p_size) const {
	ERR_FAIL_COND_V(!valid, 0.f);
	return (height - ascent) * (float(p_size) / float(base_size));
}

float BitmapFontDataFallback::get_underline_position(int p_size) const {
	ERR_FAIL_COND_V(!valid, 0.f);
	return 2 * (float(p_size) / float(base_size));
}

float BitmapFontDataFallback::get_underline_thickness(int p_size) const {
	ERR_FAIL_COND_V(!valid, 0.f);
	return 1 * (float(p_size) / float(base_size));
}

void BitmapFontDataFallback::set_distance_field_hint(bool p_distance_field) {
	distance_field_hint = p_distance_field;
}

bool BitmapFontDataFallback::get_distance_field_hint() const {
	return distance_field_hint;
}

float BitmapFontDataFallback::get_base_size() const {
	return base_size;
}

bool BitmapFontDataFallback::has_char(char32_t p_char) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, false);
	return char_map.has(p_char);
}

String BitmapFontDataFallback::get_supported_chars() const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, String());
	String chars;
	const char32_t *k = nullptr;
	while ((k = char_map.next(k))) {
		chars += char32_t(*k);
	}
	return chars;
}

Vector2 BitmapFontDataFallback::get_advance(char32_t p_char, int p_size) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, Vector2());
	const Character *c = char_map.getptr(p_char);
	ERR_FAIL_COND_V(c == nullptr, Vector2());

	return c->advance * (float(p_size) / float(base_size));
}

Vector2 BitmapFontDataFallback::get_kerning(char32_t p_char, char32_t p_next, int p_size) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!valid, Vector2());
	KerningPairKey kpk;
	kpk.A = p_char;
	kpk.B = p_next;

	const Map<KerningPairKey, int>::Element *E = kerning_map.find(kpk);
	if (E) {
		return Vector2(-E->get() * (float(p_size) / float(base_size)), 0);
	} else {
		return Vector2();
	}
}

Vector2 BitmapFontDataFallback::draw_glyph(RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
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

Vector2 BitmapFontDataFallback::draw_glyph_outline(RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
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
