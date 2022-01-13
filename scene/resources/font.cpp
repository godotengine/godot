/*************************************************************************/
/*  font.cpp                                                             */
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

#include "font.h"

#include "core/io/resource_loader.h"
#include "core/method_bind_ext.gen.inc"
#include "core/os/file_access.h"

void Font::draw_halign(RID p_canvas_item, const Point2 &p_pos, HAlign p_align, float p_width, const String &p_text, const Color &p_modulate, const Color &p_outline_modulate) const {
	float length = get_string_size(p_text).width;
	if (length >= p_width) {
		draw(p_canvas_item, p_pos, p_text, p_modulate, p_width, p_outline_modulate);
		return;
	}

	float ofs = 0.f;
	switch (p_align) {
		case HALIGN_LEFT: {
			ofs = 0;
		} break;
		case HALIGN_CENTER: {
			ofs = Math::floor((p_width - length) / 2.0);
		} break;
		case HALIGN_RIGHT: {
			ofs = p_width - length;
		} break;
		default: {
			ERR_PRINT("Unknown halignment type");
		} break;
	}
	draw(p_canvas_item, p_pos + Point2(ofs, 0), p_text, p_modulate, p_width, p_outline_modulate);
}

void Font::draw(RID p_canvas_item, const Point2 &p_pos, const String &p_text, const Color &p_modulate, int p_clip_w, const Color &p_outline_modulate) const {
	Vector2 ofs;

	int chars_drawn = 0;
	bool with_outline = has_outline();
	for (int i = 0; i < p_text.length(); i++) {
		int width = get_char_size(p_text[i]).width;

		if (p_clip_w >= 0 && (ofs.x + width) > p_clip_w) {
			break; //clip
		}

		ofs.x += draw_char(p_canvas_item, p_pos + ofs, p_text[i], p_text[i + 1], with_outline ? p_outline_modulate : p_modulate, with_outline);
		++chars_drawn;
	}

	if (has_outline()) {
		ofs = Vector2(0, 0);
		for (int i = 0; i < chars_drawn; i++) {
			ofs.x += draw_char(p_canvas_item, p_pos + ofs, p_text[i], p_text[i + 1], p_modulate, false);
		}
	}
}

void Font::update_changes() {
	emit_changed();
}

void Font::_bind_methods() {
	ClassDB::bind_method(D_METHOD("draw", "canvas_item", "position", "string", "modulate", "clip_w", "outline_modulate"), &Font::draw, DEFVAL(Color(1, 1, 1)), DEFVAL(-1), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("get_ascent"), &Font::get_ascent);
	ClassDB::bind_method(D_METHOD("get_descent"), &Font::get_descent);
	ClassDB::bind_method(D_METHOD("get_height"), &Font::get_height);
	ClassDB::bind_method(D_METHOD("is_distance_field_hint"), &Font::is_distance_field_hint);
	ClassDB::bind_method(D_METHOD("get_char_size", "char", "next"), &Font::get_char_size, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_string_size", "string"), &Font::get_string_size);
	ClassDB::bind_method(D_METHOD("get_wordwrap_string_size", "string", "width"), &Font::get_wordwrap_string_size);
	ClassDB::bind_method(D_METHOD("has_outline"), &Font::has_outline);
	ClassDB::bind_method(D_METHOD("draw_char", "canvas_item", "position", "char", "next", "modulate", "outline"), &Font::draw_char, DEFVAL(-1), DEFVAL(Color(1, 1, 1)), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("update_changes"), &Font::update_changes);
}

Font::Font() {
}

/////////////////////////////////////////////////////////////////

void BitmapFont::_set_chars(const PoolVector<int> &p_chars) {
	int len = p_chars.size();
	//char 1 charsize 1 texture, 4 rect, 2 align, advance 1
	ERR_FAIL_COND(len % 9);
	if (!len) {
		return; //none to do
	}
	int chars = len / 9;

	PoolVector<int>::Read r = p_chars.read();
	for (int i = 0; i < chars; i++) {
		const int *data = &r[i * 9];
		add_char(data[0], data[1], Rect2(data[2], data[3], data[4], data[5]), Size2(data[6], data[7]), data[8]);
	}
}

PoolVector<int> BitmapFont::_get_chars() const {
	PoolVector<int> chars;

	const CharType *key = nullptr;

	while ((key = char_map.next(key))) {
		const Character *c = char_map.getptr(*key);
		ERR_FAIL_COND_V(!c, PoolVector<int>());
		chars.push_back(*key);
		chars.push_back(c->texture_idx);
		chars.push_back(c->rect.position.x);
		chars.push_back(c->rect.position.y);

		chars.push_back(c->rect.size.x);
		chars.push_back(c->rect.size.y);
		chars.push_back(c->h_align);
		chars.push_back(c->v_align);
		chars.push_back(c->advance);
	}

	return chars;
}

void BitmapFont::_set_kernings(const PoolVector<int> &p_kernings) {
	int len = p_kernings.size();
	ERR_FAIL_COND(len % 3);
	if (!len) {
		return;
	}
	PoolVector<int>::Read r = p_kernings.read();

	for (int i = 0; i < len / 3; i++) {
		const int *data = &r[i * 3];
		add_kerning_pair(data[0], data[1], data[2]);
	}
}

PoolVector<int> BitmapFont::_get_kernings() const {
	PoolVector<int> kernings;

	for (Map<KerningPairKey, int>::Element *E = kerning_map.front(); E; E = E->next()) {
		kernings.push_back(E->key().A);
		kernings.push_back(E->key().B);
		kernings.push_back(E->get());
	}

	return kernings;
}

void BitmapFont::_set_textures(const Vector<Variant> &p_textures) {
	textures.clear();
	for (int i = 0; i < p_textures.size(); i++) {
		Ref<Texture> tex = p_textures[i];
		ERR_CONTINUE(!tex.is_valid());
		add_texture(tex);
	}
}

Vector<Variant> BitmapFont::_get_textures() const {
	Vector<Variant> rtextures;
	for (int i = 0; i < textures.size(); i++) {
		rtextures.push_back(textures[i].get_ref_ptr());
	}
	return rtextures;
}

Error BitmapFont::create_from_fnt(const String &p_file) {
	//fnt format used by angelcode bmfont
	//http://www.angelcode.com/products/bmfont/

	FileAccess *f = FileAccess::open(p_file, FileAccess::READ);

	ERR_FAIL_COND_V_MSG(!f, ERR_FILE_NOT_FOUND, "Can't open font: " + p_file + ".");

	clear();

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
			if (keys.has("face")) {
				set_name(keys["face"]);
			}
			/*
			if (keys.has("size"))
				font->set_height(keys["size"].to_int());
			*/

		} else if (type == "common") {
			if (keys.has("lineHeight")) {
				set_height(keys["lineHeight"].to_int());
			}
			if (keys.has("base")) {
				set_ascent(keys["base"].to_int());
			}

		} else if (type == "page") {
			if (keys.has("file")) {
				String base_dir = p_file.get_base_dir();
				String file = base_dir.plus_file(keys["file"]);
				Ref<Texture> tex = ResourceLoader::load(file);
				if (tex.is_null()) {
					ERR_PRINT("Can't load font texture!");
				} else {
					add_texture(tex);
				}
			}
		} else if (type == "char") {
			CharType idx = 0;
			if (keys.has("id")) {
				idx = keys["id"].to_int();
			}

			Rect2 rect;

			if (keys.has("x")) {
				rect.position.x = keys["x"].to_int();
			}
			if (keys.has("y")) {
				rect.position.y = keys["y"].to_int();
			}
			if (keys.has("width")) {
				rect.size.width = keys["width"].to_int();
			}
			if (keys.has("height")) {
				rect.size.height = keys["height"].to_int();
			}

			Point2 ofs;

			if (keys.has("xoffset")) {
				ofs.x = keys["xoffset"].to_int();
			}
			if (keys.has("yoffset")) {
				ofs.y = keys["yoffset"].to_int();
			}

			int texture = 0;
			if (keys.has("page")) {
				texture = keys["page"].to_int();
			}
			int advance = -1;
			if (keys.has("xadvance")) {
				advance = keys["xadvance"].to_int();
			}

			add_char(idx, texture, rect, ofs, advance);

		} else if (type == "kerning") {
			CharType first = 0, second = 0;
			int k = 0;

			if (keys.has("first")) {
				first = keys["first"].to_int();
			}
			if (keys.has("second")) {
				second = keys["second"].to_int();
			}
			if (keys.has("amount")) {
				k = keys["amount"].to_int();
			}

			add_kerning_pair(first, second, -k);
		}

		if (f->eof_reached()) {
			break;
		}
	}

	memdelete(f);

	return OK;
}

void BitmapFont::set_height(float p_height) {
	height = p_height;
}
float BitmapFont::get_height() const {
	return height;
}

void BitmapFont::set_ascent(float p_ascent) {
	ascent = p_ascent;
}
float BitmapFont::get_ascent() const {
	return ascent;
}
float BitmapFont::get_descent() const {
	return height - ascent;
}

void BitmapFont::add_texture(const Ref<Texture> &p_texture) {
	ERR_FAIL_COND_MSG(p_texture.is_null(), "It's not a reference to a valid Texture object.");
	textures.push_back(p_texture);
}

int BitmapFont::get_texture_count() const {
	return textures.size();
};

Ref<Texture> BitmapFont::get_texture(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, textures.size(), Ref<Texture>());
	return textures[p_idx];
};

int BitmapFont::get_character_count() const {
	return char_map.size();
};

Vector<CharType> BitmapFont::get_char_keys() const {
	Vector<CharType> chars;
	chars.resize(char_map.size());
	const CharType *ct = nullptr;
	int count = 0;
	while ((ct = char_map.next(ct))) {
		chars.write[count++] = *ct;
	};

	return chars;
};

BitmapFont::Character BitmapFont::get_character(CharType p_char) const {
	if (!char_map.has(p_char)) {
		ERR_FAIL_V(Character());
	};

	return char_map[p_char];
};

void BitmapFont::add_char(CharType p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance) {
	if (p_advance < 0) {
		p_advance = p_rect.size.width;
	}

	Character c;
	c.rect = p_rect;
	c.texture_idx = p_texture_idx;
	c.v_align = p_align.y;
	c.advance = p_advance;
	c.h_align = p_align.x;

	char_map[p_char] = c;
}

void BitmapFont::add_kerning_pair(CharType p_A, CharType p_B, int p_kerning) {
	KerningPairKey kpk;
	kpk.A = p_A;
	kpk.B = p_B;

	if (p_kerning == 0 && kerning_map.has(kpk)) {
		kerning_map.erase(kpk);
	} else {
		kerning_map[kpk] = p_kerning;
	}
}

Vector<BitmapFont::KerningPairKey> BitmapFont::get_kerning_pair_keys() const {
	Vector<BitmapFont::KerningPairKey> ret;
	ret.resize(kerning_map.size());
	int i = 0;

	for (Map<KerningPairKey, int>::Element *E = kerning_map.front(); E; E = E->next()) {
		ret.write[i++] = E->key();
	}

	return ret;
}

int BitmapFont::get_kerning_pair(CharType p_A, CharType p_B) const {
	KerningPairKey kpk;
	kpk.A = p_A;
	kpk.B = p_B;

	const Map<KerningPairKey, int>::Element *E = kerning_map.find(kpk);
	if (E) {
		return E->get();
	}

	return 0;
}

void BitmapFont::set_distance_field_hint(bool p_distance_field) {
	distance_field_hint = p_distance_field;
	emit_changed();
}

bool BitmapFont::is_distance_field_hint() const {
	return distance_field_hint;
}

void BitmapFont::clear() {
	height = 1;
	ascent = 0;
	char_map.clear();
	textures.clear();
	kerning_map.clear();
	distance_field_hint = false;
}

Size2 Font::get_string_size(const String &p_string) const {
	float w = 0;

	int l = p_string.length();
	if (l == 0) {
		return Size2(0, get_height());
	}
	const CharType *sptr = &p_string[0];

	for (int i = 0; i < l; i++) {
		w += get_char_size(sptr[i], sptr[i + 1]).width;
	}

	return Size2(w, get_height());
}

Size2 Font::get_wordwrap_string_size(const String &p_string, float p_width) const {
	ERR_FAIL_COND_V(p_width <= 0, Vector2(0, get_height()));

	int l = p_string.length();
	if (l == 0) {
		return Size2(p_width, get_height());
	}

	float line_w = 0;
	float h = 0;
	float space_w = get_char_size(' ').width;
	Vector<String> lines = p_string.split("\n");
	for (int i = 0; i < lines.size(); i++) {
		h += get_height();
		String t = lines[i];
		line_w = 0;
		Vector<String> words = t.split(" ");
		for (int j = 0; j < words.size(); j++) {
			line_w += get_string_size(words[j]).x;
			if (line_w > p_width) {
				h += get_height();
				line_w = get_string_size(words[j]).x;
			} else {
				line_w += space_w;
			}
		}
	}

	return Size2(p_width, h);
}

void BitmapFont::set_fallback(const Ref<BitmapFont> &p_fallback) {
	for (Ref<BitmapFont> fallback_child = p_fallback; fallback_child != nullptr; fallback_child = fallback_child->get_fallback()) {
		ERR_FAIL_COND_MSG(fallback_child == this, "Can't set as fallback one of its parents to prevent crashes due to recursive loop.");
	}

	fallback = p_fallback;
}

Ref<BitmapFont> BitmapFont::get_fallback() const {
	return fallback;
}

float BitmapFont::draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next, const Color &p_modulate, bool p_outline) const {
	const Character *c = char_map.getptr(p_char);

	if (!c) {
		if (fallback.is_valid()) {
			return fallback->draw_char(p_canvas_item, p_pos, p_char, p_next, p_modulate, p_outline);
		}
		return 0;
	}

	ERR_FAIL_COND_V(c->texture_idx < -1 || c->texture_idx >= textures.size(), 0);
	if (!p_outline && c->texture_idx != -1) {
		Point2 cpos = p_pos;
		cpos.x += c->h_align;
		cpos.y -= ascent;
		cpos.y += c->v_align;
		VisualServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, Rect2(cpos, c->rect.size), textures[c->texture_idx]->get_rid(), c->rect, p_modulate, false, RID(), false);
	}

	return get_char_size(p_char, p_next).width;
}

Size2 BitmapFont::get_char_size(CharType p_char, CharType p_next) const {
	const Character *c = char_map.getptr(p_char);

	if (!c) {
		if (fallback.is_valid()) {
			return fallback->get_char_size(p_char, p_next);
		}
		return Size2();
	}

	Size2 ret(c->advance, c->rect.size.y);

	if (p_next) {
		KerningPairKey kpk;
		kpk.A = p_char;
		kpk.B = p_next;

		const Map<KerningPairKey, int>::Element *E = kerning_map.find(kpk);
		if (E) {
			ret.width -= E->get();
		}
	}

	return ret;
}

void BitmapFont::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_from_fnt", "path"), &BitmapFont::create_from_fnt);
	ClassDB::bind_method(D_METHOD("set_height", "px"), &BitmapFont::set_height);

	ClassDB::bind_method(D_METHOD("set_ascent", "px"), &BitmapFont::set_ascent);

	ClassDB::bind_method(D_METHOD("add_kerning_pair", "char_a", "char_b", "kerning"), &BitmapFont::add_kerning_pair);
	ClassDB::bind_method(D_METHOD("get_kerning_pair", "char_a", "char_b"), &BitmapFont::get_kerning_pair);

	ClassDB::bind_method(D_METHOD("add_texture", "texture"), &BitmapFont::add_texture);
	ClassDB::bind_method(D_METHOD("add_char", "character", "texture", "rect", "align", "advance"), &BitmapFont::add_char, DEFVAL(Point2()), DEFVAL(-1));

	ClassDB::bind_method(D_METHOD("get_texture_count"), &BitmapFont::get_texture_count);
	ClassDB::bind_method(D_METHOD("get_texture", "idx"), &BitmapFont::get_texture);

	ClassDB::bind_method(D_METHOD("set_distance_field_hint", "enable"), &BitmapFont::set_distance_field_hint);

	ClassDB::bind_method(D_METHOD("clear"), &BitmapFont::clear);

	ClassDB::bind_method(D_METHOD("_set_chars"), &BitmapFont::_set_chars);
	ClassDB::bind_method(D_METHOD("_get_chars"), &BitmapFont::_get_chars);

	ClassDB::bind_method(D_METHOD("_set_kernings"), &BitmapFont::_set_kernings);
	ClassDB::bind_method(D_METHOD("_get_kernings"), &BitmapFont::_get_kernings);

	ClassDB::bind_method(D_METHOD("_set_textures"), &BitmapFont::_set_textures);
	ClassDB::bind_method(D_METHOD("_get_textures"), &BitmapFont::_get_textures);

	ClassDB::bind_method(D_METHOD("set_fallback", "fallback"), &BitmapFont::set_fallback);
	ClassDB::bind_method(D_METHOD("get_fallback"), &BitmapFont::get_fallback);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "textures", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_textures", "_get_textures");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_INT_ARRAY, "chars", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_chars", "_get_chars");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_INT_ARRAY, "kernings", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_kernings", "_get_kernings");

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height", PROPERTY_HINT_RANGE, "1,1024,1"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ascent", PROPERTY_HINT_RANGE, "0,1024,1"), "set_ascent", "get_ascent");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "distance_field"), "set_distance_field_hint", "is_distance_field_hint");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "fallback", PROPERTY_HINT_RESOURCE_TYPE, "BitmapFont"), "set_fallback", "get_fallback");
}

BitmapFont::BitmapFont() {
	clear();
}

BitmapFont::~BitmapFont() {
	clear();
}

////////////

RES ResourceFormatLoaderBMFont::load(const String &p_path, const String &p_original_path, Error *r_error) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	Ref<BitmapFont> font;
	font.instance();

	Error err = font->create_from_fnt(p_path);

	if (err) {
		if (r_error) {
			*r_error = err;
		}
		return RES();
	}

	return font;
}

void ResourceFormatLoaderBMFont::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("fnt");
}

bool ResourceFormatLoaderBMFont::handles_type(const String &p_type) const {
	return (p_type == "BitmapFont");
}

String ResourceFormatLoaderBMFont::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "fnt") {
		return "BitmapFont";
	}
	return "";
}
