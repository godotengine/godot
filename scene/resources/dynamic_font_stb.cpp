/*************************************************************************/
/*  dynamic_font_stb.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "dynamic_font_stb.h"

#ifndef FREETYPE_ENABLED

#define STB_TRUETYPE_IMPLEMENTATION
#include "os/file_access.h"

void DynamicFontData::lock() {

	fr = font_data.read();

	if (fr.ptr() != last_data_ptr) {

		last_data_ptr = fr.ptr();

		if (!stbtt_InitFont(&info, last_data_ptr, 0)) {
			valid = false;
		} else {
			valid = true;
		}

		last_data_ptr = fr.ptr();
	}
}

void DynamicFontData::unlock() {

	fr = PoolVector<uint8_t>::Read();
}

void DynamicFontData::set_font_data(const PoolVector<uint8_t> &p_font) {
	//clear caches and stuff
	ERR_FAIL_COND(font_data.size());
	font_data = p_font;

	lock();

	if (valid) {
		stbtt_GetFontVMetrics(&info, &ascent, &descent, &linegap);
		descent = -descent + linegap;

		for (int i = 32; i < 1024; i++) {
			for (int j = 32; j < 1024; j++) {

				int kern = stbtt_GetCodepointKernAdvance(&info, i, j);
				if (kern != 0) {
					KerningPairKey kpk;
					kpk.A = i;
					kpk.B = j;
					kerning_map[kpk] = kern;
				}
			}
		}
	}

	unlock();
	//clear existing stuff

	ERR_FAIL_COND(!valid);
}

Ref<DynamicFontAtSize> DynamicFontData::_get_dynamic_font_at_size(int p_size) {

	ERR_FAIL_COND_V(!valid, Ref<DynamicFontAtSize>());

	if (size_cache.has(p_size)) {
		return Ref<DynamicFontAtSize>(size_cache[p_size]);
	}

	Ref<DynamicFontAtSize> dfas;
	dfas.instance();

	dfas->font = Ref<DynamicFontData>(this);

	size_cache[p_size] = dfas.ptr();

	dfas->size = p_size;

	lock();

	dfas->scale = stbtt_ScaleForPixelHeight(&info, p_size);

	unlock();

	return dfas;
}

DynamicFontData::DynamicFontData() {
	last_data_ptr = NULL;
	valid = false;
}

DynamicFontData::~DynamicFontData() {
}

////////////////////

float DynamicFontAtSize::get_height() const {

	return (font->ascent + font->descent) * scale;
}

float DynamicFontAtSize::get_ascent() const {

	return font->ascent * scale;
}
float DynamicFontAtSize::get_descent() const {

	return font->descent * scale;
}

Size2 DynamicFontAtSize::get_char_size(CharType p_char, CharType p_next) const {

	const_cast<DynamicFontAtSize *>(this)->_update_char(p_char);

	const Character *c = char_map.getptr(p_char);
	ERR_FAIL_COND_V(!c, Size2());

	Size2 ret(c->advance, get_height());

	if (p_next) {
		DynamicFontData::KerningPairKey kpk;
		kpk.A = p_char;
		kpk.B = p_next;

		const Map<DynamicFontData::KerningPairKey, int>::Element *K = font->kerning_map.find(kpk);
		if (K) {
			ret.x += K->get() * scale;
		}
	}

	return ret;
}

float DynamicFontAtSize::draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next, const Color &p_modulate) const {

	const_cast<DynamicFontAtSize *>(this)->_update_char(p_char);

	const Character *c = char_map.getptr(p_char);

	if (!c) {
		return 0;
	}

	Point2 cpos = p_pos;
	cpos.x += c->h_align;
	cpos.y -= get_ascent();
	cpos.y += c->v_align;
	ERR_FAIL_COND_V(c->texture_idx < -1 || c->texture_idx >= textures.size(), 0);
	if (c->texture_idx != -1)
		VisualServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, Rect2(cpos, c->rect.size), textures[c->texture_idx].texture->get_rid(), c->rect, p_modulate);

	//textures[c->texture_idx].texture->draw(p_canvas_item,Vector2());

	float ret = c->advance;
	if (p_next) {
		DynamicFontData::KerningPairKey kpk;
		kpk.A = p_char;
		kpk.B = p_next;

		const Map<DynamicFontData::KerningPairKey, int>::Element *K = font->kerning_map.find(kpk);
		if (K) {
			ret += K->get() * scale;
		}
	}

	return ret;
}

void DynamicFontAtSize::_update_char(CharType p_char) {

	if (char_map.has(p_char))
		return;

	font->lock();

	int w, h, xofs, yofs;
	unsigned char *cpbitmap = stbtt_GetCodepointBitmap(&font->info, scale, scale, p_char, &w, &h, &xofs, &yofs);

	if (!cpbitmap) {
		//no glyph

		int advance;
		stbtt_GetCodepointHMetrics(&font->info, p_char, &advance, 0);
		//print_line("char has no bitmap: "+itos(p_char)+" but advance is "+itos(advance*scale));
		Character ch;
		ch.texture_idx = -1;
		ch.advance = advance * scale;
		ch.h_align = 0;
		ch.v_align = 0;

		char_map[p_char] = ch;

		font->unlock();

		return;
	}

	int mw = w + rect_margin * 2;
	int mh = h + rect_margin * 2;

	if (mw > 4096 || mh > 4096) {

		stbtt_FreeBitmap(cpbitmap, NULL);
		font->unlock();
		ERR_FAIL_COND(mw > 4096);
		ERR_FAIL_COND(mh > 4096);
	}

	//find a texture to fit this...

	int tex_index = -1;
	int tex_x = 0;
	int tex_y = 0;

	for (int i = 0; i < textures.size(); i++) {

		CharTexture &ct = textures[i];

		if (mw > ct.texture_size || mh > ct.texture_size) //too big for this texture
			continue;

		tex_y = 0x7FFFFFFF;
		tex_x = 0;

		for (int j = 0; j < ct.texture_size - mw; j++) {

			int max_y = 0;

			for (int k = j; k < j + mw; k++) {

				int y = ct.offsets[k];
				if (y > max_y)
					max_y = y;
			}

			if (max_y < tex_y) {
				tex_y = max_y;
				tex_x = j;
			}
		}

		if (tex_y == 0x7FFFFFFF || tex_y + mh > ct.texture_size)
			continue; //fail, could not fit it here

		tex_index = i;
		break;
	}

	//print_line("CHAR: "+String::chr(p_char)+" TEX INDEX: "+itos(tex_index)+" X: "+itos(tex_x)+" Y: "+itos(tex_y));

	if (tex_index == -1) {
		//could not find texture to fit, create one
		tex_x = 0;
		tex_y = 0;

		int texsize = MAX(size * 8, 256);
		if (mw > texsize)
			texsize = mw; //special case, adapt to it?
		if (mh > texsize)
			texsize = mh; //special case, adapt to it?

		texsize = next_power_of_2(texsize);

		texsize = MIN(texsize, 4096);

		CharTexture tex;
		tex.texture_size = texsize;
		tex.imgdata.resize(texsize * texsize * 2); //grayscale alpha

		{
			//zero texture
			PoolVector<uint8_t>::Write w = tex.imgdata.write();
			ERR_FAIL_COND(texsize * texsize * 2 > tex.imgdata.size());
			for (int i = 0; i < texsize * texsize * 2; i++) {
				w[i] = 0;
			}
		}
		tex.offsets.resize(texsize);
		for (int i = 0; i < texsize; i++) //zero offsets
			tex.offsets[i] = 0;

		textures.push_back(tex);
		tex_index = textures.size() - 1;
	}

	//fit character in char texture

	CharTexture &tex = textures[tex_index];

	{
		PoolVector<uint8_t>::Write wr = tex.imgdata.write();

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

				int ofs = ((i + tex_y + rect_margin) * tex.texture_size + j + tex_x + rect_margin) * 2;
				ERR_FAIL_COND(ofs >= tex.imgdata.size());
				wr[ofs + 0] = 255; //grayscale as 1
				wr[ofs + 1] = cpbitmap[i * w + j]; //alpha as 0
			}
		}
	}

	//blit to image and texture
	{
		Ref<Image> img = memnew(Image(tex.texture_size, tex.texture_size, 0, Image::FORMAT_LA8, tex.imgdata));

		if (tex.texture.is_null()) {
			tex.texture.instance();
			tex.texture->create_from_image(img, Texture::FLAG_FILTER);
		} else {
			tex.texture->set_data(img); //update
		}
	}

	// update height array

	for (int k = tex_x; k < tex_x + mw; k++) {

		tex.offsets[k] = tex_y + mh;
	}

	int advance;
	stbtt_GetCodepointHMetrics(&font->info, p_char, &advance, 0);

	Character chr;
	chr.h_align = xofs;
	chr.v_align = yofs + get_ascent();
	chr.advance = advance * scale;
	chr.texture_idx = tex_index;

	chr.rect = Rect2(tex_x + rect_margin, tex_y + rect_margin, w, h);

	//print_line("CHAR: "+String::chr(p_char)+" TEX INDEX: "+itos(tex_index)+" RECT: "+chr.rect+" X OFS: "+itos(xofs)+" Y OFS: "+itos(yofs));

	char_map[p_char] = chr;

	stbtt_FreeBitmap(cpbitmap, NULL);

	font->unlock();
}

DynamicFontAtSize::DynamicFontAtSize() {

	rect_margin = 1;
}

DynamicFontAtSize::~DynamicFontAtSize() {

	ERR_FAIL_COND(!font.ptr());
	font->size_cache.erase(size);
}

/////////////////////////

void DynamicFont::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_font_data", "data"), &DynamicFont::set_font_data);
	ClassDB::bind_method(D_METHOD("get_font_data"), &DynamicFont::get_font_data);

	ClassDB::bind_method(D_METHOD("set_size", "data"), &DynamicFont::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &DynamicFont::get_size);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "font/size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "font/font", PROPERTY_HINT_RESOURCE_TYPE, "DynamicFontData"), "set_font_data", "get_font_data");
}

void DynamicFont::set_font_data(const Ref<DynamicFontData> &p_data) {

	data = p_data;
	data_at_size = data->_get_dynamic_font_at_size(size);
}

Ref<DynamicFontData> DynamicFont::get_font_data() const {

	return data;
}

void DynamicFont::set_size(int p_size) {

	if (size == p_size)
		return;
	size = p_size;
	ERR_FAIL_COND(p_size < 1);
	if (!data.is_valid())
		return;
	data_at_size = data->_get_dynamic_font_at_size(size);
}
int DynamicFont::get_size() const {

	return size;
}

float DynamicFont::get_height() const {

	if (!data_at_size.is_valid())
		return 1;

	return data_at_size->get_height();
}

float DynamicFont::get_ascent() const {

	if (!data_at_size.is_valid())
		return 1;

	return data_at_size->get_ascent();
}

float DynamicFont::get_descent() const {

	if (!data_at_size.is_valid())
		return 1;

	return data_at_size->get_descent();
}

Size2 DynamicFont::get_char_size(CharType p_char, CharType p_next) const {

	if (!data_at_size.is_valid())
		return Size2(1, 1);

	return data_at_size->get_char_size(p_char, p_next);
}

bool DynamicFont::is_distance_field_hint() const {

	return false;
}

float DynamicFont::draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next, const Color &p_modulate) const {

	if (!data_at_size.is_valid())
		return 0;

	return data_at_size->draw_char(p_canvas_item, p_pos, p_char, p_next, p_modulate);
}

DynamicFont::DynamicFont() {

	size = 16;
}

DynamicFont::~DynamicFont() {
}

/////////////////////////

RES ResourceFormatLoaderDynamicFont::load(const String &p_path, const String &p_original_path, Error *r_error) {

	if (r_error)
		*r_error = ERR_FILE_CANT_OPEN;

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V(!f, RES());

	PoolVector<uint8_t> data;

	data.resize(f->get_len());

	ERR_FAIL_COND_V(data.size() == 0, RES());

	{
		PoolVector<uint8_t>::Write w = data.write();
		f->get_buffer(w.ptr(), data.size());
	}

	Ref<DynamicFontData> dfd;
	dfd.instance();
	dfd->set_font_data(data);

	if (r_error)
		*r_error = OK;

	return dfd;
}

void ResourceFormatLoaderDynamicFont::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("ttf");
}

bool ResourceFormatLoaderDynamicFont::handles_type(const String &p_type) const {

	return (p_type == "DynamicFontData");
}

String ResourceFormatLoaderDynamicFont::get_resource_type(const String &p_path) const {

	String el = p_path.get_extension().to_lower();
	if (el == "ttf")
		return "DynamicFontData";
	return "";
}

#endif
