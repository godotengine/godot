/*************************************************************************/
/*  dynamic_font.cpp                                                     */
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

#ifdef FREETYPE_ENABLED
#include "dynamic_font.h"
#include "core/os/file_access.h"
#include "core/os/os.h"

#include FT_STROKER_H

#define __STDC_LIMIT_MACROS
#include <stdint.h>

bool DynamicFontData::CacheID::operator<(CacheID right) const {
	return key < right.key;
}

Ref<DynamicFontAtSize> DynamicFontData::_get_dynamic_font_at_size(CacheID p_cache_id) {

	if (size_cache.has(p_cache_id)) {
		return Ref<DynamicFontAtSize>(size_cache[p_cache_id]);
	}

	Ref<DynamicFontAtSize> dfas;

	dfas.instance();

	dfas->font = Ref<DynamicFontData>(this);

	size_cache[p_cache_id] = dfas.ptr();
	dfas->id = p_cache_id;
	dfas->_load();

	return dfas;
}

void DynamicFontData::set_font_ptr(const uint8_t *p_font_mem, int p_font_mem_size) {

	font_mem = p_font_mem;
	font_mem_size = p_font_mem_size;
}

void DynamicFontData::set_font_path(const String &p_path) {

	font_path = p_path;
}

String DynamicFontData::get_font_path() const {
	return font_path;
}

void DynamicFontData::set_force_autohinter(bool p_force) {

	force_autohinter = p_force;
}

void DynamicFontData::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_antialiased", "antialiased"), &DynamicFontData::set_antialiased);
	ClassDB::bind_method(D_METHOD("is_antialiased"), &DynamicFontData::is_antialiased);
	ClassDB::bind_method(D_METHOD("set_font_path", "path"), &DynamicFontData::set_font_path);
	ClassDB::bind_method(D_METHOD("get_font_path"), &DynamicFontData::get_font_path);
	ClassDB::bind_method(D_METHOD("set_hinting", "mode"), &DynamicFontData::set_hinting);
	ClassDB::bind_method(D_METHOD("get_hinting"), &DynamicFontData::get_hinting);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "antialiased"), "set_antialiased", "is_antialiased");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "hinting", PROPERTY_HINT_ENUM, "None,Light,Normal"), "set_hinting", "get_hinting");

	BIND_ENUM_CONSTANT(HINTING_NONE);
	BIND_ENUM_CONSTANT(HINTING_LIGHT);
	BIND_ENUM_CONSTANT(HINTING_NORMAL);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "font_path", PROPERTY_HINT_FILE, "*.ttf,*.otf"), "set_font_path", "get_font_path");
}

DynamicFontData::DynamicFontData() {

	antialiased = true;
	force_autohinter = false;
	hinting = DynamicFontData::HINTING_NORMAL;
	font_mem = NULL;
	font_mem_size = 0;
}

DynamicFontData::~DynamicFontData() {
}

////////////////////

Error DynamicFontAtSize::_load() {

	int error = FT_Init_FreeType(&library);

	ERR_FAIL_COND_V_MSG(error != 0, ERR_CANT_CREATE, "Error initializing FreeType.");

	if (font->font_mem == NULL && font->font_path != String()) {
		FileAccess *f = FileAccess::open(font->font_path, FileAccess::READ);
		if (!f) {
			FT_Done_FreeType(library);
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Cannot open font file '" + font->font_path + "'.");
		}

		size_t len = f->get_len();
		font->_fontdata = Vector<uint8_t>();
		font->_fontdata.resize(len);
		f->get_buffer(font->_fontdata.ptrw(), len);
		font->set_font_ptr(font->_fontdata.ptr(), len);
		f->close();
		memdelete(f);
	}

	if (font->font_mem) {
		memset(&stream, 0, sizeof(FT_StreamRec));
		stream.base = (unsigned char *)font->font_mem;
		stream.size = font->font_mem_size;
		stream.pos = 0;

		FT_Open_Args fargs;
		memset(&fargs, 0, sizeof(FT_Open_Args));
		fargs.memory_base = (unsigned char *)font->font_mem;
		fargs.memory_size = font->font_mem_size;
		fargs.flags = FT_OPEN_MEMORY;
		fargs.stream = &stream;
		error = FT_Open_Face(library, &fargs, 0, &face);

	} else {
		FT_Done_FreeType(library);
		ERR_FAIL_V_MSG(ERR_UNCONFIGURED, "DynamicFont uninitialized.");
	}

	//error = FT_New_Face( library, src_path.utf8().get_data(),0,&face );

	if (error == FT_Err_Unknown_File_Format) {

		FT_Done_FreeType(library);
		ERR_FAIL_V_MSG(ERR_FILE_CANT_OPEN, "Unknown font format.");

	} else if (error) {

		FT_Done_FreeType(library);
		ERR_FAIL_V_MSG(ERR_FILE_CANT_OPEN, "Error loading font.");
	}

	if (FT_HAS_COLOR(face) && face->num_fixed_sizes > 0) {
		int best_match = 0;
		int diff = ABS(id.size - ((int64_t)face->available_sizes[0].width));
		scale_color_font = float(id.size * oversampling) / face->available_sizes[0].width;
		for (int i = 1; i < face->num_fixed_sizes; i++) {
			int ndiff = ABS(id.size - ((int64_t)face->available_sizes[i].width));
			if (ndiff < diff) {
				best_match = i;
				diff = ndiff;
				scale_color_font = float(id.size * oversampling) / face->available_sizes[i].width;
			}
		}
		FT_Select_Size(face, best_match);
	} else {
		FT_Set_Pixel_Sizes(face, 0, id.size * oversampling);
	}

	ascent = (face->size->metrics.ascender / 64.0) / oversampling * scale_color_font;
	descent = (-face->size->metrics.descender / 64.0) / oversampling * scale_color_font;
	linegap = 0;
	texture_flags = 0;
	if (id.mipmaps)
		texture_flags |= Texture::FLAG_MIPMAPS;
	if (id.filter)
		texture_flags |= Texture::FLAG_FILTER;

	valid = true;
	return OK;
}

float DynamicFontAtSize::font_oversampling = 1.0;

float DynamicFontAtSize::get_height() const {

	return ascent + descent;
}

float DynamicFontAtSize::get_ascent() const {

	return ascent;
}
float DynamicFontAtSize::get_descent() const {

	return descent;
}

const Pair<const DynamicFontAtSize::Character *, DynamicFontAtSize *> DynamicFontAtSize::_find_char_with_font(CharType p_char, const Vector<Ref<DynamicFontAtSize> > &p_fallbacks) const {
	const Character *chr = char_map.getptr(p_char);
	ERR_FAIL_COND_V(!chr, (Pair<const Character *, DynamicFontAtSize *>(NULL, NULL)));

	if (!chr->found) {

		//not found, try in fallbacks
		for (int i = 0; i < p_fallbacks.size(); i++) {

			DynamicFontAtSize *fb = const_cast<DynamicFontAtSize *>(p_fallbacks[i].ptr());
			if (!fb->valid)
				continue;

			fb->_update_char(p_char);
			const Character *fallback_chr = fb->char_map.getptr(p_char);
			ERR_CONTINUE(!fallback_chr);

			if (!fallback_chr->found)
				continue;

			return Pair<const Character *, DynamicFontAtSize *>(fallback_chr, fb);
		}

		//not found, try 0xFFFD to display 'not found'.
		const_cast<DynamicFontAtSize *>(this)->_update_char(0xFFFD);
		chr = char_map.getptr(0xFFFD);
		ERR_FAIL_COND_V(!chr, (Pair<const Character *, DynamicFontAtSize *>(NULL, NULL)));
	}

	return Pair<const Character *, DynamicFontAtSize *>(chr, const_cast<DynamicFontAtSize *>(this));
}

Size2 DynamicFontAtSize::get_char_size(CharType p_char, CharType p_next, const Vector<Ref<DynamicFontAtSize> > &p_fallbacks) const {

	if (!valid)
		return Size2(1, 1);
	const_cast<DynamicFontAtSize *>(this)->_update_char(p_char);

	Pair<const Character *, DynamicFontAtSize *> char_pair_with_font = _find_char_with_font(p_char, p_fallbacks);
	const Character *ch = char_pair_with_font.first;
	ERR_FAIL_COND_V(!ch, Size2());

	Size2 ret(0, get_height());

	if (ch->found) {
		ret.x = ch->advance;
	}

	return ret;
}

String DynamicFontAtSize::get_available_chars() const {
	String chars;

	FT_UInt gindex;
	FT_ULong charcode = FT_Get_First_Char(face, &gindex);
	while (gindex != 0) {
		if (charcode != 0) {
			chars += CharType(charcode);
		}
		charcode = FT_Get_Next_Char(face, charcode, &gindex);
	}

	return chars;
}

void DynamicFontAtSize::set_texture_flags(uint32_t p_flags) {

	texture_flags = p_flags;
	for (int i = 0; i < textures.size(); i++) {
		Ref<ImageTexture> &tex = textures.write[i].texture;
		if (!tex.is_null())
			tex->set_flags(p_flags);
	}
}

float DynamicFontAtSize::draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next, const Color &p_modulate, const Vector<Ref<DynamicFontAtSize> > &p_fallbacks, bool p_advance_only, bool p_outline) const {

	if (!valid)
		return 0;

	const_cast<DynamicFontAtSize *>(this)->_update_char(p_char);

	Pair<const Character *, DynamicFontAtSize *> char_pair_with_font = _find_char_with_font(p_char, p_fallbacks);
	const Character *ch = char_pair_with_font.first;
	DynamicFontAtSize *font = char_pair_with_font.second;

	ERR_FAIL_COND_V(!ch, 0.0);

	float advance = 0.0;

	// use normal character size if there's no outline character
	if (p_outline && !ch->found) {
		FT_GlyphSlot slot = face->glyph;
		int error = FT_Load_Char(face, p_char, FT_HAS_COLOR(face) ? FT_LOAD_COLOR : FT_LOAD_DEFAULT);
		if (!error) {
			error = FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
			if (!error) {
				Character character = Character::not_found();
				character = const_cast<DynamicFontAtSize *>(this)->_bitmap_to_character(slot->bitmap, slot->bitmap_top, slot->bitmap_left, slot->advance.x / 64.0);
				advance = character.advance;
			}
		}
	}

	if (ch->found) {
		ERR_FAIL_COND_V(ch->texture_idx < -1 || ch->texture_idx >= font->textures.size(), 0);

		if (!p_advance_only && ch->texture_idx != -1) {
			Point2 cpos = p_pos;
			cpos.x += ch->h_align;
			cpos.y -= font->get_ascent();
			cpos.y += ch->v_align;
			Color modulate = p_modulate;
			if (FT_HAS_COLOR(font->face)) {
				modulate.r = modulate.g = modulate.b = 1.0;
			}
			RID texture = font->textures[ch->texture_idx].texture->get_rid();
			VisualServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, Rect2(cpos, ch->rect.size), texture, ch->rect_uv, modulate, false, RID(), false);
		}

		advance = ch->advance;
	}

	return advance;
}

DynamicFontAtSize::Character DynamicFontAtSize::Character::not_found() {
	Character ch;
	ch.texture_idx = -1;
	ch.advance = 0;
	ch.h_align = 0;
	ch.v_align = 0;
	ch.found = false;
	return ch;
}

DynamicFontAtSize::TexturePosition DynamicFontAtSize::_find_texture_pos_for_glyph(int p_color_size, Image::Format p_image_format, int p_width, int p_height) {
	TexturePosition ret;
	ret.index = -1;
	ret.x = 0;
	ret.y = 0;

	int mw = p_width;
	int mh = p_height;

	for (int i = 0; i < textures.size(); i++) {

		const CharTexture &ct = textures[i];

		if (ct.texture->get_format() != p_image_format)
			continue;

		if (mw > ct.texture_size || mh > ct.texture_size) //too big for this texture
			continue;

		ret.y = 0x7FFFFFFF;
		ret.x = 0;

		for (int j = 0; j < ct.texture_size - mw; j++) {

			int max_y = 0;

			for (int k = j; k < j + mw; k++) {

				int y = ct.offsets[k];
				if (y > max_y)
					max_y = y;
			}

			if (max_y < ret.y) {
				ret.y = max_y;
				ret.x = j;
			}
		}

		if (ret.y == 0x7FFFFFFF || ret.y + mh > ct.texture_size)
			continue; //fail, could not fit it here

		ret.index = i;
		break;
	}

	if (ret.index == -1) {
		//could not find texture to fit, create one
		ret.x = 0;
		ret.y = 0;

		int texsize = MAX(id.size * oversampling * 8, 256);
		if (mw > texsize)
			texsize = mw; //special case, adapt to it?
		if (mh > texsize)
			texsize = mh; //special case, adapt to it?

		texsize = next_power_of_2(texsize);

		texsize = MIN(texsize, 4096);

		CharTexture tex;
		tex.texture_size = texsize;
		tex.imgdata.resize(texsize * texsize * p_color_size); //grayscale alpha

		{
			//zero texture
			PoolVector<uint8_t>::Write w = tex.imgdata.write();
			ERR_FAIL_COND_V(texsize * texsize * p_color_size > tex.imgdata.size(), ret);

			// Initialize the texture to all-white pixels to prevent artifacts when the
			// font is displayed at a non-default scale with filtering enabled.
			if (p_color_size == 2) {
				for (int i = 0; i < texsize * texsize * p_color_size; i += 2) {
					w[i + 0] = 255;
					w[i + 1] = 0;
				}
			} else {
				for (int i = 0; i < texsize * texsize * p_color_size; i += 4) {
					w[i + 0] = 255;
					w[i + 1] = 255;
					w[i + 2] = 255;
					w[i + 3] = 0;
				}
			}
		}
		tex.offsets.resize(texsize);
		for (int i = 0; i < texsize; i++) //zero offsets
			tex.offsets.write[i] = 0;

		textures.push_back(tex);
		ret.index = textures.size() - 1;
	}

	return ret;
}

DynamicFontAtSize::Character DynamicFontAtSize::_bitmap_to_character(FT_Bitmap bitmap, int yofs, int xofs, float advance) {
	int w = bitmap.width;
	int h = bitmap.rows;

	int mw = w + rect_margin * 2;
	int mh = h + rect_margin * 2;

	ERR_FAIL_COND_V(mw > 4096, Character::not_found());
	ERR_FAIL_COND_V(mh > 4096, Character::not_found());

	int color_size = bitmap.pixel_mode == FT_PIXEL_MODE_BGRA ? 4 : 2;
	Image::Format require_format = color_size == 4 ? Image::FORMAT_RGBA8 : Image::FORMAT_LA8;

	TexturePosition tex_pos = _find_texture_pos_for_glyph(color_size, require_format, mw, mh);
	ERR_FAIL_COND_V(tex_pos.index < 0, Character::not_found());

	//fit character in char texture

	CharTexture &tex = textures.write[tex_pos.index];

	{
		PoolVector<uint8_t>::Write wr = tex.imgdata.write();

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

				int ofs = ((i + tex_pos.y + rect_margin) * tex.texture_size + j + tex_pos.x + rect_margin) * color_size;
				ERR_FAIL_COND_V(ofs >= tex.imgdata.size(), Character::not_found());
				switch (bitmap.pixel_mode) {
					case FT_PIXEL_MODE_MONO: {
						int byte = i * bitmap.pitch + (j >> 3);
						int bit = 1 << (7 - (j % 8));
						wr[ofs + 0] = 255; //grayscale as 1
						wr[ofs + 1] = (bitmap.buffer[byte] & bit) ? 255 : 0;
					} break;
					case FT_PIXEL_MODE_GRAY:
						wr[ofs + 0] = 255; //grayscale as 1
						wr[ofs + 1] = bitmap.buffer[i * bitmap.pitch + j];
						break;
					case FT_PIXEL_MODE_BGRA: {
						int ofs_color = i * bitmap.pitch + (j << 2);
						wr[ofs + 2] = bitmap.buffer[ofs_color + 0];
						wr[ofs + 1] = bitmap.buffer[ofs_color + 1];
						wr[ofs + 0] = bitmap.buffer[ofs_color + 2];
						wr[ofs + 3] = bitmap.buffer[ofs_color + 3];
					} break;
					// TODO: FT_PIXEL_MODE_LCD
					default:
						ERR_FAIL_V_MSG(Character::not_found(), "Font uses unsupported pixel format: " + itos(bitmap.pixel_mode) + ".");
						break;
				}
			}
		}
	}

	//blit to image and texture
	{

		Ref<Image> img = memnew(Image(tex.texture_size, tex.texture_size, 0, require_format, tex.imgdata));

		if (tex.texture.is_null()) {
			tex.texture.instance();
			tex.texture->create_from_image(img, Texture::FLAG_VIDEO_SURFACE | texture_flags);
		} else {
			tex.texture->set_data(img); //update
		}
	}

	// update height array

	for (int k = tex_pos.x; k < tex_pos.x + mw; k++) {
		tex.offsets.write[k] = tex_pos.y + mh;
	}

	Character chr;
	chr.h_align = xofs * scale_color_font / oversampling;
	chr.v_align = ascent - (yofs * scale_color_font / oversampling); // + ascent - descent;
	chr.advance = advance * scale_color_font / oversampling;
	chr.texture_idx = tex_pos.index;
	chr.found = true;

	chr.rect_uv = Rect2(tex_pos.x + rect_margin, tex_pos.y + rect_margin, w, h);
	chr.rect = chr.rect_uv;
	chr.rect.position /= oversampling;
	chr.rect.size = chr.rect.size * scale_color_font / oversampling;
	return chr;
}

DynamicFontAtSize::Character DynamicFontAtSize::_make_outline_char(CharType p_char) {
	Character ret = Character::not_found();

	if (FT_Load_Char(face, p_char, FT_LOAD_NO_BITMAP | (font->force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0)) != 0)
		return ret;

	FT_Stroker stroker;
	if (FT_Stroker_New(library, &stroker) != 0)
		return ret;

	FT_Stroker_Set(stroker, (int)(id.outline_size * oversampling * 64.0), FT_STROKER_LINECAP_BUTT, FT_STROKER_LINEJOIN_ROUND, 0);
	FT_Glyph glyph;
	FT_BitmapGlyph glyph_bitmap;

	if (FT_Get_Glyph(face->glyph, &glyph) != 0)
		goto cleanup_stroker;
	if (FT_Glyph_Stroke(&glyph, stroker, 1) != 0)
		goto cleanup_glyph;
	if (FT_Glyph_To_Bitmap(&glyph, font->antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO, nullptr, 1) != 0)
		goto cleanup_glyph;

	glyph_bitmap = (FT_BitmapGlyph)glyph;
	ret = _bitmap_to_character(glyph_bitmap->bitmap, glyph_bitmap->top, glyph_bitmap->left, glyph->advance.x / 65536.0);

cleanup_glyph:
	FT_Done_Glyph(glyph);
cleanup_stroker:
	FT_Stroker_Done(stroker);
	return ret;
}

void DynamicFontAtSize::_update_char(CharType p_char) {

	if (char_map.has(p_char))
		return;

	_THREAD_SAFE_METHOD_

	Character character = Character::not_found();

	FT_GlyphSlot slot = face->glyph;

	if (FT_Get_Char_Index(face, p_char) == 0) {
		char_map[p_char] = character;
		return;
	}

	int ft_hinting;

	switch (font->hinting) {
		case DynamicFontData::HINTING_NONE:
			ft_hinting = FT_LOAD_NO_HINTING;
			break;
		case DynamicFontData::HINTING_LIGHT:
			ft_hinting = FT_LOAD_TARGET_LIGHT;
			break;
		default:
			ft_hinting = FT_LOAD_TARGET_NORMAL;
			break;
	}

	int error = FT_Load_Char(face, p_char, FT_HAS_COLOR(face) ? FT_LOAD_COLOR : FT_LOAD_DEFAULT | (font->force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0) | ft_hinting);
	if (error) {
		char_map[p_char] = character;
		return;
	}

	if (id.outline_size > 0) {
		character = _make_outline_char(p_char);
	} else {
		error = FT_Render_Glyph(face->glyph, font->antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO);
		if (!error)
			character = _bitmap_to_character(slot->bitmap, slot->bitmap_top, slot->bitmap_left, slot->advance.x / 64.0);
	}

	char_map[p_char] = character;
}

void DynamicFontAtSize::update_oversampling() {
	if (oversampling == font_oversampling || !valid)
		return;

	FT_Done_FreeType(library);
	textures.clear();
	char_map.clear();
	oversampling = font_oversampling;
	valid = false;
	_load();
}

DynamicFontAtSize::DynamicFontAtSize() {

	valid = false;
	rect_margin = 1;
	ascent = 1;
	descent = 1;
	linegap = 1;
	texture_flags = 0;
	oversampling = font_oversampling;
	scale_color_font = 1;
}

DynamicFontAtSize::~DynamicFontAtSize() {

	if (valid) {
		FT_Done_FreeType(library);
	}
	font->size_cache.erase(id);
	font.unref();
}

/////////////////////////

void DynamicFont::_reload_cache() {

	ERR_FAIL_COND(cache_id.size < 1);
	if (!data.is_valid()) {
		data_at_size.unref();
		outline_data_at_size.unref();
		fallbacks.resize(0);
		fallback_data_at_size.resize(0);
		fallback_outline_data_at_size.resize(0);
		return;
	}

	data_at_size = data->_get_dynamic_font_at_size(cache_id);
	if (outline_cache_id.outline_size > 0) {
		outline_data_at_size = data->_get_dynamic_font_at_size(outline_cache_id);
		fallback_outline_data_at_size.resize(fallback_data_at_size.size());
	} else {
		outline_data_at_size.unref();
		fallback_outline_data_at_size.resize(0);
	}

	for (int i = 0; i < fallbacks.size(); i++) {
		fallback_data_at_size.write[i] = fallbacks.write[i]->_get_dynamic_font_at_size(cache_id);
		if (outline_cache_id.outline_size > 0)
			fallback_outline_data_at_size.write[i] = fallbacks.write[i]->_get_dynamic_font_at_size(outline_cache_id);
	}

	emit_changed();
	_change_notify();
}

void DynamicFont::set_font_data(const Ref<DynamicFontData> &p_data) {

	data = p_data;
	_reload_cache();

	emit_changed();
	_change_notify();
}

Ref<DynamicFontData> DynamicFont::get_font_data() const {

	return data;
}

void DynamicFont::set_size(int p_size) {

	if (cache_id.size == p_size)
		return;
	cache_id.size = p_size;
	outline_cache_id.size = p_size;
	_reload_cache();
}

int DynamicFont::get_size() const {

	return cache_id.size;
}

void DynamicFont::set_outline_size(int p_size) {
	if (outline_cache_id.outline_size == p_size)
		return;
	ERR_FAIL_COND(p_size < 0 || p_size > UINT8_MAX);
	outline_cache_id.outline_size = p_size;
	_reload_cache();
}

int DynamicFont::get_outline_size() const {
	return outline_cache_id.outline_size;
}

void DynamicFont::set_outline_color(Color p_color) {
	if (p_color != outline_color) {
		outline_color = p_color;
		emit_changed();
		_change_notify();
	}
}

Color DynamicFont::get_outline_color() const {
	return outline_color;
}

bool DynamicFont::get_use_mipmaps() const {

	return cache_id.mipmaps;
}

void DynamicFont::set_use_mipmaps(bool p_enable) {

	if (cache_id.mipmaps == p_enable)
		return;
	cache_id.mipmaps = p_enable;
	outline_cache_id.mipmaps = p_enable;
	_reload_cache();
}

bool DynamicFont::get_use_filter() const {

	return cache_id.filter;
}

void DynamicFont::set_use_filter(bool p_enable) {

	if (cache_id.filter == p_enable)
		return;
	cache_id.filter = p_enable;
	outline_cache_id.filter = p_enable;
	_reload_cache();
}

bool DynamicFontData::is_antialiased() const {

	return antialiased;
}

void DynamicFontData::set_antialiased(bool p_antialiased) {

	if (antialiased == p_antialiased)
		return;
	antialiased = p_antialiased;
}

DynamicFontData::Hinting DynamicFontData::get_hinting() const {

	return hinting;
}

void DynamicFontData::set_hinting(Hinting p_hinting) {

	if (hinting == p_hinting)
		return;
	hinting = p_hinting;
}

int DynamicFont::get_spacing(int p_type) const {

	if (p_type == SPACING_TOP) {
		return spacing_top;
	} else if (p_type == SPACING_BOTTOM) {
		return spacing_bottom;
	} else if (p_type == SPACING_CHAR) {
		return spacing_char;
	} else if (p_type == SPACING_SPACE) {
		return spacing_space;
	}

	return 0;
}

void DynamicFont::set_spacing(int p_type, int p_value) {

	if (p_type == SPACING_TOP) {
		spacing_top = p_value;
	} else if (p_type == SPACING_BOTTOM) {
		spacing_bottom = p_value;
	} else if (p_type == SPACING_CHAR) {
		spacing_char = p_value;
	} else if (p_type == SPACING_SPACE) {
		spacing_space = p_value;
	}

	emit_changed();
	_change_notify();
}

float DynamicFont::get_height() const {

	if (!data_at_size.is_valid())
		return 1;

	return data_at_size->get_height() + spacing_top + spacing_bottom;
}

float DynamicFont::get_ascent() const {

	if (!data_at_size.is_valid())
		return 1;

	return data_at_size->get_ascent() + spacing_top;
}

float DynamicFont::get_descent() const {

	if (!data_at_size.is_valid())
		return 1;

	return data_at_size->get_descent() + spacing_bottom;
}

Size2 DynamicFont::get_char_size(CharType p_char, CharType p_next) const {

	if (!data_at_size.is_valid())
		return Size2(1, 1);

	Size2 ret = data_at_size->get_char_size(p_char, p_next, fallback_data_at_size);
	if (p_char == ' ')
		ret.width += spacing_space + spacing_char;
	else if (p_next)
		ret.width += spacing_char;

	return ret;
}

String DynamicFont::get_available_chars() const {

	if (!data_at_size.is_valid())
		return "";

	String chars = data_at_size->get_available_chars();

	for (int i = 0; i < fallback_data_at_size.size(); i++) {
		String fallback_chars = fallback_data_at_size[i]->get_available_chars();
		for (int j = 0; j < fallback_chars.length(); j++) {
			if (chars.find_char(fallback_chars[j]) == -1) {
				chars += fallback_chars[j];
			}
		}
	}

	return chars;
}

bool DynamicFont::is_distance_field_hint() const {

	return false;
}

bool DynamicFont::has_outline() const {
	return outline_cache_id.outline_size > 0;
}

float DynamicFont::draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next, const Color &p_modulate, bool p_outline) const {

	if (!data_at_size.is_valid())
		return 0;

	int spacing = spacing_char;
	if (p_char == ' ') {
		spacing += spacing_space;
	}

	if (p_outline) {
		if (outline_data_at_size.is_valid() && outline_cache_id.outline_size > 0) {
			outline_data_at_size->draw_char(p_canvas_item, p_pos, p_char, p_next, p_modulate * outline_color, fallback_outline_data_at_size, false, true); // Draw glpyh outline.
		}
		return data_at_size->draw_char(p_canvas_item, p_pos, p_char, p_next, p_modulate, fallback_data_at_size, true, false) + spacing; // Return advance of the base glyph.
	} else {
		return data_at_size->draw_char(p_canvas_item, p_pos, p_char, p_next, p_modulate, fallback_data_at_size, false, false) + spacing; // Draw base glyph and return advance.
	}
}

void DynamicFont::set_fallback(int p_idx, const Ref<DynamicFontData> &p_data) {

	ERR_FAIL_COND(p_data.is_null());
	ERR_FAIL_INDEX(p_idx, fallbacks.size());
	fallbacks.write[p_idx] = p_data;
	fallback_data_at_size.write[p_idx] = fallbacks.write[p_idx]->_get_dynamic_font_at_size(cache_id);
}

void DynamicFont::add_fallback(const Ref<DynamicFontData> &p_data) {

	ERR_FAIL_COND(p_data.is_null());
	fallbacks.push_back(p_data);
	fallback_data_at_size.push_back(fallbacks.write[fallbacks.size() - 1]->_get_dynamic_font_at_size(cache_id)); //const..
	if (outline_cache_id.outline_size > 0)
		fallback_outline_data_at_size.push_back(fallbacks.write[fallbacks.size() - 1]->_get_dynamic_font_at_size(outline_cache_id));

	_change_notify();
	emit_changed();
	_change_notify();
}

int DynamicFont::get_fallback_count() const {
	return fallbacks.size();
}
Ref<DynamicFontData> DynamicFont::get_fallback(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, fallbacks.size(), Ref<DynamicFontData>());

	return fallbacks[p_idx];
}
void DynamicFont::remove_fallback(int p_idx) {

	ERR_FAIL_INDEX(p_idx, fallbacks.size());
	fallbacks.remove(p_idx);
	fallback_data_at_size.remove(p_idx);
	emit_changed();
	_change_notify();
}

bool DynamicFont::_set(const StringName &p_name, const Variant &p_value) {

	String str = p_name;
	if (str.begins_with("fallback/")) {
		int idx = str.get_slicec('/', 1).to_int();
		Ref<DynamicFontData> fd = p_value;

		if (fd.is_valid()) {
			if (idx == fallbacks.size()) {
				add_fallback(fd);
				return true;
			} else if (idx >= 0 && idx < fallbacks.size()) {
				set_fallback(idx, fd);
				return true;
			} else {
				return false;
			}
		} else if (idx >= 0 && idx < fallbacks.size()) {
			remove_fallback(idx);
			return true;
		}
	}

	return false;
}

bool DynamicFont::_get(const StringName &p_name, Variant &r_ret) const {

	String str = p_name;
	if (str.begins_with("fallback/")) {
		int idx = str.get_slicec('/', 1).to_int();

		if (idx == fallbacks.size()) {
			r_ret = Ref<DynamicFontData>();
			return true;
		} else if (idx >= 0 && idx < fallbacks.size()) {
			r_ret = get_fallback(idx);
			return true;
		}
	}

	return false;
}
void DynamicFont::_get_property_list(List<PropertyInfo> *p_list) const {

	for (int i = 0; i < fallbacks.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "fallback/" + itos(i), PROPERTY_HINT_RESOURCE_TYPE, "DynamicFontData"));
	}

	p_list->push_back(PropertyInfo(Variant::OBJECT, "fallback/" + itos(fallbacks.size()), PROPERTY_HINT_RESOURCE_TYPE, "DynamicFontData"));
}

void DynamicFont::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_font_data", "data"), &DynamicFont::set_font_data);
	ClassDB::bind_method(D_METHOD("get_font_data"), &DynamicFont::get_font_data);

	ClassDB::bind_method(D_METHOD("get_available_chars"), &DynamicFont::get_available_chars);

	ClassDB::bind_method(D_METHOD("set_size", "data"), &DynamicFont::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &DynamicFont::get_size);

	ClassDB::bind_method(D_METHOD("set_outline_size", "size"), &DynamicFont::set_outline_size);
	ClassDB::bind_method(D_METHOD("get_outline_size"), &DynamicFont::get_outline_size);

	ClassDB::bind_method(D_METHOD("set_outline_color", "color"), &DynamicFont::set_outline_color);
	ClassDB::bind_method(D_METHOD("get_outline_color"), &DynamicFont::get_outline_color);

	ClassDB::bind_method(D_METHOD("set_use_mipmaps", "enable"), &DynamicFont::set_use_mipmaps);
	ClassDB::bind_method(D_METHOD("get_use_mipmaps"), &DynamicFont::get_use_mipmaps);
	ClassDB::bind_method(D_METHOD("set_use_filter", "enable"), &DynamicFont::set_use_filter);
	ClassDB::bind_method(D_METHOD("get_use_filter"), &DynamicFont::get_use_filter);
	ClassDB::bind_method(D_METHOD("set_spacing", "type", "value"), &DynamicFont::set_spacing);
	ClassDB::bind_method(D_METHOD("get_spacing", "type"), &DynamicFont::get_spacing);

	ClassDB::bind_method(D_METHOD("add_fallback", "data"), &DynamicFont::add_fallback);
	ClassDB::bind_method(D_METHOD("set_fallback", "idx", "data"), &DynamicFont::set_fallback);
	ClassDB::bind_method(D_METHOD("get_fallback", "idx"), &DynamicFont::get_fallback);
	ClassDB::bind_method(D_METHOD("remove_fallback", "idx"), &DynamicFont::remove_fallback);
	ClassDB::bind_method(D_METHOD("get_fallback_count"), &DynamicFont::get_fallback_count);

	ADD_GROUP("Settings", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "size", PROPERTY_HINT_RANGE, "1,1024,1"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "outline_size", PROPERTY_HINT_RANGE, "0,1024,1"), "set_outline_size", "get_outline_size");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "outline_color"), "set_outline_color", "get_outline_color");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_mipmaps"), "set_use_mipmaps", "get_use_mipmaps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_filter"), "set_use_filter", "get_use_filter");
	ADD_GROUP("Extra Spacing", "extra_spacing");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "extra_spacing_top"), "set_spacing", "get_spacing", SPACING_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "extra_spacing_bottom"), "set_spacing", "get_spacing", SPACING_BOTTOM);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "extra_spacing_char"), "set_spacing", "get_spacing", SPACING_CHAR);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "extra_spacing_space"), "set_spacing", "get_spacing", SPACING_SPACE);
	ADD_GROUP("Font", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "font_data", PROPERTY_HINT_RESOURCE_TYPE, "DynamicFontData"), "set_font_data", "get_font_data");

	BIND_ENUM_CONSTANT(SPACING_TOP);
	BIND_ENUM_CONSTANT(SPACING_BOTTOM);
	BIND_ENUM_CONSTANT(SPACING_CHAR);
	BIND_ENUM_CONSTANT(SPACING_SPACE);
}

Mutex DynamicFont::dynamic_font_mutex;

SelfList<DynamicFont>::List *DynamicFont::dynamic_fonts = NULL;

DynamicFont::DynamicFont() :
		font_list(this) {

	cache_id.size = 16;
	outline_cache_id.size = 16;
	spacing_top = 0;
	spacing_bottom = 0;
	spacing_char = 0;
	spacing_space = 0;
	outline_color = Color(1, 1, 1);
	dynamic_font_mutex.lock();
	dynamic_fonts->add(&font_list);
	dynamic_font_mutex.unlock();
}

DynamicFont::~DynamicFont() {
	dynamic_font_mutex.lock();
	dynamic_fonts->remove(&font_list);
	dynamic_font_mutex.unlock();
}

void DynamicFont::initialize_dynamic_fonts() {
	dynamic_fonts = memnew(SelfList<DynamicFont>::List());
}

void DynamicFont::finish_dynamic_fonts() {
	memdelete(dynamic_fonts);
	dynamic_fonts = NULL;
}

void DynamicFont::update_oversampling() {

	Vector<Ref<DynamicFont> > changed;

	dynamic_font_mutex.lock();

	SelfList<DynamicFont> *E = dynamic_fonts->first();
	while (E) {

		if (E->self()->data_at_size.is_valid()) {
			E->self()->data_at_size->update_oversampling();

			if (E->self()->outline_data_at_size.is_valid()) {
				E->self()->outline_data_at_size->update_oversampling();
			}

			for (int i = 0; i < E->self()->fallback_data_at_size.size(); i++) {
				if (E->self()->fallback_data_at_size[i].is_valid()) {
					E->self()->fallback_data_at_size.write[i]->update_oversampling();

					if (E->self()->has_outline() && E->self()->fallback_outline_data_at_size[i].is_valid()) {
						E->self()->fallback_outline_data_at_size.write[i]->update_oversampling();
					}
				}
			}

			changed.push_back(Ref<DynamicFont>(E->self()));
		}

		E = E->next();
	}

	dynamic_font_mutex.unlock();

	for (int i = 0; i < changed.size(); i++) {
		changed.write[i]->emit_changed();
	}
}

/////////////////////////

RES ResourceFormatLoaderDynamicFont::load(const String &p_path, const String &p_original_path, Error *r_error) {

	if (r_error)
		*r_error = ERR_FILE_CANT_OPEN;

	Ref<DynamicFontData> dfont;
	dfont.instance();
	dfont->set_font_path(p_path);

	if (r_error)
		*r_error = OK;

	return dfont;
}

void ResourceFormatLoaderDynamicFont::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("ttf");
	p_extensions->push_back("otf");
}

bool ResourceFormatLoaderDynamicFont::handles_type(const String &p_type) const {

	return (p_type == "DynamicFontData");
}

String ResourceFormatLoaderDynamicFont::get_resource_type(const String &p_path) const {

	String el = p_path.get_extension().to_lower();
	if (el == "ttf" || el == "otf")
		return "DynamicFontData";
	return "";
}

#endif
