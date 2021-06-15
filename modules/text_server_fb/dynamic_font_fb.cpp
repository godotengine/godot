/*************************************************************************/
/*  dynamic_font_fb.cpp                                                  */
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

#include "dynamic_font_fb.h"

#ifdef MODULE_FREETYPE_ENABLED

#include FT_STROKER_H
#include FT_ADVANCES_H

DynamicFontDataFallback::DataAtSize *DynamicFontDataFallback::get_data_for_size(int p_size, int p_outline_size) {
	ERR_FAIL_COND_V(!valid, nullptr);
	ERR_FAIL_COND_V(p_size < 0 || p_size > UINT16_MAX, nullptr);
	ERR_FAIL_COND_V(p_outline_size < 0 || p_outline_size > UINT16_MAX, nullptr);

	CacheID id;
	id.size = p_size;
	id.outline_size = p_outline_size;

	DataAtSize *fds = nullptr;
	Map<CacheID, DataAtSize *>::Element *E = nullptr;
	if (p_outline_size != 0) {
		E = size_cache_outline.find(id);
	} else {
		E = size_cache.find(id);
	}

	if (E != nullptr) {
		fds = E->get();
	} else {
		if (font_mem == nullptr && font_path != String()) {
			if (!font_mem_cache.is_empty()) {
				font_mem = font_mem_cache.ptr();
				font_mem_size = font_mem_cache.size();
			} else {
				FileAccess *f = FileAccess::open(font_path, FileAccess::READ);
				if (!f) {
					ERR_FAIL_V_MSG(nullptr, "Cannot open font file '" + font_path + "'.");
				}

				uint64_t len = f->get_length();
				font_mem_cache.resize(len);
				f->get_buffer(font_mem_cache.ptrw(), len);
				font_mem = font_mem_cache.ptr();
				font_mem_size = len;
				f->close();
			}
		}

		int error = 0;
		fds = memnew(DataAtSize);
		if (font_mem) {
			memset(&fds->stream, 0, sizeof(FT_StreamRec));
			fds->stream.base = (unsigned char *)font_mem;
			fds->stream.size = font_mem_size;
			fds->stream.pos = 0;

			FT_Open_Args fargs;
			memset(&fargs, 0, sizeof(FT_Open_Args));
			fargs.memory_base = (unsigned char *)font_mem;
			fargs.memory_size = font_mem_size;
			fargs.flags = FT_OPEN_MEMORY;
			fargs.stream = &fds->stream;
			error = FT_Open_Face(library, &fargs, 0, &fds->face);

		} else {
			memdelete(fds);
			ERR_FAIL_V_MSG(nullptr, "DynamicFont uninitialized.");
		}

		if (error == FT_Err_Unknown_File_Format) {
			memdelete(fds);
			ERR_FAIL_V_MSG(nullptr, "Unknown font format.");

		} else if (error) {
			memdelete(fds);
			ERR_FAIL_V_MSG(nullptr, "Error loading font.");
		}

		oversampling = TS->font_get_oversampling();

		if (FT_HAS_COLOR(fds->face) && fds->face->num_fixed_sizes > 0) {
			int best_match = 0;
			int diff = ABS(p_size - ((int64_t)fds->face->available_sizes[0].width));
			fds->scale_color_font = float(p_size * oversampling) / fds->face->available_sizes[0].width;
			for (int i = 1; i < fds->face->num_fixed_sizes; i++) {
				int ndiff = ABS(p_size - ((int64_t)fds->face->available_sizes[i].width));
				if (ndiff < diff) {
					best_match = i;
					diff = ndiff;
					fds->scale_color_font = float(p_size * oversampling) / fds->face->available_sizes[i].width;
				}
			}
			FT_Select_Size(fds->face, best_match);
		} else {
			FT_Set_Pixel_Sizes(fds->face, 0, p_size * oversampling);
		}

		fds->size = p_size;
		fds->ascent = (fds->face->size->metrics.ascender / 64.0) / oversampling * fds->scale_color_font;
		fds->descent = (-fds->face->size->metrics.descender / 64.0) / oversampling * fds->scale_color_font;
		fds->underline_position = (-FT_MulFix(fds->face->underline_position, fds->face->size->metrics.y_scale) / 64.0) / oversampling * fds->scale_color_font;
		fds->underline_thickness = (FT_MulFix(fds->face->underline_thickness, fds->face->size->metrics.y_scale) / 64.0) / oversampling * fds->scale_color_font;

		if (p_outline_size != 0) {
			size_cache_outline[id] = fds;
		} else {
			size_cache[id] = fds;
		}
	}

	return fds;
}

DynamicFontDataFallback::TexturePosition DynamicFontDataFallback::find_texture_pos_for_glyph(DynamicFontDataFallback::DataAtSize *p_data, int p_color_size, Image::Format p_image_format, int p_width, int p_height) {
	TexturePosition ret;
	ret.index = -1;

	int mw = p_width;
	int mh = p_height;

	for (int i = 0; i < p_data->textures.size(); i++) {
		const CharTexture &ct = p_data->textures[i];

		if (RenderingServer::get_singleton() != nullptr) {
			if (ct.texture->get_format() != p_image_format) {
				continue;
			}
		}

		if (mw > ct.texture_size || mh > ct.texture_size) { //too big for this texture
			continue;
		}

		ret.y = 0x7FFFFFFF;
		ret.x = 0;

		for (int j = 0; j < ct.texture_size - mw; j++) {
			int max_y = 0;

			for (int k = j; k < j + mw; k++) {
				int y = ct.offsets[k];
				if (y > max_y) {
					max_y = y;
				}
			}

			if (max_y < ret.y) {
				ret.y = max_y;
				ret.x = j;
			}
		}

		if (ret.y == 0x7FFFFFFF || ret.y + mh > ct.texture_size) {
			continue; //fail, could not fit it here
		}

		ret.index = i;
		break;
	}

	if (ret.index == -1) {
		//could not find texture to fit, create one
		ret.x = 0;
		ret.y = 0;

		int texsize = MAX(p_data->size * oversampling * 8, 256);
		if (mw > texsize) {
			texsize = mw; //special case, adapt to it?
		}
		if (mh > texsize) {
			texsize = mh; //special case, adapt to it?
		}

		texsize = next_power_of_2(texsize);

		texsize = MIN(texsize, 4096);

		CharTexture tex;
		tex.texture_size = texsize;
		tex.imgdata.resize(texsize * texsize * p_color_size); //grayscale alpha

		{
			//zero texture
			uint8_t *w = tex.imgdata.ptrw();
			ERR_FAIL_COND_V(texsize * texsize * p_color_size > tex.imgdata.size(), ret);
			// Initialize the texture to all-white pixels to prevent artifacts when the
			// font is displayed at a non-default scale with filtering enabled.
			if (p_color_size == 2) {
				for (int i = 0; i < texsize * texsize * p_color_size; i += 2) { // FORMAT_LA8
					w[i + 0] = 255;
					w[i + 1] = 0;
				}
			} else if (p_color_size == 4) {
				for (int i = 0; i < texsize * texsize * p_color_size; i += 4) { // FORMAT_RGBA8
					w[i + 0] = 255;
					w[i + 1] = 255;
					w[i + 2] = 255;
					w[i + 3] = 0;
				}
			} else {
				ERR_FAIL_V(ret);
			}
		}
		tex.offsets.resize(texsize);
		for (int i = 0; i < texsize; i++) { //zero offsets
			tex.offsets.write[i] = 0;
		}

		p_data->textures.push_back(tex);
		ret.index = p_data->textures.size() - 1;
	}

	return ret;
}

DynamicFontDataFallback::Character DynamicFontDataFallback::Character::not_found() {
	Character ch;
	return ch;
}

DynamicFontDataFallback::Character DynamicFontDataFallback::bitmap_to_character(DynamicFontDataFallback::DataAtSize *p_data, FT_Bitmap bitmap, int yofs, int xofs, const Vector2 &advance) {
	int w = bitmap.width;
	int h = bitmap.rows;

	int mw = w + rect_margin * 2;
	int mh = h + rect_margin * 2;

	ERR_FAIL_COND_V(mw > 4096, Character::not_found());
	ERR_FAIL_COND_V(mh > 4096, Character::not_found());

	int color_size = bitmap.pixel_mode == FT_PIXEL_MODE_BGRA ? 4 : 2;
	Image::Format require_format = color_size == 4 ? Image::FORMAT_RGBA8 : Image::FORMAT_LA8;

	TexturePosition tex_pos = find_texture_pos_for_glyph(p_data, color_size, require_format, mw, mh);
	ERR_FAIL_COND_V(tex_pos.index < 0, Character::not_found());

	//fit character in char texture

	CharTexture &tex = p_data->textures.write[tex_pos.index];

	{
		uint8_t *wr = tex.imgdata.ptrw();

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
		if (RenderingServer::get_singleton() != nullptr) {
			Ref<Image> img = memnew(Image(tex.texture_size, tex.texture_size, 0, require_format, tex.imgdata));

			if (tex.texture.is_null()) {
				tex.texture.instance();
				tex.texture->create_from_image(img);
			} else {
				tex.texture->update(img); //update
			}
		}
	}

	// update height array
	for (int k = tex_pos.x; k < tex_pos.x + mw; k++) {
		tex.offsets.write[k] = tex_pos.y + mh;
	}

	Character chr;
	chr.align = (Vector2(xofs, -yofs) * p_data->scale_color_font / oversampling).round();
	chr.advance = (advance * p_data->scale_color_font / oversampling).round();
	chr.texture_idx = tex_pos.index;
	chr.found = true;

	chr.rect_uv = Rect2(tex_pos.x + rect_margin, tex_pos.y + rect_margin, w, h);
	chr.rect = chr.rect_uv;
	chr.rect.position /= oversampling;
	chr.rect.size *= (p_data->scale_color_font / oversampling);
	return chr;
}

void DynamicFontDataFallback::update_char(int p_size, char32_t p_char) {
	DataAtSize *fds = get_data_for_size(p_size, false);
	ERR_FAIL_COND(fds == nullptr);

	if (fds->char_map.has(p_char)) {
		return;
	}

	Character character = Character::not_found();

	FT_GlyphSlot slot = fds->face->glyph;
	FT_UInt gl_index = FT_Get_Char_Index(fds->face, p_char);

	if (gl_index == 0) {
		fds->char_map[p_char] = character;
		return;
	}

	int ft_hinting;
	switch (hinting) {
		case TextServer::HINTING_NONE:
			ft_hinting = FT_LOAD_NO_HINTING;
			break;
		case TextServer::HINTING_LIGHT:
			ft_hinting = FT_LOAD_TARGET_LIGHT;
			break;
		default:
			ft_hinting = FT_LOAD_TARGET_NORMAL;
			break;
	}

	FT_Fixed v, h;
	FT_Get_Advance(fds->face, gl_index, FT_HAS_COLOR(fds->face) ? FT_LOAD_COLOR : FT_LOAD_DEFAULT | (force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0) | ft_hinting, &h);
	FT_Get_Advance(fds->face, gl_index, FT_HAS_COLOR(fds->face) ? FT_LOAD_COLOR : FT_LOAD_DEFAULT | (force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0) | ft_hinting | FT_LOAD_VERTICAL_LAYOUT, &v);

	int error = FT_Load_Glyph(fds->face, gl_index, FT_HAS_COLOR(fds->face) ? FT_LOAD_COLOR : FT_LOAD_DEFAULT | (force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0) | ft_hinting);
	if (error) {
		fds->char_map[p_char] = character;
		return;
	}

	error = FT_Render_Glyph(fds->face->glyph, antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO);
	if (!error) {
		character = bitmap_to_character(fds, slot->bitmap, slot->bitmap_top, slot->bitmap_left, Vector2((h + (1 << 9)) >> 10, (v + (1 << 9)) >> 10) / 64.0);
	}

	fds->char_map[p_char] = character;
}

void DynamicFontDataFallback::update_char_outline(int p_size, int p_outline_size, char32_t p_char) {
	DataAtSize *fds = get_data_for_size(p_size, p_outline_size);
	ERR_FAIL_COND(fds == nullptr);

	if (fds->char_map.has(p_char)) {
		return;
	}

	Character character = Character::not_found();
	FT_UInt gl_index = FT_Get_Char_Index(fds->face, p_char);

	if (gl_index == 0) {
		fds->char_map[p_char] = character;
		return;
	}

	int error = FT_Load_Glyph(fds->face, gl_index, FT_LOAD_NO_BITMAP | (force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0));
	if (error) {
		fds->char_map[p_char] = character;
		return;
	}

	FT_Stroker stroker;
	if (FT_Stroker_New(library, &stroker) != 0) {
		fds->char_map[p_char] = character;
		return;
	}

	FT_Stroker_Set(stroker, (int)(p_outline_size * oversampling * 64.0), FT_STROKER_LINECAP_BUTT, FT_STROKER_LINEJOIN_ROUND, 0);
	FT_Glyph glyph;
	FT_BitmapGlyph glyph_bitmap;

	if (FT_Get_Glyph(fds->face->glyph, &glyph) != 0) {
		goto cleanup_stroker;
	}
	if (FT_Glyph_Stroke(&glyph, stroker, 1) != 0) {
		goto cleanup_glyph;
	}
	if (FT_Glyph_To_Bitmap(&glyph, antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO, nullptr, 1) != 0) {
		goto cleanup_glyph;
	}

	glyph_bitmap = (FT_BitmapGlyph)glyph;
	character = bitmap_to_character(fds, glyph_bitmap->bitmap, glyph_bitmap->top, glyph_bitmap->left, Vector2());

cleanup_glyph:
	FT_Done_Glyph(glyph);
cleanup_stroker:
	FT_Stroker_Done(stroker);

	fds->char_map[p_char] = character;
}

void DynamicFontDataFallback::clear_cache() {
	_THREAD_SAFE_METHOD_
	for (Map<CacheID, DataAtSize *>::Element *E = size_cache.front(); E; E = E->next()) {
		memdelete(E->get());
	}
	size_cache.clear();
	for (Map<CacheID, DataAtSize *>::Element *E = size_cache_outline.front(); E; E = E->next()) {
		memdelete(E->get());
	}
	size_cache_outline.clear();
}

Error DynamicFontDataFallback::load_from_file(const String &p_filename, int p_base_size) {
	_THREAD_SAFE_METHOD_
	if (library == nullptr) {
		int error = FT_Init_FreeType(&library);
		ERR_FAIL_COND_V_MSG(error != 0, ERR_CANT_CREATE, "Error initializing FreeType.");
	}
	clear_cache();

	font_path = p_filename;
	base_size = p_base_size;

	valid = true;
	DataAtSize *fds = get_data_for_size(base_size); // load base size.
	if (fds == nullptr) {
		valid = false;
		ERR_FAIL_V(ERR_CANT_CREATE);
	}

	return OK;
}

Error DynamicFontDataFallback::load_from_memory(const uint8_t *p_data, size_t p_size, int p_base_size) {
	_THREAD_SAFE_METHOD_
	if (library == nullptr) {
		int error = FT_Init_FreeType(&library);
		ERR_FAIL_COND_V_MSG(error != 0, ERR_CANT_CREATE, "Error initializing FreeType.");
	}
	clear_cache();

	font_mem = p_data;
	font_mem_size = p_size;
	base_size = p_base_size;

	valid = true;
	DataAtSize *fds = get_data_for_size(base_size); // load base size.
	if (fds == nullptr) {
		valid = false;
		ERR_FAIL_V(ERR_CANT_CREATE);
	}

	return OK;
}

float DynamicFontDataFallback::get_height(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 0.f);
	return fds->ascent + fds->descent;
}

float DynamicFontDataFallback::get_ascent(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 0.f);
	return fds->ascent;
}

float DynamicFontDataFallback::get_descent(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 0.f);
	return fds->descent;
}

float DynamicFontDataFallback::get_underline_position(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 0.f);
	return fds->underline_position;
}

float DynamicFontDataFallback::get_underline_thickness(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 0.f);
	return fds->underline_thickness;
}

void DynamicFontDataFallback::set_antialiased(bool p_antialiased) {
	if (antialiased != p_antialiased) {
		clear_cache();
		antialiased = p_antialiased;
	}
}

bool DynamicFontDataFallback::get_antialiased() const {
	return antialiased;
}

void DynamicFontDataFallback::set_force_autohinter(bool p_enabled) {
	if (force_autohinter != p_enabled) {
		clear_cache();
		force_autohinter = p_enabled;
	}
}

bool DynamicFontDataFallback::get_force_autohinter() const {
	return force_autohinter;
}

void DynamicFontDataFallback::set_hinting(TextServer::Hinting p_hinting) {
	if (hinting != p_hinting) {
		clear_cache();
		hinting = p_hinting;
	}
}

TextServer::Hinting DynamicFontDataFallback::get_hinting() const {
	return hinting;
}

bool DynamicFontDataFallback::has_outline() const {
	return true;
}

float DynamicFontDataFallback::get_base_size() const {
	return base_size;
}

bool DynamicFontDataFallback::has_char(char32_t p_char) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(base_size);
	ERR_FAIL_COND_V(fds == nullptr, false);

	const_cast<DynamicFontDataFallback *>(this)->update_char(base_size, p_char);
	Character ch = fds->char_map[p_char];

	return (ch.found);
}

String DynamicFontDataFallback::get_supported_chars() const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(base_size);
	ERR_FAIL_COND_V(fds == nullptr, String());

	String chars;

	FT_UInt gindex;
	FT_ULong charcode = FT_Get_First_Char(fds->face, &gindex);
	while (gindex != 0) {
		if (charcode != 0) {
			chars += char32_t(charcode);
		}
		charcode = FT_Get_Next_Char(fds->face, charcode, &gindex);
	}

	return chars;
}

Vector2 DynamicFontDataFallback::get_advance(char32_t p_char, int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, Vector2());

	const_cast<DynamicFontDataFallback *>(this)->update_char(p_size, p_char);
	Character ch = fds->char_map[p_char];

	return ch.advance;
}

Vector2 DynamicFontDataFallback::get_kerning(char32_t p_char, char32_t p_next, int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, Vector2());

	FT_Vector delta;
	FT_Get_Kerning(fds->face, FT_Get_Char_Index(fds->face, p_char), FT_Get_Char_Index(fds->face, p_next), FT_KERNING_DEFAULT, &delta);
	return Vector2(delta.x, delta.y);
}

Vector2 DynamicFontDataFallback::draw_glyph(RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, Vector2());

	const_cast<DynamicFontDataFallback *>(this)->update_char(p_size, p_index);
	Character ch = fds->char_map[p_index];

	Vector2 advance;
	if (ch.found) {
		ERR_FAIL_COND_V(ch.texture_idx < -1 || ch.texture_idx >= fds->textures.size(), Vector2());

		if (ch.texture_idx != -1) {
			Point2i cpos = p_pos;
			cpos += ch.align;

			Color modulate = p_color;
			if (FT_HAS_COLOR(fds->face)) {
				modulate.r = modulate.g = modulate.b = 1.0;
			}
			if (RenderingServer::get_singleton() != nullptr) {
				RID texture = fds->textures[ch.texture_idx].texture->get_rid();
				RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, Rect2(cpos, ch.rect.size), texture, ch.rect_uv, modulate, false, false);
			}
		}

		advance = ch.advance;
	}

	return advance;
}

Vector2 DynamicFontDataFallback::draw_glyph_outline(RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(p_size, p_outline_size);
	ERR_FAIL_COND_V(fds == nullptr, Vector2());

	const_cast<DynamicFontDataFallback *>(this)->update_char_outline(p_size, p_outline_size, p_index);
	Character ch = fds->char_map[p_index];

	Vector2 advance;
	if (ch.found) {
		ERR_FAIL_COND_V(ch.texture_idx < -1 || ch.texture_idx >= fds->textures.size(), Vector2());

		if (ch.texture_idx != -1) {
			Point2i cpos = p_pos;
			cpos += ch.align;

			Color modulate = p_color;
			if (FT_HAS_COLOR(fds->face)) {
				modulate.r = modulate.g = modulate.b = 1.0;
			}
			if (RenderingServer::get_singleton() != nullptr) {
				RID texture = fds->textures[ch.texture_idx].texture->get_rid();
				RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, Rect2(cpos, ch.rect.size), texture, ch.rect_uv, modulate, false, false);
			}
		}

		advance = ch.advance;
	}

	return advance;
}

bool DynamicFontDataFallback::get_glyph_contours(int p_size, uint32_t p_index, Vector<Vector3> &r_points, Vector<int32_t> &r_contours, bool &r_orientation) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataFallback *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, false);

	int error = FT_Load_Glyph(fds->face, p_index, FT_LOAD_NO_BITMAP | (force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0));
	ERR_FAIL_COND_V(error, false);

	r_points.clear();
	r_contours.clear();

	float h = fds->ascent;
	float scale = (1.0 / 64.0) / oversampling * fds->scale_color_font;
	for (short i = 0; i < fds->face->glyph->outline.n_points; i++) {
		r_points.push_back(Vector3(fds->face->glyph->outline.points[i].x * scale, h - fds->face->glyph->outline.points[i].y * scale, FT_CURVE_TAG(fds->face->glyph->outline.tags[i])));
	}
	for (short i = 0; i < fds->face->glyph->outline.n_contours; i++) {
		r_contours.push_back(fds->face->glyph->outline.contours[i]);
	}
	r_orientation = (FT_Outline_Get_Orientation(&fds->face->glyph->outline) == FT_ORIENTATION_FILL_RIGHT);
	return true;
}

DynamicFontDataFallback::~DynamicFontDataFallback() {
	clear_cache();
	if (library != nullptr) {
		FT_Done_FreeType(library);
	}
}

#endif // MODULE_FREETYPE_ENABLED
