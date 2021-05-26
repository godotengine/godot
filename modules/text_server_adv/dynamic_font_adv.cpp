/*************************************************************************/
/*  dynamic_font_adv.cpp                                                 */
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

#include "dynamic_font_adv.h"

#ifdef MODULE_FREETYPE_ENABLED

#include FT_STROKER_H
#include FT_ADVANCES_H
#include FT_MULTIPLE_MASTERS_H

DynamicFontDataAdvanced::DataAtSize *DynamicFontDataAdvanced::get_data_for_size(int p_size, int p_outline_size) {
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

		//Load os2 TTF table
		fds->os2 = (TT_OS2 *)FT_Get_Sfnt_Table(fds->face, FT_SFNT_OS2);

		fds->hb_handle = hb_ft_font_create(fds->face, nullptr);
		if (fds->hb_handle == nullptr) {
			memdelete(fds);
			ERR_FAIL_V_MSG(nullptr, "Error loading HB font.");
		}

		if (p_outline_size != 0) {
			size_cache_outline[id] = fds;
		} else {
			size_cache[id] = fds;
		}

		// Write variations.
		if (fds->face->face_flags & FT_FACE_FLAG_MULTIPLE_MASTERS) {
			FT_MM_Var *amaster;

			FT_Get_MM_Var(fds->face, &amaster);

			Vector<hb_variation_t> hb_vars;
			Vector<FT_Fixed> coords;
			coords.resize(amaster->num_axis);

			FT_Get_Var_Design_Coordinates(fds->face, coords.size(), coords.ptrw());

			for (FT_UInt i = 0; i < amaster->num_axis; i++) {
				hb_variation_t var;

				// Reset to default.
				var.tag = amaster->axis[i].tag;
				var.value = (double)amaster->axis[i].def / 65536.f;
				coords.write[i] = amaster->axis[i].def;

				if (variations.has(var.tag)) {
					var.value = variations[var.tag];
					coords.write[i] = CLAMP(variations[var.tag] * 65536.f, amaster->axis[i].minimum, amaster->axis[i].maximum);
				}

				hb_vars.push_back(var);
			}

			FT_Set_Var_Design_Coordinates(fds->face, coords.size(), coords.ptrw());
			hb_font_set_variations(fds->hb_handle, hb_vars.is_empty() ? nullptr : &hb_vars[0], hb_vars.size());

			FT_Done_MM_Var(library, amaster);
		}
	}
	return fds;
}

Dictionary DynamicFontDataAdvanced::get_variation_list() const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(base_size);
	if (fds == nullptr) {
		return Dictionary();
	}

	Dictionary ret;
	// Read variations.
	if (fds->face->face_flags & FT_FACE_FLAG_MULTIPLE_MASTERS) {
		FT_MM_Var *amaster;

		FT_Get_MM_Var(fds->face, &amaster);

		for (FT_UInt i = 0; i < amaster->num_axis; i++) {
			ret[(int32_t)amaster->axis[i].tag] = Vector3i(amaster->axis[i].minimum / 65536, amaster->axis[i].maximum / 65536, amaster->axis[i].def / 65536);
		}

		FT_Done_MM_Var(library, amaster);
	}
	return ret;
}

void DynamicFontDataAdvanced::set_variation(const String &p_name, double p_value) {
	_THREAD_SAFE_METHOD_
	int32_t tag = TS->name_to_tag(p_name);
	if (!variations.has(tag) || (variations[tag] != p_value)) {
		variations[tag] = p_value;
		clear_cache();
	}
}

double DynamicFontDataAdvanced::get_variation(const String &p_name) const {
	_THREAD_SAFE_METHOD_
	int32_t tag = TS->name_to_tag(p_name);
	if (!variations.has(tag)) {
		return 0.f;
	}
	return variations[tag];
}

Dictionary DynamicFontDataAdvanced::get_feature_list() const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(base_size);
	if (fds == nullptr) {
		return Dictionary();
	}

	Dictionary out;
	// Read feature flags.
	unsigned int count = hb_ot_layout_table_get_feature_tags(hb_font_get_face(fds->hb_handle), HB_OT_TAG_GSUB, 0, nullptr, nullptr);
	if (count != 0) {
		hb_tag_t *feature_tags = (hb_tag_t *)memalloc(count * sizeof(hb_tag_t));
		hb_ot_layout_table_get_feature_tags(hb_font_get_face(fds->hb_handle), HB_OT_TAG_GSUB, 0, &count, feature_tags);
		for (unsigned int i = 0; i < count; i++) {
			out[feature_tags[i]] = 1;
		}
		memfree(feature_tags);
	}
	count = hb_ot_layout_table_get_feature_tags(hb_font_get_face(fds->hb_handle), HB_OT_TAG_GPOS, 0, nullptr, nullptr);
	if (count != 0) {
		hb_tag_t *feature_tags = (hb_tag_t *)memalloc(count * sizeof(hb_tag_t));
		hb_ot_layout_table_get_feature_tags(hb_font_get_face(fds->hb_handle), HB_OT_TAG_GPOS, 0, &count, feature_tags);
		for (unsigned int i = 0; i < count; i++) {
			out[feature_tags[i]] = 1;
		}
		memfree(feature_tags);
	}

	return out;
}

DynamicFontDataAdvanced::TexturePosition DynamicFontDataAdvanced::find_texture_pos_for_glyph(DynamicFontDataAdvanced::DataAtSize *p_data, int p_color_size, Image::Format p_image_format, int p_width, int p_height) {
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

DynamicFontDataAdvanced::Character DynamicFontDataAdvanced::Character::not_found() {
	Character ch;
	return ch;
}

DynamicFontDataAdvanced::Character DynamicFontDataAdvanced::bitmap_to_character(DynamicFontDataAdvanced::DataAtSize *p_data, FT_Bitmap bitmap, int yofs, int xofs, const Vector2 &advance) {
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

void DynamicFontDataAdvanced::update_glyph(int p_size, uint32_t p_index) {
	DataAtSize *fds = get_data_for_size(p_size, false);
	ERR_FAIL_COND(fds == nullptr);

	if (fds->glyph_map.has(p_index)) {
		return;
	}

	Character character = Character::not_found();
	FT_GlyphSlot slot = fds->face->glyph;

	if (p_index == 0) {
		fds->glyph_map[p_index] = character;
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
	FT_Get_Advance(fds->face, p_index, FT_HAS_COLOR(fds->face) ? FT_LOAD_COLOR : FT_LOAD_DEFAULT | (force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0) | ft_hinting, &h);
	FT_Get_Advance(fds->face, p_index, FT_HAS_COLOR(fds->face) ? FT_LOAD_COLOR : FT_LOAD_DEFAULT | (force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0) | ft_hinting | FT_LOAD_VERTICAL_LAYOUT, &v);

	int error = FT_Load_Glyph(fds->face, p_index, FT_HAS_COLOR(fds->face) ? FT_LOAD_COLOR : FT_LOAD_DEFAULT | (force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0) | ft_hinting);
	if (error) {
		fds->glyph_map[p_index] = character;
		return;
	}

	error = FT_Render_Glyph(fds->face->glyph, antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO);
	if (!error) {
		character = bitmap_to_character(fds, slot->bitmap, slot->bitmap_top, slot->bitmap_left, Vector2((h + (1 << 9)) >> 10, (v + (1 << 9)) >> 10) / 64.0);
	}

	fds->glyph_map[p_index] = character;
}

void DynamicFontDataAdvanced::update_glyph_outline(int p_size, int p_outline_size, uint32_t p_index) {
	DataAtSize *fds = get_data_for_size(p_size, p_outline_size);
	ERR_FAIL_COND(fds == nullptr);

	if (fds->glyph_map.has(p_index)) {
		return;
	}

	Character character = Character::not_found();
	if (p_index == 0) {
		fds->glyph_map[p_index] = character;
		return;
	}

	int error = FT_Load_Glyph(fds->face, p_index, FT_LOAD_NO_BITMAP | (force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0));
	if (error) {
		fds->glyph_map[p_index] = character;
		return;
	}

	FT_Stroker stroker;
	if (FT_Stroker_New(library, &stroker) != 0) {
		fds->glyph_map[p_index] = character;
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

	fds->glyph_map[p_index] = character;
}

void DynamicFontDataAdvanced::clear_cache() {
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

Error DynamicFontDataAdvanced::load_from_file(const String &p_filename, int p_base_size) {
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

Error DynamicFontDataAdvanced::load_from_memory(const uint8_t *p_data, size_t p_size, int p_base_size) {
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

float DynamicFontDataAdvanced::get_height(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 0.f);
	return fds->ascent + fds->descent;
}

float DynamicFontDataAdvanced::get_ascent(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 0.f);
	return fds->ascent;
}

float DynamicFontDataAdvanced::get_descent(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 0.f);
	return fds->descent;
}

float DynamicFontDataAdvanced::get_underline_position(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 0.f);
	return fds->underline_position;
}

float DynamicFontDataAdvanced::get_underline_thickness(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 0.f);
	return fds->underline_thickness;
}

bool DynamicFontDataAdvanced::is_script_supported(uint32_t p_script) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(base_size);
	ERR_FAIL_COND_V(fds == nullptr, false);

	unsigned int count = hb_ot_layout_table_get_script_tags(hb_font_get_face(fds->hb_handle), HB_OT_TAG_GSUB, 0, nullptr, nullptr);
	if (count != 0) {
		hb_tag_t *script_tags = (hb_tag_t *)memalloc(count * sizeof(hb_tag_t));
		hb_ot_layout_table_get_script_tags(hb_font_get_face(fds->hb_handle), HB_OT_TAG_GSUB, 0, &count, script_tags);
		for (unsigned int i = 0; i < count; i++) {
			if (p_script == script_tags[i]) {
				memfree(script_tags);
				return true;
			}
		}
		memfree(script_tags);
	}
	count = hb_ot_layout_table_get_script_tags(hb_font_get_face(fds->hb_handle), HB_OT_TAG_GPOS, 0, nullptr, nullptr);
	if (count != 0) {
		hb_tag_t *script_tags = (hb_tag_t *)memalloc(count * sizeof(hb_tag_t));
		hb_ot_layout_table_get_script_tags(hb_font_get_face(fds->hb_handle), HB_OT_TAG_GPOS, 0, &count, script_tags);
		for (unsigned int i = 0; i < count; i++) {
			if (p_script == script_tags[i]) {
				memfree(script_tags);
				return true;
			}
		}
		memfree(script_tags);
	}

	if (!fds->os2) {
		return false;
	}

	switch (p_script) {
		case HB_SCRIPT_COMMON:
			return (fds->os2->ulUnicodeRange1 & 1L << 4) || (fds->os2->ulUnicodeRange1 & 1L << 5) || (fds->os2->ulUnicodeRange1 & 1L << 6) || (fds->os2->ulUnicodeRange1 & 1L << 31) || (fds->os2->ulUnicodeRange2 & 1L << 0) || (fds->os2->ulUnicodeRange2 & 1L << 1) || (fds->os2->ulUnicodeRange2 & 1L << 2) || (fds->os2->ulUnicodeRange2 & 1L << 3) || (fds->os2->ulUnicodeRange2 & 1L << 4) || (fds->os2->ulUnicodeRange2 & 1L << 5) || (fds->os2->ulUnicodeRange2 & 1L << 6) || (fds->os2->ulUnicodeRange2 & 1L << 7) || (fds->os2->ulUnicodeRange2 & 1L << 8) || (fds->os2->ulUnicodeRange2 & 1L << 9) || (fds->os2->ulUnicodeRange2 & 1L << 10) || (fds->os2->ulUnicodeRange2 & 1L << 11) || (fds->os2->ulUnicodeRange2 & 1L << 12) || (fds->os2->ulUnicodeRange2 & 1L << 13) || (fds->os2->ulUnicodeRange2 & 1L << 14) || (fds->os2->ulUnicodeRange2 & 1L << 15) || (fds->os2->ulUnicodeRange2 & 1L << 30) || (fds->os2->ulUnicodeRange3 & 1L << 0) || (fds->os2->ulUnicodeRange3 & 1L << 1) || (fds->os2->ulUnicodeRange3 & 1L << 2) || (fds->os2->ulUnicodeRange3 & 1L << 4) || (fds->os2->ulUnicodeRange3 & 1L << 5) || (fds->os2->ulUnicodeRange3 & 1L << 18) || (fds->os2->ulUnicodeRange3 & 1L << 24) || (fds->os2->ulUnicodeRange3 & 1L << 25) || (fds->os2->ulUnicodeRange3 & 1L << 26) || (fds->os2->ulUnicodeRange3 & 1L << 27) || (fds->os2->ulUnicodeRange3 & 1L << 28) || (fds->os2->ulUnicodeRange4 & 1L << 3) || (fds->os2->ulUnicodeRange4 & 1L << 6) || (fds->os2->ulUnicodeRange4 & 1L << 15) || (fds->os2->ulUnicodeRange4 & 1L << 23) || (fds->os2->ulUnicodeRange4 & 1L << 24) || (fds->os2->ulUnicodeRange4 & 1L << 26);
		case HB_SCRIPT_LATIN:
			return (fds->os2->ulUnicodeRange1 & 1L << 0) || (fds->os2->ulUnicodeRange1 & 1L << 1) || (fds->os2->ulUnicodeRange1 & 1L << 2) || (fds->os2->ulUnicodeRange1 & 1L << 3) || (fds->os2->ulUnicodeRange1 & 1L << 29);
		case HB_SCRIPT_GREEK:
			return (fds->os2->ulUnicodeRange1 & 1L << 7) || (fds->os2->ulUnicodeRange1 & 1L << 30);
		case HB_SCRIPT_COPTIC:
			return (fds->os2->ulUnicodeRange1 & 1L << 8);
		case HB_SCRIPT_CYRILLIC:
			return (fds->os2->ulUnicodeRange1 & 1L << 9);
		case HB_SCRIPT_ARMENIAN:
			return (fds->os2->ulUnicodeRange1 & 1L << 10);
		case HB_SCRIPT_HEBREW:
			return (fds->os2->ulUnicodeRange1 & 1L << 11);
		case HB_SCRIPT_VAI:
			return (fds->os2->ulUnicodeRange1 & 1L << 12);
		case HB_SCRIPT_ARABIC:
			return (fds->os2->ulUnicodeRange1 & 1L << 13) || (fds->os2->ulUnicodeRange2 & 1L << 31) || (fds->os2->ulUnicodeRange3 & 1L << 3);
		case HB_SCRIPT_NKO:
			return (fds->os2->ulUnicodeRange1 & 1L << 14);
		case HB_SCRIPT_DEVANAGARI:
			return (fds->os2->ulUnicodeRange1 & 1L << 15);
		case HB_SCRIPT_BENGALI:
			return (fds->os2->ulUnicodeRange1 & 1L << 16);
		case HB_SCRIPT_GURMUKHI:
			return (fds->os2->ulUnicodeRange1 & 1L << 17);
		case HB_SCRIPT_GUJARATI:
			return (fds->os2->ulUnicodeRange1 & 1L << 18);
		case HB_SCRIPT_ORIYA:
			return (fds->os2->ulUnicodeRange1 & 1L << 19);
		case HB_SCRIPT_TAMIL:
			return (fds->os2->ulUnicodeRange1 & 1L << 20);
		case HB_SCRIPT_TELUGU:
			return (fds->os2->ulUnicodeRange1 & 1L << 21);
		case HB_SCRIPT_KANNADA:
			return (fds->os2->ulUnicodeRange1 & 1L << 22);
		case HB_SCRIPT_MALAYALAM:
			return (fds->os2->ulUnicodeRange1 & 1L << 23);
		case HB_SCRIPT_THAI:
			return (fds->os2->ulUnicodeRange1 & 1L << 24);
		case HB_SCRIPT_LAO:
			return (fds->os2->ulUnicodeRange1 & 1L << 25);
		case HB_SCRIPT_GEORGIAN:
			return (fds->os2->ulUnicodeRange1 & 1L << 26);
		case HB_SCRIPT_BALINESE:
			return (fds->os2->ulUnicodeRange1 & 1L << 27);
		case HB_SCRIPT_HANGUL:
			return (fds->os2->ulUnicodeRange1 & 1L << 28) || (fds->os2->ulUnicodeRange2 & 1L << 20) || (fds->os2->ulUnicodeRange2 & 1L << 24);
		case HB_SCRIPT_HAN:
			return (fds->os2->ulUnicodeRange2 & 1L << 21) || (fds->os2->ulUnicodeRange2 & 1L << 22) || (fds->os2->ulUnicodeRange2 & 1L << 23) || (fds->os2->ulUnicodeRange2 & 1L << 26) || (fds->os2->ulUnicodeRange2 & 1L << 27) || (fds->os2->ulUnicodeRange2 & 1L << 29);
		case HB_SCRIPT_HIRAGANA:
			return (fds->os2->ulUnicodeRange2 & 1L << 17);
		case HB_SCRIPT_KATAKANA:
			return (fds->os2->ulUnicodeRange2 & 1L << 18);
		case HB_SCRIPT_BOPOMOFO:
			return (fds->os2->ulUnicodeRange2 & 1L << 19);
		case HB_SCRIPT_TIBETAN:
			return (fds->os2->ulUnicodeRange3 & 1L << 6);
		case HB_SCRIPT_SYRIAC:
			return (fds->os2->ulUnicodeRange3 & 1L << 7);
		case HB_SCRIPT_THAANA:
			return (fds->os2->ulUnicodeRange3 & 1L << 8);
		case HB_SCRIPT_SINHALA:
			return (fds->os2->ulUnicodeRange3 & 1L << 9);
		case HB_SCRIPT_MYANMAR:
			return (fds->os2->ulUnicodeRange3 & 1L << 10);
		case HB_SCRIPT_ETHIOPIC:
			return (fds->os2->ulUnicodeRange3 & 1L << 11);
		case HB_SCRIPT_CHEROKEE:
			return (fds->os2->ulUnicodeRange3 & 1L << 12);
		case HB_SCRIPT_CANADIAN_SYLLABICS:
			return (fds->os2->ulUnicodeRange3 & 1L << 13);
		case HB_SCRIPT_OGHAM:
			return (fds->os2->ulUnicodeRange3 & 1L << 14);
		case HB_SCRIPT_RUNIC:
			return (fds->os2->ulUnicodeRange3 & 1L << 15);
		case HB_SCRIPT_KHMER:
			return (fds->os2->ulUnicodeRange3 & 1L << 16);
		case HB_SCRIPT_MONGOLIAN:
			return (fds->os2->ulUnicodeRange3 & 1L << 17);
		case HB_SCRIPT_YI:
			return (fds->os2->ulUnicodeRange3 & 1L << 19);
		case HB_SCRIPT_HANUNOO:
		case HB_SCRIPT_TAGBANWA:
		case HB_SCRIPT_BUHID:
		case HB_SCRIPT_TAGALOG:
			return (fds->os2->ulUnicodeRange3 & 1L << 20);
		case HB_SCRIPT_OLD_ITALIC:
			return (fds->os2->ulUnicodeRange3 & 1L << 21);
		case HB_SCRIPT_GOTHIC:
			return (fds->os2->ulUnicodeRange3 & 1L << 22);
		case HB_SCRIPT_DESERET:
			return (fds->os2->ulUnicodeRange3 & 1L << 23);
		case HB_SCRIPT_LIMBU:
			return (fds->os2->ulUnicodeRange3 & 1L << 29);
		case HB_SCRIPT_TAI_LE:
			return (fds->os2->ulUnicodeRange3 & 1L << 30);
		case HB_SCRIPT_NEW_TAI_LUE:
			return (fds->os2->ulUnicodeRange3 & 1L << 31);
		case HB_SCRIPT_BUGINESE:
			return (fds->os2->ulUnicodeRange4 & 1L << 0);
		case HB_SCRIPT_GLAGOLITIC:
			return (fds->os2->ulUnicodeRange4 & 1L << 1);
		case HB_SCRIPT_TIFINAGH:
			return (fds->os2->ulUnicodeRange4 & 1L << 2);
		case HB_SCRIPT_SYLOTI_NAGRI:
			return (fds->os2->ulUnicodeRange4 & 1L << 4);
		case HB_SCRIPT_LINEAR_B:
			return (fds->os2->ulUnicodeRange4 & 1L << 5);
		case HB_SCRIPT_UGARITIC:
			return (fds->os2->ulUnicodeRange4 & 1L << 7);
		case HB_SCRIPT_OLD_PERSIAN:
			return (fds->os2->ulUnicodeRange4 & 1L << 8);
		case HB_SCRIPT_SHAVIAN:
			return (fds->os2->ulUnicodeRange4 & 1L << 9);
		case HB_SCRIPT_OSMANYA:
			return (fds->os2->ulUnicodeRange4 & 1L << 10);
		case HB_SCRIPT_CYPRIOT:
			return (fds->os2->ulUnicodeRange4 & 1L << 11);
		case HB_SCRIPT_KHAROSHTHI:
			return (fds->os2->ulUnicodeRange4 & 1L << 12);
		case HB_SCRIPT_TAI_VIET:
			return (fds->os2->ulUnicodeRange4 & 1L << 13);
		case HB_SCRIPT_CUNEIFORM:
			return (fds->os2->ulUnicodeRange4 & 1L << 14);
		case HB_SCRIPT_SUNDANESE:
			return (fds->os2->ulUnicodeRange4 & 1L << 16);
		case HB_SCRIPT_LEPCHA:
			return (fds->os2->ulUnicodeRange4 & 1L << 17);
		case HB_SCRIPT_OL_CHIKI:
			return (fds->os2->ulUnicodeRange4 & 1L << 18);
		case HB_SCRIPT_SAURASHTRA:
			return (fds->os2->ulUnicodeRange4 & 1L << 19);
		case HB_SCRIPT_KAYAH_LI:
			return (fds->os2->ulUnicodeRange4 & 1L << 20);
		case HB_SCRIPT_REJANG:
			return (fds->os2->ulUnicodeRange4 & 1L << 21);
		case HB_SCRIPT_CHAM:
			return (fds->os2->ulUnicodeRange4 & 1L << 22);
		case HB_SCRIPT_ANATOLIAN_HIEROGLYPHS:
			return (fds->os2->ulUnicodeRange4 & 1L << 25);
		default:
			return false;
	};
}

void DynamicFontDataAdvanced::set_antialiased(bool p_antialiased) {
	if (antialiased != p_antialiased) {
		clear_cache();
		antialiased = p_antialiased;
	}
}

bool DynamicFontDataAdvanced::get_antialiased() const {
	return antialiased;
}

void DynamicFontDataAdvanced::set_force_autohinter(bool p_enabled) {
	if (force_autohinter != p_enabled) {
		clear_cache();
		force_autohinter = p_enabled;
	}
}

bool DynamicFontDataAdvanced::get_force_autohinter() const {
	return force_autohinter;
}

void DynamicFontDataAdvanced::set_hinting(TextServer::Hinting p_hinting) {
	if (hinting != p_hinting) {
		clear_cache();
		hinting = p_hinting;
	}
}

TextServer::Hinting DynamicFontDataAdvanced::get_hinting() const {
	return hinting;
}

bool DynamicFontDataAdvanced::has_outline() const {
	return true;
}

float DynamicFontDataAdvanced::get_base_size() const {
	return base_size;
}

String DynamicFontDataAdvanced::get_supported_chars() const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(base_size);
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

float DynamicFontDataAdvanced::get_font_scale(int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, 1.0f);

	return fds->scale_color_font / oversampling;
}

bool DynamicFontDataAdvanced::has_char(char32_t p_char) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(base_size);
	ERR_FAIL_COND_V(fds == nullptr, false);

	const_cast<DynamicFontDataAdvanced *>(this)->update_glyph(base_size, FT_Get_Char_Index(fds->face, p_char));
	Character ch = fds->glyph_map[FT_Get_Char_Index(fds->face, p_char)];

	return (ch.found);
}

hb_font_t *DynamicFontDataAdvanced::get_hb_handle(int p_size) {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, nullptr);

	return fds->hb_handle;
}

uint32_t DynamicFontDataAdvanced::get_glyph_index(char32_t p_char, char32_t p_variation_selector) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(base_size);
	ERR_FAIL_COND_V(fds == nullptr, 0);

	if (p_variation_selector == 0x0000) {
		return FT_Get_Char_Index(fds->face, p_char);
	} else {
		return FT_Face_GetCharVariantIndex(fds->face, p_char, p_variation_selector);
	}
}

Vector2 DynamicFontDataAdvanced::get_advance(uint32_t p_index, int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, Vector2());

	const_cast<DynamicFontDataAdvanced *>(this)->update_glyph(p_size, p_index);
	Character ch = fds->glyph_map[p_index];

	return ch.advance;
}

Vector2 DynamicFontDataAdvanced::get_kerning(uint32_t p_char, uint32_t p_next, int p_size) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, Vector2());

	FT_Vector delta;
	FT_Get_Kerning(fds->face, p_char, p_next, FT_KERNING_DEFAULT, &delta);
	return Vector2(delta.x, delta.y);
}

Vector2 DynamicFontDataAdvanced::draw_glyph(RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size);
	ERR_FAIL_COND_V(fds == nullptr, Vector2());

	const_cast<DynamicFontDataAdvanced *>(this)->update_glyph(p_size, p_index);
	Character ch = fds->glyph_map[p_index];

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

Vector2 DynamicFontDataAdvanced::draw_glyph_outline(RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size, p_outline_size);
	ERR_FAIL_COND_V(fds == nullptr, Vector2());

	const_cast<DynamicFontDataAdvanced *>(this)->update_glyph_outline(p_size, p_outline_size, p_index);
	Character ch = fds->glyph_map[p_index];

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

bool DynamicFontDataAdvanced::get_glyph_contours(int p_size, uint32_t p_index, Vector<Vector3> &r_points, Vector<int32_t> &r_contours, bool &r_orientation) const {
	_THREAD_SAFE_METHOD_
	DataAtSize *fds = const_cast<DynamicFontDataAdvanced *>(this)->get_data_for_size(p_size);
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

DynamicFontDataAdvanced::~DynamicFontDataAdvanced() {
	clear_cache();
	if (library != nullptr) {
		FT_Done_FreeType(library);
	}
}

#endif // MODULE_FREETYPE_ENABLED
