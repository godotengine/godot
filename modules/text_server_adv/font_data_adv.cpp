/*************************************************************************/
/*  font_data_adv.cpp                                                    */
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

#include "font_data_adv.h"

#include "core/io/compression.h"
#include "core/io/file_access_memory.h"
#include "core/io/resource_saver.h"

#ifdef MODULE_MSDFGEN_ENABLED
#include "msdfgen.h"
#endif

#define MAGIC(c1, c2, c3, c4) ((((uint32_t)(c4)&0xFF) << 24) | (((uint32_t)(c3)&0xFF) << 16) | (((uint32_t)(c2)&0xFF) << 8) | ((uint32_t)(c1)&0xFF))

/*************************************************************************/
/*  hb_bmp_font_t HarfBuzz Bitmap font interface                         */
/*************************************************************************/

struct hb_bmp_font_t {
	FontDataAdvanced *face = nullptr;
	bool unref = false; /* Whether to destroy bm_face when done. */
};

static hb_bmp_font_t *_hb_bmp_font_create(FontDataAdvanced *p_face, bool p_unref) {
	hb_bmp_font_t *bm_font = memnew(hb_bmp_font_t);

	if (!bm_font) {
		return nullptr;
	}

	bm_font->face = p_face;
	bm_font->unref = p_unref;

	return bm_font;
}

static void _hb_bmp_font_destroy(void *data) {
	hb_bmp_font_t *bm_font = reinterpret_cast<hb_bmp_font_t *>(data);
	memdelete(bm_font);
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

	return bm_font->face->get_advance(glyph, -1).x * 64;
}

static hb_position_t hb_bmp_get_glyph_v_advance(hb_font_t *font, void *font_data, hb_codepoint_t glyph, void *user_data) {
	const hb_bmp_font_t *bm_font = reinterpret_cast<const hb_bmp_font_t *>(font_data);

	if (!bm_font->face) {
		return 0;
	}

	if (!bm_font->face->has_char(glyph)) {
		return 0;
	}

	return -bm_font->face->get_advance(glyph, -1).y * 64;
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

	return bm_font->face->get_kerning(left_glyph, right_glyph, -1).x * 64;
}

static hb_bool_t hb_bmp_get_glyph_v_origin(hb_font_t *font, void *font_data, hb_codepoint_t glyph, hb_position_t *x, hb_position_t *y, void *user_data) {
	const hb_bmp_font_t *bm_font = reinterpret_cast<const hb_bmp_font_t *>(font_data);

	if (!bm_font->face) {
		return false;
	}

	if (!bm_font->face->has_char(glyph)) {
		return false;
	}

	*x = bm_font->face->get_advance(glyph, -1).x * 32;
	*y = -bm_font->face->get_ascent(-1) * 64;

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
	extents->width = bm_font->face->get_glyph_size(glyph, -1).x * 64;
	extents->height = bm_font->face->get_glyph_size(glyph, -1).y * 64;

	return true;
}

static hb_bool_t hb_bmp_get_font_h_extents(hb_font_t *font, void *font_data, hb_font_extents_t *metrics, void *user_data) {
	const hb_bmp_font_t *bm_font = reinterpret_cast<const hb_bmp_font_t *>(font_data);

	if (!bm_font->face) {
		return false;
	}

	metrics->ascender = bm_font->face->get_ascent(-1);
	metrics->descender = bm_font->face->get_descent(-1);
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

static void _hb_bmp_font_set_funcs(hb_font_t *p_font, FontDataAdvanced *p_face, bool p_unref) {
	hb_font_set_funcs(p_font, funcs, _hb_bmp_font_create(p_face, p_unref), _hb_bmp_font_destroy);
}

hb_font_t *hb_bmp_font_create(FontDataAdvanced *p_face, hb_destroy_func_t p_destroy) {
	hb_font_t *font;
	hb_face_t *face = hb_face_create(nullptr, 0);

	font = hb_font_create(face);
	hb_face_destroy(face);
	_hb_bmp_font_set_funcs(font, p_face, false);
	return font;
}

/*************************************************************************/
/* Font Cache                                                            */
/*************************************************************************/

FontDataAdvanced::Data *FontDataAdvanced::get_cache_data(const VariationKey &p_var_id, const SizeKey &p_size_id) {
	Data *fd = nullptr;

	Map<SizeKey, Data *> &variation_cache = cache[p_var_id];

	SizeKey id = p_size_id;
	if (font_type == FONT_BITMAP || msdf) {
		id.size = base_size.size; // Use base size for bitmap fonts.
		id.outline_size = 0;
	}
	Map<SizeKey, Data *>::Element *E = variation_cache.find(id);

	if (E != nullptr) {
		fd = E->get(); // Font data found in cache.
	} else {
		fd = memnew(Data); // Create new font data cache record.

		fd->variation_id = p_var_id;
		fd->size_id = id;

		if (font_type == FONT_BITMAP) { // Load bitmap font.
			fd->hb_handle = hb_bmp_font_create(this, nullptr);
			if (fd->hb_handle == nullptr) {
				memdelete(fd);
				ERR_FAIL_V_MSG(nullptr, "Error loading HarfBuzz font object.");
			}
		} else if (font_type == FONT_DYNAMIC) { // Load dynamic font.
#ifdef MODULE_FREETYPE_ENABLED
			int error = 0;

			if (library == nullptr) {
				error = FT_Init_FreeType(&library);
				ERR_FAIL_COND_V_MSG(error != 0, nullptr, "FreeType: Error initializing libray: '" + String(FT_Error_String(error)) + "'.");
			}

			memset(&fd->stream, 0, sizeof(FT_StreamRec));
			fd->stream.base = (unsigned char *)font_mem;
			fd->stream.size = font_mem_size;
			fd->stream.pos = 0;

			FT_Open_Args fargs;
			memset(&fargs, 0, sizeof(FT_Open_Args));
			fargs.memory_base = (unsigned char *)font_mem;
			fargs.memory_size = font_mem_size;
			fargs.flags = FT_OPEN_MEMORY;
			fargs.stream = &fd->stream;
			error = FT_Open_Face(library, &fargs, 0, &fd->face);
			if (error) {
				memdelete(fd);
				ERR_FAIL_V_MSG(nullptr, "FreeType: Error loading font: '" + String(FT_Error_String(error)) + "'.");
			}

			if (msdf) {
				fd->oversampling = 1.0f;
			} else if (oversampling > 0.0f) {
				fd->oversampling = oversampling;
			} else {
				fd->oversampling = TS->font_get_global_oversampling();
			}

			if (FT_HAS_COLOR(fd->face) && fd->face->num_fixed_sizes > 0) {
				int best_match = 0;
				int diff = ABS(id.size - ((int64_t)fd->face->available_sizes[0].width));
				fd->scale_color_font = float(id.size * fd->oversampling) / fd->face->available_sizes[0].width;
				for (int i = 1; i < fd->face->num_fixed_sizes; i++) {
					int ndiff = ABS(id.size - ((int64_t)fd->face->available_sizes[i].width));
					if (ndiff < diff) {
						best_match = i;
						diff = ndiff;
						fd->scale_color_font = float(id.size * fd->oversampling) / fd->face->available_sizes[i].width;
					}
				}
				FT_Select_Size(fd->face, best_match);
			} else {
				FT_Set_Pixel_Sizes(fd->face, 0, id.size * fd->oversampling);
			}

			fd->ascent = (fd->face->size->metrics.ascender / 64.0) / fd->oversampling * fd->scale_color_font;
			fd->descent = (-fd->face->size->metrics.descender / 64.0) / fd->oversampling * fd->scale_color_font;
			fd->height = fd->ascent + fd->descent;
			fd->underline_position = (-FT_MulFix(fd->face->underline_position, fd->face->size->metrics.y_scale) / 64.0) / fd->oversampling * fd->scale_color_font;
			fd->underline_thickness = (FT_MulFix(fd->face->underline_thickness, fd->face->size->metrics.y_scale) / 64.0) / fd->oversampling * fd->scale_color_font;

			fd->os2 = (TT_OS2 *)FT_Get_Sfnt_Table(fd->face, FT_SFNT_OS2);

			fd->hb_handle = hb_ft_font_create(fd->face, nullptr);
			if (fd->hb_handle == nullptr) {
				memdelete(fd);
				ERR_FAIL_V_MSG(nullptr, "Error loading HarfBuzz font object.");
			}
			// Write variations.
			if (fd->face->face_flags & FT_FACE_FLAG_MULTIPLE_MASTERS) {
				FT_MM_Var *amaster;

				FT_Get_MM_Var(fd->face, &amaster);

				Vector<hb_variation_t> hb_vars;
				Vector<FT_Fixed> coords;
				coords.resize(amaster->num_axis);

				FT_Get_Var_Design_Coordinates(fd->face, coords.size(), coords.ptrw());

				for (FT_UInt i = 0; i < amaster->num_axis; i++) {
					hb_variation_t var;
					// Reset to default.
					var.tag = amaster->axis[i].tag;
					var.value = (double)amaster->axis[i].def / 65536.f;
					coords.write[i] = amaster->axis[i].def;

					if (p_var_id.variations.has(var.tag)) {
						var.value = p_var_id.variations[var.tag];
						coords.write[i] = CLAMP(p_var_id.variations[var.tag] * 65536.f, amaster->axis[i].minimum, amaster->axis[i].maximum);
					}
					hb_vars.push_back(var);
				}

				FT_Set_Var_Design_Coordinates(fd->face, coords.size(), coords.ptrw());
				hb_font_set_variations(fd->hb_handle, hb_vars.is_empty() ? nullptr : &hb_vars[0], hb_vars.size());

				FT_Done_MM_Var(library, amaster);
			}
#else
			ERR_FAIL_V_MSG(nullptr, "Compiled without FreeType support!");
#endif
		}
		variation_cache[id] = fd;
	}

	return fd;
}

void FontDataAdvanced::add_to_cache(const Map<int32_t, double> &p_var_id, int p_size, int p_outline_size) {
	_THREAD_SAFE_METHOD_

	VariationKey var_id;
	var_id.variations = p_var_id;
	var_id.update_key();

	SizeKey size_id;
	if (p_size <= 0) {
		size_id.size = base_size.size;
	} else {
		size_id.size = p_size;
	}
	size_id.outline_size = p_outline_size;

	get_cache_data(var_id, size_id);
}

void FontDataAdvanced::clear_cache(bool p_force) {
	_THREAD_SAFE_METHOD_

	print_verbose("Clear cache...");

	if (!p_force && font_type == FONT_BITMAP) {
		return;
	}

	for (Map<VariationKey, Map<SizeKey, Data *>>::Element *E = cache.front(); E; E = E->next()) {
		Map<SizeKey, Data *> &size_cache = E->get();
		for (Map<SizeKey, Data *>::Element *F = size_cache.front(); F; F = F->next()) {
			memdelete(F->get());
		}
		size_cache.clear();
	}
	cache.clear();
}

Error FontDataAdvanced::save_cache(const String &p_path, uint8_t p_flags, List<String> *r_gen_files) const {
	ERR_FAIL_COND_V_MSG(font_type == FONT_NONE, ERR_CANT_CREATE, "Font is not loaded.");

	FileAccess *f = FileAccess::open(p_path + ".fontdata", FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(!f, ERR_CANT_CREATE, "Cannot create file in path '" + p_path + ".fontdata'.");

	print_verbose("Saving font resource: " + p_path);

	bool save_as_bmp = ((p_flags & TextServer::FONT_CACHE_FLAGS_CONVERT_TO_BITMAP) == TextServer::FONT_CACHE_FLAGS_CONVERT_TO_BITMAP);

	// Save GDFT cache.
	f->store_32(MAGIC('G', 'D', 'F', 'T'));
	f->store_8(1); // version major
	f->store_8(0); // version minor

	if (save_as_bmp || font_type == FONT_BITMAP) {
		f->store_32(MAGIC('B', 'M', 'P', '0'));
	} else {
		f->store_32(MAGIC('D', 'Y', 'N', '0'));
	}

	if (r_gen_files) {
		r_gen_files->push_back(p_path + ".fontdata");
	}

	if (save_as_bmp) {
		f->store_32(0); // Strip raw dynamic font data.
	} else {
		Vector<uint8_t> rawdata;
		rawdata.resize(Compression::get_max_compressed_buffer_size(font_mem_size, Compression::MODE_ZSTD));
		int buf_size = Compression::compress(rawdata.ptrw(), font_mem, font_mem_size, Compression::MODE_ZSTD);
		f->store_32(font_mem_size); // Raw data, real size.
		if (font_mem_size > 0) {
			f->store_32(buf_size); // Raw data, compressed size.
			f->store_buffer(rawdata.ptr(), buf_size);
		}
	}

	print_verbose("  Raw data size: " + itos(font_mem_size));

	f->store_8((uint8_t)hinting);
	f->store_8((uint8_t)msdf);
	f->store_8((uint8_t)force_autohinter);
	f->store_8((uint8_t)antialiased);
	f->store_8((uint8_t)spacing_glyph);
	f->store_8((uint8_t)spacing_space);
	f->store_float((uint8_t)oversampling);

	f->store_32(lang_support_overrides.size());
	for (const Map<String, bool>::Element *F = lang_support_overrides.front(); F; F = F->next()) {
		f->store_pascal_string(F->key());
		f->store_8((uint8_t)F->value());
	}

	f->store_32(script_support_overrides.size());
	for (const Map<String, bool>::Element *F = script_support_overrides.front(); F; F = F->next()) {
		f->store_pascal_string(F->key());
		f->store_8((uint8_t)F->value());
	}

	f->store_float(rect_margin);
	f->store_float(msdf_margin);

	f->store_32(base_variation.variations.size());
	for (const Map<int32_t, double>::Element *F = base_variation.variations.front(); F; F = F->next()) {
		f->store_32(F->key());
		f->store_double(F->get());
	}
	f->store_16(base_size.size);

	if (save_as_bmp) {
		f->store_32(MAX(1, cache.size()));
	} else {
		f->store_32(cache.size());
	}
	print_verbose("  Variation cache records: " + itos(cache.size()));

	for (const Map<VariationKey, Map<SizeKey, Data *>>::Element *E = cache.front(); E; E = E->next()) {
		if (save_as_bmp && (E->key().key != base_variation.key)) {
			continue;
		}
		const Map<SizeKey, Data *> &size_cache = E->get();

		f->store_32(E->key().variations.size());
		for (const Map<int32_t, double>::Element *F = E->key().variations.front(); F; F = F->next()) {
			f->store_32(F->key());
			f->store_double(F->get());
		}

		if (save_as_bmp) {
			f->store_32(MAX(1, size_cache.size()));
		} else {
			f->store_32(size_cache.size());
		}
		print_verbose("    Size cache records: " + itos(size_cache.size()));

		for (const Map<SizeKey, Data *>::Element *F = size_cache.front(); F; F = F->next()) {
			if (save_as_bmp && (F->key().key != base_size.key) && (!msdf)) {
				continue;
			}
			f->store_32(F->key().key);

			const Data *fd = F->get();
			f->store_float(fd->ascent);
			f->store_float(fd->descent);
			f->store_float(fd->height);
			f->store_float(fd->underline_position);
			f->store_float(fd->underline_thickness);

			f->store_32(fd->textures.size());
			for (int i = 0; i < fd->textures.size(); i++) {
				f->store_32(fd->textures[i].texture_w);
				f->store_32(fd->textures[i].texture_h);
				f->store_32(fd->textures[i].offsets.size());
				for (int j = 0; j < fd->textures[i].offsets.size(); j++) {
					f->store_16(fd->textures[i].offsets[j]);
				}
				Ref<Image> img = memnew(Image(fd->textures[i].texture_w, fd->textures[i].texture_h, 0, fd->textures[i].texture->get_format(), fd->textures[i].imgdata));
				String save_name = p_path + "." + String::num_uint64(E->key().key, 16, false) + "." + String::num_uint64(F->key().key, 16, false) + "." + itos(i) + ".png";
				f->store_pascal_string(save_name);

				print_verbose("      Saving texture: " + itos(i) + " of " + itos(fd->textures.size()) + " to " + save_name);
				img->save_png(save_name);

				if (r_gen_files) {
					r_gen_files->push_back(save_name);
				}
			}

			f->store_32(fd->glyph_map.size());
			print_verbose("    Saving glyph map");
			for (const uint32_t *key = fd->glyph_map.next(nullptr); key; key = fd->glyph_map.next(key)) {
				const Glyph &c = fd->glyph_map[*key];
				if (save_as_bmp && (font_type == FONT_DYNAMIC)) {
#ifdef MODULE_FREETYPE_ENABLED
					FT_ULong charcode;
					FT_UInt gindex;
					charcode = FT_Get_First_Char(fd->face, &gindex);
					while (gindex != 0 && gindex != *key) {
						charcode = FT_Get_Next_Char(fd->face, charcode, &gindex);
					}
					f->store_32(charcode);
#else
					ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Compiled without FreeType support!");
#endif
				} else {
					f->store_32(*key);
				}
				f->store_8((uint8_t)c.found);
				f->store_16(c.texture_idx);
				f->store_float(c.rect.position.x);
				f->store_float(c.rect.position.y);
				f->store_float(c.rect.size.x);
				f->store_float(c.rect.size.y);
				f->store_float(c.rect_uv.position.x);
				f->store_float(c.rect_uv.position.y);
				f->store_float(c.rect_uv.size.x);
				f->store_float(c.rect_uv.size.y);
				f->store_float(c.advance.x);
				f->store_float(c.advance.y);
			}
			print_verbose("    Saving kerning map");
			if (save_as_bmp && (font_type == FONT_DYNAMIC)) {
#ifdef MODULE_FREETYPE_ENABLED
				Map<KerningPairKey, int> gen_kerning;

				for (const uint32_t *key_a = fd->glyph_map.next(nullptr); key_a; key_a = fd->glyph_map.next(key_a)) {
					for (const uint32_t *key_b = fd->glyph_map.next(nullptr); key_b; key_b = fd->glyph_map.next(key_b)) {
						KerningPairKey kp;
						kp.A = *key_a;
						kp.B = *key_b;
						FT_Vector delta;
						FT_Get_Kerning(fd->face, *key_a, *key_b, FT_KERNING_DEFAULT, &delta);
						if (delta.x != 0) {
							gen_kerning[kp] = -delta.x;
						}
					}
				}

				f->store_32(gen_kerning.size()); // Kerning, generate from dynamic font.
				for (const Map<KerningPairKey, int>::Element *G = gen_kerning.front(); G; G = G->next()) {
					f->store_32(G->key().A);
					f->store_32(G->key().B);
					f->store_32(G->get());
				}
#else
				ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Compiled without FreeType support!");
#endif
			} else {
				f->store_32(fd->kerning_map.size()); // Kerning, for bitmap fonts only.
				for (const Map<KerningPairKey, int>::Element *G = fd->kerning_map.front(); G; G = G->next()) {
					f->store_32(G->key().A);
					f->store_32(G->key().B);
					f->store_32(G->get());
				}
			}
		}
	}
	print_verbose("DONE");
	memdelete(f);

	return OK;
}

Error FontDataAdvanced::_load_bmp(const String &p_path, FileAccess *p_f, int p_base_size) {
	font_type = FONT_BITMAP;

	print_verbose("Loading BMP font: " + p_path);

	p_f->seek(0);
	hinting = TextServer::HINTING_NONE;
	force_autohinter = false;

	SizeKey id;
	Data *fd = memnew(Data);
	while (true) {
		String line = p_f->get_line();

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
				base_size.size = keys["size"].to_int();
			}
		} else if (type == "common") {
			if (keys.has("lineHeight")) {
				fd->height = keys["lineHeight"].to_int();
			}
			if (keys.has("base")) {
				fd->ascent = keys["base"].to_int();
			}
		} else if (type == "page") {
			if (keys.has("file")) {
				String base_dir = p_path.get_base_dir();
				String file = base_dir.plus_file(keys["file"]);
				if (RenderingServer::get_singleton() != nullptr) {
					FontTexture tx;
					Ref<StreamTexture2D> stex = ResourceLoader::load(file);
					ERR_FAIL_COND_V_MSG(stex.is_null(), ERR_FILE_CANT_READ, "Can't load font texture: '" + file + "'.");

					Ref<Image> img = stex->get_image();

					tx.texture.instantiate();
					tx.texture->create_from_image(img);
					tx.imgdata = img->get_data();
					tx.texture_w = img->get_width();
					tx.texture_h = img->get_height();
					ERR_FAIL_COND_V_MSG(tx.texture.is_null(), ERR_FILE_CANT_READ, "Can't load font texture: '" + file + "'.");
					fd->textures.push_back(tx);
				}
			}
		} else if (type == "char") {
			Glyph c;
			char32_t idx = 0;
			if (keys.has("id")) {
				idx = keys["id"].to_int();
			}
			if (keys.has("x")) {
				c.rect_uv.position.x = keys["x"].to_int();
			}
			if (keys.has("y")) {
				c.rect_uv.position.y = keys["y"].to_int();
			}
			if (keys.has("width")) {
				c.rect_uv.size.width = keys["width"].to_int();
				c.rect.size.width = keys["width"].to_int();
			}
			if (keys.has("height")) {
				c.rect_uv.size.height = keys["height"].to_int();
				c.rect.size.height = keys["height"].to_int();
			}
			if (keys.has("xoffset")) {
				c.rect.position.x = keys["xoffset"].to_int();
			}
			if (keys.has("yoffset")) {
				c.rect.position.y = keys["yoffset"].to_int() - fd->ascent;
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
			c.found = true;
			fd->glyph_map[idx] = c;
		} else if (type == "kerning") {
			KerningPairKey kpk;
			float k = 0.f;
			if (keys.has("first")) {
				kpk.A = keys["first"].to_int();
			}
			if (keys.has("second")) {
				kpk.B = keys["second"].to_int();
			}
			if (keys.has("amount")) {
				k = keys["amount"].to_int();
			}
			fd->kerning_map[kpk] = k;
		}

		if (p_f->eof_reached()) {
			break;
		}
	}
	fd->descent = fd->height - fd->ascent;
	if (base_size.size <= 0) {
		base_size.size = fd->height;
	}
	fd->hb_handle = hb_bmp_font_create(this, nullptr);
	if (fd->hb_handle == nullptr) {
		memdelete(fd);
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Error loading HarfBuzz font object.");
	}
	fd->size_id = base_size;
	fd->variation_id = base_variation;

	Map<SizeKey, Data *> &variation_cache = cache[base_variation];
	variation_cache[id] = fd;

	return OK;
}

Error FontDataAdvanced::_load_ttf(const String &p_path, FileAccess *p_f, int p_base_size) {
	font_type = FONT_DYNAMIC;

	print_verbose("Loading FT font: " + p_path);

	base_variation = VariationKey();
	base_size.size = p_base_size;
	if (base_size.size <= 0) {
		base_size.size = 16;
	}

	p_f->seek(0);
	font_mem_size = p_f->get_length();
	font_mem_cache.resize(font_mem_size);
	if (font_mem_size > 0) {
		p_f->get_buffer(font_mem_cache.ptrw(), font_mem_size);
	}
	font_mem = font_mem_cache.ptr();

	if (get_cache_data(base_variation, base_size) == nullptr) {
		font_type = FONT_NONE;
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Failed to create base size font data.");
	}
	return OK;
}

Error FontDataAdvanced::_load_cache(const String &p_path, FileAccess *p_f, int p_base_size) {
	print_verbose("Loading font resource: " + p_path);

	uint8_t ver_ma = p_f->get_8(); // version major
	uint8_t ver_mi = p_f->get_8(); // version minor
	if ((ver_ma != 1) || (ver_mi != 0)) {
		font_type = FONT_NONE;
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Unsupported format version.");
	}

	uint32_t type_tag = p_f->get_32();
	if (type_tag == MAGIC('D', 'Y', 'N', '0')) {
		font_type = FONT_DYNAMIC;
	} else if (type_tag == MAGIC('B', 'M', 'P', '0')) {
		font_type = FONT_BITMAP;
	} else {
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Unknown font cache format.");
	}

	font_mem_size = p_f->get_32();
	print_verbose("  Raw data size: " + itos(font_mem_size));
	font_mem_cache.resize(font_mem_size);

	if (font_mem_size > 0) {
		int comp_size = p_f->get_32();
		Vector<uint8_t> rawdata;
		rawdata.resize(comp_size);
		p_f->get_buffer(rawdata.ptrw(), comp_size);
		uint64_t read_size = (uint64_t)Compression::decompress(font_mem_cache.ptrw(), font_mem_size, rawdata.ptr(), comp_size, Compression::MODE_ZSTD);
		if (read_size != font_mem_size) {
			font_type = FONT_NONE;
			ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Failed to decompress font data.");
		}
	}
	font_mem = font_mem_cache.ptr();

	hinting = (TextServer::Hinting)p_f->get_8();
	msdf = (bool)p_f->get_8();
	force_autohinter = (bool)p_f->get_8();
	antialiased = (bool)p_f->get_8();
	spacing_glyph = p_f->get_8();
	spacing_space = p_f->get_8();
	oversampling = p_f->get_float();

	int32_t lang_ov_size = p_f->get_32();
	for (int32_t i = 0; i < lang_ov_size; i++) {
		String key = p_f->get_pascal_string();
		bool val = (bool)p_f->get_8();
		lang_support_overrides[key] = val;
	}

	int32_t script_ov_size = p_f->get_32();
	for (int32_t i = 0; i < script_ov_size; i++) {
		String key = p_f->get_pascal_string();
		bool val = (bool)p_f->get_8();
		script_support_overrides[key] = val;
	}

	rect_margin = p_f->get_float();
	msdf_margin = p_f->get_float();

	base_variation = VariationKey();
	int32_t base_variation_size = p_f->get_32();
	for (int32_t i = 0; i < base_variation_size; i++) {
		uint32_t tag = p_f->get_32();
		double value = p_f->get_double();
		base_variation.variations[tag] = value;
	}
	base_variation.update_key();
	base_size.size = p_f->get_16();
	if (base_size.size <= 0) {
		base_size.size = 16;
	}

	if (get_cache_data(base_variation, base_size) == nullptr) {
		font_type = FONT_NONE;
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Failed to create base size font data.");
	}

	int32_t var_cache_size = p_f->get_32();
	print_verbose("  Variation cache records: " + itos(var_cache_size));
	for (int32_t i = 0; i < var_cache_size; i++) {
		VariationKey vkey;
		int32_t variation_size = p_f->get_32();
		for (int32_t j = 0; j < variation_size; j++) {
			uint32_t tag = p_f->get_32();
			double value = p_f->get_double();
			vkey.variations[tag] = value;
		}
		vkey.update_key();

		Map<SizeKey, Data *> &size_cache = cache[vkey];

		int32_t size_cache_size = p_f->get_32();
		print_verbose("    Size cache records: " + itos(size_cache_size));
		for (int32_t j = 0; j < size_cache_size; j++) {
			SizeKey skey;
			skey.key = p_f->get_32();

			Data *fd = get_cache_data(vkey, skey);
			fd->ascent = p_f->get_float();
			fd->descent = p_f->get_float();
			fd->height = p_f->get_float();
			fd->underline_position = p_f->get_float();
			fd->underline_thickness = p_f->get_float();

			print_verbose("    FD: " + String::num_int64((uint64_t)fd, 16) + " " + String::num_int64(vkey.key, 16) + " " + String::num_int64(skey.key, 16));

			int32_t txt_count = p_f->get_32();
			for (int32_t k = 0; k < txt_count; k++) {
				FontTexture tx;
				tx.texture_w = p_f->get_32();
				tx.texture_h = p_f->get_32();
				int32_t off_count = p_f->get_32();
				for (int32_t l = 0; l < off_count; l++) {
					tx.offsets.push_back(p_f->get_16());
				}
				// Load textures.
				Ref<Image> img;
				img.instantiate();
				String load_name = p_f->get_pascal_string();
				print_verbose("      Loading texture: " + itos(k) + " of " + itos(txt_count) + " from file " + load_name);

				img->load(load_name);
				tx.imgdata = img->get_data();
				tx.texture.instantiate();
				tx.texture->create_from_image(img);

				fd->textures.push_back(tx);
			}

			int32_t glyph_count = p_f->get_32();
			print_verbose("    Loading glyph map: " + itos(glyph_count));
			for (int32_t k = 0; k < glyph_count; k++) {
				uint32_t ckey = p_f->get_32();

				Glyph c;
				c.found = p_f->get_8();
				c.texture_idx = p_f->get_16();
				c.rect.position.x = p_f->get_float();
				c.rect.position.y = p_f->get_float();
				c.rect.size.x = p_f->get_float();
				c.rect.size.y = p_f->get_float();
				c.rect_uv.position.x = p_f->get_float();
				c.rect_uv.position.y = p_f->get_float();
				c.rect_uv.size.x = p_f->get_float();
				c.rect_uv.size.y = p_f->get_float();
				c.advance.x = p_f->get_float();
				c.advance.y = p_f->get_float();

				fd->glyph_map[ckey] = c;
			}
			int32_t kern_count = p_f->get_32();
			print_verbose("    Loading kerning map: " + itos(kern_count));
			for (int32_t k = 0; k < kern_count; k++) {
				KerningPairKey kpk;
				kpk.A = p_f->get_32();
				kpk.B = p_f->get_32();
				fd->kerning_map[kpk] = p_f->get_32();
			}

			size_cache[skey] = fd;
		}
	}
	print_verbose("DONE");

	return OK;
}

Error FontDataAdvanced::_load(const String &p_path, FileAccess *p_f, int p_base_size) {
	_THREAD_SAFE_METHOD_

	// Clear current font.
	clear_cache();
	font_type = FONT_NONE;
	font_mem_size = 0;
	font_mem_cache.clear();
	base_size.size = p_base_size;
	base_variation = VariationKey();
	spacing_glyph = 0;
	spacing_space = 0;
	lang_support_overrides.clear();
	script_support_overrides.clear();

	// Detect new font type.
	uint32_t magic = p_f->get_32();
	switch (magic) {
		case MAGIC('G', 'D', 'F', 'T'): {
			return _load_cache(p_path, p_f, p_base_size);
		} break;
		case MAGIC('O', 'T', 'T', 'O'):
		case MAGIC('1', 0, 0, 0):
		case MAGIC('t', 'y', 'p', '1'):
		case MAGIC('w', 'O', 'F', 'F'):
		case MAGIC('w', 'O', 'F', '2'):
		case MAGIC(0, 1, 0, 0):
		case MAGIC('t', 'r', 'u', 'e'): {
			return _load_ttf(p_path, p_f, p_base_size);
		} break;
		case MAGIC('p', 'a', 'g', 'e'):
		case MAGIC('c', 'o', 'm', 'm'):
		case MAGIC('i', 'n', 'f', 'o'): {
			return _load_bmp(p_path, p_f, p_base_size);
		} break;
		default: {
			ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Unknown font format: '" + String::num_uint64(magic, 16, false) + "'.");
		} break;
	}

	return ERR_CANT_CREATE;
}

Error FontDataAdvanced::load_from_file(const String &p_path, int p_base_size) {
	_THREAD_SAFE_METHOD_

	print_verbose("FontData load from file: " + p_path);

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	if (f == nullptr) {
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Cannot open font from file '" + p_path + "'.");
	}
	Error res = _load(p_path, f, p_base_size);
	memdelete(f);
	return res;
}

Error FontDataAdvanced::load_from_memory(const uint8_t *p_data, size_t p_size, int p_base_size) {
	_THREAD_SAFE_METHOD_

	print_verbose("FontData load from mem: " + itos(p_size) + " @ " + itos((uint64_t)p_data));

	FileAccessMemory *f = memnew(FileAccessMemory);
	if ((f == nullptr) || (f->open_custom(p_data, p_size) != OK)) {
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Cannot open font from memory address '0x" + String::num_uint64((uint64_t)p_data, 16, true) + "'.");
	}
	Error res = _load("", f, p_base_size);
	memdelete(f);
	return res;
}

Error FontDataAdvanced::bitmap_new(float p_height, float p_ascent, int p_base_size) {
	_THREAD_SAFE_METHOD_

	print_verbose("FontData new bitmap");

	clear_cache();
	font_type = FONT_BITMAP;

	hinting = TextServer::HINTING_NONE;
	force_autohinter = false;
	spacing_glyph = 0;
	spacing_space = 0;
	lang_support_overrides.clear();
	script_support_overrides.clear();

	Data *fd = memnew(Data);
	fd->height = p_height;
	fd->ascent = p_ascent;
	fd->descent = fd->height - fd->ascent;
	base_size.size = p_base_size;

	if (base_size.size <= 0) {
		base_size.size = fd->height;
	}

	fd->hb_handle = hb_bmp_font_create(this, nullptr);
	if (fd->hb_handle == nullptr) {
		memdelete(fd);
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Error loading HarfBuzz font object.");
	}
	fd->size_id = base_size;
	fd->variation_id = base_variation;

	Map<SizeKey, Data *> &variation_cache = cache[base_variation];
	variation_cache[base_size] = fd;

	return OK;
}

void FontDataAdvanced::bitmap_add_texture(const Ref<Texture2D> &p_texture) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(font_type != FONT_BITMAP);
	ERR_FAIL_COND_MSG(p_texture.is_null(), "It's not a reference to a valid Texture object.");
	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, base_size);
	ERR_FAIL_COND(fd == nullptr);

	Ref<Image> img = p_texture->get_image();
	ERR_FAIL_COND(img.is_null());

	FontTexture tx;
	tx.imgdata = img->get_data();
	tx.texture.instantiate();
	tx.texture->create_from_image(img);

	fd->textures.push_back(tx);
}

void FontDataAdvanced::bitmap_add_char(char32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(font_type != FONT_BITMAP);
	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, base_size);
	ERR_FAIL_COND(fd == nullptr);
	ERR_FAIL_COND(p_texture_idx < 0 || p_texture_idx >= fd->textures.size());

	Glyph chr;
	chr.found = true;
	chr.texture_idx = p_texture_idx;
	chr.rect_uv = p_rect;
	if (p_advance < 0) {
		chr.advance.x = chr.rect.size.x;
	} else {
		chr.advance.x = p_advance;
	}
	chr.rect.position = p_align;
	chr.rect.position.y -= fd->ascent;
	chr.rect.size = chr.rect_uv.size;
	fd->glyph_map[p_char] = chr;
}

void FontDataAdvanced::bitmap_add_kerning_pair(char32_t p_A, char32_t p_B, int p_kerning) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(font_type != FONT_BITMAP);
	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, base_size);
	ERR_FAIL_COND(fd == nullptr);

	KerningPairKey kpk;
	kpk.A = p_A;
	kpk.B = p_B;

	if (p_kerning == 0 && fd->kerning_map.has(kpk)) {
		fd->kerning_map.erase(kpk);
	} else {
		fd->kerning_map[kpk] = p_kerning;
	}
}

void FontDataAdvanced::preload_range(uint32_t p_start, uint32_t p_end, bool p_glyphs) {
	_THREAD_SAFE_METHOD_
#ifdef MODULE_FREETYPE_ENABLED
	get_cache_data(base_variation, base_size);
	for (Map<VariationKey, Map<SizeKey, Data *>>::Element *E = cache.front(); E; E = E->next()) {
		Map<SizeKey, Data *> &size_cache = E->get();
		for (Map<SizeKey, Data *>::Element *F = size_cache.front(); F; F = F->next()) {
			Data *fd = F->get();
			if (fd) {
				if (p_start == 0 && p_end == 0) { // Preload all glyphs.
					FT_UInt g;
					FT_ULong c = FT_Get_First_Char(fd->face, &g);
					while (g != 0) {
						if (c != 0) {
							update_glyph(fd, g);
						}
						c = FT_Get_Next_Char(fd->face, c, &g);
					}
				} else
					for (uint32_t c = p_start; c <= p_end; c++) { // Preload glyph or char range.
						if (p_glyphs) {
							update_glyph(fd, c);
						} else {
							update_glyph(fd, FT_Get_Char_Index(fd->face, c));
						}
					}
			}
		}
	}
#else
	ERR_FAIL_MSG("Compiled without FreeType support!");
#endif
}

/*************************************************************************/
/* Glpyh Rendering                                                       */
/*************************************************************************/

FontDataAdvanced::TexturePosition FontDataAdvanced::find_texture_pos_for_glyph(FontDataAdvanced::Data *p_data, int p_color_size, Image::Format p_image_format, int p_width, int p_height) {
	TexturePosition ret;
	ret.index = -1;

	int mw = p_width;
	int mh = p_height;

	for (int i = 0; i < p_data->textures.size(); i++) {
		const FontTexture &ct = p_data->textures[i];

		if (RenderingServer::get_singleton() != nullptr) {
			if (ct.texture->get_format() != p_image_format) {
				continue;
			}
		}

		if (mw > ct.texture_w || mh > ct.texture_h) { // Too big for this texture.
			continue;
		}

		ret.y = 0x7FFFFFFF;
		ret.x = 0;

		for (int j = 0; j < ct.texture_w - mw; j++) {
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

		if (ret.y == 0x7FFFFFFF || ret.y + mh > ct.texture_h) {
			continue; // Fail, could not fit it here.
		}

		ret.index = i;
		break;
	}

	if (ret.index == -1) {
		// Could not find texture to fit, create one.
		ret.x = 0;
		ret.y = 0;

		int texsize = MAX(p_data->size_id.size * p_data->oversampling * 8, 256);
		if (mw > texsize) {
			texsize = mw; // Special case, adapt to it?
		}
		if (mh > texsize) {
			texsize = mh; // Special case, adapt to it?
		}

		texsize = next_power_of_2(texsize);

		texsize = MIN(texsize, 4096);

		FontTexture tex;
		tex.texture_w = texsize;
		tex.texture_h = texsize;
		tex.imgdata.resize(texsize * texsize * p_color_size);

		{
			// Zero texture.
			uint8_t *w = tex.imgdata.ptrw();
			ERR_FAIL_COND_V(texsize * texsize * p_color_size > tex.imgdata.size(), ret);
			// Initialize the texture to all-white pixels to prevent artifacts when the
			// font is displayed at a non-default scale with filtering enabled.
			if (p_color_size == 2) {
				for (int i = 0; i < texsize * texsize * p_color_size; i += 2) { // FORMAT_LA8, BW font.
					w[i + 0] = 255;
					w[i + 1] = 0;
				}
			} else if (p_color_size == 4) {
				for (int i = 0; i < texsize * texsize * p_color_size; i += 4) { // FORMAT_RGBA8, Color font, MTSDF.
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
		for (int i = 0; i < texsize; i++) { // Zero offsets.
			tex.offsets.write[i] = 0;
		}

		p_data->textures.push_back(tex);
		ret.index = p_data->textures.size() - 1;
	}

	return ret;
}

#ifdef MODULE_MSDFGEN_ENABLED

struct MSContext {
	msdfgen::Point2 position;
	msdfgen::Shape *shape;
	msdfgen::Contour *contour;
};

static msdfgen::Point2 ft_point2(const FT_Vector &vector) {
	return msdfgen::Point2(vector.x / 60.0f, vector.y / 60.0f);
}

static int ft_move_to(const FT_Vector *to, void *user) {
	MSContext *context = reinterpret_cast<MSContext *>(user);
	if (!(context->contour && context->contour->edges.empty())) {
		context->contour = &context->shape->addContour();
	}
	context->position = ft_point2(*to);
	return 0;
}

static int ft_line_to(const FT_Vector *to, void *user) {
	MSContext *context = reinterpret_cast<MSContext *>(user);
	msdfgen::Point2 endpoint = ft_point2(*to);
	if (endpoint != context->position) {
		context->contour->addEdge(new msdfgen::LinearSegment(context->position, endpoint));
		context->position = endpoint;
	}
	return 0;
}

static int ft_conic_to(const FT_Vector *control, const FT_Vector *to, void *user) {
	MSContext *context = reinterpret_cast<MSContext *>(user);
	context->contour->addEdge(new msdfgen::QuadraticSegment(context->position, ft_point2(*control), ft_point2(*to)));
	context->position = ft_point2(*to);
	return 0;
}

static int ft_cubic_to(const FT_Vector *control1, const FT_Vector *control2, const FT_Vector *to, void *user) {
	MSContext *context = reinterpret_cast<MSContext *>(user);
	context->contour->addEdge(new msdfgen::CubicSegment(context->position, ft_point2(*control1), ft_point2(*control2), ft_point2(*to)));
	context->position = ft_point2(*to);
	return 0;
}

FontDataAdvanced::Glyph FontDataAdvanced::rasterize_msdf(FontDataAdvanced::Data *p_data, FT_Outline *outline, const Vector2 &advance) {
	msdfgen::Shape shape;

	shape.contours.clear();
	shape.inverseYAxis = false;

	MSContext context = {};
	context.shape = &shape;
	FT_Outline_Funcs ft_functions;
	ft_functions.move_to = &ft_move_to;
	ft_functions.line_to = &ft_line_to;
	ft_functions.conic_to = &ft_conic_to;
	ft_functions.cubic_to = &ft_cubic_to;
	ft_functions.shift = 0;
	ft_functions.delta = 0;

	int error = FT_Outline_Decompose(outline, &ft_functions, &context);
	ERR_FAIL_COND_V_MSG(error, Glyph(), "FreeType: Outline decomposition error: '" + String(FT_Error_String(error)) + "'.");
	if (!shape.contours.empty() && shape.contours.back().edges.empty()) {
		shape.contours.pop_back();
	}

	if (FT_Outline_Get_Orientation(outline) == 1) {
		for (int i = 0; i < (int)shape.contours.size(); ++i) {
			shape.contours[i].reverse();
		}
	}

	shape.inverseYAxis = true;
	shape.normalize();

	msdfgen::Shape::Bounds bounds = shape.getBounds(msdf_margin);

	Glyph chr;
	chr.found = true;
	chr.advance = advance.round();

	if (shape.validate() && shape.contours.size() > 0) {
		int w = (bounds.r - bounds.l);
		int h = (bounds.t - bounds.b);

		int mw = w + rect_margin * 2;
		int mh = h + rect_margin * 2;

		ERR_FAIL_COND_V(mw > 4096, Glyph());
		ERR_FAIL_COND_V(mh > 4096, Glyph());

		TexturePosition tex_pos = find_texture_pos_for_glyph(p_data, 4, Image::FORMAT_RGBA8, mw, mh);
		ERR_FAIL_COND_V(tex_pos.index < 0, Glyph());
		FontTexture &tex = p_data->textures.write[tex_pos.index];

		edgeColoringSimple(shape, 3.0); // Max. angle.
		msdfgen::Bitmap<float, 4> image(w, h); // Texture size.
		msdfgen::generateMTSDF(image, shape, msdf_margin, 1.0, msdfgen::Vector2(-bounds.l, -bounds.b)); // Range, scale, translation.

		{
			uint8_t *wr = tex.imgdata.ptrw();

			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					int ofs = ((i + tex_pos.y + rect_margin) * tex.texture_w + j + tex_pos.x + rect_margin) * 4;
					ERR_FAIL_COND_V(ofs >= tex.imgdata.size(), Glyph());
					wr[ofs + 0] = (uint8_t)(CLAMP(image(j, i)[0] * 256.f, 0.f, 255.f));
					wr[ofs + 1] = (uint8_t)(CLAMP(image(j, i)[1] * 256.f, 0.f, 255.f));
					wr[ofs + 2] = (uint8_t)(CLAMP(image(j, i)[2] * 256.f, 0.f, 255.f));
					wr[ofs + 3] = (uint8_t)(CLAMP(image(j, i)[3] * 256.f, 0.f, 255.f));
				}
			}
		}

		// Blit to image and texture.
		{
			if (RenderingServer::get_singleton() != nullptr) {
				Ref<Image> img = memnew(Image(tex.texture_w, tex.texture_h, 0, Image::FORMAT_RGBA8, tex.imgdata));
				if (tex.texture.is_null()) {
					tex.texture.instantiate();
					tex.texture->create_from_image(img);
				} else {
					tex.texture->update(img);
				}
			}
		}

		// Update height array.
		for (int k = tex_pos.x; k < tex_pos.x + mw; k++) {
			tex.offsets.write[k] = tex_pos.y + mh;
		}

		chr.texture_idx = tex_pos.index;

		chr.rect_uv = Rect2(tex_pos.x + rect_margin, tex_pos.y + rect_margin, w, h);
		chr.rect.position = Vector2(bounds.l, -bounds.t);
		chr.rect.size = chr.rect_uv.size;
	}
	return chr;
}
#endif

#ifdef MODULE_FREETYPE_ENABLED
FontDataAdvanced::Glyph FontDataAdvanced::rasterize_bitmap(FontDataAdvanced::Data *p_data, FT_Bitmap bitmap, int yofs, int xofs, const Vector2 &advance) {
	int w = bitmap.width;
	int h = bitmap.rows;

	int mw = w + rect_margin * 2;
	int mh = h + rect_margin * 2;

	ERR_FAIL_COND_V(mw > 4096, Glyph());
	ERR_FAIL_COND_V(mh > 4096, Glyph());

	int color_size = bitmap.pixel_mode == FT_PIXEL_MODE_BGRA ? 4 : 2;
	Image::Format require_format = color_size == 4 ? Image::FORMAT_RGBA8 : Image::FORMAT_LA8;

	TexturePosition tex_pos = find_texture_pos_for_glyph(p_data, color_size, require_format, mw, mh);
	ERR_FAIL_COND_V(tex_pos.index < 0, Glyph());

	// Fit character in char texture.

	FontTexture &tex = p_data->textures.write[tex_pos.index];

	{
		uint8_t *wr = tex.imgdata.ptrw();

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				int ofs = ((i + tex_pos.y + rect_margin) * tex.texture_w + j + tex_pos.x + rect_margin) * color_size;
				ERR_FAIL_COND_V(ofs >= tex.imgdata.size(), Glyph());
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
					default:
						ERR_FAIL_V_MSG(Glyph(), "Font uses unsupported pixel format: " + itos(bitmap.pixel_mode) + ".");
						break;
				}
			}
		}
	}

	// Blit to image and texture.
	{
		if (RenderingServer::get_singleton() != nullptr) {
			Ref<Image> img = memnew(Image(tex.texture_w, tex.texture_h, 0, require_format, tex.imgdata));

			if (tex.texture.is_null()) {
				tex.texture.instantiate();
				tex.texture->create_from_image(img);
			} else {
				tex.texture->update(img);
			}
		}
	}

	// Update height array.
	for (int k = tex_pos.x; k < tex_pos.x + mw; k++) {
		tex.offsets.write[k] = tex_pos.y + mh;
	}

	Glyph chr;
	chr.advance = (advance * p_data->scale_color_font / p_data->oversampling).round();
	chr.texture_idx = tex_pos.index;
	chr.found = true;

	chr.rect_uv = Rect2(tex_pos.x + rect_margin, tex_pos.y + rect_margin, w, h);
	chr.rect.position = (Vector2(xofs, -yofs) * p_data->scale_color_font / p_data->oversampling).round();
	chr.rect.size = chr.rect_uv.size * p_data->scale_color_font / p_data->oversampling;
	return chr;
}
#endif

void FontDataAdvanced::update_glyph(FontDataAdvanced::Data *p_fd, uint32_t p_index) {
	if ((p_index == 0) || (p_fd == nullptr) || p_fd->glyph_map.has(p_index) || font_type == FONT_NONE) {
		return;
	}

#ifdef MODULE_FREETYPE_ENABLED
	Glyph gl;
	if (font_type == FONT_DYNAMIC) {
		FT_Int32 flags = FT_LOAD_DEFAULT;

		bool outline = p_fd->size_id.outline_size > 0;
		switch (hinting) {
			case TextServer::HINTING_NONE:
				flags |= FT_LOAD_NO_HINTING;
				break;
			case TextServer::HINTING_LIGHT:
				flags |= FT_LOAD_TARGET_LIGHT;
				break;
			default:
				flags |= FT_LOAD_TARGET_NORMAL;
				break;
		}
		if (force_autohinter) {
			flags |= FT_LOAD_FORCE_AUTOHINT;
		}
		if (outline) {
			flags |= FT_LOAD_NO_BITMAP;
		} else if (FT_HAS_COLOR(p_fd->face)) {
			flags |= FT_LOAD_COLOR;
		}

		FT_Fixed v, h;
		FT_Get_Advance(p_fd->face, p_index, flags, &h);
		FT_Get_Advance(p_fd->face, p_index, flags | FT_LOAD_VERTICAL_LAYOUT, &v);

		int error = FT_Load_Glyph(p_fd->face, p_index, flags);
		if (error) {
			p_fd->glyph_map[p_index] = gl;
			return;
		}

#ifdef ENABLED_MSDF_STROKER_OUTLINES
		if (!outline || msdf) {
#else
		if (!outline) {
#endif
			if (!msdf) {
				error = FT_Render_Glyph(p_fd->face->glyph, antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO);
			}
			FT_GlyphSlot slot = p_fd->face->glyph;
			if (!error) {
				if (msdf) {
#ifdef MODULE_MSDFGEN_ENABLED
					gl = rasterize_msdf(p_fd, &slot->outline, Vector2((h + (1 << 9)) >> 10, (v + (1 << 9)) >> 10) / 64.0);
#else
					ERR_PRINT("Compiled without MSDFGEN support!");
#endif
				} else {
					gl = rasterize_bitmap(p_fd, slot->bitmap, slot->bitmap_top, slot->bitmap_left, Vector2((h + (1 << 9)) >> 10, (v + (1 << 9)) >> 10) / 64.0);
				}
			}
		} else {
			FT_Stroker stroker;
			if (FT_Stroker_New(library, &stroker) != 0) {
				p_fd->glyph_map[p_index] = gl;
				return;
			}

			FT_Stroker_Set(stroker, (int)(p_fd->size_id.outline_size * p_fd->oversampling * 16.0), FT_STROKER_LINECAP_BUTT, FT_STROKER_LINEJOIN_ROUND, 0);
			FT_Glyph glyph;
			FT_BitmapGlyph glyph_bitmap;

			if (FT_Get_Glyph(p_fd->face->glyph, &glyph) != 0) {
				goto cleanup_stroker;
			}
			if (FT_Glyph_Stroke(&glyph, stroker, 1) != 0) {
				goto cleanup_glyph;
			}
			if (FT_Glyph_To_Bitmap(&glyph, antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO, nullptr, 1) != 0) {
				goto cleanup_glyph;
			}
			glyph_bitmap = (FT_BitmapGlyph)glyph;
			gl = rasterize_bitmap(p_fd, glyph_bitmap->bitmap, glyph_bitmap->top, glyph_bitmap->left, Vector2());

		cleanup_glyph:
			FT_Done_Glyph(glyph);
		cleanup_stroker:
			FT_Stroker_Done(stroker);
		}
	}
	p_fd->glyph_map[p_index] = gl;
#else
	ERR_FAIL_MSG("Compiled without FreeType support!");
#endif
}

/*************************************************************************/
/* Font API                                                              */
/*************************************************************************/

Dictionary FontDataAdvanced::get_variation_list() const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, Dictionary());
	if (font_type != FONT_DYNAMIC) {
		return Dictionary();
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, base_size);
	if (fd == nullptr) {
		return Dictionary();
	}

	Dictionary ret;
#ifdef MODULE_FREETYPE_ENABLED
	// Read variations.
	if (fd->face->face_flags & FT_FACE_FLAG_MULTIPLE_MASTERS) {
		FT_MM_Var *amaster;

		FT_Get_MM_Var(fd->face, &amaster);

		for (FT_UInt i = 0; i < amaster->num_axis; i++) {
			ret[(int32_t)amaster->axis[i].tag] = Vector3i(amaster->axis[i].minimum / 65536, amaster->axis[i].maximum / 65536, amaster->axis[i].def / 65536);
		}

		FT_Done_MM_Var(library, amaster);
	}
	return ret;
#else
	ERR_FAIL_V_MSG(ret, "Compiled without FreeType support!");
#endif
}

Dictionary FontDataAdvanced::get_feature_list() const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, Dictionary());
	if (font_type != FONT_DYNAMIC) {
		return Dictionary();
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, base_size);
	if (fd == nullptr) {
		return Dictionary();
	}

	Dictionary out;
#ifdef MODULE_FREETYPE_ENABLED
	// Read feature flags.
	unsigned int count = hb_ot_layout_table_get_feature_tags(hb_font_get_face(fd->hb_handle), HB_OT_TAG_GSUB, 0, nullptr, nullptr);
	if (count != 0) {
		hb_tag_t *feature_tags = (hb_tag_t *)memalloc(count * sizeof(hb_tag_t));
		hb_ot_layout_table_get_feature_tags(hb_font_get_face(fd->hb_handle), HB_OT_TAG_GSUB, 0, &count, feature_tags);
		for (unsigned int i = 0; i < count; i++) {
			out[feature_tags[i]] = 1;
		}
		memfree(feature_tags);
	}
	count = hb_ot_layout_table_get_feature_tags(hb_font_get_face(fd->hb_handle), HB_OT_TAG_GPOS, 0, nullptr, nullptr);
	if (count != 0) {
		hb_tag_t *feature_tags = (hb_tag_t *)memalloc(count * sizeof(hb_tag_t));
		hb_ot_layout_table_get_feature_tags(hb_font_get_face(fd->hb_handle), HB_OT_TAG_GPOS, 0, &count, feature_tags);
		for (unsigned int i = 0; i < count; i++) {
			out[feature_tags[i]] = 1;
		}
		memfree(feature_tags);
	}
	return out;
#else
	ERR_FAIL_V_MSG(out, "Compiled without FreeType support!");
#endif
}

bool FontDataAdvanced::is_script_supported(uint32_t p_script) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, false);

	if (font_type == FONT_BITMAP) {
		return true;
	}

#ifdef MODULE_FREETYPE_ENABLED
	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, base_size);
	ERR_FAIL_COND_V(fd == nullptr, false);
	unsigned int count = hb_ot_layout_table_get_script_tags(hb_font_get_face(fd->hb_handle), HB_OT_TAG_GSUB, 0, nullptr, nullptr);
	if (count != 0) {
		hb_tag_t *script_tags = (hb_tag_t *)memalloc(count * sizeof(hb_tag_t));
		hb_ot_layout_table_get_script_tags(hb_font_get_face(fd->hb_handle), HB_OT_TAG_GSUB, 0, &count, script_tags);
		for (unsigned int i = 0; i < count; i++) {
			if (p_script == script_tags[i]) {
				memfree(script_tags);
				return true;
			}
		}
		memfree(script_tags);
	}
	count = hb_ot_layout_table_get_script_tags(hb_font_get_face(fd->hb_handle), HB_OT_TAG_GPOS, 0, nullptr, nullptr);
	if (count != 0) {
		hb_tag_t *script_tags = (hb_tag_t *)memalloc(count * sizeof(hb_tag_t));
		hb_ot_layout_table_get_script_tags(hb_font_get_face(fd->hb_handle), HB_OT_TAG_GPOS, 0, &count, script_tags);
		for (unsigned int i = 0; i < count; i++) {
			if (p_script == script_tags[i]) {
				memfree(script_tags);
				return true;
			}
		}
		memfree(script_tags);
	}

	if (!fd->os2) {
		return false;
	}

	switch (p_script) {
		case HB_SCRIPT_COMMON:
			return (fd->os2->ulUnicodeRange1 & 1L << 4) || (fd->os2->ulUnicodeRange1 & 1L << 5) || (fd->os2->ulUnicodeRange1 & 1L << 6) || (fd->os2->ulUnicodeRange1 & 1L << 31) || (fd->os2->ulUnicodeRange2 & 1L << 0) || (fd->os2->ulUnicodeRange2 & 1L << 1) || (fd->os2->ulUnicodeRange2 & 1L << 2) || (fd->os2->ulUnicodeRange2 & 1L << 3) || (fd->os2->ulUnicodeRange2 & 1L << 4) || (fd->os2->ulUnicodeRange2 & 1L << 5) || (fd->os2->ulUnicodeRange2 & 1L << 6) || (fd->os2->ulUnicodeRange2 & 1L << 7) || (fd->os2->ulUnicodeRange2 & 1L << 8) || (fd->os2->ulUnicodeRange2 & 1L << 9) || (fd->os2->ulUnicodeRange2 & 1L << 10) || (fd->os2->ulUnicodeRange2 & 1L << 11) || (fd->os2->ulUnicodeRange2 & 1L << 12) || (fd->os2->ulUnicodeRange2 & 1L << 13) || (fd->os2->ulUnicodeRange2 & 1L << 14) || (fd->os2->ulUnicodeRange2 & 1L << 15) || (fd->os2->ulUnicodeRange2 & 1L << 30) || (fd->os2->ulUnicodeRange3 & 1L << 0) || (fd->os2->ulUnicodeRange3 & 1L << 1) || (fd->os2->ulUnicodeRange3 & 1L << 2) || (fd->os2->ulUnicodeRange3 & 1L << 4) || (fd->os2->ulUnicodeRange3 & 1L << 5) || (fd->os2->ulUnicodeRange3 & 1L << 18) || (fd->os2->ulUnicodeRange3 & 1L << 24) || (fd->os2->ulUnicodeRange3 & 1L << 25) || (fd->os2->ulUnicodeRange3 & 1L << 26) || (fd->os2->ulUnicodeRange3 & 1L << 27) || (fd->os2->ulUnicodeRange3 & 1L << 28) || (fd->os2->ulUnicodeRange4 & 1L << 3) || (fd->os2->ulUnicodeRange4 & 1L << 6) || (fd->os2->ulUnicodeRange4 & 1L << 15) || (fd->os2->ulUnicodeRange4 & 1L << 23) || (fd->os2->ulUnicodeRange4 & 1L << 24) || (fd->os2->ulUnicodeRange4 & 1L << 26);
		case HB_SCRIPT_LATIN:
			return (fd->os2->ulUnicodeRange1 & 1L << 0) || (fd->os2->ulUnicodeRange1 & 1L << 1) || (fd->os2->ulUnicodeRange1 & 1L << 2) || (fd->os2->ulUnicodeRange1 & 1L << 3) || (fd->os2->ulUnicodeRange1 & 1L << 29);
		case HB_SCRIPT_GREEK:
			return (fd->os2->ulUnicodeRange1 & 1L << 7) || (fd->os2->ulUnicodeRange1 & 1L << 30);
		case HB_SCRIPT_COPTIC:
			return (fd->os2->ulUnicodeRange1 & 1L << 8);
		case HB_SCRIPT_CYRILLIC:
			return (fd->os2->ulUnicodeRange1 & 1L << 9);
		case HB_SCRIPT_ARMENIAN:
			return (fd->os2->ulUnicodeRange1 & 1L << 10);
		case HB_SCRIPT_HEBREW:
			return (fd->os2->ulUnicodeRange1 & 1L << 11);
		case HB_SCRIPT_VAI:
			return (fd->os2->ulUnicodeRange1 & 1L << 12);
		case HB_SCRIPT_ARABIC:
			return (fd->os2->ulUnicodeRange1 & 1L << 13) || (fd->os2->ulUnicodeRange2 & 1L << 31) || (fd->os2->ulUnicodeRange3 & 1L << 3);
		case HB_SCRIPT_NKO:
			return (fd->os2->ulUnicodeRange1 & 1L << 14);
		case HB_SCRIPT_DEVANAGARI:
			return (fd->os2->ulUnicodeRange1 & 1L << 15);
		case HB_SCRIPT_BENGALI:
			return (fd->os2->ulUnicodeRange1 & 1L << 16);
		case HB_SCRIPT_GURMUKHI:
			return (fd->os2->ulUnicodeRange1 & 1L << 17);
		case HB_SCRIPT_GUJARATI:
			return (fd->os2->ulUnicodeRange1 & 1L << 18);
		case HB_SCRIPT_ORIYA:
			return (fd->os2->ulUnicodeRange1 & 1L << 19);
		case HB_SCRIPT_TAMIL:
			return (fd->os2->ulUnicodeRange1 & 1L << 20);
		case HB_SCRIPT_TELUGU:
			return (fd->os2->ulUnicodeRange1 & 1L << 21);
		case HB_SCRIPT_KANNADA:
			return (fd->os2->ulUnicodeRange1 & 1L << 22);
		case HB_SCRIPT_MALAYALAM:
			return (fd->os2->ulUnicodeRange1 & 1L << 23);
		case HB_SCRIPT_THAI:
			return (fd->os2->ulUnicodeRange1 & 1L << 24);
		case HB_SCRIPT_LAO:
			return (fd->os2->ulUnicodeRange1 & 1L << 25);
		case HB_SCRIPT_GEORGIAN:
			return (fd->os2->ulUnicodeRange1 & 1L << 26);
		case HB_SCRIPT_BALINESE:
			return (fd->os2->ulUnicodeRange1 & 1L << 27);
		case HB_SCRIPT_HANGUL:
			return (fd->os2->ulUnicodeRange1 & 1L << 28) || (fd->os2->ulUnicodeRange2 & 1L << 20) || (fd->os2->ulUnicodeRange2 & 1L << 24);
		case HB_SCRIPT_HAN:
			return (fd->os2->ulUnicodeRange2 & 1L << 21) || (fd->os2->ulUnicodeRange2 & 1L << 22) || (fd->os2->ulUnicodeRange2 & 1L << 23) || (fd->os2->ulUnicodeRange2 & 1L << 26) || (fd->os2->ulUnicodeRange2 & 1L << 27) || (fd->os2->ulUnicodeRange2 & 1L << 29);
		case HB_SCRIPT_HIRAGANA:
			return (fd->os2->ulUnicodeRange2 & 1L << 17);
		case HB_SCRIPT_KATAKANA:
			return (fd->os2->ulUnicodeRange2 & 1L << 18);
		case HB_SCRIPT_BOPOMOFO:
			return (fd->os2->ulUnicodeRange2 & 1L << 19);
		case HB_SCRIPT_TIBETAN:
			return (fd->os2->ulUnicodeRange3 & 1L << 6);
		case HB_SCRIPT_SYRIAC:
			return (fd->os2->ulUnicodeRange3 & 1L << 7);
		case HB_SCRIPT_THAANA:
			return (fd->os2->ulUnicodeRange3 & 1L << 8);
		case HB_SCRIPT_SINHALA:
			return (fd->os2->ulUnicodeRange3 & 1L << 9);
		case HB_SCRIPT_MYANMAR:
			return (fd->os2->ulUnicodeRange3 & 1L << 10);
		case HB_SCRIPT_ETHIOPIC:
			return (fd->os2->ulUnicodeRange3 & 1L << 11);
		case HB_SCRIPT_CHEROKEE:
			return (fd->os2->ulUnicodeRange3 & 1L << 12);
		case HB_SCRIPT_CANADIAN_SYLLABICS:
			return (fd->os2->ulUnicodeRange3 & 1L << 13);
		case HB_SCRIPT_OGHAM:
			return (fd->os2->ulUnicodeRange3 & 1L << 14);
		case HB_SCRIPT_RUNIC:
			return (fd->os2->ulUnicodeRange3 & 1L << 15);
		case HB_SCRIPT_KHMER:
			return (fd->os2->ulUnicodeRange3 & 1L << 16);
		case HB_SCRIPT_MONGOLIAN:
			return (fd->os2->ulUnicodeRange3 & 1L << 17);
		case HB_SCRIPT_YI:
			return (fd->os2->ulUnicodeRange3 & 1L << 19);
		case HB_SCRIPT_HANUNOO:
		case HB_SCRIPT_TAGBANWA:
		case HB_SCRIPT_BUHID:
		case HB_SCRIPT_TAGALOG:
			return (fd->os2->ulUnicodeRange3 & 1L << 20);
		case HB_SCRIPT_OLD_ITALIC:
			return (fd->os2->ulUnicodeRange3 & 1L << 21);
		case HB_SCRIPT_GOTHIC:
			return (fd->os2->ulUnicodeRange3 & 1L << 22);
		case HB_SCRIPT_DESERET:
			return (fd->os2->ulUnicodeRange3 & 1L << 23);
		case HB_SCRIPT_LIMBU:
			return (fd->os2->ulUnicodeRange3 & 1L << 29);
		case HB_SCRIPT_TAI_LE:
			return (fd->os2->ulUnicodeRange3 & 1L << 30);
		case HB_SCRIPT_NEW_TAI_LUE:
			return (fd->os2->ulUnicodeRange3 & 1L << 31);
		case HB_SCRIPT_BUGINESE:
			return (fd->os2->ulUnicodeRange4 & 1L << 0);
		case HB_SCRIPT_GLAGOLITIC:
			return (fd->os2->ulUnicodeRange4 & 1L << 1);
		case HB_SCRIPT_TIFINAGH:
			return (fd->os2->ulUnicodeRange4 & 1L << 2);
		case HB_SCRIPT_SYLOTI_NAGRI:
			return (fd->os2->ulUnicodeRange4 & 1L << 4);
		case HB_SCRIPT_LINEAR_B:
			return (fd->os2->ulUnicodeRange4 & 1L << 5);
		case HB_SCRIPT_UGARITIC:
			return (fd->os2->ulUnicodeRange4 & 1L << 7);
		case HB_SCRIPT_OLD_PERSIAN:
			return (fd->os2->ulUnicodeRange4 & 1L << 8);
		case HB_SCRIPT_SHAVIAN:
			return (fd->os2->ulUnicodeRange4 & 1L << 9);
		case HB_SCRIPT_OSMANYA:
			return (fd->os2->ulUnicodeRange4 & 1L << 10);
		case HB_SCRIPT_CYPRIOT:
			return (fd->os2->ulUnicodeRange4 & 1L << 11);
		case HB_SCRIPT_KHAROSHTHI:
			return (fd->os2->ulUnicodeRange4 & 1L << 12);
		case HB_SCRIPT_TAI_VIET:
			return (fd->os2->ulUnicodeRange4 & 1L << 13);
		case HB_SCRIPT_CUNEIFORM:
			return (fd->os2->ulUnicodeRange4 & 1L << 14);
		case HB_SCRIPT_SUNDANESE:
			return (fd->os2->ulUnicodeRange4 & 1L << 16);
		case HB_SCRIPT_LEPCHA:
			return (fd->os2->ulUnicodeRange4 & 1L << 17);
		case HB_SCRIPT_OL_CHIKI:
			return (fd->os2->ulUnicodeRange4 & 1L << 18);
		case HB_SCRIPT_SAURASHTRA:
			return (fd->os2->ulUnicodeRange4 & 1L << 19);
		case HB_SCRIPT_KAYAH_LI:
			return (fd->os2->ulUnicodeRange4 & 1L << 20);
		case HB_SCRIPT_REJANG:
			return (fd->os2->ulUnicodeRange4 & 1L << 21);
		case HB_SCRIPT_CHAM:
			return (fd->os2->ulUnicodeRange4 & 1L << 22);
		case HB_SCRIPT_ANATOLIAN_HIEROGLYPHS:
			return (fd->os2->ulUnicodeRange4 & 1L << 25);
		default:
			return false;
	};
#else
	ERR_FAIL_V_MSG(false, "Compiled without FreeType support!");
#endif
}

String FontDataAdvanced::get_supported_chars() const {
	_THREAD_SAFE_METHOD_

	String chars;

	ERR_FAIL_COND_V(font_type == FONT_NONE, chars);
	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, base_size);
	ERR_FAIL_COND_V(fd == nullptr, chars);

#ifdef MODULE_FREETYPE_ENABLED
	if (font_type == FONT_DYNAMIC) {
		FT_UInt gindex;
		FT_ULong charcode = FT_Get_First_Char(fd->face, &gindex);
		while (gindex != 0) {
			if (charcode != 0) {
				chars += char32_t(charcode);
			}
			charcode = FT_Get_Next_Char(fd->face, charcode, &gindex);
		}
	} else {
		for (const uint32_t *key = fd->glyph_map.next(nullptr); key; key = fd->glyph_map.next(key)) {
			const Glyph &c = fd->glyph_map[*key];
			if (c.found) {
				chars += char32_t(*key);
			}
		}
	}
	return chars;
#else
	ERR_FAIL_V_MSG(chars, "Compiled without FreeType support!");
#endif
}

void FontDataAdvanced::set_variation(const String &p_name, double p_value) {
	if (font_type != FONT_DYNAMIC) {
		return;
	}

	int32_t tag = TS->name_to_tag(p_name);
	if (!base_variation.variations.has(tag) || (base_variation.variations[tag] != p_value)) {
		base_variation.variations[tag] = p_value;
		base_variation.update_key();
	}
}

double FontDataAdvanced::get_variation(const String &p_name) const {
	if (font_type != FONT_DYNAMIC) {
		return 0.f;
	}

	int32_t tag = TS->name_to_tag(p_name);
	if (!base_variation.variations.has(tag)) {
		return 0.f;
	}
	return base_variation.variations[tag];
}

void FontDataAdvanced::set_base_size(int p_size) {
	if (font_type != FONT_DYNAMIC) {
		return;
	}

	if (base_size.size != p_size) {
		base_size.size = p_size;
		if (msdf) {
			clear_cache();
		}
	}
}

int FontDataAdvanced::get_base_size() const {
	return base_size.size;
}

void FontDataAdvanced::set_distance_field_hint(bool p_distance_field) {
	if (font_type != FONT_DYNAMIC) {
		return;
	}

	if (msdf != p_distance_field) {
		msdf = p_distance_field;
		clear_cache();
	}
}

bool FontDataAdvanced::get_distance_field_hint() const {
	return msdf;
}

void FontDataAdvanced::set_disable_distance_field_shader(bool p_disable) {
	if (font_type != FONT_DYNAMIC) {
		return;
	}

	msdf_disabled = p_disable;
}

bool FontDataAdvanced::get_disable_distance_field_shader() const {
	return msdf_disabled;
}

void FontDataAdvanced::set_antialiased(bool p_antialiased) {
	if (font_type != FONT_DYNAMIC) {
		return;
	}

	if (antialiased != p_antialiased) {
		clear_cache();
		antialiased = p_antialiased;
	}
}

bool FontDataAdvanced::get_antialiased() const {
	return antialiased;
}

void FontDataAdvanced::set_force_autohinter(bool p_enabled) {
	if (font_type != FONT_DYNAMIC) {
		return;
	}

	if (force_autohinter != p_enabled) {
		clear_cache();
		force_autohinter = p_enabled;
	}
}

bool FontDataAdvanced::get_force_autohinter() const {
	return force_autohinter;
}

void FontDataAdvanced::set_oversampling(double p_value) {
	if (font_type != FONT_DYNAMIC) {
		return;
	}

	if (oversampling != p_value) {
		if (!msdf) {
			clear_cache();
		}
		oversampling = p_value;
	}
}

double FontDataAdvanced::get_oversampling() const {
	return oversampling;
}

void FontDataAdvanced::set_msdf_px_range(double p_range) {
	if (font_type != FONT_DYNAMIC || !msdf) {
		return;
	}

	if (msdf_margin != p_range) {
		clear_cache();
		msdf_margin = p_range;
	}
}

double FontDataAdvanced::get_msdf_px_range() const {
	return msdf_margin;
}

void FontDataAdvanced::set_hinting(TextServer::Hinting p_hinting) {
	if (font_type != FONT_DYNAMIC) {
		return;
	}

	if (hinting != p_hinting) {
		clear_cache();
		hinting = p_hinting;
	}
}

TextServer::Hinting FontDataAdvanced::get_hinting() const {
	return hinting;
}

bool FontDataAdvanced::has_outline() const {
	return (font_type == FONT_DYNAMIC) || msdf;
}

float FontDataAdvanced::get_height(int p_size) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, 0.f);

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	ERR_FAIL_COND_V(fd == nullptr, 0.f);
	return fd->height * double(size.size) / double(fd->size_id.size);
}

float FontDataAdvanced::get_ascent(int p_size) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, 0.f);

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	ERR_FAIL_COND_V(fd == nullptr, 0.f);
	return fd->ascent * double(size.size) / double(fd->size_id.size);
}

float FontDataAdvanced::get_descent(int p_size) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, 0.f);

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	ERR_FAIL_COND_V(fd == nullptr, 0.f);
	return fd->descent * double(size.size) / double(fd->size_id.size);
}

float FontDataAdvanced::get_underline_position(int p_size) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, 0.f);

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	ERR_FAIL_COND_V(fd == nullptr, 0.f);
	return fd->underline_position * double(size.size) / double(fd->size_id.size);
}

float FontDataAdvanced::get_underline_thickness(int p_size) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, 0.f);

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	ERR_FAIL_COND_V(fd == nullptr, 0.f);
	return fd->underline_thickness * double(size.size) / double(fd->size_id.size);
}

float FontDataAdvanced::get_font_scale(int p_size) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, 0.f);

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	ERR_FAIL_COND_V(fd == nullptr, 1.0f);

	return (fd->scale_color_font / fd->oversampling) * double(size.size) / double(fd->size_id.size);
}

bool FontDataAdvanced::has_char(char32_t p_char) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, false);

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, base_size);
	ERR_FAIL_COND_V(fd == nullptr, false);

	uint32_t index = get_glyph_index(p_char, 0x0000);
	const_cast<FontDataAdvanced *>(this)->update_glyph(fd, index);
	Glyph gl = fd->glyph_map[index];

	return gl.found;
}

hb_font_t *FontDataAdvanced::get_hb_handle(int p_size) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, nullptr);

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = get_cache_data(base_variation, size);
	ERR_FAIL_COND_V(fd == nullptr, nullptr);

	return fd->hb_handle;
}

uint32_t FontDataAdvanced::get_glyph_index(char32_t p_char, char32_t p_variation_selector) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, 0);

	if (font_type != FONT_DYNAMIC) {
		return p_char;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, base_size);
	ERR_FAIL_COND_V(fd == nullptr, 0);

#ifdef MODULE_FREETYPE_ENABLED
	if (p_variation_selector == 0x0000) {
		return FT_Get_Char_Index(fd->face, p_char);
	} else {
		return FT_Face_GetCharVariantIndex(fd->face, p_char, p_variation_selector);
	}
#else
	ERR_FAIL_V_MSG(0, "Compiled without FreeType support!");
#endif
}

Vector2 FontDataAdvanced::get_advance(uint32_t p_index, int p_size) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, Vector2());

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	ERR_FAIL_COND_V(fd == nullptr, Vector2());

	const_cast<FontDataAdvanced *>(this)->update_glyph(fd, p_index);
	Glyph gl = fd->glyph_map[p_index];

	if (gl.found) {
		return gl.advance * double(size.size) / double(fd->size_id.size);
	} else {
		return Vector2();
	}
}

Vector2 FontDataAdvanced::get_kerning(uint32_t p_a, uint32_t p_b, int p_size) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, Vector2());

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	ERR_FAIL_COND_V(fd == nullptr, Vector2());

	if (font_type == FONT_DYNAMIC) {
#ifdef MODULE_FREETYPE_ENABLED
		FT_Vector delta;
		FT_Get_Kerning(fd->face, p_a, p_b, FT_KERNING_DEFAULT, &delta);

		return Vector2(delta.x, delta.y) * double(size.size) / double(fd->size_id.size);
#else
		ERR_FAIL_V_MSG(Vector2(), "Compiled without FreeType support!");
#endif
	} else {
		KerningPairKey kpk;
		kpk.A = p_a;
		kpk.B = p_b;

		const Map<KerningPairKey, int>::Element *E = fd->kerning_map.find(kpk);
		if (E) {
			return Vector2(-E->get() * double(size.size) / double(fd->size_id.size), 0.f);
		} else {
			return Vector2();
		}
	}
}

Vector2 FontDataAdvanced::get_glyph_size(uint32_t p_index, int p_size) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(font_type == FONT_NONE, Vector2());

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	ERR_FAIL_COND_V(fd == nullptr, Vector2());

	const_cast<FontDataAdvanced *>(this)->update_glyph(fd, p_index);
	Glyph gl = fd->glyph_map[p_index];

	if (gl.found) {
		return gl.rect.size * double(size.size) / double(fd->size_id.size);
	} else {
		return Vector2();
	}
}

void FontDataAdvanced::draw_glyph(RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(font_type == FONT_NONE);

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	ERR_FAIL_COND(fd == nullptr);

	const_cast<FontDataAdvanced *>(this)->update_glyph(fd, p_index);
	Glyph gl = fd->glyph_map[p_index];

	if (gl.found) {
		ERR_FAIL_COND(gl.texture_idx < -1 || gl.texture_idx >= fd->textures.size());

		if (gl.texture_idx != -1) {
			Color modulate = p_color;
#ifdef MODULE_FREETYPE_ENABLED
			if ((font_type == FONT_DYNAMIC) && FT_HAS_COLOR(fd->face)) {
				modulate.r = modulate.g = modulate.b = 1.0;
			}
#endif
			if (RenderingServer::get_singleton() != nullptr) {
				RID texture = fd->textures[gl.texture_idx].texture->get_rid();
				if (msdf && !msdf_disabled) {
					Point2 cpos = p_pos;
					cpos += (gl.rect.position * double(size.size) / double(fd->size_id.size));
					Size2 csize = (gl.rect.size * double(size.size) / double(fd->size_id.size));
					RenderingServer::get_singleton()->canvas_item_add_msdf_texture_rect_region(p_canvas, Rect2(cpos, csize), texture, gl.rect_uv, modulate, -1, msdf_margin);
				} else {
					Point2i cpos = p_pos;
					cpos += (gl.rect.position * double(size.size) / double(fd->size_id.size));
					Size2i csize = (gl.rect.size * double(size.size) / double(fd->size_id.size));
					RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, Rect2(cpos, csize), texture, gl.rect_uv, modulate, false, false);
				}
			}
		}
	}
}

void FontDataAdvanced::draw_glyph_outline(RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(font_type == FONT_NONE);

	if (msdf && msdf_disabled) {
		return;
	}

	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}
	size.outline_size = p_outline_size;

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	if (fd == nullptr) {
		return;
	}

	const_cast<FontDataAdvanced *>(this)->update_glyph(fd, p_index);
	Glyph gl = fd->glyph_map[p_index];

	if (gl.found) {
		ERR_FAIL_COND(gl.texture_idx < -1 || gl.texture_idx >= fd->textures.size());

		if (gl.texture_idx != -1) {
			Color modulate = p_color;
#ifdef MODULE_FREETYPE_ENABLED
			if ((font_type == FONT_DYNAMIC) && FT_HAS_COLOR(fd->face)) {
				modulate.r = modulate.g = modulate.b = 1.0;
			}
#endif
			if (RenderingServer::get_singleton() != nullptr) {
				RID texture = fd->textures[gl.texture_idx].texture->get_rid();
				if (msdf) {
					Point2 cpos = p_pos;
					cpos += (gl.rect.position * double(size.size) / double(fd->size_id.size));
					Size2 csize = (gl.rect.size * double(size.size) / double(fd->size_id.size));
					RenderingServer::get_singleton()->canvas_item_add_msdf_texture_rect_region(p_canvas, Rect2(cpos, csize), texture, gl.rect_uv, modulate, p_outline_size, msdf_margin);
				} else {
					Point2i cpos = p_pos;
					cpos += (gl.rect.position * double(size.size) / double(fd->size_id.size));
					Size2i csize = (gl.rect.size * double(size.size) / double(fd->size_id.size));
					RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, Rect2(cpos, csize), texture, gl.rect_uv, modulate, false, false);
				}
			}
		}
	}
}

bool FontDataAdvanced::get_glyph_contours(int p_size, uint32_t p_index, Vector<Vector3> &r_points, Vector<int32_t> &r_contours, bool &r_orientation) const {
	_THREAD_SAFE_METHOD_

	if (font_type != FONT_DYNAMIC) {
		return false;
	}

#ifdef MODULE_FREETYPE_ENABLED
	SizeKey size;
	if (p_size <= 0) {
		size.size = base_size.size;
	} else {
		size.size = p_size;
	}

	Data *fd = const_cast<FontDataAdvanced *>(this)->get_cache_data(base_variation, size);
	if (fd == nullptr) {
		return false;
	}
	int error = FT_Load_Glyph(fd->face, p_index, FT_LOAD_NO_BITMAP | (force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0));
	ERR_FAIL_COND_V(error, false);

	r_points.clear();
	r_contours.clear();

	float h = fd->ascent;
	float scale = (1.0 / 64.0) / fd->oversampling * fd->scale_color_font;
	for (short i = 0; i < fd->face->glyph->outline.n_points; i++) {
		r_points.push_back(Vector3(fd->face->glyph->outline.points[i].x * scale, h - fd->face->glyph->outline.points[i].y * scale, FT_CURVE_TAG(fd->face->glyph->outline.tags[i])));
	}
	for (short i = 0; i < fd->face->glyph->outline.n_contours; i++) {
		r_contours.push_back(fd->face->glyph->outline.contours[i]);
	}
	r_orientation = (FT_Outline_Get_Orientation(&fd->face->glyph->outline) == FT_ORIENTATION_FILL_RIGHT);
#else
	return false;
#endif
	return true;
}

FontDataAdvanced::~FontDataAdvanced() {
	clear_cache(true);
#ifdef MODULE_FREETYPE_ENABLED
	if (library != nullptr) {
		FT_Done_FreeType(library);
	}
#endif
}
