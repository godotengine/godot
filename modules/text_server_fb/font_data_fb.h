/*************************************************************************/
/*  font_data_fb.h                                                       */
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

#ifndef FONT_DATA_ADV_H
#define FONT_DATA_ADV_H

#include "modules/modules_enabled.gen.h"
#include "servers/text_server.h"

#ifdef MODULE_FREETYPE_ENABLED
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_TRUETYPE_TABLES_H
#include FT_STROKER_H
#include FT_ADVANCES_H
#include FT_MULTIPLE_MASTERS_H
#include FT_BBOX_H
#endif

struct FontDataFallback {
	_THREAD_SAFE_CLASS_

private:
	struct FontTexture {
		Vector<uint8_t> imgdata;
		int texture_w = 0;
		int texture_h = 0;
		Vector<int16_t> offsets;
		Ref<ImageTexture> texture;
	};

	struct TexturePosition {
		int index = 0;
		int x = 0;
		int y = 0;
	};

	struct Glyph {
		bool found = false;
		int texture_idx = 0;
		Rect2 rect;
		Rect2 rect_uv;
		Vector2 advance;
	};

	struct KerningPairKey {
		union {
			struct {
				uint32_t A, B;
			};
			uint64_t pair = 0;
		};

		_FORCE_INLINE_ bool operator<(const KerningPairKey &p_r) const { return pair < p_r.pair; }
	};

	struct VariationKey {
		uint64_t key = 5381;
		Map<int32_t, double> variations;
		void update_key() {
			key = 5381;
			for (const Map<int32_t, double>::Element *E = variations.front(); E; E = E->next()) {
				key = hash_djb2_one_32(E->key(), key);
				key = hash_djb2_one_float(E->get(), key);
			}
		}

		VariationKey() {
			key = 5381;
		}

		_FORCE_INLINE_ bool operator<(VariationKey p_r) const { return key < p_r.key; }
	};

	struct SizeKey {
		union {
			struct {
				uint32_t size : 16;
				uint32_t outline_size : 16;
			};
			uint32_t key = 0;
		};

		SizeKey() {
			size = 16;
			outline_size = 0;
		}
		SizeKey(int p_size) {
			size = p_size;
			outline_size = 0;
		}

		_FORCE_INLINE_ bool operator<(SizeKey p_r) const { return key < p_r.key; }
	};

	struct Data {
#ifdef MODULE_FREETYPE_ENABLED
		FT_Face face = nullptr;
		TT_OS2 *os2 = nullptr;
		FT_StreamRec stream;
#endif
		VariationKey variation_id;
		SizeKey size_id;

		float scale_color_font = 1.f;

		float height = 0.f;
		float ascent = 0.f;
		float descent = 0.f;
		float underline_position = 0.f;
		float underline_thickness = 0.f;
		float oversampling = 1.f;

		Vector<FontTexture> textures;
		HashMap<uint32_t, Glyph> glyph_map;
		Map<KerningPairKey, int> kerning_map;

		~Data() {
#ifdef MODULE_FREETYPE_ENABLED
			if (face != nullptr) {
				FT_Done_Face(face);
			}
#endif
		}
	};

	enum FontType {
		FONT_NONE,
		FONT_BITMAP,
		FONT_DYNAMIC,
	};
#ifdef MODULE_FREETYPE_ENABLED
	FT_Library library = nullptr;
#endif
	// Source data.
	const uint8_t *font_mem = nullptr;
	uint64_t font_mem_size = 0;
	String font_path;
	Vector<uint8_t> font_mem_cache;

	SizeKey base_size = SizeKey(16);
	VariationKey base_variation;

	float rect_margin = 1.f;
	float msdf_margin = 10.f;
	int spacing_space = 0;
	int spacing_glyph = 0;
	float oversampling = 0.f;

	bool antialiased = true;
	bool force_autohinter = false;
	bool msdf = false;
	bool msdf_disabled = false;
	TextServer::Hinting hinting = TextServer::HINTING_LIGHT;

	FontType font_type = FONT_NONE;

	Map<VariationKey, Map<SizeKey, Data *>> cache;

	// Font Cache
	Data *get_cache_data(const VariationKey &p_var_id, const SizeKey &p_size_id);

	Error _load(const String &p_path, FileAccess *p_f, int p_base_size);
	Error _load_bmp(const String &p_path, FileAccess *p_f, int p_base_size);
	Error _load_ttf(const String &p_path, FileAccess *p_f, int p_base_size);
	Error _load_cache(const String &p_path, FileAccess *p_f, int p_base_size);

	// Glpyh Rendering
	void update_glyph(Data *p_fd, uint32_t p_index);
	TexturePosition find_texture_pos_for_glyph(Data *p_data, int p_color_size, Image::Format p_image_format, int p_width, int p_height);

#ifdef MODULE_MSDFGEN_ENABLED
	Glyph rasterize_msdf(Data *p_data, FT_Outline *outline, const Vector2 &advance);
#endif

#ifdef MODULE_FREETYPE_ENABLED
	Glyph rasterize_bitmap(Data *p_data, FT_Bitmap bitmap, int yofs, int xofs, const Vector2 &advance);
#endif

public:
	Map<String, bool> lang_support_overrides;
	Map<String, bool> script_support_overrides;

	Error save_cache(const String &p_path, uint8_t p_flags, List<String> *r_gen_files) const;
	void add_to_cache(const Map<int32_t, double> &p_var_id, int p_size, int p_outline_size);
	void clear_cache(bool p_force = false);

	int get_spacing_space() const { return spacing_space; };
	void set_spacing_space(int p_value) {
		spacing_space = p_value;
		clear_cache();
	};

	int get_spacing_glyph() const { return spacing_glyph; };
	void set_spacing_glyph(int p_value) {
		spacing_glyph = p_value;
		clear_cache();
	};

	void preload_range(uint32_t p_start, uint32_t p_end, bool p_glyphs);

	Error load_from_file(const String &p_path, int p_base_size);
	Error load_from_memory(const uint8_t *p_data, size_t p_size, int p_base_size);
	Error bitmap_new(float p_height, float p_ascent, int p_base_size);

	void bitmap_add_texture(const Ref<Texture2D> &p_texture);
	void bitmap_add_char(char32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance);
	void bitmap_add_kerning_pair(char32_t p_A, char32_t p_B, int p_kerning);

	Dictionary get_variation_list() const;
	Dictionary get_feature_list() const;
	bool is_lang_supported(const String &p_lang) const { return false; };
	bool is_script_supported(uint32_t p_script) const;
	String get_supported_chars() const;

	void set_variation(const String &p_name, double p_value);
	double get_variation(const String &p_name) const;

	void set_base_size(int p_base_size);
	int get_base_size() const;

	void set_distance_field_hint(bool p_distance_field);
	bool get_distance_field_hint() const;

	void set_disable_distance_field_shader(bool p_disable);
	bool get_disable_distance_field_shader() const;

	void set_antialiased(bool p_antialiased);
	bool get_antialiased() const;

	void set_force_autohinter(bool p_enabled);
	bool get_force_autohinter() const;

	void set_oversampling(double p_value);
	double get_oversampling() const;

	void set_msdf_px_range(double p_range);
	double get_msdf_px_range() const;

	void set_hinting(TextServer::Hinting p_hinting);
	TextServer::Hinting get_hinting() const;

	bool has_outline() const;

	float get_height(int p_size) const;
	float get_ascent(int p_size) const;
	float get_descent(int p_size) const;
	float get_underline_position(int p_size) const;
	float get_underline_thickness(int p_size) const;

	bool has_char(char32_t p_char) const;
	float get_font_scale(int p_size) const;

	uint32_t get_glyph_index(char32_t p_char, char32_t p_variation_selector) const;
	Vector2 get_advance(uint32_t p_index, int p_size) const;
	Vector2 get_kerning(uint32_t p_a, uint32_t p_b, int p_size) const;
	Vector2 get_glyph_size(uint32_t p_index, int p_size) const;

	void draw_glyph(RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const;
	void draw_glyph_outline(RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const;

	bool get_glyph_contours(int p_size, uint32_t p_index, Vector<Vector3> &r_points, Vector<int32_t> &r_contours, bool &r_orientation) const;

	~FontDataFallback();
};

#endif // FONT_DATA_ADV_H
