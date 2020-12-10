/*************************************************************************/
/*  dynamic_font_fb.h                                                    */
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

#ifndef DYNAMIC_FONT_FALLBACK_H
#define DYNAMIC_FONT_FALLBACK_H

#include "font_fb.h"

#include "modules/modules_enabled.gen.h"

#ifdef MODULE_FREETYPE_ENABLED

#include <ft2build.h>
#include FT_FREETYPE_H

struct DynamicFontDataFallback : public FontDataFallback {
	_THREAD_SAFE_CLASS_

private:
	struct CharTexture {
		Vector<uint8_t> imgdata;
		int texture_size = 0;
		Vector<int> offsets;
		Ref<ImageTexture> texture;
	};

	struct Character {
		bool found = false;
		int texture_idx = 0;
		Rect2 rect;
		Rect2 rect_uv;
		Vector2 align;
		Vector2 advance = Vector2(-1, -1);

		static Character not_found();
	};

	struct TexturePosition {
		int index = 0;
		int x = 0;
		int y = 0;
	};

	struct CacheID {
		union {
			struct {
				uint32_t size : 16;
				uint32_t outline_size : 16;
			};
			uint32_t key = 0;
		};
		bool operator<(CacheID right) const {
			return key < right.key;
		}
	};

	struct DataAtSize {
		FT_Face face = nullptr;
		FT_StreamRec stream;

		int size = 0;
		float scale_color_font = 1.f;
		float ascent = 0.0;
		float descent = 0.0;
		float underline_position = 0.0;
		float underline_thickness = 0.0;

		Vector<CharTexture> textures;
		HashMap<char32_t, Character> char_map;

		~DataAtSize() {
			if (face != nullptr) {
				FT_Done_Face(face);
			}
		}
	};

	FT_Library library = nullptr;

	// Source data.
	const uint8_t *font_mem = nullptr;
	int font_mem_size = 0;
	String font_path;
	Vector<uint8_t> font_mem_cache;

	float rect_margin = 1.f;
	int base_size = 16;
	float oversampling = 1.f;
	bool antialiased = true;
	bool force_autohinter = false;
	TextServer::Hinting hinting = TextServer::HINTING_LIGHT;

	Map<CacheID, DataAtSize *> size_cache;
	Map<CacheID, DataAtSize *> size_cache_outline;

	DataAtSize *get_data_for_size(int p_size, int p_outline_size = 0);

	TexturePosition find_texture_pos_for_glyph(DataAtSize *p_data, int p_color_size, Image::Format p_image_format, int p_width, int p_height);
	Character bitmap_to_character(DataAtSize *p_data, FT_Bitmap bitmap, int yofs, int xofs, const Vector2 &advance);
	_FORCE_INLINE_ void update_char(int p_size, char32_t p_char);
	_FORCE_INLINE_ void update_char_outline(int p_size, int p_outline_size, char32_t p_char);

public:
	virtual void clear_cache() override;

	virtual Error load_from_file(const String &p_filename, int p_base_size) override;
	virtual Error load_from_memory(const uint8_t *p_data, size_t p_size, int p_base_size) override;

	virtual float get_height(int p_size) const override;
	virtual float get_ascent(int p_size) const override;
	virtual float get_descent(int p_size) const override;

	virtual float get_underline_position(int p_size) const override;
	virtual float get_underline_thickness(int p_size) const override;

	virtual void set_antialiased(bool p_antialiased) override;
	virtual bool get_antialiased() const override;

	virtual void set_hinting(TextServer::Hinting p_hinting) override;
	virtual TextServer::Hinting get_hinting() const override;

	virtual void set_force_autohinter(bool p_enabeld) override;
	virtual bool get_force_autohinter() const override;

	virtual void set_distance_field_hint(bool p_distance_field) override{};
	virtual bool get_distance_field_hint() const override { return false; };

	virtual bool has_outline() const override;
	virtual float get_base_size() const override;

	virtual bool has_char(char32_t p_char) const override;
	virtual String get_supported_chars() const override;

	virtual Vector2 get_advance(char32_t p_char, int p_size) const override;
	virtual Vector2 get_kerning(char32_t p_char, char32_t p_next, int p_size) const override;

	virtual Vector2 draw_glyph(RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const override;
	virtual Vector2 draw_glyph_outline(RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const override;

	virtual bool get_glyph_contours(int p_size, uint32_t p_index, Vector<Vector3> &r_points, Vector<int32_t> &r_contours, bool &r_orientation) const override;

	virtual ~DynamicFontDataFallback() override;
};

#endif // MODULE_FREETYPE_ENABLED

#endif // DYNAMIC_FONT_FALLBACK_H
