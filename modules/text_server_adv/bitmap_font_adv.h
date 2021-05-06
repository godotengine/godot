/*************************************************************************/
/*  bitmap_font_adv.h                                                    */
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

#ifndef BITMAP_FONT_ADV_H
#define BITMAP_FONT_ADV_H

#include "font_adv.h"

void hb_bmp_create_font_funcs();
void hb_bmp_free_font_funcs();

struct BitmapFontDataAdvanced : public FontDataAdvanced {
	_THREAD_SAFE_CLASS_

private:
	Vector<Ref<Texture2D>> textures;

	struct Character {
		int texture_idx = 0;
		Rect2 rect;
		Vector2 align;
		Vector2 advance = Vector2(-1, -1);
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

	HashMap<uint32_t, Character> char_map;
	Map<KerningPairKey, int> kerning_map;
	hb_font_t *hb_handle = nullptr;

	float height = 0.f;
	float ascent = 0.f;
	int base_size = 0;
	bool distance_field_hint = false;

public:
	virtual void clear_cache() override{};

	virtual Error load_from_file(const String &p_filename, int p_base_size) override;
	virtual Error bitmap_new(float p_height, float p_ascent, int p_base_size) override;

	virtual void bitmap_add_texture(const Ref<Texture> &p_texture) override;
	virtual void bitmap_add_char(char32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance) override;
	virtual void bitmap_add_kerning_pair(char32_t p_A, char32_t p_B, int p_kerning) override;

	virtual float get_height(int p_size) const override;
	virtual float get_ascent(int p_size) const override;
	virtual float get_descent(int p_size) const override;

	virtual float get_underline_position(int p_size) const override;
	virtual float get_underline_thickness(int p_size) const override;

	virtual void set_antialiased(bool p_antialiased) override{};
	virtual bool get_antialiased() const override { return false; };

	virtual void set_hinting(TextServer::Hinting p_hinting) override{};
	virtual TextServer::Hinting get_hinting() const override { return TextServer::HINTING_NONE; };

	virtual void set_distance_field_hint(bool p_distance_field) override;
	virtual bool get_distance_field_hint() const override;

	virtual void set_force_autohinter(bool p_enabeld) override{};
	virtual bool get_force_autohinter() const override { return false; };

	virtual bool has_outline() const override { return false; };
	virtual float get_base_size() const override;
	virtual float get_font_scale(int p_size) const override;

	virtual hb_font_t *get_hb_handle(int p_size) override;

	virtual bool has_char(char32_t p_char) const override;
	virtual String get_supported_chars() const override;

	virtual Vector2 get_advance(uint32_t p_char, int p_size) const override;
	Vector2 get_align(uint32_t p_char, int p_size) const;
	Vector2 get_size(uint32_t p_char, int p_size) const;
	virtual Vector2 get_kerning(uint32_t p_char, uint32_t p_next, int p_size) const override;
	virtual uint32_t get_glyph_index(char32_t p_char, char32_t p_variation_selector) const override { return (uint32_t)p_char; };

	virtual Vector2 draw_glyph(RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const override;
	virtual Vector2 draw_glyph_outline(RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const override;

	virtual ~BitmapFontDataAdvanced();
};

#endif // BITMAP_FONT_ADV_H
