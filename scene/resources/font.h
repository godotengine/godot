/*************************************************************************/
/*  font.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef FONT_H
#define FONT_H

#include "core/map.h"
#include "core/resource.h"
#include "scene/resources/texture.h"

#ifdef USE_TEXT_SHAPING
#include <hb.h>
#endif

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

//http://font.gohu.org/ (WTFPL) based 5x7 hex number font for char code box drawing
static const unsigned char _hex_box_img_data[167] = {
	0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0x07, 0x01, 0x03, 0x00, 0x00, 0x00, 0xA5, 0x54, 0x58, 0xA1, 0x00, 0x00, 0x00, 0x06, 0x50, 0x4C, 0x54, 0x45, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xA5, 0xD9, 0x9F, 0xDD, 0x00, 0x00, 0x00, 0x01, 0x74, 0x52, 0x4E, 0x53, 0x00, 0x40, 0xE6, 0xD8, 0x66, 0x00, 0x00, 0x00, 0x4F, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0x28, 0x94, 0x79, 0x58, 0x7B, 0xBF, 0x78, 0xEE, 0xF3, 0xEA, 0xFF, 0x0C, 0xDD, 0xCA, 0xC2, 0x4E, 0x8C, 0x3D, 0xC9, 0x12, 0xC7, 0x04, 0x18, 0xE6, 0x32, 0x89, 0x56, 0x1F, 0x02, 0x32, 0xDD, 0x04, 0x18, 0x56, 0xB2, 0x64, 0xB2, 0x29, 0x15, 0xFF, 0x7F, 0xE1, 0x7E, 0x8F, 0xE1, 0x24, 0x87, 0x7C, 0x9B, 0x4A, 0x07, 0x58, 0xB4, 0x53, 0x50, 0xD0, 0x0D, 0xC4, 0x04, 0xAA, 0x2D, 0xB4, 0x7B, 0x68, 0x79, 0xA4, 0x78, 0xF1, 0xF3, 0xEA, 0x0F, 0x00, 0x5F, 0x2A, 0x1C, 0xFE, 0x51, 0xD4, 0xC9, 0xA3, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
};

class Font : public Resource {

	GDCLASS(Font, Resource);

	static RID hex_tex;

protected:
	static void _bind_methods();

public:
	virtual float get_height() const = 0;

	virtual float get_ascent() const = 0;
	virtual float get_descent() const = 0;

	virtual Size2 get_char_size(CharType p_char, CharType p_next = 0) const = 0;

	Size2 get_string_size(const String &p_string) const;
	static void draw_hex_box(RID p_canvas_item, const Point2 &p_pos, uint32_t p_charcode, const Color &p_modulate);

	virtual bool is_distance_field_hint() const = 0;

	virtual int get_fallback_count() const = 0;

#ifdef USE_TEXT_SHAPING
	virtual hb_font_t *get_hb_font(int p_fallback_index = -1) const = 0;
#endif
	void draw(RID p_canvas_item, const Point2 &p_pos, const String &p_text, const Color &p_modulate = Color(1, 1, 1), int p_clip_w = -1, const Color &p_outline_modulate = Color(1, 1, 1)) const;
	void draw_halign(RID p_canvas_item, const Point2 &p_pos, HAlign p_align, float p_width, const String &p_text, const Color &p_modulate = Color(1, 1, 1), const Color &p_outline_modulate = Color(1, 1, 1)) const;
	void draw_paragraph(RID p_canvas_item, const Point2 &p_pos, const String &p_text, const Color &p_modulate = Color(1, 1, 1), int p_clip_w = -1, const Color &p_outline_modulate = Color(1, 1, 1), TextBreak p_bflags = TEXT_BREAK_MANDATORY_AND_WORD_BOUND, TextJustification p_jflags = TEXT_JUSTIFICATION_KASHIDA_AND_WHITESPACE) const;

	virtual bool has_outline() const { return false; }
	virtual float draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1), bool p_outline = false) const = 0;
	virtual void draw_glyph(RID p_canvas_item, const Point2 &p_pos, uint32_t p_codepoint, const Point2 &p_offset, float p_ascent, const Color &p_modulate, bool p_outline, int p_fallback_index) const = 0;

	static void initialize_hex_font();
	static void finish_hex_font();

	void update_changes();
	Font(){};
};

// Helper class to that draws outlines immediately and draws characters in its destructor.
class FontDrawer {
	const Ref<Font> &font;
	Color outline_color;
	bool has_outline;

	struct PendingDraw {
		RID canvas_item;
		Point2 pos;
		CharType chr;
		CharType next;
		Color modulate;
	};

	Vector<PendingDraw> pending_draws;

public:
	FontDrawer(const Ref<Font> &p_font, const Color &p_outline_color) :
			font(p_font),
			outline_color(p_outline_color) {
		has_outline = p_font->has_outline();
	}

	float draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1)) {
		if (has_outline) {
			PendingDraw draw = { p_canvas_item, p_pos, p_char, p_next, p_modulate };
			pending_draws.push_back(draw);
		}
		return font->draw_char(p_canvas_item, p_pos, p_char, p_next, has_outline ? outline_color : p_modulate, has_outline);
	}

	~FontDrawer() {
		for (int i = 0; i < pending_draws.size(); ++i) {
			const PendingDraw &draw = pending_draws[i];
			font->draw_char(draw.canvas_item, draw.pos, draw.chr, draw.next, draw.modulate, false);
		}
	}
};

class BitmapFont : public Font {

	GDCLASS(BitmapFont, Font);
	RES_BASE_EXTENSION("font");

#ifdef USE_TEXT_SHAPING
	hb_font_t *h_font;
#endif

	Vector<Ref<Texture> > textures;

public:
	struct Character {

		int texture_idx;
		Rect2 rect;
		float v_align;
		float h_align;
		float advance;

		Character() {
			texture_idx = 0;
			v_align = 0;
		}
	};

	struct KerningPairKey {

		union {
			struct {
				uint32_t A, B;
			};

			uint64_t pair;
		};

		_FORCE_INLINE_ bool operator<(const KerningPairKey &p_r) const { return pair < p_r.pair; }
	};

private:
	HashMap<CharType, Character> char_map;
	Map<KerningPairKey, int> kerning_map;

	float height;
	float ascent;
	bool distance_field_hint;

	void _set_chars(const PoolVector<int> &p_chars);
	PoolVector<int> _get_chars() const;
	void _set_kernings(const PoolVector<int> &p_kernings);
	PoolVector<int> _get_kernings() const;
	void _set_textures(const Vector<Variant> &p_textures);
	Vector<Variant> _get_textures() const;

	Ref<BitmapFont> fallback;

protected:
	static void _bind_methods();

public:
	Error create_from_fnt(const String &p_file);

	void set_height(float p_height);
	float get_height() const;

	void set_ascent(float p_ascent);
	float get_ascent() const;
	float get_descent() const;

	void add_texture(const Ref<Texture> &p_texture);
	void add_char(CharType p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance = -1);

	int get_character_count() const;
	Vector<CharType> get_char_keys() const;
	Character get_character(CharType p_char) const;

	bool has_character(CharType p_char) const;

	int get_texture_count() const;
	Ref<Texture> get_texture(int p_idx) const;

	void add_kerning_pair(CharType p_A, CharType p_B, int p_kerning);
	int get_kerning_pair(CharType p_A, CharType p_B) const;
	Vector<KerningPairKey> get_kerning_pair_keys() const;

	Size2 get_char_size(CharType p_char, CharType p_next = 0) const;

	void set_fallback(const Ref<BitmapFont> &p_fallback);
	Ref<BitmapFont> get_fallback() const;

	void clear();

	void set_distance_field_hint(bool p_distance_field);
	bool is_distance_field_hint() const;

	float draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1), bool p_outline = false) const;
	void draw_glyph(RID p_canvas_item, const Point2 &p_pos, uint32_t p_codepoint, const Point2 &p_offset, float p_ascent, const Color &p_modulate, bool p_outline, int p_fallback_index) const;

	virtual int get_fallback_count() const;
#ifdef USE_TEXT_SHAPING
	virtual hb_font_t *get_hb_font(int p_fallback_index) const;
#endif

	BitmapFont();
	~BitmapFont();
};

class ResourceFormatLoaderBMFont : public ResourceFormatLoader {
	GDCLASS(ResourceFormatLoaderBMFont, ResourceFormatLoader)
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#endif
