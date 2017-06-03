/*************************************************************************/
/*  font.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef FONT_H
#define FONT_H

#include "map.h"
#include "resource.h"
#include "scene/resources/texture.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Font : public Resource {

	GDCLASS(Font, Resource);

protected:
	static void _bind_methods();

public:
	virtual float get_height() const = 0;

	virtual float get_ascent() const = 0;
	virtual float get_descent() const = 0;

	virtual Size2 get_char_size(CharType p_char, CharType p_next = 0) const = 0;
	Size2 get_string_size(const String &p_string) const;

	virtual bool is_distance_field_hint() const = 0;

	void draw(RID p_canvas_item, const Point2 &p_pos, const String &p_text, const Color &p_modulate = Color(1, 1, 1), int p_clip_w = -1) const;
	void draw_halign(RID p_canvas_item, const Point2 &p_pos, HAlign p_align, float p_width, const String &p_text, const Color &p_modulate = Color(1, 1, 1)) const;
	virtual float draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1)) const = 0;

	void update_changes();
	Font();
};

class BitmapFont : public Font {

	GDCLASS(BitmapFont, Font);
	RES_BASE_EXTENSION("fnt");

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

	float draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1)) const;

	BitmapFont();
	~BitmapFont();
};

#endif
