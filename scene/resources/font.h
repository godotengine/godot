/**************************************************************************/
/*  font.h                                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef FONT_H
#define FONT_H

#include "core/map.h"
#include "core/resource.h"
#include "scene/resources/texture.h"
#include "servers/visual/visual_server_canvas_helper.h"

class Font : public Resource {
	GDCLASS(Font, Resource);

protected:
	static void _bind_methods();

public:
	enum ContourPointTag {
		CONTOUR_CURVE_TAG_ON = 0x01,
		CONTOUR_CURVE_TAG_OFF_CONIC = 0x00,
		CONTOUR_CURVE_TAG_OFF_CUBIC = 0x02
	};

	virtual float get_height() const = 0;

	virtual float get_ascent() const = 0;
	virtual float get_descent() const = 0;
	virtual int get_spacing_char() const = 0;

	virtual Size2 get_char_size(CharType p_char, CharType p_next = 0) const = 0;
	Size2 get_string_size(const String &p_string) const;
	Size2 get_wordwrap_string_size(const String &p_string, float p_width) const;
	Size2 total_size_of_lines(Vector<String> p_lines);

	virtual bool is_distance_field_hint() const = 0;

	void draw(RID p_canvas_item, const Point2 &p_pos, const String &p_text, const Color &p_modulate = Color(1, 1, 1), int p_clip_w = -1, const Color &p_outline_modulate = Color(1, 1, 1)) const;
	void draw_halign(RID p_canvas_item, const Point2 &p_pos, HAlign p_align, float p_width, const String &p_text, const Color &p_modulate = Color(1, 1, 1), const Color &p_outline_modulate = Color(1, 1, 1)) const;

	virtual bool has_outline() const { return false; }
	float draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1), bool p_outline = false) const { return draw_char_ex(p_canvas_item, p_pos, p_char, p_next, p_modulate, p_outline); }
	virtual float draw_char_ex(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1), bool p_outline = false, MultiRect *p_multirect = nullptr) const = 0;

	virtual RID get_char_texture(CharType p_char, CharType p_next, bool p_outline) const = 0;
	virtual Size2 get_char_texture_size(CharType p_char, CharType p_next, bool p_outline) const = 0;

	virtual Vector2 get_char_tx_offset(CharType p_char, CharType p_next, bool p_outline) const = 0;
	virtual Size2 get_char_tx_size(CharType p_char, CharType p_next, bool p_outline) const = 0;
	virtual Rect2 get_char_tx_uv_rect(CharType p_char, CharType p_next, bool p_outline) const = 0;

	virtual Dictionary get_char_contours(CharType p_char, CharType p_next = 0) const { return Dictionary(); }

	void update_changes();
	Font();
};

// Helper class to that draws outlines immediately and draws characters in its destructor.
class FontDrawer {
	const Ref<Font> &font;
	Color outline_color;
	bool has_outline;
	MultiRect multirect;

	struct PendingDraw {
		RID canvas_item;
		Point2 pos;
		CharType chr;
		CharType next;
		Color modulate;
	};

	Vector<PendingDraw> pending_draws;

public:
	FontDrawer(const Ref<Font> &p_font, const Color &p_outline_color);

	float draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1)) {
		if (has_outline) {
			PendingDraw draw = { p_canvas_item, p_pos, p_char, p_next, p_modulate };
			pending_draws.push_back(draw);
		}
		return font->draw_char_ex(p_canvas_item, p_pos, p_char, p_next, has_outline ? outline_color : p_modulate, has_outline, &multirect);
	}
	MultiRect &get_multirect() { return multirect; }
	void flush();

	~FontDrawer();
};

class BitmapFont : public Font {
	GDCLASS(BitmapFont, Font);
	RES_BASE_EXTENSION("font");

	Vector<Ref<Texture>> textures;

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
	HashMap<int32_t, Character> char_map;
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
	int get_spacing_char() const {
		return 0;
	}

	void add_texture(const Ref<Texture> &p_texture);
	void add_char(int32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance = -1);

	int get_character_count() const;
	Vector<int32_t> get_char_keys() const;
	Character get_character(int32_t p_char) const;

	int get_texture_count() const;
	Ref<Texture> get_texture(int p_idx) const;

	void add_kerning_pair(int32_t p_A, int32_t p_B, int p_kerning);
	int get_kerning_pair(int32_t p_A, int32_t p_B) const;
	Vector<KerningPairKey> get_kerning_pair_keys() const;

	Size2 get_char_size(CharType p_char, CharType p_next = 0) const;

	void set_fallback(const Ref<BitmapFont> &p_fallback);
	Ref<BitmapFont> get_fallback() const;

	void clear();

	void set_distance_field_hint(bool p_distance_field);
	bool is_distance_field_hint() const;

	float draw_char_ex(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1), bool p_outline = false, MultiRect *p_multirect = nullptr) const;

	RID get_char_texture(CharType p_char, CharType p_next, bool p_outline) const;
	Size2 get_char_texture_size(CharType p_char, CharType p_next, bool p_outline) const;

	Vector2 get_char_tx_offset(CharType p_char, CharType p_next, bool p_outline) const;
	Size2 get_char_tx_size(CharType p_char, CharType p_next, bool p_outline) const;
	Rect2 get_char_tx_uv_rect(CharType p_char, CharType p_next, bool p_outline) const;

	BitmapFont();
	~BitmapFont();
};

class ResourceFormatLoaderBMFont : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_no_subresource_cache = false);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

VARIANT_ENUM_CAST(Font::ContourPointTag);

#endif // FONT_H
