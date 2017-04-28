/*************************************************************************/
/*  dynamic_font_stb.h                                                   */
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
#ifndef DYNAMICFONT_STB_H
#define DYNAMICFONT_STB_H

#ifndef FREETYPE_ENABLED

#include "font.h"
#include "io/resource_loader.h"
#include "stb_truetype.h"

class DynamicFontAtSize;
class DynamicFont;

class DynamicFontData : public Resource {

	GDCLASS(DynamicFontData, Resource);

	bool valid;

	PoolVector<uint8_t> font_data;
	PoolVector<uint8_t>::Read fr;
	const uint8_t *last_data_ptr;

	struct KerningPairKey {

		union {
			struct {
				uint32_t A, B;
			};

			uint64_t pair;
		};

		_FORCE_INLINE_ bool operator<(const KerningPairKey &p_r) const { return pair < p_r.pair; }
	};

	Map<KerningPairKey, int> kerning_map;

	Map<int, DynamicFontAtSize *> size_cache;

	friend class DynamicFontAtSize;

	stbtt_fontinfo info;
	int ascent;
	int descent;
	int linegap;

	void lock();
	void unlock();

	friend class DynamicFont;

	Ref<DynamicFontAtSize> _get_dynamic_font_at_size(int p_size);

public:
	void set_font_data(const PoolVector<uint8_t> &p_font);
	DynamicFontData();
	~DynamicFontData();
};

class DynamicFontAtSize : public Reference {

	GDCLASS(DynamicFontAtSize, Reference);

	int rect_margin;

	struct CharTexture {

		PoolVector<uint8_t> imgdata;
		int texture_size;
		Vector<int> offsets;
		Ref<ImageTexture> texture;
	};

	Vector<CharTexture> textures;

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

	HashMap<CharType, Character> char_map;

	_FORCE_INLINE_ void _update_char(CharType p_char);

	friend class DynamicFontData;
	Ref<DynamicFontData> font;
	float scale;
	int size;

protected:
public:
	float get_height() const;

	float get_ascent() const;
	float get_descent() const;

	Size2 get_char_size(CharType p_char, CharType p_next = 0) const;

	float draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1)) const;

	DynamicFontAtSize();
	~DynamicFontAtSize();
};

///////////////

class DynamicFont : public Font {

	GDCLASS(DynamicFont, Font);

	Ref<DynamicFontData> data;
	Ref<DynamicFontAtSize> data_at_size;
	int size;

protected:
	static void _bind_methods();

public:
	void set_font_data(const Ref<DynamicFontData> &p_data);
	Ref<DynamicFontData> get_font_data() const;

	void set_size(int p_size);
	int get_size() const;

	virtual float get_height() const;

	virtual float get_ascent() const;
	virtual float get_descent() const;

	virtual Size2 get_char_size(CharType p_char, CharType p_next = 0) const;

	virtual bool is_distance_field_hint() const;

	virtual float draw_char(RID p_canvas_item, const Point2 &p_pos, CharType p_char, CharType p_next = 0, const Color &p_modulate = Color(1, 1, 1)) const;

	DynamicFont();
	~DynamicFont();
};

/////////////

class ResourceFormatLoaderDynamicFont : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#endif
#endif // DYNAMICFONT_H
