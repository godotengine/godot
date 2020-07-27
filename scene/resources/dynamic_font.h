/*************************************************************************/
/*  dynamic_font.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DYNAMIC_FONT_H
#define DYNAMIC_FONT_H

#include "modules/modules_enabled.gen.h"
#ifdef MODULE_FREETYPE_ENABLED

#include "core/io/resource_loader.h"
#include "core/os/mutex.h"
#include "core/os/thread_safe.h"
#include "core/pair.h"
#include "scene/resources/font.h"

#include <ft2build.h>
#include FT_FREETYPE_H

class DynamicFontAtSize;
class DynamicFont;

class DynamicFontData : public Resource {
	GDCLASS(DynamicFontData, Resource);

public:
	struct CacheID {
		union {
			struct {
				uint32_t size : 16;
				uint32_t outline_size : 8;
				uint32_t unused : 8;
			};
			uint32_t key;
		};
		bool operator<(CacheID right) const;
		CacheID() {
			key = 0;
		}
	};

	enum Hinting {
		HINTING_NONE,
		HINTING_LIGHT,
		HINTING_NORMAL
	};

	bool is_antialiased() const;
	void set_antialiased(bool p_antialiased);
	Hinting get_hinting() const;
	void set_hinting(Hinting p_hinting);

private:
	const uint8_t *font_mem;
	int font_mem_size;
	bool antialiased;
	bool force_autohinter;
	Hinting hinting;

	String font_path;
	Map<CacheID, DynamicFontAtSize *> size_cache;

	friend class DynamicFontAtSize;

	friend class DynamicFont;

	Ref<DynamicFontAtSize> _get_dynamic_font_at_size(CacheID p_cache_id);

protected:
	static void _bind_methods();

public:
	void set_font_ptr(const uint8_t *p_font_mem, int p_font_mem_size);
	void set_font_path(const String &p_path);
	String get_font_path() const;
	void set_force_autohinter(bool p_force);

	DynamicFontData();
	~DynamicFontData();
};

VARIANT_ENUM_CAST(DynamicFontData::Hinting);

class DynamicFontAtSize : public Reference {
	GDCLASS(DynamicFontAtSize, Reference);

	_THREAD_SAFE_CLASS_

	FT_Library library; /* handle to library     */
	FT_Face face; /* handle to face object */
	FT_StreamRec stream;

	float ascent;
	float descent;
	float linegap;
	float rect_margin;
	float oversampling;
	float scale_color_font;
	float underline_position;
	float underline_thickness;

	bool valid;

	struct CharTexture {
		Vector<uint8_t> imgdata;
		int texture_size;
		Vector<int> offsets;
		Ref<ImageTexture> texture;
	};

	Vector<CharTexture> textures;

	struct Character {
		bool found;
		int texture_idx;
		Rect2 rect;
		Rect2 rect_uv;
		float v_align;
		float h_align;
		float advance;

		Character() {
			texture_idx = 0;
			v_align = 0;
		}

		static Character not_found();
	};

	struct TexturePosition {
		int index;
		int x;
		int y;
	};

	const Pair<const Character *, DynamicFontAtSize *> _find_char_with_font(char32_t p_char, const Vector<Ref<DynamicFontAtSize>> &p_fallbacks) const;
	Character _make_outline_char(char32_t p_char);
	TexturePosition _find_texture_pos_for_glyph(int p_color_size, Image::Format p_image_format, int p_width, int p_height);
	Character _bitmap_to_character(FT_Bitmap bitmap, int yofs, int xofs, float advance);

	static unsigned long _ft_stream_io(FT_Stream stream, unsigned long offset, unsigned char *buffer, unsigned long count);
	static void _ft_stream_close(FT_Stream stream);

	HashMap<char32_t, Character> char_map;

	_FORCE_INLINE_ void _update_char(char32_t p_char);

	friend class DynamicFontData;
	Ref<DynamicFontData> font;
	DynamicFontData::CacheID id;

	static HashMap<String, Vector<uint8_t>> _fontdata;
	Error _load();

public:
	static float font_oversampling;

	float get_height() const;

	float get_ascent() const;
	float get_descent() const;
	float get_underline_position() const;
	float get_underline_thickness() const;

	Size2 get_char_size(char32_t p_char, char32_t p_next, const Vector<Ref<DynamicFontAtSize>> &p_fallbacks) const;
	String get_available_chars() const;

	float draw_char(RID p_canvas_item, const Point2 &p_pos, char32_t p_char, char32_t p_next, const Color &p_modulate, const Vector<Ref<DynamicFontAtSize>> &p_fallbacks, bool p_advance_only = false, bool p_outline = false) const;

	void set_texture_flags(uint32_t p_flags);
	void update_oversampling();

	DynamicFontAtSize();
	~DynamicFontAtSize();
};

///////////////

class DynamicFont : public Font {
	GDCLASS(DynamicFont, Font);

public:
	enum SpacingType {
		SPACING_TOP,
		SPACING_BOTTOM,
		SPACING_CHAR,
		SPACING_SPACE
	};

private:
	Ref<DynamicFontData> data;
	Ref<DynamicFontAtSize> data_at_size;
	Ref<DynamicFontAtSize> outline_data_at_size;

	Vector<Ref<DynamicFontData>> fallbacks;
	Vector<Ref<DynamicFontAtSize>> fallback_data_at_size;
	Vector<Ref<DynamicFontAtSize>> fallback_outline_data_at_size;

	DynamicFontData::CacheID cache_id;
	DynamicFontData::CacheID outline_cache_id;

	bool valid;
	int spacing_top;
	int spacing_bottom;
	int spacing_char;
	int spacing_space;

	Color outline_color;

protected:
	void _reload_cache();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	void set_font_data(const Ref<DynamicFontData> &p_data);
	Ref<DynamicFontData> get_font_data() const;

	void set_size(int p_size);
	int get_size() const;

	void set_outline_size(int p_size);
	int get_outline_size() const;

	void set_outline_color(Color p_color);
	Color get_outline_color() const;

	bool get_use_mipmaps() const;
	void set_use_mipmaps(bool p_enable);

	bool get_use_filter() const;
	void set_use_filter(bool p_enable);

	int get_spacing(int p_type) const;
	void set_spacing(int p_type, int p_value);

	void add_fallback(const Ref<DynamicFontData> &p_data);
	void set_fallback(int p_idx, const Ref<DynamicFontData> &p_data);
	int get_fallback_count() const;
	Ref<DynamicFontData> get_fallback(int p_idx) const;
	void remove_fallback(int p_idx);

	virtual float get_height() const override;

	virtual float get_ascent() const override;
	virtual float get_descent() const override;
	virtual float get_underline_position() const override;
	virtual float get_underline_thickness() const override;

	virtual Size2 get_char_size(char32_t p_char, char32_t p_next = 0) const override;
	String get_available_chars() const;

	virtual bool is_distance_field_hint() const override;

	virtual bool has_outline() const override;

	virtual float draw_char(RID p_canvas_item, const Point2 &p_pos, char32_t p_char, char32_t p_next = 0, const Color &p_modulate = Color(1, 1, 1), bool p_outline = false) const override;

	SelfList<DynamicFont> font_list;

	static Mutex dynamic_font_mutex;
	static SelfList<DynamicFont>::List *dynamic_fonts;

	static void initialize_dynamic_fonts();
	static void finish_dynamic_fonts();
	static void update_oversampling();

	DynamicFont();
	~DynamicFont();
};

VARIANT_ENUM_CAST(DynamicFont::SpacingType);

/////////////

class ResourceFormatLoaderDynamicFont : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, bool p_no_cache = false);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#endif

#endif
