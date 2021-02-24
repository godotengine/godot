/*************************************************************************/
/*  font.h                                                               */
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

#ifndef FONT_H
#define FONT_H

#include "core/io/resource.h"
#include "core/templates/lru.h"
#include "core/templates/map.h"
#include "scene/resources/texture.h"
#include "servers/text_server.h"

/*************************************************************************/

class FontData : public Resource {
	GDCLASS(FontData, Resource);

public:
	enum SpacingType {
		SPACING_GLYPH,
		SPACING_SPACE,
	};

private:
	RID rid;
	int base_size = 16;
	String path;

protected:
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual void reset_state() override;

public:
	virtual RID get_rid() const override;

	void load_resource(const String &p_filename, int p_base_size = 16);
	void load_memory(const uint8_t *p_data, size_t p_size, const String &p_type, int p_base_size = 16);
	void _load_memory(const PackedByteArray &p_data, const String &p_type, int p_base_size = 16);

	void new_bitmap(float p_height, float p_ascent, int p_base_size = 16);

	void bitmap_add_texture(const Ref<Texture> &p_texture);
	void bitmap_add_char(char32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance);
	void bitmap_add_kerning_pair(char32_t p_A, char32_t p_B, int p_kerning);

	void set_data_path(const String &p_path);
	String get_data_path() const;

	float get_height(int p_size) const;
	float get_ascent(int p_size) const;
	float get_descent(int p_size) const;

	Dictionary get_feature_list() const;
	Dictionary get_variation_list() const;

	void set_variation(const String &p_name, double p_value);
	double get_variation(const String &p_name) const;

	float get_underline_position(int p_size) const;
	float get_underline_thickness(int p_size) const;

	int get_spacing(int p_type) const;
	void set_spacing(int p_type, int p_value);

	void set_antialiased(bool p_antialiased);
	bool get_antialiased() const;

	void set_distance_field_hint(bool p_distance_field);
	bool get_distance_field_hint() const;

	void set_force_autohinter(bool p_enabeld);
	bool get_force_autohinter() const;

	void set_hinting(TextServer::Hinting p_hinting);
	TextServer::Hinting get_hinting() const;

	bool has_char(char32_t p_char) const;
	String get_supported_chars() const;

	Vector2 get_glyph_advance(uint32_t p_index, int p_size) const;
	Vector2 get_glyph_kerning(uint32_t p_index_a, uint32_t p_index_b, int p_size) const;

	bool has_outline() const;
	float get_base_size() const;

	bool is_language_supported(const String &p_language) const;
	void set_language_support_override(const String &p_language, bool p_supported);
	bool get_language_support_override(const String &p_language) const;
	void remove_language_support_override(const String &p_language);
	Vector<String> get_language_support_overrides() const;

	bool is_script_supported(const String &p_script) const;
	void set_script_support_override(const String &p_script, bool p_supported);
	bool get_script_support_override(const String &p_script) const;
	void remove_script_support_override(const String &p_script);
	Vector<String> get_script_support_overrides() const;

	uint32_t get_glyph_index(char32_t p_char, char32_t p_variation_selector = 0x0000) const;

	Vector2 draw_glyph(RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color = Color(1, 1, 1)) const;
	Vector2 draw_glyph_outline(RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color = Color(1, 1, 1)) const;

	FontData();
	FontData(const String &p_filename, int p_base_size);
	FontData(const PackedByteArray &p_data, const String &p_type, int p_base_size);

	~FontData();
};

/*************************************************************************/

class TextLine;
class TextParagraph;

class Font : public Resource {
	GDCLASS(Font, Resource);

public:
	enum SpacingType {
		SPACING_TOP,
		SPACING_BOTTOM,
	};

private:
	int spacing_top = 0;
	int spacing_bottom = 0;

	mutable LRUCache<uint64_t, Ref<TextLine>> cache;
	mutable LRUCache<uint64_t, Ref<TextParagraph>> cache_wrap;

	Vector<Ref<FontData>> data;

protected:
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual void reset_state() override;

	void _data_changed();

public:
	Dictionary get_feature_list() const;

	// Font data control.
	void add_data(const Ref<FontData> &p_data);
	void set_data(int p_idx, const Ref<FontData> &p_data);
	int get_data_count() const;
	Ref<FontData> get_data(int p_idx) const;
	void remove_data(int p_idx);

	float get_height(int p_size = -1) const;
	float get_ascent(int p_size = -1) const;
	float get_descent(int p_size = -1) const;

	float get_underline_position(int p_size = -1) const;
	float get_underline_thickness(int p_size = -1) const;

	int get_spacing(int p_type) const;
	void set_spacing(int p_type, int p_value);

	// Drawing string.
	Size2 get_string_size(const String &p_text, int p_size = -1) const;
	Size2 get_multiline_string_size(const String &p_text, float p_width = -1, int p_size = -1, uint8_t p_flags = TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND) const;

	void draw_string(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HAlign p_align = HALIGN_LEFT, float p_width = -1, int p_size = -1, const Color &p_modulate = Color(1, 1, 1), int p_outline_size = 0, const Color &p_outline_modulate = Color(1, 1, 1, 0), uint8_t p_flags = TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND) const;
	void draw_multiline_string(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HAlign p_align = HALIGN_LEFT, float p_width = -1, int p_max_lines = -1, int p_size = -1, const Color &p_modulate = Color(1, 1, 1), int p_outline_size = 0, const Color &p_outline_modulate = Color(1, 1, 1, 0), uint8_t p_flags = TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND) const;

	// Helper functions.
	bool has_char(char32_t p_char) const;
	String get_supported_chars() const;

	Size2 get_char_size(char32_t p_char, char32_t p_next = 0, int p_size = -1) const;
	float draw_char(RID p_canvas_item, const Point2 &p_pos, char32_t p_char, char32_t p_next = 0, int p_size = -1, const Color &p_modulate = Color(1, 1, 1), int p_outline_size = 0, const Color &p_outline_modulate = Color(1, 1, 1, 0)) const;

	Vector<RID> get_rids() const;

	void update_changes();

	Font();
	~Font();
};

VARIANT_ENUM_CAST(FontData::SpacingType);
VARIANT_ENUM_CAST(Font::SpacingType);

/*************************************************************************/

class ResourceFormatLoaderFont : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE);
	virtual void get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#ifndef DISABLE_DEPRECATED

class ResourceFormatLoaderCompatFont : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE);
	virtual void get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#endif /* DISABLE_DEPRECATED */

#endif /* FONT_H */
