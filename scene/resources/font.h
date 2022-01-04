/*************************************************************************/
/*  font.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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
	RES_BASE_EXTENSION("fontdata");

	// Font source data.
	const uint8_t *data_ptr = nullptr;
	size_t data_size = 0;
	PackedByteArray data;

	bool antialiased = true;
	bool msdf = false;
	int msdf_pixel_range = 16;
	int msdf_size = 48;
	int fixed_size = 0;
	bool force_autohinter = false;
	TextServer::Hinting hinting = TextServer::HINTING_LIGHT;
	real_t oversampling = 0.f;

	// Cache.
	mutable Vector<RID> cache;

	_FORCE_INLINE_ void _clear_cache();
	_FORCE_INLINE_ void _ensure_rid(int p_cache_index) const;

protected:
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual void reset_state() override;

public:
	// Font source data.
	virtual void set_data_ptr(const uint8_t *p_data, size_t p_size);
	virtual void set_data(const PackedByteArray &p_data);
	virtual PackedByteArray get_data() const;

	// Common properties.
	virtual void set_font_name(const String &p_name);
	virtual String get_font_name() const;

	virtual void set_font_style_name(const String &p_name);
	virtual String get_font_style_name() const;

	virtual void set_font_style(uint32_t p_style);
	virtual uint32_t get_font_style() const;

	virtual void set_antialiased(bool p_antialiased);
	virtual bool is_antialiased() const;

	virtual void set_multichannel_signed_distance_field(bool p_msdf);
	virtual bool is_multichannel_signed_distance_field() const;

	virtual void set_msdf_pixel_range(int p_msdf_pixel_range);
	virtual int get_msdf_pixel_range() const;

	virtual void set_msdf_size(int p_msdf_size);
	virtual int get_msdf_size() const;

	virtual void set_fixed_size(int p_fixed_size);
	virtual int get_fixed_size() const;

	virtual void set_force_autohinter(bool p_force_autohinter);
	virtual bool is_force_autohinter() const;

	virtual void set_hinting(TextServer::Hinting p_hinting);
	virtual TextServer::Hinting get_hinting() const;

	virtual void set_oversampling(real_t p_oversampling);
	virtual real_t get_oversampling() const;

	// Cache.
	virtual RID find_cache(const Dictionary &p_variation_coordinates) const;

	virtual int get_cache_count() const;
	virtual void clear_cache();
	virtual void remove_cache(int p_cache_index);

	virtual Array get_size_cache_list(int p_cache_index) const;
	virtual void clear_size_cache(int p_cache_index);
	virtual void remove_size_cache(int p_cache_index, const Vector2i &p_size);

	virtual void set_variation_coordinates(int p_cache_index, const Dictionary &p_variation_coordinates);
	virtual Dictionary get_variation_coordinates(int p_cache_index) const;

	virtual void set_ascent(int p_cache_index, int p_size, real_t p_ascent);
	virtual real_t get_ascent(int p_cache_index, int p_size) const;

	virtual void set_descent(int p_cache_index, int p_size, real_t p_descent);
	virtual real_t get_descent(int p_cache_index, int p_size) const;

	virtual void set_underline_position(int p_cache_index, int p_size, real_t p_underline_position);
	virtual real_t get_underline_position(int p_cache_index, int p_size) const;

	virtual void set_underline_thickness(int p_cache_index, int p_size, real_t p_underline_thickness);
	virtual real_t get_underline_thickness(int p_cache_index, int p_size) const;

	virtual void set_scale(int p_cache_index, int p_size, real_t p_scale); // Rendering scale for bitmap fonts (e.g. emoji fonts).
	virtual real_t get_scale(int p_cache_index, int p_size) const;

	virtual void set_spacing(int p_cache_index, int p_size, TextServer::SpacingType p_spacing, int p_value);
	virtual int get_spacing(int p_cache_index, int p_size, TextServer::SpacingType p_spacing) const;

	virtual int get_texture_count(int p_cache_index, const Vector2i &p_size) const;
	virtual void clear_textures(int p_cache_index, const Vector2i &p_size);
	virtual void remove_texture(int p_cache_index, const Vector2i &p_size, int p_texture_index);

	virtual void set_texture_image(int p_cache_index, const Vector2i &p_size, int p_texture_index, const Ref<Image> &p_image);
	virtual Ref<Image> get_texture_image(int p_cache_index, const Vector2i &p_size, int p_texture_index) const;

	virtual void set_texture_offsets(int p_cache_index, const Vector2i &p_size, int p_texture_index, const PackedInt32Array &p_offset);
	virtual PackedInt32Array get_texture_offsets(int p_cache_index, const Vector2i &p_size, int p_texture_index) const;

	virtual Array get_glyph_list(int p_cache_index, const Vector2i &p_size) const;
	virtual void clear_glyphs(int p_cache_index, const Vector2i &p_size);
	virtual void remove_glyph(int p_cache_index, const Vector2i &p_size, int32_t p_glyph);

	virtual void set_glyph_advance(int p_cache_index, int p_size, int32_t p_glyph, const Vector2 &p_advance);
	virtual Vector2 get_glyph_advance(int p_cache_index, int p_size, int32_t p_glyph) const;

	virtual void set_glyph_offset(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset);
	virtual Vector2 get_glyph_offset(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const;

	virtual void set_glyph_size(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size);
	virtual Vector2 get_glyph_size(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const;

	virtual void set_glyph_uv_rect(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect);
	virtual Rect2 get_glyph_uv_rect(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const;

	virtual void set_glyph_texture_idx(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, int p_texture_idx);
	virtual int get_glyph_texture_idx(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const;

	virtual Array get_kerning_list(int p_cache_index, int p_size) const;
	virtual void clear_kerning_map(int p_cache_index, int p_size);
	virtual void remove_kerning(int p_cache_index, int p_size, const Vector2i &p_glyph_pair);

	virtual void set_kerning(int p_cache_index, int p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning);
	virtual Vector2 get_kerning(int p_cache_index, int p_size, const Vector2i &p_glyph_pair) const;

	virtual void render_range(int p_cache_index, const Vector2i &p_size, char32_t p_start, char32_t p_end);
	virtual void render_glyph(int p_cache_index, const Vector2i &p_size, int32_t p_index);

	virtual RID get_cache_rid(int p_cache_index) const;

	// Language/script support override.
	virtual bool is_language_supported(const String &p_language) const;
	virtual void set_language_support_override(const String &p_language, bool p_supported);
	virtual bool get_language_support_override(const String &p_language) const;
	virtual void remove_language_support_override(const String &p_language);
	virtual Vector<String> get_language_support_overrides() const;

	virtual bool is_script_supported(const String &p_script) const;
	virtual void set_script_support_override(const String &p_script, bool p_supported);
	virtual bool get_script_support_override(const String &p_script) const;
	virtual void remove_script_support_override(const String &p_script);
	virtual Vector<String> get_script_support_overrides() const;

	// Base font properties.
	virtual bool has_char(char32_t p_char) const;
	virtual String get_supported_chars() const;

	virtual int32_t get_glyph_index(int p_size, char32_t p_char, char32_t p_variation_selector = 0x0000) const;

	virtual Dictionary get_supported_feature_list() const;
	virtual Dictionary get_supported_variation_list() const;

	FontData();
	~FontData();
};

/*************************************************************************/

class TextLine;
class TextParagraph;

class Font : public Resource {
	GDCLASS(Font, Resource);

	// Shaped string cache.
	mutable LRUCache<uint64_t, Ref<TextLine>> cache;
	mutable LRUCache<uint64_t, Ref<TextParagraph>> cache_wrap;

	// Font data cache.
	Vector<Ref<FontData>> data;
	mutable Vector<RID> rids;

	// Font config.
	Dictionary variation_coordinates;
	int spacing_bottom = 0;
	int spacing_top = 0;

	_FORCE_INLINE_ void _data_changed();
	_FORCE_INLINE_ void _ensure_rid(int p_index) const; // Find or create cache record.

protected:
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual void reset_state() override;

public:
	static const int DEFAULT_FONT_SIZE = 16;

	Dictionary get_feature_list() const;

	// Font data.
	virtual void add_data(const Ref<FontData> &p_data);
	virtual void set_data(int p_idx, const Ref<FontData> &p_data);
	virtual int get_data_count() const;
	virtual Ref<FontData> get_data(int p_idx) const;
	virtual RID get_data_rid(int p_idx) const;
	virtual void clear_data();
	virtual void remove_data(int p_idx);

	// Font configuration.
	virtual void set_variation_coordinates(const Dictionary &p_variation_coordinates);
	virtual Dictionary get_variation_coordinates() const;

	virtual void set_spacing(TextServer::SpacingType p_spacing, int p_value);
	virtual int get_spacing(TextServer::SpacingType p_spacing) const;

	// Font metrics.
	virtual real_t get_height(int p_size = DEFAULT_FONT_SIZE) const;
	virtual real_t get_ascent(int p_size = DEFAULT_FONT_SIZE) const;
	virtual real_t get_descent(int p_size = DEFAULT_FONT_SIZE) const;
	virtual real_t get_underline_position(int p_size = DEFAULT_FONT_SIZE) const;
	virtual real_t get_underline_thickness(int p_size = DEFAULT_FONT_SIZE) const;

	// Drawing string.
	virtual Size2 get_string_size(const String &p_text, int p_size = DEFAULT_FONT_SIZE, HorizontalAlignment p_alignment = HORIZONTAL_ALIGNMENT_LEFT, float p_width = -1, uint16_t p_flags = TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND) const;
	virtual Size2 get_multiline_string_size(const String &p_text, float p_width = -1, int p_size = DEFAULT_FONT_SIZE, uint16_t p_flags = TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND) const;

	virtual void draw_string(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HorizontalAlignment p_alignment = HORIZONTAL_ALIGNMENT_LEFT, float p_width = -1, int p_size = DEFAULT_FONT_SIZE, const Color &p_modulate = Color(1, 1, 1), int p_outline_size = 0, const Color &p_outline_modulate = Color(1, 1, 1, 0), uint16_t p_flags = TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND) const;
	virtual void draw_multiline_string(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HorizontalAlignment p_alignment = HORIZONTAL_ALIGNMENT_LEFT, float p_width = -1, int p_max_lines = -1, int p_size = DEFAULT_FONT_SIZE, const Color &p_modulate = Color(1, 1, 1), int p_outline_size = 0, const Color &p_outline_modulate = Color(1, 1, 1, 0), uint16_t p_flags = TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND) const;

	// Helper functions.
	virtual bool has_char(char32_t p_char) const;
	virtual String get_supported_chars() const;

	// Drawing char.
	virtual Size2 get_char_size(char32_t p_char, char32_t p_next = 0, int p_size = DEFAULT_FONT_SIZE) const;
	virtual real_t draw_char(RID p_canvas_item, const Point2 &p_pos, char32_t p_char, char32_t p_next = 0, int p_size = DEFAULT_FONT_SIZE, const Color &p_modulate = Color(1, 1, 1), int p_outline_size = 0, const Color &p_outline_modulate = Color(1, 1, 1, 0)) const;

	Vector<RID> get_rids() const;

	void update_changes();

	Font();
	~Font();
};

#endif /* FONT_H */
