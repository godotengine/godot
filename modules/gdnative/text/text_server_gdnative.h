/*************************************************************************/
/*  text_server_gdnative.h                                               */
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

#ifndef TEXT_SERVER_GDNATIVE_H
#define TEXT_SERVER_GDNATIVE_H

#include "modules/gdnative/gdnative.h"

#include "servers/text_server.h"

class TextServerGDNative : public TextServer {
	GDCLASS(TextServerGDNative, TextServer);

	const godot_text_interface_gdnative *interface = nullptr;
	void *data = nullptr;

protected:
	static void _bind_methods(){};

public:
	virtual bool has_feature(Feature p_feature) override;
	virtual String get_name() const override;

	virtual void free(RID p_rid) override;
	virtual bool has(RID p_rid) override;
	virtual bool load_support_data(const String &p_filename) override;

#ifdef TOOLS_ENABLED
	virtual String get_support_data_filename() override;
	virtual String get_support_data_info() override;
	virtual bool save_support_data(const String &p_filename) override;
#endif

	virtual bool is_locale_right_to_left(const String &p_locale) override;

	virtual int32_t name_to_tag(const String &p_name) const override;
	virtual String tag_to_name(int32_t p_tag) const override;

	/* Font interface */
	virtual RID create_font() override;

	virtual void font_set_data(RID p_font_rid, const PackedByteArray &p_data) override;
	virtual void font_set_data_ptr(RID p_font_rid, const uint8_t *p_data_ptr, size_t p_data_size) override;

	virtual void font_set_antialiased(RID p_font_rid, bool p_antialiased) override;
	virtual bool font_is_antialiased(RID p_font_rid) const override;

	virtual void font_set_multichannel_signed_distance_field(RID p_font_rid, bool p_msdf) override;
	virtual bool font_is_multichannel_signed_distance_field(RID p_font_rid) const override;

	virtual void font_set_msdf_pixel_range(RID p_font_rid, int p_msdf_pixel_range) override;
	virtual int font_get_msdf_pixel_range(RID p_font_rid) const override;

	virtual void font_set_msdf_size(RID p_font_rid, int p_msdf_size) override;
	virtual int font_get_msdf_size(RID p_font_rid) const override;

	virtual void font_set_fixed_size(RID p_font_rid, int p_fixed_size) override;
	virtual int font_get_fixed_size(RID p_font_rid) const override;

	virtual void font_set_force_autohinter(RID p_font_rid, bool p_force_autohinter) override;
	virtual bool font_is_force_autohinter(RID p_font_rid) const override;

	virtual void font_set_hinting(RID p_font_rid, TextServer::Hinting p_hinting) override;
	virtual TextServer::Hinting font_get_hinting(RID p_font_rid) const override;

	virtual void font_set_variation_coordinates(RID p_font_rid, const Dictionary &p_variation_coordinates) override;
	virtual Dictionary font_get_variation_coordinates(RID p_font_rid) const override;

	virtual void font_set_oversampling(RID p_font_rid, real_t p_oversampling) override;
	virtual real_t font_get_oversampling(RID p_font_rid) const override;

	virtual Array font_get_size_cache_list(RID p_font_rid) const override;
	virtual void font_clear_size_cache(RID p_font_rid) override;
	virtual void font_remove_size_cache(RID p_font_rid, const Vector2i &p_size) override;

	virtual void font_set_ascent(RID p_font_rid, int p_size, real_t p_ascent) override;
	virtual real_t font_get_ascent(RID p_font_rid, int p_size) const override;

	virtual void font_set_descent(RID p_font_rid, int p_size, real_t p_descent) override;
	virtual real_t font_get_descent(RID p_font_rid, int p_size) const override;

	virtual void font_set_underline_position(RID p_font_rid, int p_size, real_t p_underline_position) override;
	virtual real_t font_get_underline_position(RID p_font_rid, int p_size) const override;

	virtual void font_set_underline_thickness(RID p_font_rid, int p_size, real_t p_underline_thickness) override;
	virtual real_t font_get_underline_thickness(RID p_font_rid, int p_size) const override;

	virtual void font_set_scale(RID p_font_rid, int p_size, real_t p_scale) override;
	virtual real_t font_get_scale(RID p_font_rid, int p_size) const override;

	virtual void font_set_spacing(RID p_font_rid, int p_size, SpacingType p_spacing, int p_value) override;
	virtual int font_get_spacing(RID p_font_rid, int p_size, SpacingType p_spacing) const override;

	virtual int font_get_texture_count(RID p_font_rid, const Vector2i &p_size) const override;
	virtual void font_clear_textures(RID p_font_rid, const Vector2i &p_size) override;
	virtual void font_remove_texture(RID p_font_rid, const Vector2i &p_size, int p_texture_index) override;

	virtual void font_set_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const Ref<Image> &p_image) override;
	virtual Ref<Image> font_get_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const override;

	virtual void font_set_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const PackedInt32Array &p_offset) override;
	virtual PackedInt32Array font_get_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const override;

	virtual Array font_get_glyph_list(RID p_font_rid, const Vector2i &p_size) const override;
	virtual void font_clear_glyphs(RID p_font_rid, const Vector2i &p_size) override;
	virtual void font_remove_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) override;

	virtual Vector2 font_get_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph) const override;
	virtual void font_set_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph, const Vector2 &p_advance) override;

	virtual Vector2 font_get_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const override;
	virtual void font_set_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset) override;

	virtual Vector2 font_get_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const override;
	virtual void font_set_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size) override;

	virtual Rect2 font_get_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const override;
	virtual void font_set_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect) override;

	virtual int font_get_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const override;
	virtual void font_set_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, int p_texture_idx) override;

	virtual bool font_get_glyph_contours(RID p_font, int p_size, int32_t p_index, Vector<Vector3> &r_points, Vector<int32_t> &r_contours, bool &r_orientation) const override;

	virtual Array font_get_kerning_list(RID p_font_rid, int p_size) const override;
	virtual void font_clear_kerning_map(RID p_font_rid, int p_size) override;
	virtual void font_remove_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) override;

	virtual void font_set_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) override;
	virtual Vector2 font_get_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) const override;

	virtual int32_t font_get_glyph_index(RID p_font_rid, int p_size, char32_t p_char, char32_t p_variation_selector = 0) const override;

	virtual bool font_has_char(RID p_font_rid, char32_t p_char) const override;
	virtual String font_get_supported_chars(RID p_font_rid) const override;

	virtual void font_render_range(RID p_font, const Vector2i &p_size, char32_t p_start, char32_t p_end) override;
	virtual void font_render_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_index) override;

	virtual void font_draw_glyph(RID p_font, RID p_canvas, int p_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color = Color(1, 1, 1)) const override;
	virtual void font_draw_glyph_outline(RID p_font, RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color = Color(1, 1, 1)) const override;

	virtual bool font_is_language_supported(RID p_font_rid, const String &p_language) const override;
	virtual void font_set_language_support_override(RID p_font_rid, const String &p_language, bool p_supported) override;
	virtual bool font_get_language_support_override(RID p_font_rid, const String &p_language) override;
	virtual void font_remove_language_support_override(RID p_font_rid, const String &p_language) override;
	virtual Vector<String> font_get_language_support_overrides(RID p_font_rid) override;

	virtual bool font_is_script_supported(RID p_font_rid, const String &p_script) const override;
	virtual void font_set_script_support_override(RID p_font_rid, const String &p_script, bool p_supported) override;
	virtual bool font_get_script_support_override(RID p_font_rid, const String &p_script) override;
	virtual void font_remove_script_support_override(RID p_font_rid, const String &p_script) override;
	virtual Vector<String> font_get_script_support_overrides(RID p_font_rid) override;

	virtual Dictionary font_supported_feature_list(RID p_font_rid) const override;
	virtual Dictionary font_supported_variation_list(RID p_font_rid) const override;

	virtual real_t font_get_global_oversampling() const override;
	virtual void font_set_global_oversampling(real_t p_oversampling) override;

	/* Shaped text buffer interface */

	virtual RID create_shaped_text(Direction p_direction = DIRECTION_AUTO, Orientation p_orientation = ORIENTATION_HORIZONTAL) override;

	virtual void shaped_text_clear(RID p_shaped) override;

	virtual void shaped_text_set_direction(RID p_shaped, Direction p_direction = DIRECTION_AUTO) override;
	virtual Direction shaped_text_get_direction(RID p_shaped) const override;

	virtual void shaped_text_set_bidi_override(RID p_shaped, const Vector<Vector2i> &p_override) override;

	virtual void shaped_text_set_orientation(RID p_shaped, Orientation p_orientation = ORIENTATION_HORIZONTAL) override;
	virtual Orientation shaped_text_get_orientation(RID p_shaped) const override;

	virtual void shaped_text_set_preserve_invalid(RID p_shaped, bool p_enabled) override;
	virtual bool shaped_text_get_preserve_invalid(RID p_shaped) const override;

	virtual void shaped_text_set_preserve_control(RID p_shaped, bool p_enabled) override;
	virtual bool shaped_text_get_preserve_control(RID p_shaped) const override;

	virtual bool shaped_text_add_string(RID p_shaped, const String &p_text, const Vector<RID> &p_fonts, int p_size, const Dictionary &p_opentype_features = Dictionary(), const String &p_language = "") override;
	virtual bool shaped_text_add_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlign p_inline_align = INLINE_ALIGN_CENTER, int p_length = 1) override;
	virtual bool shaped_text_resize_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlign p_inline_align = INLINE_ALIGN_CENTER) override;

	virtual RID shaped_text_substr(RID p_shaped, int p_start, int p_length) const override;
	virtual RID shaped_text_get_parent(RID p_shaped) const override;

	virtual real_t shaped_text_fit_to_width(RID p_shaped, real_t p_width, uint8_t /*JustificationFlag*/ p_jst_flags = JUSTIFICATION_WORD_BOUND | JUSTIFICATION_KASHIDA) override;
	virtual real_t shaped_text_tab_align(RID p_shaped, const Vector<real_t> &p_tab_stops) override;

	virtual bool shaped_text_shape(RID p_shaped) override;
	virtual bool shaped_text_update_breaks(RID p_shaped) override;
	virtual bool shaped_text_update_justification_ops(RID p_shaped) override;

	virtual void shaped_text_overrun_trim_to_width(RID p_shaped, real_t p_width, uint8_t p_trim_flags) override;

	virtual bool shaped_text_is_ready(RID p_shaped) const override;

	virtual Vector<Glyph> shaped_text_get_glyphs(RID p_shaped) const override;

	virtual Vector2i shaped_text_get_range(RID p_shaped) const override;

	virtual Vector<Glyph> shaped_text_sort_logical(RID p_shaped) override;
	virtual Vector<Vector2i> shaped_text_get_line_breaks_adv(RID p_shaped, const Vector<real_t> &p_width, int p_start = 0, bool p_once = true, uint8_t /*TextBreakFlag*/ p_break_flags = BREAK_MANDATORY | BREAK_WORD_BOUND) const override;
	virtual Vector<Vector2i> shaped_text_get_line_breaks(RID p_shaped, real_t p_width, int p_start = 0, uint8_t p_break_flags = BREAK_MANDATORY | BREAK_WORD_BOUND) const override;
	virtual Vector<Vector2i> shaped_text_get_word_breaks(RID p_shaped, int p_grapheme_flags = GRAPHEME_IS_SPACE | GRAPHEME_IS_PUNCTUATION) const override;
	virtual Array shaped_text_get_objects(RID p_shaped) const override;
	virtual Rect2 shaped_text_get_object_rect(RID p_shaped, Variant p_key) const override;

	virtual Size2 shaped_text_get_size(RID p_shaped) const override;
	virtual real_t shaped_text_get_ascent(RID p_shaped) const override;
	virtual real_t shaped_text_get_descent(RID p_shaped) const override;
	virtual real_t shaped_text_get_width(RID p_shaped) const override;
	virtual real_t shaped_text_get_underline_position(RID p_shaped) const override;
	virtual real_t shaped_text_get_underline_thickness(RID p_shaped) const override;

	virtual String format_number(const String &p_string, const String &p_language = "") const override;
	virtual String parse_number(const String &p_string, const String &p_language = "") const override;
	virtual String percent_sign(const String &p_language = "") const override;

	static TextServer *create_func(Error &r_error, void *p_user_data);

	TextServerGDNative();
	~TextServerGDNative();
};

#endif // TEXT_SERVER_GDNATIVE_H
