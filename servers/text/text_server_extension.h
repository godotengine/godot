/*************************************************************************/
/*  text_server_extension.h                                              */
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

#ifndef TEXT_SERVER_EXTENSION_H
#define TEXT_SERVER_EXTENSION_H

#include "core/object/gdvirtual.gen.inc"
#include "core/object/script_language.h"
#include "core/os/thread_safe.h"
#include "core/variant/native_ptr.h"
#include "servers/text_server.h"

class TextServerExtension : public TextServer {
	GDCLASS(TextServerExtension, TextServer);

protected:
	_THREAD_SAFE_CLASS_

	static void _bind_methods();

public:
	virtual bool has_feature(Feature p_feature) const override;
	virtual String get_name() const override;
	virtual uint32_t get_features() const override;
	GDVIRTUAL1RC(bool, _has_feature, Feature);
	GDVIRTUAL0RC(String, _get_name);
	GDVIRTUAL0RC(uint32_t, _get_features);

	virtual void free(RID p_rid) override;
	virtual bool has(RID p_rid) override;
	virtual bool load_support_data(const String &p_filename) override;
	GDVIRTUAL1(_free, RID);
	GDVIRTUAL1R(bool, _has, RID);
	GDVIRTUAL1R(bool, _load_support_data, const String &);

	virtual String get_support_data_filename() const override;
	virtual String get_support_data_info() const override;
	virtual bool save_support_data(const String &p_filename) const override;
	GDVIRTUAL0RC(String, _get_support_data_filename);
	GDVIRTUAL0RC(String, _get_support_data_info);
	GDVIRTUAL1RC(bool, _save_support_data, const String &);

	virtual bool is_locale_right_to_left(const String &p_locale) const override;
	GDVIRTUAL1RC(bool, _is_locale_right_to_left, const String &);

	virtual int32_t name_to_tag(const String &p_name) const override;
	virtual String tag_to_name(int32_t p_tag) const override;
	GDVIRTUAL1RC(int32_t, _name_to_tag, const String &);
	GDVIRTUAL1RC(String, _tag_to_name, int32_t);

	/* Font interface */
	virtual RID create_font() override;
	GDVIRTUAL0R(RID, _create_font);

	virtual void font_set_data(RID p_font_rid, const PackedByteArray &p_data) override;
	virtual void font_set_data_ptr(RID p_font_rid, const uint8_t *p_data_ptr, size_t p_data_size) override;
	GDVIRTUAL2(_font_set_data, RID, const PackedByteArray &);
	GDVIRTUAL3(_font_set_data_ptr, RID, GDNativeConstPtr<const uint8_t>, uint64_t);

	virtual void font_set_style(RID p_font_rid, uint32_t /*FontStyle*/ p_style) override;
	virtual uint32_t /*FontStyle*/ font_get_style(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_style, RID, uint32_t);
	GDVIRTUAL1RC(uint32_t, _font_get_style, RID);

	virtual void font_set_name(RID p_font_rid, const String &p_name) override;
	virtual String font_get_name(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_name, RID, const String &);
	GDVIRTUAL1RC(String, _font_get_name, RID);

	virtual void font_set_style_name(RID p_font_rid, const String &p_name) override;
	virtual String font_get_style_name(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_style_name, RID, const String &);
	GDVIRTUAL1RC(String, _font_get_style_name, RID);

	virtual void font_set_antialiased(RID p_font_rid, bool p_antialiased) override;
	virtual bool font_is_antialiased(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_antialiased, RID, bool);
	GDVIRTUAL1RC(bool, _font_is_antialiased, RID);

	virtual void font_set_multichannel_signed_distance_field(RID p_font_rid, bool p_msdf) override;
	virtual bool font_is_multichannel_signed_distance_field(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_multichannel_signed_distance_field, RID, bool);
	GDVIRTUAL1RC(bool, _font_is_multichannel_signed_distance_field, RID);

	virtual void font_set_msdf_pixel_range(RID p_font_rid, int p_msdf_pixel_range) override;
	virtual int font_get_msdf_pixel_range(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_msdf_pixel_range, RID, int);
	GDVIRTUAL1RC(int, _font_get_msdf_pixel_range, RID);

	virtual void font_set_msdf_size(RID p_font_rid, int p_msdf_size) override;
	virtual int font_get_msdf_size(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_msdf_size, RID, int);
	GDVIRTUAL1RC(int, _font_get_msdf_size, RID);

	virtual void font_set_fixed_size(RID p_font_rid, int p_fixed_size) override;
	virtual int font_get_fixed_size(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_fixed_size, RID, int);
	GDVIRTUAL1RC(int, _font_get_fixed_size, RID);

	virtual void font_set_force_autohinter(RID p_font_rid, bool p_force_autohinter) override;
	virtual bool font_is_force_autohinter(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_force_autohinter, RID, bool);
	GDVIRTUAL1RC(bool, _font_is_force_autohinter, RID);

	virtual void font_set_hinting(RID p_font_rid, Hinting p_hinting) override;
	virtual Hinting font_get_hinting(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_hinting, RID, Hinting);
	GDVIRTUAL1RC(/*Hinting*/ int, _font_get_hinting, RID);

	virtual void font_set_variation_coordinates(RID p_font_rid, const Dictionary &p_variation_coordinates) override;
	virtual Dictionary font_get_variation_coordinates(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_variation_coordinates, RID, Dictionary);
	GDVIRTUAL1RC(Dictionary, _font_get_variation_coordinates, RID);

	virtual void font_set_oversampling(RID p_font_rid, float p_oversampling) override;
	virtual float font_get_oversampling(RID p_font_rid) const override;
	GDVIRTUAL2(_font_set_oversampling, RID, float);
	GDVIRTUAL1RC(float, _font_get_oversampling, RID);

	virtual Array font_get_size_cache_list(RID p_font_rid) const override;
	virtual void font_clear_size_cache(RID p_font_rid) override;
	virtual void font_remove_size_cache(RID p_font_rid, const Vector2i &p_size) override;
	GDVIRTUAL1RC(Array, _font_get_size_cache_list, RID);
	GDVIRTUAL1(_font_clear_size_cache, RID);
	GDVIRTUAL2(_font_remove_size_cache, RID, const Vector2i &);

	virtual void font_set_ascent(RID p_font_rid, int p_size, float p_ascent) override;
	virtual float font_get_ascent(RID p_font_rid, int p_size) const override;
	GDVIRTUAL3(_font_set_ascent, RID, int, float);
	GDVIRTUAL2RC(float, _font_get_ascent, RID, int);

	virtual void font_set_descent(RID p_font_rid, int p_size, float p_descent) override;
	virtual float font_get_descent(RID p_font_rid, int p_size) const override;
	GDVIRTUAL3(_font_set_descent, RID, int, float);
	GDVIRTUAL2RC(float, _font_get_descent, RID, int);

	virtual void font_set_underline_position(RID p_font_rid, int p_size, float p_underline_position) override;
	virtual float font_get_underline_position(RID p_font_rid, int p_size) const override;
	GDVIRTUAL3(_font_set_underline_position, RID, int, float);
	GDVIRTUAL2RC(float, _font_get_underline_position, RID, int);

	virtual void font_set_underline_thickness(RID p_font_rid, int p_size, float p_underline_thickness) override;
	virtual float font_get_underline_thickness(RID p_font_rid, int p_size) const override;
	GDVIRTUAL3(_font_set_underline_thickness, RID, int, float);
	GDVIRTUAL2RC(float, _font_get_underline_thickness, RID, int);

	virtual void font_set_scale(RID p_font_rid, int p_size, float p_scale) override;
	virtual float font_get_scale(RID p_font_rid, int p_size) const override;
	GDVIRTUAL3(_font_set_scale, RID, int, float);
	GDVIRTUAL2RC(float, _font_get_scale, RID, int);

	virtual void font_set_spacing(RID p_font_rid, int p_size, SpacingType p_spacing, int p_value) override;
	virtual int font_get_spacing(RID p_font_rid, int p_size, SpacingType p_spacing) const override;
	GDVIRTUAL4(_font_set_spacing, RID, int, SpacingType, int);
	GDVIRTUAL3RC(int, _font_get_spacing, RID, int, SpacingType);

	virtual int font_get_texture_count(RID p_font_rid, const Vector2i &p_size) const override;
	virtual void font_clear_textures(RID p_font_rid, const Vector2i &p_size) override;
	virtual void font_remove_texture(RID p_font_rid, const Vector2i &p_size, int p_texture_index) override;
	GDVIRTUAL2RC(int, _font_get_texture_count, RID, const Vector2i &);
	GDVIRTUAL2(_font_clear_textures, RID, const Vector2i &);
	GDVIRTUAL3(_font_remove_texture, RID, const Vector2i &, int);

	virtual void font_set_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const Ref<Image> &p_image) override;
	virtual Ref<Image> font_get_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const override;
	GDVIRTUAL4(_font_set_texture_image, RID, const Vector2i &, int, const Ref<Image> &);
	GDVIRTUAL3RC(Ref<Image>, _font_get_texture_image, RID, const Vector2i &, int);

	virtual void font_set_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const PackedInt32Array &p_offset) override;
	virtual PackedInt32Array font_get_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const override;
	GDVIRTUAL4(_font_set_texture_offsets, RID, const Vector2i &, int, const PackedInt32Array &);
	GDVIRTUAL3RC(PackedInt32Array, _font_get_texture_offsets, RID, const Vector2i &, int);

	virtual Array font_get_glyph_list(RID p_font_rid, const Vector2i &p_size) const override;
	virtual void font_clear_glyphs(RID p_font_rid, const Vector2i &p_size) override;
	virtual void font_remove_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) override;
	GDVIRTUAL2RC(Array, _font_get_glyph_list, RID, const Vector2i &);
	GDVIRTUAL2(_font_clear_glyphs, RID, const Vector2i &);
	GDVIRTUAL3(_font_remove_glyph, RID, const Vector2i &, int32_t);

	virtual Vector2 font_get_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph) const override;
	virtual void font_set_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph, const Vector2 &p_advance) override;
	GDVIRTUAL3RC(Vector2, _font_get_glyph_advance, RID, int, int32_t);
	GDVIRTUAL4(_font_set_glyph_advance, RID, int, int32_t, const Vector2 &);

	virtual Vector2 font_get_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const override;
	virtual void font_set_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset) override;
	GDVIRTUAL3RC(Vector2, _font_get_glyph_offset, RID, const Vector2i &, int32_t);
	GDVIRTUAL4(_font_set_glyph_offset, RID, const Vector2i &, int32_t, const Vector2 &);

	virtual Vector2 font_get_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const override;
	virtual void font_set_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size) override;
	GDVIRTUAL3RC(Vector2, _font_get_glyph_size, RID, const Vector2i &, int32_t);
	GDVIRTUAL4(_font_set_glyph_size, RID, const Vector2i &, int32_t, const Vector2 &);

	virtual Rect2 font_get_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const override;
	virtual void font_set_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect) override;
	GDVIRTUAL3RC(Rect2, _font_get_glyph_uv_rect, RID, const Vector2i &, int32_t);
	GDVIRTUAL4(_font_set_glyph_uv_rect, RID, const Vector2i &, int32_t, const Rect2 &);

	virtual int font_get_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const override;
	virtual void font_set_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, int p_texture_idx) override;
	GDVIRTUAL3RC(int, _font_get_glyph_texture_idx, RID, const Vector2i &, int32_t);
	GDVIRTUAL4(_font_set_glyph_texture_idx, RID, const Vector2i &, int32_t, int);

	virtual Dictionary font_get_glyph_contours(RID p_font, int p_size, int32_t p_index) const override;
	GDVIRTUAL3RC(Dictionary, _font_get_glyph_contours, RID, int, int32_t);

	virtual Array font_get_kerning_list(RID p_font_rid, int p_size) const override;
	virtual void font_clear_kerning_map(RID p_font_rid, int p_size) override;
	virtual void font_remove_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) override;
	GDVIRTUAL2RC(Array, _font_get_kerning_list, RID, int);
	GDVIRTUAL2(_font_clear_kerning_map, RID, int);
	GDVIRTUAL3(_font_remove_kerning, RID, int, const Vector2i &);

	virtual void font_set_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) override;
	virtual Vector2 font_get_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) const override;
	GDVIRTUAL4(_font_set_kerning, RID, int, const Vector2i &, const Vector2 &);
	GDVIRTUAL3RC(Vector2, _font_get_kerning, RID, int, const Vector2i &);

	virtual int32_t font_get_glyph_index(RID p_font_rid, int p_size, char32_t p_char, char32_t p_variation_selector = 0) const override;
	GDVIRTUAL4RC(int32_t, _font_get_glyph_index, RID, int, char32_t, char32_t);

	virtual bool font_has_char(RID p_font_rid, char32_t p_char) const override;
	virtual String font_get_supported_chars(RID p_font_rid) const override;
	GDVIRTUAL2RC(bool, _font_has_char, RID, char32_t);
	GDVIRTUAL1RC(String, _font_get_supported_chars, RID);

	virtual void font_render_range(RID p_font, const Vector2i &p_size, char32_t p_start, char32_t p_end) override;
	virtual void font_render_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_index) override;
	GDVIRTUAL4(_font_render_range, RID, const Vector2i &, char32_t, char32_t);
	GDVIRTUAL3(_font_render_glyph, RID, const Vector2i &, int32_t);

	virtual void font_draw_glyph(RID p_font, RID p_canvas, int p_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color = Color(1, 1, 1)) const override;
	virtual void font_draw_glyph_outline(RID p_font, RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color = Color(1, 1, 1)) const override;
	GDVIRTUAL6C(_font_draw_glyph, RID, RID, int, const Vector2 &, int32_t, const Color &);
	GDVIRTUAL7C(_font_draw_glyph_outline, RID, RID, int, int, const Vector2 &, int32_t, const Color &);

	virtual bool font_is_language_supported(RID p_font_rid, const String &p_language) const override;
	virtual void font_set_language_support_override(RID p_font_rid, const String &p_language, bool p_supported) override;
	virtual bool font_get_language_support_override(RID p_font_rid, const String &p_language) override;
	virtual void font_remove_language_support_override(RID p_font_rid, const String &p_language) override;
	virtual Vector<String> font_get_language_support_overrides(RID p_font_rid) override;
	GDVIRTUAL2RC(bool, _font_is_language_supported, RID, const String &);
	GDVIRTUAL3(_font_set_language_support_override, RID, const String &, bool);
	GDVIRTUAL2R(bool, _font_get_language_support_override, RID, const String &);
	GDVIRTUAL2(_font_remove_language_support_override, RID, const String &);
	GDVIRTUAL1R(Vector<String>, _font_get_language_support_overrides, RID);

	virtual bool font_is_script_supported(RID p_font_rid, const String &p_script) const override;
	virtual void font_set_script_support_override(RID p_font_rid, const String &p_script, bool p_supported) override;
	virtual bool font_get_script_support_override(RID p_font_rid, const String &p_script) override;
	virtual void font_remove_script_support_override(RID p_font_rid, const String &p_script) override;
	virtual Vector<String> font_get_script_support_overrides(RID p_font_rid) override;
	GDVIRTUAL2RC(bool, _font_is_script_supported, RID, const String &);
	GDVIRTUAL3(_font_set_script_support_override, RID, const String &, bool);
	GDVIRTUAL2R(bool, _font_get_script_support_override, RID, const String &);
	GDVIRTUAL2(_font_remove_script_support_override, RID, const String &);
	GDVIRTUAL1R(Vector<String>, _font_get_script_support_overrides, RID);

	virtual Dictionary font_supported_feature_list(RID p_font_rid) const override;
	virtual Dictionary font_supported_variation_list(RID p_font_rid) const override;
	GDVIRTUAL1RC(Dictionary, _font_supported_feature_list, RID);
	GDVIRTUAL1RC(Dictionary, _font_supported_variation_list, RID);

	virtual float font_get_global_oversampling() const override;
	virtual void font_set_global_oversampling(float p_oversampling) override;
	GDVIRTUAL0RC(float, _font_get_global_oversampling);
	GDVIRTUAL1(_font_set_global_oversampling, float);

	virtual Vector2 get_hex_code_box_size(int p_size, char32_t p_index) const override;
	virtual void draw_hex_code_box(RID p_canvas, int p_size, const Vector2 &p_pos, char32_t p_index, const Color &p_color) const override;
	GDVIRTUAL2RC(Vector2, _get_hex_code_box_size, int, char32_t);
	GDVIRTUAL5C(_draw_hex_code_box, RID, int, const Vector2 &, char32_t, const Color &);

	/* Shaped text buffer interface */

	virtual RID create_shaped_text(Direction p_direction = DIRECTION_AUTO, Orientation p_orientation = ORIENTATION_HORIZONTAL) override;
	GDVIRTUAL2R(RID, _create_shaped_text, Direction, Orientation);

	virtual void shaped_text_clear(RID p_shaped) override;
	GDVIRTUAL1(_shaped_text_clear, RID);

	virtual void shaped_text_set_direction(RID p_shaped, Direction p_direction = DIRECTION_AUTO) override;
	virtual Direction shaped_text_get_direction(RID p_shaped) const override;
	GDVIRTUAL2(_shaped_text_set_direction, RID, Direction);
	GDVIRTUAL1RC(/*Direction*/ int, _shaped_text_get_direction, RID);

	virtual void shaped_text_set_bidi_override(RID p_shaped, const Array &p_override) override;
	GDVIRTUAL2(_shaped_text_set_bidi_override, RID, const Array &);

	virtual void shaped_text_set_custom_punctuation(RID p_shaped, const String &p_punct) override;
	virtual String shaped_text_get_custom_punctuation(RID p_shaped) const override;
	GDVIRTUAL2(_shaped_text_set_custom_punctuation, RID, String);
	GDVIRTUAL1RC(String, _shaped_text_get_custom_punctuation, RID);

	virtual void shaped_text_set_orientation(RID p_shaped, Orientation p_orientation = ORIENTATION_HORIZONTAL) override;
	virtual Orientation shaped_text_get_orientation(RID p_shaped) const override;
	GDVIRTUAL2(_shaped_text_set_orientation, RID, Orientation);
	GDVIRTUAL1RC(/*Orientation*/ int, _shaped_text_get_orientation, RID);

	virtual void shaped_text_set_preserve_invalid(RID p_shaped, bool p_enabled) override;
	virtual bool shaped_text_get_preserve_invalid(RID p_shaped) const override;
	GDVIRTUAL2(_shaped_text_set_preserve_invalid, RID, bool);
	GDVIRTUAL1RC(bool, _shaped_text_get_preserve_invalid, RID);

	virtual void shaped_text_set_preserve_control(RID p_shaped, bool p_enabled) override;
	virtual bool shaped_text_get_preserve_control(RID p_shaped) const override;
	GDVIRTUAL2(_shaped_text_set_preserve_control, RID, bool);
	GDVIRTUAL1RC(bool, _shaped_text_get_preserve_control, RID);

	virtual bool shaped_text_add_string(RID p_shaped, const String &p_text, const Vector<RID> &p_fonts, int p_size, const Dictionary &p_opentype_features = Dictionary(), const String &p_language = "") override;
	virtual bool shaped_text_add_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlign p_inline_align = INLINE_ALIGN_CENTER, int p_length = 1) override;
	virtual bool shaped_text_resize_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlign p_inline_align = INLINE_ALIGN_CENTER) override;
	GDVIRTUAL6R(bool, _shaped_text_add_string, RID, const String &, const Array &, int, const Dictionary &, const String &);
	GDVIRTUAL5R(bool, _shaped_text_add_object, RID, Variant, const Size2 &, InlineAlign, int);
	GDVIRTUAL4R(bool, _shaped_text_resize_object, RID, Variant, const Size2 &, InlineAlign);

	virtual RID shaped_text_substr(RID p_shaped, int p_start, int p_length) const override;
	virtual RID shaped_text_get_parent(RID p_shaped) const override;
	GDVIRTUAL3RC(RID, _shaped_text_substr, RID, int, int);
	GDVIRTUAL1RC(RID, _shaped_text_get_parent, RID);

	virtual float shaped_text_fit_to_width(RID p_shaped, float p_width, uint16_t /*JustificationFlag*/ p_jst_flags = JUSTIFICATION_WORD_BOUND | JUSTIFICATION_KASHIDA) override;
	virtual float shaped_text_tab_align(RID p_shaped, const PackedFloat32Array &p_tab_stops) override;
	GDVIRTUAL3R(float, _shaped_text_fit_to_width, RID, float, uint16_t);
	GDVIRTUAL2R(float, _shaped_text_tab_align, RID, const PackedFloat32Array &);

	virtual bool shaped_text_shape(RID p_shaped) override;
	virtual bool shaped_text_update_breaks(RID p_shaped) override;
	virtual bool shaped_text_update_justification_ops(RID p_shaped) override;
	GDVIRTUAL1R(bool, _shaped_text_shape, RID);
	GDVIRTUAL1R(bool, _shaped_text_update_breaks, RID);
	GDVIRTUAL1R(bool, _shaped_text_update_justification_ops, RID);

	virtual bool shaped_text_is_ready(RID p_shaped) const override;
	GDVIRTUAL1RC(bool, _shaped_text_is_ready, RID);

	virtual const Glyph *shaped_text_get_glyphs(RID p_shaped) const override;
	virtual const Glyph *shaped_text_sort_logical(RID p_shaped) override;
	virtual int shaped_text_get_glyph_count(RID p_shaped) const override;
	GDVIRTUAL2C(_shaped_text_get_glyphs, RID, GDNativePtr<const Glyph *>);
	GDVIRTUAL2(_shaped_text_sort_logical, RID, GDNativePtr<const Glyph *>);
	GDVIRTUAL1RC(int, _shaped_text_get_glyph_count, RID);

	virtual Vector2i shaped_text_get_range(RID p_shaped) const override;
	GDVIRTUAL1RC(Vector2i, _shaped_text_get_range, RID);

	virtual PackedInt32Array shaped_text_get_line_breaks_adv(RID p_shaped, const PackedFloat32Array &p_width, int p_start = 0, bool p_once = true, uint16_t /*TextBreakFlag*/ p_break_flags = BREAK_MANDATORY | BREAK_WORD_BOUND) const override;
	virtual PackedInt32Array shaped_text_get_line_breaks(RID p_shaped, float p_width, int p_start = 0, uint16_t p_break_flags = BREAK_MANDATORY | BREAK_WORD_BOUND) const override;
	virtual PackedInt32Array shaped_text_get_word_breaks(RID p_shaped, int p_grapheme_flags = GRAPHEME_IS_SPACE | GRAPHEME_IS_PUNCTUATION) const override;
	GDVIRTUAL5RC(PackedInt32Array, _shaped_text_get_line_breaks_adv, RID, const PackedFloat32Array &, int, bool, uint16_t);
	GDVIRTUAL4RC(PackedInt32Array, _shaped_text_get_line_breaks, RID, float, int, uint16_t);
	GDVIRTUAL2RC(PackedInt32Array, _shaped_text_get_word_breaks, RID, int);

	virtual int shaped_text_get_trim_pos(RID p_shaped) const override;
	virtual int shaped_text_get_ellipsis_pos(RID p_shaped) const override;
	virtual const Glyph *shaped_text_get_ellipsis_glyphs(RID p_shaped) const override;
	virtual int shaped_text_get_ellipsis_glyph_count(RID p_shaped) const override;
	GDVIRTUAL1RC(int, _shaped_text_get_trim_pos, RID);
	GDVIRTUAL1RC(int, _shaped_text_get_ellipsis_pos, RID);
	GDVIRTUAL2C(_shaped_text_get_ellipsis_glyphs, RID, GDNativePtr<const Glyph *>);
	GDVIRTUAL1RC(int, _shaped_text_get_ellipsis_glyph_count, RID);

	virtual void shaped_text_overrun_trim_to_width(RID p_shaped, float p_width, uint16_t p_trim_flags) override;
	GDVIRTUAL3(_shaped_text_overrun_trim_to_width, RID, float, uint16_t);

	virtual Array shaped_text_get_objects(RID p_shaped) const override;
	virtual Rect2 shaped_text_get_object_rect(RID p_shaped, Variant p_key) const override;
	GDVIRTUAL1RC(Array, _shaped_text_get_objects, RID);
	GDVIRTUAL2RC(Rect2, _shaped_text_get_object_rect, RID, Variant);

	virtual Size2 shaped_text_get_size(RID p_shaped) const override;
	virtual float shaped_text_get_ascent(RID p_shaped) const override;
	virtual float shaped_text_get_descent(RID p_shaped) const override;
	virtual float shaped_text_get_width(RID p_shaped) const override;
	virtual float shaped_text_get_underline_position(RID p_shaped) const override;
	virtual float shaped_text_get_underline_thickness(RID p_shaped) const override;
	GDVIRTUAL1RC(Size2, _shaped_text_get_size, RID);
	GDVIRTUAL1RC(float, _shaped_text_get_ascent, RID);
	GDVIRTUAL1RC(float, _shaped_text_get_descent, RID);
	GDVIRTUAL1RC(float, _shaped_text_get_width, RID);
	GDVIRTUAL1RC(float, _shaped_text_get_underline_position, RID);
	GDVIRTUAL1RC(float, _shaped_text_get_underline_thickness, RID);

	virtual Direction shaped_text_get_dominant_direction_in_range(RID p_shaped, int p_start, int p_end) const override;
	GDVIRTUAL3RC(int, _shaped_text_get_dominant_direction_in_range, RID, int, int);

	virtual CaretInfo shaped_text_get_carets(RID p_shaped, int p_position) const override;
	virtual Vector<Vector2> shaped_text_get_selection(RID p_shaped, int p_start, int p_end) const override;
	GDVIRTUAL3C(_shaped_text_get_carets, RID, int, GDNativePtr<CaretInfo>);
	GDVIRTUAL3RC(Vector<Vector2>, _shaped_text_get_selection, RID, int, int);

	virtual int shaped_text_hit_test_grapheme(RID p_shaped, float p_coords) const override;
	virtual int shaped_text_hit_test_position(RID p_shaped, float p_coords) const override;
	GDVIRTUAL2RC(int, _shaped_text_hit_test_grapheme, RID, float);
	GDVIRTUAL2RC(int, _shaped_text_hit_test_position, RID, float);

	virtual void shaped_text_draw(RID p_shaped, RID p_canvas, const Vector2 &p_pos, float p_clip_l = -1.f, float p_clip_r = -1.f, const Color &p_color = Color(1, 1, 1)) const override;
	virtual void shaped_text_draw_outline(RID p_shaped, RID p_canvas, const Vector2 &p_pos, float p_clip_l = -1.f, float p_clip_r = -1.f, int p_outline_size = 1, const Color &p_color = Color(1, 1, 1)) const override;
	GDVIRTUAL6C(_shaped_text_draw, RID, RID, const Vector2 &, float, float, const Color &);
	GDVIRTUAL7C(_shaped_text_draw_outline, RID, RID, const Vector2 &, float, float, int, const Color &);

	virtual int shaped_text_next_grapheme_pos(RID p_shaped, int p_pos) const override;
	virtual int shaped_text_prev_grapheme_pos(RID p_shaped, int p_pos) const override;
	GDVIRTUAL2RC(int, _shaped_text_next_grapheme_pos, RID, int);
	GDVIRTUAL2RC(int, _shaped_text_prev_grapheme_pos, RID, int);

	virtual String format_number(const String &p_string, const String &p_language = "") const override;
	virtual String parse_number(const String &p_string, const String &p_language = "") const override;
	virtual String percent_sign(const String &p_language = "") const override;
	GDVIRTUAL2RC(String, _format_number, const String &, const String &);
	GDVIRTUAL2RC(String, _parse_number, const String &, const String &);
	GDVIRTUAL1RC(String, _percent_sign, const String &);

	TextServerExtension();
	~TextServerExtension();
};

#endif // TEXT_SERVER_EXTENSION_H
