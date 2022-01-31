/*************************************************************************/
/*  text_server.h                                                        */
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

#ifndef TEXT_SERVER_H
#define TEXT_SERVER_H

#include "core/object/ref_counted.h"
#include "core/os/os.h"
#include "core/templates/rid.h"
#include "core/variant/native_ptr.h"
#include "core/variant/variant.h"

struct Glyph;
struct CaretInfo;

class TextServer : public RefCounted {
	GDCLASS(TextServer, RefCounted);

public:
	enum Direction {
		DIRECTION_AUTO,
		DIRECTION_LTR,
		DIRECTION_RTL
	};

	enum Orientation {
		ORIENTATION_HORIZONTAL,
		ORIENTATION_VERTICAL
	};

	enum JustificationFlag {
		JUSTIFICATION_NONE = 0,
		JUSTIFICATION_KASHIDA = 1 << 0,
		JUSTIFICATION_WORD_BOUND = 1 << 1,
		JUSTIFICATION_TRIM_EDGE_SPACES = 1 << 2,
		JUSTIFICATION_AFTER_LAST_TAB = 1 << 3,
		JUSTIFICATION_CONSTRAIN_ELLIPSIS = 1 << 4,
	};

	enum LineBreakFlag { // LineBreakFlag can be passed in the same value as the JustificationFlag, do not use the same values.
		BREAK_NONE = 0,
		BREAK_MANDATORY = 1 << 5,
		BREAK_WORD_BOUND = 1 << 6,
		BREAK_GRAPHEME_BOUND = 1 << 7,
		BREAK_WORD_BOUND_ADAPTIVE = 1 << 6 | 1 << 8,
	};

	enum TextOverrunFlag {
		OVERRUN_NO_TRIMMING = 0,
		OVERRUN_TRIM = 1 << 0,
		OVERRUN_TRIM_WORD_ONLY = 1 << 1,
		OVERRUN_ADD_ELLIPSIS = 1 << 2,
		OVERRUN_ENFORCE_ELLIPSIS = 1 << 3,
		OVERRUN_JUSTIFICATION_AWARE = 1 << 4,
	};

	enum GraphemeFlag {
		GRAPHEME_IS_VALID = 1 << 0, // Glyph is valid.
		GRAPHEME_IS_RTL = 1 << 1, // Glyph is right-to-left.
		GRAPHEME_IS_VIRTUAL = 1 << 2, // Glyph is not part of source string (added by fit_to_width function, do not affect caret movement).
		GRAPHEME_IS_SPACE = 1 << 3, // Is whitespace (for justification and word breaks).
		GRAPHEME_IS_BREAK_HARD = 1 << 4, // Is line break (mandatory break, e.g. "\n").
		GRAPHEME_IS_BREAK_SOFT = 1 << 5, // Is line break (optional break, e.g. space).
		GRAPHEME_IS_TAB = 1 << 6, // Is tab or vertical tab.
		GRAPHEME_IS_ELONGATION = 1 << 7, // Elongation (e.g. kashida), glyph can be duplicated or truncated to fit line to width.
		GRAPHEME_IS_PUNCTUATION = 1 << 8, // Punctuation, except underscore (can be used as word break, but not line break or justifiction).
		GRAPHEME_IS_UNDERSCORE = 1 << 9, // Underscore (can be used as word break).
		GRAPHEME_IS_CONNECTED = 1 << 10, // Connected to previous grapheme.
	};

	enum Hinting {
		HINTING_NONE,
		HINTING_LIGHT,
		HINTING_NORMAL
	};

	enum Feature {
		FEATURE_BIDI_LAYOUT = 1 << 0,
		FEATURE_VERTICAL_LAYOUT = 1 << 1,
		FEATURE_SHAPING = 1 << 2,
		FEATURE_KASHIDA_JUSTIFICATION = 1 << 3,
		FEATURE_BREAK_ITERATORS = 1 << 4,
		FEATURE_FONT_SYSTEM = 1 << 5,
		FEATURE_FONT_VARIABLE = 1 << 6,
		FEATURE_CONTEXT_SENSITIVE_CASE_CONVERSION = 1 << 7,
		FEATURE_USE_SUPPORT_DATA = 1 << 8,
	};

	enum ContourPointTag {
		CONTOUR_CURVE_TAG_ON = 0x01,
		CONTOUR_CURVE_TAG_OFF_CONIC = 0x00,
		CONTOUR_CURVE_TAG_OFF_CUBIC = 0x02
	};

	enum SpacingType {
		SPACING_GLYPH,
		SPACING_SPACE,
		SPACING_TOP,
		SPACING_BOTTOM,
	};

	enum FontStyle {
		FONT_BOLD = 1 << 0,
		FONT_ITALIC = 1 << 1,
		FONT_FIXED_WIDTH = 1 << 2,
	};

	void _draw_hex_code_box_number(RID p_canvas, int p_size, const Vector2 &p_pos, uint8_t p_index, const Color &p_color) const;

protected:
	struct TrimData {
		int trim_pos = -1;
		int ellipsis_pos = -1;
		Vector<Glyph> ellipsis_glyph_buf;
	};

	struct ShapedTextData {
		Mutex mutex;

		/* Source data */
		RID parent; // Substring parent ShapedTextData.

		int start = 0; // Substring start offset in the parent string.
		int end = 0; // Substring end offset in the parent string.

		String text;
		String custom_punct;
		TextServer::Direction direction = DIRECTION_LTR; // Desired text direction.
		TextServer::Orientation orientation = ORIENTATION_HORIZONTAL;

		struct Span {
			int start = -1;
			int end = -1;

			Vector<RID> fonts;
			int font_size = 0;

			Variant embedded_key;

			String language;
			Dictionary features;
		};
		Vector<Span> spans;

		struct EmbeddedObject {
			int pos = 0;
			InlineAlignment inline_align = INLINE_ALIGNMENT_CENTER;
			Rect2 rect;
		};
		Map<Variant, EmbeddedObject> objects;

		/* Shaped data */
		TextServer::Direction para_direction = DIRECTION_LTR; // Detected text direction.
		bool valid = false; // String is shaped.
		bool line_breaks_valid = false; // Line and word break flags are populated (and virtual zero width spaces inserted).
		bool justification_ops_valid = false; // Virtual elongation glyphs are added to the string.
		bool sort_valid = false;
		bool text_trimmed = false;

		bool preserve_invalid = true; // Draw hex code box instead of missing characters.
		bool preserve_control = false; // Draw control characters.

		float ascent = 0.f; // Ascent for horizontal layout, 1/2 of width for vertical.
		float descent = 0.f; // Descent for horizontal layout, 1/2 of width for vertical.
		float width = 0.f; // Width for horizontal layout, height for vertical.
		float width_trimmed = 0.f;

		float upos = 0.f;
		float uthk = 0.f;

		TrimData overrun_trim_data;
		bool fit_width_minimum_reached = false;

		Vector<Glyph> glyphs;
		Vector<Glyph> glyphs_logical;
	};

	Map<char32_t, char32_t> diacritics_map;
	void _diacritics_map_add(const String &p_from, char32_t p_to);
	void _init_diacritics_map();

	static void _bind_methods();

public:
	virtual bool has_feature(Feature p_feature) const = 0;
	virtual String get_name() const = 0;
	virtual uint32_t get_features() const = 0;

	virtual void free(RID p_rid) = 0;
	virtual bool has(RID p_rid) = 0;
	virtual bool load_support_data(const String &p_filename) = 0;

	virtual String get_support_data_filename() const = 0;
	virtual String get_support_data_info() const = 0;
	virtual bool save_support_data(const String &p_filename) const = 0;

	virtual bool is_locale_right_to_left(const String &p_locale) const = 0;

	virtual int32_t name_to_tag(const String &p_name) const { return 0; };
	virtual String tag_to_name(int32_t p_tag) const { return ""; };

	/* Font interface */
	virtual RID create_font() = 0;

	virtual void font_set_data(RID p_font_rid, const PackedByteArray &p_data) = 0;
	virtual void font_set_data_ptr(RID p_font_rid, const uint8_t *p_data_ptr, size_t p_data_size) = 0;

	virtual void font_set_style(RID p_font_rid, uint32_t /*FontStyle*/ p_style) = 0;
	virtual uint32_t /*FontStyle*/ font_get_style(RID p_font_rid) const = 0;

	virtual void font_set_name(RID p_font_rid, const String &p_name) = 0;
	virtual String font_get_name(RID p_font_rid) const = 0;

	virtual void font_set_style_name(RID p_font_rid, const String &p_name) = 0;
	virtual String font_get_style_name(RID p_font_rid) const = 0;

	virtual void font_set_antialiased(RID p_font_rid, bool p_antialiased) = 0;
	virtual bool font_is_antialiased(RID p_font_rid) const = 0;

	virtual void font_set_multichannel_signed_distance_field(RID p_font_rid, bool p_msdf) = 0;
	virtual bool font_is_multichannel_signed_distance_field(RID p_font_rid) const = 0;

	virtual void font_set_msdf_pixel_range(RID p_font_rid, int p_msdf_pixel_range) = 0;
	virtual int font_get_msdf_pixel_range(RID p_font_rid) const = 0;

	virtual void font_set_msdf_size(RID p_font_rid, int p_msdf_size) = 0;
	virtual int font_get_msdf_size(RID p_font_rid) const = 0;

	virtual void font_set_fixed_size(RID p_font_rid, int p_fixed_size) = 0;
	virtual int font_get_fixed_size(RID p_font_rid) const = 0;

	virtual void font_set_force_autohinter(RID p_font_rid, bool p_force_autohinter) = 0;
	virtual bool font_is_force_autohinter(RID p_font_rid) const = 0;

	virtual void font_set_hinting(RID p_font_rid, Hinting p_hinting) = 0;
	virtual Hinting font_get_hinting(RID p_font_rid) const = 0;

	virtual void font_set_variation_coordinates(RID p_font_rid, const Dictionary &p_variation_coordinates) = 0;
	virtual Dictionary font_get_variation_coordinates(RID p_font_rid) const = 0;

	virtual void font_set_oversampling(RID p_font_rid, float p_oversampling) = 0;
	virtual float font_get_oversampling(RID p_font_rid) const = 0;

	virtual Array font_get_size_cache_list(RID p_font_rid) const = 0;
	virtual void font_clear_size_cache(RID p_font_rid) = 0;
	virtual void font_remove_size_cache(RID p_font_rid, const Vector2i &p_size) = 0;

	virtual void font_set_ascent(RID p_font_rid, int p_size, float p_ascent) = 0;
	virtual float font_get_ascent(RID p_font_rid, int p_size) const = 0;

	virtual void font_set_descent(RID p_font_rid, int p_size, float p_descent) = 0;
	virtual float font_get_descent(RID p_font_rid, int p_size) const = 0;

	virtual void font_set_underline_position(RID p_font_rid, int p_size, float p_underline_position) = 0;
	virtual float font_get_underline_position(RID p_font_rid, int p_size) const = 0;

	virtual void font_set_underline_thickness(RID p_font_rid, int p_size, float p_underline_thickness) = 0;
	virtual float font_get_underline_thickness(RID p_font_rid, int p_size) const = 0;

	virtual void font_set_scale(RID p_font_rid, int p_size, float p_scale) = 0;
	virtual float font_get_scale(RID p_font_rid, int p_size) const = 0;

	virtual void font_set_spacing(RID p_font_rid, int p_size, SpacingType p_spacing, int p_value) = 0;
	virtual int font_get_spacing(RID p_font_rid, int p_size, SpacingType p_spacing) const = 0;

	virtual int font_get_texture_count(RID p_font_rid, const Vector2i &p_size) const = 0;
	virtual void font_clear_textures(RID p_font_rid, const Vector2i &p_size) = 0;
	virtual void font_remove_texture(RID p_font_rid, const Vector2i &p_size, int p_texture_index) = 0;

	virtual void font_set_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const Ref<Image> &p_image) = 0;
	virtual Ref<Image> font_get_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const = 0;

	virtual void font_set_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const PackedInt32Array &p_offset) = 0;
	virtual PackedInt32Array font_get_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const = 0;

	virtual Array font_get_glyph_list(RID p_font_rid, const Vector2i &p_size) const = 0;
	virtual void font_clear_glyphs(RID p_font_rid, const Vector2i &p_size) = 0;
	virtual void font_remove_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) = 0;

	virtual Vector2 font_get_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph) const = 0;
	virtual void font_set_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph, const Vector2 &p_advance) = 0;

	virtual Vector2 font_get_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const = 0;
	virtual void font_set_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset) = 0;

	virtual Vector2 font_get_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const = 0;
	virtual void font_set_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size) = 0;

	virtual Rect2 font_get_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const = 0;
	virtual void font_set_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect) = 0;

	virtual int font_get_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const = 0;
	virtual void font_set_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, int p_texture_idx) = 0;

	virtual Dictionary font_get_glyph_contours(RID p_font, int p_size, int32_t p_index) const = 0;

	virtual Array font_get_kerning_list(RID p_font_rid, int p_size) const = 0;
	virtual void font_clear_kerning_map(RID p_font_rid, int p_size) = 0;
	virtual void font_remove_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) = 0;

	virtual void font_set_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) = 0;
	virtual Vector2 font_get_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) const = 0;

	virtual int32_t font_get_glyph_index(RID p_font_rid, int p_size, char32_t p_char, char32_t p_variation_selector) const = 0;

	virtual bool font_has_char(RID p_font_rid, char32_t p_char) const = 0;
	virtual String font_get_supported_chars(RID p_font_rid) const = 0;

	virtual void font_render_range(RID p_font, const Vector2i &p_size, char32_t p_start, char32_t p_end) = 0;
	virtual void font_render_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_index) = 0;

	virtual void font_draw_glyph(RID p_font, RID p_canvas, int p_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color = Color(1, 1, 1)) const = 0;
	virtual void font_draw_glyph_outline(RID p_font, RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color = Color(1, 1, 1)) const = 0;

	virtual bool font_is_language_supported(RID p_font_rid, const String &p_language) const = 0;
	virtual void font_set_language_support_override(RID p_font_rid, const String &p_language, bool p_supported) = 0;
	virtual bool font_get_language_support_override(RID p_font_rid, const String &p_language) = 0;
	virtual void font_remove_language_support_override(RID p_font_rid, const String &p_language) = 0;
	virtual Vector<String> font_get_language_support_overrides(RID p_font_rid) = 0;

	virtual bool font_is_script_supported(RID p_font_rid, const String &p_script) const = 0;
	virtual void font_set_script_support_override(RID p_font_rid, const String &p_script, bool p_supported) = 0;
	virtual bool font_get_script_support_override(RID p_font_rid, const String &p_script) = 0;
	virtual void font_remove_script_support_override(RID p_font_rid, const String &p_script) = 0;
	virtual Vector<String> font_get_script_support_overrides(RID p_font_rid) = 0;

	virtual void font_set_opentype_feature_overrides(RID p_font_rid, const Dictionary &p_overrides) = 0;
	virtual Dictionary font_get_opentype_feature_overrides(RID p_font_rid) const = 0;

	virtual Dictionary font_supported_feature_list(RID p_font_rid) const = 0;
	virtual Dictionary font_supported_variation_list(RID p_font_rid) const = 0;

	virtual float font_get_global_oversampling() const = 0;
	virtual void font_set_global_oversampling(float p_oversampling) = 0;

	virtual Vector2 get_hex_code_box_size(int p_size, char32_t p_index) const;
	virtual void draw_hex_code_box(RID p_canvas, int p_size, const Vector2 &p_pos, char32_t p_index, const Color &p_color) const;

	/* Shaped text buffer interface */

	virtual RID create_shaped_text(Direction p_direction = DIRECTION_AUTO, Orientation p_orientation = ORIENTATION_HORIZONTAL) = 0;

	virtual void shaped_text_clear(RID p_shaped) = 0;

	virtual void shaped_text_set_direction(RID p_shaped, Direction p_direction = DIRECTION_AUTO) = 0;
	virtual Direction shaped_text_get_direction(RID p_shaped) const = 0;
	virtual Direction shaped_text_get_inferred_direction(RID p_shaped) const = 0;

	virtual void shaped_text_set_bidi_override(RID p_shaped, const Array &p_override) = 0;

	virtual void shaped_text_set_custom_punctuation(RID p_shaped, const String &p_punct) = 0;
	virtual String shaped_text_get_custom_punctuation(RID p_shaped) const = 0;

	virtual void shaped_text_set_orientation(RID p_shaped, Orientation p_orientation = ORIENTATION_HORIZONTAL) = 0;
	virtual Orientation shaped_text_get_orientation(RID p_shaped) const = 0;

	virtual void shaped_text_set_preserve_invalid(RID p_shaped, bool p_enabled) = 0;
	virtual bool shaped_text_get_preserve_invalid(RID p_shaped) const = 0;

	virtual void shaped_text_set_preserve_control(RID p_shaped, bool p_enabled) = 0;
	virtual bool shaped_text_get_preserve_control(RID p_shaped) const = 0;

	virtual bool shaped_text_add_string(RID p_shaped, const String &p_text, const Vector<RID> &p_fonts, int p_size, const Dictionary &p_opentype_features = Dictionary(), const String &p_language = "") = 0;
	virtual bool shaped_text_add_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align = INLINE_ALIGNMENT_CENTER, int p_length = 1) = 0;
	virtual bool shaped_text_resize_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align = INLINE_ALIGNMENT_CENTER) = 0;

	virtual RID shaped_text_substr(RID p_shaped, int p_start, int p_length) const = 0; // Copy shaped substring (e.g. line break) without reshaping, but correctly reordered, preservers range.
	virtual RID shaped_text_get_parent(RID p_shaped) const = 0;

	virtual float shaped_text_fit_to_width(RID p_shaped, float p_width, uint16_t /*JustificationFlag*/ p_jst_flags = JUSTIFICATION_WORD_BOUND | JUSTIFICATION_KASHIDA) = 0;
	virtual float shaped_text_tab_align(RID p_shaped, const PackedFloat32Array &p_tab_stops) = 0;

	virtual bool shaped_text_shape(RID p_shaped) = 0;
	virtual bool shaped_text_update_breaks(RID p_shaped) = 0;
	virtual bool shaped_text_update_justification_ops(RID p_shaped) = 0;

	virtual bool shaped_text_is_ready(RID p_shaped) const = 0;

	virtual const Glyph *shaped_text_get_glyphs(RID p_shaped) const = 0;
	Array _shaped_text_get_glyphs_wrapper(RID p_shaped) const;
	virtual const Glyph *shaped_text_sort_logical(RID p_shaped) = 0;
	Array _shaped_text_sort_logical_wrapper(RID p_shaped);
	virtual int shaped_text_get_glyph_count(RID p_shaped) const = 0;

	virtual Vector2i shaped_text_get_range(RID p_shaped) const = 0;

	virtual PackedInt32Array shaped_text_get_line_breaks_adv(RID p_shaped, const PackedFloat32Array &p_width, int p_start = 0, bool p_once = true, uint16_t /*TextBreakFlag*/ p_break_flags = BREAK_MANDATORY | BREAK_WORD_BOUND) const;
	virtual PackedInt32Array shaped_text_get_line_breaks(RID p_shaped, float p_width, int p_start = 0, uint16_t /*TextBreakFlag*/ p_break_flags = BREAK_MANDATORY | BREAK_WORD_BOUND) const;
	virtual PackedInt32Array shaped_text_get_word_breaks(RID p_shaped, int p_grapheme_flags = GRAPHEME_IS_SPACE | GRAPHEME_IS_PUNCTUATION) const;

	virtual int shaped_text_get_trim_pos(RID p_shaped) const = 0;
	virtual int shaped_text_get_ellipsis_pos(RID p_shaped) const = 0;
	virtual const Glyph *shaped_text_get_ellipsis_glyphs(RID p_shaped) const = 0;
	Array _shaped_text_get_ellipsis_glyphs_wrapper(RID p_shaped) const;
	virtual int shaped_text_get_ellipsis_glyph_count(RID p_shaped) const = 0;

	virtual void shaped_text_overrun_trim_to_width(RID p_shaped, float p_width, uint16_t p_trim_flags) = 0;

	virtual Array shaped_text_get_objects(RID p_shaped) const = 0;
	virtual Rect2 shaped_text_get_object_rect(RID p_shaped, Variant p_key) const = 0;

	virtual Size2 shaped_text_get_size(RID p_shaped) const = 0;
	virtual float shaped_text_get_ascent(RID p_shaped) const = 0;
	virtual float shaped_text_get_descent(RID p_shaped) const = 0;
	virtual float shaped_text_get_width(RID p_shaped) const = 0;
	virtual float shaped_text_get_underline_position(RID p_shaped) const = 0;
	virtual float shaped_text_get_underline_thickness(RID p_shaped) const = 0;

	virtual Direction shaped_text_get_dominant_direction_in_range(RID p_shaped, int p_start, int p_end) const;

	virtual CaretInfo shaped_text_get_carets(RID p_shaped, int p_position) const;
	Dictionary _shaped_text_get_carets_wrapper(RID p_shaped, int p_position) const;

	virtual Vector<Vector2> shaped_text_get_selection(RID p_shaped, int p_start, int p_end) const;

	virtual int shaped_text_hit_test_grapheme(RID p_shaped, float p_coords) const; // Return grapheme index.
	virtual int shaped_text_hit_test_position(RID p_shaped, float p_coords) const; // Return caret/selection position.

	virtual Vector2 shaped_text_get_grapheme_bounds(RID p_shaped, int p_pos) const;
	virtual int shaped_text_next_grapheme_pos(RID p_shaped, int p_pos) const;
	virtual int shaped_text_prev_grapheme_pos(RID p_shaped, int p_pos) const;

	// The pen position is always placed on the baseline and moveing left to right.
	virtual void shaped_text_draw(RID p_shaped, RID p_canvas, const Vector2 &p_pos, float p_clip_l = -1.f, float p_clip_r = -1.f, const Color &p_color = Color(1, 1, 1)) const;
	virtual void shaped_text_draw_outline(RID p_shaped, RID p_canvas, const Vector2 &p_pos, float p_clip_l = -1.f, float p_clip_r = -1.f, int p_outline_size = 1, const Color &p_color = Color(1, 1, 1)) const;

	// Number conversion.
	virtual String format_number(const String &p_string, const String &p_language = "") const { return p_string; };
	virtual String parse_number(const String &p_string, const String &p_language = "") const { return p_string; };
	virtual String percent_sign(const String &p_language = "") const { return "%"; };

	virtual String strip_diacritics(const String &p_string) const;

	// Other string operations.
	virtual String string_to_upper(const String &p_string, const String &p_language = "") const = 0;
	virtual String string_to_lower(const String &p_string, const String &p_language = "") const = 0;

	TextServer();
	~TextServer();
};

/*************************************************************************/

struct Glyph {
	int start = -1; // Start offset in the source string.
	int end = -1; // End offset in the source string.

	uint8_t count = 0; // Number of glyphs in the grapheme, set in the first glyph only.
	uint8_t repeat = 1; // Draw multiple times in the row.
	uint16_t flags = 0; // Grapheme flags (valid, rtl, virtual), set in the first glyph only.

	float x_off = 0.f; // Offset from the origin of the glyph on baseline.
	float y_off = 0.f;
	float advance = 0.f; // Advance to the next glyph along baseline(x for horizontal layout, y for vertical).

	RID font_rid; // Font resource.
	int font_size = 0; // Font size;
	int32_t index = 0; // Glyph index (font specific) or UTF-32 codepoint (for the invalid glyphs).

	bool operator==(const Glyph &p_a) const;
	bool operator!=(const Glyph &p_a) const;

	bool operator<(const Glyph &p_a) const;
	bool operator>(const Glyph &p_a) const;
};

struct CaretInfo {
	Rect2 l_caret;
	Rect2 t_caret;
	TextServer::Direction l_dir;
	TextServer::Direction t_dir;
};

struct GlyphCompare { // For line breaking reordering.
	_FORCE_INLINE_ bool operator()(const Glyph &l, const Glyph &r) const {
		if (l.start == r.start) {
			if (l.count == r.count) {
				if ((l.flags & TextServer::GRAPHEME_IS_VIRTUAL) == TextServer::GRAPHEME_IS_VIRTUAL) {
					return false;
				} else {
					return true;
				}
			}
			return l.count > r.count; // Sort first glyph with count & flags, order of the rest are irrelevant.
		} else {
			return l.start < r.start;
		}
	}
};

/*************************************************************************/

class TextServerManager : public Object {
	GDCLASS(TextServerManager, Object);

protected:
	static void _bind_methods();

private:
	static TextServerManager *singleton;

	Ref<TextServer> primary_interface;
	Vector<Ref<TextServer>> interfaces;

public:
	_FORCE_INLINE_ static TextServerManager *get_singleton() {
		return singleton;
	}

	void add_interface(const Ref<TextServer> &p_interface);
	void remove_interface(const Ref<TextServer> &p_interface);
	int get_interface_count() const;
	Ref<TextServer> get_interface(int p_index) const;
	Ref<TextServer> find_interface(const String &p_name) const;
	Array get_interfaces() const;

	_FORCE_INLINE_ Ref<TextServer> get_primary_interface() const {
		return primary_interface;
	}
	void set_primary_interface(const Ref<TextServer> &p_primary_interface);

	TextServerManager();
	~TextServerManager();
};

/*************************************************************************/

#define TS TextServerManager::get_singleton()->get_primary_interface()

VARIANT_ENUM_CAST(TextServer::Direction);
VARIANT_ENUM_CAST(TextServer::Orientation);
VARIANT_ENUM_CAST(TextServer::JustificationFlag);
VARIANT_ENUM_CAST(TextServer::LineBreakFlag);
VARIANT_ENUM_CAST(TextServer::TextOverrunFlag);
VARIANT_ENUM_CAST(TextServer::GraphemeFlag);
VARIANT_ENUM_CAST(TextServer::Hinting);
VARIANT_ENUM_CAST(TextServer::Feature);
VARIANT_ENUM_CAST(TextServer::ContourPointTag);
VARIANT_ENUM_CAST(TextServer::SpacingType);
VARIANT_ENUM_CAST(TextServer::FontStyle);

GDVIRTUAL_NATIVE_PTR(Glyph);
GDVIRTUAL_NATIVE_PTR(Glyph *);
GDVIRTUAL_NATIVE_PTR(CaretInfo);

#endif // TEXT_SERVER_H
