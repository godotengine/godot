/*************************************************************************/
/*  text_server.h                                                        */
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

#ifndef TEXT_SERVER_H
#define TEXT_SERVER_H

#include "core/object/reference.h"
#include "core/os/os.h"
#include "core/templates/rid.h"
#include "core/variant/variant.h"
#include "scene/resources/texture.h"

class CanvasTexture;

class TextServer : public Object {
	GDCLASS(TextServer, Object);

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
		JUSTIFICATION_AFTER_LAST_TAB = 1 << 3
	};

	enum LineBreakFlag {
		BREAK_NONE = 0,
		BREAK_MANDATORY = 1 << 4,
		BREAK_WORD_BOUND = 1 << 5,
		BREAK_GRAPHEME_BOUND = 1 << 6
		//RESERVED = 1 << 7
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
		GRAPHEME_IS_PUNCTUATION = 1 << 8 // Punctuation (can be used as word break, but not line break or justifiction).
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
		FEATURE_USE_SUPPORT_DATA = 1 << 7
	};

	enum ContourPointTag {
		CONTOUR_CURVE_TAG_ON = 0x01,
		CONTOUR_CURVE_TAG_OFF_CONIC = 0x00,
		CONTOUR_CURVE_TAG_OFF_CUBIC = 0x02
	};

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
		uint32_t index = 0; // Glyph index (font specific) or UTF-32 codepoint (for the invalid glyphs).

		bool operator==(const Glyph &p_a) const;
		bool operator!=(const Glyph &p_a) const;

		bool operator<(const Glyph &p_a) const;
		bool operator>(const Glyph &p_a) const;
	};

	struct GlyphCompare { // For line breaking reordering.
		_FORCE_INLINE_ bool operator()(const Glyph &l, const Glyph &r) const {
			if (l.start == r.start) {
				if (l.count == r.count) {
					if ((l.flags & GRAPHEME_IS_VIRTUAL) == GRAPHEME_IS_VIRTUAL) {
						return false;
					} else {
						return true;
					}
				}
				return l.count > r.count; // Sort first glyoh with count & flags, order of the rest are irrelevant.
			} else {
				return l.start < r.start;
			}
		}
	};

	struct ShapedTextData {
		/* Source data */
		RID parent; // Substring parent ShapedTextData.

		int start = 0; // Substring start offset in the parent string.
		int end = 0; // Substring end offset in the parent string.

		String text;
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
			VAlign inline_align = VALIGN_TOP;
			Rect2 rect;
		};
		Map<Variant, EmbeddedObject> objects;

		/* Shaped data */
		TextServer::Direction para_direction = DIRECTION_LTR; // Detected text direction.
		bool valid = false; // String is shaped.
		bool line_breaks_valid = false; // Line and word break flags are populated (and virtual zero width spaces inserted).
		bool justification_ops_valid = false; // Virtual elongation glyphs are added to the string.
		bool sort_valid = false;

		bool preserve_invalid = true; // Draw hex code box instead of missing characters.
		bool preserve_control = false; // Draw control characters.

		float ascent = 0.f; // Ascent for horizontal layout, 1/2 of width for vertical.
		float descent = 0.f; // Descent for horizontal layout, 1/2 of width for vertical.
		float width = 0.f; // Width for horizontal layout, height for vertical.

		float upos = 0.f;
		float uthk = 0.f;

		Vector<TextServer::Glyph> glyphs;
		Vector<TextServer::Glyph> glyphs_logical;
	};

protected:
	static void _bind_methods();

	static Vector3 hex_code_box_font_size[2];
	static Ref<CanvasTexture> hex_code_box_font_tex[2];

public:
	static void initialize_hex_code_box_fonts();
	static void finish_hex_code_box_fonts();

	virtual bool has_feature(Feature p_feature) = 0;
	virtual String get_name() const = 0;

	virtual void free(RID p_rid) = 0;
	virtual bool has(RID p_rid) = 0;
	virtual bool load_support_data(const String &p_filename) = 0;

#ifdef TOOLS_ENABLED
	virtual String get_support_data_filename() = 0;
	virtual String get_support_data_info() = 0;
	virtual bool save_support_data(const String &p_filename) = 0;
#endif

	virtual bool is_locale_right_to_left(const String &p_locale) = 0;

	virtual int32_t name_to_tag(const String &p_name) { return 0; };
	virtual String tag_to_name(int32_t p_tag) { return ""; };

	/* Font interface */
	virtual RID create_font_system(const String &p_name, int p_base_size = 16) = 0;
	virtual RID create_font_resource(const String &p_filename, int p_base_size = 16) = 0;
	virtual RID create_font_memory(const uint8_t *p_data, size_t p_size, const String &p_type, int p_base_size = 16) = 0;
	virtual RID create_font_bitmap(float p_height, float p_ascent, int p_base_size = 16) = 0;

	virtual void font_bitmap_add_texture(RID p_font, const Ref<Texture> &p_texture) = 0;
	virtual void font_bitmap_add_char(RID p_font, char32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance) = 0;
	virtual void font_bitmap_add_kerning_pair(RID p_font, char32_t p_A, char32_t p_B, int p_kerning) = 0;

	virtual float font_get_height(RID p_font, int p_size) const = 0;
	virtual float font_get_ascent(RID p_font, int p_size) const = 0;
	virtual float font_get_descent(RID p_font, int p_size) const = 0;

	virtual int font_get_spacing_space(RID p_font) const = 0;
	virtual void font_set_spacing_space(RID p_font, int p_value) = 0;

	virtual int font_get_spacing_glyph(RID p_font) const = 0;
	virtual void font_set_spacing_glyph(RID p_font, int p_value) = 0;

	virtual float font_get_underline_position(RID p_font, int p_size) const = 0;
	virtual float font_get_underline_thickness(RID p_font, int p_size) const = 0;

	virtual void font_set_antialiased(RID p_font, bool p_antialiased) = 0;
	virtual bool font_get_antialiased(RID p_font) const = 0;

	virtual Dictionary font_get_feature_list(RID p_font) const { return Dictionary(); };
	virtual Dictionary font_get_variation_list(RID p_font) const { return Dictionary(); };

	virtual void font_set_variation(RID p_font, const String &p_name, double p_value){};
	virtual double font_get_variation(RID p_font, const String &p_name) const { return 0; };

	virtual void font_set_distance_field_hint(RID p_font, bool p_distance_field) = 0;
	virtual bool font_get_distance_field_hint(RID p_font) const = 0;

	virtual void font_set_hinting(RID p_font, Hinting p_hinting) = 0;
	virtual Hinting font_get_hinting(RID p_font) const = 0;

	virtual void font_set_force_autohinter(RID p_font, bool p_enabeld) = 0;
	virtual bool font_get_force_autohinter(RID p_font) const = 0;

	virtual bool font_has_char(RID p_font, char32_t p_char) const = 0;
	virtual String font_get_supported_chars(RID p_font) const = 0;

	virtual bool font_has_outline(RID p_font) const = 0;
	virtual float font_get_base_size(RID p_font) const = 0;

	virtual bool font_is_language_supported(RID p_font, const String &p_language) const = 0;
	virtual void font_set_language_support_override(RID p_font, const String &p_language, bool p_supported) = 0;
	virtual bool font_get_language_support_override(RID p_font, const String &p_language) = 0;
	virtual void font_remove_language_support_override(RID p_font, const String &p_language) = 0;
	virtual Vector<String> font_get_language_support_overrides(RID p_font) = 0;

	virtual bool font_is_script_supported(RID p_font, const String &p_script) const = 0;
	virtual void font_set_script_support_override(RID p_font, const String &p_script, bool p_supported) = 0;
	virtual bool font_get_script_support_override(RID p_font, const String &p_script) = 0;
	virtual void font_remove_script_support_override(RID p_font, const String &p_script) = 0;
	virtual Vector<String> font_get_script_support_overrides(RID p_font) = 0;

	virtual uint32_t font_get_glyph_index(RID p_font, char32_t p_char, char32_t p_variation_selector = 0x0000) const = 0;
	virtual Vector2 font_get_glyph_advance(RID p_font, uint32_t p_index, int p_size) const = 0;
	virtual Vector2 font_get_glyph_kerning(RID p_font, uint32_t p_index_a, uint32_t p_index_b, int p_size) const = 0;

	virtual Vector2 font_draw_glyph(RID p_font, RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color = Color(1, 1, 1)) const = 0;
	virtual Vector2 font_draw_glyph_outline(RID p_font, RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color = Color(1, 1, 1)) const = 0;

	virtual bool font_get_glyph_contours(RID p_font, int p_size, uint32_t p_index, Vector<Vector3> &r_points, Vector<int32_t> &r_contours, bool &r_orientation) const = 0;

	virtual float font_get_oversampling() const = 0;
	virtual void font_set_oversampling(float p_oversampling) = 0;

	Vector2 get_hex_code_box_size(int p_size, char32_t p_index) const;
	void draw_hex_code_box(RID p_canvas, int p_size, const Vector2 &p_pos, char32_t p_index, const Color &p_color) const;

	virtual Vector<String> get_system_fonts() const = 0;

	/* Shaped text buffer interface */

	virtual RID create_shaped_text(Direction p_direction = DIRECTION_AUTO, Orientation p_orientation = ORIENTATION_HORIZONTAL) = 0;

	virtual void shaped_text_clear(RID p_shaped) = 0;

	virtual void shaped_text_set_direction(RID p_shaped, Direction p_direction = DIRECTION_AUTO) = 0;
	virtual Direction shaped_text_get_direction(RID p_shaped) const = 0;

	virtual void shaped_text_set_bidi_override(RID p_shaped, const Vector<Vector2i> &p_override) = 0;

	virtual void shaped_text_set_orientation(RID p_shaped, Orientation p_orientation = ORIENTATION_HORIZONTAL) = 0;
	virtual Orientation shaped_text_get_orientation(RID p_shaped) const = 0;

	virtual void shaped_text_set_preserve_invalid(RID p_shaped, bool p_enabled) = 0;
	virtual bool shaped_text_get_preserve_invalid(RID p_shaped) const = 0;

	virtual void shaped_text_set_preserve_control(RID p_shaped, bool p_enabled) = 0;
	virtual bool shaped_text_get_preserve_control(RID p_shaped) const = 0;

	virtual bool shaped_text_add_string(RID p_shaped, const String &p_text, const Vector<RID> &p_fonts, int p_size, const Dictionary &p_opentype_features = Dictionary(), const String &p_language = "") = 0;
	virtual bool shaped_text_add_object(RID p_shaped, Variant p_key, const Size2 &p_size, VAlign p_inline_align = VALIGN_CENTER, int p_length = 1) = 0;
	virtual bool shaped_text_resize_object(RID p_shaped, Variant p_key, const Size2 &p_size, VAlign p_inline_align = VALIGN_CENTER) = 0;

	virtual RID shaped_text_substr(RID p_shaped, int p_start, int p_length) const = 0; // Copy shaped substring (e.g. line break) without reshaping, but correctly reordered, preservers range.
	virtual RID shaped_text_get_parent(RID p_shaped) const = 0;

	virtual float shaped_text_fit_to_width(RID p_shaped, float p_width, uint8_t /*JustificationFlag*/ p_jst_flags = JUSTIFICATION_WORD_BOUND | JUSTIFICATION_KASHIDA) = 0;
	virtual float shaped_text_tab_align(RID p_shaped, const Vector<float> &p_tab_stops) = 0;

	virtual bool shaped_text_shape(RID p_shaped) = 0;
	virtual bool shaped_text_update_breaks(RID p_shaped) = 0;
	virtual bool shaped_text_update_justification_ops(RID p_shaped) = 0;

	virtual bool shaped_text_is_ready(RID p_shaped) const = 0;

	virtual Vector<Glyph> shaped_text_get_glyphs(RID p_shaped) const = 0;

	virtual Vector2i shaped_text_get_range(RID p_shaped) const = 0;

	virtual Vector<Glyph> shaped_text_sort_logical(RID p_shaped) = 0;

	virtual Vector<Vector2i> shaped_text_get_line_breaks_adv(RID p_shaped, const Vector<float> &p_width, int p_start = 0, bool p_once = true, uint8_t /*TextBreakFlag*/ p_break_flags = BREAK_MANDATORY | BREAK_WORD_BOUND) const;
	virtual Vector<Vector2i> shaped_text_get_line_breaks(RID p_shaped, float p_width, int p_start = 0, uint8_t /*TextBreakFlag*/ p_break_flags = BREAK_MANDATORY | BREAK_WORD_BOUND) const;
	virtual Vector<Vector2i> shaped_text_get_word_breaks(RID p_shaped) const;
	virtual Array shaped_text_get_objects(RID p_shaped) const = 0;
	virtual Rect2 shaped_text_get_object_rect(RID p_shaped, Variant p_key) const = 0;

	virtual Size2 shaped_text_get_size(RID p_shaped) const = 0;
	virtual float shaped_text_get_ascent(RID p_shaped) const = 0;
	virtual float shaped_text_get_descent(RID p_shaped) const = 0;
	virtual float shaped_text_get_width(RID p_shaped) const = 0;
	virtual float shaped_text_get_underline_position(RID p_shaped) const = 0;
	virtual float shaped_text_get_underline_thickness(RID p_shaped) const = 0;

	virtual Direction shaped_text_get_dominant_direciton_in_range(RID p_shaped, int p_start, int p_end) const;

	virtual void shaped_text_get_carets(RID p_shaped, int p_position, Rect2 &p_leading_caret, Direction &p_leading_dir, Rect2 &p_trailing_caret, Direction &p_trailing_dir) const;
	virtual Vector<Vector2> shaped_text_get_selection(RID p_shaped, int p_start, int p_end) const;

	virtual int shaped_text_hit_test_grapheme(RID p_shaped, float p_coords) const; // Return grapheme index.
	virtual int shaped_text_hit_test_position(RID p_shaped, float p_coords) const; // Return caret/selection position.

	virtual int shaped_text_next_grapheme_pos(RID p_shaped, int p_pos);
	virtual int shaped_text_prev_grapheme_pos(RID p_shaped, int p_pos);

	// The pen position is always placed on the baseline and moveing left to right.
	virtual void shaped_text_draw(RID p_shaped, RID p_canvas, const Vector2 &p_pos, float p_clip_l = -1.f, float p_clip_r = -1.f, const Color &p_color = Color(1, 1, 1)) const;
	virtual void shaped_text_draw_outline(RID p_shaped, RID p_canvas, const Vector2 &p_pos, float p_clip_l = -1.f, float p_clip_r = -1.f, int p_outline_size = 1, const Color &p_color = Color(1, 1, 1)) const;

	// Number conversion.
	virtual String format_number(const String &p_string, const String &p_language = "") const { return p_string; };
	virtual String parse_number(const String &p_string, const String &p_language = "") const { return p_string; };
	virtual String percent_sign(const String &p_language = "") const { return "%"; };

	/* GDScript wrappers */
	RID _create_font_memory(const PackedByteArray &p_data, const String &p_type, int p_base_size = 16);

	Dictionary _font_get_glyph_contours(RID p_font, int p_size, uint32_t p_index) const;

	Array _shaped_text_get_glyphs(RID p_shaped) const;
	Dictionary _shaped_text_get_carets(RID p_shaped, int p_position) const;

	void _shaped_text_set_bidi_override(RID p_shaped, const Array &p_override);

	Array _shaped_text_get_line_breaks_adv(RID p_shaped, const PackedFloat32Array &p_width, int p_start, bool p_once, uint8_t p_break_flags) const;
	Array _shaped_text_get_line_breaks(RID p_shaped, float p_width, int p_start, uint8_t p_break_flags) const;
	Array _shaped_text_get_word_breaks(RID p_shaped) const;

	Array _shaped_text_get_selection(RID p_shaped, int p_start, int p_end) const;

	TextServer();
	~TextServer();
};

/*************************************************************************/

class TextServerManager : public Object {
	GDCLASS(TextServerManager, Object);

public:
	typedef TextServer *(*CreateFunction)(Error &r_error, void *p_user_data);

protected:
	static void _bind_methods();

private:
	static TextServerManager *singleton;
	static TextServer *server;
	enum {
		MAX_SERVERS = 64
	};

	struct TextServerCreate {
		String name;
		CreateFunction create_function = nullptr;
		uint32_t features = 0;
		TextServer *instance = nullptr;
		void *user_data = nullptr;
	};

	static TextServerCreate server_create_functions[MAX_SERVERS];
	static int server_create_count;

public:
	_FORCE_INLINE_ static TextServerManager *get_singleton() {
		return singleton;
	}

	static void register_create_function(const String &p_name, uint32_t p_features, CreateFunction p_function, void *p_user_data);
	static int get_interface_count();
	static String get_interface_name(int p_index);
	static uint32_t get_interface_features(int p_index);
	static TextServer *initialize(int p_index, Error &r_error);
	static TextServer *get_primary_interface();

	/* GDScript wrappers */
	int _get_interface_count() const;
	String _get_interface_name(int p_index) const;
	uint32_t _get_interface_features(int p_index) const;
	TextServer *_get_interface(int p_index) const;
	Array _get_interfaces() const;
	TextServer *_find_interface(const String &p_name) const;

	bool _set_primary_interface(int p_index);
	TextServer *_get_primary_interface() const;

	TextServerManager();
	~TextServerManager();
};

/*************************************************************************/

#define TS TextServerManager::get_primary_interface()

VARIANT_ENUM_CAST(TextServer::Direction);
VARIANT_ENUM_CAST(TextServer::Orientation);
VARIANT_ENUM_CAST(TextServer::JustificationFlag);
VARIANT_ENUM_CAST(TextServer::LineBreakFlag);
VARIANT_ENUM_CAST(TextServer::GraphemeFlag);
VARIANT_ENUM_CAST(TextServer::Hinting);
VARIANT_ENUM_CAST(TextServer::Feature);
VARIANT_ENUM_CAST(TextServer::ContourPointTag);

#endif // TEXT_SERVER_H
