/**************************************************************************/
/*  text_server.hpp                                                       */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>
#include <godot_cpp/variant/vector3i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Image;
class PackedFloat32Array;

class TextServer : public RefCounted {
	GDEXTENSION_CLASS(TextServer, RefCounted)

public:
	enum FontAntialiasing {
		FONT_ANTIALIASING_NONE = 0,
		FONT_ANTIALIASING_GRAY = 1,
		FONT_ANTIALIASING_LCD = 2,
	};

	enum FontLCDSubpixelLayout {
		FONT_LCD_SUBPIXEL_LAYOUT_NONE = 0,
		FONT_LCD_SUBPIXEL_LAYOUT_HRGB = 1,
		FONT_LCD_SUBPIXEL_LAYOUT_HBGR = 2,
		FONT_LCD_SUBPIXEL_LAYOUT_VRGB = 3,
		FONT_LCD_SUBPIXEL_LAYOUT_VBGR = 4,
		FONT_LCD_SUBPIXEL_LAYOUT_MAX = 5,
	};

	enum Direction {
		DIRECTION_AUTO = 0,
		DIRECTION_LTR = 1,
		DIRECTION_RTL = 2,
		DIRECTION_INHERITED = 3,
	};

	enum Orientation {
		ORIENTATION_HORIZONTAL = 0,
		ORIENTATION_VERTICAL = 1,
	};

	enum JustificationFlag : uint64_t {
		JUSTIFICATION_NONE = 0,
		JUSTIFICATION_KASHIDA = 1,
		JUSTIFICATION_WORD_BOUND = 2,
		JUSTIFICATION_TRIM_EDGE_SPACES = 4,
		JUSTIFICATION_AFTER_LAST_TAB = 8,
		JUSTIFICATION_CONSTRAIN_ELLIPSIS = 16,
		JUSTIFICATION_SKIP_LAST_LINE = 32,
		JUSTIFICATION_SKIP_LAST_LINE_WITH_VISIBLE_CHARS = 64,
		JUSTIFICATION_DO_NOT_SKIP_SINGLE_LINE = 128,
	};

	enum AutowrapMode {
		AUTOWRAP_OFF = 0,
		AUTOWRAP_ARBITRARY = 1,
		AUTOWRAP_WORD = 2,
		AUTOWRAP_WORD_SMART = 3,
	};

	enum LineBreakFlag : uint64_t {
		BREAK_NONE = 0,
		BREAK_MANDATORY = 1,
		BREAK_WORD_BOUND = 2,
		BREAK_GRAPHEME_BOUND = 4,
		BREAK_ADAPTIVE = 8,
		BREAK_TRIM_EDGE_SPACES = 16,
		BREAK_TRIM_INDENT = 32,
		BREAK_TRIM_START_EDGE_SPACES = 64,
		BREAK_TRIM_END_EDGE_SPACES = 128,
	};

	enum VisibleCharactersBehavior {
		VC_CHARS_BEFORE_SHAPING = 0,
		VC_CHARS_AFTER_SHAPING = 1,
		VC_GLYPHS_AUTO = 2,
		VC_GLYPHS_LTR = 3,
		VC_GLYPHS_RTL = 4,
	};

	enum OverrunBehavior {
		OVERRUN_NO_TRIMMING = 0,
		OVERRUN_TRIM_CHAR = 1,
		OVERRUN_TRIM_WORD = 2,
		OVERRUN_TRIM_ELLIPSIS = 3,
		OVERRUN_TRIM_WORD_ELLIPSIS = 4,
		OVERRUN_TRIM_ELLIPSIS_FORCE = 5,
		OVERRUN_TRIM_WORD_ELLIPSIS_FORCE = 6,
	};

	enum TextOverrunFlag : uint64_t {
		OVERRUN_NO_TRIM = 0,
		OVERRUN_TRIM = 1,
		OVERRUN_TRIM_WORD_ONLY = 2,
		OVERRUN_ADD_ELLIPSIS = 4,
		OVERRUN_ENFORCE_ELLIPSIS = 8,
		OVERRUN_JUSTIFICATION_AWARE = 16,
		OVERRUN_SHORT_STRING_ELLIPSIS = 32,
	};

	enum GraphemeFlag : uint64_t {
		GRAPHEME_IS_VALID = 1,
		GRAPHEME_IS_RTL = 2,
		GRAPHEME_IS_VIRTUAL = 4,
		GRAPHEME_IS_SPACE = 8,
		GRAPHEME_IS_BREAK_HARD = 16,
		GRAPHEME_IS_BREAK_SOFT = 32,
		GRAPHEME_IS_TAB = 64,
		GRAPHEME_IS_ELONGATION = 128,
		GRAPHEME_IS_PUNCTUATION = 256,
		GRAPHEME_IS_UNDERSCORE = 512,
		GRAPHEME_IS_CONNECTED = 1024,
		GRAPHEME_IS_SAFE_TO_INSERT_TATWEEL = 2048,
		GRAPHEME_IS_EMBEDDED_OBJECT = 4096,
		GRAPHEME_IS_SOFT_HYPHEN = 8192,
	};

	enum Hinting {
		HINTING_NONE = 0,
		HINTING_LIGHT = 1,
		HINTING_NORMAL = 2,
	};

	enum SubpixelPositioning {
		SUBPIXEL_POSITIONING_DISABLED = 0,
		SUBPIXEL_POSITIONING_AUTO = 1,
		SUBPIXEL_POSITIONING_ONE_HALF = 2,
		SUBPIXEL_POSITIONING_ONE_QUARTER = 3,
		SUBPIXEL_POSITIONING_ONE_HALF_MAX_SIZE = 20,
		SUBPIXEL_POSITIONING_ONE_QUARTER_MAX_SIZE = 16,
	};

	enum Feature {
		FEATURE_SIMPLE_LAYOUT = 1,
		FEATURE_BIDI_LAYOUT = 2,
		FEATURE_VERTICAL_LAYOUT = 4,
		FEATURE_SHAPING = 8,
		FEATURE_KASHIDA_JUSTIFICATION = 16,
		FEATURE_BREAK_ITERATORS = 32,
		FEATURE_FONT_BITMAP = 64,
		FEATURE_FONT_DYNAMIC = 128,
		FEATURE_FONT_MSDF = 256,
		FEATURE_FONT_SYSTEM = 512,
		FEATURE_FONT_VARIABLE = 1024,
		FEATURE_CONTEXT_SENSITIVE_CASE_CONVERSION = 2048,
		FEATURE_USE_SUPPORT_DATA = 4096,
		FEATURE_UNICODE_IDENTIFIERS = 8192,
		FEATURE_UNICODE_SECURITY = 16384,
	};

	enum ContourPointTag {
		CONTOUR_CURVE_TAG_ON = 1,
		CONTOUR_CURVE_TAG_OFF_CONIC = 0,
		CONTOUR_CURVE_TAG_OFF_CUBIC = 2,
	};

	enum SpacingType {
		SPACING_GLYPH = 0,
		SPACING_SPACE = 1,
		SPACING_TOP = 2,
		SPACING_BOTTOM = 3,
		SPACING_MAX = 4,
	};

	enum FontStyle : uint64_t {
		FONT_BOLD = 1,
		FONT_ITALIC = 2,
		FONT_FIXED_WIDTH = 4,
	};

	enum StructuredTextParser {
		STRUCTURED_TEXT_DEFAULT = 0,
		STRUCTURED_TEXT_URI = 1,
		STRUCTURED_TEXT_FILE = 2,
		STRUCTURED_TEXT_EMAIL = 3,
		STRUCTURED_TEXT_LIST = 4,
		STRUCTURED_TEXT_GDSCRIPT = 5,
		STRUCTURED_TEXT_CUSTOM = 6,
	};

	enum FixedSizeScaleMode {
		FIXED_SIZE_SCALE_DISABLE = 0,
		FIXED_SIZE_SCALE_INTEGER_ONLY = 1,
		FIXED_SIZE_SCALE_ENABLED = 2,
	};

	bool has_feature(TextServer::Feature p_feature) const;
	String get_name() const;
	int64_t get_features() const;
	bool load_support_data(const String &p_filename);
	String get_support_data_filename() const;
	String get_support_data_info() const;
	bool save_support_data(const String &p_filename) const;
	PackedByteArray get_support_data() const;
	bool is_locale_using_support_data(const String &p_locale) const;
	bool is_locale_right_to_left(const String &p_locale) const;
	int64_t name_to_tag(const String &p_name) const;
	String tag_to_name(int64_t p_tag) const;
	bool has(const RID &p_rid);
	void free_rid(const RID &p_rid);
	RID create_font();
	RID create_font_linked_variation(const RID &p_font_rid);
	void font_set_data(const RID &p_font_rid, const PackedByteArray &p_data);
	void font_set_face_index(const RID &p_font_rid, int64_t p_face_index);
	int64_t font_get_face_index(const RID &p_font_rid) const;
	int64_t font_get_face_count(const RID &p_font_rid) const;
	void font_set_style(const RID &p_font_rid, BitField<TextServer::FontStyle> p_style);
	BitField<TextServer::FontStyle> font_get_style(const RID &p_font_rid) const;
	void font_set_name(const RID &p_font_rid, const String &p_name);
	String font_get_name(const RID &p_font_rid) const;
	Dictionary font_get_ot_name_strings(const RID &p_font_rid) const;
	void font_set_style_name(const RID &p_font_rid, const String &p_name);
	String font_get_style_name(const RID &p_font_rid) const;
	void font_set_weight(const RID &p_font_rid, int64_t p_weight);
	int64_t font_get_weight(const RID &p_font_rid) const;
	void font_set_stretch(const RID &p_font_rid, int64_t p_weight);
	int64_t font_get_stretch(const RID &p_font_rid) const;
	void font_set_antialiasing(const RID &p_font_rid, TextServer::FontAntialiasing p_antialiasing);
	TextServer::FontAntialiasing font_get_antialiasing(const RID &p_font_rid) const;
	void font_set_disable_embedded_bitmaps(const RID &p_font_rid, bool p_disable_embedded_bitmaps);
	bool font_get_disable_embedded_bitmaps(const RID &p_font_rid) const;
	void font_set_generate_mipmaps(const RID &p_font_rid, bool p_generate_mipmaps);
	bool font_get_generate_mipmaps(const RID &p_font_rid) const;
	void font_set_multichannel_signed_distance_field(const RID &p_font_rid, bool p_msdf);
	bool font_is_multichannel_signed_distance_field(const RID &p_font_rid) const;
	void font_set_msdf_pixel_range(const RID &p_font_rid, int64_t p_msdf_pixel_range);
	int64_t font_get_msdf_pixel_range(const RID &p_font_rid) const;
	void font_set_msdf_size(const RID &p_font_rid, int64_t p_msdf_size);
	int64_t font_get_msdf_size(const RID &p_font_rid) const;
	void font_set_fixed_size(const RID &p_font_rid, int64_t p_fixed_size);
	int64_t font_get_fixed_size(const RID &p_font_rid) const;
	void font_set_fixed_size_scale_mode(const RID &p_font_rid, TextServer::FixedSizeScaleMode p_fixed_size_scale_mode);
	TextServer::FixedSizeScaleMode font_get_fixed_size_scale_mode(const RID &p_font_rid) const;
	void font_set_allow_system_fallback(const RID &p_font_rid, bool p_allow_system_fallback);
	bool font_is_allow_system_fallback(const RID &p_font_rid) const;
	void font_clear_system_fallback_cache();
	void font_set_force_autohinter(const RID &p_font_rid, bool p_force_autohinter);
	bool font_is_force_autohinter(const RID &p_font_rid) const;
	void font_set_modulate_color_glyphs(const RID &p_font_rid, bool p_force_autohinter);
	bool font_is_modulate_color_glyphs(const RID &p_font_rid) const;
	void font_set_hinting(const RID &p_font_rid, TextServer::Hinting p_hinting);
	TextServer::Hinting font_get_hinting(const RID &p_font_rid) const;
	void font_set_subpixel_positioning(const RID &p_font_rid, TextServer::SubpixelPositioning p_subpixel_positioning);
	TextServer::SubpixelPositioning font_get_subpixel_positioning(const RID &p_font_rid) const;
	void font_set_keep_rounding_remainders(const RID &p_font_rid, bool p_keep_rounding_remainders);
	bool font_get_keep_rounding_remainders(const RID &p_font_rid) const;
	void font_set_embolden(const RID &p_font_rid, double p_strength);
	double font_get_embolden(const RID &p_font_rid) const;
	void font_set_spacing(const RID &p_font_rid, TextServer::SpacingType p_spacing, int64_t p_value);
	int64_t font_get_spacing(const RID &p_font_rid, TextServer::SpacingType p_spacing) const;
	void font_set_baseline_offset(const RID &p_font_rid, double p_baseline_offset);
	double font_get_baseline_offset(const RID &p_font_rid) const;
	void font_set_transform(const RID &p_font_rid, const Transform2D &p_transform);
	Transform2D font_get_transform(const RID &p_font_rid) const;
	void font_set_variation_coordinates(const RID &p_font_rid, const Dictionary &p_variation_coordinates);
	Dictionary font_get_variation_coordinates(const RID &p_font_rid) const;
	void font_set_oversampling(const RID &p_font_rid, double p_oversampling);
	double font_get_oversampling(const RID &p_font_rid) const;
	TypedArray<Vector2i> font_get_size_cache_list(const RID &p_font_rid) const;
	void font_clear_size_cache(const RID &p_font_rid);
	void font_remove_size_cache(const RID &p_font_rid, const Vector2i &p_size);
	TypedArray<Dictionary> font_get_size_cache_info(const RID &p_font_rid) const;
	void font_set_ascent(const RID &p_font_rid, int64_t p_size, double p_ascent);
	double font_get_ascent(const RID &p_font_rid, int64_t p_size) const;
	void font_set_descent(const RID &p_font_rid, int64_t p_size, double p_descent);
	double font_get_descent(const RID &p_font_rid, int64_t p_size) const;
	void font_set_underline_position(const RID &p_font_rid, int64_t p_size, double p_underline_position);
	double font_get_underline_position(const RID &p_font_rid, int64_t p_size) const;
	void font_set_underline_thickness(const RID &p_font_rid, int64_t p_size, double p_underline_thickness);
	double font_get_underline_thickness(const RID &p_font_rid, int64_t p_size) const;
	void font_set_scale(const RID &p_font_rid, int64_t p_size, double p_scale);
	double font_get_scale(const RID &p_font_rid, int64_t p_size) const;
	int64_t font_get_texture_count(const RID &p_font_rid, const Vector2i &p_size) const;
	void font_clear_textures(const RID &p_font_rid, const Vector2i &p_size);
	void font_remove_texture(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index);
	void font_set_texture_image(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index, const Ref<Image> &p_image);
	Ref<Image> font_get_texture_image(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) const;
	void font_set_texture_offsets(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index, const PackedInt32Array &p_offset);
	PackedInt32Array font_get_texture_offsets(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) const;
	PackedInt32Array font_get_glyph_list(const RID &p_font_rid, const Vector2i &p_size) const;
	void font_clear_glyphs(const RID &p_font_rid, const Vector2i &p_size);
	void font_remove_glyph(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph);
	Vector2 font_get_glyph_advance(const RID &p_font_rid, int64_t p_size, int64_t p_glyph) const;
	void font_set_glyph_advance(const RID &p_font_rid, int64_t p_size, int64_t p_glyph, const Vector2 &p_advance);
	Vector2 font_get_glyph_offset(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const;
	void font_set_glyph_offset(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Vector2 &p_offset);
	Vector2 font_get_glyph_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const;
	void font_set_glyph_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Vector2 &p_gl_size);
	Rect2 font_get_glyph_uv_rect(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const;
	void font_set_glyph_uv_rect(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Rect2 &p_uv_rect);
	int64_t font_get_glyph_texture_idx(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const;
	void font_set_glyph_texture_idx(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, int64_t p_texture_idx);
	RID font_get_glyph_texture_rid(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const;
	Vector2 font_get_glyph_texture_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const;
	Dictionary font_get_glyph_contours(const RID &p_font, int64_t p_size, int64_t p_index) const;
	TypedArray<Vector2i> font_get_kerning_list(const RID &p_font_rid, int64_t p_size) const;
	void font_clear_kerning_map(const RID &p_font_rid, int64_t p_size);
	void font_remove_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair);
	void font_set_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning);
	Vector2 font_get_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair) const;
	int64_t font_get_glyph_index(const RID &p_font_rid, int64_t p_size, int64_t p_char, int64_t p_variation_selector) const;
	int64_t font_get_char_from_glyph_index(const RID &p_font_rid, int64_t p_size, int64_t p_glyph_index) const;
	bool font_has_char(const RID &p_font_rid, int64_t p_char) const;
	String font_get_supported_chars(const RID &p_font_rid) const;
	PackedInt32Array font_get_supported_glyphs(const RID &p_font_rid) const;
	void font_render_range(const RID &p_font_rid, const Vector2i &p_size, int64_t p_start, int64_t p_end);
	void font_render_glyph(const RID &p_font_rid, const Vector2i &p_size, int64_t p_index);
	void font_draw_glyph(const RID &p_font_rid, const RID &p_canvas, int64_t p_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	void font_draw_glyph_outline(const RID &p_font_rid, const RID &p_canvas, int64_t p_size, int64_t p_outline_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	bool font_is_language_supported(const RID &p_font_rid, const String &p_language) const;
	void font_set_language_support_override(const RID &p_font_rid, const String &p_language, bool p_supported);
	bool font_get_language_support_override(const RID &p_font_rid, const String &p_language);
	void font_remove_language_support_override(const RID &p_font_rid, const String &p_language);
	PackedStringArray font_get_language_support_overrides(const RID &p_font_rid);
	bool font_is_script_supported(const RID &p_font_rid, const String &p_script) const;
	void font_set_script_support_override(const RID &p_font_rid, const String &p_script, bool p_supported);
	bool font_get_script_support_override(const RID &p_font_rid, const String &p_script);
	void font_remove_script_support_override(const RID &p_font_rid, const String &p_script);
	PackedStringArray font_get_script_support_overrides(const RID &p_font_rid);
	void font_set_opentype_feature_overrides(const RID &p_font_rid, const Dictionary &p_overrides);
	Dictionary font_get_opentype_feature_overrides(const RID &p_font_rid) const;
	Dictionary font_supported_feature_list(const RID &p_font_rid) const;
	Dictionary font_supported_variation_list(const RID &p_font_rid) const;
	double font_get_global_oversampling() const;
	void font_set_global_oversampling(double p_oversampling);
	Vector2 get_hex_code_box_size(int64_t p_size, int64_t p_index) const;
	void draw_hex_code_box(const RID &p_canvas, int64_t p_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color) const;
	RID create_shaped_text(TextServer::Direction p_direction = (TextServer::Direction)0, TextServer::Orientation p_orientation = (TextServer::Orientation)0);
	void shaped_text_clear(const RID &p_rid);
	RID shaped_text_duplicate(const RID &p_rid);
	void shaped_text_set_direction(const RID &p_shaped, TextServer::Direction p_direction = (TextServer::Direction)0);
	TextServer::Direction shaped_text_get_direction(const RID &p_shaped) const;
	TextServer::Direction shaped_text_get_inferred_direction(const RID &p_shaped) const;
	void shaped_text_set_bidi_override(const RID &p_shaped, const Array &p_override);
	void shaped_text_set_custom_punctuation(const RID &p_shaped, const String &p_punct);
	String shaped_text_get_custom_punctuation(const RID &p_shaped) const;
	void shaped_text_set_custom_ellipsis(const RID &p_shaped, int64_t p_char);
	int64_t shaped_text_get_custom_ellipsis(const RID &p_shaped) const;
	void shaped_text_set_orientation(const RID &p_shaped, TextServer::Orientation p_orientation = (TextServer::Orientation)0);
	TextServer::Orientation shaped_text_get_orientation(const RID &p_shaped) const;
	void shaped_text_set_preserve_invalid(const RID &p_shaped, bool p_enabled);
	bool shaped_text_get_preserve_invalid(const RID &p_shaped) const;
	void shaped_text_set_preserve_control(const RID &p_shaped, bool p_enabled);
	bool shaped_text_get_preserve_control(const RID &p_shaped) const;
	void shaped_text_set_spacing(const RID &p_shaped, TextServer::SpacingType p_spacing, int64_t p_value);
	int64_t shaped_text_get_spacing(const RID &p_shaped, TextServer::SpacingType p_spacing) const;
	bool shaped_text_add_string(const RID &p_shaped, const String &p_text, const TypedArray<RID> &p_fonts, int64_t p_size, const Dictionary &p_opentype_features = Dictionary(), const String &p_language = String(), const Variant &p_meta = nullptr);
	bool shaped_text_add_object(const RID &p_shaped, const Variant &p_key, const Vector2 &p_size, InlineAlignment p_inline_align = (InlineAlignment)5, int64_t p_length = 1, double p_baseline = 0.0);
	bool shaped_text_resize_object(const RID &p_shaped, const Variant &p_key, const Vector2 &p_size, InlineAlignment p_inline_align = (InlineAlignment)5, double p_baseline = 0.0);
	bool shaped_text_has_object(const RID &p_shaped, const Variant &p_key) const;
	String shaped_get_text(const RID &p_shaped) const;
	int64_t shaped_get_span_count(const RID &p_shaped) const;
	Variant shaped_get_span_meta(const RID &p_shaped, int64_t p_index) const;
	Variant shaped_get_span_embedded_object(const RID &p_shaped, int64_t p_index) const;
	String shaped_get_span_text(const RID &p_shaped, int64_t p_index) const;
	Variant shaped_get_span_object(const RID &p_shaped, int64_t p_index) const;
	void shaped_set_span_update_font(const RID &p_shaped, int64_t p_index, const TypedArray<RID> &p_fonts, int64_t p_size, const Dictionary &p_opentype_features = Dictionary());
	int64_t shaped_get_run_count(const RID &p_shaped) const;
	String shaped_get_run_text(const RID &p_shaped, int64_t p_index) const;
	Vector2i shaped_get_run_range(const RID &p_shaped, int64_t p_index) const;
	RID shaped_get_run_font_rid(const RID &p_shaped, int64_t p_index) const;
	int32_t shaped_get_run_font_size(const RID &p_shaped, int64_t p_index) const;
	String shaped_get_run_language(const RID &p_shaped, int64_t p_index) const;
	TextServer::Direction shaped_get_run_direction(const RID &p_shaped, int64_t p_index) const;
	Variant shaped_get_run_object(const RID &p_shaped, int64_t p_index) const;
	RID shaped_text_substr(const RID &p_shaped, int64_t p_start, int64_t p_length) const;
	RID shaped_text_get_parent(const RID &p_shaped) const;
	double shaped_text_fit_to_width(const RID &p_shaped, double p_width, BitField<TextServer::JustificationFlag> p_justification_flags = (BitField<TextServer::JustificationFlag>)3);
	double shaped_text_tab_align(const RID &p_shaped, const PackedFloat32Array &p_tab_stops);
	bool shaped_text_shape(const RID &p_shaped);
	bool shaped_text_is_ready(const RID &p_shaped) const;
	bool shaped_text_has_visible_chars(const RID &p_shaped) const;
	TypedArray<Dictionary> shaped_text_get_glyphs(const RID &p_shaped) const;
	TypedArray<Dictionary> shaped_text_sort_logical(const RID &p_shaped);
	int64_t shaped_text_get_glyph_count(const RID &p_shaped) const;
	Vector2i shaped_text_get_range(const RID &p_shaped) const;
	PackedInt32Array shaped_text_get_line_breaks_adv(const RID &p_shaped, const PackedFloat32Array &p_width, int64_t p_start = 0, bool p_once = true, BitField<TextServer::LineBreakFlag> p_break_flags = (BitField<TextServer::LineBreakFlag>)3) const;
	PackedInt32Array shaped_text_get_line_breaks(const RID &p_shaped, double p_width, int64_t p_start = 0, BitField<TextServer::LineBreakFlag> p_break_flags = (BitField<TextServer::LineBreakFlag>)3) const;
	PackedInt32Array shaped_text_get_word_breaks(const RID &p_shaped, BitField<TextServer::GraphemeFlag> p_grapheme_flags = (BitField<TextServer::GraphemeFlag>)264, BitField<TextServer::GraphemeFlag> p_skip_grapheme_flags = (BitField<TextServer::GraphemeFlag>)4) const;
	int64_t shaped_text_get_trim_pos(const RID &p_shaped) const;
	int64_t shaped_text_get_ellipsis_pos(const RID &p_shaped) const;
	TypedArray<Dictionary> shaped_text_get_ellipsis_glyphs(const RID &p_shaped) const;
	int64_t shaped_text_get_ellipsis_glyph_count(const RID &p_shaped) const;
	void shaped_text_overrun_trim_to_width(const RID &p_shaped, double p_width = 0, BitField<TextServer::TextOverrunFlag> p_overrun_trim_flags = (BitField<TextServer::TextOverrunFlag>)0);
	Array shaped_text_get_objects(const RID &p_shaped) const;
	Rect2 shaped_text_get_object_rect(const RID &p_shaped, const Variant &p_key) const;
	Vector2i shaped_text_get_object_range(const RID &p_shaped, const Variant &p_key) const;
	int64_t shaped_text_get_object_glyph(const RID &p_shaped, const Variant &p_key) const;
	Vector2 shaped_text_get_size(const RID &p_shaped) const;
	double shaped_text_get_ascent(const RID &p_shaped) const;
	double shaped_text_get_descent(const RID &p_shaped) const;
	double shaped_text_get_width(const RID &p_shaped) const;
	double shaped_text_get_underline_position(const RID &p_shaped) const;
	double shaped_text_get_underline_thickness(const RID &p_shaped) const;
	Dictionary shaped_text_get_carets(const RID &p_shaped, int64_t p_position) const;
	PackedVector2Array shaped_text_get_selection(const RID &p_shaped, int64_t p_start, int64_t p_end) const;
	int64_t shaped_text_hit_test_grapheme(const RID &p_shaped, double p_coords) const;
	int64_t shaped_text_hit_test_position(const RID &p_shaped, double p_coords) const;
	Vector2 shaped_text_get_grapheme_bounds(const RID &p_shaped, int64_t p_pos) const;
	int64_t shaped_text_next_grapheme_pos(const RID &p_shaped, int64_t p_pos) const;
	int64_t shaped_text_prev_grapheme_pos(const RID &p_shaped, int64_t p_pos) const;
	PackedInt32Array shaped_text_get_character_breaks(const RID &p_shaped) const;
	int64_t shaped_text_next_character_pos(const RID &p_shaped, int64_t p_pos) const;
	int64_t shaped_text_prev_character_pos(const RID &p_shaped, int64_t p_pos) const;
	int64_t shaped_text_closest_character_pos(const RID &p_shaped, int64_t p_pos) const;
	void shaped_text_draw(const RID &p_shaped, const RID &p_canvas, const Vector2 &p_pos, double p_clip_l = -1, double p_clip_r = -1, const Color &p_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	void shaped_text_draw_outline(const RID &p_shaped, const RID &p_canvas, const Vector2 &p_pos, double p_clip_l = -1, double p_clip_r = -1, int64_t p_outline_size = 1, const Color &p_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	TextServer::Direction shaped_text_get_dominant_direction_in_range(const RID &p_shaped, int64_t p_start, int64_t p_end) const;
	String format_number(const String &p_number, const String &p_language = String()) const;
	String parse_number(const String &p_number, const String &p_language = String()) const;
	String percent_sign(const String &p_language = String()) const;
	PackedInt32Array string_get_word_breaks(const String &p_string, const String &p_language = String(), int64_t p_chars_per_line = 0) const;
	PackedInt32Array string_get_character_breaks(const String &p_string, const String &p_language = String()) const;
	int64_t is_confusable(const String &p_string, const PackedStringArray &p_dict) const;
	bool spoof_check(const String &p_string) const;
	String strip_diacritics(const String &p_string) const;
	bool is_valid_identifier(const String &p_string) const;
	bool is_valid_letter(uint64_t p_unicode) const;
	String string_to_upper(const String &p_string, const String &p_language = String()) const;
	String string_to_lower(const String &p_string, const String &p_language = String()) const;
	String string_to_title(const String &p_string, const String &p_language = String()) const;
	TypedArray<Vector3i> parse_structured_text(TextServer::StructuredTextParser p_parser_type, const Array &p_args, const String &p_text) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TextServer::FontAntialiasing);
VARIANT_ENUM_CAST(TextServer::FontLCDSubpixelLayout);
VARIANT_ENUM_CAST(TextServer::Direction);
VARIANT_ENUM_CAST(TextServer::Orientation);
VARIANT_BITFIELD_CAST(TextServer::JustificationFlag);
VARIANT_ENUM_CAST(TextServer::AutowrapMode);
VARIANT_BITFIELD_CAST(TextServer::LineBreakFlag);
VARIANT_ENUM_CAST(TextServer::VisibleCharactersBehavior);
VARIANT_ENUM_CAST(TextServer::OverrunBehavior);
VARIANT_BITFIELD_CAST(TextServer::TextOverrunFlag);
VARIANT_BITFIELD_CAST(TextServer::GraphemeFlag);
VARIANT_ENUM_CAST(TextServer::Hinting);
VARIANT_ENUM_CAST(TextServer::SubpixelPositioning);
VARIANT_ENUM_CAST(TextServer::Feature);
VARIANT_ENUM_CAST(TextServer::ContourPointTag);
VARIANT_ENUM_CAST(TextServer::SpacingType);
VARIANT_BITFIELD_CAST(TextServer::FontStyle);
VARIANT_ENUM_CAST(TextServer::StructuredTextParser);
VARIANT_ENUM_CAST(TextServer::FixedSizeScaleMode);

