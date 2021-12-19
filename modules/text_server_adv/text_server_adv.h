/*************************************************************************/
/*  text_server_adv.h                                                    */
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

#ifndef TEXT_SERVER_ADV_H
#define TEXT_SERVER_ADV_H

/*************************************************************************/
/* ICU/HarfBuzz/Graphite backed Text Server implementation with BiDi,    */
/* shaping and advanced font features support.                           */
/*************************************************************************/

#include "servers/text_server.h"

#include "core/templates/rid_owner.h"
#include "core/templates/thread_work_pool.h"
#include "scene/resources/texture.h"
#include "script_iterator.h"

#include <unicode/ubidi.h>
#include <unicode/ubrk.h>
#include <unicode/uchar.h>
#include <unicode/uclean.h>
#include <unicode/udata.h>
#include <unicode/uiter.h>
#include <unicode/uloc.h>
#include <unicode/unorm2.h>
#include <unicode/uscript.h>
#include <unicode/ustring.h>
#include <unicode/utypes.h>

#include "modules/modules_enabled.gen.h" // For freetype, msdfgen.

#ifdef MODULE_FREETYPE_ENABLED
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_TRUETYPE_TABLES_H
#include FT_STROKER_H
#include FT_ADVANCES_H
#include FT_MULTIPLE_MASTERS_H
#include FT_BBOX_H

#include <hb-ft.h>
#include <hb-ot.h>
#endif

#include <hb-icu.h>
#include <hb.h>

class TextServerAdvanced : public TextServer {
	GDCLASS(TextServerAdvanced, TextServer);
	_THREAD_SAFE_CLASS_

	static String interface_name;
	static uint32_t interface_features;

	struct NumSystemData {
		Set<StringName> lang;
		String digits;
		String percent_sign;
		String exp;
	};

	Vector<NumSystemData> num_systems;
	Map<StringName, int32_t> feature_sets;

	void _insert_num_systems_lang();
	void _insert_feature_sets();

	// ICU support data.

	uint8_t *icu_data = nullptr;

	// Font cache data.

#ifdef MODULE_FREETYPE_ENABLED
	mutable FT_Library library = nullptr;
#endif

	const int rect_range = 2;

	struct FontTexture {
		Image::Format format;
		PackedByteArray imgdata;
		int texture_w = 0;
		int texture_h = 0;
		PackedInt32Array offsets;
		Ref<ImageTexture> texture;
	};

	struct FontTexturePosition {
		int index = 0;
		int x = 0;
		int y = 0;
	};

	struct FontGlyph {
		bool found = false;
		int texture_idx = -1;
		Rect2 rect;
		Rect2 uv_rect;
		Vector2 advance;
	};

	struct FontDataForSizeAdvanced {
		float ascent = 0.f;
		float descent = 0.f;
		float underline_position = 0.f;
		float underline_thickness = 0.f;
		float scale = 1.f;
		float oversampling = 1.f;

		int spacing_glyph = 0;
		int spacing_space = 0;

		Vector2i size;

		Vector<FontTexture> textures;
		HashMap<int32_t, FontGlyph> glyph_map;
		Map<Vector2i, Vector2> kerning_map;

		hb_font_t *hb_handle = nullptr;

#ifdef MODULE_FREETYPE_ENABLED
		FT_Face face = nullptr;
		FT_StreamRec stream;
#endif

		~FontDataForSizeAdvanced() {
			if (hb_handle != nullptr) {
				hb_font_destroy(hb_handle);
			}
#ifdef MODULE_FREETYPE_ENABLED
			if (face != nullptr) {
				FT_Done_Face(face);
			}
#endif
		}
	};

	struct FontDataAdvanced {
		Mutex mutex;

		bool antialiased = true;
		bool msdf = false;
		int msdf_range = 14;
		int msdf_source_size = 48;
		int fixed_size = 0;
		bool force_autohinter = false;
		TextServer::Hinting hinting = TextServer::HINTING_LIGHT;
		Dictionary variation_coordinates;
		float oversampling = 0.f;

		uint32_t style_flags = 0;
		String font_name;
		String style_name;

		Map<Vector2i, FontDataForSizeAdvanced *> cache;

		bool face_init = false;
		Set<uint32_t> supported_scripts;
		Dictionary supported_features;
		Dictionary supported_varaitions;

		// Language/script support override.
		Map<String, bool> language_support_overrides;
		Map<String, bool> script_support_overrides;

		PackedByteArray data;
		const uint8_t *data_ptr;
		size_t data_size;
		mutable ThreadWorkPool work_pool;

		~FontDataAdvanced() {
			work_pool.finish();
			for (const Map<Vector2i, FontDataForSizeAdvanced *>::Element *E = cache.front(); E; E = E->next()) {
				memdelete(E->get());
			}
			cache.clear();
		}
	};

	_FORCE_INLINE_ FontTexturePosition find_texture_pos_for_glyph(FontDataForSizeAdvanced *p_data, int p_color_size, Image::Format p_image_format, int p_width, int p_height) const;
#ifdef MODULE_MSDFGEN_ENABLED
	_FORCE_INLINE_ FontGlyph rasterize_msdf(FontDataAdvanced *p_font_data, FontDataForSizeAdvanced *p_data, int p_pixel_range, int p_rect_margin, FT_Outline *outline, const Vector2 &advance) const;
#endif
#ifdef MODULE_FREETYPE_ENABLED
	_FORCE_INLINE_ FontGlyph rasterize_bitmap(FontDataForSizeAdvanced *p_data, int p_rect_margin, FT_Bitmap bitmap, int yofs, int xofs, const Vector2 &advance) const;
#endif
	_FORCE_INLINE_ bool _ensure_glyph(FontDataAdvanced *p_font_data, const Vector2i &p_size, int32_t p_glyph) const;
	_FORCE_INLINE_ bool _ensure_cache_for_size(FontDataAdvanced *p_font_data, const Vector2i &p_size) const;
	_FORCE_INLINE_ void _font_clear_cache(FontDataAdvanced *p_font_data);
	void _generateMTSDF_threaded(uint32_t y, void *p_td) const;

	_FORCE_INLINE_ Vector2i _get_size(const FontDataAdvanced *p_font_data, int p_size) const {
		if (p_font_data->msdf) {
			return Vector2i(p_font_data->msdf_source_size, 0);
		} else if (p_font_data->fixed_size > 0) {
			return Vector2i(p_font_data->fixed_size, 0);
		} else {
			return Vector2i(p_size, 0);
		}
	}

	_FORCE_INLINE_ Vector2i _get_size_outline(const FontDataAdvanced *p_font_data, const Vector2i &p_size) const {
		if (p_font_data->msdf) {
			return Vector2i(p_font_data->msdf_source_size, 0);
		} else if (p_font_data->fixed_size > 0) {
			return Vector2i(p_font_data->fixed_size, MIN(p_size.y, 1));
		} else {
			return p_size;
		}
	}

	// Shaped text cache data.

	struct ShapedTextDataAdvanced : public ShapedTextData {
		/* Intermediate data */
		Char16String utf16;
		Vector<UBiDi *> bidi_iter;
		Vector<Vector2i> bidi_override;
		ScriptIterator *script_iter = nullptr;
		hb_buffer_t *hb_buffer = nullptr;

		~ShapedTextDataAdvanced() {
			for (int i = 0; i < bidi_iter.size(); i++) {
				ubidi_close(bidi_iter[i]);
			}
			if (script_iter) {
				memdelete(script_iter);
			}
			if (hb_buffer) {
				hb_buffer_destroy(hb_buffer);
			}
		}
	};

	// Common data.

	float oversampling = 1.f;
	mutable RID_PtrOwner<FontDataAdvanced> font_owner;
	mutable RID_PtrOwner<ShapedTextDataAdvanced> shaped_owner;

	int _convert_pos(const ShapedTextDataAdvanced *p_sd, int p_pos) const;
	int _convert_pos_inv(const ShapedTextDataAdvanced *p_sd, int p_pos) const;
	bool _shape_substr(ShapedTextDataAdvanced *p_new_sd, const ShapedTextDataAdvanced *p_sd, int p_start, int p_length) const;
	void _shape_run(ShapedTextDataAdvanced *p_sd, int32_t p_start, int32_t p_end, hb_script_t p_script, hb_direction_t p_direction, Vector<RID> p_fonts, int p_span, int p_fb_index);
	Glyph _shape_single_glyph(ShapedTextDataAdvanced *p_sd, char32_t p_char, hb_script_t p_script, hb_direction_t p_direction, RID p_font, int p_font_size);

	// HarfBuzz bitmap font interface.

	static hb_font_funcs_t *funcs;

	struct bmp_font_t {
		TextServerAdvanced::FontDataForSizeAdvanced *face = nullptr;
		bool unref = false; /* Whether to destroy bm_face when done. */
	};

	static bmp_font_t *_bmp_font_create(TextServerAdvanced::FontDataForSizeAdvanced *p_face, bool p_unref);
	static void _bmp_font_destroy(void *p_data);
	static hb_bool_t _bmp_get_nominal_glyph(hb_font_t *p_font, void *p_font_data, hb_codepoint_t p_unicode, hb_codepoint_t *r_glyph, void *p_user_data);
	static hb_position_t _bmp_get_glyph_h_advance(hb_font_t *p_font, void *p_font_data, hb_codepoint_t p_glyph, void *p_user_data);
	static hb_position_t _bmp_get_glyph_v_advance(hb_font_t *p_font, void *p_font_data, hb_codepoint_t p_glyph, void *p_user_data);
	static hb_position_t _bmp_get_glyph_h_kerning(hb_font_t *p_font, void *p_font_data, hb_codepoint_t p_left_glyph, hb_codepoint_t p_right_glyph, void *p_user_data);
	static hb_bool_t _bmp_get_glyph_v_origin(hb_font_t *p_font, void *p_font_data, hb_codepoint_t p_glyph, hb_position_t *r_x, hb_position_t *r_y, void *p_user_data);
	static hb_bool_t _bmp_get_glyph_extents(hb_font_t *p_font, void *p_font_data, hb_codepoint_t p_glyph, hb_glyph_extents_t *r_extents, void *p_user_data);
	static hb_bool_t _bmp_get_font_h_extents(hb_font_t *p_font, void *p_font_data, hb_font_extents_t *r_metrics, void *p_user_data);
	static void _bmp_create_font_funcs();
	static void _bmp_free_font_funcs();
	static void _bmp_font_set_funcs(hb_font_t *p_font, TextServerAdvanced::FontDataForSizeAdvanced *p_face, bool p_unref);
	static hb_font_t *_bmp_font_create(TextServerAdvanced::FontDataForSizeAdvanced *p_face, hb_destroy_func_t p_destroy);

protected:
	static void _bind_methods(){};

	void full_copy(ShapedTextDataAdvanced *p_shaped);
	void invalidate(ShapedTextDataAdvanced *p_shaped);

public:
	virtual bool has_feature(Feature p_feature) const override;
	virtual String get_name() const override;
	virtual uint32_t get_features() const override;

	virtual void free(RID p_rid) override;
	virtual bool has(RID p_rid) override;
	virtual bool load_support_data(const String &p_filename) override;

	virtual String get_support_data_filename() const override;
	virtual String get_support_data_info() const override;
	virtual bool save_support_data(const String &p_filename) const override;

	virtual bool is_locale_right_to_left(const String &p_locale) const override;

	virtual int32_t name_to_tag(const String &p_name) const override;
	virtual String tag_to_name(int32_t p_tag) const override;

	/* Font interface */
	virtual RID create_font() override;

	virtual void font_set_data(RID p_font_rid, const PackedByteArray &p_data) override;
	virtual void font_set_data_ptr(RID p_font_rid, const uint8_t *p_data_ptr, size_t p_data_size) override;

	virtual void font_set_style(RID p_font_rid, uint32_t /*FontStyle*/ p_style) override;
	virtual uint32_t /*FontStyle*/ font_get_style(RID p_font_rid) const override;

	virtual void font_set_style_name(RID p_font_rid, const String &p_name) override;
	virtual String font_get_style_name(RID p_font_rid) const override;

	virtual void font_set_name(RID p_font_rid, const String &p_name) override;
	virtual String font_get_name(RID p_font_rid) const override;

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

	virtual void font_set_oversampling(RID p_font_rid, float p_oversampling) override;
	virtual float font_get_oversampling(RID p_font_rid) const override;

	virtual Array font_get_size_cache_list(RID p_font_rid) const override;
	virtual void font_clear_size_cache(RID p_font_rid) override;
	virtual void font_remove_size_cache(RID p_font_rid, const Vector2i &p_size) override;

	hb_font_t *_font_get_hb_handle(RID p_font, int p_font_size) const;

	virtual void font_set_ascent(RID p_font_rid, int p_size, float p_ascent) override;
	virtual float font_get_ascent(RID p_font_rid, int p_size) const override;

	virtual void font_set_descent(RID p_font_rid, int p_size, float p_descent) override;
	virtual float font_get_descent(RID p_font_rid, int p_size) const override;

	virtual void font_set_underline_position(RID p_font_rid, int p_size, float p_underline_position) override;
	virtual float font_get_underline_position(RID p_font_rid, int p_size) const override;

	virtual void font_set_underline_thickness(RID p_font_rid, int p_size, float p_underline_thickness) override;
	virtual float font_get_underline_thickness(RID p_font_rid, int p_size) const override;

	virtual void font_set_scale(RID p_font_rid, int p_size, float p_scale) override;
	virtual float font_get_scale(RID p_font_rid, int p_size) const override;

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

	virtual Dictionary font_get_glyph_contours(RID p_font, int p_size, int32_t p_index) const override;

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

	virtual float font_get_global_oversampling() const override;
	virtual void font_set_global_oversampling(float p_oversampling) override;

	/* Shaped text buffer interface */

	virtual RID create_shaped_text(Direction p_direction = DIRECTION_AUTO, Orientation p_orientation = ORIENTATION_HORIZONTAL) override;

	virtual void shaped_text_clear(RID p_shaped) override;

	virtual void shaped_text_set_direction(RID p_shaped, Direction p_direction = DIRECTION_AUTO) override;
	virtual Direction shaped_text_get_direction(RID p_shaped) const override;

	virtual void shaped_text_set_bidi_override(RID p_shaped, const Array &p_override) override;

	virtual void shaped_text_set_custom_punctuation(RID p_shaped, const String &p_punct) override;
	virtual String shaped_text_get_custom_punctuation(RID p_shaped) const override;

	virtual void shaped_text_set_orientation(RID p_shaped, Orientation p_orientation = ORIENTATION_HORIZONTAL) override;
	virtual Orientation shaped_text_get_orientation(RID p_shaped) const override;

	virtual void shaped_text_set_preserve_invalid(RID p_shaped, bool p_enabled) override;
	virtual bool shaped_text_get_preserve_invalid(RID p_shaped) const override;

	virtual void shaped_text_set_preserve_control(RID p_shaped, bool p_enabled) override;
	virtual bool shaped_text_get_preserve_control(RID p_shaped) const override;

	virtual bool shaped_text_add_string(RID p_shaped, const String &p_text, const Vector<RID> &p_fonts, int p_size, const Dictionary &p_opentype_features = Dictionary(), const String &p_language = "") override;
	virtual bool shaped_text_add_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align = INLINE_ALIGNMENT_CENTER, int p_length = 1) override;
	virtual bool shaped_text_resize_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align = INLINE_ALIGNMENT_CENTER) override;

	virtual RID shaped_text_substr(RID p_shaped, int p_start, int p_length) const override;
	virtual RID shaped_text_get_parent(RID p_shaped) const override;

	virtual float shaped_text_fit_to_width(RID p_shaped, float p_width, uint16_t /*JustificationFlag*/ p_jst_flags = JUSTIFICATION_WORD_BOUND | JUSTIFICATION_KASHIDA) override;
	virtual float shaped_text_tab_align(RID p_shaped, const PackedFloat32Array &p_tab_stops) override;

	virtual bool shaped_text_shape(RID p_shaped) override;
	virtual bool shaped_text_update_breaks(RID p_shaped) override;
	virtual bool shaped_text_update_justification_ops(RID p_shaped) override;

	virtual int shaped_text_get_trim_pos(RID p_shaped) const override;
	virtual int shaped_text_get_ellipsis_pos(RID p_shaped) const override;
	virtual const Glyph *shaped_text_get_ellipsis_glyphs(RID p_shaped) const override;
	virtual int shaped_text_get_ellipsis_glyph_count(RID p_shaped) const override;

	virtual void shaped_text_overrun_trim_to_width(RID p_shaped, float p_width, uint16_t p_trim_flags) override;

	virtual bool shaped_text_is_ready(RID p_shaped) const override;

	virtual const Glyph *shaped_text_get_glyphs(RID p_shaped) const override;
	virtual const Glyph *shaped_text_sort_logical(RID p_shaped) override;
	virtual int shaped_text_get_glyph_count(RID p_shaped) const override;

	virtual Vector2i shaped_text_get_range(RID p_shaped) const override;

	virtual Array shaped_text_get_objects(RID p_shaped) const override;
	virtual Rect2 shaped_text_get_object_rect(RID p_shaped, Variant p_key) const override;

	virtual Size2 shaped_text_get_size(RID p_shaped) const override;
	virtual float shaped_text_get_ascent(RID p_shaped) const override;
	virtual float shaped_text_get_descent(RID p_shaped) const override;
	virtual float shaped_text_get_width(RID p_shaped) const override;
	virtual float shaped_text_get_underline_position(RID p_shaped) const override;
	virtual float shaped_text_get_underline_thickness(RID p_shaped) const override;

	virtual String format_number(const String &p_string, const String &p_language = "") const override;
	virtual String parse_number(const String &p_string, const String &p_language = "") const override;
	virtual String percent_sign(const String &p_language = "") const override;

	virtual String strip_diacritics(const String &p_string) const override;

	TextServerAdvanced();
	~TextServerAdvanced();
};

#endif // TEXT_SERVER_ADV_H
