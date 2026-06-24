/**************************************************************************/
/*  text_server_fb.h                                                      */
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

#pragma once

/*************************************************************************/
/* Fallback Text Server provides simplified TS functionality, without    */
/* BiDi, shaping and advanced font features support.                     */
/*************************************************************************/

#include "core/os/thread_safe.h"
#include "core/templates/rid_owner.h"
#include "core/templates/safe_refcount.h"
#include "scene/resources/image_texture.h"
#include "servers/text/text_server.h"

#include "modules/modules_enabled.gen.h" // For freetype, msdfgen, svg.

// Thirdparty headers.

#ifdef MODULE_FREETYPE_ENABLED
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_TRUETYPE_TABLES_H
#include FT_STROKER_H
#include FT_ADVANCES_H
#include FT_MULTIPLE_MASTERS_H
#include FT_BBOX_H
#include FT_SIZES_H
#include FT_MODULE_H
#include FT_CONFIG_OPTIONS_H
#if !defined(FT_CONFIG_OPTION_USE_BROTLI) && !defined(_MSC_VER)
#warning FreeType is configured without Brotli support, built-in fonts will not be available.
#endif
#endif

/*************************************************************************/

class TextServerFallback : public TextServer {
	GDCLASS(TextServerFallback, TextServer);
	_THREAD_SAFE_CLASS_

	HashMap<StringName, int32_t> feature_sets;
	HashMap<int32_t, StringName> feature_sets_inv;

	SafeNumeric<TextServer::FontLCDSubpixelLayout> lcd_subpixel_layout{ TextServer::FontLCDSubpixelLayout::FONT_LCD_SUBPIXEL_LAYOUT_NONE };
	void _update_settings();

	void _insert_feature_sets();
	_FORCE_INLINE_ void _insert_feature(const StringName &p_name, int32_t p_tag);

	// Font cache data.

#ifdef MODULE_FREETYPE_ENABLED
	mutable FT_Library ft_library = nullptr;
#endif

	const int rect_range = 1;

	struct FontTexturePosition {
		int32_t index = -1;
		int32_t x = 0;
		int32_t y = 0;

		FontTexturePosition() {}
		FontTexturePosition(int32_t p_id, int32_t p_x, int32_t p_y) :
				index(p_id), x(p_x), y(p_y) {}
	};

	struct Shelf {
		int32_t x = 0;
		int32_t y = 0;
		int32_t w = 0;
		int32_t h = 0;

		FontTexturePosition alloc_shelf(int32_t p_id, int32_t p_w, int32_t p_h) {
			if (p_w > w || p_h > h) {
				return FontTexturePosition(-1, 0, 0);
			}
			int32_t xx = x;
			x += p_w;
			w -= p_w;
			return FontTexturePosition(p_id, xx, y);
		}

		Shelf() {}
		Shelf(int32_t p_x, int32_t p_y, int32_t p_w, int32_t p_h) :
				x(p_x), y(p_y), w(p_w), h(p_h) {}
	};

	struct ShelfPackTexture {
		int32_t texture_w = 1024;
		int32_t texture_h = 1024;

		Ref<Image> image;
		Ref<ImageTexture> texture;
		bool dirty = true;

		List<Shelf> shelves;

		FontTexturePosition pack_rect(int32_t p_id, int32_t p_h, int32_t p_w) {
			int32_t y = 0;
			int32_t waste = 0;
			Shelf *best_shelf = nullptr;
			int32_t best_waste = std::numeric_limits<std::int32_t>::max();

			for (Shelf &E : shelves) {
				y += E.h;
				if (p_w > E.w) {
					continue;
				}
				if (p_h == E.h) {
					return E.alloc_shelf(p_id, p_w, p_h);
				}
				if (p_h > E.h) {
					continue;
				}
				if (p_h < E.h) {
					waste = (E.h - p_h) * p_w;
					if (waste < best_waste) {
						best_waste = waste;
						best_shelf = &E;
					}
				}
			}
			if (best_shelf) {
				return best_shelf->alloc_shelf(p_id, p_w, p_h);
			}
			if (p_h <= (texture_h - y) && p_w <= texture_w) {
				List<Shelf>::Element *E = shelves.push_back(Shelf(0, y, texture_w, p_h));
				return E->get().alloc_shelf(p_id, p_w, p_h);
			}
			return FontTexturePosition(-1, 0, 0);
		}

		ShelfPackTexture() {}
		ShelfPackTexture(int32_t p_w, int32_t p_h) :
				texture_w(p_w), texture_h(p_h) {}
	};

	struct FontGlyph {
		bool found = false;
		int texture_idx = -1;
		Rect2 rect;
		Rect2 uv_rect;
		Vector2 advance;
		bool from_svg = false;
	};

	struct FontFallback;
	struct FontForSizeFallback {
		double ascent = 0.0;
		double descent = 0.0;
		double underline_position = 0.0;
		double underline_thickness = 0.0;
		double scale = 1.0;

		FontFallback *owner = nullptr;
		uint32_t viewport_oversampling = 0;

		Vector2i size;

		Vector<ShelfPackTexture> textures;
		HashMap<int32_t, FontGlyph> glyph_map;
		HashMap<Vector2i, Vector2> kerning_map;

#ifdef MODULE_FREETYPE_ENABLED
		FT_Size fsize = nullptr;
#endif

		~FontForSizeFallback() {
#ifdef MODULE_FREETYPE_ENABLED
			if (fsize != nullptr) {
				FT_Done_Size(fsize);
			}
#endif
		}
	};

	struct OversamplingLevel {
		HashSet<FontForSizeFallback *> fonts;
		int32_t refcount = 1;
	};

	mutable HashMap<uint32_t, OversamplingLevel> oversampling_levels;

	struct FontFallbackLinkedVariation {
		RID base_font;
		int extra_spacing[4] = { 0, 0, 0, 0 };
		double baseline_offset = 0.0;
	};

	struct FontFallback {
		Mutex mutex;

		TextServer::FontAntialiasing antialiasing = TextServer::FONT_ANTIALIASING_GRAY;
		bool disable_embedded_bitmaps = true;
		bool mipmaps = false;
		bool msdf = false;
		FixedSizeScaleMode fixed_size_scale_mode = FIXED_SIZE_SCALE_DISABLE;
		int msdf_range = 14;
		int msdf_source_size = 48;
		int fixed_size = 0;
		bool force_autohinter = false;
		bool allow_system_fallback = true;
		bool modulate_color_glyphs = false;
		TextServer::Hinting hinting = TextServer::HINTING_LIGHT;
		TextServer::SubpixelPositioning subpixel_positioning = TextServer::SUBPIXEL_POSITIONING_AUTO;
		bool keep_rounding_remainders = true;
		Dictionary variation_coordinates;
		double oversampling_override = 0.0;
		double embolden = 0.0;
		Transform2D transform;

		BitField<TextServer::FontStyle> style_flags = 0;
		String font_name;
		String style_name;
		int weight = 400;
		int stretch = 100;
		int extra_spacing[4] = { 0, 0, 0, 0 };
		double baseline_offset = 0.0;

		HashMap<Vector2i, FontForSizeFallback *> cache;

		bool face_init = false;
		Dictionary supported_varaitions;
		Dictionary feature_overrides;

		// Language/script support override.
		HashMap<String, bool> language_support_overrides;
		HashMap<String, bool> script_support_overrides;

		PackedByteArray data;
		const uint8_t *data_ptr = nullptr;
		size_t data_size;
		int face_index = 0;

#ifdef MODULE_FREETYPE_ENABLED
		FT_Face face = nullptr;
		FT_StreamRec stream;
#endif

		~FontFallback() {
			for (const KeyValue<Vector2i, FontForSizeFallback *> &E : cache) {
				memdelete(E.value);
			}
			cache.clear();
#ifdef MODULE_FREETYPE_ENABLED
			if (face != nullptr) {
				FT_Done_Face(face);
			}
#endif
		}
	};

	_FORCE_INLINE_ FontTexturePosition find_texture_pos_for_glyph(FontForSizeFallback *p_data, int p_color_size, Image::Format p_image_format, int p_width, int p_height, bool p_msdf) const;
#ifdef MODULE_MSDFGEN_ENABLED
	_FORCE_INLINE_ FontGlyph rasterize_msdf(FontFallback *p_font_data, FontForSizeFallback *p_data, int p_pixel_range, int p_rect_margin, FT_Outline *p_outline, const Vector2 &p_advance) const;
#endif
#ifdef MODULE_FREETYPE_ENABLED
	_FORCE_INLINE_ FontGlyph rasterize_bitmap(FontForSizeFallback *p_data, int p_rect_margin, FT_Bitmap p_bitmap, int p_yofs, int p_xofs, const Vector2 &p_advance, bool p_bgra) const;
#endif
	bool _ensure_glyph(FontFallback *p_font_data, const Vector2i &p_size, int32_t p_glyph, FontGlyph &r_glyph, uint32_t p_oversampling = 0) const;
	bool _ensure_cache_for_size(FontFallback *p_font_data, const Vector2i &p_size, FontForSizeFallback *&r_cache_for_size, bool p_silent = false, uint32_t p_oversampling = 0) const;
	_FORCE_INLINE_ bool _font_validate(const RID &p_font_rid) const;
	_FORCE_INLINE_ void _font_clear_cache(FontFallback *p_font_data);
	static void _generateMTSDF_threaded(void *p_td, uint32_t p_y);

	_FORCE_INLINE_ Vector2i _get_size(const FontFallback *p_font_data, int p_size) const {
		if (p_font_data->msdf) {
			return Vector2i(p_font_data->msdf_source_size * 64, 0);
		} else if (p_font_data->fixed_size > 0) {
			return Vector2i(p_font_data->fixed_size * 64, 0);
		} else {
			return Vector2i(p_size * 64, 0);
		}
	}

	_FORCE_INLINE_ Vector2i _get_size_outline(const FontFallback *p_font_data, const Vector2i &p_size) const {
		if (p_font_data->msdf) {
			return Vector2i(p_font_data->msdf_source_size * 64, 0);
		} else if (p_font_data->fixed_size > 0) {
			return Vector2i(p_font_data->fixed_size * 64, MIN(p_size.y, 1));
		} else {
			return Vector2i(p_size.x * 64, p_size.y);
		}
	}

	_FORCE_INLINE_ int _font_get_weight_by_name(const String &p_sty_name) const {
		String sty_name = p_sty_name.remove_chars(" -");
		if (sty_name.contains("thin") || sty_name.contains("hairline")) {
			return 100;
		} else if (sty_name.contains("extralight") || sty_name.contains("ultralight")) {
			return 200;
		} else if (sty_name.contains("light")) {
			return 300;
		} else if (sty_name.contains("semilight")) {
			return 350;
		} else if (sty_name.contains("regular")) {
			return 400;
		} else if (sty_name.contains("medium")) {
			return 500;
		} else if (sty_name.contains("semibold") || sty_name.contains("demibold")) {
			return 600;
		} else if (sty_name.contains("bold")) {
			return 700;
		} else if (sty_name.contains("extrabold") || sty_name.contains("ultrabold")) {
			return 800;
		} else if (sty_name.contains("black") || sty_name.contains("heavy")) {
			return 900;
		} else if (sty_name.contains("extrablack") || sty_name.contains("ultrablack")) {
			return 950;
		}
		return 400;
	}
	_FORCE_INLINE_ int _font_get_stretch_by_name(const String &p_sty_name) const {
		String sty_name = p_sty_name.remove_chars(" -");
		if (sty_name.contains("ultracondensed")) {
			return 50;
		} else if (sty_name.contains("extracondensed")) {
			return 63;
		} else if (sty_name.contains("condensed")) {
			return 75;
		} else if (sty_name.contains("semicondensed")) {
			return 87;
		} else if (sty_name.contains("semiexpanded")) {
			return 113;
		} else if (sty_name.contains("expanded")) {
			return 125;
		} else if (sty_name.contains("extraexpanded")) {
			return 150;
		} else if (sty_name.contains("ultraexpanded")) {
			return 200;
		}
		return 100;
	}
	_FORCE_INLINE_ bool _is_ital_style(const String &p_sty_name) const {
		return p_sty_name.contains("italic") || p_sty_name.contains("oblique");
	}

	// Shaped text cache data.
	struct TrimData {
		int trim_pos = -1;
		int ellipsis_pos = -1;
		Vector<Glyph> ellipsis_glyph_buf;
	};

	struct TextRun {
		Vector2i range;
		Vector2i gl_range;
		RID font_rid;
		int font_size = 0;
		int64_t span_index = -1;
	};

	struct ShapedTextDataFallback {
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

			Array fonts;
			int font_size = 0;

			Variant embedded_key;

			String language;
			Dictionary features;
			Variant meta;
		};
		Vector<Span> spans;
		int first_span = 0; // First span in the parent ShapedTextData.
		int last_span = 0;

		Vector<TextRun> runs;
		bool runs_dirty = true;

		struct EmbeddedObject {
			int start = -1;
			int end = -1;
			InlineAlignment inline_align = INLINE_ALIGNMENT_CENTER;
			Rect2 rect;
			double baseline = 0;
		};
		HashMap<Variant, EmbeddedObject> objects;

		/* Shaped data */
		TextServer::Direction para_direction = DIRECTION_LTR; // Detected text direction.
		SafeFlag valid{ false }; // String is shaped.
		bool line_breaks_valid = false; // Line and word break flags are populated (and virtual zero width spaces inserted).
		bool justification_ops_valid = false; // Virtual elongation glyphs are added to the string.
		bool sort_valid = false;
		bool text_trimmed = false;

		bool preserve_invalid = true; // Draw hex code box instead of missing characters.
		bool preserve_control = false; // Draw control characters.

		double ascent = 0.0; // Ascent for horizontal layout, 1/2 of width for vertical.
		double descent = 0.0; // Descent for horizontal layout, 1/2 of width for vertical.
		double width = 0.0; // Width for horizontal layout, height for vertical.
		double width_trimmed = 0.0;
		int extra_spacing[4] = { 0, 0, 0, 0 };

		double upos = 0.0;
		double uthk = 0.0;

		char32_t el_char = 0x2026;
		TrimData overrun_trim_data;
		bool fit_width_minimum_reached = false;

		Vector<Glyph> glyphs;
		Vector<Glyph> glyphs_logical;
	};

	// Common data.

	mutable RID_PtrOwner<FontFallbackLinkedVariation> font_var_owner;
	mutable RID_PtrOwner<FontFallback> font_owner;
	mutable RID_PtrOwner<ShapedTextDataFallback> shaped_owner{ 65536, 1048576 };

	_FORCE_INLINE_ FontFallback *_get_font_data(const RID &p_font_rid) const {
		RID rid = p_font_rid;
		FontFallbackLinkedVariation *fdv = font_var_owner.get_or_null(rid);
		if (unlikely(fdv)) {
			rid = fdv->base_font;
		}
		return font_owner.get_or_null(rid);
	}

	struct SystemFontKey {
		String font_name;
		TextServer::FontAntialiasing antialiasing = TextServer::FONT_ANTIALIASING_GRAY;
		bool disable_embedded_bitmaps = true;
		bool italic = false;
		bool mipmaps = false;
		bool msdf = false;
		bool force_autohinter = false;
		int weight = 400;
		int stretch = 100;
		int msdf_range = 14;
		int msdf_source_size = 48;
		int fixed_size = 0;
		TextServer::Hinting hinting = TextServer::HINTING_LIGHT;
		TextServer::SubpixelPositioning subpixel_positioning = TextServer::SUBPIXEL_POSITIONING_AUTO;
		bool keep_rounding_remainders = true;
		Dictionary variation_coordinates;
		double embolden = 0.0;
		Transform2D transform;
		int extra_spacing[4] = { 0, 0, 0, 0 };
		double baseline_offset = 0.0;

		bool operator==(const SystemFontKey &p_b) const {
			return (font_name == p_b.font_name) && (antialiasing == p_b.antialiasing) && (italic == p_b.italic) && (disable_embedded_bitmaps == p_b.disable_embedded_bitmaps) && (mipmaps == p_b.mipmaps) && (msdf == p_b.msdf) && (force_autohinter == p_b.force_autohinter) && (weight == p_b.weight) && (stretch == p_b.stretch) && (msdf_range == p_b.msdf_range) && (msdf_source_size == p_b.msdf_source_size) && (fixed_size == p_b.fixed_size) && (hinting == p_b.hinting) && (subpixel_positioning == p_b.subpixel_positioning) && (keep_rounding_remainders == p_b.keep_rounding_remainders) && (variation_coordinates == p_b.variation_coordinates) && (embolden == p_b.embolden) && (transform == p_b.transform) && (extra_spacing[SPACING_TOP] == p_b.extra_spacing[SPACING_TOP]) && (extra_spacing[SPACING_BOTTOM] == p_b.extra_spacing[SPACING_BOTTOM]) && (extra_spacing[SPACING_SPACE] == p_b.extra_spacing[SPACING_SPACE]) && (extra_spacing[SPACING_GLYPH] == p_b.extra_spacing[SPACING_GLYPH]) && (baseline_offset == p_b.baseline_offset);
		}

		SystemFontKey(const String &p_font_name, bool p_italic, int p_weight, int p_stretch, RID p_font, const TextServerFallback *p_fb) {
			font_name = p_font_name;
			italic = p_italic;
			weight = p_weight;
			stretch = p_stretch;
			antialiasing = p_fb->font_get_antialiasing(p_font);
			disable_embedded_bitmaps = p_fb->font_get_disable_embedded_bitmaps(p_font);
			mipmaps = p_fb->font_get_generate_mipmaps(p_font);
			msdf = p_fb->font_is_multichannel_signed_distance_field(p_font);
			msdf_range = p_fb->font_get_msdf_pixel_range(p_font);
			msdf_source_size = p_fb->font_get_msdf_size(p_font);
			fixed_size = p_fb->font_get_fixed_size(p_font);
			force_autohinter = p_fb->font_is_force_autohinter(p_font);
			hinting = p_fb->font_get_hinting(p_font);
			subpixel_positioning = p_fb->font_get_subpixel_positioning(p_font);
			keep_rounding_remainders = p_fb->font_get_keep_rounding_remainders(p_font);
			variation_coordinates = p_fb->font_get_variation_coordinates(p_font);
			embolden = p_fb->font_get_embolden(p_font);
			transform = p_fb->font_get_transform(p_font);
			extra_spacing[SPACING_TOP] = p_fb->font_get_spacing(p_font, SPACING_TOP);
			extra_spacing[SPACING_BOTTOM] = p_fb->font_get_spacing(p_font, SPACING_BOTTOM);
			extra_spacing[SPACING_SPACE] = p_fb->font_get_spacing(p_font, SPACING_SPACE);
			extra_spacing[SPACING_GLYPH] = p_fb->font_get_spacing(p_font, SPACING_GLYPH);
			baseline_offset = p_fb->font_get_baseline_offset(p_font);
		}
	};

	struct SystemFontCacheRec {
		RID rid;
		int index = 0;
	};

	struct SystemFontCache {
		Vector<SystemFontCacheRec> var;
		int max_var = 0;
	};

	struct SystemFontKeyHasher {
		_FORCE_INLINE_ static uint32_t hash(const SystemFontKey &p_a) {
			uint32_t hash = p_a.font_name.hash();
			hash = hash_murmur3_one_32(p_a.variation_coordinates.hash(), hash);
			hash = hash_murmur3_one_32(p_a.weight, hash);
			hash = hash_murmur3_one_32(p_a.stretch, hash);
			hash = hash_murmur3_one_32(p_a.msdf_range, hash);
			hash = hash_murmur3_one_32(p_a.msdf_source_size, hash);
			hash = hash_murmur3_one_32(p_a.fixed_size, hash);
			hash = hash_murmur3_one_double(p_a.embolden, hash);
			hash = hash_murmur3_one_real(p_a.transform[0].x, hash);
			hash = hash_murmur3_one_real(p_a.transform[0].y, hash);
			hash = hash_murmur3_one_real(p_a.transform[1].x, hash);
			hash = hash_murmur3_one_real(p_a.transform[1].y, hash);
			hash = hash_murmur3_one_32(p_a.extra_spacing[SPACING_TOP], hash);
			hash = hash_murmur3_one_32(p_a.extra_spacing[SPACING_BOTTOM], hash);
			hash = hash_murmur3_one_32(p_a.extra_spacing[SPACING_SPACE], hash);
			hash = hash_murmur3_one_32(p_a.extra_spacing[SPACING_GLYPH], hash);
			hash = hash_murmur3_one_double(p_a.baseline_offset, hash);
			return hash_fmix32(hash_murmur3_one_32(((int)p_a.mipmaps) | ((int)p_a.msdf << 1) | ((int)p_a.italic << 2) | ((int)p_a.force_autohinter << 3) | ((int)p_a.hinting << 4) | ((int)p_a.subpixel_positioning << 8) | ((int)p_a.antialiasing << 12) | ((int)p_a.disable_embedded_bitmaps << 14) | ((int)p_a.keep_rounding_remainders << 15), hash));
		}
	};
	mutable HashMap<SystemFontKey, SystemFontCache, SystemFontKeyHasher> system_fonts;
	mutable HashMap<String, PackedByteArray> system_font_data;

	void _generate_runs(ShapedTextDataFallback *p_sd) const;
	void _realign(ShapedTextDataFallback *p_sd) const;
	_FORCE_INLINE_ RID _find_sys_font_for_text(const RID &p_fdef, const String &p_script_code, const String &p_language, const String &p_text);

	Mutex ft_mutex;

protected:
	static void _bind_methods() {}

	void full_copy(ShapedTextDataFallback *p_shaped);
	void invalidate(ShapedTextDataFallback *p_shaped);

public:
	virtual bool has_feature(Feature p_feature) const override;
	virtual String get_name() const override;
	virtual int64_t get_features() const override;

	virtual void free_rid(const RID &p_rid) override;
	virtual bool has(const RID &p_rid) override;
	virtual bool load_support_data(const String &p_filename) override;

	virtual String get_support_data_filename() const override;
	virtual String get_support_data_info() const override;
	virtual bool save_support_data(const String &p_filename) const override;
	virtual PackedByteArray get_support_data() const override;
	virtual bool is_locale_using_support_data(const String &p_locale) const override;

	virtual bool is_locale_right_to_left(const String &p_locale) const override;

	virtual int64_t name_to_tag(const String &p_name) const override;
	virtual String tag_to_name(int64_t p_tag) const override;

	/* Font interface */

	virtual RID create_font() override;
	virtual RID create_font_linked_variation(const RID &p_font_rid) override;

	virtual void font_set_data(const RID &p_font_rid, const PackedByteArray &p_data) override;
	virtual void font_set_data_ptr(const RID &p_font_rid, const uint8_t *p_data_ptr, int64_t p_data_size) override;

	virtual void font_set_face_index(const RID &p_font_rid, int64_t p_index) override;
	virtual int64_t font_get_face_index(const RID &p_font_rid) const override;

	virtual int64_t font_get_face_count(const RID &p_font_rid) const override;

	virtual void font_set_style(const RID &p_font_rid, BitField<FontStyle> p_style) override;
	virtual BitField<FontStyle> font_get_style(const RID &p_font_rid) const override;

	virtual void font_set_style_name(const RID &p_font_rid, const String &p_name) override;
	virtual String font_get_style_name(const RID &p_font_rid) const override;

	virtual void font_set_weight(const RID &p_font_rid, int64_t p_weight) override;
	virtual int64_t font_get_weight(const RID &p_font_rid) const override;

	virtual void font_set_stretch(const RID &p_font_rid, int64_t p_stretch) override;
	virtual int64_t font_get_stretch(const RID &p_font_rid) const override;

	virtual void font_set_name(const RID &p_font_rid, const String &p_name) override;
	virtual String font_get_name(const RID &p_font_rid) const override;

	virtual void font_set_antialiasing(const RID &p_font_rid, TextServer::FontAntialiasing p_antialiasing) override;
	virtual TextServer::FontAntialiasing font_get_antialiasing(const RID &p_font_rid) const override;

	virtual void font_set_disable_embedded_bitmaps(const RID &p_font_rid, bool p_disable_embedded_bitmaps) override;
	virtual bool font_get_disable_embedded_bitmaps(const RID &p_font_rid) const override;

	virtual void font_set_generate_mipmaps(const RID &p_font_rid, bool p_generate_mipmaps) override;
	virtual bool font_get_generate_mipmaps(const RID &p_font_rid) const override;

	virtual void font_set_multichannel_signed_distance_field(const RID &p_font_rid, bool p_msdf) override;
	virtual bool font_is_multichannel_signed_distance_field(const RID &p_font_rid) const override;

	virtual void font_set_msdf_pixel_range(const RID &p_font_rid, int64_t p_msdf_pixel_range) override;
	virtual int64_t font_get_msdf_pixel_range(const RID &p_font_rid) const override;

	virtual void font_set_msdf_size(const RID &p_font_rid, int64_t p_msdf_size) override;
	virtual int64_t font_get_msdf_size(const RID &p_font_rid) const override;

	virtual void font_set_fixed_size(const RID &p_font_rid, int64_t p_fixed_size) override;
	virtual int64_t font_get_fixed_size(const RID &p_font_rid) const override;

	virtual void font_set_fixed_size_scale_mode(const RID &p_font_rid, FixedSizeScaleMode p_fixed_size_scale) override;
	virtual FixedSizeScaleMode font_get_fixed_size_scale_mode(const RID &p_font_rid) const override;

	virtual void font_set_subpixel_positioning(const RID &p_font_rid, SubpixelPositioning p_subpixel) override;
	virtual SubpixelPositioning font_get_subpixel_positioning(const RID &p_font_rid) const override;

	virtual void font_set_keep_rounding_remainders(const RID &p_font_rid, bool p_keep_rounding_remainders) override;
	virtual bool font_get_keep_rounding_remainders(const RID &p_font_rid) const override;

	virtual void font_set_embolden(const RID &p_font_rid, double p_strength) override;
	virtual double font_get_embolden(const RID &p_font_rid) const override;

	virtual void font_set_spacing(const RID &p_font_rid, SpacingType p_spacing, int64_t p_value) override;
	virtual int64_t font_get_spacing(const RID &p_font_rid, SpacingType p_spacing) const override;

	virtual void font_set_baseline_offset(const RID &p_font_rid, double p_baseline_offset) override;
	virtual double font_get_baseline_offset(const RID &p_font_rid) const override;

	virtual void font_set_transform(const RID &p_font_rid, const Transform2D &p_transform) override;
	virtual Transform2D font_get_transform(const RID &p_font_rid) const override;

	virtual void font_set_allow_system_fallback(const RID &p_font_rid, bool p_allow_system_fallback) override;
	virtual bool font_is_allow_system_fallback(const RID &p_font_rid) const override;
	virtual void font_clear_system_fallback_cache() override;

	virtual void font_set_force_autohinter(const RID &p_font_rid, bool p_force_autohinter) override;
	virtual bool font_is_force_autohinter(const RID &p_font_rid) const override;

	virtual void font_set_modulate_color_glyphs(const RID &p_font_rid, bool p_modulate) override;
	virtual bool font_is_modulate_color_glyphs(const RID &p_font_rid) const override;

	virtual int64_t font_get_palette_count(const RID &p_font_rid) const override;
	virtual String font_get_palette_name(const RID &p_font_rid, int64_t p_index) const override;
	virtual Vector<Color> font_get_palette_colors(const RID &p_font_rid, int64_t p_index) const override;
	virtual void font_set_palette_custom_colors(const RID &p_font_rid, const Vector<Color> &p_colors) override;
	virtual Vector<Color> font_get_palette_custom_colors(const RID &p_font_rid) const override;
	virtual int64_t font_get_used_palette(const RID &p_font_rid) const override;
	virtual void font_set_used_palette(const RID &p_font_rid, int64_t p_index) override;

	virtual void font_set_hinting(const RID &p_font_rid, Hinting p_hinting) override;
	virtual Hinting font_get_hinting(const RID &p_font_rid) const override;

	virtual void font_set_variation_coordinates(const RID &p_font_rid, const Dictionary &p_variation_coordinates) override;
	virtual Dictionary font_get_variation_coordinates(const RID &p_font_rid) const override;

	virtual void font_set_oversampling(const RID &p_font_rid, double p_oversampling) override;
	virtual double font_get_oversampling(const RID &p_font_rid) const override;

	virtual TypedArray<Vector2i> font_get_size_cache_list(const RID &p_font_rid) const override;
	virtual void font_clear_size_cache(const RID &p_font_rid) override;
	virtual void font_remove_size_cache(const RID &p_font_rid, const Vector2i &p_size) override;
	virtual TypedArray<Dictionary> font_get_size_cache_info(const RID &p_font_rid) const override;

	virtual void font_set_ascent(const RID &p_font_rid, int64_t p_size, double p_ascent) override;
	virtual double font_get_ascent(const RID &p_font_rid, int64_t p_size) const override;

	virtual void font_set_descent(const RID &p_font_rid, int64_t p_size, double p_descent) override;
	virtual double font_get_descent(const RID &p_font_rid, int64_t p_size) const override;

	virtual void font_set_underline_position(const RID &p_font_rid, int64_t p_size, double p_underline_position) override;
	virtual double font_get_underline_position(const RID &p_font_rid, int64_t p_size) const override;

	virtual void font_set_underline_thickness(const RID &p_font_rid, int64_t p_size, double p_underline_thickness) override;
	virtual double font_get_underline_thickness(const RID &p_font_rid, int64_t p_size) const override;

	virtual void font_set_scale(const RID &p_font_rid, int64_t p_size, double p_scale) override;
	virtual double font_get_scale(const RID &p_font_rid, int64_t p_size) const override;

	virtual int64_t font_get_texture_count(const RID &p_font_rid, const Vector2i &p_size) const override;
	virtual void font_clear_textures(const RID &p_font_rid, const Vector2i &p_size) override;
	virtual void font_remove_texture(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) override;

	virtual void font_set_texture_image(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index, const Ref<Image> &p_image) override;
	virtual Ref<Image> font_get_texture_image(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) const override;

	virtual void font_set_texture_offsets(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index, const PackedInt32Array &p_offset) override;
	virtual PackedInt32Array font_get_texture_offsets(const RID &p_font_rid, const Vector2i &p_size, int64_t p_texture_index) const override;

	virtual PackedInt32Array font_get_glyph_list(const RID &p_font_rid, const Vector2i &p_size) const override;
	virtual void font_clear_glyphs(const RID &p_font_rid, const Vector2i &p_size) override;
	virtual void font_remove_glyph(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) override;

	virtual Vector2 font_get_glyph_advance(const RID &p_font_rid, int64_t p_size, int64_t p_glyph) const override;
	virtual void font_set_glyph_advance(const RID &p_font_rid, int64_t p_size, int64_t p_glyph, const Vector2 &p_advance) override;

	virtual Vector2 font_get_glyph_offset(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const override;
	virtual void font_set_glyph_offset(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Vector2 &p_offset) override;

	virtual Vector2 font_get_glyph_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const override;
	virtual void font_set_glyph_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Vector2 &p_gl_size) override;

	virtual Rect2 font_get_glyph_uv_rect(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const override;
	virtual void font_set_glyph_uv_rect(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, const Rect2 &p_uv_rect) override;

	virtual int64_t font_get_glyph_texture_idx(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const override;
	virtual void font_set_glyph_texture_idx(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph, int64_t p_texture_idx) override;

	virtual RID font_get_glyph_texture_rid(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const override;
	virtual Size2 font_get_glyph_texture_size(const RID &p_font_rid, const Vector2i &p_size, int64_t p_glyph) const override;

	virtual Dictionary font_get_glyph_contours(const RID &p_font, int64_t p_size, int64_t p_index) const override;

	virtual TypedArray<Vector2i> font_get_kerning_list(const RID &p_font_rid, int64_t p_size) const override;
	virtual void font_clear_kerning_map(const RID &p_font_rid, int64_t p_size) override;
	virtual void font_remove_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair) override;

	virtual void font_set_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) override;
	virtual Vector2 font_get_kerning(const RID &p_font_rid, int64_t p_size, const Vector2i &p_glyph_pair) const override;

	virtual int64_t font_get_glyph_index(const RID &p_font_rid, int64_t p_size, int64_t p_char, int64_t p_variation_selector = 0) const override;
	virtual int64_t font_get_char_from_glyph_index(const RID &p_font_rid, int64_t p_size, int64_t p_glyph_index) const override;

	virtual bool font_has_char(const RID &p_font_rid, int64_t p_char) const override;
	virtual String font_get_supported_chars(const RID &p_font_rid) const override;
	virtual PackedInt32Array font_get_supported_glyphs(const RID &p_font_rid) const override;

	virtual void font_render_range(const RID &p_font, const Vector2i &p_size, int64_t p_start, int64_t p_end) override;
	virtual void font_render_glyph(const RID &p_font_rid, const Vector2i &p_size, int64_t p_index) override;

	virtual void font_draw_glyph(const RID &p_font, const RID &p_canvas, int64_t p_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color = Color(1, 1, 1), float p_oversampling = 0.0) const override;
	virtual void font_draw_glyph_outline(const RID &p_font, const RID &p_canvas, int64_t p_size, int64_t p_outline_size, const Vector2 &p_pos, int64_t p_index, const Color &p_color = Color(1, 1, 1), float p_oversampling = 0.0) const override;

	virtual bool font_is_language_supported(const RID &p_font_rid, const String &p_language) const override;
	virtual void font_set_language_support_override(const RID &p_font_rid, const String &p_language, bool p_supported) override;
	virtual bool font_get_language_support_override(const RID &p_font_rid, const String &p_language) override;
	virtual void font_remove_language_support_override(const RID &p_font_rid, const String &p_language) override;
	virtual PackedStringArray font_get_language_support_overrides(const RID &p_font_rid) override;

	virtual bool font_is_script_supported(const RID &p_font_rid, const String &p_script) const override;
	virtual void font_set_script_support_override(const RID &p_font_rid, const String &p_script, bool p_supported) override;
	virtual bool font_get_script_support_override(const RID &p_font_rid, const String &p_script) override;
	virtual void font_remove_script_support_override(const RID &p_font_rid, const String &p_script) override;
	virtual PackedStringArray font_get_script_support_overrides(const RID &p_font_rid) override;

	virtual void font_set_opentype_feature_overrides(const RID &p_font_rid, const Dictionary &p_overrides) override;
	virtual Dictionary font_get_opentype_feature_overrides(const RID &p_font_rid) const override;

	virtual Dictionary font_supported_feature_list(const RID &p_font_rid) const override;
	virtual Dictionary font_supported_variation_list(const RID &p_font_rid) const override;

	virtual void reference_oversampling_level(double p_oversampling) override;
	virtual void unreference_oversampling_level(double p_oversampling) override;

	/* Shaped text buffer interface */

	virtual RID create_shaped_text(Direction p_direction = DIRECTION_AUTO, Orientation p_orientation = ORIENTATION_HORIZONTAL) override;

	virtual void shaped_text_clear(const RID &p_shaped) override;
	virtual RID shaped_text_duplicate(const RID &p_shaped) override;

	virtual void shaped_text_set_direction(const RID &p_shaped, Direction p_direction = DIRECTION_AUTO) override;
	virtual Direction shaped_text_get_direction(const RID &p_shaped) const override;
	virtual Direction shaped_text_get_inferred_direction(const RID &p_shaped) const override;

	virtual void shaped_text_set_bidi_override(const RID &p_shaped, const Array &p_override) override;

	virtual void shaped_text_set_custom_punctuation(const RID &p_shaped, const String &p_punct) override;
	virtual String shaped_text_get_custom_punctuation(const RID &p_shaped) const override;

	virtual void shaped_text_set_custom_ellipsis(const RID &p_shaped, int64_t p_char) override;
	virtual int64_t shaped_text_get_custom_ellipsis(const RID &p_shaped) const override;

	virtual void shaped_text_set_orientation(const RID &p_shaped, Orientation p_orientation = ORIENTATION_HORIZONTAL) override;
	virtual Orientation shaped_text_get_orientation(const RID &p_shaped) const override;

	virtual void shaped_text_set_preserve_invalid(const RID &p_shaped, bool p_enabled) override;
	virtual bool shaped_text_get_preserve_invalid(const RID &p_shaped) const override;

	virtual void shaped_text_set_preserve_control(const RID &p_shaped, bool p_enabled) override;
	virtual bool shaped_text_get_preserve_control(const RID &p_shaped) const override;

	virtual void shaped_text_set_spacing(const RID &p_shaped, SpacingType p_spacing, int64_t p_value) override;
	virtual int64_t shaped_text_get_spacing(const RID &p_shaped, SpacingType p_spacing) const override;

	virtual bool shaped_text_add_string(const RID &p_shaped, const String &p_text, const TypedArray<RID> &p_fonts, int64_t p_size, const Dictionary &p_opentype_features = Dictionary(), const String &p_language = "", const Variant &p_meta = Variant()) override;
	virtual bool shaped_text_add_object(const RID &p_shaped, const Variant &p_key, const Size2 &p_size, InlineAlignment p_inline_align = INLINE_ALIGNMENT_CENTER, int64_t p_length = 1, double p_baseline = 0.0) override;
	virtual bool shaped_text_resize_object(const RID &p_shaped, const Variant &p_key, const Size2 &p_size, InlineAlignment p_inline_align = INLINE_ALIGNMENT_CENTER, double p_baseline = 0.0) override;
	virtual bool shaped_text_has_object(const RID &p_shaped, const Variant &p_key) const override;
	virtual String shaped_get_text(const RID &p_shaped) const override;

	virtual int64_t shaped_get_span_count(const RID &p_shaped) const override;
	virtual Variant shaped_get_span_meta(const RID &p_shaped, int64_t p_index) const override;
	virtual Variant shaped_get_span_embedded_object(const RID &p_shaped, int64_t p_index) const override;
	virtual String shaped_get_span_text(const RID &p_shaped, int64_t p_index) const override;
	virtual Variant shaped_get_span_object(const RID &p_shaped, int64_t p_index) const override;
	virtual void shaped_set_span_update_font(const RID &p_shaped, int64_t p_index, const TypedArray<RID> &p_fonts, int64_t p_size, const Dictionary &p_opentype_features = Dictionary()) override;

	virtual int64_t shaped_get_run_count(const RID &p_shaped) const override;
	virtual String shaped_get_run_text(const RID &p_shaped, int64_t p_index) const override;
	virtual Vector2i shaped_get_run_range(const RID &p_shaped, int64_t p_index) const override;
	virtual Vector2i shaped_get_run_glyph_range(const RID &p_shaped, int64_t p_index) const override;
	virtual RID shaped_get_run_font_rid(const RID &p_shaped, int64_t p_index) const override;
	virtual int shaped_get_run_font_size(const RID &p_shaped, int64_t p_index) const override;
	virtual String shaped_get_run_language(const RID &p_shaped, int64_t p_index) const override;
	virtual Direction shaped_get_run_direction(const RID &p_shaped, int64_t p_index) const override;
	virtual Variant shaped_get_run_object(const RID &p_shaped, int64_t p_index) const override;

	virtual RID shaped_text_substr(const RID &p_shaped, int64_t p_start, int64_t p_length) const override;
	virtual RID shaped_text_get_parent(const RID &p_shaped) const override;

	virtual double shaped_text_fit_to_width(const RID &p_shaped, double p_width, BitField<TextServer::JustificationFlag> p_jst_flags = JUSTIFICATION_WORD_BOUND | JUSTIFICATION_KASHIDA) override;
	virtual double shaped_text_tab_align(const RID &p_shaped, const PackedFloat32Array &p_tab_stops) override;

	virtual bool shaped_text_shape(const RID &p_shaped) override;
	virtual bool shaped_text_update_breaks(const RID &p_shaped) override;
	virtual bool shaped_text_update_justification_ops(const RID &p_shaped) override;

	virtual int64_t shaped_text_get_trim_pos(const RID &p_shaped) const override;
	virtual int64_t shaped_text_get_ellipsis_pos(const RID &p_shaped) const override;
	virtual const Glyph *shaped_text_get_ellipsis_glyphs(const RID &p_shaped) const override;
	virtual int64_t shaped_text_get_ellipsis_glyph_count(const RID &p_shaped) const override;

	virtual void shaped_text_overrun_trim_to_width(const RID &p_shaped, double p_width, BitField<TextServer::TextOverrunFlag> p_trim_flags) override;

	virtual bool shaped_text_is_ready(const RID &p_shaped) const override;

	virtual const Glyph *shaped_text_get_glyphs(const RID &p_shaped) const override;
	virtual const Glyph *shaped_text_sort_logical(const RID &p_shaped) override;
	virtual int64_t shaped_text_get_glyph_count(const RID &p_shaped) const override;

	virtual Vector2i shaped_text_get_range(const RID &p_shaped) const override;

	virtual Array shaped_text_get_objects(const RID &p_shaped) const override;
	virtual Rect2 shaped_text_get_object_rect(const RID &p_shaped, const Variant &p_key) const override;
	virtual Vector2i shaped_text_get_object_range(const RID &p_shaped, const Variant &p_key) const override;
	virtual int64_t shaped_text_get_object_glyph(const RID &p_shaped, const Variant &p_key) const override;

	virtual Size2 shaped_text_get_size(const RID &p_shaped) const override;
	virtual double shaped_text_get_ascent(const RID &p_shaped) const override;
	virtual double shaped_text_get_descent(const RID &p_shaped) const override;
	virtual double shaped_text_get_width(const RID &p_shaped) const override;
	virtual double shaped_text_get_underline_position(const RID &p_shaped) const override;
	virtual double shaped_text_get_underline_thickness(const RID &p_shaped) const override;

	virtual PackedInt32Array shaped_text_get_character_breaks(const RID &p_shaped) const override;

	virtual PackedInt32Array string_get_word_breaks(const String &p_string, const String &p_language = "", int64_t p_chars_per_line = 0) const override;

	virtual String string_to_upper(const String &p_string, const String &p_language = "") const override;
	virtual String string_to_lower(const String &p_string, const String &p_language = "") const override;
	virtual String string_to_title(const String &p_string, const String &p_language = "") const override;

	virtual void cleanup() override;

	TextServerFallback();
	~TextServerFallback();
};
