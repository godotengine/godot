/*************************************************************************/
/*  hb-bitmap.cc                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "scene/resources/font.h"

#include "hb-private.hh"

#include "hb-bitmap.h"
#include "hb-font-private.hh"
#include "hb-machinery-private.hh"

struct hb_bmp_font_t {
	BitmapFont *bm_face;
	bool unref; /* Whether to destroy bm_face when done. */
};

static hb_bmp_font_t *_hb_bmp_font_create(BitmapFont *bm_face, bool unref) {
	hb_bmp_font_t *bm_font = (hb_bmp_font_t *)calloc(1, sizeof(hb_bmp_font_t));

	if (unlikely(!bm_font))
		return nullptr;

	bm_font->bm_face = bm_face;
	bm_font->unref = unref;

	return bm_font;
}

static void _hb_bmp_font_destroy(void *data) {
	hb_bmp_font_t *bm_font = (hb_bmp_font_t *)data;
	free(bm_font);
}

static hb_bool_t hb_bmp_get_nominal_glyph(hb_font_t *font HB_UNUSED, void *font_data, hb_codepoint_t unicode, hb_codepoint_t *glyph, void *user_data HB_UNUSED) {
	const hb_bmp_font_t *bm_font = (const hb_bmp_font_t *)font_data;

	if (!bm_font->bm_face)
		return false;

	if (!bm_font->bm_face->has_character(unicode)) {
		if (bm_font->bm_face->has_character(0xF000u + unicode)) {
			*glyph = 0xF000u + unicode;
			return true;
		} else {
			return false;
		}
	}

	*glyph = unicode;
	return true;
}

static hb_position_t hb_bmp_get_glyph_h_advance(hb_font_t *font, void *font_data, hb_codepoint_t glyph, void *user_data HB_UNUSED) {
	const hb_bmp_font_t *bm_font = (const hb_bmp_font_t *)font_data;

	if (!bm_font->bm_face)
		return 0;

	if (!bm_font->bm_face->has_character(glyph))
		return 0;

	return bm_font->bm_face->get_character(glyph).advance * 64;
}

static hb_position_t hb_bmp_get_glyph_h_kerning(hb_font_t *font, void *font_data, hb_codepoint_t left_glyph, hb_codepoint_t right_glyph, void *user_data HB_UNUSED) {
	const hb_bmp_font_t *bm_font = (const hb_bmp_font_t *)font_data;

	if (!bm_font->bm_face)
		return 0;

	if (!bm_font->bm_face->has_character(left_glyph))
		return 0;

	if (!bm_font->bm_face->has_character(right_glyph))
		return 0;

	return bm_font->bm_face->get_kerning_pair(left_glyph, right_glyph) * 64;
}

static hb_bool_t hb_bmp_get_glyph_v_origin(hb_font_t *font, void *font_data, hb_codepoint_t glyph, hb_position_t *x, hb_position_t *y, void *user_data HB_UNUSED) {
	const hb_bmp_font_t *bm_font = (const hb_bmp_font_t *)font_data;

	if (!bm_font->bm_face)
		return false;

	if (!bm_font->bm_face->has_character(glyph))
		return false;

	*x = bm_font->bm_face->get_character(glyph).h_align * 64;
	*y = bm_font->bm_face->get_character(glyph).v_align * 64;

	return true;
}

static hb_bool_t hb_bmp_get_glyph_extents(hb_font_t *font, void *font_data, hb_codepoint_t glyph, hb_glyph_extents_t *extents, void *user_data HB_UNUSED) {
	const hb_bmp_font_t *bm_font = (const hb_bmp_font_t *)font_data;

	if (!bm_font->bm_face)
		return false;

	if (!bm_font->bm_face->has_character(glyph))
		return false;

	extents->x_bearing = 0;
	extents->y_bearing = 0;
	extents->width = bm_font->bm_face->get_character(glyph).rect.size.x * 64;
	extents->height = bm_font->bm_face->get_character(glyph).rect.size.y * 64;

	return true;
}

static hb_bool_t hb_bmp_get_font_h_extents(hb_font_t *font HB_UNUSED, void *font_data, hb_font_extents_t *metrics, void *user_data HB_UNUSED) {
	const hb_bmp_font_t *bm_font = (const hb_bmp_font_t *)font_data;

	if (!bm_font->bm_face)
		return false;

	metrics->ascender = bm_font->bm_face->get_ascent();
	metrics->descender = bm_font->bm_face->get_descent();
	metrics->line_gap = 0;

	return true;
}

static struct hb_bmp_font_funcs_lazy_loader_t : hb_font_funcs_lazy_loader_t<hb_bmp_font_funcs_lazy_loader_t> {
	static inline hb_font_funcs_t *create(void) {
		hb_font_funcs_t *funcs = hb_font_funcs_create();

		hb_font_funcs_set_font_h_extents_func(funcs, hb_bmp_get_font_h_extents, nullptr, nullptr);
		//hb_font_funcs_set_font_v_extents_func (funcs, hb_bmp_get_font_v_extents, nullptr, nullptr);
		hb_font_funcs_set_nominal_glyph_func(funcs, hb_bmp_get_nominal_glyph, nullptr, nullptr);
		//hb_font_funcs_set_variation_glyph_func (funcs, hb_bmp_get_variation_glyph, nullptr, nullptr);
		hb_font_funcs_set_glyph_h_advance_func(funcs, hb_bmp_get_glyph_h_advance, nullptr, nullptr);
		//hb_font_funcs_set_glyph_v_advance_func (funcs, hb_bmp_get_glyph_v_advance, nullptr, nullptr);
		//hb_font_funcs_set_glyph_h_origin_func (funcs, hb_bmp_get_glyph_h_origin, nullptr, nullptr);
		hb_font_funcs_set_glyph_v_origin_func(funcs, hb_bmp_get_glyph_v_origin, nullptr, nullptr);
		hb_font_funcs_set_glyph_h_kerning_func(funcs, hb_bmp_get_glyph_h_kerning, nullptr, nullptr);
		//hb_font_funcs_set_glyph_v_kerning_func (funcs, hb_bmp_get_glyph_v_kerning, nullptr, nullptr);
		hb_font_funcs_set_glyph_extents_func(funcs, hb_bmp_get_glyph_extents, nullptr, nullptr);
		//hb_font_funcs_set_glyph_contour_point_func (funcs, hb_bmp_get_glyph_contour_point, nullptr, nullptr);
		//hb_font_funcs_set_glyph_name_func (funcs, hb_bmp_get_glyph_name, nullptr, nullptr);
		//hb_font_funcs_set_glyph_from_name_func (funcs, hb_bmp_get_glyph_from_name, nullptr, nullptr);

		hb_font_funcs_make_immutable(funcs);

#ifdef HB_USE_ATEXIT
		atexit(free_static_bmp_funcs);
#endif

		return funcs;
	}
} static_bmp_funcs;

#ifdef HB_USE_ATEXIT
static void free_static_bmp_funcs(void) {
	static_bmp_funcs.free_instance();
}
#endif

static hb_font_funcs_t *_hb_bmp_get_font_funcs(void) {
	return static_bmp_funcs.get_unconst();
}

static void _hb_bmp_font_set_funcs(hb_font_t *font, BitmapFont *bm_face, bool unref) {
	hb_font_set_funcs(font, _hb_bmp_get_font_funcs(), _hb_bmp_font_create(bm_face, unref), _hb_bmp_font_destroy);
}

hb_font_t *hb_bmp_font_create(BitmapFont *bm_face, hb_destroy_func_t destroy) {
	hb_font_t *font;

	font = hb_font_create(hb_face_create(NULL, 0));
	_hb_bmp_font_set_funcs(font, bm_face, false);
	return font;
}
