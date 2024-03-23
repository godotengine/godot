/*
 * Copyright © 2022  Red Hat, Inc
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Matthias Clasen
 */

#ifndef HB_CAIRO_UTILS_H
#define HB_CAIRO_UTILS_H

#include "hb.hh"
#include "hb-cairo.h"


typedef struct
{
  cairo_scaled_font_t *scaled_font;
  cairo_t *cr;
  hb_map_t *color_cache;
} hb_cairo_context_t;

static inline cairo_operator_t
_hb_paint_composite_mode_to_cairo (hb_paint_composite_mode_t mode)
{
  switch (mode)
    {
    case HB_PAINT_COMPOSITE_MODE_CLEAR: return CAIRO_OPERATOR_CLEAR;
    case HB_PAINT_COMPOSITE_MODE_SRC: return CAIRO_OPERATOR_SOURCE;
    case HB_PAINT_COMPOSITE_MODE_DEST: return CAIRO_OPERATOR_DEST;
    case HB_PAINT_COMPOSITE_MODE_SRC_OVER: return CAIRO_OPERATOR_OVER;
    case HB_PAINT_COMPOSITE_MODE_DEST_OVER: return CAIRO_OPERATOR_DEST_OVER;
    case HB_PAINT_COMPOSITE_MODE_SRC_IN: return CAIRO_OPERATOR_IN;
    case HB_PAINT_COMPOSITE_MODE_DEST_IN: return CAIRO_OPERATOR_DEST_IN;
    case HB_PAINT_COMPOSITE_MODE_SRC_OUT: return CAIRO_OPERATOR_OUT;
    case HB_PAINT_COMPOSITE_MODE_DEST_OUT: return CAIRO_OPERATOR_DEST_OUT;
    case HB_PAINT_COMPOSITE_MODE_SRC_ATOP: return CAIRO_OPERATOR_ATOP;
    case HB_PAINT_COMPOSITE_MODE_DEST_ATOP: return CAIRO_OPERATOR_DEST_ATOP;
    case HB_PAINT_COMPOSITE_MODE_XOR: return CAIRO_OPERATOR_XOR;
    case HB_PAINT_COMPOSITE_MODE_PLUS: return CAIRO_OPERATOR_ADD;
    case HB_PAINT_COMPOSITE_MODE_SCREEN: return CAIRO_OPERATOR_SCREEN;
    case HB_PAINT_COMPOSITE_MODE_OVERLAY: return CAIRO_OPERATOR_OVERLAY;
    case HB_PAINT_COMPOSITE_MODE_DARKEN: return CAIRO_OPERATOR_DARKEN;
    case HB_PAINT_COMPOSITE_MODE_LIGHTEN: return CAIRO_OPERATOR_LIGHTEN;
    case HB_PAINT_COMPOSITE_MODE_COLOR_DODGE: return CAIRO_OPERATOR_COLOR_DODGE;
    case HB_PAINT_COMPOSITE_MODE_COLOR_BURN: return CAIRO_OPERATOR_COLOR_BURN;
    case HB_PAINT_COMPOSITE_MODE_HARD_LIGHT: return CAIRO_OPERATOR_HARD_LIGHT;
    case HB_PAINT_COMPOSITE_MODE_SOFT_LIGHT: return CAIRO_OPERATOR_SOFT_LIGHT;
    case HB_PAINT_COMPOSITE_MODE_DIFFERENCE: return CAIRO_OPERATOR_DIFFERENCE;
    case HB_PAINT_COMPOSITE_MODE_EXCLUSION: return CAIRO_OPERATOR_EXCLUSION;
    case HB_PAINT_COMPOSITE_MODE_MULTIPLY: return CAIRO_OPERATOR_MULTIPLY;
    case HB_PAINT_COMPOSITE_MODE_HSL_HUE: return CAIRO_OPERATOR_HSL_HUE;
    case HB_PAINT_COMPOSITE_MODE_HSL_SATURATION: return CAIRO_OPERATOR_HSL_SATURATION;
    case HB_PAINT_COMPOSITE_MODE_HSL_COLOR: return CAIRO_OPERATOR_HSL_COLOR;
    case HB_PAINT_COMPOSITE_MODE_HSL_LUMINOSITY: return CAIRO_OPERATOR_HSL_LUMINOSITY;
    default: return CAIRO_OPERATOR_CLEAR;
    }
}

HB_INTERNAL hb_bool_t
_hb_cairo_paint_glyph_image (hb_cairo_context_t *c,
			     hb_blob_t *blob,
			     unsigned width,
			     unsigned height,
			     hb_tag_t format,
			     float slant,
			     hb_glyph_extents_t *extents);

HB_INTERNAL void
_hb_cairo_paint_linear_gradient (hb_cairo_context_t *c,
				 hb_color_line_t *color_line,
				 float x0, float y0,
				 float x1, float y1,
				 float x2, float y2);

HB_INTERNAL void
_hb_cairo_paint_radial_gradient (hb_cairo_context_t *c,
				 hb_color_line_t *color_line,
				 float x0, float y0, float r0,
				 float x1, float y1, float r1);

HB_INTERNAL void
_hb_cairo_paint_sweep_gradient (hb_cairo_context_t *c,
				hb_color_line_t *color_line,
				float x0, float y0,
				float start_angle, float end_angle);


#endif /* HB_CAIRO_UTILS_H */
