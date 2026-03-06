/*
 * Copyright © 2026  Behdad Esfahbod
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
 * Author(s): Behdad Esfahbod
 */

#ifndef HB_RASTER_SVG_HH
#define HB_RASTER_SVG_HH

#include "hb.hh"

struct hb_raster_paint_t;

/* Callback type: emits SVG path draw commands to the given draw funcs */
typedef void (*hb_raster_svg_path_func_t) (hb_draw_funcs_t *dfuncs,
					   void *draw_data,
					   void *user_data);

/* Push clip from arbitrary path (reuses push_clip_glyph logic) */
HB_INTERNAL void
hb_raster_paint_push_clip_path (hb_raster_paint_t *c,
				hb_raster_svg_path_func_t func,
				void *user_data);

/* Render SVG document for a specific glyph */
#ifndef HB_NO_RASTER_SVG
HB_INTERNAL hb_bool_t
hb_raster_svg_render (hb_raster_paint_t *paint,
		      hb_blob_t *blob,
		      hb_codepoint_t glyph,
		      hb_font_t *font,
		      unsigned palette,
		      hb_color_t foreground);
#else
static inline hb_bool_t
hb_raster_svg_render (hb_raster_paint_t *paint HB_UNUSED,
		      hb_blob_t *blob HB_UNUSED,
		      hb_codepoint_t glyph HB_UNUSED,
		      hb_font_t *font HB_UNUSED,
		      unsigned palette HB_UNUSED,
		      hb_color_t foreground HB_UNUSED)
{
  return false;
}
#endif


#endif /* HB_RASTER_SVG_HH */
