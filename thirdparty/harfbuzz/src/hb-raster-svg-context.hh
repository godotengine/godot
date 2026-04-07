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

#ifndef HB_RASTER_SVG_CONTEXT_HH
#define HB_RASTER_SVG_CONTEXT_HH

#include "hb.hh"

#include "OT/Color/svg/svg.hh"
#include "hb-decycler.hh"
#include "hb-raster-paint.hh"
#include "hb-raster-svg-base.hh"
#include "hb-raster-svg-defs.hh"

struct hb_svg_fill_context_t
{
  hb_raster_paint_t *paint;
  hb_paint_funcs_t *pfuncs;
  hb_font_t *font;
  unsigned palette;
  hb_svg_defs_t *defs;
};

struct hb_svg_use_context_t
{
  hb_raster_paint_t *paint;
  hb_paint_funcs_t *pfuncs;

  const char *doc_start;
  unsigned doc_len;
  const OT::SVG::accelerator_t *svg_accel;
  const OT::SVG::svg_doc_cache_t *doc_cache;
  hb_decycler_t *use_decycler;
};

struct hb_svg_render_context_t
{
  hb_raster_paint_t *paint;
  hb_paint_funcs_t *pfuncs;
  hb_font_t *font;
  unsigned palette;
  hb_color_t foreground;
  hb_svg_defs_t defs;
  int depth = 0;

  const char *doc_start;
  unsigned doc_len;
  const OT::SVG::accelerator_t *svg_accel = nullptr;
  const OT::SVG::svg_doc_cache_t *doc_cache = nullptr;
  hb_decycler_t use_decycler;
  bool suppress_viewbox_once = false;
  bool allow_symbol_render_once = false;

  void push_transform (float xx, float yx, float xy, float yy, float dx, float dy)
  {
    hb_paint_push_transform (pfuncs, paint, xx, yx, xy, yy, dx, dy);
  }
  void pop_transform ()
  {
    hb_paint_pop_transform (pfuncs, paint);
  }
  void push_group ()
  {
    hb_paint_push_group (pfuncs, paint);
  }
  void pop_group (hb_paint_composite_mode_t mode)
  {
    hb_paint_pop_group (pfuncs, paint, mode);
  }
  void paint_color (hb_color_t color)
  {
    hb_paint_color (pfuncs, paint, false, color);
  }
  void pop_clip ()
  {
    hb_paint_pop_clip (pfuncs, paint);
  }
};

struct hb_svg_cascade_t
{
  hb_svg_str_t fill;
  float fill_opacity = 1.f;
  hb_svg_str_t clip_path;
  hb_color_t color = HB_COLOR (0, 0, 0, 255);
  bool visibility = true;
  float opacity = 1.f;
};

#endif /* HB_RASTER_SVG_CONTEXT_HH */
