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

#ifndef HB_VECTOR_PAINT_HH
#define HB_VECTOR_PAINT_HH

#include "hb.hh"

#include "hb-vector.h"
#include "hb-geometry.hh"
#include "hb-machinery.hh"
#include "hb-map.hh"
#include "hb-vector-path.hh"
#include "hb-vector-buf.hh"
#include "hb-vector-internal.hh"


struct hb_vector_paint_t
{
  hb_object_header_t header;

  hb_vector_format_t format = HB_VECTOR_FORMAT_INVALID;
  hb_transform_t<> transform = {1, 0, 0, 1, 0, 0};
  float x_scale_factor = 1.f;
  float y_scale_factor = 1.f;
  hb_vector_extents_t extents = {0, 0, 0, 0};
  bool has_extents = false;

  hb_color_t foreground = HB_COLOR (0, 0, 0, 255);
  hb_color_t background = HB_COLOR (0, 0, 0, 0);
  int palette = 0;
  hb_hashmap_t<unsigned, hb_color_t> custom_palette_colors;
  char *id_prefix = nullptr;
  unsigned id_prefix_length = 0;

  hb_vector_buf_t defs;
  hb_vector_buf_t path;
  hb_vector_t<hb_vector_buf_t> group_stack;
  uint64_t transform_group_open_mask = 0;
  unsigned transform_group_depth = 0;
  unsigned transform_group_overflow_depth = 0;

  unsigned clip_rect_counter = 0;
  unsigned clip_path_counter = 0;
  hb_vector_path_sink_t clip_path_sink = {nullptr, 0, 1.f, 1.f};
  unsigned gradient_counter = 0;
  unsigned color_glyph_depth = 0;
  unsigned path_def_count = 0;
  hb_set_t *active_color_glyphs = nullptr;
  hb_vector_t<hb_color_stop_t> color_stops_scratch;
  hb_vector_buf_t captured_scratch;
  hb_blob_t *recycled_blob = nullptr;

  hb_vector_buf_t &current_body () { return group_stack.tail (); }

  float sx (float v) const { return v / x_scale_factor; }
  float sy (float v) const { return v / y_scale_factor; }

  bool fetch_color_stops (hb_color_line_t *color_line)
  {
    unsigned count = hb_color_line_get_color_stops (color_line, 0, nullptr, nullptr);
    if (unlikely (!count || !color_stops_scratch.resize (count)))
    {
      color_stops_scratch.resize (0);
      return false;
    }
    hb_color_line_get_color_stops (color_line, 0, &count, color_stops_scratch.arrayZ);
    return true;
  }

  void set_precision (unsigned p)
  {
    p = hb_min (p, 12u);
    defs.precision = p;
    path.precision = p;
  }

  unsigned get_precision () const { return path.precision; }

  bool ensure_initialized ()
  {
    if (group_stack.length)
      return !group_stack.in_error () &&
             !group_stack.tail ().in_error ();
    if (unlikely (!group_stack.push_or_fail ()))
      return false;
    group_stack.tail ().alloc (4096);
    if (unlikely (group_stack.tail ().in_error ()))
    {
      group_stack.pop ();
      return false;
    }
    return true;
  }
};

/* Implemented in hb-vector-paint-svg.cc */
HB_INTERNAL hb_paint_funcs_t * hb_vector_paint_svg_funcs_get ();
HB_INTERNAL hb_blob_t * hb_vector_paint_render_svg (hb_vector_paint_t *paint);

/* Implemented in hb-vector-paint-pdf.cc */
HB_INTERNAL hb_paint_funcs_t * hb_vector_paint_pdf_funcs_get ();
HB_INTERNAL hb_blob_t * hb_vector_paint_render_pdf (hb_vector_paint_t *paint);
HB_INTERNAL void hb_vector_paint_pdf_free_resources (hb_vector_paint_t *paint);

#endif /* HB_VECTOR_PAINT_HH */
