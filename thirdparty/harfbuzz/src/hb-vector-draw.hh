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

#ifndef HB_VECTOR_DRAW_HH
#define HB_VECTOR_DRAW_HH

#include "hb.hh"

#include "hb-vector.h"
#include "hb-geometry.hh"
#include "hb-machinery.hh"
#include "hb-vector-buf.hh"
#include "hb-vector-internal.hh"

struct hb_vector_draw_t
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

  hb_vector_buf_t defs;
  hb_vector_buf_t body;
  hb_vector_buf_t path;
  hb_vector_buf_t pdf_extgstate_dict;
  unsigned pdf_extgstate_count = 0;

  void set_precision (unsigned p)
  {
    p = hb_min (p, 12u);
    defs.precision = p;
    body.precision = p;
    path.precision = p;
  }

  unsigned get_precision () const { return path.precision; }
  hb_blob_t *recycled_blob = nullptr;

  void new_path ()
  {
    flush_path ();
  }

  void flush_path ()
  {
    if (!path.length) return;
    switch (format)
    {
      case HB_VECTOR_FORMAT_PDF: flush_path_pdf (); break;
      case HB_VECTOR_FORMAT_SVG: flush_path_svg (); break;
      case HB_VECTOR_FORMAT_INVALID: default: break;
    }
    path.shrink (0);
  }

  void flush_path_pdf ()
  {
    unsigned a = hb_color_get_alpha (foreground);
    if (a < 255)
    {
      unsigned gs_idx = pdf_extgstate_count++;
      body.append_str ("/GS");
      body.append_unsigned (gs_idx);
      body.append_str (" gs\n");

      pdf_extgstate_dict.append_str ("/GS");
      pdf_extgstate_dict.append_unsigned (gs_idx);
      pdf_extgstate_dict.append_str (" << /Type /ExtGState /ca ");
      pdf_extgstate_dict.append_num (a / 255.f, 4);
      pdf_extgstate_dict.append_str (" >> ");
    }
    body.append_num (hb_color_get_red (foreground) / 255.f, 4);
    body.append_c (' ');
    body.append_num (hb_color_get_green (foreground) / 255.f, 4);
    body.append_c (' ');
    body.append_num (hb_color_get_blue (foreground) / 255.f, 4);
    body.append_str (" rg\n");
    body.append_len (path.arrayZ, path.length);
    body.append_str ("f\n");
  }

  void flush_path_svg ()
  {
    body.append_str ("<path d=\"");
    body.append_len (path.arrayZ, path.length);
    body.append_str ("\" fill=\"");
    body.append_svg_color (foreground, true);
    body.append_str ("\"/>\n");
  }

  void transform_xy (float x, float y, float *tx, float *ty)
  {
    hb_vector_transform_point (transform, x_scale_factor, y_scale_factor, x, y, tx, ty);
  }

  void append_xy (float x, float y, char sep)
  {
    float tx, ty;
    transform_xy (x, y, &tx, &ty);
    path.append_num (tx, path.precision);
    path.append_c (sep);
    path.append_num (ty, path.precision);
  }

  void append_xy_svg (float x, float y) { append_xy (x, y, ','); }
  void append_xy_pdf (float x, float y) { append_xy (x, y, ' '); }
};

#endif /* HB_VECTOR_DRAW_HH */
