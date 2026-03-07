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

#ifndef HB_RASTER_SVG_BASE_HH
#define HB_RASTER_SVG_BASE_HH

#include "hb.hh"
#include "OT/Color/svg/svg.hh"

#include <math.h>
#include <string.h>
#include <stdlib.h>

static inline char
hb_svg_ascii_lower (char c)
{
  if (c >= 'A' && c <= 'Z')
    return c + ('a' - 'A');
  return c;
}

struct hb_svg_str_t
{
  const char *data;
  unsigned len;

  hb_svg_str_t () : data (nullptr), len (0) {}
  hb_svg_str_t (const char *d, unsigned l) : data (d), len (l) {}

  bool is_null () const { return !data; }

  bool eq (const char *s) const
  {
    unsigned slen = (unsigned) strlen (s);
    return len == slen && memcmp (data, s, len) == 0;
  }

  bool starts_with (const char *prefix) const
  {
    unsigned plen = (unsigned) strlen (prefix);
    return len >= plen && memcmp (data, prefix, plen) == 0;
  }

  bool eq_ascii_ci (const char *lit) const
  {
    unsigned n = (unsigned) strlen (lit);
    if (len != n) return false;
    for (unsigned i = 0; i < n; i++)
      if (hb_svg_ascii_lower (data[i]) != hb_svg_ascii_lower (lit[i]))
        return false;
    return true;
  }

  bool starts_with_ascii_ci (const char *lit) const
  {
    unsigned n = (unsigned) strlen (lit);
    if (len < n) return false;
    for (unsigned i = 0; i < n; i++)
      if (hb_svg_ascii_lower (data[i]) != hb_svg_ascii_lower (lit[i]))
        return false;
    return true;
  }

  float to_float () const
  {
    if (!data || !len) return 0.f;
    char buf[64];
    unsigned n = hb_min (len, (unsigned) sizeof (buf) - 1);
    memcpy (buf, data, n);
    buf[n] = '\0';
    float v = strtof (buf, nullptr);
    return isfinite (v) ? v : 0.f;
  }

  hb_svg_str_t trim_left () const
  {
    const char *p = data;
    unsigned l = len;
    while (l && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
    {
      p++;
      l--;
    }
    return {p, l};
  }

  hb_svg_str_t trim () const
  {
    hb_svg_str_t s = trim_left ();
    while (s.len && (s.data[s.len - 1] == ' ' || s.data[s.len - 1] == '\t' ||
                     s.data[s.len - 1] == '\n' || s.data[s.len - 1] == '\r'))
      s.len--;
    return s;
  }
};

struct hb_svg_style_props_t
{
  hb_svg_str_t fill;
  hb_svg_str_t fill_opacity;
  hb_svg_str_t opacity;
  hb_svg_str_t transform;
  hb_svg_str_t clip_path;
  hb_svg_str_t display;
  hb_svg_str_t color;
  hb_svg_str_t visibility;
  hb_svg_str_t offset;
  hb_svg_str_t stop_color;
  hb_svg_str_t stop_opacity;
  hb_svg_str_t spread_method;
  hb_svg_str_t gradient_units;
  hb_svg_str_t gradient_transform;
  hb_svg_str_t x;
  hb_svg_str_t y;
  hb_svg_str_t width;
  hb_svg_str_t height;
  hb_svg_str_t cx;
  hb_svg_str_t cy;
  hb_svg_str_t r;
  hb_svg_str_t fx;
  hb_svg_str_t fy;
  hb_svg_str_t fr;
  hb_svg_str_t rx;
  hb_svg_str_t ry;
  hb_svg_str_t x1;
  hb_svg_str_t y1;
  hb_svg_str_t x2;
  hb_svg_str_t y2;
  hb_svg_str_t points;
  hb_svg_str_t d;
};

struct hb_svg_xml_parser_t;
struct hb_svg_transform_t;

HB_INTERNAL void svg_parse_style_props (hb_svg_str_t style, hb_svg_style_props_t *out);
HB_INTERNAL float svg_parse_number_or_percent (hb_svg_str_t s, bool *is_percent);
HB_INTERNAL hb_svg_str_t hb_raster_svg_find_href_attr (const hb_svg_xml_parser_t &parser);
HB_INTERNAL bool hb_raster_svg_parse_id_ref (hb_svg_str_t s,
                                              hb_svg_str_t *out_id,
                                              hb_svg_str_t *out_tail);
HB_INTERNAL bool hb_raster_svg_parse_local_id_ref (hb_svg_str_t s,
                                                    hb_svg_str_t *out_id,
                                                    hb_svg_str_t *out_tail);
HB_INTERNAL bool hb_raster_svg_find_element_by_id (const char *doc_start,
                                                    unsigned doc_len,
                                                    const OT::SVG::accelerator_t *svg_accel,
                                                    const OT::SVG::svg_doc_cache_t *doc_cache,
                                                    hb_svg_str_t id,
                                                    const char **found);
HB_INTERNAL bool hb_raster_svg_parse_viewbox (hb_svg_str_t viewbox_str,
                                               float *x,
                                               float *y,
                                               float *w,
                                               float *h);
HB_INTERNAL bool hb_raster_svg_compute_viewbox_transform (float viewport_w,
                                                           float viewport_h,
                                                           float vb_x,
                                                           float vb_y,
                                                           float vb_w,
                                                           float vb_h,
                                                           hb_svg_str_t preserve_aspect_ratio,
                                                           hb_svg_transform_t *out);
HB_INTERNAL bool hb_raster_svg_compute_use_target_viewbox_transform (hb_svg_xml_parser_t &target_parser,
                                                                      float use_w,
                                                                      float use_h,
                                                                      hb_svg_transform_t *out);
HB_INTERNAL void hb_raster_svg_parse_use_geometry (hb_svg_xml_parser_t &parser,
                                                    float *x,
                                                    float *y,
                                                    float *w,
                                                    float *h);
HB_INTERNAL float hb_raster_svg_parse_non_percent_length (hb_svg_str_t s);
static inline float
svg_parse_float_clamped01 (hb_svg_str_t s)
{
  return hb_clamp (s.to_float (), 0.f, 1.f);
}

static inline bool
svg_str_is_inherit (hb_svg_str_t s)
{
  return s.trim ().eq_ascii_ci ("inherit");
}

static inline bool
svg_str_is_none (hb_svg_str_t s)
{
  return s.trim ().eq_ascii_ci ("none");
}

static inline bool
hb_raster_svg_tag_is_container (hb_svg_str_t tag)
{
  return tag.eq ("g") || tag.eq ("a") || tag.eq ("svg") || tag.eq ("symbol");
}

static inline bool
hb_raster_svg_tag_is_container_or_use (hb_svg_str_t tag)
{
  return hb_raster_svg_tag_is_container (tag) || tag.eq ("use");
}

#endif /* HB_RASTER_SVG_BASE_HH */
