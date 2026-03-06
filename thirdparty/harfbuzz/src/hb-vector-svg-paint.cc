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

#include "hb.hh"

#include "hb-vector.h"
#include "hb-blob.hh"
#include "hb-geometry.hh"
#include "hb-machinery.hh"
#include "hb-map.hh"
#include "hb-vector-svg-path.hh"
#include "hb-vector-svg-subset.hh"
#include "hb-vector-svg-utils.hh"

#include <algorithm>
#include <math.h>
#include <string.h>

#include "hb-vector-svg.hh"

struct hb_svg_color_glyph_cache_key_t
{
  hb_codepoint_t glyph = HB_CODEPOINT_INVALID;
  unsigned palette = 0;
  hb_color_t foreground = 0;

  hb_svg_color_glyph_cache_key_t () = default;
  hb_svg_color_glyph_cache_key_t (hb_codepoint_t g, unsigned p, hb_color_t f)
    : glyph (g), palette (p), foreground (f) {}

  bool operator == (const hb_svg_color_glyph_cache_key_t &o) const
  {
    return glyph == o.glyph &&
           palette == o.palette &&
           foreground == o.foreground;
  }

  uint32_t hash () const
  {
    uint32_t h = hb_hash (glyph);
    h = h * 31u + hb_hash (palette);
    h = h * 31u + hb_hash (foreground);
    return h;
  }
};

struct hb_vector_paint_t
{
  hb_object_header_t header;

  hb_vector_format_t format = HB_VECTOR_FORMAT_SVG;
  hb_transform_t<> transform = {1, 0, 0, 1, 0, 0};
  float x_scale_factor = 1.f;
  float y_scale_factor = 1.f;
  hb_vector_extents_t extents = {0, 0, 0, 0};
  bool has_extents = false;

  hb_color_t foreground = HB_COLOR (0, 0, 0, 255);
  int palette = 0;
  hb_hashmap_t<unsigned, hb_color_t> custom_palette_colors;
  unsigned precision = 2;
  bool flat = false;

  hb_vector_t<char> defs;
  hb_vector_t<char> path;
  hb_vector_t<hb_vector_t<char>> group_stack;
  uint64_t transform_group_open_mask = 0;
  unsigned transform_group_depth = 0;
  unsigned transform_group_overflow_depth = 0;

  unsigned clip_rect_counter = 0;
  unsigned gradient_counter = 0;
  unsigned color_glyph_counter = 0;
  hb_set_t *defined_outlines = nullptr;
  hb_set_t *defined_clips = nullptr;
  hb_hashmap_t<hb_svg_color_glyph_cache_key_t, uint64_t> defined_color_glyphs;
  hb_vector_t<hb_color_stop_t> color_stops_scratch;
  hb_vector_t<char> subset_body_scratch;
  hb_vector_t<char> captured_scratch;
  hb_blob_t *recycled_blob = nullptr;
  bool current_color_glyph_has_svg_image = false;
  hb_codepoint_t current_svg_image_glyph = HB_CODEPOINT_INVALID;
  hb_face_t *current_face = nullptr;
  unsigned svg_image_counter = 0;

  hb_vector_t<char> &current_body () { return group_stack.tail (); }

  void append_global_transform_prefix (hb_vector_t<char> *buf)
  {
    if (transform.xx == 1.f && transform.yx == 0.f &&
        transform.xy == 0.f && transform.yy == 1.f &&
        transform.x0 == 0.f && transform.y0 == 0.f &&
        x_scale_factor == 1.f && y_scale_factor == 1.f)
      return;

    unsigned sprec = hb_svg_scale_precision (precision);
    hb_svg_append_str (buf, "<g transform=\"matrix(");
    hb_svg_append_num (buf, transform.xx / x_scale_factor, sprec, true);
    hb_svg_append_c (buf, ',');
    hb_svg_append_num (buf, transform.yx / y_scale_factor, sprec, true);
    hb_svg_append_c (buf, ',');
    hb_svg_append_num (buf, transform.xy / x_scale_factor, sprec, true);
    hb_svg_append_c (buf, ',');
    hb_svg_append_num (buf, transform.yy / y_scale_factor, sprec, true);
    hb_svg_append_c (buf, ',');
    hb_svg_append_num (buf, transform.x0 / x_scale_factor, precision);
    hb_svg_append_c (buf, ',');
    hb_svg_append_num (buf, transform.y0 / y_scale_factor, precision);
    hb_svg_append_str (buf, ")\">\n");
  }

  void append_global_transform_suffix (hb_vector_t<char> *buf)
  {
    if (transform.xx == 1.f && transform.yx == 0.f &&
        transform.xy == 0.f && transform.yy == 1.f &&
        transform.x0 == 0.f && transform.y0 == 0.f &&
        x_scale_factor == 1.f && y_scale_factor == 1.f)
      return;
    hb_svg_append_str (buf, "</g>\n");
  }
};

static inline uint64_t
hb_svg_pack_color_glyph_cache_entry (unsigned def_id,
                                     bool image_like)
{
  return ((uint64_t) def_id << 1) | (image_like ? 1ull : 0ull);
}

static inline unsigned
hb_svg_cache_entry_def_id (uint64_t v)
{
  return (unsigned) (v >> 1);
}

static inline bool
hb_svg_cache_entry_image_like (uint64_t v)
{
  return !!(v & 1ull);
}

static inline hb_svg_color_glyph_cache_key_t
hb_svg_color_glyph_cache_key (hb_codepoint_t glyph,
                              unsigned palette,
                              hb_color_t foreground)
{
  return {glyph, palette, foreground};
}

static hb_bool_t
hb_svg_get_color_stops (hb_vector_paint_t *paint,
                        hb_color_line_t *color_line,
                        hb_vector_t<hb_color_stop_t> *stops)
{
  unsigned len = hb_color_line_get_color_stops (color_line, 0, nullptr, nullptr);
  if (unlikely (!stops->resize (len)))
    return false;
  hb_color_line_get_color_stops (color_line, 0, &len, stops->arrayZ);

  for (unsigned i = 0; i < len; i++)
    if (stops->arrayZ[i].is_foreground)
      stops->arrayZ[i].color = HB_COLOR (hb_color_get_blue (paint->foreground),
                                         hb_color_get_green (paint->foreground),
                                         hb_color_get_red (paint->foreground),
                                         (unsigned) hb_color_get_alpha (stops->arrayZ[i].color) *
                                         hb_color_get_alpha (paint->foreground) / 255);
  return true;
}

static const char *
hb_svg_extend_mode_str (hb_paint_extend_t ext)
{
  switch (ext)
  {
    case HB_PAINT_EXTEND_PAD: return "pad";
    case HB_PAINT_EXTEND_REPEAT: return "repeat";
    case HB_PAINT_EXTEND_REFLECT: return "reflect";
    default: return "pad";
  }
}

static int
hb_svg_color_stop_cmp (const void *a, const void *b)
{
  const hb_color_stop_t *x = (const hb_color_stop_t *) a;
  const hb_color_stop_t *y = (const hb_color_stop_t *) b;
  if (x->offset < y->offset) return -1;
  if (x->offset > y->offset) return 1;
  return 0;
}

static void
hb_svg_emit_color_stops (hb_vector_paint_t *paint,
                         hb_vector_t<char> *buf,
                         hb_vector_t<hb_color_stop_t> *stops)
{
  for (unsigned i = 0; i < stops->length; i++)
  {
    hb_color_t c = stops->arrayZ[i].color;
    hb_svg_append_str (buf, "<stop offset=\"");
    hb_svg_append_num (buf, stops->arrayZ[i].offset, 4);
    hb_svg_append_str (buf, "\" stop-color=\"rgb(");
    hb_svg_append_unsigned (buf, hb_color_get_red (c));
    hb_svg_append_c (buf, ',');
    hb_svg_append_unsigned (buf, hb_color_get_green (c));
    hb_svg_append_c (buf, ',');
    hb_svg_append_unsigned (buf, hb_color_get_blue (c));
    hb_svg_append_str (buf, ")\"");
    if (hb_color_get_alpha (c) != 255)
    {
      hb_svg_append_str (buf, " stop-opacity=\"");
      hb_svg_append_num (buf, hb_color_get_alpha (c) / 255.f, 4);
      hb_svg_append_c (buf, '"');
    }
    hb_svg_append_str (buf, "/>\n");
  }
}

static const char *
hb_svg_composite_mode_str (hb_paint_composite_mode_t mode)
{
  switch (mode)
  {
    case HB_PAINT_COMPOSITE_MODE_CLEAR:
    case HB_PAINT_COMPOSITE_MODE_SRC:
    case HB_PAINT_COMPOSITE_MODE_DEST:
    case HB_PAINT_COMPOSITE_MODE_DEST_OVER:
    case HB_PAINT_COMPOSITE_MODE_SRC_IN:
    case HB_PAINT_COMPOSITE_MODE_DEST_IN:
    case HB_PAINT_COMPOSITE_MODE_SRC_OUT:
    case HB_PAINT_COMPOSITE_MODE_DEST_OUT:
    case HB_PAINT_COMPOSITE_MODE_SRC_ATOP:
    case HB_PAINT_COMPOSITE_MODE_DEST_ATOP:
    case HB_PAINT_COMPOSITE_MODE_XOR:
    case HB_PAINT_COMPOSITE_MODE_PLUS:
      return nullptr;
    case HB_PAINT_COMPOSITE_MODE_SRC_OVER: return "normal";
    case HB_PAINT_COMPOSITE_MODE_SCREEN: return "screen";
    case HB_PAINT_COMPOSITE_MODE_OVERLAY: return "overlay";
    case HB_PAINT_COMPOSITE_MODE_DARKEN: return "darken";
    case HB_PAINT_COMPOSITE_MODE_LIGHTEN: return "lighten";
    case HB_PAINT_COMPOSITE_MODE_COLOR_DODGE: return "color-dodge";
    case HB_PAINT_COMPOSITE_MODE_COLOR_BURN: return "color-burn";
    case HB_PAINT_COMPOSITE_MODE_HARD_LIGHT: return "hard-light";
    case HB_PAINT_COMPOSITE_MODE_SOFT_LIGHT: return "soft-light";
    case HB_PAINT_COMPOSITE_MODE_DIFFERENCE: return "difference";
    case HB_PAINT_COMPOSITE_MODE_EXCLUSION: return "exclusion";
    case HB_PAINT_COMPOSITE_MODE_MULTIPLY: return "multiply";
    case HB_PAINT_COMPOSITE_MODE_HSL_HUE: return "hue";
    case HB_PAINT_COMPOSITE_MODE_HSL_SATURATION: return "saturation";
    case HB_PAINT_COMPOSITE_MODE_HSL_COLOR: return "color";
    case HB_PAINT_COMPOSITE_MODE_HSL_LUMINOSITY: return "luminosity";
    default: return nullptr;
  }
}

struct hb_svg_point_t { float x, y; };
struct hb_svg_rgba_t { float r, g, b, a; };

static inline float
hb_svg_lerp (float a, float b, float t)
{ return a + (b - a) * t; }

static inline float
hb_svg_clamp01 (float v)
{
  if (v < 0.f) return 0.f;
  if (v > 1.f) return 1.f;
  return v;
}

static inline hb_svg_rgba_t
hb_svg_rgba_from_hb_color (hb_color_t c)
{
  return {(float) hb_color_get_red (c) / 255.f,
          (float) hb_color_get_green (c) / 255.f,
          (float) hb_color_get_blue (c) / 255.f,
          (float) hb_color_get_alpha (c) / 255.f};
}

static inline hb_color_t
hb_svg_hb_color_from_rgba (const hb_svg_rgba_t &c)
{
  unsigned r = (unsigned) roundf (hb_svg_clamp01 (c.r) * 255.f);
  unsigned g = (unsigned) roundf (hb_svg_clamp01 (c.g) * 255.f);
  unsigned b = (unsigned) roundf (hb_svg_clamp01 (c.b) * 255.f);
  unsigned a = (unsigned) roundf (hb_svg_clamp01 (c.a) * 255.f);
  return HB_COLOR (b, g, r, a);
}

static inline hb_svg_rgba_t
hb_svg_lerp_rgba (const hb_svg_rgba_t &c0,
                  const hb_svg_rgba_t &c1,
                  float t)
{
  return {hb_svg_lerp (c0.r, c1.r, t),
          hb_svg_lerp (c0.g, c1.g, t),
          hb_svg_lerp (c0.b, c1.b, t),
          hb_svg_lerp (c0.a, c1.a, t)};
}

static inline float hb_svg_dot (const hb_svg_point_t &p, const hb_svg_point_t &q) { return p.x * q.x + p.y * q.y; }
static inline hb_svg_point_t hb_svg_add (const hb_svg_point_t &p, const hb_svg_point_t &q) { return {p.x + q.x, p.y + q.y}; }
static inline hb_svg_point_t hb_svg_sub (const hb_svg_point_t &p, const hb_svg_point_t &q) { return {p.x - q.x, p.y - q.y}; }
static inline hb_svg_point_t hb_svg_scale (const hb_svg_point_t &p, float f) { return {p.x * f, p.y * f}; }

static inline hb_svg_point_t
hb_svg_normalize (const hb_svg_point_t &p)
{
  float len = sqrtf (hb_svg_dot (p, p));
  if (len == 0.f) return {0.f, 0.f};
  return hb_svg_scale (p, 1.f / len);
}

static void
hb_svg_add_sweep_patch (hb_vector_t<char> *body,
                        unsigned precision,
                        float cx, float cy, float radius,
                        float a0, const hb_svg_rgba_t &c0_in,
                        float a1, const hb_svg_rgba_t &c1_in)
{
  static const float max_angle = HB_PI / 16.f;
  hb_svg_point_t center = {cx, cy};
  int num_splits = (int) ceilf (fabsf (a1 - a0) / max_angle);
  if (num_splits < 1) num_splits = 1;

  hb_svg_point_t p0 = {cosf (a0), sinf (a0)};
  hb_svg_rgba_t color0 = c0_in;

  for (int a = 0; a < num_splits; a++)
  {
    float k = (a + 1.f) / num_splits;
    float angle1 = hb_svg_lerp (a0, a1, k);
    hb_svg_rgba_t color1 = hb_svg_lerp_rgba (c0_in, c1_in, k);

    hb_svg_point_t p1 = {cosf (angle1), sinf (angle1)};
    hb_svg_point_t sp0 = hb_svg_add (center, hb_svg_scale (p0, radius));
    hb_svg_point_t sp1 = hb_svg_add (center, hb_svg_scale (p1, radius));

    hb_svg_point_t A = hb_svg_normalize (hb_svg_add (p0, p1));
    hb_svg_point_t U = {-A.y, A.x};
    float up0 = hb_svg_dot (U, p0);
    float up1 = hb_svg_dot (U, p1);
    if (fabsf (up0) < 1e-6f || fabsf (up1) < 1e-6f)
    {
      p0 = p1;
      color0 = color1;
      continue;
    }
    hb_svg_point_t C0 = hb_svg_add (A, hb_svg_scale (U, hb_svg_dot (hb_svg_sub (p0, A), p0) / up0));
    hb_svg_point_t C1 = hb_svg_add (A, hb_svg_scale (U, hb_svg_dot (hb_svg_sub (p1, A), p1) / up1));

    hb_svg_point_t sc0 = hb_svg_add (center, hb_svg_scale (hb_svg_add (C0, hb_svg_scale (hb_svg_sub (C0, p0), 0.33333f)), radius));
    hb_svg_point_t sc1 = hb_svg_add (center, hb_svg_scale (hb_svg_add (C1, hb_svg_scale (hb_svg_sub (C1, p1), 0.33333f)), radius));

    hb_svg_rgba_t mid_color = hb_svg_lerp_rgba (color0, color1, 0.5f);
    hb_color_t mid = hb_svg_hb_color_from_rgba (mid_color);

    hb_svg_append_str (body, "<path d=\"M");
    hb_svg_append_num (body, center.x, precision);
    hb_svg_append_c (body, ',');
    hb_svg_append_num (body, center.y, precision);
    hb_svg_append_str (body, "L");
    hb_svg_append_num (body, sp0.x, precision);
    hb_svg_append_c (body, ',');
    hb_svg_append_num (body, sp0.y, precision);
    hb_svg_append_str (body, "C");
    hb_svg_append_num (body, sc0.x, precision);
    hb_svg_append_c (body, ',');
    hb_svg_append_num (body, sc0.y, precision);
    hb_svg_append_c (body, ' ');
    hb_svg_append_num (body, sc1.x, precision);
    hb_svg_append_c (body, ',');
    hb_svg_append_num (body, sc1.y, precision);
    hb_svg_append_c (body, ' ');
    hb_svg_append_num (body, sp1.x, precision);
    hb_svg_append_c (body, ',');
    hb_svg_append_num (body, sp1.y, precision);
    hb_svg_append_str (body, "Z\" fill=\"");
    hb_svg_append_color (body, mid, true);
    hb_svg_append_str (body, "\"/>\n");

    p0 = p1;
    color0 = color1;
  }
}

static void
hb_svg_add_sweep_gradient_patches (hb_vector_t<char> *body,
                                   unsigned precision,
                                   hb_color_stop_t *stops,
                                   unsigned n_stops,
                                   hb_paint_extend_t extend,
                                   float cx, float cy, float radius,
                                   float start_angle, float end_angle)
{
  if (!n_stops) return;

  hb_svg_rgba_t colors_buf[16];
  float angles_buf[16];
  hb_svg_rgba_t *colors = colors_buf;
  float *angles = angles_buf;
  bool dynamic = false;

  if (start_angle == end_angle)
  {
    if (extend == HB_PAINT_EXTEND_PAD)
    {
      if (start_angle > 0.f)
      {
        hb_svg_rgba_t c = hb_svg_rgba_from_hb_color (stops[0].color);
        hb_svg_add_sweep_patch (body, precision, cx, cy, radius, 0.f, c, start_angle, c);
      }
      if (end_angle < HB_2_PI)
      {
        hb_svg_rgba_t c = hb_svg_rgba_from_hb_color (stops[n_stops - 1].color);
        hb_svg_add_sweep_patch (body, precision, cx, cy, radius, end_angle, c, HB_2_PI, c);
      }
    }
    return;
  }

  if (end_angle < start_angle)
  {
    float tmp = start_angle; start_angle = end_angle; end_angle = tmp;
    for (unsigned i = 0; i < n_stops - 1 - i; i++)
    {
      hb_color_stop_t t = stops[i];
      stops[i] = stops[n_stops - 1 - i];
      stops[n_stops - 1 - i] = t;
    }
    for (unsigned i = 0; i < n_stops; i++)
      stops[i].offset = 1.f - stops[i].offset;
  }

  if (n_stops > 16)
  {
    angles = (float *) hb_malloc (sizeof (float) * n_stops);
    colors = (hb_svg_rgba_t *) hb_malloc (sizeof (hb_svg_rgba_t) * n_stops);
    if (!angles || !colors)
    {
      hb_free (angles);
      hb_free (colors);
      return;
    }
    dynamic = true;
  }

  for (unsigned i = 0; i < n_stops; i++)
  {
    angles[i] = start_angle + stops[i].offset * (end_angle - start_angle);
    colors[i] = hb_svg_rgba_from_hb_color (stops[i].color);
  }

  if (extend == HB_PAINT_EXTEND_PAD)
  {
    unsigned pos;
    hb_svg_rgba_t color0 = colors[0];
    for (pos = 0; pos < n_stops; pos++)
    {
      if (angles[pos] >= 0)
      {
        if (pos > 0)
        {
          float f = (0.f - angles[pos - 1]) / (angles[pos] - angles[pos - 1]);
          color0 = hb_svg_lerp_rgba (colors[pos - 1], colors[pos], f);
        }
        break;
      }
    }
    if (pos == n_stops)
    {
      color0 = colors[n_stops - 1];
      hb_svg_add_sweep_patch (body, precision, cx, cy, radius, 0.f, color0, HB_2_PI, color0);
      goto done;
    }
    hb_svg_add_sweep_patch (body, precision, cx, cy, radius, 0.f, color0, angles[pos], colors[pos]);
    for (pos++; pos < n_stops; pos++)
    {
      if (angles[pos] <= HB_2_PI)
        hb_svg_add_sweep_patch (body, precision, cx, cy, radius, angles[pos - 1], colors[pos - 1], angles[pos], colors[pos]);
      else
      {
        float f = (HB_2_PI - angles[pos - 1]) / (angles[pos] - angles[pos - 1]);
        hb_svg_rgba_t color1 = hb_svg_lerp_rgba (colors[pos - 1], colors[pos], f);
        hb_svg_add_sweep_patch (body, precision, cx, cy, radius, angles[pos - 1], colors[pos - 1], HB_2_PI, color1);
        break;
      }
    }
    if (pos == n_stops)
    {
      color0 = colors[n_stops - 1];
      hb_svg_add_sweep_patch (body, precision, cx, cy, radius, angles[n_stops - 1], color0, HB_2_PI, color0);
      goto done;
    }
  }
  else
  {
    float span = angles[n_stops - 1] - angles[0];
    if (fabsf (span) < 1e-6f)
      goto done;

    int k = 0;
    if (angles[0] >= 0)
    {
      float ss = angles[0];
      while (ss > 0)
      {
        if (span > 0) { ss -= span; k--; }
        else          { ss += span; k++; }
      }
    }
    else
    {
      float ee = angles[n_stops - 1];
      while (ee < 0)
      {
        if (span > 0) { ee += span; k++; }
        else          { ee -= span; k--; }
      }
    }

    span = fabsf (span);
    for (int l = k; l < 1000; l++)
    {
      for (unsigned i = 1; i < n_stops; i++)
      {
        float a0_l, a1_l;
        const hb_svg_rgba_t *col0, *col1;
        if ((l % 2 != 0) && (extend == HB_PAINT_EXTEND_REFLECT))
        {
          a0_l = angles[0] + angles[n_stops - 1] - angles[n_stops - i] + l * span;
          a1_l = angles[0] + angles[n_stops - 1] - angles[n_stops - 1 - i] + l * span;
          col0 = &colors[n_stops - i];
          col1 = &colors[n_stops - 1 - i];
        }
        else
        {
          a0_l = angles[i - 1] + l * span;
          a1_l = angles[i] + l * span;
          col0 = &colors[i - 1];
          col1 = &colors[i];
        }

        if (a1_l < 0.f) continue;
        if (a0_l < 0.f)
        {
          float f = (0.f - a0_l) / (a1_l - a0_l);
          hb_svg_rgba_t c = hb_svg_lerp_rgba (*col0, *col1, f);
          hb_svg_add_sweep_patch (body, precision, cx, cy, radius, 0.f, c, a1_l, *col1);
        }
        else if (a1_l >= HB_2_PI)
        {
          float f = (HB_2_PI - a0_l) / (a1_l - a0_l);
          hb_svg_rgba_t c = hb_svg_lerp_rgba (*col0, *col1, f);
          hb_svg_add_sweep_patch (body, precision, cx, cy, radius, a0_l, *col0, HB_2_PI, c);
          goto done;
        }
        else
          hb_svg_add_sweep_patch (body, precision, cx, cy, radius, a0_l, *col0, a1_l, *col1);
      }
    }
  }

done:
  if (dynamic)
  {
    hb_free (angles);
    hb_free (colors);
  }
}


static void hb_vector_paint_push_transform (hb_paint_funcs_t *, void *,
                                            float, float, float, float, float, float,
                                            void *);
static void hb_vector_paint_pop_transform (hb_paint_funcs_t *, void *, void *);
static void hb_vector_paint_push_clip_glyph (hb_paint_funcs_t *, void *, hb_codepoint_t, hb_font_t *, void *);
static void hb_vector_paint_push_clip_rectangle (hb_paint_funcs_t *, void *, float, float, float, float, void *);
static void hb_vector_paint_pop_clip (hb_paint_funcs_t *, void *, void *);
static void hb_vector_paint_color (hb_paint_funcs_t *, void *, hb_bool_t, hb_color_t, void *);
static hb_bool_t hb_vector_paint_image (hb_paint_funcs_t *, void *, hb_blob_t *, unsigned, unsigned, hb_tag_t, float, hb_glyph_extents_t *, void *);
static void hb_vector_paint_linear_gradient (hb_paint_funcs_t *, void *, hb_color_line_t *, float, float, float, float, float, float, void *);
static void hb_vector_paint_radial_gradient (hb_paint_funcs_t *, void *, hb_color_line_t *, float, float, float, float, float, float, void *);
static void hb_vector_paint_sweep_gradient (hb_paint_funcs_t *, void *, hb_color_line_t *, float, float, float, float, void *);
static void hb_vector_paint_push_group (hb_paint_funcs_t *, void *, void *);
static void hb_vector_paint_pop_group (hb_paint_funcs_t *, void *, hb_paint_composite_mode_t, void *);
static hb_bool_t hb_vector_paint_color_glyph (hb_paint_funcs_t *, void *, hb_codepoint_t, hb_font_t *, void *);
static hb_bool_t hb_vector_paint_custom_palette_color (hb_paint_funcs_t *, void *, unsigned, hb_color_t *, void *);

static inline void free_static_vector_paint_funcs ();
static struct hb_vector_paint_funcs_lazy_loader_t
  : hb_paint_funcs_lazy_loader_t<hb_vector_paint_funcs_lazy_loader_t>
{
  static hb_paint_funcs_t *create ()
  {
    hb_paint_funcs_t *funcs = hb_paint_funcs_create ();
    hb_paint_funcs_set_push_transform_func (funcs, (hb_paint_push_transform_func_t) hb_vector_paint_push_transform, nullptr, nullptr);
    hb_paint_funcs_set_pop_transform_func (funcs, (hb_paint_pop_transform_func_t) hb_vector_paint_pop_transform, nullptr, nullptr);
    hb_paint_funcs_set_push_clip_glyph_func (funcs, (hb_paint_push_clip_glyph_func_t) hb_vector_paint_push_clip_glyph, nullptr, nullptr);
    hb_paint_funcs_set_push_clip_rectangle_func (funcs, (hb_paint_push_clip_rectangle_func_t) hb_vector_paint_push_clip_rectangle, nullptr, nullptr);
    hb_paint_funcs_set_pop_clip_func (funcs, (hb_paint_pop_clip_func_t) hb_vector_paint_pop_clip, nullptr, nullptr);
    hb_paint_funcs_set_color_func (funcs, (hb_paint_color_func_t) hb_vector_paint_color, nullptr, nullptr);
    hb_paint_funcs_set_image_func (funcs, (hb_paint_image_func_t) hb_vector_paint_image, nullptr, nullptr);
    hb_paint_funcs_set_linear_gradient_func (funcs, (hb_paint_linear_gradient_func_t) hb_vector_paint_linear_gradient, nullptr, nullptr);
    hb_paint_funcs_set_radial_gradient_func (funcs, (hb_paint_radial_gradient_func_t) hb_vector_paint_radial_gradient, nullptr, nullptr);
    hb_paint_funcs_set_sweep_gradient_func (funcs, (hb_paint_sweep_gradient_func_t) hb_vector_paint_sweep_gradient, nullptr, nullptr);
    hb_paint_funcs_set_push_group_func (funcs, (hb_paint_push_group_func_t) hb_vector_paint_push_group, nullptr, nullptr);
    hb_paint_funcs_set_pop_group_func (funcs, (hb_paint_pop_group_func_t) hb_vector_paint_pop_group, nullptr, nullptr);
    hb_paint_funcs_set_color_glyph_func (funcs, (hb_paint_color_glyph_func_t) hb_vector_paint_color_glyph, nullptr, nullptr);
    hb_paint_funcs_set_custom_palette_color_func (funcs, (hb_paint_custom_palette_color_func_t) hb_vector_paint_custom_palette_color, nullptr, nullptr);
    hb_paint_funcs_make_immutable (funcs);
    hb_atexit (free_static_vector_paint_funcs);
    return funcs;
  }
} static_vector_paint_funcs;

static inline void
free_static_vector_paint_funcs ()
{
  static_vector_paint_funcs.free_instance ();
}

static hb_paint_funcs_t *
hb_vector_paint_funcs_get ()
{
  return static_vector_paint_funcs.get_unconst ();
}

static hb_bool_t
hb_vector_paint_ensure_initialized (hb_vector_paint_t *paint)
{
  if (paint->group_stack.length)
    return true;
  if (unlikely (!paint->group_stack.push_or_fail ()))
    return false;
  paint->group_stack.tail ().alloc (4096);
  return !paint->group_stack.in_error ();
}

static void
hb_vector_paint_push_transform (hb_paint_funcs_t *,
                                void *paint_data,
                                float xx, float yx,
                                float xy, float yy,
                                float dx, float dy,
                                void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;

  if (unlikely (paint->transform_group_overflow_depth))
  {
    paint->transform_group_overflow_depth++;
    return;
  }
  if (unlikely (paint->transform_group_depth >= 64))
  {
    paint->transform_group_overflow_depth = 1;
    return;
  }

  hb_bool_t opened =
    !(fabsf (xx - 1.f) < 1e-6f && fabsf (yx) < 1e-6f &&
      fabsf (xy) < 1e-6f && fabsf (yy - 1.f) < 1e-6f &&
      fabsf (dx) < 1e-6f && fabsf (dy) < 1e-6f);
  paint->transform_group_open_mask = (paint->transform_group_open_mask << 1) | (opened ? 1ull : 0ull);
  paint->transform_group_depth++;

  if (!opened)
    return;

  auto &body = paint->current_body ();
  unsigned sprec = hb_svg_scale_precision (paint->precision);
  hb_svg_append_str (&body, "<g transform=\"matrix(");
  hb_svg_append_num (&body, xx, sprec, true);
  hb_svg_append_c (&body, ',');
  hb_svg_append_num (&body, yx, sprec, true);
  hb_svg_append_c (&body, ',');
  hb_svg_append_num (&body, xy, sprec, true);
  hb_svg_append_c (&body, ',');
  hb_svg_append_num (&body, yy, sprec, true);
  hb_svg_append_c (&body, ',');
  hb_svg_append_num (&body, dx, paint->precision);
  hb_svg_append_c (&body, ',');
  hb_svg_append_num (&body, dy, paint->precision);
  hb_svg_append_str (&body, ")\">\n");
}

static void
hb_vector_paint_pop_transform (hb_paint_funcs_t *,
                               void *paint_data,
                               void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;
  if (unlikely (paint->transform_group_overflow_depth))
  {
    paint->transform_group_overflow_depth--;
    return;
  }
  if (!paint->transform_group_depth)
    return;
  paint->transform_group_depth--;
  hb_bool_t opened = !!(paint->transform_group_open_mask & 1ull);
  paint->transform_group_open_mask >>= 1;
  if (opened)
    hb_svg_append_str (&paint->current_body (), "</g>\n");
}

static void
hb_vector_paint_push_clip_glyph (hb_paint_funcs_t *,
                                 void *paint_data,
                                 hb_codepoint_t glyph,
                                 hb_font_t *font,
                                 void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;

  if (!hb_set_has (paint->defined_outlines, glyph))
  {
    hb_set_add (paint->defined_outlines, glyph);
    paint->path.clear ();
    hb_svg_path_sink_t sink = {&paint->path, paint->precision};
    hb_font_draw_glyph (font, glyph, hb_svg_path_draw_funcs_get (), &sink);
    hb_svg_append_str (&paint->defs, "<path id=\"p");
    hb_svg_append_unsigned (&paint->defs, glyph);
    hb_svg_append_str (&paint->defs, "\" d=\"");
    hb_svg_append_len (&paint->defs, paint->path.arrayZ, paint->path.length);
    hb_svg_append_str (&paint->defs, "\"/>\n");
  }

  if (!hb_set_has (paint->defined_clips, glyph))
  {
    hb_set_add (paint->defined_clips, glyph);
    hb_svg_append_str (&paint->defs, "<clipPath id=\"clip-g");
    hb_svg_append_unsigned (&paint->defs, glyph);
    hb_svg_append_str (&paint->defs, "\"><use href=\"#p");
    hb_svg_append_unsigned (&paint->defs, glyph);
    hb_svg_append_str (&paint->defs, "\"/></clipPath>\n");
  }

  hb_svg_append_str (&paint->current_body (), "<g clip-path=\"url(#clip-g");
  hb_svg_append_unsigned (&paint->current_body (), glyph);
  hb_svg_append_str (&paint->current_body (), ")\">\n");
}

static void
hb_vector_paint_push_clip_rectangle (hb_paint_funcs_t *,
                                     void *paint_data,
                                     float xmin, float ymin,
                                     float xmax, float ymax,
                                     void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;

  unsigned clip_id = paint->clip_rect_counter++;
  hb_svg_append_str (&paint->defs, "<clipPath id=\"c");
  hb_svg_append_unsigned (&paint->defs, clip_id);
  hb_svg_append_str (&paint->defs, "\"><rect x=\"");
  hb_svg_append_num (&paint->defs, xmin, paint->precision);
  hb_svg_append_str (&paint->defs, "\" y=\"");
  hb_svg_append_num (&paint->defs, ymin, paint->precision);
  hb_svg_append_str (&paint->defs, "\" width=\"");
  hb_svg_append_num (&paint->defs, xmax - xmin, paint->precision);
  hb_svg_append_str (&paint->defs, "\" height=\"");
  hb_svg_append_num (&paint->defs, ymax - ymin, paint->precision);
  hb_svg_append_str (&paint->defs, "\"/></clipPath>\n");

  hb_svg_append_str (&paint->current_body (), "<g clip-path=\"url(#c");
  hb_svg_append_unsigned (&paint->current_body (), clip_id);
  hb_svg_append_str (&paint->current_body (), ")\">\n");
}

static void
hb_vector_paint_pop_clip (hb_paint_funcs_t *,
                          void *paint_data,
                          void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;
  hb_svg_append_str (&paint->current_body (), "</g>\n");
}

static void
hb_vector_paint_color (hb_paint_funcs_t *,
                       void *paint_data,
                       hb_bool_t is_foreground,
                       hb_color_t color,
                       void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;

  hb_color_t c = color;
  if (is_foreground)
    c = HB_COLOR (hb_color_get_blue (paint->foreground),
                  hb_color_get_green (paint->foreground),
                  hb_color_get_red (paint->foreground),
                  (unsigned) hb_color_get_alpha (paint->foreground) * hb_color_get_alpha (color) / 255);

  auto &body = paint->current_body ();
  hb_svg_append_str (&body, "<rect x=\"-32767\" y=\"-32767\" width=\"65534\" height=\"65534\" fill=\"");
  hb_svg_append_color (&body, c, true);
  hb_svg_append_str (&body, "\"/>\n");
}

static hb_bool_t
hb_vector_paint_image (hb_paint_funcs_t *,
                       void *paint_data,
                       hb_blob_t *image,
                       unsigned width,
                       unsigned height,
                       hb_tag_t format,
                       float slant HB_UNUSED,
                       hb_glyph_extents_t *extents,
                       void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return false;

  auto &body = paint->current_body ();
  if (format == HB_TAG ('s','v','g',' '))
  {
    paint->current_color_glyph_has_svg_image = true;

    paint->subset_body_scratch.clear ();
    bool subset_ok = hb_svg_subset_glyph_image (paint->current_face,
                                                image,
                                                paint->current_svg_image_glyph,
                                                &paint->svg_image_counter,
                                                &paint->defs,
                                                &paint->subset_body_scratch);
    if (unlikely (!subset_ok))
      return false;

    if (extents)
    {
      hb_svg_append_str (&body, "<g transform=\"translate(");
      hb_svg_append_num (&body, (float) extents->x_bearing, paint->precision);
      hb_svg_append_c (&body, ',');
      hb_svg_append_num (&body, (float) extents->y_bearing, paint->precision);
      hb_svg_append_str (&body, ") scale(");
      hb_svg_append_num (&body, (float) extents->width / width, paint->precision);
      hb_svg_append_c (&body, ',');
      hb_svg_append_num (&body, (float) extents->height / height, paint->precision);
      hb_svg_append_str (&body, ")\">\n");
    }

    hb_svg_append_len (&body,
                       paint->subset_body_scratch.arrayZ,
                       paint->subset_body_scratch.length);
    hb_svg_append_c (&body, '\n');

    if (extents)
      hb_svg_append_str (&body, "</g>\n");

    return true;
  }

  if (format == HB_TAG ('p','n','g',' '))
  {
    if (!extents || !width || !height)
      return false;

    unsigned len = 0;
    const char *png_data = hb_blob_get_data (image, &len);
    if (!png_data || !len)
      return false;

    hb_svg_append_str (&body, "<g transform=\"translate(");
    hb_svg_append_num (&body, (float) extents->x_bearing, paint->precision);
    hb_svg_append_c (&body, ',');
    hb_svg_append_num (&body, (float) extents->y_bearing, paint->precision);
    hb_svg_append_str (&body, ") scale(");
    hb_svg_append_num (&body, (float) extents->width / width, paint->precision);
    hb_svg_append_c (&body, ',');
    hb_svg_append_num (&body, (float) extents->height / height, paint->precision);
    hb_svg_append_str (&body, ")\">\n");

    hb_svg_append_str (&body, "<image href=\"data:image/png;base64,");
    hb_svg_append_base64 (&body, (const uint8_t *) png_data, len);
    hb_svg_append_str (&body, "\" width=\"");
    hb_svg_append_num (&body, (float) width, paint->precision);
    hb_svg_append_str (&body, "\" height=\"");
    hb_svg_append_num (&body, (float) height, paint->precision);
    hb_svg_append_str (&body, "\"/>\n</g>\n");

    return true;
  }

  return false;
}

static void
hb_vector_paint_linear_gradient (hb_paint_funcs_t *,
                                 void *paint_data,
                                 hb_color_line_t *color_line,
                                 float x0, float y0,
                                 float x1, float y1,
                                 float x2, float y2,
                                 void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;

  hb_vector_t<hb_color_stop_t> &stops = paint->color_stops_scratch;
  if (!hb_svg_get_color_stops (paint, color_line, &stops) || !stops.length)
    return;

  qsort (stops.arrayZ, stops.length, sizeof (hb_color_stop_t), hb_svg_color_stop_cmp);

  unsigned grad_id = paint->gradient_counter++;

  hb_svg_append_str (&paint->defs, "<linearGradient id=\"gr");
  hb_svg_append_unsigned (&paint->defs, grad_id);
  hb_svg_append_str (&paint->defs, "\" gradientUnits=\"userSpaceOnUse\" x1=\"");
  hb_svg_append_num (&paint->defs, x0, paint->precision);
  hb_svg_append_str (&paint->defs, "\" y1=\"");
  hb_svg_append_num (&paint->defs, y0, paint->precision);
  hb_svg_append_str (&paint->defs, "\" x2=\"");
  hb_svg_append_num (&paint->defs, x1 + (x1 - x2), paint->precision);
  hb_svg_append_str (&paint->defs, "\" y2=\"");
  hb_svg_append_num (&paint->defs, y1 + (y1 - y2), paint->precision);
  hb_svg_append_str (&paint->defs, "\" spreadMethod=\"");
  hb_svg_append_str (&paint->defs, hb_svg_extend_mode_str (hb_color_line_get_extend (color_line)));
  hb_svg_append_str (&paint->defs, "\">\n");
  hb_svg_emit_color_stops (paint, &paint->defs, &stops);
  hb_svg_append_str (&paint->defs, "</linearGradient>\n");

  hb_svg_append_str (&paint->current_body (),
                     "<rect x=\"-32767\" y=\"-32767\" width=\"65534\" height=\"65534\" fill=\"url(#gr");
  hb_svg_append_unsigned (&paint->current_body (), grad_id);
  hb_svg_append_str (&paint->current_body (), ")\"/>\n");
}

static void
hb_vector_paint_radial_gradient (hb_paint_funcs_t *,
                                 void *paint_data,
                                 hb_color_line_t *color_line,
                                 float x0, float y0, float r0,
                                 float x1, float y1, float r1,
                                 void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;

  hb_vector_t<hb_color_stop_t> &stops = paint->color_stops_scratch;
  if (!hb_svg_get_color_stops (paint, color_line, &stops) || !stops.length)
    return;

  qsort (stops.arrayZ, stops.length, sizeof (hb_color_stop_t), hb_svg_color_stop_cmp);

  unsigned grad_id = paint->gradient_counter++;

  hb_svg_append_str (&paint->defs, "<radialGradient id=\"gr");
  hb_svg_append_unsigned (&paint->defs, grad_id);
  hb_svg_append_str (&paint->defs, "\" gradientUnits=\"userSpaceOnUse\" cx=\"");
  hb_svg_append_num (&paint->defs, x1, paint->precision);
  hb_svg_append_str (&paint->defs, "\" cy=\"");
  hb_svg_append_num (&paint->defs, y1, paint->precision);
  hb_svg_append_str (&paint->defs, "\" r=\"");
  hb_svg_append_num (&paint->defs, r1, paint->precision);
  hb_svg_append_str (&paint->defs, "\" fx=\"");
  hb_svg_append_num (&paint->defs, x0, paint->precision);
  hb_svg_append_str (&paint->defs, "\" fy=\"");
  hb_svg_append_num (&paint->defs, y0, paint->precision);
  if (r0 > 0)
  {
    hb_svg_append_str (&paint->defs, "\" fr=\"");
    hb_svg_append_num (&paint->defs, r0, paint->precision);
  }
  hb_svg_append_str (&paint->defs, "\" spreadMethod=\"");
  hb_svg_append_str (&paint->defs, hb_svg_extend_mode_str (hb_color_line_get_extend (color_line)));
  hb_svg_append_str (&paint->defs, "\">\n");
  hb_svg_emit_color_stops (paint, &paint->defs, &stops);
  hb_svg_append_str (&paint->defs, "</radialGradient>\n");

  hb_svg_append_str (&paint->current_body (),
                     "<rect x=\"-32767\" y=\"-32767\" width=\"65534\" height=\"65534\" fill=\"url(#gr");
  hb_svg_append_unsigned (&paint->current_body (), grad_id);
  hb_svg_append_str (&paint->current_body (), ")\"/>\n");
}

static void
hb_vector_paint_sweep_gradient (hb_paint_funcs_t *,
                                void *paint_data,
                                hb_color_line_t *color_line,
                                float cx, float cy,
                                float start_angle, float end_angle,
                                void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;

  hb_vector_t<hb_color_stop_t> &stops = paint->color_stops_scratch;
  if (!hb_svg_get_color_stops (paint, color_line, &stops) || !stops.length)
    return;

  qsort (stops.arrayZ, stops.length, sizeof (hb_color_stop_t), hb_svg_color_stop_cmp);
  hb_svg_add_sweep_gradient_patches (&paint->current_body (),
                                     paint->precision,
                                     stops.arrayZ,
                                     stops.length,
                                     hb_color_line_get_extend (color_line),
                                     cx, cy,
                                     32767.f,
                                     start_angle,
                                     end_angle);
}

static void
hb_vector_paint_push_group (hb_paint_funcs_t *,
                            void *paint_data,
                            void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;
  paint->group_stack.push_or_fail (hb_vector_t<char> {});
}

static void
hb_vector_paint_pop_group (hb_paint_funcs_t *,
                           void *paint_data,
                           hb_paint_composite_mode_t mode,
                           void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return;
  if (paint->group_stack.length < 2)
    return;

  hb_vector_t<char> group = paint->group_stack.pop ();
  auto &body = paint->current_body ();

  const char *blend = hb_svg_composite_mode_str (mode);
  if (blend)
  {
    hb_svg_append_str (&body, "<g style=\"mix-blend-mode:");
    hb_svg_append_str (&body, blend);
    hb_svg_append_str (&body, "\">\n");
    hb_svg_append_len (&body, group.arrayZ, group.length);
    hb_svg_append_str (&body, "</g>\n");
  }
  else
    hb_svg_append_len (&body, group.arrayZ, group.length);
}

static hb_bool_t
hb_vector_paint_color_glyph (hb_paint_funcs_t *,
                             void *paint_data,
                             hb_codepoint_t glyph,
                             hb_font_t *font,
                             void *)
{
  auto *paint = (hb_vector_paint_t *) paint_data;
  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return false;
  hb_codepoint_t old_gid = paint->current_svg_image_glyph;
  hb_face_t *old_face = paint->current_face;
  paint->current_svg_image_glyph = glyph;
  paint->current_face = hb_font_get_face (font);
  hb_font_paint_glyph (font, glyph,
                       hb_vector_paint_funcs_get (),
                       paint,
                       paint->palette,
                       paint->foreground);
  paint->current_svg_image_glyph = old_gid;
  paint->current_face = old_face;
  return true;
}



/**
 * hb_vector_paint_create_or_fail:
 * @format: output format.
 *
 * Creates a new paint context for vector output.
 *
 * Return value: (nullable): a newly allocated #hb_vector_paint_t, or `NULL` on failure.
 *
 * Since: 13.0.0
 */
hb_vector_paint_t *
hb_vector_paint_create_or_fail (hb_vector_format_t format)
{
  if (format != HB_VECTOR_FORMAT_SVG)
    return nullptr;

  hb_vector_paint_t *paint = hb_object_create<hb_vector_paint_t> ();
  if (unlikely (!paint))
    return nullptr;
  paint->format = format;

  paint->defined_outlines = hb_set_create ();
  paint->defined_clips = hb_set_create ();
  paint->defs.alloc (4096);
  paint->path.alloc (2048);
  paint->subset_body_scratch.alloc (2048);
  paint->captured_scratch.alloc (4096);
  paint->color_stops_scratch.alloc (16);

  return paint;
}

/**
 * hb_vector_paint_reference:
 * @paint: a paint context.
 *
 * Increases the reference count of @paint.
 *
 * Return value: (transfer full): referenced @paint.
 *
 * Since: 13.0.0
 */
hb_vector_paint_t *
hb_vector_paint_reference (hb_vector_paint_t *paint)
{
  return hb_object_reference (paint);
}

/**
 * hb_vector_paint_destroy:
 * @paint: a paint context.
 *
 * Decreases the reference count of @paint and destroys it when it reaches zero.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_destroy (hb_vector_paint_t *paint)
{
  if (!hb_object_should_destroy (paint))
    return;

  hb_blob_destroy (paint->recycled_blob);
  hb_set_destroy (paint->defined_outlines);
  hb_set_destroy (paint->defined_clips);
  hb_object_actually_destroy (paint);
  hb_free (paint);
}

/**
 * hb_vector_paint_set_user_data:
 * @paint: a paint context.
 * @key: user-data key.
 * @data: user-data value.
 * @destroy: (nullable): destroy callback for @data.
 * @replace: whether to replace an existing value for @key.
 *
 * Attaches user data to @paint.
 *
 * Return value: `true` on success, `false` otherwise.
 *
 * Since: 13.0.0
 */
hb_bool_t
hb_vector_paint_set_user_data (hb_vector_paint_t  *paint,
                               hb_user_data_key_t *key,
                               void               *data,
                               hb_destroy_func_t   destroy,
                               hb_bool_t           replace)
{
  return hb_object_set_user_data (paint, key, data, destroy, replace);
}

/**
 * hb_vector_paint_get_user_data:
 * @paint: a paint context.
 * @key: user-data key.
 *
 * Gets previously attached user data from @paint.
 *
 * Return value: (nullable): user-data value associated with @key.
 *
 * Since: 13.0.0
 */
void *
hb_vector_paint_get_user_data (hb_vector_paint_t  *paint,
                               hb_user_data_key_t *key)
{
  return hb_object_get_user_data (paint, key);
}

/**
 * hb_vector_paint_set_transform:
 * @paint: a paint context.
 * @xx: transform xx component.
 * @yx: transform yx component.
 * @xy: transform xy component.
 * @yy: transform yy component.
 * @dx: transform x translation.
 * @dy: transform y translation.
 *
 * Sets the affine transform used when painting glyphs.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_transform (hb_vector_paint_t *paint,
                               float xx, float yx,
                               float xy, float yy,
                               float dx, float dy)
{
  paint->transform = {xx, yx, xy, yy, dx, dy};
}

/**
 * hb_vector_paint_get_transform:
 * @paint: a paint context.
 * @xx: (out) (nullable): transform xx component.
 * @yx: (out) (nullable): transform yx component.
 * @xy: (out) (nullable): transform xy component.
 * @yy: (out) (nullable): transform yy component.
 * @dx: (out) (nullable): transform x translation.
 * @dy: (out) (nullable): transform y translation.
 *
 * Gets the affine transform used when painting glyphs.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_get_transform (hb_vector_paint_t *paint,
                               float *xx, float *yx,
                               float *xy, float *yy,
                               float *dx, float *dy)
{
  if (xx) *xx = paint->transform.xx;
  if (yx) *yx = paint->transform.yx;
  if (xy) *xy = paint->transform.xy;
  if (yy) *yy = paint->transform.yy;
  if (dx) *dx = paint->transform.x0;
  if (dy) *dy = paint->transform.y0;
}

/**
 * hb_vector_paint_set_scale_factor:
 * @paint: a paint context.
 * @x_scale_factor: x scale factor.
 * @y_scale_factor: y scale factor.
 *
 * Sets additional output scaling factors.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_scale_factor (hb_vector_paint_t *paint,
                                  float x_scale_factor,
                                  float y_scale_factor)
{
  paint->x_scale_factor = x_scale_factor > 0.f ? x_scale_factor : 1.f;
  paint->y_scale_factor = y_scale_factor > 0.f ? y_scale_factor : 1.f;
}

/**
 * hb_vector_paint_get_scale_factor:
 * @paint: a paint context.
 * @x_scale_factor: (out) (nullable): x scale factor.
 * @y_scale_factor: (out) (nullable): y scale factor.
 *
 * Gets additional output scaling factors.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_get_scale_factor (hb_vector_paint_t *paint,
                                  float *x_scale_factor,
                                  float *y_scale_factor)
{
  if (x_scale_factor) *x_scale_factor = paint->x_scale_factor;
  if (y_scale_factor) *y_scale_factor = paint->y_scale_factor;
}

/**
 * hb_vector_paint_set_extents:
 * @paint: a paint context.
 * @extents: (nullable): output extents to set or expand.
 *
 * Sets or expands output extents on @paint. Passing `NULL` clears extents.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_extents (hb_vector_paint_t *paint,
                             const hb_vector_extents_t *extents)
{
  if (!extents)
  {
    paint->extents = {0, 0, 0, 0};
    paint->has_extents = false;
    return;
  }

  if (!(extents->width > 0.f && extents->height > 0.f))
    return;

  if (paint->has_extents)
  {
    float x0 = hb_min (paint->extents.x, extents->x);
    float y0 = hb_min (paint->extents.y, extents->y);
    float x1 = hb_max (paint->extents.x + paint->extents.width,
		       extents->x + extents->width);
    float y1 = hb_max (paint->extents.y + paint->extents.height,
		       extents->y + extents->height);
    paint->extents = {x0, y0, x1 - x0, y1 - y0};
  }
  else
  {
    paint->extents = *extents;
    paint->has_extents = true;
  }
}

/**
 * hb_vector_paint_get_extents:
 * @paint: a paint context.
 * @extents: (out) (nullable): where to store current output extents.
 *
 * Gets current output extents from @paint.
 *
 * Return value: `true` if extents are set, `false` otherwise.
 *
 * Since: 13.0.0
 */
hb_bool_t
hb_vector_paint_get_extents (hb_vector_paint_t *paint,
                             hb_vector_extents_t *extents)
{
  if (!paint->has_extents)
    return false;

  if (extents)
    *extents = paint->extents;
  return true;
}

/**
 * hb_vector_paint_set_glyph_extents:
 * @paint: a paint context.
 * @glyph_extents: glyph extents in font units.
 *
 * Expands @paint extents using @glyph_extents under the current transform.
 *
 * Return value: `true` on success, `false` otherwise.
 *
 * Since: 13.0.0
 */
hb_bool_t
hb_vector_paint_set_glyph_extents (hb_vector_paint_t *paint,
                                   const hb_glyph_extents_t *glyph_extents)
{
  hb_bool_t has_extents = paint->has_extents;
  hb_bool_t ret = hb_svg_set_glyph_extents_common (paint->transform,
						   paint->x_scale_factor,
						   paint->y_scale_factor,
						   glyph_extents,
						   &paint->extents,
						   &has_extents);
  paint->has_extents = has_extents;
  return ret;
}

/**
 * hb_vector_paint_set_foreground:
 * @paint: a paint context.
 * @foreground: foreground color used for COLR foreground paints.
 *
 * Sets fallback foreground color used by paint operations.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_foreground (hb_vector_paint_t *paint,
                                hb_color_t foreground)
{
  paint->foreground = foreground;
}

/**
 * hb_vector_paint_set_palette:
 * @paint: a paint context.
 * @palette: palette index for color glyph painting.
 *
 * Sets the color palette index used by paint operations.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_palette (hb_vector_paint_t *paint,
                             int palette)
{
  paint->palette = palette;
}

/**
 * hb_vector_paint_set_custom_palette_color:
 * @paint: a paint context.
 * @color_index: color index to override.
 * @color: replacement color.
 *
 * Overrides one font palette color entry for subsequent paint operations.
 * Overrides are keyed by @color_index and persist on @paint until cleared
 * (or replaced for the same index).
 *
 * These overrides are consulted by paint operations that resolve CPAL
 * entries, including SVG glyph content using `var(--colorN)`.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_custom_palette_color (hb_vector_paint_t *paint,
                                          unsigned color_index,
                                          hb_color_t color)
{
  paint->custom_palette_colors.set (color_index, color);
}

/**
 * hb_vector_paint_clear_custom_palette_colors:
 * @paint: a paint context.
 *
 * Clears all custom palette color overrides previously set on @paint.
 *
 * After this call, palette lookups use the selected font palette without
 * custom override entries.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_clear_custom_palette_colors (hb_vector_paint_t *paint)
{
  paint->custom_palette_colors.clear ();
}

/**
 * hb_vector_paint_get_funcs:
 *
 * Gets paint callbacks implemented by the vector paint backend.
 *
 * Return value: (transfer none): immutable #hb_paint_funcs_t singleton.
 *
 * Since: 13.0.0
 */
hb_paint_funcs_t *
hb_vector_paint_get_funcs (void)
{
  return hb_vector_paint_funcs_get ();
}

static hb_bool_t
hb_vector_paint_custom_palette_color (hb_paint_funcs_t *pfuncs HB_UNUSED,
                                      void *paint_data,
                                      unsigned color_index,
                                      hb_color_t *color,
                                      void *user_data HB_UNUSED)
{
  hb_vector_paint_t *paint = (hb_vector_paint_t *) paint_data;
  if (!paint || !color)
    return false;

  hb_color_t *value = nullptr;
  if (!paint->custom_palette_colors.has (color_index, &value) || !value)
    return false;
  *color = *value;
  return true;
}

/**
 * hb_vector_paint_glyph:
 * @paint: a paint context.
 * @font: font object.
 * @glyph: glyph ID.
 * @pen_x: glyph x origin before context transform.
 * @pen_y: glyph y origin before context transform.
 * @extents_mode: extents update mode.
 *
 * Paints one color glyph into @paint.
 *
 * Return value: `true` if glyph paint data was emitted, `false` otherwise.
 *
 * Since: 13.0.0
 */
hb_bool_t
hb_vector_paint_glyph (hb_vector_paint_t *paint,
		       hb_font_t         *font,
		       hb_codepoint_t     glyph,
		       float              pen_x,
		       float              pen_y,
		       hb_vector_extents_mode_t extents_mode)
{
  float xx = paint->transform.xx;
  float yx = paint->transform.yx;
  float xy = paint->transform.xy;
  float yy = paint->transform.yy;
  float tx = paint->transform.x0 + xx * pen_x + xy * pen_y;
  float ty = paint->transform.y0 + yx * pen_x + yy * pen_y;

  if (extents_mode == HB_VECTOR_EXTENTS_MODE_EXPAND)
  {
    hb_glyph_extents_t ge;
    if (hb_font_get_glyph_extents (font, glyph, &ge))
    {
      hb_bool_t has_extents = paint->has_extents;
      hb_transform_t<> extents_transform = {xx, yx, -xy, -yy, tx, ty};
      hb_bool_t ret = hb_svg_set_glyph_extents_common (extents_transform,
							paint->x_scale_factor,
							paint->y_scale_factor,
							&ge,
							&paint->extents,
							&has_extents);
      paint->has_extents = has_extents;
      (void) ret;
    }
  }

  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return false;

  bool can_cache = !paint->flat;
  hb_svg_color_glyph_cache_key_t cache_key = hb_svg_color_glyph_cache_key (glyph,
                                                                            (unsigned) paint->palette,
                                                                            paint->foreground);
  if (can_cache)
  {
    if (paint->defined_color_glyphs.has (cache_key))
    {
      uint64_t entry = paint->defined_color_glyphs.get (cache_key);
      unsigned def_id = hb_svg_cache_entry_def_id (entry);
      bool image_like = hb_svg_cache_entry_image_like (entry);
      auto &body = paint->current_body ();
      hb_svg_append_str (&body, "<use href=\"#cg");
      hb_svg_append_unsigned (&body, def_id);
      hb_svg_append_str (&body, "\" transform=\"");
      if (image_like)
        hb_svg_append_image_instance_translate (&body, paint->precision,
                                                paint->x_scale_factor,
                                                paint->y_scale_factor,
                                                tx, ty);
      else
        hb_svg_append_instance_transform (&body, paint->precision,
                                          paint->x_scale_factor,
                                          paint->y_scale_factor,
                                          xx, yx, xy, yy, tx, ty);
      hb_svg_append_str (&body, "\"/>\n");
      return true;
    }
  }

  bool has_svg_image = false;
  if (can_cache)
  {
    if (unlikely (!paint->group_stack.push_or_fail (hb_vector_t<char> {})))
      return false;
    paint->current_color_glyph_has_svg_image = false;
  }

  if (can_cache)
  {
    hb_codepoint_t old_gid = paint->current_svg_image_glyph;
    hb_face_t *old_face = paint->current_face;
    paint->current_svg_image_glyph = glyph;
    paint->current_face = hb_font_get_face (font);
    hb_bool_t ret = hb_font_paint_glyph_or_fail (font, glyph,
						  hb_vector_paint_get_funcs (), paint,
						  (unsigned) paint->palette,
						  paint->foreground);
    paint->current_svg_image_glyph = old_gid;
    paint->current_face = old_face;
    if (unlikely (!ret))
    {
      paint->group_stack.pop ();
      return false;
    }

    paint->captured_scratch = paint->group_stack.pop ();
    has_svg_image = paint->current_color_glyph_has_svg_image;
    if (unlikely (!paint->captured_scratch.length))
      return false;

    unsigned def_id = paint->color_glyph_counter++;
    if (unlikely (!paint->defined_color_glyphs.set (cache_key,
                                                    hb_svg_pack_color_glyph_cache_entry (def_id,
                                                                                         has_svg_image))))
      return false;

    hb_svg_append_str (&paint->defs, "<g id=\"cg");
    hb_svg_append_unsigned (&paint->defs, def_id);
    hb_svg_append_str (&paint->defs, "\">\n");
    hb_svg_append_len (&paint->defs,
                       paint->captured_scratch.arrayZ,
                       paint->captured_scratch.length);
    hb_svg_append_str (&paint->defs, "</g>\n");

    auto &body = paint->current_body ();
    hb_svg_append_str (&body, "<use href=\"#cg");
    hb_svg_append_unsigned (&body, def_id);
    hb_svg_append_str (&body, "\" transform=\"");
    if (has_svg_image)
      hb_svg_append_image_instance_translate (&body, paint->precision,
                                              paint->x_scale_factor,
                                              paint->y_scale_factor,
                                              tx, ty);
    else
      hb_svg_append_instance_transform (&body, paint->precision,
                                        paint->x_scale_factor,
                                        paint->y_scale_factor,
                                        xx, yx, xy, yy, tx, ty);
    hb_svg_append_str (&body, "\"/>\n");
    return true;
  }

  auto &body = paint->current_body ();
  hb_svg_append_str (&body, "<g transform=\"");
  hb_svg_append_instance_transform (&body, paint->precision,
				    paint->x_scale_factor,
				    paint->y_scale_factor,
				    xx, yx, xy, yy, tx, ty);
  hb_svg_append_str (&body, "\">\n");
  hb_codepoint_t old_gid = paint->current_svg_image_glyph;
  hb_face_t *old_face = paint->current_face;
  paint->current_svg_image_glyph = glyph;
  paint->current_face = hb_font_get_face (font);
  hb_bool_t ret = hb_font_paint_glyph_or_fail (font, glyph,
						hb_vector_paint_get_funcs (), paint,
						(unsigned) paint->palette,
						paint->foreground);
  paint->current_svg_image_glyph = old_gid;
  paint->current_face = old_face;
  hb_svg_append_str (&body, "</g>\n");
  return ret;
}

/**
 * hb_vector_svg_paint_set_flat:
 * @paint: a paint context.
 * @flat: whether to flatten paint output and disable glyph-group reuse.
 *
 * Enables or disables SVG paint flattening.
 *
 * Since: 13.0.0
 */
void
hb_vector_svg_paint_set_flat (hb_vector_paint_t *paint,
                              hb_bool_t flat)
{
  paint->flat = !!flat;
}

/**
 * hb_vector_svg_paint_set_precision:
 * @paint: a paint context.
 * @precision: decimal precision.
 *
 * Sets numeric output precision for SVG paint output.
 *
 * Since: 13.0.0
 */
void
hb_vector_svg_paint_set_precision (hb_vector_paint_t *paint,
                                   unsigned precision)
{
  paint->precision = hb_min (precision, 12u);
}

static void
hb_vector_paint_clear_render_state (hb_vector_paint_t *paint)
{
  paint->extents = {0, 0, 0, 0};
  paint->has_extents = false;

  paint->defs.clear ();
  paint->path.clear ();
  paint->group_stack.clear ();
  paint->transform_group_open_mask = 0;
  paint->transform_group_depth = 0;
  paint->transform_group_overflow_depth = 0;
  paint->clip_rect_counter = 0;
  paint->gradient_counter = 0;
  paint->color_glyph_counter = 0;
  paint->current_color_glyph_has_svg_image = false;
  paint->current_svg_image_glyph = HB_CODEPOINT_INVALID;
  paint->current_face = nullptr;
  paint->svg_image_counter = 0;
  hb_set_clear (paint->defined_outlines);
  hb_set_clear (paint->defined_clips);
  paint->defined_color_glyphs.clear ();
  paint->color_stops_scratch.clear ();
  paint->subset_body_scratch.clear ();
  paint->captured_scratch.clear ();
}

/**
 * hb_vector_paint_render:
 * @paint: a paint context.
 *
 * Renders accumulated paint content to an SVG blob.
 *
 * Return value: (transfer full) (nullable): output blob, or `NULL` if rendering cannot proceed.
 *
 * Since: 13.0.0
 */
hb_blob_t *
hb_vector_paint_render (hb_vector_paint_t *paint)
{
  if (paint->format != HB_VECTOR_FORMAT_SVG)
    return nullptr;
  if (!paint->has_extents)
    return nullptr;

  if (unlikely (!hb_vector_paint_ensure_initialized (paint)))
    return nullptr;

  hb_vector_t<char> out;
  hb_svg_recover_recycled_buffer (paint->recycled_blob, &out);
  unsigned estimated = paint->defs.length +
                       paint->group_stack.arrayZ[0].length +
                       320;
  out.alloc (estimated);
  hb_svg_append_str (&out, "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" viewBox=\"");
  hb_svg_append_num (&out, paint->extents.x, paint->precision);
  hb_svg_append_c (&out, ' ');
  hb_svg_append_num (&out, paint->extents.y, paint->precision);
  hb_svg_append_c (&out, ' ');
  hb_svg_append_num (&out, paint->extents.width, paint->precision);
  hb_svg_append_c (&out, ' ');
  hb_svg_append_num (&out, paint->extents.height, paint->precision);
  hb_svg_append_str (&out, "\" width=\"");
  hb_svg_append_num (&out, paint->extents.width, paint->precision);
  hb_svg_append_str (&out, "\" height=\"");
  hb_svg_append_num (&out, paint->extents.height, paint->precision);
  hb_svg_append_str (&out, "\">\n");

  if (paint->defs.length)
  {
    hb_svg_append_str (&out, "<defs>\n");
    hb_svg_append_len (&out, paint->defs.arrayZ, paint->defs.length);
    hb_svg_append_str (&out, "</defs>\n");
  }

  paint->append_global_transform_prefix (&out);
  hb_svg_append_len (&out, paint->group_stack.arrayZ[0].arrayZ, paint->group_stack.arrayZ[0].length);
  paint->append_global_transform_suffix (&out);

  hb_svg_append_str (&out, "</svg>\n");

  hb_blob_t *blob = hb_svg_blob_from_buffer (&paint->recycled_blob, &out);

  hb_vector_paint_clear_render_state (paint);

  return blob;
}

/**
 * hb_vector_paint_reset:
 * @paint: a paint context.
 *
 * Resets @paint state and clears accumulated content.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_reset (hb_vector_paint_t *paint)
{
  paint->transform = {1, 0, 0, 1, 0, 0};
  paint->x_scale_factor = 1.f;
  paint->y_scale_factor = 1.f;
  paint->foreground = HB_COLOR (0, 0, 0, 255);
  paint->palette = 0;
  paint->precision = 2;
  paint->flat = false;
  hb_vector_paint_clear_render_state (paint);
}

/**
 * hb_vector_paint_recycle_blob:
 * @paint: a paint context.
 * @blob: (nullable): previously rendered blob to recycle.
 *
 * Provides a blob for internal buffer reuse by later render calls.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_recycle_blob (hb_vector_paint_t *paint,
                              hb_blob_t *blob)
{
  hb_blob_destroy (paint->recycled_blob);
  paint->recycled_blob = nullptr;
  if (!blob || blob == hb_blob_get_empty ())
    return;
  paint->recycled_blob = blob;
}
