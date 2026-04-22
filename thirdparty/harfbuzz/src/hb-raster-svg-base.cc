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

#ifndef HB_NO_RASTER_SVG

#include "hb.hh"

#include "hb-raster-svg-base.hh"
#include "hb-raster-svg-parse.hh"

void
svg_parse_style_props (hb_svg_str_t style, hb_svg_style_props_t *out)
{
  if (style.is_null ()) return;
  const char *p = style.data;
  const char *end = style.data + style.len;
  while (p < end)
  {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ';'))
      p++;
    if (p >= end) break;
    const char *name_start = p;
    while (p < end && *p != ':' && *p != ';')
      p++;
    const char *name_end = p;
    while (name_end > name_start &&
           (name_end[-1] == ' ' || name_end[-1] == '\t' || name_end[-1] == '\n' || name_end[-1] == '\r'))
      name_end--;
    if (p >= end || *p != ':')
    {
      while (p < end && *p != ';') p++;
      continue;
    }
    p++; /* skip ':' */
    const char *value_start = p;
    while (p < end && *p != ';')
      p++;
    const char *value_end = p;
    while (value_start < value_end &&
           (*value_start == ' ' || *value_start == '\t' || *value_start == '\n' || *value_start == '\r'))
      value_start++;
    while (value_end > value_start &&
           (value_end[-1] == ' ' || value_end[-1] == '\t' || value_end[-1] == '\n' || value_end[-1] == '\r'))
      value_end--;

    hb_svg_str_t prop_name = {name_start, (unsigned) (name_end - name_start)};
    hb_svg_str_t prop_value = {value_start, (unsigned) (value_end - value_start)};
    if (prop_name.len)
    {
      switch (hb_svg_ascii_lower (prop_name.data[0]))
      {
      case 'c':
        if (prop_name.len == 9 && prop_name.eq_ascii_ci ("clip-path")) out->clip_path = prop_value;
        else if (prop_name.len == 5 && prop_name.eq_ascii_ci ("color")) out->color = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("cx")) out->cx = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("cy")) out->cy = prop_value;
        break;
      case 'd':
        if (prop_name.len == 7 && prop_name.eq_ascii_ci ("display")) out->display = prop_value;
        else if (prop_name.len == 1 && prop_name.eq_ascii_ci ("d")) out->d = prop_value;
        break;
      case 'f':
        if (prop_name.len == 4 && prop_name.eq_ascii_ci ("fill")) out->fill = prop_value;
        else if (prop_name.len == 12 && prop_name.eq_ascii_ci ("fill-opacity")) out->fill_opacity = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("fx")) out->fx = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("fy")) out->fy = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("fr")) out->fr = prop_value;
        break;
      case 'g':
        if ((prop_name.len == 13 && prop_name.eq_ascii_ci ("gradientunits")) ||
            (prop_name.len == 14 && prop_name.eq_ascii_ci ("gradient-units"))) out->gradient_units = prop_value;
        else if ((prop_name.len == 17 && prop_name.eq_ascii_ci ("gradienttransform")) ||
                 (prop_name.len == 18 && prop_name.eq_ascii_ci ("gradient-transform"))) out->gradient_transform = prop_value;
        break;
      case 'h':
        if (prop_name.len == 6 && prop_name.eq_ascii_ci ("height")) out->height = prop_value;
        break;
      case 'o':
        if (prop_name.len == 7 && prop_name.eq_ascii_ci ("opacity")) out->opacity = prop_value;
        else if (prop_name.len == 6 && prop_name.eq_ascii_ci ("offset")) out->offset = prop_value;
        break;
      case 'p':
        if (prop_name.len == 6 && prop_name.eq_ascii_ci ("points")) out->points = prop_value;
        break;
      case 'r':
        if (prop_name.len == 1 && prop_name.eq_ascii_ci ("r")) out->r = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("rx")) out->rx = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("ry")) out->ry = prop_value;
        break;
      case 's':
        if (prop_name.len == 10 && prop_name.eq_ascii_ci ("stop-color")) out->stop_color = prop_value;
        else if (prop_name.len == 12 && prop_name.eq_ascii_ci ("stop-opacity")) out->stop_opacity = prop_value;
        else if ((prop_name.len == 12 && prop_name.eq_ascii_ci ("spreadmethod")) ||
                 (prop_name.len == 13 && prop_name.eq_ascii_ci ("spread-method"))) out->spread_method = prop_value;
        break;
      case 't':
        if (prop_name.len == 9 && prop_name.eq_ascii_ci ("transform")) out->transform = prop_value;
        break;
      case 'v':
        if (prop_name.len == 10 && prop_name.eq_ascii_ci ("visibility")) out->visibility = prop_value;
        break;
      case 'w':
        if (prop_name.len == 5 && prop_name.eq_ascii_ci ("width")) out->width = prop_value;
        break;
      case 'x':
        if (prop_name.len == 1 && prop_name.eq_ascii_ci ("x")) out->x = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("x1")) out->x1 = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("x2")) out->x2 = prop_value;
        break;
      case 'y':
        if (prop_name.len == 1 && prop_name.eq_ascii_ci ("y")) out->y = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("y1")) out->y1 = prop_value;
        else if (prop_name.len == 2 && prop_name.eq_ascii_ci ("y2")) out->y2 = prop_value;
        break;
      default:
        break;
      }
    }

    if (p < end && *p == ';') p++;
  }
}

float
svg_parse_number_or_percent (hb_svg_str_t s, bool *is_percent)
{
  if (is_percent) *is_percent = false;
  s = s.trim ();
  if (!s.len) return 0.f;
  if (s.data[s.len - 1] == '%')
  {
    if (is_percent) *is_percent = true;
    hb_svg_str_t n = {s.data, s.len - 1};
    return n.to_float () / 100.f;
  }
  return s.to_float ();
}

hb_svg_str_t
hb_raster_svg_find_href_attr (const hb_svg_xml_parser_t &parser)
{
  hb_svg_str_t href = parser.find_attr ("href");
  if (href.is_null ())
    href = parser.find_attr ("xlink:href");
  return href;
}

bool
hb_raster_svg_parse_id_ref (hb_svg_str_t s,
                            hb_svg_str_t *out_id,
                            hb_svg_str_t *out_tail)
{
  if (out_id) *out_id = {};
  if (out_tail) *out_tail = {};
  s = s.trim ();

  if (s.len && s.data[0] == '#')
  {
    hb_svg_str_t id = {s.data + 1, s.len - 1};
    id = id.trim ();
    if (!id.len)
      return false;
    if (out_id) *out_id = id;
    return true;
  }

  if (!s.starts_with_ascii_ci ("url("))
    return false;

  const char *p = s.data + 4;
  const char *end = s.data + s.len;
  while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;

  const char *q = p;
  char quote = 0;
  while (q < end)
  {
    char c = *q;
    if (quote)
    {
      if (c == quote) quote = 0;
    }
    else
    {
      if (c == '"' || c == '\'') quote = c;
      else if (c == ')') break;
    }
    q++;
  }
  if (q >= end || *q != ')')
    return false;

  hb_svg_str_t id = {(const char *) p, (unsigned) (q - p)};
  id = id.trim ();
  if (id.len >= 2 &&
      ((id.data[0] == '\'' && id.data[id.len - 1] == '\'') ||
       (id.data[0] == '"' && id.data[id.len - 1] == '"')))
  {
    id.data++;
    id.len -= 2;
  }
  id = id.trim ();
  if (id.len && id.data[0] == '#')
  {
    id.data++;
    id.len--;
  }
  id = id.trim ();
  if (!id.len)
    return false;

  if (out_id) *out_id = id;
  if (out_tail)
  {
    const char *tail = q + 1;
    while (tail < end && (*tail == ' ' || *tail == '\t' || *tail == '\n' || *tail == '\r')) tail++;
    *out_tail = {(const char *) tail, (unsigned) (end - tail)};
  }
  return true;
}

bool
hb_raster_svg_parse_local_id_ref (hb_svg_str_t s,
                                  hb_svg_str_t *out_id,
                                  hb_svg_str_t *out_tail)
{
  if (out_id) *out_id = {};
  if (out_tail) *out_tail = {};
  s = s.trim ();

  if (s.len && s.data[0] == '#')
  {
    hb_svg_str_t id = {s.data + 1, s.len - 1};
    id = id.trim ();
    if (!id.len)
      return false;
    if (out_id) *out_id = id;
    return true;
  }

  if (!s.starts_with_ascii_ci ("url("))
    return false;

  hb_svg_str_t id;
  if (!hb_raster_svg_parse_id_ref (s, &id, out_tail))
    return false;

  const char *p = s.data + 4;
  const char *end = s.data + s.len;
  while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  if (p < end && (*p == '\'' || *p == '"'))
  {
    p++;
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (p >= end || *p != '#')
      return false;
  }
  else if (p >= end || *p != '#')
    return false;

  if (out_id) *out_id = id;
  return true;
}

bool
hb_raster_svg_find_element_by_id (const char *doc_start,
                                  unsigned doc_len,
                                  const OT::SVG::accelerator_t *svg_accel,
                                  const OT::SVG::svg_doc_cache_t *doc_cache,
                                  hb_svg_str_t id,
                                  const char **found)
{
  *found = nullptr;
  if (!doc_start || !doc_len || !id.len)
    return false;

  if (doc_cache && svg_accel)
  {
    unsigned start = 0, end = 0;
    OT::SVG::svg_id_span_t key = {id.data, id.len};
    if (svg_accel->doc_cache_find_id_span (doc_cache, key, &start, &end))
    {
      if (start < doc_len && end <= doc_len && start < end)
      {
        *found = doc_start + start;
        return true;
      }
    }
  }

  hb_svg_xml_parser_t search (doc_start, doc_len);
  while (true)
  {
    hb_svg_token_type_t tok = search.next ();
    if (tok == SVG_TOKEN_EOF) break;
    if (tok != SVG_TOKEN_OPEN_TAG && tok != SVG_TOKEN_SELF_CLOSE_TAG) continue;
    hb_svg_str_t attr_id = search.find_attr ("id");
    if (attr_id.len == id.len && 0 == memcmp (attr_id.data, id.data, id.len))
    {
      *found = search.tag_start;
      return true;
    }
  }
  return false;
}

bool
hb_raster_svg_parse_viewbox (hb_svg_str_t viewbox_str,
                             float *x,
                             float *y,
                             float *w,
                             float *h)
{
  if (!viewbox_str.len)
    return false;
  hb_svg_float_parser_t vb_fp (viewbox_str);
  float vb_x = vb_fp.next_float ();
  float vb_y = vb_fp.next_float ();
  float vb_w = vb_fp.next_float ();
  float vb_h = vb_fp.next_float ();
  if (vb_w <= 0.f || vb_h <= 0.f)
    return false;
  if (x) *x = vb_x;
  if (y) *y = vb_y;
  if (w) *w = vb_w;
  if (h) *h = vb_h;
  return true;
}

static inline float
svg_align_offset (hb_svg_str_t align,
                  float leftover,
                  char axis)
{
  if (leftover <= 0.f) return 0.f;
  if ((axis == 'x' && align.starts_with_ascii_ci ("xMin")) ||
      (axis == 'y' && (align.eq_ascii_ci ("xMinYMin") ||
                       align.eq_ascii_ci ("xMidYMin") ||
                       align.eq_ascii_ci ("xMaxYMin"))))
    return 0.f;
  if ((axis == 'x' && align.starts_with_ascii_ci ("xMax")) ||
      (axis == 'y' && (align.eq_ascii_ci ("xMinYMax") ||
                       align.eq_ascii_ci ("xMidYMax") ||
                       align.eq_ascii_ci ("xMaxYMax"))))
    return leftover;
  return leftover * 0.5f;
}

bool
hb_raster_svg_compute_viewbox_transform (float viewport_w,
                                         float viewport_h,
                                         float vb_x,
                                         float vb_y,
                                         float vb_w,
                                         float vb_h,
                                         hb_svg_str_t preserve_aspect_ratio,
                                         hb_svg_transform_t *out)
{
  if (!(viewport_w > 0.f && viewport_h > 0.f && vb_w > 0.f && vb_h > 0.f))
    return false;

  hb_svg_str_t par = preserve_aspect_ratio.trim ();
  if (par.starts_with_ascii_ci ("defer"))
  {
    const char *p = par.data + 5;
    const char *end = par.data + par.len;
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    par = hb_svg_str_t (p, (unsigned) (end - p));
  }
  if (!par.len)
    par = hb_svg_str_t ("xMidYMid meet", 12);

  bool is_none = false;
  bool is_slice = false;
  hb_svg_str_t align = par;
  const char *p = par.data;
  const char *end = par.data + par.len;
  while (p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
  align = hb_svg_str_t (par.data, (unsigned) (p - par.data)).trim ();

  if (par.starts_with_ascii_ci ("none"))
    is_none = true;
  else if (align.starts_with_ascii_ci ("x"))
  {
    const char *mode = p;
    while (mode < end && (*mode == ' ' || *mode == '\t' || *mode == '\n' || *mode == '\r')) mode++;
    if (mode < end && hb_svg_str_t (mode, (unsigned) (end - mode)).starts_with_ascii_ci ("slice"))
      is_slice = true;
  }

  hb_svg_transform_t t;
  if (is_none)
  {
    t.xx = viewport_w / vb_w;
    t.yy = viewport_h / vb_h;
    t.dx = -vb_x * t.xx;
    t.dy = -vb_y * t.yy;
    *out = t;
    return true;
  }

  float sx = viewport_w / vb_w;
  float sy = viewport_h / vb_h;
  float s = is_slice ? hb_max (sx, sy) : hb_min (sx, sy);
  float scaled_w = vb_w * s;
  float scaled_h = vb_h * s;
  float leftover_x = viewport_w - scaled_w;
  float leftover_y = viewport_h - scaled_h;

  if (!align.starts_with_ascii_ci ("x"))
    align = hb_svg_str_t ("xMidYMid", 8);

  t.xx = s;
  t.yy = s;
  t.dx = svg_align_offset (align, leftover_x, 'x') - vb_x * s;
  t.dy = svg_align_offset (align, leftover_y, 'y') - vb_y * s;
  *out = t;
  return true;
}

bool
hb_raster_svg_compute_use_target_viewbox_transform (hb_svg_xml_parser_t &target_parser,
                                                     float use_w,
                                                     float use_h,
                                                     hb_svg_transform_t *out)
{
  if (!(target_parser.tag_name.eq ("svg") || target_parser.tag_name.eq ("symbol")))
    return false;

  float viewport_w = use_w;
  float viewport_h = use_h;
  hb_svg_style_props_t target_style_props;
  svg_parse_style_props (target_parser.find_attr ("style"), &target_style_props);
  if (viewport_w <= 0.f)
    viewport_w = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (target_parser, target_style_props.width, "width"));
  if (viewport_h <= 0.f)
    viewport_h = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (target_parser, target_style_props.height, "height"));

  float vb_x = 0.f, vb_y = 0.f, vb_w = 0.f, vb_h = 0.f;
  if (!hb_raster_svg_parse_viewbox (target_parser.find_attr ("viewBox"), &vb_x, &vb_y, &vb_w, &vb_h))
    return false;

  if (!(viewport_w > 0.f && viewport_h > 0.f))
  {
    viewport_w = vb_w;
    viewport_h = vb_h;
  }

  return hb_raster_svg_compute_viewbox_transform (viewport_w, viewport_h, vb_x, vb_y, vb_w, vb_h,
                                                   target_parser.find_attr ("preserveAspectRatio"),
                                                   out);
}

void
hb_raster_svg_parse_use_geometry (hb_svg_xml_parser_t &parser,
                                  float *x,
                                  float *y,
                                  float *w,
                                  float *h)
{
  hb_svg_style_props_t style_props;
  svg_parse_style_props (parser.find_attr ("style"), &style_props);

  if (x) *x = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, style_props.x, "x"));
  if (y) *y = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, style_props.y, "y"));
  if (w) *w = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, style_props.width, "width"));
  if (h) *h = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, style_props.height, "height"));
}

float
hb_raster_svg_parse_non_percent_length (hb_svg_str_t s)
{
  bool is_percent = false;
  float v = svg_parse_number_or_percent (s, &is_percent);
  return is_percent ? 0.f : v;
}

#endif /* !HB_NO_RASTER_SVG */
