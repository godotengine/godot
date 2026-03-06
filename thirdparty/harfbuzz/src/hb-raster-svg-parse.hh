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

#ifndef HB_RASTER_SVG_PARSE_HH
#define HB_RASTER_SVG_PARSE_HH

#include "hb.hh"

#include "hb-raster-svg-base.hh"
#include "hb-draw.h"

#include <stdlib.h>
#include <string.h>

enum hb_svg_token_type_t
{
  SVG_TOKEN_OPEN_TAG,
  SVG_TOKEN_CLOSE_TAG,
  SVG_TOKEN_SELF_CLOSE_TAG,
  SVG_TOKEN_TEXT,
  SVG_TOKEN_EOF
};

struct hb_svg_attr_t
{
  hb_svg_str_t name;
  hb_svg_str_t value;
};

struct hb_svg_xml_parser_t
{
  enum { SVG_MAX_ATTRS_PER_TAG = 256 };

  const char *p;
  const char *end;
  const char *tag_start = nullptr;

  hb_svg_str_t tag_name;
  hb_vector_t<hb_svg_attr_t> attrs;
  bool self_closing;

  hb_svg_xml_parser_t (const char *data, unsigned len)
    : p (data), end (data + len), self_closing (false) {}

  void skip_ws ()
  {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
      p++;
  }

  hb_svg_str_t read_name ()
  {
    const char *start = p;
    while (p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r' &&
           *p != '>' && *p != '/' && *p != '=')
      p++;
    return {start, (unsigned) (p - start)};
  }

  hb_svg_str_t read_attr_value ()
  {
    if (p >= end) return {};
    char quote = *p;
    if (quote != '"' && quote != '\'') return {};
    p++;
    const char *start = p;
    while (p < end && *p != quote)
      p++;
    hb_svg_str_t val = {start, (unsigned) (p - start)};
    if (p < end) p++; /* skip closing quote */
    return val;
  }

  void parse_attrs ()
  {
    attrs.clear ();
    while (p < end)
    {
      skip_ws ();
      if (p >= end || *p == '>' || *p == '/') break;
      hb_svg_str_t name = read_name ();
      if (!name.len) { p++; continue; }
      skip_ws ();
      if (p < end && *p == '=')
      {
        p++;
        skip_ws ();
        hb_svg_str_t val = read_attr_value ();
        if (attrs.length < SVG_MAX_ATTRS_PER_TAG)
        {
          hb_svg_attr_t attr = {name, val};
          attrs.push (attr);
        }
      }
      else
      {
        if (attrs.length < SVG_MAX_ATTRS_PER_TAG)
        {
          hb_svg_attr_t attr = {name, {}};
          attrs.push (attr);
        }
      }
    }
  }

  hb_svg_token_type_t next ()
  {
    while (p < end)
    {
      if (*p != '<')
      {
        while (p < end && *p != '<') p++;
        continue;
      }

      tag_start = p;
      p++; /* skip '<' */
      if (p >= end) return SVG_TOKEN_EOF;

      if (*p == '!')
      {
        if (p + 2 < end && p[1] == '-' && p[2] == '-')
        {
          p += 3;
          while (p + 2 < end && !(p[0] == '-' && p[1] == '-' && p[2] == '>'))
            p++;
          if (p + 2 < end) p += 3;
        }
        else
        {
          while (p < end && *p != '>') p++;
          if (p < end) p++;
        }
        continue;
      }
      if (*p == '?')
      {
        while (p + 1 < end && !(p[0] == '?' && p[1] == '>'))
          p++;
        if (p + 1 < end) p += 2;
        continue;
      }

      if (*p == '/')
      {
        p++;
        tag_name = read_name ();
        while (p < end && *p != '>') p++;
        if (p < end) p++;
        attrs.clear ();
        self_closing = false;
        return SVG_TOKEN_CLOSE_TAG;
      }

      tag_name = read_name ();
      parse_attrs ();

      self_closing = false;
      skip_ws ();
      if (p < end && *p == '/')
      {
        self_closing = true;
        p++;
      }
      if (p < end && *p == '>')
        p++;

      return self_closing ? SVG_TOKEN_SELF_CLOSE_TAG : SVG_TOKEN_OPEN_TAG;
    }
    return SVG_TOKEN_EOF;
  }

  hb_svg_str_t find_attr (const char *name) const
  {
    for (unsigned i = 0; i < attrs.length; i++)
      if (attrs[i].name.eq (name))
        return attrs[i].value;
    return {};
  }
};

static inline hb_svg_str_t
svg_pick_attr_or_style (const hb_svg_xml_parser_t &parser,
                        hb_svg_str_t style_value,
                        const char *attr_name)
{
  return style_value.is_null () ? parser.find_attr (attr_name) : style_value;
}

struct hb_svg_attr_view_t
{
  const hb_svg_xml_parser_t &parser;
  enum { CACHE_SIZE = 16 };
  struct entry_t
  {
    const char *name = nullptr;
    hb_svg_str_t value;
  };
  entry_t cache[CACHE_SIZE];
  unsigned cache_len = 0;

  hb_svg_attr_view_t (const hb_svg_xml_parser_t &p) : parser (p) {}

  hb_svg_str_t get (const char *name)
  {
    for (unsigned i = 0; i < cache_len; i++)
      if (strcmp (cache[i].name, name) == 0)
        return cache[i].value;

    hb_svg_str_t value = parser.find_attr (name);
    if (cache_len < CACHE_SIZE)
    {
      cache[cache_len].name = name;
      cache[cache_len].value = value;
      cache_len++;
    }
    return value;
  }
};

struct hb_svg_transform_t
{
  float xx = 1, yx = 0, xy = 0, yy = 1, dx = 0, dy = 0;

  void multiply (const hb_svg_transform_t &other)
  {
    float nxx = xx * other.xx + xy * other.yx;
    float nyx = yx * other.xx + yy * other.yx;
    float nxy = xx * other.xy + xy * other.yy;
    float nyy = yx * other.xy + yy * other.yy;
    float ndx = xx * other.dx + xy * other.dy + dx;
    float ndy = yx * other.dx + yy * other.dy + dy;
    xx = nxx; yx = nyx; xy = nxy; yy = nyy; dx = ndx; dy = ndy;
  }
};

struct hb_svg_float_parser_t
{
  const char *p;
  const char *end;

  hb_svg_float_parser_t (hb_svg_str_t s) : p (s.data), end (s.data + s.len) {}

  void skip_ws_comma ()
  {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ','))
      p++;
  }

  bool has_more () const { return p < end; }

  float next_float ()
  {
    skip_ws_comma ();
    if (p >= end) return 0.f;
    const char *start = p;
    char buf[64];
    unsigned n = 0;
    bool has_digit = false;
    if (p < end && (*p == '-' || *p == '+'))
      buf[n++] = *p++;
    while (p < end && n < sizeof (buf) - 1 && *p >= '0' && *p <= '9')
    {
      buf[n++] = *p++;
      has_digit = true;
    }
    if (p < end && n < sizeof (buf) - 1 && *p == '.')
    {
      buf[n++] = *p++;
      while (p < end && n < sizeof (buf) - 1 && *p >= '0' && *p <= '9')
      {
        buf[n++] = *p++;
        has_digit = true;
      }
    }
    if (p < end && n < sizeof (buf) - 1 && (*p == 'e' || *p == 'E'))
    {
      buf[n++] = *p++;
      if (p < end && n < sizeof (buf) - 1 && (*p == '+' || *p == '-'))
        buf[n++] = *p++;
      bool has_exp_digit = false;
      while (p < end && n < sizeof (buf) - 1 && *p >= '0' && *p <= '9')
      {
        buf[n++] = *p++;
        has_exp_digit = true;
      }
      if (!has_exp_digit)
      {
        p = start;
        has_digit = false;
      }
    }
    if (!has_digit)
    {
      if (p < end) p++;
      return 0.f;
    }
    buf[n] = '\0';
    float v = strtof (buf, nullptr);
    return isfinite (v) ? v : 0.f;
  }

  bool next_flag ()
  {
    skip_ws_comma ();
    if (p >= end) return false;
    bool v = *p != '0';
    p++;
    return v;
  }
};

static inline float
svg_parse_float (hb_svg_str_t s)
{
  return s.to_float ();
}

struct hb_svg_shape_emit_data_t
{
  enum { SHAPE_PATH, SHAPE_RECT, SHAPE_CIRCLE, SHAPE_ELLIPSE,
         SHAPE_LINE, SHAPE_POLYLINE, SHAPE_POLYGON } type;
  hb_svg_str_t str_data;
  float params[6];
};

HB_INTERNAL bool hb_raster_svg_parse_transform (hb_svg_str_t s, hb_svg_transform_t *out);
HB_INTERNAL void hb_raster_svg_parse_path_data (hb_svg_str_t d, hb_draw_funcs_t *dfuncs, void *draw_data);
HB_INTERNAL void hb_raster_svg_shape_path_emit (hb_draw_funcs_t *dfuncs, void *draw_data, void *user_data);
HB_INTERNAL bool hb_raster_svg_parse_shape_tag (hb_svg_xml_parser_t &parser, hb_svg_shape_emit_data_t *shape);

#endif /* HB_RASTER_SVG_PARSE_HH */
