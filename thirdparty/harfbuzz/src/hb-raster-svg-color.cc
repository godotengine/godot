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

#include "hb-raster-svg-color.hh"

#include "hb-raster.h"
#include "hb-raster-svg-base.hh"
#include "hb-ot-color.h"

#include <string.h>
#include <stdlib.h>

struct hb_svg_named_color_t
{
  const char *name;
  uint32_t rgb; /* 0xRRGGBB */
};

static const hb_svg_named_color_t svg_named_colors[] = {
  {"aliceblue", 0xF0F8FF},
  {"antiquewhite", 0xFAEBD7},
  {"aqua", 0x00FFFF},
  {"aquamarine", 0x7FFFD4},
  {"azure", 0xF0FFFF},
  {"beige", 0xF5F5DC},
  {"bisque", 0xFFE4C4},
  {"black", 0x000000},
  {"blanchedalmond", 0xFFEBCD},
  {"blue", 0x0000FF},
  {"blueviolet", 0x8A2BE2},
  {"brown", 0xA52A2A},
  {"burlywood", 0xDEB887},
  {"cadetblue", 0x5F9EA0},
  {"chartreuse", 0x7FFF00},
  {"chocolate", 0xD2691E},
  {"coral", 0xFF7F50},
  {"cornflowerblue", 0x6495ED},
  {"cornsilk", 0xFFF8DC},
  {"crimson", 0xDC143C},
  {"cyan", 0x00FFFF},
  {"darkblue", 0x00008B},
  {"darkcyan", 0x008B8B},
  {"darkgoldenrod", 0xB8860B},
  {"darkgray", 0xA9A9A9},
  {"darkgreen", 0x006400},
  {"darkgrey", 0xA9A9A9},
  {"darkkhaki", 0xBDB76B},
  {"darkmagenta", 0x8B008B},
  {"darkolivegreen", 0x556B2F},
  {"darkorange", 0xFF8C00},
  {"darkorchid", 0x9932CC},
  {"darkred", 0x8B0000},
  {"darksalmon", 0xE9967A},
  {"darkseagreen", 0x8FBC8F},
  {"darkslateblue", 0x483D8B},
  {"darkslategray", 0x2F4F4F},
  {"darkslategrey", 0x2F4F4F},
  {"darkturquoise", 0x00CED1},
  {"darkviolet", 0x9400D3},
  {"deeppink", 0xFF1493},
  {"deepskyblue", 0x00BFFF},
  {"dimgray", 0x696969},
  {"dimgrey", 0x696969},
  {"dodgerblue", 0x1E90FF},
  {"firebrick", 0xB22222},
  {"floralwhite", 0xFFFAF0},
  {"forestgreen", 0x228B22},
  {"fuchsia", 0xFF00FF},
  {"gainsboro", 0xDCDCDC},
  {"ghostwhite", 0xF8F8FF},
  {"gold", 0xFFD700},
  {"goldenrod", 0xDAA520},
  {"gray", 0x808080},
  {"green", 0x008000},
  {"greenyellow", 0xADFF2F},
  {"grey", 0x808080},
  {"honeydew", 0xF0FFF0},
  {"hotpink", 0xFF69B4},
  {"indianred", 0xCD5C5C},
  {"indigo", 0x4B0082},
  {"ivory", 0xFFFFF0},
  {"khaki", 0xF0E68C},
  {"lavender", 0xE6E6FA},
  {"lavenderblush", 0xFFF0F5},
  {"lawngreen", 0x7CFC00},
  {"lemonchiffon", 0xFFFACD},
  {"lightblue", 0xADD8E6},
  {"lightcoral", 0xF08080},
  {"lightcyan", 0xE0FFFF},
  {"lightgoldenrodyellow", 0xFAFAD2},
  {"lightgray", 0xD3D3D3},
  {"lightgreen", 0x90EE90},
  {"lightgrey", 0xD3D3D3},
  {"lightpink", 0xFFB6C1},
  {"lightsalmon", 0xFFA07A},
  {"lightseagreen", 0x20B2AA},
  {"lightskyblue", 0x87CEFA},
  {"lightslategray", 0x778899},
  {"lightslategrey", 0x778899},
  {"lightsteelblue", 0xB0C4DE},
  {"lightyellow", 0xFFFFE0},
  {"lime", 0x00FF00},
  {"limegreen", 0x32CD32},
  {"linen", 0xFAF0E6},
  {"magenta", 0xFF00FF},
  {"maroon", 0x800000},
  {"mediumaquamarine", 0x66CDAA},
  {"mediumblue", 0x0000CD},
  {"mediumorchid", 0xBA55D3},
  {"mediumpurple", 0x9370DB},
  {"mediumseagreen", 0x3CB371},
  {"mediumslateblue", 0x7B68EE},
  {"mediumspringgreen", 0x00FA9A},
  {"mediumturquoise", 0x48D1CC},
  {"mediumvioletred", 0xC71585},
  {"midnightblue", 0x191970},
  {"mintcream", 0xF5FFFA},
  {"mistyrose", 0xFFE4E1},
  {"moccasin", 0xFFE4B5},
  {"navajowhite", 0xFFDEAD},
  {"navy", 0x000080},
  {"oldlace", 0xFDF5E6},
  {"olive", 0x808000},
  {"olivedrab", 0x6B8E23},
  {"orange", 0xFFA500},
  {"orangered", 0xFF4500},
  {"orchid", 0xDA70D6},
  {"palegoldenrod", 0xEEE8AA},
  {"palegreen", 0x98FB98},
  {"paleturquoise", 0xAFEEEE},
  {"palevioletred", 0xDB7093},
  {"papayawhip", 0xFFEFD5},
  {"peachpuff", 0xFFDAB9},
  {"peru", 0xCD853F},
  {"pink", 0xFFC0CB},
  {"plum", 0xDDA0DD},
  {"powderblue", 0xB0E0E6},
  {"purple", 0x800080},
  {"rebeccapurple", 0x663399},
  {"red", 0xFF0000},
  {"rosybrown", 0xBC8F8F},
  {"royalblue", 0x4169E1},
  {"saddlebrown", 0x8B4513},
  {"salmon", 0xFA8072},
  {"sandybrown", 0xF4A460},
  {"seagreen", 0x2E8B57},
  {"seashell", 0xFFF5EE},
  {"sienna", 0xA0522D},
  {"silver", 0xC0C0C0},
  {"skyblue", 0x87CEEB},
  {"slateblue", 0x6A5ACD},
  {"slategray", 0x708090},
  {"slategrey", 0x708090},
  {"snow", 0xFFFAFA},
  {"springgreen", 0x00FF7F},
  {"steelblue", 0x4682B4},
  {"tan", 0xD2B48C},
  {"teal", 0x008080},
  {"thistle", 0xD8BFD8},
  {"tomato", 0xFF6347},
  {"turquoise", 0x40E0D0},
  {"violet", 0xEE82EE},
  {"wheat", 0xF5DEB3},
  {"white", 0xFFFFFF},
  {"whitesmoke", 0xF5F5F5},
  {"yellow", 0xFFFF00},
  {"yellowgreen", 0x9ACD32},
};

static int
hexval (char c)
{
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* Parse SVG color value; returns HB_COLOR with alpha = 255.
 * Sets *is_none if "none". */
hb_color_t
hb_raster_svg_parse_color (hb_svg_str_t s,
		 hb_paint_funcs_t *pfuncs,
		 void *paint_data,
		 hb_color_t foreground,
		 hb_face_t *face,
		 unsigned palette,
		 bool *is_none)
{
  *is_none = false;
  s = s.trim ();
  if (!s.len) { *is_none = true; return HB_COLOR (0, 0, 0, 0); }

  if (s.eq_ascii_ci ("none") || s.eq_ascii_ci ("transparent"))
  {
    *is_none = true;
    return HB_COLOR (0, 0, 0, 0);
  }

  if (s.eq_ascii_ci ("currentColor"))
    return foreground;

  /* var(--colorN) → CPAL palette color */
  if (s.starts_with_ascii_ci ("var("))
  {
    const char *p = s.data + 4;
    const char *e = s.data + s.len;
    while (p < e && *p == ' ') p++;
    if (p + 7 < e && p[0] == '-' && p[1] == '-' && p[2] == 'c' &&
	p[3] == 'o' && p[4] == 'l' && p[5] == 'o' && p[6] == 'r')
    {
      p += 7;
      unsigned color_index = 0;
      while (p < e && *p >= '0' && *p <= '9')
	color_index = color_index * 10 + (*p++ - '0');

      hb_color_t palette_color;
      if (hb_paint_custom_palette_color (pfuncs, paint_data, color_index, &palette_color))
	return palette_color;

      unsigned count = 1;
      hb_ot_color_palette_get_colors (face, palette, color_index, &count, &palette_color);
      if (count)
	return palette_color;
    }

    /* Fallback value after comma: var(--colorN, fallback) */
    p = s.data + 4;
    while (p < e && *p != ',') p++;
    if (p < e)
    {
      p++;
      while (p < e && *p == ' ') p++;
      const char *val_start = p;
      /* Find closing paren */
      while (e > val_start && *(e - 1) != ')') e--;
      if (e > val_start) e--;
      hb_svg_str_t fallback = {val_start, (unsigned) (e - val_start)};
      return hb_raster_svg_parse_color (fallback, pfuncs, paint_data, foreground, face, palette, is_none);
    }

    return foreground;
  }

  /* #RGB or #RRGGBB */
  if (s.data[0] == '#')
  {
    if (s.len == 4) /* #RGB */
    {
      int r = hexval (s.data[1]);
      int g = hexval (s.data[2]);
      int b = hexval (s.data[3]);
      if (r < 0 || g < 0 || b < 0) return HB_COLOR (0, 0, 0, 255);
      return HB_COLOR (b * 17, g * 17, r * 17, 255);
    }
    if (s.len == 7) /* #RRGGBB */
    {
      int r = hexval (s.data[1]) * 16 + hexval (s.data[2]);
      int g = hexval (s.data[3]) * 16 + hexval (s.data[4]);
      int b = hexval (s.data[5]) * 16 + hexval (s.data[6]);
      return HB_COLOR (b, g, r, 255);
    }
    return HB_COLOR (0, 0, 0, 255);
  }

  /* rgb(r, g, b) or rgb(r%, g%, b%) */
  if (s.starts_with ("rgb"))
  {
    const char *p = s.data + 3;
    const char *e = s.data + s.len;
    while (p < e && *p != '(') p++;
    if (p < e) p++;

    auto read_component = [&] () -> int {
      while (p < e && (*p == ' ' || *p == ',')) p++;
      char buf[32];
      unsigned n = 0;
      while (p < e && n < sizeof (buf) - 1 && ((*p >= '0' && *p <= '9') || *p == '.' || *p == '-'))
	buf[n++] = *p++;
      bool is_pct = (p < e && *p == '%');
      if (is_pct) p++;
      buf[n] = '\0';
      float val = strtof (buf, nullptr);
      if (is_pct) val = val * 255.f / 100.f;
      return hb_clamp ((int) (val + 0.5f), 0, 255);
    };

    int r = read_component ();
    int g = read_component ();
    int b = read_component ();
    return HB_COLOR (b, g, r, 255);
  }

  /* Named colors (case-insensitive comparison) */
  {
    char lower[32];
    unsigned n = hb_min (s.len, (unsigned) sizeof (lower) - 1);
    for (unsigned i = 0; i < n; i++)
      lower[i] = (s.data[i] >= 'A' && s.data[i] <= 'Z') ? s.data[i] + 32 : s.data[i];
    lower[n] = '\0';

    for (unsigned i = 0; i < ARRAY_LENGTH (svg_named_colors); i++)
      if (strcmp (lower, svg_named_colors[i].name) == 0)
      {
	uint32_t rgb = svg_named_colors[i].rgb;
	return HB_COLOR (rgb & 0xFF, (rgb >> 8) & 0xFF, (rgb >> 16) & 0xFF, 255);
      }
  }

  return HB_COLOR (0, 0, 0, 255);
}

#endif /* !HB_NO_RASTER_SVG */
