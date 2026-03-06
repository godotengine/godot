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

#ifndef HB_VECTOR_SVG_UTILS_HH
#define HB_VECTOR_SVG_UTILS_HH

#include "hb.hh"
#include "hb-vector.hh"
#include <math.h>
#include <stdio.h>
#include <string.h>

HB_INTERNAL const char *
hb_svg_decimal_point_get (void);

static inline bool
hb_svg_append_len (hb_vector_t<char> *buf,
                   const char *s,
                   unsigned len)
{
  unsigned old_len = buf->length;
  if (unlikely (!buf->resize_dirty ((int) (old_len + len))))
    return false;
  hb_memcpy (buf->arrayZ + old_len, s, len);
  return true;
}

static inline bool
hb_svg_append_c (hb_vector_t<char> *buf, char c)
{
  return !!buf->push (c);
}

static inline void
hb_svg_append_num (hb_vector_t<char> *buf,
                   float v,
                   unsigned precision,
                   bool keep_nonzero = false)
{
  unsigned effective_precision = precision;
  if (effective_precision > 12)
    effective_precision = 12;
  if (keep_nonzero && v != 0.f)
    while (effective_precision < 12)
    {
      float rounded_zero_threshold = 0.5f;
      for (unsigned i = 0; i < effective_precision; i++)
        rounded_zero_threshold *= 0.1f;
      if (fabsf (v) >= rounded_zero_threshold)
        break;
      effective_precision++;
    }

  float rounded_zero_threshold = 0.5f;
  for (unsigned i = 0; i < effective_precision; i++)
    rounded_zero_threshold *= 0.1f;
  if (fabsf (v) < rounded_zero_threshold)
    v = 0.f;

  if (!(v == v) || !isfinite (v))
  {
    hb_svg_append_c (buf, '0');
    return;
  }

  static const char float_formats[13][6] = {
    "%.0f",  "%.1f",  "%.2f",  "%.3f",  "%.4f",  "%.5f",  "%.6f",
    "%.7f",  "%.8f",  "%.9f",  "%.10f", "%.11f", "%.12f",
  };
  char out[128];
  snprintf (out, sizeof (out), float_formats[effective_precision], (double) v);

  const char *decimal_point = hb_svg_decimal_point_get ();

  if (decimal_point[0] != '.' || decimal_point[1] != '\0')
  {
    char *p = strstr (out, decimal_point);
    if (p)
    {
      unsigned dp_len = (unsigned) strlen (decimal_point);
      unsigned tail_len = (unsigned) strlen (p + dp_len);
      memmove (p + 1, p + dp_len, tail_len + 1);
      *p = '.';
    }
  }

  char *dot = strchr (out, '.');
  if (dot)
  {
    char *end = out + strlen (out) - 1;
    while (end > dot && *end == '0')
      *end-- = '\0';
    if (end == dot)
      *end = '\0';
  }

  hb_svg_append_len (buf, out, (unsigned) strlen (out));
}

static inline unsigned
hb_svg_scale_precision (unsigned precision)
{
  return precision < 7 ? 7 : precision;
}

#endif /* HB_VECTOR_SVG_UTILS_HH */
