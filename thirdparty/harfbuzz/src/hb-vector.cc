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

#include "hb-machinery.hh"
#include "hb-vector-buf.hh"

/**
 * SECTION:hb-vector
 * @title: hb-vector
 * @short_description: Glyph vector conversion
 * @include: hb-vector.h
 *
 * Functions for converting glyph outlines and color paint trees
 * into SVG or PDF vector output.
 *
 * #hb_vector_draw_t converts monochrome glyph outlines into vector
 * paths. Typical flow:
 *
 * |[<!-- language="plain" -->
 * hb_vector_draw_t *draw = hb_vector_draw_create_or_fail (HB_VECTOR_FORMAT_SVG);
 * hb_vector_draw_set_scale_factor (draw, 64.f, 64.f);
 * hb_vector_draw_set_foreground (draw, foreground);
 * hb_vector_draw_glyph (draw, font, gid, pen_x, pen_y);
 * hb_blob_t *svg = hb_vector_draw_render (draw);
 * ]|
 *
 * #hb_vector_paint_t renders color paint graphs (COLRv0/v1) and
 * embedded PNG images into vector output with gradients, layers,
 * and compositing. Typical flow:
 *
 * |[<!-- language="plain" -->
 * hb_vector_paint_t *paint = hb_vector_paint_create_or_fail (HB_VECTOR_FORMAT_SVG);
 * hb_vector_paint_set_scale_factor (paint, 64.f, 64.f);
 * hb_vector_paint_set_foreground (paint, foreground);
 * hb_vector_paint_glyph (paint, font, gid, pen_x, pen_y,
 *                        HB_VECTOR_EXTENTS_MODE_EXPAND);
 * hb_blob_t *svg = hb_vector_paint_render (paint);
 * ]|
 *
 * Both contexts accumulate multiple glyphs into a single document.
 * Call hb_vector_draw_render() / hb_vector_paint_render() to
 * retrieve the final blob.  Rendering clears all accumulated
 * content (including extents), so retrieve any needed extents
 * via hb_vector_draw_get_extents() / hb_vector_paint_get_extents()
 * before rendering.
 *
 * Each glyph is emitted as an independent element.  If glyphs
 * overlap and the foreground color is semi-transparent, the
 * overlapping regions will be composited separately rather than
 * painted as a single uniform layer.
 **/

struct hb_vector_decimal_point_t
{
  char value[8];
};

static hb_vector_decimal_point_t hb_vector_decimal_point_default = {{'.', '\0'}};

static inline void free_static_svg_decimal_point ();

static struct hb_vector_decimal_point_lazy_loader_t
  : hb_lazy_loader_t<hb_vector_decimal_point_t, hb_vector_decimal_point_lazy_loader_t>
{
  static hb_vector_decimal_point_t *create ()
  {
    auto *p = (hb_vector_decimal_point_t *) hb_calloc (1, sizeof (hb_vector_decimal_point_t));
    if (!p)
      return nullptr;

    p->value[0] = '.';
    p->value[1] = '\0';

#ifndef HB_NO_SETLOCALE
    lconv *lc = nullptr;
#ifdef HAVE_LOCALECONV_L
    hb_locale_t current_locale = hb_uselocale ((hb_locale_t) 0);
    if (current_locale)
      lc = localeconv_l (current_locale);
#endif
    if (!lc)
      lc = localeconv ();
    if (lc && lc->decimal_point && lc->decimal_point[0])
    {
      strncpy (p->value, lc->decimal_point, sizeof (p->value) - 1);
      p->value[sizeof (p->value) - 1] = '\0';
    }
#endif

    hb_atexit (free_static_svg_decimal_point);
    return p;
  }

  static void destroy (hb_vector_decimal_point_t *p)
  { hb_free (p); }

  static const hb_vector_decimal_point_t *get_null ()
  { return &hb_vector_decimal_point_default; }
} static_svg_decimal_point;

static inline void
free_static_svg_decimal_point ()
{
  static_svg_decimal_point.free_instance ();
}

const char *
hb_vector_decimal_point_get (void)
{
  return static_svg_decimal_point.get_unconst ()->value;
}
