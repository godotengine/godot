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
#include "hb-vector-svg-utils.hh"

#include <string.h>

struct hb_svg_decimal_point_t
{
  char value[8];
};

static hb_svg_decimal_point_t hb_svg_decimal_point_default = {{'.', '\0'}};

static inline void free_static_svg_decimal_point ();

static struct hb_svg_decimal_point_lazy_loader_t
  : hb_lazy_loader_t<hb_svg_decimal_point_t, hb_svg_decimal_point_lazy_loader_t>
{
  static hb_svg_decimal_point_t *create ()
  {
    auto *p = (hb_svg_decimal_point_t *) hb_calloc (1, sizeof (hb_svg_decimal_point_t));
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

  static void destroy (hb_svg_decimal_point_t *p)
  { hb_free (p); }

  static const hb_svg_decimal_point_t *get_null ()
  { return &hb_svg_decimal_point_default; }
} static_svg_decimal_point;

static inline void
free_static_svg_decimal_point ()
{
  static_svg_decimal_point.free_instance ();
}

const char *
hb_svg_decimal_point_get (void)
{
  return static_svg_decimal_point.get_unconst ()->value;
}
