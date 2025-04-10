/*
 * Copyright © 2023  Behdad Esfahbod
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
 */

#ifndef HB_OUTLINE_HH
#define HB_OUTLINE_HH

#include "hb.hh"

#include "hb-draw.hh"


struct hb_outline_point_t
{
  enum class type_t
  {
    MOVE_TO,
    LINE_TO,
    QUADRATIC_TO,
    CUBIC_TO,
  };

  hb_outline_point_t (float x, float y, type_t type) :
    x (x), y (y), type (type) {}

  float x, y;
  type_t type;
};

struct hb_outline_vector_t
{
  float normalize_len ()
  {
    float len = hypotf (x, y);
    if (len)
    {
      x /= len;
      y /= len;
    }
    return len;
  }

  float x, y;
};

struct hb_outline_t
{
  void reset () { points.shrink (0, false); contours.resize (0); }

  HB_INTERNAL void replay (hb_draw_funcs_t *pen, void *pen_data) const;
  HB_INTERNAL float control_area () const;
  HB_INTERNAL void embolden (float x_strength, float y_strength,
			     float x_shift, float y_shift);

  hb_vector_t<hb_outline_point_t> points;
  hb_vector_t<unsigned> contours;
};

HB_INTERNAL hb_draw_funcs_t *
hb_outline_recording_pen_get_funcs ();


#endif /* HB_OUTLINE_HH */
