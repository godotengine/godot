/*
 * Copyright © 2019  Ebrahim Byagowi
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

#include "hb.hh"

#ifndef HB_NO_STYLE
#ifdef HB_EXPERIMENTAL_API

#include "hb-ot-var-avar-table.hh"
#include "hb-ot-var-fvar-table.hh"
#include "hb-ot-stat-table.hh"
#include "hb-ot-os2-table.hh"
#include "hb-ot-head-table.hh"
#include "hb-ot-post-table.hh"
#include "hb-ot-face.hh"

/**
 * hb_style_tag_t:
 * @HB_STYLE_TAG_ITALIC: Used to vary between non-italic and italic.
 * A value of 0 can be interpreted as "Roman" (non-italic); a value of 1 can
 * be interpreted as (fully) italic.
 * @HB_STYLE_TAG_OPTICAL_SIZE: Used to vary design to suit different text sizes.
 * Non-zero. Values can be interpreted as text size, in points.
 * @HB_STYLE_TAG_SLANT: Used to vary between upright and slanted text. Values
 * must be greater than -90 and less than +90. Values can be interpreted as
 * the angle, in counter-clockwise degrees, of oblique slant from whatever the
 * designer considers to be upright for that font design.
 * @HB_STYLE_TAG_WIDTH: Used to vary width of text from narrower to wider.
 * Non-zero. Values can be interpreted as a percentage of whatever the font
 * designer considers “normal width” for that font design.
 * @HB_STYLE_TAG_WEIGHT: Used to vary stroke thicknesses or other design details
 * to give variation from lighter to blacker. Values can be interpreted in direct
 * comparison to values for usWeightClass in the OS/2 table,
 * or the CSS font-weight property.
 *
 * Defined by https://docs.microsoft.com/en-us/typography/opentype/spec/dvaraxisreg
 *
 * Since: EXPERIMENTAL
 **/
typedef enum {
  HB_STYLE_TAG_ITALIC		= HB_TAG ('i','t','a','l'),
  HB_STYLE_TAG_OPTICAL_SIZE	= HB_TAG ('o','p','s','z'),
  HB_STYLE_TAG_SLANT		= HB_TAG ('s','l','n','t'),
  HB_STYLE_TAG_WIDTH		= HB_TAG ('w','d','t','h'),
  HB_STYLE_TAG_WEIGHT		= HB_TAG ('w','g','h','t'),

  /*< private >*/
  _HB_STYLE_TAG_MAX_VALUE	= HB_TAG_MAX_SIGNED /*< skip >*/
} hb_style_tag_t;

/**
 * hb_style_get_value:
 * @font: a #hb_font_t object.
 * @style_tag: a style tag.
 *
 * Searches variation axes of a hb_font_t object for a specific axis first,
 * if not set, then tries to get default style values from different
 * tables of the font.
 *
 * Returns: Corresponding axis or default value to a style tag.
 *
 * Since: EXPERIMENTAL
 **/
float
hb_style_get_value (hb_font_t *font, hb_tag_t tag)
{
  hb_style_tag_t style_tag = (hb_style_tag_t) tag;
  hb_face_t *face = font->face;

#ifndef HB_NO_VAR
  hb_ot_var_axis_info_t axis;
  if (hb_ot_var_find_axis_info (face, style_tag, &axis))
  {
    if (axis.axis_index < font->num_coords) return font->design_coords[axis.axis_index];
    /* If a face is variable, fvar's default_value is better than STAT records */
    return axis.default_value;
  }
#endif

  if (style_tag == HB_STYLE_TAG_OPTICAL_SIZE && font->ptem)
    return font->ptem;

  /* STAT */
  float value;
  if (face->table.STAT->get_value (style_tag, &value))
    return value;

  switch ((unsigned) style_tag)
  {
  case HB_STYLE_TAG_ITALIC:
    return face->table.OS2->is_italic () || face->table.head->is_italic () ? 1 : 0;
  case HB_STYLE_TAG_OPTICAL_SIZE:
  {
    unsigned int lower, upper;
    return face->table.OS2->v5 ().get_optical_size (&lower, &upper)
	   ? (float) (lower + upper) / 2.f
	   : 12.f;
  }
  case HB_STYLE_TAG_SLANT:
    return face->table.post->table->italicAngle.to_float ();
  case HB_STYLE_TAG_WIDTH:
    return face->table.OS2->has_data ()
	   ? face->table.OS2->get_width ()
	   : (face->table.head->is_condensed () ? 75 : 100);
  case HB_STYLE_TAG_WEIGHT:
    return face->table.OS2->has_data ()
	   ? face->table.OS2->usWeightClass
	   : (face->table.head->is_bold () ? 700 : 400);
  default:
    return 0;
  }
}

#endif
#endif
