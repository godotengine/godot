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

#if !defined(HB_H_IN) && !defined(HB_NO_SINGLE_HEADER_ERROR)
#error "Include <hb.h> instead."
#endif

#ifndef HB_STYLE_H
#define HB_STYLE_H

#include "hb.h"

HB_BEGIN_DECLS

/**
 * hb_style_tag_t:
 * @HB_STYLE_TAG_ITALIC: Used to vary between non-italic and italic.
 * A value of 0 can be interpreted as "Roman" (non-italic); a value of 1 can
 * be interpreted as (fully) italic.
 * @HB_STYLE_TAG_OPTICAL_SIZE: Used to vary design to suit different text sizes.
 * Non-zero. Values can be interpreted as text size, in points.
 * @HB_STYLE_TAG_SLANT_ANGLE: Used to vary between upright and slanted text. Values
 * must be greater than -90 and less than +90. Values can be interpreted as
 * the angle, in counter-clockwise degrees, of oblique slant from whatever the
 * designer considers to be upright for that font design. Typical right-leaning
 * Italic fonts have a negative slant angle (typically around -12)
 * @HB_STYLE_TAG_SLANT_RATIO: same as @HB_STYLE_TAG_SLANT_ANGLE expression as ratio.
 * Typical right-leaning Italic fonts have a positive slant ratio (typically around 0.2)
 * @HB_STYLE_TAG_WIDTH: Used to vary width of text from narrower to wider.
 * Non-zero. Values can be interpreted as a percentage of whatever the font
 * designer considers “normal width” for that font design.
 * @HB_STYLE_TAG_WEIGHT: Used to vary stroke thicknesses or other design details
 * to give variation from lighter to blacker. Values can be interpreted in direct
 * comparison to values for usWeightClass in the OS/2 table,
 * or the CSS font-weight property.
 *
 * Defined by [OpenType Design-Variation Axis Tag Registry](https://docs.microsoft.com/en-us/typography/opentype/spec/dvaraxisreg).
 *
 * Since: 3.0.0
 **/
typedef enum
{
  HB_STYLE_TAG_ITALIC		= HB_TAG ('i','t','a','l'),
  HB_STYLE_TAG_OPTICAL_SIZE	= HB_TAG ('o','p','s','z'),
  HB_STYLE_TAG_SLANT_ANGLE	= HB_TAG ('s','l','n','t'),
  HB_STYLE_TAG_SLANT_RATIO	= HB_TAG ('S','l','n','t'),
  HB_STYLE_TAG_WIDTH		= HB_TAG ('w','d','t','h'),
  HB_STYLE_TAG_WEIGHT		= HB_TAG ('w','g','h','t'),

  /*< private >*/
  _HB_STYLE_TAG_MAX_VALUE	= HB_TAG_MAX_SIGNED /*< skip >*/
} hb_style_tag_t;


HB_EXTERN float
hb_style_get_value (hb_font_t *font, hb_style_tag_t style_tag);

HB_END_DECLS

#endif /* HB_STYLE_H */
