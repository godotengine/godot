/*
 * Copyright © 2018-2019  Ebrahim Byagowi
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

#include "hb-ot-var-mvar-table.hh"
#include "hb-ot-gasp-table.hh" // Just so we compile it; unused otherwise.
#include "hb-ot-os2-table.hh"
#include "hb-ot-post-table.hh"
#include "hb-ot-hhea-table.hh"
#include "hb-ot-metrics.hh"
#include "hb-ot-face.hh"


/**
 * SECTION:hb-ot-metrics
 * @title: hb-ot-metrics
 * @short_description: OpenType Metrics
 * @include: hb-ot.h
 *
 * Functions for fetching metrics from fonts.
 **/

static float
_fix_ascender_descender (float value, hb_ot_metrics_tag_t metrics_tag)
{
  if (metrics_tag == HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER ||
      metrics_tag == HB_OT_METRICS_TAG_VERTICAL_ASCENDER)
    return fabs ((double) value);
  if (metrics_tag == HB_OT_METRICS_TAG_HORIZONTAL_DESCENDER ||
      metrics_tag == HB_OT_METRICS_TAG_VERTICAL_DESCENDER)
    return -fabs ((double) value);
  return value;
}

/* The common part of _get_position logic needed on hb-ot-font and here
   to be able to have slim builds without the not always needed parts */
bool
_hb_ot_metrics_get_position_common (hb_font_t           *font,
				    hb_ot_metrics_tag_t  metrics_tag,
				    hb_position_t       *position     /* OUT.  May be NULL. */)
{
  hb_face_t *face = font->face;
  switch ((unsigned) metrics_tag)
  {
#ifndef HB_NO_VAR
#define GET_VAR face->table.MVAR->get_var (metrics_tag, font->coords, font->num_coords)
#else
#define GET_VAR .0f
#endif
#define GET_METRIC_X(TABLE, ATTR) \
  (face->table.TABLE->has_data () && \
    ((void) (position && (*position = font->em_scalef_x (_fix_ascender_descender ( \
      face->table.TABLE->ATTR + GET_VAR, metrics_tag)))), true))
#define GET_METRIC_Y(TABLE, ATTR) \
  (face->table.TABLE->has_data () && \
    ((void) (position && (*position = font->em_scalef_y (_fix_ascender_descender ( \
      face->table.TABLE->ATTR + GET_VAR, metrics_tag)))), true))

  case HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER:
    return (face->table.OS2->use_typo_metrics () && GET_METRIC_Y (OS2, sTypoAscender)) ||
	   GET_METRIC_Y (hhea, ascender);
  case HB_OT_METRICS_TAG_HORIZONTAL_DESCENDER:
    return (face->table.OS2->use_typo_metrics () && GET_METRIC_Y (OS2, sTypoDescender)) ||
	   GET_METRIC_Y (hhea, descender);
  case HB_OT_METRICS_TAG_HORIZONTAL_LINE_GAP:
    return (face->table.OS2->use_typo_metrics () && GET_METRIC_Y (OS2, sTypoLineGap)) ||
	   GET_METRIC_Y (hhea, lineGap);

#ifndef HB_NO_VERTICAL
  case HB_OT_METRICS_TAG_VERTICAL_ASCENDER:  return GET_METRIC_X (vhea, ascender);
  case HB_OT_METRICS_TAG_VERTICAL_DESCENDER: return GET_METRIC_X (vhea, descender);
  case HB_OT_METRICS_TAG_VERTICAL_LINE_GAP:  return GET_METRIC_X (vhea, lineGap);
#endif

#undef GET_METRIC_Y
#undef GET_METRIC_X
#undef GET_VAR
  default:                               assert (0); return false;
  }
}

#ifndef HB_NO_METRICS

#if 0
static bool
_get_gasp (hb_face_t *face, float *result, hb_ot_metrics_tag_t metrics_tag)
{
  const OT::GaspRange& range = face->table.gasp->get_gasp_range (metrics_tag - HB_TAG ('g','s','p','0'));
  if (&range == &Null (OT::GaspRange)) return false;
  if (result) *result = range.rangeMaxPPEM + font->face->table.MVAR->get_var (metrics_tag, font->coords, font->num_coords);
  return true;
}
#endif

/* Private tags for https://github.com/harfbuzz/harfbuzz/issues/1866 */
#define _HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER_OS2   HB_TAG ('O','a','s','c')
#define _HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER_HHEA  HB_TAG ('H','a','s','c')
#define _HB_OT_METRICS_TAG_HORIZONTAL_DESCENDER_OS2  HB_TAG ('O','d','s','c')
#define _HB_OT_METRICS_TAG_HORIZONTAL_DESCENDER_HHEA HB_TAG ('H','d','s','c')
#define _HB_OT_METRICS_TAG_HORIZONTAL_LINE_GAP_OS2   HB_TAG ('O','l','g','p')
#define _HB_OT_METRICS_TAG_HORIZONTAL_LINE_GAP_HHEA  HB_TAG ('H','l','g','p')

/**
 * hb_ot_metrics_get_position:
 * @font: an #hb_font_t object.
 * @metrics_tag: tag of metrics value you like to fetch.
 * @position: (out) (optional): result of metrics value from the font.
 *
 * Fetches metrics value corresponding to @metrics_tag from @font.
 *
 * Returns: Whether found the requested metrics in the font.
 * Since: 2.6.0
 **/
hb_bool_t
hb_ot_metrics_get_position (hb_font_t           *font,
			    hb_ot_metrics_tag_t  metrics_tag,
			    hb_position_t       *position     /* OUT.  May be NULL. */)
{
  hb_face_t *face = font->face;
  switch ((unsigned) metrics_tag)
  {
  case HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER:
  case HB_OT_METRICS_TAG_HORIZONTAL_DESCENDER:
  case HB_OT_METRICS_TAG_HORIZONTAL_LINE_GAP:
  case HB_OT_METRICS_TAG_VERTICAL_ASCENDER:
  case HB_OT_METRICS_TAG_VERTICAL_DESCENDER:
  case HB_OT_METRICS_TAG_VERTICAL_LINE_GAP:           return _hb_ot_metrics_get_position_common (font, metrics_tag, position);
#ifndef HB_NO_VAR
#define GET_VAR hb_ot_metrics_get_variation (font, metrics_tag)
#else
#define GET_VAR 0
#endif
#define GET_METRIC_X(TABLE, ATTR) \
  (face->table.TABLE->has_data () && \
    ((void) (position && (*position = font->em_scalef_x (face->table.TABLE->ATTR + GET_VAR))), true))
#define GET_METRIC_Y(TABLE, ATTR) \
  (face->table.TABLE->has_data () && \
    ((void) (position && (*position = font->em_scalef_y (face->table.TABLE->ATTR + GET_VAR))), true))
  case HB_OT_METRICS_TAG_HORIZONTAL_CLIPPING_ASCENT:  return GET_METRIC_Y (OS2, usWinAscent);
  case HB_OT_METRICS_TAG_HORIZONTAL_CLIPPING_DESCENT: return GET_METRIC_Y (OS2, usWinDescent);

  case HB_OT_METRICS_TAG_HORIZONTAL_CARET_RISE:
  case HB_OT_METRICS_TAG_HORIZONTAL_CARET_RUN:
  {
    unsigned mult = 1u;

    if (font->slant)
    {
      unsigned rise = face->table.hhea->caretSlopeRise;
      unsigned upem = face->get_upem ();
      mult = (rise && rise < upem) ? hb_min (upem / rise, 256u) : 1u;
    }

    if (metrics_tag == HB_OT_METRICS_TAG_HORIZONTAL_CARET_RISE)
    {
      bool ret = GET_METRIC_Y (hhea, caretSlopeRise);

      if (position)
	*position *= mult;

      return ret;
    }
    else
    {
      hb_position_t rise = 0;

      if (font->slant && position && GET_METRIC_Y (hhea, caretSlopeRise))
	rise = *position;

      bool ret = GET_METRIC_X (hhea, caretSlopeRun);

      if (position)
      {
	*position *= mult;

	if (font->slant)
	  *position += roundf (mult * font->slant_xy * rise);
      }

      return ret;
    }
  }
  case HB_OT_METRICS_TAG_HORIZONTAL_CARET_OFFSET:     return GET_METRIC_X (hhea, caretOffset);

#ifndef HB_NO_VERTICAL
  case HB_OT_METRICS_TAG_VERTICAL_CARET_RISE:         return GET_METRIC_X (vhea, caretSlopeRise);
  case HB_OT_METRICS_TAG_VERTICAL_CARET_RUN:          return GET_METRIC_Y (vhea, caretSlopeRun);
  case HB_OT_METRICS_TAG_VERTICAL_CARET_OFFSET:       return GET_METRIC_Y (vhea, caretOffset);
#endif
  case HB_OT_METRICS_TAG_X_HEIGHT:                    return GET_METRIC_Y (OS2->v2 (), sxHeight);
  case HB_OT_METRICS_TAG_CAP_HEIGHT:                  return GET_METRIC_Y (OS2->v2 (), sCapHeight);
  case HB_OT_METRICS_TAG_SUBSCRIPT_EM_X_SIZE:         return GET_METRIC_X (OS2, ySubscriptXSize);
  case HB_OT_METRICS_TAG_SUBSCRIPT_EM_Y_SIZE:         return GET_METRIC_Y (OS2, ySubscriptYSize);
  case HB_OT_METRICS_TAG_SUBSCRIPT_EM_X_OFFSET:       return GET_METRIC_X (OS2, ySubscriptXOffset);
  case HB_OT_METRICS_TAG_SUBSCRIPT_EM_Y_OFFSET:       return GET_METRIC_Y (OS2, ySubscriptYOffset);
  case HB_OT_METRICS_TAG_SUPERSCRIPT_EM_X_SIZE:       return GET_METRIC_X (OS2, ySuperscriptXSize);
  case HB_OT_METRICS_TAG_SUPERSCRIPT_EM_Y_SIZE:       return GET_METRIC_Y (OS2, ySuperscriptYSize);
  case HB_OT_METRICS_TAG_SUPERSCRIPT_EM_X_OFFSET:     return GET_METRIC_X (OS2, ySuperscriptXOffset);
  case HB_OT_METRICS_TAG_SUPERSCRIPT_EM_Y_OFFSET:     return GET_METRIC_Y (OS2, ySuperscriptYOffset);
  case HB_OT_METRICS_TAG_STRIKEOUT_SIZE:              return GET_METRIC_Y (OS2, yStrikeoutSize);
  case HB_OT_METRICS_TAG_STRIKEOUT_OFFSET:            return GET_METRIC_Y (OS2, yStrikeoutPosition);
  case HB_OT_METRICS_TAG_UNDERLINE_SIZE:              return GET_METRIC_Y (post->table, underlineThickness);
  case HB_OT_METRICS_TAG_UNDERLINE_OFFSET:            return GET_METRIC_Y (post->table, underlinePosition);

  /* Private tags */
  case _HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER_OS2:    return GET_METRIC_Y (OS2, sTypoAscender);
  case _HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER_HHEA:   return GET_METRIC_Y (hhea, ascender);
  case _HB_OT_METRICS_TAG_HORIZONTAL_DESCENDER_OS2:   return GET_METRIC_Y (OS2, sTypoDescender);
  case _HB_OT_METRICS_TAG_HORIZONTAL_DESCENDER_HHEA:  return GET_METRIC_Y (hhea, descender);
  case _HB_OT_METRICS_TAG_HORIZONTAL_LINE_GAP_OS2:    return GET_METRIC_Y (OS2, sTypoLineGap);
  case _HB_OT_METRICS_TAG_HORIZONTAL_LINE_GAP_HHEA:   return GET_METRIC_Y (hhea, lineGap);
#undef GET_METRIC_Y
#undef GET_METRIC_X
#undef GET_VAR
  default:                                        return false;
  }
}

/**
 * hb_ot_metrics_get_position_with_fallback:
 * @font: an #hb_font_t object.
 * @metrics_tag: tag of metrics value you like to fetch.
 * @position: (out) (optional): result of metrics value from the font.
 *
 * Fetches metrics value corresponding to @metrics_tag from @font,
 * and synthesizes a value if it the value is missing in the font.
 *
 * Since: 4.0.0
 **/
void
hb_ot_metrics_get_position_with_fallback (hb_font_t           *font,
					  hb_ot_metrics_tag_t  metrics_tag,
					  hb_position_t       *position     /* OUT */)
{
  hb_font_extents_t font_extents;
  hb_codepoint_t glyph;
  hb_glyph_extents_t extents;

  if (hb_ot_metrics_get_position (font, metrics_tag, position))
    {
      if ((metrics_tag != HB_OT_METRICS_TAG_STRIKEOUT_SIZE &&
           metrics_tag != HB_OT_METRICS_TAG_UNDERLINE_SIZE) ||
          *position != 0)
        return;
    }

  switch (metrics_tag)
  {
  case HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER:
  case HB_OT_METRICS_TAG_HORIZONTAL_CLIPPING_ASCENT:
    hb_font_get_extents_for_direction (font, HB_DIRECTION_LTR, &font_extents);
    *position = font_extents.ascender;
    break;

  case HB_OT_METRICS_TAG_VERTICAL_ASCENDER:
    hb_font_get_extents_for_direction (font, HB_DIRECTION_TTB, &font_extents);
    *position = font_extents.ascender;
    break;

  case HB_OT_METRICS_TAG_HORIZONTAL_DESCENDER:
  case HB_OT_METRICS_TAG_HORIZONTAL_CLIPPING_DESCENT:
    hb_font_get_extents_for_direction (font, HB_DIRECTION_LTR, &font_extents);
    *position = font_extents.descender;
    break;

  case HB_OT_METRICS_TAG_VERTICAL_DESCENDER:
    hb_font_get_extents_for_direction (font, HB_DIRECTION_TTB, &font_extents);
    *position = font_extents.ascender;
    break;

  case HB_OT_METRICS_TAG_HORIZONTAL_LINE_GAP:
    hb_font_get_extents_for_direction (font, HB_DIRECTION_LTR, &font_extents);
    *position = font_extents.line_gap;
    break;

  case HB_OT_METRICS_TAG_VERTICAL_LINE_GAP:
    hb_font_get_extents_for_direction (font, HB_DIRECTION_TTB, &font_extents);
    *position = font_extents.line_gap;
    break;

  case HB_OT_METRICS_TAG_HORIZONTAL_CARET_RISE:
  case HB_OT_METRICS_TAG_VERTICAL_CARET_RISE:
    *position = 1;
    break;

  case HB_OT_METRICS_TAG_HORIZONTAL_CARET_RUN:
  case HB_OT_METRICS_TAG_VERTICAL_CARET_RUN:
    *position = 0;
    break;

  case HB_OT_METRICS_TAG_HORIZONTAL_CARET_OFFSET:
  case HB_OT_METRICS_TAG_VERTICAL_CARET_OFFSET:
    *position = 0;
    break;

  case HB_OT_METRICS_TAG_X_HEIGHT:
    if (hb_font_get_nominal_glyph (font, 'x', &glyph) &&
        hb_font_get_glyph_extents (font, glyph, &extents))
      *position = extents.y_bearing;
    else
      *position = font->y_scale / 2;
    break;

  case HB_OT_METRICS_TAG_CAP_HEIGHT:
    if (hb_font_get_nominal_glyph (font, 'O', &glyph) &&
        hb_font_get_glyph_extents (font, glyph, &extents))
      *position = extents.height + 2 * extents.y_bearing;
    else
      *position = font->y_scale * 2 / 3;
    break;

  case HB_OT_METRICS_TAG_STRIKEOUT_SIZE:
  case HB_OT_METRICS_TAG_UNDERLINE_SIZE:
    *position = font->y_scale / 18;
    break;

  case HB_OT_METRICS_TAG_STRIKEOUT_OFFSET:
    {
      hb_position_t ascender;
      hb_ot_metrics_get_position_with_fallback (font,
                                                HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER,
                                                &ascender);
      *position = ascender / 2;
    }
    break;

  case HB_OT_METRICS_TAG_UNDERLINE_OFFSET:
    *position = - font->y_scale / 18;
    break;

  case HB_OT_METRICS_TAG_SUBSCRIPT_EM_X_SIZE:
  case HB_OT_METRICS_TAG_SUPERSCRIPT_EM_X_SIZE:
    *position = font->x_scale * 10 / 12;
    break;

  case HB_OT_METRICS_TAG_SUBSCRIPT_EM_Y_SIZE:
  case HB_OT_METRICS_TAG_SUPERSCRIPT_EM_Y_SIZE:
    *position = font->y_scale * 10 / 12;
    break;

  case HB_OT_METRICS_TAG_SUBSCRIPT_EM_X_OFFSET:
  case HB_OT_METRICS_TAG_SUPERSCRIPT_EM_X_OFFSET:
    *position = 0;
    break;

  case HB_OT_METRICS_TAG_SUBSCRIPT_EM_Y_OFFSET:
  case HB_OT_METRICS_TAG_SUPERSCRIPT_EM_Y_OFFSET:
    *position = font->y_scale / 5;
    break;

  case _HB_OT_METRICS_TAG_MAX_VALUE:
  default:
    *position = 0;
    break;
  }
}

#ifndef HB_NO_VAR
/**
 * hb_ot_metrics_get_variation:
 * @font: an #hb_font_t object.
 * @metrics_tag: tag of metrics value you like to fetch.
 *
 * Fetches metrics value corresponding to @metrics_tag from @font with the
 * current font variation settings applied.
 *
 * Returns: The requested metric value.
 *
 * Since: 2.6.0
 **/
float
hb_ot_metrics_get_variation (hb_font_t *font, hb_ot_metrics_tag_t metrics_tag)
{
  return font->face->table.MVAR->get_var (metrics_tag, font->coords, font->num_coords);
}

/**
 * hb_ot_metrics_get_x_variation:
 * @font: an #hb_font_t object.
 * @metrics_tag: tag of metrics value you like to fetch.
 *
 * Fetches horizontal metrics value corresponding to @metrics_tag from @font
 * with the current font variation settings applied.
 *
 * Returns: The requested metric value.
 *
 * Since: 2.6.0
 **/
hb_position_t
hb_ot_metrics_get_x_variation (hb_font_t *font, hb_ot_metrics_tag_t metrics_tag)
{
  return font->em_scalef_x (hb_ot_metrics_get_variation (font, metrics_tag));
}

/**
 * hb_ot_metrics_get_y_variation:
 * @font: an #hb_font_t object.
 * @metrics_tag: tag of metrics value you like to fetch.
 *
 * Fetches vertical metrics value corresponding to @metrics_tag from @font with
 * the current font variation settings applied.
 *
 * Returns: The requested metric value.
 *
 * Since: 2.6.0
 **/
hb_position_t
hb_ot_metrics_get_y_variation (hb_font_t *font, hb_ot_metrics_tag_t metrics_tag)
{
  return font->em_scalef_y (hb_ot_metrics_get_variation (font, metrics_tag));
}
#endif

#endif
