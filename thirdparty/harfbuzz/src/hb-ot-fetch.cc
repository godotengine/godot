/*
 * Copyright © 2026  Khaled Hosny
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

#ifndef HB_NO_OT_FETCH

#include "hb-ot.h"

#include "hb-ot-head-table.hh"
#include "hb-ot-os2-table.hh"
#include "hb-ot-post-table.hh"

/**
 * SECTION:hb-ot-fetch
 * @title: hb-ot-fetch
 * @short_description: OpenType bit fields and numbers
 * @include: hb-ot.h
 *
 * Functions for fetching various bit fields and numbers scattered around
 * OpenType tables.
 *
 * These are raw table values, and many of them are legacy or unreliable, but
 * applications might need them for various legacy reasons.
 **/


/**
 * hb_ot_fetch_bits:
 * @face: #hb_face_t to work upon
 * @tag: tag of the bit field to fetch
 *
 * Fetches a bit field of @face.
 *
 * Return value: the bit field, or zero if the font does not have it.
 *
 * Since: 14.3.0
 **/
uint32_t
hb_ot_fetch_bits (hb_face_t        *face,
		hb_ot_bits_tag_t  tag)
{
  switch ((unsigned) tag)
  {
  case HB_OT_BITS_TAG_FS_TYPE:		 return face->table.OS2->fsType;
  case HB_OT_BITS_TAG_FS_SELECTION:	 return face->table.OS2->fsSelection;
  case HB_OT_BITS_TAG_MAC_STYLE:	 return face->table.head->get_mac_style ();
  case HB_OT_BITS_TAG_IS_FIXED_PITCH:	 return face->table.post->table->isFixedPitch;
  case HB_OT_BITS_TAG_UNICODE_RANGE_1:	 return face->table.OS2->ulUnicodeRange[0];
  case HB_OT_BITS_TAG_UNICODE_RANGE_2:	 return face->table.OS2->ulUnicodeRange[1];
  case HB_OT_BITS_TAG_UNICODE_RANGE_3:	 return face->table.OS2->ulUnicodeRange[2];
  case HB_OT_BITS_TAG_UNICODE_RANGE_4:	 return face->table.OS2->ulUnicodeRange[3];
  case HB_OT_BITS_TAG_CODE_PAGE_RANGE_1: return face->table.OS2->v1 ().ulCodePageRange1;
  case HB_OT_BITS_TAG_CODE_PAGE_RANGE_2: return face->table.OS2->v1 ().ulCodePageRange2;
  default:				 return 0;
  }
}

/**
 * hb_ot_fetch_number:
 * @face: #hb_face_t to work upon
 * @tag: tag of the number to fetch
 *
 * Fetches a number of @face in font units.
 *
 * Return value: the number, or zero if the font does not have it.
 *
 * Since: 14.3.0
 **/
int32_t
hb_ot_fetch_number (hb_face_t          *face,
		  hb_ot_number_tag_t  tag)
{
  switch ((unsigned) tag)
  {
  case HB_OT_NUMBER_TAG_FONT_X_MIN: return face->table.head->xMin;
  case HB_OT_NUMBER_TAG_FONT_Y_MIN: return face->table.head->yMin;
  case HB_OT_NUMBER_TAG_FONT_X_MAX: return face->table.head->xMax;
  case HB_OT_NUMBER_TAG_FONT_Y_MAX: return face->table.head->yMax;
  default:		       return 0;
  }
}

#endif
