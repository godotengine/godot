/*
 * Copyright © 2018  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_SHAPE_COMPLEX_KHMER_HH
#define HB_OT_SHAPE_COMPLEX_KHMER_HH

#include "hb.hh"

#include "hb-ot-shape-complex-indic.hh"


/* buffer var allocations */
#define khmer_category() indic_category() /* khmer_category_t */


/* Note: This enum is duplicated in the -machine.rl source file.
 * Not sure how to avoid duplication. */
enum khmer_category_t
{
  OT_Robatic = 20,
  OT_Xgroup  = 21,
  OT_Ygroup  = 22,
  //OT_VAbv = 26,
  //OT_VBlw = 27,
  //OT_VPre = 28,
  //OT_VPst = 29,
};

static inline void
set_khmer_properties (hb_glyph_info_t &info)
{
  hb_codepoint_t u = info.codepoint;
  unsigned int type = hb_indic_get_categories (u);
  khmer_category_t cat = (khmer_category_t) (type & 0x7Fu);
  indic_position_t pos = (indic_position_t) (type >> 8);


  /*
   * Re-assign category
   *
   * These categories are experimentally extracted from what Uniscribe allows.
   */
  switch (u)
  {
    case 0x179Au:
      cat = (khmer_category_t) OT_Ra;
      break;

    case 0x17CCu:
    case 0x17C9u:
    case 0x17CAu:
      cat = OT_Robatic;
      break;

    case 0x17C6u:
    case 0x17CBu:
    case 0x17CDu:
    case 0x17CEu:
    case 0x17CFu:
    case 0x17D0u:
    case 0x17D1u:
      cat = OT_Xgroup;
      break;

    case 0x17C7u:
    case 0x17C8u:
    case 0x17DDu:
    case 0x17D3u: /* Just guessing. Uniscribe doesn't categorize it. */
      cat = OT_Ygroup;
      break;
  }

  /*
   * Re-assign position.
   */
  if (cat == (khmer_category_t) OT_M)
    switch ((int) pos)
    {
      case POS_PRE_C:	cat = (khmer_category_t) OT_VPre; break;
      case POS_BELOW_C:	cat = (khmer_category_t) OT_VBlw; break;
      case POS_ABOVE_C:	cat = (khmer_category_t) OT_VAbv; break;
      case POS_POST_C:	cat = (khmer_category_t) OT_VPst; break;
      default: assert (0);
    }

  info.khmer_category() = cat;
}


#endif /* HB_OT_SHAPE_COMPLEX_KHMER_HH */
