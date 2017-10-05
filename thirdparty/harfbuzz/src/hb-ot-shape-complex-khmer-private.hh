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

#ifndef HB_OT_SHAPE_COMPLEX_KHMER_PRIVATE_HH
#define HB_OT_SHAPE_COMPLEX_KHMER_PRIVATE_HH

#include "hb-private.hh"

#include "hb-ot-shape-complex-indic-private.hh"


/* buffer var allocations */
#define khmer_category() indic_category() /* khmer_category_t */
#define khmer_position() indic_position() /* khmer_position_t */


typedef indic_category_t khmer_category_t;
typedef indic_position_t khmer_position_t;


static inline khmer_position_t
matra_position_khmer (khmer_position_t side)
{
  switch ((int) side)
  {
    case POS_PRE_C:
      return POS_PRE_M;

    case POS_POST_C:
    case POS_ABOVE_C:
    case POS_BELOW_C:
      return POS_AFTER_POST;

    default:
      return side;
  };
}

static inline bool
is_consonant_or_vowel (const hb_glyph_info_t &info)
{
  return is_one_of (info, CONSONANT_FLAGS | FLAG (OT_V));
}

static inline bool
is_coeng (const hb_glyph_info_t &info)
{
  return is_one_of (info, FLAG (OT_Coeng));
}

static inline void
set_khmer_properties (hb_glyph_info_t &info)
{
  hb_codepoint_t u = info.codepoint;
  unsigned int type = hb_indic_get_categories (u);
  khmer_category_t cat = (khmer_category_t) (type & 0x7Fu);
  khmer_position_t pos = (khmer_position_t) (type >> 8);


  /*
   * Re-assign category
   */

  if (unlikely (u == 0x17C6u)) cat = OT_N; /* Khmer Bindu doesn't like to be repositioned. */
  else if (unlikely (hb_in_range<hb_codepoint_t> (u, 0x17CDu, 0x17D1u) ||
		     u == 0x17CBu || u == 0x17D3u || u == 0x17DDu)) /* Khmer Various signs */
  {
    /* These can occur mid-syllable (eg. before matras), even though Unicode marks them as Syllable_Modifier.
     * https://github.com/roozbehp/unicode-data/issues/5 */
    cat = OT_M;
    pos = POS_ABOVE_C;
  }
  else if (unlikely (hb_in_range<hb_codepoint_t> (u, 0x2010u, 0x2011u))) cat = OT_PLACEHOLDER;
  else if (unlikely (u == 0x25CCu)) cat = OT_DOTTEDCIRCLE;


  /*
   * Re-assign position.
   */

  if ((FLAG_UNSAFE (cat) & CONSONANT_FLAGS))
  {
    pos = POS_BASE_C;
    if (u == 0x179Au)
      cat = OT_Ra;
  }
  else if (cat == OT_M)
  {
    pos = matra_position_khmer (pos);
  }
  else if ((FLAG_UNSAFE (cat) & (FLAG (OT_SM) | FLAG (OT_A) | FLAG (OT_Symbol))))
  {
    pos = POS_SMVD;
  }

  info.khmer_category() = cat;
  info.khmer_position() = pos;
}


#endif /* HB_OT_SHAPE_COMPLEX_KHMER_PRIVATE_HH */
