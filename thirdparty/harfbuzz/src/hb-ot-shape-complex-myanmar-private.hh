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

#ifndef HB_OT_SHAPE_COMPLEX_MYANMAR_PRIVATE_HH
#define HB_OT_SHAPE_COMPLEX_MYANMAR_PRIVATE_HH

#include "hb-private.hh"

#include "hb-ot-shape-complex-indic-private.hh"


/* buffer var allocations */
#define myanmar_category() indic_category() /* myanmar_category_t */
#define myanmar_position() indic_position() /* myanmar_position_t */


/* Note: This enum is duplicated in the -machine.rl source file.
 * Not sure how to avoid duplication. */
enum myanmar_category_t {
  OT_As  = 18,  /* Asat */
  OT_D0  = 20, /* Digit zero */
  OT_DB  = OT_N, /* Dot below */
  OT_GB  = OT_PLACEHOLDER,
  OT_MH  = 21, /* Various consonant medial types */
  OT_MR  = 22, /* Various consonant medial types */
  OT_MW  = 23, /* Various consonant medial types */
  OT_MY  = 24, /* Various consonant medial types */
  OT_PT  = 25, /* Pwo and other tones */
  OT_VAbv = 26,
  OT_VBlw = 27,
  OT_VPre = 28,
  OT_VPst = 29,
  OT_VS   = 30, /* Variation selectors */
  OT_P    = 31, /* Punctuation */
  OT_D    = 32, /* Digits except zero */
};


static inline void
set_myanmar_properties (hb_glyph_info_t &info)
{
  hb_codepoint_t u = info.codepoint;
  unsigned int type = hb_indic_get_categories (u);
  indic_category_t cat = (indic_category_t) (type & 0x7Fu);
  indic_position_t pos = (indic_position_t) (type >> 8);

  /* Myanmar
   * https://docs.microsoft.com/en-us/typography/script-development/myanmar#analyze
   */
  if (unlikely (hb_in_range<hb_codepoint_t> (u, 0xFE00u, 0xFE0Fu)))
    cat = (indic_category_t) OT_VS;

  switch (u)
  {
    case 0x104Eu:
      cat = (indic_category_t) OT_C; /* The spec says C, IndicSyllableCategory doesn't have. */
      break;

    case 0x002Du: case 0x00A0u: case 0x00D7u: case 0x2012u:
    case 0x2013u: case 0x2014u: case 0x2015u: case 0x2022u:
    case 0x25CCu: case 0x25FBu: case 0x25FCu: case 0x25FDu:
    case 0x25FEu:
      cat = (indic_category_t) OT_GB;
      break;

    case 0x1004u: case 0x101Bu: case 0x105Au:
      cat = (indic_category_t) OT_Ra;
      break;

    case 0x1032u: case 0x1036u:
      cat = (indic_category_t) OT_A;
      break;

    case 0x1039u:
      cat = (indic_category_t) OT_H;
      break;

    case 0x103Au:
      cat = (indic_category_t) OT_As;
      break;

    case 0x1041u: case 0x1042u: case 0x1043u: case 0x1044u:
    case 0x1045u: case 0x1046u: case 0x1047u: case 0x1048u:
    case 0x1049u: case 0x1090u: case 0x1091u: case 0x1092u:
    case 0x1093u: case 0x1094u: case 0x1095u: case 0x1096u:
    case 0x1097u: case 0x1098u: case 0x1099u:
      cat = (indic_category_t) OT_D;
      break;

    case 0x1040u:
      cat = (indic_category_t) OT_D; /* XXX The spec says D0, but Uniscribe doesn't seem to do. */
      break;

    case 0x103Eu: case 0x1060u:
      cat = (indic_category_t) OT_MH;
      break;

    case 0x103Cu:
      cat = (indic_category_t) OT_MR;
      break;

    case 0x103Du: case 0x1082u:
      cat = (indic_category_t) OT_MW;
      break;

    case 0x103Bu: case 0x105Eu: case 0x105Fu:
      cat = (indic_category_t) OT_MY;
      break;

    case 0x1063u: case 0x1064u: case 0x1069u: case 0x106Au:
    case 0x106Bu: case 0x106Cu: case 0x106Du: case 0xAA7Bu:
      cat = (indic_category_t) OT_PT;
      break;

    case 0x1038u: case 0x1087u: case 0x1088u: case 0x1089u:
    case 0x108Au: case 0x108Bu: case 0x108Cu: case 0x108Du:
    case 0x108Fu: case 0x109Au: case 0x109Bu: case 0x109Cu:
      cat = (indic_category_t) OT_SM;
      break;

    case 0x104Au: case 0x104Bu:
      cat = (indic_category_t) OT_P;
      break;

    case 0xAA74u: case 0xAA75u: case 0xAA76u:
      /* https://github.com/roozbehp/unicode-data/issues/3 */
      cat = (indic_category_t) OT_C;
      break;
  }

  if (cat == OT_M)
  {
    switch ((int) pos)
    {
      case POS_PRE_C:	cat = (indic_category_t) OT_VPre;
			pos = POS_PRE_M;                  break;
      case POS_ABOVE_C:	cat = (indic_category_t) OT_VAbv; break;
      case POS_BELOW_C:	cat = (indic_category_t) OT_VBlw; break;
      case POS_POST_C:	cat = (indic_category_t) OT_VPst; break;
    }
  }

  info.myanmar_category() = (myanmar_category_t) cat;
  info.myanmar_position() = pos;
}


#endif /* HB_OT_SHAPE_COMPLEX_MYANMAR_PRIVATE_HH */
