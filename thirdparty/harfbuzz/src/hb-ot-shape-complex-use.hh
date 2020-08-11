/*
 * Copyright © 2015  Mozilla Foundation.
 * Copyright © 2015  Google, Inc.
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
 * Mozilla Author(s): Jonathan Kew
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_SHAPE_COMPLEX_USE_HH
#define HB_OT_SHAPE_COMPLEX_USE_HH

#include "hb.hh"


#include "hb-ot-shape-complex.hh"


#define USE_TABLE_ELEMENT_TYPE uint8_t

/* Cateories used in the Universal Shaping Engine spec:
 * https://docs.microsoft.com/en-us/typography/script-development/use
 */
/* Note: This enum is duplicated in the -machine.rl source file.
 * Not sure how to avoid duplication. */
enum use_category_t {
  USE_O		= 0,	/* OTHER */

  USE_B		= 1,	/* BASE */
  USE_IND	= 3,	/* BASE_IND */
  USE_N		= 4,	/* BASE_NUM */
  USE_GB	= 5,	/* BASE_OTHER */
  USE_CGJ	= 6,	/* CGJ */
//  USE_F		= 7,	/* CONS_FINAL */
  USE_FM	= 8,	/* CONS_FINAL_MOD */
//  USE_M		= 9,	/* CONS_MED */
//  USE_CM	= 10,	/* CONS_MOD */
  USE_SUB	= 11,	/* CONS_SUB */
  USE_H		= 12,	/* HALANT */

  USE_HN	= 13,	/* HALANT_NUM */
  USE_ZWNJ	= 14,	/* Zero width non-joiner */
  USE_ZWJ	= 15,	/* Zero width joiner */
  USE_WJ	= 16,	/* Word joiner */
  USE_Rsv	= 17,	/* Reserved characters */
  USE_R		= 18,	/* REPHA */
  USE_S		= 19,	/* SYM */
//  USE_SM	= 20,	/* SYM_MOD */
  USE_VS	= 21,	/* VARIATION_SELECTOR */
//  USE_V	= 36,	/* VOWEL */
//  USE_VM	= 40,	/* VOWEL_MOD */
  USE_CS	= 43,	/* CONS_WITH_STACKER */

  /* https://github.com/harfbuzz/harfbuzz/issues/1102 */
  USE_HVM	= 44,	/* HALANT_OR_VOWEL_MODIFIER */

  USE_Sk	= 48,	/* SAKOT */

  USE_FAbv	= 24,	/* CONS_FINAL_ABOVE */
  USE_FBlw	= 25,	/* CONS_FINAL_BELOW */
  USE_FPst	= 26,	/* CONS_FINAL_POST */
  USE_MAbv	= 27,	/* CONS_MED_ABOVE */
  USE_MBlw	= 28,	/* CONS_MED_BELOW */
  USE_MPst	= 29,	/* CONS_MED_POST */
  USE_MPre	= 30,	/* CONS_MED_PRE */
  USE_CMAbv	= 31,	/* CONS_MOD_ABOVE */
  USE_CMBlw	= 32,	/* CONS_MOD_BELOW */
  USE_VAbv	= 33,	/* VOWEL_ABOVE / VOWEL_ABOVE_BELOW / VOWEL_ABOVE_BELOW_POST / VOWEL_ABOVE_POST */
  USE_VBlw	= 34,	/* VOWEL_BELOW / VOWEL_BELOW_POST */
  USE_VPst	= 35,	/* VOWEL_POST	UIPC = Right */
  USE_VPre	= 22,	/* VOWEL_PRE / VOWEL_PRE_ABOVE / VOWEL_PRE_ABOVE_POST / VOWEL_PRE_POST */
  USE_VMAbv	= 37,	/* VOWEL_MOD_ABOVE */
  USE_VMBlw	= 38,	/* VOWEL_MOD_BELOW */
  USE_VMPst	= 39,	/* VOWEL_MOD_POST */
  USE_VMPre	= 23,	/* VOWEL_MOD_PRE */
  USE_SMAbv	= 41,	/* SYM_MOD_ABOVE */
  USE_SMBlw	= 42,	/* SYM_MOD_BELOW */
  USE_FMAbv	= 45,	/* CONS_FINAL_MOD	UIPC = Top */
  USE_FMBlw	= 46,	/* CONS_FINAL_MOD	UIPC = Bottom */
  USE_FMPst	= 47,	/* CONS_FINAL_MOD	UIPC = Not_Applicable */
};

HB_INTERNAL USE_TABLE_ELEMENT_TYPE
hb_use_get_category (hb_codepoint_t u);

#endif /* HB_OT_SHAPE_COMPLEX_USE_HH */
