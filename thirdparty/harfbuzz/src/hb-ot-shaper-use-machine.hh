
#line 1 "hb-ot-shaper-use-machine.rl"
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

#ifndef HB_OT_SHAPER_USE_MACHINE_HH
#define HB_OT_SHAPER_USE_MACHINE_HH

#include "hb.hh"

#include "hb-ot-shaper-syllabic.hh"

/* buffer var allocations */
#define use_category() ot_shaper_var_u8_category()

#define USE(Cat) use_syllable_machine_ex_##Cat

enum use_syllable_type_t {
  use_virama_terminated_cluster,
  use_sakot_terminated_cluster,
  use_standard_cluster,
  use_number_joiner_terminated_cluster,
  use_numeral_cluster,
  use_symbol_cluster,
  use_hieroglyph_cluster,
  use_broken_cluster,
  use_non_cluster,
};


#line 54 "hb-ot-shaper-use-machine.hh"
#define use_syllable_machine_ex_B 1u
#define use_syllable_machine_ex_CGJ 6u
#define use_syllable_machine_ex_CMAbv 31u
#define use_syllable_machine_ex_CMBlw 32u
#define use_syllable_machine_ex_CS 43u
#define use_syllable_machine_ex_FAbv 24u
#define use_syllable_machine_ex_FBlw 25u
#define use_syllable_machine_ex_FMAbv 45u
#define use_syllable_machine_ex_FMBlw 46u
#define use_syllable_machine_ex_FMPst 47u
#define use_syllable_machine_ex_FPst 26u
#define use_syllable_machine_ex_G 49u
#define use_syllable_machine_ex_GB 5u
#define use_syllable_machine_ex_H 12u
#define use_syllable_machine_ex_HN 13u
#define use_syllable_machine_ex_HVM 53u
#define use_syllable_machine_ex_IS 44u
#define use_syllable_machine_ex_J 50u
#define use_syllable_machine_ex_MAbv 27u
#define use_syllable_machine_ex_MBlw 28u
#define use_syllable_machine_ex_MPre 30u
#define use_syllable_machine_ex_MPst 29u
#define use_syllable_machine_ex_N 4u
#define use_syllable_machine_ex_O 0u
#define use_syllable_machine_ex_R 18u
#define use_syllable_machine_ex_SB 51u
#define use_syllable_machine_ex_SE 52u
#define use_syllable_machine_ex_SMAbv 41u
#define use_syllable_machine_ex_SMBlw 42u
#define use_syllable_machine_ex_SUB 11u
#define use_syllable_machine_ex_Sk 48u
#define use_syllable_machine_ex_VAbv 33u
#define use_syllable_machine_ex_VBlw 34u
#define use_syllable_machine_ex_VMAbv 37u
#define use_syllable_machine_ex_VMBlw 38u
#define use_syllable_machine_ex_VMPre 23u
#define use_syllable_machine_ex_VMPst 39u
#define use_syllable_machine_ex_VPre 22u
#define use_syllable_machine_ex_VPst 35u
#define use_syllable_machine_ex_WJ 16u
#define use_syllable_machine_ex_ZWNJ 14u


#line 96 "hb-ot-shaper-use-machine.hh"
static const unsigned char _use_syllable_machine_trans_keys[] = {
	0u, 53u, 11u, 53u, 11u, 53u, 1u, 53u, 23u, 48u, 24u, 47u, 25u, 47u, 26u, 47u, 
	45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 1u, 1u, 24u, 48u, 22u, 53u, 
	23u, 53u, 23u, 53u, 23u, 53u, 12u, 53u, 23u, 53u, 12u, 53u, 12u, 53u, 12u, 53u, 
	11u, 53u, 1u, 1u, 1u, 48u, 11u, 53u, 41u, 42u, 42u, 42u, 11u, 53u, 11u, 53u, 
	1u, 53u, 23u, 48u, 24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 
	24u, 48u, 24u, 48u, 1u, 1u, 24u, 48u, 22u, 53u, 23u, 53u, 23u, 53u, 23u, 53u, 
	12u, 53u, 23u, 53u, 12u, 53u, 12u, 53u, 12u, 53u, 11u, 53u, 1u, 1u, 1u, 48u, 
	13u, 13u, 4u, 4u, 11u, 53u, 11u, 53u, 1u, 53u, 23u, 48u, 24u, 47u, 25u, 47u, 
	26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 1u, 1u, 24u, 48u, 
	22u, 53u, 23u, 53u, 23u, 53u, 23u, 53u, 12u, 53u, 23u, 53u, 12u, 53u, 12u, 53u, 
	12u, 53u, 11u, 53u, 1u, 1u, 1u, 48u, 11u, 53u, 11u, 53u, 1u, 53u, 23u, 48u, 
	24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 
	1u, 1u, 24u, 48u, 22u, 53u, 23u, 53u, 23u, 53u, 23u, 53u, 12u, 53u, 23u, 53u, 
	12u, 53u, 12u, 53u, 12u, 53u, 11u, 53u, 1u, 1u, 1u, 48u, 4u, 4u, 13u, 13u, 
	1u, 53u, 11u, 53u, 41u, 42u, 42u, 42u, 1u, 5u, 50u, 52u, 49u, 52u, 49u, 51u, 
	0
};

static const char _use_syllable_machine_key_spans[] = {
	54, 43, 43, 53, 26, 24, 23, 22, 
	2, 1, 25, 25, 25, 1, 25, 32, 
	31, 31, 31, 42, 31, 42, 42, 42, 
	43, 1, 48, 43, 2, 1, 43, 43, 
	53, 26, 24, 23, 22, 2, 1, 25, 
	25, 25, 1, 25, 32, 31, 31, 31, 
	42, 31, 42, 42, 42, 43, 1, 48, 
	1, 1, 43, 43, 53, 26, 24, 23, 
	22, 2, 1, 25, 25, 25, 1, 25, 
	32, 31, 31, 31, 42, 31, 42, 42, 
	42, 43, 1, 48, 43, 43, 53, 26, 
	24, 23, 22, 2, 1, 25, 25, 25, 
	1, 25, 32, 31, 31, 31, 42, 31, 
	42, 42, 42, 43, 1, 48, 1, 1, 
	53, 43, 2, 1, 5, 3, 4, 3
};

static const short _use_syllable_machine_index_offsets[] = {
	0, 55, 99, 143, 197, 224, 249, 273, 
	296, 299, 301, 327, 353, 379, 381, 407, 
	440, 472, 504, 536, 579, 611, 654, 697, 
	740, 784, 786, 835, 879, 882, 884, 928, 
	972, 1026, 1053, 1078, 1102, 1125, 1128, 1130, 
	1156, 1182, 1208, 1210, 1236, 1269, 1301, 1333, 
	1365, 1408, 1440, 1483, 1526, 1569, 1613, 1615, 
	1664, 1666, 1668, 1712, 1756, 1810, 1837, 1862, 
	1886, 1909, 1912, 1914, 1940, 1966, 1992, 1994, 
	2020, 2053, 2085, 2117, 2149, 2192, 2224, 2267, 
	2310, 2353, 2397, 2399, 2448, 2492, 2536, 2590, 
	2617, 2642, 2666, 2689, 2692, 2694, 2720, 2746, 
	2772, 2774, 2800, 2833, 2865, 2897, 2929, 2972, 
	3004, 3047, 3090, 3133, 3177, 3179, 3228, 3230, 
	3232, 3286, 3330, 3333, 3335, 3341, 3345, 3350
};

static const unsigned char _use_syllable_machine_indicies[] = {
	0, 1, 2, 2, 3, 4, 2, 2, 
	2, 2, 2, 5, 6, 7, 2, 2, 
	2, 2, 8, 2, 2, 2, 9, 10, 
	11, 12, 13, 14, 15, 16, 17, 18, 
	19, 20, 21, 22, 2, 23, 24, 25, 
	2, 26, 27, 28, 29, 30, 31, 32, 
	29, 33, 2, 34, 2, 35, 2, 37, 
	38, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 39, 40, 41, 42, 43, 44, 
	45, 46, 47, 48, 49, 50, 51, 52, 
	36, 53, 54, 55, 36, 56, 57, 36, 
	58, 59, 60, 61, 58, 36, 36, 36, 
	36, 62, 36, 37, 38, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 39, 40, 
	41, 42, 43, 44, 45, 46, 47, 49, 
	49, 50, 51, 52, 36, 53, 54, 55, 
	36, 36, 36, 36, 58, 59, 60, 61, 
	58, 36, 36, 36, 36, 62, 36, 37, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 40, 41, 42, 
	43, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 53, 54, 55, 36, 36, 
	36, 36, 36, 59, 60, 61, 63, 36, 
	36, 36, 36, 40, 36, 40, 41, 42, 
	43, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 53, 54, 55, 36, 36, 
	36, 36, 36, 59, 60, 61, 63, 36, 
	41, 42, 43, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 59, 60, 61, 
	36, 42, 43, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 59, 60, 61, 
	36, 43, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 59, 60, 61, 36, 
	59, 60, 36, 60, 36, 41, 42, 43, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 53, 54, 55, 36, 36, 36, 
	36, 36, 59, 60, 61, 63, 36, 41, 
	42, 43, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 54, 55, 36, 
	36, 36, 36, 36, 59, 60, 61, 63, 
	36, 41, 42, 43, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	55, 36, 36, 36, 36, 36, 59, 60, 
	61, 63, 36, 64, 36, 41, 42, 43, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 59, 60, 61, 63, 36, 39, 
	40, 41, 42, 43, 36, 36, 36, 36, 
	36, 36, 50, 51, 52, 36, 53, 54, 
	55, 36, 36, 36, 36, 36, 59, 60, 
	61, 63, 36, 36, 36, 36, 40, 36, 
	40, 41, 42, 43, 36, 36, 36, 36, 
	36, 36, 50, 51, 52, 36, 53, 54, 
	55, 36, 36, 36, 36, 36, 59, 60, 
	61, 63, 36, 36, 36, 36, 40, 36, 
	40, 41, 42, 43, 36, 36, 36, 36, 
	36, 36, 36, 51, 52, 36, 53, 54, 
	55, 36, 36, 36, 36, 36, 59, 60, 
	61, 63, 36, 36, 36, 36, 40, 36, 
	40, 41, 42, 43, 36, 36, 36, 36, 
	36, 36, 36, 36, 52, 36, 53, 54, 
	55, 36, 36, 36, 36, 36, 59, 60, 
	61, 63, 36, 36, 36, 36, 40, 36, 
	65, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 39, 40, 41, 42, 43, 36, 
	45, 46, 36, 36, 36, 50, 51, 52, 
	36, 53, 54, 55, 36, 36, 36, 36, 
	36, 59, 60, 61, 63, 36, 36, 36, 
	36, 40, 36, 40, 41, 42, 43, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 53, 54, 55, 36, 36, 36, 36, 
	36, 59, 60, 61, 63, 36, 36, 36, 
	36, 40, 36, 65, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 39, 40, 41, 
	42, 43, 36, 36, 46, 36, 36, 36, 
	50, 51, 52, 36, 53, 54, 55, 36, 
	36, 36, 36, 36, 59, 60, 61, 63, 
	36, 36, 36, 36, 40, 36, 65, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	39, 40, 41, 42, 43, 36, 36, 36, 
	36, 36, 36, 50, 51, 52, 36, 53, 
	54, 55, 36, 36, 36, 36, 36, 59, 
	60, 61, 63, 36, 36, 36, 36, 40, 
	36, 65, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 39, 40, 41, 42, 43, 
	44, 45, 46, 36, 36, 36, 50, 51, 
	52, 36, 53, 54, 55, 36, 36, 36, 
	36, 36, 59, 60, 61, 63, 36, 36, 
	36, 36, 40, 36, 37, 38, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 39, 
	40, 41, 42, 43, 44, 45, 46, 47, 
	36, 49, 50, 51, 52, 36, 53, 54, 
	55, 36, 36, 36, 36, 58, 59, 60, 
	61, 58, 36, 36, 36, 36, 62, 36, 
	37, 36, 37, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	40, 41, 42, 43, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 53, 54, 
	55, 36, 36, 36, 36, 36, 59, 60, 
	61, 63, 36, 37, 38, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 39, 40, 
	41, 42, 43, 44, 45, 46, 47, 48, 
	49, 50, 51, 52, 36, 53, 54, 55, 
	36, 36, 36, 36, 58, 59, 60, 61, 
	58, 36, 36, 36, 36, 62, 36, 56, 
	57, 36, 57, 36, 67, 68, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 69, 
	70, 71, 72, 73, 74, 75, 76, 77, 
	1, 78, 79, 80, 81, 66, 82, 83, 
	84, 66, 66, 66, 66, 85, 86, 87, 
	88, 89, 66, 66, 66, 66, 90, 66, 
	67, 68, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 69, 70, 71, 72, 73, 
	74, 75, 76, 77, 78, 78, 79, 80, 
	81, 66, 82, 83, 84, 66, 66, 66, 
	66, 85, 86, 87, 88, 89, 66, 66, 
	66, 66, 90, 66, 67, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 70, 71, 72, 73, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	82, 83, 84, 66, 66, 66, 66, 66, 
	86, 87, 88, 91, 66, 66, 66, 66, 
	70, 66, 70, 71, 72, 73, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	82, 83, 84, 66, 66, 66, 66, 66, 
	86, 87, 88, 91, 66, 71, 72, 73, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 86, 87, 88, 66, 72, 73, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 86, 87, 88, 66, 73, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 86, 87, 88, 66, 86, 87, 66, 
	87, 66, 71, 72, 73, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 82, 
	83, 84, 66, 66, 66, 66, 66, 86, 
	87, 88, 91, 66, 71, 72, 73, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 83, 84, 66, 66, 66, 66, 
	66, 86, 87, 88, 91, 66, 71, 72, 
	73, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 84, 66, 66, 
	66, 66, 66, 86, 87, 88, 91, 66, 
	93, 92, 71, 72, 73, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 86, 
	87, 88, 91, 66, 69, 70, 71, 72, 
	73, 66, 66, 66, 66, 66, 66, 79, 
	80, 81, 66, 82, 83, 84, 66, 66, 
	66, 66, 66, 86, 87, 88, 91, 66, 
	66, 66, 66, 70, 66, 70, 71, 72, 
	73, 66, 66, 66, 66, 66, 66, 79, 
	80, 81, 66, 82, 83, 84, 66, 66, 
	66, 66, 66, 86, 87, 88, 91, 66, 
	66, 66, 66, 70, 66, 70, 71, 72, 
	73, 66, 66, 66, 66, 66, 66, 66, 
	80, 81, 66, 82, 83, 84, 66, 66, 
	66, 66, 66, 86, 87, 88, 91, 66, 
	66, 66, 66, 70, 66, 70, 71, 72, 
	73, 66, 66, 66, 66, 66, 66, 66, 
	66, 81, 66, 82, 83, 84, 66, 66, 
	66, 66, 66, 86, 87, 88, 91, 66, 
	66, 66, 66, 70, 66, 94, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 69, 
	70, 71, 72, 73, 66, 75, 76, 66, 
	66, 66, 79, 80, 81, 66, 82, 83, 
	84, 66, 66, 66, 66, 66, 86, 87, 
	88, 91, 66, 66, 66, 66, 70, 66, 
	70, 71, 72, 73, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 82, 83, 
	84, 66, 66, 66, 66, 66, 86, 87, 
	88, 91, 66, 66, 66, 66, 70, 66, 
	94, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 69, 70, 71, 72, 73, 66, 
	66, 76, 66, 66, 66, 79, 80, 81, 
	66, 82, 83, 84, 66, 66, 66, 66, 
	66, 86, 87, 88, 91, 66, 66, 66, 
	66, 70, 66, 94, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 69, 70, 71, 
	72, 73, 66, 66, 66, 66, 66, 66, 
	79, 80, 81, 66, 82, 83, 84, 66, 
	66, 66, 66, 66, 86, 87, 88, 91, 
	66, 66, 66, 66, 70, 66, 94, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	69, 70, 71, 72, 73, 74, 75, 76, 
	66, 66, 66, 79, 80, 81, 66, 82, 
	83, 84, 66, 66, 66, 66, 66, 86, 
	87, 88, 91, 66, 66, 66, 66, 70, 
	66, 67, 68, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 69, 70, 71, 72, 
	73, 74, 75, 76, 77, 66, 78, 79, 
	80, 81, 66, 82, 83, 84, 66, 66, 
	66, 66, 85, 86, 87, 88, 89, 66, 
	66, 66, 66, 90, 66, 67, 95, 67, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 70, 71, 72, 
	73, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 82, 83, 84, 66, 66, 
	66, 66, 66, 86, 87, 88, 91, 66, 
	97, 96, 3, 98, 99, 100, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 101, 
	102, 103, 104, 105, 106, 107, 108, 109, 
	110, 111, 112, 113, 114, 66, 115, 116, 
	117, 66, 56, 57, 66, 118, 119, 120, 
	88, 121, 66, 66, 66, 66, 122, 66, 
	99, 100, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 101, 102, 103, 104, 105, 
	106, 107, 108, 109, 111, 111, 112, 113, 
	114, 66, 115, 116, 117, 66, 66, 66, 
	66, 118, 119, 120, 88, 121, 66, 66, 
	66, 66, 122, 66, 99, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 102, 103, 104, 105, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	115, 116, 117, 66, 66, 66, 66, 66, 
	119, 120, 88, 123, 66, 66, 66, 66, 
	102, 66, 102, 103, 104, 105, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	115, 116, 117, 66, 66, 66, 66, 66, 
	119, 120, 88, 123, 66, 103, 104, 105, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 119, 120, 88, 66, 104, 105, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 119, 120, 88, 66, 105, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 119, 120, 88, 66, 119, 120, 66, 
	120, 66, 103, 104, 105, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 115, 
	116, 117, 66, 66, 66, 66, 66, 119, 
	120, 88, 123, 66, 103, 104, 105, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 116, 117, 66, 66, 66, 66, 
	66, 119, 120, 88, 123, 66, 103, 104, 
	105, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 117, 66, 66, 
	66, 66, 66, 119, 120, 88, 123, 66, 
	124, 92, 103, 104, 105, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 119, 
	120, 88, 123, 66, 101, 102, 103, 104, 
	105, 66, 66, 66, 66, 66, 66, 112, 
	113, 114, 66, 115, 116, 117, 66, 66, 
	66, 66, 66, 119, 120, 88, 123, 66, 
	66, 66, 66, 102, 66, 102, 103, 104, 
	105, 66, 66, 66, 66, 66, 66, 112, 
	113, 114, 66, 115, 116, 117, 66, 66, 
	66, 66, 66, 119, 120, 88, 123, 66, 
	66, 66, 66, 102, 66, 102, 103, 104, 
	105, 66, 66, 66, 66, 66, 66, 66, 
	113, 114, 66, 115, 116, 117, 66, 66, 
	66, 66, 66, 119, 120, 88, 123, 66, 
	66, 66, 66, 102, 66, 102, 103, 104, 
	105, 66, 66, 66, 66, 66, 66, 66, 
	66, 114, 66, 115, 116, 117, 66, 66, 
	66, 66, 66, 119, 120, 88, 123, 66, 
	66, 66, 66, 102, 66, 125, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 101, 
	102, 103, 104, 105, 66, 107, 108, 66, 
	66, 66, 112, 113, 114, 66, 115, 116, 
	117, 66, 66, 66, 66, 66, 119, 120, 
	88, 123, 66, 66, 66, 66, 102, 66, 
	102, 103, 104, 105, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 115, 116, 
	117, 66, 66, 66, 66, 66, 119, 120, 
	88, 123, 66, 66, 66, 66, 102, 66, 
	125, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 101, 102, 103, 104, 105, 66, 
	66, 108, 66, 66, 66, 112, 113, 114, 
	66, 115, 116, 117, 66, 66, 66, 66, 
	66, 119, 120, 88, 123, 66, 66, 66, 
	66, 102, 66, 125, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 101, 102, 103, 
	104, 105, 66, 66, 66, 66, 66, 66, 
	112, 113, 114, 66, 115, 116, 117, 66, 
	66, 66, 66, 66, 119, 120, 88, 123, 
	66, 66, 66, 66, 102, 66, 125, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	101, 102, 103, 104, 105, 106, 107, 108, 
	66, 66, 66, 112, 113, 114, 66, 115, 
	116, 117, 66, 66, 66, 66, 66, 119, 
	120, 88, 123, 66, 66, 66, 66, 102, 
	66, 99, 100, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 101, 102, 103, 104, 
	105, 106, 107, 108, 109, 66, 111, 112, 
	113, 114, 66, 115, 116, 117, 66, 66, 
	66, 66, 118, 119, 120, 88, 121, 66, 
	66, 66, 66, 122, 66, 99, 95, 99, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 102, 103, 104, 
	105, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 115, 116, 117, 66, 66, 
	66, 66, 66, 119, 120, 88, 123, 66, 
	99, 100, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 101, 102, 103, 104, 105, 
	106, 107, 108, 109, 110, 111, 112, 113, 
	114, 66, 115, 116, 117, 66, 66, 66, 
	66, 118, 119, 120, 88, 121, 66, 66, 
	66, 66, 122, 66, 5, 6, 126, 126, 
	126, 126, 126, 126, 126, 126, 126, 9, 
	10, 11, 12, 13, 14, 15, 16, 17, 
	19, 19, 20, 21, 22, 126, 23, 24, 
	25, 126, 126, 126, 126, 29, 30, 31, 
	32, 29, 126, 126, 126, 126, 35, 126, 
	5, 126, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 10, 11, 
	12, 13, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 23, 24, 25, 126, 
	126, 126, 126, 126, 30, 31, 32, 127, 
	126, 126, 126, 126, 10, 126, 10, 11, 
	12, 13, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 23, 24, 25, 126, 
	126, 126, 126, 126, 30, 31, 32, 127, 
	126, 11, 12, 13, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 30, 31, 
	32, 126, 12, 13, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 30, 31, 
	32, 126, 13, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 30, 31, 32, 
	126, 30, 31, 126, 31, 126, 11, 12, 
	13, 126, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 23, 24, 25, 126, 126, 
	126, 126, 126, 30, 31, 32, 127, 126, 
	11, 12, 13, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 24, 25, 
	126, 126, 126, 126, 126, 30, 31, 32, 
	127, 126, 11, 12, 13, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 25, 126, 126, 126, 126, 126, 30, 
	31, 32, 127, 126, 128, 126, 11, 12, 
	13, 126, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 30, 31, 32, 127, 126, 
	9, 10, 11, 12, 13, 126, 126, 126, 
	126, 126, 126, 20, 21, 22, 126, 23, 
	24, 25, 126, 126, 126, 126, 126, 30, 
	31, 32, 127, 126, 126, 126, 126, 10, 
	126, 10, 11, 12, 13, 126, 126, 126, 
	126, 126, 126, 20, 21, 22, 126, 23, 
	24, 25, 126, 126, 126, 126, 126, 30, 
	31, 32, 127, 126, 126, 126, 126, 10, 
	126, 10, 11, 12, 13, 126, 126, 126, 
	126, 126, 126, 126, 21, 22, 126, 23, 
	24, 25, 126, 126, 126, 126, 126, 30, 
	31, 32, 127, 126, 126, 126, 126, 10, 
	126, 10, 11, 12, 13, 126, 126, 126, 
	126, 126, 126, 126, 126, 22, 126, 23, 
	24, 25, 126, 126, 126, 126, 126, 30, 
	31, 32, 127, 126, 126, 126, 126, 10, 
	126, 129, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 9, 10, 11, 12, 13, 
	126, 15, 16, 126, 126, 126, 20, 21, 
	22, 126, 23, 24, 25, 126, 126, 126, 
	126, 126, 30, 31, 32, 127, 126, 126, 
	126, 126, 10, 126, 10, 11, 12, 13, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 126, 23, 24, 25, 126, 126, 126, 
	126, 126, 30, 31, 32, 127, 126, 126, 
	126, 126, 10, 126, 129, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 9, 10, 
	11, 12, 13, 126, 126, 16, 126, 126, 
	126, 20, 21, 22, 126, 23, 24, 25, 
	126, 126, 126, 126, 126, 30, 31, 32, 
	127, 126, 126, 126, 126, 10, 126, 129, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 9, 10, 11, 12, 13, 126, 126, 
	126, 126, 126, 126, 20, 21, 22, 126, 
	23, 24, 25, 126, 126, 126, 126, 126, 
	30, 31, 32, 127, 126, 126, 126, 126, 
	10, 126, 129, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 9, 10, 11, 12, 
	13, 14, 15, 16, 126, 126, 126, 20, 
	21, 22, 126, 23, 24, 25, 126, 126, 
	126, 126, 126, 30, 31, 32, 127, 126, 
	126, 126, 126, 10, 126, 5, 6, 126, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	9, 10, 11, 12, 13, 14, 15, 16, 
	17, 126, 19, 20, 21, 22, 126, 23, 
	24, 25, 126, 126, 126, 126, 29, 30, 
	31, 32, 29, 126, 126, 126, 126, 35, 
	126, 5, 126, 5, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 10, 11, 12, 13, 126, 126, 126, 
	126, 126, 126, 126, 126, 126, 126, 23, 
	24, 25, 126, 126, 126, 126, 126, 30, 
	31, 32, 127, 126, 130, 126, 7, 126, 
	1, 126, 126, 126, 1, 126, 126, 126, 
	126, 126, 5, 6, 7, 126, 126, 126, 
	126, 126, 126, 126, 126, 9, 10, 11, 
	12, 13, 14, 15, 16, 17, 18, 19, 
	20, 21, 22, 126, 23, 24, 25, 126, 
	26, 27, 126, 29, 30, 31, 32, 29, 
	126, 126, 126, 126, 35, 126, 5, 6, 
	126, 126, 126, 126, 126, 126, 126, 126, 
	126, 9, 10, 11, 12, 13, 14, 15, 
	16, 17, 18, 19, 20, 21, 22, 126, 
	23, 24, 25, 126, 126, 126, 126, 29, 
	30, 31, 32, 29, 126, 126, 126, 126, 
	35, 126, 26, 27, 126, 27, 126, 1, 
	131, 131, 131, 1, 131, 133, 132, 33, 
	132, 33, 133, 132, 133, 132, 33, 132, 
	34, 132, 0
};

static const char _use_syllable_machine_trans_targs[] = {
	1, 30, 0, 56, 58, 85, 86, 110, 
	112, 98, 87, 88, 89, 90, 102, 104, 
	105, 106, 113, 107, 99, 100, 101, 93, 
	94, 95, 114, 115, 116, 108, 91, 92, 
	0, 117, 119, 109, 0, 2, 3, 15, 
	4, 5, 6, 7, 19, 21, 22, 23, 
	27, 24, 16, 17, 18, 10, 11, 12, 
	28, 29, 25, 8, 9, 0, 26, 13, 
	14, 20, 0, 31, 32, 44, 33, 34, 
	35, 36, 48, 50, 51, 52, 53, 45, 
	46, 47, 39, 40, 41, 54, 37, 38, 
	0, 54, 55, 42, 0, 43, 49, 0, 
	0, 57, 0, 59, 60, 72, 61, 62, 
	63, 64, 76, 78, 79, 80, 84, 81, 
	73, 74, 75, 67, 68, 69, 82, 65, 
	66, 82, 83, 70, 71, 77, 0, 96, 
	97, 103, 111, 0, 0, 118
};

static const char _use_syllable_machine_trans_actions[] = {
	0, 0, 3, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	4, 0, 0, 0, 5, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 6, 0, 0, 
	0, 0, 7, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 8, 0, 0, 
	9, 10, 0, 0, 11, 0, 0, 12, 
	13, 0, 14, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 8, 0, 
	0, 10, 0, 0, 0, 0, 15, 0, 
	0, 0, 0, 16, 17, 0
};

static const char _use_syllable_machine_to_state_actions[] = {
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0
};

static const char _use_syllable_machine_from_state_actions[] = {
	2, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0
};

static const short _use_syllable_machine_eof_trans[] = {
	0, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 67, 67, 
	67, 67, 67, 67, 67, 67, 67, 67, 
	67, 67, 93, 67, 67, 67, 67, 67, 
	67, 67, 67, 67, 67, 67, 96, 67, 
	97, 99, 67, 67, 67, 67, 67, 67, 
	67, 67, 67, 67, 67, 67, 93, 67, 
	67, 67, 67, 67, 67, 67, 67, 67, 
	67, 67, 96, 67, 67, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 132, 133, 133, 133
};

static const int use_syllable_machine_start = 0;
static const int use_syllable_machine_first_final = 0;
static const int use_syllable_machine_error = -1;

static const int use_syllable_machine_en_main = 0;


#line 58 "hb-ot-shaper-use-machine.rl"



#line 182 "hb-ot-shaper-use-machine.rl"


#define found_syllable(syllable_type) \
  HB_STMT_START { \
    if (0) fprintf (stderr, "syllable %d..%d %s\n", (*ts).second.first, (*te).second.first, #syllable_type); \
    for (unsigned i = (*ts).second.first; i < (*te).second.first; ++i) \
      info[i].syllable() = (syllable_serial << 4) | syllable_type; \
    syllable_serial++; \
    if (unlikely (syllable_serial == 16)) syllable_serial = 1; \
  } HB_STMT_END


template <typename Iter>
struct machine_index_t :
  hb_iter_with_fallback_t<machine_index_t<Iter>,
			  typename Iter::item_t>
{
  machine_index_t (const Iter& it) : it (it) {}
  machine_index_t (const machine_index_t& o) : hb_iter_with_fallback_t<machine_index_t<Iter>,
								       typename Iter::item_t> (),
					       it (o.it), is_null (o.is_null) {}

  static constexpr bool is_random_access_iterator = Iter::is_random_access_iterator;
  static constexpr bool is_sorted_iterator = Iter::is_sorted_iterator;

  typename Iter::item_t __item__ () const { return *it; }
  typename Iter::item_t __item_at__ (unsigned i) const { return it[i]; }
  unsigned __len__ () const { return it.len (); }
  void __next__ () { ++it; }
  void __forward__ (unsigned n) { it += n; }
  void __prev__ () { --it; }
  void __rewind__ (unsigned n) { it -= n; }

  void operator = (unsigned n)
  {
    assert (n == 0);
    is_null = true;
  }
  explicit operator bool () { return !is_null; }

  void operator = (const machine_index_t& o)
  {
    is_null = o.is_null;
    unsigned index = (*it).first;
    unsigned n = (*o.it).first;
    if (index < n) it += n - index; else if (index > n) it -= index - n;
  }
  bool operator == (const machine_index_t& o) const
  { return is_null ? o.is_null : !o.is_null && (*it).first == (*o.it).first; }
  bool operator != (const machine_index_t& o) const { return !(*this == o); }

  private:
  Iter it;
  bool is_null = false;
};
struct
{
  template <typename Iter,
	    hb_requires (hb_is_iterable (Iter))>
  machine_index_t<hb_iter_type<Iter>>
  operator () (Iter&& it) const
  { return machine_index_t<hb_iter_type<Iter>> (hb_iter (it)); }
}
HB_FUNCOBJ (machine_index);



static bool
not_ccs_default_ignorable (const hb_glyph_info_t &i)
{ return i.use_category() != USE(CGJ); }

static inline void
find_syllables_use (hb_buffer_t *buffer)
{
  hb_glyph_info_t *info = buffer->info;
  auto p =
    + hb_iter (info, buffer->len)
    | hb_enumerate
    | hb_filter ([] (const hb_glyph_info_t &i) { return not_ccs_default_ignorable (i); },
		 hb_second)
    | hb_filter ([&] (const hb_pair_t<unsigned, const hb_glyph_info_t &> p)
		 {
		   if (p.second.use_category() == USE(ZWNJ))
		     for (unsigned i = p.first + 1; i < buffer->len; ++i)
		       if (not_ccs_default_ignorable (info[i]))
			 return !_hb_glyph_info_is_unicode_mark (&info[i]);
		   return true;
		 })
    | hb_enumerate
    | machine_index
    ;
  auto pe = p + p.len ();
  auto eof = +pe;
  auto ts = +p;
  auto te = +p;
  unsigned int act HB_UNUSED;
  int cs;
  
#line 773 "hb-ot-shaper-use-machine.hh"
	{
	cs = use_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 282 "hb-ot-shaper-use-machine.rl"


  unsigned int syllable_serial = 1;
  
#line 782 "hb-ot-shaper-use-machine.hh"
	{
	int _slen;
	int _trans;
	const unsigned char *_keys;
	const unsigned char *_inds;
	if ( p == pe )
		goto _test_eof;
_resume:
	switch ( _use_syllable_machine_from_state_actions[cs] ) {
	case 2:
#line 1 "NONE"
	{ts = p;}
	break;
#line 794 "hb-ot-shaper-use-machine.hh"
	}

	_keys = _use_syllable_machine_trans_keys + (cs<<1);
	_inds = _use_syllable_machine_indicies + _use_syllable_machine_index_offsets[cs];

	_slen = _use_syllable_machine_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=( (*p).second.second.use_category()) &&
		( (*p).second.second.use_category()) <= _keys[1] ?
		( (*p).second.second.use_category()) - _keys[0] : _slen ];

_eof_trans:
	cs = _use_syllable_machine_trans_targs[_trans];

	if ( _use_syllable_machine_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _use_syllable_machine_trans_actions[_trans] ) {
	case 9:
#line 172 "hb-ot-shaper-use-machine.rl"
	{te = p+1;{ found_syllable (use_standard_cluster); }}
	break;
	case 6:
#line 175 "hb-ot-shaper-use-machine.rl"
	{te = p+1;{ found_syllable (use_symbol_cluster); }}
	break;
	case 4:
#line 177 "hb-ot-shaper-use-machine.rl"
	{te = p+1;{ found_syllable (use_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }}
	break;
	case 3:
#line 178 "hb-ot-shaper-use-machine.rl"
	{te = p+1;{ found_syllable (use_non_cluster); }}
	break;
	case 11:
#line 171 "hb-ot-shaper-use-machine.rl"
	{te = p;p--;{ found_syllable (use_sakot_terminated_cluster); }}
	break;
	case 7:
#line 172 "hb-ot-shaper-use-machine.rl"
	{te = p;p--;{ found_syllable (use_standard_cluster); }}
	break;
	case 14:
#line 173 "hb-ot-shaper-use-machine.rl"
	{te = p;p--;{ found_syllable (use_number_joiner_terminated_cluster); }}
	break;
	case 13:
#line 174 "hb-ot-shaper-use-machine.rl"
	{te = p;p--;{ found_syllable (use_numeral_cluster); }}
	break;
	case 5:
#line 175 "hb-ot-shaper-use-machine.rl"
	{te = p;p--;{ found_syllable (use_symbol_cluster); }}
	break;
	case 17:
#line 176 "hb-ot-shaper-use-machine.rl"
	{te = p;p--;{ found_syllable (use_hieroglyph_cluster); }}
	break;
	case 15:
#line 177 "hb-ot-shaper-use-machine.rl"
	{te = p;p--;{ found_syllable (use_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }}
	break;
	case 16:
#line 178 "hb-ot-shaper-use-machine.rl"
	{te = p;p--;{ found_syllable (use_non_cluster); }}
	break;
	case 12:
#line 1 "NONE"
	{	switch( act ) {
	case 1:
	{{p = ((te))-1;} found_syllable (use_virama_terminated_cluster); }
	break;
	case 2:
	{{p = ((te))-1;} found_syllable (use_sakot_terminated_cluster); }
	break;
	}
	}
	break;
	case 8:
#line 1 "NONE"
	{te = p+1;}
#line 170 "hb-ot-shaper-use-machine.rl"
	{act = 1;}
	break;
	case 10:
#line 1 "NONE"
	{te = p+1;}
#line 171 "hb-ot-shaper-use-machine.rl"
	{act = 2;}
	break;
#line 866 "hb-ot-shaper-use-machine.hh"
	}

_again:
	switch ( _use_syllable_machine_to_state_actions[cs] ) {
	case 1:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 873 "hb-ot-shaper-use-machine.hh"
	}

	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	if ( _use_syllable_machine_eof_trans[cs] > 0 ) {
		_trans = _use_syllable_machine_eof_trans[cs] - 1;
		goto _eof_trans;
	}
	}

	}

#line 287 "hb-ot-shaper-use-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPER_USE_MACHINE_HH */
