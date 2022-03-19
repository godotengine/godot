
#line 1 "hb-ot-shape-complex-use-machine.rl"
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

#ifndef HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH
#define HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH

#include "hb.hh"

#include "hb-ot-shape-complex-syllabic.hh"

/* buffer var allocations */
#define use_category() complex_var_u8_category()

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


#line 57 "hb-ot-shape-complex-use-machine.hh"
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
#define use_syllable_machine_ex_HVM 44u
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


#line 100 "hb-ot-shape-complex-use-machine.hh"
static const unsigned char _use_syllable_machine_trans_keys[] = {
	0u, 51u, 11u, 48u, 11u, 48u, 1u, 1u, 22u, 48u, 23u, 48u, 24u, 47u, 25u, 47u, 
	26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 1u, 1u, 24u, 48u, 
	23u, 48u, 23u, 48u, 23u, 48u, 22u, 48u, 22u, 48u, 22u, 48u, 11u, 48u, 1u, 48u, 
	11u, 48u, 41u, 42u, 42u, 42u, 11u, 48u, 11u, 48u, 1u, 1u, 22u, 48u, 23u, 48u, 
	24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 
	1u, 1u, 24u, 48u, 23u, 48u, 23u, 48u, 23u, 48u, 22u, 48u, 22u, 48u, 22u, 48u, 
	11u, 48u, 1u, 48u, 13u, 13u, 4u, 4u, 11u, 48u, 11u, 48u, 1u, 1u, 22u, 48u, 
	23u, 48u, 24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 
	24u, 48u, 1u, 1u, 24u, 48u, 23u, 48u, 23u, 48u, 23u, 48u, 22u, 48u, 22u, 48u, 
	22u, 48u, 11u, 48u, 1u, 48u, 11u, 48u, 11u, 48u, 1u, 1u, 22u, 48u, 23u, 48u, 
	24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 
	1u, 1u, 24u, 48u, 23u, 48u, 23u, 48u, 23u, 48u, 22u, 48u, 22u, 48u, 22u, 48u, 
	11u, 48u, 1u, 48u, 4u, 4u, 13u, 13u, 1u, 48u, 11u, 48u, 41u, 42u, 42u, 42u, 
	1u, 5u, 50u, 52u, 49u, 52u, 49u, 51u, 0
};

static const char _use_syllable_machine_key_spans[] = {
	52, 38, 38, 1, 27, 26, 24, 23, 
	22, 2, 1, 25, 25, 25, 1, 25, 
	26, 26, 26, 27, 27, 27, 38, 48, 
	38, 2, 1, 38, 38, 1, 27, 26, 
	24, 23, 22, 2, 1, 25, 25, 25, 
	1, 25, 26, 26, 26, 27, 27, 27, 
	38, 48, 1, 1, 38, 38, 1, 27, 
	26, 24, 23, 22, 2, 1, 25, 25, 
	25, 1, 25, 26, 26, 26, 27, 27, 
	27, 38, 48, 38, 38, 1, 27, 26, 
	24, 23, 22, 2, 1, 25, 25, 25, 
	1, 25, 26, 26, 26, 27, 27, 27, 
	38, 48, 1, 1, 48, 38, 2, 1, 
	5, 3, 4, 3
};

static const short _use_syllable_machine_index_offsets[] = {
	0, 53, 92, 131, 133, 161, 188, 213, 
	237, 260, 263, 265, 291, 317, 343, 345, 
	371, 398, 425, 452, 480, 508, 536, 575, 
	624, 663, 666, 668, 707, 746, 748, 776, 
	803, 828, 852, 875, 878, 880, 906, 932, 
	958, 960, 986, 1013, 1040, 1067, 1095, 1123, 
	1151, 1190, 1239, 1241, 1243, 1282, 1321, 1323, 
	1351, 1378, 1403, 1427, 1450, 1453, 1455, 1481, 
	1507, 1533, 1535, 1561, 1588, 1615, 1642, 1670, 
	1698, 1726, 1765, 1814, 1853, 1892, 1894, 1922, 
	1949, 1974, 1998, 2021, 2024, 2026, 2052, 2078, 
	2104, 2106, 2132, 2159, 2186, 2213, 2241, 2269, 
	2297, 2336, 2385, 2387, 2389, 2438, 2477, 2480, 
	2482, 2488, 2492, 2497
};

static const char _use_syllable_machine_indicies[] = {
	0, 1, 2, 2, 3, 4, 2, 2, 
	2, 2, 2, 5, 6, 7, 2, 2, 
	2, 2, 8, 2, 2, 2, 9, 10, 
	11, 12, 13, 14, 15, 9, 16, 17, 
	18, 19, 20, 21, 2, 22, 23, 24, 
	2, 25, 26, 27, 28, 29, 30, 31, 
	6, 32, 2, 33, 2, 35, 36, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	37, 38, 39, 40, 41, 42, 43, 37, 
	44, 45, 46, 47, 48, 49, 34, 50, 
	51, 52, 34, 53, 54, 34, 55, 56, 
	57, 58, 36, 34, 35, 36, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 37, 
	38, 39, 40, 41, 42, 43, 37, 44, 
	46, 46, 47, 48, 49, 34, 50, 51, 
	52, 34, 34, 34, 34, 55, 56, 57, 
	58, 36, 34, 35, 34, 37, 38, 39, 
	40, 41, 34, 34, 34, 34, 34, 34, 
	47, 48, 49, 34, 50, 51, 52, 34, 
	34, 34, 34, 38, 56, 57, 58, 59, 
	34, 38, 39, 40, 41, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 50, 
	51, 52, 34, 34, 34, 34, 34, 56, 
	57, 58, 59, 34, 39, 40, 41, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 56, 57, 58, 34, 40, 41, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 56, 57, 58, 34, 41, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	56, 57, 58, 34, 56, 57, 34, 57, 
	34, 39, 40, 41, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 50, 51, 
	52, 34, 34, 34, 34, 34, 56, 57, 
	58, 59, 34, 39, 40, 41, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 51, 52, 34, 34, 34, 34, 34, 
	56, 57, 58, 59, 34, 39, 40, 41, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 52, 34, 34, 34, 
	34, 34, 56, 57, 58, 59, 34, 60, 
	34, 39, 40, 41, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 56, 57, 
	58, 59, 34, 38, 39, 40, 41, 34, 
	34, 34, 34, 34, 34, 47, 48, 49, 
	34, 50, 51, 52, 34, 34, 34, 34, 
	38, 56, 57, 58, 59, 34, 38, 39, 
	40, 41, 34, 34, 34, 34, 34, 34, 
	34, 48, 49, 34, 50, 51, 52, 34, 
	34, 34, 34, 38, 56, 57, 58, 59, 
	34, 38, 39, 40, 41, 34, 34, 34, 
	34, 34, 34, 34, 34, 49, 34, 50, 
	51, 52, 34, 34, 34, 34, 38, 56, 
	57, 58, 59, 34, 37, 38, 39, 40, 
	41, 34, 43, 37, 34, 34, 34, 47, 
	48, 49, 34, 50, 51, 52, 34, 34, 
	34, 34, 38, 56, 57, 58, 59, 34, 
	37, 38, 39, 40, 41, 34, 34, 37, 
	34, 34, 34, 47, 48, 49, 34, 50, 
	51, 52, 34, 34, 34, 34, 38, 56, 
	57, 58, 59, 34, 37, 38, 39, 40, 
	41, 42, 43, 37, 34, 34, 34, 47, 
	48, 49, 34, 50, 51, 52, 34, 34, 
	34, 34, 38, 56, 57, 58, 59, 34, 
	35, 36, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 37, 38, 39, 40, 41, 
	42, 43, 37, 44, 34, 46, 47, 48, 
	49, 34, 50, 51, 52, 34, 34, 34, 
	34, 55, 56, 57, 58, 36, 34, 35, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 38, 39, 40, 
	41, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 50, 51, 52, 34, 34, 
	34, 34, 34, 56, 57, 58, 59, 34, 
	35, 36, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 37, 38, 39, 40, 41, 
	42, 43, 37, 44, 45, 46, 47, 48, 
	49, 34, 50, 51, 52, 34, 34, 34, 
	34, 55, 56, 57, 58, 36, 34, 53, 
	54, 34, 54, 34, 62, 63, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 64, 
	65, 66, 67, 68, 69, 70, 64, 71, 
	1, 72, 73, 74, 75, 61, 76, 77, 
	78, 61, 61, 61, 61, 79, 80, 81, 
	82, 63, 61, 62, 63, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 64, 65, 
	66, 67, 68, 69, 70, 64, 71, 72, 
	72, 73, 74, 75, 61, 76, 77, 78, 
	61, 61, 61, 61, 79, 80, 81, 82, 
	63, 61, 62, 83, 64, 65, 66, 67, 
	68, 61, 61, 61, 61, 61, 61, 73, 
	74, 75, 61, 76, 77, 78, 61, 61, 
	61, 61, 65, 80, 81, 82, 84, 61, 
	65, 66, 67, 68, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 76, 77, 
	78, 61, 61, 61, 61, 61, 80, 81, 
	82, 84, 61, 66, 67, 68, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	80, 81, 82, 61, 67, 68, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	80, 81, 82, 61, 68, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 80, 
	81, 82, 61, 80, 81, 61, 81, 61, 
	66, 67, 68, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 76, 77, 78, 
	61, 61, 61, 61, 61, 80, 81, 82, 
	84, 61, 66, 67, 68, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	77, 78, 61, 61, 61, 61, 61, 80, 
	81, 82, 84, 61, 66, 67, 68, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 78, 61, 61, 61, 61, 
	61, 80, 81, 82, 84, 61, 86, 85, 
	66, 67, 68, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 80, 81, 82, 
	84, 61, 65, 66, 67, 68, 61, 61, 
	61, 61, 61, 61, 73, 74, 75, 61, 
	76, 77, 78, 61, 61, 61, 61, 65, 
	80, 81, 82, 84, 61, 65, 66, 67, 
	68, 61, 61, 61, 61, 61, 61, 61, 
	74, 75, 61, 76, 77, 78, 61, 61, 
	61, 61, 65, 80, 81, 82, 84, 61, 
	65, 66, 67, 68, 61, 61, 61, 61, 
	61, 61, 61, 61, 75, 61, 76, 77, 
	78, 61, 61, 61, 61, 65, 80, 81, 
	82, 84, 61, 64, 65, 66, 67, 68, 
	61, 70, 64, 61, 61, 61, 73, 74, 
	75, 61, 76, 77, 78, 61, 61, 61, 
	61, 65, 80, 81, 82, 84, 61, 64, 
	65, 66, 67, 68, 61, 61, 64, 61, 
	61, 61, 73, 74, 75, 61, 76, 77, 
	78, 61, 61, 61, 61, 65, 80, 81, 
	82, 84, 61, 64, 65, 66, 67, 68, 
	69, 70, 64, 61, 61, 61, 73, 74, 
	75, 61, 76, 77, 78, 61, 61, 61, 
	61, 65, 80, 81, 82, 84, 61, 62, 
	63, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 64, 65, 66, 67, 68, 69, 
	70, 64, 71, 61, 72, 73, 74, 75, 
	61, 76, 77, 78, 61, 61, 61, 61, 
	79, 80, 81, 82, 63, 61, 62, 83, 
	83, 83, 83, 83, 83, 83, 83, 83, 
	83, 83, 83, 83, 83, 83, 83, 83, 
	83, 83, 83, 83, 65, 66, 67, 68, 
	83, 83, 83, 83, 83, 83, 83, 83, 
	83, 83, 76, 77, 78, 83, 83, 83, 
	83, 83, 80, 81, 82, 84, 83, 88, 
	87, 3, 89, 90, 91, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 92, 93, 
	94, 95, 96, 97, 98, 92, 99, 100, 
	101, 102, 103, 104, 61, 105, 106, 107, 
	61, 53, 54, 61, 108, 109, 110, 82, 
	91, 61, 90, 91, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 92, 93, 94, 
	95, 96, 97, 98, 92, 99, 101, 101, 
	102, 103, 104, 61, 105, 106, 107, 61, 
	61, 61, 61, 108, 109, 110, 82, 91, 
	61, 90, 83, 92, 93, 94, 95, 96, 
	61, 61, 61, 61, 61, 61, 102, 103, 
	104, 61, 105, 106, 107, 61, 61, 61, 
	61, 93, 109, 110, 82, 111, 61, 93, 
	94, 95, 96, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 105, 106, 107, 
	61, 61, 61, 61, 61, 109, 110, 82, 
	111, 61, 94, 95, 96, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 109, 
	110, 82, 61, 95, 96, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 109, 
	110, 82, 61, 96, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 109, 110, 
	82, 61, 109, 110, 61, 110, 61, 94, 
	95, 96, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 105, 106, 107, 61, 
	61, 61, 61, 61, 109, 110, 82, 111, 
	61, 94, 95, 96, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 106, 
	107, 61, 61, 61, 61, 61, 109, 110, 
	82, 111, 61, 94, 95, 96, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 107, 61, 61, 61, 61, 61, 
	109, 110, 82, 111, 61, 112, 85, 94, 
	95, 96, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 61, 61, 109, 110, 82, 111, 
	61, 93, 94, 95, 96, 61, 61, 61, 
	61, 61, 61, 102, 103, 104, 61, 105, 
	106, 107, 61, 61, 61, 61, 93, 109, 
	110, 82, 111, 61, 93, 94, 95, 96, 
	61, 61, 61, 61, 61, 61, 61, 103, 
	104, 61, 105, 106, 107, 61, 61, 61, 
	61, 93, 109, 110, 82, 111, 61, 93, 
	94, 95, 96, 61, 61, 61, 61, 61, 
	61, 61, 61, 104, 61, 105, 106, 107, 
	61, 61, 61, 61, 93, 109, 110, 82, 
	111, 61, 92, 93, 94, 95, 96, 61, 
	98, 92, 61, 61, 61, 102, 103, 104, 
	61, 105, 106, 107, 61, 61, 61, 61, 
	93, 109, 110, 82, 111, 61, 92, 93, 
	94, 95, 96, 61, 61, 92, 61, 61, 
	61, 102, 103, 104, 61, 105, 106, 107, 
	61, 61, 61, 61, 93, 109, 110, 82, 
	111, 61, 92, 93, 94, 95, 96, 97, 
	98, 92, 61, 61, 61, 102, 103, 104, 
	61, 105, 106, 107, 61, 61, 61, 61, 
	93, 109, 110, 82, 111, 61, 90, 91, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 92, 93, 94, 95, 96, 97, 98, 
	92, 99, 61, 101, 102, 103, 104, 61, 
	105, 106, 107, 61, 61, 61, 61, 108, 
	109, 110, 82, 91, 61, 90, 83, 83, 
	83, 83, 83, 83, 83, 83, 83, 83, 
	83, 83, 83, 83, 83, 83, 83, 83, 
	83, 83, 83, 93, 94, 95, 96, 83, 
	83, 83, 83, 83, 83, 83, 83, 83, 
	83, 105, 106, 107, 83, 83, 83, 83, 
	83, 109, 110, 82, 111, 83, 90, 91, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 92, 93, 94, 95, 96, 97, 98, 
	92, 99, 100, 101, 102, 103, 104, 61, 
	105, 106, 107, 61, 61, 61, 61, 108, 
	109, 110, 82, 91, 61, 5, 6, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	9, 10, 11, 12, 13, 14, 15, 9, 
	16, 18, 18, 19, 20, 21, 113, 22, 
	23, 24, 113, 113, 113, 113, 28, 29, 
	30, 31, 6, 113, 5, 113, 9, 10, 
	11, 12, 13, 113, 113, 113, 113, 113, 
	113, 19, 20, 21, 113, 22, 23, 24, 
	113, 113, 113, 113, 10, 29, 30, 31, 
	114, 113, 10, 11, 12, 13, 113, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	22, 23, 24, 113, 113, 113, 113, 113, 
	29, 30, 31, 114, 113, 11, 12, 13, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 29, 30, 31, 113, 12, 13, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 29, 30, 31, 113, 13, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 29, 30, 31, 113, 29, 30, 113, 
	30, 113, 11, 12, 13, 113, 113, 113, 
	113, 113, 113, 113, 113, 113, 113, 22, 
	23, 24, 113, 113, 113, 113, 113, 29, 
	30, 31, 114, 113, 11, 12, 13, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 23, 24, 113, 113, 113, 113, 
	113, 29, 30, 31, 114, 113, 11, 12, 
	13, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 113, 24, 113, 113, 
	113, 113, 113, 29, 30, 31, 114, 113, 
	115, 113, 11, 12, 13, 113, 113, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 113, 113, 113, 29, 
	30, 31, 114, 113, 10, 11, 12, 13, 
	113, 113, 113, 113, 113, 113, 19, 20, 
	21, 113, 22, 23, 24, 113, 113, 113, 
	113, 10, 29, 30, 31, 114, 113, 10, 
	11, 12, 13, 113, 113, 113, 113, 113, 
	113, 113, 20, 21, 113, 22, 23, 24, 
	113, 113, 113, 113, 10, 29, 30, 31, 
	114, 113, 10, 11, 12, 13, 113, 113, 
	113, 113, 113, 113, 113, 113, 21, 113, 
	22, 23, 24, 113, 113, 113, 113, 10, 
	29, 30, 31, 114, 113, 9, 10, 11, 
	12, 13, 113, 15, 9, 113, 113, 113, 
	19, 20, 21, 113, 22, 23, 24, 113, 
	113, 113, 113, 10, 29, 30, 31, 114, 
	113, 9, 10, 11, 12, 13, 113, 113, 
	9, 113, 113, 113, 19, 20, 21, 113, 
	22, 23, 24, 113, 113, 113, 113, 10, 
	29, 30, 31, 114, 113, 9, 10, 11, 
	12, 13, 14, 15, 9, 113, 113, 113, 
	19, 20, 21, 113, 22, 23, 24, 113, 
	113, 113, 113, 10, 29, 30, 31, 114, 
	113, 5, 6, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 9, 10, 11, 12, 
	13, 14, 15, 9, 16, 113, 18, 19, 
	20, 21, 113, 22, 23, 24, 113, 113, 
	113, 113, 28, 29, 30, 31, 6, 113, 
	5, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 113, 113, 10, 11, 
	12, 13, 113, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 22, 23, 24, 113, 
	113, 113, 113, 113, 29, 30, 31, 114, 
	113, 116, 113, 7, 113, 1, 113, 113, 
	113, 1, 113, 113, 113, 113, 113, 5, 
	6, 7, 113, 113, 113, 113, 113, 113, 
	113, 113, 9, 10, 11, 12, 13, 14, 
	15, 9, 16, 17, 18, 19, 20, 21, 
	113, 22, 23, 24, 113, 25, 26, 113, 
	28, 29, 30, 31, 6, 113, 5, 6, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 9, 10, 11, 12, 13, 14, 15, 
	9, 16, 17, 18, 19, 20, 21, 113, 
	22, 23, 24, 113, 113, 113, 113, 28, 
	29, 30, 31, 6, 113, 25, 26, 113, 
	26, 113, 1, 117, 117, 117, 1, 117, 
	119, 118, 32, 118, 32, 119, 118, 119, 
	118, 32, 118, 33, 118, 0
};

static const char _use_syllable_machine_trans_targs[] = {
	1, 27, 0, 50, 52, 76, 77, 98, 
	100, 78, 79, 80, 81, 82, 93, 94, 
	95, 101, 96, 90, 91, 92, 85, 86, 
	87, 102, 103, 104, 97, 83, 84, 0, 
	105, 107, 0, 2, 3, 4, 5, 6, 
	7, 8, 19, 20, 21, 24, 22, 16, 
	17, 18, 11, 12, 13, 25, 26, 23, 
	9, 10, 0, 14, 15, 0, 28, 29, 
	30, 31, 32, 33, 34, 45, 46, 47, 
	48, 42, 43, 44, 37, 38, 39, 49, 
	35, 36, 0, 0, 40, 0, 41, 0, 
	51, 0, 53, 54, 55, 56, 57, 58, 
	59, 70, 71, 72, 75, 73, 67, 68, 
	69, 62, 63, 64, 74, 60, 61, 65, 
	66, 0, 88, 89, 99, 0, 0, 106
};

static const char _use_syllable_machine_trans_actions[] = {
	0, 0, 3, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 4, 
	0, 0, 5, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 6, 0, 0, 7, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 8, 9, 0, 10, 0, 11, 
	0, 12, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 13, 0, 0, 0, 14, 15, 0
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
	0, 0, 0, 0
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
	0, 0, 0, 0
};

static const short _use_syllable_machine_eof_trans[] = {
	0, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 62, 62, 84, 62, 62, 
	62, 62, 62, 62, 62, 62, 62, 62, 
	86, 62, 62, 62, 62, 62, 62, 62, 
	62, 84, 88, 90, 62, 62, 84, 62, 
	62, 62, 62, 62, 62, 62, 62, 62, 
	62, 86, 62, 62, 62, 62, 62, 62, 
	62, 62, 84, 62, 114, 114, 114, 114, 
	114, 114, 114, 114, 114, 114, 114, 114, 
	114, 114, 114, 114, 114, 114, 114, 114, 
	114, 114, 114, 114, 114, 114, 114, 114, 
	118, 119, 119, 119
};

static const int use_syllable_machine_start = 0;
static const int use_syllable_machine_first_final = 0;
static const int use_syllable_machine_error = -1;

static const int use_syllable_machine_en_main = 0;


#line 58 "hb-ot-shape-complex-use-machine.rl"



#line 181 "hb-ot-shape-complex-use-machine.rl"


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
					       it (o.it) {}

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
  { unsigned index = (*it).first; if (index < n) it += n - index; else if (index > n) it -= index - n; }
  void operator = (const machine_index_t& o) { *this = (*o.it).first; }
  bool operator == (const machine_index_t& o) const { return (*it).first == (*o.it).first; }
  bool operator != (const machine_index_t& o) const { return !(*this == o); }

  private:
  Iter it;
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
  
#line 651 "hb-ot-shape-complex-use-machine.hh"
	{
	cs = use_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 267 "hb-ot-shape-complex-use-machine.rl"


  unsigned int syllable_serial = 1;
  
#line 664 "hb-ot-shape-complex-use-machine.hh"
	{
	int _slen;
	int _trans;
	const unsigned char *_keys;
	const char *_inds;
	if ( p == pe )
		goto _test_eof;
_resume:
	switch ( _use_syllable_machine_from_state_actions[cs] ) {
	case 2:
#line 1 "NONE"
	{ts = p;}
	break;
#line 678 "hb-ot-shape-complex-use-machine.hh"
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
	case 8:
#line 171 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (use_standard_cluster); }}
	break;
	case 6:
#line 174 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (use_symbol_cluster); }}
	break;
	case 4:
#line 176 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (use_broken_cluster); }}
	break;
	case 3:
#line 177 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (use_non_cluster); }}
	break;
	case 9:
#line 169 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_virama_terminated_cluster); }}
	break;
	case 10:
#line 170 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_sakot_terminated_cluster); }}
	break;
	case 7:
#line 171 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_standard_cluster); }}
	break;
	case 12:
#line 172 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_number_joiner_terminated_cluster); }}
	break;
	case 11:
#line 173 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_numeral_cluster); }}
	break;
	case 5:
#line 174 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_symbol_cluster); }}
	break;
	case 15:
#line 175 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_hieroglyph_cluster); }}
	break;
	case 13:
#line 176 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_broken_cluster); }}
	break;
	case 14:
#line 177 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_non_cluster); }}
	break;
#line 748 "hb-ot-shape-complex-use-machine.hh"
	}

_again:
	switch ( _use_syllable_machine_to_state_actions[cs] ) {
	case 1:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 757 "hb-ot-shape-complex-use-machine.hh"
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

#line 272 "hb-ot-shape-complex-use-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH */
