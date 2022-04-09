
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


#line 100 "hb-ot-shape-complex-use-machine.hh"
static const unsigned char _use_syllable_machine_trans_keys[] = {
	0u, 51u, 11u, 48u, 11u, 48u, 1u, 48u, 23u, 48u, 24u, 47u, 25u, 47u, 26u, 47u, 
	45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 1u, 1u, 24u, 48u, 22u, 48u, 
	23u, 48u, 23u, 48u, 23u, 48u, 12u, 48u, 12u, 48u, 12u, 48u, 12u, 48u, 11u, 48u, 
	1u, 1u, 11u, 48u, 41u, 42u, 42u, 42u, 11u, 48u, 11u, 48u, 1u, 48u, 23u, 48u, 
	24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 
	1u, 1u, 24u, 48u, 22u, 48u, 23u, 48u, 23u, 48u, 23u, 48u, 12u, 48u, 12u, 48u, 
	12u, 48u, 12u, 48u, 11u, 48u, 1u, 1u, 13u, 13u, 4u, 4u, 11u, 48u, 11u, 48u, 
	1u, 48u, 23u, 48u, 24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 
	24u, 48u, 24u, 48u, 1u, 1u, 24u, 48u, 22u, 48u, 23u, 48u, 23u, 48u, 23u, 48u, 
	12u, 48u, 12u, 48u, 12u, 48u, 12u, 48u, 11u, 48u, 1u, 1u, 11u, 48u, 11u, 48u, 
	1u, 48u, 23u, 48u, 24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 
	24u, 48u, 24u, 48u, 1u, 1u, 24u, 48u, 22u, 48u, 23u, 48u, 23u, 48u, 23u, 48u, 
	12u, 48u, 12u, 48u, 12u, 48u, 12u, 48u, 11u, 48u, 1u, 1u, 4u, 4u, 13u, 13u, 
	1u, 48u, 11u, 48u, 41u, 42u, 42u, 42u, 1u, 5u, 50u, 52u, 49u, 52u, 49u, 51u, 
	0
};

static const char _use_syllable_machine_key_spans[] = {
	52, 38, 38, 48, 26, 24, 23, 22, 
	2, 1, 25, 25, 25, 1, 25, 27, 
	26, 26, 26, 37, 37, 37, 37, 38, 
	1, 38, 2, 1, 38, 38, 48, 26, 
	24, 23, 22, 2, 1, 25, 25, 25, 
	1, 25, 27, 26, 26, 26, 37, 37, 
	37, 37, 38, 1, 1, 1, 38, 38, 
	48, 26, 24, 23, 22, 2, 1, 25, 
	25, 25, 1, 25, 27, 26, 26, 26, 
	37, 37, 37, 37, 38, 1, 38, 38, 
	48, 26, 24, 23, 22, 2, 1, 25, 
	25, 25, 1, 25, 27, 26, 26, 26, 
	37, 37, 37, 37, 38, 1, 1, 1, 
	48, 38, 2, 1, 5, 3, 4, 3
};

static const short _use_syllable_machine_index_offsets[] = {
	0, 53, 92, 131, 180, 207, 232, 256, 
	279, 282, 284, 310, 336, 362, 364, 390, 
	418, 445, 472, 499, 537, 575, 613, 651, 
	690, 692, 731, 734, 736, 775, 814, 863, 
	890, 915, 939, 962, 965, 967, 993, 1019, 
	1045, 1047, 1073, 1101, 1128, 1155, 1182, 1220, 
	1258, 1296, 1334, 1373, 1375, 1377, 1379, 1418, 
	1457, 1506, 1533, 1558, 1582, 1605, 1608, 1610, 
	1636, 1662, 1688, 1690, 1716, 1744, 1771, 1798, 
	1825, 1863, 1901, 1939, 1977, 2016, 2018, 2057, 
	2096, 2145, 2172, 2197, 2221, 2244, 2247, 2249, 
	2275, 2301, 2327, 2329, 2355, 2383, 2410, 2437, 
	2464, 2502, 2540, 2578, 2616, 2655, 2657, 2659, 
	2661, 2710, 2749, 2752, 2754, 2760, 2764, 2769
};

static const char _use_syllable_machine_indicies[] = {
	0, 1, 2, 2, 3, 4, 2, 2, 
	2, 2, 2, 5, 6, 7, 2, 2, 
	2, 2, 8, 2, 2, 2, 9, 10, 
	11, 12, 13, 14, 15, 16, 17, 18, 
	19, 20, 21, 22, 2, 23, 24, 25, 
	2, 26, 27, 28, 29, 30, 31, 32, 
	29, 33, 2, 34, 2, 36, 37, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	38, 39, 40, 41, 42, 43, 44, 45, 
	46, 47, 48, 49, 50, 51, 35, 52, 
	53, 54, 35, 55, 56, 35, 57, 58, 
	59, 60, 57, 35, 36, 37, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 38, 
	39, 40, 41, 42, 43, 44, 45, 46, 
	48, 48, 49, 50, 51, 35, 52, 53, 
	54, 35, 35, 35, 35, 57, 58, 59, 
	60, 57, 35, 36, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 39, 40, 41, 42, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 52, 
	53, 54, 35, 35, 35, 35, 35, 58, 
	59, 60, 61, 35, 39, 40, 41, 42, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 52, 53, 54, 35, 35, 35, 
	35, 35, 58, 59, 60, 61, 35, 40, 
	41, 42, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 58, 59, 60, 35, 
	41, 42, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 58, 59, 60, 35, 
	42, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 58, 59, 60, 35, 58, 
	59, 35, 59, 35, 40, 41, 42, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 52, 53, 54, 35, 35, 35, 35, 
	35, 58, 59, 60, 61, 35, 40, 41, 
	42, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 53, 54, 35, 35, 
	35, 35, 35, 58, 59, 60, 61, 35, 
	40, 41, 42, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 54, 
	35, 35, 35, 35, 35, 58, 59, 60, 
	61, 35, 62, 35, 40, 41, 42, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 58, 59, 60, 61, 35, 38, 39, 
	40, 41, 42, 35, 35, 35, 35, 35, 
	35, 49, 50, 51, 35, 52, 53, 54, 
	35, 35, 35, 35, 35, 58, 59, 60, 
	61, 35, 39, 40, 41, 42, 35, 35, 
	35, 35, 35, 35, 49, 50, 51, 35, 
	52, 53, 54, 35, 35, 35, 35, 35, 
	58, 59, 60, 61, 35, 39, 40, 41, 
	42, 35, 35, 35, 35, 35, 35, 35, 
	50, 51, 35, 52, 53, 54, 35, 35, 
	35, 35, 35, 58, 59, 60, 61, 35, 
	39, 40, 41, 42, 35, 35, 35, 35, 
	35, 35, 35, 35, 51, 35, 52, 53, 
	54, 35, 35, 35, 35, 35, 58, 59, 
	60, 61, 35, 39, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 38, 39, 40, 
	41, 42, 35, 44, 45, 35, 35, 35, 
	49, 50, 51, 35, 52, 53, 54, 35, 
	35, 35, 35, 35, 58, 59, 60, 61, 
	35, 39, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 38, 39, 40, 41, 42, 
	35, 35, 45, 35, 35, 35, 49, 50, 
	51, 35, 52, 53, 54, 35, 35, 35, 
	35, 35, 58, 59, 60, 61, 35, 39, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 38, 39, 40, 41, 42, 35, 35, 
	35, 35, 35, 35, 49, 50, 51, 35, 
	52, 53, 54, 35, 35, 35, 35, 35, 
	58, 59, 60, 61, 35, 39, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 38, 
	39, 40, 41, 42, 43, 44, 45, 35, 
	35, 35, 49, 50, 51, 35, 52, 53, 
	54, 35, 35, 35, 35, 35, 58, 59, 
	60, 61, 35, 36, 37, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 38, 39, 
	40, 41, 42, 43, 44, 45, 46, 35, 
	48, 49, 50, 51, 35, 52, 53, 54, 
	35, 35, 35, 35, 57, 58, 59, 60, 
	57, 35, 36, 35, 36, 37, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 38, 
	39, 40, 41, 42, 43, 44, 45, 46, 
	47, 48, 49, 50, 51, 35, 52, 53, 
	54, 35, 35, 35, 35, 57, 58, 59, 
	60, 57, 35, 55, 56, 35, 56, 35, 
	64, 65, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 66, 67, 68, 69, 70, 
	71, 72, 73, 74, 1, 75, 76, 77, 
	78, 63, 79, 80, 81, 63, 63, 63, 
	63, 82, 83, 84, 85, 86, 63, 64, 
	65, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 66, 67, 68, 69, 70, 71, 
	72, 73, 74, 75, 75, 76, 77, 78, 
	63, 79, 80, 81, 63, 63, 63, 63, 
	82, 83, 84, 85, 86, 63, 64, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 67, 68, 69, 70, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 79, 80, 81, 63, 63, 63, 
	63, 63, 83, 84, 85, 87, 63, 67, 
	68, 69, 70, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 79, 80, 81, 
	63, 63, 63, 63, 63, 83, 84, 85, 
	87, 63, 68, 69, 70, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 83, 
	84, 85, 63, 69, 70, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 83, 
	84, 85, 63, 70, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 83, 84, 
	85, 63, 83, 84, 63, 84, 63, 68, 
	69, 70, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 79, 80, 81, 63, 
	63, 63, 63, 63, 83, 84, 85, 87, 
	63, 68, 69, 70, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 80, 
	81, 63, 63, 63, 63, 63, 83, 84, 
	85, 87, 63, 68, 69, 70, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 81, 63, 63, 63, 63, 63, 
	83, 84, 85, 87, 63, 89, 88, 68, 
	69, 70, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 83, 84, 85, 87, 
	63, 66, 67, 68, 69, 70, 63, 63, 
	63, 63, 63, 63, 76, 77, 78, 63, 
	79, 80, 81, 63, 63, 63, 63, 63, 
	83, 84, 85, 87, 63, 67, 68, 69, 
	70, 63, 63, 63, 63, 63, 63, 76, 
	77, 78, 63, 79, 80, 81, 63, 63, 
	63, 63, 63, 83, 84, 85, 87, 63, 
	67, 68, 69, 70, 63, 63, 63, 63, 
	63, 63, 63, 77, 78, 63, 79, 80, 
	81, 63, 63, 63, 63, 63, 83, 84, 
	85, 87, 63, 67, 68, 69, 70, 63, 
	63, 63, 63, 63, 63, 63, 63, 78, 
	63, 79, 80, 81, 63, 63, 63, 63, 
	63, 83, 84, 85, 87, 63, 67, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	66, 67, 68, 69, 70, 63, 72, 73, 
	63, 63, 63, 76, 77, 78, 63, 79, 
	80, 81, 63, 63, 63, 63, 63, 83, 
	84, 85, 87, 63, 67, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 66, 67, 
	68, 69, 70, 63, 63, 73, 63, 63, 
	63, 76, 77, 78, 63, 79, 80, 81, 
	63, 63, 63, 63, 63, 83, 84, 85, 
	87, 63, 67, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 66, 67, 68, 69, 
	70, 63, 63, 63, 63, 63, 63, 76, 
	77, 78, 63, 79, 80, 81, 63, 63, 
	63, 63, 63, 83, 84, 85, 87, 63, 
	67, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 66, 67, 68, 69, 70, 71, 
	72, 73, 63, 63, 63, 76, 77, 78, 
	63, 79, 80, 81, 63, 63, 63, 63, 
	63, 83, 84, 85, 87, 63, 64, 65, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 66, 67, 68, 69, 70, 71, 72, 
	73, 74, 63, 75, 76, 77, 78, 63, 
	79, 80, 81, 63, 63, 63, 63, 82, 
	83, 84, 85, 86, 63, 64, 90, 92, 
	91, 3, 93, 94, 95, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 96, 97, 
	98, 99, 100, 101, 102, 103, 104, 105, 
	106, 107, 108, 109, 63, 110, 111, 112, 
	63, 55, 56, 63, 113, 114, 115, 85, 
	116, 63, 94, 95, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 96, 97, 98, 
	99, 100, 101, 102, 103, 104, 106, 106, 
	107, 108, 109, 63, 110, 111, 112, 63, 
	63, 63, 63, 113, 114, 115, 85, 116, 
	63, 94, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 97, 
	98, 99, 100, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 110, 111, 112, 
	63, 63, 63, 63, 63, 114, 115, 85, 
	117, 63, 97, 98, 99, 100, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	110, 111, 112, 63, 63, 63, 63, 63, 
	114, 115, 85, 117, 63, 98, 99, 100, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 114, 115, 85, 63, 99, 100, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 114, 115, 85, 63, 100, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 114, 115, 85, 63, 114, 115, 63, 
	115, 63, 98, 99, 100, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 110, 
	111, 112, 63, 63, 63, 63, 63, 114, 
	115, 85, 117, 63, 98, 99, 100, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 111, 112, 63, 63, 63, 63, 
	63, 114, 115, 85, 117, 63, 98, 99, 
	100, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 112, 63, 63, 
	63, 63, 63, 114, 115, 85, 117, 63, 
	118, 88, 98, 99, 100, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 114, 
	115, 85, 117, 63, 96, 97, 98, 99, 
	100, 63, 63, 63, 63, 63, 63, 107, 
	108, 109, 63, 110, 111, 112, 63, 63, 
	63, 63, 63, 114, 115, 85, 117, 63, 
	97, 98, 99, 100, 63, 63, 63, 63, 
	63, 63, 107, 108, 109, 63, 110, 111, 
	112, 63, 63, 63, 63, 63, 114, 115, 
	85, 117, 63, 97, 98, 99, 100, 63, 
	63, 63, 63, 63, 63, 63, 108, 109, 
	63, 110, 111, 112, 63, 63, 63, 63, 
	63, 114, 115, 85, 117, 63, 97, 98, 
	99, 100, 63, 63, 63, 63, 63, 63, 
	63, 63, 109, 63, 110, 111, 112, 63, 
	63, 63, 63, 63, 114, 115, 85, 117, 
	63, 97, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 96, 97, 98, 99, 100, 
	63, 102, 103, 63, 63, 63, 107, 108, 
	109, 63, 110, 111, 112, 63, 63, 63, 
	63, 63, 114, 115, 85, 117, 63, 97, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 96, 97, 98, 99, 100, 63, 63, 
	103, 63, 63, 63, 107, 108, 109, 63, 
	110, 111, 112, 63, 63, 63, 63, 63, 
	114, 115, 85, 117, 63, 97, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 96, 
	97, 98, 99, 100, 63, 63, 63, 63, 
	63, 63, 107, 108, 109, 63, 110, 111, 
	112, 63, 63, 63, 63, 63, 114, 115, 
	85, 117, 63, 97, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 96, 97, 98, 
	99, 100, 101, 102, 103, 63, 63, 63, 
	107, 108, 109, 63, 110, 111, 112, 63, 
	63, 63, 63, 63, 114, 115, 85, 117, 
	63, 94, 95, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 96, 97, 98, 99, 
	100, 101, 102, 103, 104, 63, 106, 107, 
	108, 109, 63, 110, 111, 112, 63, 63, 
	63, 63, 113, 114, 115, 85, 116, 63, 
	94, 90, 94, 95, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 96, 97, 98, 
	99, 100, 101, 102, 103, 104, 105, 106, 
	107, 108, 109, 63, 110, 111, 112, 63, 
	63, 63, 63, 113, 114, 115, 85, 116, 
	63, 5, 6, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 9, 10, 11, 12, 
	13, 14, 15, 16, 17, 19, 19, 20, 
	21, 22, 119, 23, 24, 25, 119, 119, 
	119, 119, 29, 30, 31, 32, 29, 119, 
	5, 119, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 10, 11, 
	12, 13, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 23, 24, 25, 119, 
	119, 119, 119, 119, 30, 31, 32, 120, 
	119, 10, 11, 12, 13, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 119, 23, 
	24, 25, 119, 119, 119, 119, 119, 30, 
	31, 32, 120, 119, 11, 12, 13, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 30, 31, 32, 119, 12, 13, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 30, 31, 32, 119, 13, 119, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	30, 31, 32, 119, 30, 31, 119, 31, 
	119, 11, 12, 13, 119, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 23, 24, 
	25, 119, 119, 119, 119, 119, 30, 31, 
	32, 120, 119, 11, 12, 13, 119, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 24, 25, 119, 119, 119, 119, 119, 
	30, 31, 32, 120, 119, 11, 12, 13, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 25, 119, 119, 119, 
	119, 119, 30, 31, 32, 120, 119, 121, 
	119, 11, 12, 13, 119, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 30, 31, 
	32, 120, 119, 9, 10, 11, 12, 13, 
	119, 119, 119, 119, 119, 119, 20, 21, 
	22, 119, 23, 24, 25, 119, 119, 119, 
	119, 119, 30, 31, 32, 120, 119, 10, 
	11, 12, 13, 119, 119, 119, 119, 119, 
	119, 20, 21, 22, 119, 23, 24, 25, 
	119, 119, 119, 119, 119, 30, 31, 32, 
	120, 119, 10, 11, 12, 13, 119, 119, 
	119, 119, 119, 119, 119, 21, 22, 119, 
	23, 24, 25, 119, 119, 119, 119, 119, 
	30, 31, 32, 120, 119, 10, 11, 12, 
	13, 119, 119, 119, 119, 119, 119, 119, 
	119, 22, 119, 23, 24, 25, 119, 119, 
	119, 119, 119, 30, 31, 32, 120, 119, 
	10, 119, 119, 119, 119, 119, 119, 119, 
	119, 119, 9, 10, 11, 12, 13, 119, 
	15, 16, 119, 119, 119, 20, 21, 22, 
	119, 23, 24, 25, 119, 119, 119, 119, 
	119, 30, 31, 32, 120, 119, 10, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	9, 10, 11, 12, 13, 119, 119, 16, 
	119, 119, 119, 20, 21, 22, 119, 23, 
	24, 25, 119, 119, 119, 119, 119, 30, 
	31, 32, 120, 119, 10, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 9, 10, 
	11, 12, 13, 119, 119, 119, 119, 119, 
	119, 20, 21, 22, 119, 23, 24, 25, 
	119, 119, 119, 119, 119, 30, 31, 32, 
	120, 119, 10, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 9, 10, 11, 12, 
	13, 14, 15, 16, 119, 119, 119, 20, 
	21, 22, 119, 23, 24, 25, 119, 119, 
	119, 119, 119, 30, 31, 32, 120, 119, 
	5, 6, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 9, 10, 11, 12, 13, 
	14, 15, 16, 17, 119, 19, 20, 21, 
	22, 119, 23, 24, 25, 119, 119, 119, 
	119, 29, 30, 31, 32, 29, 119, 5, 
	119, 122, 119, 7, 119, 1, 119, 119, 
	119, 1, 119, 119, 119, 119, 119, 5, 
	6, 7, 119, 119, 119, 119, 119, 119, 
	119, 119, 9, 10, 11, 12, 13, 14, 
	15, 16, 17, 18, 19, 20, 21, 22, 
	119, 23, 24, 25, 119, 26, 27, 119, 
	29, 30, 31, 32, 29, 119, 5, 6, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 9, 10, 11, 12, 13, 14, 15, 
	16, 17, 18, 19, 20, 21, 22, 119, 
	23, 24, 25, 119, 119, 119, 119, 29, 
	30, 31, 32, 29, 119, 26, 27, 119, 
	27, 119, 1, 123, 123, 123, 1, 123, 
	125, 124, 33, 124, 33, 125, 124, 125, 
	124, 33, 124, 34, 124, 0
};

static const char _use_syllable_machine_trans_targs[] = {
	1, 28, 0, 52, 54, 79, 80, 102, 
	104, 92, 81, 82, 83, 84, 96, 97, 
	98, 99, 105, 100, 93, 94, 95, 87, 
	88, 89, 106, 107, 108, 101, 85, 86, 
	0, 109, 111, 0, 2, 3, 15, 4, 
	5, 6, 7, 19, 20, 21, 22, 25, 
	23, 16, 17, 18, 10, 11, 12, 26, 
	27, 24, 8, 9, 0, 13, 14, 0, 
	29, 30, 42, 31, 32, 33, 34, 46, 
	47, 48, 49, 50, 43, 44, 45, 37, 
	38, 39, 51, 35, 36, 0, 51, 40, 
	0, 41, 0, 0, 53, 0, 55, 56, 
	68, 57, 58, 59, 60, 72, 73, 74, 
	75, 78, 76, 69, 70, 71, 63, 64, 
	65, 77, 61, 62, 77, 66, 67, 0, 
	90, 91, 103, 0, 0, 110
};

static const char _use_syllable_machine_trans_actions[] = {
	0, 0, 3, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	4, 0, 0, 5, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 6, 0, 0, 7, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 8, 0, 0, 9, 10, 0, 
	11, 0, 12, 13, 0, 14, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 8, 0, 0, 10, 0, 0, 15, 
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
	0, 0, 0, 0, 0, 0, 0, 0
};

static const short _use_syllable_machine_eof_trans[] = {
	0, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	89, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 91, 92, 94, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 89, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 91, 64, 120, 
	120, 120, 120, 120, 120, 120, 120, 120, 
	120, 120, 120, 120, 120, 120, 120, 120, 
	120, 120, 120, 120, 120, 120, 120, 120, 
	120, 120, 120, 120, 124, 125, 125, 125
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
  
#line 702 "hb-ot-shape-complex-use-machine.hh"
	{
	cs = use_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 281 "hb-ot-shape-complex-use-machine.rl"


  unsigned int syllable_serial = 1;
  
#line 715 "hb-ot-shape-complex-use-machine.hh"
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
#line 729 "hb-ot-shape-complex-use-machine.hh"
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
	case 11:
#line 170 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_sakot_terminated_cluster); }}
	break;
	case 7:
#line 171 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_standard_cluster); }}
	break;
	case 14:
#line 172 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_number_joiner_terminated_cluster); }}
	break;
	case 13:
#line 173 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_numeral_cluster); }}
	break;
	case 5:
#line 174 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_symbol_cluster); }}
	break;
	case 17:
#line 175 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_hieroglyph_cluster); }}
	break;
	case 15:
#line 176 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_broken_cluster); }}
	break;
	case 16:
#line 177 "hb-ot-shape-complex-use-machine.rl"
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
#line 169 "hb-ot-shape-complex-use-machine.rl"
	{act = 1;}
	break;
	case 10:
#line 1 "NONE"
	{te = p+1;}
#line 170 "hb-ot-shape-complex-use-machine.rl"
	{act = 2;}
	break;
#line 819 "hb-ot-shape-complex-use-machine.hh"
	}

_again:
	switch ( _use_syllable_machine_to_state_actions[cs] ) {
	case 1:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 828 "hb-ot-shape-complex-use-machine.hh"
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

#line 286 "hb-ot-shape-complex-use-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH */
