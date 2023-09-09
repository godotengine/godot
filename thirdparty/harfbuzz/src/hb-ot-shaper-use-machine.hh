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
	0u, 39u, 5u, 39u, 5u, 39u, 1u, 39u,
	8u, 34u, 8u, 33u, 8u, 33u, 8u, 33u,
	8u, 32u, 8u, 32u, 8u, 8u, 8u, 34u,
	8u, 34u, 8u, 34u, 1u, 8u, 8u, 34u,
	8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u,
	6u, 39u, 8u, 39u, 6u, 39u, 6u, 39u,
	6u, 39u, 5u, 39u, 1u, 8u, 1u, 34u,
	8u, 28u, 8u, 28u, 5u, 39u, 1u, 39u,
	8u, 34u, 8u, 33u, 8u, 33u, 8u, 33u,
	8u, 32u, 8u, 32u, 8u, 8u, 8u, 34u,
	8u, 34u, 8u, 34u, 1u, 8u, 8u, 34u,
	8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u,
	6u, 39u, 8u, 39u, 6u, 39u, 6u, 39u,
	6u, 39u, 5u, 39u, 1u, 8u, 1u, 8u,
	1u, 34u, 7u, 8u, 3u, 8u, 5u, 39u,
	5u, 39u, 1u, 39u, 8u, 34u, 8u, 33u,
	8u, 33u, 8u, 33u, 8u, 32u, 8u, 32u,
	8u, 8u, 8u, 34u, 8u, 34u, 8u, 34u,
	1u, 8u, 8u, 34u, 8u, 39u, 8u, 39u,
	8u, 39u, 8u, 39u, 6u, 39u, 8u, 39u,
	6u, 39u, 6u, 39u, 6u, 39u, 5u, 39u,
	1u, 8u, 1u, 8u, 1u, 34u, 5u, 39u,
	1u, 39u, 8u, 34u, 8u, 33u, 8u, 33u,
	8u, 33u, 8u, 32u, 8u, 32u, 8u, 8u,
	8u, 34u, 8u, 34u, 8u, 34u, 1u, 8u,
	8u, 34u, 8u, 39u, 8u, 39u, 8u, 39u,
	8u, 39u, 6u, 39u, 8u, 39u, 6u, 39u,
	6u, 39u, 6u, 39u, 5u, 39u, 1u, 8u,
	1u, 34u, 3u, 8u, 7u, 8u, 1u, 39u,
	8u, 28u, 8u, 28u, 1u, 4u, 8u, 38u,
	8u, 38u, 8u, 37u, 0u
};

static const signed char _use_syllable_machine_char_class[] = {
	0, 1, 2, 2, 3, 4, 2, 2,
	2, 2, 2, 5, 6, 7, 8, 2,
	2, 2, 9, 2, 2, 2, 10, 11,
	12, 13, 14, 15, 16, 17, 18, 19,
	20, 21, 22, 23, 2, 24, 25, 26,
	2, 27, 28, 29, 30, 31, 32, 33,
	34, 35, 36, 37, 38, 39, 0
};

static const short _use_syllable_machine_index_offsets[] = {
	0, 40, 75, 110, 149, 176, 202, 228,
	254, 279, 304, 305, 332, 359, 386, 394,
	421, 453, 485, 517, 549, 583, 615, 649,
	683, 717, 752, 760, 794, 815, 836, 871,
	910, 937, 963, 989, 1015, 1040, 1065, 1066,
	1093, 1120, 1147, 1155, 1182, 1214, 1246, 1278,
	1310, 1344, 1376, 1410, 1444, 1478, 1513, 1521,
	1529, 1563, 1565, 1571, 1606, 1641, 1680, 1707,
	1733, 1759, 1785, 1810, 1835, 1836, 1863, 1890,
	1917, 1925, 1952, 1984, 2016, 2048, 2080, 2114,
	2146, 2180, 2214, 2248, 2283, 2291, 2299, 2333,
	2368, 2407, 2434, 2460, 2486, 2512, 2537, 2562,
	2563, 2590, 2617, 2644, 2652, 2679, 2711, 2743,
	2775, 2807, 2841, 2873, 2907, 2941, 2975, 3010,
	3018, 3052, 3058, 3060, 3099, 3120, 3141, 3145,
	3176, 3207, 0
};

static const short _use_syllable_machine_indicies[] = {
	1, 2, 3, 4, 5, 6, 7, 8,
	9, 10, 11, 12, 13, 14, 15, 16,
	17, 18, 19, 6, 20, 21, 22, 23,
	24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 30, 34, 3, 35, 3, 36,
	38, 39, 37, 40, 37, 41, 42, 43,
	44, 45, 46, 47, 48, 49, 38, 50,
	51, 52, 53, 54, 55, 56, 57, 58,
	37, 59, 60, 61, 62, 59, 37, 37,
	37, 37, 63, 38, 39, 37, 40, 37,
	41, 42, 43, 44, 45, 46, 47, 48,
	49, 38, 50, 51, 52, 53, 54, 55,
	56, 37, 37, 37, 59, 60, 61, 62,
	59, 37, 37, 37, 37, 63, 38, 37,
	37, 37, 37, 37, 37, 40, 37, 37,
	42, 43, 44, 45, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 54, 55, 56,
	37, 37, 37, 37, 60, 61, 62, 64,
	37, 37, 37, 37, 42, 40, 37, 37,
	42, 43, 44, 45, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 54, 55, 56,
	37, 37, 37, 37, 60, 61, 62, 64,
	40, 37, 37, 37, 43, 44, 45, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 60,
	61, 62, 40, 37, 37, 37, 37, 44,
	45, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 60, 61, 62, 40, 37, 37, 37,
	37, 37, 45, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 60, 61, 62, 40, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 60, 61, 40,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 61,
	40, 40, 37, 37, 37, 43, 44, 45,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 54, 55, 56, 37, 37, 37, 37,
	60, 61, 62, 64, 40, 37, 37, 37,
	43, 44, 45, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 55, 56, 37,
	37, 37, 37, 60, 61, 62, 64, 40,
	37, 37, 37, 43, 44, 45, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 56, 37, 37, 37, 37, 60, 61,
	62, 64, 65, 37, 37, 37, 37, 37,
	37, 40, 40, 37, 37, 37, 43, 44,
	45, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 60, 61, 62, 64, 40, 37, 41,
	42, 43, 44, 45, 37, 37, 37, 37,
	37, 37, 51, 52, 53, 54, 55, 56,
	37, 37, 37, 37, 60, 61, 62, 64,
	37, 37, 37, 37, 42, 40, 37, 37,
	42, 43, 44, 45, 37, 37, 37, 37,
	37, 37, 51, 52, 53, 54, 55, 56,
	37, 37, 37, 37, 60, 61, 62, 64,
	37, 37, 37, 37, 42, 40, 37, 37,
	42, 43, 44, 45, 37, 37, 37, 37,
	37, 37, 37, 52, 53, 54, 55, 56,
	37, 37, 37, 37, 60, 61, 62, 64,
	37, 37, 37, 37, 42, 40, 37, 37,
	42, 43, 44, 45, 37, 37, 37, 37,
	37, 37, 37, 37, 53, 54, 55, 56,
	37, 37, 37, 37, 60, 61, 62, 64,
	37, 37, 37, 37, 42, 66, 37, 40,
	37, 41, 42, 43, 44, 45, 37, 47,
	48, 37, 37, 37, 51, 52, 53, 54,
	55, 56, 37, 37, 37, 37, 60, 61,
	62, 64, 37, 37, 37, 37, 42, 40,
	37, 37, 42, 43, 44, 45, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 54,
	55, 56, 37, 37, 37, 37, 60, 61,
	62, 64, 37, 37, 37, 37, 42, 66,
	37, 40, 37, 41, 42, 43, 44, 45,
	37, 37, 48, 37, 37, 37, 51, 52,
	53, 54, 55, 56, 37, 37, 37, 37,
	60, 61, 62, 64, 37, 37, 37, 37,
	42, 66, 37, 40, 37, 41, 42, 43,
	44, 45, 37, 37, 37, 37, 37, 37,
	51, 52, 53, 54, 55, 56, 37, 37,
	37, 37, 60, 61, 62, 64, 37, 37,
	37, 37, 42, 66, 37, 40, 37, 41,
	42, 43, 44, 45, 46, 47, 48, 37,
	37, 37, 51, 52, 53, 54, 55, 56,
	37, 37, 37, 37, 60, 61, 62, 64,
	37, 37, 37, 37, 42, 38, 39, 37,
	40, 37, 41, 42, 43, 44, 45, 46,
	47, 48, 49, 37, 50, 51, 52, 53,
	54, 55, 56, 37, 37, 37, 59, 60,
	61, 62, 59, 37, 37, 37, 37, 63,
	38, 37, 37, 37, 37, 37, 37, 40,
	38, 37, 37, 37, 37, 37, 37, 40,
	37, 37, 42, 43, 44, 45, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 54,
	55, 56, 37, 37, 37, 37, 60, 61,
	62, 64, 40, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 57, 58, 40,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 58, 2, 68, 67, 69,
	67, 70, 71, 72, 73, 74, 75, 76,
	77, 78, 2, 79, 80, 81, 82, 83,
	84, 85, 67, 67, 67, 86, 87, 88,
	89, 90, 67, 67, 67, 67, 91, 2,
	67, 67, 67, 67, 67, 67, 69, 67,
	67, 71, 72, 73, 74, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 83, 84,
	85, 67, 67, 67, 67, 87, 88, 89,
	92, 67, 67, 67, 67, 71, 69, 67,
	67, 71, 72, 73, 74, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 83, 84,
	85, 67, 67, 67, 67, 87, 88, 89,
	92, 69, 67, 67, 67, 72, 73, 74,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	87, 88, 89, 69, 67, 67, 67, 67,
	73, 74, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 87, 88, 89, 69, 67, 67,
	67, 67, 67, 74, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 87, 88, 89, 69,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 87, 88,
	69, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	88, 69, 69, 67, 67, 67, 72, 73,
	74, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 83, 84, 85, 67, 67, 67,
	67, 87, 88, 89, 92, 69, 67, 67,
	67, 72, 73, 74, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 84, 85,
	67, 67, 67, 67, 87, 88, 89, 92,
	69, 67, 67, 67, 72, 73, 74, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 85, 67, 67, 67, 67, 87,
	88, 89, 92, 94, 93, 93, 93, 93,
	93, 93, 95, 69, 67, 67, 67, 72,
	73, 74, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 87, 88, 89, 92, 69, 67,
	70, 71, 72, 73, 74, 67, 67, 67,
	67, 67, 67, 80, 81, 82, 83, 84,
	85, 67, 67, 67, 67, 87, 88, 89,
	92, 67, 67, 67, 67, 71, 69, 67,
	67, 71, 72, 73, 74, 67, 67, 67,
	67, 67, 67, 80, 81, 82, 83, 84,
	85, 67, 67, 67, 67, 87, 88, 89,
	92, 67, 67, 67, 67, 71, 69, 67,
	67, 71, 72, 73, 74, 67, 67, 67,
	67, 67, 67, 67, 81, 82, 83, 84,
	85, 67, 67, 67, 67, 87, 88, 89,
	92, 67, 67, 67, 67, 71, 69, 67,
	67, 71, 72, 73, 74, 67, 67, 67,
	67, 67, 67, 67, 67, 82, 83, 84,
	85, 67, 67, 67, 67, 87, 88, 89,
	92, 67, 67, 67, 67, 71, 96, 67,
	69, 67, 70, 71, 72, 73, 74, 67,
	76, 77, 67, 67, 67, 80, 81, 82,
	83, 84, 85, 67, 67, 67, 67, 87,
	88, 89, 92, 67, 67, 67, 67, 71,
	69, 67, 67, 71, 72, 73, 74, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	83, 84, 85, 67, 67, 67, 67, 87,
	88, 89, 92, 67, 67, 67, 67, 71,
	96, 67, 69, 67, 70, 71, 72, 73,
	74, 67, 67, 77, 67, 67, 67, 80,
	81, 82, 83, 84, 85, 67, 67, 67,
	67, 87, 88, 89, 92, 67, 67, 67,
	67, 71, 96, 67, 69, 67, 70, 71,
	72, 73, 74, 67, 67, 67, 67, 67,
	67, 80, 81, 82, 83, 84, 85, 67,
	67, 67, 67, 87, 88, 89, 92, 67,
	67, 67, 67, 71, 96, 67, 69, 67,
	70, 71, 72, 73, 74, 75, 76, 77,
	67, 67, 67, 80, 81, 82, 83, 84,
	85, 67, 67, 67, 67, 87, 88, 89,
	92, 67, 67, 67, 67, 71, 2, 68,
	67, 69, 67, 70, 71, 72, 73, 74,
	75, 76, 77, 78, 67, 79, 80, 81,
	82, 83, 84, 85, 67, 67, 67, 86,
	87, 88, 89, 90, 67, 67, 67, 67,
	91, 2, 97, 97, 97, 97, 97, 97,
	98, 2, 93, 93, 93, 93, 93, 93,
	95, 2, 67, 67, 67, 67, 67, 67,
	69, 67, 67, 71, 72, 73, 74, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	83, 84, 85, 67, 67, 67, 67, 87,
	88, 89, 92, 100, 101, 4, 102, 102,
	102, 102, 103, 104, 105, 67, 69, 67,
	106, 107, 108, 109, 110, 111, 112, 113,
	114, 104, 115, 116, 117, 118, 119, 120,
	121, 57, 58, 67, 122, 123, 124, 125,
	126, 67, 67, 67, 67, 127, 104, 105,
	67, 69, 67, 106, 107, 108, 109, 110,
	111, 112, 113, 114, 104, 115, 116, 117,
	118, 119, 120, 121, 67, 67, 67, 122,
	123, 124, 125, 126, 67, 67, 67, 67,
	127, 104, 67, 67, 67, 67, 67, 67,
	69, 67, 67, 107, 108, 109, 110, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	119, 120, 121, 67, 67, 67, 67, 123,
	124, 125, 128, 67, 67, 67, 67, 107,
	69, 67, 67, 107, 108, 109, 110, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	119, 120, 121, 67, 67, 67, 67, 123,
	124, 125, 128, 69, 67, 67, 67, 108,
	109, 110, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 123, 124, 125, 69, 67, 67,
	67, 67, 109, 110, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 123, 124, 125, 69,
	67, 67, 67, 67, 67, 110, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 123, 124,
	125, 69, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	123, 124, 69, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 124, 69, 69, 67, 67, 67,
	108, 109, 110, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 119, 120, 121, 67,
	67, 67, 67, 123, 124, 125, 128, 69,
	67, 67, 67, 108, 109, 110, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	120, 121, 67, 67, 67, 67, 123, 124,
	125, 128, 69, 67, 67, 67, 108, 109,
	110, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 121, 67, 67, 67,
	67, 123, 124, 125, 128, 129, 93, 93,
	93, 93, 93, 93, 95, 69, 67, 67,
	67, 108, 109, 110, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 123, 124, 125, 128,
	69, 67, 106, 107, 108, 109, 110, 67,
	67, 67, 67, 67, 67, 116, 117, 118,
	119, 120, 121, 67, 67, 67, 67, 123,
	124, 125, 128, 67, 67, 67, 67, 107,
	69, 67, 67, 107, 108, 109, 110, 67,
	67, 67, 67, 67, 67, 116, 117, 118,
	119, 120, 121, 67, 67, 67, 67, 123,
	124, 125, 128, 67, 67, 67, 67, 107,
	69, 67, 67, 107, 108, 109, 110, 67,
	67, 67, 67, 67, 67, 67, 117, 118,
	119, 120, 121, 67, 67, 67, 67, 123,
	124, 125, 128, 67, 67, 67, 67, 107,
	69, 67, 67, 107, 108, 109, 110, 67,
	67, 67, 67, 67, 67, 67, 67, 118,
	119, 120, 121, 67, 67, 67, 67, 123,
	124, 125, 128, 67, 67, 67, 67, 107,
	130, 67, 69, 67, 106, 107, 108, 109,
	110, 67, 112, 113, 67, 67, 67, 116,
	117, 118, 119, 120, 121, 67, 67, 67,
	67, 123, 124, 125, 128, 67, 67, 67,
	67, 107, 69, 67, 67, 107, 108, 109,
	110, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 119, 120, 121, 67, 67, 67,
	67, 123, 124, 125, 128, 67, 67, 67,
	67, 107, 130, 67, 69, 67, 106, 107,
	108, 109, 110, 67, 67, 113, 67, 67,
	67, 116, 117, 118, 119, 120, 121, 67,
	67, 67, 67, 123, 124, 125, 128, 67,
	67, 67, 67, 107, 130, 67, 69, 67,
	106, 107, 108, 109, 110, 67, 67, 67,
	67, 67, 67, 116, 117, 118, 119, 120,
	121, 67, 67, 67, 67, 123, 124, 125,
	128, 67, 67, 67, 67, 107, 130, 67,
	69, 67, 106, 107, 108, 109, 110, 111,
	112, 113, 67, 67, 67, 116, 117, 118,
	119, 120, 121, 67, 67, 67, 67, 123,
	124, 125, 128, 67, 67, 67, 67, 107,
	104, 105, 67, 69, 67, 106, 107, 108,
	109, 110, 111, 112, 113, 114, 67, 115,
	116, 117, 118, 119, 120, 121, 67, 67,
	67, 122, 123, 124, 125, 126, 67, 67,
	67, 67, 127, 104, 97, 97, 97, 97,
	97, 97, 98, 104, 93, 93, 93, 93,
	93, 93, 95, 104, 67, 67, 67, 67,
	67, 67, 69, 67, 67, 107, 108, 109,
	110, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 119, 120, 121, 67, 67, 67,
	67, 123, 124, 125, 128, 6, 7, 131,
	9, 131, 11, 12, 13, 14, 15, 16,
	17, 18, 19, 6, 20, 21, 22, 23,
	24, 25, 26, 131, 131, 131, 30, 31,
	32, 33, 30, 131, 131, 131, 131, 36,
	6, 131, 131, 131, 131, 131, 131, 9,
	131, 131, 12, 13, 14, 15, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 24,
	25, 26, 131, 131, 131, 131, 31, 32,
	33, 132, 131, 131, 131, 131, 12, 9,
	131, 131, 12, 13, 14, 15, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 24,
	25, 26, 131, 131, 131, 131, 31, 32,
	33, 132, 9, 131, 131, 131, 13, 14,
	15, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 31, 32, 33, 9, 131, 131, 131,
	131, 14, 15, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 31, 32, 33, 9, 131,
	131, 131, 131, 131, 15, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 31, 32, 33,
	9, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 31,
	32, 9, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 32, 9, 9, 131, 131, 131, 13,
	14, 15, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 24, 25, 26, 131, 131,
	131, 131, 31, 32, 33, 132, 9, 131,
	131, 131, 13, 14, 15, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 25,
	26, 131, 131, 131, 131, 31, 32, 33,
	132, 9, 131, 131, 131, 13, 14, 15,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 26, 131, 131, 131, 131,
	31, 32, 33, 132, 133, 131, 131, 131,
	131, 131, 131, 9, 9, 131, 131, 131,
	13, 14, 15, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 31, 32, 33, 132, 9,
	131, 11, 12, 13, 14, 15, 131, 131,
	131, 131, 131, 131, 21, 22, 23, 24,
	25, 26, 131, 131, 131, 131, 31, 32,
	33, 132, 131, 131, 131, 131, 12, 9,
	131, 131, 12, 13, 14, 15, 131, 131,
	131, 131, 131, 131, 21, 22, 23, 24,
	25, 26, 131, 131, 131, 131, 31, 32,
	33, 132, 131, 131, 131, 131, 12, 9,
	131, 131, 12, 13, 14, 15, 131, 131,
	131, 131, 131, 131, 131, 22, 23, 24,
	25, 26, 131, 131, 131, 131, 31, 32,
	33, 132, 131, 131, 131, 131, 12, 9,
	131, 131, 12, 13, 14, 15, 131, 131,
	131, 131, 131, 131, 131, 131, 23, 24,
	25, 26, 131, 131, 131, 131, 31, 32,
	33, 132, 131, 131, 131, 131, 12, 134,
	131, 9, 131, 11, 12, 13, 14, 15,
	131, 17, 18, 131, 131, 131, 21, 22,
	23, 24, 25, 26, 131, 131, 131, 131,
	31, 32, 33, 132, 131, 131, 131, 131,
	12, 9, 131, 131, 12, 13, 14, 15,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 24, 25, 26, 131, 131, 131, 131,
	31, 32, 33, 132, 131, 131, 131, 131,
	12, 134, 131, 9, 131, 11, 12, 13,
	14, 15, 131, 131, 18, 131, 131, 131,
	21, 22, 23, 24, 25, 26, 131, 131,
	131, 131, 31, 32, 33, 132, 131, 131,
	131, 131, 12, 134, 131, 9, 131, 11,
	12, 13, 14, 15, 131, 131, 131, 131,
	131, 131, 21, 22, 23, 24, 25, 26,
	131, 131, 131, 131, 31, 32, 33, 132,
	131, 131, 131, 131, 12, 134, 131, 9,
	131, 11, 12, 13, 14, 15, 16, 17,
	18, 131, 131, 131, 21, 22, 23, 24,
	25, 26, 131, 131, 131, 131, 31, 32,
	33, 132, 131, 131, 131, 131, 12, 6,
	7, 131, 9, 131, 11, 12, 13, 14,
	15, 16, 17, 18, 19, 131, 20, 21,
	22, 23, 24, 25, 26, 131, 131, 131,
	30, 31, 32, 33, 30, 131, 131, 131,
	131, 36, 6, 131, 131, 131, 131, 131,
	131, 9, 6, 131, 131, 131, 131, 131,
	131, 9, 131, 131, 12, 13, 14, 15,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 24, 25, 26, 131, 131, 131, 131,
	31, 32, 33, 132, 135, 131, 131, 131,
	131, 9, 8, 9, 2, 131, 131, 2,
	6, 7, 8, 9, 131, 11, 12, 13,
	14, 15, 16, 17, 18, 19, 6, 20,
	21, 22, 23, 24, 25, 26, 27, 28,
	131, 30, 31, 32, 33, 30, 131, 131,
	131, 131, 36, 9, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 27, 28,
	9, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 28, 2, 136, 136,
	2, 138, 137, 137, 137, 137, 137, 137,
	137, 137, 137, 137, 137, 137, 137, 137,
	137, 137, 137, 137, 137, 137, 137, 137,
	137, 137, 137, 137, 137, 139, 137, 34,
	138, 137, 137, 137, 137, 137, 137, 137,
	137, 137, 137, 137, 137, 137, 137, 137,
	137, 137, 137, 137, 137, 137, 137, 137,
	137, 137, 137, 34, 139, 137, 139, 138,
	137, 137, 137, 137, 137, 137, 137, 137,
	137, 137, 137, 137, 137, 137, 137, 137,
	137, 137, 137, 137, 137, 137, 137, 137,
	137, 137, 34, 137, 35, 0
};

static const short _use_syllable_machine_index_defaults[] = {
	3, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 37, 37,
	37, 37, 37, 37, 37, 37, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 93, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 97, 93,
	67, 99, 102, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 67, 67, 67, 67,
	93, 67, 67, 67, 67, 67, 67, 67,
	67, 67, 67, 67, 97, 93, 67, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 131, 131,
	131, 131, 131, 131, 131, 131, 136, 137,
	137, 137, 0
};

static const signed char _use_syllable_machine_cond_targs[] = {
	0, 1, 30, 0, 57, 59, 87, 88,
	113, 0, 115, 101, 89, 90, 91, 92,
	105, 107, 108, 109, 110, 102, 103, 104,
	96, 97, 98, 116, 117, 118, 111, 93,
	94, 95, 119, 121, 112, 0, 2, 3,
	0, 16, 4, 5, 6, 7, 20, 22,
	23, 24, 25, 17, 18, 19, 11, 12,
	13, 28, 29, 26, 8, 9, 10, 27,
	14, 15, 21, 0, 31, 0, 44, 32,
	33, 34, 35, 48, 50, 51, 52, 53,
	45, 46, 47, 39, 40, 41, 54, 36,
	37, 38, 55, 56, 42, 0, 43, 0,
	49, 0, 0, 0, 58, 0, 0, 0,
	60, 61, 74, 62, 63, 64, 65, 78,
	80, 81, 82, 83, 75, 76, 77, 69,
	70, 71, 84, 66, 67, 68, 85, 86,
	72, 73, 79, 0, 99, 100, 106, 114,
	0, 0, 0, 120, 0
};

static const signed char _use_syllable_machine_cond_actions[] = {
	0, 0, 0, 3, 0, 0, 0, 0,
	0, 4, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 5, 0, 0,
	6, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 7, 0, 8, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 9, 0, 10,
	0, 11, 12, 13, 0, 14, 15, 16,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 17, 0, 0, 0, 0,
	18, 19, 20, 0, 0
};

static const signed char _use_syllable_machine_to_state_actions[] = {
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
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0
};

static const signed char _use_syllable_machine_from_state_actions[] = {
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
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0
};

static const short _use_syllable_machine_eof_trans[] = {
	1, 38, 38, 38, 38, 38, 38, 38,
	38, 38, 38, 38, 38, 38, 38, 38,
	38, 38, 38, 38, 38, 38, 38, 38,
	38, 38, 38, 38, 38, 38, 68, 68,
	68, 68, 68, 68, 68, 68, 68, 68,
	68, 68, 94, 68, 68, 68, 68, 68,
	68, 68, 68, 68, 68, 68, 98, 94,
	68, 100, 103, 68, 68, 68, 68, 68,
	68, 68, 68, 68, 68, 68, 68, 68,
	94, 68, 68, 68, 68, 68, 68, 68,
	68, 68, 68, 68, 98, 94, 68, 132,
	132, 132, 132, 132, 132, 132, 132, 132,
	132, 132, 132, 132, 132, 132, 132, 132,
	132, 132, 132, 132, 132, 132, 132, 132,
	132, 132, 132, 132, 132, 132, 137, 138,
	138, 138, 0
};

static const int use_syllable_machine_start = 0;
static const int use_syllable_machine_first_final = 0;
static const int use_syllable_machine_error = -1;

static const int use_syllable_machine_en_main = 0;


#line 58 "hb-ot-shaper-use-machine.rl"



#line 182 "hb-ot-shaper-use-machine.rl"


#define found_syllable(syllable_type) \
HB_STMT_START { \
	if (0) fprintf (stderr, "syllable %u..%u %s\n", (*ts).second.first, (*te).second.first, #syllable_type); \
		for (unsigned i = (*ts).second.first; i < (*te).second.first; ++i) \
	info[i].syllable() = (syllable_serial << 4) | syllable_type; \
	syllable_serial++; \
	if (syllable_serial == 16) syllable_serial = 1; \
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

#line 792 "hb-ot-shaper-use-machine.hh"
	{
		cs = (int)use_syllable_machine_start;
		ts = 0;
		te = 0;
	}
	
#line 282 "hb-ot-shaper-use-machine.rl"

	
	unsigned int syllable_serial = 1;

#line 801 "hb-ot-shaper-use-machine.hh"
	{
		unsigned int _trans = 0;
		const unsigned char * _keys;
		const short * _inds;
		int _ic;
		_resume: {}
		if ( p == pe && p != eof )
			goto _out;
		switch ( _use_syllable_machine_from_state_actions[cs] ) {
			case 2:  {
					{
#line 1 "NONE"
					{ts = p;}}
				
#line 815 "hb-ot-shaper-use-machine.hh"

				
				break; 
			}
		}
		
		if ( p == eof ) {
			if ( _use_syllable_machine_eof_trans[cs] > 0 ) {
				_trans = (unsigned int)_use_syllable_machine_eof_trans[cs] - 1;
			}
		}
		else {
			_keys = ( _use_syllable_machine_trans_keys + ((cs<<1)));
			_inds = ( _use_syllable_machine_indicies + (_use_syllable_machine_index_offsets[cs]));
			
			if ( ((*p).second.second.use_category()) <= 53 ) {
				_ic = (int)_use_syllable_machine_char_class[(int)((*p).second.second.use_category()) - 0];
				if ( _ic <= (int)(*( _keys+1)) && _ic >= (int)(*( _keys)) )
					_trans = (unsigned int)(*( _inds + (int)( _ic - (int)(*( _keys)) ) )); 
				else
					_trans = (unsigned int)_use_syllable_machine_index_defaults[cs];
			}
			else {
				_trans = (unsigned int)_use_syllable_machine_index_defaults[cs];
			}
			
		}
		cs = (int)_use_syllable_machine_cond_targs[_trans];
		
		if ( _use_syllable_machine_cond_actions[_trans] != 0 ) {
			
			switch ( _use_syllable_machine_cond_actions[_trans] ) {
				case 12:  {
						{
#line 170 "hb-ot-shaper-use-machine.rl"
						{te = p+1;{
#line 170 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_virama_terminated_cluster); }
						}}
					
#line 855 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 10:  {
						{
#line 171 "hb-ot-shaper-use-machine.rl"
						{te = p+1;{
#line 171 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_sakot_terminated_cluster); }
						}}
					
#line 867 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 8:  {
						{
#line 172 "hb-ot-shaper-use-machine.rl"
						{te = p+1;{
#line 172 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_standard_cluster); }
						}}
					
#line 879 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 16:  {
						{
#line 173 "hb-ot-shaper-use-machine.rl"
						{te = p+1;{
#line 173 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_number_joiner_terminated_cluster); }
						}}
					
#line 891 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 14:  {
						{
#line 174 "hb-ot-shaper-use-machine.rl"
						{te = p+1;{
#line 174 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_numeral_cluster); }
						}}
					
#line 903 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 6:  {
						{
#line 175 "hb-ot-shaper-use-machine.rl"
						{te = p+1;{
#line 175 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_symbol_cluster); }
						}}
					
#line 915 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 20:  {
						{
#line 176 "hb-ot-shaper-use-machine.rl"
						{te = p+1;{
#line 176 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_hieroglyph_cluster); }
						}}
					
#line 927 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 4:  {
						{
#line 177 "hb-ot-shaper-use-machine.rl"
						{te = p+1;{
#line 177 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }
						}}
					
#line 939 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 3:  {
						{
#line 178 "hb-ot-shaper-use-machine.rl"
						{te = p+1;{
#line 178 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_non_cluster); }
						}}
					
#line 951 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 11:  {
						{
#line 170 "hb-ot-shaper-use-machine.rl"
						{te = p;p = p - 1;{
#line 170 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_virama_terminated_cluster); }
						}}
					
#line 963 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 9:  {
						{
#line 171 "hb-ot-shaper-use-machine.rl"
						{te = p;p = p - 1;{
#line 171 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_sakot_terminated_cluster); }
						}}
					
#line 975 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 7:  {
						{
#line 172 "hb-ot-shaper-use-machine.rl"
						{te = p;p = p - 1;{
#line 172 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_standard_cluster); }
						}}
					
#line 987 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 15:  {
						{
#line 173 "hb-ot-shaper-use-machine.rl"
						{te = p;p = p - 1;{
#line 173 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_number_joiner_terminated_cluster); }
						}}
					
#line 999 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 13:  {
						{
#line 174 "hb-ot-shaper-use-machine.rl"
						{te = p;p = p - 1;{
#line 174 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_numeral_cluster); }
						}}
					
#line 1011 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 5:  {
						{
#line 175 "hb-ot-shaper-use-machine.rl"
						{te = p;p = p - 1;{
#line 175 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_symbol_cluster); }
						}}
					
#line 1023 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 19:  {
						{
#line 176 "hb-ot-shaper-use-machine.rl"
						{te = p;p = p - 1;{
#line 176 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_hieroglyph_cluster); }
						}}
					
#line 1035 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 17:  {
						{
#line 177 "hb-ot-shaper-use-machine.rl"
						{te = p;p = p - 1;{
#line 177 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }
						}}
					
#line 1047 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
				case 18:  {
						{
#line 178 "hb-ot-shaper-use-machine.rl"
						{te = p;p = p - 1;{
#line 178 "hb-ot-shaper-use-machine.rl"
								found_syllable (use_non_cluster); }
						}}
					
#line 1059 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
			}
			
		}
		
		if ( p == eof ) {
			if ( cs >= 0 )
				goto _out;
		}
		else {
			switch ( _use_syllable_machine_to_state_actions[cs] ) {
				case 1:  {
						{
#line 1 "NONE"
						{ts = 0;}}
					
#line 1078 "hb-ot-shaper-use-machine.hh"

					
					break; 
				}
			}
			
			p += 1;
			goto _resume;
		}
		_out: {}
	}
	
#line 287 "hb-ot-shaper-use-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPER_USE_MACHINE_HH */
