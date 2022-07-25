
#line 1 "hb-ot-shaper-indic-machine.rl"
/*
 * Copyright © 2011,2012  Google, Inc.
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

#ifndef HB_OT_SHAPER_INDIC_MACHINE_HH
#define HB_OT_SHAPER_INDIC_MACHINE_HH

#include "hb.hh"

#include "hb-ot-layout.hh"
#include "hb-ot-shaper-indic.hh"

/* buffer var allocations */
#define indic_category() ot_shaper_var_u8_category() /* indic_category_t */
#define indic_position() ot_shaper_var_u8_auxiliary() /* indic_position_t */

using indic_category_t = unsigned;
using indic_position_t = ot_position_t;

#define I_Cat(Cat) indic_syllable_machine_ex_##Cat

enum indic_syllable_type_t {
  indic_consonant_syllable,
  indic_vowel_syllable,
  indic_standalone_cluster,
  indic_symbol_cluster,
  indic_broken_cluster,
  indic_non_indic_cluster,
};


#line 54 "hb-ot-shaper-indic-machine.hh"
#define indic_syllable_machine_ex_A 9u
#define indic_syllable_machine_ex_C 1u
#define indic_syllable_machine_ex_CM 16u
#define indic_syllable_machine_ex_CS 18u
#define indic_syllable_machine_ex_DOTTEDCIRCLE 11u
#define indic_syllable_machine_ex_H 4u
#define indic_syllable_machine_ex_M 7u
#define indic_syllable_machine_ex_N 3u
#define indic_syllable_machine_ex_PLACEHOLDER 10u
#define indic_syllable_machine_ex_RS 12u
#define indic_syllable_machine_ex_Ra 15u
#define indic_syllable_machine_ex_Repha 14u
#define indic_syllable_machine_ex_SM 8u
#define indic_syllable_machine_ex_Symbol 17u
#define indic_syllable_machine_ex_V 2u
#define indic_syllable_machine_ex_VD 9u
#define indic_syllable_machine_ex_X 0u
#define indic_syllable_machine_ex_ZWJ 6u
#define indic_syllable_machine_ex_ZWNJ 5u


#line 74 "hb-ot-shaper-indic-machine.hh"
static const unsigned char _indic_syllable_machine_trans_keys[] = {
	8u, 8u, 4u, 8u, 5u, 7u, 5u, 8u, 4u, 8u, 4u, 12u, 4u, 8u, 8u, 8u, 
	5u, 7u, 5u, 8u, 4u, 8u, 4u, 12u, 4u, 12u, 4u, 12u, 8u, 8u, 5u, 7u, 
	5u, 8u, 4u, 8u, 4u, 8u, 4u, 12u, 8u, 8u, 5u, 7u, 5u, 8u, 4u, 8u, 
	4u, 8u, 5u, 8u, 8u, 8u, 1u, 18u, 3u, 16u, 3u, 16u, 4u, 16u, 1u, 15u, 
	5u, 9u, 5u, 9u, 9u, 9u, 5u, 9u, 1u, 15u, 1u, 15u, 1u, 15u, 3u, 9u, 
	4u, 9u, 5u, 9u, 4u, 9u, 5u, 9u, 3u, 9u, 5u, 9u, 3u, 16u, 3u, 16u, 
	3u, 16u, 3u, 16u, 4u, 16u, 1u, 15u, 3u, 16u, 3u, 16u, 4u, 16u, 1u, 15u, 
	5u, 9u, 9u, 9u, 5u, 9u, 1u, 15u, 1u, 15u, 3u, 9u, 4u, 9u, 5u, 9u, 
	4u, 9u, 5u, 9u, 5u, 9u, 3u, 9u, 5u, 9u, 3u, 16u, 3u, 16u, 4u, 8u, 
	3u, 16u, 3u, 16u, 4u, 16u, 1u, 15u, 3u, 16u, 1u, 15u, 5u, 9u, 9u, 9u, 
	5u, 9u, 1u, 15u, 1u, 15u, 3u, 9u, 4u, 9u, 5u, 9u, 3u, 16u, 4u, 9u, 
	5u, 9u, 5u, 9u, 3u, 9u, 5u, 9u, 3u, 16u, 4u, 12u, 4u, 8u, 3u, 16u, 
	3u, 16u, 4u, 16u, 1u, 15u, 3u, 16u, 1u, 15u, 5u, 9u, 9u, 9u, 5u, 9u, 
	1u, 15u, 1u, 15u, 3u, 9u, 4u, 9u, 5u, 9u, 3u, 16u, 4u, 9u, 5u, 9u, 
	5u, 9u, 3u, 9u, 5u, 9u, 1u, 16u, 3u, 16u, 1u, 16u, 4u, 12u, 5u, 9u, 
	9u, 9u, 5u, 9u, 1u, 15u, 3u, 9u, 5u, 9u, 5u, 9u, 9u, 9u, 5u, 9u, 
	1u, 15u, 0
};

static const char _indic_syllable_machine_key_spans[] = {
	1, 5, 3, 4, 5, 9, 5, 1, 
	3, 4, 5, 9, 9, 9, 1, 3, 
	4, 5, 5, 9, 1, 3, 4, 5, 
	5, 4, 1, 18, 14, 14, 13, 15, 
	5, 5, 1, 5, 15, 15, 15, 7, 
	6, 5, 6, 5, 7, 5, 14, 14, 
	14, 14, 13, 15, 14, 14, 13, 15, 
	5, 1, 5, 15, 15, 7, 6, 5, 
	6, 5, 5, 7, 5, 14, 14, 5, 
	14, 14, 13, 15, 14, 15, 5, 1, 
	5, 15, 15, 7, 6, 5, 14, 6, 
	5, 5, 7, 5, 14, 9, 5, 14, 
	14, 13, 15, 14, 15, 5, 1, 5, 
	15, 15, 7, 6, 5, 14, 6, 5, 
	5, 7, 5, 16, 14, 16, 9, 5, 
	1, 5, 15, 7, 5, 5, 1, 5, 
	15
};

static const short _indic_syllable_machine_index_offsets[] = {
	0, 2, 8, 12, 17, 23, 33, 39, 
	41, 45, 50, 56, 66, 76, 86, 88, 
	92, 97, 103, 109, 119, 121, 125, 130, 
	136, 142, 147, 149, 168, 183, 198, 212, 
	228, 234, 240, 242, 248, 264, 280, 296, 
	304, 311, 317, 324, 330, 338, 344, 359, 
	374, 389, 404, 418, 434, 449, 464, 478, 
	494, 500, 502, 508, 524, 540, 548, 555, 
	561, 568, 574, 580, 588, 594, 609, 624, 
	630, 645, 660, 674, 690, 705, 721, 727, 
	729, 735, 751, 767, 775, 782, 788, 803, 
	810, 816, 822, 830, 836, 851, 861, 867, 
	882, 897, 911, 927, 942, 958, 964, 966, 
	972, 988, 1004, 1012, 1019, 1025, 1040, 1047, 
	1053, 1059, 1067, 1073, 1090, 1105, 1122, 1132, 
	1138, 1140, 1146, 1162, 1170, 1176, 1182, 1184, 
	1190
};

static const unsigned char _indic_syllable_machine_indicies[] = {
	1, 0, 2, 3, 3, 4, 1, 0, 
	3, 3, 4, 0, 3, 3, 4, 1, 
	0, 5, 3, 3, 4, 1, 0, 2, 
	3, 3, 4, 1, 0, 0, 0, 6, 
	0, 8, 9, 9, 10, 11, 7, 11, 
	7, 9, 9, 10, 7, 9, 9, 10, 
	11, 7, 12, 9, 9, 10, 11, 7, 
	8, 9, 9, 10, 11, 7, 7, 7, 
	13, 7, 8, 9, 9, 10, 11, 7, 
	7, 7, 14, 7, 16, 17, 17, 18, 
	19, 15, 15, 15, 20, 15, 19, 15, 
	17, 17, 18, 21, 17, 17, 18, 19, 
	15, 16, 17, 17, 18, 19, 15, 22, 
	17, 17, 18, 19, 15, 24, 25, 25, 
	26, 27, 23, 23, 23, 28, 23, 27, 
	23, 25, 25, 26, 23, 25, 25, 26, 
	27, 23, 24, 25, 25, 26, 27, 23, 
	29, 25, 25, 26, 27, 23, 17, 17, 
	18, 1, 0, 31, 30, 33, 34, 35, 
	36, 37, 38, 18, 19, 39, 40, 40, 
	20, 32, 41, 42, 43, 44, 45, 32, 
	47, 48, 49, 50, 4, 1, 51, 46, 
	46, 6, 46, 46, 46, 52, 46, 53, 
	48, 54, 54, 4, 1, 51, 46, 46, 
	46, 46, 46, 46, 52, 46, 48, 54, 
	54, 4, 1, 51, 46, 46, 46, 46, 
	46, 46, 52, 46, 33, 46, 46, 46, 
	55, 56, 46, 1, 51, 46, 46, 46, 
	46, 46, 33, 46, 57, 57, 46, 1, 
	51, 46, 51, 46, 46, 58, 51, 46, 
	51, 46, 51, 46, 46, 46, 51, 46, 
	33, 46, 59, 46, 57, 57, 46, 1, 
	51, 46, 46, 46, 46, 46, 33, 46, 
	33, 46, 46, 46, 57, 57, 46, 1, 
	51, 46, 46, 46, 46, 46, 33, 46, 
	33, 46, 46, 46, 57, 56, 46, 1, 
	51, 46, 46, 46, 46, 46, 33, 46, 
	60, 61, 62, 62, 4, 1, 51, 46, 
	61, 62, 62, 4, 1, 51, 46, 62, 
	62, 4, 1, 51, 46, 63, 64, 64, 
	4, 1, 51, 46, 55, 65, 46, 1, 
	51, 46, 55, 46, 57, 57, 46, 1, 
	51, 46, 57, 65, 46, 1, 51, 46, 
	47, 48, 54, 54, 4, 1, 51, 46, 
	46, 46, 46, 46, 46, 52, 46, 47, 
	48, 49, 54, 4, 1, 51, 46, 46, 
	6, 46, 46, 46, 52, 46, 67, 68, 
	69, 70, 10, 11, 71, 66, 66, 14, 
	66, 66, 66, 72, 66, 73, 68, 74, 
	70, 10, 11, 71, 66, 66, 66, 66, 
	66, 66, 72, 66, 68, 74, 70, 10, 
	11, 71, 66, 66, 66, 66, 66, 66, 
	72, 66, 75, 66, 66, 66, 76, 77, 
	66, 11, 71, 66, 66, 66, 66, 66, 
	75, 66, 78, 68, 79, 80, 10, 11, 
	71, 66, 66, 13, 66, 66, 66, 72, 
	66, 81, 68, 74, 74, 10, 11, 71, 
	66, 66, 66, 66, 66, 66, 72, 66, 
	68, 74, 74, 10, 11, 71, 66, 66, 
	66, 66, 66, 66, 72, 66, 75, 66, 
	66, 66, 82, 77, 66, 11, 71, 66, 
	66, 66, 66, 66, 75, 66, 71, 66, 
	66, 83, 71, 66, 71, 66, 71, 66, 
	66, 66, 71, 66, 75, 66, 84, 66, 
	82, 82, 66, 11, 71, 66, 66, 66, 
	66, 66, 75, 66, 75, 66, 66, 66, 
	82, 82, 66, 11, 71, 66, 66, 66, 
	66, 66, 75, 66, 85, 86, 87, 87, 
	10, 11, 71, 66, 86, 87, 87, 10, 
	11, 71, 66, 87, 87, 10, 11, 71, 
	66, 88, 89, 89, 10, 11, 71, 66, 
	76, 90, 66, 11, 71, 66, 82, 82, 
	66, 11, 71, 66, 76, 66, 82, 82, 
	66, 11, 71, 66, 82, 90, 66, 11, 
	71, 66, 78, 68, 74, 74, 10, 11, 
	71, 66, 66, 66, 66, 66, 66, 72, 
	66, 78, 68, 79, 74, 10, 11, 71, 
	66, 66, 13, 66, 66, 66, 72, 66, 
	8, 9, 9, 10, 11, 66, 67, 68, 
	74, 70, 10, 11, 71, 66, 66, 66, 
	66, 66, 66, 72, 66, 92, 36, 93, 
	93, 18, 19, 39, 91, 91, 91, 91, 
	91, 91, 43, 91, 36, 93, 93, 18, 
	19, 39, 91, 91, 91, 91, 91, 91, 
	43, 91, 94, 91, 91, 91, 95, 96, 
	91, 19, 39, 91, 91, 91, 91, 91, 
	94, 91, 35, 36, 97, 98, 18, 19, 
	39, 91, 91, 20, 91, 91, 91, 43, 
	91, 94, 91, 91, 91, 99, 96, 91, 
	19, 39, 91, 91, 91, 91, 91, 94, 
	91, 39, 91, 91, 100, 39, 91, 39, 
	91, 39, 91, 91, 91, 39, 91, 94, 
	91, 101, 91, 99, 99, 91, 19, 39, 
	91, 91, 91, 91, 91, 94, 91, 94, 
	91, 91, 91, 99, 99, 91, 19, 39, 
	91, 91, 91, 91, 91, 94, 91, 102, 
	103, 104, 104, 18, 19, 39, 91, 103, 
	104, 104, 18, 19, 39, 91, 104, 104, 
	18, 19, 39, 91, 35, 36, 93, 93, 
	18, 19, 39, 91, 91, 91, 91, 91, 
	91, 43, 91, 105, 106, 106, 18, 19, 
	39, 91, 95, 107, 91, 19, 39, 91, 
	99, 99, 91, 19, 39, 91, 95, 91, 
	99, 99, 91, 19, 39, 91, 99, 107, 
	91, 19, 39, 91, 35, 36, 97, 93, 
	18, 19, 39, 91, 91, 20, 91, 91, 
	91, 43, 91, 16, 17, 17, 18, 19, 
	108, 108, 108, 20, 108, 16, 17, 17, 
	18, 19, 108, 110, 111, 112, 113, 26, 
	27, 114, 109, 109, 28, 109, 109, 109, 
	115, 109, 116, 111, 113, 113, 26, 27, 
	114, 109, 109, 109, 109, 109, 109, 115, 
	109, 111, 113, 113, 26, 27, 114, 109, 
	109, 109, 109, 109, 109, 115, 109, 117, 
	109, 109, 109, 118, 119, 109, 27, 114, 
	109, 109, 109, 109, 109, 117, 109, 110, 
	111, 112, 40, 26, 27, 114, 109, 109, 
	28, 109, 109, 109, 115, 109, 117, 109, 
	109, 109, 120, 119, 109, 27, 114, 109, 
	109, 109, 109, 109, 117, 109, 114, 109, 
	109, 121, 114, 109, 114, 109, 114, 109, 
	109, 109, 114, 109, 117, 109, 122, 109, 
	120, 120, 109, 27, 114, 109, 109, 109, 
	109, 109, 117, 109, 117, 109, 109, 109, 
	120, 120, 109, 27, 114, 109, 109, 109, 
	109, 109, 117, 109, 123, 124, 125, 125, 
	26, 27, 114, 109, 124, 125, 125, 26, 
	27, 114, 109, 125, 125, 26, 27, 114, 
	109, 110, 111, 113, 113, 26, 27, 114, 
	109, 109, 109, 109, 109, 109, 115, 109, 
	126, 127, 127, 26, 27, 114, 109, 118, 
	128, 109, 27, 114, 109, 120, 120, 109, 
	27, 114, 109, 118, 109, 120, 120, 109, 
	27, 114, 109, 120, 128, 109, 27, 114, 
	109, 33, 34, 35, 36, 97, 93, 18, 
	19, 39, 40, 40, 20, 91, 91, 33, 
	43, 91, 47, 129, 49, 50, 4, 1, 
	51, 46, 46, 6, 46, 46, 46, 52, 
	46, 33, 34, 35, 36, 130, 131, 18, 
	132, 133, 46, 40, 20, 46, 46, 33, 
	43, 46, 16, 134, 134, 18, 132, 51, 
	46, 46, 20, 46, 133, 46, 46, 135, 
	133, 46, 133, 46, 133, 46, 46, 46, 
	133, 46, 33, 46, 59, 16, 134, 134, 
	18, 132, 51, 46, 46, 46, 46, 46, 
	33, 46, 137, 136, 138, 138, 136, 31, 
	139, 136, 138, 138, 136, 31, 139, 136, 
	139, 136, 136, 140, 139, 136, 139, 136, 
	139, 136, 136, 136, 139, 136, 33, 108, 
	108, 108, 108, 108, 108, 108, 108, 40, 
	108, 108, 108, 108, 33, 108, 0
};

static const unsigned char _indic_syllable_machine_trans_targs[] = {
	27, 33, 38, 2, 39, 45, 46, 27, 
	55, 8, 61, 56, 68, 69, 72, 27, 
	77, 15, 83, 78, 86, 27, 91, 27, 
	100, 21, 106, 101, 109, 114, 27, 125, 
	27, 28, 48, 73, 75, 93, 94, 79, 
	95, 115, 116, 87, 123, 128, 27, 29, 
	31, 5, 47, 34, 42, 30, 1, 32, 
	36, 0, 35, 37, 40, 41, 3, 43, 
	4, 44, 27, 49, 51, 12, 71, 57, 
	64, 50, 6, 52, 66, 59, 53, 11, 
	70, 54, 7, 58, 60, 62, 63, 9, 
	65, 10, 67, 27, 74, 17, 76, 89, 
	81, 13, 92, 14, 80, 82, 84, 85, 
	16, 88, 18, 90, 27, 27, 96, 98, 
	19, 23, 102, 110, 97, 99, 112, 104, 
	20, 103, 105, 107, 108, 22, 111, 24, 
	113, 117, 118, 122, 119, 120, 25, 121, 
	27, 124, 26, 126, 127
};

static const char _indic_syllable_machine_trans_actions[] = {
	1, 0, 2, 0, 2, 2, 2, 3, 
	2, 0, 2, 0, 2, 2, 2, 4, 
	2, 0, 5, 0, 5, 6, 2, 7, 
	2, 0, 2, 0, 2, 2, 8, 0, 
	11, 2, 2, 5, 0, 12, 12, 0, 
	2, 5, 2, 5, 2, 0, 13, 2, 
	0, 0, 2, 0, 2, 2, 0, 2, 
	2, 0, 0, 2, 2, 2, 0, 0, 
	0, 2, 14, 2, 0, 0, 2, 0, 
	2, 2, 0, 2, 2, 2, 2, 0, 
	2, 2, 0, 0, 2, 2, 2, 0, 
	0, 0, 2, 15, 5, 0, 5, 2, 
	2, 0, 5, 0, 0, 2, 5, 5, 
	0, 0, 0, 2, 16, 17, 2, 0, 
	0, 0, 0, 2, 2, 2, 2, 2, 
	0, 0, 2, 2, 2, 0, 0, 0, 
	2, 0, 18, 18, 0, 0, 0, 0, 
	19, 2, 0, 0, 0
};

static const char _indic_syllable_machine_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 9, 0, 0, 0, 0, 
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
	0
};

static const char _indic_syllable_machine_from_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 10, 0, 0, 0, 0, 
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
	0
};

static const short _indic_syllable_machine_eof_trans[] = {
	1, 1, 1, 1, 1, 1, 8, 8, 
	8, 8, 8, 8, 8, 16, 16, 22, 
	16, 16, 16, 24, 24, 24, 24, 24, 
	24, 1, 31, 0, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	67, 67, 67, 67, 67, 67, 67, 67, 
	67, 67, 67, 67, 67, 67, 67, 67, 
	67, 67, 67, 67, 67, 67, 67, 67, 
	67, 92, 92, 92, 92, 92, 92, 92, 
	92, 92, 92, 92, 92, 92, 92, 92, 
	92, 92, 92, 92, 92, 109, 109, 110, 
	110, 110, 110, 110, 110, 110, 110, 110, 
	110, 110, 110, 110, 110, 110, 110, 110, 
	110, 110, 110, 92, 47, 47, 47, 47, 
	47, 47, 47, 137, 137, 137, 137, 137, 
	109
};

static const int indic_syllable_machine_start = 27;
static const int indic_syllable_machine_first_final = 27;
static const int indic_syllable_machine_error = -1;

static const int indic_syllable_machine_en_main = 27;


#line 58 "hb-ot-shaper-indic-machine.rl"



#line 117 "hb-ot-shaper-indic-machine.rl"


#define found_syllable(syllable_type) \
  HB_STMT_START { \
    if (0) fprintf (stderr, "syllable %d..%d %s\n", ts, te, #syllable_type); \
    for (unsigned int i = ts; i < te; i++) \
      info[i].syllable() = (syllable_serial << 4) | syllable_type; \
    syllable_serial++; \
    if (unlikely (syllable_serial == 16)) syllable_serial = 1; \
  } HB_STMT_END

inline void
find_syllables_indic (hb_buffer_t *buffer)
{
  unsigned int p, pe, eof, ts, te, act;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  
#line 415 "hb-ot-shaper-indic-machine.hh"
	{
	cs = indic_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 137 "hb-ot-shaper-indic-machine.rl"


  p = 0;
  pe = eof = buffer->len;

  unsigned int syllable_serial = 1;
  
#line 427 "hb-ot-shaper-indic-machine.hh"
	{
	int _slen;
	int _trans;
	const unsigned char *_keys;
	const unsigned char *_inds;
	if ( p == pe )
		goto _test_eof;
_resume:
	switch ( _indic_syllable_machine_from_state_actions[cs] ) {
	case 10:
#line 1 "NONE"
	{ts = p;}
	break;
#line 439 "hb-ot-shaper-indic-machine.hh"
	}

	_keys = _indic_syllable_machine_trans_keys + (cs<<1);
	_inds = _indic_syllable_machine_indicies + _indic_syllable_machine_index_offsets[cs];

	_slen = _indic_syllable_machine_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=( info[p].indic_category()) &&
		( info[p].indic_category()) <= _keys[1] ?
		( info[p].indic_category()) - _keys[0] : _slen ];

_eof_trans:
	cs = _indic_syllable_machine_trans_targs[_trans];

	if ( _indic_syllable_machine_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _indic_syllable_machine_trans_actions[_trans] ) {
	case 2:
#line 1 "NONE"
	{te = p+1;}
	break;
	case 11:
#line 113 "hb-ot-shaper-indic-machine.rl"
	{te = p+1;{ found_syllable (indic_non_indic_cluster); }}
	break;
	case 13:
#line 108 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_consonant_syllable); }}
	break;
	case 14:
#line 109 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_vowel_syllable); }}
	break;
	case 17:
#line 110 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_standalone_cluster); }}
	break;
	case 19:
#line 111 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_symbol_cluster); }}
	break;
	case 15:
#line 112 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }}
	break;
	case 16:
#line 113 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_non_indic_cluster); }}
	break;
	case 1:
#line 108 "hb-ot-shaper-indic-machine.rl"
	{{p = ((te))-1;}{ found_syllable (indic_consonant_syllable); }}
	break;
	case 3:
#line 109 "hb-ot-shaper-indic-machine.rl"
	{{p = ((te))-1;}{ found_syllable (indic_vowel_syllable); }}
	break;
	case 7:
#line 110 "hb-ot-shaper-indic-machine.rl"
	{{p = ((te))-1;}{ found_syllable (indic_standalone_cluster); }}
	break;
	case 8:
#line 111 "hb-ot-shaper-indic-machine.rl"
	{{p = ((te))-1;}{ found_syllable (indic_symbol_cluster); }}
	break;
	case 4:
#line 112 "hb-ot-shaper-indic-machine.rl"
	{{p = ((te))-1;}{ found_syllable (indic_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }}
	break;
	case 6:
#line 1 "NONE"
	{	switch( act ) {
	case 1:
	{{p = ((te))-1;} found_syllable (indic_consonant_syllable); }
	break;
	case 5:
	{{p = ((te))-1;} found_syllable (indic_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }
	break;
	case 6:
	{{p = ((te))-1;} found_syllable (indic_non_indic_cluster); }
	break;
	}
	}
	break;
	case 18:
#line 1 "NONE"
	{te = p+1;}
#line 108 "hb-ot-shaper-indic-machine.rl"
	{act = 1;}
	break;
	case 5:
#line 1 "NONE"
	{te = p+1;}
#line 112 "hb-ot-shaper-indic-machine.rl"
	{act = 5;}
	break;
	case 12:
#line 1 "NONE"
	{te = p+1;}
#line 113 "hb-ot-shaper-indic-machine.rl"
	{act = 6;}
	break;
#line 521 "hb-ot-shaper-indic-machine.hh"
	}

_again:
	switch ( _indic_syllable_machine_to_state_actions[cs] ) {
	case 9:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 528 "hb-ot-shaper-indic-machine.hh"
	}

	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	if ( _indic_syllable_machine_eof_trans[cs] > 0 ) {
		_trans = _indic_syllable_machine_eof_trans[cs] - 1;
		goto _eof_trans;
	}
	}

	}

#line 145 "hb-ot-shaper-indic-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPER_INDIC_MACHINE_HH */
