
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


#line 57 "hb-ot-shaper-indic-machine.hh"
#define indic_syllable_machine_ex_A 9u
#define indic_syllable_machine_ex_C 1u
#define indic_syllable_machine_ex_CM 16u
#define indic_syllable_machine_ex_CS 18u
#define indic_syllable_machine_ex_DOTTEDCIRCLE 11u
#define indic_syllable_machine_ex_H 4u
#define indic_syllable_machine_ex_M 7u
#define indic_syllable_machine_ex_MPst 13u
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


#line 80 "hb-ot-shaper-indic-machine.hh"
static const unsigned char _indic_syllable_machine_trans_keys[] = {
	8u, 8u, 4u, 13u, 5u, 13u, 5u, 13u, 13u, 13u, 4u, 13u, 4u, 13u, 4u, 13u, 
	8u, 8u, 5u, 13u, 5u, 13u, 13u, 13u, 4u, 13u, 4u, 13u, 4u, 13u, 4u, 13u, 
	8u, 8u, 5u, 13u, 5u, 13u, 13u, 13u, 4u, 13u, 4u, 13u, 4u, 13u, 8u, 8u, 
	5u, 13u, 5u, 13u, 13u, 13u, 4u, 13u, 4u, 13u, 5u, 13u, 8u, 8u, 1u, 18u, 
	3u, 16u, 3u, 16u, 4u, 16u, 1u, 15u, 5u, 9u, 5u, 9u, 9u, 9u, 5u, 9u, 
	1u, 15u, 1u, 15u, 1u, 15u, 3u, 13u, 4u, 13u, 5u, 13u, 5u, 13u, 4u, 13u, 
	5u, 9u, 3u, 9u, 5u, 9u, 3u, 16u, 3u, 16u, 3u, 16u, 3u, 16u, 4u, 16u, 
	1u, 15u, 3u, 16u, 3u, 16u, 4u, 16u, 1u, 15u, 5u, 9u, 9u, 9u, 5u, 9u, 
	1u, 15u, 1u, 15u, 3u, 13u, 4u, 13u, 5u, 13u, 5u, 13u, 4u, 13u, 5u, 9u, 
	5u, 9u, 3u, 9u, 5u, 9u, 3u, 16u, 3u, 16u, 4u, 13u, 3u, 16u, 3u, 16u, 
	4u, 16u, 1u, 15u, 3u, 16u, 1u, 15u, 5u, 9u, 9u, 9u, 5u, 9u, 1u, 15u, 
	1u, 15u, 3u, 13u, 4u, 13u, 5u, 13u, 5u, 13u, 3u, 16u, 4u, 13u, 5u, 9u, 
	5u, 9u, 3u, 9u, 5u, 9u, 3u, 16u, 4u, 13u, 4u, 13u, 3u, 16u, 3u, 16u, 
	4u, 16u, 1u, 15u, 3u, 16u, 1u, 15u, 5u, 9u, 9u, 9u, 5u, 9u, 1u, 15u, 
	1u, 15u, 3u, 13u, 4u, 13u, 5u, 13u, 5u, 13u, 3u, 16u, 4u, 13u, 5u, 9u, 
	5u, 9u, 3u, 9u, 5u, 9u, 1u, 16u, 3u, 16u, 1u, 16u, 4u, 13u, 5u, 13u, 
	5u, 13u, 9u, 9u, 5u, 9u, 1u, 15u, 3u, 9u, 5u, 9u, 5u, 9u, 9u, 9u, 
	5u, 9u, 1u, 15u, 0
};

static const char _indic_syllable_machine_key_spans[] = {
	1, 10, 9, 9, 1, 10, 10, 10, 
	1, 9, 9, 1, 10, 10, 10, 10, 
	1, 9, 9, 1, 10, 10, 10, 1, 
	9, 9, 1, 10, 10, 9, 1, 18, 
	14, 14, 13, 15, 5, 5, 1, 5, 
	15, 15, 15, 11, 10, 9, 9, 10, 
	5, 7, 5, 14, 14, 14, 14, 13, 
	15, 14, 14, 13, 15, 5, 1, 5, 
	15, 15, 11, 10, 9, 9, 10, 5, 
	5, 7, 5, 14, 14, 10, 14, 14, 
	13, 15, 14, 15, 5, 1, 5, 15, 
	15, 11, 10, 9, 9, 14, 10, 5, 
	5, 7, 5, 14, 10, 10, 14, 14, 
	13, 15, 14, 15, 5, 1, 5, 15, 
	15, 11, 10, 9, 9, 14, 10, 5, 
	5, 7, 5, 16, 14, 16, 10, 9, 
	9, 1, 5, 15, 7, 5, 5, 1, 
	5, 15
};

static const short _indic_syllable_machine_index_offsets[] = {
	0, 2, 13, 23, 33, 35, 46, 57, 
	68, 70, 80, 90, 92, 103, 114, 125, 
	136, 138, 148, 158, 160, 171, 182, 193, 
	195, 205, 215, 217, 228, 239, 249, 251, 
	270, 285, 300, 314, 330, 336, 342, 344, 
	350, 366, 382, 398, 410, 421, 431, 441, 
	452, 458, 466, 472, 487, 502, 517, 532, 
	546, 562, 577, 592, 606, 622, 628, 630, 
	636, 652, 668, 680, 691, 701, 711, 722, 
	728, 734, 742, 748, 763, 778, 789, 804, 
	819, 833, 849, 864, 880, 886, 888, 894, 
	910, 926, 938, 949, 959, 969, 984, 995, 
	1001, 1007, 1015, 1021, 1036, 1047, 1058, 1073, 
	1088, 1102, 1118, 1133, 1149, 1155, 1157, 1163, 
	1179, 1195, 1207, 1218, 1228, 1238, 1253, 1264, 
	1270, 1276, 1284, 1290, 1307, 1322, 1339, 1350, 
	1360, 1370, 1372, 1378, 1394, 1402, 1408, 1414, 
	1416, 1422
};

static const unsigned char _indic_syllable_machine_indicies[] = {
	1, 0, 2, 3, 3, 4, 5, 0, 
	0, 0, 0, 4, 0, 3, 3, 4, 
	6, 0, 0, 0, 0, 4, 0, 3, 
	3, 4, 5, 0, 0, 0, 0, 4, 
	0, 4, 0, 7, 3, 3, 4, 5, 
	0, 0, 0, 0, 4, 0, 2, 3, 
	3, 4, 5, 0, 0, 0, 8, 4, 
	0, 10, 11, 11, 12, 13, 9, 9, 
	9, 9, 12, 9, 14, 9, 11, 11, 
	12, 15, 9, 9, 9, 9, 12, 9, 
	11, 11, 12, 13, 9, 9, 9, 9, 
	12, 9, 12, 9, 16, 11, 11, 12, 
	13, 9, 9, 9, 9, 12, 9, 10, 
	11, 11, 12, 13, 9, 9, 9, 17, 
	12, 9, 10, 11, 11, 12, 13, 9, 
	9, 9, 18, 12, 9, 20, 21, 21, 
	22, 23, 19, 19, 19, 24, 22, 19, 
	25, 19, 21, 21, 22, 27, 26, 26, 
	26, 26, 22, 26, 21, 21, 22, 23, 
	19, 19, 19, 19, 22, 19, 22, 26, 
	20, 21, 21, 22, 23, 19, 19, 19, 
	19, 22, 19, 28, 21, 21, 22, 23, 
	19, 19, 19, 19, 22, 19, 30, 31, 
	31, 32, 33, 29, 29, 29, 34, 32, 
	29, 35, 29, 31, 31, 32, 36, 29, 
	29, 29, 29, 32, 29, 31, 31, 32, 
	33, 29, 29, 29, 29, 32, 29, 32, 
	29, 30, 31, 31, 32, 33, 29, 29, 
	29, 29, 32, 29, 37, 31, 31, 32, 
	33, 29, 29, 29, 29, 32, 29, 21, 
	21, 22, 38, 0, 0, 0, 0, 22, 
	0, 40, 39, 42, 43, 44, 45, 46, 
	47, 22, 23, 48, 49, 49, 24, 22, 
	50, 51, 52, 53, 54, 41, 56, 57, 
	58, 59, 4, 5, 60, 55, 55, 8, 
	4, 55, 55, 61, 55, 62, 57, 63, 
	63, 4, 5, 60, 55, 55, 55, 4, 
	55, 55, 61, 55, 57, 63, 63, 4, 
	5, 60, 55, 55, 55, 4, 55, 55, 
	61, 55, 42, 55, 55, 55, 64, 65, 
	55, 1, 60, 55, 55, 55, 55, 55, 
	42, 55, 66, 66, 55, 1, 60, 55, 
	60, 55, 55, 67, 60, 55, 60, 55, 
	60, 55, 55, 55, 60, 55, 42, 55, 
	68, 55, 66, 66, 55, 1, 60, 55, 
	55, 55, 55, 55, 42, 55, 42, 55, 
	55, 55, 66, 66, 55, 1, 60, 55, 
	55, 55, 55, 55, 42, 55, 42, 55, 
	55, 55, 66, 65, 55, 1, 60, 55, 
	55, 55, 55, 55, 42, 55, 69, 70, 
	71, 71, 4, 5, 60, 55, 55, 55, 
	4, 55, 70, 71, 71, 4, 5, 60, 
	55, 55, 55, 4, 55, 71, 71, 4, 
	5, 60, 55, 55, 55, 4, 55, 60, 
	55, 55, 67, 60, 55, 55, 55, 4, 
	55, 72, 73, 73, 4, 5, 60, 55, 
	55, 55, 4, 55, 64, 74, 55, 1, 
	60, 55, 64, 55, 66, 66, 55, 1, 
	60, 55, 66, 74, 55, 1, 60, 55, 
	56, 57, 63, 63, 4, 5, 60, 55, 
	55, 55, 4, 55, 55, 61, 55, 56, 
	57, 58, 63, 4, 5, 60, 55, 55, 
	8, 4, 55, 55, 61, 55, 76, 77, 
	78, 79, 12, 13, 80, 75, 75, 18, 
	12, 75, 75, 81, 75, 82, 77, 83, 
	79, 12, 13, 80, 75, 75, 75, 12, 
	75, 75, 81, 75, 77, 83, 79, 12, 
	13, 80, 75, 75, 75, 12, 75, 75, 
	81, 75, 84, 75, 75, 75, 85, 86, 
	75, 14, 80, 75, 75, 75, 75, 75, 
	84, 75, 87, 77, 88, 89, 12, 13, 
	80, 75, 75, 17, 12, 75, 75, 81, 
	75, 90, 77, 83, 83, 12, 13, 80, 
	75, 75, 75, 12, 75, 75, 81, 75, 
	77, 83, 83, 12, 13, 80, 75, 75, 
	75, 12, 75, 75, 81, 75, 84, 75, 
	75, 75, 91, 86, 75, 14, 80, 75, 
	75, 75, 75, 75, 84, 75, 80, 75, 
	75, 92, 80, 75, 80, 75, 80, 75, 
	75, 75, 80, 75, 84, 75, 93, 75, 
	91, 91, 75, 14, 80, 75, 75, 75, 
	75, 75, 84, 75, 84, 75, 75, 75, 
	91, 91, 75, 14, 80, 75, 75, 75, 
	75, 75, 84, 75, 94, 95, 96, 96, 
	12, 13, 80, 75, 75, 75, 12, 75, 
	95, 96, 96, 12, 13, 80, 75, 75, 
	75, 12, 75, 96, 96, 12, 13, 80, 
	75, 75, 75, 12, 75, 80, 75, 75, 
	92, 80, 75, 75, 75, 12, 75, 97, 
	98, 98, 12, 13, 80, 75, 75, 75, 
	12, 75, 85, 99, 75, 14, 80, 75, 
	91, 91, 75, 14, 80, 75, 85, 75, 
	91, 91, 75, 14, 80, 75, 91, 99, 
	75, 14, 80, 75, 87, 77, 83, 83, 
	12, 13, 80, 75, 75, 75, 12, 75, 
	75, 81, 75, 87, 77, 88, 83, 12, 
	13, 80, 75, 75, 17, 12, 75, 75, 
	81, 75, 10, 11, 11, 12, 13, 75, 
	75, 75, 75, 12, 75, 76, 77, 83, 
	79, 12, 13, 80, 75, 75, 75, 12, 
	75, 75, 81, 75, 101, 45, 102, 102, 
	22, 23, 48, 100, 100, 100, 22, 100, 
	100, 52, 100, 45, 102, 102, 22, 23, 
	48, 100, 100, 100, 22, 100, 100, 52, 
	100, 103, 100, 100, 100, 104, 105, 100, 
	25, 48, 100, 100, 100, 100, 100, 103, 
	100, 44, 45, 106, 107, 22, 23, 48, 
	100, 100, 24, 22, 100, 100, 52, 100, 
	103, 100, 100, 100, 108, 105, 100, 25, 
	48, 100, 100, 100, 100, 100, 103, 100, 
	48, 100, 100, 109, 48, 100, 48, 100, 
	48, 100, 100, 100, 48, 100, 103, 100, 
	110, 100, 108, 108, 100, 25, 48, 100, 
	100, 100, 100, 100, 103, 100, 103, 100, 
	100, 100, 108, 108, 100, 25, 48, 100, 
	100, 100, 100, 100, 103, 100, 111, 112, 
	113, 113, 22, 23, 48, 100, 100, 100, 
	22, 100, 112, 113, 113, 22, 23, 48, 
	100, 100, 100, 22, 100, 113, 113, 22, 
	23, 48, 100, 100, 100, 22, 100, 48, 
	100, 100, 109, 48, 100, 100, 100, 22, 
	100, 44, 45, 102, 102, 22, 23, 48, 
	100, 100, 100, 22, 100, 100, 52, 100, 
	114, 115, 115, 22, 23, 48, 100, 100, 
	100, 22, 100, 104, 116, 100, 25, 48, 
	100, 108, 108, 100, 25, 48, 100, 104, 
	100, 108, 108, 100, 25, 48, 100, 108, 
	116, 100, 25, 48, 100, 44, 45, 106, 
	102, 22, 23, 48, 100, 100, 24, 22, 
	100, 100, 52, 100, 20, 21, 21, 22, 
	23, 117, 117, 117, 24, 22, 117, 20, 
	21, 21, 22, 23, 117, 117, 117, 117, 
	22, 117, 119, 120, 121, 122, 32, 33, 
	123, 118, 118, 34, 32, 118, 118, 124, 
	118, 125, 120, 122, 122, 32, 33, 123, 
	118, 118, 118, 32, 118, 118, 124, 118, 
	120, 122, 122, 32, 33, 123, 118, 118, 
	118, 32, 118, 118, 124, 118, 126, 118, 
	118, 118, 127, 128, 118, 35, 123, 118, 
	118, 118, 118, 118, 126, 118, 119, 120, 
	121, 49, 32, 33, 123, 118, 118, 34, 
	32, 118, 118, 124, 118, 126, 118, 118, 
	118, 129, 128, 118, 35, 123, 118, 118, 
	118, 118, 118, 126, 118, 123, 118, 118, 
	130, 123, 118, 123, 118, 123, 118, 118, 
	118, 123, 118, 126, 118, 131, 118, 129, 
	129, 118, 35, 123, 118, 118, 118, 118, 
	118, 126, 118, 126, 118, 118, 118, 129, 
	129, 118, 35, 123, 118, 118, 118, 118, 
	118, 126, 118, 132, 133, 134, 134, 32, 
	33, 123, 118, 118, 118, 32, 118, 133, 
	134, 134, 32, 33, 123, 118, 118, 118, 
	32, 118, 134, 134, 32, 33, 123, 118, 
	118, 118, 32, 118, 123, 118, 118, 130, 
	123, 118, 118, 118, 32, 118, 119, 120, 
	122, 122, 32, 33, 123, 118, 118, 118, 
	32, 118, 118, 124, 118, 135, 136, 136, 
	32, 33, 123, 118, 118, 118, 32, 118, 
	127, 137, 118, 35, 123, 118, 129, 129, 
	118, 35, 123, 118, 127, 118, 129, 129, 
	118, 35, 123, 118, 129, 137, 118, 35, 
	123, 118, 42, 43, 44, 45, 106, 102, 
	22, 23, 48, 49, 49, 24, 22, 100, 
	42, 52, 100, 56, 138, 58, 59, 4, 
	5, 60, 55, 55, 8, 4, 55, 55, 
	61, 55, 42, 43, 44, 45, 139, 140, 
	22, 141, 142, 55, 49, 24, 22, 55, 
	42, 52, 55, 20, 143, 143, 22, 141, 
	60, 55, 55, 24, 22, 55, 60, 55, 
	55, 67, 60, 55, 55, 55, 22, 55, 
	142, 55, 55, 144, 142, 55, 55, 55, 
	22, 55, 142, 55, 142, 55, 55, 55, 
	142, 55, 42, 55, 68, 20, 143, 143, 
	22, 141, 60, 55, 55, 55, 22, 55, 
	42, 55, 146, 145, 147, 147, 145, 40, 
	148, 145, 147, 147, 145, 40, 148, 145, 
	148, 145, 145, 149, 148, 145, 148, 145, 
	148, 145, 145, 145, 148, 145, 42, 117, 
	117, 117, 117, 117, 117, 117, 117, 49, 
	117, 117, 117, 117, 42, 117, 0
};

static const unsigned char _indic_syllable_machine_trans_targs[] = {
	31, 37, 42, 2, 43, 46, 4, 50, 
	51, 31, 60, 9, 66, 69, 61, 11, 
	74, 75, 78, 31, 83, 17, 89, 92, 
	93, 84, 31, 19, 98, 31, 107, 24, 
	113, 116, 117, 108, 26, 122, 127, 31, 
	134, 31, 32, 53, 79, 81, 100, 101, 
	85, 102, 123, 124, 94, 132, 137, 31, 
	33, 35, 6, 52, 38, 47, 34, 1, 
	36, 40, 0, 39, 41, 44, 45, 3, 
	48, 5, 49, 31, 54, 56, 14, 77, 
	62, 70, 55, 7, 57, 72, 64, 58, 
	13, 76, 59, 8, 63, 65, 67, 68, 
	10, 71, 12, 73, 31, 80, 20, 82, 
	96, 87, 15, 99, 16, 86, 88, 90, 
	91, 18, 95, 21, 97, 31, 31, 103, 
	105, 22, 27, 109, 118, 104, 106, 120, 
	111, 23, 110, 112, 114, 115, 25, 119, 
	28, 121, 125, 126, 131, 128, 129, 29, 
	130, 31, 133, 30, 135, 136
};

static const char _indic_syllable_machine_trans_actions[] = {
	1, 0, 2, 0, 2, 0, 0, 2, 
	2, 3, 2, 0, 2, 0, 0, 0, 
	2, 2, 2, 4, 2, 0, 5, 0, 
	5, 0, 6, 0, 2, 7, 2, 0, 
	2, 0, 2, 0, 0, 2, 0, 8, 
	0, 11, 2, 2, 5, 0, 12, 12, 
	0, 2, 5, 2, 5, 2, 0, 13, 
	2, 0, 0, 2, 0, 2, 2, 0, 
	2, 2, 0, 0, 2, 2, 2, 0, 
	0, 0, 2, 14, 2, 0, 0, 2, 
	0, 2, 2, 0, 2, 2, 2, 2, 
	0, 2, 2, 0, 0, 2, 2, 2, 
	0, 0, 0, 2, 15, 5, 0, 5, 
	2, 2, 0, 5, 0, 0, 2, 5, 
	5, 0, 0, 0, 2, 16, 17, 2, 
	0, 0, 0, 0, 2, 2, 2, 2, 
	2, 0, 0, 2, 2, 2, 0, 0, 
	0, 2, 0, 18, 18, 0, 0, 0, 
	0, 19, 2, 0, 0, 0
};

static const char _indic_syllable_machine_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 9, 
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
	0, 0
};

static const char _indic_syllable_machine_from_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 10, 
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
	0, 0
};

static const short _indic_syllable_machine_eof_trans[] = {
	1, 1, 1, 1, 1, 1, 1, 10, 
	10, 10, 10, 10, 10, 10, 10, 20, 
	20, 27, 20, 27, 20, 20, 30, 30, 
	30, 30, 30, 30, 30, 1, 40, 0, 
	56, 56, 56, 56, 56, 56, 56, 56, 
	56, 56, 56, 56, 56, 56, 56, 56, 
	56, 56, 56, 56, 56, 76, 76, 76, 
	76, 76, 76, 76, 76, 76, 76, 76, 
	76, 76, 76, 76, 76, 76, 76, 76, 
	76, 76, 76, 76, 76, 76, 76, 101, 
	101, 101, 101, 101, 101, 101, 101, 101, 
	101, 101, 101, 101, 101, 101, 101, 101, 
	101, 101, 101, 101, 118, 118, 119, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 119, 119, 119, 119, 119, 
	119, 119, 119, 101, 56, 56, 56, 56, 
	56, 56, 56, 56, 146, 146, 146, 146, 
	146, 118
};

static const int indic_syllable_machine_start = 31;
static const int indic_syllable_machine_first_final = 31;
static const int indic_syllable_machine_error = -1;

static const int indic_syllable_machine_en_main = 31;


#line 58 "hb-ot-shaper-indic-machine.rl"



#line 118 "hb-ot-shaper-indic-machine.rl"


#define found_syllable(syllable_type) \
  HB_STMT_START { \
    if (0) fprintf (stderr, "syllable %u..%u %s\n", ts, te, #syllable_type); \
    for (unsigned int i = ts; i < te; i++) \
      info[i].syllable() = (syllable_serial << 4) | syllable_type; \
    syllable_serial++; \
    if (syllable_serial == 16) syllable_serial = 1; \
  } HB_STMT_END

inline void
find_syllables_indic (hb_buffer_t *buffer)
{
  unsigned int p, pe, eof, ts, te, act;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  
#line 464 "hb-ot-shaper-indic-machine.hh"
	{
	cs = indic_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 138 "hb-ot-shaper-indic-machine.rl"


  p = 0;
  pe = eof = buffer->len;

  unsigned int syllable_serial = 1;
  
#line 480 "hb-ot-shaper-indic-machine.hh"
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
#line 494 "hb-ot-shaper-indic-machine.hh"
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
#line 114 "hb-ot-shaper-indic-machine.rl"
	{te = p+1;{ found_syllable (indic_non_indic_cluster); }}
	break;
	case 13:
#line 109 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_consonant_syllable); }}
	break;
	case 14:
#line 110 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_vowel_syllable); }}
	break;
	case 17:
#line 111 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_standalone_cluster); }}
	break;
	case 19:
#line 112 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_symbol_cluster); }}
	break;
	case 15:
#line 113 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }}
	break;
	case 16:
#line 114 "hb-ot-shaper-indic-machine.rl"
	{te = p;p--;{ found_syllable (indic_non_indic_cluster); }}
	break;
	case 1:
#line 109 "hb-ot-shaper-indic-machine.rl"
	{{p = ((te))-1;}{ found_syllable (indic_consonant_syllable); }}
	break;
	case 3:
#line 110 "hb-ot-shaper-indic-machine.rl"
	{{p = ((te))-1;}{ found_syllable (indic_vowel_syllable); }}
	break;
	case 7:
#line 111 "hb-ot-shaper-indic-machine.rl"
	{{p = ((te))-1;}{ found_syllable (indic_standalone_cluster); }}
	break;
	case 8:
#line 112 "hb-ot-shaper-indic-machine.rl"
	{{p = ((te))-1;}{ found_syllable (indic_symbol_cluster); }}
	break;
	case 4:
#line 113 "hb-ot-shaper-indic-machine.rl"
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
#line 109 "hb-ot-shaper-indic-machine.rl"
	{act = 1;}
	break;
	case 5:
#line 1 "NONE"
	{te = p+1;}
#line 113 "hb-ot-shaper-indic-machine.rl"
	{act = 5;}
	break;
	case 12:
#line 1 "NONE"
	{te = p+1;}
#line 114 "hb-ot-shaper-indic-machine.rl"
	{act = 6;}
	break;
#line 597 "hb-ot-shaper-indic-machine.hh"
	}

_again:
	switch ( _indic_syllable_machine_to_state_actions[cs] ) {
	case 9:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 606 "hb-ot-shaper-indic-machine.hh"
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

#line 146 "hb-ot-shaper-indic-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPER_INDIC_MACHINE_HH */
