
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


#line 38 "hb-ot-shape-complex-use-machine.hh"
static const unsigned char _use_syllable_machine_trans_keys[] = {
	12u, 48u, 1u, 15u, 1u, 1u, 12u, 48u, 1u, 1u, 0u, 48u, 21u, 21u, 11u, 48u, 
	11u, 48u, 1u, 15u, 1u, 1u, 11u, 48u, 22u, 48u, 23u, 48u, 24u, 47u, 25u, 47u, 
	26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 1u, 1u, 24u, 48u, 
	23u, 48u, 23u, 48u, 23u, 48u, 22u, 48u, 22u, 48u, 22u, 48u, 22u, 48u, 11u, 48u, 
	1u, 48u, 11u, 48u, 13u, 21u, 4u, 4u, 13u, 13u, 11u, 48u, 11u, 48u, 41u, 42u, 
	42u, 42u, 11u, 48u, 11u, 48u, 22u, 48u, 23u, 48u, 24u, 47u, 25u, 47u, 26u, 47u, 
	45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 24u, 48u, 23u, 48u, 23u, 48u, 
	23u, 48u, 22u, 48u, 22u, 48u, 22u, 48u, 22u, 48u, 11u, 48u, 1u, 48u, 1u, 15u, 
	4u, 4u, 13u, 21u, 13u, 13u, 12u, 48u, 1u, 48u, 11u, 48u, 41u, 42u, 42u, 42u, 
	21u, 42u, 1u, 5u, 0
};

static const char _use_syllable_machine_key_spans[] = {
	37, 15, 1, 37, 1, 49, 1, 38, 
	38, 15, 1, 38, 27, 26, 24, 23, 
	22, 2, 1, 25, 25, 25, 1, 25, 
	26, 26, 26, 27, 27, 27, 27, 38, 
	48, 38, 9, 1, 1, 38, 38, 2, 
	1, 38, 38, 27, 26, 24, 23, 22, 
	2, 1, 25, 25, 25, 25, 26, 26, 
	26, 27, 27, 27, 27, 38, 48, 15, 
	1, 9, 1, 37, 48, 38, 2, 1, 
	22, 5
};

static const short _use_syllable_machine_index_offsets[] = {
	0, 38, 54, 56, 94, 96, 146, 148, 
	187, 226, 242, 244, 283, 311, 338, 363, 
	387, 410, 413, 415, 441, 467, 493, 495, 
	521, 548, 575, 602, 630, 658, 686, 714, 
	753, 802, 841, 851, 853, 855, 894, 933, 
	936, 938, 977, 1016, 1044, 1071, 1096, 1120, 
	1143, 1146, 1148, 1174, 1200, 1226, 1252, 1279, 
	1306, 1333, 1361, 1389, 1417, 1445, 1484, 1533, 
	1549, 1551, 1561, 1563, 1601, 1650, 1689, 1692, 
	1694, 1717
};

static const char _use_syllable_machine_indicies[] = {
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 1, 0, 3, 2, 
	2, 2, 2, 2, 2, 2, 2, 2, 
	2, 2, 2, 2, 4, 2, 3, 2, 
	6, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	6, 5, 5, 5, 6, 5, 7, 5, 
	8, 9, 10, 8, 11, 12, 10, 10, 
	10, 10, 10, 3, 13, 14, 10, 15, 
	8, 8, 16, 17, 10, 10, 18, 19, 
	20, 21, 22, 23, 24, 18, 25, 26, 
	27, 28, 29, 30, 10, 31, 32, 33, 
	10, 34, 35, 36, 37, 38, 39, 40, 
	13, 10, 42, 41, 44, 1, 43, 43, 
	45, 43, 43, 43, 43, 43, 46, 47, 
	48, 49, 50, 51, 52, 53, 47, 54, 
	46, 55, 56, 57, 58, 43, 59, 60, 
	61, 43, 43, 43, 43, 62, 63, 64, 
	65, 1, 43, 44, 1, 43, 43, 45, 
	43, 43, 43, 43, 43, 66, 47, 48, 
	49, 50, 51, 52, 53, 47, 54, 55, 
	55, 56, 57, 58, 43, 59, 60, 61, 
	43, 43, 43, 43, 62, 63, 64, 65, 
	1, 43, 44, 67, 67, 67, 67, 67, 
	67, 67, 67, 67, 67, 67, 67, 67, 
	68, 67, 44, 67, 44, 1, 43, 43, 
	45, 43, 43, 43, 43, 43, 43, 47, 
	48, 49, 50, 51, 52, 53, 47, 54, 
	55, 55, 56, 57, 58, 43, 59, 60, 
	61, 43, 43, 43, 43, 62, 63, 64, 
	65, 1, 43, 47, 48, 49, 50, 51, 
	43, 43, 43, 43, 43, 43, 56, 57, 
	58, 43, 59, 60, 61, 43, 43, 43, 
	43, 48, 63, 64, 65, 69, 43, 48, 
	49, 50, 51, 43, 43, 43, 43, 43, 
	43, 43, 43, 43, 43, 59, 60, 61, 
	43, 43, 43, 43, 43, 63, 64, 65, 
	69, 43, 49, 50, 51, 43, 43, 43, 
	43, 43, 43, 43, 43, 43, 43, 43, 
	43, 43, 43, 43, 43, 43, 43, 63, 
	64, 65, 43, 50, 51, 43, 43, 43, 
	43, 43, 43, 43, 43, 43, 43, 43, 
	43, 43, 43, 43, 43, 43, 43, 63, 
	64, 65, 43, 51, 43, 43, 43, 43, 
	43, 43, 43, 43, 43, 43, 43, 43, 
	43, 43, 43, 43, 43, 43, 63, 64, 
	65, 43, 63, 64, 43, 64, 43, 49, 
	50, 51, 43, 43, 43, 43, 43, 43, 
	43, 43, 43, 43, 59, 60, 61, 43, 
	43, 43, 43, 43, 63, 64, 65, 69, 
	43, 49, 50, 51, 43, 43, 43, 43, 
	43, 43, 43, 43, 43, 43, 43, 60, 
	61, 43, 43, 43, 43, 43, 63, 64, 
	65, 69, 43, 49, 50, 51, 43, 43, 
	43, 43, 43, 43, 43, 43, 43, 43, 
	43, 43, 61, 43, 43, 43, 43, 43, 
	63, 64, 65, 69, 43, 71, 70, 49, 
	50, 51, 43, 43, 43, 43, 43, 43, 
	43, 43, 43, 43, 43, 43, 43, 43, 
	43, 43, 43, 43, 63, 64, 65, 69, 
	43, 48, 49, 50, 51, 43, 43, 43, 
	43, 43, 43, 56, 57, 58, 43, 59, 
	60, 61, 43, 43, 43, 43, 48, 63, 
	64, 65, 69, 43, 48, 49, 50, 51, 
	43, 43, 43, 43, 43, 43, 43, 57, 
	58, 43, 59, 60, 61, 43, 43, 43, 
	43, 48, 63, 64, 65, 69, 43, 48, 
	49, 50, 51, 43, 43, 43, 43, 43, 
	43, 43, 43, 58, 43, 59, 60, 61, 
	43, 43, 43, 43, 48, 63, 64, 65, 
	69, 43, 47, 48, 49, 50, 51, 43, 
	53, 47, 43, 43, 43, 56, 57, 58, 
	43, 59, 60, 61, 43, 43, 43, 43, 
	48, 63, 64, 65, 69, 43, 47, 48, 
	49, 50, 51, 43, 72, 47, 43, 43, 
	43, 56, 57, 58, 43, 59, 60, 61, 
	43, 43, 43, 43, 48, 63, 64, 65, 
	69, 43, 47, 48, 49, 50, 51, 43, 
	43, 47, 43, 43, 43, 56, 57, 58, 
	43, 59, 60, 61, 43, 43, 43, 43, 
	48, 63, 64, 65, 69, 43, 47, 48, 
	49, 50, 51, 52, 53, 47, 43, 43, 
	43, 56, 57, 58, 43, 59, 60, 61, 
	43, 43, 43, 43, 48, 63, 64, 65, 
	69, 43, 44, 1, 43, 43, 45, 43, 
	43, 43, 43, 43, 43, 47, 48, 49, 
	50, 51, 52, 53, 47, 54, 43, 55, 
	56, 57, 58, 43, 59, 60, 61, 43, 
	43, 43, 43, 62, 63, 64, 65, 1, 
	43, 44, 67, 67, 67, 67, 67, 67, 
	67, 67, 67, 67, 67, 67, 67, 68, 
	67, 67, 67, 67, 67, 67, 67, 48, 
	49, 50, 51, 67, 67, 67, 67, 67, 
	67, 67, 67, 67, 67, 59, 60, 61, 
	67, 67, 67, 67, 67, 63, 64, 65, 
	69, 67, 44, 1, 43, 43, 45, 43, 
	43, 43, 43, 43, 43, 47, 48, 49, 
	50, 51, 52, 53, 47, 54, 46, 55, 
	56, 57, 58, 43, 59, 60, 61, 43, 
	43, 43, 43, 62, 63, 64, 65, 1, 
	43, 74, 73, 73, 73, 73, 73, 73, 
	73, 75, 73, 11, 76, 74, 73, 44, 
	1, 43, 43, 45, 43, 43, 43, 43, 
	43, 77, 47, 48, 49, 50, 51, 52, 
	53, 47, 54, 46, 55, 56, 57, 58, 
	43, 59, 60, 61, 43, 78, 79, 43, 
	62, 63, 64, 65, 1, 43, 44, 1, 
	43, 43, 45, 43, 43, 43, 43, 43, 
	43, 47, 48, 49, 50, 51, 52, 53, 
	47, 54, 46, 55, 56, 57, 58, 43, 
	59, 60, 61, 43, 78, 79, 43, 62, 
	63, 64, 65, 1, 43, 78, 79, 80, 
	79, 80, 3, 6, 81, 81, 82, 81, 
	81, 81, 81, 81, 83, 18, 19, 20, 
	21, 22, 23, 24, 18, 25, 27, 27, 
	28, 29, 30, 81, 31, 32, 33, 81, 
	81, 81, 81, 37, 38, 39, 40, 6, 
	81, 3, 6, 81, 81, 82, 81, 81, 
	81, 81, 81, 81, 18, 19, 20, 21, 
	22, 23, 24, 18, 25, 27, 27, 28, 
	29, 30, 81, 31, 32, 33, 81, 81, 
	81, 81, 37, 38, 39, 40, 6, 81, 
	18, 19, 20, 21, 22, 81, 81, 81, 
	81, 81, 81, 28, 29, 30, 81, 31, 
	32, 33, 81, 81, 81, 81, 19, 38, 
	39, 40, 84, 81, 19, 20, 21, 22, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 31, 32, 33, 81, 81, 81, 
	81, 81, 38, 39, 40, 84, 81, 20, 
	21, 22, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 38, 39, 40, 81, 
	21, 22, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 38, 39, 40, 81, 
	22, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 38, 39, 40, 81, 38, 
	39, 81, 39, 81, 20, 21, 22, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 31, 32, 33, 81, 81, 81, 81, 
	81, 38, 39, 40, 84, 81, 20, 21, 
	22, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 32, 33, 81, 81, 
	81, 81, 81, 38, 39, 40, 84, 81, 
	20, 21, 22, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 33, 
	81, 81, 81, 81, 81, 38, 39, 40, 
	84, 81, 20, 21, 22, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 38, 
	39, 40, 84, 81, 19, 20, 21, 22, 
	81, 81, 81, 81, 81, 81, 28, 29, 
	30, 81, 31, 32, 33, 81, 81, 81, 
	81, 19, 38, 39, 40, 84, 81, 19, 
	20, 21, 22, 81, 81, 81, 81, 81, 
	81, 81, 29, 30, 81, 31, 32, 33, 
	81, 81, 81, 81, 19, 38, 39, 40, 
	84, 81, 19, 20, 21, 22, 81, 81, 
	81, 81, 81, 81, 81, 81, 30, 81, 
	31, 32, 33, 81, 81, 81, 81, 19, 
	38, 39, 40, 84, 81, 18, 19, 20, 
	21, 22, 81, 24, 18, 81, 81, 81, 
	28, 29, 30, 81, 31, 32, 33, 81, 
	81, 81, 81, 19, 38, 39, 40, 84, 
	81, 18, 19, 20, 21, 22, 81, 85, 
	18, 81, 81, 81, 28, 29, 30, 81, 
	31, 32, 33, 81, 81, 81, 81, 19, 
	38, 39, 40, 84, 81, 18, 19, 20, 
	21, 22, 81, 81, 18, 81, 81, 81, 
	28, 29, 30, 81, 31, 32, 33, 81, 
	81, 81, 81, 19, 38, 39, 40, 84, 
	81, 18, 19, 20, 21, 22, 23, 24, 
	18, 81, 81, 81, 28, 29, 30, 81, 
	31, 32, 33, 81, 81, 81, 81, 19, 
	38, 39, 40, 84, 81, 3, 6, 81, 
	81, 82, 81, 81, 81, 81, 81, 81, 
	18, 19, 20, 21, 22, 23, 24, 18, 
	25, 81, 27, 28, 29, 30, 81, 31, 
	32, 33, 81, 81, 81, 81, 37, 38, 
	39, 40, 6, 81, 3, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 4, 81, 81, 81, 81, 81, 
	81, 81, 19, 20, 21, 22, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	31, 32, 33, 81, 81, 81, 81, 81, 
	38, 39, 40, 84, 81, 3, 86, 86, 
	86, 86, 86, 86, 86, 86, 86, 86, 
	86, 86, 86, 4, 86, 87, 81, 14, 
	81, 81, 81, 81, 81, 81, 81, 88, 
	81, 14, 81, 6, 86, 86, 86, 86, 
	86, 86, 86, 86, 86, 86, 86, 86, 
	86, 86, 86, 86, 86, 86, 86, 86, 
	86, 86, 86, 86, 86, 86, 86, 86, 
	86, 86, 86, 6, 86, 86, 86, 6, 
	86, 9, 81, 81, 81, 9, 81, 81, 
	81, 81, 81, 3, 6, 14, 81, 82, 
	81, 81, 81, 81, 81, 81, 18, 19, 
	20, 21, 22, 23, 24, 18, 25, 26, 
	27, 28, 29, 30, 81, 31, 32, 33, 
	81, 34, 35, 81, 37, 38, 39, 40, 
	6, 81, 3, 6, 81, 81, 82, 81, 
	81, 81, 81, 81, 81, 18, 19, 20, 
	21, 22, 23, 24, 18, 25, 26, 27, 
	28, 29, 30, 81, 31, 32, 33, 81, 
	81, 81, 81, 37, 38, 39, 40, 6, 
	81, 34, 35, 81, 35, 81, 78, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 78, 79, 80, 9, 86, 86, 
	86, 9, 86, 0
};

static const char _use_syllable_machine_trans_targs[] = {
	5, 9, 5, 41, 2, 5, 1, 53, 
	6, 7, 5, 34, 37, 63, 64, 67, 
	68, 72, 43, 44, 45, 46, 47, 57, 
	58, 60, 69, 61, 54, 55, 56, 50, 
	51, 52, 70, 71, 73, 62, 48, 49, 
	5, 5, 5, 5, 8, 0, 33, 12, 
	13, 14, 15, 16, 27, 28, 30, 31, 
	24, 25, 26, 19, 20, 21, 32, 17, 
	18, 5, 11, 5, 10, 22, 5, 23, 
	29, 5, 35, 36, 5, 38, 39, 40, 
	5, 5, 3, 42, 4, 59, 5, 65, 
	66
};

static const char _use_syllable_machine_trans_actions[] = {
	1, 0, 2, 3, 0, 4, 0, 5, 
	0, 5, 8, 0, 5, 9, 0, 9, 
	3, 0, 5, 5, 0, 0, 0, 5, 
	5, 5, 3, 3, 5, 5, 5, 5, 
	5, 5, 0, 0, 0, 3, 0, 0, 
	10, 11, 12, 13, 5, 0, 5, 0, 
	0, 0, 0, 0, 0, 0, 0, 5, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 14, 5, 15, 0, 0, 16, 0, 
	0, 17, 0, 0, 18, 5, 0, 0, 
	19, 20, 0, 3, 0, 5, 21, 0, 
	0
};

static const char _use_syllable_machine_to_state_actions[] = {
	0, 0, 0, 0, 0, 6, 0, 0, 
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

static const char _use_syllable_machine_from_state_actions[] = {
	0, 0, 0, 0, 0, 7, 0, 0, 
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

static const short _use_syllable_machine_eof_trans[] = {
	1, 3, 3, 6, 6, 0, 42, 44, 
	44, 68, 68, 44, 44, 44, 44, 44, 
	44, 44, 44, 44, 44, 44, 71, 44, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	68, 44, 74, 77, 74, 44, 44, 81, 
	81, 82, 82, 82, 82, 82, 82, 82, 
	82, 82, 82, 82, 82, 82, 82, 82, 
	82, 82, 82, 82, 82, 82, 82, 87, 
	82, 82, 82, 87, 82, 82, 82, 82, 
	81, 87
};

static const int use_syllable_machine_start = 5;
static const int use_syllable_machine_first_final = 5;
static const int use_syllable_machine_error = -1;

static const int use_syllable_machine_en_main = 5;


#line 38 "hb-ot-shape-complex-use-machine.rl"



#line 162 "hb-ot-shape-complex-use-machine.rl"


#define found_syllable(syllable_type) \
  HB_STMT_START { \
    if (0) fprintf (stderr, "syllable %d..%d %s\n", ts, te, #syllable_type); \
    for (unsigned int i = ts; i < te; i++) \
      info[i].syllable() = (syllable_serial << 4) | use_##syllable_type; \
    syllable_serial++; \
    if (unlikely (syllable_serial == 16)) syllable_serial = 1; \
  } HB_STMT_END

static void
find_syllables_use (hb_buffer_t *buffer)
{
  unsigned int p, pe, eof, ts, te, act;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  
#line 396 "hb-ot-shape-complex-use-machine.hh"
	{
	cs = use_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 182 "hb-ot-shape-complex-use-machine.rl"


  p = 0;
  pe = eof = buffer->len;

  unsigned int syllable_serial = 1;
  
#line 412 "hb-ot-shape-complex-use-machine.hh"
	{
	int _slen;
	int _trans;
	const unsigned char *_keys;
	const char *_inds;
	if ( p == pe )
		goto _test_eof;
_resume:
	switch ( _use_syllable_machine_from_state_actions[cs] ) {
	case 7:
#line 1 "NONE"
	{ts = p;}
	break;
#line 426 "hb-ot-shape-complex-use-machine.hh"
	}

	_keys = _use_syllable_machine_trans_keys + (cs<<1);
	_inds = _use_syllable_machine_indicies + _use_syllable_machine_index_offsets[cs];

	_slen = _use_syllable_machine_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=( info[p].use_category()) &&
		( info[p].use_category()) <= _keys[1] ?
		( info[p].use_category()) - _keys[0] : _slen ];

_eof_trans:
	cs = _use_syllable_machine_trans_targs[_trans];

	if ( _use_syllable_machine_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _use_syllable_machine_trans_actions[_trans] ) {
	case 5:
#line 1 "NONE"
	{te = p+1;}
	break;
	case 12:
#line 150 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (independent_cluster); }}
	break;
	case 14:
#line 153 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (standard_cluster); }}
	break;
	case 10:
#line 157 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (broken_cluster); }}
	break;
	case 8:
#line 158 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (non_cluster); }}
	break;
	case 11:
#line 150 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (independent_cluster); }}
	break;
	case 15:
#line 151 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (virama_terminated_cluster); }}
	break;
	case 16:
#line 152 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (sakot_terminated_cluster); }}
	break;
	case 13:
#line 153 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (standard_cluster); }}
	break;
	case 18:
#line 154 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (number_joiner_terminated_cluster); }}
	break;
	case 17:
#line 155 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (numeral_cluster); }}
	break;
	case 19:
#line 156 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (symbol_cluster); }}
	break;
	case 20:
#line 157 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (broken_cluster); }}
	break;
	case 21:
#line 158 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (non_cluster); }}
	break;
	case 1:
#line 153 "hb-ot-shape-complex-use-machine.rl"
	{{p = ((te))-1;}{ found_syllable (standard_cluster); }}
	break;
	case 4:
#line 157 "hb-ot-shape-complex-use-machine.rl"
	{{p = ((te))-1;}{ found_syllable (broken_cluster); }}
	break;
	case 2:
#line 1 "NONE"
	{	switch( act ) {
	case 8:
	{{p = ((te))-1;} found_syllable (broken_cluster); }
	break;
	case 9:
	{{p = ((te))-1;} found_syllable (non_cluster); }
	break;
	}
	}
	break;
	case 3:
#line 1 "NONE"
	{te = p+1;}
#line 157 "hb-ot-shape-complex-use-machine.rl"
	{act = 8;}
	break;
	case 9:
#line 1 "NONE"
	{te = p+1;}
#line 158 "hb-ot-shape-complex-use-machine.rl"
	{act = 9;}
	break;
#line 532 "hb-ot-shape-complex-use-machine.hh"
	}

_again:
	switch ( _use_syllable_machine_to_state_actions[cs] ) {
	case 6:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 541 "hb-ot-shape-complex-use-machine.hh"
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

#line 190 "hb-ot-shape-complex-use-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH */
