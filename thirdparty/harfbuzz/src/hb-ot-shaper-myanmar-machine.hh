
#line 1 "hb-ot-shaper-myanmar-machine.rl"
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

#ifndef HB_OT_SHAPER_MYANMAR_MACHINE_HH
#define HB_OT_SHAPER_MYANMAR_MACHINE_HH

#include "hb.hh"

#include "hb-ot-layout.hh"
#include "hb-ot-shaper-indic.hh"

/* buffer var allocations */
#define myanmar_category() ot_shaper_var_u8_category() /* myanmar_category_t */
#define myanmar_position() ot_shaper_var_u8_auxiliary() /* myanmar_position_t */

using myanmar_category_t = unsigned;
using myanmar_position_t = ot_position_t;

#define M_Cat(Cat) myanmar_syllable_machine_ex_##Cat

enum myanmar_syllable_type_t {
  myanmar_consonant_syllable,
  myanmar_broken_cluster,
  myanmar_non_myanmar_cluster,
};


#line 51 "hb-ot-shaper-myanmar-machine.hh"
#define myanmar_syllable_machine_ex_A 9u
#define myanmar_syllable_machine_ex_As 32u
#define myanmar_syllable_machine_ex_C 1u
#define myanmar_syllable_machine_ex_CS 18u
#define myanmar_syllable_machine_ex_DB 3u
#define myanmar_syllable_machine_ex_DOTTEDCIRCLE 11u
#define myanmar_syllable_machine_ex_GB 10u
#define myanmar_syllable_machine_ex_H 4u
#define myanmar_syllable_machine_ex_IV 2u
#define myanmar_syllable_machine_ex_MH 35u
#define myanmar_syllable_machine_ex_ML 41u
#define myanmar_syllable_machine_ex_MR 36u
#define myanmar_syllable_machine_ex_MW 37u
#define myanmar_syllable_machine_ex_MY 38u
#define myanmar_syllable_machine_ex_PT 39u
#define myanmar_syllable_machine_ex_Ra 15u
#define myanmar_syllable_machine_ex_SM 8u
#define myanmar_syllable_machine_ex_VAbv 20u
#define myanmar_syllable_machine_ex_VBlw 21u
#define myanmar_syllable_machine_ex_VPre 22u
#define myanmar_syllable_machine_ex_VPst 23u
#define myanmar_syllable_machine_ex_VS 40u
#define myanmar_syllable_machine_ex_ZWJ 6u
#define myanmar_syllable_machine_ex_ZWNJ 5u


#line 76 "hb-ot-shaper-myanmar-machine.hh"
static const unsigned char _myanmar_syllable_machine_trans_keys[] = {
	1u, 41u, 3u, 41u, 5u, 39u, 5u, 8u, 3u, 41u, 3u, 39u, 3u, 39u, 5u, 39u, 
	5u, 39u, 3u, 39u, 3u, 39u, 3u, 41u, 5u, 39u, 1u, 15u, 3u, 39u, 3u, 39u, 
	3u, 40u, 3u, 39u, 3u, 41u, 3u, 41u, 3u, 39u, 3u, 41u, 3u, 41u, 3u, 41u, 
	3u, 41u, 3u, 41u, 5u, 39u, 5u, 8u, 3u, 41u, 3u, 39u, 3u, 39u, 5u, 39u, 
	5u, 39u, 3u, 39u, 3u, 39u, 3u, 41u, 5u, 39u, 1u, 15u, 3u, 41u, 3u, 39u, 
	3u, 39u, 3u, 40u, 3u, 39u, 3u, 41u, 3u, 41u, 3u, 39u, 3u, 41u, 3u, 41u, 
	3u, 41u, 3u, 41u, 3u, 41u, 3u, 41u, 3u, 41u, 1u, 41u, 1u, 15u, 0
};

static const char _myanmar_syllable_machine_key_spans[] = {
	41, 39, 35, 4, 39, 37, 37, 35, 
	35, 37, 37, 39, 35, 15, 37, 37, 
	38, 37, 39, 39, 37, 39, 39, 39, 
	39, 39, 35, 4, 39, 37, 37, 35, 
	35, 37, 37, 39, 35, 15, 39, 37, 
	37, 38, 37, 39, 39, 37, 39, 39, 
	39, 39, 39, 39, 39, 41, 15
};

static const short _myanmar_syllable_machine_index_offsets[] = {
	0, 42, 82, 118, 123, 163, 201, 239, 
	275, 311, 349, 387, 427, 463, 479, 517, 
	555, 594, 632, 672, 712, 750, 790, 830, 
	870, 910, 950, 986, 991, 1031, 1069, 1107, 
	1143, 1179, 1217, 1255, 1295, 1331, 1347, 1387, 
	1425, 1463, 1502, 1540, 1580, 1620, 1658, 1698, 
	1738, 1778, 1818, 1858, 1898, 1938, 1980
};

static const char _myanmar_syllable_machine_indicies[] = {
	1, 1, 2, 3, 4, 4, 0, 5, 
	6, 1, 1, 0, 0, 0, 7, 0, 
	0, 8, 0, 9, 10, 11, 12, 0, 
	0, 0, 0, 0, 0, 0, 0, 13, 
	0, 0, 14, 15, 16, 17, 18, 19, 
	20, 0, 22, 23, 24, 24, 21, 25, 
	26, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 27, 28, 29, 30, 21, 
	21, 21, 21, 21, 21, 21, 21, 31, 
	21, 21, 32, 33, 34, 35, 36, 37, 
	38, 21, 24, 24, 21, 25, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 30, 21, 21, 21, 
	21, 21, 21, 21, 21, 39, 21, 21, 
	21, 21, 21, 21, 36, 21, 24, 24, 
	21, 25, 21, 22, 21, 24, 24, 21, 
	25, 26, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 40, 21, 21, 30, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	41, 21, 21, 42, 21, 21, 21, 36, 
	21, 41, 21, 22, 21, 24, 24, 21, 
	25, 26, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 30, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 36, 
	21, 43, 21, 24, 24, 21, 25, 36, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 44, 21, 
	21, 21, 21, 21, 21, 36, 21, 24, 
	24, 21, 25, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 44, 21, 21, 21, 21, 21, 
	21, 36, 21, 24, 24, 21, 25, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 36, 21, 22, 
	21, 24, 24, 21, 25, 26, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	40, 21, 21, 30, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 36, 21, 22, 21, 24, 
	24, 21, 25, 26, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 40, 21, 
	21, 30, 21, 21, 21, 21, 21, 21, 
	21, 21, 41, 21, 21, 21, 21, 21, 
	21, 36, 21, 22, 21, 24, 24, 21, 
	25, 26, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 40, 21, 21, 30, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	41, 21, 21, 21, 21, 21, 21, 36, 
	21, 41, 21, 24, 24, 21, 25, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 30, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 36, 21, 1, 
	1, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 1, 21, 22, 
	21, 24, 24, 21, 25, 26, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	27, 28, 21, 30, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 36, 21, 22, 21, 24, 
	24, 21, 25, 26, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 28, 
	21, 30, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 36, 21, 22, 21, 24, 24, 21, 
	25, 26, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 27, 28, 29, 30, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 36, 
	45, 21, 22, 21, 24, 24, 21, 25, 
	26, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 27, 28, 29, 30, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 36, 21, 
	22, 21, 24, 24, 21, 25, 26, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 27, 28, 29, 30, 21, 21, 21, 
	21, 21, 21, 21, 21, 31, 21, 21, 
	32, 33, 34, 35, 36, 21, 38, 21, 
	22, 21, 24, 24, 21, 25, 26, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 27, 28, 29, 30, 21, 21, 21, 
	21, 21, 21, 21, 21, 45, 21, 21, 
	21, 21, 21, 21, 36, 21, 38, 21, 
	22, 21, 24, 24, 21, 25, 26, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 27, 28, 29, 30, 21, 21, 21, 
	21, 21, 21, 21, 21, 45, 21, 21, 
	21, 21, 21, 21, 36, 21, 22, 21, 
	24, 24, 21, 25, 26, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 27, 
	28, 29, 30, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 32, 21, 
	34, 21, 36, 21, 38, 21, 22, 21, 
	24, 24, 21, 25, 26, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 27, 
	28, 29, 30, 21, 21, 21, 21, 21, 
	21, 21, 21, 45, 21, 21, 32, 21, 
	21, 21, 36, 21, 38, 21, 22, 21, 
	24, 24, 21, 25, 26, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 27, 
	28, 29, 30, 21, 21, 21, 21, 21, 
	21, 21, 21, 46, 21, 21, 32, 33, 
	34, 21, 36, 21, 38, 21, 22, 21, 
	24, 24, 21, 25, 26, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 27, 
	28, 29, 30, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 32, 33, 
	34, 21, 36, 21, 38, 21, 22, 23, 
	24, 24, 21, 25, 26, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 27, 
	28, 29, 30, 21, 21, 21, 21, 21, 
	21, 21, 21, 31, 21, 21, 32, 33, 
	34, 35, 36, 21, 38, 21, 48, 48, 
	47, 5, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 49, 47, 47, 47, 47, 47, 47, 
	18, 47, 48, 48, 47, 5, 47, 2, 
	47, 48, 48, 47, 5, 6, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	50, 47, 47, 12, 47, 47, 47, 47, 
	47, 47, 47, 47, 51, 47, 47, 52, 
	47, 47, 47, 18, 47, 51, 47, 2, 
	47, 48, 48, 47, 5, 6, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 12, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 18, 47, 53, 47, 48, 
	48, 47, 5, 18, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 54, 47, 47, 47, 47, 47, 
	47, 18, 47, 48, 48, 47, 5, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 54, 47, 
	47, 47, 47, 47, 47, 18, 47, 48, 
	48, 47, 5, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 18, 47, 2, 47, 48, 48, 47, 
	5, 6, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 50, 47, 47, 12, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 18, 
	47, 2, 47, 48, 48, 47, 5, 6, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 50, 47, 47, 12, 47, 47, 
	47, 47, 47, 47, 47, 47, 51, 47, 
	47, 47, 47, 47, 47, 18, 47, 2, 
	47, 48, 48, 47, 5, 6, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	50, 47, 47, 12, 47, 47, 47, 47, 
	47, 47, 47, 47, 51, 47, 47, 47, 
	47, 47, 47, 18, 47, 51, 47, 48, 
	48, 47, 5, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 12, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 18, 47, 55, 55, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 55, 47, 2, 3, 48, 48, 47, 
	5, 6, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 9, 10, 11, 12, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	13, 47, 47, 14, 15, 16, 17, 18, 
	19, 20, 47, 2, 47, 48, 48, 47, 
	5, 6, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 9, 10, 47, 12, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 18, 
	47, 2, 47, 48, 48, 47, 5, 6, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 10, 47, 12, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 18, 47, 2, 
	47, 48, 48, 47, 5, 6, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	9, 10, 11, 12, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 18, 56, 47, 2, 47, 
	48, 48, 47, 5, 6, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 9, 
	10, 11, 12, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 18, 47, 2, 47, 48, 48, 
	47, 5, 6, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 9, 10, 11, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 13, 47, 47, 14, 15, 16, 17, 
	18, 47, 20, 47, 2, 47, 48, 48, 
	47, 5, 6, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 9, 10, 11, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 56, 47, 47, 47, 47, 47, 47, 
	18, 47, 20, 47, 2, 47, 48, 48, 
	47, 5, 6, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 9, 10, 11, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 56, 47, 47, 47, 47, 47, 47, 
	18, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 14, 47, 16, 47, 18, 47, 
	20, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 56, 
	47, 47, 14, 47, 47, 47, 18, 47, 
	20, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 57, 
	47, 47, 14, 15, 16, 47, 18, 47, 
	20, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 14, 15, 16, 47, 18, 47, 
	20, 47, 2, 3, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 13, 
	47, 47, 14, 15, 16, 17, 18, 47, 
	20, 47, 22, 23, 24, 24, 21, 25, 
	26, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 27, 28, 29, 30, 21, 
	21, 21, 21, 21, 21, 21, 21, 58, 
	21, 21, 32, 33, 34, 35, 36, 37, 
	38, 21, 22, 59, 24, 24, 21, 25, 
	26, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 27, 28, 29, 30, 21, 
	21, 21, 21, 21, 21, 21, 21, 31, 
	21, 21, 32, 33, 34, 35, 36, 21, 
	38, 21, 1, 1, 2, 3, 48, 48, 
	47, 5, 6, 1, 1, 47, 47, 47, 
	1, 47, 47, 47, 47, 9, 10, 11, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 13, 47, 47, 14, 15, 16, 17, 
	18, 19, 20, 47, 1, 1, 60, 60, 
	60, 60, 60, 60, 60, 1, 1, 60, 
	60, 60, 1, 60, 0
};

static const char _myanmar_syllable_machine_trans_targs[] = {
	0, 1, 26, 37, 0, 27, 29, 51, 
	54, 39, 40, 41, 28, 43, 44, 46, 
	47, 48, 30, 50, 45, 0, 2, 13, 
	0, 3, 5, 14, 15, 16, 4, 18, 
	19, 21, 22, 23, 6, 25, 20, 12, 
	9, 10, 11, 7, 8, 17, 24, 0, 
	0, 36, 33, 34, 35, 31, 32, 38, 
	42, 49, 52, 53, 0
};

static const char _myanmar_syllable_machine_trans_actions[] = {
	3, 0, 0, 0, 4, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 5, 0, 0, 
	6, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 7, 
	8, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 9
};

static const char _myanmar_syllable_machine_to_state_actions[] = {
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0
};

static const char _myanmar_syllable_machine_from_state_actions[] = {
	2, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0
};

static const short _myanmar_syllable_machine_eof_trans[] = {
	0, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 22, 22, 48, 61
};

static const int myanmar_syllable_machine_start = 0;
static const int myanmar_syllable_machine_first_final = 0;
static const int myanmar_syllable_machine_error = -1;

static const int myanmar_syllable_machine_en_main = 0;


#line 55 "hb-ot-shaper-myanmar-machine.rl"



#line 117 "hb-ot-shaper-myanmar-machine.rl"


#define found_syllable(syllable_type) \
  HB_STMT_START { \
    if (0) fprintf (stderr, "syllable %d..%d %s\n", ts, te, #syllable_type); \
    for (unsigned int i = ts; i < te; i++) \
      info[i].syllable() = (syllable_serial << 4) | syllable_type; \
    syllable_serial++; \
    if (unlikely (syllable_serial == 16)) syllable_serial = 1; \
  } HB_STMT_END

inline void
find_syllables_myanmar (hb_buffer_t *buffer)
{
  unsigned int p, pe, eof, ts, te, act HB_UNUSED;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  
#line 436 "hb-ot-shaper-myanmar-machine.hh"
	{
	cs = myanmar_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 137 "hb-ot-shaper-myanmar-machine.rl"


  p = 0;
  pe = eof = buffer->len;

  unsigned int syllable_serial = 1;
  
#line 448 "hb-ot-shaper-myanmar-machine.hh"
	{
	int _slen;
	int _trans;
	const unsigned char *_keys;
	const char *_inds;
	if ( p == pe )
		goto _test_eof;
_resume:
	switch ( _myanmar_syllable_machine_from_state_actions[cs] ) {
	case 2:
#line 1 "NONE"
	{ts = p;}
	break;
#line 460 "hb-ot-shaper-myanmar-machine.hh"
	}

	_keys = _myanmar_syllable_machine_trans_keys + (cs<<1);
	_inds = _myanmar_syllable_machine_indicies + _myanmar_syllable_machine_index_offsets[cs];

	_slen = _myanmar_syllable_machine_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=( info[p].myanmar_category()) &&
		( info[p].myanmar_category()) <= _keys[1] ?
		( info[p].myanmar_category()) - _keys[0] : _slen ];

_eof_trans:
	cs = _myanmar_syllable_machine_trans_targs[_trans];

	if ( _myanmar_syllable_machine_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _myanmar_syllable_machine_trans_actions[_trans] ) {
	case 6:
#line 110 "hb-ot-shaper-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_consonant_syllable); }}
	break;
	case 4:
#line 111 "hb-ot-shaper-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_non_myanmar_cluster); }}
	break;
	case 8:
#line 112 "hb-ot-shaper-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }}
	break;
	case 3:
#line 113 "hb-ot-shaper-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_non_myanmar_cluster); }}
	break;
	case 5:
#line 110 "hb-ot-shaper-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (myanmar_consonant_syllable); }}
	break;
	case 7:
#line 112 "hb-ot-shaper-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (myanmar_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }}
	break;
	case 9:
#line 113 "hb-ot-shaper-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (myanmar_non_myanmar_cluster); }}
	break;
#line 498 "hb-ot-shaper-myanmar-machine.hh"
	}

_again:
	switch ( _myanmar_syllable_machine_to_state_actions[cs] ) {
	case 1:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 505 "hb-ot-shaper-myanmar-machine.hh"
	}

	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	if ( _myanmar_syllable_machine_eof_trans[cs] > 0 ) {
		_trans = _myanmar_syllable_machine_eof_trans[cs] - 1;
		goto _eof_trans;
	}
	}

	}

#line 145 "hb-ot-shaper-myanmar-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPER_MYANMAR_MACHINE_HH */
