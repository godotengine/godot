
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


#line 54 "hb-ot-shaper-myanmar-machine.hh"
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
#define myanmar_syllable_machine_ex_SMPst 57u
#define myanmar_syllable_machine_ex_VAbv 20u
#define myanmar_syllable_machine_ex_VBlw 21u
#define myanmar_syllable_machine_ex_VPre 22u
#define myanmar_syllable_machine_ex_VPst 23u
#define myanmar_syllable_machine_ex_VS 40u
#define myanmar_syllable_machine_ex_ZWJ 6u
#define myanmar_syllable_machine_ex_ZWNJ 5u


#line 82 "hb-ot-shaper-myanmar-machine.hh"
static const unsigned char _myanmar_syllable_machine_trans_keys[] = {
	1u, 57u, 3u, 57u, 5u, 57u, 5u, 57u, 3u, 57u, 5u, 57u, 3u, 57u, 3u, 57u, 
	3u, 57u, 3u, 57u, 3u, 57u, 5u, 57u, 1u, 15u, 3u, 57u, 3u, 57u, 3u, 57u, 
	3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 
	3u, 57u, 5u, 57u, 5u, 57u, 3u, 57u, 5u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 
	3u, 57u, 3u, 57u, 5u, 57u, 1u, 15u, 3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 
	3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 3u, 57u, 
	3u, 57u, 3u, 57u, 3u, 57u, 1u, 57u, 1u, 15u, 0
};

static const char _myanmar_syllable_machine_key_spans[] = {
	57, 55, 53, 53, 55, 53, 55, 55, 
	55, 55, 55, 53, 15, 55, 55, 55, 
	55, 55, 55, 55, 55, 55, 55, 55, 
	55, 53, 53, 55, 53, 55, 55, 55, 
	55, 55, 53, 15, 55, 55, 55, 55, 
	55, 55, 55, 55, 55, 55, 55, 55, 
	55, 55, 55, 57, 15
};

static const short _myanmar_syllable_machine_index_offsets[] = {
	0, 58, 114, 168, 222, 278, 332, 388, 
	444, 500, 556, 612, 666, 682, 738, 794, 
	850, 906, 962, 1018, 1074, 1130, 1186, 1242, 
	1298, 1354, 1408, 1462, 1518, 1572, 1628, 1684, 
	1740, 1796, 1852, 1906, 1922, 1978, 2034, 2090, 
	2146, 2202, 2258, 2314, 2370, 2426, 2482, 2538, 
	2594, 2650, 2706, 2762, 2820
};

static const char _myanmar_syllable_machine_indicies[] = {
	1, 1, 2, 3, 4, 4, 0, 5, 
	6, 1, 1, 0, 0, 0, 7, 0, 
	0, 8, 0, 9, 10, 11, 12, 0, 
	0, 0, 0, 0, 0, 0, 0, 13, 
	0, 0, 14, 15, 16, 17, 18, 19, 
	20, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	21, 0, 23, 24, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 32, 
	22, 22, 33, 34, 35, 36, 37, 38, 
	39, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 25, 25, 22, 26, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 31, 22, 22, 22, 
	22, 22, 22, 22, 22, 40, 22, 22, 
	22, 22, 22, 22, 37, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 26, 22, 
	25, 25, 22, 26, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 37, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 26, 22, 41, 22, 
	25, 25, 22, 26, 37, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 26, 22, 22, 22, 22, 
	22, 22, 37, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 26, 22, 25, 25, 
	22, 26, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 26, 22, 22, 22, 22, 22, 22, 
	37, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 26, 22, 23, 22, 25, 25, 
	22, 26, 27, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 42, 22, 22, 
	31, 22, 22, 22, 22, 22, 22, 22, 
	22, 43, 22, 22, 44, 22, 22, 22, 
	37, 22, 43, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 26, 22, 23, 22, 25, 25, 
	22, 26, 27, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	31, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	37, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 26, 22, 23, 22, 25, 25, 
	22, 26, 27, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 42, 22, 22, 
	31, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	37, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 26, 22, 23, 22, 25, 25, 
	22, 26, 27, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 42, 22, 22, 
	31, 22, 22, 22, 22, 22, 22, 22, 
	22, 43, 22, 22, 22, 22, 22, 22, 
	37, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 26, 22, 23, 22, 25, 25, 
	22, 26, 27, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 42, 22, 22, 
	31, 22, 22, 22, 22, 22, 22, 22, 
	22, 43, 22, 22, 22, 22, 22, 22, 
	37, 22, 43, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 26, 22, 25, 25, 22, 26, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 37, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 1, 1, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	1, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 22, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 37, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 29, 22, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 37, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 37, 45, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 37, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 32, 
	22, 22, 33, 34, 35, 36, 37, 22, 
	39, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 45, 
	22, 22, 22, 22, 22, 22, 37, 22, 
	39, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 45, 
	22, 22, 22, 22, 22, 22, 37, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 33, 22, 35, 22, 37, 22, 
	39, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 45, 
	22, 22, 33, 22, 22, 22, 37, 22, 
	39, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 46, 
	22, 22, 33, 34, 35, 22, 37, 22, 
	39, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 22, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 33, 34, 35, 22, 37, 22, 
	39, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 24, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 32, 
	22, 22, 33, 34, 35, 36, 37, 22, 
	39, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 48, 48, 47, 5, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 12, 47, 47, 47, 
	47, 47, 47, 47, 47, 49, 47, 47, 
	47, 47, 47, 47, 18, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 5, 47, 
	48, 48, 50, 5, 50, 50, 50, 50, 
	50, 50, 50, 50, 50, 50, 50, 50, 
	50, 50, 50, 50, 50, 50, 50, 50, 
	50, 50, 50, 50, 50, 50, 50, 50, 
	50, 50, 18, 50, 50, 50, 50, 50, 
	50, 50, 50, 50, 50, 50, 50, 50, 
	50, 50, 50, 50, 5, 50, 51, 47, 
	48, 48, 47, 5, 18, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 5, 47, 47, 47, 47, 
	47, 47, 18, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 5, 47, 48, 48, 
	47, 5, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 5, 47, 47, 47, 47, 47, 47, 
	18, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 5, 47, 2, 47, 48, 48, 
	47, 5, 6, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 52, 47, 47, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 53, 47, 47, 54, 47, 47, 47, 
	18, 47, 53, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 5, 47, 2, 47, 48, 48, 
	47, 5, 6, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	18, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 5, 47, 2, 47, 48, 48, 
	47, 5, 6, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 52, 47, 47, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	18, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 5, 47, 2, 47, 48, 48, 
	47, 5, 6, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 52, 47, 47, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 53, 47, 47, 47, 47, 47, 47, 
	18, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 5, 47, 2, 47, 48, 48, 
	47, 5, 6, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 52, 47, 47, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 53, 47, 47, 47, 47, 47, 47, 
	18, 47, 53, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 5, 47, 48, 48, 47, 5, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 18, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 55, 55, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	55, 47, 2, 3, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 13, 
	47, 47, 14, 15, 16, 17, 18, 19, 
	20, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 47, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 18, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 10, 47, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 18, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 18, 56, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 18, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 13, 
	47, 47, 14, 15, 16, 17, 18, 47, 
	20, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 56, 
	47, 47, 47, 47, 47, 47, 18, 47, 
	20, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 56, 
	47, 47, 47, 47, 47, 47, 18, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 14, 47, 16, 47, 18, 47, 
	20, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 56, 
	47, 47, 14, 47, 47, 47, 18, 47, 
	20, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 57, 
	47, 47, 14, 15, 16, 47, 18, 47, 
	20, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 47, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 14, 15, 16, 47, 18, 47, 
	20, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 2, 3, 48, 48, 47, 5, 
	6, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 9, 10, 11, 12, 47, 
	47, 47, 47, 47, 47, 47, 47, 13, 
	47, 47, 14, 15, 16, 17, 18, 47, 
	20, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	5, 47, 23, 24, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 58, 
	22, 22, 33, 34, 35, 36, 37, 38, 
	39, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 23, 59, 25, 25, 22, 26, 
	27, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 28, 29, 30, 31, 22, 
	22, 22, 22, 22, 22, 22, 22, 32, 
	22, 22, 33, 34, 35, 36, 37, 22, 
	39, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	26, 22, 1, 1, 2, 3, 48, 48, 
	47, 5, 6, 1, 1, 47, 47, 47, 
	1, 47, 47, 47, 47, 9, 10, 11, 
	12, 47, 47, 47, 47, 47, 47, 47, 
	47, 13, 47, 47, 14, 15, 16, 17, 
	18, 19, 20, 47, 47, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 47, 
	47, 47, 5, 47, 1, 1, 60, 60, 
	60, 60, 60, 60, 60, 1, 1, 60, 
	60, 60, 1, 60, 0
};

static const char _myanmar_syllable_machine_trans_targs[] = {
	0, 1, 25, 35, 0, 26, 30, 49, 
	52, 37, 38, 39, 29, 41, 42, 44, 
	45, 46, 27, 48, 43, 26, 0, 2, 
	12, 0, 3, 7, 13, 14, 15, 6, 
	17, 18, 20, 21, 22, 4, 24, 19, 
	11, 5, 8, 9, 10, 16, 23, 0, 
	0, 34, 0, 28, 31, 32, 33, 36, 
	40, 47, 50, 51, 0
};

static const char _myanmar_syllable_machine_trans_actions[] = {
	3, 0, 0, 0, 4, 5, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 6, 7, 0, 
	0, 8, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 9, 
	10, 0, 11, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 12
};

static const char _myanmar_syllable_machine_to_state_actions[] = {
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0
};

static const char _myanmar_syllable_machine_from_state_actions[] = {
	2, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0
};

static const short _myanmar_syllable_machine_eof_trans[] = {
	0, 23, 23, 23, 23, 23, 23, 23, 
	23, 23, 23, 23, 23, 23, 23, 23, 
	23, 23, 23, 23, 23, 23, 23, 23, 
	23, 48, 51, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 23, 23, 48, 61
};

static const int myanmar_syllable_machine_start = 0;
static const int myanmar_syllable_machine_first_final = 0;
static const int myanmar_syllable_machine_error = -1;

static const int myanmar_syllable_machine_en_main = 0;


#line 55 "hb-ot-shaper-myanmar-machine.rl"



#line 118 "hb-ot-shaper-myanmar-machine.rl"


#define found_syllable(syllable_type) \
  HB_STMT_START { \
    if (0) fprintf (stderr, "syllable %u..%u %s\n", ts, te, #syllable_type); \
    for (unsigned int i = ts; i < te; i++) \
      info[i].syllable() = (syllable_serial << 4) | syllable_type; \
    syllable_serial++; \
    if (syllable_serial == 16) syllable_serial = 1; \
  } HB_STMT_END

inline void
find_syllables_myanmar (hb_buffer_t *buffer)
{
  unsigned int p, pe, eof, ts, te, act HB_UNUSED;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  
#line 553 "hb-ot-shaper-myanmar-machine.hh"
	{
	cs = myanmar_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 138 "hb-ot-shaper-myanmar-machine.rl"


  p = 0;
  pe = eof = buffer->len;

  unsigned int syllable_serial = 1;
  
#line 569 "hb-ot-shaper-myanmar-machine.hh"
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
#line 583 "hb-ot-shaper-myanmar-machine.hh"
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
	case 8:
#line 111 "hb-ot-shaper-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_consonant_syllable); }}
	break;
	case 4:
#line 112 "hb-ot-shaper-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_non_myanmar_cluster); }}
	break;
	case 10:
#line 113 "hb-ot-shaper-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }}
	break;
	case 3:
#line 114 "hb-ot-shaper-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_non_myanmar_cluster); }}
	break;
	case 7:
#line 111 "hb-ot-shaper-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (myanmar_consonant_syllable); }}
	break;
	case 9:
#line 113 "hb-ot-shaper-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (myanmar_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }}
	break;
	case 12:
#line 114 "hb-ot-shaper-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (myanmar_non_myanmar_cluster); }}
	break;
	case 11:
#line 1 "NONE"
	{	switch( act ) {
	case 2:
	{{p = ((te))-1;} found_syllable (myanmar_non_myanmar_cluster); }
	break;
	case 3:
	{{p = ((te))-1;} found_syllable (myanmar_broken_cluster); buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE; }
	break;
	}
	}
	break;
	case 6:
#line 1 "NONE"
	{te = p+1;}
#line 112 "hb-ot-shaper-myanmar-machine.rl"
	{act = 2;}
	break;
	case 5:
#line 1 "NONE"
	{te = p+1;}
#line 113 "hb-ot-shaper-myanmar-machine.rl"
	{act = 3;}
	break;
#line 653 "hb-ot-shaper-myanmar-machine.hh"
	}

_again:
	switch ( _myanmar_syllable_machine_to_state_actions[cs] ) {
	case 1:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 662 "hb-ot-shaper-myanmar-machine.hh"
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

#line 146 "hb-ot-shaper-myanmar-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPER_MYANMAR_MACHINE_HH */
