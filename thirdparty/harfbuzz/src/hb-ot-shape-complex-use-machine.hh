
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
#define use_syllable_machine_ex_ZWNJ 14u


#line 99 "hb-ot-shape-complex-use-machine.hh"
static const unsigned char _use_syllable_machine_trans_keys[] = {
	0u, 51u, 41u, 42u, 42u, 42u, 11u, 48u, 11u, 48u, 1u, 1u, 22u, 48u, 23u, 48u, 
	24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 
	1u, 1u, 24u, 48u, 23u, 48u, 23u, 48u, 23u, 48u, 22u, 48u, 22u, 48u, 22u, 48u, 
	11u, 48u, 1u, 48u, 13u, 13u, 4u, 4u, 11u, 48u, 11u, 48u, 1u, 1u, 22u, 48u, 
	23u, 48u, 24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 
	24u, 48u, 1u, 1u, 24u, 48u, 23u, 48u, 23u, 48u, 23u, 48u, 22u, 48u, 22u, 48u, 
	22u, 48u, 11u, 48u, 1u, 48u, 4u, 4u, 13u, 13u, 1u, 48u, 11u, 48u, 41u, 42u, 
	42u, 42u, 1u, 5u, 50u, 52u, 49u, 52u, 49u, 51u, 0
};

static const char _use_syllable_machine_key_spans[] = {
	52, 2, 1, 38, 38, 1, 27, 26, 
	24, 23, 22, 2, 1, 25, 25, 25, 
	1, 25, 26, 26, 26, 27, 27, 27, 
	38, 48, 1, 1, 38, 38, 1, 27, 
	26, 24, 23, 22, 2, 1, 25, 25, 
	25, 1, 25, 26, 26, 26, 27, 27, 
	27, 38, 48, 1, 1, 48, 38, 2, 
	1, 5, 3, 4, 3
};

static const short _use_syllable_machine_index_offsets[] = {
	0, 53, 56, 58, 97, 136, 138, 166, 
	193, 218, 242, 265, 268, 270, 296, 322, 
	348, 350, 376, 403, 430, 457, 485, 513, 
	541, 580, 629, 631, 633, 672, 711, 713, 
	741, 768, 793, 817, 840, 843, 845, 871, 
	897, 923, 925, 951, 978, 1005, 1032, 1060, 
	1088, 1116, 1155, 1204, 1206, 1208, 1257, 1296, 
	1299, 1301, 1307, 1311, 1316
};

static const char _use_syllable_machine_indicies[] = {
	0, 1, 2, 2, 3, 4, 2, 2, 
	2, 2, 2, 5, 6, 7, 2, 2, 
	2, 2, 8, 2, 2, 2, 9, 10, 
	11, 12, 13, 14, 15, 9, 16, 17, 
	18, 19, 20, 21, 2, 22, 23, 24, 
	2, 25, 26, 27, 28, 29, 30, 31, 
	6, 32, 2, 33, 2, 0, 35, 34, 
	35, 34, 37, 38, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 39, 40, 41, 
	42, 43, 44, 45, 39, 46, 1, 47, 
	48, 49, 50, 36, 51, 52, 53, 36, 
	36, 36, 36, 54, 55, 56, 57, 38, 
	36, 37, 38, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 39, 40, 41, 42, 
	43, 44, 45, 39, 46, 47, 47, 48, 
	49, 50, 36, 51, 52, 53, 36, 36, 
	36, 36, 54, 55, 56, 57, 38, 36, 
	37, 58, 39, 40, 41, 42, 43, 36, 
	36, 36, 36, 36, 36, 48, 49, 50, 
	36, 51, 52, 53, 36, 36, 36, 36, 
	40, 55, 56, 57, 59, 36, 40, 41, 
	42, 43, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 51, 52, 53, 36, 
	36, 36, 36, 36, 55, 56, 57, 59, 
	36, 41, 42, 43, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 55, 56, 
	57, 36, 42, 43, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 55, 56, 
	57, 36, 43, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 55, 56, 57, 
	36, 55, 56, 36, 56, 36, 41, 42, 
	43, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 51, 52, 53, 36, 36, 
	36, 36, 36, 55, 56, 57, 59, 36, 
	41, 42, 43, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 52, 53, 
	36, 36, 36, 36, 36, 55, 56, 57, 
	59, 36, 41, 42, 43, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 53, 36, 36, 36, 36, 36, 55, 
	56, 57, 59, 36, 61, 60, 41, 42, 
	43, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 55, 56, 57, 59, 36, 
	40, 41, 42, 43, 36, 36, 36, 36, 
	36, 36, 48, 49, 50, 36, 51, 52, 
	53, 36, 36, 36, 36, 40, 55, 56, 
	57, 59, 36, 40, 41, 42, 43, 36, 
	36, 36, 36, 36, 36, 36, 49, 50, 
	36, 51, 52, 53, 36, 36, 36, 36, 
	40, 55, 56, 57, 59, 36, 40, 41, 
	42, 43, 36, 36, 36, 36, 36, 36, 
	36, 36, 50, 36, 51, 52, 53, 36, 
	36, 36, 36, 40, 55, 56, 57, 59, 
	36, 39, 40, 41, 42, 43, 36, 45, 
	39, 36, 36, 36, 48, 49, 50, 36, 
	51, 52, 53, 36, 36, 36, 36, 40, 
	55, 56, 57, 59, 36, 39, 40, 41, 
	42, 43, 36, 36, 39, 36, 36, 36, 
	48, 49, 50, 36, 51, 52, 53, 36, 
	36, 36, 36, 40, 55, 56, 57, 59, 
	36, 39, 40, 41, 42, 43, 44, 45, 
	39, 36, 36, 36, 48, 49, 50, 36, 
	51, 52, 53, 36, 36, 36, 36, 40, 
	55, 56, 57, 59, 36, 37, 38, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	39, 40, 41, 42, 43, 44, 45, 39, 
	46, 36, 47, 48, 49, 50, 36, 51, 
	52, 53, 36, 36, 36, 36, 54, 55, 
	56, 57, 38, 36, 37, 58, 58, 58, 
	58, 58, 58, 58, 58, 58, 58, 58, 
	58, 58, 58, 58, 58, 58, 58, 58, 
	58, 58, 40, 41, 42, 43, 58, 58, 
	58, 58, 58, 58, 58, 58, 58, 58, 
	51, 52, 53, 58, 58, 58, 58, 58, 
	55, 56, 57, 59, 58, 63, 62, 3, 
	64, 37, 38, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 39, 40, 41, 42, 
	43, 44, 45, 39, 46, 1, 47, 48, 
	49, 50, 36, 51, 52, 53, 36, 0, 
	35, 36, 54, 55, 56, 57, 38, 36, 
	5, 6, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 9, 10, 11, 12, 13, 
	14, 15, 9, 16, 18, 18, 19, 20, 
	21, 65, 22, 23, 24, 65, 65, 65, 
	65, 28, 29, 30, 31, 6, 65, 5, 
	65, 9, 10, 11, 12, 13, 65, 65, 
	65, 65, 65, 65, 19, 20, 21, 65, 
	22, 23, 24, 65, 65, 65, 65, 10, 
	29, 30, 31, 66, 65, 10, 11, 12, 
	13, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 22, 23, 24, 65, 65, 
	65, 65, 65, 29, 30, 31, 66, 65, 
	11, 12, 13, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 29, 30, 31, 
	65, 12, 13, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 29, 30, 31, 
	65, 13, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 29, 30, 31, 65, 
	29, 30, 65, 30, 65, 11, 12, 13, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 22, 23, 24, 65, 65, 65, 
	65, 65, 29, 30, 31, 66, 65, 11, 
	12, 13, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 23, 24, 65, 
	65, 65, 65, 65, 29, 30, 31, 66, 
	65, 11, 12, 13, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	24, 65, 65, 65, 65, 65, 29, 30, 
	31, 66, 65, 67, 65, 11, 12, 13, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 29, 30, 31, 66, 65, 10, 
	11, 12, 13, 65, 65, 65, 65, 65, 
	65, 19, 20, 21, 65, 22, 23, 24, 
	65, 65, 65, 65, 10, 29, 30, 31, 
	66, 65, 10, 11, 12, 13, 65, 65, 
	65, 65, 65, 65, 65, 20, 21, 65, 
	22, 23, 24, 65, 65, 65, 65, 10, 
	29, 30, 31, 66, 65, 10, 11, 12, 
	13, 65, 65, 65, 65, 65, 65, 65, 
	65, 21, 65, 22, 23, 24, 65, 65, 
	65, 65, 10, 29, 30, 31, 66, 65, 
	9, 10, 11, 12, 13, 65, 15, 9, 
	65, 65, 65, 19, 20, 21, 65, 22, 
	23, 24, 65, 65, 65, 65, 10, 29, 
	30, 31, 66, 65, 9, 10, 11, 12, 
	13, 65, 65, 9, 65, 65, 65, 19, 
	20, 21, 65, 22, 23, 24, 65, 65, 
	65, 65, 10, 29, 30, 31, 66, 65, 
	9, 10, 11, 12, 13, 14, 15, 9, 
	65, 65, 65, 19, 20, 21, 65, 22, 
	23, 24, 65, 65, 65, 65, 10, 29, 
	30, 31, 66, 65, 5, 6, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 9, 
	10, 11, 12, 13, 14, 15, 9, 16, 
	65, 18, 19, 20, 21, 65, 22, 23, 
	24, 65, 65, 65, 65, 28, 29, 30, 
	31, 6, 65, 5, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 10, 11, 12, 13, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 22, 
	23, 24, 65, 65, 65, 65, 65, 29, 
	30, 31, 66, 65, 68, 65, 7, 65, 
	1, 65, 65, 65, 1, 65, 65, 65, 
	65, 65, 5, 6, 7, 65, 65, 65, 
	65, 65, 65, 65, 65, 9, 10, 11, 
	12, 13, 14, 15, 9, 16, 17, 18, 
	19, 20, 21, 65, 22, 23, 24, 65, 
	25, 26, 65, 28, 29, 30, 31, 6, 
	65, 5, 6, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 9, 10, 11, 12, 
	13, 14, 15, 9, 16, 17, 18, 19, 
	20, 21, 65, 22, 23, 24, 65, 65, 
	65, 65, 28, 29, 30, 31, 6, 65, 
	25, 26, 65, 26, 65, 1, 69, 69, 
	69, 1, 69, 71, 70, 32, 70, 32, 
	71, 70, 71, 70, 32, 70, 33, 70, 
	0
};

static const char _use_syllable_machine_trans_targs[] = {
	1, 3, 0, 26, 28, 29, 30, 51, 
	53, 31, 32, 33, 34, 35, 46, 47, 
	48, 54, 49, 43, 44, 45, 38, 39, 
	40, 55, 56, 57, 50, 36, 37, 0, 
	58, 60, 0, 2, 0, 4, 5, 6, 
	7, 8, 9, 10, 21, 22, 23, 24, 
	18, 19, 20, 13, 14, 15, 25, 11, 
	12, 0, 0, 16, 0, 17, 0, 27, 
	0, 0, 41, 42, 52, 0, 0, 59
};

static const char _use_syllable_machine_trans_actions[] = {
	0, 0, 3, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 4, 
	0, 0, 5, 0, 6, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 7, 8, 0, 9, 0, 10, 0, 
	11, 12, 0, 0, 0, 13, 14, 0
};

static const char _use_syllable_machine_to_state_actions[] = {
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0
};

static const char _use_syllable_machine_from_state_actions[] = {
	2, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0
};

static const short _use_syllable_machine_eof_trans[] = {
	0, 35, 35, 37, 37, 59, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	61, 37, 37, 37, 37, 37, 37, 37, 
	37, 59, 63, 65, 37, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 66, 66, 
	66, 70, 71, 71, 71
};

static const int use_syllable_machine_start = 0;
static const int use_syllable_machine_first_final = 0;
static const int use_syllable_machine_error = -1;

static const int use_syllable_machine_en_main = 0;


#line 58 "hb-ot-shape-complex-use-machine.rl"



#line 179 "hb-ot-shape-complex-use-machine.rl"


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
  machine_index_t (const machine_index_t& o) : it (o.it) {}

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
{ return !(i.use_category() == USE(CGJ) && _hb_glyph_info_is_default_ignorable (&i)); }

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
  
#line 453 "hb-ot-shape-complex-use-machine.hh"
	{
	cs = use_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 263 "hb-ot-shape-complex-use-machine.rl"


  unsigned int syllable_serial = 1;
  
#line 466 "hb-ot-shape-complex-use-machine.hh"
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
#line 480 "hb-ot-shape-complex-use-machine.hh"
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
	case 7:
#line 169 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (use_standard_cluster); }}
	break;
	case 4:
#line 174 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (use_broken_cluster); }}
	break;
	case 3:
#line 175 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (use_non_cluster); }}
	break;
	case 8:
#line 167 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_virama_terminated_cluster); }}
	break;
	case 9:
#line 168 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_sakot_terminated_cluster); }}
	break;
	case 6:
#line 169 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_standard_cluster); }}
	break;
	case 11:
#line 170 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_number_joiner_terminated_cluster); }}
	break;
	case 10:
#line 171 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_numeral_cluster); }}
	break;
	case 5:
#line 172 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_symbol_cluster); }}
	break;
	case 14:
#line 173 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_hieroglyph_cluster); }}
	break;
	case 12:
#line 174 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_broken_cluster); }}
	break;
	case 13:
#line 175 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (use_non_cluster); }}
	break;
#line 546 "hb-ot-shape-complex-use-machine.hh"
	}

_again:
	switch ( _use_syllable_machine_to_state_actions[cs] ) {
	case 1:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 555 "hb-ot-shape-complex-use-machine.hh"
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

#line 268 "hb-ot-shape-complex-use-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH */
