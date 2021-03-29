
#line 1 "hb-ot-shape-complex-myanmar-machine.rl"
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

#ifndef HB_OT_SHAPE_COMPLEX_MYANMAR_MACHINE_HH
#define HB_OT_SHAPE_COMPLEX_MYANMAR_MACHINE_HH

#include "hb.hh"

enum myanmar_syllable_type_t {
  myanmar_consonant_syllable,
  myanmar_punctuation_cluster,
  myanmar_broken_cluster,
  myanmar_non_myanmar_cluster,
};


#line 43 "hb-ot-shape-complex-myanmar-machine.hh"
#define myanmar_syllable_machine_ex_A 10u
#define myanmar_syllable_machine_ex_As 18u
#define myanmar_syllable_machine_ex_C 1u
#define myanmar_syllable_machine_ex_CS 19u
#define myanmar_syllable_machine_ex_D 32u
#define myanmar_syllable_machine_ex_D0 20u
#define myanmar_syllable_machine_ex_DB 3u
#define myanmar_syllable_machine_ex_GB 11u
#define myanmar_syllable_machine_ex_H 4u
#define myanmar_syllable_machine_ex_IV 2u
#define myanmar_syllable_machine_ex_MH 21u
#define myanmar_syllable_machine_ex_MR 22u
#define myanmar_syllable_machine_ex_MW 23u
#define myanmar_syllable_machine_ex_MY 24u
#define myanmar_syllable_machine_ex_P 31u
#define myanmar_syllable_machine_ex_PT 25u
#define myanmar_syllable_machine_ex_Ra 16u
#define myanmar_syllable_machine_ex_V 8u
#define myanmar_syllable_machine_ex_VAbv 26u
#define myanmar_syllable_machine_ex_VBlw 27u
#define myanmar_syllable_machine_ex_VPre 28u
#define myanmar_syllable_machine_ex_VPst 29u
#define myanmar_syllable_machine_ex_VS 30u
#define myanmar_syllable_machine_ex_ZWJ 6u
#define myanmar_syllable_machine_ex_ZWNJ 5u


#line 71 "hb-ot-shape-complex-myanmar-machine.hh"
static const unsigned char _myanmar_syllable_machine_trans_keys[] = {
	1u, 32u, 3u, 30u, 5u, 29u, 5u, 8u, 5u, 29u, 3u, 25u, 5u, 25u, 5u, 25u, 
	3u, 29u, 3u, 29u, 3u, 29u, 3u, 29u, 1u, 16u, 3u, 29u, 3u, 29u, 3u, 29u, 
	3u, 29u, 3u, 29u, 3u, 30u, 3u, 29u, 3u, 29u, 3u, 29u, 3u, 29u, 3u, 29u, 
	5u, 29u, 5u, 8u, 5u, 29u, 3u, 25u, 5u, 25u, 5u, 25u, 3u, 29u, 3u, 29u, 
	3u, 29u, 3u, 29u, 1u, 16u, 3u, 30u, 3u, 29u, 3u, 29u, 3u, 29u, 3u, 29u, 
	3u, 29u, 3u, 30u, 3u, 29u, 3u, 29u, 3u, 29u, 3u, 29u, 3u, 29u, 3u, 30u, 
	3u, 29u, 1u, 32u, 1u, 32u, 8u, 8u, 0
};

static const char _myanmar_syllable_machine_key_spans[] = {
	32, 28, 25, 4, 25, 23, 21, 21, 
	27, 27, 27, 27, 16, 27, 27, 27, 
	27, 27, 28, 27, 27, 27, 27, 27, 
	25, 4, 25, 23, 21, 21, 27, 27, 
	27, 27, 16, 28, 27, 27, 27, 27, 
	27, 28, 27, 27, 27, 27, 27, 28, 
	27, 32, 32, 1
};

static const short _myanmar_syllable_machine_index_offsets[] = {
	0, 33, 62, 88, 93, 119, 143, 165, 
	187, 215, 243, 271, 299, 316, 344, 372, 
	400, 428, 456, 485, 513, 541, 569, 597, 
	625, 651, 656, 682, 706, 728, 750, 778, 
	806, 834, 862, 879, 908, 936, 964, 992, 
	1020, 1048, 1077, 1105, 1133, 1161, 1189, 1217, 
	1246, 1274, 1307, 1340
};

static const char _myanmar_syllable_machine_indicies[] = {
	1, 1, 2, 3, 4, 4, 0, 5, 
	0, 6, 1, 0, 0, 0, 0, 7, 
	0, 8, 9, 0, 10, 11, 12, 13, 
	14, 15, 16, 17, 18, 19, 20, 1, 
	0, 22, 23, 24, 24, 21, 25, 21, 
	26, 21, 21, 21, 21, 21, 21, 21, 
	27, 21, 21, 28, 29, 30, 31, 32, 
	33, 34, 35, 36, 37, 21, 24, 24, 
	21, 25, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 38, 21, 21, 21, 21, 
	21, 21, 32, 21, 21, 21, 36, 21, 
	24, 24, 21, 25, 21, 24, 24, 21, 
	25, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 32, 21, 21, 21, 36, 21, 39, 
	21, 24, 24, 21, 25, 21, 32, 21, 
	21, 21, 21, 21, 21, 21, 40, 21, 
	21, 21, 21, 21, 21, 32, 21, 24, 
	24, 21, 25, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 40, 21, 21, 21, 
	21, 21, 21, 32, 21, 24, 24, 21, 
	25, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 32, 21, 22, 21, 24, 24, 21, 
	25, 21, 26, 21, 21, 21, 21, 21, 
	21, 21, 41, 21, 21, 41, 21, 21, 
	21, 32, 42, 21, 21, 36, 21, 22, 
	21, 24, 24, 21, 25, 21, 26, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 32, 21, 21, 
	21, 36, 21, 22, 21, 24, 24, 21, 
	25, 21, 26, 21, 21, 21, 21, 21, 
	21, 21, 41, 21, 21, 21, 21, 21, 
	21, 32, 42, 21, 21, 36, 21, 22, 
	21, 24, 24, 21, 25, 21, 26, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 32, 42, 21, 
	21, 36, 21, 1, 1, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 1, 21, 22, 21, 24, 24, 
	21, 25, 21, 26, 21, 21, 21, 21, 
	21, 21, 21, 27, 21, 21, 28, 29, 
	30, 31, 32, 33, 34, 35, 36, 21, 
	22, 21, 24, 24, 21, 25, 21, 26, 
	21, 21, 21, 21, 21, 21, 21, 43, 
	21, 21, 21, 21, 21, 21, 32, 33, 
	34, 35, 36, 21, 22, 21, 24, 24, 
	21, 25, 21, 26, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 32, 33, 34, 35, 36, 21, 
	22, 21, 24, 24, 21, 25, 21, 26, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 32, 33, 
	34, 21, 36, 21, 22, 21, 24, 24, 
	21, 25, 21, 26, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 32, 21, 34, 21, 36, 21, 
	22, 21, 24, 24, 21, 25, 21, 26, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 32, 33, 
	34, 35, 36, 43, 21, 22, 21, 24, 
	24, 21, 25, 21, 26, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 28, 
	21, 30, 21, 32, 33, 34, 35, 36, 
	21, 22, 21, 24, 24, 21, 25, 21, 
	26, 21, 21, 21, 21, 21, 21, 21, 
	43, 21, 21, 28, 21, 21, 21, 32, 
	33, 34, 35, 36, 21, 22, 21, 24, 
	24, 21, 25, 21, 26, 21, 21, 21, 
	21, 21, 21, 21, 44, 21, 21, 28, 
	29, 30, 21, 32, 33, 34, 35, 36, 
	21, 22, 21, 24, 24, 21, 25, 21, 
	26, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 28, 29, 30, 21, 32, 
	33, 34, 35, 36, 21, 22, 23, 24, 
	24, 21, 25, 21, 26, 21, 21, 21, 
	21, 21, 21, 21, 27, 21, 21, 28, 
	29, 30, 31, 32, 33, 34, 35, 36, 
	21, 46, 46, 45, 5, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 47, 45, 
	45, 45, 45, 45, 45, 14, 45, 45, 
	45, 18, 45, 46, 46, 45, 5, 45, 
	46, 46, 45, 5, 45, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 45, 45, 
	45, 45, 45, 45, 14, 45, 45, 45, 
	18, 45, 48, 45, 46, 46, 45, 5, 
	45, 14, 45, 45, 45, 45, 45, 45, 
	45, 49, 45, 45, 45, 45, 45, 45, 
	14, 45, 46, 46, 45, 5, 45, 45, 
	45, 45, 45, 45, 45, 45, 45, 49, 
	45, 45, 45, 45, 45, 45, 14, 45, 
	46, 46, 45, 5, 45, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 45, 45, 
	45, 45, 45, 45, 14, 45, 2, 45, 
	46, 46, 45, 5, 45, 6, 45, 45, 
	45, 45, 45, 45, 45, 50, 45, 45, 
	50, 45, 45, 45, 14, 51, 45, 45, 
	18, 45, 2, 45, 46, 46, 45, 5, 
	45, 6, 45, 45, 45, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 45, 45, 
	14, 45, 45, 45, 18, 45, 2, 45, 
	46, 46, 45, 5, 45, 6, 45, 45, 
	45, 45, 45, 45, 45, 50, 45, 45, 
	45, 45, 45, 45, 14, 51, 45, 45, 
	18, 45, 2, 45, 46, 46, 45, 5, 
	45, 6, 45, 45, 45, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 45, 45, 
	14, 51, 45, 45, 18, 45, 52, 52, 
	45, 45, 45, 45, 45, 45, 45, 45, 
	45, 45, 45, 45, 45, 52, 45, 2, 
	3, 46, 46, 45, 5, 45, 6, 45, 
	45, 45, 45, 45, 45, 45, 8, 45, 
	45, 10, 11, 12, 13, 14, 15, 16, 
	17, 18, 19, 45, 2, 45, 46, 46, 
	45, 5, 45, 6, 45, 45, 45, 45, 
	45, 45, 45, 8, 45, 45, 10, 11, 
	12, 13, 14, 15, 16, 17, 18, 45, 
	2, 45, 46, 46, 45, 5, 45, 6, 
	45, 45, 45, 45, 45, 45, 45, 53, 
	45, 45, 45, 45, 45, 45, 14, 15, 
	16, 17, 18, 45, 2, 45, 46, 46, 
	45, 5, 45, 6, 45, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 45, 45, 
	45, 45, 14, 15, 16, 17, 18, 45, 
	2, 45, 46, 46, 45, 5, 45, 6, 
	45, 45, 45, 45, 45, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 14, 15, 
	16, 45, 18, 45, 2, 45, 46, 46, 
	45, 5, 45, 6, 45, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 45, 45, 
	45, 45, 14, 45, 16, 45, 18, 45, 
	2, 45, 46, 46, 45, 5, 45, 6, 
	45, 45, 45, 45, 45, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 14, 15, 
	16, 17, 18, 53, 45, 2, 45, 46, 
	46, 45, 5, 45, 6, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 45, 10, 
	45, 12, 45, 14, 15, 16, 17, 18, 
	45, 2, 45, 46, 46, 45, 5, 45, 
	6, 45, 45, 45, 45, 45, 45, 45, 
	53, 45, 45, 10, 45, 45, 45, 14, 
	15, 16, 17, 18, 45, 2, 45, 46, 
	46, 45, 5, 45, 6, 45, 45, 45, 
	45, 45, 45, 45, 54, 45, 45, 10, 
	11, 12, 45, 14, 15, 16, 17, 18, 
	45, 2, 45, 46, 46, 45, 5, 45, 
	6, 45, 45, 45, 45, 45, 45, 45, 
	45, 45, 45, 10, 11, 12, 45, 14, 
	15, 16, 17, 18, 45, 2, 3, 46, 
	46, 45, 5, 45, 6, 45, 45, 45, 
	45, 45, 45, 45, 8, 45, 45, 10, 
	11, 12, 13, 14, 15, 16, 17, 18, 
	45, 22, 23, 24, 24, 21, 25, 21, 
	26, 21, 21, 21, 21, 21, 21, 21, 
	55, 21, 21, 28, 29, 30, 31, 32, 
	33, 34, 35, 36, 37, 21, 22, 56, 
	24, 24, 21, 25, 21, 26, 21, 21, 
	21, 21, 21, 21, 21, 27, 21, 21, 
	28, 29, 30, 31, 32, 33, 34, 35, 
	36, 21, 1, 1, 2, 3, 46, 46, 
	45, 5, 45, 6, 1, 45, 45, 45, 
	45, 1, 45, 8, 45, 45, 10, 11, 
	12, 13, 14, 15, 16, 17, 18, 19, 
	45, 1, 45, 1, 1, 57, 57, 57, 
	57, 57, 57, 57, 57, 1, 57, 57, 
	57, 57, 1, 57, 57, 57, 57, 57, 
	57, 57, 57, 57, 57, 57, 57, 57, 
	57, 57, 1, 57, 58, 57, 0
};

static const char _myanmar_syllable_machine_trans_targs[] = {
	0, 1, 24, 34, 0, 25, 31, 47, 
	36, 50, 37, 42, 43, 44, 27, 39, 
	40, 41, 30, 46, 51, 0, 2, 12, 
	0, 3, 9, 13, 14, 19, 20, 21, 
	5, 16, 17, 18, 8, 23, 4, 6, 
	7, 10, 11, 15, 22, 0, 0, 26, 
	28, 29, 32, 33, 35, 38, 45, 48, 
	49, 0, 0
};

static const char _myanmar_syllable_machine_trans_actions[] = {
	3, 0, 0, 0, 4, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 5, 0, 0, 
	6, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 7, 8, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 9, 10
};

static const char _myanmar_syllable_machine_to_state_actions[] = {
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0
};

static const char _myanmar_syllable_machine_from_state_actions[] = {
	2, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0
};

static const short _myanmar_syllable_machine_eof_trans[] = {
	0, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 22, 
	22, 46, 58, 58
};

static const int myanmar_syllable_machine_start = 0;
static const int myanmar_syllable_machine_first_final = 0;
static const int myanmar_syllable_machine_error = -1;

static const int myanmar_syllable_machine_en_main = 0;


#line 44 "hb-ot-shape-complex-myanmar-machine.rl"



#line 101 "hb-ot-shape-complex-myanmar-machine.rl"


#define found_syllable(syllable_type) \
  HB_STMT_START { \
    if (0) fprintf (stderr, "syllable %d..%d %s\n", ts, te, #syllable_type); \
    for (unsigned int i = ts; i < te; i++) \
      info[i].syllable() = (syllable_serial << 4) | syllable_type; \
    syllable_serial++; \
    if (unlikely (syllable_serial == 16)) syllable_serial = 1; \
  } HB_STMT_END

static void
find_syllables_myanmar (hb_buffer_t *buffer)
{
  unsigned int p, pe, eof, ts, te, act HB_UNUSED;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  
#line 355 "hb-ot-shape-complex-myanmar-machine.hh"
	{
	cs = myanmar_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 121 "hb-ot-shape-complex-myanmar-machine.rl"


  p = 0;
  pe = eof = buffer->len;

  unsigned int syllable_serial = 1;
  
#line 371 "hb-ot-shape-complex-myanmar-machine.hh"
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
#line 385 "hb-ot-shape-complex-myanmar-machine.hh"
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
#line 93 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_consonant_syllable); }}
	break;
	case 4:
#line 94 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_non_myanmar_cluster); }}
	break;
	case 10:
#line 95 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_punctuation_cluster); }}
	break;
	case 8:
#line 96 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_broken_cluster); }}
	break;
	case 3:
#line 97 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p+1;{ found_syllable (myanmar_non_myanmar_cluster); }}
	break;
	case 5:
#line 93 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (myanmar_consonant_syllable); }}
	break;
	case 7:
#line 96 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (myanmar_broken_cluster); }}
	break;
	case 9:
#line 97 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (myanmar_non_myanmar_cluster); }}
	break;
#line 435 "hb-ot-shape-complex-myanmar-machine.hh"
	}

_again:
	switch ( _myanmar_syllable_machine_to_state_actions[cs] ) {
	case 1:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 444 "hb-ot-shape-complex-myanmar-machine.hh"
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

#line 129 "hb-ot-shape-complex-myanmar-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_MYANMAR_MACHINE_HH */
