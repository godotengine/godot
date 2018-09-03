
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

#include "hb-private.hh"


#line 38 "hb-ot-shape-complex-use-machine.hh"
static const unsigned char _use_syllable_machine_trans_keys[] = {
	1u, 1u, 12u, 12u, 1u, 15u, 1u, 15u, 1u, 1u, 12u, 12u, 0u, 43u, 21u, 21u, 
	8u, 39u, 8u, 39u, 1u, 15u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 26u, 8u, 26u, 
	8u, 26u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 
	8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 13u, 21u, 4u, 4u, 13u, 13u, 
	8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 26u, 8u, 26u, 8u, 26u, 8u, 39u, 
	8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 
	8u, 39u, 8u, 39u, 1u, 15u, 12u, 12u, 1u, 39u, 8u, 39u, 21u, 42u, 41u, 42u, 
	42u, 42u, 1u, 5u, 0
};

static const char _use_syllable_machine_key_spans[] = {
	1, 1, 15, 15, 1, 1, 44, 1, 
	32, 32, 15, 32, 32, 32, 19, 19, 
	19, 32, 32, 32, 32, 32, 32, 32, 
	32, 32, 32, 32, 32, 9, 1, 1, 
	32, 32, 32, 32, 19, 19, 19, 32, 
	32, 32, 32, 32, 32, 32, 32, 32, 
	32, 32, 15, 1, 39, 32, 22, 2, 
	1, 5
};

static const short _use_syllable_machine_index_offsets[] = {
	0, 2, 4, 20, 36, 38, 40, 85, 
	87, 120, 153, 169, 202, 235, 268, 288, 
	308, 328, 361, 394, 427, 460, 493, 526, 
	559, 592, 625, 658, 691, 724, 734, 736, 
	738, 771, 804, 837, 870, 890, 910, 930, 
	963, 996, 1029, 1062, 1095, 1128, 1161, 1194, 
	1227, 1260, 1293, 1309, 1311, 1351, 1384, 1407, 
	1410, 1412
};

static const char _use_syllable_machine_indicies[] = {
	1, 0, 3, 2, 1, 2, 2, 2, 
	2, 2, 2, 2, 2, 2, 2, 2, 
	2, 2, 4, 2, 5, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 6, 0, 5, 0, 8, 7, 
	9, 10, 11, 9, 12, 10, 11, 11, 
	13, 11, 11, 5, 14, 11, 11, 15, 
	9, 9, 16, 17, 11, 11, 18, 19, 
	20, 21, 22, 23, 24, 18, 25, 26, 
	27, 28, 29, 30, 11, 31, 32, 33, 
	11, 11, 11, 34, 11, 36, 35, 38, 
	37, 37, 1, 39, 37, 37, 40, 37, 
	37, 37, 37, 37, 41, 42, 43, 44, 
	45, 46, 47, 48, 42, 49, 41, 50, 
	51, 52, 53, 37, 54, 55, 56, 37, 
	38, 37, 37, 1, 39, 37, 37, 40, 
	37, 37, 37, 37, 37, 57, 42, 43, 
	44, 45, 46, 47, 48, 42, 49, 50, 
	50, 51, 52, 53, 37, 54, 55, 56, 
	37, 1, 58, 58, 58, 58, 58, 58, 
	58, 58, 58, 58, 58, 58, 58, 4, 
	58, 38, 37, 37, 1, 39, 37, 37, 
	40, 37, 37, 37, 37, 37, 37, 42, 
	43, 44, 45, 46, 47, 48, 42, 49, 
	50, 50, 51, 52, 53, 37, 54, 55, 
	56, 37, 38, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	42, 43, 44, 45, 46, 37, 37, 37, 
	37, 37, 37, 51, 52, 53, 37, 54, 
	55, 56, 37, 38, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 43, 44, 45, 46, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	54, 55, 56, 37, 38, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 44, 45, 46, 37, 
	38, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 45, 46, 37, 38, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 46, 37, 
	38, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	44, 45, 46, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 54, 55, 56, 
	37, 38, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 44, 45, 46, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 55, 
	56, 37, 38, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 44, 45, 46, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 56, 37, 38, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 43, 44, 45, 46, 37, 37, 
	37, 37, 37, 37, 51, 52, 53, 37, 
	54, 55, 56, 37, 38, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 43, 44, 45, 46, 37, 
	37, 37, 37, 37, 37, 37, 52, 53, 
	37, 54, 55, 56, 37, 38, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 43, 44, 45, 46, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	53, 37, 54, 55, 56, 37, 38, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 42, 43, 44, 45, 
	46, 37, 48, 42, 37, 37, 37, 51, 
	52, 53, 37, 54, 55, 56, 37, 38, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 42, 43, 44, 
	45, 46, 37, 59, 42, 37, 37, 37, 
	51, 52, 53, 37, 54, 55, 56, 37, 
	38, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 42, 43, 
	44, 45, 46, 37, 37, 42, 37, 37, 
	37, 51, 52, 53, 37, 54, 55, 56, 
	37, 38, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 42, 
	43, 44, 45, 46, 47, 48, 42, 37, 
	37, 37, 51, 52, 53, 37, 54, 55, 
	56, 37, 38, 37, 37, 1, 39, 37, 
	37, 40, 37, 37, 37, 37, 37, 37, 
	42, 43, 44, 45, 46, 47, 48, 42, 
	49, 37, 50, 51, 52, 53, 37, 54, 
	55, 56, 37, 38, 37, 37, 1, 39, 
	37, 37, 40, 37, 37, 37, 37, 37, 
	37, 42, 43, 44, 45, 46, 47, 48, 
	42, 49, 41, 50, 51, 52, 53, 37, 
	54, 55, 56, 37, 61, 60, 60, 60, 
	60, 60, 60, 60, 62, 60, 12, 63, 
	61, 60, 13, 64, 64, 5, 8, 64, 
	64, 65, 64, 64, 64, 64, 64, 66, 
	18, 19, 20, 21, 22, 23, 24, 18, 
	25, 27, 27, 28, 29, 30, 64, 31, 
	32, 33, 64, 13, 64, 64, 5, 8, 
	64, 64, 65, 64, 64, 64, 64, 64, 
	64, 18, 19, 20, 21, 22, 23, 24, 
	18, 25, 27, 27, 28, 29, 30, 64, 
	31, 32, 33, 64, 13, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 18, 19, 20, 21, 22, 64, 
	64, 64, 64, 64, 64, 28, 29, 30, 
	64, 31, 32, 33, 64, 13, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 19, 20, 21, 22, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 31, 32, 33, 64, 13, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 20, 21, 
	22, 64, 13, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 21, 22, 64, 13, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	22, 64, 13, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 20, 21, 22, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 31, 
	32, 33, 64, 13, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 20, 21, 22, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 32, 33, 64, 13, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 20, 21, 22, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 33, 64, 13, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 19, 20, 21, 22, 
	64, 64, 64, 64, 64, 64, 28, 29, 
	30, 64, 31, 32, 33, 64, 13, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 19, 20, 21, 
	22, 64, 64, 64, 64, 64, 64, 64, 
	29, 30, 64, 31, 32, 33, 64, 13, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 19, 20, 
	21, 22, 64, 64, 64, 64, 64, 64, 
	64, 64, 30, 64, 31, 32, 33, 64, 
	13, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 18, 19, 
	20, 21, 22, 64, 24, 18, 64, 64, 
	64, 28, 29, 30, 64, 31, 32, 33, 
	64, 13, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 18, 
	19, 20, 21, 22, 64, 67, 18, 64, 
	64, 64, 28, 29, 30, 64, 31, 32, 
	33, 64, 13, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	18, 19, 20, 21, 22, 64, 64, 18, 
	64, 64, 64, 28, 29, 30, 64, 31, 
	32, 33, 64, 13, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 18, 19, 20, 21, 22, 23, 24, 
	18, 64, 64, 64, 28, 29, 30, 64, 
	31, 32, 33, 64, 13, 64, 64, 5, 
	8, 64, 64, 65, 64, 64, 64, 64, 
	64, 64, 18, 19, 20, 21, 22, 23, 
	24, 18, 25, 64, 27, 28, 29, 30, 
	64, 31, 32, 33, 64, 5, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 6, 68, 8, 68, 10, 
	64, 64, 64, 10, 64, 64, 13, 64, 
	64, 5, 8, 64, 64, 65, 64, 64, 
	64, 64, 64, 64, 18, 19, 20, 21, 
	22, 23, 24, 18, 25, 26, 27, 28, 
	29, 30, 64, 31, 32, 33, 64, 13, 
	64, 64, 5, 8, 64, 64, 65, 64, 
	64, 64, 64, 64, 64, 18, 19, 20, 
	21, 22, 23, 24, 18, 25, 26, 27, 
	28, 29, 30, 64, 31, 32, 33, 64, 
	70, 69, 69, 69, 69, 69, 69, 69, 
	69, 69, 69, 69, 69, 69, 69, 69, 
	69, 69, 69, 69, 70, 71, 69, 70, 
	71, 69, 71, 69, 10, 68, 68, 68, 
	10, 68, 0
};

static const char _use_syllable_machine_trans_targs[] = {
	6, 9, 6, 2, 0, 32, 4, 6, 
	3, 7, 8, 6, 29, 6, 50, 51, 
	52, 54, 34, 35, 36, 37, 38, 45, 
	46, 48, 53, 49, 42, 43, 44, 39, 
	40, 41, 57, 6, 6, 6, 6, 10, 
	1, 28, 12, 13, 14, 15, 16, 23, 
	24, 26, 27, 20, 21, 22, 17, 18, 
	19, 11, 6, 25, 6, 30, 31, 6, 
	6, 5, 33, 47, 6, 6, 55, 56
};

static const char _use_syllable_machine_trans_actions[] = {
	1, 2, 3, 0, 0, 4, 0, 5, 
	0, 0, 2, 8, 0, 9, 10, 10, 
	4, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 4, 4, 0, 0, 0, 0, 
	0, 0, 0, 11, 12, 13, 14, 15, 
	0, 2, 0, 0, 0, 0, 0, 0, 
	0, 0, 2, 0, 0, 0, 0, 0, 
	0, 2, 16, 0, 17, 0, 0, 18, 
	19, 0, 4, 0, 20, 21, 0, 0
};

static const char _use_syllable_machine_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 6, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0
};

static const char _use_syllable_machine_from_state_actions[] = {
	0, 0, 0, 0, 0, 0, 7, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0
};

static const short _use_syllable_machine_eof_trans[] = {
	1, 3, 3, 1, 1, 8, 0, 36, 
	38, 38, 59, 38, 38, 38, 38, 38, 
	38, 38, 38, 38, 38, 38, 38, 38, 
	38, 38, 38, 38, 38, 61, 64, 61, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 69, 69, 65, 65, 70, 70, 
	70, 69
};

static const int use_syllable_machine_start = 6;
static const int use_syllable_machine_first_final = 6;
static const int use_syllable_machine_error = -1;

static const int use_syllable_machine_en_main = 6;


#line 38 "hb-ot-shape-complex-use-machine.rl"



#line 141 "hb-ot-shape-complex-use-machine.rl"


#define found_syllable(syllable_type) \
  HB_STMT_START { \
    if (0) fprintf (stderr, "syllable %d..%d %s\n", last, p+1, #syllable_type); \
    for (unsigned int i = last; i < p+1; i++) \
      info[i].syllable() = (syllable_serial << 4) | syllable_type; \
    last = p+1; \
    syllable_serial++; \
    if (unlikely (syllable_serial == 16)) syllable_serial = 1; \
  } HB_STMT_END

static void
find_syllables (hb_buffer_t *buffer)
{
  unsigned int p, pe, eof, ts HB_UNUSED, te HB_UNUSED, act HB_UNUSED;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  
#line 341 "hb-ot-shape-complex-use-machine.hh"
	{
	cs = use_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 162 "hb-ot-shape-complex-use-machine.rl"


  p = 0;
  pe = eof = buffer->len;

  unsigned int last = 0;
  unsigned int syllable_serial = 1;
  
#line 358 "hb-ot-shape-complex-use-machine.hh"
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
#line 372 "hb-ot-shape-complex-use-machine.hh"
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
	case 12:
#line 130 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (independent_cluster); }}
	break;
	case 14:
#line 132 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (standard_cluster); }}
	break;
	case 9:
#line 136 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (broken_cluster); }}
	break;
	case 8:
#line 137 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (non_cluster); }}
	break;
	case 11:
#line 130 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (independent_cluster); }}
	break;
	case 16:
#line 131 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (virama_terminated_cluster); }}
	break;
	case 13:
#line 132 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (standard_cluster); }}
	break;
	case 18:
#line 133 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (number_joiner_terminated_cluster); }}
	break;
	case 17:
#line 134 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (numeral_cluster); }}
	break;
	case 21:
#line 135 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (symbol_cluster); }}
	break;
	case 19:
#line 136 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (broken_cluster); }}
	break;
	case 20:
#line 137 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (non_cluster); }}
	break;
	case 3:
#line 132 "hb-ot-shape-complex-use-machine.rl"
	{{p = ((te))-1;}{ found_syllable (standard_cluster); }}
	break;
	case 5:
#line 136 "hb-ot-shape-complex-use-machine.rl"
	{{p = ((te))-1;}{ found_syllable (broken_cluster); }}
	break;
	case 1:
#line 1 "NONE"
	{	switch( act ) {
	case 2:
	{{p = ((te))-1;} found_syllable (virama_terminated_cluster); }
	break;
	case 3:
	{{p = ((te))-1;} found_syllable (standard_cluster); }
	break;
	case 7:
	{{p = ((te))-1;} found_syllable (broken_cluster); }
	break;
	case 8:
	{{p = ((te))-1;} found_syllable (non_cluster); }
	break;
	}
	}
	break;
	case 15:
#line 1 "NONE"
	{te = p+1;}
#line 131 "hb-ot-shape-complex-use-machine.rl"
	{act = 2;}
	break;
	case 2:
#line 1 "NONE"
	{te = p+1;}
#line 132 "hb-ot-shape-complex-use-machine.rl"
	{act = 3;}
	break;
	case 4:
#line 1 "NONE"
	{te = p+1;}
#line 136 "hb-ot-shape-complex-use-machine.rl"
	{act = 7;}
	break;
	case 10:
#line 1 "NONE"
	{te = p+1;}
#line 137 "hb-ot-shape-complex-use-machine.rl"
	{act = 8;}
	break;
#line 488 "hb-ot-shape-complex-use-machine.hh"
	}

_again:
	switch ( _use_syllable_machine_to_state_actions[cs] ) {
	case 6:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 497 "hb-ot-shape-complex-use-machine.hh"
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

#line 171 "hb-ot-shape-complex-use-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH */
