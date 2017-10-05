
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
	12u, 12u, 1u, 15u, 1u, 1u, 12u, 12u, 0u, 43u, 21u, 21u, 8u, 39u, 8u, 39u, 
	1u, 15u, 1u, 1u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 26u, 8u, 26u, 8u, 26u, 
	8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 
	8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 13u, 21u, 4u, 4u, 13u, 13u, 8u, 39u, 
	8u, 39u, 8u, 39u, 8u, 39u, 8u, 26u, 8u, 26u, 8u, 26u, 8u, 39u, 8u, 39u, 
	8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 8u, 39u, 
	8u, 39u, 1u, 15u, 12u, 12u, 1u, 39u, 8u, 39u, 21u, 42u, 41u, 42u, 42u, 42u, 
	1u, 5u, 0
};

static const char _use_syllable_machine_key_spans[] = {
	1, 15, 1, 1, 44, 1, 32, 32, 
	15, 1, 32, 32, 32, 19, 19, 19, 
	32, 32, 32, 32, 32, 32, 32, 32, 
	32, 32, 32, 32, 9, 1, 1, 32, 
	32, 32, 32, 19, 19, 19, 32, 32, 
	32, 32, 32, 32, 32, 32, 32, 32, 
	32, 15, 1, 39, 32, 22, 2, 1, 
	5
};

static const short _use_syllable_machine_index_offsets[] = {
	0, 2, 18, 20, 22, 67, 69, 102, 
	135, 151, 153, 186, 219, 252, 272, 292, 
	312, 345, 378, 411, 444, 477, 510, 543, 
	576, 609, 642, 675, 708, 718, 720, 722, 
	755, 788, 821, 854, 874, 894, 914, 947, 
	980, 1013, 1046, 1079, 1112, 1145, 1178, 1211, 
	1244, 1277, 1293, 1295, 1335, 1368, 1391, 1394, 
	1396
};

static const char _use_syllable_machine_indicies[] = {
	1, 0, 3, 2, 2, 2, 2, 2, 
	2, 2, 2, 2, 2, 2, 2, 2, 
	4, 2, 3, 2, 6, 5, 7, 8, 
	9, 7, 10, 8, 9, 9, 11, 9, 
	9, 3, 12, 9, 9, 13, 7, 7, 
	14, 15, 9, 9, 16, 17, 18, 19, 
	20, 21, 22, 16, 23, 24, 25, 26, 
	27, 28, 9, 29, 30, 31, 9, 9, 
	9, 32, 9, 34, 33, 36, 35, 35, 
	37, 1, 35, 35, 38, 35, 35, 35, 
	35, 35, 39, 40, 41, 42, 43, 44, 
	45, 46, 40, 47, 39, 48, 49, 50, 
	51, 35, 52, 53, 54, 35, 36, 35, 
	35, 37, 1, 35, 35, 38, 35, 35, 
	35, 35, 35, 55, 40, 41, 42, 43, 
	44, 45, 46, 40, 47, 48, 48, 49, 
	50, 51, 35, 52, 53, 54, 35, 37, 
	56, 56, 56, 56, 56, 56, 56, 56, 
	56, 56, 56, 56, 56, 57, 56, 37, 
	56, 36, 35, 35, 37, 1, 35, 35, 
	38, 35, 35, 35, 35, 35, 35, 40, 
	41, 42, 43, 44, 45, 46, 40, 47, 
	48, 48, 49, 50, 51, 35, 52, 53, 
	54, 35, 36, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	40, 41, 42, 43, 44, 35, 35, 35, 
	35, 35, 35, 49, 50, 51, 35, 52, 
	53, 54, 35, 36, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 41, 42, 43, 44, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	52, 53, 54, 35, 36, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 42, 43, 44, 35, 
	36, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 43, 44, 35, 36, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 44, 35, 
	36, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	42, 43, 44, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 52, 53, 54, 
	35, 36, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 42, 43, 44, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 53, 
	54, 35, 36, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 42, 43, 44, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 54, 35, 36, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 41, 42, 43, 44, 35, 35, 
	35, 35, 35, 35, 49, 50, 51, 35, 
	52, 53, 54, 35, 36, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 41, 42, 43, 44, 35, 
	35, 35, 35, 35, 35, 35, 50, 51, 
	35, 52, 53, 54, 35, 36, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 41, 42, 43, 44, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	51, 35, 52, 53, 54, 35, 36, 35, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 40, 41, 42, 43, 
	44, 35, 46, 40, 35, 35, 35, 49, 
	50, 51, 35, 52, 53, 54, 35, 36, 
	35, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 40, 41, 42, 
	43, 44, 35, 58, 40, 35, 35, 35, 
	49, 50, 51, 35, 52, 53, 54, 35, 
	36, 35, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 40, 41, 
	42, 43, 44, 35, 35, 40, 35, 35, 
	35, 49, 50, 51, 35, 52, 53, 54, 
	35, 36, 35, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 40, 
	41, 42, 43, 44, 45, 46, 40, 35, 
	35, 35, 49, 50, 51, 35, 52, 53, 
	54, 35, 36, 35, 35, 37, 1, 35, 
	35, 38, 35, 35, 35, 35, 35, 35, 
	40, 41, 42, 43, 44, 45, 46, 40, 
	47, 35, 48, 49, 50, 51, 35, 52, 
	53, 54, 35, 36, 35, 35, 37, 1, 
	35, 35, 38, 35, 35, 35, 35, 35, 
	35, 40, 41, 42, 43, 44, 45, 46, 
	40, 47, 39, 48, 49, 50, 51, 35, 
	52, 53, 54, 35, 60, 59, 59, 59, 
	59, 59, 59, 59, 61, 59, 10, 62, 
	60, 59, 11, 63, 63, 3, 6, 63, 
	63, 64, 63, 63, 63, 63, 63, 65, 
	16, 17, 18, 19, 20, 21, 22, 16, 
	23, 25, 25, 26, 27, 28, 63, 29, 
	30, 31, 63, 11, 63, 63, 3, 6, 
	63, 63, 64, 63, 63, 63, 63, 63, 
	63, 16, 17, 18, 19, 20, 21, 22, 
	16, 23, 25, 25, 26, 27, 28, 63, 
	29, 30, 31, 63, 11, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 16, 17, 18, 19, 20, 63, 
	63, 63, 63, 63, 63, 26, 27, 28, 
	63, 29, 30, 31, 63, 11, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 17, 18, 19, 20, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 29, 30, 31, 63, 11, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 18, 19, 
	20, 63, 11, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 19, 20, 63, 11, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	20, 63, 11, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 18, 19, 20, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 29, 
	30, 31, 63, 11, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 18, 19, 20, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 30, 31, 63, 11, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 18, 19, 20, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 31, 63, 11, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 17, 18, 19, 20, 
	63, 63, 63, 63, 63, 63, 26, 27, 
	28, 63, 29, 30, 31, 63, 11, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 17, 18, 19, 
	20, 63, 63, 63, 63, 63, 63, 63, 
	27, 28, 63, 29, 30, 31, 63, 11, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 17, 18, 
	19, 20, 63, 63, 63, 63, 63, 63, 
	63, 63, 28, 63, 29, 30, 31, 63, 
	11, 63, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 16, 17, 
	18, 19, 20, 63, 22, 16, 63, 63, 
	63, 26, 27, 28, 63, 29, 30, 31, 
	63, 11, 63, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 16, 
	17, 18, 19, 20, 63, 66, 16, 63, 
	63, 63, 26, 27, 28, 63, 29, 30, 
	31, 63, 11, 63, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	16, 17, 18, 19, 20, 63, 63, 16, 
	63, 63, 63, 26, 27, 28, 63, 29, 
	30, 31, 63, 11, 63, 63, 63, 63, 
	63, 63, 63, 63, 63, 63, 63, 63, 
	63, 16, 17, 18, 19, 20, 21, 22, 
	16, 63, 63, 63, 26, 27, 28, 63, 
	29, 30, 31, 63, 11, 63, 63, 3, 
	6, 63, 63, 64, 63, 63, 63, 63, 
	63, 63, 16, 17, 18, 19, 20, 21, 
	22, 16, 23, 63, 25, 26, 27, 28, 
	63, 29, 30, 31, 63, 3, 67, 67, 
	67, 67, 67, 67, 67, 67, 67, 67, 
	67, 67, 67, 4, 67, 6, 67, 8, 
	63, 63, 63, 8, 63, 63, 11, 63, 
	63, 3, 6, 63, 63, 64, 63, 63, 
	63, 63, 63, 63, 16, 17, 18, 19, 
	20, 21, 22, 16, 23, 24, 25, 26, 
	27, 28, 63, 29, 30, 31, 63, 11, 
	63, 63, 3, 6, 63, 63, 64, 63, 
	63, 63, 63, 63, 63, 16, 17, 18, 
	19, 20, 21, 22, 16, 23, 24, 25, 
	26, 27, 28, 63, 29, 30, 31, 63, 
	69, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 69, 70, 68, 69, 
	70, 68, 70, 68, 8, 67, 67, 67, 
	8, 67, 0
};

static const char _use_syllable_machine_trans_targs[] = {
	4, 8, 4, 31, 2, 4, 1, 5, 
	6, 4, 28, 4, 49, 50, 51, 53, 
	33, 34, 35, 36, 37, 44, 45, 47, 
	52, 48, 41, 42, 43, 38, 39, 40, 
	56, 4, 4, 4, 4, 7, 0, 27, 
	11, 12, 13, 14, 15, 22, 23, 25, 
	26, 19, 20, 21, 16, 17, 18, 10, 
	4, 9, 24, 4, 29, 30, 4, 4, 
	3, 32, 46, 4, 4, 54, 55
};

static const char _use_syllable_machine_trans_actions[] = {
	1, 0, 2, 3, 0, 4, 0, 0, 
	7, 8, 0, 9, 10, 10, 3, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	3, 3, 0, 0, 0, 0, 0, 0, 
	0, 11, 12, 13, 14, 7, 0, 7, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	7, 0, 0, 0, 0, 0, 0, 7, 
	15, 0, 0, 16, 0, 0, 17, 18, 
	0, 3, 0, 19, 20, 0, 0
};

static const char _use_syllable_machine_to_state_actions[] = {
	0, 0, 0, 0, 5, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0
};

static const char _use_syllable_machine_from_state_actions[] = {
	0, 0, 0, 0, 6, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0
};

static const short _use_syllable_machine_eof_trans[] = {
	1, 3, 3, 6, 0, 34, 36, 36, 
	57, 57, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 36, 36, 36, 36, 
	36, 36, 36, 36, 60, 63, 60, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 64, 64, 64, 64, 64, 64, 64, 
	64, 68, 68, 64, 64, 69, 69, 69, 
	68
};

static const int use_syllable_machine_start = 4;
static const int use_syllable_machine_first_final = 4;
static const int use_syllable_machine_error = -1;

static const int use_syllable_machine_en_main = 4;


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
  unsigned int p, pe, eof, ts HB_UNUSED, te, act;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  
#line 339 "hb-ot-shape-complex-use-machine.hh"
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
  
#line 356 "hb-ot-shape-complex-use-machine.hh"
	{
	int _slen;
	int _trans;
	const unsigned char *_keys;
	const char *_inds;
	if ( p == pe )
		goto _test_eof;
_resume:
	switch ( _use_syllable_machine_from_state_actions[cs] ) {
	case 6:
#line 1 "NONE"
	{ts = p;}
	break;
#line 370 "hb-ot-shape-complex-use-machine.hh"
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
	case 7:
#line 1 "NONE"
	{te = p+1;}
	break;
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
	case 15:
#line 131 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (virama_terminated_cluster); }}
	break;
	case 13:
#line 132 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (standard_cluster); }}
	break;
	case 17:
#line 133 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (number_joiner_terminated_cluster); }}
	break;
	case 16:
#line 134 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (numeral_cluster); }}
	break;
	case 20:
#line 135 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (symbol_cluster); }}
	break;
	case 18:
#line 136 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (broken_cluster); }}
	break;
	case 19:
#line 137 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (non_cluster); }}
	break;
	case 1:
#line 132 "hb-ot-shape-complex-use-machine.rl"
	{{p = ((te))-1;}{ found_syllable (standard_cluster); }}
	break;
	case 4:
#line 136 "hb-ot-shape-complex-use-machine.rl"
	{{p = ((te))-1;}{ found_syllable (broken_cluster); }}
	break;
	case 2:
#line 1 "NONE"
	{	switch( act ) {
	case 7:
	{{p = ((te))-1;} found_syllable (broken_cluster); }
	break;
	case 8:
	{{p = ((te))-1;} found_syllable (non_cluster); }
	break;
	}
	}
	break;
	case 3:
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
#line 472 "hb-ot-shape-complex-use-machine.hh"
	}

_again:
	switch ( _use_syllable_machine_to_state_actions[cs] ) {
	case 5:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 481 "hb-ot-shape-complex-use-machine.hh"
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
