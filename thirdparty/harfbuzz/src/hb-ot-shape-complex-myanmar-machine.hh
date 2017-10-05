
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

#include "hb-private.hh"


#line 36 "hb-ot-shape-complex-myanmar-machine.hh"
static const unsigned char _myanmar_syllable_machine_trans_keys[] = {
	1u, 32u, 3u, 30u, 5u, 29u, 5u, 8u, 5u, 29u, 3u, 25u, 5u, 25u, 5u, 25u, 
	3u, 29u, 3u, 29u, 3u, 29u, 3u, 29u, 1u, 16u, 3u, 29u, 3u, 29u, 3u, 29u, 
	3u, 29u, 3u, 29u, 3u, 30u, 3u, 29u, 3u, 29u, 3u, 29u, 3u, 29u, 5u, 29u, 
	5u, 8u, 5u, 29u, 3u, 25u, 5u, 25u, 5u, 25u, 3u, 29u, 3u, 29u, 3u, 29u, 
	3u, 29u, 3u, 30u, 3u, 29u, 1u, 32u, 3u, 29u, 3u, 29u, 3u, 29u, 3u, 29u, 
	3u, 29u, 3u, 30u, 3u, 29u, 3u, 29u, 3u, 29u, 3u, 29u, 1u, 32u, 8u, 8u, 
	0
};

static const char _myanmar_syllable_machine_key_spans[] = {
	32, 28, 25, 4, 25, 23, 21, 21, 
	27, 27, 27, 27, 16, 27, 27, 27, 
	27, 27, 28, 27, 27, 27, 27, 25, 
	4, 25, 23, 21, 21, 27, 27, 27, 
	27, 28, 27, 32, 27, 27, 27, 27, 
	27, 28, 27, 27, 27, 27, 32, 1
};

static const short _myanmar_syllable_machine_index_offsets[] = {
	0, 33, 62, 88, 93, 119, 143, 165, 
	187, 215, 243, 271, 299, 316, 344, 372, 
	400, 428, 456, 485, 513, 541, 569, 597, 
	623, 628, 654, 678, 700, 722, 750, 778, 
	806, 834, 863, 891, 924, 952, 980, 1008, 
	1036, 1064, 1093, 1121, 1149, 1177, 1205, 1238
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
	21, 21, 21, 21, 43, 21, 21, 28, 
	21, 30, 21, 32, 33, 34, 35, 36, 
	21, 22, 21, 24, 24, 21, 25, 21, 
	26, 21, 21, 21, 21, 21, 21, 21, 
	43, 21, 21, 28, 21, 21, 21, 32, 
	33, 34, 35, 36, 21, 22, 21, 24, 
	24, 21, 25, 21, 26, 21, 21, 21, 
	21, 21, 21, 21, 43, 21, 21, 28, 
	29, 30, 21, 32, 33, 34, 35, 36, 
	21, 22, 23, 24, 24, 21, 25, 21, 
	26, 21, 21, 21, 21, 21, 21, 21, 
	27, 21, 21, 28, 29, 30, 31, 32, 
	33, 34, 35, 36, 21, 3, 3, 44, 
	5, 44, 44, 44, 44, 44, 44, 44, 
	44, 44, 45, 44, 44, 44, 44, 44, 
	44, 14, 44, 44, 44, 18, 44, 3, 
	3, 44, 5, 44, 3, 3, 44, 5, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	14, 44, 44, 44, 18, 44, 46, 44, 
	3, 3, 44, 5, 44, 14, 44, 44, 
	44, 44, 44, 44, 44, 47, 44, 44, 
	44, 44, 44, 44, 14, 44, 3, 3, 
	44, 5, 44, 44, 44, 44, 44, 44, 
	44, 44, 44, 47, 44, 44, 44, 44, 
	44, 44, 14, 44, 3, 3, 44, 5, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	14, 44, 2, 44, 3, 3, 44, 5, 
	44, 6, 44, 44, 44, 44, 44, 44, 
	44, 48, 44, 44, 48, 44, 44, 44, 
	14, 49, 44, 44, 18, 44, 2, 44, 
	3, 3, 44, 5, 44, 6, 44, 44, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	44, 44, 44, 44, 14, 44, 44, 44, 
	18, 44, 2, 44, 3, 3, 44, 5, 
	44, 6, 44, 44, 44, 44, 44, 44, 
	44, 48, 44, 44, 44, 44, 44, 44, 
	14, 49, 44, 44, 18, 44, 2, 44, 
	3, 3, 44, 5, 44, 6, 44, 44, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	44, 44, 44, 44, 14, 49, 44, 44, 
	18, 44, 22, 23, 24, 24, 21, 25, 
	21, 26, 21, 21, 21, 21, 21, 21, 
	21, 50, 21, 21, 28, 29, 30, 31, 
	32, 33, 34, 35, 36, 37, 21, 22, 
	51, 24, 24, 21, 25, 21, 26, 21, 
	21, 21, 21, 21, 21, 21, 27, 21, 
	21, 28, 29, 30, 31, 32, 33, 34, 
	35, 36, 21, 1, 1, 2, 3, 3, 
	3, 44, 5, 44, 6, 1, 44, 44, 
	44, 44, 1, 44, 8, 44, 44, 10, 
	11, 12, 13, 14, 15, 16, 17, 18, 
	19, 44, 1, 44, 2, 44, 3, 3, 
	44, 5, 44, 6, 44, 44, 44, 44, 
	44, 44, 44, 8, 44, 44, 10, 11, 
	12, 13, 14, 15, 16, 17, 18, 44, 
	2, 44, 3, 3, 44, 5, 44, 6, 
	44, 44, 44, 44, 44, 44, 44, 52, 
	44, 44, 44, 44, 44, 44, 14, 15, 
	16, 17, 18, 44, 2, 44, 3, 3, 
	44, 5, 44, 6, 44, 44, 44, 44, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	44, 44, 14, 15, 16, 17, 18, 44, 
	2, 44, 3, 3, 44, 5, 44, 6, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	44, 44, 44, 44, 44, 44, 14, 15, 
	16, 44, 18, 44, 2, 44, 3, 3, 
	44, 5, 44, 6, 44, 44, 44, 44, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	44, 44, 14, 44, 16, 44, 18, 44, 
	2, 44, 3, 3, 44, 5, 44, 6, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	44, 44, 44, 44, 44, 44, 14, 15, 
	16, 17, 18, 52, 44, 2, 44, 3, 
	3, 44, 5, 44, 6, 44, 44, 44, 
	44, 44, 44, 44, 52, 44, 44, 10, 
	44, 12, 44, 14, 15, 16, 17, 18, 
	44, 2, 44, 3, 3, 44, 5, 44, 
	6, 44, 44, 44, 44, 44, 44, 44, 
	52, 44, 44, 10, 44, 44, 44, 14, 
	15, 16, 17, 18, 44, 2, 44, 3, 
	3, 44, 5, 44, 6, 44, 44, 44, 
	44, 44, 44, 44, 52, 44, 44, 10, 
	11, 12, 44, 14, 15, 16, 17, 18, 
	44, 2, 3, 3, 3, 44, 5, 44, 
	6, 44, 44, 44, 44, 44, 44, 44, 
	8, 44, 44, 10, 11, 12, 13, 14, 
	15, 16, 17, 18, 44, 1, 1, 53, 
	53, 53, 53, 53, 53, 53, 53, 1, 
	53, 53, 53, 53, 1, 53, 53, 53, 
	53, 53, 53, 53, 53, 53, 53, 53, 
	53, 53, 53, 53, 1, 53, 54, 53, 
	0
};

static const char _myanmar_syllable_machine_trans_targs[] = {
	0, 1, 23, 0, 0, 24, 30, 33, 
	36, 46, 37, 42, 43, 44, 26, 39, 
	40, 41, 29, 45, 47, 0, 2, 12, 
	0, 3, 9, 13, 14, 19, 20, 21, 
	5, 16, 17, 18, 8, 22, 4, 6, 
	7, 10, 11, 15, 0, 25, 27, 28, 
	31, 32, 34, 35, 38, 0, 0
};

static const char _myanmar_syllable_machine_trans_actions[] = {
	3, 0, 0, 4, 5, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 6, 0, 0, 
	7, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 8, 0, 0, 0, 
	0, 0, 0, 0, 0, 9, 10
};

static const char _myanmar_syllable_machine_to_state_actions[] = {
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0
};

static const char _myanmar_syllable_machine_from_state_actions[] = {
	2, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0
};

static const short _myanmar_syllable_machine_eof_trans[] = {
	0, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 22, 
	22, 22, 22, 22, 22, 22, 22, 45, 
	45, 45, 45, 45, 45, 45, 45, 45, 
	45, 22, 22, 45, 45, 45, 45, 45, 
	45, 45, 45, 45, 45, 45, 54, 54
};

static const int myanmar_syllable_machine_start = 0;
static const int myanmar_syllable_machine_first_final = 0;
static const int myanmar_syllable_machine_error = -1;

static const int myanmar_syllable_machine_en_main = 0;


#line 36 "hb-ot-shape-complex-myanmar-machine.rl"



#line 94 "hb-ot-shape-complex-myanmar-machine.rl"


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
  
#line 302 "hb-ot-shape-complex-myanmar-machine.hh"
	{
	cs = myanmar_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 115 "hb-ot-shape-complex-myanmar-machine.rl"


  p = 0;
  pe = eof = buffer->len;

  unsigned int last = 0;
  unsigned int syllable_serial = 1;
  
#line 319 "hb-ot-shape-complex-myanmar-machine.hh"
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
#line 333 "hb-ot-shape-complex-myanmar-machine.hh"
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
	case 7:
#line 86 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p+1;{ found_syllable (consonant_syllable); }}
	break;
	case 5:
#line 87 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p+1;{ found_syllable (non_myanmar_cluster); }}
	break;
	case 10:
#line 88 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p+1;{ found_syllable (punctuation_cluster); }}
	break;
	case 4:
#line 89 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p+1;{ found_syllable (broken_cluster); }}
	break;
	case 3:
#line 90 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p+1;{ found_syllable (non_myanmar_cluster); }}
	break;
	case 6:
#line 86 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (consonant_syllable); }}
	break;
	case 8:
#line 89 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (broken_cluster); }}
	break;
	case 9:
#line 90 "hb-ot-shape-complex-myanmar-machine.rl"
	{te = p;p--;{ found_syllable (non_myanmar_cluster); }}
	break;
#line 383 "hb-ot-shape-complex-myanmar-machine.hh"
	}

_again:
	switch ( _myanmar_syllable_machine_to_state_actions[cs] ) {
	case 1:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 392 "hb-ot-shape-complex-myanmar-machine.hh"
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

#line 124 "hb-ot-shape-complex-myanmar-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_MYANMAR_MACHINE_HH */
