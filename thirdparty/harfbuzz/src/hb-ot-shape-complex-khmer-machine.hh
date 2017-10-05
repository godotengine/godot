
#line 1 "hb-ot-shape-complex-khmer-machine.rl"
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

#ifndef HB_OT_SHAPE_COMPLEX_KHMER_MACHINE_HH
#define HB_OT_SHAPE_COMPLEX_KHMER_MACHINE_HH

#include "hb-private.hh"


#line 36 "hb-ot-shape-complex-khmer-machine.hh"
static const unsigned char _khmer_syllable_machine_trans_keys[] = {
	7u, 7u, 1u, 16u, 13u, 13u, 1u, 16u, 7u, 13u, 7u, 7u, 1u, 16u, 13u, 13u, 
	1u, 16u, 7u, 13u, 1u, 16u, 3u, 14u, 3u, 14u, 5u, 14u, 3u, 14u, 5u, 14u, 
	8u, 8u, 3u, 13u, 3u, 8u, 8u, 8u, 3u, 8u, 3u, 14u, 3u, 14u, 5u, 14u, 
	3u, 14u, 5u, 14u, 8u, 8u, 3u, 13u, 3u, 8u, 8u, 8u, 3u, 8u, 3u, 14u, 
	3u, 14u, 7u, 13u, 7u, 7u, 1u, 16u, 0
};

static const char _khmer_syllable_machine_key_spans[] = {
	1, 16, 1, 16, 7, 1, 16, 1, 
	16, 7, 16, 12, 12, 10, 12, 10, 
	1, 11, 6, 1, 6, 12, 12, 10, 
	12, 10, 1, 11, 6, 1, 6, 12, 
	12, 7, 1, 16
};

static const short _khmer_syllable_machine_index_offsets[] = {
	0, 2, 19, 21, 38, 46, 48, 65, 
	67, 84, 92, 109, 122, 135, 146, 159, 
	170, 172, 184, 191, 193, 200, 213, 226, 
	237, 250, 261, 263, 275, 282, 284, 291, 
	304, 317, 325, 327
};

static const char _khmer_syllable_machine_indicies[] = {
	1, 0, 2, 2, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 2, 0, 3, 0, 4, 4, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 4, 0, 1, 0, 
	0, 0, 0, 0, 5, 0, 7, 6, 
	8, 8, 6, 6, 6, 6, 6, 6, 
	6, 6, 6, 6, 6, 6, 6, 8, 
	6, 9, 6, 10, 10, 6, 6, 6, 
	6, 6, 6, 6, 6, 6, 6, 6, 
	6, 6, 10, 6, 7, 6, 6, 6, 
	6, 6, 11, 6, 4, 4, 13, 12, 
	14, 15, 7, 16, 12, 12, 4, 4, 
	11, 17, 12, 4, 12, 19, 18, 20, 
	21, 1, 22, 18, 18, 18, 18, 5, 
	23, 18, 24, 18, 21, 21, 1, 22, 
	18, 18, 18, 18, 18, 23, 18, 21, 
	21, 1, 22, 18, 18, 18, 18, 18, 
	23, 18, 25, 18, 21, 21, 1, 22, 
	18, 18, 18, 18, 18, 26, 18, 21, 
	21, 1, 22, 18, 18, 18, 18, 18, 
	26, 18, 27, 18, 28, 18, 29, 18, 
	18, 22, 18, 18, 18, 18, 3, 18, 
	30, 18, 18, 18, 18, 22, 18, 22, 
	18, 28, 18, 18, 18, 18, 22, 18, 
	19, 18, 21, 21, 1, 22, 18, 18, 
	18, 18, 18, 23, 18, 32, 31, 33, 
	33, 7, 16, 31, 31, 31, 31, 31, 
	34, 31, 33, 33, 7, 16, 31, 31, 
	31, 31, 31, 34, 31, 35, 31, 33, 
	33, 7, 16, 31, 31, 31, 31, 31, 
	36, 31, 33, 33, 7, 16, 31, 31, 
	31, 31, 31, 36, 31, 37, 31, 38, 
	31, 39, 31, 31, 16, 31, 31, 31, 
	31, 9, 31, 40, 31, 31, 31, 31, 
	16, 31, 16, 31, 38, 31, 31, 31, 
	31, 16, 31, 13, 31, 41, 33, 7, 
	16, 31, 31, 31, 31, 11, 34, 31, 
	13, 31, 33, 33, 7, 16, 31, 31, 
	31, 31, 31, 34, 31, 7, 42, 42, 
	42, 42, 42, 11, 42, 7, 42, 10, 
	10, 42, 42, 42, 42, 42, 42, 42, 
	42, 42, 42, 42, 42, 42, 10, 42, 
	0
};

static const char _khmer_syllable_machine_trans_targs[] = {
	10, 14, 17, 20, 11, 21, 10, 24, 
	27, 30, 31, 32, 10, 22, 33, 34, 
	26, 35, 10, 12, 4, 0, 16, 3, 
	13, 15, 1, 10, 18, 2, 19, 10, 
	23, 5, 8, 25, 6, 10, 28, 7, 
	29, 9, 10
};

static const char _khmer_syllable_machine_trans_actions[] = {
	1, 2, 2, 0, 2, 2, 3, 2, 
	2, 0, 2, 2, 6, 2, 0, 0, 
	0, 0, 7, 2, 0, 0, 0, 0, 
	2, 2, 0, 8, 0, 0, 0, 9, 
	2, 0, 0, 2, 0, 10, 0, 0, 
	0, 0, 11
};

static const char _khmer_syllable_machine_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 4, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0
};

static const char _khmer_syllable_machine_from_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 5, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0
};

static const unsigned char _khmer_syllable_machine_eof_trans[] = {
	1, 1, 1, 1, 1, 7, 7, 7, 
	7, 7, 0, 19, 19, 19, 19, 19, 
	19, 19, 19, 19, 19, 19, 32, 32, 
	32, 32, 32, 32, 32, 32, 32, 32, 
	32, 43, 43, 43
};

static const int khmer_syllable_machine_start = 10;
static const int khmer_syllable_machine_first_final = 10;
static const int khmer_syllable_machine_error = -1;

static const int khmer_syllable_machine_en_main = 10;


#line 36 "hb-ot-shape-complex-khmer-machine.rl"



#line 74 "hb-ot-shape-complex-khmer-machine.rl"


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
  unsigned int p, pe, eof, ts HB_UNUSED, te, act HB_UNUSED;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  
#line 181 "hb-ot-shape-complex-khmer-machine.hh"
	{
	cs = khmer_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 95 "hb-ot-shape-complex-khmer-machine.rl"


  p = 0;
  pe = eof = buffer->len;

  unsigned int last = 0;
  unsigned int syllable_serial = 1;
  
#line 198 "hb-ot-shape-complex-khmer-machine.hh"
	{
	int _slen;
	int _trans;
	const unsigned char *_keys;
	const char *_inds;
	if ( p == pe )
		goto _test_eof;
_resume:
	switch ( _khmer_syllable_machine_from_state_actions[cs] ) {
	case 5:
#line 1 "NONE"
	{ts = p;}
	break;
#line 212 "hb-ot-shape-complex-khmer-machine.hh"
	}

	_keys = _khmer_syllable_machine_trans_keys + (cs<<1);
	_inds = _khmer_syllable_machine_indicies + _khmer_syllable_machine_index_offsets[cs];

	_slen = _khmer_syllable_machine_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=( info[p].khmer_category()) &&
		( info[p].khmer_category()) <= _keys[1] ?
		( info[p].khmer_category()) - _keys[0] : _slen ];

_eof_trans:
	cs = _khmer_syllable_machine_trans_targs[_trans];

	if ( _khmer_syllable_machine_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _khmer_syllable_machine_trans_actions[_trans] ) {
	case 2:
#line 1 "NONE"
	{te = p+1;}
	break;
	case 8:
#line 68 "hb-ot-shape-complex-khmer-machine.rl"
	{te = p+1;{ found_syllable (consonant_syllable); }}
	break;
	case 10:
#line 69 "hb-ot-shape-complex-khmer-machine.rl"
	{te = p+1;{ found_syllable (broken_cluster); }}
	break;
	case 6:
#line 70 "hb-ot-shape-complex-khmer-machine.rl"
	{te = p+1;{ found_syllable (non_khmer_cluster); }}
	break;
	case 7:
#line 68 "hb-ot-shape-complex-khmer-machine.rl"
	{te = p;p--;{ found_syllable (consonant_syllable); }}
	break;
	case 9:
#line 69 "hb-ot-shape-complex-khmer-machine.rl"
	{te = p;p--;{ found_syllable (broken_cluster); }}
	break;
	case 11:
#line 70 "hb-ot-shape-complex-khmer-machine.rl"
	{te = p;p--;{ found_syllable (non_khmer_cluster); }}
	break;
	case 1:
#line 68 "hb-ot-shape-complex-khmer-machine.rl"
	{{p = ((te))-1;}{ found_syllable (consonant_syllable); }}
	break;
	case 3:
#line 69 "hb-ot-shape-complex-khmer-machine.rl"
	{{p = ((te))-1;}{ found_syllable (broken_cluster); }}
	break;
#line 266 "hb-ot-shape-complex-khmer-machine.hh"
	}

_again:
	switch ( _khmer_syllable_machine_to_state_actions[cs] ) {
	case 4:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 275 "hb-ot-shape-complex-khmer-machine.hh"
	}

	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	if ( _khmer_syllable_machine_eof_trans[cs] > 0 ) {
		_trans = _khmer_syllable_machine_eof_trans[cs] - 1;
		goto _eof_trans;
	}
	}

	}

#line 104 "hb-ot-shape-complex-khmer-machine.rl"

}

#endif /* HB_OT_SHAPE_COMPLEX_KHMER_MACHINE_HH */
