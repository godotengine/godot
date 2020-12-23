
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
#include "hb-ot-shape-complex-machine-index.hh"


#line 39 "hb-ot-shape-complex-use-machine.hh"
static const unsigned char _use_syllable_machine_trans_keys[] = {
	1u, 1u, 1u, 1u, 0u, 51u, 11u, 48u, 11u, 48u, 1u, 1u, 22u, 48u, 23u, 48u, 
	24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 24u, 48u, 24u, 48u, 
	1u, 1u, 24u, 48u, 23u, 48u, 23u, 48u, 23u, 48u, 22u, 48u, 22u, 48u, 22u, 48u, 
	11u, 48u, 1u, 48u, 13u, 13u, 4u, 4u, 11u, 48u, 41u, 42u, 42u, 42u, 11u, 48u, 
	22u, 48u, 23u, 48u, 24u, 47u, 25u, 47u, 26u, 47u, 45u, 46u, 46u, 46u, 24u, 48u, 
	24u, 48u, 24u, 48u, 24u, 48u, 23u, 48u, 23u, 48u, 23u, 48u, 22u, 48u, 22u, 48u, 
	22u, 48u, 11u, 48u, 1u, 48u, 1u, 1u, 4u, 4u, 13u, 13u, 1u, 48u, 11u, 48u, 
	41u, 42u, 42u, 42u, 1u, 5u, 50u, 52u, 49u, 52u, 49u, 51u, 0
};

static const char _use_syllable_machine_key_spans[] = {
	1, 1, 52, 38, 38, 1, 27, 26, 
	24, 23, 22, 2, 1, 25, 25, 25, 
	1, 25, 26, 26, 26, 27, 27, 27, 
	38, 48, 1, 1, 38, 2, 1, 38, 
	27, 26, 24, 23, 22, 2, 1, 25, 
	25, 25, 25, 26, 26, 26, 27, 27, 
	27, 38, 48, 1, 1, 1, 48, 38, 
	2, 1, 5, 3, 4, 3
};

static const short _use_syllable_machine_index_offsets[] = {
	0, 2, 4, 57, 96, 135, 137, 165, 
	192, 217, 241, 264, 267, 269, 295, 321, 
	347, 349, 375, 402, 429, 456, 484, 512, 
	540, 579, 628, 630, 632, 671, 674, 676, 
	715, 743, 770, 795, 819, 842, 845, 847, 
	873, 899, 925, 951, 978, 1005, 1032, 1060, 
	1088, 1116, 1155, 1204, 1206, 1208, 1210, 1259, 
	1298, 1301, 1303, 1309, 1313, 1318
};

static const char _use_syllable_machine_indicies[] = {
	1, 0, 2, 0, 3, 4, 5, 5, 
	6, 7, 5, 5, 5, 5, 5, 1, 
	8, 9, 5, 5, 5, 5, 10, 11, 
	5, 5, 12, 13, 14, 15, 16, 17, 
	18, 12, 19, 20, 21, 22, 23, 24, 
	5, 25, 26, 27, 5, 28, 29, 30, 
	31, 32, 33, 34, 8, 35, 5, 36, 
	5, 38, 39, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 40, 41, 42, 43, 
	44, 45, 46, 40, 47, 4, 48, 49, 
	50, 51, 37, 52, 53, 54, 37, 37, 
	37, 37, 55, 56, 57, 58, 39, 37, 
	38, 39, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 40, 41, 42, 43, 44, 
	45, 46, 40, 47, 48, 48, 49, 50, 
	51, 37, 52, 53, 54, 37, 37, 37, 
	37, 55, 56, 57, 58, 39, 37, 38, 
	59, 40, 41, 42, 43, 44, 37, 37, 
	37, 37, 37, 37, 49, 50, 51, 37, 
	52, 53, 54, 37, 37, 37, 37, 41, 
	56, 57, 58, 60, 37, 41, 42, 43, 
	44, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 52, 53, 54, 37, 37, 
	37, 37, 37, 56, 57, 58, 60, 37, 
	42, 43, 44, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 56, 57, 58, 
	37, 43, 44, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 56, 57, 58, 
	37, 44, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 56, 57, 58, 37, 
	56, 57, 37, 57, 37, 42, 43, 44, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 52, 53, 54, 37, 37, 37, 
	37, 37, 56, 57, 58, 60, 37, 42, 
	43, 44, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 53, 54, 37, 
	37, 37, 37, 37, 56, 57, 58, 60, 
	37, 42, 43, 44, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	54, 37, 37, 37, 37, 37, 56, 57, 
	58, 60, 37, 62, 61, 42, 43, 44, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 56, 57, 58, 60, 37, 41, 
	42, 43, 44, 37, 37, 37, 37, 37, 
	37, 49, 50, 51, 37, 52, 53, 54, 
	37, 37, 37, 37, 41, 56, 57, 58, 
	60, 37, 41, 42, 43, 44, 37, 37, 
	37, 37, 37, 37, 37, 50, 51, 37, 
	52, 53, 54, 37, 37, 37, 37, 41, 
	56, 57, 58, 60, 37, 41, 42, 43, 
	44, 37, 37, 37, 37, 37, 37, 37, 
	37, 51, 37, 52, 53, 54, 37, 37, 
	37, 37, 41, 56, 57, 58, 60, 37, 
	40, 41, 42, 43, 44, 37, 46, 40, 
	37, 37, 37, 49, 50, 51, 37, 52, 
	53, 54, 37, 37, 37, 37, 41, 56, 
	57, 58, 60, 37, 40, 41, 42, 43, 
	44, 37, 37, 40, 37, 37, 37, 49, 
	50, 51, 37, 52, 53, 54, 37, 37, 
	37, 37, 41, 56, 57, 58, 60, 37, 
	40, 41, 42, 43, 44, 45, 46, 40, 
	37, 37, 37, 49, 50, 51, 37, 52, 
	53, 54, 37, 37, 37, 37, 41, 56, 
	57, 58, 60, 37, 38, 39, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 40, 
	41, 42, 43, 44, 45, 46, 40, 47, 
	37, 48, 49, 50, 51, 37, 52, 53, 
	54, 37, 37, 37, 37, 55, 56, 57, 
	58, 39, 37, 38, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 41, 42, 43, 44, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 52, 
	53, 54, 59, 59, 59, 59, 59, 56, 
	57, 58, 60, 59, 64, 63, 6, 65, 
	38, 39, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 40, 41, 42, 43, 44, 
	45, 46, 40, 47, 4, 48, 49, 50, 
	51, 37, 52, 53, 54, 37, 11, 66, 
	37, 55, 56, 57, 58, 39, 37, 11, 
	66, 67, 66, 67, 1, 69, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 12, 
	13, 14, 15, 16, 17, 18, 12, 19, 
	21, 21, 22, 23, 24, 68, 25, 26, 
	27, 68, 68, 68, 68, 31, 32, 33, 
	34, 69, 68, 12, 13, 14, 15, 16, 
	68, 68, 68, 68, 68, 68, 22, 23, 
	24, 68, 25, 26, 27, 68, 68, 68, 
	68, 13, 32, 33, 34, 70, 68, 13, 
	14, 15, 16, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 25, 26, 27, 
	68, 68, 68, 68, 68, 32, 33, 34, 
	70, 68, 14, 15, 16, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 32, 
	33, 34, 68, 15, 16, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 32, 
	33, 34, 68, 16, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 32, 33, 
	34, 68, 32, 33, 68, 33, 68, 14, 
	15, 16, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 25, 26, 27, 68, 
	68, 68, 68, 68, 32, 33, 34, 70, 
	68, 14, 15, 16, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 26, 
	27, 68, 68, 68, 68, 68, 32, 33, 
	34, 70, 68, 14, 15, 16, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 27, 68, 68, 68, 68, 68, 
	32, 33, 34, 70, 68, 14, 15, 16, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 32, 33, 34, 70, 68, 13, 
	14, 15, 16, 68, 68, 68, 68, 68, 
	68, 22, 23, 24, 68, 25, 26, 27, 
	68, 68, 68, 68, 13, 32, 33, 34, 
	70, 68, 13, 14, 15, 16, 68, 68, 
	68, 68, 68, 68, 68, 23, 24, 68, 
	25, 26, 27, 68, 68, 68, 68, 13, 
	32, 33, 34, 70, 68, 13, 14, 15, 
	16, 68, 68, 68, 68, 68, 68, 68, 
	68, 24, 68, 25, 26, 27, 68, 68, 
	68, 68, 13, 32, 33, 34, 70, 68, 
	12, 13, 14, 15, 16, 68, 18, 12, 
	68, 68, 68, 22, 23, 24, 68, 25, 
	26, 27, 68, 68, 68, 68, 13, 32, 
	33, 34, 70, 68, 12, 13, 14, 15, 
	16, 68, 68, 12, 68, 68, 68, 22, 
	23, 24, 68, 25, 26, 27, 68, 68, 
	68, 68, 13, 32, 33, 34, 70, 68, 
	12, 13, 14, 15, 16, 17, 18, 12, 
	68, 68, 68, 22, 23, 24, 68, 25, 
	26, 27, 68, 68, 68, 68, 13, 32, 
	33, 34, 70, 68, 1, 69, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 12, 
	13, 14, 15, 16, 17, 18, 12, 19, 
	68, 21, 22, 23, 24, 68, 25, 26, 
	27, 68, 68, 68, 68, 31, 32, 33, 
	34, 69, 68, 1, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 13, 14, 15, 16, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 25, 
	26, 27, 68, 68, 68, 68, 68, 32, 
	33, 34, 70, 68, 1, 71, 72, 68, 
	9, 68, 4, 68, 68, 68, 4, 68, 
	68, 68, 68, 68, 1, 69, 9, 68, 
	68, 68, 68, 68, 68, 68, 68, 12, 
	13, 14, 15, 16, 17, 18, 12, 19, 
	20, 21, 22, 23, 24, 68, 25, 26, 
	27, 68, 28, 29, 68, 31, 32, 33, 
	34, 69, 68, 1, 69, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 12, 13, 
	14, 15, 16, 17, 18, 12, 19, 20, 
	21, 22, 23, 24, 68, 25, 26, 27, 
	68, 68, 68, 68, 31, 32, 33, 34, 
	69, 68, 28, 29, 68, 29, 68, 4, 
	71, 71, 71, 4, 71, 74, 73, 35, 
	73, 35, 74, 73, 74, 73, 35, 73, 
	36, 73, 0
};

static const char _use_syllable_machine_trans_targs[] = {
	2, 31, 42, 2, 3, 2, 26, 28, 
	51, 52, 54, 29, 32, 33, 34, 35, 
	36, 46, 47, 48, 55, 49, 43, 44, 
	45, 39, 40, 41, 56, 57, 58, 50, 
	37, 38, 2, 59, 61, 2, 4, 5, 
	6, 7, 8, 9, 10, 21, 22, 23, 
	24, 18, 19, 20, 13, 14, 15, 25, 
	11, 12, 2, 2, 16, 2, 17, 2, 
	27, 2, 30, 2, 2, 0, 1, 2, 
	53, 2, 60
};

static const char _use_syllable_machine_trans_actions[] = {
	1, 2, 2, 5, 0, 6, 0, 0, 
	0, 0, 2, 0, 2, 2, 0, 0, 
	0, 2, 2, 2, 2, 2, 2, 2, 
	2, 2, 2, 2, 0, 0, 0, 2, 
	0, 0, 7, 0, 0, 8, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 9, 10, 0, 11, 0, 12, 
	0, 13, 0, 14, 15, 0, 0, 16, 
	0, 17, 0
};

static const char _use_syllable_machine_to_state_actions[] = {
	0, 0, 3, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0
};

static const char _use_syllable_machine_from_state_actions[] = {
	0, 0, 4, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0
};

static const short _use_syllable_machine_eof_trans[] = {
	1, 1, 0, 38, 38, 60, 38, 38, 
	38, 38, 38, 38, 38, 38, 38, 38, 
	62, 38, 38, 38, 38, 38, 38, 38, 
	38, 60, 64, 66, 38, 68, 68, 69, 
	69, 69, 69, 69, 69, 69, 69, 69, 
	69, 69, 69, 69, 69, 69, 69, 69, 
	69, 69, 69, 72, 69, 69, 69, 69, 
	69, 69, 72, 74, 74, 74
};

static const int use_syllable_machine_start = 2;
static const int use_syllable_machine_first_final = 2;
static const int use_syllable_machine_error = -1;

static const int use_syllable_machine_en_main = 2;


#line 39 "hb-ot-shape-complex-use-machine.rl"



#line 154 "hb-ot-shape-complex-use-machine.rl"


#define found_syllable(syllable_type) \
  HB_STMT_START { \
    if (0) fprintf (stderr, "syllable %d..%d %s\n", (*ts).second.first, (*te).second.first, #syllable_type); \
    for (unsigned i = (*ts).second.first; i < (*te).second.first; ++i) \
      info[i].syllable() = (syllable_serial << 4) | use_##syllable_type; \
    syllable_serial++; \
    if (unlikely (syllable_serial == 16)) syllable_serial = 1; \
  } HB_STMT_END

static bool
not_standard_default_ignorable (const hb_glyph_info_t &i)
{ return !(i.use_category() == USE_O && _hb_glyph_info_is_default_ignorable (&i)); }

static void
find_syllables_use (hb_buffer_t *buffer)
{
  hb_glyph_info_t *info = buffer->info;
  auto p =
    + hb_iter (info, buffer->len)
    | hb_enumerate
    | hb_filter ([] (const hb_glyph_info_t &i) { return not_standard_default_ignorable (i); },
		 hb_second)
    | hb_filter ([&] (const hb_pair_t<unsigned, const hb_glyph_info_t &> p)
		 {
		   if (p.second.use_category() == USE_ZWNJ)
		     for (unsigned i = p.first + 1; i < buffer->len; ++i)
		       if (not_standard_default_ignorable (info[i]))
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
  
#line 355 "hb-ot-shape-complex-use-machine.hh"
	{
	cs = use_syllable_machine_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 198 "hb-ot-shape-complex-use-machine.rl"


  unsigned int syllable_serial = 1;
  
#line 368 "hb-ot-shape-complex-use-machine.hh"
	{
	int _slen;
	int _trans;
	const unsigned char *_keys;
	const char *_inds;
	if ( p == pe )
		goto _test_eof;
_resume:
	switch ( _use_syllable_machine_from_state_actions[cs] ) {
	case 4:
#line 1 "NONE"
	{ts = p;}
	break;
#line 382 "hb-ot-shape-complex-use-machine.hh"
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
	case 2:
#line 1 "NONE"
	{te = p+1;}
	break;
	case 5:
#line 141 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (independent_cluster); }}
	break;
	case 9:
#line 144 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (standard_cluster); }}
	break;
	case 7:
#line 149 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (broken_cluster); }}
	break;
	case 6:
#line 150 "hb-ot-shape-complex-use-machine.rl"
	{te = p+1;{ found_syllable (non_cluster); }}
	break;
	case 10:
#line 142 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (virama_terminated_cluster); }}
	break;
	case 11:
#line 143 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (sakot_terminated_cluster); }}
	break;
	case 8:
#line 144 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (standard_cluster); }}
	break;
	case 13:
#line 145 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (number_joiner_terminated_cluster); }}
	break;
	case 12:
#line 146 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (numeral_cluster); }}
	break;
	case 14:
#line 147 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (symbol_cluster); }}
	break;
	case 17:
#line 148 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (hieroglyph_cluster); }}
	break;
	case 15:
#line 149 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (broken_cluster); }}
	break;
	case 16:
#line 150 "hb-ot-shape-complex-use-machine.rl"
	{te = p;p--;{ found_syllable (non_cluster); }}
	break;
	case 1:
#line 149 "hb-ot-shape-complex-use-machine.rl"
	{{p = ((te))-1;}{ found_syllable (broken_cluster); }}
	break;
#line 460 "hb-ot-shape-complex-use-machine.hh"
	}

_again:
	switch ( _use_syllable_machine_to_state_actions[cs] ) {
	case 3:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 469 "hb-ot-shape-complex-use-machine.hh"
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

#line 203 "hb-ot-shape-complex-use-machine.rl"

}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH */
