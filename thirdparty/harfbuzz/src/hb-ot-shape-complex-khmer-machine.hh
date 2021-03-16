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

#include "hb.hh"

enum khmer_syllable_type_t {
	khmer_consonant_syllable,
	khmer_broken_cluster,
	khmer_non_khmer_cluster,
};


#line 41 "hb-ot-shape-complex-khmer-machine.hh"
#define khmer_syllable_machine_ex_C 1u
#define khmer_syllable_machine_ex_Coeng 14u
#define khmer_syllable_machine_ex_DOTTEDCIRCLE 12u
#define khmer_syllable_machine_ex_PLACEHOLDER 11u
#define khmer_syllable_machine_ex_Ra 16u
#define khmer_syllable_machine_ex_Robatic 20u
#define khmer_syllable_machine_ex_V 2u
#define khmer_syllable_machine_ex_VAbv 26u
#define khmer_syllable_machine_ex_VBlw 27u
#define khmer_syllable_machine_ex_VPre 28u
#define khmer_syllable_machine_ex_VPst 29u
#define khmer_syllable_machine_ex_Xgroup 21u
#define khmer_syllable_machine_ex_Ygroup 22u
#define khmer_syllable_machine_ex_ZWJ 6u
#define khmer_syllable_machine_ex_ZWNJ 5u


#line 59 "hb-ot-shape-complex-khmer-machine.hh"
static const unsigned char _khmer_syllable_machine_trans_keys[] = {
	2u, 8u, 2u, 6u, 2u, 8u, 2u, 6u,
	0u, 0u, 2u, 6u, 2u, 8u, 2u, 6u,
	2u, 8u, 2u, 6u, 2u, 6u, 2u, 8u,
	2u, 6u, 0u, 0u, 2u, 6u, 2u, 8u,
	2u, 6u, 2u, 8u, 2u, 6u, 2u, 8u,
	0u, 11u, 2u, 11u, 2u, 11u, 2u, 11u,
	7u, 7u, 2u, 7u, 2u, 11u, 2u, 11u,
	2u, 11u, 0u, 0u, 2u, 8u, 2u, 11u,
	2u, 11u, 7u, 7u, 2u, 7u, 2u, 11u,
	2u, 11u, 0u, 0u, 2u, 11u, 2u, 11u,
	0u
};

static const signed char _khmer_syllable_machine_char_class[] = {
	0, 0, 1, 1, 2, 2, 1, 1,
	1, 1, 3, 3, 1, 4, 1, 0,
	1, 1, 1, 5, 6, 7, 1, 1,
	1, 8, 9, 10, 11, 0
};

static const short _khmer_syllable_machine_index_offsets[] = {
	0, 7, 12, 19, 24, 25, 30, 37,
	42, 49, 54, 59, 66, 71, 72, 77,
	84, 89, 96, 101, 108, 120, 130, 140,
	150, 151, 157, 167, 177, 187, 188, 195,
	205, 215, 216, 222, 232, 242, 243, 253,
	0
};

static const signed char _khmer_syllable_machine_indicies[] = {
	1, 0, 0, 2, 3, 0, 4, 1,
	0, 0, 0, 3, 1, 0, 0, 0,
	3, 0, 4, 5, 0, 0, 0, 4,
	6, 7, 0, 0, 0, 8, 9, 0,
	0, 0, 10, 0, 4, 9, 0, 0,
	0, 10, 11, 0, 0, 0, 12, 0,
	4, 11, 0, 0, 0, 12, 14, 13,
	13, 13, 15, 14, 16, 16, 16, 15,
	16, 17, 18, 16, 16, 16, 17, 19,
	20, 16, 16, 16, 21, 22, 16, 16,
	16, 23, 16, 17, 22, 16, 16, 16,
	23, 24, 16, 16, 16, 25, 16, 17,
	24, 16, 16, 16, 25, 14, 16, 16,
	26, 15, 16, 17, 29, 28, 30, 2,
	31, 28, 15, 19, 17, 23, 25, 21,
	33, 32, 34, 2, 3, 6, 4, 10,
	12, 8, 35, 32, 36, 32, 3, 6,
	4, 10, 12, 8, 5, 32, 36, 32,
	4, 6, 32, 32, 32, 8, 6, 7,
	32, 36, 32, 8, 6, 37, 32, 36,
	32, 10, 6, 4, 32, 32, 8, 38,
	32, 36, 32, 12, 6, 4, 10, 32,
	8, 35, 32, 34, 32, 3, 6, 4,
	10, 12, 8, 29, 14, 39, 39, 39,
	15, 39, 17, 41, 40, 42, 40, 15,
	19, 17, 23, 25, 21, 18, 40, 42,
	40, 17, 19, 40, 40, 40, 21, 19,
	20, 40, 42, 40, 21, 19, 43, 40,
	42, 40, 23, 19, 17, 40, 40, 21,
	44, 40, 42, 40, 25, 19, 17, 23,
	40, 21, 45, 46, 40, 31, 26, 15,
	19, 17, 23, 25, 21, 41, 40, 31,
	40, 15, 19, 17, 23, 25, 21, 0
};

static const signed char _khmer_syllable_machine_index_defaults[] = {
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 13, 16, 16, 16, 16, 16,
	16, 16, 16, 16, 28, 32, 32, 32,
	32, 32, 32, 32, 32, 32, 39, 40,
	40, 40, 40, 40, 40, 40, 40, 40,
	0
};

static const signed char _khmer_syllable_machine_cond_targs[] = {
	20, 1, 28, 22, 23, 3, 24, 5,
	25, 7, 26, 9, 27, 20, 10, 31,
	20, 32, 12, 33, 14, 34, 16, 35,
	18, 36, 39, 20, 20, 21, 30, 37,
	20, 0, 29, 2, 4, 6, 8, 20,
	20, 11, 13, 15, 17, 38, 19, 0
};

static const signed char _khmer_syllable_machine_cond_actions[] = {
	1, 0, 2, 2, 2, 0, 0, 0,
	2, 0, 2, 0, 2, 3, 0, 4,
	5, 2, 0, 0, 0, 2, 0, 2,
	0, 2, 4, 0, 8, 2, 9, 0,
	10, 0, 0, 0, 0, 0, 0, 11,
	12, 0, 0, 0, 0, 4, 0, 0
};

static const signed char _khmer_syllable_machine_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 6, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0
};

static const signed char _khmer_syllable_machine_from_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 7, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0
};

static const signed char _khmer_syllable_machine_eof_trans[] = {
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 14, 17, 17, 17, 17, 17,
	17, 17, 17, 17, 28, 33, 33, 33,
	33, 33, 33, 33, 33, 33, 40, 41,
	41, 41, 41, 41, 41, 41, 41, 41,
	0
};

static const int khmer_syllable_machine_start = 20;
static const int khmer_syllable_machine_first_final = 20;
static const int khmer_syllable_machine_error = -1;

static const int khmer_syllable_machine_en_main = 20;


#line 43 "hb-ot-shape-complex-khmer-machine.rl"



#line 86 "hb-ot-shape-complex-khmer-machine.rl"


#define found_syllable(syllable_type) \
HB_STMT_START { \
	if (0) fprintf (stderr, "syllable %d..%d %s\n", ts, te, #syllable_type); \
		for (unsigned int i = ts; i < te; i++) \
	info[i].syllable() = (syllable_serial << 4) | syllable_type; \
	syllable_serial++; \
	if (unlikely (syllable_serial == 16)) syllable_serial = 1; \
	} HB_STMT_END

static void
find_syllables_khmer (hb_buffer_t *buffer)
{
	unsigned int p, pe, eof, ts, te, act HB_UNUSED;
	int cs;
	hb_glyph_info_t *info = buffer->info;
	
#line 210 "hb-ot-shape-complex-khmer-machine.hh"
	{
		cs = (int)khmer_syllable_machine_start;
		ts = 0;
		te = 0;
		act = 0;
	}
	
#line 106 "hb-ot-shape-complex-khmer-machine.rl"
	
	
	p = 0;
	pe = eof = buffer->len;
	
	unsigned int syllable_serial = 1;
	
#line 226 "hb-ot-shape-complex-khmer-machine.hh"
	{
		unsigned int _trans = 0;
		const unsigned char * _keys;
		const signed char * _inds;
		int _ic;
		_resume: {}
		if ( p == pe && p != eof )
			goto _out;
		switch ( _khmer_syllable_machine_from_state_actions[cs] ) {
			case 7:  {
				{
#line 1 "NONE"
					{ts = p;}}
				
#line 241 "hb-ot-shape-complex-khmer-machine.hh"
				
				
				break; 
			}
		}
		
		if ( p == eof ) {
			if ( _khmer_syllable_machine_eof_trans[cs] > 0 ) {
				_trans = (unsigned int)_khmer_syllable_machine_eof_trans[cs] - 1;
			}
		}
		else {
			_keys = ( _khmer_syllable_machine_trans_keys + ((cs<<1)));
			_inds = ( _khmer_syllable_machine_indicies + (_khmer_syllable_machine_index_offsets[cs]));
			
			if ( (info[p].khmer_category()) <= 29 && (info[p].khmer_category()) >= 1 ) {
				_ic = (int)_khmer_syllable_machine_char_class[(int)(info[p].khmer_category()) - 1];
				if ( _ic <= (int)(*( _keys+1)) && _ic >= (int)(*( _keys)) )
					_trans = (unsigned int)(*( _inds + (int)( _ic - (int)(*( _keys)) ) )); 
				else
					_trans = (unsigned int)_khmer_syllable_machine_index_defaults[cs];
			}
			else {
				_trans = (unsigned int)_khmer_syllable_machine_index_defaults[cs];
			}
			
		}
		cs = (int)_khmer_syllable_machine_cond_targs[_trans];
		
		if ( _khmer_syllable_machine_cond_actions[_trans] != 0 ) {
			
			switch ( _khmer_syllable_machine_cond_actions[_trans] ) {
				case 2:  {
					{
#line 1 "NONE"
						{te = p+1;}}
					
#line 279 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
				case 8:  {
					{
#line 82 "hb-ot-shape-complex-khmer-machine.rl"
						{te = p+1;{
#line 82 "hb-ot-shape-complex-khmer-machine.rl"
								found_syllable (khmer_non_khmer_cluster); }
						}}
					
#line 292 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
				case 10:  {
					{
#line 80 "hb-ot-shape-complex-khmer-machine.rl"
						{te = p;p = p - 1;{
#line 80 "hb-ot-shape-complex-khmer-machine.rl"
								found_syllable (khmer_consonant_syllable); }
						}}
					
#line 305 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
				case 12:  {
					{
#line 81 "hb-ot-shape-complex-khmer-machine.rl"
						{te = p;p = p - 1;{
#line 81 "hb-ot-shape-complex-khmer-machine.rl"
								found_syllable (khmer_broken_cluster); }
						}}
					
#line 318 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
				case 11:  {
					{
#line 82 "hb-ot-shape-complex-khmer-machine.rl"
						{te = p;p = p - 1;{
#line 82 "hb-ot-shape-complex-khmer-machine.rl"
								found_syllable (khmer_non_khmer_cluster); }
						}}
					
#line 331 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
				case 1:  {
					{
#line 80 "hb-ot-shape-complex-khmer-machine.rl"
						{p = ((te))-1;
							{
#line 80 "hb-ot-shape-complex-khmer-machine.rl"
								found_syllable (khmer_consonant_syllable); }
						}}
					
#line 345 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
				case 5:  {
					{
#line 81 "hb-ot-shape-complex-khmer-machine.rl"
						{p = ((te))-1;
							{
#line 81 "hb-ot-shape-complex-khmer-machine.rl"
								found_syllable (khmer_broken_cluster); }
						}}
					
#line 359 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
				case 3:  {
					{
#line 1 "NONE"
						{switch( act ) {
								case 2:  {
									p = ((te))-1;
									{
#line 81 "hb-ot-shape-complex-khmer-machine.rl"
										found_syllable (khmer_broken_cluster); }
									break; 
								}
								case 3:  {
									p = ((te))-1;
									{
#line 82 "hb-ot-shape-complex-khmer-machine.rl"
										found_syllable (khmer_non_khmer_cluster); }
									break; 
								}
							}}
					}
					
#line 385 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
				case 4:  {
					{
#line 1 "NONE"
						{te = p+1;}}
					
#line 395 "hb-ot-shape-complex-khmer-machine.hh"
					
					{
#line 81 "hb-ot-shape-complex-khmer-machine.rl"
						{act = 2;}}
					
#line 401 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
				case 9:  {
					{
#line 1 "NONE"
						{te = p+1;}}
					
#line 411 "hb-ot-shape-complex-khmer-machine.hh"
					
					{
#line 82 "hb-ot-shape-complex-khmer-machine.rl"
						{act = 3;}}
					
#line 417 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
			}
			
		}
		
		if ( p == eof ) {
			if ( cs >= 20 )
				goto _out;
		}
		else {
			switch ( _khmer_syllable_machine_to_state_actions[cs] ) {
				case 6:  {
					{
#line 1 "NONE"
						{ts = 0;}}
					
#line 437 "hb-ot-shape-complex-khmer-machine.hh"
					
					
					break; 
				}
			}
			
			p += 1;
			goto _resume;
		}
		_out: {}
	}
	
#line 114 "hb-ot-shape-complex-khmer-machine.rl"
	
}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_KHMER_MACHINE_HH */
