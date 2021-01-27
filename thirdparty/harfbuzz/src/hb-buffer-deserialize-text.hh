#line 1 "hb-buffer-deserialize-text.rl"
/*
* Copyright © 2013  Google, Inc.
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

#ifndef HB_BUFFER_DESERIALIZE_TEXT_HH
#define HB_BUFFER_DESERIALIZE_TEXT_HH

#include "hb.hh"


#line 35 "hb-buffer-deserialize-text.hh"
static const unsigned char _deserialize_text_trans_keys[] = {
	1u, 0u, 0u, 13u, 12u, 12u, 2u, 2u,
	5u, 11u, 0u, 12u, 5u, 6u, 4u, 6u,
	5u, 6u, 5u, 6u, 4u, 6u, 5u, 6u,
	3u, 3u, 4u, 6u, 5u, 6u, 3u, 6u,
	2u, 16u, 4u, 6u, 5u, 6u, 0u, 16u,
	0u, 16u, 1u, 0u, 0u, 12u, 0u, 16u,
	0u, 16u, 0u, 16u, 0u, 16u, 0u, 16u,
	0u, 16u, 0u, 16u, 0u, 16u, 0u, 16u,
	0u, 16u, 0u, 16u, 0u, 16u, 0u, 16u,
	0u, 16u, 0u
};

static const signed char _deserialize_text_char_class[] = {
	0, 0, 0, 0, 0, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 0,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 2, 3, 4, 1, 1, 5,
	6, 6, 6, 6, 6, 6, 6, 6,
	6, 1, 1, 7, 8, 9, 1, 10,
	11, 11, 11, 11, 11, 11, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 12, 1, 1, 1,
	1, 1, 13, 14, 15, 1, 1, 1,
	11, 11, 11, 11, 11, 11, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 16, 0
};

static const short _deserialize_text_index_offsets[] = {
	0, 0, 14, 15, 16, 23, 36, 38,
	41, 43, 45, 48, 50, 51, 54, 56,
	60, 75, 78, 80, 97, 114, 114, 127,
	144, 161, 178, 195, 212, 229, 246, 263,
	280, 297, 314, 331, 348, 0
};

static const signed char _deserialize_text_indicies[] = {
	1, 0, 0, 0, 0, 0, 0, 2,
	0, 0, 0, 0, 0, 3, 4, 6,
	7, 7, 0, 0, 0, 0, 7, 8,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 4, 10, 11, 13, 14,
	15, 17, 18, 20, 21, 23, 24, 25,
	27, 28, 29, 31, 32, 33, 35, 36,
	29, 0, 28, 28, 38, 38, 0, 0,
	0, 0, 38, 0, 38, 0, 0, 0,
	38, 38, 38, 40, 41, 42, 44, 45,
	47, 0, 0, 0, 0, 48, 48, 0,
	49, 50, 0, 48, 0, 0, 0, 0,
	51, 52, 0, 0, 0, 0, 0, 0,
	0, 0, 53, 0, 0, 0, 0, 0,
	0, 54, 8, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 4, 56,
	0, 0, 0, 0, 0, 0, 0, 0,
	57, 0, 0, 0, 0, 0, 0, 58,
	56, 0, 0, 0, 0, 60, 60, 0,
	0, 57, 0, 0, 0, 0, 0, 0,
	58, 63, 62, 64, 0, 62, 62, 62,
	62, 65, 62, 66, 62, 62, 62, 67,
	68, 69, 71, 38, 72, 0, 38, 38,
	38, 38, 73, 38, 74, 38, 38, 38,
	37, 75, 76, 78, 0, 0, 79, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 80, 81, 82, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 53, 83, 84, 62, 64,
	0, 62, 62, 62, 62, 65, 62, 66,
	62, 62, 62, 67, 68, 69, 86, 0,
	87, 0, 0, 0, 0, 0, 0, 0,
	88, 0, 0, 0, 0, 57, 89, 91,
	0, 92, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 93, 94,
	91, 0, 92, 0, 0, 36, 36, 0,
	0, 0, 0, 0, 0, 0, 0, 93,
	94, 86, 0, 87, 0, 0, 97, 97,
	0, 0, 0, 88, 0, 0, 0, 0,
	57, 89, 99, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 100, 101, 99, 0, 0, 0, 0,
	45, 45, 0, 0, 0, 0, 0, 0,
	0, 0, 100, 101, 78, 0, 0, 79,
	0, 18, 18, 0, 0, 0, 0, 0,
	0, 0, 0, 80, 81, 0
};

static const signed char _deserialize_text_index_defaults[] = {
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 62, 38, 0, 0, 62, 0, 0,
	0, 0, 0, 0, 0, 0
};

static const signed char _deserialize_text_cond_targs[] = {
	0, 1, 2, 25, 3, 3, 4, 19,
	5, 6, 23, 24, 7, 8, 27, 36,
	8, 27, 36, 9, 30, 33, 10, 11,
	12, 15, 11, 12, 15, 13, 13, 14,
	31, 32, 14, 31, 32, 16, 26, 17,
	18, 34, 35, 18, 34, 35, 19, 20,
	19, 6, 21, 22, 20, 21, 22, 23,
	20, 21, 22, 24, 24, 25, 26, 26,
	7, 9, 10, 16, 21, 29, 26, 26,
	7, 9, 10, 21, 29, 27, 28, 17,
	21, 29, 28, 29, 29, 30, 28, 7,
	10, 29, 31, 28, 7, 21, 29, 32,
	33, 33, 34, 28, 21, 29, 35, 36,
	0
};

static const signed char _deserialize_text_cond_actions[] = {
	0, 0, 0, 0, 1, 0, 0, 2,
	0, 0, 2, 2, 0, 3, 4, 4,
	0, 5, 5, 0, 4, 4, 0, 3,
	3, 3, 0, 0, 0, 6, 0, 3,
	4, 4, 0, 5, 5, 0, 5, 0,
	3, 4, 4, 0, 5, 5, 7, 7,
	8, 9, 7, 7, 0, 0, 0, 10,
	10, 10, 10, 10, 8, 11, 12, 13,
	14, 14, 14, 15, 11, 11, 16, 17,
	18, 18, 18, 16, 16, 19, 19, 20,
	19, 19, 0, 0, 13, 10, 10, 21,
	21, 10, 22, 22, 23, 22, 22, 22,
	10, 5, 24, 24, 24, 24, 24, 19,
	0
};

static const signed char _deserialize_text_eof_trans[] = {
	1, 2, 3, 6, 7, 9, 10, 13,
	17, 20, 23, 27, 28, 31, 35, 29,
	38, 40, 44, 47, 53, 54, 55, 56,
	60, 62, 71, 78, 83, 70, 86, 91,
	96, 97, 99, 103, 104, 0
};

static const int deserialize_text_start = 1;
static const int deserialize_text_first_final = 19;
static const int deserialize_text_error = 0;

static const int deserialize_text_en_main = 1;


#line 114 "hb-buffer-deserialize-text.rl"


static hb_bool_t
_hb_buffer_deserialize_text (hb_buffer_t *buffer,
const char *buf,
unsigned int buf_len,
const char **end_ptr,
hb_font_t *font)
{
	const char *p = buf, *pe = buf + buf_len;
	
	/* Ensure we have positions. */
	(void) hb_buffer_get_glyph_positions (buffer, nullptr);
	
	while (p < pe && ISSPACE (*p))
	p++;
	
	const char *eof = pe, *tok = nullptr;
	int cs;
	hb_glyph_info_t info = {0};
	hb_glyph_position_t pos = {0};
	
#line 204 "hb-buffer-deserialize-text.hh"
	{
		cs = (int)deserialize_text_start;
	}
	
#line 209 "hb-buffer-deserialize-text.hh"
	{
		unsigned int _trans = 0;
		const unsigned char * _keys;
		const signed char * _inds;
		int _ic;
		_resume: {}
		if ( p == pe && p != eof )
			goto _out;
		if ( p == eof ) {
			if ( _deserialize_text_eof_trans[cs] > 0 ) {
				_trans = (unsigned int)_deserialize_text_eof_trans[cs] - 1;
			}
		}
		else {
			_keys = ( _deserialize_text_trans_keys + ((cs<<1)));
			_inds = ( _deserialize_text_indicies + (_deserialize_text_index_offsets[cs]));
			
			if ( ( (*( p))) <= 124 && ( (*( p))) >= 9 ) {
				_ic = (int)_deserialize_text_char_class[(int)( (*( p))) - 9];
				if ( _ic <= (int)(*( _keys+1)) && _ic >= (int)(*( _keys)) )
					_trans = (unsigned int)(*( _inds + (int)( _ic - (int)(*( _keys)) ) )); 
				else
					_trans = (unsigned int)_deserialize_text_index_defaults[cs];
			}
			else {
				_trans = (unsigned int)_deserialize_text_index_defaults[cs];
			}
			
		}
		cs = (int)_deserialize_text_cond_targs[_trans];
		
		if ( _deserialize_text_cond_actions[_trans] != 0 ) {
			
			switch ( _deserialize_text_cond_actions[_trans] ) {
				case 1:  {
					{
#line 38 "hb-buffer-deserialize-text.rl"
						
						memset (&info, 0, sizeof (info));
						memset (&pos , 0, sizeof (pos ));
					}
					
#line 252 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 3:  {
					{
#line 51 "hb-buffer-deserialize-text.rl"
						
						tok = p;
					}
					
#line 264 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 5:  {
					{
#line 55 "hb-buffer-deserialize-text.rl"
						if (unlikely (!buffer->ensure_glyphs ())) return false; }
					
#line 274 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 8:  {
					{
#line 56 "hb-buffer-deserialize-text.rl"
						if (unlikely (!buffer->ensure_unicode ())) return false; }
					
#line 284 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 18:  {
					{
#line 58 "hb-buffer-deserialize-text.rl"
						
						/* TODO Unescape delimeters. */
						if (!hb_font_glyph_from_string (font,
						tok, p - tok,
						&info.codepoint))
						return false;
					}
					
#line 300 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 9:  {
					{
#line 66 "hb-buffer-deserialize-text.rl"
						if (!parse_hex (tok, p, &info.codepoint )) return false; }
					
#line 310 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 21:  {
					{
#line 68 "hb-buffer-deserialize-text.rl"
						if (!parse_uint (tok, p, &info.cluster )) return false; }
					
#line 320 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 6:  {
					{
#line 69 "hb-buffer-deserialize-text.rl"
						if (!parse_int  (tok, p, &pos.x_offset )) return false; }
					
#line 330 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 23:  {
					{
#line 70 "hb-buffer-deserialize-text.rl"
						if (!parse_int  (tok, p, &pos.y_offset )) return false; }
					
#line 340 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 20:  {
					{
#line 71 "hb-buffer-deserialize-text.rl"
						if (!parse_int  (tok, p, &pos.x_advance)) return false; }
					
#line 350 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 15:  {
					{
#line 38 "hb-buffer-deserialize-text.rl"
						
						memset (&info, 0, sizeof (info));
						memset (&pos , 0, sizeof (pos ));
					}
					
#line 363 "hb-buffer-deserialize-text.hh"
					
					{
#line 51 "hb-buffer-deserialize-text.rl"
						
						tok = p;
					}
					
#line 371 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 4:  {
					{
#line 51 "hb-buffer-deserialize-text.rl"
						
						tok = p;
					}
					
#line 383 "hb-buffer-deserialize-text.hh"
					
					{
#line 55 "hb-buffer-deserialize-text.rl"
						if (unlikely (!buffer->ensure_glyphs ())) return false; }
					
#line 389 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 2:  {
					{
#line 51 "hb-buffer-deserialize-text.rl"
						
						tok = p;
					}
					
#line 401 "hb-buffer-deserialize-text.hh"
					
					{
#line 56 "hb-buffer-deserialize-text.rl"
						if (unlikely (!buffer->ensure_unicode ())) return false; }
					
#line 407 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 16:  {
					{
#line 58 "hb-buffer-deserialize-text.rl"
						
						/* TODO Unescape delimeters. */
						if (!hb_font_glyph_from_string (font,
						tok, p - tok,
						&info.codepoint))
						return false;
					}
					
#line 423 "hb-buffer-deserialize-text.hh"
					
					{
#line 43 "hb-buffer-deserialize-text.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 435 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 7:  {
					{
#line 66 "hb-buffer-deserialize-text.rl"
						if (!parse_hex (tok, p, &info.codepoint )) return false; }
					
#line 445 "hb-buffer-deserialize-text.hh"
					
					{
#line 43 "hb-buffer-deserialize-text.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 457 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 10:  {
					{
#line 68 "hb-buffer-deserialize-text.rl"
						if (!parse_uint (tok, p, &info.cluster )) return false; }
					
#line 467 "hb-buffer-deserialize-text.hh"
					
					{
#line 43 "hb-buffer-deserialize-text.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 479 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 22:  {
					{
#line 70 "hb-buffer-deserialize-text.rl"
						if (!parse_int  (tok, p, &pos.y_offset )) return false; }
					
#line 489 "hb-buffer-deserialize-text.hh"
					
					{
#line 43 "hb-buffer-deserialize-text.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 501 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 19:  {
					{
#line 71 "hb-buffer-deserialize-text.rl"
						if (!parse_int  (tok, p, &pos.x_advance)) return false; }
					
#line 511 "hb-buffer-deserialize-text.hh"
					
					{
#line 43 "hb-buffer-deserialize-text.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 523 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 24:  {
					{
#line 72 "hb-buffer-deserialize-text.rl"
						if (!parse_int  (tok, p, &pos.y_advance)) return false; }
					
#line 533 "hb-buffer-deserialize-text.hh"
					
					{
#line 43 "hb-buffer-deserialize-text.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 545 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 12:  {
					{
#line 38 "hb-buffer-deserialize-text.rl"
						
						memset (&info, 0, sizeof (info));
						memset (&pos , 0, sizeof (pos ));
					}
					
#line 558 "hb-buffer-deserialize-text.hh"
					
					{
#line 51 "hb-buffer-deserialize-text.rl"
						
						tok = p;
					}
					
#line 566 "hb-buffer-deserialize-text.hh"
					
					{
#line 55 "hb-buffer-deserialize-text.rl"
						if (unlikely (!buffer->ensure_glyphs ())) return false; }
					
#line 572 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 14:  {
					{
#line 38 "hb-buffer-deserialize-text.rl"
						
						memset (&info, 0, sizeof (info));
						memset (&pos , 0, sizeof (pos ));
					}
					
#line 585 "hb-buffer-deserialize-text.hh"
					
					{
#line 51 "hb-buffer-deserialize-text.rl"
						
						tok = p;
					}
					
#line 593 "hb-buffer-deserialize-text.hh"
					
					{
#line 58 "hb-buffer-deserialize-text.rl"
						
						/* TODO Unescape delimeters. */
						if (!hb_font_glyph_from_string (font,
						tok, p - tok,
						&info.codepoint))
						return false;
					}
					
#line 605 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 17:  {
					{
#line 58 "hb-buffer-deserialize-text.rl"
						
						/* TODO Unescape delimeters. */
						if (!hb_font_glyph_from_string (font,
						tok, p - tok,
						&info.codepoint))
						return false;
					}
					
#line 621 "hb-buffer-deserialize-text.hh"
					
					{
#line 55 "hb-buffer-deserialize-text.rl"
						if (unlikely (!buffer->ensure_glyphs ())) return false; }
					
#line 627 "hb-buffer-deserialize-text.hh"
					
					{
#line 43 "hb-buffer-deserialize-text.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 639 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 11:  {
					{
#line 38 "hb-buffer-deserialize-text.rl"
						
						memset (&info, 0, sizeof (info));
						memset (&pos , 0, sizeof (pos ));
					}
					
#line 652 "hb-buffer-deserialize-text.hh"
					
					{
#line 51 "hb-buffer-deserialize-text.rl"
						
						tok = p;
					}
					
#line 660 "hb-buffer-deserialize-text.hh"
					
					{
#line 58 "hb-buffer-deserialize-text.rl"
						
						/* TODO Unescape delimeters. */
						if (!hb_font_glyph_from_string (font,
						tok, p - tok,
						&info.codepoint))
						return false;
					}
					
#line 672 "hb-buffer-deserialize-text.hh"
					
					{
#line 43 "hb-buffer-deserialize-text.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 684 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
				case 13:  {
					{
#line 38 "hb-buffer-deserialize-text.rl"
						
						memset (&info, 0, sizeof (info));
						memset (&pos , 0, sizeof (pos ));
					}
					
#line 697 "hb-buffer-deserialize-text.hh"
					
					{
#line 51 "hb-buffer-deserialize-text.rl"
						
						tok = p;
					}
					
#line 705 "hb-buffer-deserialize-text.hh"
					
					{
#line 58 "hb-buffer-deserialize-text.rl"
						
						/* TODO Unescape delimeters. */
						if (!hb_font_glyph_from_string (font,
						tok, p - tok,
						&info.codepoint))
						return false;
					}
					
#line 717 "hb-buffer-deserialize-text.hh"
					
					{
#line 55 "hb-buffer-deserialize-text.rl"
						if (unlikely (!buffer->ensure_glyphs ())) return false; }
					
#line 723 "hb-buffer-deserialize-text.hh"
					
					{
#line 43 "hb-buffer-deserialize-text.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 735 "hb-buffer-deserialize-text.hh"
					
					
					break; 
				}
			}
			
		}
		
		if ( p == eof ) {
			if ( cs >= 19 )
				goto _out;
		}
		else {
			if ( cs != 0 ) {
				p += 1;
				goto _resume;
			}
		}
		_out: {}
	}
	
#line 138 "hb-buffer-deserialize-text.rl"
	
	
	*end_ptr = p;
	
	return p == pe && *(p-1) != ']';
}

#endif /* HB_BUFFER_DESERIALIZE_TEXT_HH */
