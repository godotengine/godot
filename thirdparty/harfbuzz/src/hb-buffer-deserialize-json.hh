#line 1 "hb-buffer-deserialize-json.rl"
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

#ifndef HB_BUFFER_DESERIALIZE_JSON_HH
#define HB_BUFFER_DESERIALIZE_JSON_HH

#include "hb.hh"


#line 35 "hb-buffer-deserialize-json.hh"
static const unsigned char _deserialize_json_trans_keys[] = {
	1u, 0u, 0u, 18u, 0u, 2u, 10u, 15u,
	16u, 17u, 2u, 2u, 0u, 7u, 0u, 6u,
	5u, 6u, 0u, 19u, 0u, 19u, 0u, 19u,
	2u, 2u, 0u, 7u, 0u, 6u, 5u, 6u,
	0u, 19u, 0u, 19u, 14u, 14u, 2u, 2u,
	0u, 7u, 0u, 6u, 0u, 19u, 0u, 19u,
	16u, 17u, 2u, 2u, 0u, 7u, 0u, 6u,
	5u, 6u, 0u, 19u, 0u, 19u, 2u, 2u,
	0u, 7u, 0u, 6u, 5u, 6u, 0u, 19u,
	0u, 19u, 2u, 2u, 0u, 7u, 0u, 6u,
	2u, 8u, 0u, 19u, 2u, 8u, 0u, 19u,
	0u, 19u, 2u, 2u, 0u, 7u, 0u, 6u,
	0u, 19u, 0u, 9u, 0u, 18u, 1u, 0u,
	0u
};

static const signed char _deserialize_json_char_class[] = {
	0, 0, 0, 0, 0, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 0,
	1, 2, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 3, 4, 1, 1, 5,
	6, 6, 6, 6, 6, 6, 6, 6,
	6, 7, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 8, 9, 1, 1, 1,
	10, 1, 11, 12, 1, 1, 13, 1,
	1, 1, 1, 14, 1, 1, 1, 1,
	1, 1, 1, 1, 15, 1, 1, 16,
	17, 1, 18, 1, 19, 0
};

static const short _deserialize_json_index_offsets[] = {
	0, 0, 19, 22, 28, 30, 31, 39,
	46, 48, 68, 88, 108, 109, 117, 124,
	126, 146, 166, 167, 168, 176, 183, 203,
	223, 225, 226, 234, 241, 243, 263, 283,
	284, 292, 299, 301, 321, 341, 342, 350,
	357, 364, 384, 391, 411, 431, 432, 440,
	447, 467, 477, 496, 0
};

static const signed char _deserialize_json_indicies[] = {
	1, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 2, 3, 0, 4, 5, 6,
	7, 8, 0, 9, 10, 11, 12, 12,
	0, 0, 0, 0, 0, 0, 13, 13,
	0, 0, 0, 14, 15, 16, 18, 19,
	20, 0, 0, 21, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 22, 23, 0, 0, 3,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 24,
	20, 0, 0, 21, 0, 19, 19, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 22, 25, 25, 0, 0,
	0, 0, 0, 0, 26, 26, 0, 0,
	0, 27, 28, 29, 31, 32, 33, 0,
	0, 34, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 35, 33, 0, 0, 34, 0, 32,
	32, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 35, 36, 37,
	37, 0, 0, 0, 0, 0, 0, 38,
	38, 0, 0, 0, 0, 39, 40, 42,
	0, 0, 43, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 44, 42, 0, 0, 43, 0,
	45, 45, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 44, 46,
	47, 48, 48, 0, 0, 0, 0, 0,
	0, 49, 49, 0, 0, 0, 50, 51,
	52, 54, 55, 56, 0, 0, 57, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 58, 56,
	0, 0, 57, 0, 55, 55, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 58, 59, 59, 0, 0, 0,
	0, 0, 0, 60, 60, 0, 0, 0,
	61, 62, 63, 65, 66, 67, 0, 0,
	68, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	69, 67, 0, 0, 68, 0, 66, 66,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 69, 70, 70, 0,
	0, 0, 0, 0, 0, 71, 71, 0,
	72, 0, 0, 73, 74, 76, 75, 75,
	75, 75, 75, 77, 79, 0, 0, 80,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 81,
	75, 0, 0, 0, 0, 0, 75, 83,
	0, 0, 84, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 85, 83, 0, 0, 84, 0,
	87, 87, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 85, 88,
	88, 0, 0, 0, 0, 0, 0, 89,
	89, 0, 0, 0, 0, 90, 91, 83,
	0, 0, 84, 0, 93, 93, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 85, 94, 0, 0, 95, 0,
	0, 0, 0, 0, 96, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 2,
	0
};

static const signed char _deserialize_json_index_defaults[] = {
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	75, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0
};

static const signed char _deserialize_json_cond_targs[] = {
	0, 1, 2, 2, 3, 4, 18, 24,
	37, 45, 5, 12, 6, 7, 8, 9,
	11, 8, 9, 11, 10, 2, 49, 10,
	49, 13, 14, 15, 16, 17, 15, 16,
	17, 10, 2, 49, 19, 20, 21, 22,
	23, 22, 10, 2, 49, 23, 25, 31,
	26, 27, 28, 29, 30, 28, 29, 30,
	10, 2, 49, 32, 33, 34, 35, 36,
	34, 35, 36, 10, 2, 49, 38, 39,
	40, 43, 44, 40, 41, 42, 41, 10,
	2, 49, 43, 10, 2, 49, 44, 44,
	46, 47, 43, 48, 48, 48, 49, 50,
	51, 0
};

static const signed char _deserialize_json_cond_actions[] = {
	0, 0, 1, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 2, 2,
	2, 0, 0, 0, 3, 3, 4, 0,
	5, 0, 0, 2, 2, 2, 0, 0,
	0, 6, 6, 7, 0, 0, 0, 2,
	2, 0, 8, 8, 9, 0, 0, 0,
	0, 0, 2, 2, 2, 0, 0, 0,
	10, 10, 11, 0, 0, 2, 2, 2,
	0, 0, 0, 12, 12, 13, 0, 0,
	2, 14, 14, 0, 15, 0, 0, 16,
	16, 17, 0, 18, 18, 19, 0, 15,
	0, 0, 20, 20, 0, 21, 0, 0,
	0, 0
};

static const int deserialize_json_start = 1;
static const int deserialize_json_first_final = 49;
static const int deserialize_json_error = 0;

static const int deserialize_json_en_main = 1;


#line 108 "hb-buffer-deserialize-json.rl"


static hb_bool_t
_hb_buffer_deserialize_json (hb_buffer_t *buffer,
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
	if (p < pe && *p == (buffer->len ? ',' : '['))
		{
		*end_ptr = ++p;
	}
	
	const char *tok = nullptr;
	int cs;
	hb_glyph_info_t info = {0};
	hb_glyph_position_t pos = {0};
	
#line 223 "hb-buffer-deserialize-json.hh"
	{
		cs = (int)deserialize_json_start;
	}
	
#line 228 "hb-buffer-deserialize-json.hh"
	{
		unsigned int _trans = 0;
		const unsigned char * _keys;
		const signed char * _inds;
		int _ic;
		_resume: {}
		if ( p == pe )
			goto _out;
		_keys = ( _deserialize_json_trans_keys + ((cs<<1)));
		_inds = ( _deserialize_json_indicies + (_deserialize_json_index_offsets[cs]));
		
		if ( ( (*( p))) <= 125 && ( (*( p))) >= 9 ) {
			_ic = (int)_deserialize_json_char_class[(int)( (*( p))) - 9];
			if ( _ic <= (int)(*( _keys+1)) && _ic >= (int)(*( _keys)) )
				_trans = (unsigned int)(*( _inds + (int)( _ic - (int)(*( _keys)) ) )); 
			else
				_trans = (unsigned int)_deserialize_json_index_defaults[cs];
		}
		else {
			_trans = (unsigned int)_deserialize_json_index_defaults[cs];
		}
		
		cs = (int)_deserialize_json_cond_targs[_trans];
		
		if ( _deserialize_json_cond_actions[_trans] != 0 ) {
			
			switch ( _deserialize_json_cond_actions[_trans] ) {
				case 1:  {
					{
#line 38 "hb-buffer-deserialize-json.rl"
						
						memset (&info, 0, sizeof (info));
						memset (&pos , 0, sizeof (pos ));
					}
					
#line 264 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 5:  {
					{
#line 43 "hb-buffer-deserialize-json.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 280 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 2:  {
					{
#line 51 "hb-buffer-deserialize-json.rl"
						
						tok = p;
					}
					
#line 292 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 15:  {
					{
#line 55 "hb-buffer-deserialize-json.rl"
						if (unlikely (!buffer->ensure_glyphs ())) return false; }
					
#line 302 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 21:  {
					{
#line 56 "hb-buffer-deserialize-json.rl"
						if (unlikely (!buffer->ensure_unicode ())) return false; }
					
#line 312 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 16:  {
					{
#line 58 "hb-buffer-deserialize-json.rl"
						
						/* TODO Unescape \" and \\ if found. */
						if (!hb_font_glyph_from_string (font,
						tok, p - tok,
						&info.codepoint))
						return false;
					}
					
#line 328 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 18:  {
					{
#line 66 "hb-buffer-deserialize-json.rl"
						if (!parse_uint (tok, p, &info.codepoint)) return false; }
					
#line 338 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 8:  {
					{
#line 67 "hb-buffer-deserialize-json.rl"
						if (!parse_uint (tok, p, &info.cluster )) return false; }
					
#line 348 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 10:  {
					{
#line 68 "hb-buffer-deserialize-json.rl"
						if (!parse_int  (tok, p, &pos.x_offset )) return false; }
					
#line 358 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 12:  {
					{
#line 69 "hb-buffer-deserialize-json.rl"
						if (!parse_int  (tok, p, &pos.y_offset )) return false; }
					
#line 368 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 3:  {
					{
#line 70 "hb-buffer-deserialize-json.rl"
						if (!parse_int  (tok, p, &pos.x_advance)) return false; }
					
#line 378 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 6:  {
					{
#line 71 "hb-buffer-deserialize-json.rl"
						if (!parse_int  (tok, p, &pos.y_advance)) return false; }
					
#line 388 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 14:  {
					{
#line 51 "hb-buffer-deserialize-json.rl"
						
						tok = p;
					}
					
#line 400 "hb-buffer-deserialize-json.hh"
					
					{
#line 55 "hb-buffer-deserialize-json.rl"
						if (unlikely (!buffer->ensure_glyphs ())) return false; }
					
#line 406 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 20:  {
					{
#line 51 "hb-buffer-deserialize-json.rl"
						
						tok = p;
					}
					
#line 418 "hb-buffer-deserialize-json.hh"
					
					{
#line 56 "hb-buffer-deserialize-json.rl"
						if (unlikely (!buffer->ensure_unicode ())) return false; }
					
#line 424 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 17:  {
					{
#line 58 "hb-buffer-deserialize-json.rl"
						
						/* TODO Unescape \" and \\ if found. */
						if (!hb_font_glyph_from_string (font,
						tok, p - tok,
						&info.codepoint))
						return false;
					}
					
#line 440 "hb-buffer-deserialize-json.hh"
					
					{
#line 43 "hb-buffer-deserialize-json.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 452 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 19:  {
					{
#line 66 "hb-buffer-deserialize-json.rl"
						if (!parse_uint (tok, p, &info.codepoint)) return false; }
					
#line 462 "hb-buffer-deserialize-json.hh"
					
					{
#line 43 "hb-buffer-deserialize-json.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 474 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 9:  {
					{
#line 67 "hb-buffer-deserialize-json.rl"
						if (!parse_uint (tok, p, &info.cluster )) return false; }
					
#line 484 "hb-buffer-deserialize-json.hh"
					
					{
#line 43 "hb-buffer-deserialize-json.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 496 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 11:  {
					{
#line 68 "hb-buffer-deserialize-json.rl"
						if (!parse_int  (tok, p, &pos.x_offset )) return false; }
					
#line 506 "hb-buffer-deserialize-json.hh"
					
					{
#line 43 "hb-buffer-deserialize-json.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 518 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 13:  {
					{
#line 69 "hb-buffer-deserialize-json.rl"
						if (!parse_int  (tok, p, &pos.y_offset )) return false; }
					
#line 528 "hb-buffer-deserialize-json.hh"
					
					{
#line 43 "hb-buffer-deserialize-json.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 540 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 4:  {
					{
#line 70 "hb-buffer-deserialize-json.rl"
						if (!parse_int  (tok, p, &pos.x_advance)) return false; }
					
#line 550 "hb-buffer-deserialize-json.hh"
					
					{
#line 43 "hb-buffer-deserialize-json.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 562 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
				case 7:  {
					{
#line 71 "hb-buffer-deserialize-json.rl"
						if (!parse_int  (tok, p, &pos.y_advance)) return false; }
					
#line 572 "hb-buffer-deserialize-json.hh"
					
					{
#line 43 "hb-buffer-deserialize-json.rl"
						
						buffer->add_info (info);
						if (unlikely (!buffer->successful))
						return false;
						buffer->pos[buffer->len - 1] = pos;
						*end_ptr = p;
					}
					
#line 584 "hb-buffer-deserialize-json.hh"
					
					
					break; 
				}
			}
			
		}
		
		if ( cs != 0 ) {
			p += 1;
			goto _resume;
		}
		_out: {}
	}
	
#line 136 "hb-buffer-deserialize-json.rl"
	
	
	*end_ptr = p;
	
	return p == pe && *(p-1) != ']';
}

#endif /* HB_BUFFER_DESERIALIZE_JSON_HH */
