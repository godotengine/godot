
#line 1 "hb-buffer-deserialize-text-glyphs.rl"
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

#ifndef HB_BUFFER_DESERIALIZE_TEXT_GLYPHS_HH
#define HB_BUFFER_DESERIALIZE_TEXT_GLYPHS_HH

#include "hb.hh"


#line 36 "hb-buffer-deserialize-text-glyphs.hh"
static const unsigned char _deserialize_text_glyphs_trans_keys[] = {
	0u, 0u, 48u, 57u, 45u, 57u, 48u, 57u, 45u, 57u, 48u, 57u, 48u, 57u, 45u, 57u, 
	48u, 57u, 44u, 44u, 45u, 57u, 48u, 57u, 44u, 57u, 43u, 124u, 9u, 124u, 9u, 124u, 
	9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 
	9u, 124u, 9u, 124u, 9u, 124u, 0
};

static const char _deserialize_text_glyphs_key_spans[] = {
	0, 10, 13, 10, 13, 10, 10, 13, 
	10, 1, 13, 10, 14, 82, 116, 116, 
	116, 116, 116, 116, 116, 116, 116, 116, 
	116, 116, 116
};

static const short _deserialize_text_glyphs_index_offsets[] = {
	0, 0, 11, 25, 36, 50, 61, 72, 
	86, 97, 99, 113, 124, 139, 222, 339, 
	456, 573, 690, 807, 924, 1041, 1158, 1275, 
	1392, 1509, 1626
};

static const char _deserialize_text_glyphs_indicies[] = {
	0, 2, 2, 2, 2, 2, 2, 
	2, 2, 2, 1, 3, 1, 1, 4, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 1, 6, 7, 7, 7, 7, 7, 
	7, 7, 7, 7, 1, 8, 1, 1, 
	9, 10, 10, 10, 10, 10, 10, 10, 
	10, 10, 1, 11, 12, 12, 12, 12, 
	12, 12, 12, 12, 12, 1, 13, 14, 
	14, 14, 14, 14, 14, 14, 14, 14, 
	1, 15, 1, 1, 16, 17, 17, 17, 
	17, 17, 17, 17, 17, 17, 1, 18, 
	19, 19, 19, 19, 19, 19, 19, 19, 
	19, 1, 20, 1, 21, 1, 1, 22, 
	23, 23, 23, 23, 23, 23, 23, 23, 
	23, 1, 24, 25, 25, 25, 25, 25, 
	25, 25, 25, 25, 1, 20, 1, 1, 
	1, 19, 19, 19, 19, 19, 19, 19, 
	19, 19, 19, 1, 26, 26, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 26, 1, 
	1, 26, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 26, 26, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 26, 1, 28, 
	28, 28, 28, 28, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 27, 28, 27, 
	27, 29, 27, 27, 27, 27, 27, 27, 
	27, 30, 1, 27, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 27, 27, 31, 27, 27, 32, 27, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 27, 33, 1, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 27, 28, 27, 34, 34, 34, 34, 
	34, 26, 26, 26, 26, 26, 26, 26, 
	26, 26, 26, 26, 26, 26, 26, 26, 
	26, 26, 26, 34, 26, 26, 35, 26, 
	26, 26, 26, 26, 26, 26, 36, 1, 
	26, 26, 26, 26, 26, 26, 26, 26, 
	26, 26, 26, 26, 26, 26, 26, 26, 
	37, 26, 26, 38, 26, 26, 26, 26, 
	26, 26, 26, 26, 26, 26, 26, 26, 
	26, 26, 26, 26, 26, 26, 26, 26, 
	26, 26, 26, 26, 26, 26, 26, 39, 
	1, 26, 26, 26, 26, 26, 26, 26, 
	26, 26, 26, 26, 26, 26, 26, 26, 
	26, 26, 26, 26, 26, 26, 26, 26, 
	26, 26, 26, 26, 26, 26, 26, 40, 
	26, 41, 41, 41, 41, 41, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	41, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 42, 1, 43, 43, 
	43, 43, 43, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 43, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 44, 1, 41, 41, 41, 41, 41, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 41, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 45, 45, 45, 45, 45, 45, 
	45, 45, 45, 45, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 42, 1, 
	46, 46, 46, 46, 46, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 46, 
	1, 1, 47, 1, 1, 1, 1, 1, 
	1, 1, 1, 48, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 49, 1, 50, 50, 50, 
	50, 50, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 50, 1, 1, 51, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	52, 1, 50, 50, 50, 50, 50, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 50, 1, 1, 51, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 12, 12, 12, 12, 12, 12, 12, 
	12, 12, 12, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 52, 1, 46, 
	46, 46, 46, 46, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 46, 1, 
	1, 47, 1, 1, 1, 1, 1, 1, 
	1, 1, 48, 1, 1, 1, 7, 7, 
	7, 7, 7, 7, 7, 7, 7, 7, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 49, 1, 53, 53, 53, 53, 
	53, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 53, 1, 1, 54, 1, 
	1, 1, 1, 1, 1, 1, 55, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 56, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 57, 
	1, 58, 58, 58, 58, 58, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	58, 1, 1, 59, 1, 1, 1, 1, 
	1, 1, 1, 60, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 61, 1, 58, 58, 
	58, 58, 58, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 58, 1, 1, 
	59, 1, 1, 1, 1, 1, 1, 1, 
	60, 1, 1, 1, 1, 25, 25, 25, 
	25, 25, 25, 25, 25, 25, 25, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 61, 1, 53, 53, 53, 53, 53, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 53, 1, 1, 54, 1, 1, 
	1, 1, 1, 1, 1, 55, 1, 1, 
	1, 1, 62, 62, 62, 62, 62, 62, 
	62, 62, 62, 62, 1, 1, 1, 1, 
	1, 1, 56, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 57, 1, 
	0
};

static const char _deserialize_text_glyphs_trans_targs[] = {
	16, 0, 18, 3, 19, 22, 19, 22, 
	5, 20, 21, 20, 21, 23, 26, 8, 
	9, 12, 9, 12, 10, 11, 24, 25, 
	24, 25, 15, 15, 14, 1, 2, 6, 
	7, 13, 15, 1, 2, 6, 7, 13, 
	14, 17, 14, 17, 14, 18, 17, 1, 
	4, 14, 17, 1, 14, 17, 1, 2, 
	7, 14, 17, 1, 2, 14, 26
};

static const char _deserialize_text_glyphs_trans_actions[] = {
	1, 0, 1, 1, 1, 1, 0, 0, 
	1, 1, 1, 0, 0, 1, 1, 1, 
	1, 1, 0, 0, 2, 1, 1, 1, 
	0, 0, 0, 4, 3, 5, 5, 5, 
	5, 4, 6, 7, 7, 7, 7, 0, 
	6, 8, 8, 0, 0, 0, 9, 10, 
	10, 9, 11, 12, 11, 13, 14, 14, 
	14, 13, 15, 16, 16, 15, 0
};

static const char _deserialize_text_glyphs_eof_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 3, 6, 
	8, 0, 8, 9, 11, 11, 9, 13, 
	15, 15, 13
};

static const int deserialize_text_glyphs_start = 14;
static const int deserialize_text_glyphs_first_final = 14;
static const int deserialize_text_glyphs_error = 0;

static const int deserialize_text_glyphs_en_main = 14;


#line 98 "hb-buffer-deserialize-text-glyphs.rl"


static hb_bool_t
_hb_buffer_deserialize_text_glyphs (hb_buffer_t *buffer,
				    const char *buf,
				    unsigned int buf_len,
				    const char **end_ptr,
				    hb_font_t *font)
{
  const char *p = buf, *pe = buf + buf_len, *eof = pe, *orig_pe = pe;

  /* Ensure we have positions. */
  (void) hb_buffer_get_glyph_positions (buffer, nullptr);

  while (p < pe && ISSPACE (*p))
    p++;
  if (p < pe && *p == (buffer->len ? '|' : '['))
    *end_ptr = ++p;

  const char *end = strchr ((char *) p, ']');
  if (end)
    pe = eof = end;
  else
  {
    end = strrchr ((char *) p, '|');
    if (end)
      pe = eof = end;
    else
      pe = eof = p;
  }

  const char *tok = nullptr;
  int cs;
  hb_glyph_info_t info = {0};
  hb_glyph_position_t pos = {0};
  
#line 353 "hb-buffer-deserialize-text-glyphs.hh"
	{
	cs = deserialize_text_glyphs_start;
	}

#line 358 "hb-buffer-deserialize-text-glyphs.hh"
	{
	int _slen;
	int _trans;
	const unsigned char *_keys;
	const char *_inds;
	if ( p == pe )
		goto _test_eof;
	if ( cs == 0 )
		goto _out;
_resume:
	_keys = _deserialize_text_glyphs_trans_keys + (cs<<1);
	_inds = _deserialize_text_glyphs_indicies + _deserialize_text_glyphs_index_offsets[cs];

	_slen = _deserialize_text_glyphs_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=(*p) &&
		(*p) <= _keys[1] ?
		(*p) - _keys[0] : _slen ];

	cs = _deserialize_text_glyphs_trans_targs[_trans];

	if ( _deserialize_text_glyphs_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _deserialize_text_glyphs_trans_actions[_trans] ) {
	case 1:
#line 51 "hb-buffer-deserialize-text-glyphs.rl"
	{
	tok = p;
}
	break;
	case 7:
#line 55 "hb-buffer-deserialize-text-glyphs.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
	break;
	case 14:
#line 63 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_uint (tok, p, &info.cluster )) return false; }
	break;
	case 2:
#line 64 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_int  (tok, p, &pos.x_offset )) return false; }
	break;
	case 16:
#line 65 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_int  (tok, p, &pos.y_offset )) return false; }
	break;
	case 10:
#line 66 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_int  (tok, p, &pos.x_advance)) return false; }
	break;
	case 12:
#line 67 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_int  (tok, p, &pos.y_advance)) return false; }
	break;
	case 4:
#line 38 "hb-buffer-deserialize-text-glyphs.rl"
	{
	hb_memset (&info, 0, sizeof (info));
	hb_memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text-glyphs.rl"
	{
	tok = p;
}
	break;
	case 6:
#line 55 "hb-buffer-deserialize-text-glyphs.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 13:
#line 63 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_uint (tok, p, &info.cluster )) return false; }
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 15:
#line 65 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_int  (tok, p, &pos.y_offset )) return false; }
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 9:
#line 66 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_int  (tok, p, &pos.x_advance)) return false; }
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 11:
#line 67 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_int  (tok, p, &pos.y_advance)) return false; }
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 8:
#line 68 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_uint (tok, p, &info.mask    )) return false; }
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 5:
#line 38 "hb-buffer-deserialize-text-glyphs.rl"
	{
	hb_memset (&info, 0, sizeof (info));
	hb_memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text-glyphs.rl"
	{
	tok = p;
}
#line 55 "hb-buffer-deserialize-text-glyphs.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
	break;
	case 3:
#line 38 "hb-buffer-deserialize-text-glyphs.rl"
	{
	hb_memset (&info, 0, sizeof (info));
	hb_memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text-glyphs.rl"
	{
	tok = p;
}
#line 55 "hb-buffer-deserialize-text-glyphs.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
#line 554 "hb-buffer-deserialize-text-glyphs.hh"
	}

_again:
	if ( cs == 0 )
		goto _out;
	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	switch ( _deserialize_text_glyphs_eof_actions[cs] ) {
	case 6:
#line 55 "hb-buffer-deserialize-text-glyphs.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 13:
#line 63 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_uint (tok, p, &info.cluster )) return false; }
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 15:
#line 65 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_int  (tok, p, &pos.y_offset )) return false; }
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 9:
#line 66 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_int  (tok, p, &pos.x_advance)) return false; }
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 11:
#line 67 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_int  (tok, p, &pos.y_advance)) return false; }
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 8:
#line 68 "hb-buffer-deserialize-text-glyphs.rl"
	{ if (!parse_uint (tok, p, &info.mask    )) return false; }
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 3:
#line 38 "hb-buffer-deserialize-text-glyphs.rl"
	{
	hb_memset (&info, 0, sizeof (info));
	hb_memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text-glyphs.rl"
	{
	tok = p;
}
#line 55 "hb-buffer-deserialize-text-glyphs.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 43 "hb-buffer-deserialize-text-glyphs.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
#line 671 "hb-buffer-deserialize-text-glyphs.hh"
	}
	}

	_out: {}
	}

#line 136 "hb-buffer-deserialize-text-glyphs.rl"


  if (pe < orig_pe && *pe == ']')
  {
    pe++;
    if (p == pe)
      p++;
  }

  *end_ptr = p;

  return p == pe;
}

#endif /* HB_BUFFER_DESERIALIZE_TEXT_GLYPHS_HH */
