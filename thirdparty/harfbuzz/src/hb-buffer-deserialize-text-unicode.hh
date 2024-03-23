
#line 1 "hb-buffer-deserialize-text-unicode.rl"
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

#ifndef HB_BUFFER_DESERIALIZE_TEXT_UNICODE_HH
#define HB_BUFFER_DESERIALIZE_TEXT_UNICODE_HH

#include "hb.hh"


#line 36 "hb-buffer-deserialize-text-unicode.hh"
static const unsigned char _deserialize_text_unicode_trans_keys[] = {
	0u, 0u, 9u, 117u, 43u, 102u, 48u, 102u, 48u, 57u, 9u, 124u, 9u, 124u, 9u, 124u, 
	9u, 124u, 0
};

static const char _deserialize_text_unicode_key_spans[] = {
	0, 109, 60, 55, 10, 116, 116, 116, 
	116
};

static const short _deserialize_text_unicode_index_offsets[] = {
	0, 0, 110, 171, 227, 238, 355, 472, 
	589
};

static const char _deserialize_text_unicode_indicies[] = {
	0, 0, 0, 0, 0, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	0, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 2, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 2, 1, 3, 
	1, 1, 1, 1, 4, 4, 4, 4, 
	4, 4, 4, 4, 4, 4, 1, 1, 
	1, 1, 1, 1, 1, 4, 4, 4, 
	4, 4, 4, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 4, 4, 4, 
	4, 4, 4, 1, 4, 4, 4, 4, 
	4, 4, 4, 4, 4, 4, 1, 1, 
	1, 1, 1, 1, 1, 4, 4, 4, 
	4, 4, 4, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 4, 4, 4, 
	4, 4, 4, 1, 5, 6, 6, 6, 
	6, 6, 6, 6, 6, 6, 1, 7, 
	7, 7, 7, 7, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 7, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 8, 8, 
	8, 8, 8, 8, 8, 8, 8, 8, 
	1, 1, 1, 9, 1, 1, 1, 8, 
	8, 8, 8, 8, 8, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 8, 
	8, 8, 8, 8, 8, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 10, 1, 11, 11, 11, 11, 
	11, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 11, 1, 1, 1, 1, 
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
	1, 1, 1, 1, 1, 1, 1, 0, 
	1, 12, 12, 12, 12, 12, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	12, 1, 1, 1, 1, 1, 1, 1, 
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
	1, 1, 1, 1, 13, 1, 12, 12, 
	12, 12, 12, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 12, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 14, 14, 14, 
	14, 14, 14, 14, 14, 14, 14, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 13, 1, 0
};

static const char _deserialize_text_unicode_trans_targs[] = {
	1, 0, 2, 3, 5, 7, 8, 6, 
	5, 4, 1, 6, 6, 1, 8
};

static const char _deserialize_text_unicode_trans_actions[] = {
	0, 0, 1, 0, 2, 2, 2, 3, 
	0, 4, 3, 0, 5, 5, 0
};

static const char _deserialize_text_unicode_eof_actions[] = {
	0, 0, 0, 0, 0, 3, 0, 5, 
	5
};

static const int deserialize_text_unicode_start = 1;
static const int deserialize_text_unicode_first_final = 5;
static const int deserialize_text_unicode_error = 0;

static const int deserialize_text_unicode_en_main = 1;


#line 79 "hb-buffer-deserialize-text-unicode.rl"


static hb_bool_t
_hb_buffer_deserialize_text_unicode (hb_buffer_t *buffer,
				     const char *buf,
				     unsigned int buf_len,
				     const char **end_ptr,
				     hb_font_t *font)
{
  const char *p = buf, *pe = buf + buf_len, *eof = pe, *orig_pe = pe;

  while (p < pe && ISSPACE (*p))
    p++;
  if (p < pe && *p == (buffer->len ? '|' : '<'))
    *end_ptr = ++p;

  const char *end = strchr ((char *) p, '>');
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
  const hb_glyph_position_t pos = {0};
  
#line 201 "hb-buffer-deserialize-text-unicode.hh"
	{
	cs = deserialize_text_unicode_start;
	}

#line 206 "hb-buffer-deserialize-text-unicode.hh"
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
	_keys = _deserialize_text_unicode_trans_keys + (cs<<1);
	_inds = _deserialize_text_unicode_indicies + _deserialize_text_unicode_index_offsets[cs];

	_slen = _deserialize_text_unicode_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=(*p) &&
		(*p) <= _keys[1] ?
		(*p) - _keys[0] : _slen ];

	cs = _deserialize_text_unicode_trans_targs[_trans];

	if ( _deserialize_text_unicode_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _deserialize_text_unicode_trans_actions[_trans] ) {
	case 1:
#line 38 "hb-buffer-deserialize-text-unicode.rl"
	{
	hb_memset (&info, 0, sizeof (info));
}
	break;
	case 2:
#line 51 "hb-buffer-deserialize-text-unicode.rl"
	{
	tok = p;
}
	break;
	case 4:
#line 55 "hb-buffer-deserialize-text-unicode.rl"
	{if (!parse_hex (tok, p, &info.codepoint )) return false; }
	break;
	case 3:
#line 55 "hb-buffer-deserialize-text-unicode.rl"
	{if (!parse_hex (tok, p, &info.codepoint )) return false; }
#line 42 "hb-buffer-deserialize-text-unicode.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	if (buffer->have_positions)
	  buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 5:
#line 57 "hb-buffer-deserialize-text-unicode.rl"
	{ if (!parse_uint (tok, p, &info.cluster )) return false; }
#line 42 "hb-buffer-deserialize-text-unicode.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	if (buffer->have_positions)
	  buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
#line 273 "hb-buffer-deserialize-text-unicode.hh"
	}

_again:
	if ( cs == 0 )
		goto _out;
	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	switch ( _deserialize_text_unicode_eof_actions[cs] ) {
	case 3:
#line 55 "hb-buffer-deserialize-text-unicode.rl"
	{if (!parse_hex (tok, p, &info.codepoint )) return false; }
#line 42 "hb-buffer-deserialize-text-unicode.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	if (buffer->have_positions)
	  buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 5:
#line 57 "hb-buffer-deserialize-text-unicode.rl"
	{ if (!parse_uint (tok, p, &info.cluster )) return false; }
#line 42 "hb-buffer-deserialize-text-unicode.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	if (buffer->have_positions)
	  buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
#line 311 "hb-buffer-deserialize-text-unicode.hh"
	}
	}

	_out: {}
	}

#line 115 "hb-buffer-deserialize-text-unicode.rl"


  if (pe < orig_pe && *pe == '>')
  {
    pe++;
    if (p == pe)
      p++;
  }

  *end_ptr = p;

  return p == pe;
}

#endif /* HB_BUFFER_DESERIALIZE_TEXT_UNICODE_HH */
