
#line 1 "hb-number-parser.rl"
/*
 * Copyright © 2019  Ebrahim Byagowi
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
 */

#ifndef HB_NUMBER_PARSER_HH
#define HB_NUMBER_PARSER_HH

#include "hb.hh"


#line 32 "hb-number-parser.hh"
static const unsigned char _double_parser_trans_keys[] = {
	0u, 0u, 43u, 57u, 46u, 57u, 48u, 57u, 43u, 57u, 48u, 57u, 48u, 101u, 48u, 57u, 
	46u, 101u, 0
};

static const char _double_parser_key_spans[] = {
	0, 15, 12, 10, 15, 10, 54, 10, 
	56
};

static const unsigned char _double_parser_index_offsets[] = {
	0, 0, 16, 29, 40, 56, 67, 122, 
	133
};

static const char _double_parser_indicies[] = {
	0, 1, 2, 3, 1, 4, 4, 
	4, 4, 4, 4, 4, 4, 4, 4, 
	1, 3, 1, 4, 4, 4, 4, 4, 
	4, 4, 4, 4, 4, 1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	1, 6, 1, 7, 1, 1, 8, 8, 
	8, 8, 8, 8, 8, 8, 8, 8, 
	1, 8, 8, 8, 8, 8, 8, 8, 
	8, 8, 8, 1, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 9, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 9, 1, 8, 8, 8, 8, 8, 
	8, 8, 8, 8, 8, 1, 3, 1, 
	4, 4, 4, 4, 4, 4, 4, 4, 
	4, 4, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 9, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 9, 1, 0
};

static const char _double_parser_trans_targs[] = {
	2, 0, 2, 3, 8, 6, 5, 5, 
	7, 4
};

static const char _double_parser_trans_actions[] = {
	0, 0, 1, 0, 2, 3, 0, 4, 
	5, 0
};

static const int double_parser_start = 1;
static const int double_parser_first_final = 6;
static const int double_parser_error = 0;

static const int double_parser_en_main = 1;


#line 68 "hb-number-parser.rl"


/* Works only for n < 512 */
static inline double
_pow10 (unsigned exponent)
{
  static const double _powers_of_10[] =
  {
    1.0e+256,
    1.0e+128,
    1.0e+64,
    1.0e+32,
    1.0e+16,
    1.0e+8,
    10000.,
    100.,
    10.
  };
  unsigned mask = 1 << (ARRAY_LENGTH (_powers_of_10) - 1);
  double result = 1;
  for (const double *power = _powers_of_10; mask; ++power, mask >>= 1)
    if (exponent & mask) result *= *power;
  return result;
}

/* a variant of strtod that also gets end of buffer in its second argument */
static inline double
strtod_rl (const char *p, const char **end_ptr /* IN/OUT */)
{
  double value = 0;
  double frac = 0;
  double frac_count = 0;
  unsigned exp = 0;
  bool neg = false, exp_neg = false, exp_overflow = false;
  const unsigned long long MAX_FRACT = 0xFFFFFFFFFFFFFull; /* 2^52-1 */
  const unsigned MAX_EXP = 0x7FFu; /* 2^11-1 */

  const char *pe = *end_ptr;
  while (p < pe && ISSPACE (*p))
    p++;

  int cs;
  
#line 132 "hb-number-parser.hh"
	{
	cs = double_parser_start;
	}

#line 135 "hb-number-parser.hh"
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
	_keys = _double_parser_trans_keys + (cs<<1);
	_inds = _double_parser_indicies + _double_parser_index_offsets[cs];

	_slen = _double_parser_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=(*p) &&
		(*p) <= _keys[1] ?
		(*p) - _keys[0] : _slen ];

	cs = _double_parser_trans_targs[_trans];

	if ( _double_parser_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _double_parser_trans_actions[_trans] ) {
	case 1:
#line 37 "hb-number-parser.rl"
	{ neg = true; }
	break;
	case 4:
#line 38 "hb-number-parser.rl"
	{ exp_neg = true; }
	break;
	case 2:
#line 40 "hb-number-parser.rl"
	{
	value = value * 10. + ((*p) - '0');
}
	break;
	case 3:
#line 43 "hb-number-parser.rl"
	{
	if (likely (frac <= MAX_FRACT / 10))
	{
	  frac = frac * 10. + ((*p) - '0');
	  ++frac_count;
	}
}
	break;
	case 5:
#line 50 "hb-number-parser.rl"
	{
	if (likely (exp * 10 + ((*p) - '0') <= MAX_EXP))
	  exp = exp * 10 + ((*p) - '0');
	else
	  exp_overflow = true;
}
	break;
#line 187 "hb-number-parser.hh"
	}

_again:
	if ( cs == 0 )
		goto _out;
	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	_out: {}
	}

#line 113 "hb-number-parser.rl"


  *end_ptr = p;

  if (frac_count) value += frac / _pow10 (frac_count);
  if (neg) value *= -1.;

  if (unlikely (exp_overflow))
  {
    if (value == 0) return value;
    if (exp_neg)    return neg ? -DBL_MIN : DBL_MIN;
    else            return neg ? -DBL_MAX : DBL_MAX;
  }

  if (exp)
  {
    if (exp_neg) value /= _pow10 (exp);
    else         value *= _pow10 (exp);
  }

  return value;
}

#endif /* HB_NUMBER_PARSER_HH */
