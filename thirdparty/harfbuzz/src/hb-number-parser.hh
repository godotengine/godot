/*
 * Copyright © 2019-2026  Ebrahim Byagowi
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

// Handles 0..308; anything larger overflows double and returns HUGE_VAL.
static inline double
_pow10 (unsigned exponent)
{
  // DBL_MAX is between 10^308 and 10^309
  if (unlikely (exponent > 308)) return HUGE_VAL;

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

struct _hb_fraction
{
  void add (int digit)
  {
    if (likely (fraction <= MAX_FRACT / 10))
    {
      fraction = fraction * 10. + digit;
      ++count;
    }
    else if (drop < 0) drop = digit;
    else if (digit != 0) sticky = true;
  }

  double get_value () const
  {
    double result = fraction;
    // Round-to-nearest-even using the recorded tail
    if (drop > 5 || (drop == 5 && ((((uint64_t) fraction) & 1) || sticky)))
      result += 1;
    return result / _pow10 (count);
  }

  bool is_empty () const { return !count; }

private:
  unsigned count = 0;
  double fraction = 0;
  int drop = -1; // drop is the first digit we couldn't accumulate
  bool sticky = false; // sticky is set when any later digit was nonzero.
  static constexpr uint64_t MAX_FRACT = 0xFFFFFFFFFFFFFull; /* 2^52-1 */
};

// A variant of strtod that also gets end of buffer in its second argument.
//
// Grammar (with backtracking on partial exponents):
//
//   number  := sign? mantissa exponent?
//   sign    := '+' | '-'
//   mantissa:= digits ('.' digits?)?
//	    | '.' digits
//   exponent:= ('e'|'E') sign? digits
//
// On success, *end_ptr is set just past the longest well-formed prefix.
// On no match, *end_ptr is left at the original p (matching strtod's
// "no conversion performed" behavior, so the caller's pointer is unchanged).
static inline double
hb_strtod (const char *p, const char **end_ptr /* IN/OUT */)
{
  const char *p_original = p;
  const char *pe = *end_ptr;

  // Skip leading whitespace
  while (p < pe && ISSPACE (*p))
    p++;

  // Sign
  bool neg = false;
  if (p < pe && (*p == '+' || *p == '-'))
  {
    neg = (*p == '-');
    p++;
  }

  double value = 0;

  // Integer and fraction
  {
    const char *int_start = p;
    while (p < pe && '0' <= *p && *p <= '9')
    {
      value = value * 10. + (*p - '0');
      p++;
    }
    bool has_int = p > int_start;

    _hb_fraction frac;
    if (p < pe && *p == '.')
    {
      p++;
      while (p < pe && '0' <= *p && *p <= '9')
      {
	frac.add (*p - '0');
	p++;
      }
    }

    if (!frac.is_empty ()) value += frac.get_value ();
    else if (!has_int)
    {
      // Nothing could be parsed
      *end_ptr = p_original;
      return .0;
    }
  }

  // Exponent
  unsigned exp = 0;
  bool exp_neg = false, exp_overflow = false;
  if (p < pe && (*p == 'e' || *p == 'E'))
  {
    constexpr unsigned MAX_EXP = 0x7FFu; /* 2^11-1 */
    const char *before_exponent = p;
    p++;
    if (p < pe && (*p == '+' || *p == '-'))
    {
      exp_neg = (*p == '-');
      p++;
    }

    if (p < pe && '0' <= *p && *p <= '9')
      do
      {
	if (likely (exp * 10 + (*p - '0') <= MAX_EXP))
	  exp = exp * 10 + (*p - '0');
	else
	  exp_overflow = true;
	p++;
      } while (p < pe && '0' <= *p && *p <= '9');
    else
      p = before_exponent; // rewind
  }
  *end_ptr = p;

  if (neg) value *= -1.;

  if (unlikely (exp_overflow))
  {
    if (value == 0) return value;
    if (exp_neg)    return neg ? -.0 : .0;
    else            return neg ? -DBL_MAX : DBL_MAX;
  }

  if (exp)
  {
    if (exp_neg)
    {
      // 10^308 is the largest finite power of ten, so two reductions of 308 suffice.
      if (unlikely (exp > 308))
      {
	value /= 1e308;
	exp -= 308;
	if (unlikely (exp > 308))
	{
	  value /= 1e308;
	  exp -= 308;
	}
      }
      value /= _pow10 (exp);
    }
    else
      value *= _pow10 (exp);
  }

  return value;
}

#endif /* HB_NUMBER_PARSER_HH */
