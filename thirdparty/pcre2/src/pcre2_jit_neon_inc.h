/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
            This module by Zoltan Herczeg and Sebastian Pop
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2019 University of Cambridge

-----------------------------------------------------------------------------
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the University of Cambridge nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------------
*/

# if defined(FFCS)
#  if defined(FF_UTF)
#   define FF_FUN ffcs_utf
#  else
#   define FF_FUN ffcs
#  endif

# elif defined(FFCS_2)
#  if defined(FF_UTF)
#   define FF_FUN ffcs_2_utf
#  else
#   define FF_FUN ffcs_2
#  endif

# elif defined(FFCS_MASK)
#  if defined(FF_UTF)
#   define FF_FUN ffcs_mask_utf
#  else
#   define FF_FUN ffcs_mask
#  endif

# elif defined(FFCPS_0)
#  if defined (FF_UTF)
#   define FF_FUN ffcps_0_utf
#  else
#   define FF_FUN ffcps_0
#  endif

# elif defined (FFCPS_1)
#  if defined (FF_UTF)
#   define FF_FUN ffcps_1_utf
#  else
#   define FF_FUN ffcps_1
#  endif

# elif defined (FFCPS_DEFAULT)
#  if defined (FF_UTF)
#   define FF_FUN ffcps_default_utf
#  else
#   define FF_FUN ffcps_default
#  endif
# endif

static sljit_u8* SLJIT_FUNC FF_FUN(sljit_u8 *str_end, sljit_u8 *str_ptr, sljit_uw offs1, sljit_uw offs2, sljit_uw chars)
#undef FF_FUN
{
quad_word qw;
int_char ic;
ic.x = chars;

#if defined(FFCS)
sljit_u8 c1 = ic.c.c1;
vect_t vc1 = VDUPQ(c1);

#elif defined(FFCS_2)
sljit_u8 c1 = ic.c.c1;
vect_t vc1 = VDUPQ(c1);
sljit_u8 c2 = ic.c.c2;
vect_t vc2 = VDUPQ(c2);

#elif defined(FFCS_MASK)
sljit_u8 c1 = ic.c.c1;
vect_t vc1 = VDUPQ(c1);
sljit_u8 mask = ic.c.c2;
vect_t vmask = VDUPQ(mask);
#endif

#if defined(FFCPS)
compare_type compare1_type = compare_match1;
compare_type compare2_type = compare_match1;
vect_t cmp1a, cmp1b, cmp2a, cmp2b;
const sljit_u32 diff = IN_UCHARS(offs1 - offs2);
PCRE2_UCHAR char1a = ic.c.c1;
PCRE2_UCHAR char2a = ic.c.c3;

# ifdef FFCPS_CHAR1A2A
cmp1a = VDUPQ(char1a);
cmp2a = VDUPQ(char2a);
# else
PCRE2_UCHAR char1b = ic.c.c2;
PCRE2_UCHAR char2b = ic.c.c4;
if (char1a == char1b)
  cmp1a = VDUPQ(char1a);
else
  {
  sljit_u32 bit1 = char1a ^ char1b;
  if (is_powerof2(bit1))
    {
    compare1_type = compare_match1i;
    cmp1a = VDUPQ(char1a | bit1);
    cmp1b = VDUPQ(bit1);
    }
  else
    {
    compare1_type = compare_match2;
    cmp1a = VDUPQ(char1a);
    cmp1b = VDUPQ(char1b);
    }
  }

if (char2a == char2b)
  cmp2a = VDUPQ(char2a);
else
  {
  sljit_u32 bit2 = char2a ^ char2b;
  if (is_powerof2(bit2))
    {
    compare2_type = compare_match1i;
    cmp2a = VDUPQ(char2a | bit2);
    cmp2b = VDUPQ(bit2);
    }
  else
    {
    compare2_type = compare_match2;
    cmp2a = VDUPQ(char2a);
    cmp2b = VDUPQ(char2b);
    }
  }
# endif

str_ptr += IN_UCHARS(offs1);
#endif

#if PCRE2_CODE_UNIT_WIDTH != 8
vect_t char_mask = VDUPQ(0xff);
#endif

#if defined(FF_UTF)
restart:;
#endif

#if defined(FFCPS)
sljit_u8 *p1 = str_ptr - diff;
#endif
sljit_s32 align_offset = ((uint64_t)str_ptr & 0xf);
str_ptr = (sljit_u8 *) ((uint64_t)str_ptr & ~0xf);
vect_t data = VLD1Q(str_ptr);
#if PCRE2_CODE_UNIT_WIDTH != 8
data = VANDQ(data, char_mask);
#endif
 
#if defined(FFCS)
vect_t eq = VCEQQ(data, vc1);

#elif defined(FFCS_2)
vect_t eq1 = VCEQQ(data, vc1);
vect_t eq2 = VCEQQ(data, vc2);
vect_t eq = VORRQ(eq1, eq2);    

#elif defined(FFCS_MASK)
vect_t eq = VORRQ(data, vmask);
eq = VCEQQ(eq, vc1);

#elif defined(FFCPS)
# if defined(FFCPS_DIFF1)
vect_t prev_data = data;
# endif

vect_t data2;
if (p1 < str_ptr)
  {
  data2 = VLD1Q(str_ptr - diff);
#if PCRE2_CODE_UNIT_WIDTH != 8
  data2 = VANDQ(data2, char_mask);
#endif
  }
else
  data2 = shift_left_n_lanes(data, offs1 - offs2);
 
data = fast_forward_char_pair_compare(compare1_type, data, cmp1a, cmp1b);
data2 = fast_forward_char_pair_compare(compare2_type, data2, cmp2a, cmp2b);
vect_t eq = VANDQ(data, data2);
#endif

VST1Q(qw.mem, eq);
/* Ignore matches before the first STR_PTR. */
if (align_offset < 8)
  {
  qw.dw[0] >>= align_offset * 8;
  if (qw.dw[0])
    {
    str_ptr += align_offset + __builtin_ctzll(qw.dw[0]) / 8;
    goto match;
    }
  if (qw.dw[1])
    {
    str_ptr += 8 + __builtin_ctzll(qw.dw[1]) / 8;
    goto match;
    }
  }
else
  {
  qw.dw[1] >>= (align_offset - 8) * 8;
  if (qw.dw[1])
    {
    str_ptr += align_offset + __builtin_ctzll(qw.dw[1]) / 8;
    goto match;
    }
  }
str_ptr += 16;

while (str_ptr < str_end)
  {
  vect_t orig_data = VLD1Q(str_ptr);
#if PCRE2_CODE_UNIT_WIDTH != 8
  orig_data = VANDQ(orig_data, char_mask);
#endif
  data = orig_data;

#if defined(FFCS)
  eq = VCEQQ(data, vc1);

#elif defined(FFCS_2)
  eq1 = VCEQQ(data, vc1);
  eq2 = VCEQQ(data, vc2);
  eq = VORRQ(eq1, eq2);    

#elif defined(FFCS_MASK)
  eq = VORRQ(data, vmask);
  eq = VCEQQ(eq, vc1);
#endif

#if defined(FFCPS)
# if defined (FFCPS_DIFF1)
  data2 = VEXTQ(prev_data, data, VECTOR_FACTOR - 1);
# else
  data2 = VLD1Q(str_ptr - diff);
#  if PCRE2_CODE_UNIT_WIDTH != 8
  data2 = VANDQ(data2, char_mask);
#  endif
# endif

# ifdef FFCPS_CHAR1A2A
  data = VCEQQ(data, cmp1a);
  data2 = VCEQQ(data2, cmp2a);
# else
  data = fast_forward_char_pair_compare(compare1_type, data, cmp1a, cmp1b);
  data2 = fast_forward_char_pair_compare(compare2_type, data2, cmp2a, cmp2b);
# endif

  eq = VANDQ(data, data2);
#endif

  VST1Q(qw.mem, eq);
  if (qw.dw[0])
    str_ptr += __builtin_ctzll(qw.dw[0]) / 8;
  else if (qw.dw[1])
    str_ptr += 8 + __builtin_ctzll(qw.dw[1]) / 8;
  else {
    str_ptr += 16;
#if defined (FFCPS_DIFF1)
    prev_data = orig_data;
#endif
    continue;
  }

match:;
  if (str_ptr >= str_end)
    /* Failed match. */
    return NULL;

#if defined(FF_UTF)
  if (utf_continue(str_ptr + IN_UCHARS(-offs1)))
    {
    /* Not a match. */
    str_ptr += IN_UCHARS(1);
    goto restart;
    }
#endif

  /* Match. */
#if defined (FFCPS)
  str_ptr -= IN_UCHARS(offs1);
#endif
  return str_ptr;
  }

/* Failed match. */
return NULL;
}
