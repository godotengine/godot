/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2024 University of Cambridge

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


/* This module contains two internal functions that are used to match
OP_XCLASS and OP_ECLASS. It is used by pcre2_auto_possessify() and by both
pcre2_match() and pcre2_dfa_match(). */


#include "pcre2_internal.h"



/*************************************************
*       Match character against an XCLASS        *
*************************************************/

/* This function is called to match a character against an extended class that
might contain codepoints above 255 and/or Unicode properties.

Arguments:
  c           the character
  data        points to the flag code unit of the XCLASS data
  utf         TRUE if in UTF mode

Returns:      TRUE if character matches, else FALSE
*/

BOOL
PRIV(xclass)(uint32_t c, PCRE2_SPTR data, const uint8_t *char_lists_end, BOOL utf)
{
/* Update PRIV(update_classbits) when this function is changed. */
PCRE2_UCHAR t;
BOOL not_negated = (*data & XCL_NOT) == 0;
uint32_t type, max_index, min_index, value;
const uint8_t *next_char;

#if PCRE2_CODE_UNIT_WIDTH == 8
/* In 8 bit mode, this must always be TRUE. Help the compiler to know that. */
utf = TRUE;
#endif

/* Code points < 256 are matched against a bitmap, if one is present. */

if ((*data++ & XCL_MAP) != 0)
  {
  if (c < 256)
    return (((const uint8_t *)data)[c/8] & (1u << (c&7))) != 0;
  /* Skip bitmap. */
  data += 32 / sizeof(PCRE2_UCHAR);
  }

/* Match against the list of Unicode properties. We won't ever
encounter XCL_PROP or XCL_NOTPROP when UTF support is not compiled. */
#ifdef SUPPORT_UNICODE
if (*data == XCL_PROP || *data == XCL_NOTPROP)
  {
  /* The UCD record is the same for all properties. */
  const ucd_record *prop = GET_UCD(c);

  do
    {
    int chartype;
    BOOL isprop = (*data++) == XCL_PROP;
    BOOL ok;

    switch(*data)
      {
      case PT_LAMP:
      chartype = prop->chartype;
      if ((chartype == ucp_Lu || chartype == ucp_Ll ||
           chartype == ucp_Lt) == isprop) return not_negated;
      break;

      case PT_GC:
      if ((data[1] == PRIV(ucp_gentype)[prop->chartype]) == isprop)
        return not_negated;
      break;

      case PT_PC:
      if ((data[1] == prop->chartype) == isprop) return not_negated;
      break;

      case PT_SC:
      if ((data[1] == prop->script) == isprop) return not_negated;
      break;

      case PT_SCX:
      ok = (data[1] == prop->script ||
            MAPBIT(PRIV(ucd_script_sets) + UCD_SCRIPTX_PROP(prop), data[1]) != 0);
      if (ok == isprop) return not_negated;
      break;

      case PT_ALNUM:
      chartype = prop->chartype;
      if ((PRIV(ucp_gentype)[chartype] == ucp_L ||
           PRIV(ucp_gentype)[chartype] == ucp_N) == isprop)
        return not_negated;
      break;

      /* Perl space used to exclude VT, but from Perl 5.18 it is included,
      which means that Perl space and POSIX space are now identical. PCRE
      was changed at release 8.34. */

      case PT_SPACE:    /* Perl space */
      case PT_PXSPACE:  /* POSIX space */
      switch(c)
        {
        HSPACE_CASES:
        VSPACE_CASES:
        if (isprop) return not_negated;
        break;

        default:
        if ((PRIV(ucp_gentype)[prop->chartype] == ucp_Z) == isprop)
          return not_negated;
        break;
        }
      break;

      case PT_WORD:
      chartype = prop->chartype;
      if ((PRIV(ucp_gentype)[chartype] == ucp_L ||
           PRIV(ucp_gentype)[chartype] == ucp_N ||
           chartype == ucp_Mn || chartype == ucp_Pc) == isprop)
        return not_negated;
      break;

      case PT_UCNC:
      if (c < 0xa0)
        {
        if ((c == CHAR_DOLLAR_SIGN || c == CHAR_COMMERCIAL_AT ||
             c == CHAR_GRAVE_ACCENT) == isprop)
          return not_negated;
        }
      else
        {
        if ((c < 0xd800 || c > 0xdfff) == isprop)
          return not_negated;
        }
      break;

      case PT_BIDICL:
      if ((UCD_BIDICLASS_PROP(prop) == data[1]) == isprop)
        return not_negated;
      break;

      case PT_BOOL:
      ok = MAPBIT(PRIV(ucd_boolprop_sets) +
        UCD_BPROPS_PROP(prop), data[1]) != 0;
      if (ok == isprop) return not_negated;
      break;

      /* The following three properties can occur only in an XCLASS, as there
      is no \p or \P coding for them. */

      /* Graphic character. Implement this as not Z (space or separator) and
      not C (other), except for Cf (format) with a few exceptions. This seems
      to be what Perl does. The exceptional characters are:

      U+061C           Arabic Letter Mark
      U+180E           Mongolian Vowel Separator
      U+2066 - U+2069  Various "isolate"s
      */

      case PT_PXGRAPH:
      chartype = prop->chartype;
      if ((PRIV(ucp_gentype)[chartype] != ucp_Z &&
            (PRIV(ucp_gentype)[chartype] != ucp_C ||
              (chartype == ucp_Cf &&
                c != 0x061c && c != 0x180e && (c < 0x2066 || c > 0x2069))
         )) == isprop)
        return not_negated;
      break;

      /* Printable character: same as graphic, with the addition of Zs, i.e.
      not Zl and not Zp, and U+180E. */

      case PT_PXPRINT:
      chartype = prop->chartype;
      if ((chartype != ucp_Zl &&
           chartype != ucp_Zp &&
            (PRIV(ucp_gentype)[chartype] != ucp_C ||
              (chartype == ucp_Cf &&
                c != 0x061c && (c < 0x2066 || c > 0x2069))
         )) == isprop)
        return not_negated;
      break;

      /* Punctuation: all Unicode punctuation, plus ASCII characters that
      Unicode treats as symbols rather than punctuation, for Perl
      compatibility (these are $+<=>^`|~). */

      case PT_PXPUNCT:
      chartype = prop->chartype;
      if ((PRIV(ucp_gentype)[chartype] == ucp_P ||
            (c < 128 && PRIV(ucp_gentype)[chartype] == ucp_S)) == isprop)
        return not_negated;
      break;

      /* Perl has two sets of hex digits */

      case PT_PXXDIGIT:
      if (((c >= CHAR_0 && c <= CHAR_9) ||
           (c >= CHAR_A && c <= CHAR_F) ||
           (c >= CHAR_a && c <= CHAR_f) ||
           (c >= 0xff10 && c <= 0xff19) ||  /* Fullwidth digits */
           (c >= 0xff21 && c <= 0xff26) ||  /* Fullwidth letters */
           (c >= 0xff41 && c <= 0xff46)) == isprop)
        return not_negated;
      break;

      /* This should never occur, but compilers may mutter if there is no
      default. */

      /* LCOV_EXCL_START */
      default:
      PCRE2_DEBUG_UNREACHABLE();
      return FALSE;
      /* LCOV_EXCL_STOP */
      }

    data += 2;
    }
  while (*data == XCL_PROP || *data == XCL_NOTPROP);
  }
#else
  (void)utf;  /* Avoid compiler warning */
#endif  /* SUPPORT_UNICODE */

/* Match against large chars or ranges that end with a large char. */
if (*data < XCL_LIST)
  {
  while ((t = *data++) != XCL_END)
    {
    uint32_t x, y;

#ifdef SUPPORT_UNICODE
    if (utf)
      {
      GETCHARINC(x, data); /* macro generates multiple statements */
      }
    else
#endif
      x = *data++;

    if (t == XCL_SINGLE)
      {
      /* Since character ranges follow the properties, and they are
      sorted, early return is possible for all characters <= x. */
      if (c <= x) return (c == x) ? not_negated : !not_negated;
      continue;
      }

    PCRE2_ASSERT(t == XCL_RANGE);
#ifdef SUPPORT_UNICODE
    if (utf)
      {
      GETCHARINC(y, data); /* macro generates multiple statements */
      }
    else
#endif
      y = *data++;

    /* Since character ranges follow the properties, and they are
    sorted, early return is possible for all characters <= y. */
    if (c <= y) return (c >= x) ? not_negated : !not_negated;
    }

  return !not_negated;   /* char did not match */
  }

#if PCRE2_CODE_UNIT_WIDTH == 8
type = (uint32_t)(data[0] << 8) | data[1];
data += 2;
#else
type = data[0];
data++;
#endif  /* CODE_UNIT_WIDTH */

/* Align characters. */
next_char = char_lists_end - (GET(data, 0) << 1);
type &= XCL_TYPE_MASK;

/* Alignment check. */
PCRE2_ASSERT(((uintptr_t)next_char & 0x1) == 0);

if (c >= XCL_CHAR_LIST_HIGH_16_START)
  {
  max_index = type & XCL_ITEM_COUNT_MASK;
  if (max_index == XCL_ITEM_COUNT_MASK)
    {
    max_index = *(const uint16_t*)next_char;
    PCRE2_ASSERT(max_index >= XCL_ITEM_COUNT_MASK);
    next_char += 2;
    }

  next_char += max_index << 1;
  type >>= XCL_TYPE_BIT_LEN;
  }

if (c < XCL_CHAR_LIST_LOW_32_START)
  {
  max_index = type & XCL_ITEM_COUNT_MASK;

  c = (uint16_t)((c << XCL_CHAR_SHIFT) | XCL_CHAR_END);

  if (max_index == XCL_ITEM_COUNT_MASK)
    {
    max_index = *(const uint16_t*)next_char;
    PCRE2_ASSERT(max_index >= XCL_ITEM_COUNT_MASK);
    next_char += 2;
    }

  if (max_index == 0 || c < *(const uint16_t*)next_char)
    return ((type & XCL_BEGIN_WITH_RANGE) != 0) == not_negated;

  min_index = 0;
  value = ((const uint16_t*)next_char)[--max_index];
  if (c >= value)
    return (value == c || (value & XCL_CHAR_END) == 0) == not_negated;

  max_index--;

  /* Binary search of a range. */
  while (TRUE)
    {
    uint32_t mid_index = (min_index + max_index) >> 1;
    value = ((const uint16_t*)next_char)[mid_index];

    if (c < value)
      max_index = mid_index - 1;
    else if (((const uint16_t*)next_char)[mid_index + 1] <= c)
      min_index = mid_index + 1;
    else
      return (value == c || (value & XCL_CHAR_END) == 0) == not_negated;
    }
  }

/* Skip the 16 bit ranges. */
max_index = type & XCL_ITEM_COUNT_MASK;
if (max_index == XCL_ITEM_COUNT_MASK)
  {
  max_index = *(const uint16_t*)next_char;
  PCRE2_ASSERT(max_index >= XCL_ITEM_COUNT_MASK);
  next_char += 2;
  }

next_char += (max_index << 1);
type >>= XCL_TYPE_BIT_LEN;

/* Alignment check. */
PCRE2_ASSERT(((uintptr_t)next_char & 0x3) == 0);

max_index = type & XCL_ITEM_COUNT_MASK;

#if PCRE2_CODE_UNIT_WIDTH == 32
if (c >= XCL_CHAR_LIST_HIGH_32_START)
  {
  if (max_index == XCL_ITEM_COUNT_MASK)
    {
    max_index = *(const uint32_t*)next_char;
    PCRE2_ASSERT(max_index >= XCL_ITEM_COUNT_MASK);
    next_char += 4;
    }

  next_char += max_index << 2;
  type >>= XCL_TYPE_BIT_LEN;
  max_index = type & XCL_ITEM_COUNT_MASK;
  }
#endif

c = (uint32_t)((c << XCL_CHAR_SHIFT) | XCL_CHAR_END);

if (max_index == XCL_ITEM_COUNT_MASK)
  {
  max_index = *(const uint32_t*)next_char;
  next_char += 4;
  }

if (max_index == 0 || c < *(const uint32_t*)next_char)
  return ((type & XCL_BEGIN_WITH_RANGE) != 0) == not_negated;

min_index = 0;
value = ((const uint32_t*)next_char)[--max_index];
if (c >= value)
  return (value == c || (value & XCL_CHAR_END) == 0) == not_negated;

max_index--;

/* Binary search of a range. */
while (TRUE)
  {
  uint32_t mid_index = (min_index + max_index) >> 1;
  value = ((const uint32_t*)next_char)[mid_index];

  if (c < value)
    max_index = mid_index - 1;
  else if (((const uint32_t*)next_char)[mid_index + 1] <= c)
    min_index = mid_index + 1;
  else
    return (value == c || (value & XCL_CHAR_END) == 0) == not_negated;
  }
}



/*************************************************
*       Match character against an ECLASS        *
*************************************************/

/* This function is called to match a character against an extended class
used for describing characters using boolean operations on sets.

Arguments:
  c           the character
  data_start  points to the start of the ECLASS data
  data_end    points one-past-the-last of the ECLASS data
  utf         TRUE if in UTF mode

Returns:      TRUE if character matches, else FALSE
*/

BOOL
PRIV(eclass)(uint32_t c, PCRE2_SPTR data_start, PCRE2_SPTR data_end,
  const uint8_t *char_lists_end, BOOL utf)
{
PCRE2_SPTR ptr = data_start;
PCRE2_UCHAR flags;
uint32_t stack = 0;
int stack_depth = 0;

PCRE2_ASSERT(data_start < data_end);
flags = *ptr++;
PCRE2_ASSERT((flags & ECL_MAP) == 0 ||
             (data_end - ptr) >= 32 / (int)sizeof(PCRE2_UCHAR));

/* Code points < 256 are matched against a bitmap, if one is present.
Otherwise all codepoints are checked later. */

if ((flags & ECL_MAP) != 0)
  {
  if (c < 256)
    return (((const uint8_t *)ptr)[c/8] & (1u << (c&7))) != 0;

  /* Skip the bitmap. */
  ptr += 32 / sizeof(PCRE2_UCHAR);
  }

/* Do a little loop, until we reach the end of the ECLASS. */
while (ptr < data_end)
  {
  switch (*ptr)
    {
    case ECL_AND:
    ++ptr;
    stack = (stack >> 1) & (stack | ~(uint32_t)1u);
    PCRE2_ASSERT(stack_depth >= 2);
    --stack_depth;
    break;

    case ECL_OR:
    ++ptr;
    stack = (stack >> 1) | (stack & (uint32_t)1u);
    PCRE2_ASSERT(stack_depth >= 2);
    --stack_depth;
    break;

    case ECL_XOR:
    ++ptr;
    stack = (stack >> 1) ^ (stack & (uint32_t)1u);
    PCRE2_ASSERT(stack_depth >= 2);
    --stack_depth;
    break;

    case ECL_NOT:
    ++ptr;
    stack ^= (uint32_t)1u;
    PCRE2_ASSERT(stack_depth >= 1);
    break;

    case ECL_XCLASS:
      {
      uint32_t matched = PRIV(xclass)(c, ptr + 1 + LINK_SIZE, char_lists_end, utf);

      ptr += GET(ptr, 1);
      stack = (stack << 1) | matched;
      ++stack_depth;
      break;
      }

    /* This should never occur, but compilers may mutter if there is no
    default. */

    /* LCOV_EXCL_START */
    default:
    PCRE2_DEBUG_UNREACHABLE();
    return FALSE;
    /* LCOV_EXCL_STOP */
    }
  }

PCRE2_ASSERT(stack_depth == 1);
(void)stack_depth;  /* Ignore unused variable, if assertions are disabled. */

/* The final bit left on the stack now holds the match result. */
return (stack & 1u) != 0;
}

/* End of pcre2_xclass.c */
