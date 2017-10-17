/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
         New API code Copyright (c) 2016 University of Cambridge

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

/* This module contains an internal function that is used to match an extended
class. It is used by pcre2_auto_possessify() and by both pcre2_match() and
pcre2_def_match(). */


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


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
PRIV(xclass)(uint32_t c, PCRE2_SPTR data, BOOL utf)
{
PCRE2_UCHAR t;
BOOL negated = (*data & XCL_NOT) != 0;

#if PCRE2_CODE_UNIT_WIDTH == 8
/* In 8 bit mode, this must always be TRUE. Help the compiler to know that. */
utf = TRUE;
#endif

/* Code points < 256 are matched against a bitmap, if one is present. If not,
we still carry on, because there may be ranges that start below 256 in the
additional data. */

if (c < 256)
  {
  if ((*data & XCL_HASPROP) == 0)
    {
    if ((*data & XCL_MAP) == 0) return negated;
    return (((uint8_t *)(data + 1))[c/8] & (1 << (c&7))) != 0;
    }
  if ((*data & XCL_MAP) != 0 &&
    (((uint8_t *)(data + 1))[c/8] & (1 << (c&7))) != 0)
    return !negated; /* char found */
  }

/* First skip the bit map if present. Then match against the list of Unicode
properties or large chars or ranges that end with a large char. We won't ever
encounter XCL_PROP or XCL_NOTPROP when UTF support is not compiled. */

if ((*data++ & XCL_MAP) != 0) data += 32 / sizeof(PCRE2_UCHAR);

while ((t = *data++) != XCL_END)
  {
  uint32_t x, y;
  if (t == XCL_SINGLE)
    {
#ifdef SUPPORT_UNICODE
    if (utf)
      {
      GETCHARINC(x, data); /* macro generates multiple statements */
      }
    else
#endif
    x = *data++;
    if (c == x) return !negated;
    }
  else if (t == XCL_RANGE)
    {
#ifdef SUPPORT_UNICODE
    if (utf)
      {
      GETCHARINC(x, data); /* macro generates multiple statements */
      GETCHARINC(y, data); /* macro generates multiple statements */
      }
    else
#endif
      {
      x = *data++;
      y = *data++;
      }
    if (c >= x && c <= y) return !negated;
    }

#ifdef SUPPORT_UNICODE
  else  /* XCL_PROP & XCL_NOTPROP */
    {
    const ucd_record *prop = GET_UCD(c);
    BOOL isprop = t == XCL_PROP;

    switch(*data)
      {
      case PT_ANY:
      if (isprop) return !negated;
      break;

      case PT_LAMP:
      if ((prop->chartype == ucp_Lu || prop->chartype == ucp_Ll ||
           prop->chartype == ucp_Lt) == isprop) return !negated;
      break;

      case PT_GC:
      if ((data[1] == PRIV(ucp_gentype)[prop->chartype]) == isprop)
        return !negated;
      break;

      case PT_PC:
      if ((data[1] == prop->chartype) == isprop) return !negated;
      break;

      case PT_SC:
      if ((data[1] == prop->script) == isprop) return !negated;
      break;

      case PT_ALNUM:
      if ((PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
           PRIV(ucp_gentype)[prop->chartype] == ucp_N) == isprop)
        return !negated;
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
        if (isprop) return !negated;
        break;

        default:
        if ((PRIV(ucp_gentype)[prop->chartype] == ucp_Z) == isprop)
          return !negated;
        break;
        }
      break;

      case PT_WORD:
      if ((PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
           PRIV(ucp_gentype)[prop->chartype] == ucp_N || c == CHAR_UNDERSCORE)
             == isprop)
        return !negated;
      break;

      case PT_UCNC:
      if (c < 0xa0)
        {
        if ((c == CHAR_DOLLAR_SIGN || c == CHAR_COMMERCIAL_AT ||
             c == CHAR_GRAVE_ACCENT) == isprop)
          return !negated;
        }
      else
        {
        if ((c < 0xd800 || c > 0xdfff) == isprop)
          return !negated;
        }
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
      if ((PRIV(ucp_gentype)[prop->chartype] != ucp_Z &&
            (PRIV(ucp_gentype)[prop->chartype] != ucp_C ||
              (prop->chartype == ucp_Cf &&
                c != 0x061c && c != 0x180e && (c < 0x2066 || c > 0x2069))
         )) == isprop)
        return !negated;
      break;

      /* Printable character: same as graphic, with the addition of Zs, i.e.
      not Zl and not Zp, and U+180E. */

      case PT_PXPRINT:
      if ((prop->chartype != ucp_Zl &&
           prop->chartype != ucp_Zp &&
            (PRIV(ucp_gentype)[prop->chartype] != ucp_C ||
              (prop->chartype == ucp_Cf &&
                c != 0x061c && (c < 0x2066 || c > 0x2069))
         )) == isprop)
        return !negated;
      break;

      /* Punctuation: all Unicode punctuation, plus ASCII characters that
      Unicode treats as symbols rather than punctuation, for Perl
      compatibility (these are $+<=>^`|~). */

      case PT_PXPUNCT:
      if ((PRIV(ucp_gentype)[prop->chartype] == ucp_P ||
            (c < 128 && PRIV(ucp_gentype)[prop->chartype] == ucp_S)) == isprop)
        return !negated;
      break;

      /* This should never occur, but compilers may mutter if there is no
      default. */

      default:
      return FALSE;
      }

    data += 2;
    }
#else
  (void)utf;  /* Avoid compiler warning */
#endif  /* SUPPORT_UNICODE */
  }

return negated;   /* char did not match */
}

/* End of pcre2_xclass.c */
