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


/* This module contains internal functions for testing newlines when more than
one kind of newline is to be recognized. When a newline is found, its length is
returned. In principle, we could implement several newline "types", each
referring to a different set of newline characters. At present, PCRE2 supports
only NLTYPE_FIXED, which gets handled without these functions, NLTYPE_ANYCRLF,
and NLTYPE_ANY. The full list of Unicode newline characters is taken from
http://unicode.org/unicode/reports/tr18/. */


#include "pcre2_internal.h"



/*************************************************
*      Check for newline at given position       *
*************************************************/

/* This function is called only via the IS_NEWLINE macro, which does so only
when the newline type is NLTYPE_ANY or NLTYPE_ANYCRLF. The case of a fixed
newline (NLTYPE_FIXED) is handled inline. It is guaranteed that the code unit
pointed to by ptr is less than the end of the string.

Arguments:
  ptr          pointer to possible newline
  type         the newline type
  endptr       pointer to the end of the string
  lenptr       where to return the length
  utf          TRUE if in utf mode

Returns:       TRUE or FALSE
*/

BOOL
PRIV(is_newline)(PCRE2_SPTR ptr, uint32_t type, PCRE2_SPTR endptr,
  uint32_t *lenptr, BOOL utf)
{
uint32_t c;

#ifdef SUPPORT_UNICODE
if (utf) { GETCHAR(c, ptr); } else c = *ptr;
#else
(void)utf;
c = *ptr;
#endif  /* SUPPORT_UNICODE */

if (type == NLTYPE_ANYCRLF) switch(c)
  {
  case CHAR_LF:
  *lenptr = 1;
  return TRUE;

  case CHAR_CR:
  *lenptr = (ptr < endptr - 1 && ptr[1] == CHAR_LF)? 2 : 1;
  return TRUE;

  default:
  return FALSE;
  }

/* NLTYPE_ANY */

else switch(c)
  {
#ifdef EBCDIC
  case CHAR_NEL:
#endif
  case CHAR_LF:
  case CHAR_VT:
  case CHAR_FF:
  *lenptr = 1;
  return TRUE;

  case CHAR_CR:
  *lenptr = (ptr < endptr - 1 && ptr[1] == CHAR_LF)? 2 : 1;
  return TRUE;

#ifndef EBCDIC
#if PCRE2_CODE_UNIT_WIDTH == 8
  case CHAR_NEL:
  *lenptr = utf? 2 : 1;
  return TRUE;

  case 0x2028:   /* LS */
  case 0x2029:   /* PS */
  *lenptr = 3;
  return TRUE;

#else  /* 16-bit or 32-bit code units */
  case CHAR_NEL:
  case 0x2028:   /* LS */
  case 0x2029:   /* PS */
  *lenptr = 1;
  return TRUE;
#endif
#endif /* Not EBCDIC */

  default:
  return FALSE;
  }
}



/*************************************************
*     Check for newline at previous position     *
*************************************************/

/* This function is called only via the WAS_NEWLINE macro, which does so only
when the newline type is NLTYPE_ANY or NLTYPE_ANYCRLF. The case of a fixed
newline (NLTYPE_FIXED) is handled inline. It is guaranteed that the initial
value of ptr is greater than the start of the string that is being processed.

Arguments:
  ptr          pointer to possible newline
  type         the newline type
  startptr     pointer to the start of the string
  lenptr       where to return the length
  utf          TRUE if in utf mode

Returns:       TRUE or FALSE
*/

BOOL
PRIV(was_newline)(PCRE2_SPTR ptr, uint32_t type, PCRE2_SPTR startptr,
  uint32_t *lenptr, BOOL utf)
{
uint32_t c;
ptr--;

#ifdef SUPPORT_UNICODE
if (utf)
  {
  BACKCHAR(ptr);
  GETCHAR(c, ptr);
  }
else c = *ptr;
#else
(void)utf;
c = *ptr;
#endif  /* SUPPORT_UNICODE */

if (type == NLTYPE_ANYCRLF) switch(c)
  {
  case CHAR_LF:
  *lenptr = (ptr > startptr && ptr[-1] == CHAR_CR)? 2 : 1;
  return TRUE;

  case CHAR_CR:
  *lenptr = 1;
  return TRUE;

  default:
  return FALSE;
  }

/* NLTYPE_ANY */

else switch(c)
  {
  case CHAR_LF:
  *lenptr = (ptr > startptr && ptr[-1] == CHAR_CR)? 2 : 1;
  return TRUE;

#ifdef EBCDIC
  case CHAR_NEL:
#endif
  case CHAR_VT:
  case CHAR_FF:
  case CHAR_CR:
  *lenptr = 1;
  return TRUE;

#ifndef EBCDIC
#if PCRE2_CODE_UNIT_WIDTH == 8
  case CHAR_NEL:
  *lenptr = utf? 2 : 1;
  return TRUE;

  case 0x2028:   /* LS */
  case 0x2029:   /* PS */
  *lenptr = 3;
  return TRUE;

#else /* 16-bit or 32-bit code units */
  case CHAR_NEL:
  case 0x2028:   /* LS */
  case 0x2029:   /* PS */
  *lenptr = 1;
  return TRUE;
#endif
#endif /* Not EBCDIC */

  default:
  return FALSE;
  }
}

/* End of pcre2_newline.c */
