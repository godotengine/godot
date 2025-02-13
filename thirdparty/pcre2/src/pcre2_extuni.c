/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2021 University of Cambridge

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

/* This module contains an internal function that is used to match a Unicode
extended grapheme sequence. It is used by both pcre2_match() and
pcre2_def_match(). However, it is called only when Unicode support is being
compiled. Nevertheless, we provide a dummy function when there is no Unicode
support, because some compilers do not like functionless source files. */


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include "pcre2_internal.h"


/* Dummy function */

#ifndef SUPPORT_UNICODE
PCRE2_SPTR
PRIV(extuni)(uint32_t c, PCRE2_SPTR eptr, PCRE2_SPTR start_subject,
  PCRE2_SPTR end_subject, BOOL utf, int *xcount)
{
(void)c;
(void)eptr;
(void)start_subject;
(void)end_subject;
(void)utf;
(void)xcount;
return NULL;
}
#else


/*************************************************
*      Match an extended grapheme sequence       *
*************************************************/

/*
Arguments:
  c              the first character
  eptr           pointer to next character
  start_subject  pointer to start of subject
  end_subject    pointer to end of subject
  utf            TRUE if in UTF mode
  xcount         pointer to count of additional characters,
                   or NULL if count not needed

Returns:         pointer after the end of the sequence
*/

PCRE2_SPTR
PRIV(extuni)(uint32_t c, PCRE2_SPTR eptr, PCRE2_SPTR start_subject,
  PCRE2_SPTR end_subject, BOOL utf, int *xcount)
{
int lgb = UCD_GRAPHBREAK(c);

while (eptr < end_subject)
  {
  int rgb;
  int len = 1;
  if (!utf) c = *eptr; else { GETCHARLEN(c, eptr, len); }
  rgb = UCD_GRAPHBREAK(c);
  if ((PRIV(ucp_gbtable)[lgb] & (1u << rgb)) == 0) break;

  /* Not breaking between Regional Indicators is allowed only if there
  are an even number of preceding RIs. */

  if (lgb == ucp_gbRegional_Indicator && rgb == ucp_gbRegional_Indicator)
    {
    int ricount = 0;
    PCRE2_SPTR bptr = eptr - 1;
    if (utf) BACKCHAR(bptr);

    /* bptr is pointing to the left-hand character */

    while (bptr > start_subject)
      {
      bptr--;
      if (utf)
        {
        BACKCHAR(bptr);
        GETCHAR(c, bptr);
        }
      else
      c = *bptr;
      if (UCD_GRAPHBREAK(c) != ucp_gbRegional_Indicator) break;
      ricount++;
      }
    if ((ricount & 1) != 0) break;  /* Grapheme break required */
    }

  /* If Extend or ZWJ follows Extended_Pictographic, do not update lgb; this
  allows any number of them before a following Extended_Pictographic. */

  if ((rgb != ucp_gbExtend && rgb != ucp_gbZWJ) ||
       lgb != ucp_gbExtended_Pictographic)
    lgb = rgb;

  eptr += len;
  if (xcount != NULL) *xcount += 1;
  }

return eptr;
}

#endif  /* SUPPORT_UNICODE */

/* End of pcre2_extuni.c */
