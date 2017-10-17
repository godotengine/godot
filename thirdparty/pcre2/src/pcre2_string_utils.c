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

/* This module contains internal functions for comparing and finding the length
of strings. These are used instead of strcmp() etc because the standard
functions work only on 8-bit data. */


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "pcre2_internal.h"


/*************************************************
*    Compare two zero-terminated PCRE2 strings   *
*************************************************/

/*
Arguments:
  str1        first string
  str2        second string

Returns:      0, 1, or -1
*/

int
PRIV(strcmp)(PCRE2_SPTR str1, PCRE2_SPTR str2)
{
PCRE2_UCHAR c1, c2;
while (*str1 != '\0' || *str2 != '\0')
  {
  c1 = *str1++;
  c2 = *str2++;
  if (c1 != c2) return ((c1 > c2) << 1) - 1;
  }
return 0;
}


/*************************************************
*  Compare zero-terminated PCRE2 & 8-bit strings *
*************************************************/

/* As the 8-bit string is almost always a literal, its type is specified as
const char *.

Arguments:
  str1        first string
  str2        second string

Returns:      0, 1, or -1
*/

int
PRIV(strcmp_c8)(PCRE2_SPTR str1, const char *str2)
{
PCRE2_UCHAR c1, c2;
while (*str1 != '\0' || *str2 != '\0')
  {
  c1 = *str1++;
  c2 = *str2++;
  if (c1 != c2) return ((c1 > c2) << 1) - 1;
  }
return 0;
}


/*************************************************
*    Compare two PCRE2 strings, given a length   *
*************************************************/

/*
Arguments:
  str1        first string
  str2        second string
  len         the length

Returns:      0, 1, or -1
*/

int
PRIV(strncmp)(PCRE2_SPTR str1, PCRE2_SPTR str2, size_t len)
{
PCRE2_UCHAR c1, c2;
for (; len > 0; len--)
  {
  c1 = *str1++;
  c2 = *str2++;
  if (c1 != c2) return ((c1 > c2) << 1) - 1;
  }
return 0;
}


/*************************************************
* Compare PCRE2 string to 8-bit string by length *
*************************************************/

/* As the 8-bit string is almost always a literal, its type is specified as
const char *.

Arguments:
  str1        first string
  str2        second string
  len         the length

Returns:      0, 1, or -1
*/

int
PRIV(strncmp_c8)(PCRE2_SPTR str1, const char *str2, size_t len)
{
PCRE2_UCHAR c1, c2;
for (; len > 0; len--)
  {
  c1 = *str1++;
  c2 = *str2++;
  if (c1 != c2) return ((c1 > c2) << 1) - 1;
  }
return 0;
}


/*************************************************
*        Find the length of a PCRE2 string       *
*************************************************/

/*
Argument:    the string
Returns:     the length
*/

PCRE2_SIZE
PRIV(strlen)(PCRE2_SPTR str)
{
PCRE2_SIZE c = 0;
while (*str++ != 0) c++;
return c;
}


/*************************************************
* Copy 8-bit 0-terminated string to PCRE2 string *
*************************************************/

/* Arguments:
  str1     buffer to receive the string
  str2     8-bit string to be copied

Returns:   the number of code units used (excluding trailing zero)
*/

PCRE2_SIZE
PRIV(strcpy_c8)(PCRE2_UCHAR *str1, const char *str2)
{
PCRE2_UCHAR *t = str1;
while (*str2 != 0) *t++ = *str2++;
*t = 0;
return t - str1;
}

/* End of pcre2_string_utils.c */
