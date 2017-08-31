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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

/* Save the configured link size, which is in bytes. In 16-bit and 32-bit modes
its value gets changed by pcre2_internal.h to be in code units. */

static int configured_link_size = LINK_SIZE;

#include "pcre2_internal.h"

/* These macros are the standard way of turning unquoted text into C strings.
They allow macros like PCRE2_MAJOR to be defined without quotes, which is
convenient for user programs that want to test their values. */

#define STRING(a)  # a
#define XSTRING(s) STRING(s)


/*************************************************
* Return info about what features are configured *
*************************************************/

/* If where is NULL, the length of memory required is returned.

Arguments:
  what             what information is required
  where            where to put the information

Returns:           0 if a numerical value is returned
                   >= 0 if a string value
                   PCRE2_ERROR_BADOPTION if "where" not recognized
                     or JIT target requested when JIT not enabled
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_config(uint32_t what, void *where)
{
if (where == NULL)  /* Requests a length */
  {
  switch(what)
    {
    default:
    return PCRE2_ERROR_BADOPTION;

    case PCRE2_CONFIG_BSR:
    case PCRE2_CONFIG_JIT:
    case PCRE2_CONFIG_LINKSIZE:
    case PCRE2_CONFIG_MATCHLIMIT:
    case PCRE2_CONFIG_NEWLINE:
    case PCRE2_CONFIG_PARENSLIMIT:
    case PCRE2_CONFIG_RECURSIONLIMIT:
    case PCRE2_CONFIG_STACKRECURSE:
    case PCRE2_CONFIG_UNICODE:
    return sizeof(uint32_t);

    /* These are handled below */

    case PCRE2_CONFIG_JITTARGET:
    case PCRE2_CONFIG_UNICODE_VERSION:
    case PCRE2_CONFIG_VERSION:
    break;
    }
  }

switch (what)
  {
  default:
  return PCRE2_ERROR_BADOPTION;

  case PCRE2_CONFIG_BSR:
#ifdef BSR_ANYCRLF
  *((uint32_t *)where) = PCRE2_BSR_ANYCRLF;
#else
  *((uint32_t *)where) = PCRE2_BSR_UNICODE;
#endif
  break;

  case PCRE2_CONFIG_JIT:
#ifdef SUPPORT_JIT
  *((uint32_t *)where) = 1;
#else
  *((uint32_t *)where) = 0;
#endif
  break;

  case PCRE2_CONFIG_JITTARGET:
#ifdef SUPPORT_JIT
    {
    const char *v = PRIV(jit_get_target)();
    return (int)(1 + ((where == NULL)?
      strlen(v) : PRIV(strcpy_c8)((PCRE2_UCHAR *)where, v)));
    }
#else
  return PCRE2_ERROR_BADOPTION;
#endif

  case PCRE2_CONFIG_LINKSIZE:
  *((uint32_t *)where) = (uint32_t)configured_link_size;
  break;

  case PCRE2_CONFIG_MATCHLIMIT:
  *((uint32_t *)where) = MATCH_LIMIT;
  break;

  case PCRE2_CONFIG_NEWLINE:
  *((uint32_t *)where) = NEWLINE_DEFAULT;
  break;

  case PCRE2_CONFIG_PARENSLIMIT:
  *((uint32_t *)where) = PARENS_NEST_LIMIT;
  break;

  case PCRE2_CONFIG_RECURSIONLIMIT:
  *((uint32_t *)where) = MATCH_LIMIT_RECURSION;
  break;

  case PCRE2_CONFIG_STACKRECURSE:
#ifdef HEAP_MATCH_RECURSE
  *((uint32_t *)where) = 0;
#else
  *((uint32_t *)where) = 1;
#endif
  break;

  case PCRE2_CONFIG_UNICODE_VERSION:
    {
#if defined SUPPORT_UNICODE
    const char *v = PRIV(unicode_version);
#else
    const char *v = "Unicode not supported";
#endif
    return (int)(1 + ((where == NULL)?
      strlen(v) : PRIV(strcpy_c8)((PCRE2_UCHAR *)where, v)));
   }
  break;

  case PCRE2_CONFIG_UNICODE:
#if defined SUPPORT_UNICODE
  *((uint32_t *)where) = 1;
#else
  *((uint32_t *)where) = 0;
#endif
  break;

  /* The hackery in setting "v" below is to cope with the case when
  PCRE2_PRERELEASE is set to an empty string (which it is for real releases).
  If the second alternative is used in this case, it does not leave a space
  before the date. On the other hand, if all four macros are put into a single
  XSTRING when PCRE2_PRERELEASE is not empty, an unwanted space is inserted.
  There are problems using an "obvious" approach like this:

     XSTRING(PCRE2_MAJOR) "." XSTRING(PCRE_MINOR)
     XSTRING(PCRE2_PRERELEASE) " " XSTRING(PCRE_DATE)

  because, when PCRE2_PRERELEASE is empty, this leads to an attempted expansion
  of STRING(). The C standard states: "If (before argument substitution) any
  argument consists of no preprocessing tokens, the behavior is undefined." It
  turns out the gcc treats this case as a single empty string - which is what
  we really want - but Visual C grumbles about the lack of an argument for the
  macro. Unfortunately, both are within their rights. As there seems to be no
  way to test for a macro's value being empty at compile time, we have to
  resort to a runtime test. */

  case PCRE2_CONFIG_VERSION:
    {
    const char *v = (XSTRING(Z PCRE2_PRERELEASE)[1] == 0)?
      XSTRING(PCRE2_MAJOR.PCRE2_MINOR PCRE2_DATE) :
      XSTRING(PCRE2_MAJOR.PCRE2_MINOR) XSTRING(PCRE2_PRERELEASE PCRE2_DATE);
    return (int)(1 + ((where == NULL)?
      strlen(v) : PRIV(strcpy_c8)((PCRE2_UCHAR *)where, v)));
    }
  }

return 0;
}

/* End of pcre2_config.c */
