/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

#include <errno.h>
#include "curl_setup.h"

#include "strtoofft.h"

/*
 * NOTE:
 *
 * In the ISO C standard (IEEE Std 1003.1), there is a strtoimax() function we
 * could use in case strtoll() doesn't exist...  See
 * https://www.opengroup.org/onlinepubs/009695399/functions/strtoimax.html
 */

#if (SIZEOF_CURL_OFF_T > SIZEOF_LONG)
#  ifdef HAVE_STRTOLL
#    define strtooff strtoll
#  else
#    if defined(_MSC_VER) && (_MSC_VER >= 1300) && (_INTEGRAL_MAX_BITS >= 64)
#      if defined(_SAL_VERSION)
         _Check_return_ _CRTIMP __int64 __cdecl _strtoi64(
             _In_z_ const char *_String,
             _Out_opt_ _Deref_post_z_ char **_EndPtr, _In_ int _Radix);
#      else
         _CRTIMP __int64 __cdecl _strtoi64(const char *_String,
                                           char **_EndPtr, int _Radix);
#      endif
#      define strtooff _strtoi64
#    else
#      define PRIVATE_STRTOOFF 1
#    endif
#  endif
#else
#  define strtooff strtol
#endif

#ifdef PRIVATE_STRTOOFF

/* Range tests can be used for alphanum decoding if characters are consecutive,
   like in ASCII. Else an array is scanned. Determine this condition now. */

#if('9' - '0') != 9 || ('Z' - 'A') != 25 || ('z' - 'a') != 25

#define NO_RANGE_TEST

static const char valchars[] =
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
#endif

static int get_char(char c, int base);

/**
 * Custom version of the strtooff function.  This extracts a curl_off_t
 * value from the given input string and returns it.
 */
static curl_off_t strtooff(const char *nptr, char **endptr, int base)
{
  char *end;
  int is_negative = 0;
  int overflow;
  int i;
  curl_off_t value = 0;
  curl_off_t newval;

  /* Skip leading whitespace. */
  end = (char *)nptr;
  while(ISSPACE(end[0])) {
    end++;
  }

  /* Handle the sign, if any. */
  if(end[0] == '-') {
    is_negative = 1;
    end++;
  }
  else if(end[0] == '+') {
    end++;
  }
  else if(end[0] == '\0') {
    /* We had nothing but perhaps some whitespace -- there was no number. */
    if(endptr) {
      *endptr = end;
    }
    return 0;
  }

  /* Handle special beginnings, if present and allowed. */
  if(end[0] == '0' && end[1] == 'x') {
    if(base == 16 || base == 0) {
      end += 2;
      base = 16;
    }
  }
  else if(end[0] == '0') {
    if(base == 8 || base == 0) {
      end++;
      base = 8;
    }
  }

  /* Matching strtol, if the base is 0 and it doesn't look like
   * the number is octal or hex, we assume it's base 10.
   */
  if(base == 0) {
    base = 10;
  }

  /* Loop handling digits. */
  value = 0;
  overflow = 0;
  for(i = get_char(end[0], base);
      i != -1;
      end++, i = get_char(end[0], base)) {
    newval = base * value + i;
    if(newval < value) {
      /* We've overflowed. */
      overflow = 1;
      break;
    }
    else
      value = newval;
  }

  if(!overflow) {
    if(is_negative) {
      /* Fix the sign. */
      value *= -1;
    }
  }
  else {
    if(is_negative)
      value = CURL_OFF_T_MIN;
    else
      value = CURL_OFF_T_MAX;

    errno = ERANGE;
  }

  if(endptr)
    *endptr = end;

  return value;
}

/**
 * Returns the value of c in the given base, or -1 if c cannot
 * be interpreted properly in that base (i.e., is out of range,
 * is a null, etc.).
 *
 * @param c     the character to interpret according to base
 * @param base  the base in which to interpret c
 *
 * @return  the value of c in base, or -1 if c isn't in range
 */
static int get_char(char c, int base)
{
#ifndef NO_RANGE_TEST
  int value = -1;
  if(c <= '9' && c >= '0') {
    value = c - '0';
  }
  else if(c <= 'Z' && c >= 'A') {
    value = c - 'A' + 10;
  }
  else if(c <= 'z' && c >= 'a') {
    value = c - 'a' + 10;
  }
#else
  const char *cp;
  int value;

  cp = memchr(valchars, c, 10 + 26 + 26);

  if(!cp)
    return -1;

  value = cp - valchars;

  if(value >= 10 + 26)
    value -= 26;                /* Lowercase. */
#endif

  if(value >= base) {
    value = -1;
  }

  return value;
}
#endif  /* Only present if we need strtoll, but don't have it. */

/*
 * Parse a *positive* up to 64 bit number written in ascii.
 */
CURLofft curlx_strtoofft(const char *str, char **endp, int base,
                         curl_off_t *num)
{
  char *end;
  curl_off_t number;
  errno = 0;
  *num = 0; /* clear by default */

  while(*str && ISSPACE(*str))
    str++;
  if('-' == *str) {
    if(endp)
      *endp = (char *)str; /* didn't actually move */
    return CURL_OFFT_INVAL; /* nothing parsed */
  }
  number = strtooff(str, &end, base);
  if(endp)
    *endp = end;
  if(errno == ERANGE)
    /* overflow/underflow */
    return CURL_OFFT_FLOW;
  else if(str == end)
    /* nothing parsed */
    return CURL_OFFT_INVAL;

  *num = number;
  return CURL_OFFT_OK;
}
