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

#include "curl_setup.h"

#ifndef HAVE_STRTOK_R
#include <stddef.h>

#include "strtok.h"

char *
Curl_strtok_r(char *ptr, const char *sep, char **end)
{
  if(!ptr)
    /* we got NULL input so then we get our last position instead */
    ptr = *end;

  /* pass all letters that are including in the separator string */
  while(*ptr && strchr(sep, *ptr))
    ++ptr;

  if(*ptr) {
    /* so this is where the next piece of string starts */
    char *start = ptr;

    /* set the end pointer to the first byte after the start */
    *end = start + 1;

    /* scan through the string to find where it ends, it ends on a
       null byte or a character that exists in the separator string */
    while(**end && !strchr(sep, **end))
      ++*end;

    if(**end) {
      /* the end is not a null byte */
      **end = '\0';  /* null-terminate it! */
      ++*end;        /* advance the last pointer to beyond the null byte */
    }

    return start; /* return the position where the string starts */
  }

  /* we ended up on a null byte, there are no more strings to find! */
  return NULL;
}

#endif /* this was only compiled if strtok_r wasn't present */
