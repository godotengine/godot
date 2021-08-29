/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

#if !defined(CURL_DISABLE_COOKIES) || !defined(CURL_DISABLE_ALTSVC) ||  \
  !defined(CURL_DISABLE_HSTS)

#include "curl_get_line.h"
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

/*
 * get_line() makes sure to only return complete whole lines that fit in 'len'
 * bytes and end with a newline.
 */
char *Curl_get_line(char *buf, int len, FILE *input)
{
  bool partial = FALSE;
  while(1) {
    char *b = fgets(buf, len, input);
    if(b) {
      size_t rlen = strlen(b);
      if(rlen && (b[rlen-1] == '\n')) {
        if(partial) {
          partial = FALSE;
          continue;
        }
        return b;
      }
      /* read a partial, discard the next piece that ends with newline */
      partial = TRUE;
    }
    else
      break;
  }
  return NULL;
}

#endif /* if not disabled */
