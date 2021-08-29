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

#include <curl/curl.h>

#include "dotdot.h"
#include "curl_memory.h"

/* The last #include file should be: */
#include "memdebug.h"

/*
 * "Remove Dot Segments"
 * https://tools.ietf.org/html/rfc3986#section-5.2.4
 */

/*
 * Curl_dedotdotify()
 * @unittest: 1395
 *
 * This function gets a null-terminated path with dot and dotdot sequences
 * passed in and strips them off according to the rules in RFC 3986 section
 * 5.2.4.
 *
 * The function handles a query part ('?' + stuff) appended but it expects
 * that fragments ('#' + stuff) have already been cut off.
 *
 * RETURNS
 *
 * an allocated dedotdotified output string
 */
char *Curl_dedotdotify(const char *input)
{
  size_t inlen = strlen(input);
  char *clone;
  size_t clen = inlen; /* the length of the cloned input */
  char *out = malloc(inlen + 1);
  char *outptr;
  char *orgclone;
  char *queryp;
  if(!out)
    return NULL; /* out of memory */

  *out = 0; /* null-terminates, for inputs like "./" */

  /* get a cloned copy of the input */
  clone = strdup(input);
  if(!clone) {
    free(out);
    return NULL;
  }
  orgclone = clone;
  outptr = out;

  if(!*clone) {
    /* zero length string, return that */
    free(out);
    return clone;
  }

  /*
   * To handle query-parts properly, we must find it and remove it during the
   * dotdot-operation and then append it again at the end to the output
   * string.
   */
  queryp = strchr(clone, '?');
  if(queryp)
    *queryp = 0;

  do {

    /*  A.  If the input buffer begins with a prefix of "../" or "./", then
        remove that prefix from the input buffer; otherwise, */

    if(!strncmp("./", clone, 2)) {
      clone += 2;
      clen -= 2;
    }
    else if(!strncmp("../", clone, 3)) {
      clone += 3;
      clen -= 3;
    }

    /*  B.  if the input buffer begins with a prefix of "/./" or "/.", where
        "."  is a complete path segment, then replace that prefix with "/" in
        the input buffer; otherwise, */
    else if(!strncmp("/./", clone, 3)) {
      clone += 2;
      clen -= 2;
    }
    else if(!strcmp("/.", clone)) {
      clone[1]='/';
      clone++;
      clen -= 1;
    }

    /*  C.  if the input buffer begins with a prefix of "/../" or "/..", where
        ".." is a complete path segment, then replace that prefix with "/" in
        the input buffer and remove the last segment and its preceding "/" (if
        any) from the output buffer; otherwise, */

    else if(!strncmp("/../", clone, 4)) {
      clone += 3;
      clen -= 3;
      /* remove the last segment from the output buffer */
      while(outptr > out) {
        outptr--;
        if(*outptr == '/')
          break;
      }
      *outptr = 0; /* null-terminate where it stops */
    }
    else if(!strcmp("/..", clone)) {
      clone[2]='/';
      clone += 2;
      clen -= 2;
      /* remove the last segment from the output buffer */
      while(outptr > out) {
        outptr--;
        if(*outptr == '/')
          break;
      }
      *outptr = 0; /* null-terminate where it stops */
    }

    /*  D.  if the input buffer consists only of "." or "..", then remove
        that from the input buffer; otherwise, */

    else if(!strcmp(".", clone) || !strcmp("..", clone)) {
      *clone = 0;
      *out = 0;
    }

    else {
      /*  E.  move the first path segment in the input buffer to the end of
          the output buffer, including the initial "/" character (if any) and
          any subsequent characters up to, but not including, the next "/"
          character or the end of the input buffer. */

      do {
        *outptr++ = *clone++;
        clen--;
      } while(*clone && (*clone != '/'));
      *outptr = 0;
    }

  } while(*clone);

  if(queryp) {
    size_t qlen;
    /* There was a query part, append that to the output. The 'clone' string
       may now have been altered so we copy from the original input string
       from the correct index. */
    size_t oindex = queryp - orgclone;
    qlen = strlen(&input[oindex]);
    memcpy(outptr, &input[oindex], qlen + 1); /* include the end zero byte */
  }

  free(orgclone);
  return out;
}
