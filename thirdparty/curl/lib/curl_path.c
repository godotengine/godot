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

#if defined(USE_SSH)

#include <curl/curl.h>
#include "curl_memory.h"
#include "curl_path.h"
#include "escape.h"
#include "memdebug.h"

/* figure out the path to work with in this particular request */
CURLcode Curl_getworkingpath(struct Curl_easy *data,
                             char *homedir,  /* when SFTP is used */
                             char **path) /* returns the  allocated
                                             real path to work with */
{
  char *real_path = NULL;
  char *working_path;
  size_t working_path_len;
  CURLcode result =
    Curl_urldecode(data, data->state.up.path, 0, &working_path,
                   &working_path_len, REJECT_ZERO);
  if(result)
    return result;

  /* Check for /~/, indicating relative to the user's home directory */
  if(data->conn->handler->protocol & CURLPROTO_SCP) {
    real_path = malloc(working_path_len + 1);
    if(!real_path) {
      free(working_path);
      return CURLE_OUT_OF_MEMORY;
    }
    if((working_path_len > 3) && (!memcmp(working_path, "/~/", 3)))
      /* It is referenced to the home directory, so strip the leading '/~/' */
      memcpy(real_path, working_path + 3, working_path_len - 2);
    else
      memcpy(real_path, working_path, 1 + working_path_len);
  }
  else if(data->conn->handler->protocol & CURLPROTO_SFTP) {
    if((working_path_len > 1) && (working_path[1] == '~')) {
      size_t homelen = strlen(homedir);
      real_path = malloc(homelen + working_path_len + 1);
      if(!real_path) {
        free(working_path);
        return CURLE_OUT_OF_MEMORY;
      }
      /* It is referenced to the home directory, so strip the
         leading '/' */
      memcpy(real_path, homedir, homelen);
      real_path[homelen] = '/';
      real_path[homelen + 1] = '\0';
      if(working_path_len > 3) {
        memcpy(real_path + homelen + 1, working_path + 3,
               1 + working_path_len -3);
      }
    }
    else {
      real_path = malloc(working_path_len + 1);
      if(!real_path) {
        free(working_path);
        return CURLE_OUT_OF_MEMORY;
      }
      memcpy(real_path, working_path, 1 + working_path_len);
    }
  }

  free(working_path);

  /* store the pointer for the caller to receive */
  *path = real_path;

  return CURLE_OK;
}

/* The get_pathname() function is being borrowed from OpenSSH sftp.c
   version 4.6p1. */
/*
 * Copyright (c) 2001-2004 Damien Miller <djm@openbsd.org>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */
CURLcode Curl_get_pathname(const char **cpp, char **path, char *homedir)
{
  const char *cp = *cpp, *end;
  char quot;
  unsigned int i, j;
  size_t fullPathLength, pathLength;
  bool relativePath = false;
  static const char WHITESPACE[] = " \t\r\n";

  if(!*cp) {
    *cpp = NULL;
    *path = NULL;
    return CURLE_QUOTE_ERROR;
  }
  /* Ignore leading whitespace */
  cp += strspn(cp, WHITESPACE);
  /* Allocate enough space for home directory and filename + separator */
  fullPathLength = strlen(cp) + strlen(homedir) + 2;
  *path = malloc(fullPathLength);
  if(!*path)
    return CURLE_OUT_OF_MEMORY;

  /* Check for quoted filenames */
  if(*cp == '\"' || *cp == '\'') {
    quot = *cp++;

    /* Search for terminating quote, unescape some chars */
    for(i = j = 0; i <= strlen(cp); i++) {
      if(cp[i] == quot) {  /* Found quote */
        i++;
        (*path)[j] = '\0';
        break;
      }
      if(cp[i] == '\0') {  /* End of string */
        /*error("Unterminated quote");*/
        goto fail;
      }
      if(cp[i] == '\\') {  /* Escaped characters */
        i++;
        if(cp[i] != '\'' && cp[i] != '\"' &&
            cp[i] != '\\') {
          /*error("Bad escaped character '\\%c'",
              cp[i]);*/
          goto fail;
        }
      }
      (*path)[j++] = cp[i];
    }

    if(j == 0) {
      /*error("Empty quotes");*/
      goto fail;
    }
    *cpp = cp + i + strspn(cp + i, WHITESPACE);
  }
  else {
    /* Read to end of filename - either to whitespace or terminator */
    end = strpbrk(cp, WHITESPACE);
    if(!end)
      end = strchr(cp, '\0');
    /* return pointer to second parameter if it exists */
    *cpp = end + strspn(end, WHITESPACE);
    pathLength = 0;
    relativePath = (cp[0] == '/' && cp[1] == '~' && cp[2] == '/');
    /* Handling for relative path - prepend home directory */
    if(relativePath) {
      strcpy(*path, homedir);
      pathLength = strlen(homedir);
      (*path)[pathLength++] = '/';
      (*path)[pathLength] = '\0';
      cp += 3;
    }
    /* Copy path name up until first "whitespace" */
    memcpy(&(*path)[pathLength], cp, (int)(end - cp));
    pathLength += (int)(end - cp);
    (*path)[pathLength] = '\0';
  }
  return CURLE_OK;

  fail:
  Curl_safefree(*path);
  return CURLE_QUOTE_ERROR;
}

#endif /* if SSH is used */
