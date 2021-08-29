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
#include "tool_setup.h"

#include "strcase.h"

#define ENABLE_CURLX_PRINTF
/* use our own printf() functions */
#include "curlx.h"

#include "tool_cfgable.h"
#include "tool_convert.h"
#include "tool_doswin.h"
#include "tool_operhlp.h"

#include "memdebug.h" /* keep this as LAST include */

void clean_getout(struct OperationConfig *config)
{
  if(config) {
    struct getout *next;
    struct getout *node = config->url_list;

    while(node) {
      next = node->next;
      Curl_safefree(node->url);
      Curl_safefree(node->outfile);
      Curl_safefree(node->infile);
      Curl_safefree(node);
      node = next;
    }
    config->url_list = NULL;
  }
}

bool output_expected(const char *url, const char *uploadfile)
{
  if(!uploadfile)
    return TRUE;  /* download */
  if(checkprefix("http://", url) || checkprefix("https://", url))
    return TRUE;   /* HTTP(S) upload */

  return FALSE; /* non-HTTP upload, probably no output should be expected */
}

bool stdin_upload(const char *uploadfile)
{
  return (!strcmp(uploadfile, "-") ||
          !strcmp(uploadfile, ".")) ? TRUE : FALSE;
}

/*
 * Adds the file name to the URL if it doesn't already have one.
 * url will be freed before return if the returned pointer is different
 */
char *add_file_name_to_url(char *url, const char *filename)
{
  /* If no file name part is given in the URL, we add this file name */
  char *ptr = strstr(url, "://");
  CURL *curl = curl_easy_init(); /* for url escaping */
  if(!curl)
    return NULL; /* error! */
  if(ptr)
    ptr += 3;
  else
    ptr = url;
  ptr = strrchr(ptr, '/');
  if(!ptr || !*++ptr) {
    /* The URL has no file name part, add the local file name. In order
       to be able to do so, we have to create a new URL in another
       buffer.*/

    /* We only want the part of the local path that is on the right
       side of the rightmost slash and backslash. */
    const char *filep = strrchr(filename, '/');
    char *file2 = strrchr(filep?filep:filename, '\\');
    char *encfile;

    if(file2)
      filep = file2 + 1;
    else if(filep)
      filep++;
    else
      filep = filename;

    /* URL encode the file name */
    encfile = curl_easy_escape(curl, filep, 0 /* use strlen */);
    if(encfile) {
      char *urlbuffer;
      if(ptr)
        /* there is a trailing slash on the URL */
        urlbuffer = aprintf("%s%s", url, encfile);
      else
        /* there is no trailing slash on the URL */
        urlbuffer = aprintf("%s/%s", url, encfile);

      curl_free(encfile);

      if(!urlbuffer) {
        url = NULL;
        goto end;
      }

      Curl_safefree(url);
      url = urlbuffer; /* use our new URL instead! */
    }
  }
  end:
  curl_easy_cleanup(curl);
  return url;
}

/* Extracts the name portion of the URL.
 * Returns a pointer to a heap-allocated string or NULL if
 * no name part, at location indicated by first argument.
 */
CURLcode get_url_file_name(char **filename, const char *url)
{
  const char *pc, *pc2;

  *filename = NULL;

  /* Find and get the remote file name */
  pc = strstr(url, "://");
  if(pc)
    pc += 3;
  else
    pc = url;

  pc2 = strrchr(pc, '\\');
  pc = strrchr(pc, '/');
  if(pc2 && (!pc || pc < pc2))
    pc = pc2;

  if(pc)
    /* duplicate the string beyond the slash */
    pc++;
  else
    /* no slash => empty string */
    pc = "";

  *filename = strdup(pc);
  if(!*filename)
    return CURLE_OUT_OF_MEMORY;

#if defined(MSDOS) || defined(WIN32)
  {
    char *sanitized;
    SANITIZEcode sc = sanitize_file_name(&sanitized, *filename, 0);
    Curl_safefree(*filename);
    if(sc)
      return CURLE_URL_MALFORMAT;
    *filename = sanitized;
  }
#endif /* MSDOS || WIN32 */

  /* in case we built debug enabled, we allow an environment variable
   * named CURL_TESTDIR to prefix the given file name to put it into a
   * specific directory
   */
#ifdef DEBUGBUILD
  {
    char *tdir = curlx_getenv("CURL_TESTDIR");
    if(tdir) {
      char buffer[512]; /* suitably large */
      msnprintf(buffer, sizeof(buffer), "%s/%s", tdir, *filename);
      Curl_safefree(*filename);
      *filename = strdup(buffer); /* clone the buffer */
      curl_free(tdir);
      if(!*filename)
        return CURLE_OUT_OF_MEMORY;
    }
  }
#endif

  return CURLE_OK;
}
