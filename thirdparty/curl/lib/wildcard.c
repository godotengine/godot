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

#ifndef CURL_DISABLE_FTP

#include "wildcard.h"
#include "llist.h"
#include "fileinfo.h"
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

static void fileinfo_dtor(void *user, void *element)
{
  (void)user;
  Curl_fileinfo_cleanup(element);
}

CURLcode Curl_wildcard_init(struct WildcardData *wc)
{
  Curl_llist_init(&wc->filelist, fileinfo_dtor);
  wc->state = CURLWC_INIT;

  return CURLE_OK;
}

void Curl_wildcard_dtor(struct WildcardData *wc)
{
  if(!wc)
    return;

  if(wc->dtor) {
    wc->dtor(wc->protdata);
    wc->dtor = ZERO_NULL;
    wc->protdata = NULL;
  }
  DEBUGASSERT(wc->protdata == NULL);

  Curl_llist_destroy(&wc->filelist, NULL);


  free(wc->path);
  wc->path = NULL;
  free(wc->pattern);
  wc->pattern = NULL;

  wc->customptr = NULL;
  wc->state = CURLWC_INIT;
}

#endif /* if disabled */
