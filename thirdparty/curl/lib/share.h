#ifndef HEADER_CURL_SHARE_H
#define HEADER_CURL_SHARE_H
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
#include <curl/curl.h>
#include "cookie.h"
#include "psl.h"
#include "urldata.h"
#include "conncache.h"

/* SalfordC says "A structure member may not be volatile". Hence:
 */
#ifdef __SALFORDC__
#define CURL_VOLATILE
#else
#define CURL_VOLATILE volatile
#endif

#define CURL_GOOD_SHARE 0x7e117a1e
#define GOOD_SHARE_HANDLE(x) ((x) && (x)->magic == CURL_GOOD_SHARE)

/* this struct is libcurl-private, don't export details */
struct Curl_share {
  unsigned int magic; /* CURL_GOOD_SHARE */
  unsigned int specifier;
  CURL_VOLATILE unsigned int dirty;

  curl_lock_function lockfunc;
  curl_unlock_function unlockfunc;
  void *clientdata;
  struct conncache conn_cache;
  struct Curl_hash hostcache;
#if !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_COOKIES)
  struct CookieInfo *cookies;
#endif
#ifdef USE_LIBPSL
  struct PslCache psl;
#endif

  struct Curl_ssl_session *sslsession;
  size_t max_ssl_sessions;
  long sessionage;
};

CURLSHcode Curl_share_lock(struct Curl_easy *, curl_lock_data,
                           curl_lock_access);
CURLSHcode Curl_share_unlock(struct Curl_easy *, curl_lock_data);

#endif /* HEADER_CURL_SHARE_H */
