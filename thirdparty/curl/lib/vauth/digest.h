#ifndef HEADER_CURL_DIGEST_H
#define HEADER_CURL_DIGEST_H
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

#include <curl/curl.h>

#if !defined(CURL_DISABLE_CRYPTO_AUTH)

#define DIGEST_MAX_VALUE_LENGTH           256
#define DIGEST_MAX_CONTENT_LENGTH         1024

enum {
  CURLDIGESTALGO_MD5,
  CURLDIGESTALGO_MD5SESS,
  CURLDIGESTALGO_SHA256,
  CURLDIGESTALGO_SHA256SESS,
  CURLDIGESTALGO_SHA512_256,
  CURLDIGESTALGO_SHA512_256SESS
};

/* This is used to extract the realm from a challenge message */
bool Curl_auth_digest_get_pair(const char *str, char *value, char *content,
                               const char **endptr);

#endif

#endif /* HEADER_CURL_DIGEST_H */
