#ifndef HEADER_CURL_HTTP_DIGEST_H
#define HEADER_CURL_HTTP_DIGEST_H
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

#if !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_CRYPTO_AUTH)

/* this is for digest header input */
CURLcode Curl_input_digest(struct Curl_easy *data,
                           bool proxy, const char *header);

/* this is for creating digest header output */
CURLcode Curl_output_digest(struct Curl_easy *data,
                            bool proxy,
                            const unsigned char *request,
                            const unsigned char *uripath);

void Curl_http_auth_cleanup_digest(struct Curl_easy *data);

#endif /* !CURL_DISABLE_HTTP && !CURL_DISABLE_CRYPTO_AUTH */

#endif /* HEADER_CURL_HTTP_DIGEST_H */
