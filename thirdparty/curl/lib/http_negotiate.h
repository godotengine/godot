#ifndef HEADER_CURL_HTTP_NEGOTIATE_H
#define HEADER_CURL_HTTP_NEGOTIATE_H
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

#if !defined(CURL_DISABLE_HTTP) && defined(USE_SPNEGO)

/* this is for Negotiate header input */
CURLcode Curl_input_negotiate(struct Curl_easy *data, struct connectdata *conn,
                              bool proxy, const char *header);

/* this is for creating Negotiate header output */
CURLcode Curl_output_negotiate(struct Curl_easy *data,
                               struct connectdata *conn, bool proxy);

void Curl_http_auth_cleanup_negotiate(struct connectdata *conn);

#else /* !CURL_DISABLE_HTTP && USE_SPNEGO */
#define Curl_http_auth_cleanup_negotiate(x)
#endif

#endif /* HEADER_CURL_HTTP_NEGOTIATE_H */
