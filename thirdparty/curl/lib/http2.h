#ifndef HEADER_CURL_HTTP2_H
#define HEADER_CURL_HTTP2_H
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

#ifdef USE_NGHTTP2
#include "http.h"

/* value for MAX_CONCURRENT_STREAMS we use until we get an updated setting
   from the peer */
#define DEFAULT_MAX_CONCURRENT_STREAMS 100

/*
 * Store nghttp2 version info in this buffer.
 */
void Curl_http2_ver(char *p, size_t len);

const char *Curl_http2_strerror(uint32_t err);

CURLcode Curl_http2_init(struct connectdata *conn);
void Curl_http2_init_state(struct UrlState *state);
void Curl_http2_init_userset(struct UserDefined *set);
CURLcode Curl_http2_request_upgrade(struct dynbuf *req,
                                    struct Curl_easy *data);
CURLcode Curl_http2_setup(struct Curl_easy *data, struct connectdata *conn);
CURLcode Curl_http2_switched(struct Curl_easy *data,
                             const char *ptr, size_t nread);
/* called from http_setup_conn */
void Curl_http2_setup_conn(struct connectdata *conn);
void Curl_http2_setup_req(struct Curl_easy *data);
void Curl_http2_done(struct Curl_easy *data, bool premature);
CURLcode Curl_http2_done_sending(struct Curl_easy *data,
                                 struct connectdata *conn);
CURLcode Curl_http2_add_child(struct Curl_easy *parent,
                              struct Curl_easy *child,
                              bool exclusive);
void Curl_http2_remove_child(struct Curl_easy *parent,
                             struct Curl_easy *child);
void Curl_http2_cleanup_dependencies(struct Curl_easy *data);
CURLcode Curl_http2_stream_pause(struct Curl_easy *data, bool pause);

/* returns true if the HTTP/2 stream error was HTTP_1_1_REQUIRED */
bool Curl_h2_http_1_1_error(struct Curl_easy *data);
#else /* USE_NGHTTP2 */
#define Curl_http2_request_upgrade(x,y) CURLE_UNSUPPORTED_PROTOCOL
#define Curl_http2_setup(x,y) CURLE_UNSUPPORTED_PROTOCOL
#define Curl_http2_switched(x,y,z) CURLE_UNSUPPORTED_PROTOCOL
#define Curl_http2_setup_conn(x) Curl_nop_stmt
#define Curl_http2_setup_req(x)
#define Curl_http2_init_state(x)
#define Curl_http2_init_userset(x)
#define Curl_http2_done(x,y)
#define Curl_http2_done_sending(x,y)
#define Curl_http2_add_child(x, y, z)
#define Curl_http2_remove_child(x, y)
#define Curl_http2_cleanup_dependencies(x)
#define Curl_http2_stream_pause(x, y)
#define Curl_h2_http_1_1_error(x) 0
#endif

#endif /* HEADER_CURL_HTTP2_H */
