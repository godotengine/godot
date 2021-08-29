#ifndef HEADER_CURL_QUIC_H
#define HEADER_CURL_QUIC_H
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

#ifdef ENABLE_QUIC
#ifdef USE_NGTCP2
#include "vquic/ngtcp2.h"
#endif
#ifdef USE_QUICHE
#include "vquic/quiche.h"
#endif

#include "urldata.h"

/* functions provided by the specific backends */
CURLcode Curl_quic_connect(struct Curl_easy *data,
                           struct connectdata *conn,
                           curl_socket_t sockfd,
                           int sockindex,
                           const struct sockaddr *addr,
                           socklen_t addrlen);
CURLcode Curl_quic_is_connected(struct Curl_easy *data,
                                struct connectdata *conn,
                                int sockindex,
                                bool *connected);
void Curl_quic_ver(char *p, size_t len);
CURLcode Curl_quic_done_sending(struct Curl_easy *data);
void Curl_quic_done(struct Curl_easy *data, bool premature);
bool Curl_quic_data_pending(const struct Curl_easy *data);
void Curl_quic_disconnect(struct Curl_easy *data,
                          struct connectdata *conn, int tempindex);

#else /* ENABLE_QUIC */
#define Curl_quic_done_sending(x)
#define Curl_quic_done(x,y)
#define Curl_quic_data_pending(x)
#define Curl_quic_disconnect(x,y,z)
#endif /* !ENABLE_QUIC */

#endif /* HEADER_CURL_QUIC_H */
