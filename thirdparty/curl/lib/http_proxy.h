#ifndef HEADER_CURL_HTTP_PROXY_H
#define HEADER_CURL_HTTP_PROXY_H
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
#include "urldata.h"

#if !defined(CURL_DISABLE_PROXY) && !defined(CURL_DISABLE_HTTP)
/* ftp can use this as well */
CURLcode Curl_proxyCONNECT(struct Curl_easy *data,
                           int tunnelsocket,
                           const char *hostname, int remote_port);

/* Default proxy timeout in milliseconds */
#define PROXY_TIMEOUT (3600*1000)

CURLcode Curl_proxy_connect(struct Curl_easy *data, int sockindex);

bool Curl_connect_complete(struct connectdata *conn);
bool Curl_connect_ongoing(struct connectdata *conn);
int Curl_connect_getsock(struct connectdata *conn);
void Curl_connect_done(struct Curl_easy *data);

#else
#define Curl_proxyCONNECT(x,y,z,w) CURLE_NOT_BUILT_IN
#define Curl_proxy_connect(x,y) CURLE_OK
#define Curl_connect_complete(x) CURLE_OK
#define Curl_connect_ongoing(x) FALSE
#define Curl_connect_getsock(x) 0
#define Curl_connect_done(x)
#endif

void Curl_connect_free(struct Curl_easy *data);

/* struct for HTTP CONNECT state data */
struct http_connect_state {
  struct HTTP http_proxy;
  struct HTTP *prot_save;
  struct dynbuf rcvbuf;
  struct dynbuf req;
  size_t nsend;
  enum keeponval {
    KEEPON_DONE,
    KEEPON_CONNECT,
    KEEPON_IGNORE
  } keepon;
  curl_off_t cl; /* size of content to read and ignore */
  enum {
    TUNNEL_INIT,     /* init/default/no tunnel state */
    TUNNEL_CONNECT,  /* CONNECT has been sent off */
    TUNNEL_COMPLETE, /* CONNECT response received completely */
    TUNNEL_EXIT
  } tunnel_state;
  BIT(chunked_encoding);
  BIT(close_connection);
};

#endif /* HEADER_CURL_HTTP_PROXY_H */
