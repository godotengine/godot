#ifndef HEADER_CURL_HYPER_H
#define HEADER_CURL_HYPER_H
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
 * are also available at https://curl.haxx.se/docs/copyright.html.
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

#if !defined(CURL_DISABLE_HTTP) && defined(USE_HYPER)

#include <hyper.h>

/* per-transfer data for the Hyper backend */
struct hyptransfer {
  hyper_waker *write_waker;
  hyper_waker *read_waker;
  const hyper_executor *exec;
  hyper_task *endtask;
  hyper_waker *exp100_waker;
};

size_t Curl_hyper_recv(void *userp, hyper_context *ctx,
                       uint8_t *buf, size_t buflen);
size_t Curl_hyper_send(void *userp, hyper_context *ctx,
                       const uint8_t *buf, size_t buflen);
CURLcode Curl_hyper_stream(struct Curl_easy *data,
                           struct connectdata *conn,
                           int *didwhat,
                           bool *done,
                           int select_res);

CURLcode Curl_hyper_header(struct Curl_easy *data, hyper_headers *headers,
                           const char *line);
void Curl_hyper_done(struct Curl_easy *);

#else
#define Curl_hyper_done(x)

#endif /* !defined(CURL_DISABLE_HTTP) && defined(USE_HYPER) */
#endif /* HEADER_CURL_HYPER_H */
