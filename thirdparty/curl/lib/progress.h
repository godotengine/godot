#ifndef HEADER_CURL_PROGRESS_H
#define HEADER_CURL_PROGRESS_H
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

#include "timeval.h"


typedef enum {
  TIMER_NONE,
  TIMER_STARTOP,
  TIMER_STARTSINGLE,
  TIMER_NAMELOOKUP,
  TIMER_CONNECT,
  TIMER_APPCONNECT,
  TIMER_PRETRANSFER,
  TIMER_STARTTRANSFER,
  TIMER_POSTRANSFER,
  TIMER_STARTACCEPT,
  TIMER_REDIRECT,
  TIMER_LAST /* must be last */
} timerid;

int Curl_pgrsDone(struct Curl_easy *data);
void Curl_pgrsStartNow(struct Curl_easy *data);
void Curl_pgrsSetDownloadSize(struct Curl_easy *data, curl_off_t size);
void Curl_pgrsSetUploadSize(struct Curl_easy *data, curl_off_t size);
void Curl_pgrsSetDownloadCounter(struct Curl_easy *data, curl_off_t size);
void Curl_pgrsSetUploadCounter(struct Curl_easy *data, curl_off_t size);
void Curl_ratelimit(struct Curl_easy *data, struct curltime now);
int Curl_pgrsUpdate(struct Curl_easy *data);
void Curl_pgrsResetTransferSizes(struct Curl_easy *data);
struct curltime Curl_pgrsTime(struct Curl_easy *data, timerid timer);
timediff_t Curl_pgrsLimitWaitTime(curl_off_t cursize,
                                  curl_off_t startsize,
                                  curl_off_t limit,
                                  struct curltime start,
                                  struct curltime now);

#define PGRS_HIDE    (1<<4)
#define PGRS_UL_SIZE_KNOWN (1<<5)
#define PGRS_DL_SIZE_KNOWN (1<<6)
#define PGRS_HEADERS_OUT (1<<7) /* set when the headers have been written */

#endif /* HEADER_CURL_PROGRESS_H */
