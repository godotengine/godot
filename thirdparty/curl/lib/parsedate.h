#ifndef HEADER_CURL_PARSEDATE_H
#define HEADER_CURL_PARSEDATE_H
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

extern const char * const Curl_wkday[7];
extern const char * const Curl_month[12];

CURLcode Curl_gmtime(time_t intime, struct tm *store);

/* Curl_getdate_capped() differs from curl_getdate() in that this will return
   TIME_T_MAX in case the parsed time value was too big, instead of an
   error. */

time_t Curl_getdate_capped(const char *p);

#endif /* HEADER_CURL_PARSEDATE_H */
