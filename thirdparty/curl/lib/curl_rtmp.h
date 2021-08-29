#ifndef HEADER_CURL_RTMP_H
#define HEADER_CURL_RTMP_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2010 - 2020, Howard Chu, <hyc@highlandsun.com>
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
#ifdef USE_LIBRTMP
extern const struct Curl_handler Curl_handler_rtmp;
extern const struct Curl_handler Curl_handler_rtmpt;
extern const struct Curl_handler Curl_handler_rtmpe;
extern const struct Curl_handler Curl_handler_rtmpte;
extern const struct Curl_handler Curl_handler_rtmps;
extern const struct Curl_handler Curl_handler_rtmpts;
#endif

#endif /* HEADER_CURL_RTMP_H */
