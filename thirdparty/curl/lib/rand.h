#ifndef HEADER_CURL_RAND_H
#define HEADER_CURL_RAND_H
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

/*
 * Curl_rand() stores 'num' number of random unsigned characters in the buffer
 * 'rnd' points to.
 *
 * If libcurl is built without TLS support or with a TLS backend that lacks a
 * proper random API (Gskit or mbedTLS), this function will use "weak" random.
 *
 * When built *with* TLS support and a backend that offers strong random, it
 * will return error if it cannot provide strong random values.
 *
 * NOTE: 'data' may be passed in as NULL when coming from external API without
 * easy handle!
 *
 */
CURLcode Curl_rand(struct Curl_easy *data, unsigned char *rnd, size_t num);

/*
 * Curl_rand_hex() fills the 'rnd' buffer with a given 'num' size with random
 * hexadecimal digits PLUS a zero terminating byte. It must be an odd number
 * size.
 */
CURLcode Curl_rand_hex(struct Curl_easy *data, unsigned char *rnd,
                       size_t num);

#endif /* HEADER_CURL_RAND_H */
