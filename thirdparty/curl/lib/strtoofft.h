#ifndef HEADER_CURL_STRTOOFFT_H
#define HEADER_CURL_STRTOOFFT_H
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

#include "curl_setup.h"

/*
 * Determine which string to integral data type conversion function we use
 * to implement string conversion to our curl_off_t integral data type.
 *
 * Notice that curl_off_t might be 64 or 32 bit wide, and that it might use
 * an underlying data type which might be 'long', 'int64_t', 'long long' or
 * '__int64' and more remotely other data types.
 *
 * On systems where the size of curl_off_t is greater than the size of 'long'
 * the conversion function to use is strtoll() if it is available, otherwise,
 * we emulate its functionality with our own clone.
 *
 * On systems where the size of curl_off_t is smaller or equal than the size
 * of 'long' the conversion function to use is strtol().
 */

typedef enum {
  CURL_OFFT_OK,    /* parsed fine */
  CURL_OFFT_FLOW,  /* over or underflow */
  CURL_OFFT_INVAL  /* nothing was parsed */
} CURLofft;

CURLofft curlx_strtoofft(const char *str, char **endp, int base,
                         curl_off_t *num);

#endif /* HEADER_CURL_STRTOOFFT_H */
