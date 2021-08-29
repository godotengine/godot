#ifndef HEADER_CURL_BASE64_H
#define HEADER_CURL_BASE64_H
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

CURLcode Curl_base64_encode(struct Curl_easy *data,
                            const char *inputbuff, size_t insize,
                            char **outptr, size_t *outlen);
CURLcode Curl_base64url_encode(struct Curl_easy *data,
                               const char *inputbuff, size_t insize,
                               char **outptr, size_t *outlen);

CURLcode Curl_base64_decode(const char *src,
                            unsigned char **outptr, size_t *outlen);

#endif /* HEADER_CURL_BASE64_H */
