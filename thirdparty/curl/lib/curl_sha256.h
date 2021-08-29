#ifndef HEADER_CURL_SHA256_H
#define HEADER_CURL_SHA256_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2017, Florin Petriuc, <petriuc.florin@gmail.com>
 * Copyright (C) 2018 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

#ifndef CURL_DISABLE_CRYPTO_AUTH
#include "curl_hmac.h"

extern const struct HMAC_params Curl_HMAC_SHA256[1];

#ifdef USE_WOLFSSL
/* SHA256_DIGEST_LENGTH is an enum value in wolfSSL. Need to import it from
 * sha.h*/
#include <wolfssl/options.h>
#include <openssl/sha.h>
#else
#define SHA256_DIGEST_LENGTH 32
#endif

void Curl_sha256it(unsigned char *outbuffer, const unsigned char *input,
                   const size_t len);

#endif

#endif /* HEADER_CURL_SHA256_H */
