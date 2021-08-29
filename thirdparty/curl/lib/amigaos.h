#ifndef HEADER_CURL_AMIGAOS_H
#define HEADER_CURL_AMIGAOS_H
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

#if defined(__AMIGA__) && defined(HAVE_BSDSOCKET_H) && !defined(USE_AMISSL)

bool Curl_amiga_init();
void Curl_amiga_cleanup();

#else

#define Curl_amiga_init() 1
#define Curl_amiga_cleanup() Curl_nop_stmt

#endif

#ifdef USE_AMISSL
#include <openssl/x509v3.h>
void Curl_amiga_X509_free(X509 *a);
#endif /* USE_AMISSL */

#endif /* HEADER_CURL_AMIGAOS_H */

