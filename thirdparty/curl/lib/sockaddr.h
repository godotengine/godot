#ifndef HEADER_CURL_SOCKADDR_H
#define HEADER_CURL_SOCKADDR_H
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

struct Curl_sockaddr_storage {
  union {
    struct sockaddr sa;
    struct sockaddr_in sa_in;
#ifdef ENABLE_IPV6
    struct sockaddr_in6 sa_in6;
#endif
#ifdef HAVE_STRUCT_SOCKADDR_STORAGE
    struct sockaddr_storage sa_stor;
#else
    char cbuf[256];   /* this should be big enough to fit a lot */
#endif
  } buffer;
};

#endif /* HEADER_CURL_SOCKADDR_H */
