#ifndef HEADER_CURL_INET_NTOP_H
#define HEADER_CURL_INET_NTOP_H
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

char *Curl_inet_ntop(int af, const void *addr, char *buf, size_t size);

#ifdef HAVE_INET_NTOP
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#define Curl_inet_ntop(af,addr,buf,size) \
        inet_ntop(af, addr, buf, (curl_socklen_t)size)
#endif

#endif /* HEADER_CURL_INET_NTOP_H */
