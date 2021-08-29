#ifndef HEADER_CURL_STRDUP_H
#define HEADER_CURL_STRDUP_H
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
#include "curl_setup.h"

#ifndef HAVE_STRDUP
extern char *curlx_strdup(const char *str);
#endif
#ifdef WIN32
wchar_t* Curl_wcsdup(const wchar_t* src);
#endif
void *Curl_memdup(const void *src, size_t buffer_length);
void *Curl_saferealloc(void *ptr, size_t size);

#endif /* HEADER_CURL_STRDUP_H */
