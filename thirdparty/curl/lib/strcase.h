#ifndef HEADER_CURL_STRCASE_H
#define HEADER_CURL_STRCASE_H
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

#include <curl/curl.h>

/*
 * Only "raw" case insensitive strings. This is meant to be locale independent
 * and only compare strings we know are safe for this.
 *
 * The function is capable of comparing a-z case insensitively even for
 * non-ascii.
 */

#define strcasecompare(a,b) Curl_strcasecompare(a,b)
#define strncasecompare(a,b,c) Curl_strncasecompare(a,b,c)

int Curl_strcasecompare(const char *first, const char *second);
int Curl_safe_strcasecompare(const char *first, const char *second);
int Curl_strncasecompare(const char *first, const char *second, size_t max);

char Curl_raw_toupper(char in);

/* checkprefix() is a shorter version of the above, used when the first
   argument is zero-byte terminated */
#define checkprefix(a,b)    curl_strnequal(a,b,strlen(a))

void Curl_strntoupper(char *dest, const char *src, size_t n);
void Curl_strntolower(char *dest, const char *src, size_t n);

#endif /* HEADER_CURL_STRCASE_H */
