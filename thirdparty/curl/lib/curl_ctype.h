#ifndef HEADER_CURL_CTYPE_H
#define HEADER_CURL_CTYPE_H
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

#ifdef CURL_DOES_CONVERSIONS

/*
 * Uppercase macro versions of ANSI/ISO is*() functions/macros which
 * avoid negative number inputs with argument byte codes > 127.
 *
 * For non-ASCII platforms the C library character classification routines
 * are used despite being locale-dependent, because this is better than
 * not to work at all.
 */
#include <ctype.h>

#define ISSPACE(x)  (isspace((int)  ((unsigned char)x)))
#define ISDIGIT(x)  (isdigit((int)  ((unsigned char)x)))
#define ISALNUM(x)  (isalnum((int)  ((unsigned char)x)))
#define ISXDIGIT(x) (isxdigit((int) ((unsigned char)x)))
#define ISGRAPH(x)  (isgraph((int)  ((unsigned char)x)))
#define ISALPHA(x)  (isalpha((int)  ((unsigned char)x)))
#define ISPRINT(x)  (isprint((int)  ((unsigned char)x)))
#define ISUPPER(x)  (isupper((int)  ((unsigned char)x)))
#define ISLOWER(x)  (islower((int)  ((unsigned char)x)))
#define ISCNTRL(x)  (iscntrl((int)  ((unsigned char)x)))
#define ISASCII(x)  (isascii((int)  ((unsigned char)x)))

#else

int Curl_isspace(int c);
int Curl_isdigit(int c);
int Curl_isalnum(int c);
int Curl_isxdigit(int c);
int Curl_isgraph(int c);
int Curl_isprint(int c);
int Curl_isalpha(int c);
int Curl_isupper(int c);
int Curl_islower(int c);
int Curl_iscntrl(int c);

#define ISSPACE(x)  (Curl_isspace((int)  ((unsigned char)x)))
#define ISDIGIT(x)  (Curl_isdigit((int)  ((unsigned char)x)))
#define ISALNUM(x)  (Curl_isalnum((int)  ((unsigned char)x)))
#define ISXDIGIT(x) (Curl_isxdigit((int) ((unsigned char)x)))
#define ISGRAPH(x)  (Curl_isgraph((int)  ((unsigned char)x)))
#define ISALPHA(x)  (Curl_isalpha((int)  ((unsigned char)x)))
#define ISPRINT(x)  (Curl_isprint((int)  ((unsigned char)x)))
#define ISUPPER(x)  (Curl_isupper((int)  ((unsigned char)x)))
#define ISLOWER(x)  (Curl_islower((int)  ((unsigned char)x)))
#define ISCNTRL(x)  (Curl_iscntrl((int)  ((unsigned char)x)))
#define ISASCII(x)  (((x) >= 0) && ((x) <= 0x80))

#endif

#define ISBLANK(x)  (int)((((unsigned char)x) == ' ') ||        \
                          (((unsigned char)x) == '\t'))

#endif /* HEADER_CURL_CTYPE_H */
