#ifndef CURLINC_MPRINTF_H
#define CURLINC_MPRINTF_H
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

#include <stdarg.h>
#include <stdio.h> /* needed for FILE */
#include "curl.h"  /* for CURL_EXTERN */

#ifdef  __cplusplus
extern "C" {
#endif

CURL_EXTERN int curl_mprintf(const char *format, ...);
CURL_EXTERN int curl_mfprintf(FILE *fd, const char *format, ...);
CURL_EXTERN int curl_msprintf(char *buffer, const char *format, ...);
CURL_EXTERN int curl_msnprintf(char *buffer, size_t maxlength,
                               const char *format, ...);
CURL_EXTERN int curl_mvprintf(const char *format, va_list args);
CURL_EXTERN int curl_mvfprintf(FILE *fd, const char *format, va_list args);
CURL_EXTERN int curl_mvsprintf(char *buffer, const char *format, va_list args);
CURL_EXTERN int curl_mvsnprintf(char *buffer, size_t maxlength,
                                const char *format, va_list args);
CURL_EXTERN char *curl_maprintf(const char *format, ...);
CURL_EXTERN char *curl_mvaprintf(const char *format, va_list args);

#ifdef  __cplusplus
}
#endif

#endif /* CURLINC_MPRINTF_H */
