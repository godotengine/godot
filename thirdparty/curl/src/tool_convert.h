#ifndef HEADER_CURL_TOOL_CONVERT_H
#define HEADER_CURL_TOOL_CONVERT_H
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
#include "tool_setup.h"

#ifdef CURL_DOES_CONVERSIONS

#ifdef HAVE_ICONV

CURLcode convert_to_network(char *buffer, size_t length);
CURLcode convert_from_network(char *buffer, size_t length);
void convert_cleanup(void);

#endif /* HAVE_ICONV */

char convert_char(curl_infotype infotype, char this_char);

#endif /* CURL_DOES_CONVERSIONS */

#if !defined(CURL_DOES_CONVERSIONS) || !defined(HAVE_ICONV)
#define convert_cleanup() Curl_nop_stmt
#endif

#endif /* HEADER_CURL_TOOL_CONVERT_H */
