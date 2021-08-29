#ifndef HEADER_CURL_TOOL_FILETIME_H
#define HEADER_CURL_TOOL_FILETIME_H
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
#include "tool_setup.h"

struct GlobalConfig;

curl_off_t getfiletime(const char *filename, struct GlobalConfig *global);

#if defined(HAVE_UTIME) || defined(HAVE_UTIMES) ||      \
  (defined(WIN32) && (SIZEOF_CURL_OFF_T >= 8))
void setfiletime(curl_off_t filetime, const char *filename,
                 struct GlobalConfig *global);
#else
#define setfiletime(a,b,c) Curl_nop_stmt
#endif /* defined(HAVE_UTIME) || defined(HAVE_UTIMES) ||        \
          (defined(WIN32) && (SIZEOF_CURL_OFF_T >= 8)) */

#endif /* HEADER_CURL_TOOL_FILETIME_H */
