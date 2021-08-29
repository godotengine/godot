#ifndef HEADER_CURL_PATH_H
#define HEADER_CURL_PATH_H
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
#include <curl/curl.h>
#include "urldata.h"

#ifdef WIN32
#  undef  PATH_MAX
#  define PATH_MAX MAX_PATH
#  ifndef R_OK
#    define R_OK 4
#  endif
#endif

#ifndef PATH_MAX
#define PATH_MAX 1024 /* just an extra precaution since there are systems that
                         have their definition hidden well */
#endif

CURLcode Curl_getworkingpath(struct Curl_easy *data,
                             char *homedir,
                             char **path);

CURLcode Curl_get_pathname(const char **cpp, char **path, char *homedir);
#endif /* HEADER_CURL_PATH_H */
