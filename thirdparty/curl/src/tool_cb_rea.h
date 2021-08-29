#ifndef HEADER_CURL_TOOL_CB_REA_H
#define HEADER_CURL_TOOL_CB_REA_H
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

/*
** callback for CURLOPT_READFUNCTION
*/

size_t tool_read_cb(char *buffer, size_t sz, size_t nmemb, void *userdata);

/*
** callback for CURLOPT_XFERINFOFUNCTION used to unpause busy reads
*/

int tool_readbusy_cb(void *clientp,
                     curl_off_t dltotal, curl_off_t dlnow,
                     curl_off_t ultotal, curl_off_t ulnow);

#endif /* HEADER_CURL_TOOL_CB_REA_H */
