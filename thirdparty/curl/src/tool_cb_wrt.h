#ifndef HEADER_CURL_TOOL_CB_WRT_H
#define HEADER_CURL_TOOL_CB_WRT_H
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
** callback for CURLOPT_WRITEFUNCTION
*/

size_t tool_write_cb(char *buffer, size_t sz, size_t nmemb, void *userdata);

/* create a local file for writing, return TRUE on success */
bool tool_create_output_file(struct OutStruct *outs,
                             struct OperationConfig *config);

#endif /* HEADER_CURL_TOOL_CB_WRT_H */
