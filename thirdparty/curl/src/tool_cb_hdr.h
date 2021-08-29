#ifndef HEADER_CURL_TOOL_CB_HDR_H
#define HEADER_CURL_TOOL_CB_HDR_H
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
 * curl operates using a single HdrCbData struct variable, a
 * pointer to this is passed as userdata pointer to tool_header_cb.
 *
 * 'outs' member is a pointer to the OutStruct variable used to keep
 * track of information relative to curl's output writing.
 *
 * 'heads' member is a pointer to the OutStruct variable used to keep
 * track of information relative to header response writing.
 *
 * 'honor_cd_filename' member is TRUE when tool_header_cb is allowed
 * to honor Content-Disposition filename property and accordingly
 * set 'outs' filename, otherwise FALSE;
 */

struct HdrCbData {
  struct GlobalConfig *global;
  struct OperationConfig *config;
  struct OutStruct *outs;
  struct OutStruct *heads;
  struct OutStruct *etag_save;
  bool honor_cd_filename;
};

/*
** callback for CURLOPT_HEADERFUNCTION
*/

size_t tool_header_cb(char *ptr, size_t size, size_t nmemb, void *userdata);

#endif /* HEADER_CURL_TOOL_CB_HDR_H */
