#ifndef HEADER_CURL_TOOL_CB_SEE_H
#define HEADER_CURL_TOOL_CB_SEE_H
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

#if defined(WIN32) && !defined(HAVE_FTRUNCATE)

int tool_ftruncate64(int fd, curl_off_t where);

#undef  ftruncate
#define ftruncate(fd,where) tool_ftruncate64(fd,where)

#define HAVE_FTRUNCATE 1
#define USE_TOOL_FTRUNCATE 1

#endif /* WIN32  && ! HAVE_FTRUNCATE */

/*
** callback for CURLOPT_SEEKFUNCTION
*/

int tool_seek_cb(void *userdata, curl_off_t offset, int whence);

#endif /* HEADER_CURL_TOOL_CB_SEE_H */
