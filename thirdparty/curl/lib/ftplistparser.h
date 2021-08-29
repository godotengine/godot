#ifndef HEADER_CURL_FTPLISTPARSER_H
#define HEADER_CURL_FTPLISTPARSER_H
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

#ifndef CURL_DISABLE_FTP

/* WRITEFUNCTION callback for parsing LIST responses */
size_t Curl_ftp_parselist(char *buffer, size_t size, size_t nmemb,
                          void *connptr);

struct ftp_parselist_data; /* defined inside ftplibparser.c */

CURLcode Curl_ftp_parselist_geterror(struct ftp_parselist_data *pl_data);

struct ftp_parselist_data *Curl_ftp_parselist_data_alloc(void);

void Curl_ftp_parselist_data_free(struct ftp_parselist_data **pl_data);

#endif /* CURL_DISABLE_FTP */
#endif /* HEADER_CURL_FTPLISTPARSER_H */
