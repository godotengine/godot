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

#ifdef ENABLE_QUIC

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
#include "urldata.h"
#include "dynbuf.h"
#include "curl_printf.h"
#include "vquic.h"

#ifdef O_BINARY
#define QLOGMODE O_WRONLY|O_CREAT|O_BINARY
#else
#define QLOGMODE O_WRONLY|O_CREAT
#endif

/*
 * If the QLOGDIR environment variable is set, open and return a file
 * descriptor to write the log to.
 *
 * This function returns error if something failed outside of failing to
 * create the file. Open file success is deemed by seeing if the returned fd
 * is != -1.
 */
CURLcode Curl_qlogdir(struct Curl_easy *data,
                      unsigned char *scid,
                      size_t scidlen,
                      int *qlogfdp)
{
  const char *qlog_dir = getenv("QLOGDIR");
  *qlogfdp = -1;
  if(qlog_dir) {
    struct dynbuf fname;
    CURLcode result;
    unsigned int i;
    Curl_dyn_init(&fname, DYN_QLOG_NAME);
    result = Curl_dyn_add(&fname, qlog_dir);
    if(!result)
      result = Curl_dyn_add(&fname, "/");
    for(i = 0; (i < scidlen) && !result; i++) {
      char hex[3];
      msnprintf(hex, 3, "%02x", scid[i]);
      result = Curl_dyn_add(&fname, hex);
    }
    if(!result)
      result = Curl_dyn_add(&fname, ".qlog");

    if(!result) {
      int qlogfd = open(Curl_dyn_ptr(&fname), QLOGMODE,
                        data->set.new_file_perms);
      if(qlogfd != -1)
        *qlogfdp = qlogfd;
    }
    Curl_dyn_free(&fname);
    if(result)
      return result;
  }

  return CURLE_OK;
}
#endif
