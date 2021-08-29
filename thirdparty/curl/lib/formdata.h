#ifndef HEADER_CURL_FORMDATA_H
#define HEADER_CURL_FORMDATA_H
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

#ifndef CURL_DISABLE_MIME

/* used by FormAdd for temporary storage */
struct FormInfo {
  char *name;
  size_t namelength;
  char *value;
  curl_off_t contentslength;
  char *contenttype;
  long flags;
  char *buffer;      /* pointer to existing buffer used for file upload */
  size_t bufferlength;
  char *showfilename; /* The file name to show. If not set, the actual
                         file name will be used */
  char *userp;        /* pointer for the read callback */
  struct curl_slist *contentheader;
  struct FormInfo *more;
  bool name_alloc;
  bool value_alloc;
  bool contenttype_alloc;
  bool showfilename_alloc;
};

CURLcode Curl_getformdata(struct Curl_easy *data,
                          curl_mimepart *,
                          struct curl_httppost *post,
                          curl_read_callback fread_func);
#else
/* disabled */
#define Curl_getformdata(a,b,c,d) CURLE_NOT_BUILT_IN
#endif


#endif /* HEADER_CURL_FORMDATA_H */
