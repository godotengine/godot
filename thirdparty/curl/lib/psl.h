#ifndef HEADER_PSL_H
#define HEADER_PSL_H
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

#ifdef USE_LIBPSL
#include <libpsl.h>

#define PSL_TTL (72 * 3600)     /* PSL time to live before a refresh. */

struct PslCache {
  const psl_ctx_t *psl; /* The PSL. */
  time_t expires; /* Time this PSL life expires. */
  bool dynamic; /* PSL should be released when no longer needed. */
};

const psl_ctx_t *Curl_psl_use(struct Curl_easy *easy);
void Curl_psl_release(struct Curl_easy *easy);
void Curl_psl_destroy(struct PslCache *pslcache);

#else

#define Curl_psl_use(easy) NULL
#define Curl_psl_release(easy)
#define Curl_psl_destroy(pslcache)

#endif /* USE_LIBPSL */
#endif /* HEADER_PSL_H */
