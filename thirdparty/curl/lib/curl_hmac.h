#ifndef HEADER_CURL_HMAC_H
#define HEADER_CURL_HMAC_H
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

#ifndef CURL_DISABLE_CRYPTO_AUTH

#define HMAC_MD5_LENGTH 16

typedef void    (* HMAC_hinit_func)(void *context);
typedef void    (* HMAC_hupdate_func)(void *context,
                                      const unsigned char *data,
                                      unsigned int len);
typedef void    (* HMAC_hfinal_func)(unsigned char *result, void *context);


/* Per-hash function HMAC parameters. */
struct HMAC_params {
  HMAC_hinit_func
  hmac_hinit;     /* Initialize context procedure. */
  HMAC_hupdate_func     hmac_hupdate;   /* Update context with data. */
  HMAC_hfinal_func      hmac_hfinal;    /* Get final result procedure. */
  unsigned int          hmac_ctxtsize;  /* Context structure size. */
  unsigned int          hmac_maxkeylen; /* Maximum key length (bytes). */
  unsigned int          hmac_resultlen; /* Result length (bytes). */
};


/* HMAC computation context. */
struct HMAC_context {
  const struct HMAC_params *hmac_hash; /* Hash function definition. */
  void *hmac_hashctxt1;         /* Hash function context 1. */
  void *hmac_hashctxt2;         /* Hash function context 2. */
};


/* Prototypes. */
struct HMAC_context *Curl_HMAC_init(const struct HMAC_params *hashparams,
                                    const unsigned char *key,
                                    unsigned int keylen);
int Curl_HMAC_update(struct HMAC_context *context,
                     const unsigned char *data,
                     unsigned int len);
int Curl_HMAC_final(struct HMAC_context *context, unsigned char *result);

CURLcode Curl_hmacit(const struct HMAC_params *hashparams,
                     const unsigned char *key, const size_t keylen,
                     const unsigned char *data, const size_t datalen,
                     unsigned char *output);

#endif

#endif /* HEADER_CURL_HMAC_H */
