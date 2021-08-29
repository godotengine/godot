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
 * RFC2104 Keyed-Hashing for Message Authentication
 *
 ***************************************************************************/

#include "curl_setup.h"

#ifndef CURL_DISABLE_CRYPTO_AUTH

#include <curl/curl.h>

#include "curl_hmac.h"
#include "curl_memory.h"
#include "warnless.h"

/* The last #include file should be: */
#include "memdebug.h"

/*
 * Generic HMAC algorithm.
 *
 *   This module computes HMAC digests based on any hash function. Parameters
 * and computing procedures are set-up dynamically at HMAC computation
 * context initialisation.
 */

static const unsigned char hmac_ipad = 0x36;
static const unsigned char hmac_opad = 0x5C;



struct HMAC_context *
Curl_HMAC_init(const struct HMAC_params *hashparams,
               const unsigned char *key,
               unsigned int keylen)
{
  size_t i;
  struct HMAC_context *ctxt;
  unsigned char *hkey;
  unsigned char b;

  /* Create HMAC context. */
  i = sizeof(*ctxt) + 2 * hashparams->hmac_ctxtsize +
    hashparams->hmac_resultlen;
  ctxt = malloc(i);

  if(!ctxt)
    return ctxt;

  ctxt->hmac_hash = hashparams;
  ctxt->hmac_hashctxt1 = (void *) (ctxt + 1);
  ctxt->hmac_hashctxt2 = (void *) ((char *) ctxt->hmac_hashctxt1 +
      hashparams->hmac_ctxtsize);

  /* If the key is too long, replace it by its hash digest. */
  if(keylen > hashparams->hmac_maxkeylen) {
    (*hashparams->hmac_hinit)(ctxt->hmac_hashctxt1);
    (*hashparams->hmac_hupdate)(ctxt->hmac_hashctxt1, key, keylen);
    hkey = (unsigned char *) ctxt->hmac_hashctxt2 + hashparams->hmac_ctxtsize;
    (*hashparams->hmac_hfinal)(hkey, ctxt->hmac_hashctxt1);
    key = hkey;
    keylen = hashparams->hmac_resultlen;
  }

  /* Prime the two hash contexts with the modified key. */
  (*hashparams->hmac_hinit)(ctxt->hmac_hashctxt1);
  (*hashparams->hmac_hinit)(ctxt->hmac_hashctxt2);

  for(i = 0; i < keylen; i++) {
    b = (unsigned char)(*key ^ hmac_ipad);
    (*hashparams->hmac_hupdate)(ctxt->hmac_hashctxt1, &b, 1);
    b = (unsigned char)(*key++ ^ hmac_opad);
    (*hashparams->hmac_hupdate)(ctxt->hmac_hashctxt2, &b, 1);
  }

  for(; i < hashparams->hmac_maxkeylen; i++) {
    (*hashparams->hmac_hupdate)(ctxt->hmac_hashctxt1, &hmac_ipad, 1);
    (*hashparams->hmac_hupdate)(ctxt->hmac_hashctxt2, &hmac_opad, 1);
  }

  /* Done, return pointer to HMAC context. */
  return ctxt;
}

int Curl_HMAC_update(struct HMAC_context *ctxt,
                     const unsigned char *data,
                     unsigned int len)
{
  /* Update first hash calculation. */
  (*ctxt->hmac_hash->hmac_hupdate)(ctxt->hmac_hashctxt1, data, len);
  return 0;
}


int Curl_HMAC_final(struct HMAC_context *ctxt, unsigned char *result)
{
  const struct HMAC_params *hashparams = ctxt->hmac_hash;

  /* Do not get result if called with a null parameter: only release
     storage. */

  if(!result)
    result = (unsigned char *) ctxt->hmac_hashctxt2 +
     ctxt->hmac_hash->hmac_ctxtsize;

  (*hashparams->hmac_hfinal)(result, ctxt->hmac_hashctxt1);
  (*hashparams->hmac_hupdate)(ctxt->hmac_hashctxt2,
   result, hashparams->hmac_resultlen);
  (*hashparams->hmac_hfinal)(result, ctxt->hmac_hashctxt2);
  free((char *) ctxt);
  return 0;
}

/*
 * Curl_hmacit()
 *
 * This is used to generate a HMAC hash, for the specified input data, given
 * the specified hash function and key.
 *
 * Parameters:
 *
 * hashparams [in]     - The hash function (Curl_HMAC_MD5).
 * key        [in]     - The key to use.
 * keylen     [in]     - The length of the key.
 * data       [in]     - The data to encrypt.
 * datalen    [in]     - The length of the data.
 * output     [in/out] - The output buffer.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_hmacit(const struct HMAC_params *hashparams,
                     const unsigned char *key, const size_t keylen,
                     const unsigned char *data, const size_t datalen,
                     unsigned char *output)
{
  struct HMAC_context *ctxt =
    Curl_HMAC_init(hashparams, key, curlx_uztoui(keylen));

  if(!ctxt)
    return CURLE_OUT_OF_MEMORY;

  /* Update the digest with the given challenge */
  Curl_HMAC_update(ctxt, data, curlx_uztoui(datalen));

  /* Finalise the digest */
  Curl_HMAC_final(ctxt, output);

  return CURLE_OK;
}

#endif /* CURL_DISABLE_CRYPTO_AUTH */
