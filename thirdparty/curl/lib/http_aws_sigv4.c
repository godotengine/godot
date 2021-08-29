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
 * are also available at https://curl.haxx.se/docs/copyright.html.
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

#if !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_CRYPTO_AUTH)

#include "urldata.h"
#include "strcase.h"
#include "strdup.h"
#include "vauth/vauth.h"
#include "vauth/digest.h"
#include "http_aws_sigv4.h"
#include "curl_sha256.h"
#include "transfer.h"

#include "strcase.h"
#include "parsedate.h"
#include "sendf.h"

#include <time.h>

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#define HMAC_SHA256(k, kl, d, dl, o)        \
  do {                                      \
    ret = Curl_hmacit(Curl_HMAC_SHA256,     \
                      (unsigned char *)k,   \
                      (unsigned int)kl,     \
                      (unsigned char *)d,   \
                      (unsigned int)dl, o); \
    if(ret != CURLE_OK) {                   \
      goto fail;                            \
    }                                       \
  } while(0)

static void sha256_to_hex(char *dst, unsigned char *sha, size_t dst_l)
{
  int i;

  DEBUGASSERT(dst_l >= 65);
  for(i = 0; i < 32; ++i) {
    curl_msnprintf(dst + (i * 2), dst_l - (i * 2), "%02x", sha[i]);
  }
}

CURLcode Curl_output_aws_sigv4(struct Curl_easy *data, bool proxy)
{
  CURLcode ret = CURLE_OUT_OF_MEMORY;
  struct connectdata *conn = data->conn;
  size_t len;
  const char *tmp0;
  const char *tmp1;
  char *provider0_low = NULL;
  char *provider0_up = NULL;
  char *provider1_low = NULL;
  char *provider1_mid = NULL;
  char *region = NULL;
  char *service = NULL;
  const char *hostname = conn->host.name;
#ifdef DEBUGBUILD
  char *force_timestamp;
#endif
  time_t clock;
  struct tm tm;
  char timestamp[17];
  char date[9];
  const char *content_type = Curl_checkheaders(data, "Content-Type");
  char *canonical_headers = NULL;
  char *signed_headers = NULL;
  Curl_HttpReq httpreq;
  const char *method;
  size_t post_data_len;
  const char *post_data = data->set.postfields ? data->set.postfields : "";
  unsigned char sha_hash[32];
  char sha_hex[65];
  char *canonical_request = NULL;
  char *request_type = NULL;
  char *credential_scope = NULL;
  char *str_to_sign = NULL;
  const char *user = data->state.aptr.user ? data->state.aptr.user : "";
  const char *passwd = data->state.aptr.passwd ? data->state.aptr.passwd : "";
  char *secret = NULL;
  unsigned char tmp_sign0[32] = {0};
  unsigned char tmp_sign1[32] = {0};
  char *auth_headers = NULL;

  DEBUGASSERT(!proxy);
  (void)proxy;

  if(Curl_checkheaders(data, "Authorization")) {
    /* Authorization already present, Bailing out */
    return CURLE_OK;
  }

  /*
   * Parameters parsing
   * Google and Outscale use the same OSC or GOOG,
   * but Amazon uses AWS and AMZ for header arguments.
   * AWS is the default because most of non-amazon providers
   * are still using aws:amz as a prefix.
   */
  tmp0 = data->set.str[STRING_AWS_SIGV4] ?
    data->set.str[STRING_AWS_SIGV4] : "aws:amz";
  tmp1 = strchr(tmp0, ':');
  len = tmp1 ? (size_t)(tmp1 - tmp0) : strlen(tmp0);
  if(len < 1) {
    infof(data, "first provider can't be empty");
    ret = CURLE_BAD_FUNCTION_ARGUMENT;
    goto fail;
  }
  provider0_low = malloc(len + 1);
  provider0_up = malloc(len + 1);
  if(!provider0_low || !provider0_up) {
    goto fail;
  }
  Curl_strntolower(provider0_low, tmp0, len);
  provider0_low[len] = '\0';
  Curl_strntoupper(provider0_up, tmp0, len);
  provider0_up[len] = '\0';

  if(tmp1) {
    tmp0 = tmp1 + 1;
    tmp1 = strchr(tmp0, ':');
    len = tmp1 ? (size_t)(tmp1 - tmp0) : strlen(tmp0);
    if(len < 1) {
      infof(data, "second provider can't be empty");
      ret = CURLE_BAD_FUNCTION_ARGUMENT;
      goto fail;
    }
    provider1_low = malloc(len + 1);
    provider1_mid = malloc(len + 1);
    if(!provider1_low || !provider1_mid) {
      goto fail;
    }
    Curl_strntolower(provider1_low, tmp0, len);
    provider1_low[len] = '\0';
    Curl_strntolower(provider1_mid, tmp0, len);
    provider1_mid[0] = Curl_raw_toupper(provider1_mid[0]);
    provider1_mid[len] = '\0';

    if(tmp1) {
      tmp0 = tmp1 + 1;
      tmp1 = strchr(tmp0, ':');
      len = tmp1 ? (size_t)(tmp1 - tmp0) : strlen(tmp0);
      if(len < 1) {
        infof(data, "region can't be empty");
        ret = CURLE_BAD_FUNCTION_ARGUMENT;
        goto fail;
      }
      region = Curl_memdup(tmp0, len + 1);
      if(!region) {
        goto fail;
      }
      region[len] = '\0';

      if(tmp1) {
        tmp0 = tmp1 + 1;
        service = strdup(tmp0);
        if(!service) {
          goto fail;
        }
        if(strlen(service) < 1) {
          infof(data, "service can't be empty");
          ret = CURLE_BAD_FUNCTION_ARGUMENT;
          goto fail;
        }
      }
    }
  }
  else {
    provider1_low = Curl_memdup(provider0_low, len + 1);
    provider1_mid = Curl_memdup(provider0_low, len + 1);
    if(!provider1_low || !provider1_mid) {
      goto fail;
    }
    provider1_mid[0] = Curl_raw_toupper(provider1_mid[0]);
  }

  if(!service) {
    tmp0 = hostname;
    tmp1 = strchr(tmp0, '.');
    len = tmp1 - tmp0;
    if(!tmp1 || len < 1) {
      infof(data, "service missing in parameters or hostname");
      ret = CURLE_URL_MALFORMAT;
      goto fail;
    }
    service = Curl_memdup(tmp0, len + 1);
    if(!service) {
      goto fail;
    }
    service[len] = '\0';

    if(!region) {
      tmp0 = tmp1 + 1;
      tmp1 = strchr(tmp0, '.');
      len = tmp1 - tmp0;
      if(!tmp1 || len < 1) {
        infof(data, "region missing in parameters or hostname");
        ret = CURLE_URL_MALFORMAT;
        goto fail;
      }
      region = Curl_memdup(tmp0, len + 1);
      if(!region) {
        goto fail;
      }
      region[len] = '\0';
    }
  }

#ifdef DEBUGBUILD
  force_timestamp = getenv("CURL_FORCETIME");
  if(force_timestamp)
    clock = 0;
  else
    time(&clock);
#else
  time(&clock);
#endif
  ret = Curl_gmtime(clock, &tm);
  if(ret != CURLE_OK) {
    goto fail;
  }
  if(!strftime(timestamp, sizeof(timestamp), "%Y%m%dT%H%M%SZ", &tm)) {
    goto fail;
  }
  memcpy(date, timestamp, sizeof(date));
  date[sizeof(date) - 1] = 0;

  if(content_type) {
    content_type = strchr(content_type, ':');
    if(!content_type) {
      ret = CURLE_FAILED_INIT;
      goto fail;
    }
    content_type++;
    /* Skip whitespace now */
    while(*content_type == ' ' || *content_type == '\t')
      ++content_type;

    canonical_headers = curl_maprintf("content-type:%s\n"
                                      "host:%s\n"
                                      "x-%s-date:%s\n",
                                      content_type,
                                      hostname,
                                      provider1_low, timestamp);
    signed_headers = curl_maprintf("content-type;host;x-%s-date",
                                   provider1_low);
  }
  else {
    canonical_headers = curl_maprintf("host:%s\n"
                                      "x-%s-date:%s\n",
                                      hostname,
                                      provider1_low, timestamp);
    signed_headers = curl_maprintf("host;x-%s-date", provider1_low);
  }

  if(!canonical_headers || !signed_headers) {
    goto fail;
  }

  if(data->set.postfieldsize < 0)
    post_data_len = strlen(post_data);
  else
    post_data_len = (size_t)data->set.postfieldsize;
  Curl_sha256it(sha_hash,
                (const unsigned char *) post_data, post_data_len);
  sha256_to_hex(sha_hex, sha_hash, sizeof(sha_hex));

  Curl_http_method(data, conn, &method, &httpreq);

  canonical_request =
    curl_maprintf("%s\n" /* HTTPRequestMethod */
                  "%s\n" /* CanonicalURI */
                  "%s\n" /* CanonicalQueryString */
                  "%s\n" /* CanonicalHeaders */
                  "%s\n" /* SignedHeaders */
                  "%s",  /* HashedRequestPayload in hex */
                  method,
                  data->state.up.path,
                  data->state.up.query ? data->state.up.query : "",
                  canonical_headers,
                  signed_headers,
                  sha_hex);
  if(!canonical_request) {
    goto fail;
  }

  request_type = curl_maprintf("%s4_request", provider0_low);
  if(!request_type) {
    goto fail;
  }

  credential_scope = curl_maprintf("%s/%s/%s/%s",
                                   date, region, service, request_type);
  if(!credential_scope) {
    goto fail;
  }

  Curl_sha256it(sha_hash, (unsigned char *) canonical_request,
                strlen(canonical_request));
  sha256_to_hex(sha_hex, sha_hash, sizeof(sha_hex));

  /*
   * Google allow to use rsa key instead of HMAC, so this code might change
   * In the future, but for now we support only HMAC version
   */
  str_to_sign = curl_maprintf("%s4-HMAC-SHA256\n" /* Algorithm */
                              "%s\n" /* RequestDateTime */
                              "%s\n" /* CredentialScope */
                              "%s",  /* HashedCanonicalRequest in hex */
                              provider0_up,
                              timestamp,
                              credential_scope,
                              sha_hex);
  if(!str_to_sign) {
    goto fail;
  }

  secret = curl_maprintf("%s4%s", provider0_up, passwd);
  if(!secret) {
    goto fail;
  }

  HMAC_SHA256(secret, strlen(secret),
              date, strlen(date), tmp_sign0);
  HMAC_SHA256(tmp_sign0, sizeof(tmp_sign0),
              region, strlen(region), tmp_sign1);
  HMAC_SHA256(tmp_sign1, sizeof(tmp_sign1),
              service, strlen(service), tmp_sign0);
  HMAC_SHA256(tmp_sign0, sizeof(tmp_sign0),
              request_type, strlen(request_type), tmp_sign1);
  HMAC_SHA256(tmp_sign1, sizeof(tmp_sign1),
              str_to_sign, strlen(str_to_sign), tmp_sign0);

  sha256_to_hex(sha_hex, tmp_sign0, sizeof(sha_hex));

  auth_headers = curl_maprintf("Authorization: %s4-HMAC-SHA256 "
                               "Credential=%s/%s, "
                               "SignedHeaders=%s, "
                               "Signature=%s\r\n"
                               "X-%s-Date: %s\r\n",
                               provider0_up,
                               user,
                               credential_scope,
                               signed_headers,
                               sha_hex,
                               provider1_mid,
                               timestamp);
  if(!auth_headers) {
    goto fail;
  }

  Curl_safefree(data->state.aptr.userpwd);
  data->state.aptr.userpwd = auth_headers;
  data->state.authhost.done = TRUE;
  ret = CURLE_OK;

fail:
  free(provider0_low);
  free(provider0_up);
  free(provider1_low);
  free(provider1_mid);
  free(region);
  free(service);
  free(canonical_headers);
  free(signed_headers);
  free(canonical_request);
  free(request_type);
  free(credential_scope);
  free(str_to_sign);
  free(secret);
  return ret;
}

#endif /* !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_CRYPTO_AUTH) */
