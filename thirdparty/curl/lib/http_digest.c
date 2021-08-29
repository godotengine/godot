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

#if !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_CRYPTO_AUTH)

#include "urldata.h"
#include "strcase.h"
#include "vauth/vauth.h"
#include "http_digest.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/* Test example headers:

WWW-Authenticate: Digest realm="testrealm", nonce="1053604598"
Proxy-Authenticate: Digest realm="testrealm", nonce="1053604598"

*/

CURLcode Curl_input_digest(struct Curl_easy *data,
                           bool proxy,
                           const char *header) /* rest of the *-authenticate:
                                                  header */
{
  /* Point to the correct struct with this */
  struct digestdata *digest;

  if(proxy) {
    digest = &data->state.proxydigest;
  }
  else {
    digest = &data->state.digest;
  }

  if(!checkprefix("Digest", header) || !ISSPACE(header[6]))
    return CURLE_BAD_CONTENT_ENCODING;

  header += strlen("Digest");
  while(*header && ISSPACE(*header))
    header++;

  return Curl_auth_decode_digest_http_message(header, digest);
}

CURLcode Curl_output_digest(struct Curl_easy *data,
                            bool proxy,
                            const unsigned char *request,
                            const unsigned char *uripath)
{
  CURLcode result;
  unsigned char *path = NULL;
  char *tmp = NULL;
  char *response;
  size_t len;
  bool have_chlg;

  /* Point to the address of the pointer that holds the string to send to the
     server, which is for a plain host or for a HTTP proxy */
  char **allocuserpwd;

  /* Point to the name and password for this */
  const char *userp;
  const char *passwdp;

  /* Point to the correct struct with this */
  struct digestdata *digest;
  struct auth *authp;

  if(proxy) {
#ifdef CURL_DISABLE_PROXY
    return CURLE_NOT_BUILT_IN;
#else
    digest = &data->state.proxydigest;
    allocuserpwd = &data->state.aptr.proxyuserpwd;
    userp = data->state.aptr.proxyuser;
    passwdp = data->state.aptr.proxypasswd;
    authp = &data->state.authproxy;
#endif
  }
  else {
    digest = &data->state.digest;
    allocuserpwd = &data->state.aptr.userpwd;
    userp = data->state.aptr.user;
    passwdp = data->state.aptr.passwd;
    authp = &data->state.authhost;
  }

  Curl_safefree(*allocuserpwd);

  /* not set means empty */
  if(!userp)
    userp = "";

  if(!passwdp)
    passwdp = "";

#if defined(USE_WINDOWS_SSPI)
  have_chlg = digest->input_token ? TRUE : FALSE;
#else
  have_chlg = digest->nonce ? TRUE : FALSE;
#endif

  if(!have_chlg) {
    authp->done = FALSE;
    return CURLE_OK;
  }

  /* So IE browsers < v7 cut off the URI part at the query part when they
     evaluate the MD5 and some (IIS?) servers work with them so we may need to
     do the Digest IE-style. Note that the different ways cause different MD5
     sums to get sent.

     Apache servers can be set to do the Digest IE-style automatically using
     the BrowserMatch feature:
     https://httpd.apache.org/docs/2.2/mod/mod_auth_digest.html#msie

     Further details on Digest implementation differences:
     http://www.fngtps.com/2006/09/http-authentication
  */

  if(authp->iestyle) {
    tmp = strchr((char *)uripath, '?');
    if(tmp) {
      size_t urilen = tmp - (char *)uripath;
      /* typecast is fine here since the value is always less than 32 bits */
      path = (unsigned char *) aprintf("%.*s", (int)urilen, uripath);
    }
  }
  if(!tmp)
    path = (unsigned char *) strdup((char *) uripath);

  if(!path)
    return CURLE_OUT_OF_MEMORY;

  result = Curl_auth_create_digest_http_message(data, userp, passwdp, request,
                                                path, digest, &response, &len);
  free(path);
  if(result)
    return result;

  *allocuserpwd = aprintf("%sAuthorization: Digest %s\r\n",
                          proxy ? "Proxy-" : "",
                          response);
  free(response);
  if(!*allocuserpwd)
    return CURLE_OUT_OF_MEMORY;

  authp->done = TRUE;

  return CURLE_OK;
}

void Curl_http_auth_cleanup_digest(struct Curl_easy *data)
{
  Curl_auth_digest_cleanup(&data->state.digest);
  Curl_auth_digest_cleanup(&data->state.proxydigest);
}

#endif
