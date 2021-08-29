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

#if !defined(CURL_DISABLE_HTTP) && defined(USE_SPNEGO)

#include "urldata.h"
#include "sendf.h"
#include "http_negotiate.h"
#include "vauth/vauth.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

CURLcode Curl_input_negotiate(struct Curl_easy *data, struct connectdata *conn,
                              bool proxy, const char *header)
{
  CURLcode result;
  size_t len;

  /* Point to the username, password, service and host */
  const char *userp;
  const char *passwdp;
  const char *service;
  const char *host;

  /* Point to the correct struct with this */
  struct negotiatedata *neg_ctx;
  curlnegotiate state;

  if(proxy) {
#ifndef CURL_DISABLE_PROXY
    userp = conn->http_proxy.user;
    passwdp = conn->http_proxy.passwd;
    service = data->set.str[STRING_PROXY_SERVICE_NAME] ?
              data->set.str[STRING_PROXY_SERVICE_NAME] : "HTTP";
    host = conn->http_proxy.host.name;
    neg_ctx = &conn->proxyneg;
    state = conn->proxy_negotiate_state;
#else
    return CURLE_NOT_BUILT_IN;
#endif
  }
  else {
    userp = conn->user;
    passwdp = conn->passwd;
    service = data->set.str[STRING_SERVICE_NAME] ?
              data->set.str[STRING_SERVICE_NAME] : "HTTP";
    host = conn->host.name;
    neg_ctx = &conn->negotiate;
    state = conn->http_negotiate_state;
  }

  /* Not set means empty */
  if(!userp)
    userp = "";

  if(!passwdp)
    passwdp = "";

  /* Obtain the input token, if any */
  header += strlen("Negotiate");
  while(*header && ISSPACE(*header))
    header++;

  len = strlen(header);
  neg_ctx->havenegdata = len != 0;
  if(!len) {
    if(state == GSS_AUTHSUCC) {
      infof(data, "Negotiate auth restarted");
      Curl_http_auth_cleanup_negotiate(conn);
    }
    else if(state != GSS_AUTHNONE) {
      /* The server rejected our authentication and hasn't supplied any more
      negotiation mechanisms */
      Curl_http_auth_cleanup_negotiate(conn);
      return CURLE_LOGIN_DENIED;
    }
  }

  /* Supports SSL channel binding for Windows ISS extended protection */
#if defined(USE_WINDOWS_SSPI) && defined(SECPKG_ATTR_ENDPOINT_BINDINGS)
  neg_ctx->sslContext = conn->sslContext;
#endif

  /* Initialize the security context and decode our challenge */
  result = Curl_auth_decode_spnego_message(data, userp, passwdp, service,
                                           host, header, neg_ctx);

  if(result)
    Curl_http_auth_cleanup_negotiate(conn);

  return result;
}

CURLcode Curl_output_negotiate(struct Curl_easy *data,
                               struct connectdata *conn, bool proxy)
{
  struct negotiatedata *neg_ctx = proxy ? &conn->proxyneg :
    &conn->negotiate;
  struct auth *authp = proxy ? &data->state.authproxy : &data->state.authhost;
  curlnegotiate *state = proxy ? &conn->proxy_negotiate_state :
    &conn->http_negotiate_state;
  char *base64 = NULL;
  size_t len = 0;
  char *userp;
  CURLcode result;

  authp->done = FALSE;

  if(*state == GSS_AUTHRECV) {
    if(neg_ctx->havenegdata) {
      neg_ctx->havemultiplerequests = TRUE;
    }
  }
  else if(*state == GSS_AUTHSUCC) {
    if(!neg_ctx->havenoauthpersist) {
      neg_ctx->noauthpersist = !neg_ctx->havemultiplerequests;
    }
  }

  if(neg_ctx->noauthpersist ||
     (*state != GSS_AUTHDONE && *state != GSS_AUTHSUCC)) {

    if(neg_ctx->noauthpersist && *state == GSS_AUTHSUCC) {
      infof(data, "Curl_output_negotiate, "
            "no persistent authentication: cleanup existing context");
      Curl_http_auth_cleanup_negotiate(conn);
    }
    if(!neg_ctx->context) {
      result = Curl_input_negotiate(data, conn, proxy, "Negotiate");
      if(result == CURLE_AUTH_ERROR) {
        /* negotiate auth failed, let's continue unauthenticated to stay
         * compatible with the behavior before curl-7_64_0-158-g6c6035532 */
        authp->done = TRUE;
        return CURLE_OK;
      }
      else if(result)
        return result;
    }

    result = Curl_auth_create_spnego_message(data, neg_ctx, &base64, &len);
    if(result)
      return result;

    userp = aprintf("%sAuthorization: Negotiate %s\r\n", proxy ? "Proxy-" : "",
                    base64);

    if(proxy) {
      Curl_safefree(data->state.aptr.proxyuserpwd);
      data->state.aptr.proxyuserpwd = userp;
    }
    else {
      Curl_safefree(data->state.aptr.userpwd);
      data->state.aptr.userpwd = userp;
    }

    free(base64);

    if(!userp) {
      return CURLE_OUT_OF_MEMORY;
    }

    *state = GSS_AUTHSENT;
  #ifdef HAVE_GSSAPI
    if(neg_ctx->status == GSS_S_COMPLETE ||
       neg_ctx->status == GSS_S_CONTINUE_NEEDED) {
      *state = GSS_AUTHDONE;
    }
  #else
  #ifdef USE_WINDOWS_SSPI
    if(neg_ctx->status == SEC_E_OK ||
       neg_ctx->status == SEC_I_CONTINUE_NEEDED) {
      *state = GSS_AUTHDONE;
    }
  #endif
  #endif
  }

  if(*state == GSS_AUTHDONE || *state == GSS_AUTHSUCC) {
    /* connection is already authenticated,
     * don't send a header in future requests */
    authp->done = TRUE;
  }

  neg_ctx->havenegdata = FALSE;

  return CURLE_OK;
}

void Curl_http_auth_cleanup_negotiate(struct connectdata *conn)
{
  conn->http_negotiate_state = GSS_AUTHNONE;
  conn->proxy_negotiate_state = GSS_AUTHNONE;

  Curl_auth_cleanup_spnego(&conn->negotiate);
  Curl_auth_cleanup_spnego(&conn->proxyneg);
}

#endif /* !CURL_DISABLE_HTTP && USE_SPNEGO */
