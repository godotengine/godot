/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2012 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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
 * RFC2195 CRAM-MD5 authentication
 * RFC2617 Basic and Digest Access Authentication
 * RFC2831 DIGEST-MD5 authentication
 * RFC4422 Simple Authentication and Security Layer (SASL)
 * RFC4616 PLAIN authentication
 * RFC5802 SCRAM-SHA-1 authentication
 * RFC7677 SCRAM-SHA-256 authentication
 * RFC6749 OAuth 2.0 Authorization Framework
 * RFC7628 A Set of SASL Mechanisms for OAuth
 * Draft   LOGIN SASL Mechanism <draft-murchison-sasl-login-00.txt>
 *
 ***************************************************************************/

#include "curl_setup.h"

#if !defined(CURL_DISABLE_IMAP) || !defined(CURL_DISABLE_SMTP) || \
  !defined(CURL_DISABLE_POP3)

#include <curl/curl.h>
#include "urldata.h"

#include "curl_base64.h"
#include "curl_md5.h"
#include "vauth/vauth.h"
#include "vtls/vtls.h"
#include "curl_hmac.h"
#include "curl_sasl.h"
#include "warnless.h"
#include "strtok.h"
#include "sendf.h"
#include "non-ascii.h" /* included for Curl_convert_... prototypes */
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/* Supported mechanisms */
static const struct {
  const char    *name;  /* Name */
  size_t         len;   /* Name length */
  unsigned short bit;   /* Flag bit */
} mechtable[] = {
  { "LOGIN",        5,  SASL_MECH_LOGIN },
  { "PLAIN",        5,  SASL_MECH_PLAIN },
  { "CRAM-MD5",     8,  SASL_MECH_CRAM_MD5 },
  { "DIGEST-MD5",   10, SASL_MECH_DIGEST_MD5 },
  { "GSSAPI",       6,  SASL_MECH_GSSAPI },
  { "EXTERNAL",     8,  SASL_MECH_EXTERNAL },
  { "NTLM",         4,  SASL_MECH_NTLM },
  { "XOAUTH2",      7,  SASL_MECH_XOAUTH2 },
  { "OAUTHBEARER",  11, SASL_MECH_OAUTHBEARER },
  { "SCRAM-SHA-1",  11, SASL_MECH_SCRAM_SHA_1 },
  { "SCRAM-SHA-256",13, SASL_MECH_SCRAM_SHA_256 },
  { ZERO_NULL,      0,  0 }
};

/*
 * Curl_sasl_cleanup()
 *
 * This is used to cleanup any libraries or curl modules used by the sasl
 * functions.
 *
 * Parameters:
 *
 * conn     [in]     - The connection data.
 * authused [in]     - The authentication mechanism used.
 */
void Curl_sasl_cleanup(struct connectdata *conn, unsigned short authused)
{
  (void)conn;
  (void)authused;

#if defined(USE_KERBEROS5)
  /* Cleanup the gssapi structure */
  if(authused == SASL_MECH_GSSAPI) {
    Curl_auth_cleanup_gssapi(&conn->krb5);
  }
#endif

#if defined(USE_GSASL)
  /* Cleanup the GSASL structure */
  if(authused & (SASL_MECH_SCRAM_SHA_1 | SASL_MECH_SCRAM_SHA_256)) {
    Curl_auth_gsasl_cleanup(&conn->gsasl);
  }
#endif

#if defined(USE_NTLM)
  /* Cleanup the NTLM structure */
  if(authused == SASL_MECH_NTLM) {
    Curl_auth_cleanup_ntlm(&conn->ntlm);
  }
#endif
}

/*
 * Curl_sasl_decode_mech()
 *
 * Convert a SASL mechanism name into a token.
 *
 * Parameters:
 *
 * ptr    [in]     - The mechanism string.
 * maxlen [in]     - Maximum mechanism string length.
 * len    [out]    - If not NULL, effective name length.
 *
 * Returns the SASL mechanism token or 0 if no match.
 */
unsigned short Curl_sasl_decode_mech(const char *ptr, size_t maxlen,
                                     size_t *len)
{
  unsigned int i;
  char c;

  for(i = 0; mechtable[i].name; i++) {
    if(maxlen >= mechtable[i].len &&
       !memcmp(ptr, mechtable[i].name, mechtable[i].len)) {
      if(len)
        *len = mechtable[i].len;

      if(maxlen == mechtable[i].len)
        return mechtable[i].bit;

      c = ptr[mechtable[i].len];
      if(!ISUPPER(c) && !ISDIGIT(c) && c != '-' && c != '_')
        return mechtable[i].bit;
    }
  }

  return 0;
}

/*
 * Curl_sasl_parse_url_auth_option()
 *
 * Parse the URL login options.
 */
CURLcode Curl_sasl_parse_url_auth_option(struct SASL *sasl,
                                         const char *value, size_t len)
{
  CURLcode result = CURLE_OK;
  size_t mechlen;

  if(!len)
    return CURLE_URL_MALFORMAT;

  if(sasl->resetprefs) {
    sasl->resetprefs = FALSE;
    sasl->prefmech = SASL_AUTH_NONE;
  }

  if(!strncmp(value, "*", len))
    sasl->prefmech = SASL_AUTH_DEFAULT;
  else {
    unsigned short mechbit = Curl_sasl_decode_mech(value, len, &mechlen);
    if(mechbit && mechlen == len)
      sasl->prefmech |= mechbit;
    else
      result = CURLE_URL_MALFORMAT;
  }

  return result;
}

/*
 * Curl_sasl_init()
 *
 * Initializes the SASL structure.
 */
void Curl_sasl_init(struct SASL *sasl, struct Curl_easy *data,
                    const struct SASLproto *params)
{
  unsigned long auth = data->set.httpauth;

  sasl->params = params;           /* Set protocol dependent parameters */
  sasl->state = SASL_STOP;         /* Not yet running */
  sasl->curmech = NULL;            /* No mechanism yet. */
  sasl->authmechs = SASL_AUTH_NONE; /* No known authentication mechanism yet */
  sasl->prefmech = params->defmechs; /* Default preferred mechanisms */
  sasl->authused = SASL_AUTH_NONE; /* The authentication mechanism used */
  sasl->resetprefs = TRUE;         /* Reset prefmech upon AUTH parsing. */
  sasl->mutual_auth = FALSE;       /* No mutual authentication (GSSAPI only) */
  sasl->force_ir = FALSE;          /* Respect external option */

  if(auth != CURLAUTH_BASIC) {
    sasl->resetprefs = FALSE;
    sasl->prefmech = SASL_AUTH_NONE;
    if(auth & CURLAUTH_BASIC)
      sasl->prefmech |= SASL_MECH_PLAIN | SASL_MECH_LOGIN;
    if(auth & CURLAUTH_DIGEST)
      sasl->prefmech |= SASL_MECH_DIGEST_MD5;
    if(auth & CURLAUTH_NTLM)
      sasl->prefmech |= SASL_MECH_NTLM;
    if(auth & CURLAUTH_BEARER)
      sasl->prefmech |= SASL_MECH_OAUTHBEARER | SASL_MECH_XOAUTH2;
    if(auth & CURLAUTH_GSSAPI)
      sasl->prefmech |= SASL_MECH_GSSAPI;
  }
}

/*
 * state()
 *
 * This is the ONLY way to change SASL state!
 */
static void state(struct SASL *sasl, struct Curl_easy *data,
                  saslstate newstate)
{
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  /* for debug purposes */
  static const char * const names[]={
    "STOP",
    "PLAIN",
    "LOGIN",
    "LOGIN_PASSWD",
    "EXTERNAL",
    "CRAMMD5",
    "DIGESTMD5",
    "DIGESTMD5_RESP",
    "NTLM",
    "NTLM_TYPE2MSG",
    "GSSAPI",
    "GSSAPI_TOKEN",
    "GSSAPI_NO_DATA",
    "OAUTH2",
    "OAUTH2_RESP",
    "GSASL",
    "CANCEL",
    "FINAL",
    /* LAST */
  };

  if(sasl->state != newstate)
    infof(data, "SASL %p state change from %s to %s",
          (void *)sasl, names[sasl->state], names[newstate]);
#else
  (void) data;
#endif

  sasl->state = newstate;
}

/* Get the SASL server message and convert it to binary. */
static CURLcode get_server_message(struct SASL *sasl, struct Curl_easy *data,
                                   struct bufref *out)
{
  CURLcode result = CURLE_OK;

  result = sasl->params->getmessage(data, out);
  if(!result && (sasl->params->flags & SASL_FLAG_BASE64)) {
    unsigned char *msg;
    size_t msglen;
    const char *serverdata = (const char *) Curl_bufref_ptr(out);

    if(!*serverdata || *serverdata == '=')
      Curl_bufref_set(out, NULL, 0, NULL);
    else {
      result = Curl_base64_decode(serverdata, &msg, &msglen);
      if(!result)
        Curl_bufref_set(out, msg, msglen, curl_free);
    }
  }
  return result;
}

/* Encode the outgoing SASL message. */
static CURLcode build_message(struct SASL *sasl, struct Curl_easy *data,
                              struct bufref *msg)
{
  CURLcode result = CURLE_OK;

  if(sasl->params->flags & SASL_FLAG_BASE64) {
    if(!Curl_bufref_ptr(msg))                   /* Empty message. */
      Curl_bufref_set(msg, "", 0, NULL);
    else if(!Curl_bufref_len(msg))              /* Explicit empty response. */
      Curl_bufref_set(msg, "=", 1, NULL);
    else {
      char *base64;
      size_t base64len;

      result = Curl_base64_encode(data, (const char *) Curl_bufref_ptr(msg),
                                  Curl_bufref_len(msg), &base64, &base64len);
      if(!result)
        Curl_bufref_set(msg, base64, base64len, curl_free);
    }
  }

  return result;
}

/*
 * Curl_sasl_can_authenticate()
 *
 * Check if we have enough auth data and capabilities to authenticate.
 */
bool Curl_sasl_can_authenticate(struct SASL *sasl, struct connectdata *conn)
{
  /* Have credentials been provided? */
  if(conn->bits.user_passwd)
    return TRUE;

  /* EXTERNAL can authenticate without a user name and/or password */
  if(sasl->authmechs & sasl->prefmech & SASL_MECH_EXTERNAL)
    return TRUE;

  return FALSE;
}

/*
 * Curl_sasl_start()
 *
 * Calculate the required login details for SASL authentication.
 */
CURLcode Curl_sasl_start(struct SASL *sasl, struct Curl_easy *data,
                         bool force_ir, saslprogress *progress)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  unsigned short enabledmechs;
  const char *mech = NULL;
  struct bufref resp;
  saslstate state1 = SASL_STOP;
  saslstate state2 = SASL_FINAL;
  const char * const hostname = SSL_HOST_NAME();
  const long int port = SSL_HOST_PORT();
#if defined(USE_KERBEROS5) || defined(USE_NTLM)
  const char *service = data->set.str[STRING_SERVICE_NAME] ?
    data->set.str[STRING_SERVICE_NAME] :
    sasl->params->service;
#endif
  const char *oauth_bearer = data->set.str[STRING_BEARER];
  struct bufref nullmsg;

  Curl_bufref_init(&nullmsg);
  Curl_bufref_init(&resp);
  sasl->force_ir = force_ir;    /* Latch for future use */
  sasl->authused = 0;           /* No mechanism used yet */
  enabledmechs = sasl->authmechs & sasl->prefmech;
  *progress = SASL_IDLE;

  /* Calculate the supported authentication mechanism, by decreasing order of
     security, as well as the initial response where appropriate */
  if((enabledmechs & SASL_MECH_EXTERNAL) && !conn->passwd[0]) {
    mech = SASL_MECH_STRING_EXTERNAL;
    state1 = SASL_EXTERNAL;
    sasl->authused = SASL_MECH_EXTERNAL;

    if(force_ir || data->set.sasl_ir)
      result = Curl_auth_create_external_message(conn->user, &resp);
  }
  else if(conn->bits.user_passwd) {
#if defined(USE_KERBEROS5)
    if((enabledmechs & SASL_MECH_GSSAPI) && Curl_auth_is_gssapi_supported() &&
       Curl_auth_user_contains_domain(conn->user)) {
      sasl->mutual_auth = FALSE;
      mech = SASL_MECH_STRING_GSSAPI;
      state1 = SASL_GSSAPI;
      state2 = SASL_GSSAPI_TOKEN;
      sasl->authused = SASL_MECH_GSSAPI;

      if(force_ir || data->set.sasl_ir)
        result = Curl_auth_create_gssapi_user_message(data, conn->user,
                                                      conn->passwd,
                                                      service,
                                                      conn->host.name,
                                                      sasl->mutual_auth,
                                                      NULL, &conn->krb5,
                                                      &resp);
    }
    else
#endif
#ifdef USE_GSASL
    if((enabledmechs & SASL_MECH_SCRAM_SHA_256) &&
       Curl_auth_gsasl_is_supported(data, SASL_MECH_STRING_SCRAM_SHA_256,
                                    &conn->gsasl)) {
      mech = SASL_MECH_STRING_SCRAM_SHA_256;
      sasl->authused = SASL_MECH_SCRAM_SHA_256;
      state1 = SASL_GSASL;
      state2 = SASL_GSASL;

      result = Curl_auth_gsasl_start(data, conn->user,
                                     conn->passwd, &conn->gsasl);
      if(result == CURLE_OK && (force_ir || data->set.sasl_ir))
        result = Curl_auth_gsasl_token(data, &nullmsg, &conn->gsasl, &resp);
    }
    else if((enabledmechs & SASL_MECH_SCRAM_SHA_1) &&
            Curl_auth_gsasl_is_supported(data, SASL_MECH_STRING_SCRAM_SHA_1,
                                         &conn->gsasl)) {
      mech = SASL_MECH_STRING_SCRAM_SHA_1;
      sasl->authused = SASL_MECH_SCRAM_SHA_1;
      state1 = SASL_GSASL;
      state2 = SASL_GSASL;

      result = Curl_auth_gsasl_start(data, conn->user,
                                     conn->passwd, &conn->gsasl);
      if(result == CURLE_OK && (force_ir || data->set.sasl_ir))
        result = Curl_auth_gsasl_token(data, &nullmsg, &conn->gsasl, &resp);
    }
    else
#endif
#ifndef CURL_DISABLE_CRYPTO_AUTH
    if((enabledmechs & SASL_MECH_DIGEST_MD5) &&
       Curl_auth_is_digest_supported()) {
      mech = SASL_MECH_STRING_DIGEST_MD5;
      state1 = SASL_DIGESTMD5;
      sasl->authused = SASL_MECH_DIGEST_MD5;
    }
    else if(enabledmechs & SASL_MECH_CRAM_MD5) {
      mech = SASL_MECH_STRING_CRAM_MD5;
      state1 = SASL_CRAMMD5;
      sasl->authused = SASL_MECH_CRAM_MD5;
    }
    else
#endif
#ifdef USE_NTLM
    if((enabledmechs & SASL_MECH_NTLM) && Curl_auth_is_ntlm_supported()) {
      mech = SASL_MECH_STRING_NTLM;
      state1 = SASL_NTLM;
      state2 = SASL_NTLM_TYPE2MSG;
      sasl->authused = SASL_MECH_NTLM;

      if(force_ir || data->set.sasl_ir)
        result = Curl_auth_create_ntlm_type1_message(data,
                                                     conn->user, conn->passwd,
                                                     service,
                                                     hostname,
                                                     &conn->ntlm, &resp);
      }
    else
#endif
    if((enabledmechs & SASL_MECH_OAUTHBEARER) && oauth_bearer) {
      mech = SASL_MECH_STRING_OAUTHBEARER;
      state1 = SASL_OAUTH2;
      state2 = SASL_OAUTH2_RESP;
      sasl->authused = SASL_MECH_OAUTHBEARER;

      if(force_ir || data->set.sasl_ir)
        result = Curl_auth_create_oauth_bearer_message(conn->user,
                                                       hostname,
                                                       port,
                                                       oauth_bearer,
                                                       &resp);
    }
    else if((enabledmechs & SASL_MECH_XOAUTH2) && oauth_bearer) {
      mech = SASL_MECH_STRING_XOAUTH2;
      state1 = SASL_OAUTH2;
      sasl->authused = SASL_MECH_XOAUTH2;

      if(force_ir || data->set.sasl_ir)
        result = Curl_auth_create_xoauth_bearer_message(conn->user,
                                                        oauth_bearer,
                                                        &resp);
    }
    else if(enabledmechs & SASL_MECH_PLAIN) {
      mech = SASL_MECH_STRING_PLAIN;
      state1 = SASL_PLAIN;
      sasl->authused = SASL_MECH_PLAIN;

      if(force_ir || data->set.sasl_ir)
        result = Curl_auth_create_plain_message(conn->sasl_authzid,
                                                conn->user, conn->passwd,
                                                &resp);
    }
    else if(enabledmechs & SASL_MECH_LOGIN) {
      mech = SASL_MECH_STRING_LOGIN;
      state1 = SASL_LOGIN;
      state2 = SASL_LOGIN_PASSWD;
      sasl->authused = SASL_MECH_LOGIN;

      if(force_ir || data->set.sasl_ir)
        result = Curl_auth_create_login_message(conn->user, &resp);
    }
  }

  if(!result && mech) {
    sasl->curmech = mech;
    if(Curl_bufref_ptr(&resp))
      result = build_message(sasl, data, &resp);

    if(sasl->params->maxirlen &&
       strlen(mech) + Curl_bufref_len(&resp) > sasl->params->maxirlen)
      Curl_bufref_free(&resp);

    if(!result)
      result = sasl->params->sendauth(data, mech, &resp);

    if(!result) {
      *progress = SASL_INPROGRESS;
      state(sasl, data, Curl_bufref_ptr(&resp) ? state2 : state1);
    }
  }

  Curl_bufref_free(&resp);
  return result;
}

/*
 * Curl_sasl_continue()
 *
 * Continue the authentication.
 */
CURLcode Curl_sasl_continue(struct SASL *sasl, struct Curl_easy *data,
                            int code, saslprogress *progress)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  saslstate newstate = SASL_FINAL;
  struct bufref resp;
  const char * const hostname = SSL_HOST_NAME();
  const long int port = SSL_HOST_PORT();
#if !defined(CURL_DISABLE_CRYPTO_AUTH) || defined(USE_KERBEROS5) ||     \
  defined(USE_NTLM)
  const char *service = data->set.str[STRING_SERVICE_NAME] ?
    data->set.str[STRING_SERVICE_NAME] :
    sasl->params->service;
#endif
  const char *oauth_bearer = data->set.str[STRING_BEARER];
  struct bufref serverdata;

  Curl_bufref_init(&serverdata);
  Curl_bufref_init(&resp);
  *progress = SASL_INPROGRESS;

  if(sasl->state == SASL_FINAL) {
    if(code != sasl->params->finalcode)
      result = CURLE_LOGIN_DENIED;
    *progress = SASL_DONE;
    state(sasl, data, SASL_STOP);
    return result;
  }

  if(sasl->state != SASL_CANCEL && sasl->state != SASL_OAUTH2_RESP &&
     code != sasl->params->contcode) {
    *progress = SASL_DONE;
    state(sasl, data, SASL_STOP);
    return CURLE_LOGIN_DENIED;
  }

  switch(sasl->state) {
  case SASL_STOP:
    *progress = SASL_DONE;
    return result;
  case SASL_PLAIN:
    result = Curl_auth_create_plain_message(conn->sasl_authzid,
                                            conn->user, conn->passwd, &resp);
    break;
  case SASL_LOGIN:
    result = Curl_auth_create_login_message(conn->user, &resp);
    newstate = SASL_LOGIN_PASSWD;
    break;
  case SASL_LOGIN_PASSWD:
    result = Curl_auth_create_login_message(conn->passwd, &resp);
    break;
  case SASL_EXTERNAL:
    result = Curl_auth_create_external_message(conn->user, &resp);
    break;
#ifndef CURL_DISABLE_CRYPTO_AUTH
#ifdef USE_GSASL
  case SASL_GSASL:
    result = get_server_message(sasl, data, &serverdata);
    if(!result)
      result = Curl_auth_gsasl_token(data, &serverdata, &conn->gsasl, &resp);
    if(!result && Curl_bufref_len(&resp) > 0)
      newstate = SASL_GSASL;
    break;
#endif
  case SASL_CRAMMD5:
    result = get_server_message(sasl, data, &serverdata);
    if(!result)
      result = Curl_auth_create_cram_md5_message(&serverdata, conn->user,
                                                 conn->passwd, &resp);
    break;
  case SASL_DIGESTMD5:
    result = get_server_message(sasl, data, &serverdata);
    if(!result)
      result = Curl_auth_create_digest_md5_message(data, &serverdata,
                                                   conn->user, conn->passwd,
                                                   service, &resp);
    if(!result && (sasl->params->flags & SASL_FLAG_BASE64))
      newstate = SASL_DIGESTMD5_RESP;
    break;
  case SASL_DIGESTMD5_RESP:
    /* Keep response NULL to output an empty line. */
    break;
#endif

#ifdef USE_NTLM
  case SASL_NTLM:
    /* Create the type-1 message */
    result = Curl_auth_create_ntlm_type1_message(data,
                                                 conn->user, conn->passwd,
                                                 service, hostname,
                                                 &conn->ntlm, &resp);
    newstate = SASL_NTLM_TYPE2MSG;
    break;
  case SASL_NTLM_TYPE2MSG:
    /* Decode the type-2 message */
    result = get_server_message(sasl, data, &serverdata);
    if(!result)
      result = Curl_auth_decode_ntlm_type2_message(data, &serverdata,
                                                   &conn->ntlm);
    if(!result)
      result = Curl_auth_create_ntlm_type3_message(data, conn->user,
                                                   conn->passwd, &conn->ntlm,
                                                   &resp);
    break;
#endif

#if defined(USE_KERBEROS5)
  case SASL_GSSAPI:
    result = Curl_auth_create_gssapi_user_message(data, conn->user,
                                                  conn->passwd,
                                                  service,
                                                  conn->host.name,
                                                  sasl->mutual_auth, NULL,
                                                  &conn->krb5,
                                                  &resp);
    newstate = SASL_GSSAPI_TOKEN;
    break;
  case SASL_GSSAPI_TOKEN:
    result = get_server_message(sasl, data, &serverdata);
    if(!result) {
      if(sasl->mutual_auth) {
        /* Decode the user token challenge and create the optional response
           message */
        result = Curl_auth_create_gssapi_user_message(data, NULL, NULL,
                                                      NULL, NULL,
                                                      sasl->mutual_auth,
                                                      &serverdata,
                                                      &conn->krb5,
                                                      &resp);
        newstate = SASL_GSSAPI_NO_DATA;
      }
      else
        /* Decode the security challenge and create the response message */
        result = Curl_auth_create_gssapi_security_message(data,
                                                          conn->sasl_authzid,
                                                          &serverdata,
                                                          &conn->krb5,
                                                          &resp);
    }
    break;
  case SASL_GSSAPI_NO_DATA:
    /* Decode the security challenge and create the response message */
    result = get_server_message(sasl, data, &serverdata);
    if(!result)
      result = Curl_auth_create_gssapi_security_message(data,
                                                        conn->sasl_authzid,
                                                        &serverdata,
                                                        &conn->krb5,
                                                        &resp);
    break;
#endif

  case SASL_OAUTH2:
    /* Create the authorisation message */
    if(sasl->authused == SASL_MECH_OAUTHBEARER) {
      result = Curl_auth_create_oauth_bearer_message(conn->user,
                                                     hostname,
                                                     port,
                                                     oauth_bearer,
                                                     &resp);

      /* Failures maybe sent by the server as continuations for OAUTHBEARER */
      newstate = SASL_OAUTH2_RESP;
    }
    else
      result = Curl_auth_create_xoauth_bearer_message(conn->user,
                                                      oauth_bearer,
                                                      &resp);
    break;

  case SASL_OAUTH2_RESP:
    /* The continuation is optional so check the response code */
    if(code == sasl->params->finalcode) {
      /* Final response was received so we are done */
      *progress = SASL_DONE;
      state(sasl, data, SASL_STOP);
      return result;
    }
    else if(code == sasl->params->contcode) {
      /* Acknowledge the continuation by sending a 0x01 response. */
      Curl_bufref_set(&resp, "\x01", 1, NULL);
      break;
    }
    else {
      *progress = SASL_DONE;
      state(sasl, data, SASL_STOP);
      return CURLE_LOGIN_DENIED;
    }

  case SASL_CANCEL:
    /* Remove the offending mechanism from the supported list */
    sasl->authmechs ^= sasl->authused;

    /* Start an alternative SASL authentication */
    return Curl_sasl_start(sasl, data, sasl->force_ir, progress);
  default:
    failf(data, "Unsupported SASL authentication mechanism");
    result = CURLE_UNSUPPORTED_PROTOCOL;  /* Should not happen */
    break;
  }

  Curl_bufref_free(&serverdata);

  switch(result) {
  case CURLE_BAD_CONTENT_ENCODING:
    /* Cancel dialog */
    result = sasl->params->cancelauth(data, sasl->curmech);
    newstate = SASL_CANCEL;
    break;
  case CURLE_OK:
    result = build_message(sasl, data, &resp);
    if(!result)
      result = sasl->params->contauth(data, sasl->curmech, &resp);
    break;
  default:
    newstate = SASL_STOP;    /* Stop on error */
    *progress = SASL_DONE;
    break;
  }

  Curl_bufref_free(&resp);

  state(sasl, data, newstate);

  return result;
}
#endif /* protocols are enabled that use SASL */
