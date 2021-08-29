/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2014 - 2021, Steve Holme, <steve_holme@hotmail.com>.
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
 * RFC4752 The Kerberos V5 ("GSSAPI") SASL Mechanism
 *
 ***************************************************************************/

#include "curl_setup.h"

#if defined(USE_WINDOWS_SSPI) && defined(USE_KERBEROS5)

#include <curl/curl.h>

#include "vauth/vauth.h"
#include "urldata.h"
#include "warnless.h"
#include "curl_multibyte.h"
#include "sendf.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/*
 * Curl_auth_is_gssapi_supported()
 *
 * This is used to evaluate if GSSAPI (Kerberos V5) is supported.
 *
 * Parameters: None
 *
 * Returns TRUE if Kerberos V5 is supported by Windows SSPI.
 */
bool Curl_auth_is_gssapi_supported(void)
{
  PSecPkgInfo SecurityPackage;
  SECURITY_STATUS status;

  /* Query the security package for Kerberos */
  status = s_pSecFn->QuerySecurityPackageInfo((TCHAR *)
                                              TEXT(SP_NAME_KERBEROS),
                                              &SecurityPackage);

  /* Release the package buffer as it is not required anymore */
  if(status == SEC_E_OK) {
    s_pSecFn->FreeContextBuffer(SecurityPackage);
  }

  return (status == SEC_E_OK ? TRUE : FALSE);
}

/*
 * Curl_auth_create_gssapi_user_message()
 *
 * This is used to generate an already encoded GSSAPI (Kerberos V5) user token
 * message ready for sending to the recipient.
 *
 * Parameters:
 *
 * data        [in]     - The session handle.
 * userp       [in]     - The user name in the format User or Domain\User.
 * passwdp     [in]     - The user's password.
 * service     [in]     - The service type such as http, smtp, pop or imap.
 * host        [in]     - The host name.
 * mutual_auth [in]     - Flag specifying whether or not mutual authentication
 *                        is enabled.
 * chlg        [in]     - Optional challenge message.
 * krb5        [in/out] - The Kerberos 5 data struct being used and modified.
 * out         [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_gssapi_user_message(struct Curl_easy *data,
                                              const char *userp,
                                              const char *passwdp,
                                              const char *service,
                                              const char *host,
                                              const bool mutual_auth,
                                              const struct bufref *chlg,
                                              struct kerberos5data *krb5,
                                              struct bufref *out)
{
  CURLcode result = CURLE_OK;
  CtxtHandle context;
  PSecPkgInfo SecurityPackage;
  SecBuffer chlg_buf;
  SecBuffer resp_buf;
  SecBufferDesc chlg_desc;
  SecBufferDesc resp_desc;
  SECURITY_STATUS status;
  unsigned long attrs;
  TimeStamp expiry; /* For Windows 9x compatibility of SSPI calls */

  if(!krb5->spn) {
    /* Generate our SPN */
    krb5->spn = Curl_auth_build_spn(service, host, NULL);
    if(!krb5->spn)
      return CURLE_OUT_OF_MEMORY;
  }

  if(!krb5->output_token) {
    /* Query the security package for Kerberos */
    status = s_pSecFn->QuerySecurityPackageInfo((TCHAR *)
                                                TEXT(SP_NAME_KERBEROS),
                                                &SecurityPackage);
    if(status != SEC_E_OK) {
      failf(data, "SSPI: couldn't get auth info");
      return CURLE_AUTH_ERROR;
    }

    krb5->token_max = SecurityPackage->cbMaxToken;

    /* Release the package buffer as it is not required anymore */
    s_pSecFn->FreeContextBuffer(SecurityPackage);

    /* Allocate our response buffer */
    krb5->output_token = malloc(krb5->token_max);
    if(!krb5->output_token)
      return CURLE_OUT_OF_MEMORY;
  }

  if(!krb5->credentials) {
    /* Do we have credentials to use or are we using single sign-on? */
    if(userp && *userp) {
      /* Populate our identity structure */
      result = Curl_create_sspi_identity(userp, passwdp, &krb5->identity);
      if(result)
        return result;

      /* Allow proper cleanup of the identity structure */
      krb5->p_identity = &krb5->identity;
    }
    else
      /* Use the current Windows user */
      krb5->p_identity = NULL;

    /* Allocate our credentials handle */
    krb5->credentials = calloc(1, sizeof(CredHandle));
    if(!krb5->credentials)
      return CURLE_OUT_OF_MEMORY;

    /* Acquire our credentials handle */
    status = s_pSecFn->AcquireCredentialsHandle(NULL,
                                                (TCHAR *)
                                                TEXT(SP_NAME_KERBEROS),
                                                SECPKG_CRED_OUTBOUND, NULL,
                                                krb5->p_identity, NULL, NULL,
                                                krb5->credentials, &expiry);
    if(status != SEC_E_OK)
      return CURLE_LOGIN_DENIED;

    /* Allocate our new context handle */
    krb5->context = calloc(1, sizeof(CtxtHandle));
    if(!krb5->context)
      return CURLE_OUT_OF_MEMORY;
  }

  if(chlg) {
    if(!Curl_bufref_len(chlg)) {
      infof(data, "GSSAPI handshake failure (empty challenge message)");
      return CURLE_BAD_CONTENT_ENCODING;
    }

    /* Setup the challenge "input" security buffer */
    chlg_desc.ulVersion = SECBUFFER_VERSION;
    chlg_desc.cBuffers  = 1;
    chlg_desc.pBuffers  = &chlg_buf;
    chlg_buf.BufferType = SECBUFFER_TOKEN;
    chlg_buf.pvBuffer   = (void *) Curl_bufref_ptr(chlg);
    chlg_buf.cbBuffer   = curlx_uztoul(Curl_bufref_len(chlg));
  }

  /* Setup the response "output" security buffer */
  resp_desc.ulVersion = SECBUFFER_VERSION;
  resp_desc.cBuffers  = 1;
  resp_desc.pBuffers  = &resp_buf;
  resp_buf.BufferType = SECBUFFER_TOKEN;
  resp_buf.pvBuffer   = krb5->output_token;
  resp_buf.cbBuffer   = curlx_uztoul(krb5->token_max);

  /* Generate our challenge-response message */
  status = s_pSecFn->InitializeSecurityContext(krb5->credentials,
                                               chlg ? krb5->context : NULL,
                                               krb5->spn,
                                               (mutual_auth ?
                                                ISC_REQ_MUTUAL_AUTH : 0),
                                               0, SECURITY_NATIVE_DREP,
                                               chlg ? &chlg_desc : NULL, 0,
                                               &context,
                                               &resp_desc, &attrs,
                                               &expiry);

  if(status == SEC_E_INSUFFICIENT_MEMORY)
    return CURLE_OUT_OF_MEMORY;

  if(status != SEC_E_OK && status != SEC_I_CONTINUE_NEEDED)
    return CURLE_AUTH_ERROR;

  if(memcmp(&context, krb5->context, sizeof(context))) {
    s_pSecFn->DeleteSecurityContext(krb5->context);

    memcpy(krb5->context, &context, sizeof(context));
  }

  if(resp_buf.cbBuffer) {
    result = Curl_bufref_memdup(out, resp_buf.pvBuffer, resp_buf.cbBuffer);
  }
  else if(mutual_auth)
    Curl_bufref_set(out, "", 0, NULL);
  else
    Curl_bufref_set(out, NULL, 0, NULL);

  return result;
}

/*
 * Curl_auth_create_gssapi_security_message()
 *
 * This is used to generate an already encoded GSSAPI (Kerberos V5) security
 * token message ready for sending to the recipient.
 *
 * Parameters:
 *
 * data    [in]     - The session handle.
 * authzid [in]     - The authorization identity if some.
 * chlg    [in]     - The optional challenge message.
 * krb5    [in/out] - The Kerberos 5 data struct being used and modified.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_gssapi_security_message(struct Curl_easy *data,
                                                  const char *authzid,
                                                  const struct bufref *chlg,
                                                  struct kerberos5data *krb5,
                                                  struct bufref *out)
{
  size_t offset = 0;
  size_t messagelen = 0;
  size_t appdatalen = 0;
  unsigned char *trailer = NULL;
  unsigned char *message = NULL;
  unsigned char *padding = NULL;
  unsigned char *appdata = NULL;
  SecBuffer input_buf[2];
  SecBuffer wrap_buf[3];
  SecBufferDesc input_desc;
  SecBufferDesc wrap_desc;
  unsigned char *indata;
  unsigned long qop = 0;
  unsigned long sec_layer = 0;
  unsigned long max_size = 0;
  SecPkgContext_Sizes sizes;
  SECURITY_STATUS status;

#if defined(CURL_DISABLE_VERBOSE_STRINGS)
  (void) data;
#endif

  /* Ensure we have a valid challenge message */
  if(!Curl_bufref_len(chlg)) {
    infof(data, "GSSAPI handshake failure (empty security message)");
    return CURLE_BAD_CONTENT_ENCODING;
  }

  /* Get our response size information */
  status = s_pSecFn->QueryContextAttributes(krb5->context,
                                            SECPKG_ATTR_SIZES,
                                            &sizes);

  if(status == SEC_E_INSUFFICIENT_MEMORY)
    return CURLE_OUT_OF_MEMORY;

  if(status != SEC_E_OK)
    return CURLE_AUTH_ERROR;

  /* Setup the "input" security buffer */
  input_desc.ulVersion = SECBUFFER_VERSION;
  input_desc.cBuffers = 2;
  input_desc.pBuffers = input_buf;
  input_buf[0].BufferType = SECBUFFER_STREAM;
  input_buf[0].pvBuffer = (void *) Curl_bufref_ptr(chlg);
  input_buf[0].cbBuffer = curlx_uztoul(Curl_bufref_len(chlg));
  input_buf[1].BufferType = SECBUFFER_DATA;
  input_buf[1].pvBuffer = NULL;
  input_buf[1].cbBuffer = 0;

  /* Decrypt the inbound challenge and obtain the qop */
  status = s_pSecFn->DecryptMessage(krb5->context, &input_desc, 0, &qop);
  if(status != SEC_E_OK) {
    infof(data, "GSSAPI handshake failure (empty security message)");
    return CURLE_BAD_CONTENT_ENCODING;
  }

  /* Not 4 octets long so fail as per RFC4752 Section 3.1 */
  if(input_buf[1].cbBuffer != 4) {
    infof(data, "GSSAPI handshake failure (invalid security data)");
    return CURLE_BAD_CONTENT_ENCODING;
  }

  /* Extract the security layer and the maximum message size */
  indata = input_buf[1].pvBuffer;
  sec_layer = indata[0];
  max_size = (indata[1] << 16) | (indata[2] << 8) | indata[3];

  /* Free the challenge as it is not required anymore */
  s_pSecFn->FreeContextBuffer(input_buf[1].pvBuffer);

  /* Process the security layer */
  if(!(sec_layer & KERB_WRAP_NO_ENCRYPT)) {
    infof(data, "GSSAPI handshake failure (invalid security layer)");
    return CURLE_BAD_CONTENT_ENCODING;
  }
  sec_layer &= KERB_WRAP_NO_ENCRYPT;  /* We do not support a security layer */

  /* Process the maximum message size the server can receive */
  if(max_size > 0) {
    /* The server has told us it supports a maximum receive buffer, however, as
       we don't require one unless we are encrypting data, we tell the server
       our receive buffer is zero. */
    max_size = 0;
  }

  /* Allocate the trailer */
  trailer = malloc(sizes.cbSecurityTrailer);
  if(!trailer)
    return CURLE_OUT_OF_MEMORY;

  /* Allocate our message */
  messagelen = 4;
  if(authzid)
    messagelen += strlen(authzid);
  message = malloc(messagelen);
  if(!message) {
    free(trailer);

    return CURLE_OUT_OF_MEMORY;
  }

  /* Populate the message with the security layer and client supported receive
     message size. */
  message[0] = sec_layer & 0xFF;
  message[1] = (max_size >> 16) & 0xFF;
  message[2] = (max_size >> 8) & 0xFF;
  message[3] = max_size & 0xFF;

  /* If given, append the authorization identity. */

  if(authzid && *authzid)
    memcpy(message + 4, authzid, messagelen - 4);

  /* Allocate the padding */
  padding = malloc(sizes.cbBlockSize);
  if(!padding) {
    free(message);
    free(trailer);

    return CURLE_OUT_OF_MEMORY;
  }

  /* Setup the "authentication data" security buffer */
  wrap_desc.ulVersion    = SECBUFFER_VERSION;
  wrap_desc.cBuffers     = 3;
  wrap_desc.pBuffers     = wrap_buf;
  wrap_buf[0].BufferType = SECBUFFER_TOKEN;
  wrap_buf[0].pvBuffer   = trailer;
  wrap_buf[0].cbBuffer   = sizes.cbSecurityTrailer;
  wrap_buf[1].BufferType = SECBUFFER_DATA;
  wrap_buf[1].pvBuffer   = message;
  wrap_buf[1].cbBuffer   = curlx_uztoul(messagelen);
  wrap_buf[2].BufferType = SECBUFFER_PADDING;
  wrap_buf[2].pvBuffer   = padding;
  wrap_buf[2].cbBuffer   = sizes.cbBlockSize;

  /* Encrypt the data */
  status = s_pSecFn->EncryptMessage(krb5->context, KERB_WRAP_NO_ENCRYPT,
                                    &wrap_desc, 0);
  if(status != SEC_E_OK) {
    free(padding);
    free(message);
    free(trailer);

    if(status == SEC_E_INSUFFICIENT_MEMORY)
      return CURLE_OUT_OF_MEMORY;

    return CURLE_AUTH_ERROR;
  }

  /* Allocate the encryption (wrap) buffer */
  appdatalen = wrap_buf[0].cbBuffer + wrap_buf[1].cbBuffer +
               wrap_buf[2].cbBuffer;
  appdata = malloc(appdatalen);
  if(!appdata) {
    free(padding);
    free(message);
    free(trailer);

    return CURLE_OUT_OF_MEMORY;
  }

  /* Populate the encryption buffer */
  memcpy(appdata, wrap_buf[0].pvBuffer, wrap_buf[0].cbBuffer);
  offset += wrap_buf[0].cbBuffer;
  memcpy(appdata + offset, wrap_buf[1].pvBuffer, wrap_buf[1].cbBuffer);
  offset += wrap_buf[1].cbBuffer;
  memcpy(appdata + offset, wrap_buf[2].pvBuffer, wrap_buf[2].cbBuffer);

  /* Free all of our local buffers */
  free(padding);
  free(message);
  free(trailer);

  /* Return the response. */
  Curl_bufref_set(out, appdata, appdatalen, curl_free);
  return CURLE_OK;
}

/*
 * Curl_auth_cleanup_gssapi()
 *
 * This is used to clean up the GSSAPI (Kerberos V5) specific data.
 *
 * Parameters:
 *
 * krb5     [in/out] - The Kerberos 5 data struct being cleaned up.
 *
 */
void Curl_auth_cleanup_gssapi(struct kerberos5data *krb5)
{
  /* Free our security context */
  if(krb5->context) {
    s_pSecFn->DeleteSecurityContext(krb5->context);
    free(krb5->context);
    krb5->context = NULL;
  }

  /* Free our credentials handle */
  if(krb5->credentials) {
    s_pSecFn->FreeCredentialsHandle(krb5->credentials);
    free(krb5->credentials);
    krb5->credentials = NULL;
  }

  /* Free our identity */
  Curl_sspi_free_identity(krb5->p_identity);
  krb5->p_identity = NULL;

  /* Free the SPN and output token */
  Curl_safefree(krb5->spn);
  Curl_safefree(krb5->output_token);

  /* Reset any variables */
  krb5->token_max = 0;
}

#endif /* USE_WINDOWS_SSPI && USE_KERBEROS5*/
