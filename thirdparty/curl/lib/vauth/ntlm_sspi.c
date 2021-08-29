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

#if defined(USE_WINDOWS_SSPI) && defined(USE_NTLM)

#include <curl/curl.h>

#include "vauth/vauth.h"
#include "urldata.h"
#include "curl_ntlm_core.h"
#include "warnless.h"
#include "curl_multibyte.h"
#include "sendf.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/*
 * Curl_auth_is_ntlm_supported()
 *
 * This is used to evaluate if NTLM is supported.
 *
 * Parameters: None
 *
 * Returns TRUE if NTLM is supported by Windows SSPI.
 */
bool Curl_auth_is_ntlm_supported(void)
{
  PSecPkgInfo SecurityPackage;
  SECURITY_STATUS status;

  /* Query the security package for NTLM */
  status = s_pSecFn->QuerySecurityPackageInfo((TCHAR *) TEXT(SP_NAME_NTLM),
                                              &SecurityPackage);

  /* Release the package buffer as it is not required anymore */
  if(status == SEC_E_OK) {
    s_pSecFn->FreeContextBuffer(SecurityPackage);
  }

  return (status == SEC_E_OK ? TRUE : FALSE);
}

/*
 * Curl_auth_create_ntlm_type1_message()
 *
 * This is used to generate an already encoded NTLM type-1 message ready for
 * sending to the recipient.
 *
 * Parameters:
 *
 * data    [in]     - The session handle.
 * userp   [in]     - The user name in the format User or Domain\User.
 * passwdp [in]     - The user's password.
 * service [in]     - The service type such as http, smtp, pop or imap.
 * host    [in]     - The host name.
 * ntlm    [in/out] - The NTLM data struct being used and modified.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_ntlm_type1_message(struct Curl_easy *data,
                                             const char *userp,
                                             const char *passwdp,
                                             const char *service,
                                             const char *host,
                                             struct ntlmdata *ntlm,
                                             struct bufref *out)
{
  PSecPkgInfo SecurityPackage;
  SecBuffer type_1_buf;
  SecBufferDesc type_1_desc;
  SECURITY_STATUS status;
  unsigned long attrs;
  TimeStamp expiry; /* For Windows 9x compatibility of SSPI calls */

  /* Clean up any former leftovers and initialise to defaults */
  Curl_auth_cleanup_ntlm(ntlm);

  /* Query the security package for NTLM */
  status = s_pSecFn->QuerySecurityPackageInfo((TCHAR *) TEXT(SP_NAME_NTLM),
                                              &SecurityPackage);
  if(status != SEC_E_OK) {
    failf(data, "SSPI: couldn't get auth info");
    return CURLE_AUTH_ERROR;
  }

  ntlm->token_max = SecurityPackage->cbMaxToken;

  /* Release the package buffer as it is not required anymore */
  s_pSecFn->FreeContextBuffer(SecurityPackage);

  /* Allocate our output buffer */
  ntlm->output_token = malloc(ntlm->token_max);
  if(!ntlm->output_token)
    return CURLE_OUT_OF_MEMORY;

  if(userp && *userp) {
    CURLcode result;

    /* Populate our identity structure */
    result = Curl_create_sspi_identity(userp, passwdp, &ntlm->identity);
    if(result)
      return result;

    /* Allow proper cleanup of the identity structure */
    ntlm->p_identity = &ntlm->identity;
  }
  else
    /* Use the current Windows user */
    ntlm->p_identity = NULL;

  /* Allocate our credentials handle */
  ntlm->credentials = calloc(1, sizeof(CredHandle));
  if(!ntlm->credentials)
    return CURLE_OUT_OF_MEMORY;

  /* Acquire our credentials handle */
  status = s_pSecFn->AcquireCredentialsHandle(NULL,
                                              (TCHAR *) TEXT(SP_NAME_NTLM),
                                              SECPKG_CRED_OUTBOUND, NULL,
                                              ntlm->p_identity, NULL, NULL,
                                              ntlm->credentials, &expiry);
  if(status != SEC_E_OK)
    return CURLE_LOGIN_DENIED;

  /* Allocate our new context handle */
  ntlm->context = calloc(1, sizeof(CtxtHandle));
  if(!ntlm->context)
    return CURLE_OUT_OF_MEMORY;

  ntlm->spn = Curl_auth_build_spn(service, host, NULL);
  if(!ntlm->spn)
    return CURLE_OUT_OF_MEMORY;

  /* Setup the type-1 "output" security buffer */
  type_1_desc.ulVersion = SECBUFFER_VERSION;
  type_1_desc.cBuffers  = 1;
  type_1_desc.pBuffers  = &type_1_buf;
  type_1_buf.BufferType = SECBUFFER_TOKEN;
  type_1_buf.pvBuffer   = ntlm->output_token;
  type_1_buf.cbBuffer   = curlx_uztoul(ntlm->token_max);

  /* Generate our type-1 message */
  status = s_pSecFn->InitializeSecurityContext(ntlm->credentials, NULL,
                                               ntlm->spn,
                                               0, 0, SECURITY_NETWORK_DREP,
                                               NULL, 0,
                                               ntlm->context, &type_1_desc,
                                               &attrs, &expiry);
  if(status == SEC_I_COMPLETE_NEEDED ||
    status == SEC_I_COMPLETE_AND_CONTINUE)
    s_pSecFn->CompleteAuthToken(ntlm->context, &type_1_desc);
  else if(status == SEC_E_INSUFFICIENT_MEMORY)
    return CURLE_OUT_OF_MEMORY;
  else if(status != SEC_E_OK && status != SEC_I_CONTINUE_NEEDED)
    return CURLE_AUTH_ERROR;

  /* Return the response. */
  Curl_bufref_set(out, ntlm->output_token, type_1_buf.cbBuffer, NULL);
  return CURLE_OK;
}

/*
 * Curl_auth_decode_ntlm_type2_message()
 *
 * This is used to decode an already encoded NTLM type-2 message.
 *
 * Parameters:
 *
 * data     [in]     - The session handle.
 * type2    [in]     - The type-2 message.
 * ntlm     [in/out] - The NTLM data struct being used and modified.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_decode_ntlm_type2_message(struct Curl_easy *data,
                                             const struct bufref *type2,
                                             struct ntlmdata *ntlm)
{
#if defined(CURL_DISABLE_VERBOSE_STRINGS)
  (void) data;
#endif

  /* Ensure we have a valid type-2 message */
  if(!Curl_bufref_len(type2)) {
    infof(data, "NTLM handshake failure (empty type-2 message)");
    return CURLE_BAD_CONTENT_ENCODING;
  }

  /* Store the challenge for later use */
  ntlm->input_token = malloc(Curl_bufref_len(type2) + 1);
  if(!ntlm->input_token)
    return CURLE_OUT_OF_MEMORY;
  memcpy(ntlm->input_token, Curl_bufref_ptr(type2), Curl_bufref_len(type2));
  ntlm->input_token[Curl_bufref_len(type2)] = '\0';
  ntlm->input_token_len = Curl_bufref_len(type2);

  return CURLE_OK;
}

/*
* Curl_auth_create_ntlm_type3_message()
 * Curl_auth_create_ntlm_type3_message()
 *
 * This is used to generate an already encoded NTLM type-3 message ready for
 * sending to the recipient.
 *
 * Parameters:
 *
 * data    [in]     - The session handle.
 * userp   [in]     - The user name in the format User or Domain\User.
 * passwdp [in]     - The user's password.
 * ntlm    [in/out] - The NTLM data struct being used and modified.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_ntlm_type3_message(struct Curl_easy *data,
                                             const char *userp,
                                             const char *passwdp,
                                             struct ntlmdata *ntlm,
                                             struct bufref *out)
{
  CURLcode result = CURLE_OK;
  SecBuffer type_2_bufs[2];
  SecBuffer type_3_buf;
  SecBufferDesc type_2_desc;
  SecBufferDesc type_3_desc;
  SECURITY_STATUS status;
  unsigned long attrs;
  TimeStamp expiry; /* For Windows 9x compatibility of SSPI calls */

#if defined(CURL_DISABLE_VERBOSE_STRINGS)
  (void) data;
#endif
  (void) passwdp;
  (void) userp;

  /* Setup the type-2 "input" security buffer */
  type_2_desc.ulVersion     = SECBUFFER_VERSION;
  type_2_desc.cBuffers      = 1;
  type_2_desc.pBuffers      = &type_2_bufs[0];
  type_2_bufs[0].BufferType = SECBUFFER_TOKEN;
  type_2_bufs[0].pvBuffer   = ntlm->input_token;
  type_2_bufs[0].cbBuffer   = curlx_uztoul(ntlm->input_token_len);

#ifdef SECPKG_ATTR_ENDPOINT_BINDINGS
  /* ssl context comes from schannel.
  * When extended protection is used in IIS server,
  * we have to pass a second SecBuffer to the SecBufferDesc
  * otherwise IIS will not pass the authentication (401 response).
  * Minimum supported version is Windows 7.
  * https://docs.microsoft.com/en-us/security-updates
  * /SecurityAdvisories/2009/973811
  */
  if(ntlm->sslContext) {
    SEC_CHANNEL_BINDINGS channelBindings;
    SecPkgContext_Bindings pkgBindings;
    pkgBindings.Bindings = &channelBindings;
    status = s_pSecFn->QueryContextAttributes(
      ntlm->sslContext,
      SECPKG_ATTR_ENDPOINT_BINDINGS,
      &pkgBindings
    );
    if(status == SEC_E_OK) {
      type_2_desc.cBuffers++;
      type_2_bufs[1].BufferType = SECBUFFER_CHANNEL_BINDINGS;
      type_2_bufs[1].cbBuffer = pkgBindings.BindingsLength;
      type_2_bufs[1].pvBuffer = pkgBindings.Bindings;
    }
  }
#endif

  /* Setup the type-3 "output" security buffer */
  type_3_desc.ulVersion = SECBUFFER_VERSION;
  type_3_desc.cBuffers  = 1;
  type_3_desc.pBuffers  = &type_3_buf;
  type_3_buf.BufferType = SECBUFFER_TOKEN;
  type_3_buf.pvBuffer   = ntlm->output_token;
  type_3_buf.cbBuffer   = curlx_uztoul(ntlm->token_max);

  /* Generate our type-3 message */
  status = s_pSecFn->InitializeSecurityContext(ntlm->credentials,
                                               ntlm->context,
                                               ntlm->spn,
                                               0, 0, SECURITY_NETWORK_DREP,
                                               &type_2_desc,
                                               0, ntlm->context,
                                               &type_3_desc,
                                               &attrs, &expiry);
  if(status != SEC_E_OK) {
    infof(data, "NTLM handshake failure (type-3 message): Status=%x",
          status);

    if(status == SEC_E_INSUFFICIENT_MEMORY)
      return CURLE_OUT_OF_MEMORY;

    return CURLE_AUTH_ERROR;
  }

  /* Return the response. */
  result = Curl_bufref_memdup(out, ntlm->output_token, type_3_buf.cbBuffer);
  Curl_auth_cleanup_ntlm(ntlm);
  return result;
}

/*
 * Curl_auth_cleanup_ntlm()
 *
 * This is used to clean up the NTLM specific data.
 *
 * Parameters:
 *
 * ntlm    [in/out] - The NTLM data struct being cleaned up.
 *
 */
void Curl_auth_cleanup_ntlm(struct ntlmdata *ntlm)
{
  /* Free our security context */
  if(ntlm->context) {
    s_pSecFn->DeleteSecurityContext(ntlm->context);
    free(ntlm->context);
    ntlm->context = NULL;
  }

  /* Free our credentials handle */
  if(ntlm->credentials) {
    s_pSecFn->FreeCredentialsHandle(ntlm->credentials);
    free(ntlm->credentials);
    ntlm->credentials = NULL;
  }

  /* Free our identity */
  Curl_sspi_free_identity(ntlm->p_identity);
  ntlm->p_identity = NULL;

  /* Free the input and output tokens */
  Curl_safefree(ntlm->input_token);
  Curl_safefree(ntlm->output_token);

  /* Reset any variables */
  ntlm->token_max = 0;

  Curl_safefree(ntlm->spn);
}

#endif /* USE_WINDOWS_SSPI && USE_NTLM */
