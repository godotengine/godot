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
 * RFC4178 Simple and Protected GSS-API Negotiation Mechanism
 *
 ***************************************************************************/

#include "curl_setup.h"

#if defined(USE_WINDOWS_SSPI) && defined(USE_SPNEGO)

#include <curl/curl.h>

#include "vauth/vauth.h"
#include "urldata.h"
#include "curl_base64.h"
#include "warnless.h"
#include "curl_multibyte.h"
#include "sendf.h"
#include "strerror.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/*
 * Curl_auth_is_spnego_supported()
 *
 * This is used to evaluate if SPNEGO (Negotiate) is supported.
 *
 * Parameters: None
 *
 * Returns TRUE if Negotiate is supported by Windows SSPI.
 */
bool Curl_auth_is_spnego_supported(void)
{
  PSecPkgInfo SecurityPackage;
  SECURITY_STATUS status;

  /* Query the security package for Negotiate */
  status = s_pSecFn->QuerySecurityPackageInfo((TCHAR *)
                                              TEXT(SP_NAME_NEGOTIATE),
                                              &SecurityPackage);

  /* Release the package buffer as it is not required anymore */
  if(status == SEC_E_OK) {
    s_pSecFn->FreeContextBuffer(SecurityPackage);
  }


  return (status == SEC_E_OK ? TRUE : FALSE);
}

/*
 * Curl_auth_decode_spnego_message()
 *
 * This is used to decode an already encoded SPNEGO (Negotiate) challenge
 * message.
 *
 * Parameters:
 *
 * data        [in]     - The session handle.
 * user        [in]     - The user name in the format User or Domain\User.
 * password    [in]     - The user's password.
 * service     [in]     - The service type such as http, smtp, pop or imap.
 * host        [in]     - The host name.
 * chlg64      [in]     - The optional base64 encoded challenge message.
 * nego        [in/out] - The Negotiate data struct being used and modified.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_decode_spnego_message(struct Curl_easy *data,
                                         const char *user,
                                         const char *password,
                                         const char *service,
                                         const char *host,
                                         const char *chlg64,
                                         struct negotiatedata *nego)
{
  CURLcode result = CURLE_OK;
  size_t chlglen = 0;
  unsigned char *chlg = NULL;
  PSecPkgInfo SecurityPackage;
  SecBuffer chlg_buf[2];
  SecBuffer resp_buf;
  SecBufferDesc chlg_desc;
  SecBufferDesc resp_desc;
  unsigned long attrs;
  TimeStamp expiry; /* For Windows 9x compatibility of SSPI calls */

#if defined(CURL_DISABLE_VERBOSE_STRINGS)
  (void) data;
#endif

  if(nego->context && nego->status == SEC_E_OK) {
    /* We finished successfully our part of authentication, but server
     * rejected it (since we're again here). Exit with an error since we
     * can't invent anything better */
    Curl_auth_cleanup_spnego(nego);
    return CURLE_LOGIN_DENIED;
  }

  if(!nego->spn) {
    /* Generate our SPN */
    nego->spn = Curl_auth_build_spn(service, host, NULL);
    if(!nego->spn)
      return CURLE_OUT_OF_MEMORY;
  }

  if(!nego->output_token) {
    /* Query the security package for Negotiate */
    nego->status = s_pSecFn->QuerySecurityPackageInfo((TCHAR *)
                                                      TEXT(SP_NAME_NEGOTIATE),
                                                      &SecurityPackage);
    if(nego->status != SEC_E_OK) {
      failf(data, "SSPI: couldn't get auth info");
      return CURLE_AUTH_ERROR;
    }

    nego->token_max = SecurityPackage->cbMaxToken;

    /* Release the package buffer as it is not required anymore */
    s_pSecFn->FreeContextBuffer(SecurityPackage);

    /* Allocate our output buffer */
    nego->output_token = malloc(nego->token_max);
    if(!nego->output_token)
      return CURLE_OUT_OF_MEMORY;
 }

  if(!nego->credentials) {
    /* Do we have credentials to use or are we using single sign-on? */
    if(user && *user) {
      /* Populate our identity structure */
      result = Curl_create_sspi_identity(user, password, &nego->identity);
      if(result)
        return result;

      /* Allow proper cleanup of the identity structure */
      nego->p_identity = &nego->identity;
    }
    else
      /* Use the current Windows user */
      nego->p_identity = NULL;

    /* Allocate our credentials handle */
    nego->credentials = calloc(1, sizeof(CredHandle));
    if(!nego->credentials)
      return CURLE_OUT_OF_MEMORY;

    /* Acquire our credentials handle */
    nego->status =
      s_pSecFn->AcquireCredentialsHandle(NULL,
                                         (TCHAR *)TEXT(SP_NAME_NEGOTIATE),
                                         SECPKG_CRED_OUTBOUND, NULL,
                                         nego->p_identity, NULL, NULL,
                                         nego->credentials, &expiry);
    if(nego->status != SEC_E_OK)
      return CURLE_AUTH_ERROR;

    /* Allocate our new context handle */
    nego->context = calloc(1, sizeof(CtxtHandle));
    if(!nego->context)
      return CURLE_OUT_OF_MEMORY;
  }

  if(chlg64 && *chlg64) {
    /* Decode the base-64 encoded challenge message */
    if(*chlg64 != '=') {
      result = Curl_base64_decode(chlg64, &chlg, &chlglen);
      if(result)
        return result;
    }

    /* Ensure we have a valid challenge message */
    if(!chlg) {
      infof(data, "SPNEGO handshake failure (empty challenge message)");
      return CURLE_BAD_CONTENT_ENCODING;
    }

    /* Setup the challenge "input" security buffer */
    chlg_desc.ulVersion    = SECBUFFER_VERSION;
    chlg_desc.cBuffers     = 1;
    chlg_desc.pBuffers     = &chlg_buf[0];
    chlg_buf[0].BufferType = SECBUFFER_TOKEN;
    chlg_buf[0].pvBuffer   = chlg;
    chlg_buf[0].cbBuffer   = curlx_uztoul(chlglen);

#ifdef SECPKG_ATTR_ENDPOINT_BINDINGS
    /* ssl context comes from Schannel.
    * When extended protection is used in IIS server,
    * we have to pass a second SecBuffer to the SecBufferDesc
    * otherwise IIS will not pass the authentication (401 response).
    * Minimum supported version is Windows 7.
    * https://docs.microsoft.com/en-us/security-updates
    * /SecurityAdvisories/2009/973811
    */
    if(nego->sslContext) {
      SEC_CHANNEL_BINDINGS channelBindings;
      SecPkgContext_Bindings pkgBindings;
      pkgBindings.Bindings = &channelBindings;
      nego->status = s_pSecFn->QueryContextAttributes(
          nego->sslContext,
          SECPKG_ATTR_ENDPOINT_BINDINGS,
          &pkgBindings
      );
      if(nego->status == SEC_E_OK) {
        chlg_desc.cBuffers++;
        chlg_buf[1].BufferType = SECBUFFER_CHANNEL_BINDINGS;
        chlg_buf[1].cbBuffer   = pkgBindings.BindingsLength;
        chlg_buf[1].pvBuffer   = pkgBindings.Bindings;
      }
    }
#endif
  }

  /* Setup the response "output" security buffer */
  resp_desc.ulVersion = SECBUFFER_VERSION;
  resp_desc.cBuffers  = 1;
  resp_desc.pBuffers  = &resp_buf;
  resp_buf.BufferType = SECBUFFER_TOKEN;
  resp_buf.pvBuffer   = nego->output_token;
  resp_buf.cbBuffer   = curlx_uztoul(nego->token_max);

  /* Generate our challenge-response message */
  nego->status = s_pSecFn->InitializeSecurityContext(nego->credentials,
                                                     chlg ? nego->context :
                                                            NULL,
                                                     nego->spn,
                                                     ISC_REQ_CONFIDENTIALITY,
                                                     0, SECURITY_NATIVE_DREP,
                                                     chlg ? &chlg_desc : NULL,
                                                     0, nego->context,
                                                     &resp_desc, &attrs,
                                                     &expiry);

  /* Free the decoded challenge as it is not required anymore */
  free(chlg);

  if(GSS_ERROR(nego->status)) {
    char buffer[STRERROR_LEN];
    failf(data, "InitializeSecurityContext failed: %s",
          Curl_sspi_strerror(nego->status, buffer, sizeof(buffer)));

    if(nego->status == (DWORD)SEC_E_INSUFFICIENT_MEMORY)
      return CURLE_OUT_OF_MEMORY;

    return CURLE_AUTH_ERROR;
  }

  if(nego->status == SEC_I_COMPLETE_NEEDED ||
     nego->status == SEC_I_COMPLETE_AND_CONTINUE) {
    nego->status = s_pSecFn->CompleteAuthToken(nego->context, &resp_desc);
    if(GSS_ERROR(nego->status)) {
      char buffer[STRERROR_LEN];
      failf(data, "CompleteAuthToken failed: %s",
            Curl_sspi_strerror(nego->status, buffer, sizeof(buffer)));

      if(nego->status == (DWORD)SEC_E_INSUFFICIENT_MEMORY)
        return CURLE_OUT_OF_MEMORY;

      return CURLE_AUTH_ERROR;
    }
  }

  nego->output_token_length = resp_buf.cbBuffer;

  return result;
}

/*
 * Curl_auth_create_spnego_message()
 *
 * This is used to generate an already encoded SPNEGO (Negotiate) response
 * message ready for sending to the recipient.
 *
 * Parameters:
 *
 * data        [in]     - The session handle.
 * nego        [in/out] - The Negotiate data struct being used and modified.
 * outptr      [in/out] - The address where a pointer to newly allocated memory
 *                        holding the result will be stored upon completion.
 * outlen      [out]    - The length of the output message.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_spnego_message(struct Curl_easy *data,
                                         struct negotiatedata *nego,
                                         char **outptr, size_t *outlen)
{
  CURLcode result;

  /* Base64 encode the already generated response */
  result = Curl_base64_encode(data,
                              (const char *) nego->output_token,
                              nego->output_token_length,
                              outptr, outlen);

  if(result)
    return result;

  if(!*outptr || !*outlen) {
    free(*outptr);
    return CURLE_REMOTE_ACCESS_DENIED;
  }

  return CURLE_OK;
}

/*
 * Curl_auth_cleanup_spnego()
 *
 * This is used to clean up the SPNEGO (Negotiate) specific data.
 *
 * Parameters:
 *
 * nego     [in/out] - The Negotiate data struct being cleaned up.
 *
 */
void Curl_auth_cleanup_spnego(struct negotiatedata *nego)
{
  /* Free our security context */
  if(nego->context) {
    s_pSecFn->DeleteSecurityContext(nego->context);
    free(nego->context);
    nego->context = NULL;
  }

  /* Free our credentials handle */
  if(nego->credentials) {
    s_pSecFn->FreeCredentialsHandle(nego->credentials);
    free(nego->credentials);
    nego->credentials = NULL;
  }

  /* Free our identity */
  Curl_sspi_free_identity(nego->p_identity);
  nego->p_identity = NULL;

  /* Free the SPN and output token */
  Curl_safefree(nego->spn);
  Curl_safefree(nego->output_token);

  /* Reset any variables */
  nego->status = 0;
  nego->token_max = 0;
  nego->noauthpersist = FALSE;
  nego->havenoauthpersist = FALSE;
  nego->havenegdata = FALSE;
  nego->havemultiplerequests = FALSE;
}

#endif /* USE_WINDOWS_SSPI && USE_SPNEGO */
