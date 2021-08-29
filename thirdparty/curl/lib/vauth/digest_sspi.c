/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2014 - 2016, Steve Holme, <steve_holme@hotmail.com>.
 * Copyright (C) 2015 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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
 * RFC2831 DIGEST-MD5 authentication
 *
 ***************************************************************************/

#include "curl_setup.h"

#if defined(USE_WINDOWS_SSPI) && !defined(CURL_DISABLE_CRYPTO_AUTH)

#include <curl/curl.h>

#include "vauth/vauth.h"
#include "vauth/digest.h"
#include "urldata.h"
#include "warnless.h"
#include "curl_multibyte.h"
#include "sendf.h"
#include "strdup.h"
#include "strcase.h"
#include "strerror.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/*
* Curl_auth_is_digest_supported()
*
* This is used to evaluate if DIGEST is supported.
*
* Parameters: None
*
* Returns TRUE if DIGEST is supported by Windows SSPI.
*/
bool Curl_auth_is_digest_supported(void)
{
  PSecPkgInfo SecurityPackage;
  SECURITY_STATUS status;

  /* Query the security package for Digest */
  status = s_pSecFn->QuerySecurityPackageInfo((TCHAR *) TEXT(SP_NAME_DIGEST),
                                              &SecurityPackage);

  /* Release the package buffer as it is not required anymore */
  if(status == SEC_E_OK) {
    s_pSecFn->FreeContextBuffer(SecurityPackage);
  }

  return (status == SEC_E_OK ? TRUE : FALSE);
}

/*
 * Curl_auth_create_digest_md5_message()
 *
 * This is used to generate an already encoded DIGEST-MD5 response message
 * ready for sending to the recipient.
 *
 * Parameters:
 *
 * data    [in]     - The session handle.
 * chlg    [in]     - The challenge message.
 * userp   [in]     - The user name in the format User or Domain\User.
 * passwdp [in]     - The user's password.
 * service [in]     - The service type such as http, smtp, pop or imap.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_digest_md5_message(struct Curl_easy *data,
                                             const struct bufref *chlg,
                                             const char *userp,
                                             const char *passwdp,
                                             const char *service,
                                             struct bufref *out)
{
  CURLcode result = CURLE_OK;
  TCHAR *spn = NULL;
  size_t token_max = 0;
  unsigned char *output_token = NULL;
  CredHandle credentials;
  CtxtHandle context;
  PSecPkgInfo SecurityPackage;
  SEC_WINNT_AUTH_IDENTITY identity;
  SEC_WINNT_AUTH_IDENTITY *p_identity;
  SecBuffer chlg_buf;
  SecBuffer resp_buf;
  SecBufferDesc chlg_desc;
  SecBufferDesc resp_desc;
  SECURITY_STATUS status;
  unsigned long attrs;
  TimeStamp expiry; /* For Windows 9x compatibility of SSPI calls */

  /* Ensure we have a valid challenge message */
  if(!Curl_bufref_len(chlg)) {
    infof(data, "DIGEST-MD5 handshake failure (empty challenge message)");
    return CURLE_BAD_CONTENT_ENCODING;
  }

  /* Query the security package for DigestSSP */
  status = s_pSecFn->QuerySecurityPackageInfo((TCHAR *) TEXT(SP_NAME_DIGEST),
                                              &SecurityPackage);
  if(status != SEC_E_OK) {
    failf(data, "SSPI: couldn't get auth info");
    return CURLE_AUTH_ERROR;
  }

  token_max = SecurityPackage->cbMaxToken;

  /* Release the package buffer as it is not required anymore */
  s_pSecFn->FreeContextBuffer(SecurityPackage);

  /* Allocate our response buffer */
  output_token = malloc(token_max);
  if(!output_token)
    return CURLE_OUT_OF_MEMORY;

  /* Generate our SPN */
  spn = Curl_auth_build_spn(service, data->conn->host.name, NULL);
  if(!spn) {
    free(output_token);
    return CURLE_OUT_OF_MEMORY;
  }

  if(userp && *userp) {
    /* Populate our identity structure */
    result = Curl_create_sspi_identity(userp, passwdp, &identity);
    if(result) {
      free(spn);
      free(output_token);
      return result;
    }

    /* Allow proper cleanup of the identity structure */
    p_identity = &identity;
  }
  else
    /* Use the current Windows user */
    p_identity = NULL;

  /* Acquire our credentials handle */
  status = s_pSecFn->AcquireCredentialsHandle(NULL,
                                              (TCHAR *) TEXT(SP_NAME_DIGEST),
                                              SECPKG_CRED_OUTBOUND, NULL,
                                              p_identity, NULL, NULL,
                                              &credentials, &expiry);

  if(status != SEC_E_OK) {
    Curl_sspi_free_identity(p_identity);
    free(spn);
    free(output_token);
    return CURLE_LOGIN_DENIED;
  }

  /* Setup the challenge "input" security buffer */
  chlg_desc.ulVersion = SECBUFFER_VERSION;
  chlg_desc.cBuffers  = 1;
  chlg_desc.pBuffers  = &chlg_buf;
  chlg_buf.BufferType = SECBUFFER_TOKEN;
  chlg_buf.pvBuffer   = (void *) Curl_bufref_ptr(chlg);
  chlg_buf.cbBuffer   = curlx_uztoul(Curl_bufref_len(chlg));

  /* Setup the response "output" security buffer */
  resp_desc.ulVersion = SECBUFFER_VERSION;
  resp_desc.cBuffers  = 1;
  resp_desc.pBuffers  = &resp_buf;
  resp_buf.BufferType = SECBUFFER_TOKEN;
  resp_buf.pvBuffer   = output_token;
  resp_buf.cbBuffer   = curlx_uztoul(token_max);

  /* Generate our response message */
  status = s_pSecFn->InitializeSecurityContext(&credentials, NULL, spn,
                                               0, 0, 0, &chlg_desc, 0,
                                               &context, &resp_desc, &attrs,
                                               &expiry);

  if(status == SEC_I_COMPLETE_NEEDED ||
     status == SEC_I_COMPLETE_AND_CONTINUE)
    s_pSecFn->CompleteAuthToken(&credentials, &resp_desc);
  else if(status != SEC_E_OK && status != SEC_I_CONTINUE_NEEDED) {
#if !defined(CURL_DISABLE_VERBOSE_STRINGS)
    char buffer[STRERROR_LEN];
#endif

    s_pSecFn->FreeCredentialsHandle(&credentials);
    Curl_sspi_free_identity(p_identity);
    free(spn);
    free(output_token);

    if(status == SEC_E_INSUFFICIENT_MEMORY)
      return CURLE_OUT_OF_MEMORY;

    infof(data, "schannel: InitializeSecurityContext failed: %s",
          Curl_sspi_strerror(status, buffer, sizeof(buffer)));

    return CURLE_AUTH_ERROR;
  }

  /* Return the response. */
  Curl_bufref_set(out, output_token, resp_buf.cbBuffer, curl_free);

  /* Free our handles */
  s_pSecFn->DeleteSecurityContext(&context);
  s_pSecFn->FreeCredentialsHandle(&credentials);

  /* Free the identity structure */
  Curl_sspi_free_identity(p_identity);

  /* Free the SPN */
  free(spn);

  return result;
}

/*
 * Curl_override_sspi_http_realm()
 *
 * This is used to populate the domain in a SSPI identity structure
 * The realm is extracted from the challenge message and used as the
 * domain if it is not already explicitly set.
 *
 * Parameters:
 *
 * chlg     [in]     - The challenge message.
 * identity [in/out] - The identity structure.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_override_sspi_http_realm(const char *chlg,
                                       SEC_WINNT_AUTH_IDENTITY *identity)
{
  xcharp_u domain, dup_domain;

  /* If domain is blank or unset, check challenge message for realm */
  if(!identity->Domain || !identity->DomainLength) {
    for(;;) {
      char value[DIGEST_MAX_VALUE_LENGTH];
      char content[DIGEST_MAX_CONTENT_LENGTH];

      /* Pass all additional spaces here */
      while(*chlg && ISSPACE(*chlg))
        chlg++;

      /* Extract a value=content pair */
      if(Curl_auth_digest_get_pair(chlg, value, content, &chlg)) {
        if(strcasecompare(value, "realm")) {

          /* Setup identity's domain and length */
          domain.tchar_ptr = curlx_convert_UTF8_to_tchar((char *) content);
          if(!domain.tchar_ptr)
            return CURLE_OUT_OF_MEMORY;

          dup_domain.tchar_ptr = _tcsdup(domain.tchar_ptr);
          if(!dup_domain.tchar_ptr) {
            curlx_unicodefree(domain.tchar_ptr);
            return CURLE_OUT_OF_MEMORY;
          }

          free(identity->Domain);
          identity->Domain = dup_domain.tbyte_ptr;
          identity->DomainLength = curlx_uztoul(_tcslen(dup_domain.tchar_ptr));
          dup_domain.tchar_ptr = NULL;

          curlx_unicodefree(domain.tchar_ptr);
        }
        else {
          /* Unknown specifier, ignore it! */
        }
      }
      else
        break; /* We're done here */

      /* Pass all additional spaces here */
      while(*chlg && ISSPACE(*chlg))
        chlg++;

      /* Allow the list to be comma-separated */
      if(',' == *chlg)
        chlg++;
    }
  }

  return CURLE_OK;
}

/*
 * Curl_auth_decode_digest_http_message()
 *
 * This is used to decode a HTTP DIGEST challenge message into the separate
 * attributes.
 *
 * Parameters:
 *
 * chlg    [in]     - The challenge message.
 * digest  [in/out] - The digest data struct being used and modified.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_decode_digest_http_message(const char *chlg,
                                              struct digestdata *digest)
{
  size_t chlglen = strlen(chlg);

  /* We had an input token before so if there's another one now that means we
     provided bad credentials in the previous request or it's stale. */
  if(digest->input_token) {
    bool stale = false;
    const char *p = chlg;

    /* Check for the 'stale' directive */
    for(;;) {
      char value[DIGEST_MAX_VALUE_LENGTH];
      char content[DIGEST_MAX_CONTENT_LENGTH];

      while(*p && ISSPACE(*p))
        p++;

      if(!Curl_auth_digest_get_pair(p, value, content, &p))
        break;

      if(strcasecompare(value, "stale") &&
         strcasecompare(content, "true")) {
        stale = true;
        break;
      }

      while(*p && ISSPACE(*p))
        p++;

      if(',' == *p)
        p++;
    }

    if(stale)
      Curl_auth_digest_cleanup(digest);
    else
      return CURLE_LOGIN_DENIED;
  }

  /* Store the challenge for use later */
  digest->input_token = (BYTE *) Curl_memdup(chlg, chlglen + 1);
  if(!digest->input_token)
    return CURLE_OUT_OF_MEMORY;

  digest->input_token_len = chlglen;

  return CURLE_OK;
}

/*
 * Curl_auth_create_digest_http_message()
 *
 * This is used to generate a HTTP DIGEST response message ready for sending
 * to the recipient.
 *
 * Parameters:
 *
 * data    [in]     - The session handle.
 * userp   [in]     - The user name in the format User or Domain\User.
 * passwdp [in]     - The user's password.
 * request [in]     - The HTTP request.
 * uripath [in]     - The path of the HTTP uri.
 * digest  [in/out] - The digest data struct being used and modified.
 * outptr  [in/out] - The address where a pointer to newly allocated memory
 *                    holding the result will be stored upon completion.
 * outlen  [out]    - The length of the output message.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_digest_http_message(struct Curl_easy *data,
                                              const char *userp,
                                              const char *passwdp,
                                              const unsigned char *request,
                                              const unsigned char *uripath,
                                              struct digestdata *digest,
                                              char **outptr, size_t *outlen)
{
  size_t token_max;
  char *resp;
  BYTE *output_token;
  size_t output_token_len = 0;
  PSecPkgInfo SecurityPackage;
  SecBuffer chlg_buf[5];
  SecBufferDesc chlg_desc;
  SECURITY_STATUS status;

  (void) data;

  /* Query the security package for DigestSSP */
  status = s_pSecFn->QuerySecurityPackageInfo((TCHAR *) TEXT(SP_NAME_DIGEST),
                                              &SecurityPackage);
  if(status != SEC_E_OK) {
    failf(data, "SSPI: couldn't get auth info");
    return CURLE_AUTH_ERROR;
  }

  token_max = SecurityPackage->cbMaxToken;

  /* Release the package buffer as it is not required anymore */
  s_pSecFn->FreeContextBuffer(SecurityPackage);

  /* Allocate the output buffer according to the max token size as indicated
     by the security package */
  output_token = malloc(token_max);
  if(!output_token) {
    return CURLE_OUT_OF_MEMORY;
  }

  /* If the user/passwd that was used to make the identity for http_context
     has changed then delete that context. */
  if((userp && !digest->user) || (!userp && digest->user) ||
     (passwdp && !digest->passwd) || (!passwdp && digest->passwd) ||
     (userp && digest->user && strcmp(userp, digest->user)) ||
     (passwdp && digest->passwd && strcmp(passwdp, digest->passwd))) {
    if(digest->http_context) {
      s_pSecFn->DeleteSecurityContext(digest->http_context);
      Curl_safefree(digest->http_context);
    }
    Curl_safefree(digest->user);
    Curl_safefree(digest->passwd);
  }

  if(digest->http_context) {
    chlg_desc.ulVersion    = SECBUFFER_VERSION;
    chlg_desc.cBuffers     = 5;
    chlg_desc.pBuffers     = chlg_buf;
    chlg_buf[0].BufferType = SECBUFFER_TOKEN;
    chlg_buf[0].pvBuffer   = NULL;
    chlg_buf[0].cbBuffer   = 0;
    chlg_buf[1].BufferType = SECBUFFER_PKG_PARAMS;
    chlg_buf[1].pvBuffer   = (void *) request;
    chlg_buf[1].cbBuffer   = curlx_uztoul(strlen((const char *) request));
    chlg_buf[2].BufferType = SECBUFFER_PKG_PARAMS;
    chlg_buf[2].pvBuffer   = (void *) uripath;
    chlg_buf[2].cbBuffer   = curlx_uztoul(strlen((const char *) uripath));
    chlg_buf[3].BufferType = SECBUFFER_PKG_PARAMS;
    chlg_buf[3].pvBuffer   = NULL;
    chlg_buf[3].cbBuffer   = 0;
    chlg_buf[4].BufferType = SECBUFFER_PADDING;
    chlg_buf[4].pvBuffer   = output_token;
    chlg_buf[4].cbBuffer   = curlx_uztoul(token_max);

    status = s_pSecFn->MakeSignature(digest->http_context, 0, &chlg_desc, 0);
    if(status == SEC_E_OK)
      output_token_len = chlg_buf[4].cbBuffer;
    else { /* delete the context so a new one can be made */
      infof(data, "digest_sspi: MakeSignature failed, error 0x%08lx",
            (long)status);
      s_pSecFn->DeleteSecurityContext(digest->http_context);
      Curl_safefree(digest->http_context);
    }
  }

  if(!digest->http_context) {
    CredHandle credentials;
    SEC_WINNT_AUTH_IDENTITY identity;
    SEC_WINNT_AUTH_IDENTITY *p_identity;
    SecBuffer resp_buf;
    SecBufferDesc resp_desc;
    unsigned long attrs;
    TimeStamp expiry; /* For Windows 9x compatibility of SSPI calls */
    TCHAR *spn;

    /* free the copy of user/passwd used to make the previous identity */
    Curl_safefree(digest->user);
    Curl_safefree(digest->passwd);

    if(userp && *userp) {
      /* Populate our identity structure */
      if(Curl_create_sspi_identity(userp, passwdp, &identity)) {
        free(output_token);
        return CURLE_OUT_OF_MEMORY;
      }

      /* Populate our identity domain */
      if(Curl_override_sspi_http_realm((const char *) digest->input_token,
                                       &identity)) {
        free(output_token);
        return CURLE_OUT_OF_MEMORY;
      }

      /* Allow proper cleanup of the identity structure */
      p_identity = &identity;
    }
    else
      /* Use the current Windows user */
      p_identity = NULL;

    if(userp) {
      digest->user = strdup(userp);

      if(!digest->user) {
        free(output_token);
        return CURLE_OUT_OF_MEMORY;
      }
    }

    if(passwdp) {
      digest->passwd = strdup(passwdp);

      if(!digest->passwd) {
        free(output_token);
        Curl_safefree(digest->user);
        return CURLE_OUT_OF_MEMORY;
      }
    }

    /* Acquire our credentials handle */
    status = s_pSecFn->AcquireCredentialsHandle(NULL,
                                                (TCHAR *) TEXT(SP_NAME_DIGEST),
                                                SECPKG_CRED_OUTBOUND, NULL,
                                                p_identity, NULL, NULL,
                                                &credentials, &expiry);
    if(status != SEC_E_OK) {
      Curl_sspi_free_identity(p_identity);
      free(output_token);

      return CURLE_LOGIN_DENIED;
    }

    /* Setup the challenge "input" security buffer if present */
    chlg_desc.ulVersion    = SECBUFFER_VERSION;
    chlg_desc.cBuffers     = 3;
    chlg_desc.pBuffers     = chlg_buf;
    chlg_buf[0].BufferType = SECBUFFER_TOKEN;
    chlg_buf[0].pvBuffer   = digest->input_token;
    chlg_buf[0].cbBuffer   = curlx_uztoul(digest->input_token_len);
    chlg_buf[1].BufferType = SECBUFFER_PKG_PARAMS;
    chlg_buf[1].pvBuffer   = (void *) request;
    chlg_buf[1].cbBuffer   = curlx_uztoul(strlen((const char *) request));
    chlg_buf[2].BufferType = SECBUFFER_PKG_PARAMS;
    chlg_buf[2].pvBuffer   = NULL;
    chlg_buf[2].cbBuffer   = 0;

    /* Setup the response "output" security buffer */
    resp_desc.ulVersion = SECBUFFER_VERSION;
    resp_desc.cBuffers  = 1;
    resp_desc.pBuffers  = &resp_buf;
    resp_buf.BufferType = SECBUFFER_TOKEN;
    resp_buf.pvBuffer   = output_token;
    resp_buf.cbBuffer   = curlx_uztoul(token_max);

    spn = curlx_convert_UTF8_to_tchar((char *) uripath);
    if(!spn) {
      s_pSecFn->FreeCredentialsHandle(&credentials);

      Curl_sspi_free_identity(p_identity);
      free(output_token);

      return CURLE_OUT_OF_MEMORY;
    }

    /* Allocate our new context handle */
    digest->http_context = calloc(1, sizeof(CtxtHandle));
    if(!digest->http_context)
      return CURLE_OUT_OF_MEMORY;

    /* Generate our response message */
    status = s_pSecFn->InitializeSecurityContext(&credentials, NULL,
                                                 spn,
                                                 ISC_REQ_USE_HTTP_STYLE, 0, 0,
                                                 &chlg_desc, 0,
                                                 digest->http_context,
                                                 &resp_desc, &attrs, &expiry);
    curlx_unicodefree(spn);

    if(status == SEC_I_COMPLETE_NEEDED ||
       status == SEC_I_COMPLETE_AND_CONTINUE)
      s_pSecFn->CompleteAuthToken(&credentials, &resp_desc);
    else if(status != SEC_E_OK && status != SEC_I_CONTINUE_NEEDED) {
#if !defined(CURL_DISABLE_VERBOSE_STRINGS)
      char buffer[STRERROR_LEN];
#endif

      s_pSecFn->FreeCredentialsHandle(&credentials);

      Curl_sspi_free_identity(p_identity);
      free(output_token);

      Curl_safefree(digest->http_context);

      if(status == SEC_E_INSUFFICIENT_MEMORY)
        return CURLE_OUT_OF_MEMORY;

      infof(data, "schannel: InitializeSecurityContext failed: %s",
            Curl_sspi_strerror(status, buffer, sizeof(buffer)));

      return CURLE_AUTH_ERROR;
    }

    output_token_len = resp_buf.cbBuffer;

    s_pSecFn->FreeCredentialsHandle(&credentials);
    Curl_sspi_free_identity(p_identity);
  }

  resp = malloc(output_token_len + 1);
  if(!resp) {
    free(output_token);

    return CURLE_OUT_OF_MEMORY;
  }

  /* Copy the generated response */
  memcpy(resp, output_token, output_token_len);
  resp[output_token_len] = 0;

  /* Return the response */
  *outptr = resp;
  *outlen = output_token_len;

  /* Free the response buffer */
  free(output_token);

  return CURLE_OK;
}

/*
 * Curl_auth_digest_cleanup()
 *
 * This is used to clean up the digest specific data.
 *
 * Parameters:
 *
 * digest    [in/out] - The digest data struct being cleaned up.
 *
 */
void Curl_auth_digest_cleanup(struct digestdata *digest)
{
  /* Free the input token */
  Curl_safefree(digest->input_token);

  /* Reset any variables */
  digest->input_token_len = 0;

  /* Delete security context */
  if(digest->http_context) {
    s_pSecFn->DeleteSecurityContext(digest->http_context);
    Curl_safefree(digest->http_context);
  }

  /* Free the copy of user/passwd used to make the identity for http_context */
  Curl_safefree(digest->user);
  Curl_safefree(digest->passwd);
}

#endif /* USE_WINDOWS_SSPI && !CURL_DISABLE_CRYPTO_AUTH */
