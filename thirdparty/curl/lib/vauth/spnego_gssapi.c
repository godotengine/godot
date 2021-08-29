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

#if defined(HAVE_GSSAPI) && defined(USE_SPNEGO)

#include <curl/curl.h>

#include "vauth/vauth.h"
#include "urldata.h"
#include "curl_base64.h"
#include "curl_gssapi.h"
#include "warnless.h"
#include "curl_multibyte.h"
#include "sendf.h"

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
 * Returns TRUE if Negotiate supported by the GSS-API library.
 */
bool Curl_auth_is_spnego_supported(void)
{
  return TRUE;
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
 * userp       [in]     - The user name in the format User or Domain\User.
 * passwdp     [in]     - The user's password.
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
  OM_uint32 major_status;
  OM_uint32 minor_status;
  OM_uint32 unused_status;
  gss_buffer_desc spn_token = GSS_C_EMPTY_BUFFER;
  gss_buffer_desc input_token = GSS_C_EMPTY_BUFFER;
  gss_buffer_desc output_token = GSS_C_EMPTY_BUFFER;

  (void) user;
  (void) password;

  if(nego->context && nego->status == GSS_S_COMPLETE) {
    /* We finished successfully our part of authentication, but server
     * rejected it (since we're again here). Exit with an error since we
     * can't invent anything better */
    Curl_auth_cleanup_spnego(nego);
    return CURLE_LOGIN_DENIED;
  }

  if(!nego->spn) {
    /* Generate our SPN */
    char *spn = Curl_auth_build_spn(service, NULL, host);
    if(!spn)
      return CURLE_OUT_OF_MEMORY;

    /* Populate the SPN structure */
    spn_token.value = spn;
    spn_token.length = strlen(spn);

    /* Import the SPN */
    major_status = gss_import_name(&minor_status, &spn_token,
                                   GSS_C_NT_HOSTBASED_SERVICE,
                                   &nego->spn);
    if(GSS_ERROR(major_status)) {
      Curl_gss_log_error(data, "gss_import_name() failed: ",
                         major_status, minor_status);

      free(spn);

      return CURLE_AUTH_ERROR;
    }

    free(spn);
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
    input_token.value = chlg;
    input_token.length = chlglen;
  }

  /* Generate our challenge-response message */
  major_status = Curl_gss_init_sec_context(data,
                                           &minor_status,
                                           &nego->context,
                                           nego->spn,
                                           &Curl_spnego_mech_oid,
                                           GSS_C_NO_CHANNEL_BINDINGS,
                                           &input_token,
                                           &output_token,
                                           TRUE,
                                           NULL);

  /* Free the decoded challenge as it is not required anymore */
  Curl_safefree(input_token.value);

  nego->status = major_status;
  if(GSS_ERROR(major_status)) {
    if(output_token.value)
      gss_release_buffer(&unused_status, &output_token);

    Curl_gss_log_error(data, "gss_init_sec_context() failed: ",
                       major_status, minor_status);

    return CURLE_AUTH_ERROR;
  }

  if(!output_token.value || !output_token.length) {
    if(output_token.value)
      gss_release_buffer(&unused_status, &output_token);

    return CURLE_AUTH_ERROR;
  }

  /* Free previous token */
  if(nego->output_token.length && nego->output_token.value)
    gss_release_buffer(&unused_status, &nego->output_token);

  nego->output_token = output_token;

  return CURLE_OK;
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
  OM_uint32 minor_status;

  /* Base64 encode the already generated response */
  result = Curl_base64_encode(data,
                              nego->output_token.value,
                              nego->output_token.length,
                              outptr, outlen);

  if(result) {
    gss_release_buffer(&minor_status, &nego->output_token);
    nego->output_token.value = NULL;
    nego->output_token.length = 0;

    return result;
  }

  if(!*outptr || !*outlen) {
    gss_release_buffer(&minor_status, &nego->output_token);
    nego->output_token.value = NULL;
    nego->output_token.length = 0;

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
  OM_uint32 minor_status;

  /* Free our security context */
  if(nego->context != GSS_C_NO_CONTEXT) {
    gss_delete_sec_context(&minor_status, &nego->context, GSS_C_NO_BUFFER);
    nego->context = GSS_C_NO_CONTEXT;
  }

  /* Free the output token */
  if(nego->output_token.value) {
    gss_release_buffer(&minor_status, &nego->output_token);
    nego->output_token.value = NULL;
    nego->output_token.length = 0;

  }

  /* Free the SPN */
  if(nego->spn != GSS_C_NO_NAME) {
    gss_release_name(&minor_status, &nego->spn);
    nego->spn = GSS_C_NO_NAME;
  }

  /* Reset any variables */
  nego->status = 0;
  nego->noauthpersist = FALSE;
  nego->havenoauthpersist = FALSE;
  nego->havenegdata = FALSE;
  nego->havemultiplerequests = FALSE;
}

#endif /* HAVE_GSSAPI && USE_SPNEGO */
