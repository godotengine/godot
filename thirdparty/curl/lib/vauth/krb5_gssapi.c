/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2014 - 2019, Steve Holme, <steve_holme@hotmail.com>.
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
 * RFC4752 The Kerberos V5 ("GSSAPI") SASL Mechanism
 *
 ***************************************************************************/

#include "curl_setup.h"

#if defined(HAVE_GSSAPI) && defined(USE_KERBEROS5)

#include <curl/curl.h>

#include "vauth/vauth.h"
#include "curl_sasl.h"
#include "urldata.h"
#include "curl_gssapi.h"
#include "sendf.h"
#include "curl_printf.h"

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
 * Returns TRUE if Kerberos V5 is supported by the GSS-API library.
 */
bool Curl_auth_is_gssapi_supported(void)
{
  return TRUE;
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
 * userp       [in]     - The user name.
 * passwdp     [in]     - The user's password.
 * service     [in]     - The service type such as http, smtp, pop or imap.
 * host        [in[     - The host name.
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
  OM_uint32 major_status;
  OM_uint32 minor_status;
  OM_uint32 unused_status;
  gss_buffer_desc spn_token = GSS_C_EMPTY_BUFFER;
  gss_buffer_desc input_token = GSS_C_EMPTY_BUFFER;
  gss_buffer_desc output_token = GSS_C_EMPTY_BUFFER;

  (void) userp;
  (void) passwdp;

  if(!krb5->spn) {
    /* Generate our SPN */
    char *spn = Curl_auth_build_spn(service, NULL, host);
    if(!spn)
      return CURLE_OUT_OF_MEMORY;

    /* Populate the SPN structure */
    spn_token.value = spn;
    spn_token.length = strlen(spn);

    /* Import the SPN */
    major_status = gss_import_name(&minor_status, &spn_token,
                                   GSS_C_NT_HOSTBASED_SERVICE, &krb5->spn);
    if(GSS_ERROR(major_status)) {
      Curl_gss_log_error(data, "gss_import_name() failed: ",
                         major_status, minor_status);

      free(spn);

      return CURLE_AUTH_ERROR;
    }

    free(spn);
  }

  if(chlg) {
    if(!Curl_bufref_len(chlg)) {
      infof(data, "GSSAPI handshake failure (empty challenge message)");
      return CURLE_BAD_CONTENT_ENCODING;
    }
    input_token.value = (void *) Curl_bufref_ptr(chlg);
    input_token.length = Curl_bufref_len(chlg);
  }

  major_status = Curl_gss_init_sec_context(data,
                                           &minor_status,
                                           &krb5->context,
                                           krb5->spn,
                                           &Curl_krb5_mech_oid,
                                           GSS_C_NO_CHANNEL_BINDINGS,
                                           &input_token,
                                           &output_token,
                                           mutual_auth,
                                           NULL);

  if(GSS_ERROR(major_status)) {
    if(output_token.value)
      gss_release_buffer(&unused_status, &output_token);

    Curl_gss_log_error(data, "gss_init_sec_context() failed: ",
                       major_status, minor_status);

    return CURLE_AUTH_ERROR;
  }

  if(output_token.value && output_token.length) {
    result = Curl_bufref_memdup(out, output_token.value, output_token.length);
    gss_release_buffer(&unused_status, &output_token);
  }
  else
    Curl_bufref_set(out, mutual_auth? "": NULL, 0, NULL);

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
 * chlg    [in]     - Optional challenge message.
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
  CURLcode result = CURLE_OK;
  size_t messagelen = 0;
  unsigned char *message = NULL;
  OM_uint32 major_status;
  OM_uint32 minor_status;
  OM_uint32 unused_status;
  gss_buffer_desc input_token = GSS_C_EMPTY_BUFFER;
  gss_buffer_desc output_token = GSS_C_EMPTY_BUFFER;
  unsigned char *indata;
  gss_qop_t qop = GSS_C_QOP_DEFAULT;
  unsigned int sec_layer = 0;
  unsigned int max_size = 0;

  /* Ensure we have a valid challenge message */
  if(!Curl_bufref_len(chlg)) {
    infof(data, "GSSAPI handshake failure (empty security message)");
    return CURLE_BAD_CONTENT_ENCODING;
  }

  /* Setup the challenge "input" security buffer */
  input_token.value = (void *) Curl_bufref_ptr(chlg);
  input_token.length = Curl_bufref_len(chlg);

  /* Decrypt the inbound challenge and obtain the qop */
  major_status = gss_unwrap(&minor_status, krb5->context, &input_token,
                            &output_token, NULL, &qop);
  if(GSS_ERROR(major_status)) {
    Curl_gss_log_error(data, "gss_unwrap() failed: ",
                       major_status, minor_status);
    return CURLE_BAD_CONTENT_ENCODING;
  }

  /* Not 4 octets long so fail as per RFC4752 Section 3.1 */
  if(output_token.length != 4) {
    infof(data, "GSSAPI handshake failure (invalid security data)");
    return CURLE_BAD_CONTENT_ENCODING;
  }

  /* Extract the security layer and the maximum message size */
  indata = output_token.value;
  sec_layer = indata[0];
  max_size = (indata[1] << 16) | (indata[2] << 8) | indata[3];

  /* Free the challenge as it is not required anymore */
  gss_release_buffer(&unused_status, &output_token);

  /* Process the security layer */
  if(!(sec_layer & GSSAUTH_P_NONE)) {
    infof(data, "GSSAPI handshake failure (invalid security layer)");

    return CURLE_BAD_CONTENT_ENCODING;
  }
  sec_layer &= GSSAUTH_P_NONE;  /* We do not support a security layer */

  /* Process the maximum message size the server can receive */
  if(max_size > 0) {
    /* The server has told us it supports a maximum receive buffer, however, as
       we don't require one unless we are encrypting data, we tell the server
       our receive buffer is zero. */
    max_size = 0;
  }

  /* Allocate our message */
  messagelen = 4;
  if(authzid)
    messagelen += strlen(authzid);
  message = malloc(messagelen);
  if(!message)
    return CURLE_OUT_OF_MEMORY;

  /* Populate the message with the security layer and client supported receive
     message size. */
  message[0] = sec_layer & 0xFF;
  message[1] = (max_size >> 16) & 0xFF;
  message[2] = (max_size >> 8) & 0xFF;
  message[3] = max_size & 0xFF;

  /* If given, append the authorization identity. */

  if(authzid && *authzid)
    memcpy(message + 4, authzid, messagelen - 4);

  /* Setup the "authentication data" security buffer */
  input_token.value = message;
  input_token.length = messagelen;

  /* Encrypt the data */
  major_status = gss_wrap(&minor_status, krb5->context, 0,
                          GSS_C_QOP_DEFAULT, &input_token, NULL,
                          &output_token);
  if(GSS_ERROR(major_status)) {
    Curl_gss_log_error(data, "gss_wrap() failed: ",
                       major_status, minor_status);
    free(message);
    return CURLE_AUTH_ERROR;
  }

  /* Return the response. */
  result = Curl_bufref_memdup(out, output_token.value, output_token.length);
  /* Free the output buffer */
  gss_release_buffer(&unused_status, &output_token);

  /* Free the message buffer */
  free(message);

  return result;
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
  OM_uint32 minor_status;

  /* Free our security context */
  if(krb5->context != GSS_C_NO_CONTEXT) {
    gss_delete_sec_context(&minor_status, &krb5->context, GSS_C_NO_BUFFER);
    krb5->context = GSS_C_NO_CONTEXT;
  }

  /* Free the SPN */
  if(krb5->spn != GSS_C_NO_NAME) {
    gss_release_name(&minor_status, &krb5->spn);
    krb5->spn = GSS_C_NO_NAME;
  }
}

#endif /* HAVE_GSSAPI && USE_KERBEROS5 */
