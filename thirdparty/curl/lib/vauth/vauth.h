#ifndef HEADER_CURL_VAUTH_H
#define HEADER_CURL_VAUTH_H
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
 ***************************************************************************/

#include <curl/curl.h>

#include "bufref.h"

struct Curl_easy;

#if !defined(CURL_DISABLE_CRYPTO_AUTH)
struct digestdata;
#endif

#if defined(USE_NTLM)
struct ntlmdata;
#endif

#if defined(USE_KERBEROS5)
struct kerberos5data;
#endif

#if (defined(HAVE_GSSAPI) || defined(USE_WINDOWS_SSPI)) && defined(USE_SPNEGO)
struct negotiatedata;
#endif

#if defined(USE_GSASL)
struct gsasldata;
#endif

#if defined(USE_WINDOWS_SSPI)
#define GSS_ERROR(status) ((status) & 0x80000000)
#endif

/* This is used to build a SPN string */
#if !defined(USE_WINDOWS_SSPI)
char *Curl_auth_build_spn(const char *service, const char *host,
                          const char *realm);
#else
TCHAR *Curl_auth_build_spn(const char *service, const char *host,
                           const char *realm);
#endif

/* This is used to test if the user contains a Windows domain name */
bool Curl_auth_user_contains_domain(const char *user);

/* This is used to generate a PLAIN cleartext message */
CURLcode Curl_auth_create_plain_message(const char *authzid,
                                        const char *authcid,
                                        const char *passwd,
                                        struct bufref *out);

/* This is used to generate a LOGIN cleartext message */
CURLcode Curl_auth_create_login_message(const char *value,
                                        struct bufref *out);

/* This is used to generate an EXTERNAL cleartext message */
CURLcode Curl_auth_create_external_message(const char *user,
                                           struct bufref *out);

#if !defined(CURL_DISABLE_CRYPTO_AUTH)
/* This is used to generate a CRAM-MD5 response message */
CURLcode Curl_auth_create_cram_md5_message(const struct bufref *chlg,
                                           const char *userp,
                                           const char *passwdp,
                                           struct bufref *out);

/* This is used to evaluate if DIGEST is supported */
bool Curl_auth_is_digest_supported(void);

/* This is used to generate a base64 encoded DIGEST-MD5 response message */
CURLcode Curl_auth_create_digest_md5_message(struct Curl_easy *data,
                                             const struct bufref *chlg,
                                             const char *userp,
                                             const char *passwdp,
                                             const char *service,
                                             struct bufref *out);

/* This is used to decode a HTTP DIGEST challenge message */
CURLcode Curl_auth_decode_digest_http_message(const char *chlg,
                                              struct digestdata *digest);

/* This is used to generate a HTTP DIGEST response message */
CURLcode Curl_auth_create_digest_http_message(struct Curl_easy *data,
                                              const char *userp,
                                              const char *passwdp,
                                              const unsigned char *request,
                                              const unsigned char *uri,
                                              struct digestdata *digest,
                                              char **outptr, size_t *outlen);

/* This is used to clean up the digest specific data */
void Curl_auth_digest_cleanup(struct digestdata *digest);
#endif /* !CURL_DISABLE_CRYPTO_AUTH */

#ifdef USE_GSASL
/* This is used to evaluate if MECH is supported by gsasl */
bool Curl_auth_gsasl_is_supported(struct Curl_easy *data,
                                  const char *mech,
                                  struct gsasldata *gsasl);
/* This is used to start a gsasl method */
CURLcode Curl_auth_gsasl_start(struct Curl_easy *data,
                               const char *userp,
                               const char *passwdp,
                               struct gsasldata *gsasl);

/* This is used to process and generate a new SASL token */
CURLcode Curl_auth_gsasl_token(struct Curl_easy *data,
                               const struct bufref *chlg,
                               struct gsasldata *gsasl,
                               struct bufref *out);

/* This is used to clean up the gsasl specific data */
void Curl_auth_gsasl_cleanup(struct gsasldata *digest);
#endif

#if defined(USE_NTLM)
/* This is used to evaluate if NTLM is supported */
bool Curl_auth_is_ntlm_supported(void);

/* This is used to generate a base64 encoded NTLM type-1 message */
CURLcode Curl_auth_create_ntlm_type1_message(struct Curl_easy *data,
                                             const char *userp,
                                             const char *passwdp,
                                             const char *service,
                                             const char *host,
                                             struct ntlmdata *ntlm,
                                             struct bufref *out);

/* This is used to decode a base64 encoded NTLM type-2 message */
CURLcode Curl_auth_decode_ntlm_type2_message(struct Curl_easy *data,
                                             const struct bufref *type2,
                                             struct ntlmdata *ntlm);

/* This is used to generate a base64 encoded NTLM type-3 message */
CURLcode Curl_auth_create_ntlm_type3_message(struct Curl_easy *data,
                                             const char *userp,
                                             const char *passwdp,
                                             struct ntlmdata *ntlm,
                                             struct bufref *out);

/* This is used to clean up the NTLM specific data */
void Curl_auth_cleanup_ntlm(struct ntlmdata *ntlm);
#endif /* USE_NTLM */

/* This is used to generate a base64 encoded OAuth 2.0 message */
CURLcode Curl_auth_create_oauth_bearer_message(const char *user,
                                               const char *host,
                                               const long port,
                                               const char *bearer,
                                               struct bufref *out);

/* This is used to generate a base64 encoded XOAuth 2.0 message */
CURLcode Curl_auth_create_xoauth_bearer_message(const char *user,
                                                const char *bearer,
                                                struct bufref *out);

#if defined(USE_KERBEROS5)
/* This is used to evaluate if GSSAPI (Kerberos V5) is supported */
bool Curl_auth_is_gssapi_supported(void);

/* This is used to generate a base64 encoded GSSAPI (Kerberos V5) user token
   message */
CURLcode Curl_auth_create_gssapi_user_message(struct Curl_easy *data,
                                              const char *userp,
                                              const char *passwdp,
                                              const char *service,
                                              const char *host,
                                              const bool mutual,
                                              const struct bufref *chlg,
                                              struct kerberos5data *krb5,
                                              struct bufref *out);

/* This is used to generate a base64 encoded GSSAPI (Kerberos V5) security
   token message */
CURLcode Curl_auth_create_gssapi_security_message(struct Curl_easy *data,
                                                  const char *authzid,
                                                  const struct bufref *chlg,
                                                  struct kerberos5data *krb5,
                                                  struct bufref *out);

/* This is used to clean up the GSSAPI specific data */
void Curl_auth_cleanup_gssapi(struct kerberos5data *krb5);
#endif /* USE_KERBEROS5 */

#if defined(USE_SPNEGO)
/* This is used to evaluate if SPNEGO (Negotiate) is supported */
bool Curl_auth_is_spnego_supported(void);

/* This is used to decode a base64 encoded SPNEGO (Negotiate) challenge
   message */
CURLcode Curl_auth_decode_spnego_message(struct Curl_easy *data,
                                         const char *user,
                                         const char *passwood,
                                         const char *service,
                                         const char *host,
                                         const char *chlg64,
                                         struct negotiatedata *nego);

/* This is used to generate a base64 encoded SPNEGO (Negotiate) response
   message */
CURLcode Curl_auth_create_spnego_message(struct Curl_easy *data,
                                         struct negotiatedata *nego,
                                         char **outptr, size_t *outlen);

/* This is used to clean up the SPNEGO specifiec data */
void Curl_auth_cleanup_spnego(struct negotiatedata *nego);

#endif /* USE_SPNEGO */

#endif /* HEADER_CURL_VAUTH_H */
