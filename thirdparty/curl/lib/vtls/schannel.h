#ifndef HEADER_CURL_SCHANNEL_H
#define HEADER_CURL_SCHANNEL_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2012, Marc Hoersken, <info@marc-hoersken.de>, et al.
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
 ***************************************************************************/
#include "curl_setup.h"

#ifdef USE_SCHANNEL

#include <schnlsp.h>
#include <schannel.h>
#include "curl_sspi.h"

#include "urldata.h"

/* <wincrypt.h> has been included via the above <schnlsp.h>.
 * Or in case of ldap.c, it was included via <winldap.h>.
 * And since <wincrypt.h> has this:
 *   #define X509_NAME  ((LPCSTR) 7)
 *
 * And in BoringSSL's <openssl/base.h> there is:
 *  typedef struct X509_name_st X509_NAME;
 *  etc.
 *
 * this will cause all kinds of C-preprocessing paste errors in
 * BoringSSL's <openssl/x509.h>: So just undefine those defines here
 * (and only here).
 */
#if defined(HAVE_BORINGSSL) || defined(OPENSSL_IS_BORINGSSL)
# undef X509_NAME
# undef X509_CERT_PAIR
# undef X509_EXTENSIONS
#endif

extern const struct Curl_ssl Curl_ssl_schannel;

CURLcode Curl_verify_certificate(struct Curl_easy *data,
                                 struct connectdata *conn, int sockindex);

/* structs to expose only in schannel.c and schannel_verify.c */
#ifdef EXPOSE_SCHANNEL_INTERNAL_STRUCTS

#ifdef __MINGW32__
#include <_mingw.h>
#ifdef __MINGW64_VERSION_MAJOR
#define HAS_MANUAL_VERIFY_API
#endif
#else
#include <wincrypt.h>
#ifdef CERT_CHAIN_REVOCATION_CHECK_CHAIN
#define HAS_MANUAL_VERIFY_API
#endif
#endif

#define NUMOF_CIPHERS 45 /* There are 45 listed in the MS headers */

struct Curl_schannel_cred {
  CredHandle cred_handle;
  TimeStamp time_stamp;
  int refcount;
};

struct Curl_schannel_ctxt {
  CtxtHandle ctxt_handle;
  TimeStamp time_stamp;
};

struct ssl_backend_data {
  struct Curl_schannel_cred *cred;
  struct Curl_schannel_ctxt *ctxt;
  SecPkgContext_StreamSizes stream_sizes;
  size_t encdata_length, decdata_length;
  size_t encdata_offset, decdata_offset;
  unsigned char *encdata_buffer, *decdata_buffer;
  /* encdata_is_incomplete: if encdata contains only a partial record that
     can't be decrypted without another Curl_read_plain (that is, status is
     SEC_E_INCOMPLETE_MESSAGE) then set this true. after Curl_read_plain writes
     more bytes into encdata then set this back to false. */
  bool encdata_is_incomplete;
  unsigned long req_flags, ret_flags;
  CURLcode recv_unrecoverable_err; /* schannel_recv had an unrecoverable err */
  bool recv_sspi_close_notify; /* true if connection closed by close_notify */
  bool recv_connection_closed; /* true if connection closed, regardless how */
  bool use_alpn; /* true if ALPN is used for this connection */
#ifdef HAS_MANUAL_VERIFY_API
  bool use_manual_cred_validation; /* true if manual cred validation is used */
#endif
  ALG_ID algIds[NUMOF_CIPHERS];
};
#endif /* EXPOSE_SCHANNEL_INTERNAL_STRUCTS */

#endif /* USE_SCHANNEL */
#endif /* HEADER_CURL_SCHANNEL_H */
