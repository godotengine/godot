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

/*
 * Source file for all OpenSSL-specific code for the TLS/SSL layer. No code
 * but vtls.c should ever call or use these functions.
 */

#include "curl_setup.h"

#ifdef USE_OPENSSL

#include <limits.h>

/* Wincrypt must be included before anything that could include OpenSSL. */
#if defined(USE_WIN32_CRYPTO)
#include <wincrypt.h>
/* Undefine wincrypt conflicting symbols for BoringSSL. */
#undef X509_NAME
#undef X509_EXTENSIONS
#undef PKCS7_ISSUER_AND_SERIAL
#undef PKCS7_SIGNER_INFO
#undef OCSP_REQUEST
#undef OCSP_RESPONSE
#endif

#include "urldata.h"
#include "sendf.h"
#include "formdata.h" /* for the boundary function */
#include "url.h" /* for the ssl config check function */
#include "inet_pton.h"
#include "openssl.h"
#include "connect.h"
#include "slist.h"
#include "select.h"
#include "vtls.h"
#include "keylog.h"
#include "strcase.h"
#include "hostcheck.h"
#include "multiif.h"
#include "strerror.h"
#include "curl_printf.h"

#include <openssl/ssl.h>
#include <openssl/rand.h>
#include <openssl/x509v3.h>
#ifndef OPENSSL_NO_DSA
#include <openssl/dsa.h>
#endif
#include <openssl/dh.h>
#include <openssl/err.h>
#include <openssl/md5.h>
#include <openssl/conf.h>
#include <openssl/bn.h>
#include <openssl/rsa.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/pkcs12.h>

#ifdef USE_AMISSL
#include "amigaos.h"
#endif

#if (OPENSSL_VERSION_NUMBER >= 0x0090808fL) && !defined(OPENSSL_NO_OCSP)
#include <openssl/ocsp.h>
#endif

#if (OPENSSL_VERSION_NUMBER >= 0x0090700fL) && /* 0.9.7 or later */     \
  !defined(OPENSSL_NO_ENGINE) && !defined(OPENSSL_NO_UI_CONSOLE)
#define USE_OPENSSL_ENGINE
#include <openssl/engine.h>
#endif

#include "warnless.h"
#include "non-ascii.h" /* for Curl_convert_from_utf8 prototype */

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/* Uncomment the ALLOW_RENEG line to a real #define if you want to allow TLS
   renegotiations when built with BoringSSL. Renegotiating is non-compliant
   with HTTP/2 and "an extremely dangerous protocol feature". Beware.

#define ALLOW_RENEG 1
 */

#ifndef OPENSSL_VERSION_NUMBER
#error "OPENSSL_VERSION_NUMBER not defined"
#endif

#ifdef USE_OPENSSL_ENGINE
#include <openssl/ui.h>
#endif

#if OPENSSL_VERSION_NUMBER >= 0x00909000L
#define SSL_METHOD_QUAL const
#else
#define SSL_METHOD_QUAL
#endif

#if (OPENSSL_VERSION_NUMBER >= 0x10000000L)
#define HAVE_ERR_REMOVE_THREAD_STATE 1
#endif

#if (OPENSSL_VERSION_NUMBER >= 0x10100000L) && /* OpenSSL 1.1.0+ */ \
    !(defined(LIBRESSL_VERSION_NUMBER) && \
      LIBRESSL_VERSION_NUMBER < 0x20700000L)
#define SSLEAY_VERSION_NUMBER OPENSSL_VERSION_NUMBER
#define HAVE_X509_GET0_EXTENSIONS 1 /* added in 1.1.0 -pre1 */
#define HAVE_OPAQUE_EVP_PKEY 1 /* since 1.1.0 -pre3 */
#define HAVE_OPAQUE_RSA_DSA_DH 1 /* since 1.1.0 -pre5 */
#define CONST_EXTS const
#define HAVE_ERR_REMOVE_THREAD_STATE_DEPRECATED 1

/* funny typecast define due to difference in API */
#ifdef LIBRESSL_VERSION_NUMBER
#define ARG2_X509_signature_print (X509_ALGOR *)
#else
#define ARG2_X509_signature_print
#endif

#else
/* For OpenSSL before 1.1.0 */
#define ASN1_STRING_get0_data(x) ASN1_STRING_data(x)
#define X509_get0_notBefore(x) X509_get_notBefore(x)
#define X509_get0_notAfter(x) X509_get_notAfter(x)
#define CONST_EXTS /* nope */
#ifndef LIBRESSL_VERSION_NUMBER
#define OpenSSL_version_num() SSLeay()
#endif
#endif

#if (OPENSSL_VERSION_NUMBER >= 0x1000200fL) && /* 1.0.2 or later */ \
    !(defined(LIBRESSL_VERSION_NUMBER) && \
      LIBRESSL_VERSION_NUMBER < 0x20700000L)
#define HAVE_X509_GET0_SIGNATURE 1
#endif

#if (OPENSSL_VERSION_NUMBER >= 0x1000200fL) /* 1.0.2 or later */
#define HAVE_SSL_GET_SHUTDOWN 1
#endif

#if OPENSSL_VERSION_NUMBER >= 0x10002003L && \
  OPENSSL_VERSION_NUMBER <= 0x10002FFFL && \
  !defined(OPENSSL_NO_COMP)
#define HAVE_SSL_COMP_FREE_COMPRESSION_METHODS 1
#endif

#if (OPENSSL_VERSION_NUMBER < 0x0090808fL)
/* not present in older OpenSSL */
#define OPENSSL_load_builtin_modules(x)
#endif

#if (OPENSSL_VERSION_NUMBER >= 0x30000000L)
#define HAVE_EVP_PKEY_GET_PARAMS 1
#else
#define SSL_get1_peer_certificate SSL_get_peer_certificate
#endif

#ifdef HAVE_EVP_PKEY_GET_PARAMS
#include <openssl/core_names.h>
#define DECLARE_PKEY_PARAM_BIGNUM(name) BIGNUM *name = NULL
#define FREE_PKEY_PARAM_BIGNUM(name) BN_clear_free(name)
#else
#define DECLARE_PKEY_PARAM_BIGNUM(name) const BIGNUM *name
#define FREE_PKEY_PARAM_BIGNUM(name)
#endif

/*
 * Whether SSL_CTX_set_keylog_callback is available.
 * OpenSSL: supported since 1.1.1 https://github.com/openssl/openssl/pull/2287
 * BoringSSL: supported since d28f59c27bac (committed 2015-11-19)
 * LibreSSL: unsupported in at least 2.7.2 (explicitly check for it since it
 *           lies and pretends to be OpenSSL 2.0.0).
 */
#if (OPENSSL_VERSION_NUMBER >= 0x10101000L && \
     !defined(LIBRESSL_VERSION_NUMBER)) || \
    defined(OPENSSL_IS_BORINGSSL)
#define HAVE_KEYLOG_CALLBACK
#endif

/* Whether SSL_CTX_set_ciphersuites is available.
 * OpenSSL: supported since 1.1.1 (commit a53b5be6a05)
 * BoringSSL: no
 * LibreSSL: no
 */
#if ((OPENSSL_VERSION_NUMBER >= 0x10101000L) && \
     !defined(LIBRESSL_VERSION_NUMBER) &&       \
     !defined(OPENSSL_IS_BORINGSSL))
#define HAVE_SSL_CTX_SET_CIPHERSUITES
#define HAVE_SSL_CTX_SET_POST_HANDSHAKE_AUTH
/* SET_EC_CURVES is available under the same preconditions: see
 * https://www.openssl.org/docs/manmaster/man3/SSL_CTX_set1_groups.html
 */
#define HAVE_SSL_CTX_SET_EC_CURVES
#endif

#if defined(LIBRESSL_VERSION_NUMBER)
#define OSSL_PACKAGE "LibreSSL"
#elif defined(OPENSSL_IS_BORINGSSL)
#define OSSL_PACKAGE "BoringSSL"
#else
#define OSSL_PACKAGE "OpenSSL"
#endif

#if (OPENSSL_VERSION_NUMBER >= 0x10100000L)
/* up2date versions of OpenSSL maintain reasonably secure defaults without
 * breaking compatibility, so it is better not to override the defaults in curl
 */
#define DEFAULT_CIPHER_SELECTION NULL
#else
/* ... but it is not the case with old versions of OpenSSL */
#define DEFAULT_CIPHER_SELECTION \
  "ALL:!EXPORT:!EXPORT40:!EXPORT56:!aNULL:!LOW:!RC4:@STRENGTH"
#endif

#ifdef HAVE_OPENSSL_SRP
/* the function exists */
#ifdef USE_TLS_SRP
/* the functionality is not disabled */
#define USE_OPENSSL_SRP
#endif
#endif

#if (OPENSSL_VERSION_NUMBER >= 0x10100000L)
#define HAVE_RANDOM_INIT_BY_DEFAULT 1
#endif

struct ssl_backend_data {
  struct Curl_easy *logger; /* transfer handle to pass trace logs to, only
                               using sockindex 0 */
  /* these ones requires specific SSL-types */
  SSL_CTX* ctx;
  SSL*     handle;
  X509*    server_cert;
#ifndef HAVE_KEYLOG_CALLBACK
  /* Set to true once a valid keylog entry has been created to avoid dupes. */
  bool     keylog_done;
#endif
};

static void ossl_associate_connection(struct Curl_easy *data,
                                      struct connectdata *conn,
                                      int sockindex);

/*
 * Number of bytes to read from the random number seed file. This must be
 * a finite value (because some entropy "files" like /dev/urandom have
 * an infinite length), but must be large enough to provide enough
 * entropy to properly seed OpenSSL's PRNG.
 */
#define RAND_LOAD_LENGTH 1024

#ifdef HAVE_KEYLOG_CALLBACK
static void ossl_keylog_callback(const SSL *ssl, const char *line)
{
  (void)ssl;

  Curl_tls_keylog_write_line(line);
}
#else
/*
 * ossl_log_tls12_secret is called by libcurl to make the CLIENT_RANDOMs if the
 * OpenSSL being used doesn't have native support for doing that.
 */
static void
ossl_log_tls12_secret(const SSL *ssl, bool *keylog_done)
{
  const SSL_SESSION *session = SSL_get_session(ssl);
  unsigned char client_random[SSL3_RANDOM_SIZE];
  unsigned char master_key[SSL_MAX_MASTER_KEY_LENGTH];
  int master_key_length = 0;

  if(!session || *keylog_done)
    return;

#if OPENSSL_VERSION_NUMBER >= 0x10100000L && \
    !(defined(LIBRESSL_VERSION_NUMBER) && \
      LIBRESSL_VERSION_NUMBER < 0x20700000L)
  /* ssl->s3 is not checked in openssl 1.1.0-pre6, but let's assume that
   * we have a valid SSL context if we have a non-NULL session. */
  SSL_get_client_random(ssl, client_random, SSL3_RANDOM_SIZE);
  master_key_length = (int)
    SSL_SESSION_get_master_key(session, master_key, SSL_MAX_MASTER_KEY_LENGTH);
#else
  if(ssl->s3 && session->master_key_length > 0) {
    master_key_length = session->master_key_length;
    memcpy(master_key, session->master_key, session->master_key_length);
    memcpy(client_random, ssl->s3->client_random, SSL3_RANDOM_SIZE);
  }
#endif

  /* The handshake has not progressed sufficiently yet, or this is a TLS 1.3
   * session (when curl was built with older OpenSSL headers and running with
   * newer OpenSSL runtime libraries). */
  if(master_key_length <= 0)
    return;

  *keylog_done = true;
  Curl_tls_keylog_write("CLIENT_RANDOM", client_random,
                        master_key, master_key_length);
}
#endif /* !HAVE_KEYLOG_CALLBACK */

static const char *SSL_ERROR_to_str(int err)
{
  switch(err) {
  case SSL_ERROR_NONE:
    return "SSL_ERROR_NONE";
  case SSL_ERROR_SSL:
    return "SSL_ERROR_SSL";
  case SSL_ERROR_WANT_READ:
    return "SSL_ERROR_WANT_READ";
  case SSL_ERROR_WANT_WRITE:
    return "SSL_ERROR_WANT_WRITE";
  case SSL_ERROR_WANT_X509_LOOKUP:
    return "SSL_ERROR_WANT_X509_LOOKUP";
  case SSL_ERROR_SYSCALL:
    return "SSL_ERROR_SYSCALL";
  case SSL_ERROR_ZERO_RETURN:
    return "SSL_ERROR_ZERO_RETURN";
  case SSL_ERROR_WANT_CONNECT:
    return "SSL_ERROR_WANT_CONNECT";
  case SSL_ERROR_WANT_ACCEPT:
    return "SSL_ERROR_WANT_ACCEPT";
#if defined(SSL_ERROR_WANT_ASYNC)
  case SSL_ERROR_WANT_ASYNC:
    return "SSL_ERROR_WANT_ASYNC";
#endif
#if defined(SSL_ERROR_WANT_ASYNC_JOB)
  case SSL_ERROR_WANT_ASYNC_JOB:
    return "SSL_ERROR_WANT_ASYNC_JOB";
#endif
#if defined(SSL_ERROR_WANT_EARLY)
  case SSL_ERROR_WANT_EARLY:
    return "SSL_ERROR_WANT_EARLY";
#endif
  default:
    return "SSL_ERROR unknown";
  }
}

/* Return error string for last OpenSSL error
 */
static char *ossl_strerror(unsigned long error, char *buf, size_t size)
{
  if(size)
    *buf = '\0';

#ifdef OPENSSL_IS_BORINGSSL
  ERR_error_string_n((uint32_t)error, buf, size);
#else
  ERR_error_string_n(error, buf, size);
#endif

  if(size > 1 && !*buf) {
    strncpy(buf, (error ? "Unknown error" : "No error"), size);
    buf[size - 1] = '\0';
  }

  return buf;
}

/* Return an extra data index for the transfer data.
 * This index can be used with SSL_get_ex_data() and SSL_set_ex_data().
 */
static int ossl_get_ssl_data_index(void)
{
  static int ssl_ex_data_data_index = -1;
  if(ssl_ex_data_data_index < 0) {
    ssl_ex_data_data_index = SSL_get_ex_new_index(0, NULL, NULL, NULL, NULL);
  }
  return ssl_ex_data_data_index;
}

/* Return an extra data index for the connection data.
 * This index can be used with SSL_get_ex_data() and SSL_set_ex_data().
 */
static int ossl_get_ssl_conn_index(void)
{
  static int ssl_ex_data_conn_index = -1;
  if(ssl_ex_data_conn_index < 0) {
    ssl_ex_data_conn_index = SSL_get_ex_new_index(0, NULL, NULL, NULL, NULL);
  }
  return ssl_ex_data_conn_index;
}

/* Return an extra data index for the sockindex.
 * This index can be used with SSL_get_ex_data() and SSL_set_ex_data().
 */
static int ossl_get_ssl_sockindex_index(void)
{
  static int sockindex_index = -1;
  if(sockindex_index < 0) {
    sockindex_index = SSL_get_ex_new_index(0, NULL, NULL, NULL, NULL);
  }
  return sockindex_index;
}

/* Return an extra data index for proxy boolean.
 * This index can be used with SSL_get_ex_data() and SSL_set_ex_data().
 */
static int ossl_get_proxy_index(void)
{
  static int proxy_index = -1;
  if(proxy_index < 0) {
    proxy_index = SSL_get_ex_new_index(0, NULL, NULL, NULL, NULL);
  }
  return proxy_index;
}

static int passwd_callback(char *buf, int num, int encrypting,
                           void *global_passwd)
{
  DEBUGASSERT(0 == encrypting);

  if(!encrypting) {
    int klen = curlx_uztosi(strlen((char *)global_passwd));
    if(num > klen) {
      memcpy(buf, global_passwd, klen + 1);
      return klen;
    }
  }
  return 0;
}

/*
 * rand_enough() returns TRUE if we have seeded the random engine properly.
 */
static bool rand_enough(void)
{
  return (0 != RAND_status()) ? TRUE : FALSE;
}

static CURLcode ossl_seed(struct Curl_easy *data)
{
  /* This might get called before it has been added to a multi handle */
  if(data->multi && data->multi->ssl_seeded)
    return CURLE_OK;

  if(rand_enough()) {
    /* OpenSSL 1.1.0+ should return here */
    if(data->multi)
      data->multi->ssl_seeded = TRUE;
    return CURLE_OK;
  }
#ifdef HAVE_RANDOM_INIT_BY_DEFAULT
  /* with OpenSSL 1.1.0+, a failed RAND_status is a showstopper */
  failf(data, "Insufficient randomness");
  return CURLE_SSL_CONNECT_ERROR;
#else

#ifndef RANDOM_FILE
  /* if RANDOM_FILE isn't defined, we only perform this if an option tells
     us to! */
  if(data->set.str[STRING_SSL_RANDOM_FILE])
#define RANDOM_FILE "" /* doesn't matter won't be used */
#endif
  {
    /* let the option override the define */
    RAND_load_file((data->set.str[STRING_SSL_RANDOM_FILE]?
                    data->set.str[STRING_SSL_RANDOM_FILE]:
                    RANDOM_FILE),
                   RAND_LOAD_LENGTH);
    if(rand_enough())
      return CURLE_OK;
  }

#if defined(HAVE_RAND_EGD)
  /* only available in OpenSSL 0.9.5 and later */
  /* EGD_SOCKET is set at configure time or not at all */
#ifndef EGD_SOCKET
  /* If we don't have the define set, we only do this if the egd-option
     is set */
  if(data->set.str[STRING_SSL_EGDSOCKET])
#define EGD_SOCKET "" /* doesn't matter won't be used */
#endif
  {
    /* If there's an option and a define, the option overrides the
       define */
    int ret = RAND_egd(data->set.str[STRING_SSL_EGDSOCKET]?
                       data->set.str[STRING_SSL_EGDSOCKET]:EGD_SOCKET);
    if(-1 != ret) {
      if(rand_enough())
        return CURLE_OK;
    }
  }
#endif

  /* fallback to a custom seeding of the PRNG using a hash based on a current
     time */
  do {
    unsigned char randb[64];
    size_t len = sizeof(randb);
    size_t i, i_max;
    for(i = 0, i_max = len / sizeof(struct curltime); i < i_max; ++i) {
      struct curltime tv = Curl_now();
      Curl_wait_ms(1);
      tv.tv_sec *= i + 1;
      tv.tv_usec *= (unsigned int)i + 2;
      tv.tv_sec ^= ((Curl_now().tv_sec + Curl_now().tv_usec) *
                    (i + 3)) << 8;
      tv.tv_usec ^= (unsigned int) ((Curl_now().tv_sec +
                                     Curl_now().tv_usec) *
                                    (i + 4)) << 16;
      memcpy(&randb[i * sizeof(struct curltime)], &tv,
             sizeof(struct curltime));
    }
    RAND_add(randb, (int)len, (double)len/2);
  } while(!rand_enough());

  {
    /* generates a default path for the random seed file */
    char fname[256];
    fname[0] = 0; /* blank it first */
    RAND_file_name(fname, sizeof(fname));
    if(fname[0]) {
      /* we got a file name to try */
      RAND_load_file(fname, RAND_LOAD_LENGTH);
      if(rand_enough())
        return CURLE_OK;
    }
  }

  infof(data, "libcurl is now using a weak random seed!");
  return (rand_enough() ? CURLE_OK :
          CURLE_SSL_CONNECT_ERROR /* confusing error code */);
#endif
}

#ifndef SSL_FILETYPE_ENGINE
#define SSL_FILETYPE_ENGINE 42
#endif
#ifndef SSL_FILETYPE_PKCS12
#define SSL_FILETYPE_PKCS12 43
#endif
static int do_file_type(const char *type)
{
  if(!type || !type[0])
    return SSL_FILETYPE_PEM;
  if(strcasecompare(type, "PEM"))
    return SSL_FILETYPE_PEM;
  if(strcasecompare(type, "DER"))
    return SSL_FILETYPE_ASN1;
  if(strcasecompare(type, "ENG"))
    return SSL_FILETYPE_ENGINE;
  if(strcasecompare(type, "P12"))
    return SSL_FILETYPE_PKCS12;
  return -1;
}

#ifdef USE_OPENSSL_ENGINE
/*
 * Supply default password to the engine user interface conversation.
 * The password is passed by OpenSSL engine from ENGINE_load_private_key()
 * last argument to the ui and can be obtained by UI_get0_user_data(ui) here.
 */
static int ssl_ui_reader(UI *ui, UI_STRING *uis)
{
  const char *password;
  switch(UI_get_string_type(uis)) {
  case UIT_PROMPT:
  case UIT_VERIFY:
    password = (const char *)UI_get0_user_data(ui);
    if(password && (UI_get_input_flags(uis) & UI_INPUT_FLAG_DEFAULT_PWD)) {
      UI_set_result(ui, uis, password);
      return 1;
    }
  default:
    break;
  }
  return (UI_method_get_reader(UI_OpenSSL()))(ui, uis);
}

/*
 * Suppress interactive request for a default password if available.
 */
static int ssl_ui_writer(UI *ui, UI_STRING *uis)
{
  switch(UI_get_string_type(uis)) {
  case UIT_PROMPT:
  case UIT_VERIFY:
    if(UI_get0_user_data(ui) &&
       (UI_get_input_flags(uis) & UI_INPUT_FLAG_DEFAULT_PWD)) {
      return 1;
    }
  default:
    break;
  }
  return (UI_method_get_writer(UI_OpenSSL()))(ui, uis);
}

/*
 * Check if a given string is a PKCS#11 URI
 */
static bool is_pkcs11_uri(const char *string)
{
  return (string && strncasecompare(string, "pkcs11:", 7));
}

#endif

static CURLcode ossl_set_engine(struct Curl_easy *data, const char *engine);

static int
SSL_CTX_use_certificate_blob(SSL_CTX *ctx, const struct curl_blob *blob,
                             int type, const char *key_passwd)
{
  int ret = 0;
  X509 *x = NULL;
  /* the typecast of blob->len is fine since it is guaranteed to never be
     larger than CURL_MAX_INPUT_LENGTH */
  BIO *in = BIO_new_mem_buf(blob->data, (int)(blob->len));
  if(!in)
    return CURLE_OUT_OF_MEMORY;

  if(type == SSL_FILETYPE_ASN1) {
    /* j = ERR_R_ASN1_LIB; */
    x = d2i_X509_bio(in, NULL);
  }
  else if(type == SSL_FILETYPE_PEM) {
    /* ERR_R_PEM_LIB; */
    x = PEM_read_bio_X509(in, NULL,
                          passwd_callback, (void *)key_passwd);
  }
  else {
    ret = 0;
    goto end;
  }

  if(!x) {
    ret = 0;
    goto end;
  }

  ret = SSL_CTX_use_certificate(ctx, x);
 end:
  X509_free(x);
  BIO_free(in);
  return ret;
}

static int
SSL_CTX_use_PrivateKey_blob(SSL_CTX *ctx, const struct curl_blob *blob,
                           int type, const char *key_passwd)
{
  int ret = 0;
  EVP_PKEY *pkey = NULL;
  BIO *in = BIO_new_mem_buf(blob->data, (int)(blob->len));
  if(!in)
    return CURLE_OUT_OF_MEMORY;

  if(type == SSL_FILETYPE_PEM)
    pkey = PEM_read_bio_PrivateKey(in, NULL, passwd_callback,
                                   (void *)key_passwd);
  else if(type == SSL_FILETYPE_ASN1)
    pkey = d2i_PrivateKey_bio(in, NULL);
  else {
    ret = 0;
    goto end;
  }
  if(!pkey) {
    ret = 0;
    goto end;
  }
  ret = SSL_CTX_use_PrivateKey(ctx, pkey);
  EVP_PKEY_free(pkey);
  end:
  BIO_free(in);
  return ret;
}

static int
SSL_CTX_use_certificate_chain_blob(SSL_CTX *ctx, const struct curl_blob *blob,
                                   const char *key_passwd)
{
/* SSL_CTX_add1_chain_cert introduced in OpenSSL 1.0.2 */
#if (OPENSSL_VERSION_NUMBER >= 0x1000200fL) && /* OpenSSL 1.0.2 or later */ \
    !(defined(LIBRESSL_VERSION_NUMBER) && \
      (LIBRESSL_VERSION_NUMBER < 0x2090100fL)) /* LibreSSL 2.9.1 or later */
  int ret = 0;
  X509 *x = NULL;
  void *passwd_callback_userdata = (void *)key_passwd;
  BIO *in = BIO_new_mem_buf(blob->data, (int)(blob->len));
  if(!in)
    return CURLE_OUT_OF_MEMORY;

  ERR_clear_error();

  x = PEM_read_bio_X509_AUX(in, NULL,
                            passwd_callback, (void *)key_passwd);

  if(!x) {
    ret = 0;
    goto end;
  }

  ret = SSL_CTX_use_certificate(ctx, x);

  if(ERR_peek_error() != 0)
    ret = 0;

  if(ret) {
    X509 *ca;
    unsigned long err;

    if(!SSL_CTX_clear_chain_certs(ctx)) {
      ret = 0;
      goto end;
    }

    while((ca = PEM_read_bio_X509(in, NULL, passwd_callback,
                                  passwd_callback_userdata))
          != NULL) {

      if(!SSL_CTX_add0_chain_cert(ctx, ca)) {
        X509_free(ca);
        ret = 0;
        goto end;
      }
    }

    err = ERR_peek_last_error();
    if((ERR_GET_LIB(err) == ERR_LIB_PEM) &&
       (ERR_GET_REASON(err) == PEM_R_NO_START_LINE))
      ERR_clear_error();
    else
      ret = 0;
  }

 end:
  X509_free(x);
  BIO_free(in);
  return ret;
#else
  (void)ctx; /* unused */
  (void)blob; /* unused */
  (void)key_passwd; /* unused */
  return 0;
#endif
}

static
int cert_stuff(struct Curl_easy *data,
               SSL_CTX* ctx,
               char *cert_file,
               const struct curl_blob *cert_blob,
               const char *cert_type,
               char *key_file,
               const struct curl_blob *key_blob,
               const char *key_type,
               char *key_passwd)
{
  char error_buffer[256];
  bool check_privkey = TRUE;

  int file_type = do_file_type(cert_type);

  if(cert_file || cert_blob || (file_type == SSL_FILETYPE_ENGINE)) {
    SSL *ssl;
    X509 *x509;
    int cert_done = 0;
    int cert_use_result;

    if(key_passwd) {
      /* set the password in the callback userdata */
      SSL_CTX_set_default_passwd_cb_userdata(ctx, key_passwd);
      /* Set passwd callback: */
      SSL_CTX_set_default_passwd_cb(ctx, passwd_callback);
    }


    switch(file_type) {
    case SSL_FILETYPE_PEM:
      /* SSL_CTX_use_certificate_chain_file() only works on PEM files */
      cert_use_result = cert_blob ?
        SSL_CTX_use_certificate_chain_blob(ctx, cert_blob, key_passwd) :
        SSL_CTX_use_certificate_chain_file(ctx, cert_file);
      if(cert_use_result != 1) {
        failf(data,
              "could not load PEM client certificate, " OSSL_PACKAGE
              " error %s, "
              "(no key found, wrong pass phrase, or wrong file format?)",
              ossl_strerror(ERR_get_error(), error_buffer,
                            sizeof(error_buffer)) );
        return 0;
      }
      break;

    case SSL_FILETYPE_ASN1:
      /* SSL_CTX_use_certificate_file() works with either PEM or ASN1, but
         we use the case above for PEM so this can only be performed with
         ASN1 files. */

      cert_use_result = cert_blob ?
        SSL_CTX_use_certificate_blob(ctx, cert_blob,
                                     file_type, key_passwd) :
        SSL_CTX_use_certificate_file(ctx, cert_file, file_type);
      if(cert_use_result != 1) {
        failf(data,
              "could not load ASN1 client certificate, " OSSL_PACKAGE
              " error %s, "
              "(no key found, wrong pass phrase, or wrong file format?)",
              ossl_strerror(ERR_get_error(), error_buffer,
                            sizeof(error_buffer)) );
        return 0;
      }
      break;
    case SSL_FILETYPE_ENGINE:
#if defined(USE_OPENSSL_ENGINE) && defined(ENGINE_CTRL_GET_CMD_FROM_NAME)
      {
        /* Implicitly use pkcs11 engine if none was provided and the
         * cert_file is a PKCS#11 URI */
        if(!data->state.engine) {
          if(is_pkcs11_uri(cert_file)) {
            if(ossl_set_engine(data, "pkcs11") != CURLE_OK) {
              return 0;
            }
          }
        }

        if(data->state.engine) {
          const char *cmd_name = "LOAD_CERT_CTRL";
          struct {
            const char *cert_id;
            X509 *cert;
          } params;

          params.cert_id = cert_file;
          params.cert = NULL;

          /* Does the engine supports LOAD_CERT_CTRL ? */
          if(!ENGINE_ctrl(data->state.engine, ENGINE_CTRL_GET_CMD_FROM_NAME,
                          0, (void *)cmd_name, NULL)) {
            failf(data, "ssl engine does not support loading certificates");
            return 0;
          }

          /* Load the certificate from the engine */
          if(!ENGINE_ctrl_cmd(data->state.engine, cmd_name,
                              0, &params, NULL, 1)) {
            failf(data, "ssl engine cannot load client cert with id"
                  " '%s' [%s]", cert_file,
                  ossl_strerror(ERR_get_error(), error_buffer,
                                sizeof(error_buffer)));
            return 0;
          }

          if(!params.cert) {
            failf(data, "ssl engine didn't initialized the certificate "
                  "properly.");
            return 0;
          }

          if(SSL_CTX_use_certificate(ctx, params.cert) != 1) {
            failf(data, "unable to set client certificate");
            X509_free(params.cert);
            return 0;
          }
          X509_free(params.cert); /* we don't need the handle any more... */
        }
        else {
          failf(data, "crypto engine not set, can't load certificate");
          return 0;
        }
      }
      break;
#else
      failf(data, "file type ENG for certificate not implemented");
      return 0;
#endif

    case SSL_FILETYPE_PKCS12:
    {
      BIO *cert_bio = NULL;
      PKCS12 *p12 = NULL;
      EVP_PKEY *pri;
      STACK_OF(X509) *ca = NULL;
      if(cert_blob) {
        cert_bio = BIO_new_mem_buf(cert_blob->data, (int)(cert_blob->len));
        if(!cert_bio) {
          failf(data,
                "BIO_new_mem_buf NULL, " OSSL_PACKAGE
                " error %s",
                ossl_strerror(ERR_get_error(), error_buffer,
                              sizeof(error_buffer)) );
          return 0;
        }
      }
      else {
        cert_bio = BIO_new(BIO_s_file());
        if(!cert_bio) {
          failf(data,
                "BIO_new return NULL, " OSSL_PACKAGE
                " error %s",
                ossl_strerror(ERR_get_error(), error_buffer,
                              sizeof(error_buffer)) );
          return 0;
        }

        if(BIO_read_filename(cert_bio, cert_file) <= 0) {
          failf(data, "could not open PKCS12 file '%s'", cert_file);
          BIO_free(cert_bio);
          return 0;
        }
      }

      p12 = d2i_PKCS12_bio(cert_bio, NULL);
      BIO_free(cert_bio);

      if(!p12) {
        failf(data, "error reading PKCS12 file '%s'",
              cert_blob ? "(memory blob)" : cert_file);
        return 0;
      }

      PKCS12_PBE_add();

      if(!PKCS12_parse(p12, key_passwd, &pri, &x509,
                       &ca)) {
        failf(data,
              "could not parse PKCS12 file, check password, " OSSL_PACKAGE
              " error %s",
              ossl_strerror(ERR_get_error(), error_buffer,
                            sizeof(error_buffer)) );
        PKCS12_free(p12);
        return 0;
      }

      PKCS12_free(p12);

      if(SSL_CTX_use_certificate(ctx, x509) != 1) {
        failf(data,
              "could not load PKCS12 client certificate, " OSSL_PACKAGE
              " error %s",
              ossl_strerror(ERR_get_error(), error_buffer,
                            sizeof(error_buffer)) );
        goto fail;
      }

      if(SSL_CTX_use_PrivateKey(ctx, pri) != 1) {
        failf(data, "unable to use private key from PKCS12 file '%s'",
              cert_file);
        goto fail;
      }

      if(!SSL_CTX_check_private_key (ctx)) {
        failf(data, "private key from PKCS12 file '%s' "
              "does not match certificate in same file", cert_file);
        goto fail;
      }
      /* Set Certificate Verification chain */
      if(ca) {
        while(sk_X509_num(ca)) {
          /*
           * Note that sk_X509_pop() is used below to make sure the cert is
           * removed from the stack properly before getting passed to
           * SSL_CTX_add_extra_chain_cert(), which takes ownership. Previously
           * we used sk_X509_value() instead, but then we'd clean it in the
           * subsequent sk_X509_pop_free() call.
           */
          X509 *x = sk_X509_pop(ca);
          if(!SSL_CTX_add_client_CA(ctx, x)) {
            X509_free(x);
            failf(data, "cannot add certificate to client CA list");
            goto fail;
          }
          if(!SSL_CTX_add_extra_chain_cert(ctx, x)) {
            X509_free(x);
            failf(data, "cannot add certificate to certificate chain");
            goto fail;
          }
        }
      }

      cert_done = 1;
  fail:
      EVP_PKEY_free(pri);
      X509_free(x509);
#ifdef USE_AMISSL
      sk_X509_pop_free(ca, Curl_amiga_X509_free);
#else
      sk_X509_pop_free(ca, X509_free);
#endif
      if(!cert_done)
        return 0; /* failure! */
      break;
    }
    default:
      failf(data, "not supported file type '%s' for certificate", cert_type);
      return 0;
    }

    if((!key_file) && (!key_blob)) {
      key_file = cert_file;
      key_blob = cert_blob;
    }
    else
      file_type = do_file_type(key_type);

    switch(file_type) {
    case SSL_FILETYPE_PEM:
      if(cert_done)
        break;
      /* FALLTHROUGH */
    case SSL_FILETYPE_ASN1:
      cert_use_result = key_blob ?
        SSL_CTX_use_PrivateKey_blob(ctx, key_blob, file_type, key_passwd) :
        SSL_CTX_use_PrivateKey_file(ctx, key_file, file_type);
      if(cert_use_result != 1) {
        failf(data, "unable to set private key file: '%s' type %s",
              key_file?key_file:"(memory blob)", key_type?key_type:"PEM");
        return 0;
      }
      break;
    case SSL_FILETYPE_ENGINE:
#ifdef USE_OPENSSL_ENGINE
      {                         /* XXXX still needs some work */
        EVP_PKEY *priv_key = NULL;

        /* Implicitly use pkcs11 engine if none was provided and the
         * key_file is a PKCS#11 URI */
        if(!data->state.engine) {
          if(is_pkcs11_uri(key_file)) {
            if(ossl_set_engine(data, "pkcs11") != CURLE_OK) {
              return 0;
            }
          }
        }

        if(data->state.engine) {
          UI_METHOD *ui_method =
            UI_create_method((char *)"curl user interface");
          if(!ui_method) {
            failf(data, "unable do create " OSSL_PACKAGE
                  " user-interface method");
            return 0;
          }
          UI_method_set_opener(ui_method, UI_method_get_opener(UI_OpenSSL()));
          UI_method_set_closer(ui_method, UI_method_get_closer(UI_OpenSSL()));
          UI_method_set_reader(ui_method, ssl_ui_reader);
          UI_method_set_writer(ui_method, ssl_ui_writer);
          /* the typecast below was added to please mingw32 */
          priv_key = (EVP_PKEY *)
            ENGINE_load_private_key(data->state.engine, key_file,
                                    ui_method,
                                    key_passwd);
          UI_destroy_method(ui_method);
          if(!priv_key) {
            failf(data, "failed to load private key from crypto engine");
            return 0;
          }
          if(SSL_CTX_use_PrivateKey(ctx, priv_key) != 1) {
            failf(data, "unable to set private key");
            EVP_PKEY_free(priv_key);
            return 0;
          }
          EVP_PKEY_free(priv_key);  /* we don't need the handle any more... */
        }
        else {
          failf(data, "crypto engine not set, can't load private key");
          return 0;
        }
      }
      break;
#else
      failf(data, "file type ENG for private key not supported");
      return 0;
#endif
    case SSL_FILETYPE_PKCS12:
      if(!cert_done) {
        failf(data, "file type P12 for private key not supported");
        return 0;
      }
      break;
    default:
      failf(data, "not supported file type for private key");
      return 0;
    }

    ssl = SSL_new(ctx);
    if(!ssl) {
      failf(data, "unable to create an SSL structure");
      return 0;
    }

    x509 = SSL_get_certificate(ssl);

    /* This version was provided by Evan Jordan and is supposed to not
       leak memory as the previous version: */
    if(x509) {
      EVP_PKEY *pktmp = X509_get_pubkey(x509);
      EVP_PKEY_copy_parameters(pktmp, SSL_get_privatekey(ssl));
      EVP_PKEY_free(pktmp);
    }

#if !defined(OPENSSL_NO_RSA) && !defined(OPENSSL_IS_BORINGSSL) && \
    !defined(OPENSSL_NO_DEPRECATED_3_0)
    {
      /* If RSA is used, don't check the private key if its flags indicate
       * it doesn't support it. */
      EVP_PKEY *priv_key = SSL_get_privatekey(ssl);
      int pktype;
#ifdef HAVE_OPAQUE_EVP_PKEY
      pktype = EVP_PKEY_id(priv_key);
#else
      pktype = priv_key->type;
#endif
      if(pktype == EVP_PKEY_RSA) {
        RSA *rsa = EVP_PKEY_get1_RSA(priv_key);
        if(RSA_flags(rsa) & RSA_METHOD_FLAG_NO_CHECK)
          check_privkey = FALSE;
        RSA_free(rsa); /* Decrement reference count */
      }
    }
#endif

    SSL_free(ssl);

    /* If we are using DSA, we can copy the parameters from
     * the private key */

    if(check_privkey == TRUE) {
      /* Now we know that a key and cert have been set against
       * the SSL context */
      if(!SSL_CTX_check_private_key(ctx)) {
        failf(data, "Private key does not match the certificate public key");
        return 0;
      }
    }
  }
  return 1;
}

/* returns non-zero on failure */
static int x509_name_oneline(X509_NAME *a, char *buf, size_t size)
{
  BIO *bio_out = BIO_new(BIO_s_mem());
  BUF_MEM *biomem;
  int rc;

  if(!bio_out)
    return 1; /* alloc failed! */

  rc = X509_NAME_print_ex(bio_out, a, 0, XN_FLAG_SEP_SPLUS_SPC);
  BIO_get_mem_ptr(bio_out, &biomem);

  if((size_t)biomem->length < size)
    size = biomem->length;
  else
    size--; /* don't overwrite the buffer end */

  memcpy(buf, biomem->data, size);
  buf[size] = 0;

  BIO_free(bio_out);

  return !rc;
}

/**
 * Global SSL init
 *
 * @retval 0 error initializing SSL
 * @retval 1 SSL initialized successfully
 */
static int ossl_init(void)
{
#if (OPENSSL_VERSION_NUMBER >= 0x10100000L) &&  \
  !defined(LIBRESSL_VERSION_NUMBER)
  const uint64_t flags =
#ifdef OPENSSL_INIT_ENGINE_ALL_BUILTIN
    /* not present in BoringSSL */
    OPENSSL_INIT_ENGINE_ALL_BUILTIN |
#endif
#ifdef CURL_DISABLE_OPENSSL_AUTO_LOAD_CONFIG
    OPENSSL_INIT_NO_LOAD_CONFIG |
#else
    OPENSSL_INIT_LOAD_CONFIG |
#endif
    0;
  OPENSSL_init_ssl(flags, NULL);
#else
  OPENSSL_load_builtin_modules();

#ifdef USE_OPENSSL_ENGINE
  ENGINE_load_builtin_engines();
#endif

/* CONF_MFLAGS_DEFAULT_SECTION was introduced some time between 0.9.8b and
   0.9.8e */
#ifndef CONF_MFLAGS_DEFAULT_SECTION
#define CONF_MFLAGS_DEFAULT_SECTION 0x0
#endif

#ifndef CURL_DISABLE_OPENSSL_AUTO_LOAD_CONFIG
  CONF_modules_load_file(NULL, NULL,
                         CONF_MFLAGS_DEFAULT_SECTION|
                         CONF_MFLAGS_IGNORE_MISSING_FILE);
#endif

  /* Let's get nice error messages */
  SSL_load_error_strings();

  /* Init the global ciphers and digests */
  if(!SSLeay_add_ssl_algorithms())
    return 0;

  OpenSSL_add_all_algorithms();
#endif

  Curl_tls_keylog_open();

  /* Initialize the extra data indexes */
  if(ossl_get_ssl_data_index() < 0 || ossl_get_ssl_conn_index() < 0 ||
     ossl_get_ssl_sockindex_index() < 0 || ossl_get_proxy_index() < 0)
    return 0;

  return 1;
}

/* Global cleanup */
static void ossl_cleanup(void)
{
#if (OPENSSL_VERSION_NUMBER >= 0x10100000L) && \
    !defined(LIBRESSL_VERSION_NUMBER)
  /* OpenSSL 1.1 deprecates all these cleanup functions and
     turns them into no-ops in OpenSSL 1.0 compatibility mode */
#else
  /* Free ciphers and digests lists */
  EVP_cleanup();

#ifdef USE_OPENSSL_ENGINE
  /* Free engine list */
  ENGINE_cleanup();
#endif

  /* Free OpenSSL error strings */
  ERR_free_strings();

  /* Free thread local error state, destroying hash upon zero refcount */
#ifdef HAVE_ERR_REMOVE_THREAD_STATE
  ERR_remove_thread_state(NULL);
#else
  ERR_remove_state(0);
#endif

  /* Free all memory allocated by all configuration modules */
  CONF_modules_free();

#ifdef HAVE_SSL_COMP_FREE_COMPRESSION_METHODS
  SSL_COMP_free_compression_methods();
#endif
#endif

  Curl_tls_keylog_close();
}

/*
 * This function is used to determine connection status.
 *
 * Return codes:
 *     1 means the connection is still in place
 *     0 means the connection has been closed
 *    -1 means the connection status is unknown
 */
static int ossl_check_cxn(struct connectdata *conn)
{
  /* SSL_peek takes data out of the raw recv buffer without peeking so we use
     recv MSG_PEEK instead. Bug #795 */
#ifdef MSG_PEEK
  char buf;
  ssize_t nread;
  nread = recv((RECV_TYPE_ARG1)conn->sock[FIRSTSOCKET], (RECV_TYPE_ARG2)&buf,
               (RECV_TYPE_ARG3)1, (RECV_TYPE_ARG4)MSG_PEEK);
  if(nread == 0)
    return 0; /* connection has been closed */
  if(nread == 1)
    return 1; /* connection still in place */
  else if(nread == -1) {
      int err = SOCKERRNO;
      if(err == EINPROGRESS ||
#if defined(EAGAIN) && (EAGAIN != EWOULDBLOCK)
         err == EAGAIN ||
#endif
         err == EWOULDBLOCK)
        return 1; /* connection still in place */
      if(err == ECONNRESET ||
#ifdef ECONNABORTED
         err == ECONNABORTED ||
#endif
#ifdef ENETDOWN
         err == ENETDOWN ||
#endif
#ifdef ENETRESET
         err == ENETRESET ||
#endif
#ifdef ESHUTDOWN
         err == ESHUTDOWN ||
#endif
#ifdef ETIMEDOUT
         err == ETIMEDOUT ||
#endif
         err == ENOTCONN)
        return 0; /* connection has been closed */
  }
#endif
  return -1; /* connection status unknown */
}

/* Selects an OpenSSL crypto engine
 */
static CURLcode ossl_set_engine(struct Curl_easy *data, const char *engine)
{
#ifdef USE_OPENSSL_ENGINE
  ENGINE *e;

#if OPENSSL_VERSION_NUMBER >= 0x00909000L
  e = ENGINE_by_id(engine);
#else
  /* avoid memory leak */
  for(e = ENGINE_get_first(); e; e = ENGINE_get_next(e)) {
    const char *e_id = ENGINE_get_id(e);
    if(!strcmp(engine, e_id))
      break;
  }
#endif

  if(!e) {
    failf(data, "SSL Engine '%s' not found", engine);
    return CURLE_SSL_ENGINE_NOTFOUND;
  }

  if(data->state.engine) {
    ENGINE_finish(data->state.engine);
    ENGINE_free(data->state.engine);
    data->state.engine = NULL;
  }
  if(!ENGINE_init(e)) {
    char buf[256];

    ENGINE_free(e);
    failf(data, "Failed to initialise SSL Engine '%s': %s",
          engine, ossl_strerror(ERR_get_error(), buf, sizeof(buf)));
    return CURLE_SSL_ENGINE_INITFAILED;
  }
  data->state.engine = e;
  return CURLE_OK;
#else
  (void)engine;
  failf(data, "SSL Engine not supported");
  return CURLE_SSL_ENGINE_NOTFOUND;
#endif
}

/* Sets engine as default for all SSL operations
 */
static CURLcode ossl_set_engine_default(struct Curl_easy *data)
{
#ifdef USE_OPENSSL_ENGINE
  if(data->state.engine) {
    if(ENGINE_set_default(data->state.engine, ENGINE_METHOD_ALL) > 0) {
      infof(data, "set default crypto engine '%s'",
            ENGINE_get_id(data->state.engine));
    }
    else {
      failf(data, "set default crypto engine '%s' failed",
            ENGINE_get_id(data->state.engine));
      return CURLE_SSL_ENGINE_SETFAILED;
    }
  }
#else
  (void) data;
#endif
  return CURLE_OK;
}

/* Return list of OpenSSL crypto engine names.
 */
static struct curl_slist *ossl_engines_list(struct Curl_easy *data)
{
  struct curl_slist *list = NULL;
#ifdef USE_OPENSSL_ENGINE
  struct curl_slist *beg;
  ENGINE *e;

  for(e = ENGINE_get_first(); e; e = ENGINE_get_next(e)) {
    beg = curl_slist_append(list, ENGINE_get_id(e));
    if(!beg) {
      curl_slist_free_all(list);
      return NULL;
    }
    list = beg;
  }
#endif
  (void) data;
  return list;
}

#define set_logger(conn, data)                  \
  conn->ssl[0].backend->logger = data

static void ossl_closeone(struct Curl_easy *data,
                          struct connectdata *conn,
                          struct ssl_connect_data *connssl)
{
  struct ssl_backend_data *backend = connssl->backend;
  if(backend->handle) {
    char buf[32];
    set_logger(conn, data);

    /* Maybe the server has already sent a close notify alert.
       Read it to avoid an RST on the TCP connection. */
    (void)SSL_read(backend->handle, buf, (int)sizeof(buf));

    (void)SSL_shutdown(backend->handle);
    SSL_set_connect_state(backend->handle);

    SSL_free(backend->handle);
    backend->handle = NULL;
  }
  if(backend->ctx) {
    SSL_CTX_free(backend->ctx);
    backend->ctx = NULL;
  }
}

/*
 * This function is called when an SSL connection is closed.
 */
static void ossl_close(struct Curl_easy *data, struct connectdata *conn,
                       int sockindex)
{
  ossl_closeone(data, conn, &conn->ssl[sockindex]);
#ifndef CURL_DISABLE_PROXY
  ossl_closeone(data, conn, &conn->proxy_ssl[sockindex]);
#endif
}

/*
 * This function is called to shut down the SSL layer but keep the
 * socket open (CCC - Clear Command Channel)
 */
static int ossl_shutdown(struct Curl_easy *data,
                         struct connectdata *conn, int sockindex)
{
  int retval = 0;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  char buf[256]; /* We will use this for the OpenSSL error buffer, so it has
                    to be at least 256 bytes long. */
  unsigned long sslerror;
  ssize_t nread;
  int buffsize;
  int err;
  bool done = FALSE;
  struct ssl_backend_data *backend = connssl->backend;
  int loop = 10;

#ifndef CURL_DISABLE_FTP
  /* This has only been tested on the proftpd server, and the mod_tls code
     sends a close notify alert without waiting for a close notify alert in
     response. Thus we wait for a close notify alert from the server, but
     we do not send one. Let's hope other servers do the same... */

  if(data->set.ftp_ccc == CURLFTPSSL_CCC_ACTIVE)
      (void)SSL_shutdown(backend->handle);
#endif

  if(backend->handle) {
    buffsize = (int)sizeof(buf);
    while(!done && loop--) {
      int what = SOCKET_READABLE(conn->sock[sockindex],
                                 SSL_SHUTDOWN_TIMEOUT);
      if(what > 0) {
        ERR_clear_error();

        /* Something to read, let's do it and hope that it is the close
           notify alert from the server */
        nread = (ssize_t)SSL_read(backend->handle, buf, buffsize);
        err = SSL_get_error(backend->handle, (int)nread);

        switch(err) {
        case SSL_ERROR_NONE: /* this is not an error */
        case SSL_ERROR_ZERO_RETURN: /* no more data */
          /* This is the expected response. There was no data but only
             the close notify alert */
          done = TRUE;
          break;
        case SSL_ERROR_WANT_READ:
          /* there's data pending, re-invoke SSL_read() */
          infof(data, "SSL_ERROR_WANT_READ");
          break;
        case SSL_ERROR_WANT_WRITE:
          /* SSL wants a write. Really odd. Let's bail out. */
          infof(data, "SSL_ERROR_WANT_WRITE");
          done = TRUE;
          break;
        default:
          /* openssl/ssl.h says "look at error stack/return value/errno" */
          sslerror = ERR_get_error();
          failf(data, OSSL_PACKAGE " SSL_read on shutdown: %s, errno %d",
                (sslerror ?
                 ossl_strerror(sslerror, buf, sizeof(buf)) :
                 SSL_ERROR_to_str(err)),
                SOCKERRNO);
          done = TRUE;
          break;
        }
      }
      else if(0 == what) {
        /* timeout */
        failf(data, "SSL shutdown timeout");
        done = TRUE;
      }
      else {
        /* anything that gets here is fatally bad */
        failf(data, "select/poll on SSL socket, errno: %d", SOCKERRNO);
        retval = -1;
        done = TRUE;
      }
    } /* while()-loop for the select() */

    if(data->set.verbose) {
#ifdef HAVE_SSL_GET_SHUTDOWN
      switch(SSL_get_shutdown(backend->handle)) {
      case SSL_SENT_SHUTDOWN:
        infof(data, "SSL_get_shutdown() returned SSL_SENT_SHUTDOWN");
        break;
      case SSL_RECEIVED_SHUTDOWN:
        infof(data, "SSL_get_shutdown() returned SSL_RECEIVED_SHUTDOWN");
        break;
      case SSL_SENT_SHUTDOWN|SSL_RECEIVED_SHUTDOWN:
        infof(data, "SSL_get_shutdown() returned SSL_SENT_SHUTDOWN|"
              "SSL_RECEIVED__SHUTDOWN");
        break;
      }
#endif
    }

    SSL_free(backend->handle);
    backend->handle = NULL;
  }
  return retval;
}

static void ossl_session_free(void *ptr)
{
  /* free the ID */
  SSL_SESSION_free(ptr);
}

/*
 * This function is called when the 'data' struct is going away. Close
 * down everything and free all resources!
 */
static void ossl_close_all(struct Curl_easy *data)
{
#ifdef USE_OPENSSL_ENGINE
  if(data->state.engine) {
    ENGINE_finish(data->state.engine);
    ENGINE_free(data->state.engine);
    data->state.engine = NULL;
  }
#else
  (void)data;
#endif
#if !defined(HAVE_ERR_REMOVE_THREAD_STATE_DEPRECATED) && \
  defined(HAVE_ERR_REMOVE_THREAD_STATE)
  /* OpenSSL 1.0.1 and 1.0.2 build an error queue that is stored per-thread
     so we need to clean it here in case the thread will be killed. All OpenSSL
     code should extract the error in association with the error so clearing
     this queue here should be harmless at worst. */
  ERR_remove_thread_state(NULL);
#endif
}

/* ====================================================== */

/*
 * Match subjectAltName against the host name. This requires a conversion
 * in CURL_DOES_CONVERSIONS builds.
 */
static bool subj_alt_hostcheck(struct Curl_easy *data,
                               const char *match_pattern, const char *hostname,
                               const char *dispname)
#ifdef CURL_DOES_CONVERSIONS
{
  bool res = FALSE;

  /* Curl_cert_hostcheck uses host encoding, but we get ASCII from
     OpenSSl.
   */
  char *match_pattern2 = strdup(match_pattern);

  if(match_pattern2) {
    if(Curl_convert_from_network(data, match_pattern2,
                                strlen(match_pattern2)) == CURLE_OK) {
      if(Curl_cert_hostcheck(match_pattern2, hostname)) {
        res = TRUE;
        infof(data,
                " subjectAltName: host \"%s\" matched cert's \"%s\"",
                dispname, match_pattern2);
      }
    }
    free(match_pattern2);
  }
  else {
    failf(data,
        "SSL: out of memory when allocating temporary for subjectAltName");
  }
  return res;
}
#else
{
#ifdef CURL_DISABLE_VERBOSE_STRINGS
  (void)dispname;
  (void)data;
#endif
  if(Curl_cert_hostcheck(match_pattern, hostname)) {
    infof(data, " subjectAltName: host \"%s\" matched cert's \"%s\"",
                  dispname, match_pattern);
    return TRUE;
  }
  return FALSE;
}
#endif


/* Quote from RFC2818 section 3.1 "Server Identity"

   If a subjectAltName extension of type dNSName is present, that MUST
   be used as the identity. Otherwise, the (most specific) Common Name
   field in the Subject field of the certificate MUST be used. Although
   the use of the Common Name is existing practice, it is deprecated and
   Certification Authorities are encouraged to use the dNSName instead.

   Matching is performed using the matching rules specified by
   [RFC2459].  If more than one identity of a given type is present in
   the certificate (e.g., more than one dNSName name, a match in any one
   of the set is considered acceptable.) Names may contain the wildcard
   character * which is considered to match any single domain name
   component or component fragment. E.g., *.a.com matches foo.a.com but
   not bar.foo.a.com. f*.com matches foo.com but not bar.com.

   In some cases, the URI is specified as an IP address rather than a
   hostname. In this case, the iPAddress subjectAltName must be present
   in the certificate and must exactly match the IP in the URI.

*/
static CURLcode verifyhost(struct Curl_easy *data, struct connectdata *conn,
                           X509 *server_cert)
{
  bool matched = FALSE;
  int target = GEN_DNS; /* target type, GEN_DNS or GEN_IPADD */
  size_t addrlen = 0;
  STACK_OF(GENERAL_NAME) *altnames;
#ifdef ENABLE_IPV6
  struct in6_addr addr;
#else
  struct in_addr addr;
#endif
  CURLcode result = CURLE_OK;
  bool dNSName = FALSE; /* if a dNSName field exists in the cert */
  bool iPAddress = FALSE; /* if a iPAddress field exists in the cert */
  const char * const hostname = SSL_HOST_NAME();
  const char * const dispname = SSL_HOST_DISPNAME();

#ifdef ENABLE_IPV6
  if(conn->bits.ipv6_ip &&
     Curl_inet_pton(AF_INET6, hostname, &addr)) {
    target = GEN_IPADD;
    addrlen = sizeof(struct in6_addr);
  }
  else
#endif
    if(Curl_inet_pton(AF_INET, hostname, &addr)) {
      target = GEN_IPADD;
      addrlen = sizeof(struct in_addr);
    }

  /* get a "list" of alternative names */
  altnames = X509_get_ext_d2i(server_cert, NID_subject_alt_name, NULL, NULL);

  if(altnames) {
#ifdef OPENSSL_IS_BORINGSSL
    size_t numalts;
    size_t i;
#else
    int numalts;
    int i;
#endif
    bool dnsmatched = FALSE;
    bool ipmatched = FALSE;

    /* get amount of alternatives, RFC2459 claims there MUST be at least
       one, but we don't depend on it... */
    numalts = sk_GENERAL_NAME_num(altnames);

    /* loop through all alternatives - until a dnsmatch */
    for(i = 0; (i < numalts) && !dnsmatched; i++) {
      /* get a handle to alternative name number i */
      const GENERAL_NAME *check = sk_GENERAL_NAME_value(altnames, i);

      if(check->type == GEN_DNS)
        dNSName = TRUE;
      else if(check->type == GEN_IPADD)
        iPAddress = TRUE;

      /* only check alternatives of the same type the target is */
      if(check->type == target) {
        /* get data and length */
        const char *altptr = (char *)ASN1_STRING_get0_data(check->d.ia5);
        size_t altlen = (size_t) ASN1_STRING_length(check->d.ia5);

        switch(target) {
        case GEN_DNS: /* name/pattern comparison */
          /* The OpenSSL man page explicitly says: "In general it cannot be
             assumed that the data returned by ASN1_STRING_data() is null
             terminated or does not contain embedded nulls." But also that
             "The actual format of the data will depend on the actual string
             type itself: for example for an IA5String the data will be ASCII"

             It has been however verified that in 0.9.6 and 0.9.7, IA5String
             is always null-terminated.
          */
          if((altlen == strlen(altptr)) &&
             /* if this isn't true, there was an embedded zero in the name
                string and we cannot match it. */
             subj_alt_hostcheck(data, altptr, hostname, dispname)) {
            dnsmatched = TRUE;
          }
          break;

        case GEN_IPADD: /* IP address comparison */
          /* compare alternative IP address if the data chunk is the same size
             our server IP address is */
          if((altlen == addrlen) && !memcmp(altptr, &addr, altlen)) {
            ipmatched = TRUE;
            infof(data,
                  " subjectAltName: host \"%s\" matched cert's IP address!",
                  dispname);
          }
          break;
        }
      }
    }
    GENERAL_NAMES_free(altnames);

    if(dnsmatched || ipmatched)
      matched = TRUE;
  }

  if(matched)
    /* an alternative name matched */
    ;
  else if(dNSName || iPAddress) {
    infof(data, " subjectAltName does not match %s", dispname);
    failf(data, "SSL: no alternative certificate subject name matches "
          "target host name '%s'", dispname);
    result = CURLE_PEER_FAILED_VERIFICATION;
  }
  else {
    /* we have to look to the last occurrence of a commonName in the
       distinguished one to get the most significant one. */
    int j, i = -1;

    /* The following is done because of a bug in 0.9.6b */

    unsigned char *nulstr = (unsigned char *)"";
    unsigned char *peer_CN = nulstr;

    X509_NAME *name = X509_get_subject_name(server_cert);
    if(name)
      while((j = X509_NAME_get_index_by_NID(name, NID_commonName, i)) >= 0)
        i = j;

    /* we have the name entry and we will now convert this to a string
       that we can use for comparison. Doing this we support BMPstring,
       UTF8, etc. */

    if(i >= 0) {
      ASN1_STRING *tmp =
        X509_NAME_ENTRY_get_data(X509_NAME_get_entry(name, i));

      /* In OpenSSL 0.9.7d and earlier, ASN1_STRING_to_UTF8 fails if the input
         is already UTF-8 encoded. We check for this case and copy the raw
         string manually to avoid the problem. This code can be made
         conditional in the future when OpenSSL has been fixed. */
      if(tmp) {
        if(ASN1_STRING_type(tmp) == V_ASN1_UTF8STRING) {
          j = ASN1_STRING_length(tmp);
          if(j >= 0) {
            peer_CN = OPENSSL_malloc(j + 1);
            if(peer_CN) {
              memcpy(peer_CN, ASN1_STRING_get0_data(tmp), j);
              peer_CN[j] = '\0';
            }
          }
        }
        else /* not a UTF8 name */
          j = ASN1_STRING_to_UTF8(&peer_CN, tmp);

        if(peer_CN && (curlx_uztosi(strlen((char *)peer_CN)) != j)) {
          /* there was a terminating zero before the end of string, this
             cannot match and we return failure! */
          failf(data, "SSL: illegal cert name field");
          result = CURLE_PEER_FAILED_VERIFICATION;
        }
      }
    }

    if(peer_CN == nulstr)
       peer_CN = NULL;
    else {
      /* convert peer_CN from UTF8 */
      CURLcode rc = Curl_convert_from_utf8(data, (char *)peer_CN,
                                           strlen((char *)peer_CN));
      /* Curl_convert_from_utf8 calls failf if unsuccessful */
      if(rc) {
        OPENSSL_free(peer_CN);
        return rc;
      }
    }

    if(result)
      /* error already detected, pass through */
      ;
    else if(!peer_CN) {
      failf(data,
            "SSL: unable to obtain common name from peer certificate");
      result = CURLE_PEER_FAILED_VERIFICATION;
    }
    else if(!Curl_cert_hostcheck((const char *)peer_CN, hostname)) {
      failf(data, "SSL: certificate subject name '%s' does not match "
            "target host name '%s'", peer_CN, dispname);
      result = CURLE_PEER_FAILED_VERIFICATION;
    }
    else {
      infof(data, " common name: %s (matched)", peer_CN);
    }
    if(peer_CN)
      OPENSSL_free(peer_CN);
  }

  return result;
}

#if (OPENSSL_VERSION_NUMBER >= 0x0090808fL) && !defined(OPENSSL_NO_TLSEXT) && \
    !defined(OPENSSL_NO_OCSP)
static CURLcode verifystatus(struct Curl_easy *data,
                             struct ssl_connect_data *connssl)
{
  int i, ocsp_status;
  unsigned char *status;
  const unsigned char *p;
  CURLcode result = CURLE_OK;
  OCSP_RESPONSE *rsp = NULL;
  OCSP_BASICRESP *br = NULL;
  X509_STORE     *st = NULL;
  STACK_OF(X509) *ch = NULL;
  struct ssl_backend_data *backend = connssl->backend;
  X509 *cert;
  OCSP_CERTID *id = NULL;
  int cert_status, crl_reason;
  ASN1_GENERALIZEDTIME *rev, *thisupd, *nextupd;
  int ret;

  long len = SSL_get_tlsext_status_ocsp_resp(backend->handle, &status);

  if(!status) {
    failf(data, "No OCSP response received");
    result = CURLE_SSL_INVALIDCERTSTATUS;
    goto end;
  }
  p = status;
  rsp = d2i_OCSP_RESPONSE(NULL, &p, len);
  if(!rsp) {
    failf(data, "Invalid OCSP response");
    result = CURLE_SSL_INVALIDCERTSTATUS;
    goto end;
  }

  ocsp_status = OCSP_response_status(rsp);
  if(ocsp_status != OCSP_RESPONSE_STATUS_SUCCESSFUL) {
    failf(data, "Invalid OCSP response status: %s (%d)",
          OCSP_response_status_str(ocsp_status), ocsp_status);
    result = CURLE_SSL_INVALIDCERTSTATUS;
    goto end;
  }

  br = OCSP_response_get1_basic(rsp);
  if(!br) {
    failf(data, "Invalid OCSP response");
    result = CURLE_SSL_INVALIDCERTSTATUS;
    goto end;
  }

  ch = SSL_get_peer_cert_chain(backend->handle);
  st = SSL_CTX_get_cert_store(backend->ctx);

#if ((OPENSSL_VERSION_NUMBER <= 0x1000201fL) /* Fixed after 1.0.2a */ || \
     (defined(LIBRESSL_VERSION_NUMBER) &&                               \
      LIBRESSL_VERSION_NUMBER <= 0x2040200fL))
  /* The authorized responder cert in the OCSP response MUST be signed by the
     peer cert's issuer (see RFC6960 section 4.2.2.2). If that's a root cert,
     no problem, but if it's an intermediate cert OpenSSL has a bug where it
     expects this issuer to be present in the chain embedded in the OCSP
     response. So we add it if necessary. */

  /* First make sure the peer cert chain includes both a peer and an issuer,
     and the OCSP response contains a responder cert. */
  if(sk_X509_num(ch) >= 2 && sk_X509_num(br->certs) >= 1) {
    X509 *responder = sk_X509_value(br->certs, sk_X509_num(br->certs) - 1);

    /* Find issuer of responder cert and add it to the OCSP response chain */
    for(i = 0; i < sk_X509_num(ch); i++) {
      X509 *issuer = sk_X509_value(ch, i);
      if(X509_check_issued(issuer, responder) == X509_V_OK) {
        if(!OCSP_basic_add1_cert(br, issuer)) {
          failf(data, "Could not add issuer cert to OCSP response");
          result = CURLE_SSL_INVALIDCERTSTATUS;
          goto end;
        }
      }
    }
  }
#endif

  if(OCSP_basic_verify(br, ch, st, 0) <= 0) {
    failf(data, "OCSP response verification failed");
    result = CURLE_SSL_INVALIDCERTSTATUS;
    goto end;
  }

  /* Compute the certificate's ID */
  cert = SSL_get1_peer_certificate(backend->handle);
  if(!cert) {
    failf(data, "Error getting peer certificate");
    result = CURLE_SSL_INVALIDCERTSTATUS;
    goto end;
  }

  for(i = 0; i < sk_X509_num(ch); i++) {
    X509 *issuer = sk_X509_value(ch, i);
    if(X509_check_issued(issuer, cert) == X509_V_OK) {
      id = OCSP_cert_to_id(EVP_sha1(), cert, issuer);
      break;
    }
  }
  X509_free(cert);

  if(!id) {
    failf(data, "Error computing OCSP ID");
    result = CURLE_SSL_INVALIDCERTSTATUS;
    goto end;
  }

  /* Find the single OCSP response corresponding to the certificate ID */
  ret = OCSP_resp_find_status(br, id, &cert_status, &crl_reason, &rev,
                              &thisupd, &nextupd);
  OCSP_CERTID_free(id);
  if(ret != 1) {
    failf(data, "Could not find certificate ID in OCSP response");
    result = CURLE_SSL_INVALIDCERTSTATUS;
    goto end;
  }

  /* Validate the corresponding single OCSP response */
  if(!OCSP_check_validity(thisupd, nextupd, 300L, -1L)) {
    failf(data, "OCSP response has expired");
    result = CURLE_SSL_INVALIDCERTSTATUS;
    goto end;
  }

  infof(data, "SSL certificate status: %s (%d)",
        OCSP_cert_status_str(cert_status), cert_status);

  switch(cert_status) {
  case V_OCSP_CERTSTATUS_GOOD:
    break;

  case V_OCSP_CERTSTATUS_REVOKED:
    result = CURLE_SSL_INVALIDCERTSTATUS;
    failf(data, "SSL certificate revocation reason: %s (%d)",
          OCSP_crl_reason_str(crl_reason), crl_reason);
    goto end;

  case V_OCSP_CERTSTATUS_UNKNOWN:
  default:
    result = CURLE_SSL_INVALIDCERTSTATUS;
    goto end;
  }

end:
  if(br)
    OCSP_BASICRESP_free(br);
  OCSP_RESPONSE_free(rsp);

  return result;
}
#endif

#endif /* USE_OPENSSL */

/* The SSL_CTRL_SET_MSG_CALLBACK doesn't exist in ancient OpenSSL versions
   and thus this cannot be done there. */
#ifdef SSL_CTRL_SET_MSG_CALLBACK

static const char *ssl_msg_type(int ssl_ver, int msg)
{
#ifdef SSL2_VERSION_MAJOR
  if(ssl_ver == SSL2_VERSION_MAJOR) {
    switch(msg) {
      case SSL2_MT_ERROR:
        return "Error";
      case SSL2_MT_CLIENT_HELLO:
        return "Client hello";
      case SSL2_MT_CLIENT_MASTER_KEY:
        return "Client key";
      case SSL2_MT_CLIENT_FINISHED:
        return "Client finished";
      case SSL2_MT_SERVER_HELLO:
        return "Server hello";
      case SSL2_MT_SERVER_VERIFY:
        return "Server verify";
      case SSL2_MT_SERVER_FINISHED:
        return "Server finished";
      case SSL2_MT_REQUEST_CERTIFICATE:
        return "Request CERT";
      case SSL2_MT_CLIENT_CERTIFICATE:
        return "Client CERT";
    }
  }
  else
#endif
  if(ssl_ver == SSL3_VERSION_MAJOR) {
    switch(msg) {
      case SSL3_MT_HELLO_REQUEST:
        return "Hello request";
      case SSL3_MT_CLIENT_HELLO:
        return "Client hello";
      case SSL3_MT_SERVER_HELLO:
        return "Server hello";
#ifdef SSL3_MT_NEWSESSION_TICKET
      case SSL3_MT_NEWSESSION_TICKET:
        return "Newsession Ticket";
#endif
      case SSL3_MT_CERTIFICATE:
        return "Certificate";
      case SSL3_MT_SERVER_KEY_EXCHANGE:
        return "Server key exchange";
      case SSL3_MT_CLIENT_KEY_EXCHANGE:
        return "Client key exchange";
      case SSL3_MT_CERTIFICATE_REQUEST:
        return "Request CERT";
      case SSL3_MT_SERVER_DONE:
        return "Server finished";
      case SSL3_MT_CERTIFICATE_VERIFY:
        return "CERT verify";
      case SSL3_MT_FINISHED:
        return "Finished";
#ifdef SSL3_MT_CERTIFICATE_STATUS
      case SSL3_MT_CERTIFICATE_STATUS:
        return "Certificate Status";
#endif
#ifdef SSL3_MT_ENCRYPTED_EXTENSIONS
      case SSL3_MT_ENCRYPTED_EXTENSIONS:
        return "Encrypted Extensions";
#endif
#ifdef SSL3_MT_SUPPLEMENTAL_DATA
      case SSL3_MT_SUPPLEMENTAL_DATA:
        return "Supplemental data";
#endif
#ifdef SSL3_MT_END_OF_EARLY_DATA
      case SSL3_MT_END_OF_EARLY_DATA:
        return "End of early data";
#endif
#ifdef SSL3_MT_KEY_UPDATE
      case SSL3_MT_KEY_UPDATE:
        return "Key update";
#endif
#ifdef SSL3_MT_NEXT_PROTO
      case SSL3_MT_NEXT_PROTO:
        return "Next protocol";
#endif
#ifdef SSL3_MT_MESSAGE_HASH
      case SSL3_MT_MESSAGE_HASH:
        return "Message hash";
#endif
    }
  }
  return "Unknown";
}

static const char *tls_rt_type(int type)
{
  switch(type) {
#ifdef SSL3_RT_HEADER
  case SSL3_RT_HEADER:
    return "TLS header";
#endif
  case SSL3_RT_CHANGE_CIPHER_SPEC:
    return "TLS change cipher";
  case SSL3_RT_ALERT:
    return "TLS alert";
  case SSL3_RT_HANDSHAKE:
    return "TLS handshake";
  case SSL3_RT_APPLICATION_DATA:
    return "TLS app data";
  default:
    return "TLS Unknown";
  }
}

/*
 * Our callback from the SSL/TLS layers.
 */
static void ossl_trace(int direction, int ssl_ver, int content_type,
                       const void *buf, size_t len, SSL *ssl,
                       void *userp)
{
  char unknown[32];
  const char *verstr = NULL;
  struct connectdata *conn = userp;
  struct ssl_connect_data *connssl = &conn->ssl[0];
  struct ssl_backend_data *backend = connssl->backend;
  struct Curl_easy *data = backend->logger;

  if(!conn || !data || !data->set.fdebug ||
     (direction != 0 && direction != 1))
    return;

  switch(ssl_ver) {
#ifdef SSL2_VERSION /* removed in recent versions */
  case SSL2_VERSION:
    verstr = "SSLv2";
    break;
#endif
#ifdef SSL3_VERSION
  case SSL3_VERSION:
    verstr = "SSLv3";
    break;
#endif
  case TLS1_VERSION:
    verstr = "TLSv1.0";
    break;
#ifdef TLS1_1_VERSION
  case TLS1_1_VERSION:
    verstr = "TLSv1.1";
    break;
#endif
#ifdef TLS1_2_VERSION
  case TLS1_2_VERSION:
    verstr = "TLSv1.2";
    break;
#endif
#ifdef TLS1_3_VERSION
  case TLS1_3_VERSION:
    verstr = "TLSv1.3";
    break;
#endif
  case 0:
    break;
  default:
    msnprintf(unknown, sizeof(unknown), "(%x)", ssl_ver);
    verstr = unknown;
    break;
  }

  /* Log progress for interesting records only (like Handshake or Alert), skip
   * all raw record headers (content_type == SSL3_RT_HEADER or ssl_ver == 0).
   * For TLS 1.3, skip notification of the decrypted inner Content-Type.
   */
  if(ssl_ver
#ifdef SSL3_RT_INNER_CONTENT_TYPE
     && content_type != SSL3_RT_INNER_CONTENT_TYPE
#endif
    ) {
    const char *msg_name, *tls_rt_name;
    char ssl_buf[1024];
    int msg_type, txt_len;

    /* the info given when the version is zero is not that useful for us */

    ssl_ver >>= 8; /* check the upper 8 bits only below */

    /* SSLv2 doesn't seem to have TLS record-type headers, so OpenSSL
     * always pass-up content-type as 0. But the interesting message-type
     * is at 'buf[0]'.
     */
    if(ssl_ver == SSL3_VERSION_MAJOR && content_type)
      tls_rt_name = tls_rt_type(content_type);
    else
      tls_rt_name = "";

    if(content_type == SSL3_RT_CHANGE_CIPHER_SPEC) {
      msg_type = *(char *)buf;
      msg_name = "Change cipher spec";
    }
    else if(content_type == SSL3_RT_ALERT) {
      msg_type = (((char *)buf)[0] << 8) + ((char *)buf)[1];
      msg_name = SSL_alert_desc_string_long(msg_type);
    }
    else {
      msg_type = *(char *)buf;
      msg_name = ssl_msg_type(ssl_ver, msg_type);
    }

    txt_len = msnprintf(ssl_buf, sizeof(ssl_buf), "%s (%s), %s, %s (%d):\n",
                        verstr, direction?"OUT":"IN",
                        tls_rt_name, msg_name, msg_type);
    if(0 <= txt_len && (unsigned)txt_len < sizeof(ssl_buf)) {
      Curl_debug(data, CURLINFO_TEXT, ssl_buf, (size_t)txt_len);
    }
  }

  Curl_debug(data, (direction == 1) ? CURLINFO_SSL_DATA_OUT :
             CURLINFO_SSL_DATA_IN, (char *)buf, len);
  (void) ssl;
}
#endif

#ifdef USE_OPENSSL
/* ====================================================== */

#ifdef SSL_CTRL_SET_TLSEXT_HOSTNAME
#  define use_sni(x)  sni = (x)
#else
#  define use_sni(x)  Curl_nop_stmt
#endif

/* Check for OpenSSL 1.0.2 which has ALPN support. */
#undef HAS_ALPN
#if OPENSSL_VERSION_NUMBER >= 0x10002000L \
    && !defined(OPENSSL_NO_TLSEXT)
#  define HAS_ALPN 1
#endif

/* Check for OpenSSL 1.0.1 which has NPN support. */
#undef HAS_NPN
#if OPENSSL_VERSION_NUMBER >= 0x10001000L \
    && !defined(OPENSSL_NO_TLSEXT) \
    && !defined(OPENSSL_NO_NEXTPROTONEG)
#  define HAS_NPN 1
#endif

#ifdef HAS_NPN

/*
 * in is a list of length prefixed strings. this function has to select
 * the protocol we want to use from the list and write its string into out.
 */

static int
select_next_protocol(unsigned char **out, unsigned char *outlen,
                     const unsigned char *in, unsigned int inlen,
                     const char *key, unsigned int keylen)
{
  unsigned int i;
  for(i = 0; i + keylen <= inlen; i += in[i] + 1) {
    if(memcmp(&in[i + 1], key, keylen) == 0) {
      *out = (unsigned char *) &in[i + 1];
      *outlen = in[i];
      return 0;
    }
  }
  return -1;
}

static int
select_next_proto_cb(SSL *ssl,
                     unsigned char **out, unsigned char *outlen,
                     const unsigned char *in, unsigned int inlen,
                     void *arg)
{
  struct Curl_easy *data = (struct Curl_easy *)arg;
  struct connectdata *conn = data->conn;
  (void)ssl;

#ifdef USE_HTTP2
  if(data->state.httpwant >= CURL_HTTP_VERSION_2 &&
     !select_next_protocol(out, outlen, in, inlen, ALPN_H2, ALPN_H2_LENGTH)) {
    infof(data, "NPN, negotiated HTTP2 (%s)", ALPN_H2);
    conn->negnpn = CURL_HTTP_VERSION_2;
    return SSL_TLSEXT_ERR_OK;
  }
#endif

  if(!select_next_protocol(out, outlen, in, inlen, ALPN_HTTP_1_1,
                           ALPN_HTTP_1_1_LENGTH)) {
    infof(data, "NPN, negotiated HTTP1.1");
    conn->negnpn = CURL_HTTP_VERSION_1_1;
    return SSL_TLSEXT_ERR_OK;
  }

  infof(data, "NPN, no overlap, use HTTP1.1");
  *out = (unsigned char *)ALPN_HTTP_1_1;
  *outlen = ALPN_HTTP_1_1_LENGTH;
  conn->negnpn = CURL_HTTP_VERSION_1_1;

  return SSL_TLSEXT_ERR_OK;
}
#endif /* HAS_NPN */

#if (OPENSSL_VERSION_NUMBER >= 0x10100000L) /* 1.1.0 */
static CURLcode
set_ssl_version_min_max(SSL_CTX *ctx, struct connectdata *conn)
{
  /* first, TLS min version... */
  long curl_ssl_version_min = SSL_CONN_CONFIG(version);
  long curl_ssl_version_max;

  /* convert curl min SSL version option to OpenSSL constant */
#if defined(OPENSSL_IS_BORINGSSL) || defined(LIBRESSL_VERSION_NUMBER)
  uint16_t ossl_ssl_version_min = 0;
  uint16_t ossl_ssl_version_max = 0;
#else
  long ossl_ssl_version_min = 0;
  long ossl_ssl_version_max = 0;
#endif
  switch(curl_ssl_version_min) {
    case CURL_SSLVERSION_TLSv1: /* TLS 1.x */
    case CURL_SSLVERSION_TLSv1_0:
      ossl_ssl_version_min = TLS1_VERSION;
      break;
    case CURL_SSLVERSION_TLSv1_1:
      ossl_ssl_version_min = TLS1_1_VERSION;
      break;
    case CURL_SSLVERSION_TLSv1_2:
      ossl_ssl_version_min = TLS1_2_VERSION;
      break;
#ifdef TLS1_3_VERSION
    case CURL_SSLVERSION_TLSv1_3:
      ossl_ssl_version_min = TLS1_3_VERSION;
      break;
#endif
  }

  /* CURL_SSLVERSION_DEFAULT means that no option was selected.
     We don't want to pass 0 to SSL_CTX_set_min_proto_version as
     it would enable all versions down to the lowest supported by
     the library.
     So we skip this, and stay with the library default
  */
  if(curl_ssl_version_min != CURL_SSLVERSION_DEFAULT) {
    if(!SSL_CTX_set_min_proto_version(ctx, ossl_ssl_version_min)) {
      return CURLE_SSL_CONNECT_ERROR;
    }
  }

  /* ... then, TLS max version */
  curl_ssl_version_max = SSL_CONN_CONFIG(version_max);

  /* convert curl max SSL version option to OpenSSL constant */
  switch(curl_ssl_version_max) {
    case CURL_SSLVERSION_MAX_TLSv1_0:
      ossl_ssl_version_max = TLS1_VERSION;
      break;
    case CURL_SSLVERSION_MAX_TLSv1_1:
      ossl_ssl_version_max = TLS1_1_VERSION;
      break;
    case CURL_SSLVERSION_MAX_TLSv1_2:
      ossl_ssl_version_max = TLS1_2_VERSION;
      break;
#ifdef TLS1_3_VERSION
    case CURL_SSLVERSION_MAX_TLSv1_3:
      ossl_ssl_version_max = TLS1_3_VERSION;
      break;
#endif
    case CURL_SSLVERSION_MAX_NONE:  /* none selected */
    case CURL_SSLVERSION_MAX_DEFAULT:  /* max selected */
    default:
      /* SSL_CTX_set_max_proto_version states that:
        setting the maximum to 0 will enable
        protocol versions up to the highest version
        supported by the library */
      ossl_ssl_version_max = 0;
      break;
  }

  if(!SSL_CTX_set_max_proto_version(ctx, ossl_ssl_version_max)) {
    return CURLE_SSL_CONNECT_ERROR;
  }

  return CURLE_OK;
}
#endif

#ifdef OPENSSL_IS_BORINGSSL
typedef uint32_t ctx_option_t;
#else
typedef long ctx_option_t;
#endif

#if (OPENSSL_VERSION_NUMBER < 0x10100000L) /* 1.1.0 */
static CURLcode
set_ssl_version_min_max_legacy(ctx_option_t *ctx_options,
                               struct Curl_easy *data,
                               struct connectdata *conn, int sockindex)
{
  long ssl_version = SSL_CONN_CONFIG(version);
  long ssl_version_max = SSL_CONN_CONFIG(version_max);

  (void) data; /* In case it's unused. */

  switch(ssl_version) {
    case CURL_SSLVERSION_TLSv1_3:
#ifdef TLS1_3_VERSION
    {
      struct ssl_connect_data *connssl = &conn->ssl[sockindex];
      SSL_CTX_set_max_proto_version(backend->ctx, TLS1_3_VERSION);
      *ctx_options |= SSL_OP_NO_TLSv1_2;
    }
#else
      (void)sockindex;
      (void)ctx_options;
      failf(data, OSSL_PACKAGE " was built without TLS 1.3 support");
      return CURLE_NOT_BUILT_IN;
#endif
      /* FALLTHROUGH */
    case CURL_SSLVERSION_TLSv1_2:
#if OPENSSL_VERSION_NUMBER >= 0x1000100FL
      *ctx_options |= SSL_OP_NO_TLSv1_1;
#else
      failf(data, OSSL_PACKAGE " was built without TLS 1.2 support");
      return CURLE_NOT_BUILT_IN;
#endif
      /* FALLTHROUGH */
    case CURL_SSLVERSION_TLSv1_1:
#if OPENSSL_VERSION_NUMBER >= 0x1000100FL
      *ctx_options |= SSL_OP_NO_TLSv1;
#else
      failf(data, OSSL_PACKAGE " was built without TLS 1.1 support");
      return CURLE_NOT_BUILT_IN;
#endif
      /* FALLTHROUGH */
    case CURL_SSLVERSION_TLSv1_0:
    case CURL_SSLVERSION_TLSv1:
      break;
  }

  switch(ssl_version_max) {
    case CURL_SSLVERSION_MAX_TLSv1_0:
#if OPENSSL_VERSION_NUMBER >= 0x1000100FL
      *ctx_options |= SSL_OP_NO_TLSv1_1;
#endif
      /* FALLTHROUGH */
    case CURL_SSLVERSION_MAX_TLSv1_1:
#if OPENSSL_VERSION_NUMBER >= 0x1000100FL
      *ctx_options |= SSL_OP_NO_TLSv1_2;
#endif
      /* FALLTHROUGH */
    case CURL_SSLVERSION_MAX_TLSv1_2:
#ifdef TLS1_3_VERSION
      *ctx_options |= SSL_OP_NO_TLSv1_3;
#endif
      break;
    case CURL_SSLVERSION_MAX_TLSv1_3:
#ifdef TLS1_3_VERSION
      break;
#else
      failf(data, OSSL_PACKAGE " was built without TLS 1.3 support");
      return CURLE_NOT_BUILT_IN;
#endif
  }
  return CURLE_OK;
}
#endif

/* The "new session" callback must return zero if the session can be removed
 * or non-zero if the session has been put into the session cache.
 */
static int ossl_new_session_cb(SSL *ssl, SSL_SESSION *ssl_sessionid)
{
  int res = 0;
  struct connectdata *conn;
  struct Curl_easy *data;
  int sockindex;
  curl_socket_t *sockindex_ptr;
  int data_idx = ossl_get_ssl_data_index();
  int connectdata_idx = ossl_get_ssl_conn_index();
  int sockindex_idx = ossl_get_ssl_sockindex_index();
  int proxy_idx = ossl_get_proxy_index();
  bool isproxy;

  if(data_idx < 0 || connectdata_idx < 0 || sockindex_idx < 0 || proxy_idx < 0)
    return 0;

  conn = (struct connectdata*) SSL_get_ex_data(ssl, connectdata_idx);
  if(!conn)
    return 0;

  data = (struct Curl_easy *) SSL_get_ex_data(ssl, data_idx);

  /* The sockindex has been stored as a pointer to an array element */
  sockindex_ptr = (curl_socket_t*) SSL_get_ex_data(ssl, sockindex_idx);
  sockindex = (int)(sockindex_ptr - conn->sock);

  isproxy = SSL_get_ex_data(ssl, proxy_idx) ? TRUE : FALSE;

  if(SSL_SET_OPTION(primary.sessionid)) {
    bool incache;
    bool added = FALSE;
    void *old_ssl_sessionid = NULL;

    Curl_ssl_sessionid_lock(data);
    if(isproxy)
      incache = FALSE;
    else
      incache = !(Curl_ssl_getsessionid(data, conn, isproxy,
                                        &old_ssl_sessionid, NULL, sockindex));
    if(incache) {
      if(old_ssl_sessionid != ssl_sessionid) {
        infof(data, "old SSL session ID is stale, removing");
        Curl_ssl_delsessionid(data, old_ssl_sessionid);
        incache = FALSE;
      }
    }

    if(!incache) {
      if(!Curl_ssl_addsessionid(data, conn, isproxy, ssl_sessionid,
                                0 /* unknown size */, sockindex, &added)) {
        if(added) {
          /* the session has been put into the session cache */
          res = 1;
        }
      }
      else
        failf(data, "failed to store ssl session");
    }
    Curl_ssl_sessionid_unlock(data);
  }

  return res;
}

static CURLcode load_cacert_from_memory(SSL_CTX *ctx,
                                        const struct curl_blob *ca_info_blob)
{
  /* these need to be freed at the end */
  BIO *cbio = NULL;
  STACK_OF(X509_INFO) *inf = NULL;

  /* everything else is just a reference */
  int i, count = 0;
  X509_STORE *cts = NULL;
  X509_INFO *itmp = NULL;

  if(ca_info_blob->len > (size_t)INT_MAX)
    return CURLE_SSL_CACERT_BADFILE;

  cts = SSL_CTX_get_cert_store(ctx);
  if(!cts)
    return CURLE_OUT_OF_MEMORY;

  cbio = BIO_new_mem_buf(ca_info_blob->data, (int)ca_info_blob->len);
  if(!cbio)
    return CURLE_OUT_OF_MEMORY;

  inf = PEM_X509_INFO_read_bio(cbio, NULL, NULL, NULL);
  if(!inf) {
    BIO_free(cbio);
    return CURLE_SSL_CACERT_BADFILE;
  }

  /* add each entry from PEM file to x509_store */
  for(i = 0; i < (int)sk_X509_INFO_num(inf); ++i) {
    itmp = sk_X509_INFO_value(inf, i);
    if(itmp->x509) {
      if(X509_STORE_add_cert(cts, itmp->x509)) {
        ++count;
      }
      else {
        /* set count to 0 to return an error */
        count = 0;
        break;
      }
    }
    if(itmp->crl) {
      if(X509_STORE_add_crl(cts, itmp->crl)) {
        ++count;
      }
      else {
        /* set count to 0 to return an error */
        count = 0;
        break;
      }
    }
  }

  sk_X509_INFO_pop_free(inf, X509_INFO_free);
  BIO_free(cbio);

  /* if we didn't end up importing anything, treat that as an error */
  return (count > 0 ? CURLE_OK : CURLE_SSL_CACERT_BADFILE);
}

static CURLcode ossl_connect_step1(struct Curl_easy *data,
                                   struct connectdata *conn, int sockindex)
{
  CURLcode result = CURLE_OK;
  char *ciphers;
  SSL_METHOD_QUAL SSL_METHOD *req_method = NULL;
  X509_LOOKUP *lookup = NULL;
  curl_socket_t sockfd = conn->sock[sockindex];
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  ctx_option_t ctx_options = 0;
  void *ssl_sessionid = NULL;

#ifdef SSL_CTRL_SET_TLSEXT_HOSTNAME
  bool sni;
  const char * const hostname = SSL_HOST_NAME();

#ifdef ENABLE_IPV6
  struct in6_addr addr;
#else
  struct in_addr addr;
#endif
#endif
  const long int ssl_version = SSL_CONN_CONFIG(version);
#ifdef USE_OPENSSL_SRP
  const enum CURL_TLSAUTH ssl_authtype = SSL_SET_OPTION(authtype);
#endif
  char * const ssl_cert = SSL_SET_OPTION(primary.clientcert);
  const struct curl_blob *ssl_cert_blob = SSL_SET_OPTION(primary.cert_blob);
  const struct curl_blob *ca_info_blob = SSL_CONN_CONFIG(ca_info_blob);
  const char * const ssl_cert_type = SSL_SET_OPTION(cert_type);
  const char * const ssl_cafile =
    /* CURLOPT_CAINFO_BLOB overrides CURLOPT_CAINFO */
    (ca_info_blob ? NULL : SSL_CONN_CONFIG(CAfile));
  const char * const ssl_capath = SSL_CONN_CONFIG(CApath);
  const bool verifypeer = SSL_CONN_CONFIG(verifypeer);
  const char * const ssl_crlfile = SSL_SET_OPTION(CRLfile);
  char error_buffer[256];
  struct ssl_backend_data *backend = connssl->backend;
  bool imported_native_ca = false;

  DEBUGASSERT(ssl_connect_1 == connssl->connecting_state);

  /* Make funny stuff to get random input */
  result = ossl_seed(data);
  if(result)
    return result;

  SSL_SET_OPTION_LVALUE(certverifyresult) = !X509_V_OK;

  /* check to see if we've been told to use an explicit SSL/TLS version */

  switch(ssl_version) {
  case CURL_SSLVERSION_DEFAULT:
  case CURL_SSLVERSION_TLSv1:
  case CURL_SSLVERSION_TLSv1_0:
  case CURL_SSLVERSION_TLSv1_1:
  case CURL_SSLVERSION_TLSv1_2:
  case CURL_SSLVERSION_TLSv1_3:
    /* it will be handled later with the context options */
#if (OPENSSL_VERSION_NUMBER >= 0x10100000L)
    req_method = TLS_client_method();
#else
    req_method = SSLv23_client_method();
#endif
    use_sni(TRUE);
    break;
  case CURL_SSLVERSION_SSLv2:
    failf(data, "No SSLv2 support");
    return CURLE_NOT_BUILT_IN;
  case CURL_SSLVERSION_SSLv3:
    failf(data, "No SSLv3 support");
    return CURLE_NOT_BUILT_IN;
  default:
    failf(data, "Unrecognized parameter passed via CURLOPT_SSLVERSION");
    return CURLE_SSL_CONNECT_ERROR;
  }

  DEBUGASSERT(!backend->ctx);
  backend->ctx = SSL_CTX_new(req_method);

  if(!backend->ctx) {
    failf(data, "SSL: couldn't create a context: %s",
          ossl_strerror(ERR_peek_error(), error_buffer, sizeof(error_buffer)));
    return CURLE_OUT_OF_MEMORY;
  }

#ifdef SSL_MODE_RELEASE_BUFFERS
  SSL_CTX_set_mode(backend->ctx, SSL_MODE_RELEASE_BUFFERS);
#endif

#ifdef SSL_CTRL_SET_MSG_CALLBACK
  if(data->set.fdebug && data->set.verbose) {
    /* the SSL trace callback is only used for verbose logging */
    SSL_CTX_set_msg_callback(backend->ctx, ossl_trace);
    SSL_CTX_set_msg_callback_arg(backend->ctx, conn);
    set_logger(conn, data);
  }
#endif

  /* OpenSSL contains code to work around lots of bugs and flaws in various
     SSL-implementations. SSL_CTX_set_options() is used to enabled those
     work-arounds. The man page for this option states that SSL_OP_ALL enables
     all the work-arounds and that "It is usually safe to use SSL_OP_ALL to
     enable the bug workaround options if compatibility with somewhat broken
     implementations is desired."

     The "-no_ticket" option was introduced in OpenSSL 0.9.8j. It's a flag to
     disable "rfc4507bis session ticket support".  rfc4507bis was later turned
     into the proper RFC5077 it seems: https://tools.ietf.org/html/rfc5077

     The enabled extension concerns the session management. I wonder how often
     libcurl stops a connection and then resumes a TLS session. Also, sending
     the session data is some overhead. I suggest that you just use your
     proposed patch (which explicitly disables TICKET).

     If someone writes an application with libcurl and OpenSSL who wants to
     enable the feature, one can do this in the SSL callback.

     SSL_OP_NETSCAPE_REUSE_CIPHER_CHANGE_BUG option enabling allowed proper
     interoperability with web server Netscape Enterprise Server 2.0.1 which
     was released back in 1996.

     Due to CVE-2010-4180, option SSL_OP_NETSCAPE_REUSE_CIPHER_CHANGE_BUG has
     become ineffective as of OpenSSL 0.9.8q and 1.0.0c. In order to mitigate
     CVE-2010-4180 when using previous OpenSSL versions we no longer enable
     this option regardless of OpenSSL version and SSL_OP_ALL definition.

     OpenSSL added a work-around for a SSL 3.0/TLS 1.0 CBC vulnerability
     (https://www.openssl.org/~bodo/tls-cbc.txt). In 0.9.6e they added a bit to
     SSL_OP_ALL that _disables_ that work-around despite the fact that
     SSL_OP_ALL is documented to do "rather harmless" workarounds. In order to
     keep the secure work-around, the SSL_OP_DONT_INSERT_EMPTY_FRAGMENTS bit
     must not be set.
  */

  ctx_options = SSL_OP_ALL;

#ifdef SSL_OP_NO_TICKET
  ctx_options |= SSL_OP_NO_TICKET;
#endif

#ifdef SSL_OP_NO_COMPRESSION
  ctx_options |= SSL_OP_NO_COMPRESSION;
#endif

#ifdef SSL_OP_NETSCAPE_REUSE_CIPHER_CHANGE_BUG
  /* mitigate CVE-2010-4180 */
  ctx_options &= ~SSL_OP_NETSCAPE_REUSE_CIPHER_CHANGE_BUG;
#endif

#ifdef SSL_OP_DONT_INSERT_EMPTY_FRAGMENTS
  /* unless the user explicitly asks to allow the protocol vulnerability we
     use the work-around */
  if(!SSL_SET_OPTION(enable_beast))
    ctx_options &= ~SSL_OP_DONT_INSERT_EMPTY_FRAGMENTS;
#endif

  switch(ssl_version) {
    case CURL_SSLVERSION_SSLv2:
    case CURL_SSLVERSION_SSLv3:
      return CURLE_NOT_BUILT_IN;

    /* "--tlsv<x.y>" options mean TLS >= version <x.y> */
    case CURL_SSLVERSION_DEFAULT:
    case CURL_SSLVERSION_TLSv1: /* TLS >= version 1.0 */
    case CURL_SSLVERSION_TLSv1_0: /* TLS >= version 1.0 */
    case CURL_SSLVERSION_TLSv1_1: /* TLS >= version 1.1 */
    case CURL_SSLVERSION_TLSv1_2: /* TLS >= version 1.2 */
    case CURL_SSLVERSION_TLSv1_3: /* TLS >= version 1.3 */
      /* asking for any TLS version as the minimum, means no SSL versions
        allowed */
      ctx_options |= SSL_OP_NO_SSLv2;
      ctx_options |= SSL_OP_NO_SSLv3;

#if (OPENSSL_VERSION_NUMBER >= 0x10100000L) /* 1.1.0 */
      result = set_ssl_version_min_max(backend->ctx, conn);
#else
      result = set_ssl_version_min_max_legacy(&ctx_options, data, conn,
                                              sockindex);
#endif
      if(result != CURLE_OK)
        return result;
      break;

    default:
      failf(data, "Unrecognized parameter passed via CURLOPT_SSLVERSION");
      return CURLE_SSL_CONNECT_ERROR;
  }

  SSL_CTX_set_options(backend->ctx, ctx_options);

#ifdef HAS_NPN
  if(conn->bits.tls_enable_npn)
    SSL_CTX_set_next_proto_select_cb(backend->ctx, select_next_proto_cb, data);
#endif

#ifdef HAS_ALPN
  if(conn->bits.tls_enable_alpn) {
    int cur = 0;
    unsigned char protocols[128];

#ifdef USE_HTTP2
    if(data->state.httpwant >= CURL_HTTP_VERSION_2
#ifndef CURL_DISABLE_PROXY
       && (!SSL_IS_PROXY() || !conn->bits.tunnel_proxy)
#endif
      ) {
      protocols[cur++] = ALPN_H2_LENGTH;

      memcpy(&protocols[cur], ALPN_H2, ALPN_H2_LENGTH);
      cur += ALPN_H2_LENGTH;
      infof(data, "ALPN, offering %s", ALPN_H2);
    }
#endif

    protocols[cur++] = ALPN_HTTP_1_1_LENGTH;
    memcpy(&protocols[cur], ALPN_HTTP_1_1, ALPN_HTTP_1_1_LENGTH);
    cur += ALPN_HTTP_1_1_LENGTH;
    infof(data, "ALPN, offering %s", ALPN_HTTP_1_1);

    /* expects length prefixed preference ordered list of protocols in wire
     * format
     */
    if(SSL_CTX_set_alpn_protos(backend->ctx, protocols, cur)) {
      failf(data, "Error setting ALPN");
      return CURLE_SSL_CONNECT_ERROR;
    }
  }
#endif

  if(ssl_cert || ssl_cert_blob || ssl_cert_type) {
    if(!result &&
       !cert_stuff(data, backend->ctx,
                   ssl_cert, ssl_cert_blob, ssl_cert_type,
                   SSL_SET_OPTION(key), SSL_SET_OPTION(key_blob),
                   SSL_SET_OPTION(key_type), SSL_SET_OPTION(key_passwd)))
      result = CURLE_SSL_CERTPROBLEM;
    if(result)
      /* failf() is already done in cert_stuff() */
      return result;
  }

  ciphers = SSL_CONN_CONFIG(cipher_list);
  if(!ciphers)
    ciphers = (char *)DEFAULT_CIPHER_SELECTION;
  if(ciphers) {
    if(!SSL_CTX_set_cipher_list(backend->ctx, ciphers)) {
      failf(data, "failed setting cipher list: %s", ciphers);
      return CURLE_SSL_CIPHER;
    }
    infof(data, "Cipher selection: %s", ciphers);
  }

#ifdef HAVE_SSL_CTX_SET_CIPHERSUITES
  {
    char *ciphers13 = SSL_CONN_CONFIG(cipher_list13);
    if(ciphers13) {
      if(!SSL_CTX_set_ciphersuites(backend->ctx, ciphers13)) {
        failf(data, "failed setting TLS 1.3 cipher suite: %s", ciphers13);
        return CURLE_SSL_CIPHER;
      }
      infof(data, "TLS 1.3 cipher selection: %s", ciphers13);
    }
  }
#endif

#ifdef HAVE_SSL_CTX_SET_POST_HANDSHAKE_AUTH
  /* OpenSSL 1.1.1 requires clients to opt-in for PHA */
  SSL_CTX_set_post_handshake_auth(backend->ctx, 1);
#endif

#ifdef HAVE_SSL_CTX_SET_EC_CURVES
  {
    char *curves = SSL_CONN_CONFIG(curves);
    if(curves) {
      if(!SSL_CTX_set1_curves_list(backend->ctx, curves)) {
        failf(data, "failed setting curves list: '%s'", curves);
        return CURLE_SSL_CIPHER;
      }
    }
  }
#endif

#ifdef USE_OPENSSL_SRP
  if(ssl_authtype == CURL_TLSAUTH_SRP) {
    char * const ssl_username = SSL_SET_OPTION(username);

    infof(data, "Using TLS-SRP username: %s", ssl_username);

    if(!SSL_CTX_set_srp_username(backend->ctx, ssl_username)) {
      failf(data, "Unable to set SRP user name");
      return CURLE_BAD_FUNCTION_ARGUMENT;
    }
    if(!SSL_CTX_set_srp_password(backend->ctx, SSL_SET_OPTION(password))) {
      failf(data, "failed setting SRP password");
      return CURLE_BAD_FUNCTION_ARGUMENT;
    }
    if(!SSL_CONN_CONFIG(cipher_list)) {
      infof(data, "Setting cipher list SRP");

      if(!SSL_CTX_set_cipher_list(backend->ctx, "SRP")) {
        failf(data, "failed setting SRP cipher list");
        return CURLE_SSL_CIPHER;
      }
    }
  }
#endif


#if defined(USE_WIN32_CRYPTO)
  /* Import certificates from the Windows root certificate store if requested.
     https://stackoverflow.com/questions/9507184/
     https://github.com/d3x0r/SACK/blob/master/src/netlib/ssl_layer.c#L1037
     https://tools.ietf.org/html/rfc5280 */
  if((SSL_CONN_CONFIG(verifypeer) || SSL_CONN_CONFIG(verifyhost)) &&
     (SSL_SET_OPTION(native_ca_store))) {
    X509_STORE *store = SSL_CTX_get_cert_store(backend->ctx);
    HCERTSTORE hStore = CertOpenSystemStore(0, TEXT("ROOT"));

    if(hStore) {
      PCCERT_CONTEXT pContext = NULL;
      /* The array of enhanced key usage OIDs will vary per certificate and is
         declared outside of the loop so that rather than malloc/free each
         iteration we can grow it with realloc, when necessary. */
      CERT_ENHKEY_USAGE *enhkey_usage = NULL;
      DWORD enhkey_usage_size = 0;

      /* This loop makes a best effort to import all valid certificates from
         the MS root store. If a certificate cannot be imported it is skipped.
         'result' is used to store only hard-fail conditions (such as out of
         memory) that cause an early break. */
      result = CURLE_OK;
      for(;;) {
        X509 *x509;
        FILETIME now;
        BYTE key_usage[2];
        DWORD req_size;
        const unsigned char *encoded_cert;
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
        char cert_name[256];
#endif

        pContext = CertEnumCertificatesInStore(hStore, pContext);
        if(!pContext)
          break;

#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
        if(!CertGetNameStringA(pContext, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0,
                               NULL, cert_name, sizeof(cert_name))) {
          strcpy(cert_name, "Unknown");
        }
        infof(data, "SSL: Checking cert \"%s\"", cert_name);
#endif

        encoded_cert = (const unsigned char *)pContext->pbCertEncoded;
        if(!encoded_cert)
          continue;

        GetSystemTimeAsFileTime(&now);
        if(CompareFileTime(&pContext->pCertInfo->NotBefore, &now) > 0 ||
           CompareFileTime(&now, &pContext->pCertInfo->NotAfter) > 0)
          continue;

        /* If key usage exists check for signing attribute */
        if(CertGetIntendedKeyUsage(pContext->dwCertEncodingType,
                                   pContext->pCertInfo,
                                   key_usage, sizeof(key_usage))) {
          if(!(key_usage[0] & CERT_KEY_CERT_SIGN_KEY_USAGE))
            continue;
        }
        else if(GetLastError())
          continue;

        /* If enhanced key usage exists check for server auth attribute.
         *
         * Note "In a Microsoft environment, a certificate might also have EKU
         * extended properties that specify valid uses for the certificate."
         * The call below checks both, and behavior varies depending on what is
         * found. For more details see CertGetEnhancedKeyUsage doc.
         */
        if(CertGetEnhancedKeyUsage(pContext, 0, NULL, &req_size)) {
          if(req_size && req_size > enhkey_usage_size) {
            void *tmp = realloc(enhkey_usage, req_size);

            if(!tmp) {
              failf(data, "SSL: Out of memory allocating for OID list");
              result = CURLE_OUT_OF_MEMORY;
              break;
            }

            enhkey_usage = (CERT_ENHKEY_USAGE *)tmp;
            enhkey_usage_size = req_size;
          }

          if(CertGetEnhancedKeyUsage(pContext, 0, enhkey_usage, &req_size)) {
            if(!enhkey_usage->cUsageIdentifier) {
              /* "If GetLastError returns CRYPT_E_NOT_FOUND, the certificate is
                 good for all uses. If it returns zero, the certificate has no
                 valid uses." */
              if((HRESULT)GetLastError() != CRYPT_E_NOT_FOUND)
                continue;
            }
            else {
              DWORD i;
              bool found = false;

              for(i = 0; i < enhkey_usage->cUsageIdentifier; ++i) {
                if(!strcmp("1.3.6.1.5.5.7.3.1" /* OID server auth */,
                           enhkey_usage->rgpszUsageIdentifier[i])) {
                  found = true;
                  break;
                }
              }

              if(!found)
                continue;
            }
          }
          else
            continue;
        }
        else
          continue;

        x509 = d2i_X509(NULL, &encoded_cert, pContext->cbCertEncoded);
        if(!x509)
          continue;

        /* Try to import the certificate. This may fail for legitimate reasons
           such as duplicate certificate, which is allowed by MS but not
           OpenSSL. */
        if(X509_STORE_add_cert(store, x509) == 1) {
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
          infof(data, "SSL: Imported cert \"%s\"", cert_name);
#endif
          imported_native_ca = true;
        }
        X509_free(x509);
      }

      free(enhkey_usage);
      CertFreeCertificateContext(pContext);
      CertCloseStore(hStore, 0);

      if(result)
        return result;
    }
    if(imported_native_ca)
      infof(data, "successfully imported Windows CA store");
    else
      infof(data, "error importing Windows CA store, continuing anyway");
  }
#endif

  if(ca_info_blob) {
    result = load_cacert_from_memory(backend->ctx, ca_info_blob);
    if(result) {
      if(result == CURLE_OUT_OF_MEMORY ||
         (verifypeer && !imported_native_ca)) {
        failf(data, "error importing CA certificate blob");
        return result;
      }
      /* Only warn if no certificate verification is required. */
      infof(data, "error importing CA certificate blob, continuing anyway");
    }
  }

  if(verifypeer && !imported_native_ca && (ssl_cafile || ssl_capath)) {
#if defined(OPENSSL_VERSION_MAJOR) && (OPENSSL_VERSION_MAJOR >= 3)
  /* OpenSSL 3.0.0 has deprecated SSL_CTX_load_verify_locations */
    if(ssl_cafile &&
       !SSL_CTX_load_verify_file(backend->ctx, ssl_cafile)) {
      /* Fail if we insist on successfully verifying the server. */
      failf(data, "error setting certificate file: %s", ssl_cafile);
      return CURLE_SSL_CACERT_BADFILE;
    }
    if(ssl_capath &&
       !SSL_CTX_load_verify_dir(backend->ctx, ssl_capath)) {
      /* Fail if we insist on successfully verifying the server. */
      failf(data, "error setting certificate path: %s", ssl_capath);
      return CURLE_SSL_CACERT_BADFILE;
    }
#else
    /* tell OpenSSL where to find CA certificates that are used to verify the
       server's certificate. */
    if(!SSL_CTX_load_verify_locations(backend->ctx, ssl_cafile, ssl_capath)) {
      /* Fail if we insist on successfully verifying the server. */
      failf(data, "error setting certificate verify locations:"
            "  CAfile: %s CApath: %s",
            ssl_cafile ? ssl_cafile : "none",
            ssl_capath ? ssl_capath : "none");
      return CURLE_SSL_CACERT_BADFILE;
    }
#endif
    infof(data, " CAfile: %s", ssl_cafile ? ssl_cafile : "none");
    infof(data, " CApath: %s", ssl_capath ? ssl_capath : "none");
  }

#ifdef CURL_CA_FALLBACK
  if(verifypeer &&
     !ca_info_blob && !ssl_cafile && !ssl_capath && !imported_native_ca) {
    /* verifying the peer without any CA certificates won't
       work so use openssl's built-in default as fallback */
    SSL_CTX_set_default_verify_paths(backend->ctx);
  }
#endif

  if(ssl_crlfile) {
    /* tell OpenSSL where to find CRL file that is used to check certificate
     * revocation */
    lookup = X509_STORE_add_lookup(SSL_CTX_get_cert_store(backend->ctx),
                                 X509_LOOKUP_file());
    if(!lookup ||
       (!X509_load_crl_file(lookup, ssl_crlfile, X509_FILETYPE_PEM)) ) {
      failf(data, "error loading CRL file: %s", ssl_crlfile);
      return CURLE_SSL_CRL_BADFILE;
    }
    /* Everything is fine. */
    infof(data, "successfully loaded CRL file:");
    X509_STORE_set_flags(SSL_CTX_get_cert_store(backend->ctx),
                         X509_V_FLAG_CRL_CHECK|X509_V_FLAG_CRL_CHECK_ALL);

    infof(data, "  CRLfile: %s", ssl_crlfile);
  }

  if(verifypeer) {
    /* Try building a chain using issuers in the trusted store first to avoid
       problems with server-sent legacy intermediates.  Newer versions of
       OpenSSL do alternate chain checking by default but we do not know how to
       determine that in a reliable manner.
       https://rt.openssl.org/Ticket/Display.html?id=3621&user=guest&pass=guest
    */
#if defined(X509_V_FLAG_TRUSTED_FIRST)
    X509_STORE_set_flags(SSL_CTX_get_cert_store(backend->ctx),
                         X509_V_FLAG_TRUSTED_FIRST);
#endif
#ifdef X509_V_FLAG_PARTIAL_CHAIN
    if(!SSL_SET_OPTION(no_partialchain) && !ssl_crlfile) {
      /* Have intermediate certificates in the trust store be treated as
         trust-anchors, in the same way as self-signed root CA certificates
         are. This allows users to verify servers using the intermediate cert
         only, instead of needing the whole chain.

         Due to OpenSSL bug https://github.com/openssl/openssl/issues/5081 we
         cannot do partial chains with a CRL check.
      */
      X509_STORE_set_flags(SSL_CTX_get_cert_store(backend->ctx),
                           X509_V_FLAG_PARTIAL_CHAIN);
    }
#endif
  }

  /* OpenSSL always tries to verify the peer, this only says whether it should
   * fail to connect if the verification fails, or if it should continue
   * anyway. In the latter case the result of the verification is checked with
   * SSL_get_verify_result() below. */
  SSL_CTX_set_verify(backend->ctx,
                     verifypeer ? SSL_VERIFY_PEER : SSL_VERIFY_NONE, NULL);

  /* Enable logging of secrets to the file specified in env SSLKEYLOGFILE. */
#ifdef HAVE_KEYLOG_CALLBACK
  if(Curl_tls_keylog_enabled()) {
    SSL_CTX_set_keylog_callback(backend->ctx, ossl_keylog_callback);
  }
#endif

  /* Enable the session cache because it's a prerequisite for the "new session"
   * callback. Use the "external storage" mode to prevent OpenSSL from creating
   * an internal session cache.
   */
  SSL_CTX_set_session_cache_mode(backend->ctx,
      SSL_SESS_CACHE_CLIENT | SSL_SESS_CACHE_NO_INTERNAL);
  SSL_CTX_sess_set_new_cb(backend->ctx, ossl_new_session_cb);

  /* give application a chance to interfere with SSL set up. */
  if(data->set.ssl.fsslctx) {
    Curl_set_in_callback(data, true);
    result = (*data->set.ssl.fsslctx)(data, backend->ctx,
                                      data->set.ssl.fsslctxp);
    Curl_set_in_callback(data, false);
    if(result) {
      failf(data, "error signaled by ssl ctx callback");
      return result;
    }
  }

  /* Let's make an SSL structure */
  if(backend->handle)
    SSL_free(backend->handle);
  backend->handle = SSL_new(backend->ctx);
  if(!backend->handle) {
    failf(data, "SSL: couldn't create a context (handle)!");
    return CURLE_OUT_OF_MEMORY;
  }

#if (OPENSSL_VERSION_NUMBER >= 0x0090808fL) && !defined(OPENSSL_NO_TLSEXT) && \
    !defined(OPENSSL_NO_OCSP)
  if(SSL_CONN_CONFIG(verifystatus))
    SSL_set_tlsext_status_type(backend->handle, TLSEXT_STATUSTYPE_ocsp);
#endif

#if defined(OPENSSL_IS_BORINGSSL) && defined(ALLOW_RENEG)
  SSL_set_renegotiate_mode(backend->handle, ssl_renegotiate_freely);
#endif

  SSL_set_connect_state(backend->handle);

  backend->server_cert = 0x0;
#ifdef SSL_CTRL_SET_TLSEXT_HOSTNAME
  if((0 == Curl_inet_pton(AF_INET, hostname, &addr)) &&
#ifdef ENABLE_IPV6
     (0 == Curl_inet_pton(AF_INET6, hostname, &addr)) &&
#endif
     sni) {
    size_t nlen = strlen(hostname);
    if((long)nlen >= data->set.buffer_size)
      /* this is seriously messed up */
      return CURLE_SSL_CONNECT_ERROR;

    /* RFC 6066 section 3 says the SNI field is case insensitive, but browsers
       send the data lowercase and subsequently there are now numerous servers
       out there that don't work unless the name is lowercased */
    Curl_strntolower(data->state.buffer, hostname, nlen);
    data->state.buffer[nlen] = 0;
    if(!SSL_set_tlsext_host_name(backend->handle, data->state.buffer))
      infof(data, "WARNING: failed to configure server name indication (SNI) "
            "TLS extension");
  }
#endif

  ossl_associate_connection(data, conn, sockindex);

  Curl_ssl_sessionid_lock(data);
  if(!Curl_ssl_getsessionid(data, conn, SSL_IS_PROXY() ? TRUE : FALSE,
                            &ssl_sessionid, NULL, sockindex)) {
    /* we got a session id, use it! */
    if(!SSL_set_session(backend->handle, ssl_sessionid)) {
      Curl_ssl_sessionid_unlock(data);
      failf(data, "SSL: SSL_set_session failed: %s",
            ossl_strerror(ERR_get_error(), error_buffer,
                          sizeof(error_buffer)));
      return CURLE_SSL_CONNECT_ERROR;
    }
    /* Informational message */
    infof(data, "SSL re-using session ID");
  }
  Curl_ssl_sessionid_unlock(data);

#ifndef CURL_DISABLE_PROXY
  if(conn->proxy_ssl[sockindex].use) {
    BIO *const bio = BIO_new(BIO_f_ssl());
    SSL *handle = conn->proxy_ssl[sockindex].backend->handle;
    DEBUGASSERT(ssl_connection_complete == conn->proxy_ssl[sockindex].state);
    DEBUGASSERT(handle != NULL);
    DEBUGASSERT(bio != NULL);
    BIO_set_ssl(bio, handle, FALSE);
    SSL_set_bio(backend->handle, bio, bio);
  }
  else
#endif
    if(!SSL_set_fd(backend->handle, (int)sockfd)) {
    /* pass the raw socket into the SSL layers */
    failf(data, "SSL: SSL_set_fd failed: %s",
          ossl_strerror(ERR_get_error(), error_buffer, sizeof(error_buffer)));
    return CURLE_SSL_CONNECT_ERROR;
  }

  connssl->connecting_state = ssl_connect_2;

  return CURLE_OK;
}

static CURLcode ossl_connect_step2(struct Curl_easy *data,
                                   struct connectdata *conn, int sockindex)
{
  int err;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  DEBUGASSERT(ssl_connect_2 == connssl->connecting_state
              || ssl_connect_2_reading == connssl->connecting_state
              || ssl_connect_2_writing == connssl->connecting_state);

  ERR_clear_error();

  err = SSL_connect(backend->handle);
#ifndef HAVE_KEYLOG_CALLBACK
  if(Curl_tls_keylog_enabled()) {
    /* If key logging is enabled, wait for the handshake to complete and then
     * proceed with logging secrets (for TLS 1.2 or older).
     */
    ossl_log_tls12_secret(backend->handle, &backend->keylog_done);
  }
#endif

  /* 1  is fine
     0  is "not successful but was shut down controlled"
     <0 is "handshake was not successful, because a fatal error occurred" */
  if(1 != err) {
    int detail = SSL_get_error(backend->handle, err);

    if(SSL_ERROR_WANT_READ == detail) {
      connssl->connecting_state = ssl_connect_2_reading;
      return CURLE_OK;
    }
    if(SSL_ERROR_WANT_WRITE == detail) {
      connssl->connecting_state = ssl_connect_2_writing;
      return CURLE_OK;
    }
#ifdef SSL_ERROR_WANT_ASYNC
    if(SSL_ERROR_WANT_ASYNC == detail) {
      connssl->connecting_state = ssl_connect_2;
      return CURLE_OK;
    }
#endif
    else {
      /* untreated error */
      unsigned long errdetail;
      char error_buffer[256]="";
      CURLcode result;
      long lerr;
      int lib;
      int reason;

      /* the connection failed, we're not waiting for anything else. */
      connssl->connecting_state = ssl_connect_2;

      /* Get the earliest error code from the thread's error queue and remove
         the entry. */
      errdetail = ERR_get_error();

      /* Extract which lib and reason */
      lib = ERR_GET_LIB(errdetail);
      reason = ERR_GET_REASON(errdetail);

      if((lib == ERR_LIB_SSL) &&
         ((reason == SSL_R_CERTIFICATE_VERIFY_FAILED) ||
          (reason == SSL_R_SSLV3_ALERT_CERTIFICATE_EXPIRED))) {
        result = CURLE_PEER_FAILED_VERIFICATION;

        lerr = SSL_get_verify_result(backend->handle);
        if(lerr != X509_V_OK) {
          SSL_SET_OPTION_LVALUE(certverifyresult) = lerr;
          msnprintf(error_buffer, sizeof(error_buffer),
                    "SSL certificate problem: %s",
                    X509_verify_cert_error_string(lerr));
        }
        else
          /* strcpy() is fine here as long as the string fits within
             error_buffer */
          strcpy(error_buffer, "SSL certificate verification failed");
      }
#if (OPENSSL_VERSION_NUMBER >= 0x10101000L && \
    !defined(LIBRESSL_VERSION_NUMBER) && \
    !defined(OPENSSL_IS_BORINGSSL))
      /* SSL_R_TLSV13_ALERT_CERTIFICATE_REQUIRED is only available on
         OpenSSL version above v1.1.1, not LibreSSL nor BoringSSL */
      else if((lib == ERR_LIB_SSL) &&
              (reason == SSL_R_TLSV13_ALERT_CERTIFICATE_REQUIRED)) {
          /* If client certificate is required, communicate the
             error to client */
          result = CURLE_SSL_CLIENTCERT;
          ossl_strerror(errdetail, error_buffer, sizeof(error_buffer));
      }
#endif
      else {
        result = CURLE_SSL_CONNECT_ERROR;
        ossl_strerror(errdetail, error_buffer, sizeof(error_buffer));
      }

      /* detail is already set to the SSL error above */

      /* If we e.g. use SSLv2 request-method and the server doesn't like us
       * (RST connection, etc.), OpenSSL gives no explanation whatsoever and
       * the SO_ERROR is also lost.
       */
      if(CURLE_SSL_CONNECT_ERROR == result && errdetail == 0) {
        const char * const hostname = SSL_HOST_NAME();
        const long int port = SSL_HOST_PORT();
        char extramsg[80]="";
        int sockerr = SOCKERRNO;
        if(sockerr && detail == SSL_ERROR_SYSCALL)
          Curl_strerror(sockerr, extramsg, sizeof(extramsg));
        failf(data, OSSL_PACKAGE " SSL_connect: %s in connection to %s:%ld ",
              extramsg[0] ? extramsg : SSL_ERROR_to_str(detail),
              hostname, port);
        return result;
      }

      /* Could be a CERT problem */
      failf(data, "%s", error_buffer);

      return result;
    }
  }
  else {
    /* we connected fine, we're not waiting for anything else. */
    connssl->connecting_state = ssl_connect_3;

    /* Informational message */
    infof(data, "SSL connection using %s / %s",
          SSL_get_version(backend->handle),
          SSL_get_cipher(backend->handle));

#ifdef HAS_ALPN
    /* Sets data and len to negotiated protocol, len is 0 if no protocol was
     * negotiated
     */
    if(conn->bits.tls_enable_alpn) {
      const unsigned char *neg_protocol;
      unsigned int len;
      SSL_get0_alpn_selected(backend->handle, &neg_protocol, &len);
      if(len) {
        infof(data, "ALPN, server accepted to use %.*s", len, neg_protocol);

#ifdef USE_HTTP2
        if(len == ALPN_H2_LENGTH &&
           !memcmp(ALPN_H2, neg_protocol, len)) {
          conn->negnpn = CURL_HTTP_VERSION_2;
        }
        else
#endif
        if(len == ALPN_HTTP_1_1_LENGTH &&
           !memcmp(ALPN_HTTP_1_1, neg_protocol, ALPN_HTTP_1_1_LENGTH)) {
          conn->negnpn = CURL_HTTP_VERSION_1_1;
        }
      }
      else
        infof(data, "ALPN, server did not agree to a protocol");

      Curl_multiuse_state(data, conn->negnpn == CURL_HTTP_VERSION_2 ?
                          BUNDLE_MULTIPLEX : BUNDLE_NO_MULTIUSE);
    }
#endif

    return CURLE_OK;
  }
}

static int asn1_object_dump(ASN1_OBJECT *a, char *buf, size_t len)
{
  int i, ilen;

  ilen = (int)len;
  if(ilen < 0)
    return 1; /* buffer too big */

  i = i2t_ASN1_OBJECT(buf, ilen, a);

  if(i >= ilen)
    return 1; /* buffer too small */

  return 0;
}

#define push_certinfo(_label, _num) \
do {                              \
  long info_len = BIO_get_mem_data(mem, &ptr); \
  Curl_ssl_push_certinfo_len(data, _num, _label, ptr, info_len); \
  if(1 != BIO_reset(mem))                                        \
    break;                                                       \
} while(0)

static void pubkey_show(struct Curl_easy *data,
                        BIO *mem,
                        int num,
                        const char *type,
                        const char *name,
                        const BIGNUM *bn)
{
  char *ptr;
  char namebuf[32];

  msnprintf(namebuf, sizeof(namebuf), "%s(%s)", type, name);

  if(bn)
    BN_print(mem, bn);
  push_certinfo(namebuf, num);
}

#ifdef HAVE_OPAQUE_RSA_DSA_DH
#define print_pubkey_BN(_type, _name, _num)              \
  pubkey_show(data, mem, _num, #_type, #_name, _name)

#else
#define print_pubkey_BN(_type, _name, _num)    \
do {                              \
  if(_type->_name) { \
    pubkey_show(data, mem, _num, #_type, #_name, _type->_name); \
  } \
} while(0)
#endif

static void X509V3_ext(struct Curl_easy *data,
                      int certnum,
                      CONST_EXTS STACK_OF(X509_EXTENSION) *exts)
{
  int i;

  if((int)sk_X509_EXTENSION_num(exts) <= 0)
    /* no extensions, bail out */
    return;

  for(i = 0; i < (int)sk_X509_EXTENSION_num(exts); i++) {
    ASN1_OBJECT *obj;
    X509_EXTENSION *ext = sk_X509_EXTENSION_value(exts, i);
    BUF_MEM *biomem;
    char namebuf[128];
    BIO *bio_out = BIO_new(BIO_s_mem());

    if(!bio_out)
      return;

    obj = X509_EXTENSION_get_object(ext);

    asn1_object_dump(obj, namebuf, sizeof(namebuf));

    if(!X509V3_EXT_print(bio_out, ext, 0, 0))
      ASN1_STRING_print(bio_out, (ASN1_STRING *)X509_EXTENSION_get_data(ext));

    BIO_get_mem_ptr(bio_out, &biomem);
    Curl_ssl_push_certinfo_len(data, certnum, namebuf, biomem->data,
                               biomem->length);
    BIO_free(bio_out);
  }
}

#ifdef OPENSSL_IS_BORINGSSL
typedef size_t numcert_t;
#else
typedef int numcert_t;
#endif

static CURLcode get_cert_chain(struct Curl_easy *data,
                               struct ssl_connect_data *connssl)
{
  CURLcode result;
  STACK_OF(X509) *sk;
  int i;
  numcert_t numcerts;
  BIO *mem;
  struct ssl_backend_data *backend = connssl->backend;

  sk = SSL_get_peer_cert_chain(backend->handle);
  if(!sk) {
    return CURLE_OUT_OF_MEMORY;
  }

  numcerts = sk_X509_num(sk);

  result = Curl_ssl_init_certinfo(data, (int)numcerts);
  if(result) {
    return result;
  }

  mem = BIO_new(BIO_s_mem());
  if(!mem) {
    return CURLE_OUT_OF_MEMORY;
  }

  for(i = 0; i < (int)numcerts; i++) {
    ASN1_INTEGER *num;
    X509 *x = sk_X509_value(sk, i);
    EVP_PKEY *pubkey = NULL;
    int j;
    char *ptr;
    const ASN1_BIT_STRING *psig = NULL;

    X509_NAME_print_ex(mem, X509_get_subject_name(x), 0, XN_FLAG_ONELINE);
    push_certinfo("Subject", i);

    X509_NAME_print_ex(mem, X509_get_issuer_name(x), 0, XN_FLAG_ONELINE);
    push_certinfo("Issuer", i);

    BIO_printf(mem, "%lx", X509_get_version(x));
    push_certinfo("Version", i);

    num = X509_get_serialNumber(x);
    if(num->type == V_ASN1_NEG_INTEGER)
      BIO_puts(mem, "-");
    for(j = 0; j < num->length; j++)
      BIO_printf(mem, "%02x", num->data[j]);
    push_certinfo("Serial Number", i);

#if defined(HAVE_X509_GET0_SIGNATURE) && defined(HAVE_X509_GET0_EXTENSIONS)
    {
      const X509_ALGOR *sigalg = NULL;
      X509_PUBKEY *xpubkey = NULL;
      ASN1_OBJECT *pubkeyoid = NULL;

      X509_get0_signature(&psig, &sigalg, x);
      if(sigalg) {
        i2a_ASN1_OBJECT(mem, sigalg->algorithm);
        push_certinfo("Signature Algorithm", i);
      }

      xpubkey = X509_get_X509_PUBKEY(x);
      if(xpubkey) {
        X509_PUBKEY_get0_param(&pubkeyoid, NULL, NULL, NULL, xpubkey);
        if(pubkeyoid) {
          i2a_ASN1_OBJECT(mem, pubkeyoid);
          push_certinfo("Public Key Algorithm", i);
        }
      }

      X509V3_ext(data, i, X509_get0_extensions(x));
    }
#else
    {
      /* before OpenSSL 1.0.2 */
      X509_CINF *cinf = x->cert_info;

      i2a_ASN1_OBJECT(mem, cinf->signature->algorithm);
      push_certinfo("Signature Algorithm", i);

      i2a_ASN1_OBJECT(mem, cinf->key->algor->algorithm);
      push_certinfo("Public Key Algorithm", i);

      X509V3_ext(data, i, cinf->extensions);

      psig = x->signature;
    }
#endif

    ASN1_TIME_print(mem, X509_get0_notBefore(x));
    push_certinfo("Start date", i);

    ASN1_TIME_print(mem, X509_get0_notAfter(x));
    push_certinfo("Expire date", i);

    pubkey = X509_get_pubkey(x);
    if(!pubkey)
      infof(data, "   Unable to load public key");
    else {
      int pktype;
#ifdef HAVE_OPAQUE_EVP_PKEY
      pktype = EVP_PKEY_id(pubkey);
#else
      pktype = pubkey->type;
#endif
      switch(pktype) {
      case EVP_PKEY_RSA:
      {
#ifndef HAVE_EVP_PKEY_GET_PARAMS
        RSA *rsa;
#ifdef HAVE_OPAQUE_EVP_PKEY
        rsa = EVP_PKEY_get0_RSA(pubkey);
#else
        rsa = pubkey->pkey.rsa;
#endif /* HAVE_OPAQUE_EVP_PKEY */
#endif /* !HAVE_EVP_PKEY_GET_PARAMS */

        {
#ifdef HAVE_OPAQUE_RSA_DSA_DH
          DECLARE_PKEY_PARAM_BIGNUM(n);
          DECLARE_PKEY_PARAM_BIGNUM(e);
#ifdef HAVE_EVP_PKEY_GET_PARAMS
          EVP_PKEY_get_bn_param(pubkey, OSSL_PKEY_PARAM_RSA_N, &n);
          EVP_PKEY_get_bn_param(pubkey, OSSL_PKEY_PARAM_RSA_E, &e);
#else
          RSA_get0_key(rsa, &n, &e, NULL);
#endif /* HAVE_EVP_PKEY_GET_PARAMS */
          BIO_printf(mem, "%d", BN_num_bits(n));
#else
          BIO_printf(mem, "%d", BN_num_bits(rsa->n));
#endif /* HAVE_OPAQUE_RSA_DSA_DH */
          push_certinfo("RSA Public Key", i);
          print_pubkey_BN(rsa, n, i);
          print_pubkey_BN(rsa, e, i);
          FREE_PKEY_PARAM_BIGNUM(n);
          FREE_PKEY_PARAM_BIGNUM(e);
        }

        break;
      }
      case EVP_PKEY_DSA:
      {
#ifndef OPENSSL_NO_DSA
#ifndef HAVE_EVP_PKEY_GET_PARAMS
        DSA *dsa;
#ifdef HAVE_OPAQUE_EVP_PKEY
        dsa = EVP_PKEY_get0_DSA(pubkey);
#else
        dsa = pubkey->pkey.dsa;
#endif /* HAVE_OPAQUE_EVP_PKEY */
#endif /* !HAVE_EVP_PKEY_GET_PARAMS */
        {
#ifdef HAVE_OPAQUE_RSA_DSA_DH
          DECLARE_PKEY_PARAM_BIGNUM(p);
          DECLARE_PKEY_PARAM_BIGNUM(q);
          DECLARE_PKEY_PARAM_BIGNUM(g);
          DECLARE_PKEY_PARAM_BIGNUM(pub_key);
#ifdef HAVE_EVP_PKEY_GET_PARAMS
          EVP_PKEY_get_bn_param(pubkey, OSSL_PKEY_PARAM_FFC_P, &p);
          EVP_PKEY_get_bn_param(pubkey, OSSL_PKEY_PARAM_FFC_Q, &q);
          EVP_PKEY_get_bn_param(pubkey, OSSL_PKEY_PARAM_FFC_G, &g);
          EVP_PKEY_get_bn_param(pubkey, OSSL_PKEY_PARAM_PUB_KEY, &pub_key);
#else
          DSA_get0_pqg(dsa, &p, &q, &g);
          DSA_get0_key(dsa, &pub_key, NULL);
#endif /* HAVE_EVP_PKEY_GET_PARAMS */
#endif /* HAVE_OPAQUE_RSA_DSA_DH */
          print_pubkey_BN(dsa, p, i);
          print_pubkey_BN(dsa, q, i);
          print_pubkey_BN(dsa, g, i);
          print_pubkey_BN(dsa, pub_key, i);
          FREE_PKEY_PARAM_BIGNUM(p);
          FREE_PKEY_PARAM_BIGNUM(q);
          FREE_PKEY_PARAM_BIGNUM(g);
          FREE_PKEY_PARAM_BIGNUM(pub_key);
        }
#endif /* !OPENSSL_NO_DSA */
        break;
      }
      case EVP_PKEY_DH:
      {
#ifndef HAVE_EVP_PKEY_GET_PARAMS
        DH *dh;
#ifdef HAVE_OPAQUE_EVP_PKEY
        dh = EVP_PKEY_get0_DH(pubkey);
#else
        dh = pubkey->pkey.dh;
#endif /* HAVE_OPAQUE_EVP_PKEY */
#endif /* !HAVE_EVP_PKEY_GET_PARAMS */
        {
#ifdef HAVE_OPAQUE_RSA_DSA_DH
          DECLARE_PKEY_PARAM_BIGNUM(p);
          DECLARE_PKEY_PARAM_BIGNUM(q);
          DECLARE_PKEY_PARAM_BIGNUM(g);
          DECLARE_PKEY_PARAM_BIGNUM(pub_key);
#ifdef HAVE_EVP_PKEY_GET_PARAMS
          EVP_PKEY_get_bn_param(pubkey, OSSL_PKEY_PARAM_FFC_P, &p);
          EVP_PKEY_get_bn_param(pubkey, OSSL_PKEY_PARAM_FFC_Q, &q);
          EVP_PKEY_get_bn_param(pubkey, OSSL_PKEY_PARAM_FFC_G, &g);
          EVP_PKEY_get_bn_param(pubkey, OSSL_PKEY_PARAM_PUB_KEY, &pub_key);
#else
          DH_get0_pqg(dh, &p, &q, &g);
          DH_get0_key(dh, &pub_key, NULL);
#endif /* HAVE_EVP_PKEY_GET_PARAMS */
          print_pubkey_BN(dh, p, i);
          print_pubkey_BN(dh, q, i);
          print_pubkey_BN(dh, g, i);
#else
          print_pubkey_BN(dh, p, i);
          print_pubkey_BN(dh, g, i);
#endif /* HAVE_OPAQUE_RSA_DSA_DH */
          print_pubkey_BN(dh, pub_key, i);
          FREE_PKEY_PARAM_BIGNUM(p);
          FREE_PKEY_PARAM_BIGNUM(q);
          FREE_PKEY_PARAM_BIGNUM(g);
          FREE_PKEY_PARAM_BIGNUM(pub_key);
       }
        break;
      }
      }
      EVP_PKEY_free(pubkey);
    }

    if(psig) {
      for(j = 0; j < psig->length; j++)
        BIO_printf(mem, "%02x:", psig->data[j]);
      push_certinfo("Signature", i);
    }

    PEM_write_bio_X509(mem, x);
    push_certinfo("Cert", i);
  }

  BIO_free(mem);

  return CURLE_OK;
}

/*
 * Heavily modified from:
 * https://www.owasp.org/index.php/Certificate_and_Public_Key_Pinning#OpenSSL
 */
static CURLcode pkp_pin_peer_pubkey(struct Curl_easy *data, X509* cert,
                                    const char *pinnedpubkey)
{
  /* Scratch */
  int len1 = 0, len2 = 0;
  unsigned char *buff1 = NULL, *temp = NULL;

  /* Result is returned to caller */
  CURLcode result = CURLE_SSL_PINNEDPUBKEYNOTMATCH;

  /* if a path wasn't specified, don't pin */
  if(!pinnedpubkey)
    return CURLE_OK;

  if(!cert)
    return result;

  do {
    /* Begin Gyrations to get the subjectPublicKeyInfo     */
    /* Thanks to Viktor Dukhovni on the OpenSSL mailing list */

    /* https://groups.google.com/group/mailing.openssl.users/browse_thread
     /thread/d61858dae102c6c7 */
    len1 = i2d_X509_PUBKEY(X509_get_X509_PUBKEY(cert), NULL);
    if(len1 < 1)
      break; /* failed */

    buff1 = temp = malloc(len1);
    if(!buff1)
      break; /* failed */

    /* https://www.openssl.org/docs/crypto/d2i_X509.html */
    len2 = i2d_X509_PUBKEY(X509_get_X509_PUBKEY(cert), &temp);

    /*
     * These checks are verifying we got back the same values as when we
     * sized the buffer. It's pretty weak since they should always be the
     * same. But it gives us something to test.
     */
    if((len1 != len2) || !temp || ((temp - buff1) != len1))
      break; /* failed */

    /* End Gyrations */

    /* The one good exit point */
    result = Curl_pin_peer_pubkey(data, pinnedpubkey, buff1, len1);
  } while(0);

  if(buff1)
    free(buff1);

  return result;
}

/*
 * Get the server cert, verify it and show it, etc., only call failf() if the
 * 'strict' argument is TRUE as otherwise all this is for informational
 * purposes only!
 *
 * We check certificates to authenticate the server; otherwise we risk
 * man-in-the-middle attack.
 */
static CURLcode servercert(struct Curl_easy *data,
                           struct connectdata *conn,
                           struct ssl_connect_data *connssl,
                           bool strict)
{
  CURLcode result = CURLE_OK;
  int rc;
  long lerr;
  X509 *issuer;
  BIO *fp = NULL;
  char error_buffer[256]="";
  char buffer[2048];
  const char *ptr;
  BIO *mem = BIO_new(BIO_s_mem());
  struct ssl_backend_data *backend = connssl->backend;

  if(!mem) {
    failf(data,
          "BIO_new return NULL, " OSSL_PACKAGE
          " error %s",
          ossl_strerror(ERR_get_error(), error_buffer,
                        sizeof(error_buffer)) );
    return CURLE_OUT_OF_MEMORY;
  }

  if(data->set.ssl.certinfo)
    /* we've been asked to gather certificate info! */
    (void)get_cert_chain(data, connssl);

  backend->server_cert = SSL_get1_peer_certificate(backend->handle);
  if(!backend->server_cert) {
    BIO_free(mem);
    if(!strict)
      return CURLE_OK;

    failf(data, "SSL: couldn't get peer certificate!");
    return CURLE_PEER_FAILED_VERIFICATION;
  }

  infof(data, "%s certificate:", SSL_IS_PROXY() ? "Proxy" : "Server");

  rc = x509_name_oneline(X509_get_subject_name(backend->server_cert),
                         buffer, sizeof(buffer));
  infof(data, " subject: %s", rc?"[NONE]":buffer);

#ifndef CURL_DISABLE_VERBOSE_STRINGS
  {
    long len;
    ASN1_TIME_print(mem, X509_get0_notBefore(backend->server_cert));
    len = BIO_get_mem_data(mem, (char **) &ptr);
    infof(data, " start date: %.*s", (int)len, ptr);
    (void)BIO_reset(mem);

    ASN1_TIME_print(mem, X509_get0_notAfter(backend->server_cert));
    len = BIO_get_mem_data(mem, (char **) &ptr);
    infof(data, " expire date: %.*s", (int)len, ptr);
    (void)BIO_reset(mem);
  }
#endif

  BIO_free(mem);

  if(SSL_CONN_CONFIG(verifyhost)) {
    result = verifyhost(data, conn, backend->server_cert);
    if(result) {
      X509_free(backend->server_cert);
      backend->server_cert = NULL;
      return result;
    }
  }

  rc = x509_name_oneline(X509_get_issuer_name(backend->server_cert),
                         buffer, sizeof(buffer));
  if(rc) {
    if(strict)
      failf(data, "SSL: couldn't get X509-issuer name!");
    result = CURLE_PEER_FAILED_VERIFICATION;
  }
  else {
    infof(data, " issuer: %s", buffer);

    /* We could do all sorts of certificate verification stuff here before
       deallocating the certificate. */

    /* e.g. match issuer name with provided issuer certificate */
    if(SSL_CONN_CONFIG(issuercert) || SSL_CONN_CONFIG(issuercert_blob)) {
      if(SSL_CONN_CONFIG(issuercert_blob))
        fp = BIO_new_mem_buf(SSL_CONN_CONFIG(issuercert_blob)->data,
                             (int)SSL_CONN_CONFIG(issuercert_blob)->len);
      else {
        fp = BIO_new(BIO_s_file());
        if(!fp) {
          failf(data,
                "BIO_new return NULL, " OSSL_PACKAGE
                " error %s",
                ossl_strerror(ERR_get_error(), error_buffer,
                              sizeof(error_buffer)) );
          X509_free(backend->server_cert);
          backend->server_cert = NULL;
          return CURLE_OUT_OF_MEMORY;
        }

        if(BIO_read_filename(fp, SSL_CONN_CONFIG(issuercert)) <= 0) {
          if(strict)
            failf(data, "SSL: Unable to open issuer cert (%s)",
                  SSL_CONN_CONFIG(issuercert));
          BIO_free(fp);
          X509_free(backend->server_cert);
          backend->server_cert = NULL;
          return CURLE_SSL_ISSUER_ERROR;
        }
      }

      issuer = PEM_read_bio_X509(fp, NULL, ZERO_NULL, NULL);
      if(!issuer) {
        if(strict)
          failf(data, "SSL: Unable to read issuer cert (%s)",
                SSL_CONN_CONFIG(issuercert));
        BIO_free(fp);
        X509_free(issuer);
        X509_free(backend->server_cert);
        backend->server_cert = NULL;
        return CURLE_SSL_ISSUER_ERROR;
      }

      if(X509_check_issued(issuer, backend->server_cert) != X509_V_OK) {
        if(strict)
          failf(data, "SSL: Certificate issuer check failed (%s)",
                SSL_CONN_CONFIG(issuercert));
        BIO_free(fp);
        X509_free(issuer);
        X509_free(backend->server_cert);
        backend->server_cert = NULL;
        return CURLE_SSL_ISSUER_ERROR;
      }

      infof(data, " SSL certificate issuer check ok (%s)",
            SSL_CONN_CONFIG(issuercert));
      BIO_free(fp);
      X509_free(issuer);
    }

    lerr = SSL_get_verify_result(backend->handle);
    SSL_SET_OPTION_LVALUE(certverifyresult) = lerr;
    if(lerr != X509_V_OK) {
      if(SSL_CONN_CONFIG(verifypeer)) {
        /* We probably never reach this, because SSL_connect() will fail
           and we return earlier if verifypeer is set? */
        if(strict)
          failf(data, "SSL certificate verify result: %s (%ld)",
                X509_verify_cert_error_string(lerr), lerr);
        result = CURLE_PEER_FAILED_VERIFICATION;
      }
      else
        infof(data, " SSL certificate verify result: %s (%ld),"
              " continuing anyway.",
              X509_verify_cert_error_string(lerr), lerr);
    }
    else
      infof(data, " SSL certificate verify ok.");
  }

#if (OPENSSL_VERSION_NUMBER >= 0x0090808fL) && !defined(OPENSSL_NO_TLSEXT) && \
    !defined(OPENSSL_NO_OCSP)
  if(SSL_CONN_CONFIG(verifystatus)) {
    result = verifystatus(data, connssl);
    if(result) {
      X509_free(backend->server_cert);
      backend->server_cert = NULL;
      return result;
    }
  }
#endif

  if(!strict)
    /* when not strict, we don't bother about the verify cert problems */
    result = CURLE_OK;

  ptr = SSL_PINNED_PUB_KEY();
  if(!result && ptr) {
    result = pkp_pin_peer_pubkey(data, backend->server_cert, ptr);
    if(result)
      failf(data, "SSL: public key does not match pinned public key!");
  }

  X509_free(backend->server_cert);
  backend->server_cert = NULL;
  connssl->connecting_state = ssl_connect_done;

  return result;
}

static CURLcode ossl_connect_step3(struct Curl_easy *data,
                                   struct connectdata *conn, int sockindex)
{
  CURLcode result = CURLE_OK;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];

  DEBUGASSERT(ssl_connect_3 == connssl->connecting_state);

  /*
   * We check certificates to authenticate the server; otherwise we risk
   * man-in-the-middle attack; NEVERTHELESS, if we're told explicitly not to
   * verify the peer, ignore faults and failures from the server cert
   * operations.
   */

  result = servercert(data, conn, connssl, (SSL_CONN_CONFIG(verifypeer) ||
                                            SSL_CONN_CONFIG(verifyhost)));

  if(!result)
    connssl->connecting_state = ssl_connect_done;

  return result;
}

static Curl_recv ossl_recv;
static Curl_send ossl_send;

static CURLcode ossl_connect_common(struct Curl_easy *data,
                                    struct connectdata *conn,
                                    int sockindex,
                                    bool nonblocking,
                                    bool *done)
{
  CURLcode result;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  curl_socket_t sockfd = conn->sock[sockindex];
  int what;

  /* check if the connection has already been established */
  if(ssl_connection_complete == connssl->state) {
    *done = TRUE;
    return CURLE_OK;
  }

  if(ssl_connect_1 == connssl->connecting_state) {
    /* Find out how much more time we're allowed */
    const timediff_t timeout_ms = Curl_timeleft(data, NULL, TRUE);

    if(timeout_ms < 0) {
      /* no need to continue if time is already up */
      failf(data, "SSL connection timeout");
      return CURLE_OPERATION_TIMEDOUT;
    }

    result = ossl_connect_step1(data, conn, sockindex);
    if(result)
      return result;
  }

  while(ssl_connect_2 == connssl->connecting_state ||
        ssl_connect_2_reading == connssl->connecting_state ||
        ssl_connect_2_writing == connssl->connecting_state) {

    /* check allowed time left */
    const timediff_t timeout_ms = Curl_timeleft(data, NULL, TRUE);

    if(timeout_ms < 0) {
      /* no need to continue if time already is up */
      failf(data, "SSL connection timeout");
      return CURLE_OPERATION_TIMEDOUT;
    }

    /* if ssl is expecting something, check if it's available. */
    if(connssl->connecting_state == ssl_connect_2_reading ||
       connssl->connecting_state == ssl_connect_2_writing) {

      curl_socket_t writefd = ssl_connect_2_writing ==
        connssl->connecting_state?sockfd:CURL_SOCKET_BAD;
      curl_socket_t readfd = ssl_connect_2_reading ==
        connssl->connecting_state?sockfd:CURL_SOCKET_BAD;

      what = Curl_socket_check(readfd, CURL_SOCKET_BAD, writefd,
                               nonblocking?0:timeout_ms);
      if(what < 0) {
        /* fatal error */
        failf(data, "select/poll on SSL socket, errno: %d", SOCKERRNO);
        return CURLE_SSL_CONNECT_ERROR;
      }
      if(0 == what) {
        if(nonblocking) {
          *done = FALSE;
          return CURLE_OK;
        }
        /* timeout */
        failf(data, "SSL connection timeout");
        return CURLE_OPERATION_TIMEDOUT;
      }
      /* socket is readable or writable */
    }

    /* Run transaction, and return to the caller if it failed or if this
     * connection is done nonblocking and this loop would execute again. This
     * permits the owner of a multi handle to abort a connection attempt
     * before step2 has completed while ensuring that a client using select()
     * or epoll() will always have a valid fdset to wait on.
     */
    result = ossl_connect_step2(data, conn, sockindex);
    if(result || (nonblocking &&
                  (ssl_connect_2 == connssl->connecting_state ||
                   ssl_connect_2_reading == connssl->connecting_state ||
                   ssl_connect_2_writing == connssl->connecting_state)))
      return result;

  } /* repeat step2 until all transactions are done. */

  if(ssl_connect_3 == connssl->connecting_state) {
    result = ossl_connect_step3(data, conn, sockindex);
    if(result)
      return result;
  }

  if(ssl_connect_done == connssl->connecting_state) {
    connssl->state = ssl_connection_complete;
    conn->recv[sockindex] = ossl_recv;
    conn->send[sockindex] = ossl_send;
    *done = TRUE;
  }
  else
    *done = FALSE;

  /* Reset our connect state machine */
  connssl->connecting_state = ssl_connect_1;

  return CURLE_OK;
}

static CURLcode ossl_connect_nonblocking(struct Curl_easy *data,
                                         struct connectdata *conn,
                                         int sockindex,
                                         bool *done)
{
  return ossl_connect_common(data, conn, sockindex, TRUE, done);
}

static CURLcode ossl_connect(struct Curl_easy *data, struct connectdata *conn,
                             int sockindex)
{
  CURLcode result;
  bool done = FALSE;

  result = ossl_connect_common(data, conn, sockindex, FALSE, &done);
  if(result)
    return result;

  DEBUGASSERT(done);

  return CURLE_OK;
}

static bool ossl_data_pending(const struct connectdata *conn,
                              int connindex)
{
  const struct ssl_connect_data *connssl = &conn->ssl[connindex];
  if(connssl->backend->handle && SSL_pending(connssl->backend->handle))
    return TRUE;
#ifndef CURL_DISABLE_PROXY
  {
    const struct ssl_connect_data *proxyssl = &conn->proxy_ssl[connindex];
    if(proxyssl->backend->handle && SSL_pending(proxyssl->backend->handle))
      return TRUE;
  }
#endif
  return FALSE;
}

static size_t ossl_version(char *buffer, size_t size);

static ssize_t ossl_send(struct Curl_easy *data,
                         int sockindex,
                         const void *mem,
                         size_t len,
                         CURLcode *curlcode)
{
  /* SSL_write() is said to return 'int' while write() and send() returns
     'size_t' */
  int err;
  char error_buffer[256];
  unsigned long sslerror;
  int memlen;
  int rc;
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;

  ERR_clear_error();

  memlen = (len > (size_t)INT_MAX) ? INT_MAX : (int)len;
  set_logger(conn, data);
  rc = SSL_write(backend->handle, mem, memlen);

  if(rc <= 0) {
    err = SSL_get_error(backend->handle, rc);

    switch(err) {
    case SSL_ERROR_WANT_READ:
    case SSL_ERROR_WANT_WRITE:
      /* The operation did not complete; the same TLS/SSL I/O function
         should be called again later. This is basically an EWOULDBLOCK
         equivalent. */
      *curlcode = CURLE_AGAIN;
      return -1;
    case SSL_ERROR_SYSCALL:
      {
        int sockerr = SOCKERRNO;
        sslerror = ERR_get_error();
        if(sslerror)
          ossl_strerror(sslerror, error_buffer, sizeof(error_buffer));
        else if(sockerr)
          Curl_strerror(sockerr, error_buffer, sizeof(error_buffer));
        else {
          strncpy(error_buffer, SSL_ERROR_to_str(err), sizeof(error_buffer));
          error_buffer[sizeof(error_buffer) - 1] = '\0';
        }
        failf(data, OSSL_PACKAGE " SSL_write: %s, errno %d",
              error_buffer, sockerr);
        *curlcode = CURLE_SEND_ERROR;
        return -1;
      }
    case SSL_ERROR_SSL:
      /*  A failure in the SSL library occurred, usually a protocol error.
          The OpenSSL error queue contains more information on the error. */
      sslerror = ERR_get_error();
      if(ERR_GET_LIB(sslerror) == ERR_LIB_SSL &&
         ERR_GET_REASON(sslerror) == SSL_R_BIO_NOT_SET &&
         conn->ssl[sockindex].state == ssl_connection_complete
#ifndef CURL_DISABLE_PROXY
         && conn->proxy_ssl[sockindex].state == ssl_connection_complete
#endif
        ) {
        char ver[120];
        (void)ossl_version(ver, sizeof(ver));
        failf(data, "Error: %s does not support double SSL tunneling.", ver);
      }
      else
        failf(data, "SSL_write() error: %s",
              ossl_strerror(sslerror, error_buffer, sizeof(error_buffer)));
      *curlcode = CURLE_SEND_ERROR;
      return -1;
    }
    /* a true error */
    failf(data, OSSL_PACKAGE " SSL_write: %s, errno %d",
          SSL_ERROR_to_str(err), SOCKERRNO);
    *curlcode = CURLE_SEND_ERROR;
    return -1;
  }
  *curlcode = CURLE_OK;
  return (ssize_t)rc; /* number of bytes */
}

static ssize_t ossl_recv(struct Curl_easy *data,   /* transfer */
                         int num,                  /* socketindex */
                         char *buf,                /* store read data here */
                         size_t buffersize,        /* max amount to read */
                         CURLcode *curlcode)
{
  char error_buffer[256];
  unsigned long sslerror;
  ssize_t nread;
  int buffsize;
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *connssl = &conn->ssl[num];
  struct ssl_backend_data *backend = connssl->backend;

  ERR_clear_error();

  buffsize = (buffersize > (size_t)INT_MAX) ? INT_MAX : (int)buffersize;
  set_logger(conn, data);
  nread = (ssize_t)SSL_read(backend->handle, buf, buffsize);
  if(nread <= 0) {
    /* failed SSL_read */
    int err = SSL_get_error(backend->handle, (int)nread);

    switch(err) {
    case SSL_ERROR_NONE: /* this is not an error */
      break;
    case SSL_ERROR_ZERO_RETURN: /* no more data */
      /* close_notify alert */
      if(num == FIRSTSOCKET)
        /* mark the connection for close if it is indeed the control
           connection */
        connclose(conn, "TLS close_notify");
      break;
    case SSL_ERROR_WANT_READ:
    case SSL_ERROR_WANT_WRITE:
      /* there's data pending, re-invoke SSL_read() */
      *curlcode = CURLE_AGAIN;
      return -1;
    default:
      /* openssl/ssl.h for SSL_ERROR_SYSCALL says "look at error stack/return
         value/errno" */
      /* https://www.openssl.org/docs/crypto/ERR_get_error.html */
      sslerror = ERR_get_error();
      if((nread < 0) || sslerror) {
        /* If the return code was negative or there actually is an error in the
           queue */
        int sockerr = SOCKERRNO;
        if(sslerror)
          ossl_strerror(sslerror, error_buffer, sizeof(error_buffer));
        else if(sockerr && err == SSL_ERROR_SYSCALL)
          Curl_strerror(sockerr, error_buffer, sizeof(error_buffer));
        else {
          strncpy(error_buffer, SSL_ERROR_to_str(err), sizeof(error_buffer));
          error_buffer[sizeof(error_buffer) - 1] = '\0';
        }
        failf(data, OSSL_PACKAGE " SSL_read: %s, errno %d",
              error_buffer, sockerr);
        *curlcode = CURLE_RECV_ERROR;
        return -1;
      }
      /* For debug builds be a little stricter and error on any
         SSL_ERROR_SYSCALL. For example a server may have closed the connection
         abruptly without a close_notify alert. For compatibility with older
         peers we don't do this by default. #4624

         We can use this to gauge how many users may be affected, and
         if it goes ok eventually transition to allow in dev and release with
         the newest OpenSSL: #if (OPENSSL_VERSION_NUMBER >= 0x10101000L) */
#ifdef DEBUGBUILD
      if(err == SSL_ERROR_SYSCALL) {
        int sockerr = SOCKERRNO;
        if(sockerr)
          Curl_strerror(sockerr, error_buffer, sizeof(error_buffer));
        else {
          msnprintf(error_buffer, sizeof(error_buffer),
                    "Connection closed abruptly");
        }
        failf(data, OSSL_PACKAGE " SSL_read: %s, errno %d"
              " (Fatal because this is a curl debug build)",
              error_buffer, sockerr);
        *curlcode = CURLE_RECV_ERROR;
        return -1;
      }
#endif
    }
  }
  return nread;
}

static size_t ossl_version(char *buffer, size_t size)
{
#ifdef LIBRESSL_VERSION_NUMBER
#if LIBRESSL_VERSION_NUMBER < 0x2070100fL
  return msnprintf(buffer, size, "%s/%lx.%lx.%lx",
                   OSSL_PACKAGE,
                   (LIBRESSL_VERSION_NUMBER>>28)&0xf,
                   (LIBRESSL_VERSION_NUMBER>>20)&0xff,
                   (LIBRESSL_VERSION_NUMBER>>12)&0xff);
#else /* OpenSSL_version() first appeared in LibreSSL 2.7.1 */
  char *p;
  int count;
  const char *ver = OpenSSL_version(OPENSSL_VERSION);
  const char expected[] = OSSL_PACKAGE " "; /* ie "LibreSSL " */
  if(Curl_strncasecompare(ver, expected, sizeof(expected) - 1)) {
    ver += sizeof(expected) - 1;
  }
  count = msnprintf(buffer, size, "%s/%s", OSSL_PACKAGE, ver);
  for(p = buffer; *p; ++p) {
    if(ISSPACE(*p))
      *p = '_';
  }
  return count;
#endif
#elif defined(OPENSSL_IS_BORINGSSL)
  return msnprintf(buffer, size, OSSL_PACKAGE);
#elif defined(HAVE_OPENSSL_VERSION) && defined(OPENSSL_VERSION_STRING)
  return msnprintf(buffer, size, "%s/%s",
                   OSSL_PACKAGE, OpenSSL_version(OPENSSL_VERSION_STRING));
#else
  /* not LibreSSL, BoringSSL and not using OpenSSL_version */

  char sub[3];
  unsigned long ssleay_value;
  sub[2]='\0';
  sub[1]='\0';
  ssleay_value = OpenSSL_version_num();
  if(ssleay_value < 0x906000) {
    ssleay_value = SSLEAY_VERSION_NUMBER;
    sub[0]='\0';
  }
  else {
    if(ssleay_value&0xff0) {
      int minor_ver = (ssleay_value >> 4) & 0xff;
      if(minor_ver > 26) {
        /* handle extended version introduced for 0.9.8za */
        sub[1] = (char) ((minor_ver - 1) % 26 + 'a' + 1);
        sub[0] = 'z';
      }
      else {
        sub[0] = (char) (minor_ver + 'a' - 1);
      }
    }
    else
      sub[0]='\0';
  }

  return msnprintf(buffer, size, "%s/%lx.%lx.%lx%s"
#ifdef OPENSSL_FIPS
                   "-fips"
#endif
                   ,
                   OSSL_PACKAGE,
                   (ssleay_value>>28)&0xf,
                   (ssleay_value>>20)&0xff,
                   (ssleay_value>>12)&0xff,
                   sub);
#endif /* OPENSSL_IS_BORINGSSL */
}

/* can be called with data == NULL */
static CURLcode ossl_random(struct Curl_easy *data,
                            unsigned char *entropy, size_t length)
{
  int rc;
  if(data) {
    if(ossl_seed(data)) /* Initiate the seed if not already done */
      return CURLE_FAILED_INIT; /* couldn't seed for some reason */
  }
  else {
    if(!rand_enough())
      return CURLE_FAILED_INIT;
  }
  /* RAND_bytes() returns 1 on success, 0 otherwise.  */
  rc = RAND_bytes(entropy, curlx_uztosi(length));
  return (rc == 1 ? CURLE_OK : CURLE_FAILED_INIT);
}

#if (OPENSSL_VERSION_NUMBER >= 0x0090800fL) && !defined(OPENSSL_NO_SHA256)
static CURLcode ossl_sha256sum(const unsigned char *tmp, /* input */
                               size_t tmplen,
                               unsigned char *sha256sum /* output */,
                               size_t unused)
{
  EVP_MD_CTX *mdctx;
  unsigned int len = 0;
  (void) unused;

  mdctx = EVP_MD_CTX_create();
  if(!mdctx)
    return CURLE_OUT_OF_MEMORY;
  EVP_DigestInit(mdctx, EVP_sha256());
  EVP_DigestUpdate(mdctx, tmp, tmplen);
  EVP_DigestFinal_ex(mdctx, sha256sum, &len);
  EVP_MD_CTX_destroy(mdctx);
  return CURLE_OK;
}
#endif

static bool ossl_cert_status_request(void)
{
#if (OPENSSL_VERSION_NUMBER >= 0x0090808fL) && !defined(OPENSSL_NO_TLSEXT) && \
    !defined(OPENSSL_NO_OCSP)
  return TRUE;
#else
  return FALSE;
#endif
}

static void *ossl_get_internals(struct ssl_connect_data *connssl,
                                CURLINFO info)
{
  /* Legacy: CURLINFO_TLS_SESSION must return an SSL_CTX pointer. */
  struct ssl_backend_data *backend = connssl->backend;
  return info == CURLINFO_TLS_SESSION ?
         (void *)backend->ctx : (void *)backend->handle;
}

static void ossl_associate_connection(struct Curl_easy *data,
                                      struct connectdata *conn,
                                      int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;

  /* If we don't have SSL context, do nothing. */
  if(!backend->handle)
    return;

  if(SSL_SET_OPTION(primary.sessionid)) {
    int data_idx = ossl_get_ssl_data_index();
    int connectdata_idx = ossl_get_ssl_conn_index();
    int sockindex_idx = ossl_get_ssl_sockindex_index();
    int proxy_idx = ossl_get_proxy_index();

    if(data_idx >= 0 && connectdata_idx >= 0 && sockindex_idx >= 0 &&
       proxy_idx >= 0) {
      /* Store the data needed for the "new session" callback.
       * The sockindex is stored as a pointer to an array element. */
      SSL_set_ex_data(backend->handle, data_idx, data);
      SSL_set_ex_data(backend->handle, connectdata_idx, conn);
      SSL_set_ex_data(backend->handle, sockindex_idx, conn->sock + sockindex);
#ifndef CURL_DISABLE_PROXY
      SSL_set_ex_data(backend->handle, proxy_idx, SSL_IS_PROXY() ? (void *) 1:
                      NULL);
#else
      SSL_set_ex_data(backend->handle, proxy_idx, NULL);
#endif
    }
  }
}

/*
 * Starting with TLS 1.3, the ossl_new_session_cb callback gets called after
 * the handshake. If the transfer that sets up the callback gets killed before
 * this callback arrives, we must make sure to properly clear the data to
 * avoid UAF problems. A future optimization could be to instead store another
 * transfer that might still be using the same connection.
 */

static void ossl_disassociate_connection(struct Curl_easy *data,
                                         int sockindex)
{
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;

  /* If we don't have SSL context, do nothing. */
  if(!backend->handle)
    return;

  if(SSL_SET_OPTION(primary.sessionid)) {
    int data_idx = ossl_get_ssl_data_index();
    int connectdata_idx = ossl_get_ssl_conn_index();
    int sockindex_idx = ossl_get_ssl_sockindex_index();
    int proxy_idx = ossl_get_proxy_index();

    if(data_idx >= 0 && connectdata_idx >= 0 && sockindex_idx >= 0 &&
       proxy_idx >= 0) {
      /* Disable references to data in "new session" callback to avoid
       * accessing a stale pointer. */
      SSL_set_ex_data(backend->handle, data_idx, NULL);
      SSL_set_ex_data(backend->handle, connectdata_idx, NULL);
      SSL_set_ex_data(backend->handle, sockindex_idx, NULL);
      SSL_set_ex_data(backend->handle, proxy_idx, NULL);
    }
  }
}

const struct Curl_ssl Curl_ssl_openssl = {
  { CURLSSLBACKEND_OPENSSL, "openssl" }, /* info */

  SSLSUPP_CA_PATH |
  SSLSUPP_CAINFO_BLOB |
  SSLSUPP_CERTINFO |
  SSLSUPP_PINNEDPUBKEY |
  SSLSUPP_SSL_CTX |
#ifdef HAVE_SSL_CTX_SET_CIPHERSUITES
  SSLSUPP_TLS13_CIPHERSUITES |
#endif
  SSLSUPP_HTTPS_PROXY,

  sizeof(struct ssl_backend_data),

  ossl_init,                /* init */
  ossl_cleanup,             /* cleanup */
  ossl_version,             /* version */
  ossl_check_cxn,           /* check_cxn */
  ossl_shutdown,            /* shutdown */
  ossl_data_pending,        /* data_pending */
  ossl_random,              /* random */
  ossl_cert_status_request, /* cert_status_request */
  ossl_connect,             /* connect */
  ossl_connect_nonblocking, /* connect_nonblocking */
  Curl_ssl_getsock,         /* getsock */
  ossl_get_internals,       /* get_internals */
  ossl_close,               /* close_one */
  ossl_close_all,           /* close_all */
  ossl_session_free,        /* session_free */
  ossl_set_engine,          /* set_engine */
  ossl_set_engine_default,  /* set_engine_default */
  ossl_engines_list,        /* engines_list */
  Curl_none_false_start,    /* false_start */
#if (OPENSSL_VERSION_NUMBER >= 0x0090800fL) && !defined(OPENSSL_NO_SHA256)
  ossl_sha256sum,           /* sha256sum */
#else
  NULL,                     /* sha256sum */
#endif
  ossl_associate_connection, /* associate_connection */
  ossl_disassociate_connection /* disassociate_connection */
};

#endif /* USE_OPENSSL */
