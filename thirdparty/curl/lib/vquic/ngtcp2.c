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

#ifdef USE_NGTCP2
#include <ngtcp2/ngtcp2.h>
#include <ngtcp2/ngtcp2_crypto.h>
#include <nghttp3/nghttp3.h>
#ifdef USE_OPENSSL
#include <openssl/err.h>
#include <ngtcp2/ngtcp2_crypto_openssl.h>
#elif defined(USE_GNUTLS)
#include <ngtcp2/ngtcp2_crypto_gnutls.h>
#endif
#include "urldata.h"
#include "sendf.h"
#include "strdup.h"
#include "rand.h"
#include "ngtcp2.h"
#include "multiif.h"
#include "strcase.h"
#include "connect.h"
#include "strerror.h"
#include "dynbuf.h"
#include "vquic.h"
#include "vtls/keylog.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/* #define DEBUG_NGTCP2 */
#ifdef CURLDEBUG
#define DEBUG_HTTP3
#endif
#ifdef DEBUG_HTTP3
#define H3BUGF(x) x
#else
#define H3BUGF(x) do { } while(0)
#endif

#define H3_ALPN_H3_29 "\x5h3-29"
#define H3_ALPN_H3 "\x2h3"

/*
 * This holds outgoing HTTP/3 stream data that is used by nghttp3 until acked.
 * It is used as a circular buffer. Add new bytes at the end until it reaches
 * the far end, then start over at index 0 again.
 */

#define H3_SEND_SIZE (20*1024)
struct h3out {
  uint8_t buf[H3_SEND_SIZE];
  size_t used;   /* number of bytes used in the buffer */
  size_t windex; /* index in the buffer where to start writing the next
                    data block */
};

#define QUIC_MAX_STREAMS (256*1024)
#define QUIC_MAX_DATA (1*1024*1024)
#define QUIC_IDLE_TIMEOUT 60000 /* milliseconds */

#ifdef USE_OPENSSL
#define QUIC_CIPHERS                                                          \
  "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_"               \
  "POLY1305_SHA256:TLS_AES_128_CCM_SHA256"
#define QUIC_GROUPS "P-256:X25519:P-384:P-521"
#elif defined(USE_GNUTLS)
#define QUIC_PRIORITY \
  "NORMAL:-VERS-ALL:+VERS-TLS1.3:-CIPHER-ALL:+AES-128-GCM:+AES-256-GCM:" \
  "+CHACHA20-POLY1305:+AES-128-CCM:-GROUP-ALL:+GROUP-SECP256R1:" \
  "+GROUP-X25519:+GROUP-SECP384R1:+GROUP-SECP521R1:" \
  "%DISABLE_TLS13_COMPAT_MODE"
#endif

static CURLcode ng_process_ingress(struct Curl_easy *data,
                                   curl_socket_t sockfd,
                                   struct quicsocket *qs);
static CURLcode ng_flush_egress(struct Curl_easy *data, int sockfd,
                                struct quicsocket *qs);
static int cb_h3_acked_stream_data(nghttp3_conn *conn, int64_t stream_id,
                                   size_t datalen, void *user_data,
                                   void *stream_user_data);

static ngtcp2_tstamp timestamp(void)
{
  struct curltime ct = Curl_now();
  return ct.tv_sec * NGTCP2_SECONDS + ct.tv_usec * NGTCP2_MICROSECONDS;
}

#ifdef DEBUG_NGTCP2
static void quic_printf(void *user_data, const char *fmt, ...)
{
  va_list ap;
  (void)user_data; /* TODO, use this to do infof() instead long-term */
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
}
#endif

static void qlog_callback(void *user_data, uint32_t flags,
                          const void *data, size_t datalen)
{
  struct quicsocket *qs = (struct quicsocket *)user_data;
  (void)flags;
  if(qs->qlogfd != -1) {
    ssize_t rc = write(qs->qlogfd, data, datalen);
    if(rc == -1) {
      /* on write error, stop further write attempts */
      close(qs->qlogfd);
      qs->qlogfd = -1;
    }
  }

}

static void quic_settings(struct quicsocket *qs,
                          uint64_t stream_buffer_size)
{
  ngtcp2_settings *s = &qs->settings;
  ngtcp2_transport_params *t = &qs->transport_params;
  ngtcp2_settings_default(s);
  ngtcp2_transport_params_default(t);
#ifdef DEBUG_NGTCP2
  s->log_printf = quic_printf;
#else
  s->log_printf = NULL;
#endif
  s->initial_ts = timestamp();
  t->initial_max_stream_data_bidi_local = stream_buffer_size;
  t->initial_max_stream_data_bidi_remote = QUIC_MAX_STREAMS;
  t->initial_max_stream_data_uni = QUIC_MAX_STREAMS;
  t->initial_max_data = QUIC_MAX_DATA;
  t->initial_max_streams_bidi = 1;
  t->initial_max_streams_uni = 3;
  t->max_idle_timeout = QUIC_IDLE_TIMEOUT;
  if(qs->qlogfd != -1) {
    s->qlog.write = qlog_callback;
  }
}

#ifdef USE_OPENSSL
static void keylog_callback(const SSL *ssl, const char *line)
{
  (void)ssl;
  Curl_tls_keylog_write_line(line);
}
#elif defined(USE_GNUTLS)
static int keylog_callback(gnutls_session_t session, const char *label,
                    const gnutls_datum_t *secret)
{
  gnutls_datum_t crandom;
  gnutls_datum_t srandom;

  gnutls_session_get_random(session, &crandom, &srandom);
  if(crandom.size != 32) {
    return -1;
  }

  Curl_tls_keylog_write(label, crandom.data, secret->data, secret->size);
  return 0;
}
#endif

static int init_ngh3_conn(struct quicsocket *qs);

static int write_client_handshake(struct quicsocket *qs,
                                  ngtcp2_crypto_level level,
                                  const uint8_t *data, size_t len)
{
  int rv;

  rv = ngtcp2_conn_submit_crypto_data(qs->qconn, level, data, len);
  if(rv) {
    H3BUGF(fprintf(stderr, "write_client_handshake failed\n"));
  }
  assert(0 == rv);

  return 1;
}

#ifdef USE_OPENSSL
static int quic_set_encryption_secrets(SSL *ssl,
                                       OSSL_ENCRYPTION_LEVEL ossl_level,
                                       const uint8_t *rx_secret,
                                       const uint8_t *tx_secret,
                                       size_t secretlen)
{
  struct quicsocket *qs = (struct quicsocket *)SSL_get_app_data(ssl);
  int level = ngtcp2_crypto_openssl_from_ossl_encryption_level(ossl_level);

  if(ngtcp2_crypto_derive_and_install_rx_key(
       qs->qconn, NULL, NULL, NULL, level, rx_secret, secretlen) != 0)
    return 0;

  if(ngtcp2_crypto_derive_and_install_tx_key(
       qs->qconn, NULL, NULL, NULL, level, tx_secret, secretlen) != 0)
    return 0;

  if(level == NGTCP2_CRYPTO_LEVEL_APPLICATION) {
    if(init_ngh3_conn(qs) != CURLE_OK)
      return 0;
  }

  return 1;
}

static int quic_add_handshake_data(SSL *ssl, OSSL_ENCRYPTION_LEVEL ossl_level,
                                   const uint8_t *data, size_t len)
{
  struct quicsocket *qs = (struct quicsocket *)SSL_get_app_data(ssl);
  ngtcp2_crypto_level level =
      ngtcp2_crypto_openssl_from_ossl_encryption_level(ossl_level);

  return write_client_handshake(qs, level, data, len);
}

static int quic_flush_flight(SSL *ssl)
{
  (void)ssl;
  return 1;
}

static int quic_send_alert(SSL *ssl, enum ssl_encryption_level_t level,
                           uint8_t alert)
{
  struct quicsocket *qs = (struct quicsocket *)SSL_get_app_data(ssl);
  (void)level;

  qs->tls_alert = alert;
  return 1;
}

static SSL_QUIC_METHOD quic_method = {quic_set_encryption_secrets,
                                      quic_add_handshake_data,
                                      quic_flush_flight, quic_send_alert};

static SSL_CTX *quic_ssl_ctx(struct Curl_easy *data)
{
  SSL_CTX *ssl_ctx = SSL_CTX_new(TLS_method());

  SSL_CTX_set_min_proto_version(ssl_ctx, TLS1_3_VERSION);
  SSL_CTX_set_max_proto_version(ssl_ctx, TLS1_3_VERSION);

  SSL_CTX_set_default_verify_paths(ssl_ctx);

  if(SSL_CTX_set_ciphersuites(ssl_ctx, QUIC_CIPHERS) != 1) {
    char error_buffer[256];
    ERR_error_string_n(ERR_get_error(), error_buffer, sizeof(error_buffer));
    failf(data, "SSL_CTX_set_ciphersuites: %s", error_buffer);
    return NULL;
  }

  if(SSL_CTX_set1_groups_list(ssl_ctx, QUIC_GROUPS) != 1) {
    failf(data, "SSL_CTX_set1_groups_list failed");
    return NULL;
  }

  SSL_CTX_set_quic_method(ssl_ctx, &quic_method);

  /* Open the file if a TLS or QUIC backend has not done this before. */
  Curl_tls_keylog_open();
  if(Curl_tls_keylog_enabled()) {
    SSL_CTX_set_keylog_callback(ssl_ctx, keylog_callback);
  }

  return ssl_ctx;
}

/** SSL callbacks ***/

static int quic_init_ssl(struct quicsocket *qs)
{
  const uint8_t *alpn = NULL;
  size_t alpnlen = 0;
  /* this will need some attention when HTTPS proxy over QUIC get fixed */
  const char * const hostname = qs->conn->host.name;

  DEBUGASSERT(!qs->ssl);
  qs->ssl = SSL_new(qs->sslctx);

  SSL_set_app_data(qs->ssl, qs);
  SSL_set_connect_state(qs->ssl);
  SSL_set_quic_use_legacy_codepoint(qs->ssl, 0);

  alpn = (const uint8_t *)H3_ALPN_H3_29 H3_ALPN_H3;
  alpnlen = sizeof(H3_ALPN_H3_29) - 1 + sizeof(H3_ALPN_H3) - 1;
  if(alpn)
    SSL_set_alpn_protos(qs->ssl, alpn, (int)alpnlen);

  /* set SNI */
  SSL_set_tlsext_host_name(qs->ssl, hostname);
  return 0;
}
#elif defined(USE_GNUTLS)
static int secret_func(gnutls_session_t ssl,
                       gnutls_record_encryption_level_t gtls_level,
                       const void *rx_secret,
                       const void *tx_secret, size_t secretlen)
{
  struct quicsocket *qs = gnutls_session_get_ptr(ssl);
  int level =
      ngtcp2_crypto_gnutls_from_gnutls_record_encryption_level(gtls_level);

  if(level != NGTCP2_CRYPTO_LEVEL_EARLY &&
     ngtcp2_crypto_derive_and_install_rx_key(
       qs->qconn, NULL, NULL, NULL, level, rx_secret, secretlen) != 0)
    return 0;

  if(ngtcp2_crypto_derive_and_install_tx_key(
       qs->qconn, NULL, NULL, NULL, level, tx_secret, secretlen) != 0)
    return 0;

  if(level == NGTCP2_CRYPTO_LEVEL_APPLICATION) {
    if(init_ngh3_conn(qs) != CURLE_OK)
      return -1;
  }

  return 0;
}

static int read_func(gnutls_session_t ssl,
                     gnutls_record_encryption_level_t gtls_level,
                     gnutls_handshake_description_t htype, const void *data,
                     size_t len)
{
  struct quicsocket *qs = gnutls_session_get_ptr(ssl);
  ngtcp2_crypto_level level =
      ngtcp2_crypto_gnutls_from_gnutls_record_encryption_level(gtls_level);
  int rv;

  if(htype == GNUTLS_HANDSHAKE_CHANGE_CIPHER_SPEC)
    return 0;

  rv = write_client_handshake(qs, level, data, len);
  if(rv == 0)
    return -1;

  return 0;
}

static int alert_read_func(gnutls_session_t ssl,
                           gnutls_record_encryption_level_t gtls_level,
                           gnutls_alert_level_t alert_level,
                           gnutls_alert_description_t alert_desc)
{
  struct quicsocket *qs = gnutls_session_get_ptr(ssl);
  (void)gtls_level;
  (void)alert_level;

  qs->tls_alert = alert_desc;
  return 1;
}

static int tp_recv_func(gnutls_session_t ssl, const uint8_t *data,
                        size_t data_size)
{
  struct quicsocket *qs = gnutls_session_get_ptr(ssl);
  ngtcp2_transport_params params;

  if(ngtcp2_decode_transport_params(
       &params, NGTCP2_TRANSPORT_PARAMS_TYPE_ENCRYPTED_EXTENSIONS,
       data, data_size) != 0)
    return -1;

  if(ngtcp2_conn_set_remote_transport_params(qs->qconn, &params) != 0)
    return -1;

  return 0;
}

static int tp_send_func(gnutls_session_t ssl, gnutls_buffer_t extdata)
{
  struct quicsocket *qs = gnutls_session_get_ptr(ssl);
  uint8_t paramsbuf[64];
  ngtcp2_transport_params params;
  ssize_t nwrite;
  int rc;

  ngtcp2_conn_get_local_transport_params(qs->qconn, &params);
  nwrite = ngtcp2_encode_transport_params(
    paramsbuf, sizeof(paramsbuf), NGTCP2_TRANSPORT_PARAMS_TYPE_CLIENT_HELLO,
    &params);
  if(nwrite < 0) {
    H3BUGF(fprintf(stderr, "ngtcp2_encode_transport_params: %s\n",
                   ngtcp2_strerror((int)nwrite)));
    return -1;
  }

  rc = gnutls_buffer_append_data(extdata, paramsbuf, nwrite);
  if(rc < 0)
    return rc;

  return (int)nwrite;
}

static int quic_init_ssl(struct quicsocket *qs)
{
  gnutls_datum_t alpn[2];
  /* this will need some attention when HTTPS proxy over QUIC get fixed */
  const char * const hostname = qs->conn->host.name;
  int rc;

  DEBUGASSERT(!qs->ssl);

  gnutls_init(&qs->ssl, GNUTLS_CLIENT);
  gnutls_session_set_ptr(qs->ssl, qs);

  rc = gnutls_priority_set_direct(qs->ssl, QUIC_PRIORITY, NULL);
  if(rc < 0) {
    H3BUGF(fprintf(stderr, "gnutls_priority_set_direct failed: %s\n",
                   gnutls_strerror(rc)));
    return 1;
  }

  gnutls_handshake_set_secret_function(qs->ssl, secret_func);
  gnutls_handshake_set_read_function(qs->ssl, read_func);
  gnutls_alert_set_read_function(qs->ssl, alert_read_func);

  rc = gnutls_session_ext_register(qs->ssl, "QUIC Transport Parameters",
         NGTCP2_TLSEXT_QUIC_TRANSPORT_PARAMETERS_V1, GNUTLS_EXT_TLS,
         tp_recv_func, tp_send_func, NULL, NULL, NULL,
         GNUTLS_EXT_FLAG_TLS | GNUTLS_EXT_FLAG_CLIENT_HELLO |
         GNUTLS_EXT_FLAG_EE);
  if(rc < 0) {
    H3BUGF(fprintf(stderr, "gnutls_session_ext_register failed: %s\n",
                   gnutls_strerror(rc)));
    return 1;
  }

  /* Open the file if a TLS or QUIC backend has not done this before. */
  Curl_tls_keylog_open();
  if(Curl_tls_keylog_enabled()) {
    gnutls_session_set_keylog_function(qs->ssl, keylog_callback);
  }

  if(qs->cred)
    gnutls_certificate_free_credentials(qs->cred);

  rc = gnutls_certificate_allocate_credentials(&qs->cred);
  if(rc < 0) {
    H3BUGF(fprintf(stderr,
                   "gnutls_certificate_allocate_credentials failed: %s\n",
                   gnutls_strerror(rc)));
    return 1;
  }

  rc = gnutls_certificate_set_x509_system_trust(qs->cred);
  if(rc < 0) {
    H3BUGF(fprintf(stderr,
                   "gnutls_certificate_set_x509_system_trust failed: %s\n",
                   gnutls_strerror(rc)));
    return 1;
  }

  rc = gnutls_credentials_set(qs->ssl, GNUTLS_CRD_CERTIFICATE, qs->cred);
  if(rc < 0) {
    H3BUGF(fprintf(stderr, "gnutls_credentials_set failed: %s\n",
                   gnutls_strerror(rc)));
    return 1;
  }

  /* strip the first byte (the length) from NGHTTP3_ALPN_H3 */
  alpn[0].data = (unsigned char *)H3_ALPN_H3_29 + 1;
  alpn[0].size = sizeof(H3_ALPN_H3_29) - 2;
  alpn[1].data = (unsigned char *)H3_ALPN_H3 + 1;
  alpn[1].size = sizeof(H3_ALPN_H3) - 2;

  gnutls_alpn_set_protocols(qs->ssl, alpn, 2, GNUTLS_ALPN_MANDATORY);

  /* set SNI */
  gnutls_server_name_set(qs->ssl, GNUTLS_NAME_DNS, hostname, strlen(hostname));
  return 0;
}
#endif

static int cb_handshake_completed(ngtcp2_conn *tconn, void *user_data)
{
  (void)user_data;
  (void)tconn;
  return 0;
}

static void extend_stream_window(ngtcp2_conn *tconn,
                                 struct HTTP *stream)
{
  size_t thismuch = stream->unacked_window;
  ngtcp2_conn_extend_max_stream_offset(tconn, stream->stream3_id, thismuch);
  ngtcp2_conn_extend_max_offset(tconn, thismuch);
  stream->unacked_window = 0;
}


static int cb_recv_stream_data(ngtcp2_conn *tconn, uint32_t flags,
                               int64_t stream_id, uint64_t offset,
                               const uint8_t *buf, size_t buflen,
                               void *user_data, void *stream_user_data)
{
  struct quicsocket *qs = (struct quicsocket *)user_data;
  ssize_t nconsumed;
  int fin = (flags & NGTCP2_STREAM_DATA_FLAG_FIN) ? 1 : 0;
  (void)offset;
  (void)stream_user_data;

  nconsumed =
    nghttp3_conn_read_stream(qs->h3conn, stream_id, buf, buflen, fin);
  if(nconsumed < 0) {
    return NGTCP2_ERR_CALLBACK_FAILURE;
  }

  /* number of bytes inside buflen which consists of framing overhead
   * including QPACK HEADERS. In other words, it does not consume payload of
   * DATA frame. */
  ngtcp2_conn_extend_max_stream_offset(tconn, stream_id, nconsumed);
  ngtcp2_conn_extend_max_offset(tconn, nconsumed);

  return 0;
}

static int
cb_acked_stream_data_offset(ngtcp2_conn *tconn, int64_t stream_id,
                            uint64_t offset, uint64_t datalen, void *user_data,
                            void *stream_user_data)
{
  struct quicsocket *qs = (struct quicsocket *)user_data;
  int rv;
  (void)stream_id;
  (void)tconn;
  (void)offset;
  (void)datalen;
  (void)stream_user_data;

  rv = nghttp3_conn_add_ack_offset(qs->h3conn, stream_id, datalen);
  if(rv) {
    return NGTCP2_ERR_CALLBACK_FAILURE;
  }

  return 0;
}

static int cb_stream_close(ngtcp2_conn *tconn, uint32_t flags,
                           int64_t stream_id, uint64_t app_error_code,
                           void *user_data, void *stream_user_data)
{
  struct quicsocket *qs = (struct quicsocket *)user_data;
  int rv;
  (void)tconn;
  (void)stream_user_data;
  /* stream is closed... */

  if(!(flags & NGTCP2_STREAM_CLOSE_FLAG_APP_ERROR_CODE_SET)) {
    app_error_code = NGHTTP3_H3_NO_ERROR;
  }

  rv = nghttp3_conn_close_stream(qs->h3conn, stream_id,
                                 app_error_code);
  if(rv) {
    return NGTCP2_ERR_CALLBACK_FAILURE;
  }

  return 0;
}

static int cb_stream_reset(ngtcp2_conn *tconn, int64_t stream_id,
                           uint64_t final_size, uint64_t app_error_code,
                           void *user_data, void *stream_user_data)
{
  struct quicsocket *qs = (struct quicsocket *)user_data;
  int rv;
  (void)tconn;
  (void)final_size;
  (void)app_error_code;
  (void)stream_user_data;

  rv = nghttp3_conn_shutdown_stream_read(qs->h3conn, stream_id);
  if(rv) {
    return NGTCP2_ERR_CALLBACK_FAILURE;
  }

  return 0;
}

static int cb_stream_stop_sending(ngtcp2_conn *tconn, int64_t stream_id,
                                  uint64_t app_error_code, void *user_data,
                                  void *stream_user_data)
{
  struct quicsocket *qs = (struct quicsocket *)user_data;
  int rv;
  (void)tconn;
  (void)app_error_code;
  (void)stream_user_data;

  rv = nghttp3_conn_shutdown_stream_read(qs->h3conn, stream_id);
  if(rv) {
    return NGTCP2_ERR_CALLBACK_FAILURE;
  }

  return 0;
}

static int cb_extend_max_local_streams_bidi(ngtcp2_conn *tconn,
                                            uint64_t max_streams,
                                            void *user_data)
{
  (void)tconn;
  (void)max_streams;
  (void)user_data;

  return 0;
}

static int cb_extend_max_stream_data(ngtcp2_conn *tconn, int64_t stream_id,
                                     uint64_t max_data, void *user_data,
                                     void *stream_user_data)
{
  struct quicsocket *qs = (struct quicsocket *)user_data;
  int rv;
  (void)tconn;
  (void)max_data;
  (void)stream_user_data;

  rv = nghttp3_conn_unblock_stream(qs->h3conn, stream_id);
  if(rv) {
    return NGTCP2_ERR_CALLBACK_FAILURE;
  }

  return 0;
}

static void cb_rand(uint8_t *dest, size_t destlen,
                    const ngtcp2_rand_ctx *rand_ctx)
{
  CURLcode result;
  (void)rand_ctx;

  result = Curl_rand(NULL, dest, destlen);
  if(result) {
    /* cb_rand is only used for non-cryptographic context.  If Curl_rand
       failed, just fill 0 and call it *random*. */
    memset(dest, 0, destlen);
  }
}

static int cb_get_new_connection_id(ngtcp2_conn *tconn, ngtcp2_cid *cid,
                                    uint8_t *token, size_t cidlen,
                                    void *user_data)
{
  CURLcode result;
  (void)tconn;
  (void)user_data;

  result = Curl_rand(NULL, cid->data, cidlen);
  if(result)
    return NGTCP2_ERR_CALLBACK_FAILURE;
  cid->datalen = cidlen;

  result = Curl_rand(NULL, token, NGTCP2_STATELESS_RESET_TOKENLEN);
  if(result)
    return NGTCP2_ERR_CALLBACK_FAILURE;

  return 0;
}

static ngtcp2_callbacks ng_callbacks = {
  ngtcp2_crypto_client_initial_cb,
  NULL, /* recv_client_initial */
  ngtcp2_crypto_recv_crypto_data_cb,
  cb_handshake_completed,
  NULL, /* recv_version_negotiation */
  ngtcp2_crypto_encrypt_cb,
  ngtcp2_crypto_decrypt_cb,
  ngtcp2_crypto_hp_mask_cb,
  cb_recv_stream_data,
  cb_acked_stream_data_offset,
  NULL, /* stream_open */
  cb_stream_close,
  NULL, /* recv_stateless_reset */
  ngtcp2_crypto_recv_retry_cb,
  cb_extend_max_local_streams_bidi,
  NULL, /* extend_max_local_streams_uni */
  cb_rand,
  cb_get_new_connection_id,
  NULL, /* remove_connection_id */
  ngtcp2_crypto_update_key_cb, /* update_key */
  NULL, /* path_validation */
  NULL, /* select_preferred_addr */
  cb_stream_reset,
  NULL, /* extend_max_remote_streams_bidi */
  NULL, /* extend_max_remote_streams_uni */
  cb_extend_max_stream_data,
  NULL, /* dcid_status */
  NULL, /* handshake_confirmed */
  NULL, /* recv_new_token */
  ngtcp2_crypto_delete_crypto_aead_ctx_cb,
  ngtcp2_crypto_delete_crypto_cipher_ctx_cb,
  NULL, /* recv_datagram */
  NULL, /* ack_datagram */
  NULL, /* lost_datagram */
  ngtcp2_crypto_get_path_challenge_data_cb,
  cb_stream_stop_sending
};

/*
 * Might be called twice for happy eyeballs.
 */
CURLcode Curl_quic_connect(struct Curl_easy *data,
                           struct connectdata *conn,
                           curl_socket_t sockfd,
                           int sockindex,
                           const struct sockaddr *addr,
                           socklen_t addrlen)
{
  int rc;
  int rv;
  CURLcode result;
  ngtcp2_path path; /* TODO: this must be initialized properly */
  struct quicsocket *qs = &conn->hequic[sockindex];
  char ipbuf[40];
  int port;
  int qfd;

  if(qs->conn)
    Curl_quic_disconnect(data, conn, sockindex);
  qs->conn = conn;

  /* extract the used address as a string */
  if(!Curl_addr2string((struct sockaddr*)addr, addrlen, ipbuf, &port)) {
    char buffer[STRERROR_LEN];
    failf(data, "ssrem inet_ntop() failed with errno %d: %s",
          SOCKERRNO, Curl_strerror(SOCKERRNO, buffer, sizeof(buffer)));
    return CURLE_BAD_FUNCTION_ARGUMENT;
  }

  infof(data, "Connect socket %d over QUIC to %s:%d",
        sockfd, ipbuf, port);

  qs->version = NGTCP2_PROTO_VER_MAX;
#ifdef USE_OPENSSL
  qs->sslctx = quic_ssl_ctx(data);
  if(!qs->sslctx)
    return CURLE_QUIC_CONNECT_ERROR;
#endif

  if(quic_init_ssl(qs))
    return CURLE_QUIC_CONNECT_ERROR;

  qs->dcid.datalen = NGTCP2_MAX_CIDLEN;
  result = Curl_rand(data, qs->dcid.data, NGTCP2_MAX_CIDLEN);
  if(result)
    return result;

  qs->scid.datalen = NGTCP2_MAX_CIDLEN;
  result = Curl_rand(data, qs->scid.data, NGTCP2_MAX_CIDLEN);
  if(result)
    return result;

  (void)Curl_qlogdir(data, qs->scid.data, NGTCP2_MAX_CIDLEN, &qfd);
  qs->qlogfd = qfd; /* -1 if failure above */
  quic_settings(qs, data->set.buffer_size);

  qs->local_addrlen = sizeof(qs->local_addr);
  rv = getsockname(sockfd, (struct sockaddr *)&qs->local_addr,
                   &qs->local_addrlen);
  if(rv == -1)
    return CURLE_QUIC_CONNECT_ERROR;

  ngtcp2_addr_init(&path.local, (struct sockaddr *)&qs->local_addr,
                   qs->local_addrlen);
  ngtcp2_addr_init(&path.remote, addr, addrlen);

  rc = ngtcp2_conn_client_new(&qs->qconn, &qs->dcid, &qs->scid, &path,
                              NGTCP2_PROTO_VER_V1, &ng_callbacks,
                              &qs->settings, &qs->transport_params, NULL, qs);
  if(rc)
    return CURLE_QUIC_CONNECT_ERROR;

  ngtcp2_conn_set_tls_native_handle(qs->qconn, qs->ssl);

  return CURLE_OK;
}

/*
 * Store ngtcp2 version info in this buffer.
 */
void Curl_quic_ver(char *p, size_t len)
{
  const ngtcp2_info *ng2 = ngtcp2_version(0);
  const nghttp3_info *ht3 = nghttp3_version(0);
  (void)msnprintf(p, len, "ngtcp2/%s nghttp3/%s",
                  ng2->version_str, ht3->version_str);
}

static int ng_getsock(struct Curl_easy *data, struct connectdata *conn,
                      curl_socket_t *socks)
{
  struct SingleRequest *k = &data->req;
  int bitmap = GETSOCK_BLANK;

  socks[0] = conn->sock[FIRSTSOCKET];

  /* in a HTTP/2 connection we can basically always get a frame so we should
     always be ready for one */
  bitmap |= GETSOCK_READSOCK(FIRSTSOCKET);

  /* we're still uploading or the HTTP/2 layer wants to send data */
  if((k->keepon & (KEEP_SEND|KEEP_SEND_PAUSE)) == KEEP_SEND)
    bitmap |= GETSOCK_WRITESOCK(FIRSTSOCKET);

  return bitmap;
}

static void qs_disconnect(struct quicsocket *qs)
{
  if(!qs->conn) /* already closed */
    return;
  qs->conn = NULL;
  if(qs->qlogfd != -1) {
    close(qs->qlogfd);
    qs->qlogfd = -1;
  }
  if(qs->ssl)
#ifdef USE_OPENSSL
    SSL_free(qs->ssl);
#elif defined(USE_GNUTLS)
    gnutls_deinit(qs->ssl);
#endif
  qs->ssl = NULL;
#ifdef USE_GNUTLS
  if(qs->cred) {
    gnutls_certificate_free_credentials(qs->cred);
    qs->cred = NULL;
  }
#endif
  nghttp3_conn_del(qs->h3conn);
  ngtcp2_conn_del(qs->qconn);
#ifdef USE_OPENSSL
  SSL_CTX_free(qs->sslctx);
#endif
}

void Curl_quic_disconnect(struct Curl_easy *data,
                          struct connectdata *conn,
                          int tempindex)
{
  (void)data;
  if(conn->transport == TRNSPRT_QUIC)
    qs_disconnect(&conn->hequic[tempindex]);
}

static CURLcode ng_disconnect(struct Curl_easy *data,
                              struct connectdata *conn,
                              bool dead_connection)
{
  (void)dead_connection;
  Curl_quic_disconnect(data, conn, 0);
  Curl_quic_disconnect(data, conn, 1);
  return CURLE_OK;
}

static unsigned int ng_conncheck(struct Curl_easy *data,
                                 struct connectdata *conn,
                                 unsigned int checks_to_perform)
{
  (void)data;
  (void)conn;
  (void)checks_to_perform;
  return CONNRESULT_NONE;
}

static const struct Curl_handler Curl_handler_http3 = {
  "HTTPS",                              /* scheme */
  ZERO_NULL,                            /* setup_connection */
  Curl_http,                            /* do_it */
  Curl_http_done,                       /* done */
  ZERO_NULL,                            /* do_more */
  ZERO_NULL,                            /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ng_getsock,                           /* proto_getsock */
  ng_getsock,                           /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ng_getsock,                           /* perform_getsock */
  ng_disconnect,                        /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ng_conncheck,                         /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_HTTP,                            /* defport */
  CURLPROTO_HTTPS,                      /* protocol */
  CURLPROTO_HTTP,                       /* family */
  PROTOPT_SSL | PROTOPT_STREAM          /* flags */
};

static int cb_h3_stream_close(nghttp3_conn *conn, int64_t stream_id,
                              uint64_t app_error_code, void *user_data,
                              void *stream_user_data)
{
  struct Curl_easy *data = stream_user_data;
  struct HTTP *stream = data->req.p.http;
  (void)conn;
  (void)stream_id;
  (void)app_error_code;
  (void)user_data;
  H3BUGF(infof(data, "cb_h3_stream_close CALLED"));

  stream->closed = TRUE;
  Curl_expire(data, 0, EXPIRE_QUIC);
  /* make sure that ngh3_stream_recv is called again to complete the transfer
     even if there are no more packets to be received from the server. */
  data->state.drain = 1;
  return 0;
}

/*
 * write_data() copies data to the stream's receive buffer. If not enough
 * space is available in the receive buffer, it copies the rest to the
 * stream's overflow buffer.
 */
static CURLcode write_data(struct HTTP *stream, const void *mem, size_t memlen)
{
  CURLcode result = CURLE_OK;
  const char *buf = mem;
  size_t ncopy = memlen;
  /* copy as much as possible to the receive buffer */
  if(stream->len) {
    size_t len = CURLMIN(ncopy, stream->len);
    memcpy(stream->mem, buf, len);
    stream->len -= len;
    stream->memlen += len;
    stream->mem += len;
    buf += len;
    ncopy -= len;
  }
  /* copy the rest to the overflow buffer */
  if(ncopy)
    result = Curl_dyn_addn(&stream->overflow, buf, ncopy);
  return result;
}

static int cb_h3_recv_data(nghttp3_conn *conn, int64_t stream_id,
                           const uint8_t *buf, size_t buflen,
                           void *user_data, void *stream_user_data)
{
  struct Curl_easy *data = stream_user_data;
  struct HTTP *stream = data->req.p.http;
  CURLcode result = CURLE_OK;
  (void)conn;

  result = write_data(stream, buf, buflen);
  if(result) {
    return -1;
  }
  stream->unacked_window += buflen;
  (void)stream_id;
  (void)user_data;
  return 0;
}

static int cb_h3_deferred_consume(nghttp3_conn *conn, int64_t stream_id,
                                  size_t consumed, void *user_data,
                                  void *stream_user_data)
{
  struct quicsocket *qs = user_data;
  (void)conn;
  (void)stream_user_data;
  (void)stream_id;

  ngtcp2_conn_extend_max_stream_offset(qs->qconn, stream_id, consumed);
  ngtcp2_conn_extend_max_offset(qs->qconn, consumed);
  return 0;
}

/* Decode HTTP status code.  Returns -1 if no valid status code was
   decoded. (duplicate from http2.c) */
static int decode_status_code(const uint8_t *value, size_t len)
{
  int i;
  int res;

  if(len != 3) {
    return -1;
  }

  res = 0;

  for(i = 0; i < 3; ++i) {
    char c = value[i];

    if(c < '0' || c > '9') {
      return -1;
    }

    res *= 10;
    res += c - '0';
  }

  return res;
}

static int cb_h3_end_headers(nghttp3_conn *conn, int64_t stream_id,
                             void *user_data, void *stream_user_data)
{
  struct Curl_easy *data = stream_user_data;
  struct HTTP *stream = data->req.p.http;
  CURLcode result = CURLE_OK;
  (void)conn;
  (void)stream_id;
  (void)user_data;

  /* add a CRLF only if we've received some headers */
  if(stream->firstheader) {
    result = write_data(stream, "\r\n", 2);
    if(result) {
      return -1;
    }
  }
  return 0;
}

static int cb_h3_recv_header(nghttp3_conn *conn, int64_t stream_id,
                             int32_t token, nghttp3_rcbuf *name,
                             nghttp3_rcbuf *value, uint8_t flags,
                             void *user_data, void *stream_user_data)
{
  nghttp3_vec h3name = nghttp3_rcbuf_get_buf(name);
  nghttp3_vec h3val = nghttp3_rcbuf_get_buf(value);
  struct Curl_easy *data = stream_user_data;
  struct HTTP *stream = data->req.p.http;
  CURLcode result = CURLE_OK;
  (void)conn;
  (void)stream_id;
  (void)token;
  (void)flags;
  (void)user_data;

  if(h3name.len == sizeof(":status") - 1 &&
     !memcmp(":status", h3name.base, h3name.len)) {
    char line[14]; /* status line is always 13 characters long */
    size_t ncopy;
    int status = decode_status_code(h3val.base, h3val.len);
    DEBUGASSERT(status != -1);
    ncopy = msnprintf(line, sizeof(line), "HTTP/3 %03d \r\n", status);
    result = write_data(stream, line, ncopy);
    if(result) {
      return -1;
    }
  }
  else {
    /* store as a HTTP1-style header */
    result = write_data(stream, h3name.base, h3name.len);
    if(result) {
      return -1;
    }
    result = write_data(stream, ": ", 2);
    if(result) {
      return -1;
    }
    result = write_data(stream, h3val.base, h3val.len);
    if(result) {
      return -1;
    }
    result = write_data(stream, "\r\n", 2);
    if(result) {
      return -1;
    }
  }

  stream->firstheader = TRUE;
  return 0;
}

static int cb_h3_send_stop_sending(nghttp3_conn *conn, int64_t stream_id,
                                   uint64_t app_error_code,
                                   void *user_data,
                                   void *stream_user_data)
{
  (void)conn;
  (void)stream_id;
  (void)app_error_code;
  (void)user_data;
  (void)stream_user_data;
  return 0;
}

static nghttp3_callbacks ngh3_callbacks = {
  cb_h3_acked_stream_data, /* acked_stream_data */
  cb_h3_stream_close,
  cb_h3_recv_data,
  cb_h3_deferred_consume,
  NULL, /* begin_headers */
  cb_h3_recv_header,
  cb_h3_end_headers,
  NULL, /* begin_trailers */
  cb_h3_recv_header,
  NULL, /* end_trailers */
  cb_h3_send_stop_sending,
  NULL, /* end_stream */
  NULL, /* reset_stream */
  NULL /* shutdown */
};

static int init_ngh3_conn(struct quicsocket *qs)
{
  CURLcode result;
  int rc;
  int64_t ctrl_stream_id, qpack_enc_stream_id, qpack_dec_stream_id;

  if(ngtcp2_conn_get_max_local_streams_uni(qs->qconn) < 3) {
    return CURLE_QUIC_CONNECT_ERROR;
  }

  nghttp3_settings_default(&qs->h3settings);

  rc = nghttp3_conn_client_new(&qs->h3conn,
                               &ngh3_callbacks,
                               &qs->h3settings,
                               nghttp3_mem_default(),
                               qs);
  if(rc) {
    result = CURLE_OUT_OF_MEMORY;
    goto fail;
  }

  rc = ngtcp2_conn_open_uni_stream(qs->qconn, &ctrl_stream_id, NULL);
  if(rc) {
    result = CURLE_QUIC_CONNECT_ERROR;
    goto fail;
  }

  rc = nghttp3_conn_bind_control_stream(qs->h3conn, ctrl_stream_id);
  if(rc) {
    result = CURLE_QUIC_CONNECT_ERROR;
    goto fail;
  }

  rc = ngtcp2_conn_open_uni_stream(qs->qconn, &qpack_enc_stream_id, NULL);
  if(rc) {
    result = CURLE_QUIC_CONNECT_ERROR;
    goto fail;
  }

  rc = ngtcp2_conn_open_uni_stream(qs->qconn, &qpack_dec_stream_id, NULL);
  if(rc) {
    result = CURLE_QUIC_CONNECT_ERROR;
    goto fail;
  }

  rc = nghttp3_conn_bind_qpack_streams(qs->h3conn, qpack_enc_stream_id,
                                       qpack_dec_stream_id);
  if(rc) {
    result = CURLE_QUIC_CONNECT_ERROR;
    goto fail;
  }

  return CURLE_OK;
  fail:

  return result;
}

static Curl_recv ngh3_stream_recv;
static Curl_send ngh3_stream_send;

static size_t drain_overflow_buffer(struct HTTP *stream)
{
  size_t overlen = Curl_dyn_len(&stream->overflow);
  size_t ncopy = CURLMIN(overlen, stream->len);
  if(ncopy > 0) {
    memcpy(stream->mem, Curl_dyn_ptr(&stream->overflow), ncopy);
    stream->len -= ncopy;
    stream->mem += ncopy;
    stream->memlen += ncopy;
    if(ncopy != overlen)
      /* make the buffer only keep the tail */
      (void)Curl_dyn_tail(&stream->overflow, overlen - ncopy);
  }
  return ncopy;
}

/* incoming data frames on the h3 stream */
static ssize_t ngh3_stream_recv(struct Curl_easy *data,
                                int sockindex,
                                char *buf,
                                size_t buffersize,
                                CURLcode *curlcode)
{
  struct connectdata *conn = data->conn;
  curl_socket_t sockfd = conn->sock[sockindex];
  struct HTTP *stream = data->req.p.http;
  struct quicsocket *qs = conn->quic;

  if(!stream->memlen) {
    /* remember where to store incoming data for this stream and how big the
       buffer is */
    stream->mem = buf;
    stream->len = buffersize;
  }
  /* else, there's data in the buffer already */

  /* if there's data in the overflow buffer from a previous call, copy as much
     as possible to the receive buffer before receiving more */
  drain_overflow_buffer(stream);

  if(ng_process_ingress(data, sockfd, qs)) {
    *curlcode = CURLE_RECV_ERROR;
    return -1;
  }
  if(ng_flush_egress(data, sockfd, qs)) {
    *curlcode = CURLE_SEND_ERROR;
    return -1;
  }

  if(stream->memlen) {
    ssize_t memlen = stream->memlen;
    /* data arrived */
    *curlcode = CURLE_OK;
    /* reset to allow more data to come */
    stream->memlen = 0;
    stream->mem = buf;
    stream->len = buffersize;
    /* extend the stream window with the data we're consuming and send out
       any additional packets to tell the server that we can receive more */
    extend_stream_window(qs->qconn, stream);
    if(ng_flush_egress(data, sockfd, qs)) {
      *curlcode = CURLE_SEND_ERROR;
      return -1;
    }
    return memlen;
  }

  if(stream->closed) {
    *curlcode = CURLE_OK;
    return 0;
  }

  infof(data, "ngh3_stream_recv returns 0 bytes and EAGAIN");
  *curlcode = CURLE_AGAIN;
  return -1;
}

/* this amount of data has now been acked on this stream */
static int cb_h3_acked_stream_data(nghttp3_conn *conn, int64_t stream_id,
                                   size_t datalen, void *user_data,
                                   void *stream_user_data)
{
  struct Curl_easy *data = stream_user_data;
  struct HTTP *stream = data->req.p.http;
  (void)user_data;

  if(!data->set.postfields) {
    stream->h3out->used -= datalen;
    H3BUGF(infof(data,
                 "cb_h3_acked_stream_data, %zd bytes, %zd left unacked",
                 datalen, stream->h3out->used));
    DEBUGASSERT(stream->h3out->used < H3_SEND_SIZE);

    if(stream->h3out->used == 0) {
      int rv = nghttp3_conn_resume_stream(conn, stream_id);
      if(rv) {
        return NGTCP2_ERR_CALLBACK_FAILURE;
      }
    }
  }
  return 0;
}

static ssize_t cb_h3_readfunction(nghttp3_conn *conn, int64_t stream_id,
                                  nghttp3_vec *vec, size_t veccnt,
                                  uint32_t *pflags, void *user_data,
                                  void *stream_user_data)
{
  struct Curl_easy *data = stream_user_data;
  size_t nread;
  struct HTTP *stream = data->req.p.http;
  (void)conn;
  (void)stream_id;
  (void)user_data;
  (void)veccnt;

  if(data->set.postfields) {
    vec[0].base = data->set.postfields;
    vec[0].len = data->state.infilesize;
    *pflags = NGHTTP3_DATA_FLAG_EOF;
    return 1;
  }

  nread = CURLMIN(stream->upload_len, H3_SEND_SIZE - stream->h3out->used);
  if(nread > 0) {
    /* nghttp3 wants us to hold on to the data until it tells us it is okay to
       delete it. Append the data at the end of the h3out buffer. Since we can
       only return consecutive data, copy the amount that fits and the next
       part comes in next invoke. */
    struct h3out *out = stream->h3out;
    if(nread + out->windex > H3_SEND_SIZE)
      nread = H3_SEND_SIZE - out->windex;

    memcpy(&out->buf[out->windex], stream->upload_mem, nread);

    /* that's the chunk we return to nghttp3 */
    vec[0].base = &out->buf[out->windex];
    vec[0].len = nread;

    out->windex += nread;
    out->used += nread;

    if(out->windex == H3_SEND_SIZE)
      out->windex = 0; /* wrap */
    stream->upload_mem += nread;
    stream->upload_len -= nread;
    if(data->state.infilesize != -1) {
      stream->upload_left -= nread;
      if(!stream->upload_left)
        *pflags = NGHTTP3_DATA_FLAG_EOF;
    }
    H3BUGF(infof(data, "cb_h3_readfunction %zd bytes%s (at %zd unacked)",
                 nread, *pflags == NGHTTP3_DATA_FLAG_EOF?" EOF":"",
                 out->used));
  }
  if(stream->upload_done && !stream->upload_len &&
     (stream->upload_left <= 0)) {
    H3BUGF(infof(data, "!!!!!!!!! cb_h3_readfunction sets EOF"));
    *pflags = NGHTTP3_DATA_FLAG_EOF;
    return nread ? 1 : 0;
  }
  else if(!nread) {
    return NGHTTP3_ERR_WOULDBLOCK;
  }
  return 1;
}

/* Index where :authority header field will appear in request header
   field list. */
#define AUTHORITY_DST_IDX 3

static CURLcode http_request(struct Curl_easy *data, const void *mem,
                             size_t len)
{
  struct connectdata *conn = data->conn;
  struct HTTP *stream = data->req.p.http;
  size_t nheader;
  size_t i;
  size_t authority_idx;
  char *hdbuf = (char *)mem;
  char *end, *line_end;
  struct quicsocket *qs = conn->quic;
  CURLcode result = CURLE_OK;
  nghttp3_nv *nva = NULL;
  int64_t stream3_id;
  int rc;
  struct h3out *h3out = NULL;

  rc = ngtcp2_conn_open_bidi_stream(qs->qconn, &stream3_id, NULL);
  if(rc) {
    failf(data, "can get bidi streams");
    result = CURLE_SEND_ERROR;
    goto fail;
  }

  stream->stream3_id = stream3_id;
  stream->h3req = TRUE; /* senf off! */
  Curl_dyn_init(&stream->overflow, CURL_MAX_READ_SIZE);

  /* Calculate number of headers contained in [mem, mem + len). Assumes a
     correctly generated HTTP header field block. */
  nheader = 0;
  for(i = 1; i < len; ++i) {
    if(hdbuf[i] == '\n' && hdbuf[i - 1] == '\r') {
      ++nheader;
      ++i;
    }
  }
  if(nheader < 2)
    goto fail;

  /* We counted additional 2 \r\n in the first and last line. We need 3
     new headers: :method, :path and :scheme. Therefore we need one
     more space. */
  nheader += 1;
  nva = malloc(sizeof(nghttp3_nv) * nheader);
  if(!nva) {
    result = CURLE_OUT_OF_MEMORY;
    goto fail;
  }

  /* Extract :method, :path from request line
     We do line endings with CRLF so checking for CR is enough */
  line_end = memchr(hdbuf, '\r', len);
  if(!line_end) {
    result = CURLE_BAD_FUNCTION_ARGUMENT; /* internal error */
    goto fail;
  }

  /* Method does not contain spaces */
  end = memchr(hdbuf, ' ', line_end - hdbuf);
  if(!end || end == hdbuf)
    goto fail;
  nva[0].name = (unsigned char *)":method";
  nva[0].namelen = strlen((char *)nva[0].name);
  nva[0].value = (unsigned char *)hdbuf;
  nva[0].valuelen = (size_t)(end - hdbuf);
  nva[0].flags = NGHTTP3_NV_FLAG_NONE;

  hdbuf = end + 1;

  /* Path may contain spaces so scan backwards */
  end = NULL;
  for(i = (size_t)(line_end - hdbuf); i; --i) {
    if(hdbuf[i - 1] == ' ') {
      end = &hdbuf[i - 1];
      break;
    }
  }
  if(!end || end == hdbuf)
    goto fail;
  nva[1].name = (unsigned char *)":path";
  nva[1].namelen = strlen((char *)nva[1].name);
  nva[1].value = (unsigned char *)hdbuf;
  nva[1].valuelen = (size_t)(end - hdbuf);
  nva[1].flags = NGHTTP3_NV_FLAG_NONE;

  nva[2].name = (unsigned char *)":scheme";
  nva[2].namelen = strlen((char *)nva[2].name);
  if(conn->handler->flags & PROTOPT_SSL)
    nva[2].value = (unsigned char *)"https";
  else
    nva[2].value = (unsigned char *)"http";
  nva[2].valuelen = strlen((char *)nva[2].value);
  nva[2].flags = NGHTTP3_NV_FLAG_NONE;


  authority_idx = 0;
  i = 3;
  while(i < nheader) {
    size_t hlen;

    hdbuf = line_end + 2;

    /* check for next CR, but only within the piece of data left in the given
       buffer */
    line_end = memchr(hdbuf, '\r', len - (hdbuf - (char *)mem));
    if(!line_end || (line_end == hdbuf))
      goto fail;

    /* header continuation lines are not supported */
    if(*hdbuf == ' ' || *hdbuf == '\t')
      goto fail;

    for(end = hdbuf; end < line_end && *end != ':'; ++end)
      ;
    if(end == hdbuf || end == line_end)
      goto fail;
    hlen = end - hdbuf;

    if(hlen == 4 && strncasecompare("host", hdbuf, 4)) {
      authority_idx = i;
      nva[i].name = (unsigned char *)":authority";
      nva[i].namelen = strlen((char *)nva[i].name);
    }
    else {
      nva[i].namelen = (size_t)(end - hdbuf);
      /* Lower case the header name for HTTP/3 */
      Curl_strntolower((char *)hdbuf, hdbuf, nva[i].namelen);
      nva[i].name = (unsigned char *)hdbuf;
    }
    nva[i].flags = NGHTTP3_NV_FLAG_NONE;
    hdbuf = end + 1;
    while(*hdbuf == ' ' || *hdbuf == '\t')
      ++hdbuf;
    end = line_end;

#if 0 /* This should probably go in more or less like this */
    switch(inspect_header((const char *)nva[i].name, nva[i].namelen, hdbuf,
                          end - hdbuf)) {
    case HEADERINST_IGNORE:
      /* skip header fields prohibited by HTTP/2 specification. */
      --nheader;
      continue;
    case HEADERINST_TE_TRAILERS:
      nva[i].value = (uint8_t*)"trailers";
      nva[i].value_len = sizeof("trailers") - 1;
      break;
    default:
      nva[i].value = (unsigned char *)hdbuf;
      nva[i].value_len = (size_t)(end - hdbuf);
    }
#endif
    nva[i].value = (unsigned char *)hdbuf;
    nva[i].valuelen = (size_t)(end - hdbuf);
    nva[i].flags = NGHTTP3_NV_FLAG_NONE;

    ++i;
  }

  /* :authority must come before non-pseudo header fields */
  if(authority_idx && authority_idx != AUTHORITY_DST_IDX) {
    nghttp3_nv authority = nva[authority_idx];
    for(i = authority_idx; i > AUTHORITY_DST_IDX; --i) {
      nva[i] = nva[i - 1];
    }
    nva[i] = authority;
  }

  /* Warn stream may be rejected if cumulative length of headers is too
     large. */
#define MAX_ACC 60000  /* <64KB to account for some overhead */
  {
    size_t acc = 0;
    for(i = 0; i < nheader; ++i)
      acc += nva[i].namelen + nva[i].valuelen;

    if(acc > MAX_ACC) {
      infof(data, "http_request: Warning: The cumulative length of all "
            "headers exceeds %d bytes and that could cause the "
            "stream to be rejected.", MAX_ACC);
    }
  }

  switch(data->state.httpreq) {
  case HTTPREQ_POST:
  case HTTPREQ_POST_FORM:
  case HTTPREQ_POST_MIME:
  case HTTPREQ_PUT: {
    nghttp3_data_reader data_reader;
    if(data->state.infilesize != -1)
      stream->upload_left = data->state.infilesize;
    else
      /* data sending without specifying the data amount up front */
      stream->upload_left = -1; /* unknown, but not zero */

    data_reader.read_data = cb_h3_readfunction;

    h3out = calloc(sizeof(struct h3out), 1);
    if(!h3out) {
      result = CURLE_OUT_OF_MEMORY;
      goto fail;
    }
    stream->h3out = h3out;

    rc = nghttp3_conn_submit_request(qs->h3conn, stream->stream3_id,
                                     nva, nheader, &data_reader, data);
    if(rc) {
      result = CURLE_SEND_ERROR;
      goto fail;
    }
    break;
  }
  default:
    stream->upload_left = 0; /* nothing left to send */
    rc = nghttp3_conn_submit_request(qs->h3conn, stream->stream3_id,
                                     nva, nheader, NULL, data);
    if(rc) {
      result = CURLE_SEND_ERROR;
      goto fail;
    }
    break;
  }

  Curl_safefree(nva);

  infof(data, "Using HTTP/3 Stream ID: %x (easy handle %p)",
        stream3_id, (void *)data);

  return CURLE_OK;

fail:
  free(nva);
  return result;
}
static ssize_t ngh3_stream_send(struct Curl_easy *data,
                                int sockindex,
                                const void *mem,
                                size_t len,
                                CURLcode *curlcode)
{
  ssize_t sent;
  struct connectdata *conn = data->conn;
  struct quicsocket *qs = conn->quic;
  curl_socket_t sockfd = conn->sock[sockindex];
  struct HTTP *stream = data->req.p.http;

  if(!stream->h3req) {
    CURLcode result = http_request(data, mem, len);
    if(result) {
      *curlcode = CURLE_SEND_ERROR;
      return -1;
    }
    sent = len;
  }
  else {
    H3BUGF(infof(data, "ngh3_stream_send() wants to send %zd bytes",
                 len));
    if(!stream->upload_len) {
      stream->upload_mem = mem;
      stream->upload_len = len;
      (void)nghttp3_conn_resume_stream(qs->h3conn, stream->stream3_id);
      sent = len;
    }
    else {
      *curlcode = CURLE_AGAIN;
      return -1;
    }
  }

  if(ng_flush_egress(data, sockfd, qs)) {
    *curlcode = CURLE_SEND_ERROR;
    return -1;
  }

  /* Reset post upload buffer after resumed. */
  if(stream->upload_mem) {
    stream->upload_mem = NULL;
    stream->upload_len = 0;
  }

  *curlcode = CURLE_OK;
  return sent;
}

static void ng_has_connected(struct connectdata *conn, int tempindex)
{
  conn->recv[FIRSTSOCKET] = ngh3_stream_recv;
  conn->send[FIRSTSOCKET] = ngh3_stream_send;
  conn->handler = &Curl_handler_http3;
  conn->bits.multiplex = TRUE; /* at least potentially multiplexed */
  conn->httpversion = 30;
  conn->bundle->multiuse = BUNDLE_MULTIPLEX;
  conn->quic = &conn->hequic[tempindex];
}

/*
 * There can be multiple connection attempts going on in parallel.
 */
CURLcode Curl_quic_is_connected(struct Curl_easy *data,
                                struct connectdata *conn,
                                int sockindex,
                                bool *done)
{
  CURLcode result;
  struct quicsocket *qs = &conn->hequic[sockindex];
  curl_socket_t sockfd = conn->tempsock[sockindex];

  result = ng_process_ingress(data, sockfd, qs);
  if(result)
    goto error;

  result = ng_flush_egress(data, sockfd, qs);
  if(result)
    goto error;

  if(ngtcp2_conn_get_handshake_completed(qs->qconn)) {
    *done = TRUE;
    ng_has_connected(conn, sockindex);
  }

  return result;
  error:
  (void)qs_disconnect(qs);
  return result;

}

static CURLcode ng_process_ingress(struct Curl_easy *data,
                                   curl_socket_t sockfd,
                                   struct quicsocket *qs)
{
  ssize_t recvd;
  int rv;
  uint8_t buf[65536];
  size_t bufsize = sizeof(buf);
  struct sockaddr_storage remote_addr;
  socklen_t remote_addrlen;
  ngtcp2_path path;
  ngtcp2_tstamp ts = timestamp();
  ngtcp2_pkt_info pi = { 0 };

  for(;;) {
    remote_addrlen = sizeof(remote_addr);
    while((recvd = recvfrom(sockfd, (char *)buf, bufsize, 0,
                            (struct sockaddr *)&remote_addr,
                            &remote_addrlen)) == -1 &&
          SOCKERRNO == EINTR)
      ;
    if(recvd == -1) {
      if(SOCKERRNO == EAGAIN || SOCKERRNO == EWOULDBLOCK)
        break;

      failf(data, "ngtcp2: recvfrom() unexpectedly returned %zd", recvd);
      return CURLE_RECV_ERROR;
    }

    ngtcp2_addr_init(&path.local, (struct sockaddr *)&qs->local_addr,
                     qs->local_addrlen);
    ngtcp2_addr_init(&path.remote, (struct sockaddr *)&remote_addr,
                     remote_addrlen);

    rv = ngtcp2_conn_read_pkt(qs->qconn, &path, &pi, buf, recvd, ts);
    if(rv) {
      /* TODO Send CONNECTION_CLOSE if possible */
      return CURLE_RECV_ERROR;
    }
  }

  return CURLE_OK;
}

static CURLcode ng_flush_egress(struct Curl_easy *data,
                                int sockfd,
                                struct quicsocket *qs)
{
  int rv;
  ssize_t sent;
  ssize_t outlen;
  uint8_t out[NGTCP2_MAX_UDP_PAYLOAD_SIZE];
  ngtcp2_path_storage ps;
  ngtcp2_tstamp ts = timestamp();
  struct sockaddr_storage remote_addr;
  ngtcp2_tstamp expiry;
  ngtcp2_duration timeout;
  int64_t stream_id;
  ssize_t veccnt;
  int fin;
  nghttp3_vec vec[16];
  ssize_t ndatalen;
  uint32_t flags;

  rv = ngtcp2_conn_handle_expiry(qs->qconn, ts);
  if(rv) {
    failf(data, "ngtcp2_conn_handle_expiry returned error: %s",
          ngtcp2_strerror(rv));
    return CURLE_SEND_ERROR;
  }

  ngtcp2_path_storage_zero(&ps);

  for(;;) {
    veccnt = 0;
    stream_id = -1;
    fin = 0;

    if(qs->h3conn && ngtcp2_conn_get_max_data_left(qs->qconn)) {
      veccnt = nghttp3_conn_writev_stream(qs->h3conn, &stream_id, &fin, vec,
                                          sizeof(vec) / sizeof(vec[0]));
      if(veccnt < 0) {
        failf(data, "nghttp3_conn_writev_stream returned error: %s",
              nghttp3_strerror((int)veccnt));
        return CURLE_SEND_ERROR;
      }
    }

    flags = NGTCP2_WRITE_STREAM_FLAG_MORE |
            (fin ? NGTCP2_WRITE_STREAM_FLAG_FIN : 0);
    outlen = ngtcp2_conn_writev_stream(qs->qconn, &ps.path, NULL, out,
                                       sizeof(out),
                                       &ndatalen, flags, stream_id,
                                       (const ngtcp2_vec *)vec, veccnt, ts);
    if(outlen == 0) {
      break;
    }
    if(outlen < 0) {
      switch(outlen) {
      case NGTCP2_ERR_STREAM_DATA_BLOCKED:
        assert(ndatalen == -1);
        rv = nghttp3_conn_block_stream(qs->h3conn, stream_id);
        if(rv) {
          failf(data, "nghttp3_conn_block_stream returned error: %s\n",
                nghttp3_strerror(rv));
          return CURLE_SEND_ERROR;
        }
        continue;
      case NGTCP2_ERR_STREAM_SHUT_WR:
        assert(ndatalen == -1);
        rv = nghttp3_conn_shutdown_stream_write(qs->h3conn, stream_id);
        if(rv) {
          failf(data,
                "nghttp3_conn_shutdown_stream_write returned error: %s\n",
                nghttp3_strerror(rv));
          return CURLE_SEND_ERROR;
        }
        continue;
      case NGTCP2_ERR_WRITE_MORE:
        assert(ndatalen >= 0);
        rv = nghttp3_conn_add_write_offset(qs->h3conn, stream_id, ndatalen);
        if(rv) {
          failf(data, "nghttp3_conn_add_write_offset returned error: %s\n",
                nghttp3_strerror(rv));
          return CURLE_SEND_ERROR;
        }
        continue;
      default:
        assert(ndatalen == -1);
        failf(data, "ngtcp2_conn_writev_stream returned error: %s",
              ngtcp2_strerror((int)outlen));
        return CURLE_SEND_ERROR;
      }
    }
    else if(ndatalen >= 0) {
      rv = nghttp3_conn_add_write_offset(qs->h3conn, stream_id, ndatalen);
      if(rv) {
        failf(data, "nghttp3_conn_add_write_offset returned error: %s\n",
              nghttp3_strerror(rv));
        return CURLE_SEND_ERROR;
      }
    }

    memcpy(&remote_addr, ps.path.remote.addr, ps.path.remote.addrlen);
    while((sent = send(sockfd, (const char *)out, outlen, 0)) == -1 &&
          SOCKERRNO == EINTR)
      ;

    if(sent == -1) {
      if(SOCKERRNO == EAGAIN || SOCKERRNO == EWOULDBLOCK) {
        /* TODO Cache packet */
        break;
      }
      else {
        failf(data, "send() returned %zd (errno %d)", sent,
              SOCKERRNO);
        return CURLE_SEND_ERROR;
      }
    }
  }

  expiry = ngtcp2_conn_get_expiry(qs->qconn);
  if(expiry != UINT64_MAX) {
    if(expiry <= ts) {
      timeout = NGTCP2_MILLISECONDS;
    }
    else {
      timeout = expiry - ts;
    }
    Curl_expire(data, timeout / NGTCP2_MILLISECONDS, EXPIRE_QUIC);
  }

  return CURLE_OK;
}

/*
 * Called from transfer.c:done_sending when we stop HTTP/3 uploading.
 */
CURLcode Curl_quic_done_sending(struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;
  DEBUGASSERT(conn);
  if(conn->handler == &Curl_handler_http3) {
    /* only for HTTP/3 transfers */
    struct HTTP *stream = data->req.p.http;
    struct quicsocket *qs = conn->quic;
    stream->upload_done = TRUE;
    (void)nghttp3_conn_resume_stream(qs->h3conn, stream->stream3_id);
  }

  return CURLE_OK;
}

/*
 * Called from http.c:Curl_http_done when a request completes.
 */
void Curl_quic_done(struct Curl_easy *data, bool premature)
{
  (void)premature;
  if(data->conn->handler == &Curl_handler_http3) {
    /* only for HTTP/3 transfers */
    struct HTTP *stream = data->req.p.http;
    Curl_dyn_free(&stream->overflow);
  }
}

/*
 * Called from transfer.c:data_pending to know if we should keep looping
 * to receive more data from the connection.
 */
bool Curl_quic_data_pending(const struct Curl_easy *data)
{
  /* We may have received more data than we're able to hold in the receive
     buffer and allocated an overflow buffer. Since it's possible that
     there's no more data coming on the socket, we need to keep reading
     until the overflow buffer is empty. */
  const struct HTTP *stream = data->req.p.http;
  return Curl_dyn_len(&stream->overflow) > 0;
}

#endif
