/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010 - 2018 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 *
 *  This is included from core/private.h if LWS_WITH_TLS
 */

#if defined(LWS_WITH_TLS)

#if defined(USE_WOLFSSL)
 #if defined(USE_OLD_CYASSL)
  #if defined(_WIN32)
   #include <IDE/WIN/user_settings.h>
   #include <cyassl/ctaocrypt/settings.h>
  #else
   #include <cyassl/options.h>
  #endif
  #include <cyassl/openssl/ssl.h>
  #include <cyassl/error-ssl.h>
 #else
  #if defined(_WIN32)
   #include <IDE/WIN/user_settings.h>
   #include <wolfssl/wolfcrypt/settings.h>
  #else
   #include <wolfssl/options.h>
  #endif
  #include <wolfssl/openssl/ssl.h>
  #include <wolfssl/error-ssl.h>
  #define OPENSSL_NO_TLSEXT
 #endif /* not USE_OLD_CYASSL */
#else /* WOLFSSL */
 #if defined(LWS_WITH_ESP32)
  #define OPENSSL_NO_TLSEXT
  #undef MBEDTLS_CONFIG_FILE
  #define MBEDTLS_CONFIG_FILE <mbedtls/esp_config.h>
  #include <mbedtls/ssl.h>
  #include <mbedtls/x509_crt.h>
  #include "tls/mbedtls/wrapper/include/openssl/ssl.h" /* wrapper !!!! */
 #else /* not esp32 */
  #if defined(LWS_WITH_MBEDTLS)
   #include <mbedtls/ssl.h>
   #include <mbedtls/x509_crt.h>
   #include <mbedtls/x509_csr.h>
   #include "tls/mbedtls/wrapper/include/openssl/ssl.h" /* wrapper !!!! */
  #else
   #include <openssl/ssl.h>
   #include <openssl/evp.h>
   #include <openssl/err.h>
   #include <openssl/md5.h>
   #include <openssl/sha.h>
   #ifdef LWS_HAVE_OPENSSL_ECDH_H
    #include <openssl/ecdh.h>
   #endif
   #include <openssl/x509v3.h>
  #endif /* not mbedtls */
  #if defined(OPENSSL_VERSION_NUMBER)
   #if (OPENSSL_VERSION_NUMBER < 0x0009080afL)
/* later openssl defines this to negate the presence of tlsext... but it was only
 * introduced at 0.9.8j.  Earlier versions don't know it exists so don't
 * define it... making it look like the feature exists...
 */
    #define OPENSSL_NO_TLSEXT
   #endif
  #endif
 #endif /* not ESP32 */
#endif /* not USE_WOLFSSL */

#endif /* LWS_WITH_TLS */

enum lws_tls_extant {
	LWS_TLS_EXTANT_NO,
	LWS_TLS_EXTANT_YES,
	LWS_TLS_EXTANT_ALTERNATIVE
};

struct lws_context_per_thread;

struct lws_tls_ops {
	int (*fake_POLLIN_for_buffered)(struct lws_context_per_thread *pt);
	int (*periodic_housekeeping)(struct lws_context *context, time_t now);
};

#if defined(LWS_WITH_TLS)

typedef SSL lws_tls_conn;
typedef SSL_CTX lws_tls_ctx;
typedef BIO lws_tls_bio;
typedef X509 lws_tls_x509;


#define LWS_SSL_ENABLED(context) (context->tls.use_ssl)

extern const struct lws_tls_ops tls_ops_openssl, tls_ops_mbedtls;

struct lws_context_tls {
	char alpn_discovered[32];
	const char *alpn_default;
	time_t last_cert_check_s;
};

struct lws_pt_tls {
	struct lws *pending_read_list; /* linked list */
};

struct lws_tls_ss_pieces;

struct alpn_ctx {
	uint8_t data[23];
	uint8_t len;
};

struct lws_vhost_tls {
	lws_tls_ctx *ssl_ctx;
	lws_tls_ctx *ssl_client_ctx;
	const char *alpn;
	struct lws_tls_ss_pieces *ss; /* for acme tls certs */
	char *alloc_cert_path;
	char *key_path;
#if defined(LWS_WITH_MBEDTLS)
	lws_tls_x509 *x509_client_CA;
#endif
	char ecdh_curve[16];
	struct alpn_ctx alpn_ctx;

	int use_ssl;
	int allow_non_ssl_on_ssl_port;
	int ssl_info_event_mask;

	unsigned int user_supplied_ssl_ctx:1;
	unsigned int skipped_certs:1;
};

struct lws_lws_tls {
	lws_tls_conn *ssl;
	lws_tls_bio *client_bio;
	struct lws *pending_read_list_prev, *pending_read_list_next;
	unsigned int use_ssl;
	unsigned int redirect_to_https:1;
};

LWS_EXTERN void
lws_context_init_alpn(struct lws_vhost *vhost);
LWS_EXTERN enum lws_tls_extant
lws_tls_use_any_upgrade_check_extant(const char *name);
LWS_EXTERN int openssl_websocket_private_data_index;
LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ssl_capable_read(struct lws *wsi, unsigned char *buf, int len);
LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ssl_capable_write(struct lws *wsi, unsigned char *buf, int len);
LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ssl_pending(struct lws *wsi);
LWS_EXTERN int
lws_context_init_ssl_library(const struct lws_context_creation_info *info);
LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_server_socket_service_ssl(struct lws *new_wsi, lws_sockfd_type accept_fd);
LWS_EXTERN int
lws_ssl_close(struct lws *wsi);
LWS_EXTERN void
lws_ssl_SSL_CTX_destroy(struct lws_vhost *vhost);
LWS_EXTERN void
lws_ssl_context_destroy(struct lws_context *context);
void
__lws_ssl_remove_wsi_from_buffered_list(struct lws *wsi);
LWS_VISIBLE void
lws_ssl_remove_wsi_from_buffered_list(struct lws *wsi);
LWS_EXTERN int
lws_ssl_client_bio_create(struct lws *wsi);
LWS_EXTERN int
lws_ssl_client_connect1(struct lws *wsi);
LWS_EXTERN int
lws_ssl_client_connect2(struct lws *wsi, char *errbuf, int len);
LWS_EXTERN void
lws_ssl_elaborate_error(void);
LWS_EXTERN int
lws_tls_fake_POLLIN_for_buffered(struct lws_context_per_thread *pt);
LWS_EXTERN int
lws_gate_accepts(struct lws_context *context, int on);
LWS_EXTERN void
lws_ssl_bind_passphrase(lws_tls_ctx *ssl_ctx,
			const struct lws_context_creation_info *info);
LWS_EXTERN void
lws_ssl_info_callback(const lws_tls_conn *ssl, int where, int ret);
LWS_EXTERN int
lws_tls_openssl_cert_info(X509 *x509, enum lws_tls_cert_info type,
			  union lws_tls_cert_info_results *buf, size_t len);
LWS_EXTERN int
lws_tls_check_all_cert_lifetimes(struct lws_context *context);
LWS_EXTERN int
lws_tls_server_certs_load(struct lws_vhost *vhost, struct lws *wsi,
			  const char *cert, const char *private_key,
			  const char *mem_cert, size_t len_mem_cert,
			  const char *mem_privkey, size_t mem_privkey_len);
LWS_EXTERN enum lws_tls_extant
lws_tls_generic_cert_checks(struct lws_vhost *vhost, const char *cert,
			    const char *private_key);
LWS_EXTERN int
lws_tls_alloc_pem_to_der_file(struct lws_context *context, const char *filename,
			const char *inbuf, lws_filepos_t inlen,
		      uint8_t **buf, lws_filepos_t *amount);

#if !defined(LWS_NO_SERVER)
 LWS_EXTERN int
 lws_context_init_server_ssl(const struct lws_context_creation_info *info,
			     struct lws_vhost *vhost);
 void
 lws_tls_acme_sni_cert_destroy(struct lws_vhost *vhost);
#else
 #define lws_context_init_server_ssl(_a, _b) (0)
 #define lws_tls_acme_sni_cert_destroy(_a)
#endif

LWS_EXTERN void
lws_ssl_destroy(struct lws_vhost *vhost);
LWS_EXTERN char *
lws_ssl_get_error_string(int status, int ret, char *buf, size_t len);

/*
 * lws_tls_ abstract backend implementations
 */

LWS_EXTERN int
lws_tls_server_client_cert_verify_config(struct lws_vhost *vh);
LWS_EXTERN int
lws_tls_server_vhost_backend_init(const struct lws_context_creation_info *info,
				  struct lws_vhost *vhost, struct lws *wsi);
LWS_EXTERN int
lws_tls_server_new_nonblocking(struct lws *wsi, lws_sockfd_type accept_fd);

LWS_EXTERN enum lws_ssl_capable_status
lws_tls_server_accept(struct lws *wsi);

LWS_EXTERN enum lws_ssl_capable_status
lws_tls_server_abort_connection(struct lws *wsi);

LWS_EXTERN enum lws_ssl_capable_status
__lws_tls_shutdown(struct lws *wsi);

LWS_EXTERN enum lws_ssl_capable_status
lws_tls_client_connect(struct lws *wsi);
LWS_EXTERN int
lws_tls_client_confirm_peer_cert(struct lws *wsi, char *ebuf, int ebuf_len);
LWS_EXTERN int
lws_tls_client_create_vhost_context(struct lws_vhost *vh,
				    const struct lws_context_creation_info *info,
				    const char *cipher_list,
				    const char *ca_filepath,
				    const char *cert_filepath,
				    const char *private_key_filepath);

LWS_EXTERN lws_tls_ctx *
lws_tls_ctx_from_wsi(struct lws *wsi);
LWS_EXTERN int
lws_ssl_get_error(struct lws *wsi, int n);

LWS_EXTERN int
lws_context_init_client_ssl(const struct lws_context_creation_info *info,
			    struct lws_vhost *vhost);

LWS_EXTERN void
lws_ssl_info_callback(const lws_tls_conn *ssl, int where, int ret);

int
lws_tls_fake_POLLIN_for_buffered(struct lws_context_per_thread *pt);

#endif