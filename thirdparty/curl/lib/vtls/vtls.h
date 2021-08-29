#ifndef HEADER_CURL_VTLS_H
#define HEADER_CURL_VTLS_H
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

struct connectdata;
struct ssl_connect_data;

#define SSLSUPP_CA_PATH      (1<<0) /* supports CAPATH */
#define SSLSUPP_CERTINFO     (1<<1) /* supports CURLOPT_CERTINFO */
#define SSLSUPP_PINNEDPUBKEY (1<<2) /* supports CURLOPT_PINNEDPUBLICKEY */
#define SSLSUPP_SSL_CTX      (1<<3) /* supports CURLOPT_SSL_CTX */
#define SSLSUPP_HTTPS_PROXY  (1<<4) /* supports access via HTTPS proxies */
#define SSLSUPP_TLS13_CIPHERSUITES (1<<5) /* supports TLS 1.3 ciphersuites */
#define SSLSUPP_CAINFO_BLOB  (1<<6)

struct Curl_ssl {
  /*
   * This *must* be the first entry to allow returning the list of available
   * backends in curl_global_sslset().
   */
  curl_ssl_backend info;
  unsigned int supports; /* bitfield, see above */
  size_t sizeof_ssl_backend_data;

  int (*init)(void);
  void (*cleanup)(void);

  size_t (*version)(char *buffer, size_t size);
  int (*check_cxn)(struct connectdata *cxn);
  int (*shut_down)(struct Curl_easy *data, struct connectdata *conn,
                   int sockindex);
  bool (*data_pending)(const struct connectdata *conn,
                       int connindex);

  /* return 0 if a find random is filled in */
  CURLcode (*random)(struct Curl_easy *data, unsigned char *entropy,
                     size_t length);
  bool (*cert_status_request)(void);

  CURLcode (*connect_blocking)(struct Curl_easy *data,
                               struct connectdata *conn, int sockindex);
  CURLcode (*connect_nonblocking)(struct Curl_easy *data,
                                  struct connectdata *conn, int sockindex,
                                  bool *done);

  /* If the SSL backend wants to read or write on this connection during a
     handshake, set socks[0] to the connection's FIRSTSOCKET, and return
     a bitmap indicating read or write with GETSOCK_WRITESOCK(0) or
     GETSOCK_READSOCK(0). Otherwise return GETSOCK_BLANK.
     Mandatory. */
  int (*getsock)(struct connectdata *conn, curl_socket_t *socks);

  void *(*get_internals)(struct ssl_connect_data *connssl, CURLINFO info);
  void (*close_one)(struct Curl_easy *data, struct connectdata *conn,
                    int sockindex);
  void (*close_all)(struct Curl_easy *data);
  void (*session_free)(void *ptr);

  CURLcode (*set_engine)(struct Curl_easy *data, const char *engine);
  CURLcode (*set_engine_default)(struct Curl_easy *data);
  struct curl_slist *(*engines_list)(struct Curl_easy *data);

  bool (*false_start)(void);
  CURLcode (*sha256sum)(const unsigned char *input, size_t inputlen,
                    unsigned char *sha256sum, size_t sha256sumlen);

  void (*associate_connection)(struct Curl_easy *data,
                               struct connectdata *conn,
                               int sockindex);
  void (*disassociate_connection)(struct Curl_easy *data, int sockindex);
};

#ifdef USE_SSL
extern const struct Curl_ssl *Curl_ssl;
#endif

int Curl_none_init(void);
void Curl_none_cleanup(void);
int Curl_none_shutdown(struct Curl_easy *data, struct connectdata *conn,
                       int sockindex);
int Curl_none_check_cxn(struct connectdata *conn);
CURLcode Curl_none_random(struct Curl_easy *data, unsigned char *entropy,
                          size_t length);
void Curl_none_close_all(struct Curl_easy *data);
void Curl_none_session_free(void *ptr);
bool Curl_none_data_pending(const struct connectdata *conn, int connindex);
bool Curl_none_cert_status_request(void);
CURLcode Curl_none_set_engine(struct Curl_easy *data, const char *engine);
CURLcode Curl_none_set_engine_default(struct Curl_easy *data);
struct curl_slist *Curl_none_engines_list(struct Curl_easy *data);
bool Curl_none_false_start(void);
bool Curl_ssl_tls13_ciphersuites(void);

#include "openssl.h"        /* OpenSSL versions */
#include "gtls.h"           /* GnuTLS versions */
#include "nssg.h"           /* NSS versions */
#include "gskit.h"          /* Global Secure ToolKit versions */
#include "wolfssl.h"        /* wolfSSL versions */
#include "schannel.h"       /* Schannel SSPI version */
#include "sectransp.h"      /* SecureTransport (Darwin) version */
#include "mbedtls.h"        /* mbedTLS versions */
#include "mesalink.h"       /* MesaLink versions */
#include "bearssl.h"        /* BearSSL versions */
#include "rustls.h"         /* rustls versions */

#ifndef MAX_PINNED_PUBKEY_SIZE
#define MAX_PINNED_PUBKEY_SIZE 1048576 /* 1MB */
#endif

#ifndef CURL_SHA256_DIGEST_LENGTH
#define CURL_SHA256_DIGEST_LENGTH 32 /* fixed size */
#endif

/* see https://www.iana.org/assignments/tls-extensiontype-values/ */
#define ALPN_HTTP_1_1_LENGTH 8
#define ALPN_HTTP_1_1 "http/1.1"
#define ALPN_H2_LENGTH 2
#define ALPN_H2 "h2"

/* set of helper macros for the backends to access the correct fields. For the
   proxy or for the remote host - to properly support HTTPS proxy */
#ifndef CURL_DISABLE_PROXY
#define SSL_IS_PROXY()                                                  \
  (CURLPROXY_HTTPS == conn->http_proxy.proxytype &&                     \
   ssl_connection_complete !=                                           \
   conn->proxy_ssl[conn->sock[SECONDARYSOCKET] ==                       \
                   CURL_SOCKET_BAD ? FIRSTSOCKET : SECONDARYSOCKET].state)
#define SSL_SET_OPTION(var)                                             \
  (SSL_IS_PROXY() ? data->set.proxy_ssl.var : data->set.ssl.var)
#define SSL_SET_OPTION_LVALUE(var)                                      \
  (*(SSL_IS_PROXY() ? &data->set.proxy_ssl.var : &data->set.ssl.var))
#define SSL_CONN_CONFIG(var)                                            \
  (SSL_IS_PROXY() ? conn->proxy_ssl_config.var : conn->ssl_config.var)
#define SSL_HOST_NAME()                                                 \
  (SSL_IS_PROXY() ? conn->http_proxy.host.name : conn->host.name)
#define SSL_HOST_DISPNAME()                                             \
  (SSL_IS_PROXY() ? conn->http_proxy.host.dispname : conn->host.dispname)
#define SSL_HOST_PORT()                                                 \
  (SSL_IS_PROXY() ? conn->port : conn->remote_port)
#define SSL_PINNED_PUB_KEY() (SSL_IS_PROXY()                            \
  ? data->set.str[STRING_SSL_PINNEDPUBLICKEY_PROXY]                     \
  : data->set.str[STRING_SSL_PINNEDPUBLICKEY])
#else
#define SSL_IS_PROXY() FALSE
#define SSL_SET_OPTION(var) data->set.ssl.var
#define SSL_SET_OPTION_LVALUE(var) data->set.ssl.var
#define SSL_CONN_CONFIG(var) conn->ssl_config.var
#define SSL_HOST_NAME() conn->host.name
#define SSL_HOST_DISPNAME() conn->host.dispname
#define SSL_HOST_PORT() conn->remote_port
#define SSL_PINNED_PUB_KEY()                                            \
  data->set.str[STRING_SSL_PINNEDPUBLICKEY]
#endif

bool Curl_ssl_config_matches(struct ssl_primary_config *data,
                             struct ssl_primary_config *needle);
bool Curl_clone_primary_ssl_config(struct ssl_primary_config *source,
                                   struct ssl_primary_config *dest);
void Curl_free_primary_ssl_config(struct ssl_primary_config *sslc);
/* An implementation of the getsock field of Curl_ssl that relies
   on the ssl_connect_state enum. Asks for read or write depending
   on whether conn->state is ssl_connect_2_reading or
   ssl_connect_2_writing. */
int Curl_ssl_getsock(struct connectdata *conn, curl_socket_t *socks);

int Curl_ssl_backend(void);

#ifdef USE_SSL
int Curl_ssl_init(void);
void Curl_ssl_cleanup(void);
CURLcode Curl_ssl_connect(struct Curl_easy *data, struct connectdata *conn,
                          int sockindex);
CURLcode Curl_ssl_connect_nonblocking(struct Curl_easy *data,
                                      struct connectdata *conn,
                                      bool isproxy,
                                      int sockindex,
                                      bool *done);
/* tell the SSL stuff to close down all open information regarding
   connections (and thus session ID caching etc) */
void Curl_ssl_close_all(struct Curl_easy *data);
void Curl_ssl_close(struct Curl_easy *data, struct connectdata *conn,
                    int sockindex);
CURLcode Curl_ssl_shutdown(struct Curl_easy *data, struct connectdata *conn,
                           int sockindex);
CURLcode Curl_ssl_set_engine(struct Curl_easy *data, const char *engine);
/* Sets engine as default for all SSL operations */
CURLcode Curl_ssl_set_engine_default(struct Curl_easy *data);
struct curl_slist *Curl_ssl_engines_list(struct Curl_easy *data);

/* init the SSL session ID cache */
CURLcode Curl_ssl_initsessions(struct Curl_easy *, size_t);
void Curl_ssl_version(char *buffer, size_t size);
bool Curl_ssl_data_pending(const struct connectdata *conn,
                           int connindex);
int Curl_ssl_check_cxn(struct connectdata *conn);

/* Certificate information list handling. */

void Curl_ssl_free_certinfo(struct Curl_easy *data);
CURLcode Curl_ssl_init_certinfo(struct Curl_easy *data, int num);
CURLcode Curl_ssl_push_certinfo_len(struct Curl_easy *data, int certnum,
                                    const char *label, const char *value,
                                    size_t valuelen);
CURLcode Curl_ssl_push_certinfo(struct Curl_easy *data, int certnum,
                                const char *label, const char *value);

/* Functions to be used by SSL library adaptation functions */

/* Lock session cache mutex.
 * Call this before calling other Curl_ssl_*session* functions
 * Caller should unlock this mutex as soon as possible, as it may block
 * other SSL connection from making progress.
 * The purpose of explicitly locking SSL session cache data is to allow
 * individual SSL engines to manage session lifetime in their specific way.
 */
void Curl_ssl_sessionid_lock(struct Curl_easy *data);

/* Unlock session cache mutex */
void Curl_ssl_sessionid_unlock(struct Curl_easy *data);

/* extract a session ID
 * Sessionid mutex must be locked (see Curl_ssl_sessionid_lock).
 * Caller must make sure that the ownership of returned sessionid object
 * is properly taken (e.g. its refcount is incremented
 * under sessionid mutex).
 */
bool Curl_ssl_getsessionid(struct Curl_easy *data,
                           struct connectdata *conn,
                           const bool isProxy,
                           void **ssl_sessionid,
                           size_t *idsize, /* set 0 if unknown */
                           int sockindex);
/* add a new session ID
 * Sessionid mutex must be locked (see Curl_ssl_sessionid_lock).
 * Caller must ensure that it has properly shared ownership of this sessionid
 * object with cache (e.g. incrementing refcount on success)
 */
CURLcode Curl_ssl_addsessionid(struct Curl_easy *data,
                               struct connectdata *conn,
                               const bool isProxy,
                               void *ssl_sessionid,
                               size_t idsize,
                               int sockindex,
                               bool *added);
/* Kill a single session ID entry in the cache
 * Sessionid mutex must be locked (see Curl_ssl_sessionid_lock).
 * This will call engine-specific curlssl_session_free function, which must
 * take sessionid object ownership from sessionid cache
 * (e.g. decrement refcount).
 */
void Curl_ssl_kill_session(struct Curl_ssl_session *session);
/* delete a session from the cache
 * Sessionid mutex must be locked (see Curl_ssl_sessionid_lock).
 * This will call engine-specific curlssl_session_free function, which must
 * take sessionid object ownership from sessionid cache
 * (e.g. decrement refcount).
 */
void Curl_ssl_delsessionid(struct Curl_easy *data, void *ssl_sessionid);

/* get N random bytes into the buffer */
CURLcode Curl_ssl_random(struct Curl_easy *data, unsigned char *buffer,
                         size_t length);
/* Check pinned public key. */
CURLcode Curl_pin_peer_pubkey(struct Curl_easy *data,
                              const char *pinnedpubkey,
                              const unsigned char *pubkey, size_t pubkeylen);

bool Curl_ssl_cert_status_request(void);

bool Curl_ssl_false_start(void);

void Curl_ssl_associate_conn(struct Curl_easy *data,
                             struct connectdata *conn);
void Curl_ssl_detach_conn(struct Curl_easy *data,
                          struct connectdata *conn);

#define SSL_SHUTDOWN_TIMEOUT 10000 /* ms */

#else /* if not USE_SSL */

/* When SSL support is not present, just define away these function calls */
#define Curl_ssl_init() 1
#define Curl_ssl_cleanup() Curl_nop_stmt
#define Curl_ssl_connect(x,y,z) CURLE_NOT_BUILT_IN
#define Curl_ssl_close_all(x) Curl_nop_stmt
#define Curl_ssl_close(x,y,z) Curl_nop_stmt
#define Curl_ssl_shutdown(x,y,z) CURLE_NOT_BUILT_IN
#define Curl_ssl_set_engine(x,y) CURLE_NOT_BUILT_IN
#define Curl_ssl_set_engine_default(x) CURLE_NOT_BUILT_IN
#define Curl_ssl_engines_list(x) NULL
#define Curl_ssl_send(a,b,c,d,e) -1
#define Curl_ssl_recv(a,b,c,d,e) -1
#define Curl_ssl_initsessions(x,y) CURLE_OK
#define Curl_ssl_data_pending(x,y) 0
#define Curl_ssl_check_cxn(x) 0
#define Curl_ssl_free_certinfo(x) Curl_nop_stmt
#define Curl_ssl_connect_nonblocking(x,y,z,w,a) CURLE_NOT_BUILT_IN
#define Curl_ssl_kill_session(x) Curl_nop_stmt
#define Curl_ssl_random(x,y,z) ((void)x, CURLE_NOT_BUILT_IN)
#define Curl_ssl_cert_status_request() FALSE
#define Curl_ssl_false_start() FALSE
#define Curl_ssl_tls13_ciphersuites() FALSE
#define Curl_ssl_associate_conn(a,b) Curl_nop_stmt
#define Curl_ssl_detach_conn(a,b) Curl_nop_stmt
#endif

#endif /* HEADER_CURL_VTLS_H */
