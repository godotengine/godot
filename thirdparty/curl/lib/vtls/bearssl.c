/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2019 - 2021, Michael Forney, <mforney@mforney.org>
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

#ifdef USE_BEARSSL

#include <bearssl.h>

#include "bearssl.h"
#include "urldata.h"
#include "sendf.h"
#include "inet_pton.h"
#include "vtls.h"
#include "connect.h"
#include "select.h"
#include "multiif.h"
#include "curl_printf.h"
#include "curl_memory.h"

struct x509_context {
  const br_x509_class *vtable;
  br_x509_minimal_context minimal;
  bool verifyhost;
  bool verifypeer;
};

struct ssl_backend_data {
  br_ssl_client_context ctx;
  struct x509_context x509;
  unsigned char buf[BR_SSL_BUFSIZE_BIDI];
  br_x509_trust_anchor *anchors;
  size_t anchors_len;
  const char *protocols[2];
  /* SSL client context is active */
  bool active;
  /* size of pending write, yet to be flushed */
  size_t pending_write;
};

struct cafile_parser {
  CURLcode err;
  bool in_cert;
  br_x509_decoder_context xc;
  /* array of trust anchors loaded from CAfile */
  br_x509_trust_anchor *anchors;
  size_t anchors_len;
  /* buffer for DN data */
  unsigned char dn[1024];
  size_t dn_len;
};

#define CAFILE_SOURCE_PATH 1
#define CAFILE_SOURCE_BLOB 2
struct cafile_source {
  const int type;
  const char * const data;
  const size_t len;
};

static void append_dn(void *ctx, const void *buf, size_t len)
{
  struct cafile_parser *ca = ctx;

  if(ca->err != CURLE_OK || !ca->in_cert)
    return;
  if(sizeof(ca->dn) - ca->dn_len < len) {
    ca->err = CURLE_FAILED_INIT;
    return;
  }
  memcpy(ca->dn + ca->dn_len, buf, len);
  ca->dn_len += len;
}

static void x509_push(void *ctx, const void *buf, size_t len)
{
  struct cafile_parser *ca = ctx;

  if(ca->in_cert)
    br_x509_decoder_push(&ca->xc, buf, len);
}

static CURLcode load_cafile(struct cafile_source *source,
                            br_x509_trust_anchor **anchors,
                            size_t *anchors_len)
{
  struct cafile_parser ca;
  br_pem_decoder_context pc;
  br_x509_trust_anchor *ta;
  size_t ta_size;
  br_x509_trust_anchor *new_anchors;
  size_t new_anchors_len;
  br_x509_pkey *pkey;
  FILE *fp = 0;
  unsigned char buf[BUFSIZ];
  const unsigned char *p;
  const char *name;
  size_t n, i, pushed;

  DEBUGASSERT(source->type == CAFILE_SOURCE_PATH
              || source->type == CAFILE_SOURCE_BLOB);

  if(source->type == CAFILE_SOURCE_PATH) {
    fp = fopen(source->data, "rb");
    if(!fp)
      return CURLE_SSL_CACERT_BADFILE;
  }

  if(source->type == CAFILE_SOURCE_BLOB && source->len > (size_t)INT_MAX)
    return CURLE_SSL_CACERT_BADFILE;

  ca.err = CURLE_OK;
  ca.in_cert = FALSE;
  ca.anchors = NULL;
  ca.anchors_len = 0;
  br_pem_decoder_init(&pc);
  br_pem_decoder_setdest(&pc, x509_push, &ca);
  do {
    if(source->type == CAFILE_SOURCE_PATH) {
      n = fread(buf, 1, sizeof(buf), fp);
      if(n == 0)
        break;
      p = buf;
    }
    else if(source->type == CAFILE_SOURCE_BLOB) {
      n = source->len;
      p = (unsigned char *) source->data;
    }
    while(n) {
      pushed = br_pem_decoder_push(&pc, p, n);
      if(ca.err)
        goto fail;
      p += pushed;
      n -= pushed;

      switch(br_pem_decoder_event(&pc)) {
      case 0:
        break;
      case BR_PEM_BEGIN_OBJ:
        name = br_pem_decoder_name(&pc);
        if(strcmp(name, "CERTIFICATE") && strcmp(name, "X509 CERTIFICATE"))
          break;
        br_x509_decoder_init(&ca.xc, append_dn, &ca);
        if(ca.anchors_len == SIZE_MAX / sizeof(ca.anchors[0])) {
          ca.err = CURLE_OUT_OF_MEMORY;
          goto fail;
        }
        new_anchors_len = ca.anchors_len + 1;
        new_anchors = realloc(ca.anchors,
                              new_anchors_len * sizeof(ca.anchors[0]));
        if(!new_anchors) {
          ca.err = CURLE_OUT_OF_MEMORY;
          goto fail;
        }
        ca.anchors = new_anchors;
        ca.anchors_len = new_anchors_len;
        ca.in_cert = TRUE;
        ca.dn_len = 0;
        ta = &ca.anchors[ca.anchors_len - 1];
        ta->dn.data = NULL;
        break;
      case BR_PEM_END_OBJ:
        if(!ca.in_cert)
          break;
        ca.in_cert = FALSE;
        if(br_x509_decoder_last_error(&ca.xc)) {
          ca.err = CURLE_SSL_CACERT_BADFILE;
          goto fail;
        }
        ta->flags = 0;
        if(br_x509_decoder_isCA(&ca.xc))
          ta->flags |= BR_X509_TA_CA;
        pkey = br_x509_decoder_get_pkey(&ca.xc);
        if(!pkey) {
          ca.err = CURLE_SSL_CACERT_BADFILE;
          goto fail;
        }
        ta->pkey = *pkey;

        /* calculate space needed for trust anchor data */
        ta_size = ca.dn_len;
        switch(pkey->key_type) {
        case BR_KEYTYPE_RSA:
          ta_size += pkey->key.rsa.nlen + pkey->key.rsa.elen;
          break;
        case BR_KEYTYPE_EC:
          ta_size += pkey->key.ec.qlen;
          break;
        default:
          ca.err = CURLE_FAILED_INIT;
          goto fail;
        }

        /* fill in trust anchor DN and public key data */
        ta->dn.data = malloc(ta_size);
        if(!ta->dn.data) {
          ca.err = CURLE_OUT_OF_MEMORY;
          goto fail;
        }
        memcpy(ta->dn.data, ca.dn, ca.dn_len);
        ta->dn.len = ca.dn_len;
        switch(pkey->key_type) {
        case BR_KEYTYPE_RSA:
          ta->pkey.key.rsa.n = ta->dn.data + ta->dn.len;
          memcpy(ta->pkey.key.rsa.n, pkey->key.rsa.n, pkey->key.rsa.nlen);
          ta->pkey.key.rsa.e = ta->pkey.key.rsa.n + ta->pkey.key.rsa.nlen;
          memcpy(ta->pkey.key.rsa.e, pkey->key.rsa.e, pkey->key.rsa.elen);
          break;
        case BR_KEYTYPE_EC:
          ta->pkey.key.ec.q = ta->dn.data + ta->dn.len;
          memcpy(ta->pkey.key.ec.q, pkey->key.ec.q, pkey->key.ec.qlen);
          break;
        }
        break;
      default:
        ca.err = CURLE_SSL_CACERT_BADFILE;
        goto fail;
      }
    }
  } while(source->type != CAFILE_SOURCE_BLOB);
  if(fp && ferror(fp))
    ca.err = CURLE_READ_ERROR;

fail:
  if(fp)
    fclose(fp);
  if(ca.err == CURLE_OK) {
    *anchors = ca.anchors;
    *anchors_len = ca.anchors_len;
  }
  else {
    for(i = 0; i < ca.anchors_len; ++i)
      free(ca.anchors[i].dn.data);
    free(ca.anchors);
  }

  return ca.err;
}

static void x509_start_chain(const br_x509_class **ctx,
                             const char *server_name)
{
  struct x509_context *x509 = (struct x509_context *)ctx;

  if(!x509->verifyhost)
    server_name = NULL;
  x509->minimal.vtable->start_chain(&x509->minimal.vtable, server_name);
}

static void x509_start_cert(const br_x509_class **ctx, uint32_t length)
{
  struct x509_context *x509 = (struct x509_context *)ctx;

  x509->minimal.vtable->start_cert(&x509->minimal.vtable, length);
}

static void x509_append(const br_x509_class **ctx, const unsigned char *buf,
                        size_t len)
{
  struct x509_context *x509 = (struct x509_context *)ctx;

  x509->minimal.vtable->append(&x509->minimal.vtable, buf, len);
}

static void x509_end_cert(const br_x509_class **ctx)
{
  struct x509_context *x509 = (struct x509_context *)ctx;

  x509->minimal.vtable->end_cert(&x509->minimal.vtable);
}

static unsigned x509_end_chain(const br_x509_class **ctx)
{
  struct x509_context *x509 = (struct x509_context *)ctx;
  unsigned err;

  err = x509->minimal.vtable->end_chain(&x509->minimal.vtable);
  if(err && !x509->verifypeer) {
    /* ignore any X.509 errors */
    err = BR_ERR_OK;
  }

  return err;
}

static const br_x509_pkey *x509_get_pkey(const br_x509_class *const *ctx,
                                         unsigned *usages)
{
  struct x509_context *x509 = (struct x509_context *)ctx;

  return x509->minimal.vtable->get_pkey(&x509->minimal.vtable, usages);
}

static const br_x509_class x509_vtable = {
  sizeof(struct x509_context),
  x509_start_chain,
  x509_start_cert,
  x509_append,
  x509_end_cert,
  x509_end_chain,
  x509_get_pkey
};

static CURLcode bearssl_connect_step1(struct Curl_easy *data,
                                      struct connectdata *conn, int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  const struct curl_blob *ca_info_blob = SSL_CONN_CONFIG(ca_info_blob);
  const char * const ssl_cafile =
    /* CURLOPT_CAINFO_BLOB overrides CURLOPT_CAINFO */
    (ca_info_blob ? NULL : SSL_CONN_CONFIG(CAfile));
  const char *hostname = SSL_HOST_NAME();
  const bool verifypeer = SSL_CONN_CONFIG(verifypeer);
  const bool verifyhost = SSL_CONN_CONFIG(verifyhost);
  CURLcode ret;
  unsigned version_min, version_max;
#ifdef ENABLE_IPV6
  struct in6_addr addr;
#else
  struct in_addr addr;
#endif

  switch(SSL_CONN_CONFIG(version)) {
  case CURL_SSLVERSION_SSLv2:
    failf(data, "BearSSL does not support SSLv2");
    return CURLE_SSL_CONNECT_ERROR;
  case CURL_SSLVERSION_SSLv3:
    failf(data, "BearSSL does not support SSLv3");
    return CURLE_SSL_CONNECT_ERROR;
  case CURL_SSLVERSION_TLSv1_0:
    version_min = BR_TLS10;
    version_max = BR_TLS10;
    break;
  case CURL_SSLVERSION_TLSv1_1:
    version_min = BR_TLS11;
    version_max = BR_TLS11;
    break;
  case CURL_SSLVERSION_TLSv1_2:
    version_min = BR_TLS12;
    version_max = BR_TLS12;
    break;
  case CURL_SSLVERSION_DEFAULT:
  case CURL_SSLVERSION_TLSv1:
    version_min = BR_TLS10;
    version_max = BR_TLS12;
    break;
  default:
    failf(data, "BearSSL: unknown CURLOPT_SSLVERSION");
    return CURLE_SSL_CONNECT_ERROR;
  }

  if(ca_info_blob) {
    struct cafile_source source = {
      CAFILE_SOURCE_BLOB,
      ca_info_blob->data,
      ca_info_blob->len,
    };
    ret = load_cafile(&source, &backend->anchors, &backend->anchors_len);
    if(ret != CURLE_OK) {
      if(verifypeer) {
        failf(data, "error importing CA certificate blob");
        return ret;
      }
      /* Only warn if no certificate verification is required. */
      infof(data, "error importing CA certificate blob, continuing anyway");
    }
  }

  if(ssl_cafile) {
    struct cafile_source source = {
      CAFILE_SOURCE_PATH,
      ssl_cafile,
      0,
    };
    ret = load_cafile(&source, &backend->anchors, &backend->anchors_len);
    if(ret != CURLE_OK) {
      if(verifypeer) {
        failf(data, "error setting certificate verify locations."
              " CAfile: %s", ssl_cafile);
        return ret;
      }
      infof(data, "error setting certificate verify locations,"
            " continuing anyway:");
    }
  }

  /* initialize SSL context */
  br_ssl_client_init_full(&backend->ctx, &backend->x509.minimal,
                          backend->anchors, backend->anchors_len);
  br_ssl_engine_set_versions(&backend->ctx.eng, version_min, version_max);
  br_ssl_engine_set_buffer(&backend->ctx.eng, backend->buf,
                           sizeof(backend->buf), 1);

  /* initialize X.509 context */
  backend->x509.vtable = &x509_vtable;
  backend->x509.verifypeer = verifypeer;
  backend->x509.verifyhost = verifyhost;
  br_ssl_engine_set_x509(&backend->ctx.eng, &backend->x509.vtable);

  if(SSL_SET_OPTION(primary.sessionid)) {
    void *session;

    Curl_ssl_sessionid_lock(data);
    if(!Curl_ssl_getsessionid(data, conn, SSL_IS_PROXY() ? TRUE : FALSE,
                              &session, NULL, sockindex)) {
      br_ssl_engine_set_session_parameters(&backend->ctx.eng, session);
      infof(data, "BearSSL: re-using session ID");
    }
    Curl_ssl_sessionid_unlock(data);
  }

  if(conn->bits.tls_enable_alpn) {
    int cur = 0;

    /* NOTE: when adding more protocols here, increase the size of the
     * protocols array in `struct ssl_backend_data`.
     */

#ifdef USE_HTTP2
    if(data->state.httpwant >= CURL_HTTP_VERSION_2
#ifndef CURL_DISABLE_PROXY
      && (!SSL_IS_PROXY() || !conn->bits.tunnel_proxy)
#endif
      ) {
      backend->protocols[cur++] = ALPN_H2;
      infof(data, "ALPN, offering %s", ALPN_H2);
    }
#endif

    backend->protocols[cur++] = ALPN_HTTP_1_1;
    infof(data, "ALPN, offering %s", ALPN_HTTP_1_1);

    br_ssl_engine_set_protocol_names(&backend->ctx.eng,
                                     backend->protocols, cur);
  }

  if((1 == Curl_inet_pton(AF_INET, hostname, &addr))
#ifdef ENABLE_IPV6
      || (1 == Curl_inet_pton(AF_INET6, hostname, &addr))
#endif
     ) {
    if(verifyhost) {
      failf(data, "BearSSL: "
            "host verification of IP address is not supported");
      return CURLE_PEER_FAILED_VERIFICATION;
    }
    hostname = NULL;
  }

  if(!br_ssl_client_reset(&backend->ctx, hostname, 0))
    return CURLE_FAILED_INIT;
  backend->active = TRUE;

  connssl->connecting_state = ssl_connect_2;

  return CURLE_OK;
}

static CURLcode bearssl_run_until(struct Curl_easy *data,
                                  struct connectdata *conn, int sockindex,
                                  unsigned target)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  curl_socket_t sockfd = conn->sock[sockindex];
  unsigned state;
  unsigned char *buf;
  size_t len;
  ssize_t ret;
  int err;

  for(;;) {
    state = br_ssl_engine_current_state(&backend->ctx.eng);
    if(state & BR_SSL_CLOSED) {
      err = br_ssl_engine_last_error(&backend->ctx.eng);
      switch(err) {
      case BR_ERR_OK:
        /* TLS close notify */
        if(connssl->state != ssl_connection_complete) {
          failf(data, "SSL: connection closed during handshake");
          return CURLE_SSL_CONNECT_ERROR;
        }
        return CURLE_OK;
      case BR_ERR_X509_EXPIRED:
        failf(data, "SSL: X.509 verification: "
              "certificate is expired or not yet valid");
        return CURLE_PEER_FAILED_VERIFICATION;
      case BR_ERR_X509_BAD_SERVER_NAME:
        failf(data, "SSL: X.509 verification: "
              "expected server name was not found in the chain");
        return CURLE_PEER_FAILED_VERIFICATION;
      case BR_ERR_X509_NOT_TRUSTED:
        failf(data, "SSL: X.509 verification: "
              "chain could not be linked to a trust anchor");
        return CURLE_PEER_FAILED_VERIFICATION;
      }
      /* X.509 errors are documented to have the range 32..63 */
      if(err >= 32 && err < 64)
        return CURLE_PEER_FAILED_VERIFICATION;
      return CURLE_SSL_CONNECT_ERROR;
    }
    if(state & target)
      return CURLE_OK;
    if(state & BR_SSL_SENDREC) {
      buf = br_ssl_engine_sendrec_buf(&backend->ctx.eng, &len);
      ret = swrite(sockfd, buf, len);
      if(ret == -1) {
        if(SOCKERRNO == EAGAIN || SOCKERRNO == EWOULDBLOCK) {
          if(connssl->state != ssl_connection_complete)
            connssl->connecting_state = ssl_connect_2_writing;
          return CURLE_AGAIN;
        }
        return CURLE_WRITE_ERROR;
      }
      br_ssl_engine_sendrec_ack(&backend->ctx.eng, ret);
    }
    else if(state & BR_SSL_RECVREC) {
      buf = br_ssl_engine_recvrec_buf(&backend->ctx.eng, &len);
      ret = sread(sockfd, buf, len);
      if(ret == 0) {
        failf(data, "SSL: EOF without close notify");
        return CURLE_READ_ERROR;
      }
      if(ret == -1) {
        if(SOCKERRNO == EAGAIN || SOCKERRNO == EWOULDBLOCK) {
          if(connssl->state != ssl_connection_complete)
            connssl->connecting_state = ssl_connect_2_reading;
          return CURLE_AGAIN;
        }
        return CURLE_READ_ERROR;
      }
      br_ssl_engine_recvrec_ack(&backend->ctx.eng, ret);
    }
  }
}

static CURLcode bearssl_connect_step2(struct Curl_easy *data,
                                      struct connectdata *conn, int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  CURLcode ret;

  ret = bearssl_run_until(data, conn, sockindex,
                          BR_SSL_SENDAPP | BR_SSL_RECVAPP);
  if(ret == CURLE_AGAIN)
    return CURLE_OK;
  if(ret == CURLE_OK) {
    if(br_ssl_engine_current_state(&backend->ctx.eng) == BR_SSL_CLOSED) {
      failf(data, "SSL: connection closed during handshake");
      return CURLE_SSL_CONNECT_ERROR;
    }
    connssl->connecting_state = ssl_connect_3;
  }
  return ret;
}

static CURLcode bearssl_connect_step3(struct Curl_easy *data,
                                      struct connectdata *conn, int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  CURLcode ret;

  DEBUGASSERT(ssl_connect_3 == connssl->connecting_state);

  if(conn->bits.tls_enable_alpn) {
    const char *protocol;

    protocol = br_ssl_engine_get_selected_protocol(&backend->ctx.eng);
    if(protocol) {
      infof(data, "ALPN, server accepted to use %s", protocol);

#ifdef USE_HTTP2
      if(!strcmp(protocol, ALPN_H2))
        conn->negnpn = CURL_HTTP_VERSION_2;
      else
#endif
      if(!strcmp(protocol, ALPN_HTTP_1_1))
        conn->negnpn = CURL_HTTP_VERSION_1_1;
      else
        infof(data, "ALPN, unrecognized protocol %s", protocol);
      Curl_multiuse_state(data, conn->negnpn == CURL_HTTP_VERSION_2 ?
                          BUNDLE_MULTIPLEX : BUNDLE_NO_MULTIUSE);
    }
    else
      infof(data, "ALPN, server did not agree to a protocol");
  }

  if(SSL_SET_OPTION(primary.sessionid)) {
    bool incache;
    bool added = FALSE;
    void *oldsession;
    br_ssl_session_parameters *session;

    session = malloc(sizeof(*session));
    if(!session)
      return CURLE_OUT_OF_MEMORY;
    br_ssl_engine_get_session_parameters(&backend->ctx.eng, session);
    Curl_ssl_sessionid_lock(data);
    incache = !(Curl_ssl_getsessionid(data, conn,
                                      SSL_IS_PROXY() ? TRUE : FALSE,
                                      &oldsession, NULL, sockindex));
    if(incache)
      Curl_ssl_delsessionid(data, oldsession);
    ret = Curl_ssl_addsessionid(data, conn,
                                SSL_IS_PROXY() ? TRUE : FALSE,
                                session, 0, sockindex, &added);
    Curl_ssl_sessionid_unlock(data);
    if(!added)
      free(session);
    if(ret) {
      return CURLE_OUT_OF_MEMORY;
    }
  }

  connssl->connecting_state = ssl_connect_done;

  return CURLE_OK;
}

static ssize_t bearssl_send(struct Curl_easy *data, int sockindex,
                            const void *buf, size_t len, CURLcode *err)
{
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  unsigned char *app;
  size_t applen;

  for(;;) {
    *err = bearssl_run_until(data, conn, sockindex, BR_SSL_SENDAPP);
    if (*err != CURLE_OK)
      return -1;
    app = br_ssl_engine_sendapp_buf(&backend->ctx.eng, &applen);
    if(!app) {
      failf(data, "SSL: connection closed during write");
      *err = CURLE_SEND_ERROR;
      return -1;
    }
    if(backend->pending_write) {
      applen = backend->pending_write;
      backend->pending_write = 0;
      return applen;
    }
    if(applen > len)
      applen = len;
    memcpy(app, buf, applen);
    br_ssl_engine_sendapp_ack(&backend->ctx.eng, applen);
    br_ssl_engine_flush(&backend->ctx.eng, 0);
    backend->pending_write = applen;
  }
}

static ssize_t bearssl_recv(struct Curl_easy *data, int sockindex,
                            char *buf, size_t len, CURLcode *err)
{
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  unsigned char *app;
  size_t applen;

  *err = bearssl_run_until(data, conn, sockindex, BR_SSL_RECVAPP);
  if(*err != CURLE_OK)
    return -1;
  app = br_ssl_engine_recvapp_buf(&backend->ctx.eng, &applen);
  if(!app)
    return 0;
  if(applen > len)
    applen = len;
  memcpy(buf, app, applen);
  br_ssl_engine_recvapp_ack(&backend->ctx.eng, applen);

  return applen;
}

static CURLcode bearssl_connect_common(struct Curl_easy *data,
                                       struct connectdata *conn,
                                       int sockindex,
                                       bool nonblocking,
                                       bool *done)
{
  CURLcode ret;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  curl_socket_t sockfd = conn->sock[sockindex];
  timediff_t timeout_ms;
  int what;

  /* check if the connection has already been established */
  if(ssl_connection_complete == connssl->state) {
    *done = TRUE;
    return CURLE_OK;
  }

  if(ssl_connect_1 == connssl->connecting_state) {
    ret = bearssl_connect_step1(data, conn, sockindex);
    if(ret)
      return ret;
  }

  while(ssl_connect_2 == connssl->connecting_state ||
        ssl_connect_2_reading == connssl->connecting_state ||
        ssl_connect_2_writing == connssl->connecting_state) {
    /* check allowed time left */
    timeout_ms = Curl_timeleft(data, NULL, TRUE);

    if(timeout_ms < 0) {
      /* no need to continue if time already is up */
      failf(data, "SSL connection timeout");
      return CURLE_OPERATION_TIMEDOUT;
    }

    /* if ssl is expecting something, check if it's available. */
    if(ssl_connect_2_reading == connssl->connecting_state ||
       ssl_connect_2_writing == connssl->connecting_state) {

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
      else if(0 == what) {
        if(nonblocking) {
          *done = FALSE;
          return CURLE_OK;
        }
        else {
          /* timeout */
          failf(data, "SSL connection timeout");
          return CURLE_OPERATION_TIMEDOUT;
        }
      }
      /* socket is readable or writable */
    }

    /* Run transaction, and return to the caller if it failed or if this
     * connection is done nonblocking and this loop would execute again. This
     * permits the owner of a multi handle to abort a connection attempt
     * before step2 has completed while ensuring that a client using select()
     * or epoll() will always have a valid fdset to wait on.
     */
    ret = bearssl_connect_step2(data, conn, sockindex);
    if(ret || (nonblocking &&
               (ssl_connect_2 == connssl->connecting_state ||
                ssl_connect_2_reading == connssl->connecting_state ||
                ssl_connect_2_writing == connssl->connecting_state)))
      return ret;
  }

  if(ssl_connect_3 == connssl->connecting_state) {
    ret = bearssl_connect_step3(data, conn, sockindex);
    if(ret)
      return ret;
  }

  if(ssl_connect_done == connssl->connecting_state) {
    connssl->state = ssl_connection_complete;
    conn->recv[sockindex] = bearssl_recv;
    conn->send[sockindex] = bearssl_send;
    *done = TRUE;
  }
  else
    *done = FALSE;

  /* Reset our connect state machine */
  connssl->connecting_state = ssl_connect_1;

  return CURLE_OK;
}

static size_t bearssl_version(char *buffer, size_t size)
{
  return msnprintf(buffer, size, "BearSSL");
}

static bool bearssl_data_pending(const struct connectdata *conn,
                                 int connindex)
{
  const struct ssl_connect_data *connssl = &conn->ssl[connindex];
  struct ssl_backend_data *backend = connssl->backend;
  return br_ssl_engine_current_state(&backend->ctx.eng) & BR_SSL_RECVAPP;
}

static CURLcode bearssl_random(struct Curl_easy *data UNUSED_PARAM,
                               unsigned char *entropy, size_t length)
{
  static br_hmac_drbg_context ctx;
  static bool seeded = FALSE;

  if(!seeded) {
    br_prng_seeder seeder;

    br_hmac_drbg_init(&ctx, &br_sha256_vtable, NULL, 0);
    seeder = br_prng_seeder_system(NULL);
    if(!seeder || !seeder(&ctx.vtable))
      return CURLE_FAILED_INIT;
    seeded = TRUE;
  }
  br_hmac_drbg_generate(&ctx, entropy, length);

  return CURLE_OK;
}

static CURLcode bearssl_connect(struct Curl_easy *data,
                                struct connectdata *conn, int sockindex)
{
  CURLcode ret;
  bool done = FALSE;

  ret = bearssl_connect_common(data, conn, sockindex, FALSE, &done);
  if(ret)
    return ret;

  DEBUGASSERT(done);

  return CURLE_OK;
}

static CURLcode bearssl_connect_nonblocking(struct Curl_easy *data,
                                            struct connectdata *conn,
                                            int sockindex, bool *done)
{
  return bearssl_connect_common(data, conn, sockindex, TRUE, done);
}

static void *bearssl_get_internals(struct ssl_connect_data *connssl,
                                   CURLINFO info UNUSED_PARAM)
{
  struct ssl_backend_data *backend = connssl->backend;
  return &backend->ctx;
}

static void bearssl_close(struct Curl_easy *data,
                          struct connectdata *conn, int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  size_t i;

  if(backend->active) {
    br_ssl_engine_close(&backend->ctx.eng);
    (void)bearssl_run_until(data, conn, sockindex, BR_SSL_CLOSED);
  }
  for(i = 0; i < backend->anchors_len; ++i)
    free(backend->anchors[i].dn.data);
  free(backend->anchors);
}

static void bearssl_session_free(void *ptr)
{
  free(ptr);
}

static CURLcode bearssl_sha256sum(const unsigned char *input,
                                  size_t inputlen,
                                  unsigned char *sha256sum,
                                  size_t sha256len UNUSED_PARAM)
{
  br_sha256_context ctx;

  br_sha256_init(&ctx);
  br_sha256_update(&ctx, input, inputlen);
  br_sha256_out(&ctx, sha256sum);
  return CURLE_OK;
}

const struct Curl_ssl Curl_ssl_bearssl = {
  { CURLSSLBACKEND_BEARSSL, "bearssl" }, /* info */
  SSLSUPP_CAINFO_BLOB,
  sizeof(struct ssl_backend_data),

  Curl_none_init,                  /* init */
  Curl_none_cleanup,               /* cleanup */
  bearssl_version,                 /* version */
  Curl_none_check_cxn,             /* check_cxn */
  Curl_none_shutdown,              /* shutdown */
  bearssl_data_pending,            /* data_pending */
  bearssl_random,                  /* random */
  Curl_none_cert_status_request,   /* cert_status_request */
  bearssl_connect,                 /* connect */
  bearssl_connect_nonblocking,     /* connect_nonblocking */
  Curl_ssl_getsock,                /* getsock */
  bearssl_get_internals,           /* get_internals */
  bearssl_close,                   /* close_one */
  Curl_none_close_all,             /* close_all */
  bearssl_session_free,            /* session_free */
  Curl_none_set_engine,            /* set_engine */
  Curl_none_set_engine_default,    /* set_engine_default */
  Curl_none_engines_list,          /* engines_list */
  Curl_none_false_start,           /* false_start */
  bearssl_sha256sum,               /* sha256sum */
  NULL,                            /* associate_connection */
  NULL                             /* disassociate_connection */
};

#endif /* USE_BEARSSL */
