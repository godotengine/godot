/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2020 - 2021, Jacob Hoffman-Andrews,
 * <github@hoffman-andrews.com>
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

#ifdef USE_RUSTLS

#include "curl_printf.h"

#include <errno.h>
#include <rustls.h>

#include "inet_pton.h"
#include "urldata.h"
#include "sendf.h"
#include "vtls.h"
#include "select.h"
#include "strerror.h"
#include "multiif.h"

struct ssl_backend_data
{
  const struct rustls_client_config *config;
  struct rustls_connection *conn;
  bool data_pending;
};

/* For a given rustls_result error code, return the best-matching CURLcode. */
static CURLcode map_error(rustls_result r)
{
  if(rustls_result_is_cert_error(r)) {
    return CURLE_PEER_FAILED_VERIFICATION;
  }
  switch(r) {
    case RUSTLS_RESULT_OK:
      return CURLE_OK;
    case RUSTLS_RESULT_NULL_PARAMETER:
      return CURLE_BAD_FUNCTION_ARGUMENT;
    default:
      return CURLE_READ_ERROR;
  }
}

static bool
cr_data_pending(const struct connectdata *conn, int sockindex)
{
  const struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  return backend->data_pending;
}

static CURLcode
cr_connect(struct Curl_easy *data UNUSED_PARAM,
                    struct connectdata *conn UNUSED_PARAM,
                    int sockindex UNUSED_PARAM)
{
  infof(data, "rustls_connect: unimplemented");
  return CURLE_SSL_CONNECT_ERROR;
}

static int
read_cb(void *userdata, uint8_t *buf, uintptr_t len, uintptr_t *out_n)
{
  ssize_t n = sread(*(int *)userdata, buf, len);
  if(n < 0) {
    return SOCKERRNO;
  }
  *out_n = n;
  return 0;
}

static int
write_cb(void *userdata, const uint8_t *buf, uintptr_t len, uintptr_t *out_n)
{
  ssize_t n = swrite(*(int *)userdata, buf, len);
  if(n < 0) {
    return SOCKERRNO;
  }
  *out_n = n;
  return 0;
}

/*
 * On each run:
 *  - Read a chunk of bytes from the socket into rustls' TLS input buffer.
 *  - Tell rustls to process any new packets.
 *  - Read out as many plaintext bytes from rustls as possible, until hitting
 *    error, EOF, or EAGAIN/EWOULDBLOCK, or plainbuf/plainlen is filled up.
 *
 * It's okay to call this function with plainbuf == NULL and plainlen == 0.
 * In that case, it will copy bytes from the socket into rustls' TLS input
 * buffer, and process packets, but won't consume bytes from rustls' plaintext
 * output buffer.
 */
static ssize_t
cr_recv(struct Curl_easy *data, int sockindex,
            char *plainbuf, size_t plainlen, CURLcode *err)
{
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *const connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *const backend = connssl->backend;
  struct rustls_connection *const rconn = backend->conn;
  size_t n = 0;
  size_t tls_bytes_read = 0;
  size_t plain_bytes_copied = 0;
  rustls_result rresult = 0;
  char errorbuf[255];
  rustls_io_result io_error;

  io_error = rustls_connection_read_tls(rconn, read_cb,
    &conn->sock[sockindex], &tls_bytes_read);
  if(io_error == EAGAIN || io_error == EWOULDBLOCK) {
    infof(data, "sread: EAGAIN or EWOULDBLOCK");
  }
  else if(io_error) {
    char buffer[STRERROR_LEN];
    failf(data, "reading from socket: %s",
          Curl_strerror(io_error, buffer, sizeof(buffer)));
    *err = CURLE_READ_ERROR;
    return -1;
  }

  infof(data, "cr_recv read %ld bytes from the network", tls_bytes_read);

  rresult = rustls_connection_process_new_packets(rconn);
  if(rresult != RUSTLS_RESULT_OK) {
    rustls_error(rresult, errorbuf, sizeof(errorbuf), &n);
    failf(data, "%.*s", n, errorbuf);
    *err = map_error(rresult);
    return -1;
  }

  backend->data_pending = TRUE;

  while(plain_bytes_copied < plainlen) {
    rresult = rustls_connection_read(rconn,
      (uint8_t *)plainbuf + plain_bytes_copied,
      plainlen - plain_bytes_copied,
      &n);
    if(rresult == RUSTLS_RESULT_PLAINTEXT_EMPTY) {
      infof(data, "cr_recv got PLAINTEXT_EMPTY. will try again later.");
      backend->data_pending = FALSE;
      break;
    }
    else if(rresult != RUSTLS_RESULT_OK) {
      /* n always equals 0 in this case, don't need to check it */
      failf(data, "error in rustls_connection_read: %d", rresult);
      *err = CURLE_READ_ERROR;
      return -1;
    }
    else if(n == 0) {
      /* n == 0 indicates clean EOF, but we may have read some other
         plaintext bytes before we reached this. Break out of the loop
         so we can figure out whether to return success or EOF. */
      break;
    }
    else {
      infof(data, "cr_recv copied out %ld bytes of plaintext", n);
      plain_bytes_copied += n;
    }
  }

  if(plain_bytes_copied) {
    *err = CURLE_OK;
    return plain_bytes_copied;
  }

  /* If we wrote out 0 plaintext bytes, that means either we hit a clean EOF,
     OR we got a RUSTLS_RESULT_PLAINTEXT_EMPTY.
     If the latter, return CURLE_AGAIN so curl doesn't treat this as EOF. */
  if(!backend->data_pending) {
    *err = CURLE_AGAIN;
    return -1;
  }

  /* Zero bytes read, and no RUSTLS_RESULT_PLAINTEXT_EMPTY, means the TCP
     connection was cleanly closed (with a close_notify alert). */
  *err = CURLE_OK;
  return 0;
}

/*
 * On each call:
 *  - Copy `plainlen` bytes into rustls' plaintext input buffer (if > 0).
 *  - Fully drain rustls' plaintext output buffer into the socket until
 *    we get either an error or EAGAIN/EWOULDBLOCK.
 *
 * It's okay to call this function with plainbuf == NULL and plainlen == 0.
 * In that case, it won't read anything into rustls' plaintext input buffer.
 * It will only drain rustls' plaintext output buffer into the socket.
 */
static ssize_t
cr_send(struct Curl_easy *data, int sockindex,
        const void *plainbuf, size_t plainlen, CURLcode *err)
{
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *const connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *const backend = connssl->backend;
  struct rustls_connection *const rconn = backend->conn;
  size_t plainwritten = 0;
  size_t tlswritten = 0;
  size_t tlswritten_total = 0;
  rustls_result rresult;
  rustls_io_result io_error;

  infof(data, "cr_send %ld bytes of plaintext", plainlen);

  if(plainlen > 0) {
    rresult = rustls_connection_write(rconn, plainbuf, plainlen,
                                      &plainwritten);
    if(rresult != RUSTLS_RESULT_OK) {
      failf(data, "error in rustls_connection_write");
      *err = CURLE_WRITE_ERROR;
      return -1;
    }
    else if(plainwritten == 0) {
      failf(data, "EOF in rustls_connection_write");
      *err = CURLE_WRITE_ERROR;
      return -1;
    }
  }

  while(rustls_connection_wants_write(rconn)) {
    io_error = rustls_connection_write_tls(rconn, write_cb,
      &conn->sock[sockindex], &tlswritten);
    if(io_error == EAGAIN || io_error == EWOULDBLOCK) {
      infof(data, "swrite: EAGAIN after %ld bytes", tlswritten_total);
      *err = CURLE_AGAIN;
      return -1;
    }
    else if(io_error) {
      char buffer[STRERROR_LEN];
      failf(data, "writing to socket: %s",
            Curl_strerror(io_error, buffer, sizeof(buffer)));
      *err = CURLE_WRITE_ERROR;
      return -1;
    }
    if(tlswritten == 0) {
      failf(data, "EOF in swrite");
      *err = CURLE_WRITE_ERROR;
      return -1;
    }
    infof(data, "cr_send wrote %ld bytes to network", tlswritten);
    tlswritten_total += tlswritten;
  }

  return plainwritten;
}

/* A server certificate verify callback for rustls that always returns
   RUSTLS_RESULT_OK, or in other words disable certificate verification. */
static enum rustls_result
cr_verify_none(void *userdata UNUSED_PARAM,
               const rustls_verify_server_cert_params *params UNUSED_PARAM)
{
  return RUSTLS_RESULT_OK;
}

static bool
cr_hostname_is_ip(const char *hostname)
{
  struct in_addr in;
#ifdef ENABLE_IPV6
  struct in6_addr in6;
  if(Curl_inet_pton(AF_INET6, hostname, &in6) > 0) {
    return true;
  }
#endif /* ENABLE_IPV6 */
  if(Curl_inet_pton(AF_INET, hostname, &in) > 0) {
    return true;
  }
  return false;
}

static CURLcode
cr_init_backend(struct Curl_easy *data, struct connectdata *conn,
                struct ssl_backend_data *const backend)
{
  struct rustls_connection *rconn = backend->conn;
  struct rustls_client_config_builder *config_builder = NULL;
  const char *const ssl_cafile = SSL_CONN_CONFIG(CAfile);
  const bool verifypeer = SSL_CONN_CONFIG(verifypeer);
  const char *hostname = conn->host.name;
  char errorbuf[256];
  size_t errorlen;
  int result;
  rustls_slice_bytes alpn[2] = {
    { (const uint8_t *)ALPN_HTTP_1_1, ALPN_HTTP_1_1_LENGTH },
    { (const uint8_t *)ALPN_H2, ALPN_H2_LENGTH },
  };

  config_builder = rustls_client_config_builder_new();
#ifdef USE_HTTP2
  infof(data, "offering ALPN for HTTP/1.1 and HTTP/2");
  rustls_client_config_builder_set_alpn_protocols(config_builder, alpn, 2);
#else
  infof(data, "offering ALPN for HTTP/1.1 only");
  rustls_client_config_builder_set_alpn_protocols(config_builder, alpn, 1);
#endif
  if(!verifypeer) {
    rustls_client_config_builder_dangerous_set_certificate_verifier(
      config_builder, cr_verify_none);
    /* rustls doesn't support IP addresses (as of 0.19.0), and will reject
     * connections created with an IP address, even when certificate
     * verification is turned off. Set a placeholder hostname and disable
     * SNI. */
    if(cr_hostname_is_ip(hostname)) {
      rustls_client_config_builder_set_enable_sni(config_builder, false);
      hostname = "example.invalid";
    }
  }
  else if(ssl_cafile) {
    result = rustls_client_config_builder_load_roots_from_file(
      config_builder, ssl_cafile);
    if(result != RUSTLS_RESULT_OK) {
      failf(data, "failed to load trusted certificates");
      rustls_client_config_free(
        rustls_client_config_builder_build(config_builder));
      return CURLE_SSL_CACERT_BADFILE;
    }
  }

  backend->config = rustls_client_config_builder_build(config_builder);
  DEBUGASSERT(rconn == NULL);
  result = rustls_client_connection_new(backend->config, hostname, &rconn);
  if(result != RUSTLS_RESULT_OK) {
    rustls_error(result, errorbuf, sizeof(errorbuf), &errorlen);
    failf(data, "rustls_client_connection_new: %.*s", errorlen, errorbuf);
    return CURLE_COULDNT_CONNECT;
  }
  rustls_connection_set_userdata(rconn, backend);
  backend->conn = rconn;
  return CURLE_OK;
}

static void
cr_set_negotiated_alpn(struct Curl_easy *data, struct connectdata *conn,
  const struct rustls_connection *rconn)
{
  const uint8_t *protocol = NULL;
  size_t len = 0;

  rustls_connection_get_alpn_protocol(rconn, &protocol, &len);
  if(NULL == protocol) {
    infof(data, "ALPN, server did not agree to a protocol");
    return;
  }

#ifdef USE_HTTP2
  if(len == ALPN_H2_LENGTH && 0 == memcmp(ALPN_H2, protocol, len)) {
    infof(data, "ALPN, negotiated h2");
    conn->negnpn = CURL_HTTP_VERSION_2;
  }
  else
#endif
  if(len == ALPN_HTTP_1_1_LENGTH &&
      0 == memcmp(ALPN_HTTP_1_1, protocol, len)) {
    infof(data, "ALPN, negotiated http/1.1");
    conn->negnpn = CURL_HTTP_VERSION_1_1;
  }
  else {
    infof(data, "ALPN, negotiated an unrecognized protocol");
  }

  Curl_multiuse_state(data, conn->negnpn == CURL_HTTP_VERSION_2 ?
                      BUNDLE_MULTIPLEX : BUNDLE_NO_MULTIUSE);
}

static CURLcode
cr_connect_nonblocking(struct Curl_easy *data, struct connectdata *conn,
                       int sockindex, bool *done)
{
  struct ssl_connect_data *const connssl = &conn->ssl[sockindex];
  curl_socket_t sockfd = conn->sock[sockindex];
  struct ssl_backend_data *const backend = connssl->backend;
  struct rustls_connection *rconn = NULL;
  CURLcode tmperr = CURLE_OK;
  int result;
  int what;
  bool wants_read;
  bool wants_write;
  curl_socket_t writefd;
  curl_socket_t readfd;

  if(ssl_connection_none == connssl->state) {
    result = cr_init_backend(data, conn, connssl->backend);
    if(result != CURLE_OK) {
      return result;
    }
    connssl->state = ssl_connection_negotiating;
  }

  rconn = backend->conn;

  /* Read/write data until the handshake is done or the socket would block. */
  for(;;) {
    /*
    * Connection has been established according to rustls. Set send/recv
    * handlers, and update the state machine.
    */
    if(!rustls_connection_is_handshaking(rconn)) {
      infof(data, "Done handshaking");
      /* Done with the handshake. Set up callbacks to send/receive data. */
      connssl->state = ssl_connection_complete;

      cr_set_negotiated_alpn(data, conn, rconn);

      conn->recv[sockindex] = cr_recv;
      conn->send[sockindex] = cr_send;
      *done = TRUE;
      return CURLE_OK;
    }

    wants_read = rustls_connection_wants_read(rconn);
    wants_write = rustls_connection_wants_write(rconn);
    DEBUGASSERT(wants_read || wants_write);
    writefd = wants_write?sockfd:CURL_SOCKET_BAD;
    readfd = wants_read?sockfd:CURL_SOCKET_BAD;

    what = Curl_socket_check(readfd, CURL_SOCKET_BAD, writefd, 0);
    if(what < 0) {
      /* fatal error */
      failf(data, "select/poll on SSL socket, errno: %d", SOCKERRNO);
      return CURLE_SSL_CONNECT_ERROR;
    }
    if(0 == what) {
      infof(data, "Curl_socket_check: %s would block",
            wants_read&&wants_write ? "writing and reading" :
            wants_write ? "writing" : "reading");
      *done = FALSE;
      return CURLE_OK;
    }
    /* socket is readable or writable */

    if(wants_write) {
      infof(data, "rustls_connection wants us to write_tls.");
      cr_send(data, sockindex, NULL, 0, &tmperr);
      if(tmperr == CURLE_AGAIN) {
        infof(data, "writing would block");
        /* fall through */
      }
      else if(tmperr != CURLE_OK) {
        return tmperr;
      }
    }

    if(wants_read) {
      infof(data, "rustls_connection wants us to read_tls.");

      cr_recv(data, sockindex, NULL, 0, &tmperr);
      if(tmperr == CURLE_AGAIN) {
        infof(data, "reading would block");
        /* fall through */
      }
      else if(tmperr != CURLE_OK) {
        if(tmperr == CURLE_READ_ERROR) {
          return CURLE_SSL_CONNECT_ERROR;
        }
        else {
          return tmperr;
        }
      }
    }
  }

  /* We should never fall through the loop. We should return either because
     the handshake is done or because we can't read/write without blocking. */
  DEBUGASSERT(false);
}

/* returns a bitmap of flags for this connection's first socket indicating
   whether we want to read or write */
static int
cr_getsock(struct connectdata *conn, curl_socket_t *socks)
{
  struct ssl_connect_data *const connssl = &conn->ssl[FIRSTSOCKET];
  curl_socket_t sockfd = conn->sock[FIRSTSOCKET];
  struct ssl_backend_data *const backend = connssl->backend;
  struct rustls_connection *rconn = backend->conn;

  if(rustls_connection_wants_write(rconn)) {
    socks[0] = sockfd;
    return GETSOCK_WRITESOCK(0);
  }
  if(rustls_connection_wants_read(rconn)) {
    socks[0] = sockfd;
    return GETSOCK_READSOCK(0);
  }

  return GETSOCK_BLANK;
}

static void *
cr_get_internals(struct ssl_connect_data *connssl,
                 CURLINFO info UNUSED_PARAM)
{
  struct ssl_backend_data *backend = connssl->backend;
  return &backend->conn;
}

static void
cr_close(struct Curl_easy *data, struct connectdata *conn,
         int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  CURLcode tmperr = CURLE_OK;
  ssize_t n = 0;

  if(backend->conn) {
    rustls_connection_send_close_notify(backend->conn);
    n = cr_send(data, sockindex, NULL, 0, &tmperr);
    if(n < 0) {
      failf(data, "error sending close notify: %d", tmperr);
    }

    rustls_connection_free(backend->conn);
    backend->conn = NULL;
  }
  if(backend->config) {
    rustls_client_config_free(backend->config);
    backend->config = NULL;
  }
}

static size_t cr_version(char *buffer, size_t size)
{
  struct rustls_str ver = rustls_version();
  return msnprintf(buffer, size, "%.*s", (int)ver.len, ver.data);
}

const struct Curl_ssl Curl_ssl_rustls = {
  { CURLSSLBACKEND_RUSTLS, "rustls" },
  SSLSUPP_TLS13_CIPHERSUITES,      /* supports */
  sizeof(struct ssl_backend_data),

  Curl_none_init,                  /* init */
  Curl_none_cleanup,               /* cleanup */
  cr_version,                      /* version */
  Curl_none_check_cxn,             /* check_cxn */
  Curl_none_shutdown,              /* shutdown */
  cr_data_pending,                 /* data_pending */
  Curl_none_random,                /* random */
  Curl_none_cert_status_request,   /* cert_status_request */
  cr_connect,                      /* connect */
  cr_connect_nonblocking,          /* connect_nonblocking */
  cr_getsock,                      /* cr_getsock */
  cr_get_internals,                /* get_internals */
  cr_close,                        /* close_one */
  Curl_none_close_all,             /* close_all */
  Curl_none_session_free,          /* session_free */
  Curl_none_set_engine,            /* set_engine */
  Curl_none_set_engine_default,    /* set_engine_default */
  Curl_none_engines_list,          /* engines_list */
  Curl_none_false_start,           /* false_start */
  NULL,                            /* sha256sum */
  NULL,                            /* associate_connection */
  NULL                             /* disassociate_connection */
};

#endif /* USE_RUSTLS */
