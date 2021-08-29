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

#ifdef USE_QUICHE
#include <quiche.h>
#include <openssl/err.h>
#include "urldata.h"
#include "sendf.h"
#include "strdup.h"
#include "rand.h"
#include "quic.h"
#include "strcase.h"
#include "multiif.h"
#include "connect.h"
#include "strerror.h"
#include "vquic.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#define DEBUG_HTTP3
/* #define DEBUG_QUICHE */
#ifdef DEBUG_HTTP3
#define H3BUGF(x) x
#else
#define H3BUGF(x) do { } while(0)
#endif

#define QUIC_MAX_STREAMS (256*1024)
#define QUIC_MAX_DATA (1*1024*1024)
#define QUIC_IDLE_TIMEOUT (60 * 1000) /* milliseconds */

static CURLcode process_ingress(struct Curl_easy *data,
                                curl_socket_t sockfd,
                                struct quicsocket *qs);

static CURLcode flush_egress(struct Curl_easy *data, curl_socket_t sockfd,
                             struct quicsocket *qs);

static CURLcode http_request(struct Curl_easy *data, const void *mem,
                             size_t len);
static Curl_recv h3_stream_recv;
static Curl_send h3_stream_send;

static int quiche_getsock(struct Curl_easy *data,
                          struct connectdata *conn, curl_socket_t *socks)
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

static CURLcode qs_disconnect(struct Curl_easy *data,
                              struct quicsocket *qs)
{
  DEBUGASSERT(qs);
  if(qs->conn) {
    (void)quiche_conn_close(qs->conn, TRUE, 0, NULL, 0);
    /* flushing the egress is not a failsafe way to deliver all the
       outstanding packets, but we also don't want to get stuck here... */
    (void)flush_egress(data, qs->sockfd, qs);
    quiche_conn_free(qs->conn);
    qs->conn = NULL;
  }
  if(qs->h3config)
    quiche_h3_config_free(qs->h3config);
  if(qs->h3c)
    quiche_h3_conn_free(qs->h3c);
  if(qs->cfg) {
    quiche_config_free(qs->cfg);
    qs->cfg = NULL;
  }
  return CURLE_OK;
}

static CURLcode quiche_disconnect(struct Curl_easy *data,
                                  struct connectdata *conn,
                                  bool dead_connection)
{
  struct quicsocket *qs = conn->quic;
  (void)dead_connection;
  return qs_disconnect(data, qs);
}

void Curl_quic_disconnect(struct Curl_easy *data,
                          struct connectdata *conn,
                          int tempindex)
{
  if(conn->transport == TRNSPRT_QUIC)
    qs_disconnect(data, &conn->hequic[tempindex]);
}

static unsigned int quiche_conncheck(struct Curl_easy *data,
                                     struct connectdata *conn,
                                     unsigned int checks_to_perform)
{
  (void)data;
  (void)conn;
  (void)checks_to_perform;
  return CONNRESULT_NONE;
}

static CURLcode quiche_do(struct Curl_easy *data, bool *done)
{
  struct HTTP *stream = data->req.p.http;
  stream->h3req = FALSE; /* not sent */
  return Curl_http(data, done);
}

static const struct Curl_handler Curl_handler_http3 = {
  "HTTPS",                              /* scheme */
  ZERO_NULL,                            /* setup_connection */
  quiche_do,                            /* do_it */
  Curl_http_done,                       /* done */
  ZERO_NULL,                            /* do_more */
  ZERO_NULL,                            /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  quiche_getsock,                       /* proto_getsock */
  quiche_getsock,                       /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  quiche_getsock,                       /* perform_getsock */
  quiche_disconnect,                    /* disconnect */
  ZERO_NULL,                            /* readwrite */
  quiche_conncheck,                     /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_HTTP,                            /* defport */
  CURLPROTO_HTTPS,                      /* protocol */
  CURLPROTO_HTTP,                       /* family */
  PROTOPT_SSL | PROTOPT_STREAM          /* flags */
};

#ifdef DEBUG_QUICHE
static void quiche_debug_log(const char *line, void *argp)
{
  (void)argp;
  fprintf(stderr, "%s\n", line);
}
#endif

CURLcode Curl_quic_connect(struct Curl_easy *data,
                           struct connectdata *conn, curl_socket_t sockfd,
                           int sockindex,
                           const struct sockaddr *addr, socklen_t addrlen)
{
  CURLcode result;
  struct quicsocket *qs = &conn->hequic[sockindex];
  char *keylog_file = NULL;
  char ipbuf[40];
  int port;

#ifdef DEBUG_QUICHE
  /* initialize debug log callback only once */
  static int debug_log_init = 0;
  if(!debug_log_init) {
    quiche_enable_debug_logging(quiche_debug_log, NULL);
    debug_log_init = 1;
  }
#endif

  (void)addr;
  (void)addrlen;

  qs->sockfd = sockfd;
  qs->cfg = quiche_config_new(QUICHE_PROTOCOL_VERSION);
  if(!qs->cfg) {
    failf(data, "can't create quiche config");
    return CURLE_FAILED_INIT;
  }

  quiche_config_set_max_idle_timeout(qs->cfg, QUIC_IDLE_TIMEOUT);
  quiche_config_set_initial_max_data(qs->cfg, QUIC_MAX_DATA);
  quiche_config_set_initial_max_stream_data_bidi_local(qs->cfg, QUIC_MAX_DATA);
  quiche_config_set_initial_max_stream_data_bidi_remote(qs->cfg,
                                                        QUIC_MAX_DATA);
  quiche_config_set_initial_max_stream_data_uni(qs->cfg, QUIC_MAX_DATA);
  quiche_config_set_initial_max_streams_bidi(qs->cfg, QUIC_MAX_STREAMS);
  quiche_config_set_initial_max_streams_uni(qs->cfg, QUIC_MAX_STREAMS);
  quiche_config_set_application_protos(qs->cfg,
                                       (uint8_t *)
                                       QUICHE_H3_APPLICATION_PROTOCOL,
                                       sizeof(QUICHE_H3_APPLICATION_PROTOCOL)
                                       - 1);

  result = Curl_rand(data, qs->scid, sizeof(qs->scid));
  if(result)
    return result;

  keylog_file = getenv("SSLKEYLOGFILE");

  if(keylog_file)
    quiche_config_log_keys(qs->cfg);

  qs->conn = quiche_connect(conn->host.name, (const uint8_t *) qs->scid,
                            sizeof(qs->scid), addr, addrlen, qs->cfg);
  if(!qs->conn) {
    failf(data, "can't create quiche connection");
    return CURLE_OUT_OF_MEMORY;
  }

  if(keylog_file)
    quiche_conn_set_keylog_path(qs->conn, keylog_file);

  /* Known to not work on Windows */
#if !defined(WIN32) && defined(HAVE_QUICHE_CONN_SET_QLOG_FD)
  {
    int qfd;
    (void)Curl_qlogdir(data, qs->scid, sizeof(qs->scid), &qfd);
    if(qfd != -1)
      quiche_conn_set_qlog_fd(qs->conn, qfd,
                              "qlog title", "curl qlog");
  }
#endif

  result = flush_egress(data, sockfd, qs);
  if(result)
    return result;

  /* extract the used address as a string */
  if(!Curl_addr2string((struct sockaddr*)addr, addrlen, ipbuf, &port)) {
    char buffer[STRERROR_LEN];
    failf(data, "ssrem inet_ntop() failed with errno %d: %s",
          SOCKERRNO, Curl_strerror(SOCKERRNO, buffer, sizeof(buffer)));
    return CURLE_BAD_FUNCTION_ARGUMENT;
  }

  infof(data, "Connect socket %d over QUIC to %s:%ld",
        sockfd, ipbuf, port);

  Curl_persistconninfo(data, conn, NULL, -1);

  /* for connection reuse purposes: */
  conn->ssl[FIRSTSOCKET].state = ssl_connection_complete;

  {
    unsigned char alpn_protocols[] = QUICHE_H3_APPLICATION_PROTOCOL;
    unsigned alpn_len, offset = 0;

    /* Replace each ALPN length prefix by a comma. */
    while(offset < sizeof(alpn_protocols) - 1) {
      alpn_len = alpn_protocols[offset];
      alpn_protocols[offset] = ',';
      offset += 1 + alpn_len;
    }

    infof(data, "Sent QUIC client Initial, ALPN: %s",
          alpn_protocols + 1);
  }

  return CURLE_OK;
}

static CURLcode quiche_has_connected(struct connectdata *conn,
                                     int sockindex,
                                     int tempindex)
{
  CURLcode result;
  struct quicsocket *qs = conn->quic = &conn->hequic[tempindex];

  conn->recv[sockindex] = h3_stream_recv;
  conn->send[sockindex] = h3_stream_send;
  conn->handler = &Curl_handler_http3;
  conn->bits.multiplex = TRUE; /* at least potentially multiplexed */
  conn->httpversion = 30;
  conn->bundle->multiuse = BUNDLE_MULTIPLEX;

  qs->h3config = quiche_h3_config_new();
  if(!qs->h3config)
    return CURLE_OUT_OF_MEMORY;

  /* Create a new HTTP/3 connection on the QUIC connection. */
  qs->h3c = quiche_h3_conn_new_with_transport(qs->conn, qs->h3config);
  if(!qs->h3c) {
    result = CURLE_OUT_OF_MEMORY;
    goto fail;
  }
  if(conn->hequic[1-tempindex].cfg) {
    qs = &conn->hequic[1-tempindex];
    quiche_config_free(qs->cfg);
    quiche_conn_free(qs->conn);
    qs->cfg = NULL;
    qs->conn = NULL;
  }
  return CURLE_OK;
  fail:
  quiche_h3_config_free(qs->h3config);
  quiche_h3_conn_free(qs->h3c);
  return result;
}

/*
 * This function gets polled to check if this QUIC connection has connected.
 */
CURLcode Curl_quic_is_connected(struct Curl_easy *data,
                                struct connectdata *conn,
                                int sockindex,
                                bool *done)
{
  CURLcode result;
  struct quicsocket *qs = &conn->hequic[sockindex];
  curl_socket_t sockfd = conn->tempsock[sockindex];

  result = process_ingress(data, sockfd, qs);
  if(result)
    goto error;

  result = flush_egress(data, sockfd, qs);
  if(result)
    goto error;

  if(quiche_conn_is_established(qs->conn)) {
    *done = TRUE;
    result = quiche_has_connected(conn, 0, sockindex);
    DEBUGF(infof(data, "quiche established connection!"));
  }

  return result;
  error:
  qs_disconnect(data, qs);
  return result;
}

static CURLcode process_ingress(struct Curl_easy *data, int sockfd,
                                struct quicsocket *qs)
{
  ssize_t recvd;
  uint8_t *buf = (uint8_t *)data->state.buffer;
  size_t bufsize = data->set.buffer_size;
  struct sockaddr_storage from;
  socklen_t from_len;
  quiche_recv_info recv_info;

  DEBUGASSERT(qs->conn);

  /* in case the timeout expired */
  quiche_conn_on_timeout(qs->conn);

  do {
    from_len = sizeof(from);

    recvd = recvfrom(sockfd, buf, bufsize, 0,
                     (struct sockaddr *)&from, &from_len);

    if((recvd < 0) && ((SOCKERRNO == EAGAIN) || (SOCKERRNO == EWOULDBLOCK)))
      break;

    if(recvd < 0) {
      failf(data, "quiche: recvfrom() unexpectedly returned %zd "
            "(errno: %d, socket %d)", recvd, SOCKERRNO, sockfd);
      return CURLE_RECV_ERROR;
    }

    recv_info.from = (struct sockaddr *) &from;
    recv_info.from_len = from_len;

    recvd = quiche_conn_recv(qs->conn, buf, recvd, &recv_info);
    if(recvd == QUICHE_ERR_DONE)
      break;

    if(recvd < 0) {
      failf(data, "quiche_conn_recv() == %zd", recvd);
      return CURLE_RECV_ERROR;
    }
  } while(1);

  return CURLE_OK;
}

/*
 * flush_egress drains the buffers and sends off data.
 * Calls failf() on errors.
 */
static CURLcode flush_egress(struct Curl_easy *data, int sockfd,
                             struct quicsocket *qs)
{
  ssize_t sent;
  uint8_t out[1200];
  int64_t timeout_ns;
  quiche_send_info send_info;

  do {
    sent = quiche_conn_send(qs->conn, out, sizeof(out), &send_info);
    if(sent == QUICHE_ERR_DONE)
      break;

    if(sent < 0) {
      failf(data, "quiche_conn_send returned %zd", sent);
      return CURLE_SEND_ERROR;
    }

    sent = send(sockfd, out, sent, 0);
    if(sent < 0) {
      failf(data, "send() returned %zd", sent);
      return CURLE_SEND_ERROR;
    }
  } while(1);

  /* time until the next timeout event, as nanoseconds. */
  timeout_ns = quiche_conn_timeout_as_nanos(qs->conn);
  if(timeout_ns)
    /* expire uses milliseconds */
    Curl_expire(data, (timeout_ns + 999999) / 1000000, EXPIRE_QUIC);

  return CURLE_OK;
}

struct h3h1header {
  char *dest;
  size_t destlen; /* left to use */
  size_t nlen; /* used */
};

static int cb_each_header(uint8_t *name, size_t name_len,
                          uint8_t *value, size_t value_len,
                          void *argp)
{
  struct h3h1header *headers = (struct h3h1header *)argp;
  size_t olen = 0;

  if((name_len == 7) && !strncmp(":status", (char *)name, 7)) {
    msnprintf(headers->dest,
              headers->destlen, "HTTP/3 %.*s\n",
              (int) value_len, value);
  }
  else if(!headers->nlen) {
    return CURLE_HTTP3;
  }
  else {
    msnprintf(headers->dest,
              headers->destlen, "%.*s: %.*s\n",
              (int)name_len, name, (int) value_len, value);
  }
  olen = strlen(headers->dest);
  headers->destlen -= olen;
  headers->nlen += olen;
  headers->dest += olen;
  return 0;
}

static ssize_t h3_stream_recv(struct Curl_easy *data,
                              int sockindex,
                              char *buf,
                              size_t buffersize,
                              CURLcode *curlcode)
{
  ssize_t recvd = -1;
  ssize_t rcode;
  struct connectdata *conn = data->conn;
  struct quicsocket *qs = conn->quic;
  curl_socket_t sockfd = conn->sock[sockindex];
  quiche_h3_event *ev;
  int rc;
  struct h3h1header headers;
  struct HTTP *stream = data->req.p.http;
  headers.dest = buf;
  headers.destlen = buffersize;
  headers.nlen = 0;

  if(process_ingress(data, sockfd, qs)) {
    infof(data, "h3_stream_recv returns on ingress");
    *curlcode = CURLE_RECV_ERROR;
    return -1;
  }

  while(recvd < 0) {
    int64_t s = quiche_h3_conn_poll(qs->h3c, qs->conn, &ev);
    if(s < 0)
      /* nothing more to do */
      break;

    if(s != stream->stream3_id) {
      /* another transfer, ignore for now */
      infof(data, "Got h3 for stream %u, expects %u",
            s, stream->stream3_id);
      continue;
    }

    switch(quiche_h3_event_type(ev)) {
    case QUICHE_H3_EVENT_HEADERS:
      rc = quiche_h3_event_for_each_header(ev, cb_each_header, &headers);
      if(rc) {
        *curlcode = rc;
        failf(data, "Error in HTTP/3 response header");
        break;
      }
      recvd = headers.nlen;
      break;
    case QUICHE_H3_EVENT_DATA:
      if(!stream->firstbody) {
        /* add a header-body separator CRLF */
        buf[0] = '\r';
        buf[1] = '\n';
        buf += 2;
        buffersize -= 2;
        stream->firstbody = TRUE;
        recvd = 2; /* two bytes already */
      }
      else
        recvd = 0;
      rcode = quiche_h3_recv_body(qs->h3c, qs->conn, s, (unsigned char *)buf,
                                  buffersize);
      if(rcode <= 0) {
        recvd = -1;
        break;
      }
      recvd += rcode;
      break;

    case QUICHE_H3_EVENT_FINISHED:
      streamclose(conn, "End of stream");
      recvd = 0; /* end of stream */
      break;
    default:
      break;
    }

    quiche_h3_event_free(ev);
  }
  if(flush_egress(data, sockfd, qs)) {
    *curlcode = CURLE_SEND_ERROR;
    return -1;
  }

  *curlcode = (-1 == recvd)? CURLE_AGAIN : CURLE_OK;
  if(recvd >= 0)
    /* Get this called again to drain the event queue */
    Curl_expire(data, 0, EXPIRE_QUIC);

  data->state.drain = (recvd >= 0) ? 1 : 0;
  return recvd;
}

static ssize_t h3_stream_send(struct Curl_easy *data,
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
    H3BUGF(infof(data, "Pass on %zd body bytes to quiche", len));
    sent = quiche_h3_send_body(qs->h3c, qs->conn, stream->stream3_id,
                               (uint8_t *)mem, len, FALSE);
    if(sent < 0) {
      *curlcode = CURLE_SEND_ERROR;
      return -1;
    }
  }

  if(flush_egress(data, sockfd, qs)) {
    *curlcode = CURLE_SEND_ERROR;
    return -1;
  }

  *curlcode = CURLE_OK;
  return sent;
}

/*
 * Store quiche version info in this buffer.
 */
void Curl_quic_ver(char *p, size_t len)
{
  (void)msnprintf(p, len, "quiche/%s", quiche_version());
}

/* Index where :authority header field will appear in request header
   field list. */
#define AUTHORITY_DST_IDX 3

static CURLcode http_request(struct Curl_easy *data, const void *mem,
                             size_t len)
{
  /*
   */
  struct connectdata *conn = data->conn;
  struct HTTP *stream = data->req.p.http;
  size_t nheader;
  size_t i;
  size_t authority_idx;
  char *hdbuf = (char *)mem;
  char *end, *line_end;
  int64_t stream3_id;
  quiche_h3_header *nva = NULL;
  struct quicsocket *qs = conn->quic;
  CURLcode result = CURLE_OK;

  stream->h3req = TRUE; /* senf off! */

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
  nva = malloc(sizeof(quiche_h3_header) * nheader);
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
  nva[0].name_len = strlen((char *)nva[0].name);
  nva[0].value = (unsigned char *)hdbuf;
  nva[0].value_len = (size_t)(end - hdbuf);

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
  nva[1].name_len = strlen((char *)nva[1].name);
  nva[1].value = (unsigned char *)hdbuf;
  nva[1].value_len = (size_t)(end - hdbuf);

  nva[2].name = (unsigned char *)":scheme";
  nva[2].name_len = strlen((char *)nva[2].name);
  if(conn->handler->flags & PROTOPT_SSL)
    nva[2].value = (unsigned char *)"https";
  else
    nva[2].value = (unsigned char *)"http";
  nva[2].value_len = strlen((char *)nva[2].value);


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
      nva[i].name_len = strlen((char *)nva[i].name);
    }
    else {
      nva[i].name_len = (size_t)(end - hdbuf);
      /* Lower case the header name for HTTP/3 */
      Curl_strntolower((char *)hdbuf, hdbuf, nva[i].name_len);
      nva[i].name = (unsigned char *)hdbuf;
    }
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
    nva[i].value_len = (size_t)(end - hdbuf);

    ++i;
  }

  /* :authority must come before non-pseudo header fields */
  if(authority_idx && authority_idx != AUTHORITY_DST_IDX) {
    quiche_h3_header authority = nva[authority_idx];
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

    for(i = 0; i < nheader; ++i) {
      acc += nva[i].name_len + nva[i].value_len;

      H3BUGF(infof(data, "h3 [%.*s: %.*s]",
                   nva[i].name_len, nva[i].name,
                   nva[i].value_len, nva[i].value));
    }

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
  case HTTPREQ_PUT:
    if(data->state.infilesize != -1)
      stream->upload_left = data->state.infilesize;
    else
      /* data sending without specifying the data amount up front */
      stream->upload_left = -1; /* unknown, but not zero */

    stream3_id = quiche_h3_send_request(qs->h3c, qs->conn, nva, nheader,
                                        stream->upload_left ? FALSE: TRUE);
    if((stream3_id >= 0) && data->set.postfields) {
      ssize_t sent = quiche_h3_send_body(qs->h3c, qs->conn, stream3_id,
                                         (uint8_t *)data->set.postfields,
                                         stream->upload_left, TRUE);
      if(sent <= 0) {
        failf(data, "quiche_h3_send_body failed!");
        result = CURLE_SEND_ERROR;
      }
      stream->upload_left = 0; /* nothing left to send */
    }
    break;
  default:
    stream3_id = quiche_h3_send_request(qs->h3c, qs->conn, nva, nheader,
                                        TRUE);
    break;
  }

  Curl_safefree(nva);

  if(stream3_id < 0) {
    H3BUGF(infof(data, "quiche_h3_send_request returned %d",
                 stream3_id));
    result = CURLE_SEND_ERROR;
    goto fail;
  }

  infof(data, "Using HTTP/3 Stream ID: %x (easy handle %p)",
        stream3_id, (void *)data);
  stream->stream3_id = stream3_id;

  return CURLE_OK;

fail:
  free(nva);
  return result;
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
    ssize_t sent;
    struct HTTP *stream = data->req.p.http;
    struct quicsocket *qs = conn->quic;
    stream->upload_done = TRUE;
    sent = quiche_h3_send_body(qs->h3c, qs->conn, stream->stream3_id,
                               NULL, 0, TRUE);
    if(sent < 0)
      return CURLE_SEND_ERROR;
  }

  return CURLE_OK;
}

/*
 * Called from http.c:Curl_http_done when a request completes.
 */
void Curl_quic_done(struct Curl_easy *data, bool premature)
{
  (void)data;
  (void)premature;
}

/*
 * Called from transfer.c:data_pending to know if we should keep looping
 * to receive more data from the connection.
 */
bool Curl_quic_data_pending(const struct Curl_easy *data)
{
  (void)data;
  return FALSE;
}

#endif
