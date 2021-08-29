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

#ifdef USE_NGHTTP2
#include <nghttp2/nghttp2.h>
#include "urldata.h"
#include "http2.h"
#include "http.h"
#include "sendf.h"
#include "select.h"
#include "curl_base64.h"
#include "strcase.h"
#include "multiif.h"
#include "url.h"
#include "connect.h"
#include "strtoofft.h"
#include "strdup.h"
#include "dynbuf.h"
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#define H2_BUFSIZE 32768

#if (NGHTTP2_VERSION_NUM < 0x010c00)
#error too old nghttp2 version, upgrade!
#endif

#ifdef CURL_DISABLE_VERBOSE_STRINGS
#define nghttp2_session_callbacks_set_error_callback(x,y)
#endif

#if (NGHTTP2_VERSION_NUM >= 0x010c00)
#define NGHTTP2_HAS_SET_LOCAL_WINDOW_SIZE 1
#endif

#define HTTP2_HUGE_WINDOW_SIZE (32 * 1024 * 1024) /* 32 MB */

#ifdef DEBUG_HTTP2
#define H2BUGF(x) x
#else
#define H2BUGF(x) do { } while(0)
#endif


static ssize_t http2_recv(struct Curl_easy *data, int sockindex,
                          char *mem, size_t len, CURLcode *err);
static bool http2_connisdead(struct Curl_easy *data,
                             struct connectdata *conn);
static int h2_session_send(struct Curl_easy *data,
                           nghttp2_session *h2);
static int h2_process_pending_input(struct Curl_easy *data,
                                    struct http_conn *httpc,
                                    CURLcode *err);

/*
 * Curl_http2_init_state() is called when the easy handle is created and
 * allows for HTTP/2 specific init of state.
 */
void Curl_http2_init_state(struct UrlState *state)
{
  state->stream_weight = NGHTTP2_DEFAULT_WEIGHT;
}

/*
 * Curl_http2_init_userset() is called when the easy handle is created and
 * allows for HTTP/2 specific user-set fields.
 */
void Curl_http2_init_userset(struct UserDefined *set)
{
  set->stream_weight = NGHTTP2_DEFAULT_WEIGHT;
}

static int http2_getsock(struct Curl_easy *data,
                         struct connectdata *conn,
                         curl_socket_t *sock)
{
  const struct http_conn *c = &conn->proto.httpc;
  struct SingleRequest *k = &data->req;
  int bitmap = GETSOCK_BLANK;
  struct HTTP *stream = data->req.p.http;

  sock[0] = conn->sock[FIRSTSOCKET];

  if(!(k->keepon & KEEP_RECV_PAUSE))
    /* Unless paused - in a HTTP/2 connection we can basically always get a
       frame so we should always be ready for one */
    bitmap |= GETSOCK_READSOCK(FIRSTSOCKET);

  /* we're (still uploading OR the HTTP/2 layer wants to send data) AND
     there's a window to send data in */
  if((((k->keepon & (KEEP_SEND|KEEP_SEND_PAUSE)) == KEEP_SEND) ||
      nghttp2_session_want_write(c->h2)) &&
     (nghttp2_session_get_remote_window_size(c->h2) &&
      nghttp2_session_get_stream_remote_window_size(c->h2,
                                                    stream->stream_id)))
    bitmap |= GETSOCK_WRITESOCK(FIRSTSOCKET);

  return bitmap;
}

/*
 * http2_stream_free() free HTTP2 stream related data
 */
static void http2_stream_free(struct HTTP *http)
{
  if(http) {
    Curl_dyn_free(&http->header_recvbuf);
    for(; http->push_headers_used > 0; --http->push_headers_used) {
      free(http->push_headers[http->push_headers_used - 1]);
    }
    free(http->push_headers);
    http->push_headers = NULL;
  }
}

/*
 * Disconnects *a* connection used for HTTP/2. It might be an old one from the
 * connection cache and not the "main" one. Don't touch the easy handle!
 */

static CURLcode http2_disconnect(struct Curl_easy *data,
                                 struct connectdata *conn,
                                 bool dead_connection)
{
  struct http_conn *c = &conn->proto.httpc;
  (void)dead_connection;
#ifndef DEBUG_HTTP2
  (void)data;
#endif

  H2BUGF(infof(data, "HTTP/2 DISCONNECT starts now"));

  nghttp2_session_del(c->h2);
  Curl_safefree(c->inbuf);

  H2BUGF(infof(data, "HTTP/2 DISCONNECT done"));

  return CURLE_OK;
}

/*
 * The server may send us data at any point (e.g. PING frames). Therefore,
 * we cannot assume that an HTTP/2 socket is dead just because it is readable.
 *
 * Instead, if it is readable, run Curl_connalive() to peek at the socket
 * and distinguish between closed and data.
 */
static bool http2_connisdead(struct Curl_easy *data, struct connectdata *conn)
{
  int sval;
  bool dead = TRUE;

  if(conn->bits.close)
    return TRUE;

  sval = SOCKET_READABLE(conn->sock[FIRSTSOCKET], 0);
  if(sval == 0) {
    /* timeout */
    dead = FALSE;
  }
  else if(sval & CURL_CSELECT_ERR) {
    /* socket is in an error state */
    dead = TRUE;
  }
  else if(sval & CURL_CSELECT_IN) {
    /* readable with no error. could still be closed */
    dead = !Curl_connalive(conn);
    if(!dead) {
      /* This happens before we've sent off a request and the connection is
         not in use by any other transfer, there shouldn't be any data here,
         only "protocol frames" */
      CURLcode result;
      struct http_conn *httpc = &conn->proto.httpc;
      ssize_t nread = -1;
      if(httpc->recv_underlying)
        /* if called "too early", this pointer isn't setup yet! */
        nread = ((Curl_recv *)httpc->recv_underlying)(
          data, FIRSTSOCKET, httpc->inbuf, H2_BUFSIZE, &result);
      if(nread != -1) {
        infof(data,
              "%d bytes stray data read before trying h2 connection",
              (int)nread);
        httpc->nread_inbuf = 0;
        httpc->inbuflen = nread;
        if(h2_process_pending_input(data, httpc, &result) < 0)
          /* immediate error, considered dead */
          dead = TRUE;
      }
      else
        /* the read failed so let's say this is dead anyway */
        dead = TRUE;
    }
  }

  return dead;
}

/*
 * Set the transfer that is currently using this HTTP/2 connection.
 */
static void set_transfer(struct http_conn *c,
                         struct Curl_easy *data)
{
  c->trnsfr = data;
}

/*
 * Get the transfer that is currently using this HTTP/2 connection.
 */
static struct Curl_easy *get_transfer(struct http_conn *c)
{
  DEBUGASSERT(c && c->trnsfr);
  return c->trnsfr;
}

static unsigned int http2_conncheck(struct Curl_easy *data,
                                    struct connectdata *conn,
                                    unsigned int checks_to_perform)
{
  unsigned int ret_val = CONNRESULT_NONE;
  struct http_conn *c = &conn->proto.httpc;
  int rc;
  bool send_frames = false;

  if(checks_to_perform & CONNCHECK_ISDEAD) {
    if(http2_connisdead(data, conn))
      ret_val |= CONNRESULT_DEAD;
  }

  if(checks_to_perform & CONNCHECK_KEEPALIVE) {
    struct curltime now = Curl_now();
    timediff_t elapsed = Curl_timediff(now, conn->keepalive);

    if(elapsed > data->set.upkeep_interval_ms) {
      /* Perform an HTTP/2 PING */
      rc = nghttp2_submit_ping(c->h2, 0, ZERO_NULL);
      if(!rc) {
        /* Successfully added a PING frame to the session. Need to flag this
           so the frame is sent. */
        send_frames = true;
      }
      else {
       failf(data, "nghttp2_submit_ping() failed: %s(%d)",
             nghttp2_strerror(rc), rc);
      }

      conn->keepalive = now;
    }
  }

  if(send_frames) {
    set_transfer(c, data); /* set the transfer */
    rc = nghttp2_session_send(c->h2);
    if(rc)
      failf(data, "nghttp2_session_send() failed: %s(%d)",
            nghttp2_strerror(rc), rc);
  }

  return ret_val;
}

/* called from http_setup_conn */
void Curl_http2_setup_req(struct Curl_easy *data)
{
  struct HTTP *http = data->req.p.http;
  http->bodystarted = FALSE;
  http->status_code = -1;
  http->pausedata = NULL;
  http->pauselen = 0;
  http->closed = FALSE;
  http->close_handled = FALSE;
  http->mem = NULL;
  http->len = 0;
  http->memlen = 0;
  http->error = NGHTTP2_NO_ERROR;
}

/* called from http_setup_conn */
void Curl_http2_setup_conn(struct connectdata *conn)
{
  conn->proto.httpc.settings.max_concurrent_streams =
    DEFAULT_MAX_CONCURRENT_STREAMS;
}

/*
 * HTTP2 handler interface. This isn't added to the general list of protocols
 * but will be used at run-time when the protocol is dynamically switched from
 * HTTP to HTTP2.
 */
static const struct Curl_handler Curl_handler_http2 = {
  "HTTP",                               /* scheme */
  ZERO_NULL,                            /* setup_connection */
  Curl_http,                            /* do_it */
  Curl_http_done,                       /* done */
  ZERO_NULL,                            /* do_more */
  ZERO_NULL,                            /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  http2_getsock,                        /* proto_getsock */
  http2_getsock,                        /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  http2_getsock,                        /* perform_getsock */
  http2_disconnect,                     /* disconnect */
  ZERO_NULL,                            /* readwrite */
  http2_conncheck,                      /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_HTTP,                            /* defport */
  CURLPROTO_HTTP,                       /* protocol */
  CURLPROTO_HTTP,                       /* family */
  PROTOPT_STREAM                        /* flags */
};

static const struct Curl_handler Curl_handler_http2_ssl = {
  "HTTPS",                              /* scheme */
  ZERO_NULL,                            /* setup_connection */
  Curl_http,                            /* do_it */
  Curl_http_done,                       /* done */
  ZERO_NULL,                            /* do_more */
  ZERO_NULL,                            /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  http2_getsock,                        /* proto_getsock */
  http2_getsock,                        /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  http2_getsock,                        /* perform_getsock */
  http2_disconnect,                     /* disconnect */
  ZERO_NULL,                            /* readwrite */
  http2_conncheck,                      /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_HTTP,                            /* defport */
  CURLPROTO_HTTPS,                      /* protocol */
  CURLPROTO_HTTP,                       /* family */
  PROTOPT_SSL | PROTOPT_STREAM          /* flags */
};

/*
 * Store nghttp2 version info in this buffer.
 */
void Curl_http2_ver(char *p, size_t len)
{
  nghttp2_info *h2 = nghttp2_version(0);
  (void)msnprintf(p, len, "nghttp2/%s", h2->version_str);
}

/*
 * The implementation of nghttp2_send_callback type. Here we write |data| with
 * size |length| to the network and return the number of bytes actually
 * written. See the documentation of nghttp2_send_callback for the details.
 */
static ssize_t send_callback(nghttp2_session *h2,
                             const uint8_t *mem, size_t length, int flags,
                             void *userp)
{
  struct connectdata *conn = (struct connectdata *)userp;
  struct http_conn *c = &conn->proto.httpc;
  struct Curl_easy *data = get_transfer(c);
  ssize_t written;
  CURLcode result = CURLE_OK;

  (void)h2;
  (void)flags;

  if(!c->send_underlying)
    /* called before setup properly! */
    return NGHTTP2_ERR_CALLBACK_FAILURE;

  written = ((Curl_send*)c->send_underlying)(data, FIRSTSOCKET,
                                             mem, length, &result);

  if(result == CURLE_AGAIN) {
    return NGHTTP2_ERR_WOULDBLOCK;
  }

  if(written == -1) {
    failf(data, "Failed sending HTTP2 data");
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }

  if(!written)
    return NGHTTP2_ERR_WOULDBLOCK;

  return written;
}


/* We pass a pointer to this struct in the push callback, but the contents of
   the struct are hidden from the user. */
struct curl_pushheaders {
  struct Curl_easy *data;
  const nghttp2_push_promise *frame;
};

/*
 * push header access function. Only to be used from within the push callback
 */
char *curl_pushheader_bynum(struct curl_pushheaders *h, size_t num)
{
  /* Verify that we got a good easy handle in the push header struct, mostly to
     detect rubbish input fast(er). */
  if(!h || !GOOD_EASY_HANDLE(h->data))
    return NULL;
  else {
    struct HTTP *stream = h->data->req.p.http;
    if(num < stream->push_headers_used)
      return stream->push_headers[num];
  }
  return NULL;
}

/*
 * push header access function. Only to be used from within the push callback
 */
char *curl_pushheader_byname(struct curl_pushheaders *h, const char *header)
{
  /* Verify that we got a good easy handle in the push header struct,
     mostly to detect rubbish input fast(er). Also empty header name
     is just a rubbish too. We have to allow ":" at the beginning of
     the header, but header == ":" must be rejected. If we have ':' in
     the middle of header, it could be matched in middle of the value,
     this is because we do prefix match.*/
  if(!h || !GOOD_EASY_HANDLE(h->data) || !header || !header[0] ||
     !strcmp(header, ":") || strchr(header + 1, ':'))
    return NULL;
  else {
    struct HTTP *stream = h->data->req.p.http;
    size_t len = strlen(header);
    size_t i;
    for(i = 0; i<stream->push_headers_used; i++) {
      if(!strncmp(header, stream->push_headers[i], len)) {
        /* sub-match, make sure that it is followed by a colon */
        if(stream->push_headers[i][len] != ':')
          continue;
        return &stream->push_headers[i][len + 1];
      }
    }
  }
  return NULL;
}

/*
 * This specific transfer on this connection has been "drained".
 */
static void drained_transfer(struct Curl_easy *data,
                             struct http_conn *httpc)
{
  DEBUGASSERT(httpc->drain_total >= data->state.drain);
  httpc->drain_total -= data->state.drain;
  data->state.drain = 0;
}

/*
 * Mark this transfer to get "drained".
 */
static void drain_this(struct Curl_easy *data,
                       struct http_conn *httpc)
{
  data->state.drain++;
  httpc->drain_total++;
  DEBUGASSERT(httpc->drain_total >= data->state.drain);
}

static struct Curl_easy *duphandle(struct Curl_easy *data)
{
  struct Curl_easy *second = curl_easy_duphandle(data);
  if(second) {
    /* setup the request struct */
    struct HTTP *http = calloc(1, sizeof(struct HTTP));
    if(!http) {
      (void)Curl_close(&second);
    }
    else {
      second->req.p.http = http;
      Curl_dyn_init(&http->header_recvbuf, DYN_H2_HEADERS);
      Curl_http2_setup_req(second);
      second->state.stream_weight = data->state.stream_weight;
    }
  }
  return second;
}

static int set_transfer_url(struct Curl_easy *data,
                            struct curl_pushheaders *hp)
{
  const char *v;
  CURLUcode uc;
  char *url = NULL;
  int rc = 0;
  CURLU *u = curl_url();

  if(!u)
    return 5;

  v = curl_pushheader_byname(hp, ":scheme");
  if(v) {
    uc = curl_url_set(u, CURLUPART_SCHEME, v, 0);
    if(uc) {
      rc = 1;
      goto fail;
    }
  }

  v = curl_pushheader_byname(hp, ":authority");
  if(v) {
    uc = curl_url_set(u, CURLUPART_HOST, v, 0);
    if(uc) {
      rc = 2;
      goto fail;
    }
  }

  v = curl_pushheader_byname(hp, ":path");
  if(v) {
    uc = curl_url_set(u, CURLUPART_PATH, v, 0);
    if(uc) {
      rc = 3;
      goto fail;
    }
  }

  uc = curl_url_get(u, CURLUPART_URL, &url, 0);
  if(uc)
    rc = 4;
  fail:
  curl_url_cleanup(u);
  if(rc)
    return rc;

  if(data->state.url_alloc)
    free(data->state.url);
  data->state.url_alloc = TRUE;
  data->state.url = url;
  return 0;
}

static int push_promise(struct Curl_easy *data,
                        struct connectdata *conn,
                        const nghttp2_push_promise *frame)
{
  int rv; /* one of the CURL_PUSH_* defines */
  H2BUGF(infof(data, "PUSH_PROMISE received, stream %u!",
               frame->promised_stream_id));
  if(data->multi->push_cb) {
    struct HTTP *stream;
    struct HTTP *newstream;
    struct curl_pushheaders heads;
    CURLMcode rc;
    struct http_conn *httpc;
    size_t i;
    /* clone the parent */
    struct Curl_easy *newhandle = duphandle(data);
    if(!newhandle) {
      infof(data, "failed to duplicate handle");
      rv = CURL_PUSH_DENY; /* FAIL HARD */
      goto fail;
    }

    heads.data = data;
    heads.frame = frame;
    /* ask the application */
    H2BUGF(infof(data, "Got PUSH_PROMISE, ask application!"));

    stream = data->req.p.http;
    if(!stream) {
      failf(data, "Internal NULL stream!");
      (void)Curl_close(&newhandle);
      rv = CURL_PUSH_DENY;
      goto fail;
    }

    rv = set_transfer_url(newhandle, &heads);
    if(rv) {
      (void)Curl_close(&newhandle);
      rv = CURL_PUSH_DENY;
      goto fail;
    }

    Curl_set_in_callback(data, true);
    rv = data->multi->push_cb(data, newhandle,
                              stream->push_headers_used, &heads,
                              data->multi->push_userp);
    Curl_set_in_callback(data, false);

    /* free the headers again */
    for(i = 0; i<stream->push_headers_used; i++)
      free(stream->push_headers[i]);
    free(stream->push_headers);
    stream->push_headers = NULL;
    stream->push_headers_used = 0;

    if(rv) {
      DEBUGASSERT((rv > CURL_PUSH_OK) && (rv <= CURL_PUSH_ERROROUT));
      /* denied, kill off the new handle again */
      http2_stream_free(newhandle->req.p.http);
      newhandle->req.p.http = NULL;
      (void)Curl_close(&newhandle);
      goto fail;
    }

    newstream = newhandle->req.p.http;
    newstream->stream_id = frame->promised_stream_id;
    newhandle->req.maxdownload = -1;
    newhandle->req.size = -1;

    /* approved, add to the multi handle and immediately switch to PERFORM
       state with the given connection !*/
    rc = Curl_multi_add_perform(data->multi, newhandle, conn);
    if(rc) {
      infof(data, "failed to add handle to multi");
      http2_stream_free(newhandle->req.p.http);
      newhandle->req.p.http = NULL;
      Curl_close(&newhandle);
      rv = CURL_PUSH_DENY;
      goto fail;
    }

    httpc = &conn->proto.httpc;
    rv = nghttp2_session_set_stream_user_data(httpc->h2,
                                              frame->promised_stream_id,
                                              newhandle);
    if(rv) {
      infof(data, "failed to set user_data for stream %d",
            frame->promised_stream_id);
      DEBUGASSERT(0);
      rv = CURL_PUSH_DENY;
      goto fail;
    }
    Curl_dyn_init(&newstream->header_recvbuf, DYN_H2_HEADERS);
    Curl_dyn_init(&newstream->trailer_recvbuf, DYN_H2_TRAILERS);
  }
  else {
    H2BUGF(infof(data, "Got PUSH_PROMISE, ignore it!"));
    rv = CURL_PUSH_DENY;
  }
  fail:
  return rv;
}

/*
 * multi_connchanged() is called to tell that there is a connection in
 * this multi handle that has changed state (multiplexing become possible, the
 * number of allowed streams changed or similar), and a subsequent use of this
 * multi handle should move CONNECT_PEND handles back to CONNECT to have them
 * retry.
 */
static void multi_connchanged(struct Curl_multi *multi)
{
  multi->recheckstate = TRUE;
}

static int on_frame_recv(nghttp2_session *session, const nghttp2_frame *frame,
                         void *userp)
{
  struct connectdata *conn = (struct connectdata *)userp;
  struct http_conn *httpc = &conn->proto.httpc;
  struct Curl_easy *data_s = NULL;
  struct HTTP *stream = NULL;
  struct Curl_easy *data = get_transfer(httpc);
  int rv;
  size_t left, ncopy;
  int32_t stream_id = frame->hd.stream_id;
  CURLcode result;

  if(!stream_id) {
    /* stream ID zero is for connection-oriented stuff */
    if(frame->hd.type == NGHTTP2_SETTINGS) {
      uint32_t max_conn = httpc->settings.max_concurrent_streams;
      H2BUGF(infof(data, "Got SETTINGS"));
      httpc->settings.max_concurrent_streams =
        nghttp2_session_get_remote_settings(
          session, NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS);
      httpc->settings.enable_push =
        nghttp2_session_get_remote_settings(
          session, NGHTTP2_SETTINGS_ENABLE_PUSH);
      H2BUGF(infof(data, "MAX_CONCURRENT_STREAMS == %d",
                   httpc->settings.max_concurrent_streams));
      H2BUGF(infof(data, "ENABLE_PUSH == %s",
                   httpc->settings.enable_push?"TRUE":"false"));
      if(max_conn != httpc->settings.max_concurrent_streams) {
        /* only signal change if the value actually changed */
        infof(data,
              "Connection state changed (MAX_CONCURRENT_STREAMS == %u)!",
              httpc->settings.max_concurrent_streams);
        multi_connchanged(data->multi);
      }
    }
    return 0;
  }
  data_s = nghttp2_session_get_stream_user_data(session, stream_id);
  if(!data_s) {
    H2BUGF(infof(data,
                 "No Curl_easy associated with stream: %x",
                 stream_id));
    return 0;
  }

  stream = data_s->req.p.http;
  if(!stream) {
    H2BUGF(infof(data_s, "No proto pointer for stream: %x",
                 stream_id));
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }

  H2BUGF(infof(data_s, "on_frame_recv() header %x stream %x",
               frame->hd.type, stream_id));

  switch(frame->hd.type) {
  case NGHTTP2_DATA:
    /* If body started on this stream, then receiving DATA is illegal. */
    if(!stream->bodystarted) {
      rv = nghttp2_submit_rst_stream(session, NGHTTP2_FLAG_NONE,
                                     stream_id, NGHTTP2_PROTOCOL_ERROR);

      if(nghttp2_is_fatal(rv)) {
        return NGHTTP2_ERR_CALLBACK_FAILURE;
      }
    }
    break;
  case NGHTTP2_HEADERS:
    if(stream->bodystarted) {
      /* Only valid HEADERS after body started is trailer HEADERS.  We
         buffer them in on_header callback. */
      break;
    }

    /* nghttp2 guarantees that :status is received, and we store it to
       stream->status_code. Fuzzing has proven this can still be reached
       without status code having been set. */
    if(stream->status_code == -1)
      return NGHTTP2_ERR_CALLBACK_FAILURE;

    /* Only final status code signals the end of header */
    if(stream->status_code / 100 != 1) {
      stream->bodystarted = TRUE;
      stream->status_code = -1;
    }

    result = Curl_dyn_add(&stream->header_recvbuf, "\r\n");
    if(result)
      return NGHTTP2_ERR_CALLBACK_FAILURE;

    left = Curl_dyn_len(&stream->header_recvbuf) -
      stream->nread_header_recvbuf;
    ncopy = CURLMIN(stream->len, left);

    memcpy(&stream->mem[stream->memlen],
           Curl_dyn_ptr(&stream->header_recvbuf) +
           stream->nread_header_recvbuf,
           ncopy);
    stream->nread_header_recvbuf += ncopy;

    DEBUGASSERT(stream->mem);
    H2BUGF(infof(data_s, "Store %zu bytes headers from stream %u at %p",
                 ncopy, stream_id, stream->mem));

    stream->len -= ncopy;
    stream->memlen += ncopy;

    drain_this(data_s, httpc);
    /* if we receive data for another handle, wake that up */
    if(get_transfer(httpc) != data_s)
      Curl_expire(data_s, 0, EXPIRE_RUN_NOW);
    break;
  case NGHTTP2_PUSH_PROMISE:
    rv = push_promise(data_s, conn, &frame->push_promise);
    if(rv) { /* deny! */
      int h2;
      DEBUGASSERT((rv > CURL_PUSH_OK) && (rv <= CURL_PUSH_ERROROUT));
      h2 = nghttp2_submit_rst_stream(session, NGHTTP2_FLAG_NONE,
                                     frame->push_promise.promised_stream_id,
                                     NGHTTP2_CANCEL);
      if(nghttp2_is_fatal(h2))
        return NGHTTP2_ERR_CALLBACK_FAILURE;
      else if(rv == CURL_PUSH_ERROROUT) {
        DEBUGF(infof(data_s, "Fail the parent stream (too)"));
        return NGHTTP2_ERR_CALLBACK_FAILURE;
      }
    }
    break;
  default:
    H2BUGF(infof(data_s, "Got frame type %x for stream %u!",
                 frame->hd.type, stream_id));
    break;
  }
  return 0;
}

static int on_data_chunk_recv(nghttp2_session *session, uint8_t flags,
                              int32_t stream_id,
                              const uint8_t *mem, size_t len, void *userp)
{
  struct HTTP *stream;
  struct Curl_easy *data_s;
  size_t nread;
  struct connectdata *conn = (struct connectdata *)userp;
  struct http_conn *httpc = &conn->proto.httpc;
  (void)session;
  (void)flags;

  DEBUGASSERT(stream_id); /* should never be a zero stream ID here */

  /* get the stream from the hash based on Stream ID */
  data_s = nghttp2_session_get_stream_user_data(session, stream_id);
  if(!data_s)
    /* Receiving a Stream ID not in the hash should not happen, this is an
       internal error more than anything else! */
    return NGHTTP2_ERR_CALLBACK_FAILURE;

  stream = data_s->req.p.http;
  if(!stream)
    return NGHTTP2_ERR_CALLBACK_FAILURE;

  nread = CURLMIN(stream->len, len);
  memcpy(&stream->mem[stream->memlen], mem, nread);

  stream->len -= nread;
  stream->memlen += nread;

  drain_this(data_s, &conn->proto.httpc);

  /* if we receive data for another handle, wake that up */
  if(get_transfer(httpc) != data_s)
    Curl_expire(data_s, 0, EXPIRE_RUN_NOW);

  H2BUGF(infof(data_s, "%zu data received for stream %u "
               "(%zu left in buffer %p, total %zu)",
               nread, stream_id,
               stream->len, stream->mem,
               stream->memlen));

  if(nread < len) {
    stream->pausedata = mem + nread;
    stream->pauselen = len - nread;
    H2BUGF(infof(data_s, "NGHTTP2_ERR_PAUSE - %zu bytes out of buffer"
                 ", stream %u",
                 len - nread, stream_id));
    data_s->conn->proto.httpc.pause_stream_id = stream_id;

    return NGHTTP2_ERR_PAUSE;
  }

  /* pause execution of nghttp2 if we received data for another handle
     in order to process them first. */
  if(get_transfer(httpc) != data_s) {
    data_s->conn->proto.httpc.pause_stream_id = stream_id;

    return NGHTTP2_ERR_PAUSE;
  }

  return 0;
}

static int on_stream_close(nghttp2_session *session, int32_t stream_id,
                           uint32_t error_code, void *userp)
{
  struct Curl_easy *data_s;
  struct HTTP *stream;
  struct connectdata *conn = (struct connectdata *)userp;
  int rv;
  (void)session;
  (void)stream_id;

  if(stream_id) {
    struct http_conn *httpc;
    /* get the stream from the hash based on Stream ID, stream ID zero is for
       connection-oriented stuff */
    data_s = nghttp2_session_get_stream_user_data(session, stream_id);
    if(!data_s) {
      /* We could get stream ID not in the hash.  For example, if we
         decided to reject stream (e.g., PUSH_PROMISE). */
      return 0;
    }
    H2BUGF(infof(data_s, "on_stream_close(), %s (err %d), stream %u",
                 nghttp2_http2_strerror(error_code), error_code, stream_id));
    stream = data_s->req.p.http;
    if(!stream)
      return NGHTTP2_ERR_CALLBACK_FAILURE;

    stream->closed = TRUE;
    httpc = &conn->proto.httpc;
    drain_this(data_s, httpc);
    Curl_expire(data_s, 0, EXPIRE_RUN_NOW);
    stream->error = error_code;

    /* remove the entry from the hash as the stream is now gone */
    rv = nghttp2_session_set_stream_user_data(session, stream_id, 0);
    if(rv) {
      infof(data_s, "http/2: failed to clear user_data for stream %d!",
            stream_id);
      DEBUGASSERT(0);
    }
    if(stream_id == httpc->pause_stream_id) {
      H2BUGF(infof(data_s, "Stopped the pause stream!"));
      httpc->pause_stream_id = 0;
    }
    H2BUGF(infof(data_s, "Removed stream %u hash!", stream_id));
    stream->stream_id = 0; /* cleared */
  }
  return 0;
}

static int on_begin_headers(nghttp2_session *session,
                            const nghttp2_frame *frame, void *userp)
{
  struct HTTP *stream;
  struct Curl_easy *data_s = NULL;
  (void)userp;

  data_s = nghttp2_session_get_stream_user_data(session, frame->hd.stream_id);
  if(!data_s) {
    return 0;
  }

  H2BUGF(infof(data_s, "on_begin_headers() was called"));

  if(frame->hd.type != NGHTTP2_HEADERS) {
    return 0;
  }

  stream = data_s->req.p.http;
  if(!stream || !stream->bodystarted) {
    return 0;
  }

  return 0;
}

/* Decode HTTP status code.  Returns -1 if no valid status code was
   decoded. */
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

/* frame->hd.type is either NGHTTP2_HEADERS or NGHTTP2_PUSH_PROMISE */
static int on_header(nghttp2_session *session, const nghttp2_frame *frame,
                     const uint8_t *name, size_t namelen,
                     const uint8_t *value, size_t valuelen,
                     uint8_t flags,
                     void *userp)
{
  struct HTTP *stream;
  struct Curl_easy *data_s;
  int32_t stream_id = frame->hd.stream_id;
  struct connectdata *conn = (struct connectdata *)userp;
  struct http_conn *httpc = &conn->proto.httpc;
  CURLcode result;
  (void)flags;

  DEBUGASSERT(stream_id); /* should never be a zero stream ID here */

  /* get the stream from the hash based on Stream ID */
  data_s = nghttp2_session_get_stream_user_data(session, stream_id);
  if(!data_s)
    /* Receiving a Stream ID not in the hash should not happen, this is an
       internal error more than anything else! */
    return NGHTTP2_ERR_CALLBACK_FAILURE;

  stream = data_s->req.p.http;
  if(!stream) {
    failf(data_s, "Internal NULL stream!");
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }

  /* Store received PUSH_PROMISE headers to be used when the subsequent
     PUSH_PROMISE callback comes */
  if(frame->hd.type == NGHTTP2_PUSH_PROMISE) {
    char *h;

    if(!strcmp(":authority", (const char *)name)) {
      /* pseudo headers are lower case */
      int rc = 0;
      char *check = aprintf("%s:%d", conn->host.name, conn->remote_port);
      if(!check)
        /* no memory */
        return NGHTTP2_ERR_CALLBACK_FAILURE;
      if(!Curl_strcasecompare(check, (const char *)value) &&
         ((conn->remote_port != conn->given->defport) ||
          !Curl_strcasecompare(conn->host.name, (const char *)value))) {
        /* This is push is not for the same authority that was asked for in
         * the URL. RFC 7540 section 8.2 says: "A client MUST treat a
         * PUSH_PROMISE for which the server is not authoritative as a stream
         * error of type PROTOCOL_ERROR."
         */
        (void)nghttp2_submit_rst_stream(session, NGHTTP2_FLAG_NONE,
                                        stream_id, NGHTTP2_PROTOCOL_ERROR);
        rc = NGHTTP2_ERR_CALLBACK_FAILURE;
      }
      free(check);
      if(rc)
        return rc;
    }

    if(!stream->push_headers) {
      stream->push_headers_alloc = 10;
      stream->push_headers = malloc(stream->push_headers_alloc *
                                    sizeof(char *));
      if(!stream->push_headers)
        return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
      stream->push_headers_used = 0;
    }
    else if(stream->push_headers_used ==
            stream->push_headers_alloc) {
      char **headp;
      stream->push_headers_alloc *= 2;
      headp = Curl_saferealloc(stream->push_headers,
                               stream->push_headers_alloc * sizeof(char *));
      if(!headp) {
        stream->push_headers = NULL;
        return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
      }
      stream->push_headers = headp;
    }
    h = aprintf("%s:%s", name, value);
    if(h)
      stream->push_headers[stream->push_headers_used++] = h;
    return 0;
  }

  if(stream->bodystarted) {
    /* This is a trailer */
    H2BUGF(infof(data_s, "h2 trailer: %.*s: %.*s", namelen, name, valuelen,
                 value));
    result = Curl_dyn_addf(&stream->trailer_recvbuf,
                           "%.*s: %.*s\r\n", namelen, name,
                           valuelen, value);
    if(result)
      return NGHTTP2_ERR_CALLBACK_FAILURE;

    return 0;
  }

  if(namelen == sizeof(":status") - 1 &&
     memcmp(":status", name, namelen) == 0) {
    /* nghttp2 guarantees :status is received first and only once, and
       value is 3 digits status code, and decode_status_code always
       succeeds. */
    stream->status_code = decode_status_code(value, valuelen);
    DEBUGASSERT(stream->status_code != -1);

    result = Curl_dyn_add(&stream->header_recvbuf, "HTTP/2 ");
    if(result)
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    result = Curl_dyn_addn(&stream->header_recvbuf, value, valuelen);
    if(result)
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    /* the space character after the status code is mandatory */
    result = Curl_dyn_add(&stream->header_recvbuf, " \r\n");
    if(result)
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    /* if we receive data for another handle, wake that up */
    if(get_transfer(httpc) != data_s)
      Curl_expire(data_s, 0, EXPIRE_RUN_NOW);

    H2BUGF(infof(data_s, "h2 status: HTTP/2 %03d (easy %p)",
                 stream->status_code, data_s));
    return 0;
  }

  /* nghttp2 guarantees that namelen > 0, and :status was already
     received, and this is not pseudo-header field . */
  /* convert to a HTTP1-style header */
  result = Curl_dyn_addn(&stream->header_recvbuf, name, namelen);
  if(result)
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  result = Curl_dyn_add(&stream->header_recvbuf, ": ");
  if(result)
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  result = Curl_dyn_addn(&stream->header_recvbuf, value, valuelen);
  if(result)
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  result = Curl_dyn_add(&stream->header_recvbuf, "\r\n");
  if(result)
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  /* if we receive data for another handle, wake that up */
  if(get_transfer(httpc) != data_s)
    Curl_expire(data_s, 0, EXPIRE_RUN_NOW);

  H2BUGF(infof(data_s, "h2 header: %.*s: %.*s", namelen, name, valuelen,
               value));

  return 0; /* 0 is successful */
}

static ssize_t data_source_read_callback(nghttp2_session *session,
                                         int32_t stream_id,
                                         uint8_t *buf, size_t length,
                                         uint32_t *data_flags,
                                         nghttp2_data_source *source,
                                         void *userp)
{
  struct Curl_easy *data_s;
  struct HTTP *stream = NULL;
  size_t nread;
  (void)source;
  (void)userp;

  if(stream_id) {
    /* get the stream from the hash based on Stream ID, stream ID zero is for
       connection-oriented stuff */
    data_s = nghttp2_session_get_stream_user_data(session, stream_id);
    if(!data_s)
      /* Receiving a Stream ID not in the hash should not happen, this is an
         internal error more than anything else! */
      return NGHTTP2_ERR_CALLBACK_FAILURE;

    stream = data_s->req.p.http;
    if(!stream)
      return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  else
    return NGHTTP2_ERR_INVALID_ARGUMENT;

  nread = CURLMIN(stream->upload_len, length);
  if(nread > 0) {
    memcpy(buf, stream->upload_mem, nread);
    stream->upload_mem += nread;
    stream->upload_len -= nread;
    if(data_s->state.infilesize != -1)
      stream->upload_left -= nread;
  }

  if(stream->upload_left == 0)
    *data_flags = NGHTTP2_DATA_FLAG_EOF;
  else if(nread == 0)
    return NGHTTP2_ERR_DEFERRED;

  H2BUGF(infof(data_s, "data_source_read_callback: "
               "returns %zu bytes stream %u",
               nread, stream_id));

  return nread;
}

#if !defined(CURL_DISABLE_VERBOSE_STRINGS)
static int error_callback(nghttp2_session *session,
                          const char *msg,
                          size_t len,
                          void *userp)
{
  (void)session;
  (void)msg;
  (void)len;
  (void)userp;
  return 0;
}
#endif

static void populate_settings(struct Curl_easy *data,
                              struct http_conn *httpc)
{
  nghttp2_settings_entry *iv = httpc->local_settings;

  iv[0].settings_id = NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS;
  iv[0].value = Curl_multi_max_concurrent_streams(data->multi);

  iv[1].settings_id = NGHTTP2_SETTINGS_INITIAL_WINDOW_SIZE;
  iv[1].value = HTTP2_HUGE_WINDOW_SIZE;

  iv[2].settings_id = NGHTTP2_SETTINGS_ENABLE_PUSH;
  iv[2].value = data->multi->push_cb != NULL;

  httpc->local_settings_num = 3;
}

void Curl_http2_done(struct Curl_easy *data, bool premature)
{
  struct HTTP *http = data->req.p.http;
  struct http_conn *httpc = &data->conn->proto.httpc;

  /* there might be allocated resources done before this got the 'h2' pointer
     setup */
  Curl_dyn_free(&http->header_recvbuf);
  Curl_dyn_free(&http->trailer_recvbuf);
  if(http->push_headers) {
    /* if they weren't used and then freed before */
    for(; http->push_headers_used > 0; --http->push_headers_used) {
      free(http->push_headers[http->push_headers_used - 1]);
    }
    free(http->push_headers);
    http->push_headers = NULL;
  }

  if(!(data->conn->handler->protocol&PROTO_FAMILY_HTTP) ||
     !httpc->h2) /* not HTTP/2 ? */
    return;

  if(premature) {
    /* RST_STREAM */
    set_transfer(httpc, data); /* set the transfer */
    if(!nghttp2_submit_rst_stream(httpc->h2, NGHTTP2_FLAG_NONE,
                                  http->stream_id, NGHTTP2_STREAM_CLOSED))
      (void)nghttp2_session_send(httpc->h2);

    if(http->stream_id == httpc->pause_stream_id) {
      infof(data, "stopped the pause stream!");
      httpc->pause_stream_id = 0;
    }
  }

  if(data->state.drain)
    drained_transfer(data, httpc);

  /* -1 means unassigned and 0 means cleared */
  if(http->stream_id > 0) {
    int rv = nghttp2_session_set_stream_user_data(httpc->h2,
                                                  http->stream_id, 0);
    if(rv) {
      infof(data, "http/2: failed to clear user_data for stream %d!",
            http->stream_id);
      DEBUGASSERT(0);
    }
    set_transfer(httpc, NULL);
    http->stream_id = 0;
  }
}

/*
 * Initialize nghttp2 for a Curl connection
 */
static CURLcode http2_init(struct Curl_easy *data, struct connectdata *conn)
{
  if(!conn->proto.httpc.h2) {
    int rc;
    nghttp2_session_callbacks *callbacks;

    conn->proto.httpc.inbuf = malloc(H2_BUFSIZE);
    if(!conn->proto.httpc.inbuf)
      return CURLE_OUT_OF_MEMORY;

    rc = nghttp2_session_callbacks_new(&callbacks);

    if(rc) {
      failf(data, "Couldn't initialize nghttp2 callbacks!");
      return CURLE_OUT_OF_MEMORY; /* most likely at least */
    }

    /* nghttp2_send_callback */
    nghttp2_session_callbacks_set_send_callback(callbacks, send_callback);
    /* nghttp2_on_frame_recv_callback */
    nghttp2_session_callbacks_set_on_frame_recv_callback
      (callbacks, on_frame_recv);
    /* nghttp2_on_data_chunk_recv_callback */
    nghttp2_session_callbacks_set_on_data_chunk_recv_callback
      (callbacks, on_data_chunk_recv);
    /* nghttp2_on_stream_close_callback */
    nghttp2_session_callbacks_set_on_stream_close_callback
      (callbacks, on_stream_close);
    /* nghttp2_on_begin_headers_callback */
    nghttp2_session_callbacks_set_on_begin_headers_callback
      (callbacks, on_begin_headers);
    /* nghttp2_on_header_callback */
    nghttp2_session_callbacks_set_on_header_callback(callbacks, on_header);

    nghttp2_session_callbacks_set_error_callback(callbacks, error_callback);

    /* The nghttp2 session is not yet setup, do it */
    rc = nghttp2_session_client_new(&conn->proto.httpc.h2, callbacks, conn);

    nghttp2_session_callbacks_del(callbacks);

    if(rc) {
      failf(data, "Couldn't initialize nghttp2!");
      return CURLE_OUT_OF_MEMORY; /* most likely at least */
    }
  }
  return CURLE_OK;
}

/*
 * Append headers to ask for a HTTP1.1 to HTTP2 upgrade.
 */
CURLcode Curl_http2_request_upgrade(struct dynbuf *req,
                                    struct Curl_easy *data)
{
  CURLcode result;
  ssize_t binlen;
  char *base64;
  size_t blen;
  struct connectdata *conn = data->conn;
  struct SingleRequest *k = &data->req;
  uint8_t *binsettings = conn->proto.httpc.binsettings;
  struct http_conn *httpc = &conn->proto.httpc;

  populate_settings(data, httpc);

  /* this returns number of bytes it wrote */
  binlen = nghttp2_pack_settings_payload(binsettings, H2_BINSETTINGS_LEN,
                                         httpc->local_settings,
                                         httpc->local_settings_num);
  if(binlen <= 0) {
    failf(data, "nghttp2 unexpectedly failed on pack_settings_payload");
    Curl_dyn_free(req);
    return CURLE_FAILED_INIT;
  }
  conn->proto.httpc.binlen = binlen;

  result = Curl_base64url_encode(data, (const char *)binsettings, binlen,
                                 &base64, &blen);
  if(result) {
    Curl_dyn_free(req);
    return result;
  }

  result = Curl_dyn_addf(req,
                         "Connection: Upgrade, HTTP2-Settings\r\n"
                         "Upgrade: %s\r\n"
                         "HTTP2-Settings: %s\r\n",
                         NGHTTP2_CLEARTEXT_PROTO_VERSION_ID, base64);
  free(base64);

  k->upgr101 = UPGR101_REQUESTED;

  return result;
}

/*
 * Returns nonzero if current HTTP/2 session should be closed.
 */
static int should_close_session(struct http_conn *httpc)
{
  return httpc->drain_total == 0 && !nghttp2_session_want_read(httpc->h2) &&
    !nghttp2_session_want_write(httpc->h2);
}

/*
 * h2_process_pending_input() processes pending input left in
 * httpc->inbuf.  Then, call h2_session_send() to send pending data.
 * This function returns 0 if it succeeds, or -1 and error code will
 * be assigned to *err.
 */
static int h2_process_pending_input(struct Curl_easy *data,
                                    struct http_conn *httpc,
                                    CURLcode *err)
{
  ssize_t nread;
  char *inbuf;
  ssize_t rv;

  nread = httpc->inbuflen - httpc->nread_inbuf;
  inbuf = httpc->inbuf + httpc->nread_inbuf;

  set_transfer(httpc, data); /* set the transfer */
  rv = nghttp2_session_mem_recv(httpc->h2, (const uint8_t *)inbuf, nread);
  if(rv < 0) {
    failf(data,
          "h2_process_pending_input: nghttp2_session_mem_recv() returned "
          "%zd:%s", rv, nghttp2_strerror((int)rv));
    *err = CURLE_RECV_ERROR;
    return -1;
  }

  if(nread == rv) {
    H2BUGF(infof(data,
                 "h2_process_pending_input: All data in connection buffer "
                 "processed"));
    httpc->inbuflen = 0;
    httpc->nread_inbuf = 0;
  }
  else {
    httpc->nread_inbuf += rv;
    H2BUGF(infof(data,
                 "h2_process_pending_input: %zu bytes left in connection "
                 "buffer",
                 httpc->inbuflen - httpc->nread_inbuf));
  }

  rv = h2_session_send(data, httpc->h2);
  if(rv) {
    *err = CURLE_SEND_ERROR;
    return -1;
  }

  if(nghttp2_session_check_request_allowed(httpc->h2) == 0) {
    /* No more requests are allowed in the current session, so
       the connection may not be reused. This is set when a
       GOAWAY frame has been received or when the limit of stream
       identifiers has been reached. */
    connclose(data->conn, "http/2: No new requests allowed");
  }

  if(should_close_session(httpc)) {
    struct HTTP *stream = data->req.p.http;
    H2BUGF(infof(data,
                 "h2_process_pending_input: nothing to do in this session"));
    if(stream->error)
      *err = CURLE_HTTP2;
    else {
      /* not an error per se, but should still close the connection */
      connclose(data->conn, "GOAWAY received");
      *err = CURLE_OK;
    }
    return -1;
  }
  return 0;
}

/*
 * Called from transfer.c:done_sending when we stop uploading.
 */
CURLcode Curl_http2_done_sending(struct Curl_easy *data,
                                 struct connectdata *conn)
{
  CURLcode result = CURLE_OK;

  if((conn->handler == &Curl_handler_http2_ssl) ||
     (conn->handler == &Curl_handler_http2)) {
    /* make sure this is only attempted for HTTP/2 transfers */
    struct HTTP *stream = data->req.p.http;
    struct http_conn *httpc = &conn->proto.httpc;
    nghttp2_session *h2 = httpc->h2;

    if(stream->upload_left) {
      /* If the stream still thinks there's data left to upload. */

      stream->upload_left = 0; /* DONE! */

      /* resume sending here to trigger the callback to get called again so
         that it can signal EOF to nghttp2 */
      (void)nghttp2_session_resume_data(h2, stream->stream_id);
      (void)h2_process_pending_input(data, httpc, &result);
    }

    /* If nghttp2 still has pending frames unsent */
    if(nghttp2_session_want_write(h2)) {
      struct SingleRequest *k = &data->req;
      int rv;

      H2BUGF(infof(data, "HTTP/2 still wants to send data (easy %p)", data));

      /* and attempt to send the pending frames */
      rv = h2_session_send(data, h2);
      if(rv)
        result = CURLE_SEND_ERROR;

      if(nghttp2_session_want_write(h2)) {
         /* re-set KEEP_SEND to make sure we are called again */
         k->keepon |= KEEP_SEND;
      }
    }
  }
  return result;
}

static ssize_t http2_handle_stream_close(struct connectdata *conn,
                                         struct Curl_easy *data,
                                         struct HTTP *stream, CURLcode *err)
{
  struct http_conn *httpc = &conn->proto.httpc;

  if(httpc->pause_stream_id == stream->stream_id) {
    httpc->pause_stream_id = 0;
  }

  drained_transfer(data, httpc);

  if(httpc->pause_stream_id == 0) {
    if(h2_process_pending_input(data, httpc, err) != 0) {
      return -1;
    }
  }

  DEBUGASSERT(data->state.drain == 0);

  /* Reset to FALSE to prevent infinite loop in readwrite_data function. */
  stream->closed = FALSE;
  if(stream->error == NGHTTP2_REFUSED_STREAM) {
    H2BUGF(infof(data, "REFUSED_STREAM (%d), try again on a new connection!",
                 stream->stream_id));
    connclose(conn, "REFUSED_STREAM"); /* don't use this anymore */
    data->state.refused_stream = TRUE;
    *err = CURLE_RECV_ERROR; /* trigger Curl_retry_request() later */
    return -1;
  }
  else if(stream->error != NGHTTP2_NO_ERROR) {
    failf(data, "HTTP/2 stream %d was not closed cleanly: %s (err %u)",
          stream->stream_id, nghttp2_http2_strerror(stream->error),
          stream->error);
    *err = CURLE_HTTP2_STREAM;
    return -1;
  }

  if(!stream->bodystarted) {
    failf(data, "HTTP/2 stream %d was closed cleanly, but before getting "
          " all response header fields, treated as error",
          stream->stream_id);
    *err = CURLE_HTTP2_STREAM;
    return -1;
  }

  if(Curl_dyn_len(&stream->trailer_recvbuf)) {
    char *trailp = Curl_dyn_ptr(&stream->trailer_recvbuf);
    char *lf;

    do {
      size_t len = 0;
      CURLcode result;
      /* each trailer line ends with a newline */
      lf = strchr(trailp, '\n');
      if(!lf)
        break;
      len = lf + 1 - trailp;

      Curl_debug(data, CURLINFO_HEADER_IN, trailp, len);
      /* pass the trailers one by one to the callback */
      result = Curl_client_write(data, CLIENTWRITE_HEADER, trailp, len);
      if(result) {
        *err = result;
        return -1;
      }
      trailp = ++lf;
    } while(lf);
  }

  stream->close_handled = TRUE;

  H2BUGF(infof(data, "http2_recv returns 0, http2_handle_stream_close"));
  return 0;
}

/*
 * h2_pri_spec() fills in the pri_spec struct, used by nghttp2 to send weight
 * and dependency to the peer. It also stores the updated values in the state
 * struct.
 */

static void h2_pri_spec(struct Curl_easy *data,
                        nghttp2_priority_spec *pri_spec)
{
  struct HTTP *depstream = (data->set.stream_depends_on?
                            data->set.stream_depends_on->req.p.http:NULL);
  int32_t depstream_id = depstream? depstream->stream_id:0;
  nghttp2_priority_spec_init(pri_spec, depstream_id, data->set.stream_weight,
                             data->set.stream_depends_e);
  data->state.stream_weight = data->set.stream_weight;
  data->state.stream_depends_e = data->set.stream_depends_e;
  data->state.stream_depends_on = data->set.stream_depends_on;
}

/*
 * h2_session_send() checks if there's been an update in the priority /
 * dependency settings and if so it submits a PRIORITY frame with the updated
 * info.
 */
static int h2_session_send(struct Curl_easy *data,
                           nghttp2_session *h2)
{
  struct HTTP *stream = data->req.p.http;
  struct http_conn *httpc = &data->conn->proto.httpc;
  set_transfer(httpc, data);
  if((data->set.stream_weight != data->state.stream_weight) ||
     (data->set.stream_depends_e != data->state.stream_depends_e) ||
     (data->set.stream_depends_on != data->state.stream_depends_on) ) {
    /* send new weight and/or dependency */
    nghttp2_priority_spec pri_spec;
    int rv;

    h2_pri_spec(data, &pri_spec);

    H2BUGF(infof(data, "Queuing PRIORITY on stream %u (easy %p)",
                 stream->stream_id, data));
    DEBUGASSERT(stream->stream_id != -1);
    rv = nghttp2_submit_priority(h2, NGHTTP2_FLAG_NONE, stream->stream_id,
                                 &pri_spec);
    if(rv)
      return rv;
  }

  return nghttp2_session_send(h2);
}

static ssize_t http2_recv(struct Curl_easy *data, int sockindex,
                          char *mem, size_t len, CURLcode *err)
{
  ssize_t nread;
  struct connectdata *conn = data->conn;
  struct http_conn *httpc = &conn->proto.httpc;
  struct HTTP *stream = data->req.p.http;

  (void)sockindex; /* we always do HTTP2 on sockindex 0 */

  if(should_close_session(httpc)) {
    H2BUGF(infof(data,
                 "http2_recv: nothing to do in this session"));
    if(conn->bits.close) {
      /* already marked for closure, return OK and we're done */
      *err = CURLE_OK;
      return 0;
    }
    *err = CURLE_HTTP2;
    return -1;
  }

  /* Nullify here because we call nghttp2_session_send() and they
     might refer to the old buffer. */
  stream->upload_mem = NULL;
  stream->upload_len = 0;

  /*
   * At this point 'stream' is just in the Curl_easy the connection
   * identifies as its owner at this time.
   */

  if(stream->bodystarted &&
     stream->nread_header_recvbuf < Curl_dyn_len(&stream->header_recvbuf)) {
    /* If there is header data pending for this stream to return, do that */
    size_t left =
      Curl_dyn_len(&stream->header_recvbuf) - stream->nread_header_recvbuf;
    size_t ncopy = CURLMIN(len, left);
    memcpy(mem, Curl_dyn_ptr(&stream->header_recvbuf) +
           stream->nread_header_recvbuf, ncopy);
    stream->nread_header_recvbuf += ncopy;

    H2BUGF(infof(data, "http2_recv: Got %d bytes from header_recvbuf",
                 (int)ncopy));
    return ncopy;
  }

  H2BUGF(infof(data, "http2_recv: easy %p (stream %u) win %u/%u",
               data, stream->stream_id,
               nghttp2_session_get_local_window_size(httpc->h2),
               nghttp2_session_get_stream_local_window_size(httpc->h2,
                                                            stream->stream_id)
           ));

  if((data->state.drain) && stream->memlen) {
    H2BUGF(infof(data, "http2_recv: DRAIN %zu bytes stream %u!! (%p => %p)",
                 stream->memlen, stream->stream_id,
                 stream->mem, mem));
    if(mem != stream->mem) {
      /* if we didn't get the same buffer this time, we must move the data to
         the beginning */
      memmove(mem, stream->mem, stream->memlen);
      stream->len = len - stream->memlen;
      stream->mem = mem;
    }
    if(httpc->pause_stream_id == stream->stream_id && !stream->pausedata) {
      /* We have paused nghttp2, but we have no pause data (see
         on_data_chunk_recv). */
      httpc->pause_stream_id = 0;
      if(h2_process_pending_input(data, httpc, err) != 0) {
        return -1;
      }
    }
  }
  else if(stream->pausedata) {
    DEBUGASSERT(httpc->pause_stream_id == stream->stream_id);
    nread = CURLMIN(len, stream->pauselen);
    memcpy(mem, stream->pausedata, nread);

    stream->pausedata += nread;
    stream->pauselen -= nread;

    if(stream->pauselen == 0) {
      H2BUGF(infof(data, "Unpaused by stream %u", stream->stream_id));
      DEBUGASSERT(httpc->pause_stream_id == stream->stream_id);
      httpc->pause_stream_id = 0;

      stream->pausedata = NULL;
      stream->pauselen = 0;

      /* When NGHTTP2_ERR_PAUSE is returned from
         data_source_read_callback, we might not process DATA frame
         fully.  Calling nghttp2_session_mem_recv() again will
         continue to process DATA frame, but if there is no incoming
         frames, then we have to call it again with 0-length data.
         Without this, on_stream_close callback will not be called,
         and stream could be hanged. */
      if(h2_process_pending_input(data, httpc, err) != 0) {
        return -1;
      }
    }
    H2BUGF(infof(data, "http2_recv: returns unpaused %zd bytes on stream %u",
                 nread, stream->stream_id));
    return nread;
  }
  else if(httpc->pause_stream_id) {
    /* If a stream paused nghttp2_session_mem_recv previously, and has
       not processed all data, it still refers to the buffer in
       nghttp2_session.  If we call nghttp2_session_mem_recv(), we may
       overwrite that buffer.  To avoid that situation, just return
       here with CURLE_AGAIN.  This could be busy loop since data in
       socket is not read.  But it seems that usually streams are
       notified with its drain property, and socket is read again
       quickly. */
    if(stream->closed)
      /* closed overrides paused */
      return 0;
    H2BUGF(infof(data, "stream %x is paused, pause id: %x",
                 stream->stream_id, httpc->pause_stream_id));
    *err = CURLE_AGAIN;
    return -1;
  }
  else {
    /* remember where to store incoming data for this stream and how big the
       buffer is */
    stream->mem = mem;
    stream->len = len;
    stream->memlen = 0;

    if(httpc->inbuflen == 0) {
      nread = ((Curl_recv *)httpc->recv_underlying)(
        data, FIRSTSOCKET, httpc->inbuf, H2_BUFSIZE, err);

      if(nread == -1) {
        if(*err != CURLE_AGAIN)
          failf(data, "Failed receiving HTTP2 data");
        else if(stream->closed)
          /* received when the stream was already closed! */
          return http2_handle_stream_close(conn, data, stream, err);

        return -1;
      }

      if(nread == 0) {
        if(!stream->closed) {
          /* This will happen when the server or proxy server is SIGKILLed
             during data transfer. We should emit an error since our data
             received may be incomplete. */
          failf(data, "HTTP/2 stream %d was not closed cleanly before"
                " end of the underlying stream",
                stream->stream_id);
          *err = CURLE_HTTP2_STREAM;
          return -1;
        }

        H2BUGF(infof(data, "end of stream"));
        *err = CURLE_OK;
        return 0;
      }

      H2BUGF(infof(data, "nread=%zd", nread));

      httpc->inbuflen = nread;

      DEBUGASSERT(httpc->nread_inbuf == 0);
    }
    else {
      nread = httpc->inbuflen - httpc->nread_inbuf;
      (void)nread;  /* silence warning, used in debug */
      H2BUGF(infof(data, "Use data left in connection buffer, nread=%zd",
                   nread));
    }

    if(h2_process_pending_input(data, httpc, err))
      return -1;
  }
  if(stream->memlen) {
    ssize_t retlen = stream->memlen;
    H2BUGF(infof(data, "http2_recv: returns %zd for stream %u",
                 retlen, stream->stream_id));
    stream->memlen = 0;

    if(httpc->pause_stream_id == stream->stream_id) {
      /* data for this stream is returned now, but this stream caused a pause
         already so we need it called again asap */
      H2BUGF(infof(data, "Data returned for PAUSED stream %u",
                   stream->stream_id));
    }
    else if(!stream->closed) {
      drained_transfer(data, httpc);
    }
    else
      /* this stream is closed, trigger a another read ASAP to detect that */
      Curl_expire(data, 0, EXPIRE_RUN_NOW);

    return retlen;
  }
  if(stream->closed)
    return http2_handle_stream_close(conn, data, stream, err);
  *err = CURLE_AGAIN;
  H2BUGF(infof(data, "http2_recv returns AGAIN for stream %u",
               stream->stream_id));
  return -1;
}

/* Index where :authority header field will appear in request header
   field list. */
#define AUTHORITY_DST_IDX 3

/* USHRT_MAX is 65535 == 0xffff */
#define HEADER_OVERFLOW(x) \
  (x.namelen > 0xffff || x.valuelen > 0xffff - x.namelen)

/*
 * Check header memory for the token "trailers".
 * Parse the tokens as separated by comma and surrounded by whitespace.
 * Returns TRUE if found or FALSE if not.
 */
static bool contains_trailers(const char *p, size_t len)
{
  const char *end = p + len;
  for(;;) {
    for(; p != end && (*p == ' ' || *p == '\t'); ++p)
      ;
    if(p == end || (size_t)(end - p) < sizeof("trailers") - 1)
      return FALSE;
    if(strncasecompare("trailers", p, sizeof("trailers") - 1)) {
      p += sizeof("trailers") - 1;
      for(; p != end && (*p == ' ' || *p == '\t'); ++p)
        ;
      if(p == end || *p == ',')
        return TRUE;
    }
    /* skip to next token */
    for(; p != end && *p != ','; ++p)
      ;
    if(p == end)
      return FALSE;
    ++p;
  }
}

typedef enum {
  /* Send header to server */
  HEADERINST_FORWARD,
  /* Don't send header to server */
  HEADERINST_IGNORE,
  /* Discard header, and replace it with "te: trailers" */
  HEADERINST_TE_TRAILERS
} header_instruction;

/* Decides how to treat given header field. */
static header_instruction inspect_header(const char *name, size_t namelen,
                                         const char *value, size_t valuelen) {
  switch(namelen) {
  case 2:
    if(!strncasecompare("te", name, namelen))
      return HEADERINST_FORWARD;

    return contains_trailers(value, valuelen) ?
           HEADERINST_TE_TRAILERS : HEADERINST_IGNORE;
  case 7:
    return strncasecompare("upgrade", name, namelen) ?
           HEADERINST_IGNORE : HEADERINST_FORWARD;
  case 10:
    return (strncasecompare("connection", name, namelen) ||
            strncasecompare("keep-alive", name, namelen)) ?
           HEADERINST_IGNORE : HEADERINST_FORWARD;
  case 16:
    return strncasecompare("proxy-connection", name, namelen) ?
           HEADERINST_IGNORE : HEADERINST_FORWARD;
  case 17:
    return strncasecompare("transfer-encoding", name, namelen) ?
           HEADERINST_IGNORE : HEADERINST_FORWARD;
  default:
    return HEADERINST_FORWARD;
  }
}

static ssize_t http2_send(struct Curl_easy *data, int sockindex,
                          const void *mem, size_t len, CURLcode *err)
{
  /*
   * Currently, we send request in this function, but this function is also
   * used to send request body. It would be nice to add dedicated function for
   * request.
   */
  int rv;
  struct connectdata *conn = data->conn;
  struct http_conn *httpc = &conn->proto.httpc;
  struct HTTP *stream = data->req.p.http;
  nghttp2_nv *nva = NULL;
  size_t nheader;
  size_t i;
  size_t authority_idx;
  char *hdbuf = (char *)mem;
  char *end, *line_end;
  nghttp2_data_provider data_prd;
  int32_t stream_id;
  nghttp2_session *h2 = httpc->h2;
  nghttp2_priority_spec pri_spec;

  (void)sockindex;

  H2BUGF(infof(data, "http2_send len=%zu", len));

  if(stream->stream_id != -1) {
    if(stream->close_handled) {
      infof(data, "stream %d closed", stream->stream_id);
      *err = CURLE_HTTP2_STREAM;
      return -1;
    }
    else if(stream->closed) {
      return http2_handle_stream_close(conn, data, stream, err);
    }
    /* If stream_id != -1, we have dispatched request HEADERS, and now
       are going to send or sending request body in DATA frame */
    stream->upload_mem = mem;
    stream->upload_len = len;
    rv = nghttp2_session_resume_data(h2, stream->stream_id);
    if(nghttp2_is_fatal(rv)) {
      *err = CURLE_SEND_ERROR;
      return -1;
    }
    rv = h2_session_send(data, h2);
    if(nghttp2_is_fatal(rv)) {
      *err = CURLE_SEND_ERROR;
      return -1;
    }
    len -= stream->upload_len;

    /* Nullify here because we call nghttp2_session_send() and they
       might refer to the old buffer. */
    stream->upload_mem = NULL;
    stream->upload_len = 0;

    if(should_close_session(httpc)) {
      H2BUGF(infof(data, "http2_send: nothing to do in this session"));
      *err = CURLE_HTTP2;
      return -1;
    }

    if(stream->upload_left) {
      /* we are sure that we have more data to send here.  Calling the
         following API will make nghttp2_session_want_write() return
         nonzero if remote window allows it, which then libcurl checks
         socket is writable or not.  See http2_perform_getsock(). */
      nghttp2_session_resume_data(h2, stream->stream_id);
    }

#ifdef DEBUG_HTTP2
    if(!len) {
      infof(data, "http2_send: easy %p (stream %u) win %u/%u",
            data, stream->stream_id,
            nghttp2_session_get_remote_window_size(httpc->h2),
            nghttp2_session_get_stream_remote_window_size(httpc->h2,
                                                          stream->stream_id)
        );

    }
    infof(data, "http2_send returns %zu for stream %u", len,
          stream->stream_id);
#endif
    return len;
  }

  /* Calculate number of headers contained in [mem, mem + len) */
  /* Here, we assume the curl http code generate *correct* HTTP header
     field block */
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
  nva = malloc(sizeof(nghttp2_nv) * nheader);
  if(!nva) {
    *err = CURLE_OUT_OF_MEMORY;
    return -1;
  }

  /* Extract :method, :path from request line
     We do line endings with CRLF so checking for CR is enough */
  line_end = memchr(hdbuf, '\r', len);
  if(!line_end)
    goto fail;

  /* Method does not contain spaces */
  end = memchr(hdbuf, ' ', line_end - hdbuf);
  if(!end || end == hdbuf)
    goto fail;
  nva[0].name = (unsigned char *)":method";
  nva[0].namelen = strlen((char *)nva[0].name);
  nva[0].value = (unsigned char *)hdbuf;
  nva[0].valuelen = (size_t)(end - hdbuf);
  nva[0].flags = NGHTTP2_NV_FLAG_NONE;
  if(HEADER_OVERFLOW(nva[0])) {
    failf(data, "Failed sending HTTP request: Header overflow");
    goto fail;
  }

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
  nva[1].flags = NGHTTP2_NV_FLAG_NONE;
  if(HEADER_OVERFLOW(nva[1])) {
    failf(data, "Failed sending HTTP request: Header overflow");
    goto fail;
  }

  nva[2].name = (unsigned char *)":scheme";
  nva[2].namelen = strlen((char *)nva[2].name);
  if(conn->handler->flags & PROTOPT_SSL)
    nva[2].value = (unsigned char *)"https";
  else
    nva[2].value = (unsigned char *)"http";
  nva[2].valuelen = strlen((char *)nva[2].value);
  nva[2].flags = NGHTTP2_NV_FLAG_NONE;
  if(HEADER_OVERFLOW(nva[2])) {
    failf(data, "Failed sending HTTP request: Header overflow");
    goto fail;
  }

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
      /* Lower case the header name for HTTP/2 */
      Curl_strntolower((char *)hdbuf, hdbuf, nva[i].namelen);
      nva[i].name = (unsigned char *)hdbuf;
    }
    hdbuf = end + 1;
    while(*hdbuf == ' ' || *hdbuf == '\t')
      ++hdbuf;
    end = line_end;

    switch(inspect_header((const char *)nva[i].name, nva[i].namelen, hdbuf,
                          end - hdbuf)) {
    case HEADERINST_IGNORE:
      /* skip header fields prohibited by HTTP/2 specification. */
      --nheader;
      continue;
    case HEADERINST_TE_TRAILERS:
      nva[i].value = (uint8_t*)"trailers";
      nva[i].valuelen = sizeof("trailers") - 1;
      break;
    default:
      nva[i].value = (unsigned char *)hdbuf;
      nva[i].valuelen = (size_t)(end - hdbuf);
    }

    nva[i].flags = NGHTTP2_NV_FLAG_NONE;
    if(HEADER_OVERFLOW(nva[i])) {
      failf(data, "Failed sending HTTP request: Header overflow");
      goto fail;
    }
    ++i;
  }

  /* :authority must come before non-pseudo header fields */
  if(authority_idx && authority_idx != AUTHORITY_DST_IDX) {
    nghttp2_nv authority = nva[authority_idx];
    for(i = authority_idx; i > AUTHORITY_DST_IDX; --i) {
      nva[i] = nva[i - 1];
    }
    nva[i] = authority;
  }

  /* Warn stream may be rejected if cumulative length of headers is too large.
     It appears nghttp2 will not send a header frame larger than 64KB. */
#define MAX_ACC 60000  /* <64KB to account for some overhead */
  {
    size_t acc = 0;

    for(i = 0; i < nheader; ++i) {
      acc += nva[i].namelen + nva[i].valuelen;

      H2BUGF(infof(data, "h2 header: %.*s:%.*s",
                   nva[i].namelen, nva[i].name,
                   nva[i].valuelen, nva[i].value));
    }

    if(acc > MAX_ACC) {
      infof(data, "http2_send: Warning: The cumulative length of all "
            "headers exceeds %d bytes and that could cause the "
            "stream to be rejected.", MAX_ACC);
    }
  }

  h2_pri_spec(data, &pri_spec);

  H2BUGF(infof(data, "http2_send request allowed %d (easy handle %p)",
               nghttp2_session_check_request_allowed(h2), (void *)data));

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

    data_prd.read_callback = data_source_read_callback;
    data_prd.source.ptr = NULL;
    stream_id = nghttp2_submit_request(h2, &pri_spec, nva, nheader,
                                       &data_prd, data);
    break;
  default:
    stream_id = nghttp2_submit_request(h2, &pri_spec, nva, nheader,
                                       NULL, data);
  }

  Curl_safefree(nva);

  if(stream_id < 0) {
    H2BUGF(infof(data,
                 "http2_send() nghttp2_submit_request error (%s)%d",
                 nghttp2_strerror(stream_id), stream_id));
    *err = CURLE_SEND_ERROR;
    return -1;
  }

  infof(data, "Using Stream ID: %x (easy handle %p)",
        stream_id, (void *)data);
  stream->stream_id = stream_id;

  rv = h2_session_send(data, h2);
  if(rv) {
    H2BUGF(infof(data,
                 "http2_send() nghttp2_session_send error (%s)%d",
                 nghttp2_strerror(rv), rv));

    *err = CURLE_SEND_ERROR;
    return -1;
  }

  if(should_close_session(httpc)) {
    H2BUGF(infof(data, "http2_send: nothing to do in this session"));
    *err = CURLE_HTTP2;
    return -1;
  }

  /* If whole HEADERS frame was sent off to the underlying socket, the nghttp2
     library calls data_source_read_callback. But only it found that no data
     available, so it deferred the DATA transmission. Which means that
     nghttp2_session_want_write() returns 0 on http2_perform_getsock(), which
     results that no writable socket check is performed. To workaround this,
     we issue nghttp2_session_resume_data() here to bring back DATA
     transmission from deferred state. */
  nghttp2_session_resume_data(h2, stream->stream_id);

  return len;

fail:
  free(nva);
  *err = CURLE_SEND_ERROR;
  return -1;
}

CURLcode Curl_http2_setup(struct Curl_easy *data,
                          struct connectdata *conn)
{
  CURLcode result;
  struct http_conn *httpc = &conn->proto.httpc;
  struct HTTP *stream = data->req.p.http;

  DEBUGASSERT(data->state.buffer);

  stream->stream_id = -1;

  Curl_dyn_init(&stream->header_recvbuf, DYN_H2_HEADERS);
  Curl_dyn_init(&stream->trailer_recvbuf, DYN_H2_TRAILERS);

  stream->upload_left = 0;
  stream->upload_mem = NULL;
  stream->upload_len = 0;
  stream->mem = data->state.buffer;
  stream->len = data->set.buffer_size;

  multi_connchanged(data->multi);
  /* below this point only connection related inits are done, which only needs
     to be done once per connection */

  if((conn->handler == &Curl_handler_http2_ssl) ||
     (conn->handler == &Curl_handler_http2))
    return CURLE_OK; /* already done */

  if(conn->handler->flags & PROTOPT_SSL)
    conn->handler = &Curl_handler_http2_ssl;
  else
    conn->handler = &Curl_handler_http2;

  result = http2_init(data, conn);
  if(result) {
    Curl_dyn_free(&stream->header_recvbuf);
    return result;
  }

  infof(data, "Using HTTP2, server supports multiplexing");

  conn->bits.multiplex = TRUE; /* at least potentially multiplexed */
  conn->httpversion = 20;
  conn->bundle->multiuse = BUNDLE_MULTIPLEX;

  httpc->inbuflen = 0;
  httpc->nread_inbuf = 0;

  httpc->pause_stream_id = 0;
  httpc->drain_total = 0;

  infof(data, "Connection state changed (HTTP/2 confirmed)");

  return CURLE_OK;
}

CURLcode Curl_http2_switched(struct Curl_easy *data,
                             const char *mem, size_t nread)
{
  CURLcode result;
  struct connectdata *conn = data->conn;
  struct http_conn *httpc = &conn->proto.httpc;
  int rv;
  struct HTTP *stream = data->req.p.http;

  result = Curl_http2_setup(data, conn);
  if(result)
    return result;

  httpc->recv_underlying = conn->recv[FIRSTSOCKET];
  httpc->send_underlying = conn->send[FIRSTSOCKET];
  conn->recv[FIRSTSOCKET] = http2_recv;
  conn->send[FIRSTSOCKET] = http2_send;

  if(data->req.upgr101 == UPGR101_RECEIVED) {
    /* stream 1 is opened implicitly on upgrade */
    stream->stream_id = 1;
    /* queue SETTINGS frame (again) */
    rv = nghttp2_session_upgrade2(httpc->h2, httpc->binsettings, httpc->binlen,
                                  data->state.httpreq == HTTPREQ_HEAD, NULL);
    if(rv) {
      failf(data, "nghttp2_session_upgrade2() failed: %s(%d)",
            nghttp2_strerror(rv), rv);
      return CURLE_HTTP2;
    }

    rv = nghttp2_session_set_stream_user_data(httpc->h2,
                                              stream->stream_id,
                                              data);
    if(rv) {
      infof(data, "http/2: failed to set user_data for stream %d!",
            stream->stream_id);
      DEBUGASSERT(0);
    }
  }
  else {
    populate_settings(data, httpc);

    /* stream ID is unknown at this point */
    stream->stream_id = -1;
    rv = nghttp2_submit_settings(httpc->h2, NGHTTP2_FLAG_NONE,
                                 httpc->local_settings,
                                 httpc->local_settings_num);
    if(rv) {
      failf(data, "nghttp2_submit_settings() failed: %s(%d)",
            nghttp2_strerror(rv), rv);
      return CURLE_HTTP2;
    }
  }

  rv = nghttp2_session_set_local_window_size(httpc->h2, NGHTTP2_FLAG_NONE, 0,
                                             HTTP2_HUGE_WINDOW_SIZE);
  if(rv) {
    failf(data, "nghttp2_session_set_local_window_size() failed: %s(%d)",
          nghttp2_strerror(rv), rv);
    return CURLE_HTTP2;
  }

  /* we are going to copy mem to httpc->inbuf.  This is required since
     mem is part of buffer pointed by stream->mem, and callbacks
     called by nghttp2_session_mem_recv() will write stream specific
     data into stream->mem, overwriting data already there. */
  if(H2_BUFSIZE < nread) {
    failf(data, "connection buffer size is too small to store data following "
          "HTTP Upgrade response header: buflen=%d, datalen=%zu",
          H2_BUFSIZE, nread);
    return CURLE_HTTP2;
  }

  infof(data, "Copying HTTP/2 data in stream buffer to connection buffer"
        " after upgrade: len=%zu",
        nread);

  if(nread)
    memcpy(httpc->inbuf, mem, nread);

  httpc->inbuflen = nread;

  DEBUGASSERT(httpc->nread_inbuf == 0);

  if(-1 == h2_process_pending_input(data, httpc, &result))
    return CURLE_HTTP2;

  return CURLE_OK;
}

CURLcode Curl_http2_stream_pause(struct Curl_easy *data, bool pause)
{
  DEBUGASSERT(data);
  DEBUGASSERT(data->conn);
  /* if it isn't HTTP/2, we're done */
  if(!(data->conn->handler->protocol & PROTO_FAMILY_HTTP) ||
     !data->conn->proto.httpc.h2)
    return CURLE_OK;
#ifdef NGHTTP2_HAS_SET_LOCAL_WINDOW_SIZE
  else {
    struct HTTP *stream = data->req.p.http;
    struct http_conn *httpc = &data->conn->proto.httpc;
    uint32_t window = !pause * HTTP2_HUGE_WINDOW_SIZE;
    int rv = nghttp2_session_set_local_window_size(httpc->h2,
                                                   NGHTTP2_FLAG_NONE,
                                                   stream->stream_id,
                                                   window);
    if(rv) {
      failf(data, "nghttp2_session_set_local_window_size() failed: %s(%d)",
            nghttp2_strerror(rv), rv);
      return CURLE_HTTP2;
    }

    /* make sure the window update gets sent */
    rv = h2_session_send(data, httpc->h2);
    if(rv)
      return CURLE_SEND_ERROR;

    DEBUGF(infof(data, "Set HTTP/2 window size to %u for stream %u",
                 window, stream->stream_id));

#ifdef DEBUGBUILD
    {
      /* read out the stream local window again */
      uint32_t window2 =
        nghttp2_session_get_stream_local_window_size(httpc->h2,
                                                     stream->stream_id);
      DEBUGF(infof(data, "HTTP/2 window size is now %u for stream %u",
                   window2, stream->stream_id));
    }
#endif
  }
#endif
  return CURLE_OK;
}

CURLcode Curl_http2_add_child(struct Curl_easy *parent,
                              struct Curl_easy *child,
                              bool exclusive)
{
  if(parent) {
    struct Curl_http2_dep **tail;
    struct Curl_http2_dep *dep = calloc(1, sizeof(struct Curl_http2_dep));
    if(!dep)
      return CURLE_OUT_OF_MEMORY;
    dep->data = child;

    if(parent->set.stream_dependents && exclusive) {
      struct Curl_http2_dep *node = parent->set.stream_dependents;
      while(node) {
        node->data->set.stream_depends_on = child;
        node = node->next;
      }

      tail = &child->set.stream_dependents;
      while(*tail)
        tail = &(*tail)->next;

      DEBUGASSERT(!*tail);
      *tail = parent->set.stream_dependents;
      parent->set.stream_dependents = 0;
    }

    tail = &parent->set.stream_dependents;
    while(*tail) {
      (*tail)->data->set.stream_depends_e = FALSE;
      tail = &(*tail)->next;
    }

    DEBUGASSERT(!*tail);
    *tail = dep;
  }

  child->set.stream_depends_on = parent;
  child->set.stream_depends_e = exclusive;
  return CURLE_OK;
}

void Curl_http2_remove_child(struct Curl_easy *parent, struct Curl_easy *child)
{
  struct Curl_http2_dep *last = 0;
  struct Curl_http2_dep *data = parent->set.stream_dependents;
  DEBUGASSERT(child->set.stream_depends_on == parent);

  while(data && data->data != child) {
    last = data;
    data = data->next;
  }

  DEBUGASSERT(data);

  if(data) {
    if(last) {
      last->next = data->next;
    }
    else {
      parent->set.stream_dependents = data->next;
    }
    free(data);
  }

  child->set.stream_depends_on = 0;
  child->set.stream_depends_e = FALSE;
}

void Curl_http2_cleanup_dependencies(struct Curl_easy *data)
{
  while(data->set.stream_dependents) {
    struct Curl_easy *tmp = data->set.stream_dependents->data;
    Curl_http2_remove_child(data, tmp);
    if(data->set.stream_depends_on)
      Curl_http2_add_child(data->set.stream_depends_on, tmp, FALSE);
  }

  if(data->set.stream_depends_on)
    Curl_http2_remove_child(data->set.stream_depends_on, data);
}

/* Only call this function for a transfer that already got a HTTP/2
   CURLE_HTTP2_STREAM error! */
bool Curl_h2_http_1_1_error(struct Curl_easy *data)
{
  struct HTTP *stream = data->req.p.http;
  return (stream->error == NGHTTP2_HTTP_1_1_REQUIRED);
}

#else /* !USE_NGHTTP2 */

/* Satisfy external references even if http2 is not compiled in. */
#include <curl/curl.h>

char *curl_pushheader_bynum(struct curl_pushheaders *h, size_t num)
{
  (void) h;
  (void) num;
  return NULL;
}

char *curl_pushheader_byname(struct curl_pushheaders *h, const char *header)
{
  (void) h;
  (void) header;
  return NULL;
}

#endif /* USE_NGHTTP2 */
