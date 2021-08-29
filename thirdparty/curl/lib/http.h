#ifndef HEADER_CURL_HTTP_H
#define HEADER_CURL_HTTP_H
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

typedef enum {
  HTTPREQ_GET,
  HTTPREQ_POST,
  HTTPREQ_POST_FORM, /* we make a difference internally */
  HTTPREQ_POST_MIME, /* we make a difference internally */
  HTTPREQ_PUT,
  HTTPREQ_HEAD
} Curl_HttpReq;

#ifndef CURL_DISABLE_HTTP

#ifdef USE_NGHTTP2
#include <nghttp2/nghttp2.h>
#endif

extern const struct Curl_handler Curl_handler_http;

#ifdef USE_SSL
extern const struct Curl_handler Curl_handler_https;
#endif

/* Header specific functions */
bool Curl_compareheader(const char *headerline,  /* line to check */
                        const char *header,   /* header keyword _with_ colon */
                        const char *content); /* content string to find */

char *Curl_copy_header_value(const char *header);

char *Curl_checkProxyheaders(struct Curl_easy *data,
                             const struct connectdata *conn,
                             const char *thisheader);
CURLcode Curl_buffer_send(struct dynbuf *in,
                          struct Curl_easy *data,
                          curl_off_t *bytes_written,
                          curl_off_t included_body_bytes,
                          int socketindex);

CURLcode Curl_add_timecondition(struct Curl_easy *data,
#ifndef USE_HYPER
                                struct dynbuf *req
#else
                                void *headers
#endif
  );
CURLcode Curl_add_custom_headers(struct Curl_easy *data,
                                 bool is_connect,
#ifndef USE_HYPER
                                 struct dynbuf *req
#else
                                 void *headers
#endif
  );
CURLcode Curl_http_compile_trailers(struct curl_slist *trailers,
                                    struct dynbuf *buf,
                                    struct Curl_easy *handle);

void Curl_http_method(struct Curl_easy *data, struct connectdata *conn,
                      const char **method, Curl_HttpReq *);
CURLcode Curl_http_useragent(struct Curl_easy *data);
CURLcode Curl_http_host(struct Curl_easy *data, struct connectdata *conn);
CURLcode Curl_http_target(struct Curl_easy *data, struct connectdata *conn,
                          struct dynbuf *req);
CURLcode Curl_http_statusline(struct Curl_easy *data,
                              struct connectdata *conn);
CURLcode Curl_http_header(struct Curl_easy *data, struct connectdata *conn,
                          char *headp);
CURLcode Curl_transferencode(struct Curl_easy *data);
CURLcode Curl_http_body(struct Curl_easy *data, struct connectdata *conn,
                        Curl_HttpReq httpreq,
                        const char **teep);
CURLcode Curl_http_bodysend(struct Curl_easy *data, struct connectdata *conn,
                            struct dynbuf *r, Curl_HttpReq httpreq);
bool Curl_use_http_1_1plus(const struct Curl_easy *data,
                           const struct connectdata *conn);
#ifndef CURL_DISABLE_COOKIES
CURLcode Curl_http_cookies(struct Curl_easy *data,
                           struct connectdata *conn,
                           struct dynbuf *r);
#else
#define Curl_http_cookies(a,b,c) CURLE_OK
#endif
CURLcode Curl_http_resume(struct Curl_easy *data,
                          struct connectdata *conn,
                          Curl_HttpReq httpreq);
CURLcode Curl_http_range(struct Curl_easy *data,
                         Curl_HttpReq httpreq);
CURLcode Curl_http_firstwrite(struct Curl_easy *data,
                              struct connectdata *conn,
                              bool *done);

/* protocol-specific functions set up to be called by the main engine */
CURLcode Curl_http(struct Curl_easy *data, bool *done);
CURLcode Curl_http_done(struct Curl_easy *data, CURLcode, bool premature);
CURLcode Curl_http_connect(struct Curl_easy *data, bool *done);

/* These functions are in http.c */
CURLcode Curl_http_input_auth(struct Curl_easy *data, bool proxy,
                              const char *auth);
CURLcode Curl_http_auth_act(struct Curl_easy *data);

/* If only the PICKNONE bit is set, there has been a round-trip and we
   selected to use no auth at all. Ie, we actively select no auth, as opposed
   to not having one selected. The other CURLAUTH_* defines are present in the
   public curl/curl.h header. */
#define CURLAUTH_PICKNONE (1<<30) /* don't use auth */

/* MAX_INITIAL_POST_SIZE indicates the number of bytes that will make the POST
   data get included in the initial data chunk sent to the server. If the
   data is larger than this, it will automatically get split up in multiple
   system calls.

   This value used to be fairly big (100K), but we must take into account that
   if the server rejects the POST due for authentication reasons, this data
   will always be unconditionally sent and thus it may not be larger than can
   always be afforded to send twice.

   It must not be greater than 64K to work on VMS.
*/
#ifndef MAX_INITIAL_POST_SIZE
#define MAX_INITIAL_POST_SIZE (64*1024)
#endif

/* EXPECT_100_THRESHOLD is the request body size limit for when libcurl will
 * automatically add an "Expect: 100-continue" header in HTTP requests. When
 * the size is unknown, it will always add it.
 *
 */
#ifndef EXPECT_100_THRESHOLD
#define EXPECT_100_THRESHOLD (1024*1024)
#endif

#endif /* CURL_DISABLE_HTTP */

#ifdef USE_NGHTTP3
struct h3out; /* see ngtcp2 */
#endif

/****************************************************************************
 * HTTP unique setup
 ***************************************************************************/
struct HTTP {
  curl_mimepart *sendit;
  curl_off_t postsize; /* off_t to handle large file sizes */
  const char *postdata;

  const char *p_pragma;      /* Pragma: string */

  /* For FORM posting */
  curl_mimepart form;

  struct back {
    curl_read_callback fread_func; /* backup storage for fread pointer */
    void *fread_in;           /* backup storage for fread_in pointer */
    const char *postdata;
    curl_off_t postsize;
  } backup;

  enum {
    HTTPSEND_NADA,    /* init */
    HTTPSEND_REQUEST, /* sending a request */
    HTTPSEND_BODY     /* sending body */
  } sending;

#ifndef CURL_DISABLE_HTTP
  struct dynbuf send_buffer; /* used if the request couldn't be sent in one
                                chunk, points to an allocated send_buffer
                                struct */
#endif
#ifdef USE_NGHTTP2
  /*********** for HTTP/2 we store stream-local data here *************/
  int32_t stream_id; /* stream we are interested in */

  bool bodystarted;
  /* We store non-final and final response headers here, per-stream */
  struct dynbuf header_recvbuf;
  size_t nread_header_recvbuf; /* number of bytes in header_recvbuf fed into
                                  upper layer */
  struct dynbuf trailer_recvbuf;
  int status_code; /* HTTP status code */
  const uint8_t *pausedata; /* pointer to data received in on_data_chunk */
  size_t pauselen; /* the number of bytes left in data */
  bool close_handled; /* TRUE if stream closure is handled by libcurl */

  char **push_headers;       /* allocated array */
  size_t push_headers_used;  /* number of entries filled in */
  size_t push_headers_alloc; /* number of entries allocated */
  uint32_t error; /* HTTP/2 stream error code */
#endif
#if defined(USE_NGHTTP2) || defined(USE_NGHTTP3)
  bool closed; /* TRUE on HTTP2 stream close */
  char *mem;     /* points to a buffer in memory to store received data */
  size_t len;    /* size of the buffer 'mem' points to */
  size_t memlen; /* size of data copied to mem */
#endif
#if defined(USE_NGHTTP2) || defined(ENABLE_QUIC)
  /* fields used by both HTTP/2 and HTTP/3 */
  const uint8_t *upload_mem; /* points to a buffer to read from */
  size_t upload_len; /* size of the buffer 'upload_mem' points to */
  curl_off_t upload_left; /* number of bytes left to upload */
#endif

#ifdef ENABLE_QUIC
  /*********** for HTTP/3 we store stream-local data here *************/
  int64_t stream3_id; /* stream we are interested in */
  bool firstheader;  /* FALSE until headers arrive */
  bool firstbody;  /* FALSE until body arrives */
  bool h3req;    /* FALSE until request is issued */
  bool upload_done;
#endif
#ifdef USE_NGHTTP3
  size_t unacked_window;
  struct h3out *h3out; /* per-stream buffers for upload */
  struct dynbuf overflow; /* excess data received during a single Curl_read */
#endif
};

#ifdef USE_NGHTTP2
/* h2 settings for this connection */
struct h2settings {
  uint32_t max_concurrent_streams;
  bool enable_push;
};
#endif

struct http_conn {
#ifdef USE_NGHTTP2
#define H2_BINSETTINGS_LEN 80
  uint8_t binsettings[H2_BINSETTINGS_LEN];
  size_t  binlen; /* length of the binsettings data */

  /* We associate the connnectdata struct with the connection, but we need to
     make sure we can identify the current "driving" transfer. This is a
     work-around for the lack of nghttp2_session_set_user_data() in older
     nghttp2 versions that we want to support. (Added in 1.31.0) */
  struct Curl_easy *trnsfr;

  nghttp2_session *h2;
  Curl_send *send_underlying; /* underlying send Curl_send callback */
  Curl_recv *recv_underlying; /* underlying recv Curl_recv callback */
  char *inbuf; /* buffer to receive data from underlying socket */
  size_t inbuflen; /* number of bytes filled in inbuf */
  size_t nread_inbuf; /* number of bytes read from in inbuf */
  /* We need separate buffer for transmission and reception because we
     may call nghttp2_session_send() after the
     nghttp2_session_mem_recv() but mem buffer is still not full. In
     this case, we wrongly sends the content of mem buffer if we share
     them for both cases. */
  int32_t pause_stream_id; /* stream ID which paused
                              nghttp2_session_mem_recv */
  size_t drain_total; /* sum of all stream's UrlState.drain */

  /* this is a hash of all individual streams (Curl_easy structs) */
  struct h2settings settings;

  /* list of settings that will be sent */
  nghttp2_settings_entry local_settings[3];
  size_t local_settings_num;
#else
  int unused; /* prevent a compiler warning */
#endif
};

CURLcode Curl_http_size(struct Curl_easy *data);

CURLcode Curl_http_readwrite_headers(struct Curl_easy *data,
                                     struct connectdata *conn,
                                     ssize_t *nread,
                                     bool *stop_reading);

/**
 * Curl_http_output_auth() setups the authentication headers for the
 * host/proxy and the correct authentication
 * method. data->state.authdone is set to TRUE when authentication is
 * done.
 *
 * @param data all information about the current transfer
 * @param conn all information about the current connection
 * @param request pointer to the request keyword
 * @param httpreq is the request type
 * @param path pointer to the requested path
 * @param proxytunnel boolean if this is the request setting up a "proxy
 * tunnel"
 *
 * @returns CURLcode
 */
CURLcode
Curl_http_output_auth(struct Curl_easy *data,
                      struct connectdata *conn,
                      const char *request,
                      Curl_HttpReq httpreq,
                      const char *path,
                      bool proxytunnel); /* TRUE if this is the request setting
                                            up the proxy tunnel */

#endif /* HEADER_CURL_HTTP_H */
