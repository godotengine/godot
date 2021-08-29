/***************************************************************************
 *                      _   _ ____  _
 *  Project         ___| | | |  _ \| |
 *                 / __| | | | |_) | |
 *                | (__| |_| |  _ <| |___
 *                 \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2012 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 * Copyright (C) 2010, Howard Chu, <hyc@highlandsun.com>
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

#ifdef USE_LIBRTMP

#include "curl_rtmp.h"
#include "urldata.h"
#include "nonblock.h" /* for curlx_nonblock */
#include "progress.h" /* for Curl_pgrsSetUploadSize */
#include "transfer.h"
#include "warnless.h"
#include <curl/curl.h>
#include <librtmp/rtmp.h>
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

#if defined(WIN32) && !defined(USE_LWIPSOCK)
#define setsockopt(a,b,c,d,e) (setsockopt)(a,b,c,(const char *)d,(int)e)
#define SET_RCVTIMEO(tv,s)   int tv = s*1000
#elif defined(LWIP_SO_SNDRCVTIMEO_NONSTANDARD)
#define SET_RCVTIMEO(tv,s)   int tv = s*1000
#else
#define SET_RCVTIMEO(tv,s)   struct timeval tv = {s,0}
#endif

#define DEF_BUFTIME    (2*60*60*1000)    /* 2 hours */

static CURLcode rtmp_setup_connection(struct Curl_easy *data,
                                      struct connectdata *conn);
static CURLcode rtmp_do(struct Curl_easy *data, bool *done);
static CURLcode rtmp_done(struct Curl_easy *data, CURLcode, bool premature);
static CURLcode rtmp_connect(struct Curl_easy *data, bool *done);
static CURLcode rtmp_disconnect(struct Curl_easy *data,
                                struct connectdata *conn, bool dead);

static Curl_recv rtmp_recv;
static Curl_send rtmp_send;

/*
 * RTMP protocol handler.h, based on https://rtmpdump.mplayerhq.hu
 */

const struct Curl_handler Curl_handler_rtmp = {
  "RTMP",                               /* scheme */
  rtmp_setup_connection,                /* setup_connection */
  rtmp_do,                              /* do_it */
  rtmp_done,                            /* done */
  ZERO_NULL,                            /* do_more */
  rtmp_connect,                         /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  rtmp_disconnect,                      /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_RTMP,                            /* defport */
  CURLPROTO_RTMP,                       /* protocol */
  CURLPROTO_RTMP,                       /* family */
  PROTOPT_NONE                          /* flags*/
};

const struct Curl_handler Curl_handler_rtmpt = {
  "RTMPT",                              /* scheme */
  rtmp_setup_connection,                /* setup_connection */
  rtmp_do,                              /* do_it */
  rtmp_done,                            /* done */
  ZERO_NULL,                            /* do_more */
  rtmp_connect,                         /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  rtmp_disconnect,                      /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_RTMPT,                           /* defport */
  CURLPROTO_RTMPT,                      /* protocol */
  CURLPROTO_RTMPT,                      /* family */
  PROTOPT_NONE                          /* flags*/
};

const struct Curl_handler Curl_handler_rtmpe = {
  "RTMPE",                              /* scheme */
  rtmp_setup_connection,                /* setup_connection */
  rtmp_do,                              /* do_it */
  rtmp_done,                            /* done */
  ZERO_NULL,                            /* do_more */
  rtmp_connect,                         /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  rtmp_disconnect,                      /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_RTMP,                            /* defport */
  CURLPROTO_RTMPE,                      /* protocol */
  CURLPROTO_RTMPE,                      /* family */
  PROTOPT_NONE                          /* flags*/
};

const struct Curl_handler Curl_handler_rtmpte = {
  "RTMPTE",                             /* scheme */
  rtmp_setup_connection,                /* setup_connection */
  rtmp_do,                              /* do_it */
  rtmp_done,                            /* done */
  ZERO_NULL,                            /* do_more */
  rtmp_connect,                         /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  rtmp_disconnect,                      /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_RTMPT,                           /* defport */
  CURLPROTO_RTMPTE,                     /* protocol */
  CURLPROTO_RTMPTE,                     /* family */
  PROTOPT_NONE                          /* flags*/
};

const struct Curl_handler Curl_handler_rtmps = {
  "RTMPS",                              /* scheme */
  rtmp_setup_connection,                /* setup_connection */
  rtmp_do,                              /* do_it */
  rtmp_done,                            /* done */
  ZERO_NULL,                            /* do_more */
  rtmp_connect,                         /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  rtmp_disconnect,                      /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_RTMPS,                           /* defport */
  CURLPROTO_RTMPS,                      /* protocol */
  CURLPROTO_RTMP,                       /* family */
  PROTOPT_NONE                          /* flags*/
};

const struct Curl_handler Curl_handler_rtmpts = {
  "RTMPTS",                             /* scheme */
  rtmp_setup_connection,                /* setup_connection */
  rtmp_do,                              /* do_it */
  rtmp_done,                            /* done */
  ZERO_NULL,                            /* do_more */
  rtmp_connect,                         /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  rtmp_disconnect,                      /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_RTMPS,                           /* defport */
  CURLPROTO_RTMPTS,                     /* protocol */
  CURLPROTO_RTMPT,                      /* family */
  PROTOPT_NONE                          /* flags*/
};

static CURLcode rtmp_setup_connection(struct Curl_easy *data,
                                      struct connectdata *conn)
{
  RTMP *r = RTMP_Alloc();
  if(!r)
    return CURLE_OUT_OF_MEMORY;

  RTMP_Init(r);
  RTMP_SetBufferMS(r, DEF_BUFTIME);
  if(!RTMP_SetupURL(r, data->state.url)) {
    RTMP_Free(r);
    return CURLE_URL_MALFORMAT;
  }
  conn->proto.rtmp = r;
  return CURLE_OK;
}

static CURLcode rtmp_connect(struct Curl_easy *data, bool *done)
{
  struct connectdata *conn = data->conn;
  RTMP *r = conn->proto.rtmp;
  SET_RCVTIMEO(tv, 10);

  r->m_sb.sb_socket = (int)conn->sock[FIRSTSOCKET];

  /* We have to know if it's a write before we send the
   * connect request packet
   */
  if(data->set.upload)
    r->Link.protocol |= RTMP_FEATURE_WRITE;

  /* For plain streams, use the buffer toggle trick to keep data flowing */
  if(!(r->Link.lFlags & RTMP_LF_LIVE) &&
     !(r->Link.protocol & RTMP_FEATURE_HTTP))
    r->Link.lFlags |= RTMP_LF_BUFX;

  (void)curlx_nonblock(r->m_sb.sb_socket, FALSE);
  setsockopt(r->m_sb.sb_socket, SOL_SOCKET, SO_RCVTIMEO,
             (char *)&tv, sizeof(tv));

  if(!RTMP_Connect1(r, NULL))
    return CURLE_FAILED_INIT;

  /* Clients must send a periodic BytesReceived report to the server */
  r->m_bSendCounter = true;

  *done = TRUE;
  conn->recv[FIRSTSOCKET] = rtmp_recv;
  conn->send[FIRSTSOCKET] = rtmp_send;
  return CURLE_OK;
}

static CURLcode rtmp_do(struct Curl_easy *data, bool *done)
{
  struct connectdata *conn = data->conn;
  RTMP *r = conn->proto.rtmp;

  if(!RTMP_ConnectStream(r, 0))
    return CURLE_FAILED_INIT;

  if(data->set.upload) {
    Curl_pgrsSetUploadSize(data, data->state.infilesize);
    Curl_setup_transfer(data, -1, -1, FALSE, FIRSTSOCKET);
  }
  else
    Curl_setup_transfer(data, FIRSTSOCKET, -1, FALSE, -1);
  *done = TRUE;
  return CURLE_OK;
}

static CURLcode rtmp_done(struct Curl_easy *data, CURLcode status,
                          bool premature)
{
  (void)data; /* unused */
  (void)status; /* unused */
  (void)premature; /* unused */

  return CURLE_OK;
}

static CURLcode rtmp_disconnect(struct Curl_easy *data,
                                struct connectdata *conn,
                                bool dead_connection)
{
  RTMP *r = conn->proto.rtmp;
  (void)data;
  (void)dead_connection;
  if(r) {
    conn->proto.rtmp = NULL;
    RTMP_Close(r);
    RTMP_Free(r);
  }
  return CURLE_OK;
}

static ssize_t rtmp_recv(struct Curl_easy *data, int sockindex, char *buf,
                         size_t len, CURLcode *err)
{
  struct connectdata *conn = data->conn;
  RTMP *r = conn->proto.rtmp;
  ssize_t nread;

  (void)sockindex; /* unused */

  nread = RTMP_Read(r, buf, curlx_uztosi(len));
  if(nread < 0) {
    if(r->m_read.status == RTMP_READ_COMPLETE ||
       r->m_read.status == RTMP_READ_EOF) {
      data->req.size = data->req.bytecount;
      nread = 0;
    }
    else
      *err = CURLE_RECV_ERROR;
  }
  return nread;
}

static ssize_t rtmp_send(struct Curl_easy *data, int sockindex,
                         const void *buf, size_t len, CURLcode *err)
{
  struct connectdata *conn = data->conn;
  RTMP *r = conn->proto.rtmp;
  ssize_t num;

  (void)sockindex; /* unused */

  num = RTMP_Write(r, (char *)buf, curlx_uztosi(len));
  if(num < 0)
    *err = CURLE_SEND_ERROR;

  return num;
}
#endif  /* USE_LIBRTMP */
