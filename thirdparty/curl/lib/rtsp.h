#ifndef HEADER_CURL_RTSP_H
#define HEADER_CURL_RTSP_H
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
#ifdef USE_HYPER
#define CURL_DISABLE_RTSP 1
#endif

#ifndef CURL_DISABLE_RTSP

extern const struct Curl_handler Curl_handler_rtsp;

CURLcode Curl_rtsp_parseheader(struct Curl_easy *data, char *header);

#else
/* disabled */
#define Curl_rtsp_parseheader(x,y) CURLE_NOT_BUILT_IN

#endif /* CURL_DISABLE_RTSP */

/*
 * RTSP Connection data
 *
 * Currently, only used for tracking incomplete RTP data reads
 */
struct rtsp_conn {
  char *rtp_buf;
  ssize_t rtp_bufsize;
  int rtp_channel;
};

/****************************************************************************
 * RTSP unique setup
 ***************************************************************************/
struct RTSP {
  /*
   * http_wrapper MUST be the first element of this structure for the wrap
   * logic to work. In this way, we get a cheap polymorphism because
   * &(data->state.proto.rtsp) == &(data->state.proto.http) per the C spec
   *
   * HTTP functions can safely treat this as an HTTP struct, but RTSP aware
   * functions can also index into the later elements.
   */
  struct HTTP http_wrapper; /*wrap HTTP to do the heavy lifting */

  long CSeq_sent; /* CSeq of this request */
  long CSeq_recv; /* CSeq received */
};


#endif /* HEADER_CURL_RTSP_H */
