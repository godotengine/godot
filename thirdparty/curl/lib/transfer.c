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
#include "strtoofft.h"

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_NET_IF_H
#include <net/if.h>
#endif
#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif
#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#elif defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif

#ifndef HAVE_SOCKET
#error "We can't compile without socket() support!"
#endif

#include "urldata.h"
#include <curl/curl.h>
#include "netrc.h"

#include "content_encoding.h"
#include "hostip.h"
#include "transfer.h"
#include "sendf.h"
#include "speedcheck.h"
#include "progress.h"
#include "http.h"
#include "url.h"
#include "getinfo.h"
#include "vtls/vtls.h"
#include "select.h"
#include "multiif.h"
#include "connect.h"
#include "non-ascii.h"
#include "http2.h"
#include "mime.h"
#include "strcase.h"
#include "urlapi-int.h"
#include "hsts.h"
#include "setopt.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#if !defined(CURL_DISABLE_HTTP) || !defined(CURL_DISABLE_SMTP) || \
    !defined(CURL_DISABLE_IMAP)
/*
 * checkheaders() checks the linked list of custom headers for a
 * particular header (prefix). Provide the prefix without colon!
 *
 * Returns a pointer to the first matching header or NULL if none matched.
 */
char *Curl_checkheaders(const struct Curl_easy *data,
                        const char *thisheader)
{
  struct curl_slist *head;
  size_t thislen = strlen(thisheader);
  DEBUGASSERT(thislen);
  DEBUGASSERT(thisheader[thislen-1] != ':');

  for(head = data->set.headers; head; head = head->next) {
    if(strncasecompare(head->data, thisheader, thislen) &&
       Curl_headersep(head->data[thislen]) )
      return head->data;
  }

  return NULL;
}
#endif

CURLcode Curl_get_upload_buffer(struct Curl_easy *data)
{
  if(!data->state.ulbuf) {
    data->state.ulbuf = malloc(data->set.upload_buffer_size);
    if(!data->state.ulbuf)
      return CURLE_OUT_OF_MEMORY;
  }
  return CURLE_OK;
}

#ifndef CURL_DISABLE_HTTP
/*
 * This function will be called to loop through the trailers buffer
 * until no more data is available for sending.
 */
static size_t trailers_read(char *buffer, size_t size, size_t nitems,
                            void *raw)
{
  struct Curl_easy *data = (struct Curl_easy *)raw;
  struct dynbuf *trailers_buf = &data->state.trailers_buf;
  size_t bytes_left = Curl_dyn_len(trailers_buf) -
    data->state.trailers_bytes_sent;
  size_t to_copy = (size*nitems < bytes_left) ? size*nitems : bytes_left;
  if(to_copy) {
    memcpy(buffer,
           Curl_dyn_ptr(trailers_buf) + data->state.trailers_bytes_sent,
           to_copy);
    data->state.trailers_bytes_sent += to_copy;
  }
  return to_copy;
}

static size_t trailers_left(void *raw)
{
  struct Curl_easy *data = (struct Curl_easy *)raw;
  struct dynbuf *trailers_buf = &data->state.trailers_buf;
  return Curl_dyn_len(trailers_buf) - data->state.trailers_bytes_sent;
}
#endif

/*
 * This function will call the read callback to fill our buffer with data
 * to upload.
 */
CURLcode Curl_fillreadbuffer(struct Curl_easy *data, size_t bytes,
                             size_t *nreadp)
{
  size_t buffersize = bytes;
  size_t nread;

  curl_read_callback readfunc = NULL;
  void *extra_data = NULL;

#ifdef CURL_DOES_CONVERSIONS
  bool sending_http_headers = FALSE;
  struct connectdata *conn = data->conn;

  if(conn->handler->protocol&(PROTO_FAMILY_HTTP|CURLPROTO_RTSP)) {
    const struct HTTP *http = data->req.p.http;

    if(http->sending == HTTPSEND_REQUEST)
      /* We're sending the HTTP request headers, not the data.
         Remember that so we don't re-translate them into garbage. */
      sending_http_headers = TRUE;
  }
#endif

#ifndef CURL_DISABLE_HTTP
  if(data->state.trailers_state == TRAILERS_INITIALIZED) {
    struct curl_slist *trailers = NULL;
    CURLcode result;
    int trailers_ret_code;

    /* at this point we already verified that the callback exists
       so we compile and store the trailers buffer, then proceed */
    infof(data,
          "Moving trailers state machine from initialized to sending.");
    data->state.trailers_state = TRAILERS_SENDING;
    Curl_dyn_init(&data->state.trailers_buf, DYN_TRAILERS);

    data->state.trailers_bytes_sent = 0;
    Curl_set_in_callback(data, true);
    trailers_ret_code = data->set.trailer_callback(&trailers,
                                                   data->set.trailer_data);
    Curl_set_in_callback(data, false);
    if(trailers_ret_code == CURL_TRAILERFUNC_OK) {
      result = Curl_http_compile_trailers(trailers, &data->state.trailers_buf,
                                          data);
    }
    else {
      failf(data, "operation aborted by trailing headers callback");
      *nreadp = 0;
      result = CURLE_ABORTED_BY_CALLBACK;
    }
    if(result) {
      Curl_dyn_free(&data->state.trailers_buf);
      curl_slist_free_all(trailers);
      return result;
    }
    infof(data, "Successfully compiled trailers.");
    curl_slist_free_all(trailers);
  }
#endif

  /* if we are transmitting trailing data, we don't need to write
     a chunk size so we skip this */
  if(data->req.upload_chunky &&
     data->state.trailers_state == TRAILERS_NONE) {
    /* if chunked Transfer-Encoding */
    buffersize -= (8 + 2 + 2);   /* 32bit hex + CRLF + CRLF */
    data->req.upload_fromhere += (8 + 2); /* 32bit hex + CRLF */
  }

#ifndef CURL_DISABLE_HTTP
  if(data->state.trailers_state == TRAILERS_SENDING) {
    /* if we're here then that means that we already sent the last empty chunk
       but we didn't send a final CR LF, so we sent 0 CR LF. We then start
       pulling trailing data until we have no more at which point we
       simply return to the previous point in the state machine as if
       nothing happened.
       */
    readfunc = trailers_read;
    extra_data = (void *)data;
  }
  else
#endif
  {
    readfunc = data->state.fread_func;
    extra_data = data->state.in;
  }

  Curl_set_in_callback(data, true);
  nread = readfunc(data->req.upload_fromhere, 1,
                   buffersize, extra_data);
  Curl_set_in_callback(data, false);

  if(nread == CURL_READFUNC_ABORT) {
    failf(data, "operation aborted by callback");
    *nreadp = 0;
    return CURLE_ABORTED_BY_CALLBACK;
  }
  if(nread == CURL_READFUNC_PAUSE) {
    struct SingleRequest *k = &data->req;

    if(data->conn->handler->flags & PROTOPT_NONETWORK) {
      /* protocols that work without network cannot be paused. This is
         actually only FILE:// just now, and it can't pause since the transfer
         isn't done using the "normal" procedure. */
      failf(data, "Read callback asked for PAUSE when not supported!");
      return CURLE_READ_ERROR;
    }

    /* CURL_READFUNC_PAUSE pauses read callbacks that feed socket writes */
    k->keepon |= KEEP_SEND_PAUSE; /* mark socket send as paused */
    if(data->req.upload_chunky) {
        /* Back out the preallocation done above */
      data->req.upload_fromhere -= (8 + 2);
    }
    *nreadp = 0;

    return CURLE_OK; /* nothing was read */
  }
  else if(nread > buffersize) {
    /* the read function returned a too large value */
    *nreadp = 0;
    failf(data, "read function returned funny value");
    return CURLE_READ_ERROR;
  }

  if(!data->req.forbidchunk && data->req.upload_chunky) {
    /* if chunked Transfer-Encoding
     *    build chunk:
     *
     *        <HEX SIZE> CRLF
     *        <DATA> CRLF
     */
    /* On non-ASCII platforms the <DATA> may or may not be
       translated based on state.prefer_ascii while the protocol
       portion must always be translated to the network encoding.
       To further complicate matters, line end conversion might be
       done later on, so we need to prevent CRLFs from becoming
       CRCRLFs if that's the case.  To do this we use bare LFs
       here, knowing they'll become CRLFs later on.
     */

    bool added_crlf = FALSE;
    int hexlen = 0;
    const char *endofline_native;
    const char *endofline_network;

    if(
#ifdef CURL_DO_LINEEND_CONV
       (data->state.prefer_ascii) ||
#endif
       (data->set.crlf)) {
      /* \n will become \r\n later on */
      endofline_native  = "\n";
      endofline_network = "\x0a";
    }
    else {
      endofline_native  = "\r\n";
      endofline_network = "\x0d\x0a";
    }

    /* if we're not handling trailing data, proceed as usual */
    if(data->state.trailers_state != TRAILERS_SENDING) {
      char hexbuffer[11] = "";
      hexlen = msnprintf(hexbuffer, sizeof(hexbuffer),
                         "%zx%s", nread, endofline_native);

      /* move buffer pointer */
      data->req.upload_fromhere -= hexlen;
      nread += hexlen;

      /* copy the prefix to the buffer, leaving out the NUL */
      memcpy(data->req.upload_fromhere, hexbuffer, hexlen);

      /* always append ASCII CRLF to the data unless
         we have a valid trailer callback */
#ifndef CURL_DISABLE_HTTP
      if((nread-hexlen) == 0 &&
          data->set.trailer_callback != NULL &&
          data->state.trailers_state == TRAILERS_NONE) {
        data->state.trailers_state = TRAILERS_INITIALIZED;
      }
      else
#endif
      {
        memcpy(data->req.upload_fromhere + nread,
               endofline_network,
               strlen(endofline_network));
        added_crlf = TRUE;
      }
    }

#ifdef CURL_DOES_CONVERSIONS
    {
      CURLcode result;
      size_t length;
      if(data->state.prefer_ascii)
        /* translate the protocol and data */
        length = nread;
      else
        /* just translate the protocol portion */
        length = hexlen;
      if(length) {
        result = Curl_convert_to_network(data, data->req.upload_fromhere,
                                         length);
        /* Curl_convert_to_network calls failf if unsuccessful */
        if(result)
          return result;
      }
    }
#endif /* CURL_DOES_CONVERSIONS */

#ifndef CURL_DISABLE_HTTP
    if(data->state.trailers_state == TRAILERS_SENDING &&
       !trailers_left(data)) {
      Curl_dyn_free(&data->state.trailers_buf);
      data->state.trailers_state = TRAILERS_DONE;
      data->set.trailer_data = NULL;
      data->set.trailer_callback = NULL;
      /* mark the transfer as done */
      data->req.upload_done = TRUE;
      infof(data, "Signaling end of chunked upload after trailers.");
    }
    else
#endif
      if((nread - hexlen) == 0 &&
         data->state.trailers_state != TRAILERS_INITIALIZED) {
        /* mark this as done once this chunk is transferred */
        data->req.upload_done = TRUE;
        infof(data,
              "Signaling end of chunked upload via terminating chunk.");
      }

    if(added_crlf)
      nread += strlen(endofline_network); /* for the added end of line */
  }
#ifdef CURL_DOES_CONVERSIONS
  else if((data->state.prefer_ascii) && (!sending_http_headers)) {
    CURLcode result;
    result = Curl_convert_to_network(data, data->req.upload_fromhere, nread);
    /* Curl_convert_to_network calls failf if unsuccessful */
    if(result)
      return result;
  }
#endif /* CURL_DOES_CONVERSIONS */

  *nreadp = nread;

  return CURLE_OK;
}


/*
 * Curl_readrewind() rewinds the read stream. This is typically used for HTTP
 * POST/PUT with multi-pass authentication when a sending was denied and a
 * resend is necessary.
 */
CURLcode Curl_readrewind(struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;
  curl_mimepart *mimepart = &data->set.mimepost;

  conn->bits.rewindaftersend = FALSE; /* we rewind now */

  /* explicitly switch off sending data on this connection now since we are
     about to restart a new transfer and thus we want to avoid inadvertently
     sending more data on the existing connection until the next transfer
     starts */
  data->req.keepon &= ~KEEP_SEND;

  /* We have sent away data. If not using CURLOPT_POSTFIELDS or
     CURLOPT_HTTPPOST, call app to rewind
  */
  if(conn->handler->protocol & PROTO_FAMILY_HTTP) {
    struct HTTP *http = data->req.p.http;

    if(http->sendit)
      mimepart = http->sendit;
  }
  if(data->set.postfields)
    ; /* do nothing */
  else if(data->state.httpreq == HTTPREQ_POST_MIME ||
          data->state.httpreq == HTTPREQ_POST_FORM) {
    CURLcode result = Curl_mime_rewind(mimepart);
    if(result) {
      failf(data, "Cannot rewind mime/post data");
      return result;
    }
  }
  else {
    if(data->set.seek_func) {
      int err;

      Curl_set_in_callback(data, true);
      err = (data->set.seek_func)(data->set.seek_client, 0, SEEK_SET);
      Curl_set_in_callback(data, false);
      if(err) {
        failf(data, "seek callback returned error %d", (int)err);
        return CURLE_SEND_FAIL_REWIND;
      }
    }
    else if(data->set.ioctl_func) {
      curlioerr err;

      Curl_set_in_callback(data, true);
      err = (data->set.ioctl_func)(data, CURLIOCMD_RESTARTREAD,
                                   data->set.ioctl_client);
      Curl_set_in_callback(data, false);
      infof(data, "the ioctl callback returned %d", (int)err);

      if(err) {
        failf(data, "ioctl callback returned error %d", (int)err);
        return CURLE_SEND_FAIL_REWIND;
      }
    }
    else {
      /* If no CURLOPT_READFUNCTION is used, we know that we operate on a
         given FILE * stream and we can actually attempt to rewind that
         ourselves with fseek() */
      if(data->state.fread_func == (curl_read_callback)fread) {
        if(-1 != fseek(data->state.in, 0, SEEK_SET))
          /* successful rewind */
          return CURLE_OK;
      }

      /* no callback set or failure above, makes us fail at once */
      failf(data, "necessary data rewind wasn't possible");
      return CURLE_SEND_FAIL_REWIND;
    }
  }
  return CURLE_OK;
}

static int data_pending(const struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;

#ifdef ENABLE_QUIC
  if(conn->transport == TRNSPRT_QUIC)
    return Curl_quic_data_pending(data);
#endif

  if(conn->handler->protocol&PROTO_FAMILY_FTP)
    return Curl_ssl_data_pending(conn, SECONDARYSOCKET);

  /* in the case of libssh2, we can never be really sure that we have emptied
     its internal buffers so we MUST always try until we get EAGAIN back */
  return conn->handler->protocol&(CURLPROTO_SCP|CURLPROTO_SFTP) ||
#if defined(USE_NGHTTP2)
    /* For HTTP/2, we may read up everything including response body
       with header fields in Curl_http_readwrite_headers. If no
       content-length is provided, curl waits for the connection
       close, which we emulate it using conn->proto.httpc.closed =
       TRUE. The thing is if we read everything, then http2_recv won't
       be called and we cannot signal the HTTP/2 stream has closed. As
       a workaround, we return nonzero here to call http2_recv. */
    ((conn->handler->protocol&PROTO_FAMILY_HTTP) && conn->httpversion >= 20) ||
#endif
    Curl_ssl_data_pending(conn, FIRSTSOCKET);
}

/*
 * Check to see if CURLOPT_TIMECONDITION was met by comparing the time of the
 * remote document with the time provided by CURLOPT_TIMEVAL
 */
bool Curl_meets_timecondition(struct Curl_easy *data, time_t timeofdoc)
{
  if((timeofdoc == 0) || (data->set.timevalue == 0))
    return TRUE;

  switch(data->set.timecondition) {
  case CURL_TIMECOND_IFMODSINCE:
  default:
    if(timeofdoc <= data->set.timevalue) {
      infof(data,
            "The requested document is not new enough");
      data->info.timecond = TRUE;
      return FALSE;
    }
    break;
  case CURL_TIMECOND_IFUNMODSINCE:
    if(timeofdoc >= data->set.timevalue) {
      infof(data,
            "The requested document is not old enough");
      data->info.timecond = TRUE;
      return FALSE;
    }
    break;
  }

  return TRUE;
}

/*
 * Go ahead and do a read if we have a readable socket or if
 * the stream was rewound (in which case we have data in a
 * buffer)
 *
 * return '*comeback' TRUE if we didn't properly drain the socket so this
 * function should get called again without select() or similar in between!
 */
static CURLcode readwrite_data(struct Curl_easy *data,
                               struct connectdata *conn,
                               struct SingleRequest *k,
                               int *didwhat, bool *done,
                               bool *comeback)
{
  CURLcode result = CURLE_OK;
  ssize_t nread; /* number of bytes read */
  size_t excess = 0; /* excess bytes read */
  bool readmore = FALSE; /* used by RTP to signal for more data */
  int maxloops = 100;
  char *buf = data->state.buffer;
  DEBUGASSERT(buf);

  *done = FALSE;
  *comeback = FALSE;

  /* This is where we loop until we have read everything there is to
     read or we get a CURLE_AGAIN */
  do {
    bool is_empty_data = FALSE;
    size_t buffersize = data->set.buffer_size;
    size_t bytestoread = buffersize;
#ifdef USE_NGHTTP2
    bool is_http2 = ((conn->handler->protocol & PROTO_FAMILY_HTTP) &&
                     (conn->httpversion == 20));
#endif

    if(
#ifdef USE_NGHTTP2
       /* For HTTP/2, read data without caring about the content
          length. This is safe because body in HTTP/2 is always
          segmented thanks to its framing layer. Meanwhile, we have to
          call Curl_read to ensure that http2_handle_stream_close is
          called when we read all incoming bytes for a particular
          stream. */
       !is_http2 &&
#endif
       k->size != -1 && !k->header) {
      /* make sure we don't read too much */
      curl_off_t totalleft = k->size - k->bytecount;
      if(totalleft < (curl_off_t)bytestoread)
        bytestoread = (size_t)totalleft;
    }

    if(bytestoread) {
      /* receive data from the network! */
      result = Curl_read(data, conn->sockfd, buf, bytestoread, &nread);

      /* read would've blocked */
      if(CURLE_AGAIN == result)
        break; /* get out of loop */

      if(result>0)
        return result;
    }
    else {
      /* read nothing but since we wanted nothing we consider this an OK
         situation to proceed from */
      DEBUGF(infof(data, "readwrite_data: we're done!"));
      nread = 0;
    }

    if(!k->bytecount) {
      Curl_pgrsTime(data, TIMER_STARTTRANSFER);
      if(k->exp100 > EXP100_SEND_DATA)
        /* set time stamp to compare with when waiting for the 100 */
        k->start100 = Curl_now();
    }

    *didwhat |= KEEP_RECV;
    /* indicates data of zero size, i.e. empty file */
    is_empty_data = ((nread == 0) && (k->bodywrites == 0)) ? TRUE : FALSE;

    if(0 < nread || is_empty_data) {
      buf[nread] = 0;
    }
    else {
      /* if we receive 0 or less here, either the http2 stream is closed or the
         server closed the connection and we bail out from this! */
#ifdef USE_NGHTTP2
      if(is_http2 && !nread)
        DEBUGF(infof(data, "nread == 0, stream closed, bailing"));
      else
#endif
        DEBUGF(infof(data, "nread <= 0, server closed connection, bailing"));
      k->keepon &= ~KEEP_RECV;
      break;
    }

    /* Default buffer to use when we write the buffer, it may be changed
       in the flow below before the actual storing is done. */
    k->str = buf;

    if(conn->handler->readwrite) {
      result = conn->handler->readwrite(data, conn, &nread, &readmore);
      if(result)
        return result;
      if(readmore)
        break;
    }

#ifndef CURL_DISABLE_HTTP
    /* Since this is a two-state thing, we check if we are parsing
       headers at the moment or not. */
    if(k->header) {
      /* we are in parse-the-header-mode */
      bool stop_reading = FALSE;
      result = Curl_http_readwrite_headers(data, conn, &nread, &stop_reading);
      if(result)
        return result;

      if(conn->handler->readwrite &&
         (k->maxdownload <= 0 && nread > 0)) {
        result = conn->handler->readwrite(data, conn, &nread, &readmore);
        if(result)
          return result;
        if(readmore)
          break;
      }

      if(stop_reading) {
        /* We've stopped dealing with input, get out of the do-while loop */

        if(nread > 0) {
          infof(data,
                "Excess found:"
                " excess = %zd"
                " url = %s (zero-length body)",
                nread, data->state.up.path);
        }

        break;
      }
    }
#endif /* CURL_DISABLE_HTTP */


    /* This is not an 'else if' since it may be a rest from the header
       parsing, where the beginning of the buffer is headers and the end
       is non-headers. */
    if(!k->header && (nread > 0 || is_empty_data)) {

      if(data->set.opt_no_body) {
        /* data arrives although we want none, bail out */
        streamclose(conn, "ignoring body");
        *done = TRUE;
        return CURLE_WEIRD_SERVER_REPLY;
      }

#ifndef CURL_DISABLE_HTTP
      if(0 == k->bodywrites && !is_empty_data) {
        /* These checks are only made the first time we are about to
           write a piece of the body */
        if(conn->handler->protocol&(PROTO_FAMILY_HTTP|CURLPROTO_RTSP)) {
          /* HTTP-only checks */
          result = Curl_http_firstwrite(data, conn, done);
          if(result || *done)
            return result;
        }
      } /* this is the first time we write a body part */
#endif /* CURL_DISABLE_HTTP */

      k->bodywrites++;

      /* pass data to the debug function before it gets "dechunked" */
      if(data->set.verbose) {
        if(k->badheader) {
          Curl_debug(data, CURLINFO_DATA_IN,
                     Curl_dyn_ptr(&data->state.headerb),
                     Curl_dyn_len(&data->state.headerb));
          if(k->badheader == HEADER_PARTHEADER)
            Curl_debug(data, CURLINFO_DATA_IN,
                       k->str, (size_t)nread);
        }
        else
          Curl_debug(data, CURLINFO_DATA_IN,
                     k->str, (size_t)nread);
      }

#ifndef CURL_DISABLE_HTTP
      if(k->chunk) {
        /*
         * Here comes a chunked transfer flying and we need to decode this
         * properly.  While the name says read, this function both reads
         * and writes away the data. The returned 'nread' holds the number
         * of actual data it wrote to the client.
         */
        CURLcode extra;
        CHUNKcode res =
          Curl_httpchunk_read(data, k->str, nread, &nread, &extra);

        if(CHUNKE_OK < res) {
          if(CHUNKE_PASSTHRU_ERROR == res) {
            failf(data, "Failed reading the chunked-encoded stream");
            return extra;
          }
          failf(data, "%s in chunked-encoding", Curl_chunked_strerror(res));
          return CURLE_RECV_ERROR;
        }
        if(CHUNKE_STOP == res) {
          /* we're done reading chunks! */
          k->keepon &= ~KEEP_RECV; /* read no more */

          /* N number of bytes at the end of the str buffer that weren't
             written to the client. */
          if(conn->chunk.datasize) {
            infof(data, "Leftovers after chunking: % "
                  CURL_FORMAT_CURL_OFF_T "u bytes",
                  conn->chunk.datasize);
          }
        }
        /* If it returned OK, we just keep going */
      }
#endif   /* CURL_DISABLE_HTTP */

      /* Account for body content stored in the header buffer */
      if((k->badheader == HEADER_PARTHEADER) && !k->ignorebody) {
        size_t headlen = Curl_dyn_len(&data->state.headerb);
        DEBUGF(infof(data, "Increasing bytecount by %zu", headlen));
        k->bytecount += headlen;
      }

      if((-1 != k->maxdownload) &&
         (k->bytecount + nread >= k->maxdownload)) {

        excess = (size_t)(k->bytecount + nread - k->maxdownload);
        if(excess > 0 && !k->ignorebody) {
          infof(data,
                "Excess found in a read:"
                " excess = %zu"
                ", size = %" CURL_FORMAT_CURL_OFF_T
                ", maxdownload = %" CURL_FORMAT_CURL_OFF_T
                ", bytecount = %" CURL_FORMAT_CURL_OFF_T,
                excess, k->size, k->maxdownload, k->bytecount);
          connclose(conn, "excess found in a read");
        }

        nread = (ssize_t) (k->maxdownload - k->bytecount);
        if(nread < 0) /* this should be unusual */
          nread = 0;

        k->keepon &= ~KEEP_RECV; /* we're done reading */
      }

      k->bytecount += nread;

      Curl_pgrsSetDownloadCounter(data, k->bytecount);

      if(!k->chunk && (nread || k->badheader || is_empty_data)) {
        /* If this is chunky transfer, it was already written */

        if(k->badheader && !k->ignorebody) {
          /* we parsed a piece of data wrongly assuming it was a header
             and now we output it as body instead */
          size_t headlen = Curl_dyn_len(&data->state.headerb);

          /* Don't let excess data pollute body writes */
          if(k->maxdownload == -1 || (curl_off_t)headlen <= k->maxdownload)
            result = Curl_client_write(data, CLIENTWRITE_BODY,
                                       Curl_dyn_ptr(&data->state.headerb),
                                       headlen);
          else
            result = Curl_client_write(data, CLIENTWRITE_BODY,
                                       Curl_dyn_ptr(&data->state.headerb),
                                       (size_t)k->maxdownload);

          if(result)
            return result;
        }
        if(k->badheader < HEADER_ALLBAD) {
          /* This switch handles various content encodings. If there's an
             error here, be sure to check over the almost identical code
             in http_chunks.c.
             Make sure that ALL_CONTENT_ENCODINGS contains all the
             encodings handled here. */
          if(data->set.http_ce_skip || !k->writer_stack) {
            if(!k->ignorebody && nread) {
#ifndef CURL_DISABLE_POP3
              if(conn->handler->protocol & PROTO_FAMILY_POP3)
                result = Curl_pop3_write(data, k->str, nread);
              else
#endif /* CURL_DISABLE_POP3 */
                result = Curl_client_write(data, CLIENTWRITE_BODY, k->str,
                                           nread);
            }
          }
          else if(!k->ignorebody && nread)
            result = Curl_unencode_write(data, k->writer_stack, k->str, nread);
        }
        k->badheader = HEADER_NORMAL; /* taken care of now */

        if(result)
          return result;
      }

    } /* if(!header and data to read) */

    if(conn->handler->readwrite && excess) {
      /* Parse the excess data */
      k->str += nread;

      if(&k->str[excess] > &buf[data->set.buffer_size]) {
        /* the excess amount was too excessive(!), make sure
           it doesn't read out of buffer */
        excess = &buf[data->set.buffer_size] - k->str;
      }
      nread = (ssize_t)excess;

      result = conn->handler->readwrite(data, conn, &nread, &readmore);
      if(result)
        return result;

      if(readmore)
        k->keepon |= KEEP_RECV; /* we're not done reading */
      break;
    }

    if(is_empty_data) {
      /* if we received nothing, the server closed the connection and we
         are done */
      k->keepon &= ~KEEP_RECV;
    }

    if(k->keepon & KEEP_RECV_PAUSE) {
      /* this is a paused transfer */
      break;
    }

  } while(data_pending(data) && maxloops--);

  if(maxloops <= 0) {
    /* we mark it as read-again-please */
    conn->cselect_bits = CURL_CSELECT_IN;
    *comeback = TRUE;
  }

  if(((k->keepon & (KEEP_RECV|KEEP_SEND)) == KEEP_SEND) &&
     conn->bits.close) {
    /* When we've read the entire thing and the close bit is set, the server
       may now close the connection. If there's now any kind of sending going
       on from our side, we need to stop that immediately. */
    infof(data, "we are done reading and this is set to close, stop send");
    k->keepon &= ~KEEP_SEND; /* no writing anymore either */
  }

  return CURLE_OK;
}

CURLcode Curl_done_sending(struct Curl_easy *data,
                           struct SingleRequest *k)
{
  struct connectdata *conn = data->conn;
  k->keepon &= ~KEEP_SEND; /* we're done writing */

  /* These functions should be moved into the handler struct! */
  Curl_http2_done_sending(data, conn);
  Curl_quic_done_sending(data);

  if(conn->bits.rewindaftersend) {
    CURLcode result = Curl_readrewind(data);
    if(result)
      return result;
  }
  return CURLE_OK;
}

#if defined(WIN32) && defined(USE_WINSOCK)
#ifndef SIO_IDEAL_SEND_BACKLOG_QUERY
#define SIO_IDEAL_SEND_BACKLOG_QUERY 0x4004747B
#endif

static void win_update_buffer_size(curl_socket_t sockfd)
{
  int result;
  ULONG ideal;
  DWORD ideallen;
  result = WSAIoctl(sockfd, SIO_IDEAL_SEND_BACKLOG_QUERY, 0, 0,
                    &ideal, sizeof(ideal), &ideallen, 0, 0);
  if(result == 0) {
    setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF,
               (const char *)&ideal, sizeof(ideal));
  }
}
#else
#define win_update_buffer_size(x)
#endif

/*
 * Send data to upload to the server, when the socket is writable.
 */
static CURLcode readwrite_upload(struct Curl_easy *data,
                                 struct connectdata *conn,
                                 int *didwhat)
{
  ssize_t i, si;
  ssize_t bytes_written;
  CURLcode result;
  ssize_t nread; /* number of bytes read */
  bool sending_http_headers = FALSE;
  struct SingleRequest *k = &data->req;

  if((k->bytecount == 0) && (k->writebytecount == 0))
    Curl_pgrsTime(data, TIMER_STARTTRANSFER);

  *didwhat |= KEEP_SEND;

  do {
    curl_off_t nbody;

    /* only read more data if there's no upload data already
       present in the upload buffer */
    if(0 == k->upload_present) {
      result = Curl_get_upload_buffer(data);
      if(result)
        return result;
      /* init the "upload from here" pointer */
      k->upload_fromhere = data->state.ulbuf;

      if(!k->upload_done) {
        /* HTTP pollution, this should be written nicer to become more
           protocol agnostic. */
        size_t fillcount;
        struct HTTP *http = k->p.http;

        if((k->exp100 == EXP100_SENDING_REQUEST) &&
           (http->sending == HTTPSEND_BODY)) {
          /* If this call is to send body data, we must take some action:
             We have sent off the full HTTP 1.1 request, and we shall now
             go into the Expect: 100 state and await such a header */
          k->exp100 = EXP100_AWAITING_CONTINUE; /* wait for the header */
          k->keepon &= ~KEEP_SEND;         /* disable writing */
          k->start100 = Curl_now();       /* timeout count starts now */
          *didwhat &= ~KEEP_SEND;  /* we didn't write anything actually */
          /* set a timeout for the multi interface */
          Curl_expire(data, data->set.expect_100_timeout, EXPIRE_100_TIMEOUT);
          break;
        }

        if(conn->handler->protocol&(PROTO_FAMILY_HTTP|CURLPROTO_RTSP)) {
          if(http->sending == HTTPSEND_REQUEST)
            /* We're sending the HTTP request headers, not the data.
               Remember that so we don't change the line endings. */
            sending_http_headers = TRUE;
          else
            sending_http_headers = FALSE;
        }

        result = Curl_fillreadbuffer(data, data->set.upload_buffer_size,
                                     &fillcount);
        if(result)
          return result;

        nread = fillcount;
      }
      else
        nread = 0; /* we're done uploading/reading */

      if(!nread && (k->keepon & KEEP_SEND_PAUSE)) {
        /* this is a paused transfer */
        break;
      }
      if(nread <= 0) {
        result = Curl_done_sending(data, k);
        if(result)
          return result;
        break;
      }

      /* store number of bytes available for upload */
      k->upload_present = nread;

      /* convert LF to CRLF if so asked */
      if((!sending_http_headers) && (
#ifdef CURL_DO_LINEEND_CONV
         /* always convert if we're FTPing in ASCII mode */
         (data->state.prefer_ascii) ||
#endif
         (data->set.crlf))) {
        /* Do we need to allocate a scratch buffer? */
        if(!data->state.scratch) {
          data->state.scratch = malloc(2 * data->set.upload_buffer_size);
          if(!data->state.scratch) {
            failf(data, "Failed to alloc scratch buffer!");

            return CURLE_OUT_OF_MEMORY;
          }
        }

        /*
         * ASCII/EBCDIC Note: This is presumably a text (not binary)
         * transfer so the data should already be in ASCII.
         * That means the hex values for ASCII CR (0x0d) & LF (0x0a)
         * must be used instead of the escape sequences \r & \n.
         */
        for(i = 0, si = 0; i < nread; i++, si++) {
          if(k->upload_fromhere[i] == 0x0a) {
            data->state.scratch[si++] = 0x0d;
            data->state.scratch[si] = 0x0a;
            if(!data->set.crlf) {
              /* we're here only because FTP is in ASCII mode...
                 bump infilesize for the LF we just added */
              if(data->state.infilesize != -1)
                data->state.infilesize++;
            }
          }
          else
            data->state.scratch[si] = k->upload_fromhere[i];
        }

        if(si != nread) {
          /* only perform the special operation if we really did replace
             anything */
          nread = si;

          /* upload from the new (replaced) buffer instead */
          k->upload_fromhere = data->state.scratch;

          /* set the new amount too */
          k->upload_present = nread;
        }
      }

#ifndef CURL_DISABLE_SMTP
      if(conn->handler->protocol & PROTO_FAMILY_SMTP) {
        result = Curl_smtp_escape_eob(data, nread);
        if(result)
          return result;
      }
#endif /* CURL_DISABLE_SMTP */
    } /* if 0 == k->upload_present */
    else {
      /* We have a partial buffer left from a previous "round". Use
         that instead of reading more data */
    }

    /* write to socket (send away data) */
    result = Curl_write(data,
                        conn->writesockfd,  /* socket to send to */
                        k->upload_fromhere, /* buffer pointer */
                        k->upload_present,  /* buffer size */
                        &bytes_written);    /* actually sent */
    if(result)
      return result;

    win_update_buffer_size(conn->writesockfd);

    if(k->pendingheader) {
      /* parts of what was sent was header */
      curl_off_t n = CURLMIN(k->pendingheader, bytes_written);
      /* show the data before we change the pointer upload_fromhere */
      Curl_debug(data, CURLINFO_HEADER_OUT, k->upload_fromhere, (size_t)n);
      k->pendingheader -= n;
      nbody = bytes_written - n; /* size of the written body part */
    }
    else
      nbody = bytes_written;

    if(nbody) {
      /* show the data before we change the pointer upload_fromhere */
      Curl_debug(data, CURLINFO_DATA_OUT,
                 &k->upload_fromhere[bytes_written - nbody],
                 (size_t)nbody);

      k->writebytecount += nbody;
      Curl_pgrsSetUploadCounter(data, k->writebytecount);
    }

    if((!k->upload_chunky || k->forbidchunk) &&
       (k->writebytecount == data->state.infilesize)) {
      /* we have sent all data we were supposed to */
      k->upload_done = TRUE;
      infof(data, "We are completely uploaded and fine");
    }

    if(k->upload_present != bytes_written) {
      /* we only wrote a part of the buffer (if anything), deal with it! */

      /* store the amount of bytes left in the buffer to write */
      k->upload_present -= bytes_written;

      /* advance the pointer where to find the buffer when the next send
         is to happen */
      k->upload_fromhere += bytes_written;
    }
    else {
      /* we've uploaded that buffer now */
      result = Curl_get_upload_buffer(data);
      if(result)
        return result;
      k->upload_fromhere = data->state.ulbuf;
      k->upload_present = 0; /* no more bytes left */

      if(k->upload_done) {
        result = Curl_done_sending(data, k);
        if(result)
          return result;
      }
    }


  } while(0); /* just to break out from! */

  return CURLE_OK;
}

/*
 * Curl_readwrite() is the low-level function to be called when data is to
 * be read and written to/from the connection.
 *
 * return '*comeback' TRUE if we didn't properly drain the socket so this
 * function should get called again without select() or similar in between!
 */
CURLcode Curl_readwrite(struct connectdata *conn,
                        struct Curl_easy *data,
                        bool *done,
                        bool *comeback)
{
  struct SingleRequest *k = &data->req;
  CURLcode result;
  int didwhat = 0;

  curl_socket_t fd_read;
  curl_socket_t fd_write;
  int select_res = conn->cselect_bits;

  conn->cselect_bits = 0;

  /* only use the proper socket if the *_HOLD bit is not set simultaneously as
     then we are in rate limiting state in that transfer direction */

  if((k->keepon & KEEP_RECVBITS) == KEEP_RECV)
    fd_read = conn->sockfd;
  else
    fd_read = CURL_SOCKET_BAD;

  if((k->keepon & KEEP_SENDBITS) == KEEP_SEND)
    fd_write = conn->writesockfd;
  else
    fd_write = CURL_SOCKET_BAD;

  if(data->state.drain) {
    select_res |= CURL_CSELECT_IN;
    DEBUGF(infof(data, "Curl_readwrite: forcibly told to drain data"));
  }

  if(!select_res) /* Call for select()/poll() only, if read/write/error
                     status is not known. */
    select_res = Curl_socket_check(fd_read, CURL_SOCKET_BAD, fd_write, 0);

  if(select_res == CURL_CSELECT_ERR) {
    failf(data, "select/poll returned error");
    return CURLE_SEND_ERROR;
  }

#ifdef USE_HYPER
  if(conn->datastream) {
    result = conn->datastream(data, conn, &didwhat, done, select_res);
    if(result || *done)
      return result;
  }
  else {
#endif
  /* We go ahead and do a read if we have a readable socket or if
     the stream was rewound (in which case we have data in a
     buffer) */
  if((k->keepon & KEEP_RECV) && (select_res & CURL_CSELECT_IN)) {
    result = readwrite_data(data, conn, k, &didwhat, done, comeback);
    if(result || *done)
      return result;
  }

  /* If we still have writing to do, we check if we have a writable socket. */
  if((k->keepon & KEEP_SEND) && (select_res & CURL_CSELECT_OUT)) {
    /* write */

    result = readwrite_upload(data, conn, &didwhat);
    if(result)
      return result;
  }
#ifdef USE_HYPER
  }
#endif

  k->now = Curl_now();
  if(!didwhat) {
    /* no read no write, this is a timeout? */
    if(k->exp100 == EXP100_AWAITING_CONTINUE) {
      /* This should allow some time for the header to arrive, but only a
         very short time as otherwise it'll be too much wasted time too
         often. */

      /* Quoting RFC2616, section "8.2.3 Use of the 100 (Continue) Status":

         Therefore, when a client sends this header field to an origin server
         (possibly via a proxy) from which it has never seen a 100 (Continue)
         status, the client SHOULD NOT wait for an indefinite period before
         sending the request body.

      */

      timediff_t ms = Curl_timediff(k->now, k->start100);
      if(ms >= data->set.expect_100_timeout) {
        /* we've waited long enough, continue anyway */
        k->exp100 = EXP100_SEND_DATA;
        k->keepon |= KEEP_SEND;
        Curl_expire_done(data, EXPIRE_100_TIMEOUT);
        infof(data, "Done waiting for 100-continue");
      }
    }
  }

  if(Curl_pgrsUpdate(data))
    result = CURLE_ABORTED_BY_CALLBACK;
  else
    result = Curl_speedcheck(data, k->now);
  if(result)
    return result;

  if(k->keepon) {
    if(0 > Curl_timeleft(data, &k->now, FALSE)) {
      if(k->size != -1) {
        failf(data, "Operation timed out after %" CURL_FORMAT_TIMEDIFF_T
              " milliseconds with %" CURL_FORMAT_CURL_OFF_T " out of %"
              CURL_FORMAT_CURL_OFF_T " bytes received",
              Curl_timediff(k->now, data->progress.t_startsingle),
              k->bytecount, k->size);
      }
      else {
        failf(data, "Operation timed out after %" CURL_FORMAT_TIMEDIFF_T
              " milliseconds with %" CURL_FORMAT_CURL_OFF_T " bytes received",
              Curl_timediff(k->now, data->progress.t_startsingle),
              k->bytecount);
      }
      return CURLE_OPERATION_TIMEDOUT;
    }
  }
  else {
    /*
     * The transfer has been performed. Just make some general checks before
     * returning.
     */

    if(!(data->set.opt_no_body) && (k->size != -1) &&
       (k->bytecount != k->size) &&
#ifdef CURL_DO_LINEEND_CONV
       /* Most FTP servers don't adjust their file SIZE response for CRLFs,
          so we'll check to see if the discrepancy can be explained
          by the number of CRLFs we've changed to LFs.
       */
       (k->bytecount != (k->size + data->state.crlf_conversions)) &&
#endif /* CURL_DO_LINEEND_CONV */
       !k->newurl) {
      failf(data, "transfer closed with %" CURL_FORMAT_CURL_OFF_T
            " bytes remaining to read", k->size - k->bytecount);
      return CURLE_PARTIAL_FILE;
    }
    if(!(data->set.opt_no_body) && k->chunk &&
       (conn->chunk.state != CHUNK_STOP)) {
      /*
       * In chunked mode, return an error if the connection is closed prior to
       * the empty (terminating) chunk is read.
       *
       * The condition above used to check for
       * conn->proto.http->chunk.datasize != 0 which is true after reading
       * *any* chunk, not just the empty chunk.
       *
       */
      failf(data, "transfer closed with outstanding read data remaining");
      return CURLE_PARTIAL_FILE;
    }
    if(Curl_pgrsUpdate(data))
      return CURLE_ABORTED_BY_CALLBACK;
  }

  /* Now update the "done" boolean we return */
  *done = (0 == (k->keepon&(KEEP_RECV|KEEP_SEND|
                            KEEP_RECV_PAUSE|KEEP_SEND_PAUSE))) ? TRUE : FALSE;

  return CURLE_OK;
}

/*
 * Curl_single_getsock() gets called by the multi interface code when the app
 * has requested to get the sockets for the current connection. This function
 * will then be called once for every connection that the multi interface
 * keeps track of. This function will only be called for connections that are
 * in the proper state to have this information available.
 */
int Curl_single_getsock(struct Curl_easy *data,
                        struct connectdata *conn,
                        curl_socket_t *sock)
{
  int bitmap = GETSOCK_BLANK;
  unsigned sockindex = 0;

  if(conn->handler->perform_getsock)
    return conn->handler->perform_getsock(data, conn, sock);

  /* don't include HOLD and PAUSE connections */
  if((data->req.keepon & KEEP_RECVBITS) == KEEP_RECV) {

    DEBUGASSERT(conn->sockfd != CURL_SOCKET_BAD);

    bitmap |= GETSOCK_READSOCK(sockindex);
    sock[sockindex] = conn->sockfd;
  }

  /* don't include HOLD and PAUSE connections */
  if((data->req.keepon & KEEP_SENDBITS) == KEEP_SEND) {

    if((conn->sockfd != conn->writesockfd) ||
       bitmap == GETSOCK_BLANK) {
      /* only if they are not the same socket and we have a readable
         one, we increase index */
      if(bitmap != GETSOCK_BLANK)
        sockindex++; /* increase index if we need two entries */

      DEBUGASSERT(conn->writesockfd != CURL_SOCKET_BAD);

      sock[sockindex] = conn->writesockfd;
    }

    bitmap |= GETSOCK_WRITESOCK(sockindex);
  }

  return bitmap;
}

/* Curl_init_CONNECT() gets called each time the handle switches to CONNECT
   which means this gets called once for each subsequent redirect etc */
void Curl_init_CONNECT(struct Curl_easy *data)
{
  data->state.fread_func = data->set.fread_func_set;
  data->state.in = data->set.in_set;
}

/*
 * Curl_pretransfer() is called immediately before a transfer starts, and only
 * once for one transfer no matter if it has redirects or do multi-pass
 * authentication etc.
 */
CURLcode Curl_pretransfer(struct Curl_easy *data)
{
  CURLcode result;

  if(!data->state.url && !data->set.uh) {
    /* we can't do anything without URL */
    failf(data, "No URL set!");
    return CURLE_URL_MALFORMAT;
  }

  /* since the URL may have been redirected in a previous use of this handle */
  if(data->state.url_alloc) {
    /* the already set URL is allocated, free it first! */
    Curl_safefree(data->state.url);
    data->state.url_alloc = FALSE;
  }

  if(!data->state.url && data->set.uh) {
    CURLUcode uc;
    free(data->set.str[STRING_SET_URL]);
    uc = curl_url_get(data->set.uh,
                      CURLUPART_URL, &data->set.str[STRING_SET_URL], 0);
    if(uc) {
      failf(data, "No URL set!");
      return CURLE_URL_MALFORMAT;
    }
  }

  data->state.prefer_ascii = data->set.prefer_ascii;
  data->state.list_only = data->set.list_only;
  data->state.httpreq = data->set.method;
  data->state.url = data->set.str[STRING_SET_URL];

  /* Init the SSL session ID cache here. We do it here since we want to do it
     after the *_setopt() calls (that could specify the size of the cache) but
     before any transfer takes place. */
  result = Curl_ssl_initsessions(data, data->set.general_ssl.max_ssl_sessions);
  if(result)
    return result;

  data->state.wildcardmatch = data->set.wildcard_enabled;
  data->state.followlocation = 0; /* reset the location-follow counter */
  data->state.this_is_a_follow = FALSE; /* reset this */
  data->state.errorbuf = FALSE; /* no error has occurred */
  data->state.httpwant = data->set.httpwant;
  data->state.httpversion = 0;
  data->state.authproblem = FALSE;
  data->state.authhost.want = data->set.httpauth;
  data->state.authproxy.want = data->set.proxyauth;
  Curl_safefree(data->info.wouldredirect);

  if(data->state.httpreq == HTTPREQ_PUT)
    data->state.infilesize = data->set.filesize;
  else if((data->state.httpreq != HTTPREQ_GET) &&
          (data->state.httpreq != HTTPREQ_HEAD)) {
    data->state.infilesize = data->set.postfieldsize;
    if(data->set.postfields && (data->state.infilesize == -1))
      data->state.infilesize = (curl_off_t)strlen(data->set.postfields);
  }
  else
    data->state.infilesize = 0;

  /* If there is a list of cookie files to read, do it now! */
  if(data->state.cookielist)
    Curl_cookie_loadfiles(data);

  /* If there is a list of host pairs to deal with */
  if(data->state.resolve)
    result = Curl_loadhostpairs(data);

  if(!result) {
    /* Allow data->set.use_port to set which port to use. This needs to be
     * disabled for example when we follow Location: headers to URLs using
     * different ports! */
    data->state.allow_port = TRUE;

#if defined(HAVE_SIGNAL) && defined(SIGPIPE) && !defined(HAVE_MSG_NOSIGNAL)
    /*************************************************************
     * Tell signal handler to ignore SIGPIPE
     *************************************************************/
    if(!data->set.no_signal)
      data->state.prev_signal = signal(SIGPIPE, SIG_IGN);
#endif

    Curl_initinfo(data); /* reset session-specific information "variables" */
    Curl_pgrsResetTransferSizes(data);
    Curl_pgrsStartNow(data);

    /* In case the handle is re-used and an authentication method was picked
       in the session we need to make sure we only use the one(s) we now
       consider to be fine */
    data->state.authhost.picked &= data->state.authhost.want;
    data->state.authproxy.picked &= data->state.authproxy.want;

#ifndef CURL_DISABLE_FTP
    if(data->state.wildcardmatch) {
      struct WildcardData *wc = &data->wildcard;
      if(wc->state < CURLWC_INIT) {
        result = Curl_wildcard_init(wc); /* init wildcard structures */
        if(result)
          return CURLE_OUT_OF_MEMORY;
      }
    }
#endif
    Curl_http2_init_state(&data->state);
    result = Curl_hsts_loadcb(data, data->hsts);
  }

  /*
   * Set user-agent. Used for HTTP, but since we can attempt to tunnel
   * basically anything through a http proxy we can't limit this based on
   * protocol.
   */
  if(data->set.str[STRING_USERAGENT]) {
    Curl_safefree(data->state.aptr.uagent);
    data->state.aptr.uagent =
      aprintf("User-Agent: %s\r\n", data->set.str[STRING_USERAGENT]);
    if(!data->state.aptr.uagent)
      return CURLE_OUT_OF_MEMORY;
  }

  if(!result)
    result = Curl_setstropt(&data->state.aptr.user,
                            data->set.str[STRING_USERNAME]);
  if(!result)
    result = Curl_setstropt(&data->state.aptr.passwd,
                            data->set.str[STRING_PASSWORD]);
  if(!result)
    result = Curl_setstropt(&data->state.aptr.proxyuser,
                            data->set.str[STRING_PROXYUSERNAME]);
  if(!result)
    result = Curl_setstropt(&data->state.aptr.proxypasswd,
                            data->set.str[STRING_PROXYPASSWORD]);

  data->req.headerbytecount = 0;
  return result;
}

/*
 * Curl_posttransfer() is called immediately after a transfer ends
 */
CURLcode Curl_posttransfer(struct Curl_easy *data)
{
#if defined(HAVE_SIGNAL) && defined(SIGPIPE) && !defined(HAVE_MSG_NOSIGNAL)
  /* restore the signal handler for SIGPIPE before we get back */
  if(!data->set.no_signal)
    signal(SIGPIPE, data->state.prev_signal);
#else
  (void)data; /* unused parameter */
#endif

  return CURLE_OK;
}

/*
 * Curl_follow() handles the URL redirect magic. Pass in the 'newurl' string
 * as given by the remote server and set up the new URL to request.
 *
 * This function DOES NOT FREE the given url.
 */
CURLcode Curl_follow(struct Curl_easy *data,
                     char *newurl,    /* the Location: string */
                     followtype type) /* see transfer.h */
{
#ifdef CURL_DISABLE_HTTP
  (void)data;
  (void)newurl;
  (void)type;
  /* Location: following will not happen when HTTP is disabled */
  return CURLE_TOO_MANY_REDIRECTS;
#else

  /* Location: redirect */
  bool disallowport = FALSE;
  bool reachedmax = FALSE;
  CURLUcode uc;

  DEBUGASSERT(type != FOLLOW_NONE);

  if(type == FOLLOW_REDIR) {
    if((data->set.maxredirs != -1) &&
       (data->state.followlocation >= data->set.maxredirs)) {
      reachedmax = TRUE;
      type = FOLLOW_FAKE; /* switch to fake to store the would-be-redirected
                             to URL */
    }
    else {
      /* mark the next request as a followed location: */
      data->state.this_is_a_follow = TRUE;

      data->state.followlocation++; /* count location-followers */

      if(data->set.http_auto_referer) {
        CURLU *u;
        char *referer = NULL;

        /* We are asked to automatically set the previous URL as the referer
           when we get the next URL. We pick the ->url field, which may or may
           not be 100% correct */

        if(data->state.referer_alloc) {
          Curl_safefree(data->state.referer);
          data->state.referer_alloc = FALSE;
        }

        /* Make a copy of the URL without crenditals and fragment */
        u = curl_url();
        if(!u)
          return CURLE_OUT_OF_MEMORY;

        uc = curl_url_set(u, CURLUPART_URL, data->state.url, 0);
        if(!uc)
          uc = curl_url_set(u, CURLUPART_FRAGMENT, NULL, 0);
        if(!uc)
          uc = curl_url_set(u, CURLUPART_USER, NULL, 0);
        if(!uc)
          uc = curl_url_set(u, CURLUPART_PASSWORD, NULL, 0);
        if(!uc)
          uc = curl_url_get(u, CURLUPART_URL, &referer, 0);

        curl_url_cleanup(u);

        if(uc || !referer)
          return CURLE_OUT_OF_MEMORY;

        data->state.referer = referer;
        data->state.referer_alloc = TRUE; /* yes, free this later */
      }
    }
  }

  if((type != FOLLOW_RETRY) &&
     (data->req.httpcode != 401) && (data->req.httpcode != 407) &&
     Curl_is_absolute_url(newurl, NULL, 0))
    /* If this is not redirect due to a 401 or 407 response and an absolute
       URL: don't allow a custom port number */
    disallowport = TRUE;

  DEBUGASSERT(data->state.uh);
  uc = curl_url_set(data->state.uh, CURLUPART_URL, newurl,
                    (type == FOLLOW_FAKE) ? CURLU_NON_SUPPORT_SCHEME :
                    ((type == FOLLOW_REDIR) ? CURLU_URLENCODE : 0) |
                    CURLU_ALLOW_SPACE);
  if(uc) {
    if(type != FOLLOW_FAKE)
      return Curl_uc_to_curlcode(uc);

    /* the URL could not be parsed for some reason, but since this is FAKE
       mode, just duplicate the field as-is */
    newurl = strdup(newurl);
    if(!newurl)
      return CURLE_OUT_OF_MEMORY;
  }
  else {

    uc = curl_url_get(data->state.uh, CURLUPART_URL, &newurl, 0);
    if(uc)
      return Curl_uc_to_curlcode(uc);
  }

  if(type == FOLLOW_FAKE) {
    /* we're only figuring out the new url if we would've followed locations
       but now we're done so we can get out! */
    data->info.wouldredirect = newurl;

    if(reachedmax) {
      failf(data, "Maximum (%ld) redirects followed", data->set.maxredirs);
      return CURLE_TOO_MANY_REDIRECTS;
    }
    return CURLE_OK;
  }

  if(disallowport)
    data->state.allow_port = FALSE;

  if(data->state.url_alloc)
    Curl_safefree(data->state.url);

  data->state.url = newurl;
  data->state.url_alloc = TRUE;

  infof(data, "Issue another request to this URL: '%s'", data->state.url);

  /*
   * We get here when the HTTP code is 300-399 (and 401). We need to perform
   * differently based on exactly what return code there was.
   *
   * News from 7.10.6: we can also get here on a 401 or 407, in case we act on
   * a HTTP (proxy-) authentication scheme other than Basic.
   */
  switch(data->info.httpcode) {
    /* 401 - Act on a WWW-Authenticate, we keep on moving and do the
       Authorization: XXXX header in the HTTP request code snippet */
    /* 407 - Act on a Proxy-Authenticate, we keep on moving and do the
       Proxy-Authorization: XXXX header in the HTTP request code snippet */
    /* 300 - Multiple Choices */
    /* 306 - Not used */
    /* 307 - Temporary Redirect */
  default:  /* for all above (and the unknown ones) */
    /* Some codes are explicitly mentioned since I've checked RFC2616 and they
     * seem to be OK to POST to.
     */
    break;
  case 301: /* Moved Permanently */
    /* (quote from RFC7231, section 6.4.2)
     *
     * Note: For historical reasons, a user agent MAY change the request
     * method from POST to GET for the subsequent request.  If this
     * behavior is undesired, the 307 (Temporary Redirect) status code
     * can be used instead.
     *
     * ----
     *
     * Many webservers expect this, so these servers often answers to a POST
     * request with an error page. To be sure that libcurl gets the page that
     * most user agents would get, libcurl has to force GET.
     *
     * This behavior is forbidden by RFC1945 and the obsolete RFC2616, and
     * can be overridden with CURLOPT_POSTREDIR.
     */
    if((data->state.httpreq == HTTPREQ_POST
        || data->state.httpreq == HTTPREQ_POST_FORM
        || data->state.httpreq == HTTPREQ_POST_MIME)
       && !(data->set.keep_post & CURL_REDIR_POST_301)) {
      infof(data, "Switch from POST to GET");
      data->state.httpreq = HTTPREQ_GET;
    }
    break;
  case 302: /* Found */
    /* (quote from RFC7231, section 6.4.3)
     *
     * Note: For historical reasons, a user agent MAY change the request
     * method from POST to GET for the subsequent request.  If this
     * behavior is undesired, the 307 (Temporary Redirect) status code
     * can be used instead.
     *
     * ----
     *
     * Many webservers expect this, so these servers often answers to a POST
     * request with an error page. To be sure that libcurl gets the page that
     * most user agents would get, libcurl has to force GET.
     *
     * This behavior is forbidden by RFC1945 and the obsolete RFC2616, and
     * can be overridden with CURLOPT_POSTREDIR.
     */
    if((data->state.httpreq == HTTPREQ_POST
        || data->state.httpreq == HTTPREQ_POST_FORM
        || data->state.httpreq == HTTPREQ_POST_MIME)
       && !(data->set.keep_post & CURL_REDIR_POST_302)) {
      infof(data, "Switch from POST to GET");
      data->state.httpreq = HTTPREQ_GET;
    }
    break;

  case 303: /* See Other */
    /* 'See Other' location is not the resource but a substitute for the
     * resource. In this case we switch the method to GET/HEAD, unless the
     * method is POST and the user specified to keep it as POST.
     * https://github.com/curl/curl/issues/5237#issuecomment-614641049
     */
    if(data->state.httpreq != HTTPREQ_GET &&
       ((data->state.httpreq != HTTPREQ_POST &&
         data->state.httpreq != HTTPREQ_POST_FORM &&
         data->state.httpreq != HTTPREQ_POST_MIME) ||
        !(data->set.keep_post & CURL_REDIR_POST_303))) {
      data->state.httpreq = HTTPREQ_GET;
      data->set.upload = false;
      infof(data, "Switch to %s",
            data->set.opt_no_body?"HEAD":"GET");
    }
    break;
  case 304: /* Not Modified */
    /* 304 means we did a conditional request and it was "Not modified".
     * We shouldn't get any Location: header in this response!
     */
    break;
  case 305: /* Use Proxy */
    /* (quote from RFC2616, section 10.3.6):
     * "The requested resource MUST be accessed through the proxy given
     * by the Location field. The Location field gives the URI of the
     * proxy.  The recipient is expected to repeat this single request
     * via the proxy. 305 responses MUST only be generated by origin
     * servers."
     */
    break;
  }
  Curl_pgrsTime(data, TIMER_REDIRECT);
  Curl_pgrsResetTransferSizes(data);

  return CURLE_OK;
#endif /* CURL_DISABLE_HTTP */
}

/* Returns CURLE_OK *and* sets '*url' if a request retry is wanted.

   NOTE: that the *url is malloc()ed. */
CURLcode Curl_retry_request(struct Curl_easy *data, char **url)
{
  struct connectdata *conn = data->conn;
  bool retry = FALSE;
  *url = NULL;

  /* if we're talking upload, we can't do the checks below, unless the protocol
     is HTTP as when uploading over HTTP we will still get a response */
  if(data->set.upload &&
     !(conn->handler->protocol&(PROTO_FAMILY_HTTP|CURLPROTO_RTSP)))
    return CURLE_OK;

  if((data->req.bytecount + data->req.headerbytecount == 0) &&
      conn->bits.reuse &&
      (!data->set.opt_no_body
        || (conn->handler->protocol & PROTO_FAMILY_HTTP)) &&
      (data->set.rtspreq != RTSPREQ_RECEIVE))
    /* We got no data, we attempted to re-use a connection. For HTTP this
       can be a retry so we try again regardless if we expected a body.
       For other protocols we only try again only if we expected a body.

       This might happen if the connection was left alive when we were
       done using it before, but that was closed when we wanted to read from
       it again. Bad luck. Retry the same request on a fresh connect! */
    retry = TRUE;
  else if(data->state.refused_stream &&
          (data->req.bytecount + data->req.headerbytecount == 0) ) {
    /* This was sent on a refused stream, safe to rerun. A refused stream
       error can typically only happen on HTTP/2 level if the stream is safe
       to issue again, but the nghttp2 API can deliver the message to other
       streams as well, which is why this adds the check the data counters
       too. */
    infof(data, "REFUSED_STREAM, retrying a fresh connect");
    data->state.refused_stream = FALSE; /* clear again */
    retry = TRUE;
  }
  if(retry) {
#define CONN_MAX_RETRIES 5
    if(data->state.retrycount++ >= CONN_MAX_RETRIES) {
      failf(data, "Connection died, tried %d times before giving up",
            CONN_MAX_RETRIES);
      data->state.retrycount = 0;
      return CURLE_SEND_ERROR;
    }
    infof(data, "Connection died, retrying a fresh connect (retry count: %d)",
          data->state.retrycount);
    *url = strdup(data->state.url);
    if(!*url)
      return CURLE_OUT_OF_MEMORY;

    connclose(conn, "retry"); /* close this connection */
    conn->bits.retry = TRUE; /* mark this as a connection we're about
                                to retry. Marking it this way should
                                prevent i.e HTTP transfers to return
                                error just because nothing has been
                                transferred! */


    if(conn->handler->protocol&PROTO_FAMILY_HTTP) {
      if(data->req.writebytecount) {
        CURLcode result = Curl_readrewind(data);
        if(result) {
          Curl_safefree(*url);
          return result;
        }
      }
    }
  }
  return CURLE_OK;
}

/*
 * Curl_setup_transfer() is called to setup some basic properties for the
 * upcoming transfer.
 */
void
Curl_setup_transfer(
  struct Curl_easy *data,   /* transfer */
  int sockindex,            /* socket index to read from or -1 */
  curl_off_t size,          /* -1 if unknown at this point */
  bool getheader,           /* TRUE if header parsing is wanted */
  int writesockindex        /* socket index to write to, it may very well be
                               the same we read from. -1 disables */
  )
{
  struct SingleRequest *k = &data->req;
  struct connectdata *conn = data->conn;
  struct HTTP *http = data->req.p.http;
  bool httpsending = ((conn->handler->protocol&PROTO_FAMILY_HTTP) &&
                      (http->sending == HTTPSEND_REQUEST));
  DEBUGASSERT(conn != NULL);
  DEBUGASSERT((sockindex <= 1) && (sockindex >= -1));

  if(conn->bits.multiplex || conn->httpversion == 20 || httpsending) {
    /* when multiplexing, the read/write sockets need to be the same! */
    conn->sockfd = sockindex == -1 ?
      ((writesockindex == -1 ? CURL_SOCKET_BAD : conn->sock[writesockindex])) :
      conn->sock[sockindex];
    conn->writesockfd = conn->sockfd;
    if(httpsending)
      /* special and very HTTP-specific */
      writesockindex = FIRSTSOCKET;
  }
  else {
    conn->sockfd = sockindex == -1 ?
      CURL_SOCKET_BAD : conn->sock[sockindex];
    conn->writesockfd = writesockindex == -1 ?
      CURL_SOCKET_BAD:conn->sock[writesockindex];
  }
  k->getheader = getheader;

  k->size = size;

  /* The code sequence below is placed in this function just because all
     necessary input is not always known in do_complete() as this function may
     be called after that */

  if(!k->getheader) {
    k->header = FALSE;
    if(size > 0)
      Curl_pgrsSetDownloadSize(data, size);
  }
  /* we want header and/or body, if neither then don't do this! */
  if(k->getheader || !data->set.opt_no_body) {

    if(sockindex != -1)
      k->keepon |= KEEP_RECV;

    if(writesockindex != -1) {
      /* HTTP 1.1 magic:

         Even if we require a 100-return code before uploading data, we might
         need to write data before that since the REQUEST may not have been
         finished sent off just yet.

         Thus, we must check if the request has been sent before we set the
         state info where we wait for the 100-return code
      */
      if((data->state.expect100header) &&
         (conn->handler->protocol&PROTO_FAMILY_HTTP) &&
         (http->sending == HTTPSEND_BODY)) {
        /* wait with write until we either got 100-continue or a timeout */
        k->exp100 = EXP100_AWAITING_CONTINUE;
        k->start100 = Curl_now();

        /* Set a timeout for the multi interface. Add the inaccuracy margin so
           that we don't fire slightly too early and get denied to run. */
        Curl_expire(data, data->set.expect_100_timeout, EXPIRE_100_TIMEOUT);
      }
      else {
        if(data->state.expect100header)
          /* when we've sent off the rest of the headers, we must await a
             100-continue but first finish sending the request */
          k->exp100 = EXP100_SENDING_REQUEST;

        /* enable the write bit when we're not waiting for continue */
        k->keepon |= KEEP_SEND;
      }
    } /* if(writesockindex != -1) */
  } /* if(k->getheader || !data->set.opt_no_body) */

}
