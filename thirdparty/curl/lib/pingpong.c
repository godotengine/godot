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
 *   'pingpong' is for generic back-and-forth support functions used by FTP,
 *   IMAP, POP3, SMTP and whatever more that likes them.
 *
 ***************************************************************************/

#include "curl_setup.h"

#include "urldata.h"
#include "sendf.h"
#include "select.h"
#include "progress.h"
#include "speedcheck.h"
#include "pingpong.h"
#include "multiif.h"
#include "non-ascii.h"
#include "vtls/vtls.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#ifdef USE_PINGPONG

/* Returns timeout in ms. 0 or negative number means the timeout has already
   triggered */
timediff_t Curl_pp_state_timeout(struct Curl_easy *data,
                                 struct pingpong *pp, bool disconnecting)
{
  struct connectdata *conn = data->conn;
  timediff_t timeout_ms; /* in milliseconds */
  timediff_t response_time = (data->set.server_response_timeout)?
    data->set.server_response_timeout: pp->response_time;

  /* if CURLOPT_SERVER_RESPONSE_TIMEOUT is set, use that to determine
     remaining time, or use pp->response because SERVER_RESPONSE_TIMEOUT is
     supposed to govern the response for any given server response, not for
     the time from connect to the given server response. */

  /* Without a requested timeout, we only wait 'response_time' seconds for the
     full response to arrive before we bail out */
  timeout_ms = response_time -
    Curl_timediff(Curl_now(), pp->response); /* spent time */

  if(data->set.timeout && !disconnecting) {
    /* if timeout is requested, find out how much remaining time we have */
    timediff_t timeout2_ms = data->set.timeout - /* timeout time */
      Curl_timediff(Curl_now(), conn->now); /* spent time */

    /* pick the lowest number */
    timeout_ms = CURLMIN(timeout_ms, timeout2_ms);
  }

  return timeout_ms;
}

/*
 * Curl_pp_statemach()
 */
CURLcode Curl_pp_statemach(struct Curl_easy *data,
                           struct pingpong *pp, bool block,
                           bool disconnecting)
{
  struct connectdata *conn = data->conn;
  curl_socket_t sock = conn->sock[FIRSTSOCKET];
  int rc;
  timediff_t interval_ms;
  timediff_t timeout_ms = Curl_pp_state_timeout(data, pp, disconnecting);
  CURLcode result = CURLE_OK;

  if(timeout_ms <= 0) {
    failf(data, "server response timeout");
    return CURLE_OPERATION_TIMEDOUT; /* already too little time */
  }

  if(block) {
    interval_ms = 1000;  /* use 1 second timeout intervals */
    if(timeout_ms < interval_ms)
      interval_ms = timeout_ms;
  }
  else
    interval_ms = 0; /* immediate */

  if(Curl_ssl_data_pending(conn, FIRSTSOCKET))
    rc = 1;
  else if(Curl_pp_moredata(pp))
    /* We are receiving and there is data in the cache so just read it */
    rc = 1;
  else if(!pp->sendleft && Curl_ssl_data_pending(conn, FIRSTSOCKET))
    /* We are receiving and there is data ready in the SSL library */
    rc = 1;
  else
    rc = Curl_socket_check(pp->sendleft?CURL_SOCKET_BAD:sock, /* reading */
                           CURL_SOCKET_BAD,
                           pp->sendleft?sock:CURL_SOCKET_BAD, /* writing */
                           interval_ms);

  if(block) {
    /* if we didn't wait, we don't have to spend time on this now */
    if(Curl_pgrsUpdate(data))
      result = CURLE_ABORTED_BY_CALLBACK;
    else
      result = Curl_speedcheck(data, Curl_now());

    if(result)
      return result;
  }

  if(rc == -1) {
    failf(data, "select/poll error");
    result = CURLE_OUT_OF_MEMORY;
  }
  else if(rc)
    result = pp->statemachine(data, data->conn);

  return result;
}

/* initialize stuff to prepare for reading a fresh new response */
void Curl_pp_init(struct Curl_easy *data, struct pingpong *pp)
{
  DEBUGASSERT(data);
  pp->nread_resp = 0;
  pp->linestart_resp = data->state.buffer;
  pp->pending_resp = TRUE;
  pp->response = Curl_now(); /* start response time-out now! */
}

/* setup for the coming transfer */
void Curl_pp_setup(struct pingpong *pp)
{
  Curl_dyn_init(&pp->sendbuf, DYN_PINGPPONG_CMD);
}

/***********************************************************************
 *
 * Curl_pp_vsendf()
 *
 * Send the formatted string as a command to a pingpong server. Note that
 * the string should not have any CRLF appended, as this function will
 * append the necessary things itself.
 *
 * made to never block
 */
CURLcode Curl_pp_vsendf(struct Curl_easy *data,
                        struct pingpong *pp,
                        const char *fmt,
                        va_list args)
{
  ssize_t bytes_written = 0;
  size_t write_len;
  char *s;
  CURLcode result;
  struct connectdata *conn = data->conn;

#ifdef HAVE_GSSAPI
  enum protection_level data_sec;
#endif

  DEBUGASSERT(pp->sendleft == 0);
  DEBUGASSERT(pp->sendsize == 0);
  DEBUGASSERT(pp->sendthis == NULL);

  if(!conn)
    /* can't send without a connection! */
    return CURLE_SEND_ERROR;

  Curl_dyn_reset(&pp->sendbuf);
  result = Curl_dyn_vaddf(&pp->sendbuf, fmt, args);
  if(result)
    return result;

  /* append CRLF */
  result = Curl_dyn_addn(&pp->sendbuf, "\r\n", 2);
  if(result)
    return result;

  write_len = Curl_dyn_len(&pp->sendbuf);
  s = Curl_dyn_ptr(&pp->sendbuf);
  Curl_pp_init(data, pp);

  result = Curl_convert_to_network(data, s, write_len);
  /* Curl_convert_to_network calls failf if unsuccessful */
  if(result)
    return result;

#ifdef HAVE_GSSAPI
  conn->data_prot = PROT_CMD;
#endif
  result = Curl_write(data, conn->sock[FIRSTSOCKET], s, write_len,
                      &bytes_written);
  if(result)
    return result;
#ifdef HAVE_GSSAPI
  data_sec = conn->data_prot;
  DEBUGASSERT(data_sec > PROT_NONE && data_sec < PROT_LAST);
  conn->data_prot = data_sec;
#endif

  Curl_debug(data, CURLINFO_HEADER_OUT, s, (size_t)bytes_written);

  if(bytes_written != (ssize_t)write_len) {
    /* the whole chunk was not sent, keep it around and adjust sizes */
    pp->sendthis = s;
    pp->sendsize = write_len;
    pp->sendleft = write_len - bytes_written;
  }
  else {
    pp->sendthis = NULL;
    pp->sendleft = pp->sendsize = 0;
    pp->response = Curl_now();
  }

  return CURLE_OK;
}


/***********************************************************************
 *
 * Curl_pp_sendf()
 *
 * Send the formatted string as a command to a pingpong server. Note that
 * the string should not have any CRLF appended, as this function will
 * append the necessary things itself.
 *
 * made to never block
 */
CURLcode Curl_pp_sendf(struct Curl_easy *data, struct pingpong *pp,
                       const char *fmt, ...)
{
  CURLcode result;
  va_list ap;
  va_start(ap, fmt);

  result = Curl_pp_vsendf(data, pp, fmt, ap);

  va_end(ap);

  return result;
}

/*
 * Curl_pp_readresp()
 *
 * Reads a piece of a server response.
 */
CURLcode Curl_pp_readresp(struct Curl_easy *data,
                          curl_socket_t sockfd,
                          struct pingpong *pp,
                          int *code, /* return the server code if done */
                          size_t *size) /* size of the response */
{
  ssize_t perline; /* count bytes per line */
  bool keepon = TRUE;
  ssize_t gotbytes;
  char *ptr;
  struct connectdata *conn = data->conn;
  char * const buf = data->state.buffer;
  CURLcode result = CURLE_OK;

  *code = 0; /* 0 for errors or not done */
  *size = 0;

  ptr = buf + pp->nread_resp;

  /* number of bytes in the current line, so far */
  perline = (ssize_t)(ptr-pp->linestart_resp);

  while((pp->nread_resp < (size_t)data->set.buffer_size) &&
        (keepon && !result)) {

    if(pp->cache) {
      /* we had data in the "cache", copy that instead of doing an actual
       * read
       *
       * pp->cache_size is cast to ssize_t here.  This should be safe, because
       * it would have been populated with something of size int to begin
       * with, even though its datatype may be larger than an int.
       */
      if((ptr + pp->cache_size) > (buf + data->set.buffer_size + 1)) {
        failf(data, "cached response data too big to handle");
        return CURLE_RECV_ERROR;
      }
      memcpy(ptr, pp->cache, pp->cache_size);
      gotbytes = (ssize_t)pp->cache_size;
      free(pp->cache);    /* free the cache */
      pp->cache = NULL;   /* clear the pointer */
      pp->cache_size = 0; /* zero the size just in case */
    }
    else {
#ifdef HAVE_GSSAPI
      enum protection_level prot = conn->data_prot;
      conn->data_prot = PROT_CLEAR;
#endif
      DEBUGASSERT((ptr + data->set.buffer_size - pp->nread_resp) <=
                  (buf + data->set.buffer_size + 1));
      result = Curl_read(data, sockfd, ptr,
                         data->set.buffer_size - pp->nread_resp,
                         &gotbytes);
#ifdef HAVE_GSSAPI
      DEBUGASSERT(prot  > PROT_NONE && prot < PROT_LAST);
      conn->data_prot = prot;
#endif
      if(result == CURLE_AGAIN)
        return CURLE_OK; /* return */

      if(!result && (gotbytes > 0))
        /* convert from the network encoding */
        result = Curl_convert_from_network(data, ptr, gotbytes);
      /* Curl_convert_from_network calls failf if unsuccessful */

      if(result)
        /* Set outer result variable to this error. */
        keepon = FALSE;
    }

    if(!keepon)
      ;
    else if(gotbytes <= 0) {
      keepon = FALSE;
      result = CURLE_RECV_ERROR;
      failf(data, "response reading failed");
    }
    else {
      /* we got a whole chunk of data, which can be anything from one
       * byte to a set of lines and possible just a piece of the last
       * line */
      ssize_t i;
      ssize_t clipamount = 0;
      bool restart = FALSE;

      data->req.headerbytecount += (long)gotbytes;

      pp->nread_resp += gotbytes;
      for(i = 0; i < gotbytes; ptr++, i++) {
        perline++;
        if(*ptr == '\n') {
          /* a newline is CRLF in pp-talk, so the CR is ignored as
             the line isn't really terminated until the LF comes */

          /* output debug output if that is requested */
#ifdef HAVE_GSSAPI
          if(!conn->sec_complete)
#endif
            Curl_debug(data, CURLINFO_HEADER_IN,
                       pp->linestart_resp, (size_t)perline);

          /*
           * We pass all response-lines to the callback function registered
           * for "headers". The response lines can be seen as a kind of
           * headers.
           */
          result = Curl_client_write(data, CLIENTWRITE_HEADER,
                                     pp->linestart_resp, perline);
          if(result)
            return result;

          if(pp->endofresp(data, conn, pp->linestart_resp, perline, code)) {
            /* This is the end of the last line, copy the last line to the
               start of the buffer and null-terminate, for old times sake */
            size_t n = ptr - pp->linestart_resp;
            memmove(buf, pp->linestart_resp, n);
            buf[n] = 0; /* null-terminate */
            keepon = FALSE;
            pp->linestart_resp = ptr + 1; /* advance pointer */
            i++; /* skip this before getting out */

            *size = pp->nread_resp; /* size of the response */
            pp->nread_resp = 0; /* restart */
            break;
          }
          perline = 0; /* line starts over here */
          pp->linestart_resp = ptr + 1;
        }
      }

      if(!keepon && (i != gotbytes)) {
        /* We found the end of the response lines, but we didn't parse the
           full chunk of data we have read from the server. We therefore need
           to store the rest of the data to be checked on the next invoke as
           it may actually contain another end of response already! */
        clipamount = gotbytes - i;
        restart = TRUE;
        DEBUGF(infof(data, "Curl_pp_readresp_ %d bytes of trailing "
                     "server response left",
                     (int)clipamount));
      }
      else if(keepon) {

        if((perline == gotbytes) && (gotbytes > data->set.buffer_size/2)) {
          /* We got an excessive line without newlines and we need to deal
             with it. We keep the first bytes of the line then we throw
             away the rest. */
          infof(data, "Excessive server response line length received, "
                "%zd bytes. Stripping", gotbytes);
          restart = TRUE;

          /* we keep 40 bytes since all our pingpong protocols are only
             interested in the first piece */
          clipamount = 40;
        }
        else if(pp->nread_resp > (size_t)data->set.buffer_size/2) {
          /* We got a large chunk of data and there's potentially still
             trailing data to take care of, so we put any such part in the
             "cache", clear the buffer to make space and restart. */
          clipamount = perline;
          restart = TRUE;
        }
      }
      else if(i == gotbytes)
        restart = TRUE;

      if(clipamount) {
        pp->cache_size = clipamount;
        pp->cache = malloc(pp->cache_size);
        if(pp->cache)
          memcpy(pp->cache, pp->linestart_resp, pp->cache_size);
        else
          return CURLE_OUT_OF_MEMORY;
      }
      if(restart) {
        /* now reset a few variables to start over nicely from the start of
           the big buffer */
        pp->nread_resp = 0; /* start over from scratch in the buffer */
        ptr = pp->linestart_resp = buf;
        perline = 0;
      }

    } /* there was data */

  } /* while there's buffer left and loop is requested */

  pp->pending_resp = FALSE;

  return result;
}

int Curl_pp_getsock(struct Curl_easy *data,
                    struct pingpong *pp, curl_socket_t *socks)
{
  struct connectdata *conn = data->conn;
  socks[0] = conn->sock[FIRSTSOCKET];

  if(pp->sendleft) {
    /* write mode */
    return GETSOCK_WRITESOCK(0);
  }

  /* read mode */
  return GETSOCK_READSOCK(0);
}

CURLcode Curl_pp_flushsend(struct Curl_easy *data,
                           struct pingpong *pp)
{
  /* we have a piece of a command still left to send */
  struct connectdata *conn = data->conn;
  ssize_t written;
  curl_socket_t sock = conn->sock[FIRSTSOCKET];
  CURLcode result = Curl_write(data, sock, pp->sendthis + pp->sendsize -
                               pp->sendleft, pp->sendleft, &written);
  if(result)
    return result;

  if(written != (ssize_t)pp->sendleft) {
    /* only a fraction was sent */
    pp->sendleft -= written;
  }
  else {
    pp->sendthis = NULL;
    pp->sendleft = pp->sendsize = 0;
    pp->response = Curl_now();
  }
  return CURLE_OK;
}

CURLcode Curl_pp_disconnect(struct pingpong *pp)
{
  Curl_dyn_free(&pp->sendbuf);
  Curl_safefree(pp->cache);
  return CURLE_OK;
}

bool Curl_pp_moredata(struct pingpong *pp)
{
  return (!pp->sendleft && pp->cache && pp->nread_resp < pp->cache_size) ?
    TRUE : FALSE;
}

#endif
