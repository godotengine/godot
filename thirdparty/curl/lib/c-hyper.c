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
 * are also available at https://curl.haxx.se/docs/copyright.html.
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

#if !defined(CURL_DISABLE_HTTP) && defined(USE_HYPER)

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

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#include <hyper.h>
#include "urldata.h"
#include "sendf.h"
#include "transfer.h"
#include "multiif.h"
#include "progress.h"
#include "content_encoding.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

size_t Curl_hyper_recv(void *userp, hyper_context *ctx,
                       uint8_t *buf, size_t buflen)
{
  struct Curl_easy *data = userp;
  struct connectdata *conn = data->conn;
  CURLcode result;
  ssize_t nread;
  DEBUGASSERT(conn);
  (void)ctx;

  result = Curl_read(data, conn->sockfd, (char *)buf, buflen, &nread);
  if(result == CURLE_AGAIN) {
    /* would block, register interest */
    if(data->hyp.read_waker)
      hyper_waker_free(data->hyp.read_waker);
    data->hyp.read_waker = hyper_context_waker(ctx);
    if(!data->hyp.read_waker) {
      failf(data, "Couldn't make the read hyper_context_waker");
      return HYPER_IO_ERROR;
    }
    return HYPER_IO_PENDING;
  }
  else if(result) {
    failf(data, "Curl_read failed");
    return HYPER_IO_ERROR;
  }
  return (size_t)nread;
}

size_t Curl_hyper_send(void *userp, hyper_context *ctx,
                       const uint8_t *buf, size_t buflen)
{
  struct Curl_easy *data = userp;
  struct connectdata *conn = data->conn;
  CURLcode result;
  ssize_t nwrote;

  result = Curl_write(data, conn->sockfd, (void *)buf, buflen, &nwrote);
  if(result == CURLE_AGAIN) {
    /* would block, register interest */
    if(data->hyp.write_waker)
      hyper_waker_free(data->hyp.write_waker);
    data->hyp.write_waker = hyper_context_waker(ctx);
    if(!data->hyp.write_waker) {
      failf(data, "Couldn't make the write hyper_context_waker");
      return HYPER_IO_ERROR;
    }
    return HYPER_IO_PENDING;
  }
  else if(result) {
    failf(data, "Curl_write failed");
    return HYPER_IO_ERROR;
  }
  return (size_t)nwrote;
}

static int hyper_each_header(void *userdata,
                             const uint8_t *name,
                             size_t name_len,
                             const uint8_t *value,
                             size_t value_len)
{
  struct Curl_easy *data = (struct Curl_easy *)userdata;
  size_t len;
  char *headp;
  CURLcode result;
  int writetype;

  if(name_len + value_len + 2 > CURL_MAX_HTTP_HEADER) {
    failf(data, "Too long response header");
    data->state.hresult = CURLE_OUT_OF_MEMORY;
    return HYPER_ITER_BREAK;
  }

  if(!data->req.bytecount)
    Curl_pgrsTime(data, TIMER_STARTTRANSFER);

  Curl_dyn_reset(&data->state.headerb);
  if(name_len) {
    if(Curl_dyn_addf(&data->state.headerb, "%.*s: %.*s\r\n",
                     (int) name_len, name, (int) value_len, value))
      return HYPER_ITER_BREAK;
  }
  else {
    if(Curl_dyn_add(&data->state.headerb, "\r\n"))
      return HYPER_ITER_BREAK;
  }
  len = Curl_dyn_len(&data->state.headerb);
  headp = Curl_dyn_ptr(&data->state.headerb);

  result = Curl_http_header(data, data->conn, headp);
  if(result) {
    data->state.hresult = result;
    return HYPER_ITER_BREAK;
  }

  Curl_debug(data, CURLINFO_HEADER_IN, headp, len);

  if(!data->state.hconnect || !data->set.suppress_connect_headers) {
    writetype = CLIENTWRITE_HEADER;
    if(data->set.include_header)
      writetype |= CLIENTWRITE_BODY;
    result = Curl_client_write(data, writetype, headp, len);
    if(result) {
      data->state.hresult = CURLE_ABORTED_BY_CALLBACK;
      return HYPER_ITER_BREAK;
    }
  }

  data->info.header_size += (long)len;
  data->req.headerbytecount += (long)len;
  return HYPER_ITER_CONTINUE;
}

static int hyper_body_chunk(void *userdata, const hyper_buf *chunk)
{
  char *buf = (char *)hyper_buf_bytes(chunk);
  size_t len = hyper_buf_len(chunk);
  struct Curl_easy *data = (struct Curl_easy *)userdata;
  struct SingleRequest *k = &data->req;
  CURLcode result = CURLE_OK;

  if(0 == k->bodywrites++) {
    bool done = FALSE;
#if defined(USE_NTLM)
    struct connectdata *conn = data->conn;
    if(conn->bits.close &&
       (((data->req.httpcode == 401) &&
         (conn->http_ntlm_state == NTLMSTATE_TYPE2)) ||
        ((data->req.httpcode == 407) &&
         (conn->proxy_ntlm_state == NTLMSTATE_TYPE2)))) {
      infof(data, "Connection closed while negotiating NTLM");
      data->state.authproblem = TRUE;
      Curl_safefree(data->req.newurl);
    }
#endif
    if(data->state.expect100header) {
      Curl_expire_done(data, EXPIRE_100_TIMEOUT);
      if(data->req.httpcode < 400) {
        k->exp100 = EXP100_SEND_DATA;
        if(data->hyp.exp100_waker) {
          hyper_waker_wake(data->hyp.exp100_waker);
          data->hyp.exp100_waker = NULL;
        }
      }
      else { /* >= 4xx */
        k->exp100 = EXP100_FAILED;
      }
    }
    if(data->state.hconnect && (data->req.httpcode/100 != 2) &&
       data->state.authproxy.done) {
      done = TRUE;
      result = CURLE_OK;
    }
    else
      result = Curl_http_firstwrite(data, data->conn, &done);
    if(result || done) {
      infof(data, "Return early from hyper_body_chunk");
      data->state.hresult = result;
      return HYPER_ITER_BREAK;
    }
  }
  if(k->ignorebody)
    return HYPER_ITER_CONTINUE;
  if(0 == len)
    return HYPER_ITER_CONTINUE;
  Curl_debug(data, CURLINFO_DATA_IN, buf, len);
  if(!data->set.http_ce_skip && k->writer_stack)
    /* content-encoded data */
    result = Curl_unencode_write(data, k->writer_stack, buf, len);
  else
    result = Curl_client_write(data, CLIENTWRITE_BODY, buf, len);

  if(result) {
    data->state.hresult = result;
    return HYPER_ITER_BREAK;
  }

  data->req.bytecount += len;
  Curl_pgrsSetDownloadCounter(data, data->req.bytecount);
  return HYPER_ITER_CONTINUE;
}

/*
 * Hyper does not consider the status line, the first line in a HTTP/1
 * response, to be a header. The libcurl API does. This function sends the
 * status line in the header callback. */
static CURLcode status_line(struct Curl_easy *data,
                            struct connectdata *conn,
                            uint16_t http_status,
                            int http_version,
                            const uint8_t *reason, size_t rlen)
{
  CURLcode result;
  size_t len;
  const char *vstr;
  int writetype;
  vstr = http_version == HYPER_HTTP_VERSION_1_1 ? "1.1" :
    (http_version == HYPER_HTTP_VERSION_2 ? "2" : "1.0");
  conn->httpversion =
    http_version == HYPER_HTTP_VERSION_1_1 ? 11 :
    (http_version == HYPER_HTTP_VERSION_2 ? 20 : 10);
  if(http_version == HYPER_HTTP_VERSION_1_0)
    data->state.httpwant = CURL_HTTP_VERSION_1_0;

  if(data->state.hconnect)
    /* CONNECT */
    data->info.httpproxycode = http_status;

  /* We need to set 'httpcodeq' for functions that check the response code in
     a single place. */
  data->req.httpcode = http_status;

  result = Curl_http_statusline(data, conn);
  if(result)
    return result;

  Curl_dyn_reset(&data->state.headerb);

  result = Curl_dyn_addf(&data->state.headerb, "HTTP/%s %03d %.*s\r\n",
                         vstr,
                         (int)http_status,
                         (int)rlen, reason);
  if(result)
    return result;
  len = Curl_dyn_len(&data->state.headerb);
  Curl_debug(data, CURLINFO_HEADER_IN, Curl_dyn_ptr(&data->state.headerb),
             len);

  if(!data->state.hconnect || !data->set.suppress_connect_headers) {
    writetype = CLIENTWRITE_HEADER;
    if(data->set.include_header)
      writetype |= CLIENTWRITE_BODY;
    result = Curl_client_write(data, writetype,
                               Curl_dyn_ptr(&data->state.headerb), len);
    if(result) {
      data->state.hresult = CURLE_ABORTED_BY_CALLBACK;
      return HYPER_ITER_BREAK;
    }
  }
  data->info.header_size += (long)len;
  data->req.headerbytecount += (long)len;
  data->req.httpcode = http_status;
  return CURLE_OK;
}

/*
 * Hyper does not pass on the last empty response header. The libcurl API
 * does. This function sends an empty header in the header callback.
 */
static CURLcode empty_header(struct Curl_easy *data)
{
  CURLcode result = Curl_http_size(data);
  if(!result) {
    result = hyper_each_header(data, NULL, 0, NULL, 0) ?
      CURLE_WRITE_ERROR : CURLE_OK;
    if(result)
      failf(data, "hyperstream: couldn't pass blank header");
  }
  return result;
}

CURLcode Curl_hyper_stream(struct Curl_easy *data,
                           struct connectdata *conn,
                           int *didwhat,
                           bool *done,
                           int select_res)
{
  hyper_response *resp = NULL;
  uint16_t http_status;
  int http_version;
  hyper_headers *headers = NULL;
  hyper_body *resp_body = NULL;
  struct hyptransfer *h = &data->hyp;
  hyper_task *task;
  hyper_task *foreach;
  hyper_error *hypererr = NULL;
  const uint8_t *reasonp;
  size_t reason_len;
  CURLcode result = CURLE_OK;
  struct SingleRequest *k = &data->req;
  (void)conn;

  if(k->exp100 > EXP100_SEND_DATA) {
    struct curltime now = Curl_now();
    timediff_t ms = Curl_timediff(now, k->start100);
    if(ms >= data->set.expect_100_timeout) {
      /* we've waited long enough, continue anyway */
      k->exp100 = EXP100_SEND_DATA;
      k->keepon |= KEEP_SEND;
      Curl_expire_done(data, EXPIRE_100_TIMEOUT);
      infof(data, "Done waiting for 100-continue");
      if(data->hyp.exp100_waker) {
        hyper_waker_wake(data->hyp.exp100_waker);
        data->hyp.exp100_waker = NULL;
      }
    }
  }

  if(select_res & CURL_CSELECT_IN) {
    if(h->read_waker)
      hyper_waker_wake(h->read_waker);
    h->read_waker = NULL;
  }
  if(select_res & CURL_CSELECT_OUT) {
    if(h->write_waker)
      hyper_waker_wake(h->write_waker);
    h->write_waker = NULL;
  }

  *done = FALSE;
  do {
    hyper_task_return_type t;
    task = hyper_executor_poll(h->exec);
    if(!task) {
      *didwhat = KEEP_RECV;
      break;
    }
    t = hyper_task_type(task);
    switch(t) {
    case HYPER_TASK_ERROR:
      hypererr = hyper_task_value(task);
      break;
    case HYPER_TASK_RESPONSE:
      resp = hyper_task_value(task);
      break;
    default:
      break;
    }
    hyper_task_free(task);

    if(t == HYPER_TASK_ERROR) {
      if(data->state.hresult) {
        /* override Hyper's view, might not even be an error */
        result = data->state.hresult;
        infof(data, "hyperstream is done (by early callback)");
      }
      else {
        uint8_t errbuf[256];
        size_t errlen = hyper_error_print(hypererr, errbuf, sizeof(errbuf));
        hyper_code code = hyper_error_code(hypererr);
        failf(data, "Hyper: [%d] %.*s", (int)code, (int)errlen, errbuf);
        if(code == HYPERE_ABORTED_BY_CALLBACK)
          result = CURLE_OK;
        else if((code == HYPERE_UNEXPECTED_EOF) && !data->req.bytecount)
          result = CURLE_GOT_NOTHING;
        else if(code == HYPERE_INVALID_PEER_MESSAGE)
          result = CURLE_UNSUPPORTED_PROTOCOL; /* maybe */
        else
          result = CURLE_RECV_ERROR;
      }
      *done = TRUE;
      hyper_error_free(hypererr);
      break;
    }
    else if(h->endtask == task) {
      /* end of transfer */
      *done = TRUE;
      infof(data, "hyperstream is done!");
      if(!k->bodywrites) {
        /* hyper doesn't always call the body write callback */
        bool stilldone;
        result = Curl_http_firstwrite(data, data->conn, &stilldone);
      }
      break;
    }
    else if(t != HYPER_TASK_RESPONSE) {
      *didwhat = KEEP_RECV;
      break;
    }
    /* HYPER_TASK_RESPONSE */

    *didwhat = KEEP_RECV;
    if(!resp) {
      failf(data, "hyperstream: couldn't get response");
      return CURLE_RECV_ERROR;
    }

    http_status = hyper_response_status(resp);
    http_version = hyper_response_version(resp);
    reasonp = hyper_response_reason_phrase(resp);
    reason_len = hyper_response_reason_phrase_len(resp);

    result = status_line(data, conn,
                         http_status, http_version, reasonp, reason_len);
    if(result)
      break;

    headers = hyper_response_headers(resp);
    if(!headers) {
      failf(data, "hyperstream: couldn't get response headers");
      result = CURLE_RECV_ERROR;
      break;
    }

    /* the headers are already received */
    hyper_headers_foreach(headers, hyper_each_header, data);
    if(data->state.hresult) {
      result = data->state.hresult;
      break;
    }

    result = empty_header(data);
    if(result)
      break;

    /* Curl_http_auth_act() checks what authentication methods that are
     * available and decides which one (if any) to use. It will set 'newurl'
     * if an auth method was picked. */
    result = Curl_http_auth_act(data);
    if(result)
      break;

    resp_body = hyper_response_body(resp);
    if(!resp_body) {
      failf(data, "hyperstream: couldn't get response body");
      result = CURLE_RECV_ERROR;
      break;
    }
    foreach = hyper_body_foreach(resp_body, hyper_body_chunk, data);
    if(!foreach) {
      failf(data, "hyperstream: body foreach failed");
      result = CURLE_OUT_OF_MEMORY;
      break;
    }
    DEBUGASSERT(hyper_task_type(foreach) == HYPER_TASK_EMPTY);
    if(HYPERE_OK != hyper_executor_push(h->exec, foreach)) {
      failf(data, "Couldn't hyper_executor_push the body-foreach");
      result = CURLE_OUT_OF_MEMORY;
      break;
    }
    h->endtask = foreach;

    hyper_response_free(resp);
    resp = NULL;
  } while(1);
  if(resp)
    hyper_response_free(resp);
  return result;
}

static CURLcode debug_request(struct Curl_easy *data,
                              const char *method,
                              const char *path,
                              bool h2)
{
  char *req = aprintf("%s %s HTTP/%s\r\n", method, path,
                      h2?"2":"1.1");
  if(!req)
    return CURLE_OUT_OF_MEMORY;
  Curl_debug(data, CURLINFO_HEADER_OUT, req, strlen(req));
  free(req);
  return CURLE_OK;
}

/*
 * Given a full header line "name: value" (optional CRLF in the input, should
 * be in the output), add to Hyper and send to the debug callback.
 *
 * Supports multiple headers.
 */

CURLcode Curl_hyper_header(struct Curl_easy *data, hyper_headers *headers,
                           const char *line)
{
  const char *p;
  const char *n;
  size_t nlen;
  const char *v;
  size_t vlen;
  bool newline = TRUE;
  int numh = 0;

  if(!line)
    return CURLE_OK;
  n = line;
  do {
    size_t linelen = 0;

    p = strchr(n, ':');
    if(!p)
      /* this is fine if we already added at least one header */
      return numh ? CURLE_OK : CURLE_BAD_FUNCTION_ARGUMENT;
    nlen = p - n;
    p++; /* move past the colon */
    while(*p == ' ')
      p++;
    v = p;
    p = strchr(v, '\r');
    if(!p) {
      p = strchr(v, '\n');
      if(p)
        linelen = 1; /* LF only */
      else {
        p = strchr(v, '\0');
        newline = FALSE; /* no newline */
      }
    }
    else
      linelen = 2; /* CRLF ending */
    linelen += (p - n);
    vlen = p - v;

    if(HYPERE_OK != hyper_headers_add(headers, (uint8_t *)n, nlen,
                                      (uint8_t *)v, vlen)) {
      failf(data, "hyper refused to add header '%s'", line);
      return CURLE_OUT_OF_MEMORY;
    }
    if(data->set.verbose) {
      char *ptr = NULL;
      if(!newline) {
        ptr = aprintf("%.*s\r\n", (int)linelen, line);
        if(!ptr)
          return CURLE_OUT_OF_MEMORY;
        Curl_debug(data, CURLINFO_HEADER_OUT, ptr, linelen + 2);
        free(ptr);
      }
      else
        Curl_debug(data, CURLINFO_HEADER_OUT, (char *)n, linelen);
    }
    numh++;
    n += linelen;
  } while(newline);
  return CURLE_OK;
}

static CURLcode request_target(struct Curl_easy *data,
                               struct connectdata *conn,
                               const char *method,
                               bool h2,
                               hyper_request *req)
{
  CURLcode result;
  struct dynbuf r;

  Curl_dyn_init(&r, DYN_HTTP_REQUEST);

  result = Curl_http_target(data, conn, &r);
  if(result)
    return result;

  if(h2 && hyper_request_set_uri_parts(req,
                                       /* scheme */
                                       (uint8_t *)data->state.up.scheme,
                                       strlen(data->state.up.scheme),
                                       /* authority */
                                       (uint8_t *)conn->host.name,
                                       strlen(conn->host.name),
                                       /* path_and_query */
                                       (uint8_t *)Curl_dyn_uptr(&r),
                                       Curl_dyn_len(&r))) {
    failf(data, "error setting uri parts to hyper");
    result = CURLE_OUT_OF_MEMORY;
  }
  else if(!h2 && hyper_request_set_uri(req, (uint8_t *)Curl_dyn_uptr(&r),
                                       Curl_dyn_len(&r))) {
    failf(data, "error setting uri to hyper");
    result = CURLE_OUT_OF_MEMORY;
  }
  else
    result = debug_request(data, method, Curl_dyn_ptr(&r), h2);

  Curl_dyn_free(&r);

  return result;
}

static int uploadpostfields(void *userdata, hyper_context *ctx,
                            hyper_buf **chunk)
{
  struct Curl_easy *data = (struct Curl_easy *)userdata;
  (void)ctx;
  if(data->req.exp100 > EXP100_SEND_DATA) {
    if(data->req.exp100 == EXP100_FAILED)
      return HYPER_POLL_ERROR;

    /* still waiting confirmation */
    if(data->hyp.exp100_waker)
      hyper_waker_free(data->hyp.exp100_waker);
    data->hyp.exp100_waker = hyper_context_waker(ctx);
    return HYPER_POLL_PENDING;
  }
  if(data->req.upload_done)
    *chunk = NULL; /* nothing more to deliver */
  else {
    /* send everything off in a single go */
    hyper_buf *copy = hyper_buf_copy(data->set.postfields,
                                     (size_t)data->req.p.http->postsize);
    if(copy)
      *chunk = copy;
    else {
      data->state.hresult = CURLE_OUT_OF_MEMORY;
      return HYPER_POLL_ERROR;
    }
    /* increasing the writebytecount here is a little premature but we
       don't know exactly when the body is sent*/
    data->req.writebytecount += (size_t)data->req.p.http->postsize;
    Curl_pgrsSetUploadCounter(data, data->req.writebytecount);
    data->req.upload_done = TRUE;
  }
  return HYPER_POLL_READY;
}

static int uploadstreamed(void *userdata, hyper_context *ctx,
                          hyper_buf **chunk)
{
  size_t fillcount;
  struct Curl_easy *data = (struct Curl_easy *)userdata;
  CURLcode result;
  (void)ctx;

  if(data->req.exp100 > EXP100_SEND_DATA) {
    if(data->req.exp100 == EXP100_FAILED)
      return HYPER_POLL_ERROR;

    /* still waiting confirmation */
    if(data->hyp.exp100_waker)
      hyper_waker_free(data->hyp.exp100_waker);
    data->hyp.exp100_waker = hyper_context_waker(ctx);
    return HYPER_POLL_PENDING;
  }

  result = Curl_fillreadbuffer(data, data->set.upload_buffer_size, &fillcount);
  if(result) {
    data->state.hresult = result;
    return HYPER_POLL_ERROR;
  }
  if(!fillcount)
    /* done! */
    *chunk = NULL;
  else {
    hyper_buf *copy = hyper_buf_copy((uint8_t *)data->state.ulbuf, fillcount);
    if(copy)
      *chunk = copy;
    else {
      data->state.hresult = CURLE_OUT_OF_MEMORY;
      return HYPER_POLL_ERROR;
    }
    /* increasing the writebytecount here is a little premature but we
       don't know exactly when the body is sent*/
    data->req.writebytecount += fillcount;
    Curl_pgrsSetUploadCounter(data, fillcount);
  }
  return HYPER_POLL_READY;
}

/*
 * bodysend() sets up headers in the outgoing request for a HTTP transfer that
 * sends a body
 */

static CURLcode bodysend(struct Curl_easy *data,
                         struct connectdata *conn,
                         hyper_headers *headers,
                         hyper_request *hyperreq,
                         Curl_HttpReq httpreq)
{
  struct HTTP *http = data->req.p.http;
  CURLcode result = CURLE_OK;
  struct dynbuf req;
  if((httpreq == HTTPREQ_GET) || (httpreq == HTTPREQ_HEAD))
    Curl_pgrsSetUploadSize(data, 0); /* no request body */
  else {
    hyper_body *body;
    Curl_dyn_init(&req, DYN_HTTP_REQUEST);
    result = Curl_http_bodysend(data, conn, &req, httpreq);

    if(!result)
      result = Curl_hyper_header(data, headers, Curl_dyn_ptr(&req));

    Curl_dyn_free(&req);

    body = hyper_body_new();
    hyper_body_set_userdata(body, data);
    if(data->set.postfields)
      hyper_body_set_data_func(body, uploadpostfields);
    else {
      result = Curl_get_upload_buffer(data);
      if(result)
        return result;
      /* init the "upload from here" pointer */
      data->req.upload_fromhere = data->state.ulbuf;
      hyper_body_set_data_func(body, uploadstreamed);
    }
    if(HYPERE_OK != hyper_request_set_body(hyperreq, body)) {
      /* fail */
      hyper_body_free(body);
      result = CURLE_OUT_OF_MEMORY;
    }
  }
  http->sending = HTTPSEND_BODY;
  return result;
}

static CURLcode cookies(struct Curl_easy *data,
                        struct connectdata *conn,
                        hyper_headers *headers)
{
  struct dynbuf req;
  CURLcode result;
  Curl_dyn_init(&req, DYN_HTTP_REQUEST);

  result = Curl_http_cookies(data, conn, &req);
  if(!result)
    result = Curl_hyper_header(data, headers, Curl_dyn_ptr(&req));
  Curl_dyn_free(&req);
  return result;
}

/* called on 1xx responses */
static void http1xx_cb(void *arg, struct hyper_response *resp)
{
  struct Curl_easy *data = (struct Curl_easy *)arg;
  hyper_headers *headers = NULL;
  CURLcode result = CURLE_OK;
  uint16_t http_status;
  int http_version;
  const uint8_t *reasonp;
  size_t reason_len;

  infof(data, "Got HTTP 1xx informational");

  http_status = hyper_response_status(resp);
  http_version = hyper_response_version(resp);
  reasonp = hyper_response_reason_phrase(resp);
  reason_len = hyper_response_reason_phrase_len(resp);

  result = status_line(data, data->conn,
                       http_status, http_version, reasonp, reason_len);
  if(!result) {
    headers = hyper_response_headers(resp);
    if(!headers) {
      failf(data, "hyperstream: couldn't get 1xx response headers");
      result = CURLE_RECV_ERROR;
    }
  }
  data->state.hresult = result;

  if(!result) {
    /* the headers are already received */
    hyper_headers_foreach(headers, hyper_each_header, data);
    /* this callback also sets data->state.hresult on error */

    if(empty_header(data))
      result = CURLE_OUT_OF_MEMORY;
  }

  if(data->state.hresult)
    infof(data, "ERROR in 1xx, bail out!");
}

/*
 * Curl_http() gets called from the generic multi_do() function when a HTTP
 * request is to be performed. This creates and sends a properly constructed
 * HTTP request.
 */
CURLcode Curl_http(struct Curl_easy *data, bool *done)
{
  struct connectdata *conn = data->conn;
  struct hyptransfer *h = &data->hyp;
  hyper_io *io = NULL;
  hyper_clientconn_options *options = NULL;
  hyper_task *task = NULL; /* for the handshake */
  hyper_task *sendtask = NULL; /* for the send */
  hyper_clientconn *client = NULL;
  hyper_request *req = NULL;
  hyper_headers *headers = NULL;
  hyper_task *handshake = NULL;
  CURLcode result;
  const char *p_accept; /* Accept: string */
  const char *method;
  Curl_HttpReq httpreq;
  bool h2 = FALSE;
  const char *te = NULL; /* transfer-encoding */
  hyper_code rc;

  /* Always consider the DO phase done after this function call, even if there
     may be parts of the request that is not yet sent, since we can deal with
     the rest of the request in the PERFORM phase. */
  *done = TRUE;

  infof(data, "Time for the Hyper dance");
  memset(h, 0, sizeof(struct hyptransfer));

  result = Curl_http_host(data, conn);
  if(result)
    return result;

  Curl_http_method(data, conn, &method, &httpreq);

  /* setup the authentication headers */
  {
    char *pq = NULL;
    if(data->state.up.query) {
      pq = aprintf("%s?%s", data->state.up.path, data->state.up.query);
      if(!pq)
        return CURLE_OUT_OF_MEMORY;
    }
    result = Curl_http_output_auth(data, conn, method, httpreq,
                                   (pq ? pq : data->state.up.path), FALSE);
    free(pq);
    if(result)
      return result;
  }

  result = Curl_http_resume(data, conn, httpreq);
  if(result)
    return result;

  result = Curl_http_range(data, httpreq);
  if(result)
    return result;

  result = Curl_http_useragent(data);
  if(result)
    return result;

  io = hyper_io_new();
  if(!io) {
    failf(data, "Couldn't create hyper IO");
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }
  /* tell Hyper how to read/write network data */
  hyper_io_set_userdata(io, data);
  hyper_io_set_read(io, Curl_hyper_recv);
  hyper_io_set_write(io, Curl_hyper_send);

  /* create an executor to poll futures */
  if(!h->exec) {
    h->exec = hyper_executor_new();
    if(!h->exec) {
      failf(data, "Couldn't create hyper executor");
      result = CURLE_OUT_OF_MEMORY;
      goto error;
    }
  }

  options = hyper_clientconn_options_new();
  if(!options) {
    failf(data, "Couldn't create hyper client options");
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }
  if(conn->negnpn == CURL_HTTP_VERSION_2) {
    hyper_clientconn_options_http2(options, 1);
    h2 = TRUE;
  }

  hyper_clientconn_options_exec(options, h->exec);

  /* "Both the `io` and the `options` are consumed in this function call" */
  handshake = hyper_clientconn_handshake(io, options);
  if(!handshake) {
    failf(data, "Couldn't create hyper client handshake");
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }
  io = NULL;
  options = NULL;

  if(HYPERE_OK != hyper_executor_push(h->exec, handshake)) {
    failf(data, "Couldn't hyper_executor_push the handshake");
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }
  handshake = NULL; /* ownership passed on */

  task = hyper_executor_poll(h->exec);
  if(!task) {
    failf(data, "Couldn't hyper_executor_poll the handshake");
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }

  client = hyper_task_value(task);
  hyper_task_free(task);

  req = hyper_request_new();
  if(!req) {
    failf(data, "Couldn't hyper_request_new");
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }

  if(!Curl_use_http_1_1plus(data, conn)) {
    if(HYPERE_OK != hyper_request_set_version(req,
                                              HYPER_HTTP_VERSION_1_0)) {
      failf(data, "error setting HTTP version");
      result = CURLE_OUT_OF_MEMORY;
      goto error;
    }
  }

  if(hyper_request_set_method(req, (uint8_t *)method, strlen(method))) {
    failf(data, "error setting method");
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }

  result = request_target(data, conn, method, h2, req);
  if(result)
    goto error;

  headers = hyper_request_headers(req);
  if(!headers) {
    failf(data, "hyper_request_headers");
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }

  rc = hyper_request_on_informational(req, http1xx_cb, data);
  if(rc) {
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }

  result = Curl_http_body(data, conn, httpreq, &te);
  if(result)
    goto error;

  if(!h2) {
    if(data->state.aptr.host) {
      result = Curl_hyper_header(data, headers, data->state.aptr.host);
      if(result)
        goto error;
    }
  }
  else {
    /* For HTTP/2, we show the Host: header as if we sent it, to make it look
       like for HTTP/1 but it isn't actually sent since :authority is then
       used. */
    result = Curl_debug(data, CURLINFO_HEADER_OUT, data->state.aptr.host,
                        strlen(data->state.aptr.host));
    if(result)
      goto error;
  }

  if(data->state.aptr.proxyuserpwd) {
    result = Curl_hyper_header(data, headers, data->state.aptr.proxyuserpwd);
    if(result)
      goto error;
  }

  if(data->state.aptr.userpwd) {
    result = Curl_hyper_header(data, headers, data->state.aptr.userpwd);
    if(result)
      goto error;
  }

  if((data->state.use_range && data->state.aptr.rangeline)) {
    result = Curl_hyper_header(data, headers, data->state.aptr.rangeline);
    if(result)
      goto error;
  }

  if(data->set.str[STRING_USERAGENT] &&
     *data->set.str[STRING_USERAGENT] &&
     data->state.aptr.uagent) {
    result = Curl_hyper_header(data, headers, data->state.aptr.uagent);
    if(result)
      goto error;
  }

  p_accept = Curl_checkheaders(data, "Accept")?NULL:"Accept: */*\r\n";
  if(p_accept) {
    result = Curl_hyper_header(data, headers, p_accept);
    if(result)
      goto error;
  }
  if(te) {
    result = Curl_hyper_header(data, headers, te);
    if(result)
      goto error;
  }

#ifndef CURL_DISABLE_PROXY
  if(conn->bits.httpproxy && !conn->bits.tunnel_proxy &&
     !Curl_checkheaders(data, "Proxy-Connection") &&
     !Curl_checkProxyheaders(data, conn, "Proxy-Connection")) {
    result = Curl_hyper_header(data, headers, "Proxy-Connection: Keep-Alive");
    if(result)
      goto error;
  }
#endif

  Curl_safefree(data->state.aptr.ref);
  if(data->state.referer && !Curl_checkheaders(data, "Referer")) {
    data->state.aptr.ref = aprintf("Referer: %s\r\n", data->state.referer);
    if(!data->state.aptr.ref)
      result = CURLE_OUT_OF_MEMORY;
    else
      result = Curl_hyper_header(data, headers, data->state.aptr.ref);
    if(result)
      goto error;
  }

  if(!Curl_checkheaders(data, "Accept-Encoding") &&
     data->set.str[STRING_ENCODING]) {
    Curl_safefree(data->state.aptr.accept_encoding);
    data->state.aptr.accept_encoding =
      aprintf("Accept-Encoding: %s\r\n", data->set.str[STRING_ENCODING]);
    if(!data->state.aptr.accept_encoding)
      result = CURLE_OUT_OF_MEMORY;
    else
      result = Curl_hyper_header(data, headers,
                                 data->state.aptr.accept_encoding);
    if(result)
      goto error;
  }
  else
    Curl_safefree(data->state.aptr.accept_encoding);

#ifdef HAVE_LIBZ
  /* we only consider transfer-encoding magic if libz support is built-in */
  result = Curl_transferencode(data);
  if(result)
    goto error;
  result = Curl_hyper_header(data, headers, data->state.aptr.te);
  if(result)
    goto error;
#endif

  result = cookies(data, conn, headers);
  if(result)
    goto error;

  result = Curl_add_timecondition(data, headers);
  if(result)
    goto error;

  result = Curl_add_custom_headers(data, FALSE, headers);
  if(result)
    goto error;

  result = bodysend(data, conn, headers, req, httpreq);
  if(result)
    goto error;

  result = Curl_debug(data, CURLINFO_HEADER_OUT, (char *)"\r\n", 2);
  if(result)
    goto error;

  data->req.upload_chunky = FALSE;
  sendtask = hyper_clientconn_send(client, req);
  if(!sendtask) {
    failf(data, "hyper_clientconn_send");
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }

  if(HYPERE_OK != hyper_executor_push(h->exec, sendtask)) {
    failf(data, "Couldn't hyper_executor_push the send");
    result = CURLE_OUT_OF_MEMORY;
    goto error;
  }

  hyper_clientconn_free(client);

  if((httpreq == HTTPREQ_GET) || (httpreq == HTTPREQ_HEAD)) {
    /* HTTP GET/HEAD download */
    Curl_pgrsSetUploadSize(data, 0); /* nothing */
    Curl_setup_transfer(data, FIRSTSOCKET, -1, TRUE, -1);
  }
  conn->datastream = Curl_hyper_stream;
  if(data->state.expect100header)
    /* Timeout count starts now since with Hyper we don't know exactly when
       the full request has been sent. */
    data->req.start100 = Curl_now();

  /* clear userpwd and proxyuserpwd to avoid re-using old credentials
   * from re-used connections */
  Curl_safefree(data->state.aptr.userpwd);
  Curl_safefree(data->state.aptr.proxyuserpwd);
  return CURLE_OK;
  error:
  DEBUGASSERT(result);
  if(io)
    hyper_io_free(io);

  if(options)
    hyper_clientconn_options_free(options);

  if(handshake)
    hyper_task_free(handshake);

  return result;
}

void Curl_hyper_done(struct Curl_easy *data)
{
  struct hyptransfer *h = &data->hyp;
  if(h->exec) {
    hyper_executor_free(h->exec);
    h->exec = NULL;
  }
  if(h->read_waker) {
    hyper_waker_free(h->read_waker);
    h->read_waker = NULL;
  }
  if(h->write_waker) {
    hyper_waker_free(h->write_waker);
    h->write_waker = NULL;
  }
  if(h->exp100_waker) {
    hyper_waker_free(h->exp100_waker);
    h->exp100_waker = NULL;
  }
}

#endif /* !defined(CURL_DISABLE_HTTP) && defined(USE_HYPER) */
