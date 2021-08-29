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

#include "http_proxy.h"

#if !defined(CURL_DISABLE_PROXY) && !defined(CURL_DISABLE_HTTP)

#include <curl/curl.h>
#ifdef USE_HYPER
#include <hyper.h>
#endif
#include "sendf.h"
#include "http.h"
#include "url.h"
#include "select.h"
#include "progress.h"
#include "non-ascii.h"
#include "connect.h"
#include "curlx.h"
#include "vtls/vtls.h"
#include "transfer.h"
#include "multiif.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/*
 * Perform SSL initialization for HTTPS proxy.  Sets
 * proxy_ssl_connected connection bit when complete.  Can be
 * called multiple times.
 */
static CURLcode https_proxy_connect(struct Curl_easy *data, int sockindex)
{
#ifdef USE_SSL
  struct connectdata *conn = data->conn;
  CURLcode result = CURLE_OK;
  DEBUGASSERT(conn->http_proxy.proxytype == CURLPROXY_HTTPS);
  if(!conn->bits.proxy_ssl_connected[sockindex]) {
    /* perform SSL initialization for this socket */
    result =
      Curl_ssl_connect_nonblocking(data, conn, TRUE, sockindex,
                                   &conn->bits.proxy_ssl_connected[sockindex]);
    if(result)
      /* a failed connection is marked for closure to prevent (bad) re-use or
         similar */
      connclose(conn, "TLS handshake failed");
  }
  return result;
#else
  (void) data;
  (void) sockindex;
  return CURLE_NOT_BUILT_IN;
#endif
}

CURLcode Curl_proxy_connect(struct Curl_easy *data, int sockindex)
{
  struct connectdata *conn = data->conn;
  if(conn->http_proxy.proxytype == CURLPROXY_HTTPS) {
    const CURLcode result = https_proxy_connect(data, sockindex);
    if(result)
      return result;
    if(!conn->bits.proxy_ssl_connected[sockindex])
      return result; /* wait for HTTPS proxy SSL initialization to complete */
  }

  if(conn->bits.tunnel_proxy && conn->bits.httpproxy) {
#ifndef CURL_DISABLE_PROXY
    /* for [protocol] tunneled through HTTP proxy */
    const char *hostname;
    int remote_port;
    CURLcode result;

    /* We want "seamless" operations through HTTP proxy tunnel */

    /* for the secondary socket (FTP), use the "connect to host"
     * but ignore the "connect to port" (use the secondary port)
     */

    if(conn->bits.conn_to_host)
      hostname = conn->conn_to_host.name;
    else if(sockindex == SECONDARYSOCKET)
      hostname = conn->secondaryhostname;
    else
      hostname = conn->host.name;

    if(sockindex == SECONDARYSOCKET)
      remote_port = conn->secondary_port;
    else if(conn->bits.conn_to_port)
      remote_port = conn->conn_to_port;
    else
      remote_port = conn->remote_port;

    result = Curl_proxyCONNECT(data, sockindex, hostname, remote_port);
    if(CURLE_OK != result)
      return result;
    Curl_safefree(data->state.aptr.proxyuserpwd);
#else
    return CURLE_NOT_BUILT_IN;
#endif
  }
  /* no HTTP tunnel proxy, just return */
  return CURLE_OK;
}

bool Curl_connect_complete(struct connectdata *conn)
{
  return !conn->connect_state ||
    (conn->connect_state->tunnel_state >= TUNNEL_COMPLETE);
}

bool Curl_connect_ongoing(struct connectdata *conn)
{
  return conn->connect_state &&
    (conn->connect_state->tunnel_state <= TUNNEL_COMPLETE);
}

/* when we've sent a CONNECT to a proxy, we should rather either wait for the
   socket to become readable to be able to get the response headers or if
   we're still sending the request, wait for write. */
int Curl_connect_getsock(struct connectdata *conn)
{
  struct HTTP *http;
  DEBUGASSERT(conn);
  DEBUGASSERT(conn->connect_state);
  http = &conn->connect_state->http_proxy;

  if(http->sending == HTTPSEND_REQUEST)
    return GETSOCK_WRITESOCK(0);

  return GETSOCK_READSOCK(0);
}

static CURLcode connect_init(struct Curl_easy *data, bool reinit)
{
  struct http_connect_state *s;
  struct connectdata *conn = data->conn;
  if(conn->handler->flags & PROTOPT_NOTCPPROXY) {
    failf(data, "%s cannot be done over CONNECT", conn->handler->scheme);
    return CURLE_UNSUPPORTED_PROTOCOL;
  }
  if(!reinit) {
    CURLcode result;
    DEBUGASSERT(!conn->connect_state);
    /* we might need the upload buffer for streaming a partial request */
    result = Curl_get_upload_buffer(data);
    if(result)
      return result;

    s = calloc(1, sizeof(struct http_connect_state));
    if(!s)
      return CURLE_OUT_OF_MEMORY;
    infof(data, "allocate connect buffer!");
    conn->connect_state = s;
    Curl_dyn_init(&s->rcvbuf, DYN_PROXY_CONNECT_HEADERS);

    /* Curl_proxyCONNECT is based on a pointer to a struct HTTP at the
     * member conn->proto.http; we want [protocol] through HTTP and we have
     * to change the member temporarily for connecting to the HTTP
     * proxy. After Curl_proxyCONNECT we have to set back the member to the
     * original pointer
     *
     * This function might be called several times in the multi interface case
     * if the proxy's CONNECT response is not instant.
     */
    s->prot_save = data->req.p.http;
    data->req.p.http = &s->http_proxy;
    connkeep(conn, "HTTP proxy CONNECT");
  }
  else {
    DEBUGASSERT(conn->connect_state);
    s = conn->connect_state;
    Curl_dyn_reset(&s->rcvbuf);
  }
  s->tunnel_state = TUNNEL_INIT;
  s->keepon = KEEPON_CONNECT;
  s->cl = 0;
  s->close_connection = FALSE;
  return CURLE_OK;
}

void Curl_connect_done(struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;
  struct http_connect_state *s = conn->connect_state;
  if(s && (s->tunnel_state != TUNNEL_EXIT)) {
    s->tunnel_state = TUNNEL_EXIT;
    Curl_dyn_free(&s->rcvbuf);
    Curl_dyn_free(&s->req);

    /* restore the protocol pointer, if not already done */
    if(s->prot_save)
      data->req.p.http = s->prot_save;
    s->prot_save = NULL;
    data->info.httpcode = 0; /* clear it as it might've been used for the
                                proxy */
    data->req.ignorebody = FALSE;
#ifdef USE_HYPER
    data->state.hconnect = FALSE;
#endif
    infof(data, "CONNECT phase completed!");
  }
}

static CURLcode CONNECT_host(struct Curl_easy *data,
                             struct connectdata *conn,
                             const char *hostname,
                             int remote_port,
                             char **connecthostp,
                             char **hostp)
{
  char *hostheader; /* for CONNECT */
  char *host = NULL; /* Host: */
  bool ipv6_ip = conn->bits.ipv6_ip;

  /* the hostname may be different */
  if(hostname != conn->host.name)
    ipv6_ip = (strchr(hostname, ':') != NULL);
  hostheader = /* host:port with IPv6 support */
    aprintf("%s%s%s:%d", ipv6_ip?"[":"", hostname, ipv6_ip?"]":"",
            remote_port);
  if(!hostheader)
    return CURLE_OUT_OF_MEMORY;

  if(!Curl_checkProxyheaders(data, conn, "Host")) {
    host = aprintf("Host: %s\r\n", hostheader);
    if(!host) {
      free(hostheader);
      return CURLE_OUT_OF_MEMORY;
    }
  }
  *connecthostp = hostheader;
  *hostp = host;
  return CURLE_OK;
}

#ifndef USE_HYPER
static CURLcode CONNECT(struct Curl_easy *data,
                        int sockindex,
                        const char *hostname,
                        int remote_port)
{
  int subversion = 0;
  struct SingleRequest *k = &data->req;
  CURLcode result;
  struct connectdata *conn = data->conn;
  curl_socket_t tunnelsocket = conn->sock[sockindex];
  struct http_connect_state *s = conn->connect_state;
  struct HTTP *http = data->req.p.http;
  char *linep;
  size_t perline;

#define SELECT_OK      0
#define SELECT_ERROR   1

  if(Curl_connect_complete(conn))
    return CURLE_OK; /* CONNECT is already completed */

  conn->bits.proxy_connect_closed = FALSE;

  do {
    timediff_t check;
    if(TUNNEL_INIT == s->tunnel_state) {
      /* BEGIN CONNECT PHASE */
      struct dynbuf *req = &s->req;
      char *hostheader = NULL;
      char *host = NULL;

      infof(data, "Establish HTTP proxy tunnel to %s:%d",
            hostname, remote_port);

        /* This only happens if we've looped here due to authentication
           reasons, and we don't really use the newly cloned URL here
           then. Just free() it. */
      Curl_safefree(data->req.newurl);

      /* initialize send-buffer */
      Curl_dyn_init(req, DYN_HTTP_REQUEST);

      result = CONNECT_host(data, conn,
                            hostname, remote_port, &hostheader, &host);
      if(result)
        return result;

      /* Setup the proxy-authorization header, if any */
      result = Curl_http_output_auth(data, conn, "CONNECT", HTTPREQ_GET,
                                     hostheader, TRUE);

      if(!result) {
        const char *httpv =
          (conn->http_proxy.proxytype == CURLPROXY_HTTP_1_0) ? "1.0" : "1.1";

        result =
          Curl_dyn_addf(req,
                        "CONNECT %s HTTP/%s\r\n"
                        "%s"  /* Host: */
                        "%s", /* Proxy-Authorization */
                        hostheader,
                        httpv,
                        host?host:"",
                        data->state.aptr.proxyuserpwd?
                        data->state.aptr.proxyuserpwd:"");

        if(!result && !Curl_checkProxyheaders(data, conn, "User-Agent") &&
           data->set.str[STRING_USERAGENT])
          result = Curl_dyn_addf(req, "User-Agent: %s\r\n",
                                 data->set.str[STRING_USERAGENT]);

        if(!result && !Curl_checkProxyheaders(data, conn, "Proxy-Connection"))
          result = Curl_dyn_add(req, "Proxy-Connection: Keep-Alive\r\n");

        if(!result)
          result = Curl_add_custom_headers(data, TRUE, req);

        if(!result)
          /* CRLF terminate the request */
          result = Curl_dyn_add(req, "\r\n");

        if(!result) {
          /* Send the connect request to the proxy */
          result = Curl_buffer_send(req, data, &data->info.request_size, 0,
                                    sockindex);
        }
        if(result)
          failf(data, "Failed sending CONNECT to proxy");
      }
      free(host);
      free(hostheader);
      if(result)
        return result;

      s->tunnel_state = TUNNEL_CONNECT;
    } /* END CONNECT PHASE */

    check = Curl_timeleft(data, NULL, TRUE);
    if(check <= 0) {
      failf(data, "Proxy CONNECT aborted due to timeout");
      return CURLE_OPERATION_TIMEDOUT;
    }

    if(!Curl_conn_data_pending(conn, sockindex) && !http->sending)
      /* return so we'll be called again polling-style */
      return CURLE_OK;

    /* at this point, the tunnel_connecting phase is over. */

    if(http->sending == HTTPSEND_REQUEST) {
      if(!s->nsend) {
        size_t fillcount;
        k->upload_fromhere = data->state.ulbuf;
        result = Curl_fillreadbuffer(data, data->set.upload_buffer_size,
                                     &fillcount);
        if(result)
          return result;
        s->nsend = fillcount;
      }
      if(s->nsend) {
        ssize_t bytes_written;
        /* write to socket (send away data) */
        result = Curl_write(data,
                            conn->writesockfd,  /* socket to send to */
                            k->upload_fromhere, /* buffer pointer */
                            s->nsend,           /* buffer size */
                            &bytes_written);    /* actually sent */

        if(!result)
          /* send to debug callback! */
          result = Curl_debug(data, CURLINFO_HEADER_OUT,
                              k->upload_fromhere, bytes_written);

        s->nsend -= bytes_written;
        k->upload_fromhere += bytes_written;
        return result;
      }
      http->sending = HTTPSEND_NADA;
      /* if nothing left to send, continue */
    }
    { /* READING RESPONSE PHASE */
      int error = SELECT_OK;

      while(s->keepon) {
        ssize_t gotbytes;
        char byte;

        /* Read one byte at a time to avoid a race condition. Wait at most one
           second before looping to ensure continuous pgrsUpdates. */
        result = Curl_read(data, tunnelsocket, &byte, 1, &gotbytes);
        if(result == CURLE_AGAIN)
          /* socket buffer drained, return */
          return CURLE_OK;

        if(Curl_pgrsUpdate(data))
          return CURLE_ABORTED_BY_CALLBACK;

        if(result) {
          s->keepon = KEEPON_DONE;
          break;
        }
        else if(gotbytes <= 0) {
          if(data->set.proxyauth && data->state.authproxy.avail &&
             data->state.aptr.proxyuserpwd) {
            /* proxy auth was requested and there was proxy auth available,
               then deem this as "mere" proxy disconnect */
            conn->bits.proxy_connect_closed = TRUE;
            infof(data, "Proxy CONNECT connection closed");
          }
          else {
            error = SELECT_ERROR;
            failf(data, "Proxy CONNECT aborted");
          }
          s->keepon = KEEPON_DONE;
          break;
        }

        if(s->keepon == KEEPON_IGNORE) {
          /* This means we are currently ignoring a response-body */

          if(s->cl) {
            /* A Content-Length based body: simply count down the counter
               and make sure to break out of the loop when we're done! */
            s->cl--;
            if(s->cl <= 0) {
              s->keepon = KEEPON_DONE;
              s->tunnel_state = TUNNEL_COMPLETE;
              break;
            }
          }
          else {
            /* chunked-encoded body, so we need to do the chunked dance
               properly to know when the end of the body is reached */
            CHUNKcode r;
            CURLcode extra;
            ssize_t tookcareof = 0;

            /* now parse the chunked piece of data so that we can
               properly tell when the stream ends */
            r = Curl_httpchunk_read(data, &byte, 1, &tookcareof, &extra);
            if(r == CHUNKE_STOP) {
              /* we're done reading chunks! */
              infof(data, "chunk reading DONE");
              s->keepon = KEEPON_DONE;
              /* we did the full CONNECT treatment, go COMPLETE */
              s->tunnel_state = TUNNEL_COMPLETE;
            }
          }
          continue;
        }

        if(Curl_dyn_addn(&s->rcvbuf, &byte, 1)) {
          failf(data, "CONNECT response too large!");
          return CURLE_RECV_ERROR;
        }

        /* if this is not the end of a header line then continue */
        if(byte != 0x0a)
          continue;

        linep = Curl_dyn_ptr(&s->rcvbuf);
        perline = Curl_dyn_len(&s->rcvbuf); /* amount of bytes in this line */

        /* convert from the network encoding */
        result = Curl_convert_from_network(data, linep, perline);
        /* Curl_convert_from_network calls failf if unsuccessful */
        if(result)
          return result;

        /* output debug if that is requested */
        Curl_debug(data, CURLINFO_HEADER_IN, linep, perline);

        if(!data->set.suppress_connect_headers) {
          /* send the header to the callback */
          int writetype = CLIENTWRITE_HEADER;
          if(data->set.include_header)
            writetype |= CLIENTWRITE_BODY;

          result = Curl_client_write(data, writetype, linep, perline);
          if(result)
            return result;
        }

        data->info.header_size += (long)perline;

        /* Newlines are CRLF, so the CR is ignored as the line isn't
           really terminated until the LF comes. Treat a following CR
           as end-of-headers as well.*/

        if(('\r' == linep[0]) ||
           ('\n' == linep[0])) {
          /* end of response-headers from the proxy */

          if((407 == k->httpcode) && !data->state.authproblem) {
            /* If we get a 407 response code with content length
               when we have no auth problem, we must ignore the
               whole response-body */
            s->keepon = KEEPON_IGNORE;

            if(s->cl) {
              infof(data, "Ignore %" CURL_FORMAT_CURL_OFF_T
                    " bytes of response-body", s->cl);
            }
            else if(s->chunked_encoding) {
              CHUNKcode r;
              CURLcode extra;

              infof(data, "Ignore chunked response-body");

              /* We set ignorebody true here since the chunked decoder
                 function will acknowledge that. Pay attention so that this is
                 cleared again when this function returns! */
              k->ignorebody = TRUE;

              if(linep[1] == '\n')
                /* this can only be a LF if the letter at index 0 was a CR */
                linep++;

              /* now parse the chunked piece of data so that we can properly
                 tell when the stream ends */
              r = Curl_httpchunk_read(data, linep + 1, 1, &gotbytes,
                                      &extra);
              if(r == CHUNKE_STOP) {
                /* we're done reading chunks! */
                infof(data, "chunk reading DONE");
                s->keepon = KEEPON_DONE;
                /* we did the full CONNECT treatment, go to COMPLETE */
                s->tunnel_state = TUNNEL_COMPLETE;
              }
            }
            else {
              /* without content-length or chunked encoding, we
                 can't keep the connection alive since the close is
                 the end signal so we bail out at once instead */
              s->keepon = KEEPON_DONE;
            }
          }
          else
            s->keepon = KEEPON_DONE;

          if(s->keepon == KEEPON_DONE && !s->cl)
            /* we did the full CONNECT treatment, go to COMPLETE */
            s->tunnel_state = TUNNEL_COMPLETE;

          DEBUGASSERT(s->keepon == KEEPON_IGNORE || s->keepon == KEEPON_DONE);
          continue;
        }

        if((checkprefix("WWW-Authenticate:", linep) &&
            (401 == k->httpcode)) ||
           (checkprefix("Proxy-authenticate:", linep) &&
            (407 == k->httpcode))) {

          bool proxy = (k->httpcode == 407) ? TRUE : FALSE;
          char *auth = Curl_copy_header_value(linep);
          if(!auth)
            return CURLE_OUT_OF_MEMORY;

          result = Curl_http_input_auth(data, proxy, auth);

          free(auth);

          if(result)
            return result;
        }
        else if(checkprefix("Content-Length:", linep)) {
          if(k->httpcode/100 == 2) {
            /* A client MUST ignore any Content-Length or Transfer-Encoding
               header fields received in a successful response to CONNECT.
               "Successful" described as: 2xx (Successful). RFC 7231 4.3.6 */
            infof(data, "Ignoring Content-Length in CONNECT %03d response",
                  k->httpcode);
          }
          else {
            (void)curlx_strtoofft(linep +
                                  strlen("Content-Length:"), NULL, 10, &s->cl);
          }
        }
        else if(Curl_compareheader(linep, "Connection:", "close"))
          s->close_connection = TRUE;
        else if(checkprefix("Transfer-Encoding:", linep)) {
          if(k->httpcode/100 == 2) {
            /* A client MUST ignore any Content-Length or Transfer-Encoding
               header fields received in a successful response to CONNECT.
               "Successful" described as: 2xx (Successful). RFC 7231 4.3.6 */
            infof(data, "Ignoring Transfer-Encoding in "
                  "CONNECT %03d response", k->httpcode);
          }
          else if(Curl_compareheader(linep,
                                     "Transfer-Encoding:", "chunked")) {
            infof(data, "CONNECT responded chunked");
            s->chunked_encoding = TRUE;
            /* init our chunky engine */
            Curl_httpchunk_init(data);
          }
        }
        else if(Curl_compareheader(linep, "Proxy-Connection:", "close"))
          s->close_connection = TRUE;
        else if(2 == sscanf(linep, "HTTP/1.%d %d",
                            &subversion,
                            &k->httpcode)) {
          /* store the HTTP code from the proxy */
          data->info.httpproxycode = k->httpcode;
        }

        Curl_dyn_reset(&s->rcvbuf);
      } /* while there's buffer left and loop is requested */

      if(Curl_pgrsUpdate(data))
        return CURLE_ABORTED_BY_CALLBACK;

      if(error)
        return CURLE_RECV_ERROR;

      if(data->info.httpproxycode/100 != 2) {
        /* Deal with the possibly already received authenticate
           headers. 'newurl' is set to a new URL if we must loop. */
        result = Curl_http_auth_act(data);
        if(result)
          return result;

        if(conn->bits.close)
          /* the connection has been marked for closure, most likely in the
             Curl_http_auth_act() function and thus we can kill it at once
             below */
          s->close_connection = TRUE;
      }

      if(s->close_connection && data->req.newurl) {
        /* Connection closed by server. Don't use it anymore */
        Curl_closesocket(data, conn, conn->sock[sockindex]);
        conn->sock[sockindex] = CURL_SOCKET_BAD;
        break;
      }
    } /* END READING RESPONSE PHASE */

    /* If we are supposed to continue and request a new URL, which basically
     * means the HTTP authentication is still going on so if the tunnel
     * is complete we start over in INIT state */
    if(data->req.newurl && (TUNNEL_COMPLETE == s->tunnel_state)) {
      connect_init(data, TRUE); /* reinit */
    }

  } while(data->req.newurl);

  if(data->info.httpproxycode/100 != 2) {
    if(s->close_connection && data->req.newurl) {
      conn->bits.proxy_connect_closed = TRUE;
      infof(data, "Connect me again please");
      Curl_connect_done(data);
    }
    else {
      free(data->req.newurl);
      data->req.newurl = NULL;
      /* failure, close this connection to avoid re-use */
      streamclose(conn, "proxy CONNECT failure");
      Curl_closesocket(data, conn, conn->sock[sockindex]);
      conn->sock[sockindex] = CURL_SOCKET_BAD;
    }

    /* to back to init state */
    s->tunnel_state = TUNNEL_INIT;

    if(conn->bits.proxy_connect_closed)
      /* this is not an error, just part of the connection negotiation */
      return CURLE_OK;
    Curl_dyn_free(&s->rcvbuf);
    failf(data, "Received HTTP code %d from proxy after CONNECT",
          data->req.httpcode);
    return CURLE_RECV_ERROR;
  }

  s->tunnel_state = TUNNEL_COMPLETE;

  /* If a proxy-authorization header was used for the proxy, then we should
     make sure that it isn't accidentally used for the document request
     after we've connected. So let's free and clear it here. */
  Curl_safefree(data->state.aptr.proxyuserpwd);
  data->state.aptr.proxyuserpwd = NULL;

  data->state.authproxy.done = TRUE;
  data->state.authproxy.multipass = FALSE;

  infof(data, "Proxy replied %d to CONNECT request",
        data->info.httpproxycode);
  data->req.ignorebody = FALSE; /* put it (back) to non-ignore state */
  conn->bits.rewindaftersend = FALSE; /* make sure this isn't set for the
                                         document request  */
  Curl_dyn_free(&s->rcvbuf);
  return CURLE_OK;
}
#else
/* The Hyper version of CONNECT */
static CURLcode CONNECT(struct Curl_easy *data,
                        int sockindex,
                        const char *hostname,
                        int remote_port)
{
  struct connectdata *conn = data->conn;
  struct hyptransfer *h = &data->hyp;
  curl_socket_t tunnelsocket = conn->sock[sockindex];
  struct http_connect_state *s = conn->connect_state;
  CURLcode result = CURLE_OUT_OF_MEMORY;
  hyper_io *io = NULL;
  hyper_request *req = NULL;
  hyper_headers *headers = NULL;
  hyper_clientconn_options *options = NULL;
  hyper_task *handshake = NULL;
  hyper_task *task = NULL; /* for the handshake */
  hyper_task *sendtask = NULL; /* for the send */
  hyper_clientconn *client = NULL;
  hyper_error *hypererr = NULL;
  char *hostheader = NULL; /* for CONNECT */
  char *host = NULL; /* Host: */

  if(Curl_connect_complete(conn))
    return CURLE_OK; /* CONNECT is already completed */

  conn->bits.proxy_connect_closed = FALSE;

  do {
    switch(s->tunnel_state) {
    case TUNNEL_INIT:
      /* BEGIN CONNECT PHASE */
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
      conn->sockfd = tunnelsocket;

      data->state.hconnect = TRUE;

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

      hyper_clientconn_options_exec(options, h->exec);

      /* "Both the `io` and the `options` are consumed in this function
         call" */
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
      if(hyper_request_set_method(req, (uint8_t *)"CONNECT",
                                  strlen("CONNECT"))) {
        failf(data, "error setting method");
        result = CURLE_OUT_OF_MEMORY;
        goto error;
      }

      infof(data, "Establish HTTP proxy tunnel to %s:%d",
            hostname, remote_port);

        /* This only happens if we've looped here due to authentication
           reasons, and we don't really use the newly cloned URL here
           then. Just free() it. */
      Curl_safefree(data->req.newurl);

      result = CONNECT_host(data, conn, hostname, remote_port,
                            &hostheader, &host);
      if(result)
        goto error;

      if(hyper_request_set_uri(req, (uint8_t *)hostheader,
                               strlen(hostheader))) {
        failf(data, "error setting path");
        result = CURLE_OUT_OF_MEMORY;
        goto error;
      }
      if(data->set.verbose) {
        char *se = aprintf("CONNECT %s HTTP/1.1\r\n", hostheader);
        if(!se) {
          result = CURLE_OUT_OF_MEMORY;
          goto error;
        }
        Curl_debug(data, CURLINFO_HEADER_OUT, se, strlen(se));
        free(se);
      }
      /* Setup the proxy-authorization header, if any */
      result = Curl_http_output_auth(data, conn, "CONNECT", HTTPREQ_GET,
                                     hostheader, TRUE);
      if(result)
        goto error;
      Curl_safefree(hostheader);

      /* default is 1.1 */
      if((conn->http_proxy.proxytype == CURLPROXY_HTTP_1_0) &&
         (HYPERE_OK != hyper_request_set_version(req,
                                                 HYPER_HTTP_VERSION_1_0))) {
        failf(data, "error setting HTTP version");
        result = CURLE_OUT_OF_MEMORY;
        goto error;
      }

      headers = hyper_request_headers(req);
      if(!headers) {
        failf(data, "hyper_request_headers");
        result = CURLE_OUT_OF_MEMORY;
        goto error;
      }
      if(host) {
        result = Curl_hyper_header(data, headers, host);
        if(result)
          goto error;
        Curl_safefree(host);
      }

      if(data->state.aptr.proxyuserpwd) {
        result = Curl_hyper_header(data, headers,
                                   data->state.aptr.proxyuserpwd);
        if(result)
          goto error;
      }

      if(!Curl_checkProxyheaders(data, conn, "User-Agent") &&
         data->set.str[STRING_USERAGENT]) {
        struct dynbuf ua;
        Curl_dyn_init(&ua, DYN_HTTP_REQUEST);
        result = Curl_dyn_addf(&ua, "User-Agent: %s\r\n",
                               data->set.str[STRING_USERAGENT]);
        if(result)
          goto error;
        result = Curl_hyper_header(data, headers, Curl_dyn_ptr(&ua));
        if(result)
          goto error;
        Curl_dyn_free(&ua);
      }

      if(!Curl_checkProxyheaders(data, conn, "Proxy-Connection")) {
        result = Curl_hyper_header(data, headers,
                                   "Proxy-Connection: Keep-Alive");
        if(result)
          goto error;
      }

      result = Curl_add_custom_headers(data, TRUE, headers);
      if(result)
        goto error;

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

      do {
        task = hyper_executor_poll(h->exec);
        if(task) {
          bool error = hyper_task_type(task) == HYPER_TASK_ERROR;
          if(error)
            hypererr = hyper_task_value(task);
          hyper_task_free(task);
          if(error) {
            /* this could probably use a better error code? */
            result = CURLE_OUT_OF_MEMORY;
            goto error;
          }
        }
      } while(task);
      s->tunnel_state = TUNNEL_CONNECT;
      /* FALLTHROUGH */
    case TUNNEL_CONNECT: {
      int didwhat;
      bool done = FALSE;
      result = Curl_hyper_stream(data, conn, &didwhat, &done,
                                 CURL_CSELECT_IN | CURL_CSELECT_OUT);
      if(result)
        goto error;
      if(!done)
        break;
      s->tunnel_state = TUNNEL_COMPLETE;
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
    }
    break;

    default:
      break;
    }

    /* If we are supposed to continue and request a new URL, which basically
     * means the HTTP authentication is still going on so if the tunnel
     * is complete we start over in INIT state */
    if(data->req.newurl && (TUNNEL_COMPLETE == s->tunnel_state)) {
      infof(data, "CONNECT request done, loop to make another");
      connect_init(data, TRUE); /* reinit */
    }
  } while(data->req.newurl);

  result = CURLE_OK;
  if(s->tunnel_state == TUNNEL_COMPLETE) {
    if(data->info.httpproxycode/100 != 2) {
      if(conn->bits.close && data->req.newurl) {
        conn->bits.proxy_connect_closed = TRUE;
        infof(data, "Connect me again please");
        Curl_connect_done(data);
      }
      else {
        free(data->req.newurl);
        data->req.newurl = NULL;
        /* failure, close this connection to avoid re-use */
        streamclose(conn, "proxy CONNECT failure");
        Curl_closesocket(data, conn, conn->sock[sockindex]);
        conn->sock[sockindex] = CURL_SOCKET_BAD;
      }

      /* to back to init state */
      s->tunnel_state = TUNNEL_INIT;

      if(!conn->bits.proxy_connect_closed) {
        failf(data, "Received HTTP code %d from proxy after CONNECT",
              data->req.httpcode);
        result = CURLE_RECV_ERROR;
      }
    }
  }
  error:
  free(host);
  free(hostheader);
  if(io)
    hyper_io_free(io);

  if(options)
    hyper_clientconn_options_free(options);

  if(handshake)
    hyper_task_free(handshake);

  if(hypererr) {
    uint8_t errbuf[256];
    size_t errlen = hyper_error_print(hypererr, errbuf, sizeof(errbuf));
    failf(data, "Hyper: %.*s", (int)errlen, errbuf);
    hyper_error_free(hypererr);
  }
  return result;
}
#endif

void Curl_connect_free(struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;
  struct http_connect_state *s = conn->connect_state;
  if(s) {
    free(s);
    conn->connect_state = NULL;
  }
}

/*
 * Curl_proxyCONNECT() requires that we're connected to a HTTP proxy. This
 * function will issue the necessary commands to get a seamless tunnel through
 * this proxy. After that, the socket can be used just as a normal socket.
 */

CURLcode Curl_proxyCONNECT(struct Curl_easy *data,
                           int sockindex,
                           const char *hostname,
                           int remote_port)
{
  CURLcode result;
  struct connectdata *conn = data->conn;
  if(!conn->connect_state) {
    result = connect_init(data, FALSE);
    if(result)
      return result;
  }
  result = CONNECT(data, sockindex, hostname, remote_port);

  if(result || Curl_connect_complete(conn))
    Curl_connect_done(data);

  return result;
}

#else
void Curl_connect_free(struct Curl_easy *data)
{
  (void)data;
}

#endif /* CURL_DISABLE_PROXY */
