/*
 * libwebsockets - lib/client/client.c
 *
 * Copyright (C) 2010-2017 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 */

#include "core/private.h"

LWS_VISIBLE LWS_EXTERN void
lws_client_http_body_pending(struct lws *wsi, int something_left_to_send)
{
	wsi->client_http_body_pending = !!something_left_to_send;
}

/*
 * return self, or queued client wsi we are acting on behalf of
 *
 * That is the TAIL of the queue (new queue elements are added at the HEAD)
 */

struct lws *
lws_client_wsi_effective(struct lws *wsi)
{
	struct lws_dll_lws *tail = NULL;

	if (!wsi->transaction_from_pipeline_queue ||
	    !wsi->dll_client_transaction_queue_head.next)
		return wsi;

	lws_start_foreach_dll_safe(struct lws_dll_lws *, d, d1,
				   wsi->dll_client_transaction_queue_head.next) {
		tail = d;
	} lws_end_foreach_dll_safe(d, d1);

	return lws_container_of(tail, struct lws,
				  dll_client_transaction_queue);
}

/*
 * return self or the guy we are queued under
 *
 * REQUIRES VHOST LOCK HELD
 */

static struct lws *
_lws_client_wsi_master(struct lws *wsi)
{
	struct lws *wsi_eff = wsi;
	struct lws_dll_lws *d;

	d = wsi->dll_client_transaction_queue.prev;
	while (d) {
		wsi_eff = lws_container_of(d, struct lws,
					dll_client_transaction_queue_head);

		d = d->prev;
	}

	return wsi_eff;
}

int
lws_client_socket_service(struct lws *wsi, struct lws_pollfd *pollfd,
			  struct lws *wsi_conn)
{
	struct lws_context *context = wsi->context;
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	char *p = (char *)&pt->serv_buf[0];
	struct lws *w;
#if defined(LWS_WITH_TLS)
	char ebuf[128];
#endif
	const char *cce = NULL;
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	ssize_t len = 0;
	unsigned char c;
#endif
	char *sb = p;
	int n = 0;
#if defined(LWS_WITH_SOCKS5)
	char conn_mode = 0, pending_timeout = 0;
#endif

	if ((pollfd->revents & LWS_POLLOUT) &&
	     wsi->keepalive_active &&
	     wsi->dll_client_transaction_queue_head.next) {
		struct lws *wfound = NULL;

		lwsl_debug("%s: pollout HANDSHAKE2\n", __func__);

		/*
		 * We have a transaction queued that wants to pipeline.
		 *
		 * We have to allow it to send headers strictly in the order
		 * that it was queued, ie, tail-first.
		 */
		lws_vhost_lock(wsi->vhost);
		lws_start_foreach_dll_safe(struct lws_dll_lws *, d, d1,
					   wsi->dll_client_transaction_queue_head.next) {
			struct lws *w = lws_container_of(d, struct lws,
						  dll_client_transaction_queue);

			lwsl_debug("%s: %p states 0x%x\n", __func__, w, w->wsistate);
			if (lwsi_state(w) == LRS_H1C_ISSUE_HANDSHAKE2)
				wfound = w;
		} lws_end_foreach_dll_safe(d, d1);

		if (wfound) {
			/*
			 * pollfd has the master sockfd in it... we
			 * need to use that in HANDSHAKE2 to understand
			 * which wsi to actually write on
			 */
			lws_client_socket_service(wfound, pollfd, wsi);
			lws_callback_on_writable(wsi);
		} else
			lwsl_debug("%s: didn't find anything in txn q in HS2\n",
							   __func__);

		lws_vhost_unlock(wsi->vhost);

		return 0;
	}

	switch (lwsi_state(wsi)) {

	case LRS_WAITING_CONNECT:

		/*
		 * we are under PENDING_TIMEOUT_SENT_CLIENT_HANDSHAKE
		 * timeout protection set in client-handshake.c
		 */

		if (!lws_client_connect_2(wsi)) {
			/* closed */
			lwsl_client("closed\n");
			return -1;
		}

		/* either still pending connection, or changed mode */
		return 0;

#if defined(LWS_WITH_SOCKS5)
	/* SOCKS Greeting Reply */
	case LRS_WAITING_SOCKS_GREETING_REPLY:
	case LRS_WAITING_SOCKS_AUTH_REPLY:
	case LRS_WAITING_SOCKS_CONNECT_REPLY:

		/* handle proxy hung up on us */

		if (pollfd->revents & LWS_POLLHUP) {
			lwsl_warn("SOCKS connection %p (fd=%d) dead\n",
				  (void *)wsi, pollfd->fd);
			goto bail3;
		}

		n = recv(wsi->desc.sockfd, sb, context->pt_serv_buf_size, 0);
		if (n < 0) {
			if (LWS_ERRNO == LWS_EAGAIN) {
				lwsl_debug("SOCKS read EAGAIN, retrying\n");
				return 0;
			}
			lwsl_err("ERROR reading from SOCKS socket\n");
			goto bail3;
		}

		switch (lwsi_state(wsi)) {

		case LRS_WAITING_SOCKS_GREETING_REPLY:
			if (pt->serv_buf[0] != SOCKS_VERSION_5)
				goto socks_reply_fail;

			if (pt->serv_buf[1] == SOCKS_AUTH_NO_AUTH) {
				lwsl_client("SOCKS GR: No Auth Method\n");
				socks_generate_msg(wsi, SOCKS_MSG_CONNECT, &len);
				conn_mode = LRS_WAITING_SOCKS_CONNECT_REPLY;
				pending_timeout =
				   PENDING_TIMEOUT_AWAITING_SOCKS_CONNECT_REPLY;
				goto socks_send;
			}

			if (pt->serv_buf[1] == SOCKS_AUTH_USERNAME_PASSWORD) {
				lwsl_client("SOCKS GR: User/Pw Method\n");
				socks_generate_msg(wsi,
						   SOCKS_MSG_USERNAME_PASSWORD,
						   &len);
				conn_mode = LRS_WAITING_SOCKS_AUTH_REPLY;
				pending_timeout =
				      PENDING_TIMEOUT_AWAITING_SOCKS_AUTH_REPLY;
				goto socks_send;
			}
			goto socks_reply_fail;

		case LRS_WAITING_SOCKS_AUTH_REPLY:
			if (pt->serv_buf[0] != SOCKS_SUBNEGOTIATION_VERSION_1 ||
			    pt->serv_buf[1] != SOCKS_SUBNEGOTIATION_STATUS_SUCCESS)
				goto socks_reply_fail;

			lwsl_client("SOCKS password OK, sending connect\n");
			socks_generate_msg(wsi, SOCKS_MSG_CONNECT, &len);
			conn_mode = LRS_WAITING_SOCKS_CONNECT_REPLY;
			pending_timeout =
				   PENDING_TIMEOUT_AWAITING_SOCKS_CONNECT_REPLY;
socks_send:
			n = send(wsi->desc.sockfd, (char *)pt->serv_buf, len,
				 MSG_NOSIGNAL);
			if (n < 0) {
				lwsl_debug("ERROR writing to socks proxy\n");
				goto bail3;
			}

			lws_set_timeout(wsi, pending_timeout, AWAITING_TIMEOUT);
			lwsi_set_state(wsi, conn_mode);
			break;

socks_reply_fail:
			lwsl_notice("socks reply: v%d, err %d\n",
				    pt->serv_buf[0], pt->serv_buf[1]);
			goto bail3;

		case LRS_WAITING_SOCKS_CONNECT_REPLY:
			if (pt->serv_buf[0] != SOCKS_VERSION_5 ||
			    pt->serv_buf[1] != SOCKS_REQUEST_REPLY_SUCCESS)
				goto socks_reply_fail;

			lwsl_client("socks connect OK\n");

			/* free stash since we are done with it */
			lws_client_stash_destroy(wsi);
			if (lws_hdr_simple_create(wsi,
						  _WSI_TOKEN_CLIENT_PEER_ADDRESS,
						  wsi->vhost->socks_proxy_address))
				goto bail3;

			wsi->c_port = wsi->vhost->socks_proxy_port;

			/* clear his proxy connection timeout */
			lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);
			goto start_ws_handshake;
		}
		break;
#endif

	case LRS_WAITING_PROXY_REPLY:

		/* handle proxy hung up on us */

		if (pollfd->revents & LWS_POLLHUP) {

			lwsl_warn("Proxy connection %p (fd=%d) dead\n",
				  (void *)wsi, pollfd->fd);

			goto bail3;
		}

		n = recv(wsi->desc.sockfd, sb, context->pt_serv_buf_size, 0);
		if (n < 0) {
			if (LWS_ERRNO == LWS_EAGAIN) {
				lwsl_debug("Proxy read EAGAIN... retrying\n");
				return 0;
			}
			lwsl_err("ERROR reading from proxy socket\n");
			goto bail3;
		}

		pt->serv_buf[13] = '\0';
		if (strcmp(sb, "HTTP/1.0 200 ") &&
		    strcmp(sb, "HTTP/1.1 200 ")) {
			lwsl_err("ERROR proxy: %s\n", sb);
			goto bail3;
		}

		/* clear his proxy connection timeout */

		lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);

		/* fallthru */

	case LRS_H1C_ISSUE_HANDSHAKE:

		/*
		 * we are under PENDING_TIMEOUT_SENT_CLIENT_HANDSHAKE
		 * timeout protection set in client-handshake.c
		 *
		 * take care of our lws_callback_on_writable
		 * happening at a time when there's no real connection yet
		 */
#if defined(LWS_WITH_SOCKS5)
start_ws_handshake:
#endif
		if (lws_change_pollfd(wsi, LWS_POLLOUT, 0))
			return -1;

#if defined(LWS_WITH_TLS)
		/* we can retry this... just cook the SSL BIO the first time */

		if ((wsi->tls.use_ssl & LCCSCF_USE_SSL) && !wsi->tls.ssl &&
		    lws_ssl_client_bio_create(wsi) < 0) {
			cce = "bio_create failed";
			goto bail3;
		}

		if (wsi->tls.use_ssl & LCCSCF_USE_SSL) {
			n = lws_ssl_client_connect1(wsi);
			if (!n)
				return 0;
			if (n < 0) {
				cce = "lws_ssl_client_connect1 failed";
				goto bail3;
			}
		} else
			wsi->tls.ssl = NULL;

		/* fallthru */

	case LRS_WAITING_SSL:

		if (wsi->tls.use_ssl & LCCSCF_USE_SSL) {
			n = lws_ssl_client_connect2(wsi, ebuf, sizeof(ebuf));
			if (!n)
				return 0;
			if (n < 0) {
				cce = ebuf;
				goto bail3;
			}
		} else
			wsi->tls.ssl = NULL;
#endif
#if defined (LWS_WITH_HTTP2)
		if (wsi->client_h2_alpn) {
			/*
			 * We connected to the server and set up tls, and
			 * negotiated "h2".
			 *
			 * So this is it, we are an h2 master client connection
			 * now, not an h1 client connection.
			 */
			lws_tls_server_conn_alpn(wsi);

			/* send the H2 preface to legitimize the connection */
			if (lws_h2_issue_preface(wsi)) {
				cce = "error sending h2 preface";
				goto bail3;
			}

			break;
		}
#endif
		lwsi_set_state(wsi, LRS_H1C_ISSUE_HANDSHAKE2);
		lws_set_timeout(wsi, PENDING_TIMEOUT_AWAITING_CLIENT_HS_SEND,
				context->timeout_secs);

		/* fallthru */

	case LRS_H1C_ISSUE_HANDSHAKE2:
		p = lws_generate_client_handshake(wsi, p);
		if (p == NULL) {
			if (wsi->role_ops == &role_ops_raw_skt ||
			    wsi->role_ops == &role_ops_raw_file)
				return 0;

			lwsl_err("Failed to generate handshake for client\n");
			lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS, "chs");
			return 0;
		}

		/* send our request to the server */
		lws_latency_pre(context, wsi);

		w = _lws_client_wsi_master(wsi);
		lwsl_info("%s: HANDSHAKE2: %p: sending headers on %p (wsistate 0x%x 0x%x)\n",
				__func__, wsi, w, wsi->wsistate, w->wsistate);

		n = lws_ssl_capable_write(w, (unsigned char *)sb, (int)(p - sb));
		lws_latency(context, wsi, "send lws_issue_raw", n,
			    n == p - sb);
		switch (n) {
		case LWS_SSL_CAPABLE_ERROR:
			lwsl_debug("ERROR writing to client socket\n");
			lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS, "cws");
			return 0;
		case LWS_SSL_CAPABLE_MORE_SERVICE:
			lws_callback_on_writable(wsi);
			break;
		}

		if (wsi->client_http_body_pending) {
			lwsi_set_state(wsi, LRS_ISSUE_HTTP_BODY);
			lws_set_timeout(wsi,
					PENDING_TIMEOUT_CLIENT_ISSUE_PAYLOAD,
					context->timeout_secs);
			/* user code must ask for writable callback */
			break;
		}

		lwsi_set_state(wsi, LRS_WAITING_SERVER_REPLY);
		wsi->hdr_parsing_completed = 0;

		if (lwsi_state(w) == LRS_IDLING) {
			lwsi_set_state(w, LRS_WAITING_SERVER_REPLY);
			w->hdr_parsing_completed = 0;
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
			w->http.ah->parser_state = WSI_TOKEN_NAME_PART;
			w->http.ah->lextable_pos = 0;
			/* If we're (re)starting on headers, need other implied init */
			wsi->http.ah->ues = URIES_IDLE;
#endif
		}

		lws_set_timeout(wsi, PENDING_TIMEOUT_AWAITING_SERVER_RESPONSE,
				wsi->context->timeout_secs);

		lws_callback_on_writable(w);

		goto client_http_body_sent;

	case LRS_ISSUE_HTTP_BODY:
		if (wsi->client_http_body_pending) {
			//lws_set_timeout(wsi,
			//		PENDING_TIMEOUT_CLIENT_ISSUE_PAYLOAD,
			//		context->timeout_secs);
			/* user code must ask for writable callback */
			break;
		}
client_http_body_sent:
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
		/* prepare ourselves to do the parsing */
		wsi->http.ah->parser_state = WSI_TOKEN_NAME_PART;
		wsi->http.ah->lextable_pos = 0;
#endif
		lwsi_set_state(wsi, LRS_WAITING_SERVER_REPLY);
		lws_set_timeout(wsi, PENDING_TIMEOUT_AWAITING_SERVER_RESPONSE,
				context->timeout_secs);
		break;

	case LRS_WAITING_SERVER_REPLY:
		/*
		 * handle server hanging up on us...
		 * but if there is POLLIN waiting, handle that first
		 */
		if ((pollfd->revents & (LWS_POLLIN | LWS_POLLHUP)) ==
								LWS_POLLHUP) {

			lwsl_debug("Server connection %p (fd=%d) dead\n",
				(void *)wsi, pollfd->fd);
			cce = "Peer hung up";
			goto bail3;
		}

		if (!(pollfd->revents & LWS_POLLIN))
			break;

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
		/* interpret the server response
		 *
		 *  HTTP/1.1 101 Switching Protocols
		 *  Upgrade: websocket
		 *  Connection: Upgrade
		 *  Sec-WebSocket-Accept: me89jWimTRKTWwrS3aRrL53YZSo=
		 *  Sec-WebSocket-Nonce: AQIDBAUGBwgJCgsMDQ4PEC==
		 *  Sec-WebSocket-Protocol: chat
		 *
		 * we have to take some care here to only take from the
		 * socket bytewise.  The browser may (and has been seen to
		 * in the case that onopen() performs websocket traffic)
		 * coalesce both handshake response and websocket traffic
		 * in one packet, since at that point the connection is
		 * definitively ready from browser pov.
		 */
		len = 1;
		while (wsi->http.ah->parser_state != WSI_PARSING_COMPLETE &&
		       len > 0) {
			int plen = 1;

			n = lws_ssl_capable_read(wsi, &c, 1);
			lws_latency(context, wsi, "send lws_issue_raw", n,
				    n == 1);
			switch (n) {
			case 0:
			case LWS_SSL_CAPABLE_ERROR:
				cce = "read failed";
				goto bail3;
			case LWS_SSL_CAPABLE_MORE_SERVICE:
				return 0;
			}

			if (lws_parse(wsi, &c, &plen)) {
				lwsl_warn("problems parsing header\n");
				goto bail3;
			}
		}

		/*
		 * hs may also be coming in multiple packets, there is a 5-sec
		 * libwebsocket timeout still active here too, so if parsing did
		 * not complete just wait for next packet coming in this state
		 */
		if (wsi->http.ah->parser_state != WSI_PARSING_COMPLETE)
			break;

#endif

		/*
		 * otherwise deal with the handshake.  If there's any
		 * packet traffic already arrived we'll trigger poll() again
		 * right away and deal with it that way
		 */
		return lws_client_interpret_server_handshake(wsi);

bail3:
		lwsl_info("closing conn at LWS_CONNMODE...SERVER_REPLY\n");
		if (cce)
			lwsl_info("reason: %s\n", cce);
		wsi->protocol->callback(wsi,
			LWS_CALLBACK_CLIENT_CONNECTION_ERROR,
			wsi->user_space, (void *)cce, cce ? strlen(cce) : 0);
		wsi->already_did_cce = 1;
		lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS, "cbail3");
		return -1;

	default:
		break;
	}

	return 0;
}

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)

int LWS_WARN_UNUSED_RESULT
lws_http_transaction_completed_client(struct lws *wsi)
{
	struct lws *wsi_eff = lws_client_wsi_effective(wsi);

	lwsl_info("%s: wsi: %p, wsi_eff: %p\n", __func__, wsi, wsi_eff);

	if (user_callback_handle_rxflow(wsi_eff->protocol->callback,
			wsi_eff, LWS_CALLBACK_COMPLETED_CLIENT_HTTP,
			wsi_eff->user_space, NULL, 0)) {
		lwsl_debug("%s: Completed call returned nonzero (role 0x%x)\n",
						__func__, lwsi_role(wsi_eff));
		return -1;
	}

	/*
	 * Are we constitutionally capable of having a queue, ie, we are on
	 * the "active client connections" list?
	 *
	 * If not, that's it for us.
	 */

	if (lws_dll_is_null(&wsi->dll_active_client_conns))
		return -1;

	/* if this was a queued guy, close him and remove from queue */

	if (wsi->transaction_from_pipeline_queue) {
		lwsl_debug("closing queued wsi %p\n", wsi_eff);
		/* so the close doesn't trigger a CCE */
		wsi_eff->already_did_cce = 1;
		__lws_close_free_wsi(wsi_eff,
			LWS_CLOSE_STATUS_CLIENT_TRANSACTION_DONE,
			"queued client done");
	}

	/* after the first one, they can only be coming from the queue */
	wsi->transaction_from_pipeline_queue = 1;

	wsi->http.rx_content_length = 0;
	wsi->hdr_parsing_completed = 0;

	/* is there a new tail after removing that one? */
	wsi_eff = lws_client_wsi_effective(wsi);

	/*
	 * Do we have something pipelined waiting?
	 * it's OK if he hasn't managed to send his headers yet... he's next
	 * in line to do that...
	 */
	if (wsi_eff == wsi) {
		/*
		 * Nothing pipelined... we should hang around a bit
		 * in case something turns up...
		 */
		lwsl_info("%s: nothing pipelined waiting\n", __func__);
		lwsi_set_state(wsi, LRS_IDLING);

		lws_set_timeout(wsi, PENDING_TIMEOUT_CLIENT_CONN_IDLE, 5);

		return 0;
	}

	/*
	 * H1: we can serialize the queued guys into the same ah
	 * H2: everybody needs their own ah until their own STREAM_END
	 */

	/* otherwise set ourselves up ready to go again */
	lwsi_set_state(wsi, LRS_WAITING_SERVER_REPLY);

	wsi->http.ah->parser_state = WSI_TOKEN_NAME_PART;
	wsi->http.ah->lextable_pos = 0;

	lws_set_timeout(wsi, PENDING_TIMEOUT_AWAITING_SERVER_RESPONSE,
			wsi->context->timeout_secs);

	/* If we're (re)starting on headers, need other implied init */
	wsi->http.ah->ues = URIES_IDLE;

	lwsl_info("%s: %p: new queued transaction as %p\n", __func__, wsi, wsi_eff);
	lws_callback_on_writable(wsi);

	return 0;
}

LWS_VISIBLE LWS_EXTERN unsigned int
lws_http_client_http_response(struct lws *wsi)
{
	if (!wsi->http.ah)
		return 0;

	return wsi->http.ah->http_response;
}
#endif
#if defined(LWS_PLAT_OPTEE)
char *
strrchr(const char *s, int c)
{
	char *hit = NULL;

	while (*s)
		if (*(s++) == (char)c)
		       hit = (char *)s - 1;

	return hit;
}

#define atoll atoi
#endif

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
int
lws_client_interpret_server_handshake(struct lws *wsi)
{
	int n, port = 0, ssl = 0;
	int close_reason = LWS_CLOSE_STATUS_PROTOCOL_ERR;
	const char *prot, *ads = NULL, *path, *cce = NULL;
	struct allocated_headers *ah = NULL;
	struct lws *w = lws_client_wsi_effective(wsi);
	char *p, *q;
	char new_path[300];

	lws_client_stash_destroy(wsi);

	ah = wsi->http.ah;
	if (!wsi->do_ws) {
		/* we are being an http client...
		 */
#if defined(LWS_ROLE_H2)
		if (wsi->client_h2_alpn || wsi->client_h2_substream) {
			lwsl_debug("%s: %p: transitioning to h2 client\n", __func__, wsi);
			lws_role_transition(wsi, LWSIFR_CLIENT,
					    LRS_ESTABLISHED, &role_ops_h2);
		} else
#endif
		{
#if defined(LWS_ROLE_H1)
			{
			lwsl_debug("%s: %p: transitioning to h1 client\n", __func__, wsi);
			lws_role_transition(wsi, LWSIFR_CLIENT,
					    LRS_ESTABLISHED, &role_ops_h1);
			}
#else
			return -1;
#endif
		}

		wsi->http.ah = ah;
		ah->http_response = 0;
	}

	/*
	 * well, what the server sent looked reasonable for syntax.
	 * Now let's confirm it sent all the necessary headers
	 *
	 * http (non-ws) client will expect something like this
	 *
	 * HTTP/1.0.200
	 * server:.libwebsockets
	 * content-type:.text/html
	 * content-length:.17703
	 * set-cookie:.test=LWS_1456736240_336776_COOKIE;Max-Age=360000
	 */

	wsi->http.connection_type = HTTP_CONNECTION_KEEP_ALIVE;
	if (!wsi->client_h2_substream) {
		p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP);
		if (wsi->do_ws && !p) {
			lwsl_info("no URI\n");
			cce = "HS: URI missing";
			goto bail3;
		}
		if (!p) {
			p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP1_0);
			wsi->http.connection_type = HTTP_CONNECTION_CLOSE;
		}
		if (!p) {
			cce = "HS: URI missing";
			lwsl_info("no URI\n");
			goto bail3;
		}
	} else {
		p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP_COLON_STATUS);
		if (!p) {
			cce = "HS: :status missing";
			lwsl_info("no status\n");
			goto bail3;
		}
	}
	n = atoi(p);
	if (ah)
		ah->http_response = n;

	if (n == 301 || n == 302 || n == 303 || n == 307 || n == 308) {
		p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP_LOCATION);
		if (!p) {
			cce = "HS: Redirect code but no Location";
			goto bail3;
		}

		/* Relative reference absolute path */
		if (p[0] == '/') {
#if defined(LWS_WITH_TLS)
			ssl = wsi->tls.use_ssl & LCCSCF_USE_SSL;
#endif
			ads = lws_hdr_simple_ptr(wsi,
						 _WSI_TOKEN_CLIENT_PEER_ADDRESS);
			port = wsi->c_port;
			/* +1 as lws_client_reset expects leading / omitted */
			path = p + 1;
		}
		/* Absolute (Full) URI */
		else if (strchr(p, ':')) {
			if (lws_parse_uri(p, &prot, &ads, &port, &path)) {
				cce = "HS: URI did not parse";
				goto bail3;
			}

			if (!strcmp(prot, "wss") || !strcmp(prot, "https"))
				ssl = 1;
		}
		/* Relative reference relative path */
		else {
			/* This doesn't try to calculate an absolute path,
			 * that will be left to the server */
#if defined(LWS_WITH_TLS)
			ssl = wsi->tls.use_ssl & LCCSCF_USE_SSL;
#endif
			ads = lws_hdr_simple_ptr(wsi,
						 _WSI_TOKEN_CLIENT_PEER_ADDRESS);
			port = wsi->c_port;
			/* +1 as lws_client_reset expects leading / omitted */
			path = new_path + 1;
			lws_strncpy(new_path, lws_hdr_simple_ptr(wsi,
				   _WSI_TOKEN_CLIENT_URI), sizeof(new_path));
			q = strrchr(new_path, '/');
			if (q)
				lws_strncpy(q + 1, p, sizeof(new_path) -
							(q - new_path));
			else
				path = p;
		}

#if defined(LWS_WITH_TLS)
		if ((wsi->tls.use_ssl & LCCSCF_USE_SSL) && !ssl) {
			cce = "HS: Redirect attempted SSL downgrade";
			goto bail3;
		}
#endif

		if (!lws_client_reset(&wsi, ssl, ads, port, path, ads)) {
			/* there are two ways to fail out with NULL return...
			 * simple, early problem where the wsi is intact, or
			 * we went through with the reconnect attempt and the
			 * wsi is already closed.  In the latter case, the wsi
			 * has beet set to NULL additionally.
			 */
			lwsl_err("Redirect failed\n");
			cce = "HS: Redirect failed";
			if (wsi)
				goto bail3;

			return 1;
		}
		return 0;
	}

	if (!wsi->do_ws) {

		/* if h1 KA is allowed, enable the queued pipeline guys */

		if (!wsi->client_h2_alpn && !wsi->client_h2_substream && w == wsi) { /* ie, coming to this for the first time */
			if (wsi->http.connection_type == HTTP_CONNECTION_KEEP_ALIVE)
				wsi->keepalive_active = 1;
			else {
				/*
				 * Ugh... now the main http connection has seen
				 * both sides, we learn the server doesn't
				 * support keepalive.
				 *
				 * That means any guys queued on us are going
				 * to have to be restarted from connect2 with
				 * their own connections.
				 */

				/*
				 * stick around telling any new guys they can't
				 * pipeline to this server
				 */
				wsi->keepalive_rejected = 1;

				lws_vhost_lock(wsi->vhost);
				lws_start_foreach_dll_safe(struct lws_dll_lws *, d, d1,
							   wsi->dll_client_transaction_queue_head.next) {
					struct lws *ww = lws_container_of(d, struct lws,
								  dll_client_transaction_queue);

					/* remove him from our queue */
					lws_dll_lws_remove(&ww->dll_client_transaction_queue);
					/* give up on pipelining */
					ww->client_pipeline = 0;

					/* go back to "trying to connect" state */
					lws_role_transition(ww, LWSIFR_CLIENT,
							    LRS_UNCONNECTED,
#if defined(LWS_ROLE_H1)
							    &role_ops_h1);
#else
#if defined (LWS_ROLE_H2)
							    &role_ops_h2);
#else
							    &role_ops_raw);
#endif
#endif
					ww->user_space = NULL;
				} lws_end_foreach_dll_safe(d, d1);
				lws_vhost_unlock(wsi->vhost);
			}
		}

#ifdef LWS_WITH_HTTP_PROXY
		wsi->http.perform_rewrite = 0;
		if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_CONTENT_TYPE)) {
			if (!strncmp(lws_hdr_simple_ptr(wsi,
						WSI_TOKEN_HTTP_CONTENT_TYPE),
						"text/html", 9))
				wsi->http.perform_rewrite = 1;
		}
#endif

		/* allocate the per-connection user memory (if any) */
		if (lws_ensure_user_space(wsi)) {
			lwsl_err("Problem allocating wsi user mem\n");
			cce = "HS: OOM";
			goto bail2;
		}

		/* he may choose to send us stuff in chunked transfer-coding */
		wsi->chunked = 0;
		wsi->chunk_remaining = 0; /* ie, next thing is chunk size */
		if (lws_hdr_total_length(wsi,
					WSI_TOKEN_HTTP_TRANSFER_ENCODING)) {
			wsi->chunked = !strcmp(lws_hdr_simple_ptr(wsi,
					       WSI_TOKEN_HTTP_TRANSFER_ENCODING),
						"chunked");
			/* first thing is hex, after payload there is crlf */
			wsi->chunk_parser = ELCP_HEX;
		}

		if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_CONTENT_LENGTH)) {
			wsi->http.rx_content_length =
					atoll(lws_hdr_simple_ptr(wsi,
						WSI_TOKEN_HTTP_CONTENT_LENGTH));
			lwsl_info("%s: incoming content length %llu\n",
				    __func__, (unsigned long long)
					    wsi->http.rx_content_length);
			wsi->http.rx_content_remain =
					wsi->http.rx_content_length;
		} else /* can't do 1.1 without a content length or chunked */
			if (!wsi->chunked)
				wsi->http.connection_type =
							HTTP_CONNECTION_CLOSE;

		/*
		 * we seem to be good to go, give client last chance to check
		 * headers and OK it
		 */
		if (wsi->protocol->callback(wsi,
				LWS_CALLBACK_CLIENT_FILTER_PRE_ESTABLISH,
					    wsi->user_space, NULL, 0)) {

			cce = "HS: disallowed by client filter";
			goto bail2;
		}

		/* clear his proxy connection timeout */
		lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);

		wsi->rxflow_change_to = LWS_RXFLOW_ALLOW;

		/* call him back to inform him he is up */
		if (wsi->protocol->callback(wsi,
					    LWS_CALLBACK_ESTABLISHED_CLIENT_HTTP,
					    wsi->user_space, NULL, 0)) {
			cce = "HS: disallowed at ESTABLISHED";
			goto bail3;
		}

		/*
		 * for pipelining, master needs to keep his ah... guys who
		 * queued on him can drop it now though.
		 */

		if (w != wsi)
			/* free up parsing allocations for queued guy */
			lws_header_table_detach(w, 0);

		lwsl_info("%s: client connection up\n", __func__);

		return 0;
	}

#if defined(LWS_ROLE_WS)
	switch (lws_client_ws_upgrade(wsi, &cce)) {
	case 2:
		goto bail2;
	case 3:
		goto bail3;
	}

	return 0;
#endif

bail3:
	close_reason = LWS_CLOSE_STATUS_NOSTATUS;

bail2:
	if (wsi->protocol) {
		n = 0;
		if (cce)
			n = (int)strlen(cce);
		wsi->protocol->callback(wsi,
				LWS_CALLBACK_CLIENT_CONNECTION_ERROR,
				wsi->user_space, (void *)cce,
				(unsigned int)n);
	}
	wsi->already_did_cce = 1;

	lwsl_info("closing connection due to bail2 connection error\n");

	/* closing will free up his parsing allocations */
	lws_close_free_wsi(wsi, close_reason, "c hs interp");

	return 1;
}
#endif

char *
lws_generate_client_handshake(struct lws *wsi, char *pkt)
{
	char *p = pkt;
	const char *meth;
	const char *pp = lws_hdr_simple_ptr(wsi,
				_WSI_TOKEN_CLIENT_SENT_PROTOCOLS);

	meth = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_METHOD);
	if (!meth) {
		meth = "GET";
		wsi->do_ws = 1;
	} else {
		wsi->do_ws = 0;
	}

	if (!strcmp(meth, "RAW")) {
		lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);
		lwsl_notice("client transition to raw\n");

		if (pp) {
			const struct lws_protocols *pr;

			pr = lws_vhost_name_to_protocol(wsi->vhost, pp);

			if (!pr) {
				lwsl_err("protocol %s not enabled on vhost\n",
					 pp);
				return NULL;
			}

			lws_bind_protocol(wsi, pr);
		}

		if ((wsi->protocol->callback)(wsi, LWS_CALLBACK_RAW_ADOPT,
					      wsi->user_space, NULL, 0))
			return NULL;

		lws_role_transition(wsi, 0, LRS_ESTABLISHED, &role_ops_raw_skt);
		lws_header_table_detach(wsi, 1);

		return NULL;
	}

	/*
	 * 04 example client handshake
	 *
	 * GET /chat HTTP/1.1
	 * Host: server.example.com
	 * Upgrade: websocket
	 * Connection: Upgrade
	 * Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
	 * Sec-WebSocket-Origin: http://example.com
	 * Sec-WebSocket-Protocol: chat, superchat
	 * Sec-WebSocket-Version: 4
	 */

	p += sprintf(p, "%s %s HTTP/1.1\x0d\x0a", meth,
		     lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_URI));

	p += sprintf(p, "Pragma: no-cache\x0d\x0a"
			"Cache-Control: no-cache\x0d\x0a");

	p += sprintf(p, "Host: %s\x0d\x0a",
		     lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_HOST));

	if (lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_ORIGIN)) {
		if (lws_check_opt(wsi->context->options,
				  LWS_SERVER_OPTION_JUST_USE_RAW_ORIGIN))
			p += sprintf(p, "Origin: %s\x0d\x0a",
				     lws_hdr_simple_ptr(wsi,
						     _WSI_TOKEN_CLIENT_ORIGIN));
		else
			p += sprintf(p, "Origin: http://%s\x0d\x0a",
				     lws_hdr_simple_ptr(wsi,
						     _WSI_TOKEN_CLIENT_ORIGIN));
	}
#if defined(LWS_ROLE_WS)
	if (wsi->do_ws)
		p = lws_generate_client_ws_handshake(wsi, p);
#endif

	/* give userland a chance to append, eg, cookies */

	if (wsi->protocol->callback(wsi,
				LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER,
				wsi->user_space, &p,
				(pkt + wsi->context->pt_serv_buf_size) - p - 12))
		return NULL;

	p += sprintf(p, "\x0d\x0a");

	return p;
}

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)

LWS_VISIBLE int
lws_http_client_read(struct lws *wsi, char **buf, int *len)
{
	int rlen, n;

	rlen = lws_ssl_capable_read(wsi, (unsigned char *)*buf, *len);
	*len = 0;

	// lwsl_notice("%s: rlen %d\n", __func__, rlen);

	/* allow the source to signal he has data again next time */
	lws_change_pollfd(wsi, 0, LWS_POLLIN);

	if (rlen == LWS_SSL_CAPABLE_ERROR) {
		lwsl_notice("%s: SSL capable error\n", __func__);
		return -1;
	}

	if (rlen == 0)
		return -1;

	if (rlen < 0)
		return 0;

	*len = rlen;
	wsi->client_rx_avail = 0;

	/*
	 * server may insist on transfer-encoding: chunked,
	 * so http client must deal with it
	 */
spin_chunks:
	while (wsi->chunked && (wsi->chunk_parser != ELCP_CONTENT) && *len) {
		switch (wsi->chunk_parser) {
		case ELCP_HEX:
			if ((*buf)[0] == '\x0d') {
				wsi->chunk_parser = ELCP_CR;
				break;
			}
			n = char_to_hex((*buf)[0]);
			if (n < 0) {
				lwsl_debug("chunking failure\n");
				return -1;
			}
			wsi->chunk_remaining <<= 4;
			wsi->chunk_remaining |= n;
			break;
		case ELCP_CR:
			if ((*buf)[0] != '\x0a') {
				lwsl_debug("chunking failure\n");
				return -1;
			}
			wsi->chunk_parser = ELCP_CONTENT;
			lwsl_info("chunk %d\n", wsi->chunk_remaining);
			if (wsi->chunk_remaining)
				break;
			lwsl_info("final chunk\n");
			goto completed;

		case ELCP_CONTENT:
			break;

		case ELCP_POST_CR:
			if ((*buf)[0] != '\x0d') {
				lwsl_debug("chunking failure\n");

				return -1;
			}

			wsi->chunk_parser = ELCP_POST_LF;
			break;

		case ELCP_POST_LF:
			if ((*buf)[0] != '\x0a')
				return -1;

			wsi->chunk_parser = ELCP_HEX;
			wsi->chunk_remaining = 0;
			break;
		}
		(*buf)++;
		(*len)--;
	}

	if (wsi->chunked && !wsi->chunk_remaining)
		return 0;

	if (wsi->http.rx_content_remain &&
	    wsi->http.rx_content_remain < (unsigned int)*len)
		n = (int)wsi->http.rx_content_remain;
	else
		n = *len;

	if (wsi->chunked && wsi->chunk_remaining &&
	    wsi->chunk_remaining < n)
		n = wsi->chunk_remaining;

#ifdef LWS_WITH_HTTP_PROXY
	/* hubbub */
	if (wsi->http.perform_rewrite)
		lws_rewrite_parse(wsi->http.rw, (unsigned char *)*buf, n);
	else
#endif
	{
		struct lws *wsi_eff = lws_client_wsi_effective(wsi);

		if (user_callback_handle_rxflow(wsi_eff->protocol->callback,
				wsi_eff, LWS_CALLBACK_RECEIVE_CLIENT_HTTP_READ,
				wsi_eff->user_space, *buf, n)) {
			lwsl_debug("%s: RECEIVE_CLIENT_HTTP_READ returned -1\n",
				   __func__);

			return -1;
		}
	}

	if (wsi->chunked && wsi->chunk_remaining) {
		(*buf) += n;
		wsi->chunk_remaining -= n;
		*len -= n;
	}

	if (wsi->chunked && !wsi->chunk_remaining)
		wsi->chunk_parser = ELCP_POST_CR;

	if (wsi->chunked && *len)
		goto spin_chunks;

	if (wsi->chunked)
		return 0;

	/* if we know the content length, decrement the content remaining */
	if (wsi->http.rx_content_length > 0)
		wsi->http.rx_content_remain -= n;

	// lwsl_notice("rx_content_remain %lld, rx_content_length %lld\n",
	//	wsi->http.rx_content_remain, wsi->http.rx_content_length);

	if (wsi->http.rx_content_remain || !wsi->http.rx_content_length)
		return 0;

completed:

	if (lws_http_transaction_completed_client(wsi)) {
		lwsl_notice("%s: transaction completed says -1\n", __func__);
		return -1;
	}

	return 0;
}

#endif