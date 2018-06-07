/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2014 Andy Green <andy@warmcat.com>
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

#include "private-libwebsockets.h"

int
lws_handshake_client(struct lws *wsi, unsigned char **buf, size_t len)
{
	int m;

	switch (wsi->mode) {
	case LWSCM_WSCL_WAITING_PROXY_REPLY:
	case LWSCM_WSCL_ISSUE_HANDSHAKE:
	case LWSCM_WSCL_WAITING_SERVER_REPLY:
	case LWSCM_WSCL_WAITING_EXTENSION_CONNECT:
	case LWSCM_WS_CLIENT:
		while (len) {
			/*
			 * we were accepting input but now we stopped doing so
			 */
			if (lws_is_flowcontrolled(wsi)) {
				lwsl_debug("%s: caching %ld\n", __func__, (long)len);
				lws_rxflow_cache(wsi, *buf, 0, len);
				return 0;
			}
			if (wsi->u.ws.rx_draining_ext) {
#if !defined(LWS_NO_CLIENT)
				if (wsi->mode == LWSCM_WS_CLIENT)
					m = lws_client_rx_sm(wsi, 0);
				else
#endif
					m = lws_rx_sm(wsi, 0);
				if (m < 0)
					return -1;
				continue;
			}
			/* account for what we're using in rxflow buffer */
			if (wsi->rxflow_buffer)
				wsi->rxflow_pos++;

			if (lws_client_rx_sm(wsi, *(*buf)++)) {
				lwsl_debug("client_rx_sm exited\n");
				return -1;
			}
			len--;
		}
		lwsl_debug("%s: finished with %ld\n", __func__, (long)len);
		return 0;
	default:
		break;
	}

	return 0;
}

LWS_VISIBLE LWS_EXTERN void
lws_client_http_body_pending(struct lws *wsi, int something_left_to_send)
{
	wsi->client_http_body_pending = !!something_left_to_send;
}

int
lws_client_socket_service(struct lws_context *context, struct lws *wsi,
			  struct lws_pollfd *pollfd)
{
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	char *p = (char *)&pt->serv_buf[0];
	const char *cce = NULL;
	unsigned char c;
	char *sb = p;
	int n = 0;
	ssize_t len = 0;
#if defined(LWS_WITH_SOCKS5)
	char conn_mode = 0, pending_timeout = 0;
#endif

	switch (wsi->mode) {

	case LWSCM_WSCL_WAITING_CONNECT:

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
	case LWSCM_WSCL_WAITING_SOCKS_GREETING_REPLY:
	case LWSCM_WSCL_WAITING_SOCKS_AUTH_REPLY:
	case LWSCM_WSCL_WAITING_SOCKS_CONNECT_REPLY:

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

		switch (wsi->mode) {

		case LWSCM_WSCL_WAITING_SOCKS_GREETING_REPLY:
			if (pt->serv_buf[0] != SOCKS_VERSION_5)
				goto socks_reply_fail;

			if (pt->serv_buf[1] == SOCKS_AUTH_NO_AUTH) {
				lwsl_client("SOCKS greeting reply: No Auth Method\n");
				socks_generate_msg(wsi, SOCKS_MSG_CONNECT, &len);
				conn_mode = LWSCM_WSCL_WAITING_SOCKS_CONNECT_REPLY;
				pending_timeout = PENDING_TIMEOUT_AWAITING_SOCKS_CONNECT_REPLY;
				goto socks_send;
			}

			if (pt->serv_buf[1] == SOCKS_AUTH_USERNAME_PASSWORD) {
				lwsl_client("SOCKS greeting reply: User/Pw Method\n");
				socks_generate_msg(wsi, SOCKS_MSG_USERNAME_PASSWORD, &len);
				conn_mode = LWSCM_WSCL_WAITING_SOCKS_AUTH_REPLY;
				pending_timeout = PENDING_TIMEOUT_AWAITING_SOCKS_AUTH_REPLY;
				goto socks_send;
			}
			goto socks_reply_fail;

		case LWSCM_WSCL_WAITING_SOCKS_AUTH_REPLY:
			if (pt->serv_buf[0] != SOCKS_SUBNEGOTIATION_VERSION_1 ||
			    pt->serv_buf[1] != SOCKS_SUBNEGOTIATION_STATUS_SUCCESS)
				goto socks_reply_fail;

			lwsl_client("SOCKS password OK, sending connect\n");
			socks_generate_msg(wsi, SOCKS_MSG_CONNECT, &len);
			conn_mode = LWSCM_WSCL_WAITING_SOCKS_CONNECT_REPLY;
			pending_timeout = PENDING_TIMEOUT_AWAITING_SOCKS_CONNECT_REPLY;
socks_send:
			n = send(wsi->desc.sockfd, (char *)pt->serv_buf, len,
				 MSG_NOSIGNAL);
			if (n < 0) {
				lwsl_debug("ERROR writing to socks proxy\n");
				goto bail3;
			}

			lws_set_timeout(wsi, pending_timeout, AWAITING_TIMEOUT);
			wsi->mode = conn_mode;
			break;

socks_reply_fail:
			lwsl_notice("socks reply: v%d, err %d\n",
				    pt->serv_buf[0], pt->serv_buf[1]);
			goto bail3;

		case LWSCM_WSCL_WAITING_SOCKS_CONNECT_REPLY:
			if (pt->serv_buf[0] != SOCKS_VERSION_5 ||
			    pt->serv_buf[1] != SOCKS_REQUEST_REPLY_SUCCESS)
				goto socks_reply_fail;

			lwsl_client("socks connect OK\n");

			/* free stash since we are done with it */
			lws_free_set_NULL(wsi->u.hdr.stash);
			if (lws_hdr_simple_create(wsi, _WSI_TOKEN_CLIENT_PEER_ADDRESS,
						  wsi->vhost->socks_proxy_address))
				goto bail3;

			wsi->c_port = wsi->vhost->socks_proxy_port;

			/* clear his proxy connection timeout */
			lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);
			goto start_ws_handshake;
		}
		break;
#endif

	case LWSCM_WSCL_WAITING_PROXY_REPLY:

		/* handle proxy hung up on us */

		if (pollfd->revents & LWS_POLLHUP) {

			lwsl_warn("Proxy connection %p (fd=%d) dead\n",
				  (void *)wsi, pollfd->fd);

			goto bail3;
		}

		n = recv(wsi->desc.sockfd, sb, context->pt_serv_buf_size, 0);
		if (n < 0) {
			if (LWS_ERRNO == LWS_EAGAIN) {
				lwsl_debug("Proxy read returned EAGAIN... retrying\n");
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

	case LWSCM_WSCL_ISSUE_HANDSHAKE:

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

#ifdef LWS_OPENSSL_SUPPORT
		/* we can retry this... just cook the SSL BIO the first time */

		if (wsi->use_ssl && !wsi->ssl &&
		    lws_ssl_client_bio_create(wsi) < 0) {
			cce = "bio_create failed";
			goto bail3;
		}

		if (wsi->use_ssl) {
			n = lws_ssl_client_connect1(wsi);
			if (!n)
				return 0;
			if (n < 0) {
				cce = "lws_ssl_client_connect1 failed";
				goto bail3;
			}
		} else
			wsi->ssl = NULL;

		/* fallthru */

	case LWSCM_WSCL_WAITING_SSL:

		if (wsi->use_ssl) {
			n = lws_ssl_client_connect2(wsi);
			if (!n)
				return 0;
			if (n < 0) {
				cce = "lws_ssl_client_connect2 failed";
				goto bail3;
			}
		} else
			wsi->ssl = NULL;
#endif

		wsi->mode = LWSCM_WSCL_ISSUE_HANDSHAKE2;
		lws_set_timeout(wsi, PENDING_TIMEOUT_AWAITING_CLIENT_HS_SEND,
				context->timeout_secs);

		/* fallthru */

	case LWSCM_WSCL_ISSUE_HANDSHAKE2:
		p = lws_generate_client_handshake(wsi, p);
		if (p == NULL) {
			if (wsi->mode == LWSCM_RAW)
				return 0;

			lwsl_err("Failed to generate handshake for client\n");
			lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS);
			return 0;
		}

		/* send our request to the server */
		lws_latency_pre(context, wsi);

		n = lws_ssl_capable_write(wsi, (unsigned char *)sb, p - sb);
		lws_latency(context, wsi, "send lws_issue_raw", n,
			    n == p - sb);
		switch (n) {
		case LWS_SSL_CAPABLE_ERROR:
			lwsl_debug("ERROR writing to client socket\n");
			lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS);
			return 0;
		case LWS_SSL_CAPABLE_MORE_SERVICE:
			lws_callback_on_writable(wsi);
			break;
		}

		if (wsi->client_http_body_pending) {
			wsi->mode = LWSCM_WSCL_ISSUE_HTTP_BODY;
			lws_set_timeout(wsi, PENDING_TIMEOUT_CLIENT_ISSUE_PAYLOAD,
					context->timeout_secs);
			/* user code must ask for writable callback */
			break;
		}

		goto client_http_body_sent;

	case LWSCM_WSCL_ISSUE_HTTP_BODY:
		if (wsi->client_http_body_pending) {
			lws_set_timeout(wsi, PENDING_TIMEOUT_CLIENT_ISSUE_PAYLOAD,
					context->timeout_secs);
			/* user code must ask for writable callback */
			break;
		}
client_http_body_sent:
		wsi->u.hdr.parser_state = WSI_TOKEN_NAME_PART;
		wsi->u.hdr.lextable_pos = 0;
		wsi->mode = LWSCM_WSCL_WAITING_SERVER_REPLY;
		lws_set_timeout(wsi, PENDING_TIMEOUT_AWAITING_SERVER_RESPONSE,
				context->timeout_secs);
		break;

	case LWSCM_WSCL_WAITING_SERVER_REPLY:
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
		while (wsi->u.hdr.parser_state != WSI_PARSING_COMPLETE &&
		       len > 0) {
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

			if (lws_parse(wsi, c)) {
				lwsl_warn("problems parsing header\n");
				goto bail3;
			}
		}

		/*
		 * hs may also be coming in multiple packets, there is a 5-sec
		 * libwebsocket timeout still active here too, so if parsing did
		 * not complete just wait for next packet coming in this state
		 */
		if (wsi->u.hdr.parser_state != WSI_PARSING_COMPLETE)
			break;

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
		lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS);
		return -1;

	case LWSCM_WSCL_WAITING_EXTENSION_CONNECT:
		lwsl_ext("LWSCM_WSCL_WAITING_EXTENSION_CONNECT\n");
		break;

	case LWSCM_WSCL_PENDING_CANDIDATE_CHILD:
		lwsl_ext("LWSCM_WSCL_PENDING_CANDIDATE_CHILD\n");
		break;
	default:
		break;
	}

	return 0;
}

/*
 * In-place str to lower case
 */

static void
strtolower(char *s)
{
	while (*s) {
#ifdef LWS_PLAT_OPTEE
		int tolower_optee(int c);
		*s = tolower_optee((int)*s);
#else
		*s = tolower((int)*s);
#endif
		s++;
	}
}

int LWS_WARN_UNUSED_RESULT
lws_http_transaction_completed_client(struct lws *wsi)
{
	lwsl_debug("%s: wsi %p\n", __func__, wsi);
	/* if we can't go back to accept new headers, drop the connection */
	if (wsi->u.http.connection_type != HTTP_CONNECTION_KEEP_ALIVE) {
		lwsl_info("%s: %p: close connection\n", __func__, wsi);
		return 1;
	}

	/* we don't support chained client connections yet */
	return 1;
#if 0
	/* otherwise set ourselves up ready to go again */
	wsi->state = LWSS_CLIENT_HTTP_ESTABLISHED;
	wsi->mode = LWSCM_HTTP_CLIENT_ACCEPTED;
	wsi->u.http.rx_content_length = 0;
	wsi->hdr_parsing_completed = 0;

	/* He asked for it to stay alive indefinitely */
	lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);

	/*
	 * As client, nothing new is going to come until we ask for it
	 * we can drop the ah, if any
	 */
	if (wsi->u.hdr.ah) {
		lws_header_table_force_to_detachable_state(wsi);
		lws_header_table_detach(wsi, 0);
	}

	/* If we're (re)starting on headers, need other implied init */
	wsi->u.hdr.ues = URIES_IDLE;

	lwsl_info("%s: %p: keep-alive await new transaction\n", __func__, wsi);

	return 0;
#endif
}

LWS_VISIBLE LWS_EXTERN unsigned int
lws_http_client_http_response(struct lws *wsi)
{
	if (!wsi->u.http.ah)
		return 0;

	return wsi->u.http.ah->http_response;
}

int
lws_client_interpret_server_handshake(struct lws *wsi)
{
	int n, len, okay = 0, port = 0, ssl = 0;
	int close_reason = LWS_CLOSE_STATUS_PROTOCOL_ERR;
	struct lws_context *context = wsi->context;
	const char *pc, *prot, *ads = NULL, *path, *cce = NULL;
	struct allocated_headers *ah = NULL;
	char *p, *q;
	char new_path[300];
#ifndef LWS_NO_EXTENSIONS
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	char *sb = (char *)&pt->serv_buf[0];
	const struct lws_ext_options *opts;
	const struct lws_extension *ext;
	char ext_name[128];
	const char *c, *a;
	char ignore;
	int more = 1;
	void *v;
#endif
	if (wsi->u.hdr.stash)
		lws_free_set_NULL(wsi->u.hdr.stash);

	ah = wsi->u.hdr.ah;
	if (!wsi->do_ws) {
		/* we are being an http client...
		 */
		lws_union_transition(wsi, LWSCM_HTTP_CLIENT_ACCEPTED);
		wsi->state = LWSS_CLIENT_HTTP_ESTABLISHED;
		wsi->u.http.ah = ah;
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
	 *
	 *
	 *
	 */

	wsi->u.http.connection_type = HTTP_CONNECTION_KEEP_ALIVE;
	p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP);
	if (wsi->do_ws && !p) {
		lwsl_info("no URI\n");
		cce = "HS: URI missing";
		goto bail3;
	}
	if (!p) {
		p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP1_0);
		wsi->u.http.connection_type = HTTP_CONNECTION_CLOSE;
	}
	if (!p) {
		cce = "HS: URI missing";
		lwsl_info("no URI\n");
		goto bail3;
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
		if (p[0] == '/')
		{
#ifdef LWS_OPENSSL_SUPPORT
			ssl = wsi->use_ssl;
#endif
			ads = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_PEER_ADDRESS);
			port = wsi->c_port;
			path = p + 1; /* +1 as lws_client_reset expects leading / to be omitted */
		}
		/* Absolute (Full) URI */
		else if (strchr(p, ':'))
		{
			if (lws_parse_uri(p, &prot, &ads, &port, &path)) {
				cce = "HS: URI did not parse";
				goto bail3;
			}

			if (!strcmp(prot, "wss") || !strcmp(prot, "https"))
				ssl = 1;
		}
		/* Relative reference relative path */
		else
		{
			/* This doesn't try to calculate an absolute path, that will be left to the server */
#ifdef LWS_OPENSSL_SUPPORT
			ssl = wsi->use_ssl;
#endif
			ads = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_PEER_ADDRESS);
			port = wsi->c_port;
			path = new_path + 1; /* +1 as lws_client_reset expects leading / to be omitted */
			strncpy(new_path, lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_URI), sizeof(new_path));
			new_path[sizeof(new_path) - 1] = '\0';
			q = strrchr(new_path, '/');
			if (q)
			{
				strncpy(q + 1, p, sizeof(new_path) - (q - new_path) - 1);
				new_path[sizeof(new_path) - 1] = '\0';
			}
			else
			{
				path = p;
			}
		}

#ifdef LWS_OPENSSL_SUPPORT
		if (wsi->use_ssl && !ssl) {
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

#ifdef LWS_WITH_HTTP_PROXY
		wsi->perform_rewrite = 0;
		if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_CONTENT_TYPE)) {
			if (!strncmp(lws_hdr_simple_ptr(wsi, WSI_TOKEN_HTTP_CONTENT_TYPE),
				     "text/html", 9))
				wsi->perform_rewrite = 1;
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
		if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_TRANSFER_ENCODING)) {
			wsi->chunked = !strcmp(lws_hdr_simple_ptr(wsi,
					       WSI_TOKEN_HTTP_TRANSFER_ENCODING),
					"chunked");
			/* first thing is hex, after payload there is crlf */
			wsi->chunk_parser = ELCP_HEX;
		}

		if (lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_CONTENT_LENGTH)) {
			wsi->u.http.rx_content_length =
					atoll(lws_hdr_simple_ptr(wsi,
						WSI_TOKEN_HTTP_CONTENT_LENGTH));
			lwsl_notice("%s: incoming content length %llu\n", __func__,
					(unsigned long long)wsi->u.http.rx_content_length);
			wsi->u.http.rx_content_remain = wsi->u.http.rx_content_length;
		} else /* can't do 1.1 without a content length or chunked */
			if (!wsi->chunked)
				wsi->u.http.connection_type = HTTP_CONNECTION_CLOSE;

		/*
		 * we seem to be good to go, give client last chance to check
		 * headers and OK it
		 */
		if (wsi->protocol->callback(wsi, LWS_CALLBACK_CLIENT_FILTER_PRE_ESTABLISH,
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

		/* free up his parsing allocations */
		lws_header_table_detach(wsi, 0);

		lwsl_notice("%s: client connection up\n", __func__);

		return 0;
	}

	if (p && !strncmp(p, "401", 3)) {
		lwsl_warn(
		       "lws_client_handshake: got bad HTTP response '%s'\n", p);
		cce = "HS: ws upgrade unauthorized";
		goto bail3;
	}

	if (p && strncmp(p, "101", 3)) {
		lwsl_warn(
		       "lws_client_handshake: got bad HTTP response '%s'\n", p);
		cce = "HS: ws upgrade response not 101";
		goto bail3;
	}

	if (lws_hdr_total_length(wsi, WSI_TOKEN_ACCEPT) == 0) {
		lwsl_info("no ACCEPT\n");
		cce = "HS: ACCEPT missing";
		goto bail3;
	}

	p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_UPGRADE);
	if (!p) {
		lwsl_info("no UPGRADE\n");
		cce = "HS: UPGRADE missing";
		goto bail3;
	}
	strtolower(p);
	if (strcmp(p, "websocket")) {
		lwsl_warn(
		      "lws_client_handshake: got bad Upgrade header '%s'\n", p);
		cce = "HS: Upgrade to something other than websocket";
		goto bail3;
	}

	p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_CONNECTION);
	if (!p) {
		lwsl_info("no Connection hdr\n");
		cce = "HS: CONNECTION missing";
		goto bail3;
	}
	strtolower(p);
	if (strcmp(p, "upgrade")) {
		lwsl_warn("lws_client_int_s_hs: bad header %s\n", p);
		cce = "HS: UPGRADE malformed";
		goto bail3;
	}

	pc = lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_SENT_PROTOCOLS);
	if (!pc) {
		lwsl_parser("lws_client_int_s_hs: no protocol list\n");
	} else
		lwsl_parser("lws_client_int_s_hs: protocol list '%s'\n", pc);

	/*
	 * confirm the protocol the server wants to talk was in the list
	 * of protocols we offered
	 */

	len = lws_hdr_total_length(wsi, WSI_TOKEN_PROTOCOL);
	if (!len) {
		lwsl_info("lws_client_int_s_hs: WSI_TOKEN_PROTOCOL is null\n");
		/*
		 * no protocol name to work from,
		 * default to first protocol
		 */
		n = 0;
		wsi->protocol = &wsi->vhost->protocols[0];
		goto check_extensions;
	}

	p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_PROTOCOL);
	len = strlen(p);

	while (pc && *pc && !okay) {
		if (!strncmp(pc, p, len) &&
		    (pc[len] == ',' || pc[len] == '\0')) {
			okay = 1;
			continue;
		}
		while (*pc && *pc++ != ',')
			;
		while (*pc && *pc == ' ')
			pc++;
	}

	if (!okay) {
		lwsl_err("lws_client_int_s_hs: got bad protocol %s\n", p);
		cce = "HS: PROTOCOL malformed";
		goto bail2;
	}

	/*
	 * identify the selected protocol struct and set it
	 */
	n = 0;
	wsi->protocol = NULL;
	while (wsi->vhost->protocols[n].callback && !wsi->protocol) {
		if (strcmp(p, wsi->vhost->protocols[n].name) == 0) {
			wsi->protocol = &wsi->vhost->protocols[n];
			break;
		}
		n++;
	}

	if (wsi->protocol == NULL) {
		lwsl_err("lws_client_int_s_hs: fail protocol %s\n", p);
		cce = "HS: Cannot match protocol";
		goto bail2;
	}

check_extensions:
	/*
	 * stitch protocol choice into the vh protocol linked list
	 * We always insert ourselves at the start of the list
	 *
	 * X <-> B
	 * X <-> pAn <-> pB
	 */
	//lwsl_err("%s: pre insert vhost start wsi %p, that wsi prev == %p\n",
	//		__func__,
	//		wsi->vhost->same_vh_protocol_list[n],
	//		wsi->same_vh_protocol_prev);
	wsi->same_vh_protocol_prev = /* guy who points to us */
		&wsi->vhost->same_vh_protocol_list[n];
	wsi->same_vh_protocol_next = /* old first guy is our next */
			wsi->vhost->same_vh_protocol_list[n];
	/* we become the new first guy */
	wsi->vhost->same_vh_protocol_list[n] = wsi;

	if (wsi->same_vh_protocol_next)
		/* old first guy points back to us now */
		wsi->same_vh_protocol_next->same_vh_protocol_prev =
				&wsi->same_vh_protocol_next;

#ifndef LWS_NO_EXTENSIONS
	/* instantiate the accepted extensions */

	if (!lws_hdr_total_length(wsi, WSI_TOKEN_EXTENSIONS)) {
		lwsl_ext("no client extensions allowed by server\n");
		goto check_accept;
	}

	/*
	 * break down the list of server accepted extensions
	 * and go through matching them or identifying bogons
	 */

	if (lws_hdr_copy(wsi, sb, context->pt_serv_buf_size, WSI_TOKEN_EXTENSIONS) < 0) {
		lwsl_warn("ext list from server failed to copy\n");
		cce = "HS: EXT: list too big";
		goto bail2;
	}

	c = sb;
	n = 0;
	ignore = 0;
	a = NULL;
	while (more) {

		if (*c && (*c != ',' && *c != '\t')) {
			if (*c == ';') {
				ignore = 1;
				if (!a)
					a = c + 1;
			}
			if (ignore || *c == ' ') {
				c++;
				continue;
			}

			ext_name[n] = *c++;
			if (n < sizeof(ext_name) - 1)
				n++;
			continue;
		}
		ext_name[n] = '\0';
		ignore = 0;
		if (!*c)
			more = 0;
		else {
			c++;
			if (!n)
				continue;
		}

		/* check we actually support it */

		lwsl_notice("checking client ext %s\n", ext_name);

		n = 0;
		ext = wsi->vhost->extensions;
		while (ext && ext->callback) {
			if (strcmp(ext_name, ext->name)) {
				ext++;
				continue;
			}

			n = 1;
			lwsl_notice("instantiating client ext %s\n", ext_name);

			/* instantiate the extension on this conn */

			wsi->active_extensions[wsi->count_act_ext] = ext;

			/* allow him to construct his ext instance */

			if (ext->callback(lws_get_context(wsi), ext, wsi,
				      LWS_EXT_CB_CLIENT_CONSTRUCT,
				      (void *)&wsi->act_ext_user[wsi->count_act_ext],
				      (void *)&opts, 0)) {
				lwsl_info(" ext %s failed construction\n", ext_name);
				ext++;
				continue;
			}

			/*
			 * allow the user code to override ext defaults if it
			 * wants to
			 */
			ext_name[0] = '\0';
			if (user_callback_handle_rxflow(wsi->protocol->callback,
					wsi, LWS_CALLBACK_WS_EXT_DEFAULTS,
					(char *)ext->name, ext_name,
					sizeof(ext_name))) {
				cce = "HS: EXT: failed setting defaults";
				goto bail2;
			}

			if (ext_name[0] &&
			    lws_ext_parse_options(ext, wsi, wsi->act_ext_user[
						  wsi->count_act_ext], opts, ext_name,
						  strlen(ext_name))) {
				lwsl_err("%s: unable to parse user defaults '%s'",
					 __func__, ext_name);
				cce = "HS: EXT: failed parsing defaults";
				goto bail2;
			}

			/*
			 * give the extension the server options
			 */
			if (a && lws_ext_parse_options(ext, wsi,
					wsi->act_ext_user[wsi->count_act_ext],
					opts, a, c - a)) {
				lwsl_err("%s: unable to parse remote def '%s'",
					 __func__, a);
				cce = "HS: EXT: failed parsing options";
				goto bail2;
			}

			if (ext->callback(lws_get_context(wsi), ext, wsi,
					LWS_EXT_CB_OPTION_CONFIRM,
				      wsi->act_ext_user[wsi->count_act_ext],
				      NULL, 0)) {
				lwsl_err("%s: ext %s rejects server options %s",
					 __func__, ext->name, a);
				cce = "HS: EXT: Rejects server options";
				goto bail2;
			}

			wsi->count_act_ext++;

			ext++;
		}

		if (n == 0) {
			lwsl_warn("Unknown ext '%s'!\n", ext_name);
			cce = "HS: EXT: unknown ext";
			goto bail2;
		}

		a = NULL;
		n = 0;
	}

check_accept:
#endif

	/*
	 * Confirm his accept token is the one we precomputed
	 */

	p = lws_hdr_simple_ptr(wsi, WSI_TOKEN_ACCEPT);
	if (strcmp(p, wsi->u.hdr.ah->initial_handshake_hash_base64)) {
		lwsl_warn("lws_client_int_s_hs: accept '%s' wrong vs '%s'\n", p,
				  wsi->u.hdr.ah->initial_handshake_hash_base64);
		cce = "HS: Accept hash wrong";
		goto bail2;
	}

	/* allocate the per-connection user memory (if any) */
	if (lws_ensure_user_space(wsi)) {
		lwsl_err("Problem allocating wsi user mem\n");
		cce = "HS: OOM";
		goto bail2;
	}

	/*
	 * we seem to be good to go, give client last chance to check
	 * headers and OK it
	 */
	if (wsi->protocol->callback(wsi, LWS_CALLBACK_CLIENT_FILTER_PRE_ESTABLISH,
				    wsi->user_space, NULL, 0)) {
		cce = "HS: Rejected by filter cb";
		goto bail2;
	}

	/* clear his proxy connection timeout */
	lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);

	/* free up his parsing allocations */
	lws_header_table_detach(wsi, 0);

	lws_union_transition(wsi, LWSCM_WS_CLIENT);
	wsi->state = LWSS_ESTABLISHED;
	lws_restart_ws_ping_pong_timer(wsi);

	wsi->rxflow_change_to = LWS_RXFLOW_ALLOW;

	/*
	 * create the frame buffer for this connection according to the
	 * size mentioned in the protocol definition.  If 0 there, then
	 * use a big default for compatibility
	 */
	n = wsi->protocol->rx_buffer_size;
	if (!n)
		n = context->pt_serv_buf_size;
	n += LWS_PRE;
	wsi->u.ws.rx_ubuf = lws_malloc(n + 4 /* 0x0000ffff zlib */, "client frame buffer");
	if (!wsi->u.ws.rx_ubuf) {
		lwsl_err("Out of Mem allocating rx buffer %d\n", n);
		cce = "HS: OOM";
		goto bail2;
	}
       wsi->u.ws.rx_ubuf_alloc = n;
	lwsl_info("Allocating client RX buffer %d\n", n);

#if !defined(LWS_WITH_ESP32)
	if (setsockopt(wsi->desc.sockfd, SOL_SOCKET, SO_SNDBUF, (const char *)&n,
		       sizeof n)) {
		lwsl_warn("Failed to set SNDBUF to %d", n);
		cce = "HS: SO_SNDBUF failed";
		goto bail3;
	}
#endif

	lwsl_debug("handshake OK for protocol %s\n", wsi->protocol->name);

	/* call him back to inform him he is up */

	if (wsi->protocol->callback(wsi, LWS_CALLBACK_CLIENT_ESTABLISHED,
				    wsi->user_space, NULL, 0)) {
		cce = "HS: Rejected at CLIENT_ESTABLISHED";
		goto bail3;
	}
#ifndef LWS_NO_EXTENSIONS
	/*
	 * inform all extensions, not just active ones since they
	 * already know
	 */
	ext = wsi->vhost->extensions;

	while (ext && ext->callback) {
		v = NULL;
		for (n = 0; n < wsi->count_act_ext; n++)
			if (wsi->active_extensions[n] == ext)
				v = wsi->act_ext_user[n];

		ext->callback(context, ext, wsi,
			  LWS_EXT_CB_ANY_WSI_ESTABLISHED, v, NULL, 0);
		ext++;
	}
#endif

	return 0;

bail3:
	close_reason = LWS_CLOSE_STATUS_NOSTATUS;

bail2:
	if (wsi->protocol)
		wsi->protocol->callback(wsi, LWS_CALLBACK_CLIENT_CONNECTION_ERROR,
				wsi->user_space, (void *)cce,
				(unsigned int)strlen(cce));
	wsi->already_did_cce = 1;

	lwsl_info("closing connection due to bail2 connection error\n");

	/* closing will free up his parsing allocations */
	lws_close_free_wsi(wsi, close_reason);

	return 1;
}


char *
lws_generate_client_handshake(struct lws *wsi, char *pkt)
{
	char buf[128], hash[20], key_b64[40], *p = pkt;
	struct lws_context *context = wsi->context;
	const char *meth;
	int n;
#ifndef LWS_NO_EXTENSIONS
	const struct lws_extension *ext;
	int ext_count = 0;
#endif
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

		if ((wsi->protocol->callback)(wsi,
				LWS_CALLBACK_RAW_ADOPT,
				wsi->user_space, NULL, 0))
			return NULL;

		lws_header_table_force_to_detachable_state(wsi);
		lws_union_transition(wsi, LWSCM_RAW);
		lws_header_table_detach(wsi, 1);

		return NULL;
	}

	if (wsi->do_ws) {
		/*
		 * create the random key
		 */
		n = lws_get_random(context, hash, 16);
		if (n != 16) {
			lwsl_err("Unable to read from random dev %s\n",
				 SYSTEM_RANDOM_FILEPATH);
			return NULL;
		}

		lws_b64_encode_string(hash, 16, key_b64, sizeof(key_b64));
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
		if (lws_check_opt(context->options, LWS_SERVER_OPTION_JUST_USE_RAW_ORIGIN))
			p += sprintf(p, "Origin: %s\x0d\x0a",
				     lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_ORIGIN));
		else
			p += sprintf(p, "Origin: http://%s\x0d\x0a",
				     lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_ORIGIN));
	}

	if (wsi->do_ws) {
		p += sprintf(p, "Upgrade: websocket\x0d\x0a"
				"Connection: Upgrade\x0d\x0a"
				"Sec-WebSocket-Key: ");
		strcpy(p, key_b64);
		p += strlen(key_b64);
		p += sprintf(p, "\x0d\x0a");
		if (lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_SENT_PROTOCOLS))
			p += sprintf(p, "Sec-WebSocket-Protocol: %s\x0d\x0a",
			     lws_hdr_simple_ptr(wsi, _WSI_TOKEN_CLIENT_SENT_PROTOCOLS));

		/* tell the server what extensions we could support */

#ifndef LWS_NO_EXTENSIONS
		ext = wsi->vhost->extensions;
		while (ext && ext->callback) {
			n = lws_ext_cb_all_exts(context, wsi,
				   LWS_EXT_CB_CHECK_OK_TO_PROPOSE_EXTENSION,
				   (char *)ext->name, 0);
			if (n) { /* an extension vetos us */
				lwsl_ext("ext %s vetoed\n", (char *)ext->name);
				ext++;
				continue;
			}
			n = wsi->vhost->protocols[0].callback(wsi,
				LWS_CALLBACK_CLIENT_CONFIRM_EXTENSION_SUPPORTED,
					wsi->user_space, (char *)ext->name, 0);

			/*
			 * zero return from callback means
			 * go ahead and allow the extension,
			 * it's what we get if the callback is
			 * unhandled
			 */

			if (n) {
				ext++;
				continue;
			}

			/* apply it */

			if (ext_count)
				*p++ = ',';
			else
				p += sprintf(p, "Sec-WebSocket-Extensions: ");
			p += sprintf(p, "%s", ext->client_offer);
			ext_count++;

			ext++;
		}
		if (ext_count)
			p += sprintf(p, "\x0d\x0a");
#endif

		if (wsi->ietf_spec_revision)
			p += sprintf(p, "Sec-WebSocket-Version: %d\x0d\x0a",
				     wsi->ietf_spec_revision);

		/* prepare the expected server accept response */

		key_b64[39] = '\0'; /* enforce composed length below buf sizeof */
		n = sprintf(buf, "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key_b64);

		lws_SHA1((unsigned char *)buf, n, (unsigned char *)hash);

		lws_b64_encode_string(hash, 20,
				      wsi->u.hdr.ah->initial_handshake_hash_base64,
				      sizeof(wsi->u.hdr.ah->initial_handshake_hash_base64));
	}

	/* give userland a chance to append, eg, cookies */

	if (wsi->protocol->callback(wsi, LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER,
				wsi->user_space, &p, (pkt + context->pt_serv_buf_size) - p - 12))
		return NULL;

	p += sprintf(p, "\x0d\x0a");

	return p;
}

