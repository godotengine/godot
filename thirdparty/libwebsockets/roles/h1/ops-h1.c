/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2018 Andy Green <andy@warmcat.com>
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

#include <core/private.h>

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif


/*
 * We have to take care about parsing because the headers may be split
 * into multiple fragments.  They may contain unknown headers with arbitrary
 * argument lengths.  So, we parse using a single-character at a time state
 * machine that is completely independent of packet size.
 *
 * Returns <0 for error or length of chars consumed from buf (up to len)
 */

int
lws_read_h1(struct lws *wsi, unsigned char *buf, lws_filepos_t len)
{
	unsigned char *last_char, *oldbuf = buf;
	lws_filepos_t body_chunk_len;
	size_t n;

	// lwsl_notice("%s: h1 path: wsi state 0x%x\n", __func__, lwsi_state(wsi));

	switch (lwsi_state(wsi)) {

	case LRS_ISSUING_FILE:
		return 0;

	case LRS_ESTABLISHED:

		if (lwsi_role_ws(wsi))
			goto ws_mode;

		if (lwsi_role_client(wsi))
			break;

		wsi->hdr_parsing_completed = 0;

		/* fallthru */

	case LRS_HEADERS:
		if (!wsi->http.ah) {
			lwsl_err("%s: LRS_HEADERS: NULL ah\n", __func__);
			assert(0);
		}
		lwsl_parser("issuing %d bytes to parser\n", (int)len);
#if defined(LWS_ROLE_WS) && !defined(LWS_NO_CLIENT)
		if (lws_ws_handshake_client(wsi, &buf, (size_t)len))
			goto bail;
#endif
		last_char = buf;
		if (lws_handshake_server(wsi, &buf, (size_t)len))
			/* Handshake indicates this session is done. */
			goto bail;

		/* we might have transitioned to RAW */
		if (wsi->role_ops == &role_ops_raw_skt ||
		    wsi->role_ops == &role_ops_raw_file)
			 /* we gave the read buffer to RAW handler already */
			goto read_ok;

		/*
		 * It's possible that we've exhausted our data already, or
		 * rx flow control has stopped us dealing with this early,
		 * but lws_handshake_server doesn't update len for us.
		 * Figure out how much was read, so that we can proceed
		 * appropriately:
		 */
		len -= (buf - last_char);
//		lwsl_debug("%s: thinks we have used %ld\n", __func__, (long)len);

		if (!wsi->hdr_parsing_completed)
			/* More header content on the way */
			goto read_ok;

		switch (lwsi_state(wsi)) {
			case LRS_ESTABLISHED:
			case LRS_HEADERS:
				goto read_ok;
			case LRS_ISSUING_FILE:
				goto read_ok;
			case LRS_BODY:
				wsi->http.rx_content_remain =
						wsi->http.rx_content_length;
				if (wsi->http.rx_content_remain)
					goto http_postbody;

				/* there is no POST content */
				goto postbody_completion;
			default:
				break;
		}
		break;

	case LRS_BODY:
http_postbody:
		lwsl_debug("%s: http post body: remain %d\n", __func__,
			    (int)wsi->http.rx_content_remain);
		while (len && wsi->http.rx_content_remain) {
			/* Copy as much as possible, up to the limit of:
			 * what we have in the read buffer (len)
			 * remaining portion of the POST body (content_remain)
			 */
			body_chunk_len = min(wsi->http.rx_content_remain, len);
			wsi->http.rx_content_remain -= body_chunk_len;
			len -= body_chunk_len;
#ifdef LWS_WITH_CGI
			if (wsi->http.cgi) {
				struct lws_cgi_args args;

				args.ch = LWS_STDIN;
				args.stdwsi = &wsi->http.cgi->stdwsi[0];
				args.data = buf;
				args.len = body_chunk_len;

				/* returns how much used */
				n = user_callback_handle_rxflow(
					wsi->protocol->callback,
					wsi, LWS_CALLBACK_CGI_STDIN_DATA,
					wsi->user_space,
					(void *)&args, 0);
				if ((int)n < 0)
					goto bail;
			} else {
#endif
				n = wsi->protocol->callback(wsi,
					LWS_CALLBACK_HTTP_BODY, wsi->user_space,
					buf, (size_t)body_chunk_len);
				if (n)
					goto bail;
				n = (size_t)body_chunk_len;
#ifdef LWS_WITH_CGI
			}
#endif
			buf += n;

			if (wsi->http.rx_content_remain)  {
				lws_set_timeout(wsi, PENDING_TIMEOUT_HTTP_CONTENT,
						wsi->context->timeout_secs);
				break;
			}
			/* he sent all the content in time */
postbody_completion:
#ifdef LWS_WITH_CGI
			/*
			 * If we're running a cgi, we can't let him off the
			 * hook just because he sent his POST data
			 */
			if (wsi->http.cgi)
				lws_set_timeout(wsi, PENDING_TIMEOUT_CGI,
						wsi->context->timeout_secs);
			else
#endif
			lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);
#ifdef LWS_WITH_CGI
			if (!wsi->http.cgi)
#endif
			{
				lwsl_info("HTTP_BODY_COMPLETION: %p (%s)\n",
					  wsi, wsi->protocol->name);
				n = wsi->protocol->callback(wsi,
					LWS_CALLBACK_HTTP_BODY_COMPLETION,
					wsi->user_space, NULL, 0);
				if (n)
					goto bail;

				if (wsi->http2_substream)
					lwsi_set_state(wsi, LRS_ESTABLISHED);
			}

			break;
		}
		break;

	case LRS_RETURNED_CLOSE:
	case LRS_AWAITING_CLOSE_ACK:
	case LRS_WAITING_TO_SEND_CLOSE:
	case LRS_SHUTDOWN:

ws_mode:
#if !defined(LWS_NO_CLIENT) && defined(LWS_ROLE_WS)
		// lwsl_notice("%s: ws_mode\n", __func__);
		if (lws_ws_handshake_client(wsi, &buf, (size_t)len))
			goto bail;
#endif
#if defined(LWS_ROLE_WS)
		if (lwsi_role_ws(wsi) && lwsi_role_server(wsi) &&
			/*
			 * for h2 we are on the swsi
			 */
		    lws_parse_ws(wsi, &buf, (size_t)len) < 0) {
			lwsl_info("%s: lws_parse_ws bailed\n", __func__);
			goto bail;
		}
#endif
		// lwsl_notice("%s: ws_mode: buf moved on by %d\n", __func__,
		//	       lws_ptr_diff(buf, oldbuf));
		break;

	case LRS_DEFERRING_ACTION:
		lwsl_debug("%s: LRS_DEFERRING_ACTION\n", __func__);
		break;

	case LRS_SSL_ACK_PENDING:
		break;

	case LRS_DEAD_SOCKET:
		lwsl_err("%s: Unhandled state LRS_DEAD_SOCKET\n", __func__);
		goto bail;
		// assert(0);
		/* fallthru */

	default:
		lwsl_err("%s: Unhandled state %d\n", __func__, lwsi_state(wsi));
		assert(0);
		goto bail;
	}

read_ok:
	/* Nothing more to do for now */
//	lwsl_info("%s: %p: read_ok, used %ld (len %d, state %d)\n", __func__,
//		  wsi, (long)(buf - oldbuf), (int)len, wsi->state);

	return lws_ptr_diff(buf, oldbuf);

bail:
	/*
	 * h2 / h2-ws calls us recursively in
	 *
	 * lws_read_h1()->
	 *   lws_h2_parser()->
	 *     lws_read_h1()
	 *
	 * pattern, having stripped the h2 framing in the middle.
	 *
	 * When taking down the whole connection, make sure that only the
	 * outer lws_read() does the wsi close.
	 */
	if (!wsi->outer_will_close)
		lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS,
				   "lws_read_h1 bail");

	return -1;
}
#if !defined(LWS_NO_SERVER)
static int
lws_h1_server_socket_service(struct lws *wsi, struct lws_pollfd *pollfd)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	struct lws_tokens ebuf;
	int n, buffered;

	if (lwsi_state(wsi) == LRS_DEFERRING_ACTION)
		goto try_pollout;

	/* any incoming data ready? */

	if (!(pollfd->revents & pollfd->events & LWS_POLLIN))
		goto try_pollout;

	/*
	 * If we previously just did POLLIN when IN and OUT were signaled
	 * (because POLLIN processing may have used up the POLLOUT), don't let
	 * that happen twice in a row... next time we see the situation favour
	 * POLLOUT
	 */

	if (wsi->favoured_pollin &&
	    (pollfd->revents & pollfd->events & LWS_POLLOUT)) {
		// lwsl_notice("favouring pollout\n");
		wsi->favoured_pollin = 0;
		goto try_pollout;
	}

	/*
	 * We haven't processed that the tunnel is set up yet, so
	 * defer reading
	 */

	if (lwsi_state(wsi) == LRS_SSL_ACK_PENDING)
		return LWS_HPI_RET_HANDLED;

	/* these states imply we MUST have an ah attached */

	if ((lwsi_state(wsi) == LRS_ESTABLISHED ||
	     lwsi_state(wsi) == LRS_ISSUING_FILE ||
	     lwsi_state(wsi) == LRS_HEADERS ||
	     lwsi_state(wsi) == LRS_BODY)) {

		if (!wsi->http.ah && lws_header_table_attach(wsi, 0)) {
			lwsl_info("%s: wsi %p: ah not available\n", __func__, wsi);
			goto try_pollout;
		}

		/*
		 * We got here because there was specifically POLLIN...
		 * regardless of our buflist state, we need to get it,
		 * and either use it, or append to the buflist and use
		 * buflist head material.
		 *
		 * We will not notice a connection close until the buflist is
		 * exhausted and we tried to do a read of some kind.
		 */

		buffered = lws_buflist_aware_read(pt, wsi, &ebuf);
		switch (ebuf.len) {
		case 0:
			lwsl_info("%s: read 0 len a\n", __func__);
			wsi->seen_zero_length_recv = 1;
			lws_change_pollfd(wsi, LWS_POLLIN, 0);
#if !defined(LWS_WITHOUT_EXTENSIONS)
			/*
			 * autobahn requires us to win the race between close
			 * and draining the extensions
			 */
			if (wsi->ws &&
			    (wsi->ws->rx_draining_ext || wsi->ws->tx_draining_ext))
				goto try_pollout;
#endif
			/*
			 * normally, we respond to close with logically closing
			 * our side immediately
			 */
			goto fail;

		case LWS_SSL_CAPABLE_ERROR:
			goto fail;
		case LWS_SSL_CAPABLE_MORE_SERVICE:
			goto try_pollout;
		}

		/* just ignore incoming if waiting for close */
		if (lwsi_state(wsi) == LRS_FLUSHING_BEFORE_CLOSE) {
			lwsl_notice("%s: just ignoring\n", __func__);
			goto try_pollout;
		}

		if (lwsi_state(wsi) == LRS_ISSUING_FILE) {
			// lwsl_notice("stashing: wsi %p: bd %d\n", wsi, buffered);
			if (lws_buflist_aware_consume(wsi, &ebuf, 0, buffered))
				return LWS_HPI_RET_PLEASE_CLOSE_ME;

			goto try_pollout;
		}

		/*
		 * Otherwise give it to whoever wants it according to the
		 * connection state
		 */
#if defined(LWS_ROLE_H2)
		if (lwsi_role_h2(wsi) && lwsi_state(wsi) != LRS_BODY)
			n = lws_read_h2(wsi, (uint8_t *)ebuf.token, ebuf.len);
		else
#endif
			n = lws_read_h1(wsi, (uint8_t *)ebuf.token, ebuf.len);
		if (n < 0) /* we closed wsi */
			return LWS_HPI_RET_WSI_ALREADY_DIED;

		lwsl_debug("%s: consumed %d\n", __func__, n);

		if (lws_buflist_aware_consume(wsi, &ebuf, n, buffered))
			return LWS_HPI_RET_PLEASE_CLOSE_ME;

		/*
		 * during the parsing our role changed to something non-http,
		 * so the ah has no further meaning
		 */

		if (wsi->http.ah &&
		    !lwsi_role_h1(wsi) &&
		    !lwsi_role_h2(wsi) &&
		    !lwsi_role_cgi(wsi))
			lws_header_table_detach(wsi, 0);

		/*
		 * He may have used up the writability above, if we will defer
		 * POLLOUT processing in favour of POLLIN, note it
		 */

		if (pollfd->revents & LWS_POLLOUT)
			wsi->favoured_pollin = 1;

		return LWS_HPI_RET_HANDLED;
	}

	/*
	 * He may have used up the writability above, if we will defer POLLOUT
	 * processing in favour of POLLIN, note it
	 */

	if (pollfd->revents & LWS_POLLOUT)
		wsi->favoured_pollin = 1;

try_pollout:

	/* this handles POLLOUT for http serving fragments */

	if (!(pollfd->revents & LWS_POLLOUT))
		return LWS_HPI_RET_HANDLED;

	/* one shot */
	if (lws_change_pollfd(wsi, LWS_POLLOUT, 0)) {
		lwsl_notice("%s a\n", __func__);
		goto fail;
	}

	/* clear back-to-back write detection */
	wsi->could_have_pending = 0;

	if (lwsi_state(wsi) == LRS_DEFERRING_ACTION) {
		lwsl_debug("%s: LRS_DEFERRING_ACTION now writable\n", __func__);

		lwsi_set_state(wsi, LRS_ESTABLISHED);
		if (lws_change_pollfd(wsi, LWS_POLLOUT, 0)) {
			lwsl_info("failed at set pollfd\n");
			goto fail;
		}
	}

	if (!wsi->hdr_parsing_completed)
		return LWS_HPI_RET_HANDLED;

	if (lwsi_state(wsi) != LRS_ISSUING_FILE) {

		if (wsi->trunc_len) {
			//lwsl_notice("%s: completing partial\n", __func__);
			if (lws_issue_raw(wsi, wsi->trunc_alloc + wsi->trunc_offset,
					  wsi->trunc_len) < 0) {
				lwsl_info("%s signalling to close\n", __func__);
				goto fail;
			}
			return LWS_HPI_RET_HANDLED;
		}

		lws_stats_atomic_bump(wsi->context, pt,
					LWSSTATS_C_WRITEABLE_CB, 1);
#if defined(LWS_WITH_STATS)
		if (wsi->active_writable_req_us) {
			uint64_t ul = time_in_microseconds() -
					wsi->active_writable_req_us;

			lws_stats_atomic_bump(wsi->context, pt,
					LWSSTATS_MS_WRITABLE_DELAY, ul);
			lws_stats_atomic_max(wsi->context, pt,
				  LWSSTATS_MS_WORST_WRITABLE_DELAY, ul);
			wsi->active_writable_req_us = 0;
		}
#endif

		n = user_callback_handle_rxflow(wsi->protocol->callback, wsi,
						LWS_CALLBACK_HTTP_WRITEABLE,
						wsi->user_space, NULL, 0);
		if (n < 0) {
			lwsl_info("writeable_fail\n");
			goto fail;
		}

		return LWS_HPI_RET_HANDLED;
	}

	/* >0 == completion, <0 == error
	 *
	 * We'll get a LWS_CALLBACK_HTTP_FILE_COMPLETION callback when
	 * it's done.  That's the case even if we just completed the
	 * send, so wait for that.
	 */
	n = lws_serve_http_file_fragment(wsi);
	if (n < 0)
		goto fail;

	return LWS_HPI_RET_HANDLED;


fail:
	lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS,
			   "server socket svc fail");

	return LWS_HPI_RET_WSI_ALREADY_DIED;
}
#endif

static int
rops_handle_POLLIN_h1(struct lws_context_per_thread *pt, struct lws *wsi,
		       struct lws_pollfd *pollfd)
{

//	lwsl_notice("%s: %p: wsistate 0x%x %s, revents 0x%x\n", __func__, wsi,
//			wsi->wsistate, wsi->role_ops->name, pollfd->revents);

#ifdef LWS_WITH_CGI
	if (wsi->http.cgi && (pollfd->revents & LWS_POLLOUT)) {
		if (lws_handle_POLLOUT_event(wsi, pollfd))
			return LWS_HPI_RET_PLEASE_CLOSE_ME;

		return LWS_HPI_RET_HANDLED;
	}
#endif

        if (lws_is_flowcontrolled(wsi))
                /* We cannot deal with any kind of new RX because we are
                 * RX-flowcontrolled.
                 */
		return LWS_HPI_RET_HANDLED;

#if !defined(LWS_NO_SERVER)
	if (!lwsi_role_client(wsi)) {
		int n;

		lwsl_debug("%s: %p: wsistate 0x%x\n", __func__, wsi, wsi->wsistate);
		n = lws_h1_server_socket_service(wsi, pollfd);
		if (n != LWS_HPI_RET_HANDLED)
			return n;
		if (lwsi_state(wsi) != LRS_SSL_INIT)
			if (lws_server_socket_service_ssl(wsi, LWS_SOCK_INVALID))
				return LWS_HPI_RET_PLEASE_CLOSE_ME;

		return LWS_HPI_RET_HANDLED;
	}
#endif

#ifndef LWS_NO_CLIENT
	if ((pollfd->revents & LWS_POLLIN) &&
	     wsi->hdr_parsing_completed && !wsi->told_user_closed) {

		/*
		 * In SSL mode we get POLLIN notification about
		 * encrypted data in.
		 *
		 * But that is not necessarily related to decrypted
		 * data out becoming available; in may need to perform
		 * other in or out before that happens.
		 *
		 * simply mark ourselves as having readable data
		 * and turn off our POLLIN
		 */
		wsi->client_rx_avail = 1;
		lws_change_pollfd(wsi, LWS_POLLIN, 0);

		//lwsl_notice("calling back %s\n", wsi->protocol->name);

		/* let user code know, he'll usually ask for writeable
		 * callback and drain / re-enable it there
		 */
		if (user_callback_handle_rxflow(
				wsi->protocol->callback,
				wsi, LWS_CALLBACK_RECEIVE_CLIENT_HTTP,
				wsi->user_space, NULL, 0)) {
			lwsl_info("RECEIVE_CLIENT_HTTP closed it\n");
			return LWS_HPI_RET_PLEASE_CLOSE_ME;
		}

		return LWS_HPI_RET_HANDLED;
	}
#endif

//	if (lwsi_state(wsi) == LRS_ESTABLISHED)
//		return LWS_HPI_RET_HANDLED;

#if !defined(LWS_NO_CLIENT)
	if ((pollfd->revents & LWS_POLLOUT) &&
	    lws_handle_POLLOUT_event(wsi, pollfd)) {
		lwsl_debug("POLLOUT event closed it\n");
		return LWS_HPI_RET_PLEASE_CLOSE_ME;
	}

	if (lws_client_socket_service(wsi, pollfd, NULL))
		return LWS_HPI_RET_WSI_ALREADY_DIED;
#endif

	return LWS_HPI_RET_HANDLED;
}

int rops_handle_POLLOUT_h1(struct lws *wsi)
{
	if (lwsi_state(wsi) == LRS_ISSUE_HTTP_BODY)
		return LWS_HP_RET_USER_SERVICE;

	if (lwsi_role_client(wsi))
		return LWS_HP_RET_USER_SERVICE;

	return LWS_HP_RET_BAIL_OK;
}

static int
rops_write_role_protocol_h1(struct lws *wsi, unsigned char *buf, size_t len,
			    enum lws_write_protocol *wp)
{
#if 0
	/* if not in a state to send stuff, then just send nothing */

	if ((lwsi_state(wsi) != LRS_RETURNED_CLOSE &&
	     lwsi_state(wsi) != LRS_WAITING_TO_SEND_CLOSE &&
	     lwsi_state(wsi) != LRS_AWAITING_CLOSE_ACK)) {
		//assert(0);
		lwsl_debug("binning %d %d\n", lwsi_state(wsi), *wp);
		return 0;
	}
#endif

	return lws_issue_raw(wsi, (unsigned char *)buf, len);
}

static int
rops_alpn_negotiated_h1(struct lws *wsi, const char *alpn)
{
	lwsl_debug("%s: client %d\n", __func__, lwsi_role_client(wsi));
#if !defined(LWS_NO_CLIENT)
	if (lwsi_role_client(wsi)) {
		/*
		 * If alpn asserts it is http/1.1, server support for KA is
		 * mandatory.
		 *
		 * Knowing this lets us proceed with sending pipelined headers
		 * before we received the first response headers.
		 */
		wsi->keepalive_active = 1;
	}
#endif

	return 0;
}

static int
rops_destroy_role_h1(struct lws *wsi)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	struct allocated_headers *ah;

	/* we may not have an ah, but may be on the waiting list... */
	lwsl_info("%s: ah det due to close\n", __func__);
	__lws_header_table_detach(wsi, 0);

	 ah = pt->http.ah_list;

	while (ah) {
		if (ah->in_use && ah->wsi == wsi) {
			lwsl_err("%s: ah leak: wsi %p\n", __func__, wsi);
			ah->in_use = 0;
			ah->wsi = NULL;
			pt->http.ah_count_in_use--;
			break;
		}
		ah = ah->next;
	}

#ifdef LWS_ROLE_WS
	lws_free_set_NULL(wsi->ws);
#endif
	return 0;
}

struct lws_role_ops role_ops_h1 = {
	/* role name */			"h1",
	/* alpn id */			"http/1.1",
	/* check_upgrades */		NULL,
	/* init_context */		NULL,
	/* init_vhost */		NULL,
	/* destroy_vhost */		NULL,
	/* periodic_checks */		NULL,
	/* service_flag_pending */	NULL,
	/* handle_POLLIN */		rops_handle_POLLIN_h1,
	/* handle_POLLOUT */		rops_handle_POLLOUT_h1,
	/* perform_user_POLLOUT */	NULL,
	/* callback_on_writable */	NULL,
	/* tx_credit */			NULL,
	/* write_role_protocol */	rops_write_role_protocol_h1,
	/* encapsulation_parent */	NULL,
	/* alpn_negotiated */		rops_alpn_negotiated_h1,
	/* close_via_role_protocol */	NULL,
	/* close_role */		NULL,
	/* close_kill_connection */	NULL,
	/* destroy_role */		rops_destroy_role_h1,
	/* writeable cb clnt, srv */	{ LWS_CALLBACK_CLIENT_HTTP_WRITEABLE,
					  LWS_CALLBACK_HTTP_WRITEABLE },
	/* close cb clnt, srv */	{ LWS_CALLBACK_CLOSED_CLIENT_HTTP,
					  LWS_CALLBACK_CLOSED_HTTP },
	/* file_handle */		0,
};
