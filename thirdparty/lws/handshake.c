/*
 * libwebsockets - small server side websockets and web server implementation
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

#include "private-libwebsockets.h"

/*
 * -04 of the protocol (actually the 80th version) has a radically different
 * handshake.  The 04 spec gives the following idea
 *
 *    The handshake from the client looks as follows:
 *
 *      GET /chat HTTP/1.1
 *      Host: server.example.com
 *      Upgrade: websocket
 *      Connection: Upgrade
 *      Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
 *      Sec-WebSocket-Origin: http://example.com
 *      Sec-WebSocket-Protocol: chat, superchat
 *	Sec-WebSocket-Version: 4
 *
 *  The handshake from the server looks as follows:
 *
 *       HTTP/1.1 101 Switching Protocols
 *       Upgrade: websocket
 *       Connection: Upgrade
 *       Sec-WebSocket-Accept: me89jWimTRKTWwrS3aRrL53YZSo=
 *       Sec-WebSocket-Nonce: AQIDBAUGBwgJCgsMDQ4PEC==
 *       Sec-WebSocket-Protocol: chat
 */

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

LWS_VISIBLE int
lws_read(struct lws *wsi, unsigned char *buf, lws_filepos_t len)
{
	unsigned char *last_char, *oldbuf = buf;
	lws_filepos_t body_chunk_len;
	size_t n;

	switch (wsi->state) {
#ifdef LWS_WITH_HTTP2
	case LWSS_HTTP2_AWAIT_CLIENT_PREFACE:
	case LWSS_HTTP2_ESTABLISHED_PRE_SETTINGS:
	case LWSS_HTTP2_ESTABLISHED:
		n = 0;
		//lwsl_debug("%s: starting new block of %d\n", __func__, (int)len);
		/*
		 * wsi here is always the network connection wsi, not a stream
		 * wsi.
		 */
		while (n < len) {
			/*
			 * we were accepting input but now we stopped doing so
			 */
			if (lws_is_flowcontrolled(wsi)) {
				lws_rxflow_cache(wsi, buf, n, len);

				return 1;
			}

			/* account for what we're using in rxflow buffer */
			if (wsi->rxflow_buffer) {
				wsi->rxflow_pos++;
				assert(wsi->rxflow_pos <= wsi->rxflow_len);
			}

			if (lws_h2_parser(wsi, buf[n++])) {
				lwsl_debug("%s: http2_parser bailed\n", __func__);
				goto bail;
			}
		}
		lwsl_debug("%s: used up block of %d\n", __func__, (int)len);
		break;
#endif

	case LWSS_HTTP_ISSUING_FILE:
		return 0;

	case LWSS_CLIENT_HTTP_ESTABLISHED:
		break;

	case LWSS_HTTP:
		wsi->hdr_parsing_completed = 0;

		/* fallthru */

	case LWSS_HTTP_HEADERS:
		if (!wsi->u.hdr.ah) {
			lwsl_err("%s: LWSS_HTTP_HEADERS: NULL ah\n", __func__);
			assert(0);
		}
		lwsl_parser("issuing %d bytes to parser\n", (int)len);

		lwsl_hexdump(buf, (size_t)len);

		if (lws_handshake_client(wsi, &buf, (size_t)len))
			goto bail;

		last_char = buf;
		if (lws_handshake_server(wsi, &buf, (size_t)len))
			/* Handshake indicates this session is done. */
			goto bail;

		/* we might have transitioned to RAW */
		if (wsi->mode == LWSCM_RAW)
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
		lwsl_debug("%s: thinks we have used %ld\n", __func__, (long)len);

		if (!wsi->hdr_parsing_completed)
			/* More header content on the way */
			goto read_ok;

		switch (wsi->state) {
			case LWSS_HTTP:
			case LWSS_HTTP_HEADERS:
				goto read_ok;
			case LWSS_HTTP_ISSUING_FILE:
				goto read_ok;
			case LWSS_HTTP_BODY:
				wsi->u.http.rx_content_remain =
						wsi->u.http.rx_content_length;
				if (wsi->u.http.rx_content_remain)
					goto http_postbody;

				/* there is no POST content */
				goto postbody_completion;
			default:
				break;
		}
		break;

	case LWSS_HTTP_BODY:
http_postbody:
		//lwsl_notice("http post body\n");
		while (len && wsi->u.http.rx_content_remain) {
			/* Copy as much as possible, up to the limit of:
			 * what we have in the read buffer (len)
			 * remaining portion of the POST body (content_remain)
			 */
			body_chunk_len = min(wsi->u.http.rx_content_remain, len);
			wsi->u.http.rx_content_remain -= body_chunk_len;
			len -= body_chunk_len;
#ifdef LWS_WITH_CGI
			if (wsi->cgi) {
				struct lws_cgi_args args;

				args.ch = LWS_STDIN;
				args.stdwsi = &wsi->cgi->stdwsi[0];
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

			if (wsi->u.http.rx_content_remain)  {
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
			if (wsi->cgi)
				lws_set_timeout(wsi, PENDING_TIMEOUT_CGI,
						wsi->context->timeout_secs);
			else
#endif
			lws_set_timeout(wsi, NO_PENDING_TIMEOUT, 0);
#ifdef LWS_WITH_CGI
			if (!wsi->cgi)
#endif
			{
				lwsl_notice("LWS_CALLBACK_HTTP_BODY_COMPLETION\n");
				n = wsi->protocol->callback(wsi,
					LWS_CALLBACK_HTTP_BODY_COMPLETION,
					wsi->user_space, NULL, 0);
				if (n)
					goto bail;

				if (wsi->http2_substream)
					wsi->state = LWSS_HTTP2_ESTABLISHED;
			}

			break;
		}
		break;

	case LWSS_ESTABLISHED:
	case LWSS_AWAITING_CLOSE_ACK:
	case LWSS_WAITING_TO_SEND_CLOSE_NOTIFICATION:
	case LWSS_SHUTDOWN:
		if (lws_handshake_client(wsi, &buf, (size_t)len))
			goto bail;
		switch (wsi->mode) {
		case LWSCM_WS_SERVING:

			if (lws_interpret_incoming_packet(wsi, &buf, (size_t)len) < 0) {
				lwsl_info("interpret_incoming_packet has bailed\n");
				goto bail;
			}
			break;
		}
		break;
	default:
		lwsl_err("%s: Unhandled state %d\n", __func__, wsi->state);
		break;
	}

read_ok:
	/* Nothing more to do for now */
	lwsl_info("%s: read_ok, used %ld\n", __func__, (long)(buf - oldbuf));

	return buf - oldbuf;

bail:
	lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS);

	return -1;
}
