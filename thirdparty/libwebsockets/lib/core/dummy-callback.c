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

#include "core/private.h"

#if defined(LWS_WITH_HTTP_PROXY)
static int
proxy_header(struct lws *wsi, struct lws *par, unsigned char *temp,
	     int temp_len, int index, unsigned char **p, unsigned char *end)
{
	int n = lws_hdr_total_length(par, index);

	if (n < 1) {
		lwsl_debug("%s: no index %d:\n", __func__, index);
		return 0;
	}

	if (lws_hdr_copy(par, (char *)temp, temp_len, index) < 0)
		return -1;

	lwsl_debug("%s: index %d: %s\n", __func__, index, (char *)temp);

	if (lws_add_http_header_by_token(wsi, index, temp, n, p, end))
		return -1;

	return 0;
}

static int
stream_close(struct lws *wsi)
{
	char buf[LWS_PRE + 6], *out = buf + LWS_PRE;

	if (wsi->http.did_stream_close)
		return 0;

	wsi->http.did_stream_close = 1;

	if (wsi->http2_substream) {
		if (lws_write(wsi, (unsigned char *)buf + LWS_PRE, 0,
			      LWS_WRITE_HTTP_FINAL) < 0) {
			lwsl_info("%s: COMPL_CLIENT_HTTP: h2 fin wr failed\n",
				  __func__);

			return -1;
		}
	} else {
		*out++ = '0';
		*out++ = '\x0d';
		*out++ = '\x0a';
		*out++ = '\x0d';
		*out++ = '\x0a';

		if (lws_write(wsi, (unsigned char *)buf + LWS_PRE, 5,
			      LWS_WRITE_HTTP_FINAL) < 0) {
			lwsl_err("%s: COMPL_CLIENT_HTTP: "
				 "h2 final write failed\n", __func__);

			return -1;
		}
	}

	return 0;
}

#endif

LWS_VISIBLE int
lws_callback_http_dummy(struct lws *wsi, enum lws_callback_reasons reason,
			void *user, void *in, size_t len)
{
	struct lws_ssl_info *si;
#ifdef LWS_WITH_CGI
	struct lws_cgi_args *args;
#endif
#if defined(LWS_WITH_CGI) || defined(LWS_WITH_HTTP_PROXY)
	char buf[8192];
	int n;
#endif
#if defined(LWS_WITH_HTTP_PROXY)
	unsigned char **p, *end;
	struct lws *parent;
#endif

	switch (reason) {
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	case LWS_CALLBACK_HTTP:
#ifndef LWS_NO_SERVER
		if (lws_return_http_status(wsi, HTTP_STATUS_NOT_FOUND, NULL))
			return -1;

		if (lws_http_transaction_completed(wsi))
#endif
			return -1;
		break;
#if !defined(LWS_NO_SERVER)
	case LWS_CALLBACK_HTTP_BODY_COMPLETION:
	case LWS_CALLBACK_HTTP_FILE_COMPLETION:
		if (lws_http_transaction_completed(wsi))
			return -1;
		break;
#endif

	case LWS_CALLBACK_HTTP_WRITEABLE:
#ifdef LWS_WITH_CGI
		if (wsi->reason_bf & (LWS_CB_REASON_AUX_BF__CGI_HEADERS |
				      LWS_CB_REASON_AUX_BF__CGI)) {
			n = lws_cgi_write_split_stdout_headers(wsi);
			if (n < 0) {
				lwsl_debug("AUX_BF__CGI forcing close\n");
				return -1;
			}
			if (!n)
				lws_rx_flow_control(
					wsi->http.cgi->stdwsi[LWS_STDOUT], 1);

			if (wsi->reason_bf & LWS_CB_REASON_AUX_BF__CGI_HEADERS)
				wsi->reason_bf &=
					~LWS_CB_REASON_AUX_BF__CGI_HEADERS;
			else
				wsi->reason_bf &= ~LWS_CB_REASON_AUX_BF__CGI;

			if (wsi->http.cgi && wsi->http.cgi->cgi_transaction_over)
				return -1;
			break;
		}

		if (wsi->reason_bf & LWS_CB_REASON_AUX_BF__CGI_CHUNK_END) {
			if (!wsi->http2_substream) {
				memcpy(buf + LWS_PRE, "0\x0d\x0a\x0d\x0a", 5);
				lwsl_debug("writing chunk term and exiting\n");
				n = lws_write(wsi, (unsigned char *)buf +
						   LWS_PRE, 5, LWS_WRITE_HTTP);
			} else
				n = lws_write(wsi, (unsigned char *)buf +
						   LWS_PRE, 0,
						   LWS_WRITE_HTTP_FINAL);

			/* always close after sending it */
			return -1;
		}
#endif
#if defined(LWS_WITH_HTTP_PROXY)

		if (wsi->reason_bf & LWS_CB_REASON_AUX_BF__PROXY_HEADERS) {

			wsi->reason_bf &= ~LWS_CB_REASON_AUX_BF__PROXY_HEADERS;

			lwsl_debug("%s: %p: issuing proxy headers\n",
				    __func__, wsi);
			n = lws_write(wsi, wsi->http.pending_return_headers +
					   LWS_PRE,
				      wsi->http.pending_return_headers_len,
				      LWS_WRITE_HTTP_HEADERS);

			lws_free_set_NULL(wsi->http.pending_return_headers);

			if (n < 0) {
				lwsl_err("%s: EST_CLIENT_HTTP: write failed\n",
					 __func__);
				return -1;
			}
			lws_callback_on_writable(wsi);
			break;
		}

		if (wsi->reason_bf & LWS_CB_REASON_AUX_BF__PROXY) {
			char *px = buf + LWS_PRE;
			int lenx = sizeof(buf) - LWS_PRE - 32;

			/*
			 * our sink is writeable and our source has something
			 * to read.  So read a lump of source material of
			 * suitable size to send or what's available, whichever
			 * is the smaller.
			 */
			wsi->reason_bf &= ~LWS_CB_REASON_AUX_BF__PROXY;
			if (!lws_get_child(wsi))
				break;

			/* this causes LWS_CALLBACK_RECEIVE_CLIENT_HTTP_READ */
			if (lws_http_client_read(lws_get_child(wsi), &px,
						 &lenx) < 0) {
				lwsl_info("%s: LWS_CB_REASON_AUX_BF__PROXY: "
					   "client closed\n", __func__);

				stream_close(wsi);

				return -1;
			}
			break;
		}

		if (wsi->reason_bf & LWS_CB_REASON_AUX_BF__PROXY_TRANS_END) {
			lwsl_info("%s: LWS_CB_REASON_AUX_BF__PROXY_TRANS_END\n",
				   __func__);

			wsi->reason_bf &= ~LWS_CB_REASON_AUX_BF__PROXY_TRANS_END;

			if (stream_close(wsi))
				return -1;

			if (lws_http_transaction_completed(wsi))
				return -1;
		}
#endif
		break;

#if defined(LWS_WITH_HTTP_PROXY)
	case LWS_CALLBACK_RECEIVE_CLIENT_HTTP:
		assert(lws_get_parent(wsi));
		if (!lws_get_parent(wsi))
			break;
		lws_get_parent(wsi)->reason_bf |= LWS_CB_REASON_AUX_BF__PROXY;
		lws_callback_on_writable(lws_get_parent(wsi));
		break;

	case LWS_CALLBACK_RECEIVE_CLIENT_HTTP_READ: {
		char *out = buf + LWS_PRE;

		assert(lws_get_parent(wsi));

		if (wsi->http.proxy_parent_chunked) {

			if (len > sizeof(buf) - LWS_PRE - 16) {
				lwsl_err("oversize buf %d %d\n", (int)len,
						(int)sizeof(buf) - LWS_PRE - 16);
				return -1;
			}

			/*
			 * this only needs dealing with on http/1.1 to allow
			 * pipelining
			 */
			n = lws_snprintf(out, 14, "%X\x0d\x0a", (int)len);
			out += n;
			memcpy(out, in, len);
			out += len;
			*out++ = '\x0d';
			*out++ = '\x0a';

			n = lws_write(lws_get_parent(wsi),
				      (unsigned char *)buf + LWS_PRE,
				      len + n + 2, LWS_WRITE_HTTP);
		} else
			n = lws_write(lws_get_parent(wsi), (unsigned char *)in,
				      len, LWS_WRITE_HTTP);
		if (n < 0)
			return -1;
		break; }


	/* this handles the proxy case... */
	case LWS_CALLBACK_ESTABLISHED_CLIENT_HTTP: {
		unsigned char *start, *p, *end;

		/*
		 * We want to proxy these headers, but we are being called
		 * at the point the onward client was established, which is
		 * unrelated to the state or writability of our proxy
		 * connection.
		 *
		 * Therefore produce the headers using the onward client ah
		 * while we have it, and stick them on the output buflist to be
		 * written on the proxy connection as soon as convenient.
		 */

		parent = lws_get_parent(wsi);

		if (!parent)
			return 0;

		start = p = (unsigned char *)buf + LWS_PRE;
		end = p + sizeof(buf) - LWS_PRE - 256;

		if (lws_add_http_header_status(lws_get_parent(wsi),
				lws_http_client_http_response(wsi), &p, end))
			return 1;

		/*
		 * copy these headers from the client connection to the parent
		 */

		proxy_header(parent, wsi, end, 256,
				WSI_TOKEN_HTTP_CONTENT_LENGTH, &p, end);
		proxy_header(parent, wsi, end, 256,
				WSI_TOKEN_HTTP_CONTENT_TYPE, &p, end);
		proxy_header(parent, wsi, end, 256,
				WSI_TOKEN_HTTP_ETAG, &p, end);
		proxy_header(parent, wsi, end, 256,
				WSI_TOKEN_HTTP_ACCEPT_LANGUAGE, &p, end);
		proxy_header(parent, wsi, end, 256,
				WSI_TOKEN_HTTP_CONTENT_ENCODING, &p, end);
		proxy_header(parent, wsi, end, 256,
				WSI_TOKEN_HTTP_CACHE_CONTROL, &p, end);

		if (!parent->http2_substream)
			if (lws_add_http_header_by_token(parent,
				WSI_TOKEN_CONNECTION, (unsigned char *)"close",
				5, &p, end))
			return -1;

		/*
		 * We proxy using h1 only atm, and strip any chunking so it
		 * can go back out on h2 just fine.
		 *
		 * However if we are actually going out on h1, we need to add
		 * our own chunking since we still don't know the size.
		 */

		if (!parent->http2_substream &&
		    !lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_CONTENT_LENGTH)) {
			lwsl_debug("downstream parent chunked\n");
			if (lws_add_http_header_by_token(parent,
					WSI_TOKEN_HTTP_TRANSFER_ENCODING,
					(unsigned char *)"chunked", 7, &p, end))
				return -1;

			wsi->http.proxy_parent_chunked = 1;
		}

		if (lws_finalize_http_header(parent, &p, end))
			return 1;

		parent->http.pending_return_headers_len =
					lws_ptr_diff(p, start);
		parent->http.pending_return_headers =
			lws_malloc(parent->http.pending_return_headers_len +
				    LWS_PRE, "return proxy headers");
		if (!parent->http.pending_return_headers)
			return -1;

		memcpy(parent->http.pending_return_headers + LWS_PRE, start,
		       parent->http.pending_return_headers_len);

		parent->reason_bf |= LWS_CB_REASON_AUX_BF__PROXY_HEADERS;

		lwsl_debug("%s: LWS_CALLBACK_ESTABLISHED_CLIENT_HTTP: "
			   "prepared headers\n", __func__);
		lws_callback_on_writable(parent);

		break; }

	case LWS_CALLBACK_COMPLETED_CLIENT_HTTP:
		lwsl_info("%s: COMPLETED_CLIENT_HTTP: %p (parent %p)\n",
					__func__, wsi, lws_get_parent(wsi));
		if (!lws_get_parent(wsi))
			break;
		lws_get_parent(wsi)->reason_bf |=
				LWS_CB_REASON_AUX_BF__PROXY_TRANS_END;
		lws_callback_on_writable(lws_get_parent(wsi));
		break;

	case LWS_CALLBACK_CLOSED_CLIENT_HTTP:
		if (!lws_get_parent(wsi))
			break;
		lwsl_err("%s: LWS_CALLBACK_CLOSED_CLIENT_HTTP\n", __func__);
		lws_set_timeout(lws_get_parent(wsi), LWS_TO_KILL_ASYNC,
				PENDING_TIMEOUT_KILLED_BY_PROXY_CLIENT_CLOSE);
		break;

	case LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER:
		parent = lws_get_parent(wsi);

		if (!parent)
			break;

		p = (unsigned char **)in;
		end = (*p) + len;

		/*
		 * copy these headers from the parent request to the client
		 * connection's request
		 */

		proxy_header(wsi, parent, (unsigned char *)buf, sizeof(buf),
				WSI_TOKEN_HOST, p, end);
		proxy_header(wsi, parent, (unsigned char *)buf, sizeof(buf),
				WSI_TOKEN_HTTP_ETAG, p, end);
		proxy_header(wsi, parent, (unsigned char *)buf, sizeof(buf),
				WSI_TOKEN_HTTP_IF_MODIFIED_SINCE, p, end);
		proxy_header(wsi, parent, (unsigned char *)buf, sizeof(buf),
				WSI_TOKEN_HTTP_ACCEPT_LANGUAGE, p, end);
		proxy_header(wsi, parent, (unsigned char *)buf, sizeof(buf),
				WSI_TOKEN_HTTP_ACCEPT_ENCODING, p, end);
		proxy_header(wsi, parent, (unsigned char *)buf, sizeof(buf),
				WSI_TOKEN_HTTP_CACHE_CONTROL, p, end);

		buf[0] = '\0';
		lws_get_peer_simple(parent, buf, sizeof(buf));
		if (lws_add_http_header_by_token(wsi, WSI_TOKEN_X_FORWARDED_FOR,
				(unsigned char *)buf, (int)strlen(buf), p, end))
			return -1;

		break;

#endif

#ifdef LWS_WITH_CGI
	/* CGI IO events (POLLIN/OUT) appear here, our default policy is:
	 *
	 *  - POST data goes on subprocess stdin
	 *  - subprocess stdout goes on http via writeable callback
	 *  - subprocess stderr goes to the logs
	 */
	case LWS_CALLBACK_CGI:
		args = (struct lws_cgi_args *)in;
		switch (args->ch) { /* which of stdin/out/err ? */
		case LWS_STDIN:
			/* TBD stdin rx flow control */
			break;
		case LWS_STDOUT:
			/* quench POLLIN on STDOUT until MASTER got writeable */
			lws_rx_flow_control(args->stdwsi[LWS_STDOUT], 0);
			wsi->reason_bf |= LWS_CB_REASON_AUX_BF__CGI;
			/* when writing to MASTER would not block */
			lws_callback_on_writable(wsi);
			break;
		case LWS_STDERR:
			n = lws_get_socket_fd(args->stdwsi[LWS_STDERR]);
			if (n < 0)
				break;
			n = read(n, buf, sizeof(buf) - 2);
			if (n > 0) {
				if (buf[n - 1] != '\n')
					buf[n++] = '\n';
				buf[n] = '\0';
				lwsl_notice("CGI-stderr: %s\n", buf);
			}
			break;
		}
		break;

	case LWS_CALLBACK_CGI_TERMINATED:
		lwsl_debug("LWS_CALLBACK_CGI_TERMINATED: %d %" PRIu64 "\n",
				wsi->http.cgi->explicitly_chunked,
				(uint64_t)wsi->http.cgi->content_length);
		if (!wsi->http.cgi->explicitly_chunked &&
		    !wsi->http.cgi->content_length) {
			/* send terminating chunk */
			lwsl_debug("LWS_CALLBACK_CGI_TERMINATED: ending\n");
			wsi->reason_bf |= LWS_CB_REASON_AUX_BF__CGI_CHUNK_END;
			lws_callback_on_writable(wsi);
			lws_set_timeout(wsi, PENDING_TIMEOUT_CGI, 3);
			break;
		}
		return -1;

	case LWS_CALLBACK_CGI_STDIN_DATA:  /* POST body for stdin */
		args = (struct lws_cgi_args *)in;
		args->data[args->len] = '\0';
		if (!args->stdwsi[LWS_STDIN])
			return -1;
		n = lws_get_socket_fd(args->stdwsi[LWS_STDIN]);
		if (n < 0)
			return -1;

#if defined(LWS_WITH_ZLIB)
		if (wsi->http.cgi->gzip_inflate) {
			/* gzip handling */

			if (!wsi->http.cgi->gzip_init) {
				lwsl_info("inflating gzip\n");

				memset(&wsi->http.cgi->inflate, 0,
				       sizeof(wsi->http.cgi->inflate));

				if (inflateInit2(&wsi->http.cgi->inflate,
						 16 + 15) != Z_OK) {
					lwsl_err("%s: iniflateInit failed\n",
						 __func__);
					return -1;
				}

				wsi->http.cgi->gzip_init = 1;
			}

			wsi->http.cgi->inflate.next_in = args->data;
			wsi->http.cgi->inflate.avail_in = args->len;

			do {

				wsi->http.cgi->inflate.next_out =
						wsi->http.cgi->inflate_buf;
				wsi->http.cgi->inflate.avail_out =
					sizeof(wsi->http.cgi->inflate_buf);

				n = inflate(&wsi->http.cgi->inflate,
					    Z_SYNC_FLUSH);

				switch (n) {
				case Z_NEED_DICT:
				case Z_STREAM_ERROR:
				case Z_DATA_ERROR:
				case Z_MEM_ERROR:
					inflateEnd(&wsi->http.cgi->inflate);
					wsi->http.cgi->gzip_init = 0;
					lwsl_err("zlib error inflate %d\n", n);
					return -1;
				}

				if (wsi->http.cgi->inflate.avail_out !=
					   sizeof(wsi->http.cgi->inflate_buf)) {
					int written;

					written = write(args->stdwsi[LWS_STDIN]->desc.filefd,
						wsi->http.cgi->inflate_buf,
						sizeof(wsi->http.cgi->inflate_buf) -
						wsi->http.cgi->inflate.avail_out);

					if (written != (int)(
						sizeof(wsi->http.cgi->inflate_buf) -
						wsi->http.cgi->inflate.avail_out)) {
						lwsl_notice("LWS_CALLBACK_CGI_STDIN_DATA: "
							"sent %d only %d went", n, args->len);
					}

					if (n == Z_STREAM_END) {
						lwsl_err("gzip inflate end\n");
						inflateEnd(&wsi->http.cgi->inflate);
						wsi->http.cgi->gzip_init = 0;
						break;
					}

				} else
					break;

				if (wsi->http.cgi->inflate.avail_out)
					break;

			} while (1);

			return args->len;
		}
#endif /* WITH_ZLIB */

		n = write(n, args->data, args->len);
//		lwsl_hexdump_notice(args->data, args->len);
		if (n < args->len)
			lwsl_notice("LWS_CALLBACK_CGI_STDIN_DATA: "
				    "sent %d only %d went", n, args->len);

		if (wsi->http.cgi->post_in_expected && args->stdwsi[LWS_STDIN] &&
		    args->stdwsi[LWS_STDIN]->desc.filefd > 0) {
			wsi->http.cgi->post_in_expected -= n;
			if (!wsi->http.cgi->post_in_expected) {
				struct lws *siwsi = args->stdwsi[LWS_STDIN];

				lwsl_debug("%s: expected POST in end: "
					   "closing stdin wsi %p, fd %d\n",
					   __func__, siwsi, siwsi->desc.sockfd);

				__remove_wsi_socket_from_fds(siwsi);
				lwsi_set_state(siwsi, LRS_DEAD_SOCKET);
				siwsi->socket_is_permanently_unusable = 1;
				lws_remove_child_from_any_parent(siwsi);
				if (wsi->context->event_loop_ops->
							close_handle_manually) {
					wsi->context->event_loop_ops->
						close_handle_manually(siwsi);
					siwsi->told_event_loop_closed = 1;
				} else {
					compatible_close(siwsi->desc.sockfd);
					__lws_free_wsi(siwsi);
				}
				wsi->http.cgi->pipe_fds[LWS_STDIN][1] = -1;

				args->stdwsi[LWS_STDIN] = NULL;
			}
		}

		return n;
#endif /* WITH_CGI */
#endif /* ROLE_ H1 / H2 */
	case LWS_CALLBACK_SSL_INFO:
		si = in;

		(void)si;
		lwsl_notice("LWS_CALLBACK_SSL_INFO: where: 0x%x, ret: 0x%x\n",
			    si->where, si->ret);
		break;

#if LWS_MAX_SMP > 1
	case LWS_CALLBACK_GET_THREAD_ID:
		return (int)(unsigned long long)pthread_self();
#endif

	default:
		break;
	}

	return 0;
}
