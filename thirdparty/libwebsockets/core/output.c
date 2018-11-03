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

#include "core/private.h"

/*
 * notice this returns number of bytes consumed, or -1
 */
int lws_issue_raw(struct lws *wsi, unsigned char *buf, size_t len)
{
	struct lws_context *context = lws_get_context(wsi);
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	size_t real_len = len;
	unsigned int n;

	// lwsl_hexdump_err(buf, len);

	/*
	 * Detect if we got called twice without going through the
	 * event loop to handle pending.  This would be caused by either
	 * back-to-back writes in one WRITABLE (illegal) or calling lws_write()
	 * from outside the WRITABLE callback (illegal).
	 */
	if (wsi->could_have_pending) {
		lwsl_hexdump_level(LLL_ERR, buf, len);
		lwsl_err("** %p: vh: %s, prot: %s, role %s: "
			 "Illegal back-to-back write of %lu detected...\n",
			 wsi, wsi->vhost->name, wsi->protocol->name,
			 wsi->role_ops->name,
			 (unsigned long)len);
		// assert(0);

		return -1;
	}

	lws_stats_atomic_bump(wsi->context, pt, LWSSTATS_C_API_WRITE, 1);

	if (!len)
		return 0;
	/* just ignore sends after we cleared the truncation buffer */
	if (lwsi_state(wsi) == LRS_FLUSHING_BEFORE_CLOSE && !wsi->trunc_len)
		return (int)len;

	if (wsi->trunc_len && (buf < wsi->trunc_alloc ||
	    buf > (wsi->trunc_alloc + wsi->trunc_len + wsi->trunc_offset))) {
		lwsl_hexdump_level(LLL_ERR, buf, len);
		lwsl_err("** %p: vh: %s, prot: %s, Sending new %lu, pending truncated ...\n"
			 "   It's illegal to do an lws_write outside of\n"
			 "   the writable callback: fix your code\n",
			 wsi, wsi->vhost->name, wsi->protocol->name,
			 (unsigned long)len);
		assert(0);

		return -1;
	}

	if (!wsi->http2_substream && !lws_socket_is_valid(wsi->desc.sockfd))
		lwsl_warn("** error invalid sock but expected to send\n");

	/* limit sending */
	if (wsi->protocol->tx_packet_size)
		n = (int)wsi->protocol->tx_packet_size;
	else {
		n = (int)wsi->protocol->rx_buffer_size;
		if (!n)
			n = context->pt_serv_buf_size;
	}
	n += LWS_PRE + 4;
	if (n > len)
		n = (int)len;

	/* nope, send it on the socket directly */
	lws_latency_pre(context, wsi);
	n = lws_ssl_capable_write(wsi, buf, n);
	lws_latency(context, wsi, "send lws_issue_raw", n, n == len);

	/* something got written, it can have been truncated now */
	wsi->could_have_pending = 1;

	switch (n) {
	case LWS_SSL_CAPABLE_ERROR:
		/* we're going to close, let close know sends aren't possible */
		wsi->socket_is_permanently_unusable = 1;
		return -1;
	case LWS_SSL_CAPABLE_MORE_SERVICE:
		/*
		 * nothing got sent, not fatal.  Retry the whole thing later,
		 * ie, implying treat it was a truncated send so it gets
		 * retried
		 */
		n = 0;
		break;
	}

	/*
	 * we were already handling a truncated send?
	 */
	if (wsi->trunc_len) {
		lwsl_info("%p partial adv %d (vs %ld)\n", wsi, n, (long)real_len);
		wsi->trunc_offset += n;
		wsi->trunc_len -= n;

		if (!wsi->trunc_len) {
			lwsl_info("** %p partial send completed\n", wsi);
			/* done with it, but don't free it */
			n = (int)real_len;
			if (lwsi_state(wsi) == LRS_FLUSHING_BEFORE_CLOSE) {
				lwsl_info("** %p signalling to close now\n", wsi);
				return -1; /* retry closing now */
			}

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
#if !defined(LWS_WITHOUT_SERVER)
			if (wsi->http.deferred_transaction_completed) {
				lwsl_notice("%s: partial completed, doing "
					    "deferred transaction completed\n",
					    __func__);
				wsi->http.deferred_transaction_completed = 0;
				return lws_http_transaction_completed(wsi);
			}
#endif
#endif
		}
		/* always callback on writeable */
		lws_callback_on_writable(wsi);

		return n;
	}

	if ((unsigned int)n == real_len)
		/* what we just sent went out cleanly */
		return n;

	/*
	 * Newly truncated send.  Buffer the remainder (it will get
	 * first priority next time the socket is writable).
	 */
	lwsl_debug("%p new partial sent %d from %lu total\n", wsi, n,
		    (unsigned long)real_len);

	lws_stats_atomic_bump(wsi->context, pt, LWSSTATS_C_WRITE_PARTIALS, 1);
	lws_stats_atomic_bump(wsi->context, pt, LWSSTATS_B_PARTIALS_ACCEPTED_PARTS, n);

	/*
	 *  - if we still have a suitable malloc lying around, use it
	 *  - or, if too small, reallocate it
	 *  - or, if no buffer, create it
	 */
	if (!wsi->trunc_alloc || real_len - n > wsi->trunc_alloc_len) {
		lws_free(wsi->trunc_alloc);

		wsi->trunc_alloc_len = (unsigned int)(real_len - n);
		wsi->trunc_alloc = lws_malloc(real_len - n,
					      "truncated send alloc");
		if (!wsi->trunc_alloc) {
			lwsl_err("truncated send: unable to malloc %lu\n",
				 (unsigned long)(real_len - n));
			return -1;
		}
	}
	wsi->trunc_offset = 0;
	wsi->trunc_len = (unsigned int)(real_len - n);
	memcpy(wsi->trunc_alloc, buf + n, real_len - n);

#if !defined(LWS_WITH_ESP32)
	if (lws_wsi_is_udp(wsi)) {
		/* stash original destination for fulfilling UDP partials */
		wsi->udp->sa_pending = wsi->udp->sa;
		wsi->udp->salen_pending = wsi->udp->salen;
	}
#endif

	/* since something buffered, force it to get another chance to send */
	lws_callback_on_writable(wsi);

	return (int)real_len;
}

LWS_VISIBLE int lws_write(struct lws *wsi, unsigned char *buf, size_t len,
			  enum lws_write_protocol wp)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];

	if (wsi->parent_carries_io) {
		struct lws_write_passthru pas;

		pas.buf = buf;
		pas.len = len;
		pas.wp = wp;
		pas.wsi = wsi;

		if (wsi->parent->protocol->callback(wsi->parent,
				LWS_CALLBACK_CHILD_WRITE_VIA_PARENT,
				wsi->parent->user_space,
				(void *)&pas, 0))
			return 1;

		return (int)len;
	}

	lws_stats_atomic_bump(wsi->context, pt, LWSSTATS_C_API_LWS_WRITE, 1);

	if ((int)len < 0) {
		lwsl_err("%s: suspicious len int %d, ulong %lu\n", __func__,
				(int)len, (unsigned long)len);
		return -1;
	}

	lws_stats_atomic_bump(wsi->context, pt, LWSSTATS_B_WRITE, len);

#ifdef LWS_WITH_ACCESS_LOG
	wsi->http.access_log.sent += len;
#endif
	if (wsi->vhost)
		wsi->vhost->conn_stats.tx += len;

	assert(wsi->role_ops);
	if (!wsi->role_ops->write_role_protocol)
		return lws_issue_raw(wsi, buf, len);

	return wsi->role_ops->write_role_protocol(wsi, buf, len, &wp);
}

LWS_VISIBLE int
lws_ssl_capable_read_no_ssl(struct lws *wsi, unsigned char *buf, int len)
{
	struct lws_context *context = wsi->context;
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	int n = 0;

	lws_stats_atomic_bump(context, pt, LWSSTATS_C_API_READ, 1);

	if (lws_wsi_is_udp(wsi)) {
#if !defined(LWS_WITH_ESP32)
		wsi->udp->salen = sizeof(wsi->udp->sa);
		n = recvfrom(wsi->desc.sockfd, (char *)buf, len, 0,
			     &wsi->udp->sa, &wsi->udp->salen);
#endif
	} else
		n = recv(wsi->desc.sockfd, (char *)buf, len, 0);

	if (n >= 0) {
		if (wsi->vhost)
			wsi->vhost->conn_stats.rx += n;
		lws_stats_atomic_bump(context, pt, LWSSTATS_B_READ, n);

		return n;
	}

	if (LWS_ERRNO == LWS_EAGAIN ||
	    LWS_ERRNO == LWS_EWOULDBLOCK ||
	    LWS_ERRNO == LWS_EINTR)
		return LWS_SSL_CAPABLE_MORE_SERVICE;

	lwsl_notice("error on reading from skt : %d\n", LWS_ERRNO);
	return LWS_SSL_CAPABLE_ERROR;
}

LWS_VISIBLE int
lws_ssl_capable_write_no_ssl(struct lws *wsi, unsigned char *buf, int len)
{
	int n = 0;

	if (lws_wsi_is_udp(wsi)) {
#if !defined(LWS_WITH_ESP32)
		if (wsi->trunc_len)
			n = sendto(wsi->desc.sockfd, buf, len, 0, &wsi->udp->sa_pending, wsi->udp->salen_pending);
		else
			n = sendto(wsi->desc.sockfd, buf, len, 0, &wsi->udp->sa, wsi->udp->salen);
#endif
	} else
		n = send(wsi->desc.sockfd, (char *)buf, len, MSG_NOSIGNAL);
//	lwsl_info("%s: sent len %d result %d", __func__, len, n);
	if (n >= 0)
		return n;

	if (LWS_ERRNO == LWS_EAGAIN ||
	    LWS_ERRNO == LWS_EWOULDBLOCK ||
	    LWS_ERRNO == LWS_EINTR) {
		if (LWS_ERRNO == LWS_EWOULDBLOCK) {
			lws_set_blocking_send(wsi);
		}

		return LWS_SSL_CAPABLE_MORE_SERVICE;
	}

	lwsl_debug("ERROR writing len %d to skt fd %d err %d / errno %d\n",
			len, wsi->desc.sockfd, n, LWS_ERRNO);

	return LWS_SSL_CAPABLE_ERROR;
}

LWS_VISIBLE int
lws_ssl_pending_no_ssl(struct lws *wsi)
{
	(void)wsi;
#if defined(LWS_WITH_ESP32)
	return 100;
#else
	return 0;
#endif
}
