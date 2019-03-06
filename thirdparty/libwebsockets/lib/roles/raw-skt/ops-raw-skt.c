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

static int
rops_handle_POLLIN_raw_skt(struct lws_context_per_thread *pt, struct lws *wsi,
			   struct lws_pollfd *pollfd)
{
	struct lws_tokens ebuf;
	int n, buffered;

	/* pending truncated sends have uber priority */

	if (lws_has_buffered_out(wsi)) {
		if (!(pollfd->revents & LWS_POLLOUT))
			return LWS_HPI_RET_HANDLED;

		/* drain the output buflist */
		if (lws_issue_raw(wsi, NULL, 0) < 0)
			goto fail;
		/*
		 * we can't afford to allow input processing to send
		 * something new, so spin around he event loop until
		 * he doesn't have any partials
		 */
		return LWS_HPI_RET_HANDLED;
	}

	if ((pollfd->revents & pollfd->events & LWS_POLLIN) &&
	    /* any tunnel has to have been established... */
	    lwsi_state(wsi) != LRS_SSL_ACK_PENDING &&
	    !(wsi->favoured_pollin &&
	      (pollfd->revents & pollfd->events & LWS_POLLOUT))) {

		buffered = lws_buflist_aware_read(pt, wsi, &ebuf);
		switch (ebuf.len) {
		case 0:
			lwsl_info("%s: read 0 len\n", __func__);
			wsi->seen_zero_length_recv = 1;
			lws_change_pollfd(wsi, LWS_POLLIN, 0);

			/*
			 * we need to go to fail here, since it's the only
			 * chance we get to understand that the socket has
			 * closed
			 */
			// goto try_pollout;
			goto fail;

		case LWS_SSL_CAPABLE_ERROR:
			goto fail;
		case LWS_SSL_CAPABLE_MORE_SERVICE:
			goto try_pollout;
		}

		n = user_callback_handle_rxflow(wsi->protocol->callback,
						wsi, LWS_CALLBACK_RAW_RX,
						wsi->user_space, ebuf.token,
						ebuf.len);
		if (n < 0) {
			lwsl_info("LWS_CALLBACK_RAW_RX_fail\n");
			goto fail;
		}

		if (lws_buflist_aware_consume(wsi, &ebuf, ebuf.len, buffered))
			return LWS_HPI_RET_PLEASE_CLOSE_ME;
	} else
		if (wsi->favoured_pollin &&
		    (pollfd->revents & pollfd->events & LWS_POLLOUT))
			/* we balanced the last favouring of pollin */
			wsi->favoured_pollin = 0;

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

	lws_stats_atomic_bump(wsi->context, pt,
				LWSSTATS_C_WRITEABLE_CB, 1);
#if defined(LWS_WITH_STATS)
	if (wsi->active_writable_req_us) {
		uint64_t ul = lws_time_in_microseconds() -
				wsi->active_writable_req_us;

		lws_stats_atomic_bump(wsi->context, pt,
				LWSSTATS_MS_WRITABLE_DELAY, ul);
		lws_stats_atomic_max(wsi->context, pt,
			  LWSSTATS_MS_WORST_WRITABLE_DELAY, ul);
		wsi->active_writable_req_us = 0;
	}
#endif
	n = user_callback_handle_rxflow(wsi->protocol->callback,
			wsi, LWS_CALLBACK_RAW_WRITEABLE,
			wsi->user_space, NULL, 0);
	if (n < 0) {
		lwsl_info("writeable_fail\n");
		goto fail;
	}

	return LWS_HPI_RET_HANDLED;

fail:
	lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS, "raw svc fail");

	return LWS_HPI_RET_WSI_ALREADY_DIED;
}

#if !defined(LWS_NO_SERVER)
static int
rops_adoption_bind_raw_skt(struct lws *wsi, int type, const char *vh_prot_name)
{
	/* no http but socket... must be raw skt */
	if ((type & LWS_ADOPT_HTTP) || !(type & LWS_ADOPT_SOCKET) ||
	    (type & _LWS_ADOPT_FINISH))
		return 0; /* no match */

#if !defined(LWS_WITH_ESP32)
	if (type & LWS_ADOPT_FLAG_UDP)
		/*
		 * these can be >128 bytes, so just alloc for UDP
		 */
		wsi->udp = lws_malloc(sizeof(*wsi->udp), "udp struct");
#endif

	lws_role_transition(wsi, 0, (type & LWS_ADOPT_ALLOW_SSL) ? LRS_SSL_INIT :
				LRS_ESTABLISHED, &role_ops_raw_skt);

	if (vh_prot_name)
		lws_bind_protocol(wsi, wsi->protocol, __func__);
	else
		/* this is the only time he will transition */
		lws_bind_protocol(wsi,
			&wsi->vhost->protocols[wsi->vhost->raw_protocol_index],
			__func__);

	return 1; /* bound */
}
#endif

#if !defined(LWS_NO_CLIENT)
static int
rops_client_bind_raw_skt(struct lws *wsi,
			 const struct lws_client_connect_info *i)
{
	if (!i) {

		/* finalize */

		if (!wsi->user_space && wsi->stash->method)
			if (lws_ensure_user_space(wsi))
				return 1;

		return 0;
	}

	/* we are a fallback if nothing else matched */

	lws_role_transition(wsi, LWSIFR_CLIENT, LRS_UNCONNECTED,
			    &role_ops_raw_skt);

	return 1; /* matched */
}
#endif

struct lws_role_ops role_ops_raw_skt = {
	/* role name */			"raw-skt",
	/* alpn id */			NULL,
	/* check_upgrades */		NULL,
	/* init_context */		NULL,
	/* init_vhost */		NULL,
	/* destroy_vhost */		NULL,
	/* periodic_checks */		NULL,
	/* service_flag_pending */	NULL,
	/* handle_POLLIN */		rops_handle_POLLIN_raw_skt,
	/* handle_POLLOUT */		NULL,
	/* perform_user_POLLOUT */	NULL,
	/* callback_on_writable */	NULL,
	/* tx_credit */			NULL,
	/* write_role_protocol */	NULL,
	/* encapsulation_parent */	NULL,
	/* alpn_negotiated */		NULL,
	/* close_via_role_protocol */	NULL,
	/* close_role */		NULL,
	/* close_kill_connection */	NULL,
	/* destroy_role */		NULL,
#if !defined(LWS_NO_SERVER)
	/* adoption_bind */		rops_adoption_bind_raw_skt,
#else
					NULL,
#endif
#if !defined(LWS_NO_CLIENT)
	/* client_bind */		rops_client_bind_raw_skt,
#else
					NULL,
#endif
	/* writeable cb clnt, srv */	{ LWS_CALLBACK_RAW_WRITEABLE, 0 },
	/* close cb clnt, srv */	{ LWS_CALLBACK_RAW_CLOSE, 0 },
	/* protocol_bind cb c, srv */	{ LWS_CALLBACK_RAW_SKT_BIND_PROTOCOL,
					  LWS_CALLBACK_RAW_SKT_BIND_PROTOCOL },
	/* protocol_unbind cb c, srv */	{ LWS_CALLBACK_RAW_SKT_DROP_PROTOCOL,
					  LWS_CALLBACK_RAW_SKT_DROP_PROTOCOL },
	/* file_handle */		0,
};
