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

int
_lws_change_pollfd(struct lws *wsi, int _and, int _or, struct lws_pollargs *pa)
{
	struct lws_context_per_thread *pt;
	struct lws_context *context;
	int ret = 0, pa_events = 1;
	struct lws_pollfd *pfd;
	int sampled_tid, tid;

	if (!wsi || wsi->position_in_fds_table < 0)
		return 0;

	if (wsi->handling_pollout && !_and && _or == LWS_POLLOUT) {
		/*
		 * Happening alongside service thread handling POLLOUT.
		 * The danger is when he is finished, he will disable POLLOUT,
		 * countermanding what we changed here.
		 *
		 * Instead of changing the fds, inform the service thread
		 * what happened, and ask it to leave POLLOUT active on exit
		 */
		wsi->leave_pollout_active = 1;
		/*
		 * by definition service thread is not in poll wait, so no need
		 * to cancel service
		 */

		lwsl_debug("%s: using leave_pollout_active\n", __func__);

		return 0;
	}

	context = wsi->context;
	pt = &context->pt[(int)wsi->tsi];
	assert(wsi->position_in_fds_table >= 0 &&
	       wsi->position_in_fds_table < pt->fds_count);

	pfd = &pt->fds[wsi->position_in_fds_table];
	pa->fd = wsi->desc.sockfd;
	pa->prev_events = pfd->events;
	pa->events = pfd->events = (pfd->events & ~_and) | _or;

	if (wsi->http2_substream)
		return 0;

	if (wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_CHANGE_MODE_POLL_FD,
					   wsi->user_space, (void *)pa, 0)) {
		ret = -1;
		goto bail;
	}

	if (_and & LWS_POLLIN) {
		lws_libev_io(wsi, LWS_EV_STOP | LWS_EV_READ);
		lws_libuv_io(wsi, LWS_EV_STOP | LWS_EV_READ);
		lws_libevent_io(wsi, LWS_EV_STOP | LWS_EV_READ);
	}
	if (_or & LWS_POLLIN) {
		lws_libev_io(wsi, LWS_EV_START | LWS_EV_READ);
		lws_libuv_io(wsi, LWS_EV_START | LWS_EV_READ);
		lws_libevent_io(wsi, LWS_EV_START | LWS_EV_READ);
	}
	if (_and & LWS_POLLOUT) {
		lws_libev_io(wsi, LWS_EV_STOP | LWS_EV_WRITE);
		lws_libuv_io(wsi, LWS_EV_STOP | LWS_EV_WRITE);
		lws_libevent_io(wsi, LWS_EV_STOP | LWS_EV_WRITE);
	}
	if (_or & LWS_POLLOUT) {
		lws_libev_io(wsi, LWS_EV_START | LWS_EV_WRITE);
		lws_libuv_io(wsi, LWS_EV_START | LWS_EV_WRITE);
		lws_libevent_io(wsi, LWS_EV_START | LWS_EV_WRITE);
	}

	/*
	 * if we changed something in this pollfd...
	 *   ... and we're running in a different thread context
	 *     than the service thread...
	 *       ... and the service thread is waiting ...
	 *         then cancel it to force a restart with our changed events
	 */
#if LWS_POSIX
	pa_events = pa->prev_events != pa->events;
#endif

	if (pa_events) {

		if (lws_plat_change_pollfd(context, wsi, pfd)) {
			lwsl_info("%s failed\n", __func__);
			ret = -1;
			goto bail;
		}

		sampled_tid = context->service_tid;
		if (sampled_tid) {
			tid = wsi->vhost->protocols[0].callback(wsi,
				     LWS_CALLBACK_GET_THREAD_ID, NULL, NULL, 0);
			if (tid == -1) {
				ret = -1;
				goto bail;
			}
			if (tid != sampled_tid)
				lws_cancel_service_pt(wsi);
		}
	}
bail:
	return ret;
}

#ifndef LWS_NO_SERVER
static void
lws_accept_modulation(struct lws_context_per_thread *pt, int allow)
{
// multithread listen seems broken
#if 0
	struct lws_vhost *vh = context->vhost_list;
	struct lws_pollargs pa1;

	while (vh) {
		if (allow)
			_lws_change_pollfd(pt->wsi_listening,
					   0, LWS_POLLIN, &pa1);
		else
			_lws_change_pollfd(pt->wsi_listening,
					   LWS_POLLIN, 0, &pa1);
		vh = vh->vhost_next;
	}
#endif
}
#endif

int
insert_wsi_socket_into_fds(struct lws_context *context, struct lws *wsi)
{
	struct lws_pollargs pa = { wsi->desc.sockfd, LWS_POLLIN, 0 };
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	int ret = 0;


	lwsl_debug("%s: %p: tsi=%d, sock=%d, pos-in-fds=%d\n",
		  __func__, wsi, wsi->tsi, wsi->desc.sockfd, pt->fds_count);

	if ((unsigned int)pt->fds_count >= context->fd_limit_per_thread) {
		lwsl_err("Too many fds (%d vs %d)\n", context->max_fds,
				context->fd_limit_per_thread	);
		return 1;
	}

#if !defined(_WIN32) && !defined(LWS_WITH_ESP8266)
	if (wsi->desc.sockfd >= context->max_fds) {
		lwsl_err("Socket fd %d is too high (%d)\n",
			 wsi->desc.sockfd, context->max_fds);
		return 1;
	}
#endif

	assert(wsi);
	assert(wsi->vhost);
	assert(lws_socket_is_valid(wsi->desc.sockfd));

	if (wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_LOCK_POLL,
					   wsi->user_space, (void *) &pa, 1))
		return -1;

	lws_pt_lock(pt);
	pt->count_conns++;
	insert_wsi(context, wsi);
#if defined(LWS_WITH_ESP8266)
	if (wsi->position_in_fds_table == -1)
#endif
		wsi->position_in_fds_table = pt->fds_count;

	pt->fds[wsi->position_in_fds_table].fd = wsi->desc.sockfd;
#if LWS_POSIX
	pt->fds[wsi->position_in_fds_table].events = LWS_POLLIN;
#else
	pt->fds[wsi->position_in_fds_table].events = 0;
#endif
	pa.events = pt->fds[pt->fds_count].events;

	lws_plat_insert_socket_into_fds(context, wsi);

	/* external POLL support via protocol 0 */
	if (wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_ADD_POLL_FD,
					   wsi->user_space, (void *) &pa, 0))
		ret =  -1;
#ifndef LWS_NO_SERVER
	/* if no more room, defeat accepts on this thread */
	if ((unsigned int)pt->fds_count == context->fd_limit_per_thread - 1)
		lws_accept_modulation(pt, 0);
#endif
	lws_pt_unlock(pt);

	if (wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_UNLOCK_POLL,
					   wsi->user_space, (void *)&pa, 1))
		ret = -1;

	return ret;
}

int
remove_wsi_socket_from_fds(struct lws *wsi)
{
	struct lws_context *context = wsi->context;
	struct lws_pollargs pa = { wsi->desc.sockfd, 0, 0 };
#if !defined(LWS_WITH_ESP8266)
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	struct lws *end_wsi;
	int v;
#endif
	int m, ret = 0;

	if (wsi->parent_carries_io) {
		lws_same_vh_protocol_remove(wsi);
		return 0;
	}

#if !defined(_WIN32) && !defined(LWS_WITH_ESP8266)
	if (wsi->desc.sockfd > context->max_fds) {
		lwsl_err("fd %d too high (%d)\n", wsi->desc.sockfd,
			 context->max_fds);
		return 1;
	}
#endif

	if (wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_LOCK_POLL,
					   wsi->user_space, (void *)&pa, 1))
		return -1;

	lws_same_vh_protocol_remove(wsi);

	/* the guy who is to be deleted's slot index in pt->fds */
	m = wsi->position_in_fds_table;
	
#if !defined(LWS_WITH_ESP8266)
	lws_libev_io(wsi, LWS_EV_STOP | LWS_EV_READ | LWS_EV_WRITE |
			  LWS_EV_PREPARE_DELETION);
	lws_libuv_io(wsi, LWS_EV_STOP | LWS_EV_READ | LWS_EV_WRITE |
			  LWS_EV_PREPARE_DELETION);

	lws_pt_lock(pt);

	lwsl_debug("%s: wsi=%p, sock=%d, fds pos=%d, end guy pos=%d, endfd=%d\n",
		  __func__, wsi, wsi->desc.sockfd, wsi->position_in_fds_table,
		  pt->fds_count, pt->fds[pt->fds_count].fd);

	/* have the last guy take up the now vacant slot */
	pt->fds[m] = pt->fds[pt->fds_count - 1];
#endif
	/* this decrements pt->fds_count */
	lws_plat_delete_socket_from_fds(context, wsi, m);
#if !defined(LWS_WITH_ESP8266)
	v = (int) pt->fds[m].fd;
	/* end guy's "position in fds table" is now the deletion guy's old one */
	end_wsi = wsi_from_fd(context, v);
	if (!end_wsi) {
		lwsl_err("no wsi found for sock fd %d at pos %d, pt->fds_count=%d\n",
				(int)pt->fds[m].fd, m, pt->fds_count);
		assert(0);
	} else
		end_wsi->position_in_fds_table = m;

	/* deletion guy's lws_lookup entry needs nuking */
	delete_from_fd(context, wsi->desc.sockfd);
	/* removed wsi has no position any more */
	wsi->position_in_fds_table = -1;

	/* remove also from external POLL support via protocol 0 */
	if (lws_socket_is_valid(wsi->desc.sockfd))
		if (wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_DEL_POLL_FD,
						   wsi->user_space, (void *) &pa, 0))
			ret = -1;
#ifndef LWS_NO_SERVER
	if (!context->being_destroyed)
		/* if this made some room, accept connects on this thread */
		if ((unsigned int)pt->fds_count < context->fd_limit_per_thread - 1)
			lws_accept_modulation(pt, 1);
#endif
	lws_pt_unlock(pt);

	if (wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_UNLOCK_POLL,
					      wsi->user_space, (void *) &pa, 1))
		ret = -1;
#endif
	return ret;
}

int
lws_change_pollfd(struct lws *wsi, int _and, int _or)
{
	struct lws_context_per_thread *pt;
	struct lws_context *context;
	struct lws_pollargs pa;
	int ret = 0;

	if (!wsi || !wsi->protocol || wsi->position_in_fds_table < 0)
		return 1;

	context = lws_get_context(wsi);
	if (!context)
		return 1;

	if (wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_LOCK_POLL,
					      wsi->user_space,  (void *) &pa, 0))
		return -1;

	pt = &context->pt[(int)wsi->tsi];

	lws_pt_lock(pt);
	ret = _lws_change_pollfd(wsi, _and, _or, &pa);
	lws_pt_unlock(pt);
	if (wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_UNLOCK_POLL,
					   wsi->user_space, (void *) &pa, 0))
		ret = -1;

	return ret;
}

LWS_VISIBLE int
lws_callback_on_writable(struct lws *wsi)
{
	struct lws_context_per_thread *pt;
#ifdef LWS_WITH_HTTP2
	struct lws *network_wsi, *wsi2;
	int already;
#endif
	int n;

	if (wsi->state == LWSS_SHUTDOWN)
		return 0;

	if (wsi->socket_is_permanently_unusable)
		return 0;

	pt = &wsi->context->pt[(int)wsi->tsi];

	if (wsi->parent_carries_io) {
#if defined(LWS_WITH_STATS)
		if (!wsi->active_writable_req_us) {
			wsi->active_writable_req_us = time_in_microseconds();
			lws_stats_atomic_bump(wsi->context, pt,
					      LWSSTATS_C_WRITEABLE_CB_EFF_REQ, 1);
		}
#endif
		n = lws_callback_on_writable(wsi->parent);
		if (n < 0)
			return n;

		wsi->parent_pending_cb_on_writable = 1;
		return 1;
	}

	lws_stats_atomic_bump(wsi->context, pt, LWSSTATS_C_WRITEABLE_CB_REQ, 1);
#if defined(LWS_WITH_STATS)
	if (!wsi->active_writable_req_us) {
		wsi->active_writable_req_us = time_in_microseconds();
		lws_stats_atomic_bump(wsi->context, pt, LWSSTATS_C_WRITEABLE_CB_EFF_REQ, 1);
	}
#endif

#ifdef LWS_WITH_HTTP2
	lwsl_info("%s: %p\n", __func__, wsi);

	if (wsi->mode != LWSCM_HTTP2_SERVING)
		goto network_sock;

	if (wsi->u.h2.requested_POLLOUT) {
		lwsl_info("already pending writable\n");
		return 1;
	}

	/* is this for DATA or for control messages? */
	if (wsi->upgraded_to_http2 && !wsi->u.h2.h2n->pps &&
	    !lws_h2_tx_cr_get(wsi)) {
		/*
		 * other side is not able to cope with us sending DATA
		 * anything so no matter if we have POLLOUT on our side if it's
		 * DATA we want to send.
		 *
		 * Delay waiting for our POLLOUT until peer indicates he has
		 * space for more using tx window command in http2 layer
		 */
		lwsl_notice("%s: %p: skint (%d)\n", __func__, wsi, wsi->u.h2.tx_cr);
		wsi->u.h2.skint = 1;
		return 0;
	}

	wsi->u.h2.skint = 0;
	network_wsi = lws_get_network_wsi(wsi);
	already = network_wsi->u.h2.requested_POLLOUT;

	/* mark everybody above him as requesting pollout */

	wsi2 = wsi;
	while (wsi2) {
		wsi2->u.h2.requested_POLLOUT = 1;
		lwsl_info("mark %p pending writable\n", wsi2);
		wsi2 = wsi2->u.h2.parent_wsi;
	}

	/* for network action, act only on the network wsi */

	wsi = network_wsi;
	if (already)
		return 1;
network_sock:
#endif

	if (lws_ext_cb_active(wsi, LWS_EXT_CB_REQUEST_ON_WRITEABLE, NULL, 0))
		return 1;

	if (wsi->position_in_fds_table < 0) {
		lwsl_debug("%s: failed to find socket %d\n", __func__, wsi->desc.sockfd);
		return -1;
	}

	if (lws_change_pollfd(wsi, 0, LWS_POLLOUT))
		return -1;

	return 1;
}

/*
 * stitch protocol choice into the vh protocol linked list
 * We always insert ourselves at the start of the list
 *
 * X <-> B
 * X <-> pAn <-> pB
 *
 * Illegal to attach more than once without detach inbetween
 */
void
lws_same_vh_protocol_insert(struct lws *wsi, int n)
{
	if (wsi->same_vh_protocol_prev || wsi->same_vh_protocol_next) {
		lws_same_vh_protocol_remove(wsi);
		lwsl_notice("Attempted to attach wsi twice to same vh prot\n");
	}

	wsi->same_vh_protocol_prev = &wsi->vhost->same_vh_protocol_list[n];
	/* old first guy is our next */
	wsi->same_vh_protocol_next =  wsi->vhost->same_vh_protocol_list[n];
	/* we become the new first guy */
	wsi->vhost->same_vh_protocol_list[n] = wsi;

	if (wsi->same_vh_protocol_next)
		/* old first guy points back to us now */
		wsi->same_vh_protocol_next->same_vh_protocol_prev =
				&wsi->same_vh_protocol_next;
}

void
lws_same_vh_protocol_remove(struct lws *wsi)
{
	/*
	 * detach ourselves from vh protocol list if we're on one
	 * A -> B -> C
	 * A -> C , or, B -> C, or A -> B
	 *
	 * OK to call on already-detached wsi
	 */
	lwsl_info("%s: removing same prot wsi %p\n", __func__, wsi);

	if (wsi->same_vh_protocol_prev) {
		assert (*(wsi->same_vh_protocol_prev) == wsi);
		lwsl_info("have prev %p, setting him to our next %p\n",
			 wsi->same_vh_protocol_prev,
			 wsi->same_vh_protocol_next);

		/* guy who pointed to us should point to our next */
		*(wsi->same_vh_protocol_prev) = wsi->same_vh_protocol_next;
	}

	/* our next should point back to our prev */
	if (wsi->same_vh_protocol_next) {
		wsi->same_vh_protocol_next->same_vh_protocol_prev =
				wsi->same_vh_protocol_prev;
	}

	wsi->same_vh_protocol_prev = NULL;
	wsi->same_vh_protocol_next = NULL;
}


LWS_VISIBLE int
lws_callback_on_writable_all_protocol_vhost(const struct lws_vhost *vhost,
				      const struct lws_protocols *protocol)
{
	struct lws *wsi;

	if (protocol < vhost->protocols ||
	    protocol >= (vhost->protocols + vhost->count_protocols)) {
		lwsl_err("%s: protocol %p is not from vhost %p (%p - %p)\n",
			__func__, protocol, vhost->protocols, vhost,
			(vhost->protocols + vhost->count_protocols));

		return -1;
	}

	wsi = vhost->same_vh_protocol_list[protocol - vhost->protocols];
	while (wsi) {
		assert(wsi->protocol == protocol);
		assert(*wsi->same_vh_protocol_prev == wsi);
		if (wsi->same_vh_protocol_next)
			assert(wsi->same_vh_protocol_next->same_vh_protocol_prev ==
					&wsi->same_vh_protocol_next);

		lws_callback_on_writable(wsi);
		wsi = wsi->same_vh_protocol_next;
	}

	return 0;
}

LWS_VISIBLE int
lws_callback_on_writable_all_protocol(const struct lws_context *context,
				      const struct lws_protocols *protocol)
{
	struct lws_vhost *vhost;
	int n;

	if (!context)
		return 0;

	vhost = context->vhost_list;

	while (vhost) {
		for (n = 0; n < vhost->count_protocols; n++)
			if (protocol->callback ==
			    vhost->protocols[n].callback &&
			    !strcmp(protocol->name, vhost->protocols[n].name))
				break;
		if (n != vhost->count_protocols)
			lws_callback_on_writable_all_protocol_vhost(
				vhost, &vhost->protocols[n]);

		vhost = vhost->vhost_next;
	}

	return 0;
}
