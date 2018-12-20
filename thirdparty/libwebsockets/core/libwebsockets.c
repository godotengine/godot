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

#ifdef LWS_HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef LWS_WITH_IPV6
#if defined(WIN32) || defined(_WIN32)
#include <wincrypt.h>
#include <iphlpapi.h>
#else
#include <net/if.h>
#endif
#endif

int log_level = LLL_ERR | LLL_WARN | LLL_NOTICE;
static void (*lwsl_emit)(int level, const char *line)
#ifndef LWS_PLAT_OPTEE
	= lwsl_emit_stderr
#endif
	;
#ifndef LWS_PLAT_OPTEE
static const char * const log_level_names[] = {
	"ERR",
	"WARN",
	"NOTICE",
	"INFO",
	"DEBUG",
	"PARSER",
	"HEADER",
	"EXTENSION",
	"CLIENT",
	"LATENCY",
	"USER",
	"?",
	"?"
};
#endif

int lws_open(const char *__file, int __oflag, ...)
{
	va_list ap;
	int n;

	va_start(ap, __oflag);
	if (((__oflag & O_CREAT) == O_CREAT)
#if defined(O_TMPFILE)
		|| ((__oflag & O_TMPFILE) == O_TMPFILE)
#endif
	)
		/* last arg is really a mode_t.  But windows... */
		n = open(__file, __oflag, va_arg(ap, uint32_t));
	else
		n = open(__file, __oflag);
	va_end(ap);

	lws_plat_apply_FD_CLOEXEC(n);

	return n;
}

#if defined (_DEBUG)
void lwsi_set_role(struct lws *wsi, lws_wsi_state_t role)
{
	wsi->wsistate = (wsi->wsistate & (~LWSI_ROLE_MASK)) | role;

	lwsl_debug("lwsi_set_role(%p, 0x%x)\n", wsi, wsi->wsistate);
}

void lwsi_set_state(struct lws *wsi, lws_wsi_state_t lrs)
{
	wsi->wsistate = (wsi->wsistate & (~LRS_MASK)) | lrs;

	lwsl_debug("lwsi_set_state(%p, 0x%x)\n", wsi, wsi->wsistate);
}
#endif

signed char char_to_hex(const char c)
{
	if (c >= '0' && c <= '9')
		return c - '0';

	if (c >= 'a' && c <= 'f')
		return c - 'a' + 10;

	if (c >= 'A' && c <= 'F')
		return c - 'A' + 10;

	return -1;
}

void
__lws_free_wsi(struct lws *wsi)
{
	if (!wsi)
		return;

	/*
	 * Protocol user data may be allocated either internally by lws
	 * or by specified the user. We should only free what we allocated.
	 */
	if (wsi->protocol && wsi->protocol->per_session_data_size &&
	    wsi->user_space && !wsi->user_space_externally_allocated)
		lws_free(wsi->user_space);

	lws_buflist_destroy_all_segments(&wsi->buflist);
	lws_free_set_NULL(wsi->trunc_alloc);
	lws_free_set_NULL(wsi->udp);

	if (wsi->vhost && wsi->vhost->lserv_wsi == wsi)
		wsi->vhost->lserv_wsi = NULL;

	// lws_peer_dump_from_wsi(wsi);

	if (wsi->role_ops->destroy_role)
		wsi->role_ops->destroy_role(wsi);

#if defined(LWS_WITH_PEER_LIMITS)
	lws_peer_track_wsi_close(wsi->context, wsi->peer);
	wsi->peer = NULL;
#endif

	/* since we will destroy the wsi, make absolutely sure now */

#if defined(LWS_WITH_OPENSSL)
	__lws_ssl_remove_wsi_from_buffered_list(wsi);
#endif
	__lws_remove_from_timeout_list(wsi);

	if (wsi->context->event_loop_ops->destroy_wsi)
		wsi->context->event_loop_ops->destroy_wsi(wsi);

	wsi->context->count_wsi_allocated--;
	lwsl_debug("%s: %p, remaining wsi %d\n", __func__, wsi,
			wsi->context->count_wsi_allocated);

	lws_free(wsi);
}

void
lws_dll_add_front(struct lws_dll *d, struct lws_dll *phead)
{
	if (d->prev)
		return;

	/* our next guy is current first guy */
	d->next = phead->next;
	/* if there is a next guy, set his prev ptr to our next ptr */
	if (d->next)
		d->next->prev = d;
	/* our prev ptr is first ptr */
	d->prev = phead;
	/* set the first guy to be us */
	phead->next = d;
}

/* situation is:
 *
 *  HEAD: struct lws_dll * = &entry1
 *
 *  Entry 1: struct lws_dll  .pprev = &HEAD , .next = Entry 2
 *  Entry 2: struct lws_dll  .pprev = &entry1 , .next = &entry2
 *  Entry 3: struct lws_dll  .pprev = &entry2 , .next = NULL
 *
 *  Delete Entry1:
 *
 *   - HEAD = &entry2
 *   - Entry2: .pprev = &HEAD, .next = &entry3
 *   - Entry3: .pprev = &entry2, .next = NULL
 *
 *  Delete Entry2:
 *
 *   - HEAD = &entry1
 *   - Entry1: .pprev = &HEAD, .next = &entry3
 *   - Entry3: .pprev = &entry1, .next = NULL
 *
 *  Delete Entry3:
 *
 *   - HEAD = &entry1
 *   - Entry1: .pprev = &HEAD, .next = &entry2
 *   - Entry2: .pprev = &entry1, .next = NULL
 *
 */

void
lws_dll_remove(struct lws_dll *d)
{
	if (!d->prev) /* ie, not part of the list */
		return;

	/*
	 *  remove us
	 *
	 *  USp <-> us <-> USn  -->  USp <-> USn
	 */

	/* if we have a next guy, set his prev to our prev */
	if (d->next)
		d->next->prev = d->prev;

	/* set our prev guy to our next guy instead of us */
	if (d->prev)
		d->prev->next = d->next;

	/* we're out of the list, we should not point anywhere any more */
	d->prev = NULL;
	d->next = NULL;
}

void
__lws_remove_from_timeout_list(struct lws *wsi)
{
	lws_dll_lws_remove(&wsi->dll_timeout);
}

void
lws_remove_from_timeout_list(struct lws *wsi)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];

	lws_pt_lock(pt, __func__);
	__lws_remove_from_timeout_list(wsi);
	lws_pt_unlock(pt);
}

void
lws_dll_dump(struct lws_dll_lws *head, const char *title)
{
	int n = 0;

	(void)n;
	lwsl_notice("%s: %s (head.next %p)\n", __func__, title, head->next);

	lws_start_foreach_dll_safe(struct lws_dll_lws *, d, d1, head->next) {
		struct lws *wsi = lws_container_of(d, struct lws, dll_hrtimer);

		(void)wsi;

		lwsl_notice("  %d: wsi %p: %llu\n", n++, wsi,
				(unsigned long long)wsi->pending_timer);
	} lws_end_foreach_dll_safe(d, d1);
}

void
__lws_set_timer_usecs(struct lws *wsi, lws_usec_t usecs)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	struct lws_dll_lws *dd = &pt->dll_head_hrtimer;
	struct timeval now;
	struct lws *wsi1;
	int bef = 0;

	lws_dll_lws_remove(&wsi->dll_hrtimer);

	if (usecs == LWS_SET_TIMER_USEC_CANCEL)
		return;

	gettimeofday(&now, NULL);
	wsi->pending_timer = ((now.tv_sec * 1000000ll) + now.tv_usec) + usecs;

	/*
	 * we sort the hrtimer list with the earliest timeout first
	 */

	lws_start_foreach_dll_safe(struct lws_dll_lws *, d, d1,
				   pt->dll_head_hrtimer.next) {
		dd = d;
		wsi1 = lws_container_of(d, struct lws, dll_hrtimer);

		if (wsi1->pending_timer >= wsi->pending_timer) {
			/* d, dprev's next, is >= our time */
			bef = 1;
			break;
		}
	} lws_end_foreach_dll_safe(d, d1);

	if (bef) {
		/*
		 *  we go before dd
		 *  DDp <-> DD <-> DDn --> DDp <-> us <-> DD <-> DDn
		 */
		/* we point forward to dd */
		wsi->dll_hrtimer.next = dd;
		/* we point back to what dd used to point back to */
		wsi->dll_hrtimer.prev = dd->prev;
		/* DDp points forward to us now */
		dd->prev->next = &wsi->dll_hrtimer;
		/* DD points back to us now */
		dd->prev = &wsi->dll_hrtimer;
	} else {
		/*
		 *  we go after dd
		 *  DDp <-> DD <-> DDn --> DDp <-> DD <-> us <-> DDn
		 */
		/* we point forward to what dd used to point forward to */
		wsi->dll_hrtimer.next = dd->next;
		/* we point back to dd */
		wsi->dll_hrtimer.prev = dd;
		/* DDn points back to us */
		if (dd->next)
			dd->next->prev = &wsi->dll_hrtimer;
		/* DD points forward to us */
		dd->next = &wsi->dll_hrtimer;
	}

//	lws_dll_dump(&pt->dll_head_hrtimer, "after set_timer_usec");
}

LWS_VISIBLE void
lws_set_timer_usecs(struct lws *wsi, lws_usec_t usecs)
{
	__lws_set_timer_usecs(wsi, usecs);
}

lws_usec_t
__lws_hrtimer_service(struct lws_context_per_thread *pt)
{
	struct timeval now;
	struct lws *wsi;
	lws_usec_t t;

	gettimeofday(&now, NULL);
	t = (now.tv_sec * 1000000ll) + now.tv_usec;

	lws_start_foreach_dll_safe(struct lws_dll_lws *, d, d1,
				   pt->dll_head_hrtimer.next) {
		wsi = lws_container_of(d, struct lws, dll_hrtimer);

		/*
		 * if we met one in the future, we are done, because the list
		 * is sorted by time in the future.
		 */
		if (wsi->pending_timer > t)
			break;

		lws_set_timer_usecs(wsi, LWS_SET_TIMER_USEC_CANCEL);

		/* it's time for the timer to be serviced */

		if (wsi->protocol &&
		    wsi->protocol->callback(wsi, LWS_CALLBACK_TIMER,
					    wsi->user_space, NULL, 0))
			__lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS,
					     "timer cb errored");
	} lws_end_foreach_dll_safe(d, d1);

	/* return an estimate how many us until next timer hit */

	if (!pt->dll_head_hrtimer.next)
		return LWS_HRTIMER_NOWAIT;

	wsi = lws_container_of(pt->dll_head_hrtimer.next, struct lws, dll_hrtimer);

	gettimeofday(&now, NULL);
	t = (now.tv_sec * 1000000ll) + now.tv_usec;

	if (wsi->pending_timer < t)
		return 0;

	return wsi->pending_timer - t;
}

void
__lws_set_timeout(struct lws *wsi, enum pending_timeout reason, int secs)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	time_t now;

	time(&now);

	lwsl_debug("%s: %p: %d secs\n", __func__, wsi, secs);
	wsi->pending_timeout_limit = secs;
	wsi->pending_timeout_set = now;
	wsi->pending_timeout = reason;

	if (!reason)
		lws_dll_lws_remove(&wsi->dll_timeout);
	else
		lws_dll_lws_add_front(&wsi->dll_timeout, &pt->dll_head_timeout);
}

LWS_VISIBLE void
lws_set_timeout(struct lws *wsi, enum pending_timeout reason, int secs)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];

	if (secs == LWS_TO_KILL_SYNC) {
		lws_remove_from_timeout_list(wsi);
		lwsl_debug("synchronously killing %p\n", wsi);
		lws_close_free_wsi(wsi, LWS_CLOSE_STATUS_NOSTATUS, "to sync kill");
		return;
	}

	if (secs == LWS_TO_KILL_ASYNC)
		secs = 0;

	lws_pt_lock(pt, __func__);
	__lws_set_timeout(wsi, reason, secs);
	lws_pt_unlock(pt);
}

int
lws_timed_callback_remove(struct lws_vhost *vh, struct lws_timed_vh_protocol *p)
{
	lws_start_foreach_llp(struct lws_timed_vh_protocol **, pt,
			      vh->timed_vh_protocol_list) {
		if (*pt == p) {
			*pt = p->next;
			lws_free(p);

			return 0;
		}
	} lws_end_foreach_llp(pt, next);

	return 1;
}

LWS_VISIBLE LWS_EXTERN int
lws_timed_callback_vh_protocol(struct lws_vhost *vh, const struct lws_protocols *prot,
			       int reason, int secs)
{
	struct lws_timed_vh_protocol *p = (struct lws_timed_vh_protocol *)
			lws_malloc(sizeof(*p), "timed_vh");

	if (!p)
		return 1;

	p->protocol = prot;
	p->reason = reason;
	p->time = lws_now_secs() + secs;
	p->next = vh->timed_vh_protocol_list;

	vh->timed_vh_protocol_list = p;

	return 0;
}

static void
lws_remove_child_from_any_parent(struct lws *wsi)
{
	struct lws **pwsi;
	int seen = 0;

	if (!wsi->parent)
		return;

	/* detach ourselves from parent's child list */
	pwsi = &wsi->parent->child_list;
	while (*pwsi) {
		if (*pwsi == wsi) {
			lwsl_info("%s: detach %p from parent %p\n", __func__,
				  wsi, wsi->parent);

			if (wsi->parent->protocol)
				wsi->parent->protocol->callback(wsi,
						LWS_CALLBACK_CHILD_CLOSING,
					       wsi->parent->user_space, wsi, 0);

			*pwsi = wsi->sibling_list;
			seen = 1;
			break;
		}
		pwsi = &(*pwsi)->sibling_list;
	}
	if (!seen)
		lwsl_err("%s: failed to detach from parent\n", __func__);

	wsi->parent = NULL;
}

int
lws_bind_protocol(struct lws *wsi, const struct lws_protocols *p)
{
//	if (wsi->protocol == p)
//		return 0;
	const struct lws_protocols *vp = wsi->vhost->protocols, *vpo;

	if (wsi->protocol && wsi->protocol_bind_balance) {
		wsi->protocol->callback(wsi, LWS_CALLBACK_HTTP_DROP_PROTOCOL,
					wsi->user_space, NULL, 0);
		wsi->protocol_bind_balance = 0;
	}
	if (!wsi->user_space_externally_allocated)
		lws_free_set_NULL(wsi->user_space);

	lws_same_vh_protocol_remove(wsi);

	wsi->protocol = p;
	if (!p)
		return 0;

	if (lws_ensure_user_space(wsi))
		return 1;

	if (p > vp && p < &vp[wsi->vhost->count_protocols])
		lws_same_vh_protocol_insert(wsi, (int)(p - vp));
	else {
		int n = wsi->vhost->count_protocols;
		int hit = 0;

		vpo = vp;

		while (n--) {
			if (p->name && vp->name && !strcmp(p->name, vp->name)) {
				hit = 1;
				lws_same_vh_protocol_insert(wsi, (int)(vp - vpo));
				break;
			}
			vp++;
		}
		if (!hit)
			lwsl_err("%s: %p is not in vhost '%s' protocols list\n",
				 __func__, p, wsi->vhost->name);
	}

	if (wsi->protocol->callback(wsi, LWS_CALLBACK_HTTP_BIND_PROTOCOL,
				    wsi->user_space, NULL, 0))
		return 1;

	wsi->protocol_bind_balance = 1;

	return 0;
}

void
__lws_close_free_wsi(struct lws *wsi, enum lws_close_status reason, const char *caller)
{
	struct lws_context_per_thread *pt;
	struct lws *wsi1, *wsi2;
	struct lws_context *context;
	int n;

	lwsl_info("%s: %p: caller: %s\n", __func__, wsi, caller);

	if (!wsi)
		return;

	lws_access_log(wsi);

	context = wsi->context;
	pt = &context->pt[(int)wsi->tsi];
	lws_stats_atomic_bump(wsi->context, pt, LWSSTATS_C_API_CLOSE, 1);

#if !defined(LWS_NO_CLIENT)

	lws_free_set_NULL(wsi->client_hostname_copy);
	/* we are no longer an active client connection that can piggyback */
	lws_dll_lws_remove(&wsi->dll_active_client_conns);

	/*
	 * if we have wsi in our transaction queue, if we are closing we
	 * must go through and close all those first
	 */
	if (wsi->vhost) {
		if ((int)reason != -1)
			lws_vhost_lock(wsi->vhost);
		lws_start_foreach_dll_safe(struct lws_dll_lws *, d, d1,
					wsi->dll_client_transaction_queue_head.next) {
			struct lws *w = lws_container_of(d, struct lws,
							 dll_client_transaction_queue);

			__lws_close_free_wsi(w, -1, "trans q leader closing");
		} lws_end_foreach_dll_safe(d, d1);

		/*
		 * !!! If we are closing, but we have pending pipelined transaction
		 * results we already sent headers for, that's going to destroy sync
		 * for HTTP/1 and leave H2 stream with no live swsi.
		 *
		 * However this is normal if we are being closed because the transaction
		 * queue leader is closing.
		 */
		lws_dll_lws_remove(&wsi->dll_client_transaction_queue);
		if ((int)reason !=-1)
			lws_vhost_unlock(wsi->vhost);
	}
#endif

	/* if we have children, close them first */
	if (wsi->child_list) {
		wsi2 = wsi->child_list;
		while (wsi2) {
			wsi1 = wsi2->sibling_list;
			wsi2->parent = NULL;
			/* stop it doing shutdown processing */
			wsi2->socket_is_permanently_unusable = 1;
			__lws_close_free_wsi(wsi2, reason, "general child recurse");
			wsi2 = wsi1;
		}
		wsi->child_list = NULL;
	}

	if (wsi->role_ops == &role_ops_raw_file) {
		lws_remove_child_from_any_parent(wsi);
		__remove_wsi_socket_from_fds(wsi);
		wsi->protocol->callback(wsi, wsi->role_ops->close_cb[0],
					wsi->user_space, NULL, 0);
		goto async_close;
	}

	wsi->wsistate_pre_close = wsi->wsistate;

#ifdef LWS_WITH_CGI
	if (wsi->role_ops == &role_ops_cgi) {
		/* we are not a network connection, but a handler for CGI io */
		if (wsi->parent && wsi->parent->http.cgi) {

			if (wsi->cgi_channel == LWS_STDOUT)
				lws_cgi_remove_and_kill(wsi->parent);

			/* end the binding between us and master */
			wsi->parent->http.cgi->stdwsi[(int)wsi->cgi_channel] = NULL;
		}
		wsi->socket_is_permanently_unusable = 1;

		goto just_kill_connection;
	}

	if (wsi->http.cgi)
		lws_cgi_remove_and_kill(wsi);
#endif

#if !defined(LWS_NO_CLIENT)
	lws_client_stash_destroy(wsi);
#endif

	if (wsi->role_ops == &role_ops_raw_skt) {
		wsi->socket_is_permanently_unusable = 1;
		goto just_kill_connection;
	}
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	if (lwsi_role_http(wsi) && lwsi_role_server(wsi) &&
	    wsi->http.fop_fd != NULL)
		lws_vfs_file_close(&wsi->http.fop_fd);
#endif

	if (lwsi_state(wsi) == LRS_DEAD_SOCKET)
		return;

	if (wsi->socket_is_permanently_unusable ||
	    reason == LWS_CLOSE_STATUS_NOSTATUS_CONTEXT_DESTROY ||
	    lwsi_state(wsi) == LRS_SHUTDOWN)
		goto just_kill_connection;

	switch (lwsi_state_PRE_CLOSE(wsi)) {
	case LRS_DEAD_SOCKET:
		return;

	/* we tried the polite way... */
	case LRS_WAITING_TO_SEND_CLOSE:
	case LRS_AWAITING_CLOSE_ACK:
	case LRS_RETURNED_CLOSE:
		goto just_kill_connection;

	case LRS_FLUSHING_BEFORE_CLOSE:
		if (wsi->trunc_len) {
			lws_callback_on_writable(wsi);
			return;
		}
		lwsl_info("%p: end LRS_FLUSHING_BEFORE_CLOSE\n", wsi);
		goto just_kill_connection;
	default:
		if (wsi->trunc_len) {
			lwsl_info("%p: LRS_FLUSHING_BEFORE_CLOSE\n", wsi);
			lwsi_set_state(wsi, LRS_FLUSHING_BEFORE_CLOSE);
			__lws_set_timeout(wsi,
				PENDING_FLUSH_STORED_SEND_BEFORE_CLOSE, 5);
			return;
		}
		break;
	}

	if (lwsi_state(wsi) == LRS_WAITING_CONNECT ||
	    lwsi_state(wsi) == LRS_H1C_ISSUE_HANDSHAKE)
		goto just_kill_connection;

	if (!wsi->told_user_closed && lwsi_role_http(wsi) &&
	    lwsi_role_server(wsi)) {
		if (wsi->user_space && wsi->protocol &&
		    wsi->protocol_bind_balance) {
			wsi->protocol->callback(wsi,
						LWS_CALLBACK_HTTP_DROP_PROTOCOL,
					       wsi->user_space, NULL, 0);
			wsi->protocol_bind_balance = 0;
		}
	}

	/*
	 * signal we are closing, lws_write will
	 * add any necessary version-specific stuff.  If the write fails,
	 * no worries we are closing anyway.  If we didn't initiate this
	 * close, then our state has been changed to
	 * LRS_RETURNED_CLOSE and we will skip this.
	 *
	 * Likewise if it's a second call to close this connection after we
	 * sent the close indication to the peer already, we are in state
	 * LRS_AWAITING_CLOSE_ACK and will skip doing this a second time.
	 */

	if (wsi->role_ops->close_via_role_protocol &&
	    wsi->role_ops->close_via_role_protocol(wsi, reason))
		return;

just_kill_connection:

	if (wsi->role_ops->close_kill_connection)
		wsi->role_ops->close_kill_connection(wsi, reason);

	lws_remove_child_from_any_parent(wsi);
	n = 0;

	if (!wsi->told_user_closed && wsi->user_space &&
	    wsi->protocol_bind_balance) {
		lwsl_debug("%s: %p: DROP_PROTOCOL %s\n", __func__, wsi,
		       wsi->protocol->name);
		wsi->protocol->callback(wsi, LWS_CALLBACK_HTTP_DROP_PROTOCOL,
				        wsi->user_space, NULL, 0);
		wsi->protocol_bind_balance = 0;
	}

	if ((lwsi_state(wsi) == LRS_WAITING_SERVER_REPLY ||
	     lwsi_state(wsi) == LRS_WAITING_CONNECT) && !wsi->already_did_cce)
		wsi->protocol->callback(wsi,
				        LWS_CALLBACK_CLIENT_CONNECTION_ERROR,
						wsi->user_space, NULL, 0);

	/*
	 * Testing with ab shows that we have to stage the socket close when
	 * the system is under stress... shutdown any further TX, change the
	 * state to one that won't emit anything more, and wait with a timeout
	 * for the POLLIN to show a zero-size rx before coming back and doing
	 * the actual close.
	 */
	if (wsi->role_ops != &role_ops_raw_skt && !lwsi_role_client(wsi) &&
	    lwsi_state(wsi) != LRS_SHUTDOWN &&
	    lwsi_state(wsi) != LRS_UNCONNECTED &&
	    reason != LWS_CLOSE_STATUS_NOSTATUS_CONTEXT_DESTROY &&
	    !wsi->socket_is_permanently_unusable) {

#if defined(LWS_WITH_TLS)
	if (lws_is_ssl(wsi) && wsi->tls.ssl) {
		n = 0;
		switch (__lws_tls_shutdown(wsi)) {
		case LWS_SSL_CAPABLE_DONE:
		case LWS_SSL_CAPABLE_ERROR:
		case LWS_SSL_CAPABLE_MORE_SERVICE_READ:
		case LWS_SSL_CAPABLE_MORE_SERVICE_WRITE:
		case LWS_SSL_CAPABLE_MORE_SERVICE:
			break;
		}
	} else
#endif
		{
			lwsl_info("%s: shutdown conn: %p (sock %d, state 0x%x)\n",
				  __func__, wsi, (int)(long)wsi->desc.sockfd,
				  lwsi_state(wsi));
			if (!wsi->socket_is_permanently_unusable &&
			    lws_socket_is_valid(wsi->desc.sockfd)) {
				wsi->socket_is_permanently_unusable = 1;
				n = shutdown(wsi->desc.sockfd, SHUT_WR);
			}
		}
		if (n)
			lwsl_debug("closing: shutdown (state 0x%x) ret %d\n",
				   lwsi_state(wsi), LWS_ERRNO);

		/*
		 * This causes problems on WINCE / ESP32 with disconnection
		 * when the events are half closing connection
		 */
#if !defined(_WIN32_WCE) && !defined(LWS_WITH_ESP32)
		/* libuv: no event available to guarantee completion */
		if (!wsi->socket_is_permanently_unusable &&
		    lws_socket_is_valid(wsi->desc.sockfd) &&
		    lwsi_state(wsi) != LRS_SHUTDOWN &&
		    context->event_loop_ops->periodic_events_available) {
			__lws_change_pollfd(wsi, LWS_POLLOUT, LWS_POLLIN);
			lwsi_set_state(wsi, LRS_SHUTDOWN);
			__lws_set_timeout(wsi, PENDING_TIMEOUT_SHUTDOWN_FLUSH,
					  context->timeout_secs);

			return;
		}
#endif
	}

	lwsl_debug("%s: real just_kill_connection: %p (sockfd %d)\n", __func__,
		   wsi, wsi->desc.sockfd);
	
#ifdef LWS_WITH_HTTP_PROXY
	if (wsi->http.rw) {
		lws_rewrite_destroy(wsi->http.rw);
		wsi->http.rw = NULL;
	}
#endif
	/*
	 * we won't be servicing or receiving anything further from this guy
	 * delete socket from the internal poll list if still present
	 */
	__lws_ssl_remove_wsi_from_buffered_list(wsi);
	__lws_remove_from_timeout_list(wsi);
	lws_dll_lws_remove(&wsi->dll_hrtimer);

	/* don't repeat event loop stuff */
	if (wsi->told_event_loop_closed)
		return;

	/* checking return redundant since we anyway close */
	if (wsi->desc.sockfd != LWS_SOCK_INVALID)
		__remove_wsi_socket_from_fds(wsi);
	else
		lws_same_vh_protocol_remove(wsi);

	lwsi_set_state(wsi, LRS_DEAD_SOCKET);
	lws_buflist_destroy_all_segments(&wsi->buflist);
	lws_dll_lws_remove(&wsi->dll_buflist);

	if (wsi->role_ops->close_role)
	    wsi->role_ops->close_role(pt, wsi);

	/* tell the user it's all over for this guy */

	if (lwsi_state_est_PRE_CLOSE(wsi) && !wsi->told_user_closed &&
	    wsi->role_ops->close_cb[lwsi_role_server(wsi)]) {
		const struct lws_protocols *pro = wsi->protocol;

		if (!wsi->protocol)
			pro = &wsi->vhost->protocols[0];

		if (!wsi->upgraded_to_http2 || !lwsi_role_client(wsi))
			/*
			 * The network wsi for a client h2 connection shouldn't
			 * call back for its role: the child stream connections
			 * own the role.  Otherwise h2 will call back closed
			 * one too many times as the children do it and then
			 * the closing network stream.
			 */
			pro->callback(wsi,
			      wsi->role_ops->close_cb[lwsi_role_server(wsi)],
			      wsi->user_space, NULL, 0);
		wsi->told_user_closed = 1;
	}

async_close:
	wsi->socket_is_permanently_unusable = 1;

	if (wsi->context->event_loop_ops->wsi_logical_close)
		if (wsi->context->event_loop_ops->wsi_logical_close(wsi))
			return;

	__lws_close_free_wsi_final(wsi);
}

void
__lws_close_free_wsi_final(struct lws *wsi)
{
	int n;

	if (lws_socket_is_valid(wsi->desc.sockfd) && !lws_ssl_close(wsi)) {
		n = compatible_close(wsi->desc.sockfd);
		if (n)
			lwsl_debug("closing: close ret %d\n", LWS_ERRNO);

		wsi->desc.sockfd = LWS_SOCK_INVALID;
	}

	/* outermost destroy notification for wsi (user_space still intact) */
	if (wsi->vhost)
		wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_WSI_DESTROY,
						  wsi->user_space, NULL, 0);

#ifdef LWS_WITH_CGI
	if (wsi->http.cgi) {

		for (n = 0; n < 3; n++) {
			if (wsi->http.cgi->pipe_fds[n][!!(n == 0)] == 0)
				lwsl_err("ZERO FD IN CGI CLOSE");

			if (wsi->http.cgi->pipe_fds[n][!!(n == 0)] >= 0)
				close(wsi->http.cgi->pipe_fds[n][!!(n == 0)]);
		}

		lws_free(wsi->http.cgi);
	}
#endif

	__lws_free_wsi(wsi);
}


void
lws_close_free_wsi(struct lws *wsi, enum lws_close_status reason, const char *caller)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];

	lws_pt_lock(pt, __func__);
	__lws_close_free_wsi(wsi, reason, caller);
	lws_pt_unlock(pt);
}

/* lws_buflist */

int
lws_buflist_append_segment(struct lws_buflist **head, const uint8_t *buf,
			   size_t len)
{
	struct lws_buflist *nbuf;
	int first = !*head;
	void *p = *head;
	int sanity = 1024;

	assert(buf);
	assert(len);

	/* append at the tail */
	while (*head) {
		if (!--sanity || head == &((*head)->next)) {
			lwsl_err("%s: corrupt list points to self\n", __func__);
			return -1;
		}
		head = &((*head)->next);
	}

	lwsl_info("%s: len %u first %d %p\n", __func__, (uint32_t)len, first, p);

	nbuf = (struct lws_buflist *)
			lws_malloc(sizeof(**head) + len, __func__);
	if (!nbuf) {
		lwsl_err("%s: OOM\n", __func__);
		return -1;
	}

	nbuf->len = len;
	nbuf->pos = 0;
	nbuf->next = NULL;

	p = (void *)nbuf->buf;
	memcpy(p, buf, len);

	*head = nbuf;

	return first; /* returns 1 if first segment just created */
}

static int
lws_buflist_destroy_segment(struct lws_buflist **head)
{
	struct lws_buflist *old = *head;

	assert(*head);
	*head = (*head)->next;
	old->next = NULL;
	lws_free(old);

	return !*head; /* returns 1 if last segment just destroyed */
}

void
lws_buflist_destroy_all_segments(struct lws_buflist **head)
{
	struct lws_buflist *p = *head, *p1;

	while (p) {
		p1 = p->next;
		p->next = NULL;
		lws_free(p);
		p = p1;
	}

	*head = NULL;
}

size_t
lws_buflist_next_segment_len(struct lws_buflist **head, uint8_t **buf)
{
	if (!*head) {
		if (buf)
			*buf = NULL;

		return 0;
	}

	if (!(*head)->len && (*head)->next)
		lws_buflist_destroy_segment(head);

	if (!*head) {
		if (buf)
			*buf = NULL;

		return 0;
	}

	assert((*head)->pos < (*head)->len);

	if (buf)
		*buf = (*head)->buf + (*head)->pos;

	return (*head)->len - (*head)->pos;
}

int
lws_buflist_use_segment(struct lws_buflist **head, size_t len)
{
	assert(*head);
	assert(len);
	assert((*head)->pos + len <= (*head)->len);

	(*head)->pos += len;
	if ((*head)->pos == (*head)->len)
		lws_buflist_destroy_segment(head);

	if (!*head)
		return 0;

	return (int)((*head)->len - (*head)->pos);
}

void
lws_buflist_describe(struct lws_buflist **head, void *id)
{
	struct lws_buflist *old;
	int n = 0;

	if (*head == NULL)
		lwsl_notice("%p: buflist empty\n", id);

	while (*head) {
		lwsl_notice("%p: %d: %llu / %llu (%llu left)\n", id, n,
			    (unsigned long long)(*head)->pos,
			    (unsigned long long)(*head)->len,
			    (unsigned long long)(*head)->len - (*head)->pos);
		old = *head;
		head = &((*head)->next);
		if (*head == old) {
			lwsl_err("%s: next points to self\n", __func__);
			break;
		}
		n++;
	}
}

/* ... */

LWS_VISIBLE LWS_EXTERN const char *
lws_get_urlarg_by_name(struct lws *wsi, const char *name, char *buf, int len)
{
	int n = 0, sl = (int)strlen(name);

	while (lws_hdr_copy_fragment(wsi, buf, len,
			  WSI_TOKEN_HTTP_URI_ARGS, n) >= 0) {

		if (!strncmp(buf, name, sl))
			return buf + sl;

		n++;
	}

	return NULL;
}

#if !defined(LWS_WITH_ESP32)
LWS_VISIBLE int
interface_to_sa(struct lws_vhost *vh, const char *ifname,
		struct sockaddr_in *addr, size_t addrlen)
{
	int ipv6 = 0;
#ifdef LWS_WITH_IPV6
	ipv6 = LWS_IPV6_ENABLED(vh);
#endif
	(void)vh;

	return lws_interface_to_sa(ipv6, ifname, addr, addrlen);
}
#endif

#ifndef LWS_PLAT_OPTEE
static int
lws_get_addresses(struct lws_vhost *vh, void *ads, char *name,
		  int name_len, char *rip, int rip_len)
{
	struct addrinfo ai, *res;
	struct sockaddr_in addr4;

	rip[0] = '\0';
	name[0] = '\0';
	addr4.sin_family = AF_UNSPEC;

#ifdef LWS_WITH_IPV6
	if (LWS_IPV6_ENABLED(vh)) {
		if (!lws_plat_inet_ntop(AF_INET6,
					&((struct sockaddr_in6 *)ads)->sin6_addr,
					rip, rip_len)) {
			lwsl_err("inet_ntop: %s", strerror(LWS_ERRNO));
			return -1;
		}

		// Strip off the IPv4 to IPv6 header if one exists
		if (strncmp(rip, "::ffff:", 7) == 0)
			memmove(rip, rip + 7, strlen(rip) - 6);

		getnameinfo((struct sockaddr *)ads, sizeof(struct sockaddr_in6),
			    name, name_len, NULL, 0, 0);

		return 0;
	} else
#endif
	{
		struct addrinfo *result;

		memset(&ai, 0, sizeof ai);
		ai.ai_family = PF_UNSPEC;
		ai.ai_socktype = SOCK_STREAM;
#if !defined(LWS_WITH_ESP32)
		if (getnameinfo((struct sockaddr *)ads,
				sizeof(struct sockaddr_in),
				name, name_len, NULL, 0, 0))
			return -1;
#endif

		if (getaddrinfo(name, NULL, &ai, &result))
			return -1;

		res = result;
		while (addr4.sin_family == AF_UNSPEC && res) {
			switch (res->ai_family) {
			case AF_INET:
				addr4.sin_addr =
				 ((struct sockaddr_in *)res->ai_addr)->sin_addr;
				addr4.sin_family = AF_INET;
				break;
			}

			res = res->ai_next;
		}
		freeaddrinfo(result);
	}

	if (addr4.sin_family == AF_UNSPEC)
		return -1;

	if (lws_plat_inet_ntop(AF_INET, &addr4.sin_addr, rip, rip_len) == NULL)
		return -1;

	return 0;
}


LWS_VISIBLE const char *
lws_get_peer_simple(struct lws *wsi, char *name, int namelen)
{
	socklen_t len, olen;
#ifdef LWS_WITH_IPV6
	struct sockaddr_in6 sin6;
#endif
	struct sockaddr_in sin4;
	int af = AF_INET;
	void *p, *q;

	wsi = lws_get_network_wsi(wsi);

	if (wsi->parent_carries_io)
		wsi = wsi->parent;

#ifdef LWS_WITH_IPV6
	if (LWS_IPV6_ENABLED(wsi->vhost)) {
		len = sizeof(sin6);
		p = &sin6;
		af = AF_INET6;
		q = &sin6.sin6_addr;
	} else
#endif
	{
		len = sizeof(sin4);
		p = &sin4;
		q = &sin4.sin_addr;
	}

	olen = len;
	if (getpeername(wsi->desc.sockfd, p, &len) < 0 || len > olen) {
		lwsl_warn("getpeername: %s\n", strerror(LWS_ERRNO));
		return NULL;
	}

	return lws_plat_inet_ntop(af, q, name, namelen);
}
#endif

LWS_VISIBLE void
lws_get_peer_addresses(struct lws *wsi, lws_sockfd_type fd, char *name,
		       int name_len, char *rip, int rip_len)
{
#ifndef LWS_PLAT_OPTEE
	socklen_t len;
#ifdef LWS_WITH_IPV6
	struct sockaddr_in6 sin6;
#endif
	struct sockaddr_in sin4;
	struct lws_context *context = wsi->context;
	int ret = -1;
	void *p;

	rip[0] = '\0';
	name[0] = '\0';

	lws_latency_pre(context, wsi);

#ifdef LWS_WITH_IPV6
	if (LWS_IPV6_ENABLED(wsi->vhost)) {
		len = sizeof(sin6);
		p = &sin6;
	} else
#endif
	{
		len = sizeof(sin4);
		p = &sin4;
	}

	if (getpeername(fd, p, &len) < 0) {
		lwsl_warn("getpeername: %s\n", strerror(LWS_ERRNO));
		goto bail;
	}

	ret = lws_get_addresses(wsi->vhost, p, name, name_len, rip, rip_len);

bail:
	lws_latency(context, wsi, "lws_get_peer_addresses", ret, 1);
#endif
	(void)wsi;
	(void)fd;
	(void)name;
	(void)name_len;
	(void)rip;
	(void)rip_len;

}

LWS_EXTERN void *
lws_vhost_user(struct lws_vhost *vhost)
{
	return vhost->user;
}

LWS_EXTERN void *
lws_context_user(struct lws_context *context)
{
	return context->user_space;
}

LWS_VISIBLE struct lws_vhost *
lws_vhost_get(struct lws *wsi)
{
	return wsi->vhost;
}

LWS_VISIBLE struct lws_vhost *
lws_get_vhost(struct lws *wsi)
{
	return wsi->vhost;
}

LWS_VISIBLE const struct lws_protocols *
lws_protocol_get(struct lws *wsi)
{
	return wsi->protocol;
}

LWS_VISIBLE const struct lws_udp *
lws_get_udp(const struct lws *wsi)
{
	return wsi->udp;
}

LWS_VISIBLE struct lws *
lws_get_network_wsi(struct lws *wsi)
{
	if (!wsi)
		return NULL;

#if defined(LWS_WITH_HTTP2)
	if (!wsi->http2_substream && !wsi->client_h2_substream)
		return wsi;

	while (wsi->h2.parent_wsi)
		wsi = wsi->h2.parent_wsi;
#endif

	return wsi;
}

LWS_VISIBLE LWS_EXTERN const struct lws_protocols *
lws_vhost_name_to_protocol(struct lws_vhost *vh, const char *name)
{
	int n;

	for (n = 0; n < vh->count_protocols; n++)
		if (!strcmp(name, vh->protocols[n].name))
			return &vh->protocols[n];

	return NULL;
}

LWS_VISIBLE int
lws_callback_all_protocol(struct lws_context *context,
			  const struct lws_protocols *protocol, int reason)
{
	struct lws_context_per_thread *pt = &context->pt[0];
	unsigned int n, m = context->count_threads;
	struct lws *wsi;

	while (m--) {
		for (n = 0; n < pt->fds_count; n++) {
			wsi = wsi_from_fd(context, pt->fds[n].fd);
			if (!wsi)
				continue;
			if (wsi->protocol == protocol)
				protocol->callback(wsi, reason, wsi->user_space,
						   NULL, 0);
		}
		pt++;
	}

	return 0;
}

LWS_VISIBLE int
lws_callback_all_protocol_vhost_args(struct lws_vhost *vh,
			  const struct lws_protocols *protocol, int reason,
			  void *argp, size_t len)
{
	struct lws_context *context = vh->context;
	struct lws_context_per_thread *pt = &context->pt[0];
	unsigned int n, m = context->count_threads;
	struct lws *wsi;

	while (m--) {
		for (n = 0; n < pt->fds_count; n++) {
			wsi = wsi_from_fd(context, pt->fds[n].fd);
			if (!wsi)
				continue;
			if (wsi->vhost == vh && (wsi->protocol == protocol ||
						 !protocol))
				wsi->protocol->callback(wsi, reason,
						wsi->user_space, argp, len);
		}
		pt++;
	}

	return 0;
}

LWS_VISIBLE int
lws_callback_all_protocol_vhost(struct lws_vhost *vh,
			  const struct lws_protocols *protocol, int reason)
{
	return lws_callback_all_protocol_vhost_args(vh, protocol, reason, NULL, 0);
}

LWS_VISIBLE LWS_EXTERN int
lws_callback_vhost_protocols(struct lws *wsi, int reason, void *in, int len)
{
	int n;

	for (n = 0; n < wsi->vhost->count_protocols; n++)
		if (wsi->vhost->protocols[n].callback(wsi, reason, NULL, in, len))
			return 1;

	return 0;
}

LWS_VISIBLE LWS_EXTERN int
lws_callback_vhost_protocols_vhost(struct lws_vhost *vh, int reason, void *in,
				   size_t len)
{
	int n;
	struct lws *wsi = lws_zalloc(sizeof(*wsi), "fake wsi");

	wsi->context = vh->context;
	wsi->vhost = vh;

	for (n = 0; n < wsi->vhost->count_protocols; n++) {
		wsi->protocol = &vh->protocols[n];
		if (wsi->protocol->callback(wsi, reason, NULL, in, len)) {
			lws_free(wsi);
			return 1;
		}
	}

	lws_free(wsi);

	return 0;
}

LWS_VISIBLE LWS_EXTERN void
lws_set_fops(struct lws_context *context, const struct lws_plat_file_ops *fops)
{
	context->fops = fops;
}

LWS_VISIBLE LWS_EXTERN lws_filepos_t
lws_vfs_tell(lws_fop_fd_t fop_fd)
{
	return fop_fd->pos;
}

LWS_VISIBLE LWS_EXTERN lws_filepos_t
lws_vfs_get_length(lws_fop_fd_t fop_fd)
{
	return fop_fd->len;
}

LWS_VISIBLE LWS_EXTERN uint32_t
lws_vfs_get_mod_time(lws_fop_fd_t fop_fd)
{
	return fop_fd->mod_time;
}

LWS_VISIBLE lws_fileofs_t
lws_vfs_file_seek_set(lws_fop_fd_t fop_fd, lws_fileofs_t offset)
{
	lws_fileofs_t ofs;

	ofs = fop_fd->fops->LWS_FOP_SEEK_CUR(fop_fd, offset - fop_fd->pos);

	return ofs;
}


LWS_VISIBLE lws_fileofs_t
lws_vfs_file_seek_end(lws_fop_fd_t fop_fd, lws_fileofs_t offset)
{
	return fop_fd->fops->LWS_FOP_SEEK_CUR(fop_fd, fop_fd->len +
					      fop_fd->pos + offset);
}


const struct lws_plat_file_ops *
lws_vfs_select_fops(const struct lws_plat_file_ops *fops, const char *vfs_path,
		    const char **vpath)
{
	const struct lws_plat_file_ops *pf;
	const char *p = vfs_path;
	int n;

	*vpath = NULL;

	/* no non-platform fops, just use that */

	if (!fops->next)
		return fops;

	/*
	 *  scan the vfs path looking for indications we are to be
	 * handled by a specific fops
	 */

	while (p && *p) {
		if (*p != '/') {
			p++;
			continue;
		}
		/* the first one is always platform fops, so skip */
		pf = fops->next;
		while (pf) {
			n = 0;
			while (n < (int)LWS_ARRAY_SIZE(pf->fi) && pf->fi[n].sig) {
				if (p >= vfs_path + pf->fi[n].len)
					if (!strncmp(p - (pf->fi[n].len - 1),
						    pf->fi[n].sig,
						    pf->fi[n].len - 1)) {
						*vpath = p + 1;
						return pf;
					}

				n++;
			}
			pf = pf->next;
		}
		p++;
	}

	return fops;
}

LWS_VISIBLE LWS_EXTERN lws_fop_fd_t LWS_WARN_UNUSED_RESULT
lws_vfs_file_open(const struct lws_plat_file_ops *fops, const char *vfs_path,
		  lws_fop_flags_t *flags)
{
	const char *vpath = "";
	const struct lws_plat_file_ops *selected;

	selected = lws_vfs_select_fops(fops, vfs_path, &vpath);

	return selected->LWS_FOP_OPEN(fops, vfs_path, vpath, flags);
}


/**
 * lws_now_secs() - seconds since 1970-1-1
 *
 */
LWS_VISIBLE LWS_EXTERN unsigned long
lws_now_secs(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);

	return tv.tv_sec;
}

LWS_VISIBLE LWS_EXTERN int
lws_compare_time_t(struct lws_context *context, time_t t1, time_t t2)
{
	if (t1 < context->time_discontiguity)
		t1 += context->time_fixup;

	if (t2 < context->time_discontiguity)
		t2 += context->time_fixup;

	return (int)(t1 - t2);
}

LWS_VISIBLE lws_sockfd_type
lws_get_socket_fd(struct lws *wsi)
{
	if (!wsi)
		return -1;
	return wsi->desc.sockfd;
}

#ifdef LWS_LATENCY
void
lws_latency(struct lws_context *context, struct lws *wsi, const char *action,
	    int ret, int completed)
{
	unsigned long long u;
	char buf[256];

	u = time_in_microseconds();

	if (!action) {
		wsi->latency_start = u;
		if (!wsi->action_start)
			wsi->action_start = u;
		return;
	}
	if (completed) {
		if (wsi->action_start == wsi->latency_start)
			sprintf(buf,
			  "Completion first try lat %lluus: %p: ret %d: %s\n",
					u - wsi->latency_start,
						      (void *)wsi, ret, action);
		else
			sprintf(buf,
			  "Completion %lluus: lat %lluus: %p: ret %d: %s\n",
				u - wsi->action_start,
					u - wsi->latency_start,
						      (void *)wsi, ret, action);
		wsi->action_start = 0;
	} else
		sprintf(buf, "lat %lluus: %p: ret %d: %s\n",
			      u - wsi->latency_start, (void *)wsi, ret, action);

	if (u - wsi->latency_start > context->worst_latency) {
		context->worst_latency = u - wsi->latency_start;
		strcpy(context->worst_latency_info, buf);
	}
	lwsl_latency("%s", buf);
}
#endif

LWS_VISIBLE int
lws_rx_flow_control(struct lws *wsi, int _enable)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	int en = _enable;

	// h2 ignores rx flow control atm
	if (lwsi_role_h2(wsi) || wsi->http2_substream ||
	    lwsi_role_h2_ENCAPSULATION(wsi))
		return 0; // !!!

	lwsl_info("%s: %p 0x%x\n", __func__, wsi, _enable);

	if (!(_enable & LWS_RXFLOW_REASON_APPLIES)) {
		/*
		 * convert user bool style to bitmap style... in user simple
		 * bool style _enable = 0 = flow control it, = 1 = allow rx
		 */
		en = LWS_RXFLOW_REASON_APPLIES | LWS_RXFLOW_REASON_USER_BOOL;
		if (_enable & 1)
			en |= LWS_RXFLOW_REASON_APPLIES_ENABLE_BIT;
	}

	lws_pt_lock(pt, __func__);

	/* any bit set in rxflow_bitmap DISABLEs rxflow control */
	if (en & LWS_RXFLOW_REASON_APPLIES_ENABLE_BIT)
		wsi->rxflow_bitmap &= ~(en & 0xff);
	else
		wsi->rxflow_bitmap |= en & 0xff;

	if ((LWS_RXFLOW_PENDING_CHANGE | (!wsi->rxflow_bitmap)) ==
	    wsi->rxflow_change_to)
		goto skip;

	wsi->rxflow_change_to = LWS_RXFLOW_PENDING_CHANGE | !wsi->rxflow_bitmap;

	lwsl_info("%s: %p: bitmap 0x%x: en 0x%x, ch 0x%x\n", __func__, wsi,
		  wsi->rxflow_bitmap, en, wsi->rxflow_change_to);

	if (_enable & LWS_RXFLOW_REASON_FLAG_PROCESS_NOW ||
	    !wsi->rxflow_will_be_applied) {
		en = __lws_rx_flow_control(wsi);
		lws_pt_unlock(pt);

		return en;
	}

skip:
	lws_pt_unlock(pt);

	return 0;
}

LWS_VISIBLE void
lws_rx_flow_allow_all_protocol(const struct lws_context *context,
			       const struct lws_protocols *protocol)
{
	const struct lws_context_per_thread *pt = &context->pt[0];
	struct lws *wsi;
	unsigned int n, m = context->count_threads;

	while (m--) {
		for (n = 0; n < pt->fds_count; n++) {
			wsi = wsi_from_fd(context, pt->fds[n].fd);
			if (!wsi)
				continue;
			if (wsi->protocol == protocol)
				lws_rx_flow_control(wsi, LWS_RXFLOW_ALLOW);
		}
		pt++;
	}
}

int
lws_broadcast(struct lws_context *context, int reason, void *in, size_t len)
{
	struct lws_vhost *v = context->vhost_list;
	struct lws wsi;
	int n, ret = 0;

	memset(&wsi, 0, sizeof(wsi));
	wsi.context = context;

	while (v) {
		const struct lws_protocols *p = v->protocols;
		wsi.vhost = v;

		for (n = 0; n < v->count_protocols; n++) {
			wsi.protocol = p;
			if (p->callback &&
			    p->callback(&wsi, reason, NULL, in, len))
				ret |= 1;
			p++;
		}
		v = v->vhost_next;
	}

	return ret;
}

LWS_VISIBLE extern const char *
lws_canonical_hostname(struct lws_context *context)
{
	return (const char *)context->canonical_hostname;
}

LWS_VISIBLE LWS_EXTERN const char *
lws_get_vhost_name(struct lws_vhost *vhost)
{
	return vhost->name;
}

LWS_VISIBLE LWS_EXTERN int
lws_get_vhost_port(struct lws_vhost *vhost)
{
	return vhost->listen_port;
}

LWS_VISIBLE LWS_EXTERN void *
lws_get_vhost_user(struct lws_vhost *vhost)
{
	return vhost->user;
}

LWS_VISIBLE LWS_EXTERN const char *
lws_get_vhost_iface(struct lws_vhost *vhost)
{
	return vhost->iface;
}

int user_callback_handle_rxflow(lws_callback_function callback_function,
				struct lws *wsi,
				enum lws_callback_reasons reason, void *user,
				void *in, size_t len)
{
	int n;

	wsi->rxflow_will_be_applied = 1;
	n = callback_function(wsi, reason, user, in, len);
	wsi->rxflow_will_be_applied = 0;
	if (!n)
		n = __lws_rx_flow_control(wsi);

	return n;
}

#if !defined(LWS_WITHOUT_CLIENT)
LWS_VISIBLE int
lws_set_proxy(struct lws_vhost *vhost, const char *proxy)
{
	char *p;
	char authstring[96];

	if (!proxy)
		return -1;

	/* we have to deal with a possible redundant leading http:// */
	if (!strncmp(proxy, "http://", 7))
		proxy += 7;

	p = strrchr(proxy, '@');
	if (p) { /* auth is around */

		if ((unsigned int)(p - proxy) > sizeof(authstring) - 1)
			goto auth_too_long;

		lws_strncpy(authstring, proxy, p - proxy + 1);
		// null termination not needed on input
		if (lws_b64_encode_string(authstring, lws_ptr_diff(p, proxy),
				vhost->proxy_basic_auth_token,
		    sizeof vhost->proxy_basic_auth_token) < 0)
			goto auth_too_long;

		lwsl_info(" Proxy auth in use\n");

		proxy = p + 1;
	} else
		vhost->proxy_basic_auth_token[0] = '\0';

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	lws_strncpy(vhost->http.http_proxy_address, proxy,
		    sizeof(vhost->http.http_proxy_address));

	p = strchr(vhost->http.http_proxy_address, ':');
	if (!p && !vhost->http.http_proxy_port) {
		lwsl_err("http_proxy needs to be ads:port\n");

		return -1;
	} else {
		if (p) {
			*p = '\0';
			vhost->http.http_proxy_port = atoi(p + 1);
		}
	}

	lwsl_info(" Proxy %s:%u\n", vhost->http.http_proxy_address,
			vhost->http.http_proxy_port);
#endif
	return 0;

auth_too_long:
	lwsl_err("proxy auth too long\n");

	return -1;
}
#endif

#if defined(LWS_WITH_SOCKS5)
LWS_VISIBLE int
lws_set_socks(struct lws_vhost *vhost, const char *socks)
{
	char *p_at, *p_colon;
	char user[96];
	char password[96];

	if (!socks)
		return -1;

	vhost->socks_user[0] = '\0';
	vhost->socks_password[0] = '\0';

	p_at = strrchr(socks, '@');
	if (p_at) { /* auth is around */
		if ((unsigned int)(p_at - socks) > (sizeof(user)
			+ sizeof(password) - 2)) {
			lwsl_err("Socks auth too long\n");
			goto bail;
		}

		p_colon = strchr(socks, ':');
		if (p_colon) {
			if ((unsigned int)(p_colon - socks) > (sizeof(user)
				- 1) ) {
				lwsl_err("Socks user too long\n");
				goto bail;
			}
			if ((unsigned int)(p_at - p_colon) > (sizeof(password)
				- 1) ) {
				lwsl_err("Socks password too long\n");
				goto bail;
			}

			lws_strncpy(vhost->socks_user, socks, p_colon - socks + 1);
			lws_strncpy(vhost->socks_password, p_colon + 1,
				p_at - (p_colon + 1) + 1);
		}

		lwsl_info(" Socks auth, user: %s, password: %s\n",
			vhost->socks_user, vhost->socks_password );

		socks = p_at + 1;
	}

	lws_strncpy(vhost->socks_proxy_address, socks,
		    sizeof(vhost->socks_proxy_address));

	p_colon = strchr(vhost->socks_proxy_address, ':');
	if (!p_colon && !vhost->socks_proxy_port) {
		lwsl_err("socks_proxy needs to be address:port\n");
		return -1;
	} else {
		if (p_colon) {
			*p_colon = '\0';
			vhost->socks_proxy_port = atoi(p_colon + 1);
		}
	}

	lwsl_info(" Socks %s:%u\n", vhost->socks_proxy_address,
			vhost->socks_proxy_port);

	return 0;

bail:
	return -1;
}
#endif

LWS_VISIBLE const struct lws_protocols *
lws_get_protocol(struct lws *wsi)
{
	return wsi->protocol;
}


int
lws_ensure_user_space(struct lws *wsi)
{
	if (!wsi->protocol)
		return 0;

	/* allocate the per-connection user memory (if any) */

	if (wsi->protocol->per_session_data_size && !wsi->user_space) {
		wsi->user_space = lws_zalloc(
			    wsi->protocol->per_session_data_size, "user space");
		if (wsi->user_space == NULL) {
			lwsl_err("%s: OOM\n", __func__);
			return 1;
		}
	} else
		lwsl_debug("%s: %p protocol pss %lu, user_space=%p\n", __func__,
			   wsi, (long)wsi->protocol->per_session_data_size,
			   wsi->user_space);
	return 0;
}

LWS_VISIBLE void *
lws_adjust_protocol_psds(struct lws *wsi, size_t new_size)
{
	((struct lws_protocols *)lws_get_protocol(wsi))->per_session_data_size =
		new_size;

	if (lws_ensure_user_space(wsi))
			return NULL;

	return wsi->user_space;
}

LWS_VISIBLE int
lwsl_timestamp(int level, char *p, int len)
{
#ifndef LWS_PLAT_OPTEE
	time_t o_now = time(NULL);
	unsigned long long now;
	struct tm *ptm = NULL;
#ifndef WIN32
	struct tm tm;
#endif
	int n;

#ifndef _WIN32_WCE
#ifdef WIN32
	ptm = localtime(&o_now);
#else
	if (localtime_r(&o_now, &tm))
		ptm = &tm;
#endif
#endif
	p[0] = '\0';
	for (n = 0; n < LLL_COUNT; n++) {
		if (level != (1 << n))
			continue;
		now = time_in_microseconds() / 100;
		if (ptm)
			n = lws_snprintf(p, len,
				"[%04d/%02d/%02d %02d:%02d:%02d:%04d] %s: ",
				ptm->tm_year + 1900,
				ptm->tm_mon + 1,
				ptm->tm_mday,
				ptm->tm_hour,
				ptm->tm_min,
				ptm->tm_sec,
				(int)(now % 10000), log_level_names[n]);
		else
			n = lws_snprintf(p, len, "[%llu:%04d] %s: ",
					(unsigned long long) now / 10000,
					(int)(now % 10000), log_level_names[n]);
		return n;
	}
#else
	p[0] = '\0';
#endif

	return 0;
}

#ifndef LWS_PLAT_OPTEE
static const char * const colours[] = {
	"[31;1m", /* LLL_ERR */
	"[36;1m", /* LLL_WARN */
	"[35;1m", /* LLL_NOTICE */
	"[32;1m", /* LLL_INFO */
	"[34;1m", /* LLL_DEBUG */
	"[33;1m", /* LLL_PARSER */
	"[33m", /* LLL_HEADER */
	"[33m", /* LLL_EXT */
	"[33m", /* LLL_CLIENT */
	"[33;1m", /* LLL_LATENCY */
	"[30;1m", /* LLL_USER */
};

LWS_VISIBLE void lwsl_emit_stderr(int level, const char *line)
{
	char buf[50];
	static char tty = 3;
	int n, m = LWS_ARRAY_SIZE(colours) - 1;

	if (!tty)
		tty = isatty(2) | 2;
	lwsl_timestamp(level, buf, sizeof(buf));

	if (tty == 3) {
		n = 1 << (LWS_ARRAY_SIZE(colours) - 1);
		while (n) {
			if (level & n)
				break;
			m--;
			n >>= 1;
		}
		fprintf(stderr, "%c%s%s%s%c[0m", 27, colours[m], buf, line, 27);
	} else
		fprintf(stderr, "%s%s", buf, line);
}
#endif

LWS_VISIBLE void _lws_logv(int filter, const char *format, va_list vl)
{
	char buf[256];
	int n;

	if (!(log_level & filter))
		return;

	n = vsnprintf(buf, sizeof(buf) - 1, format, vl);
	(void)n;
	/* vnsprintf returns what it would have written, even if truncated */
	if (n > (int)sizeof(buf) - 1)
		n = sizeof(buf) - 1;
	if (n > 0)
		buf[n] = '\0';

	lwsl_emit(filter, buf);
}

LWS_VISIBLE void _lws_log(int filter, const char *format, ...)
{
	va_list ap;

	va_start(ap, format);
	_lws_logv(filter, format, ap);
	va_end(ap);
}

LWS_VISIBLE void lws_set_log_level(int level,
				   void (*func)(int level, const char *line))
{
	log_level = level;
	if (func)
		lwsl_emit = func;
}

LWS_VISIBLE int lwsl_visible(int level)
{
	return log_level & level;
}

LWS_VISIBLE void
lwsl_hexdump_level(int hexdump_level, const void *vbuf, size_t len)
{
	unsigned char *buf = (unsigned char *)vbuf;
	unsigned int n, m, start;
	char line[80];
	char *p;

	if (!lwsl_visible(hexdump_level))
		return;

	if (!len)
		return;

	if (!vbuf)
		return;

	_lws_log(hexdump_level, "\n");

	for (n = 0; n < len;) {
		start = n;
		p = line;

		p += sprintf(p, "%04X: ", start);

		for (m = 0; m < 16 && n < len; m++)
			p += sprintf(p, "%02X ", buf[n++]);
		while (m++ < 16)
			p += sprintf(p, "   ");

		p += sprintf(p, "   ");

		for (m = 0; m < 16 && (start + m) < len; m++) {
			if (buf[start + m] >= ' ' && buf[start + m] < 127)
				*p++ = buf[start + m];
			else
				*p++ = '.';
		}
		while (m++ < 16)
			*p++ = ' ';

		*p++ = '\n';
		*p = '\0';
		_lws_log(hexdump_level, "%s", line);
		(void)line;
	}

	_lws_log(hexdump_level, "\n");
}

LWS_VISIBLE void
lwsl_hexdump(const void *vbuf, size_t len)
{
#if defined(_DEBUG)
	lwsl_hexdump_level(LLL_DEBUG, vbuf, len);
#endif
}

LWS_VISIBLE int
lws_is_ssl(struct lws *wsi)
{
#if defined(LWS_WITH_TLS)
	return wsi->tls.use_ssl & LCCSCF_USE_SSL;
#else
	(void)wsi;
	return 0;
#endif
}

#if defined(LWS_WITH_TLS) && !defined(LWS_WITH_MBEDTLS)
LWS_VISIBLE lws_tls_conn*
lws_get_ssl(struct lws *wsi)
{
	return wsi->tls.ssl;
}
#endif

LWS_VISIBLE int
lws_partial_buffered(struct lws *wsi)
{
	return !!wsi->trunc_len;
}

LWS_VISIBLE lws_fileofs_t
lws_get_peer_write_allowance(struct lws *wsi)
{
	if (!wsi->role_ops->tx_credit)
		return -1;
	return wsi->role_ops->tx_credit(wsi);
}

LWS_VISIBLE void
lws_role_transition(struct lws *wsi, enum lwsi_role role, enum lwsi_state state,
		struct lws_role_ops *ops)
{
#if defined(_DEBUG)
	const char *name = "(unset)";
#endif
	wsi->wsistate = role | state;
	if (ops)
		wsi->role_ops = ops;
#if defined(_DEBUG)
	if (wsi->role_ops)
		name = wsi->role_ops->name;
	lwsl_debug("%s: %p: wsistate 0x%x, ops %s\n", __func__, wsi,
		   wsi->wsistate, name);
#endif
}

LWS_VISIBLE struct lws_plat_file_ops *
lws_get_fops(struct lws_context *context)
{
	return (struct lws_plat_file_ops *)context->fops;
}

LWS_VISIBLE LWS_EXTERN struct lws_context *
lws_get_context(const struct lws *wsi)
{
	return wsi->context;
}

LWS_VISIBLE LWS_EXTERN int
lws_get_count_threads(struct lws_context *context)
{
	return context->count_threads;
}

LWS_VISIBLE LWS_EXTERN void *
lws_wsi_user(struct lws *wsi)
{
	return wsi->user_space;
}

LWS_VISIBLE LWS_EXTERN void
lws_set_wsi_user(struct lws *wsi, void *data)
{
	if (wsi->user_space_externally_allocated)
		wsi->user_space = data;
	else
		lwsl_err("%s: Cannot set internally-allocated user_space\n",
			 __func__);
}

LWS_VISIBLE LWS_EXTERN struct lws *
lws_get_parent(const struct lws *wsi)
{
	return wsi->parent;
}

LWS_VISIBLE LWS_EXTERN struct lws *
lws_get_child(const struct lws *wsi)
{
	return wsi->child_list;
}

LWS_VISIBLE LWS_EXTERN void
lws_set_parent_carries_io(struct lws *wsi)
{
	wsi->parent_carries_io = 1;
}

LWS_VISIBLE LWS_EXTERN void *
lws_get_opaque_parent_data(const struct lws *wsi)
{
	return wsi->opaque_parent_data;
}

LWS_VISIBLE LWS_EXTERN void
lws_set_opaque_parent_data(struct lws *wsi, void *data)
{
	wsi->opaque_parent_data = data;
}

LWS_VISIBLE LWS_EXTERN int
lws_get_child_pending_on_writable(const struct lws *wsi)
{
	return wsi->parent_pending_cb_on_writable;
}

LWS_VISIBLE LWS_EXTERN void
lws_clear_child_pending_on_writable(struct lws *wsi)
{
	wsi->parent_pending_cb_on_writable = 0;
}


LWS_EXTERN int
__lws_rx_flow_control(struct lws *wsi)
{
	struct lws *wsic = wsi->child_list;

	// h2 ignores rx flow control atm
	if (lwsi_role_h2(wsi) || wsi->http2_substream ||
	    lwsi_role_h2_ENCAPSULATION(wsi))
		return 0; // !!!

	/* if he has children, do those if they were changed */
	while (wsic) {
		if (wsic->rxflow_change_to & LWS_RXFLOW_PENDING_CHANGE)
			__lws_rx_flow_control(wsic);

		wsic = wsic->sibling_list;
	}

	/* there is no pending change */
	if (!(wsi->rxflow_change_to & LWS_RXFLOW_PENDING_CHANGE))
		return 0;

	/* stuff is still buffered, not ready to really accept new input */
	if (lws_buflist_next_segment_len(&wsi->buflist, NULL)) {
		/* get ourselves called back to deal with stashed buffer */
		lws_callback_on_writable(wsi);
		return 0;
	}

	/* now the pending is cleared, we can change rxflow state */

	wsi->rxflow_change_to &= ~LWS_RXFLOW_PENDING_CHANGE;

	lwsl_info("rxflow: wsi %p change_to %d\n", wsi,
			      wsi->rxflow_change_to & LWS_RXFLOW_ALLOW);

	/* adjust the pollfd for this wsi */

	if (wsi->rxflow_change_to & LWS_RXFLOW_ALLOW) {
		if (__lws_change_pollfd(wsi, 0, LWS_POLLIN)) {
			lwsl_info("%s: fail\n", __func__);
			return -1;
		}
	} else
		if (__lws_change_pollfd(wsi, LWS_POLLIN, 0))
			return -1;

	return 0;
}

LWS_EXTERN int
lws_check_utf8(unsigned char *state, unsigned char *buf, size_t len)
{
	static const unsigned char e0f4[] = {
		0xa0 | ((2 - 1) << 2) | 1, /* e0 */
		0x80 | ((4 - 1) << 2) | 1, /* e1 */
		0x80 | ((4 - 1) << 2) | 1, /* e2 */
		0x80 | ((4 - 1) << 2) | 1, /* e3 */
		0x80 | ((4 - 1) << 2) | 1, /* e4 */
		0x80 | ((4 - 1) << 2) | 1, /* e5 */
		0x80 | ((4 - 1) << 2) | 1, /* e6 */
		0x80 | ((4 - 1) << 2) | 1, /* e7 */
		0x80 | ((4 - 1) << 2) | 1, /* e8 */
		0x80 | ((4 - 1) << 2) | 1, /* e9 */
		0x80 | ((4 - 1) << 2) | 1, /* ea */
		0x80 | ((4 - 1) << 2) | 1, /* eb */
		0x80 | ((4 - 1) << 2) | 1, /* ec */
		0x80 | ((2 - 1) << 2) | 1, /* ed */
		0x80 | ((4 - 1) << 2) | 1, /* ee */
		0x80 | ((4 - 1) << 2) | 1, /* ef */
		0x90 | ((3 - 1) << 2) | 2, /* f0 */
		0x80 | ((4 - 1) << 2) | 2, /* f1 */
		0x80 | ((4 - 1) << 2) | 2, /* f2 */
		0x80 | ((4 - 1) << 2) | 2, /* f3 */
		0x80 | ((1 - 1) << 2) | 2, /* f4 */

		0,			   /* s0 */
		0x80 | ((4 - 1) << 2) | 0, /* s2 */
		0x80 | ((4 - 1) << 2) | 1, /* s3 */
	};
	unsigned char s = *state;

	while (len--) {
		unsigned char c = *buf++;

		if (!s) {
			if (c >= 0x80) {
				if (c < 0xc2 || c > 0xf4)
					return 1;
				if (c < 0xe0)
					s = 0x80 | ((4 - 1) << 2);
				else
					s = e0f4[c - 0xe0];
			}
		} else {
			if (c < (s & 0xf0) ||
			    c >= (s & 0xf0) + 0x10 + ((s << 2) & 0x30))
				return 1;
			s = e0f4[21 + (s & 3)];
		}
	}

	*state = s;

	return 0;
}

LWS_VISIBLE LWS_EXTERN int
lws_parse_uri(char *p, const char **prot, const char **ads, int *port,
	      const char **path)
{
	const char *end;
	static const char *slash = "/";

	/* cut up the location into address, port and path */
	*prot = p;
	while (*p && (*p != ':' || p[1] != '/' || p[2] != '/'))
		p++;
	if (!*p) {
		end = p;
		p = (char *)*prot;
		*prot = end;
	} else {
		*p = '\0';
		p += 3;
	}
	*ads = p;
	if (!strcmp(*prot, "http") || !strcmp(*prot, "ws"))
		*port = 80;
	else if (!strcmp(*prot, "https") || !strcmp(*prot, "wss"))
		*port = 443;

       if (*p == '[')
       {
               ++(*ads);
               while (*p && *p != ']')
                       p++;
               if (*p)
                       *p++ = '\0';
       }
       else
       {
               while (*p && *p != ':' && *p != '/')
                       p++;
       }
	if (*p == ':') {
		*p++ = '\0';
		*port = atoi(p);
		while (*p && *p != '/')
			p++;
	}
	*path = slash;
	if (*p) {
		*p++ = '\0';
		if (*p)
			*path = p;
	}

	return 0;
}

#if defined(LWS_WITHOUT_EXTENSIONS)

/* we need to provide dummy callbacks for internal exts
 * so user code runs when faced with a lib compiled with
 * extensions disabled.
 */

LWS_VISIBLE int
lws_extension_callback_pm_deflate(struct lws_context *context,
                                  const struct lws_extension *ext,
                                  struct lws *wsi,
                                  enum lws_extension_callback_reasons reason,
                                  void *user, void *in, size_t len)
{
	(void)context;
	(void)ext;
	(void)wsi;
	(void)reason;
	(void)user;
	(void)in;
	(void)len;

	return 0;
}

LWS_EXTERN int
lws_set_extension_option(struct lws *wsi, const char *ext_name,
			 const char *opt_name, const char *opt_val)
{
	return -1;
}
#endif

LWS_EXTERN int
lws_socket_bind(struct lws_vhost *vhost, lws_sockfd_type sockfd, int port,
		const char *iface)
{
#ifdef LWS_WITH_UNIX_SOCK
	struct sockaddr_un serv_unix;
#endif
#ifdef LWS_WITH_IPV6
	struct sockaddr_in6 serv_addr6;
#endif
	struct sockaddr_in serv_addr4;
#ifndef LWS_PLAT_OPTEE
	socklen_t len = sizeof(struct sockaddr_storage);
#endif
	int n;
#if !defined(LWS_WITH_ESP32)
	int m;
#endif
	struct sockaddr_storage sin;
	struct sockaddr *v;

#ifdef LWS_WITH_UNIX_SOCK
	if (LWS_UNIX_SOCK_ENABLED(vhost)) {
		v = (struct sockaddr *)&serv_unix;
		n = sizeof(struct sockaddr_un);
		bzero((char *) &serv_unix, sizeof(serv_unix));
		serv_unix.sun_family = AF_UNIX;
		if (!iface)
			return -1;
		if (sizeof(serv_unix.sun_path) <= strlen(iface)) {
			lwsl_err("\"%s\" too long for UNIX domain socket\n",
			         iface);
			return -1;
		}
		strcpy(serv_unix.sun_path, iface);
		if (serv_unix.sun_path[0] == '@')
			serv_unix.sun_path[0] = '\0';

	} else
#endif
#if defined(LWS_WITH_IPV6) && !defined(LWS_WITH_ESP32)
	if (LWS_IPV6_ENABLED(vhost)) {
		v = (struct sockaddr *)&serv_addr6;
		n = sizeof(struct sockaddr_in6);
		bzero((char *) &serv_addr6, sizeof(serv_addr6));
		if (iface) {
			m = interface_to_sa(vhost, iface,
				    (struct sockaddr_in *)v, n);
			if (m == LWS_ITOSA_NOT_USABLE) {
				lwsl_info("%s: netif %s: Not usable\n",
					 __func__, iface);
				return m;
			}
			if (m == LWS_ITOSA_NOT_EXIST) {
				lwsl_info("%s: netif %s: Does not exist\n",
					 __func__, iface);
				return m;
			}
			serv_addr6.sin6_scope_id = lws_get_addr_scope(iface);
		}

		serv_addr6.sin6_family = AF_INET6;
		serv_addr6.sin6_port = htons(port);
	} else
#endif
	{
		v = (struct sockaddr *)&serv_addr4;
		n = sizeof(serv_addr4);
		bzero((char *) &serv_addr4, sizeof(serv_addr4));
		serv_addr4.sin_addr.s_addr = INADDR_ANY;
		serv_addr4.sin_family = AF_INET;
#if !defined(LWS_WITH_ESP32)

		if (iface) {
		    m = interface_to_sa(vhost, iface,
				    (struct sockaddr_in *)v, n);
			if (m == LWS_ITOSA_NOT_USABLE) {
				lwsl_info("%s: netif %s: Not usable\n",
					 __func__, iface);
				return m;
			}
			if (m == LWS_ITOSA_NOT_EXIST) {
				lwsl_info("%s: netif %s: Does not exist\n",
					 __func__, iface);
				return m;
			}
		}
#endif
		serv_addr4.sin_port = htons(port);
	} /* ipv4 */

	/* just checking for the interface extant */
	if (sockfd == LWS_SOCK_INVALID)
		return 0;

	n = bind(sockfd, v, n);
#ifdef LWS_WITH_UNIX_SOCK
	if (n < 0 && LWS_UNIX_SOCK_ENABLED(vhost)) {
		lwsl_err("ERROR on binding fd %d to \"%s\" (%d %d)\n",
				sockfd, iface, n, LWS_ERRNO);
		return -1;
	} else
#endif
	if (n < 0) {
		lwsl_err("ERROR on binding fd %d to port %d (%d %d)\n",
				sockfd, port, n, LWS_ERRNO);
		return -1;
	}

#ifndef LWS_PLAT_OPTEE
	if (getsockname(sockfd, (struct sockaddr *)&sin, &len) == -1)
		lwsl_warn("getsockname: %s\n", strerror(LWS_ERRNO));
	else
#endif
#if defined(LWS_WITH_IPV6)
		port = (sin.ss_family == AF_INET6) ?
			ntohs(((struct sockaddr_in6 *) &sin)->sin6_port) :
			ntohs(((struct sockaddr_in *) &sin)->sin_port);
#else
		{
			struct sockaddr_in sain;
			memcpy(&sain, &sin, sizeof(sain));
			port = ntohs(sain.sin_port);
		}
#endif

	return port;
}

LWS_VISIBLE LWS_EXTERN int
lws_get_vhost_listen_port(struct lws_vhost *vhost)
{
	return vhost->listen_port;
}

#if defined(LWS_WITH_IPV6)
LWS_EXTERN unsigned long
lws_get_addr_scope(const char *ipaddr)
{
	unsigned long scope = 0;

#ifndef WIN32
	struct ifaddrs *addrs, *addr;
	char ip[NI_MAXHOST];
	unsigned int i;

	getifaddrs(&addrs);
	for (addr = addrs; addr; addr = addr->ifa_next) {
		if (!addr->ifa_addr ||
			addr->ifa_addr->sa_family != AF_INET6)
			continue;

		getnameinfo(addr->ifa_addr,
				sizeof(struct sockaddr_in6),
				ip, sizeof(ip),
				NULL, 0, NI_NUMERICHOST);

		i = 0;
		while (ip[i])
			if (ip[i++] == '%') {
				ip[i - 1] = '\0';
				break;
			}

		if (!strcmp(ip, ipaddr)) {
			scope = if_nametoindex(addr->ifa_name);
			break;
		}
	}
	freeifaddrs(addrs);
#else
	PIP_ADAPTER_ADDRESSES adapter, addrs = NULL;
	PIP_ADAPTER_UNICAST_ADDRESS addr;
	ULONG size = 0;
	DWORD ret;
	struct sockaddr_in6 *sockaddr;
	char ip[NI_MAXHOST];
	unsigned int i;
	int found = 0;

	for (i = 0; i < 5; i++)
	{
		ret = GetAdaptersAddresses(AF_INET6, GAA_FLAG_INCLUDE_PREFIX,
				NULL, addrs, &size);
		if ((ret == NO_ERROR) || (ret == ERROR_NO_DATA)) {
			break;
		} else if (ret == ERROR_BUFFER_OVERFLOW)
		{
			if (addrs)
				free(addrs);
			addrs = (IP_ADAPTER_ADDRESSES *)malloc(size);
		} else
		{
			if (addrs)
			{
				free(addrs);
				addrs = NULL;
			}
			lwsl_err("Failed to get IPv6 address table (%d)", ret);
			break;
		}
	}

	if ((ret == NO_ERROR) && (addrs)) {
		adapter = addrs;
		while (adapter && !found) {
			addr = adapter->FirstUnicastAddress;
			while (addr && !found) {
				if (addr->Address.lpSockaddr->sa_family ==
				    AF_INET6) {
					sockaddr = (struct sockaddr_in6 *)
						(addr->Address.lpSockaddr);

					lws_plat_inet_ntop(sockaddr->sin6_family,
							&sockaddr->sin6_addr,
							ip, sizeof(ip));

					if (!strcmp(ip, ipaddr)) {
						scope = sockaddr->sin6_scope_id;
						found = 1;
						break;
					}
				}
				addr = addr->Next;
			}
			adapter = adapter->Next;
		}
	}
	if (addrs)
		free(addrs);
#endif

	return scope;
}
#endif

#if !defined(LWS_NO_SERVER)

LWS_EXTERN struct lws *
lws_create_adopt_udp(struct lws_vhost *vhost, int port, int flags,
		     const char *protocol_name, struct lws *parent_wsi)
{
	lws_sock_file_fd_type sock;
	struct addrinfo h, *r, *rp;
	struct lws *wsi = NULL;
	char buf[16];
	int n;

	memset(&h, 0, sizeof(h));
	h.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
	h.ai_socktype = SOCK_DGRAM;
	h.ai_protocol = IPPROTO_UDP;
	h.ai_flags = AI_PASSIVE | AI_ADDRCONFIG;

	lws_snprintf(buf, sizeof(buf), "%u", port);
	n = getaddrinfo(NULL, buf, &h, &r);
	if (n) {
		lwsl_info("%s: getaddrinfo error: %s\n", __func__,
			  gai_strerror(n));
		goto bail;
	}

	for (rp = r; rp; rp = rp->ai_next) {
		sock.sockfd = socket(rp->ai_family, rp->ai_socktype,
				     rp->ai_protocol);
		if (sock.sockfd >= 0)
			break;
	}
	if (!rp) {
		lwsl_err("%s: unable to create INET socket\n", __func__);
		goto bail1;
	}

	if ((flags & LWS_CAUDP_BIND) && bind(sock.sockfd, rp->ai_addr,
#if defined(_WIN32)
			    (int)rp->ai_addrlen
#else
			    rp->ai_addrlen
#endif
	   ) == -1) {
		lwsl_err("%s: bind failed\n", __func__);
		goto bail2;
	}

	wsi = lws_adopt_descriptor_vhost(vhost, LWS_ADOPT_RAW_SOCKET_UDP, sock,
				        protocol_name, parent_wsi);
	if (!wsi)
		lwsl_err("%s: udp adoption failed\n", __func__);

bail2:
	if (!wsi)
		close((int)sock.sockfd);
bail1:
	freeaddrinfo(r);

bail:
	return wsi;
}

#endif



static const char *hex = "0123456789ABCDEF";

LWS_VISIBLE LWS_EXTERN const char *
lws_sql_purify(char *escaped, const char *string, int len)
{
	const char *p = string;
	char *q = escaped;

	while (*p && len-- > 2) {
		if (*p == '\'') {
			*q++ = '\'';
			*q++ = '\'';
			len --;
			p++;
		} else
			*q++ = *p++;
	}
	*q = '\0';

	return escaped;
}

LWS_VISIBLE LWS_EXTERN const char *
lws_json_purify(char *escaped, const char *string, int len)
{
	const char *p = string;
	char *q = escaped;

	if (!p) {
		escaped[0] = '\0';
		return escaped;
	}

	while (*p && len-- > 6) {
		if (*p == '\"' || *p == '\\' || *p < 0x20) {
			*q++ = '\\';
			*q++ = 'u';
			*q++ = '0';
			*q++ = '0';
			*q++ = hex[((*p) >> 4) & 15];
			*q++ = hex[(*p) & 15];
			len -= 5;
			p++;
		} else
			*q++ = *p++;
	}
	*q = '\0';

	return escaped;
}

LWS_VISIBLE LWS_EXTERN void
lws_filename_purify_inplace(char *filename)
{
	while (*filename) {

		if (*filename == '.' && filename[1] == '.') {
			*filename = '_';
			filename[1] = '_';
		}

		if (*filename == ':' ||
		    *filename == '/' ||
		    *filename == '\\' ||
		    *filename == '$' ||
		    *filename == '%')
			*filename = '_';

		filename++;
	}
}

LWS_VISIBLE LWS_EXTERN const char *
lws_urlencode(char *escaped, const char *string, int len)
{
	const char *p = string;
	char *q = escaped;

	while (*p && len-- > 3) {
		if (*p == ' ') {
			*q++ = '+';
			p++;
			continue;
		}
		if ((*p >= '0' && *p <= '9') ||
		    (*p >= 'A' && *p <= 'Z') ||
		    (*p >= 'a' && *p <= 'z')) {
			*q++ = *p++;
			continue;
		}
		*q++ = '%';
		*q++ = hex[(*p >> 4) & 0xf];
		*q++ = hex[*p & 0xf];

		len -= 2;
		p++;
	}
	*q = '\0';

	return escaped;
}

LWS_VISIBLE LWS_EXTERN int
lws_urldecode(char *string, const char *escaped, int len)
{
	int state = 0, n;
	char sum = 0;

	while (*escaped && len) {
		switch (state) {
		case 0:
			if (*escaped == '%') {
				state++;
				escaped++;
				continue;
			}
			if (*escaped == '+') {
				escaped++;
				*string++ = ' ';
				len--;
				continue;
			}
			*string++ = *escaped++;
			len--;
			break;
		case 1:
			n = char_to_hex(*escaped);
			if (n < 0)
				return -1;
			escaped++;
			sum = n << 4;
			state++;
			break;

		case 2:
			n = char_to_hex(*escaped);
			if (n < 0)
				return -1;
			escaped++;
			*string++ = sum | n;
			len--;
			state = 0;
			break;
		}

	}
	*string = '\0';

	return 0;
}

LWS_VISIBLE LWS_EXTERN int
lws_finalize_startup(struct lws_context *context)
{
	struct lws_context_creation_info info;

	info.uid = context->uid;
	info.gid = context->gid;

#if defined(LWS_HAVE_SYS_CAPABILITY_H) && defined(LWS_HAVE_LIBCAP)
	memcpy(info.caps, context->caps, sizeof(info.caps));
	info.count_caps = context->count_caps;
#endif

	if (lws_check_opt(context->options, LWS_SERVER_OPTION_EXPLICIT_VHOSTS))
		lws_plat_drop_app_privileges(&info);

	return 0;
}

int
lws_snprintf(char *str, size_t size, const char *format, ...)
{
	va_list ap;
	int n;

	if (!size)
		return 0;

	va_start(ap, format);
	n = vsnprintf(str, size, format, ap);
	va_end(ap);

	if (n >= (int)size)
		return (int)size;

	return n;
}

char *
lws_strncpy(char *dest, const char *src, size_t size)
{
	strncpy(dest, src, size - 1);
	dest[size - 1] = '\0';

	return dest;
}


LWS_VISIBLE LWS_EXTERN int
lws_is_cgi(struct lws *wsi) {
#ifdef LWS_WITH_CGI
	return !!wsi->http.cgi;
#else
	return 0;
#endif
}

const struct lws_protocol_vhost_options *
lws_pvo_search(const struct lws_protocol_vhost_options *pvo, const char *name)
{
	while (pvo) {
		if (!strcmp(pvo->name, name))
			break;

		pvo = pvo->next;
	}

	return pvo;
}

void
lws_sum_stats(const struct lws_context *ctx, struct lws_conn_stats *cs)
{
	const struct lws_vhost *vh = ctx->vhost_list;

	while (vh) {

		cs->rx += vh->conn_stats.rx;
		cs->tx += vh->conn_stats.tx;
		cs->h1_conn += vh->conn_stats.h1_conn;
		cs->h1_trans += vh->conn_stats.h1_trans;
		cs->h2_trans += vh->conn_stats.h2_trans;
		cs->ws_upg += vh->conn_stats.ws_upg;
		cs->h2_upg += vh->conn_stats.h2_upg;
		cs->h2_alpn += vh->conn_stats.h2_alpn;
		cs->h2_subs += vh->conn_stats.h2_subs;
		cs->rejected += vh->conn_stats.rejected;

		vh = vh->vhost_next;
	}
}

const char *
lws_cmdline_option(int argc, const char **argv, const char *val)
{
	int n = (int)strlen(val), c = argc;

	while (--c > 0) {

		if (!strncmp(argv[c], val, n)) {
			if (!*(argv[c] + n) && c < argc - 1) {
				/* coverity treats unchecked argv as "tainted" */
				if (!argv[c + 1] || strlen(argv[c + 1]) > 1024)
					return NULL;
				return argv[c + 1];
			}

			return argv[c] + n;
		}
	}

	return NULL;
}

#ifdef LWS_WITH_SERVER_STATUS

LWS_EXTERN int
lws_json_dump_vhost(const struct lws_vhost *vh, char *buf, int len)
{
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	static const char * const prots[] = {
		"http://",
		"https://",
		"file://",
		"cgi://",
		">http://",
		">https://",
		"callback://"
	};
#endif
	char *orig = buf, *end = buf + len - 1, first = 1;
	int n = 0;

	if (len < 100)
		return 0;

	buf += lws_snprintf(buf, end - buf,
			"{\n \"name\":\"%s\",\n"
			" \"port\":\"%d\",\n"
			" \"use_ssl\":\"%d\",\n"
			" \"sts\":\"%d\",\n"
			" \"rx\":\"%llu\",\n"
			" \"tx\":\"%llu\",\n"
			" \"h1_conn\":\"%lu\",\n"
			" \"h1_trans\":\"%lu\",\n"
			" \"h2_trans\":\"%lu\",\n"
			" \"ws_upg\":\"%lu\",\n"
			" \"rejected\":\"%lu\",\n"
			" \"h2_upg\":\"%lu\",\n"
			" \"h2_alpn\":\"%lu\",\n"
			" \"h2_subs\":\"%lu\""
			,
			vh->name, vh->listen_port,
#if defined(LWS_WITH_TLS)
			vh->tls.use_ssl & LCCSCF_USE_SSL,
#else
			0,
#endif
			!!(vh->options & LWS_SERVER_OPTION_STS),
			vh->conn_stats.rx, vh->conn_stats.tx,
			vh->conn_stats.h1_conn,
			vh->conn_stats.h1_trans,
			vh->conn_stats.h2_trans,
			vh->conn_stats.ws_upg,
			vh->conn_stats.rejected,
			vh->conn_stats.h2_upg,
			vh->conn_stats.h2_alpn,
			vh->conn_stats.h2_subs
	);
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	if (vh->http.mount_list) {
		const struct lws_http_mount *m = vh->http.mount_list;

		buf += lws_snprintf(buf, end - buf, ",\n \"mounts\":[");
		while (m) {
			if (!first)
				buf += lws_snprintf(buf, end - buf, ",");
			buf += lws_snprintf(buf, end - buf,
					"\n  {\n   \"mountpoint\":\"%s\",\n"
					"  \"origin\":\"%s%s\",\n"
					"  \"cache_max_age\":\"%d\",\n"
					"  \"cache_reuse\":\"%d\",\n"
					"  \"cache_revalidate\":\"%d\",\n"
					"  \"cache_intermediaries\":\"%d\"\n"
					,
					m->mountpoint,
					prots[m->origin_protocol],
					m->origin,
					m->cache_max_age,
					m->cache_reusable,
					m->cache_revalidate,
					m->cache_intermediaries);
			if (m->def)
				buf += lws_snprintf(buf, end - buf,
						",\n  \"default\":\"%s\"",
						m->def);
			buf += lws_snprintf(buf, end - buf, "\n  }");
			first = 0;
			m = m->mount_next;
		}
		buf += lws_snprintf(buf, end - buf, "\n ]");
	}
#endif
	if (vh->protocols) {
		n = 0;
		first = 1;

		buf += lws_snprintf(buf, end - buf, ",\n \"ws-protocols\":[");
		while (n < vh->count_protocols) {
			if (!first)
				buf += lws_snprintf(buf, end - buf, ",");
			buf += lws_snprintf(buf, end - buf,
					"\n  {\n   \"%s\":{\n"
					"    \"status\":\"ok\"\n   }\n  }"
					,
					vh->protocols[n].name);
			first = 0;
			n++;
		}
		buf += lws_snprintf(buf, end - buf, "\n ]");
	}

	buf += lws_snprintf(buf, end - buf, "\n}");

	return buf - orig;
}


LWS_EXTERN LWS_VISIBLE int
lws_json_dump_context(const struct lws_context *context, char *buf, int len,
		int hide_vhosts)
{
	char *orig = buf, *end = buf + len - 1, first = 1;
	const struct lws_vhost *vh = context->vhost_list;
	const struct lws_context_per_thread *pt;
	time_t t = time(NULL);
	int n, listening = 0, cgi_count = 0;
	struct lws_conn_stats cs;
	double d = 0;
#ifdef LWS_WITH_CGI
	struct lws_cgi * const *pcgi;
#endif

#ifdef LWS_WITH_LIBUV
	uv_uptime(&d);
#endif

	buf += lws_snprintf(buf, end - buf, "{ "
			    "\"version\":\"%s\",\n"
			    "\"uptime\":\"%ld\",\n",
			    lws_get_library_version(),
			    (long)d);

#ifdef LWS_HAVE_GETLOADAVG
	{
		double d[3];
		int m;

		m = getloadavg(d, 3);
		for (n = 0; n < m; n++) {
			buf += lws_snprintf(buf, end - buf,
				"\"l%d\":\"%.2f\",\n",
				n + 1, d[n]);
		}
	}
#endif

	buf += lws_snprintf(buf, end - buf, "\"contexts\":[\n");

	buf += lws_snprintf(buf, end - buf, "{ "
				"\"context_uptime\":\"%ld\",\n"
				"\"cgi_spawned\":\"%d\",\n"
				"\"pt_fd_max\":\"%d\",\n"
				"\"ah_pool_max\":\"%d\",\n"
				"\"deprecated\":\"%d\",\n"
				"\"wsi_alive\":\"%d\",\n",
				(unsigned long)(t - context->time_up),
				context->count_cgi_spawned,
				context->fd_limit_per_thread,
				context->max_http_header_pool,
				context->deprecated,
				context->count_wsi_allocated);

	buf += lws_snprintf(buf, end - buf, "\"pt\":[\n ");
	for (n = 0; n < context->count_threads; n++) {
		pt = &context->pt[n];
		if (n)
			buf += lws_snprintf(buf, end - buf, ",");
		buf += lws_snprintf(buf, end - buf,
				"\n  {\n"
				"    \"fds_count\":\"%d\",\n"
				"    \"ah_pool_inuse\":\"%d\",\n"
				"    \"ah_wait_list\":\"%d\"\n"
				"    }",
				pt->fds_count,
				pt->http.ah_count_in_use,
				pt->http.ah_wait_list_length);
	}

	buf += lws_snprintf(buf, end - buf, "]");

	buf += lws_snprintf(buf, end - buf, ", \"vhosts\":[\n ");

	first = 1;
	vh = context->vhost_list;
	listening = 0;
	cs = context->conn_stats;
	lws_sum_stats(context, &cs);
	while (vh) {

		if (!hide_vhosts) {
			if (!first)
				if(buf != end)
					*buf++ = ',';
			buf += lws_json_dump_vhost(vh, buf, end - buf);
			first = 0;
		}
		if (vh->lserv_wsi)
			listening++;
		vh = vh->vhost_next;
	}

	buf += lws_snprintf(buf, end - buf,
			"],\n\"listen_wsi\":\"%d\",\n"
			" \"rx\":\"%llu\",\n"
			" \"tx\":\"%llu\",\n"
			" \"h1_conn\":\"%lu\",\n"
			" \"h1_trans\":\"%lu\",\n"
			" \"h2_trans\":\"%lu\",\n"
			" \"ws_upg\":\"%lu\",\n"
			" \"rejected\":\"%lu\",\n"
			" \"h2_alpn\":\"%lu\",\n"
			" \"h2_subs\":\"%lu\",\n"
			" \"h2_upg\":\"%lu\"",
			listening, cs.rx, cs.tx,
			cs.h1_conn,
			cs.h1_trans,
			cs.h2_trans,
			cs.ws_upg,
			cs.rejected,
			cs.h2_alpn,
			cs.h2_subs,
			cs.h2_upg);

#ifdef LWS_WITH_CGI
	for (n = 0; n < context->count_threads; n++) {
		pt = &context->pt[n];
		pcgi = &pt->http.cgi_list;

		while (*pcgi) {
			pcgi = &(*pcgi)->cgi_list;

			cgi_count++;
		}
	}
#endif
	buf += lws_snprintf(buf, end - buf, ",\n \"cgi_alive\":\"%d\"\n ",
			cgi_count);

	buf += lws_snprintf(buf, end - buf, "}");


	buf += lws_snprintf(buf, end - buf, "]}\n ");

	return buf - orig;
}

#endif

#if defined(LWS_WITH_STATS)

LWS_VISIBLE LWS_EXTERN uint64_t
lws_stats_get(struct lws_context *context, int index)
{
	if (index >= LWSSTATS_SIZE)
		return 0;

	return context->lws_stats[index];
}

LWS_VISIBLE LWS_EXTERN void
lws_stats_log_dump(struct lws_context *context)
{
	struct lws_vhost *v = context->vhost_list;
	int n, m;

	(void)m;

	if (!context->updated)
		return;

	context->updated = 0;

	lwsl_notice("\n");
	lwsl_notice("LWS internal statistics dump ----->\n");
	lwsl_notice("LWSSTATS_C_CONNECTIONS:                     %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_CONNECTIONS));
	lwsl_notice("LWSSTATS_C_API_CLOSE:                       %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_API_CLOSE));
	lwsl_notice("LWSSTATS_C_API_READ:                        %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_API_READ));
	lwsl_notice("LWSSTATS_C_API_LWS_WRITE:                   %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_API_LWS_WRITE));
	lwsl_notice("LWSSTATS_C_API_WRITE:                       %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_API_WRITE));
	lwsl_notice("LWSSTATS_C_WRITE_PARTIALS:                  %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_WRITE_PARTIALS));
	lwsl_notice("LWSSTATS_C_WRITEABLE_CB_REQ:                %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_WRITEABLE_CB_REQ));
	lwsl_notice("LWSSTATS_C_WRITEABLE_CB_EFF_REQ:            %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_WRITEABLE_CB_EFF_REQ));
	lwsl_notice("LWSSTATS_C_WRITEABLE_CB:                    %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_WRITEABLE_CB));
	lwsl_notice("LWSSTATS_C_SSL_CONNECTIONS_ACCEPT_SPIN:     %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_SSL_CONNECTIONS_ACCEPT_SPIN));
	lwsl_notice("LWSSTATS_C_SSL_CONNECTIONS_FAILED:          %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_SSL_CONNECTIONS_FAILED));
	lwsl_notice("LWSSTATS_C_SSL_CONNECTIONS_ACCEPTED:        %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_SSL_CONNECTIONS_ACCEPTED));
	lwsl_notice("LWSSTATS_C_SSL_CONNS_HAD_RX:                %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_SSL_CONNS_HAD_RX));
	lwsl_notice("LWSSTATS_C_PEER_LIMIT_AH_DENIED:            %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_PEER_LIMIT_AH_DENIED));
	lwsl_notice("LWSSTATS_C_PEER_LIMIT_WSI_DENIED:           %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_PEER_LIMIT_WSI_DENIED));

	lwsl_notice("LWSSTATS_C_TIMEOUTS:                        %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_TIMEOUTS));
	lwsl_notice("LWSSTATS_C_SERVICE_ENTRY:                   %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_C_SERVICE_ENTRY));
	lwsl_notice("LWSSTATS_B_READ:                            %8llu\n",
		(unsigned long long)lws_stats_get(context, LWSSTATS_B_READ));
	lwsl_notice("LWSSTATS_B_WRITE:                           %8llu\n",
		(unsigned long long)lws_stats_get(context, LWSSTATS_B_WRITE));
	lwsl_notice("LWSSTATS_B_PARTIALS_ACCEPTED_PARTS:         %8llu\n",
		(unsigned long long)lws_stats_get(context,
					LWSSTATS_B_PARTIALS_ACCEPTED_PARTS));
	lwsl_notice("LWSSTATS_MS_SSL_CONNECTIONS_ACCEPTED_DELAY: %8llums\n",
		(unsigned long long)lws_stats_get(context,
			LWSSTATS_MS_SSL_CONNECTIONS_ACCEPTED_DELAY) / 1000);
	if (lws_stats_get(context, LWSSTATS_C_SSL_CONNECTIONS_ACCEPTED))
		lwsl_notice("  Avg accept delay:                         %8llums\n",
			(unsigned long long)(lws_stats_get(context,
				LWSSTATS_MS_SSL_CONNECTIONS_ACCEPTED_DELAY) /
					lws_stats_get(context,
				LWSSTATS_C_SSL_CONNECTIONS_ACCEPTED)) / 1000);
	lwsl_notice("LWSSTATS_MS_SSL_RX_DELAY:                   %8llums\n",
			(unsigned long long)lws_stats_get(context,
					LWSSTATS_MS_SSL_RX_DELAY) / 1000);
	if (lws_stats_get(context, LWSSTATS_C_SSL_CONNS_HAD_RX))
		lwsl_notice("  Avg accept-rx delay:                      %8llums\n",
			(unsigned long long)(lws_stats_get(context,
					LWSSTATS_MS_SSL_RX_DELAY) /
			lws_stats_get(context,
					LWSSTATS_C_SSL_CONNS_HAD_RX)) / 1000);

	lwsl_notice("LWSSTATS_MS_WRITABLE_DELAY:                 %8lluus\n",
			(unsigned long long)lws_stats_get(context,
					LWSSTATS_MS_WRITABLE_DELAY));
	lwsl_notice("LWSSTATS_MS_WORST_WRITABLE_DELAY:           %8lluus\n",
				(unsigned long long)lws_stats_get(context,
					LWSSTATS_MS_WORST_WRITABLE_DELAY));
	if (lws_stats_get(context, LWSSTATS_C_WRITEABLE_CB))
		lwsl_notice("  Avg writable delay:                       %8lluus\n",
			(unsigned long long)(lws_stats_get(context,
					LWSSTATS_MS_WRITABLE_DELAY) /
			lws_stats_get(context, LWSSTATS_C_WRITEABLE_CB)));
	lwsl_notice("Simultaneous SSL restriction:               %8d/%d\n",
			context->simultaneous_ssl,
			context->simultaneous_ssl_restriction);

	lwsl_notice("Live wsi:                                   %8d\n",
			context->count_wsi_allocated);

	context->updated = 1;

	while (v) {
		if (v->lserv_wsi &&
		    v->lserv_wsi->position_in_fds_table != LWS_NO_FDS_POS) {

			struct lws_context_per_thread *pt =
					&context->pt[(int)v->lserv_wsi->tsi];
			struct lws_pollfd *pfd;

			pfd = &pt->fds[v->lserv_wsi->position_in_fds_table];

			lwsl_notice("  Listen port %d actual POLLIN:   %d\n",
				    v->listen_port,
				    (int)pfd->events & LWS_POLLIN);
		}

		v = v->vhost_next;
	}

	for (n = 0; n < context->count_threads; n++) {
		struct lws_context_per_thread *pt = &context->pt[n];
		struct lws *wl;
		int m = 0;

		lwsl_notice("PT %d\n", n + 1);

		lws_pt_lock(pt, __func__);

		lwsl_notice("  AH in use / max:                  %d / %d\n",
				pt->http.ah_count_in_use,
				context->max_http_header_pool);

		wl = pt->http.ah_wait_list;
		while (wl) {
			m++;
			wl = wl->http.ah_wait_list;
		}

		lwsl_notice("  AH wait list count / actual:      %d / %d\n",
				pt->http.ah_wait_list_length, m);

		lws_pt_unlock(pt);
	}

#if defined(LWS_WITH_PEER_LIMITS)
	m = 0;
	for (n = 0; n < (int)context->pl_hash_elements; n++) {
		lws_start_foreach_llp(struct lws_peer **, peer,
				      context->pl_hash_table[n]) {
			m++;
		} lws_end_foreach_llp(peer, next);
	}

	lwsl_notice(" Peers: total active %d\n", m);
	if (m > 10) {
		m = 10;
		lwsl_notice("  (showing 10 peers only)\n");
	}

	if (m) {
		for (n = 0; n < (int)context->pl_hash_elements; n++) {
			char buf[72];

			lws_start_foreach_llp(struct lws_peer **, peer,
					      context->pl_hash_table[n]) {
				struct lws_peer *df = *peer;

				if (!lws_plat_inet_ntop(df->af, df->addr, buf,
							sizeof(buf) - 1))
					strcpy(buf, "unknown");
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
				lwsl_notice("  peer %s: count wsi: %d, count ah: %d\n",
					    buf, df->count_wsi,
					    df->http.count_ah);
#else
				lwsl_notice("  peer %s: count wsi: %d\n",
					    buf, df->count_wsi);
#endif

				if (!--m)
					break;
			} lws_end_foreach_llp(peer, next);
		}
	}
#endif

	lwsl_notice("\n");
}

void
lws_stats_atomic_bump(struct lws_context * context,
		struct lws_context_per_thread *pt, int index, uint64_t bump)
{
	lws_pt_stats_lock(pt);
	context->lws_stats[index] += bump;
	if (index != LWSSTATS_C_SERVICE_ENTRY)
		context->updated = 1;
	lws_pt_stats_unlock(pt);
}

void
lws_stats_atomic_max(struct lws_context * context,
		struct lws_context_per_thread *pt, int index, uint64_t val)
{
	lws_pt_stats_lock(pt);
	if (val > context->lws_stats[index]) {
		context->lws_stats[index] = val;
		context->updated = 1;
	}
	lws_pt_stats_unlock(pt);
}

#endif

