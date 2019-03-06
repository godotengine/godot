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

void
lws_client_stash_destroy(struct lws *wsi)
{
	if (!wsi || !wsi->stash)
		return;

	lws_free_set_NULL(wsi->stash->address);
	lws_free_set_NULL(wsi->stash->path);
	lws_free_set_NULL(wsi->stash->host);
	lws_free_set_NULL(wsi->stash->origin);
	lws_free_set_NULL(wsi->stash->protocol);
	lws_free_set_NULL(wsi->stash->method);
	lws_free_set_NULL(wsi->stash->iface);
	lws_free_set_NULL(wsi->stash->alpn);

	lws_free_set_NULL(wsi->stash);
}

LWS_VISIBLE struct lws *
lws_client_connect_via_info(const struct lws_client_connect_info *i)
{
	struct lws *wsi, *safe = NULL;
	const struct lws_protocols *p;
	const char *local = i->protocol;
#if LWS_MAX_SMP > 1
	int n, tid;
#endif

	if (i->context->requested_kill)
		return NULL;

	if (!i->context->protocol_init_done)
		lws_protocol_init(i->context);
	/*
	 * If we have .local_protocol_name, use it to select the local protocol
	 * handler to bind to.  Otherwise use .protocol if http[s].
	 */
	if (i->local_protocol_name)
		local = i->local_protocol_name;

	/* PHASE 1: create a bare wsi */

	wsi = lws_zalloc(sizeof(struct lws), "client wsi");
	if (wsi == NULL)
		goto bail;

	wsi->context = i->context;
	wsi->desc.sockfd = LWS_SOCK_INVALID;

	wsi->vhost = NULL;
	if (!i->vhost)
		lws_vhost_bind_wsi(i->context->vhost_list, wsi);
	else
		lws_vhost_bind_wsi(i->vhost, wsi);

	if (!wsi->vhost) {
		lwsl_err("%s: No vhost in the context\n", __func__);

		goto bail;
	}

	/*
	 * PHASE 2: if SMP, bind the client to whatever tsi the current thread
	 * represents
	 */

#if LWS_MAX_SMP > 1
	tid = wsi->vhost->protocols[0].callback(wsi, LWS_CALLBACK_GET_THREAD_ID,
						NULL, NULL, 0);

	lws_context_lock(i->context, "client find tsi");

	for (n = 0; n < i->context->count_threads; n++)
		if (i->context->pt[n].service_tid == tid) {
			lwsl_info("%s: client binds to caller tsi %d\n",
				  __func__, n);
			wsi->tsi = n;
			break;
		}

	/*
	 * this binding is sort of provisional, since when we try to insert
	 * into the pt fds, there may be no space and it will fail
	 */

	lws_context_unlock(i->context);
#endif

	/*
	 * PHASE 3: Choose an initial role for the wsi and do role-specific init
	 *
	 * Note the initial role may not reflect the final role, eg,
	 * we may want ws, but first we have to go through h1 to get that
	 */

	lws_role_call_client_bind(wsi, i);

	/*
	 * PHASE 4: fill up the wsi with stuff from the connect_info as far as
	 * it can go.  It's uncertain because not only is our connection
	 * going to complete asynchronously, we might have bound to h1 and not
	 * even be able to get ahold of an ah immediately.
	 */

	wsi->user_space = NULL;
	wsi->pending_timeout = NO_PENDING_TIMEOUT;
	wsi->position_in_fds_table = LWS_NO_FDS_POS;
	wsi->c_port = i->port;

	wsi->protocol = &wsi->vhost->protocols[0];
	wsi->client_pipeline = !!(i->ssl_connection & LCCSCF_PIPELINE);

	/*
	 * PHASE 5: handle external user_space now, generic alloc is done in
	 * role finalization
	 */

	if (!wsi->user_space && i->userdata) {
		wsi->user_space_externally_allocated = 1;
		wsi->user_space = i->userdata;
	}

	if (local) {
		lwsl_info("%s: protocol binding to %s\n", __func__, local);
		p = lws_vhost_name_to_protocol(wsi->vhost, local);
		if (p)
			lws_bind_protocol(wsi, p, __func__);
	}

	/*
	 * PHASE 5: handle external user_space now, generic alloc is done in
	 * role finalization
	 */

	if (!wsi->user_space && i->userdata) {
		wsi->user_space_externally_allocated = 1;
		wsi->user_space = i->userdata;
	}

#if defined(LWS_WITH_TLS)
	wsi->tls.use_ssl = i->ssl_connection;
#else
	if (i->ssl_connection & LCCSCF_USE_SSL) {
		lwsl_err("%s: lws not configured for tls\n", __func__);
		goto bail;
	}
#endif

	/*
	 * PHASE 6: stash the things from connect_info that we can't process
	 * right now, eg, if http binding, without an ah.  If h1 and no ah, we
	 * will go on the ah waiting list and process those things later (after
	 * the connect_info and maybe the things pointed to have gone out of
	 * scope)
	 *
	 * However these things are stashed in a generic way at this point,
	 * with no relationship to http or ah
	 */

	wsi->stash = lws_zalloc(sizeof(*wsi->stash), "client stash");
	if (!wsi->stash) {
		lwsl_err("%s: OOM\n", __func__);
		goto bail1;
	}

	wsi->stash->address = lws_strdup(i->address);
	wsi->stash->path = lws_strdup(i->path);
	wsi->stash->host = lws_strdup(i->host);

	if (!wsi->stash->address || !wsi->stash->path || !wsi->stash->host)
		goto bail1;

	if (i->origin) {
		wsi->stash->origin = lws_strdup(i->origin);
		if (!wsi->stash->origin)
			goto bail1;
	}
	if (i->protocol) {
		wsi->stash->protocol = lws_strdup(i->protocol);
		if (!wsi->stash->protocol)
			goto bail1;
	}
	if (i->method) {
		wsi->stash->method = lws_strdup(i->method);
		if (!wsi->stash->method)
			goto bail1;
	}
	if (i->iface) {
		wsi->stash->iface = lws_strdup(i->iface);
		if (!wsi->stash->iface)
			goto bail1;
	}
	if (i->alpn) {
		wsi->stash->alpn = lws_strdup(i->alpn);
		if (!wsi->stash->alpn)
			goto bail1;
	}

	/*
	 * at this point user callbacks like
	 * LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER will be interested to
	 * know the parent... eg for proxying we can grab extra headers from
	 * the parent's incoming ah and add them to the child client handshake
	 */

	if (i->parent_wsi) {
		lwsl_info("%s: created child %p of parent %p\n", __func__,
			  wsi, i->parent_wsi);
		wsi->parent = i->parent_wsi;
		safe = wsi->sibling_list = i->parent_wsi->child_list;
		i->parent_wsi->child_list = wsi;
	}

	/*
	 * PHASE 7: Do any role-specific finalization processing.  We can still
	 * see important info things via wsi->stash
	 */

	if (wsi->role_ops->client_bind) {
		int n = wsi->role_ops->client_bind(wsi, NULL);

		if (n && i->parent_wsi) {
			/* unpick from parent */

			i->parent_wsi->child_list = safe;
		}

		if (n < 0)
			/* we didn't survive, wsi is freed */
			goto bail2;

		if (n)
			/* something else failed, wsi needs freeing */
			goto bail;
	}

	/* let the caller's optional wsi storage have the wsi we created */

	if (i->pwsi)
		*i->pwsi = wsi;


#if defined(LWS_WITH_HUBBUB)
	if (i->uri_replace_to)
		wsi->http.rw = lws_rewrite_create(wsi, html_parser_cb,
					     i->uri_replace_from,
					     i->uri_replace_to);
#endif

	return wsi;

bail1:
	lws_client_stash_destroy(wsi);

bail:
	lws_free(wsi);
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
bail2:
#endif
	if (i->pwsi)
		*i->pwsi = NULL;

	return NULL;
}
