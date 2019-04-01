/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010 - 2018 Andy Green <andy@warmcat.com>
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
 *
 *  This is included from core/private.h
 */

struct lws_event_loop_ops {
	const char *name;
	/* event loop-specific context init during context creation */
	int (*init_context)(struct lws_context *context,
			    const struct lws_context_creation_info *info);
	/* called during lws_destroy_context */
	int (*destroy_context1)(struct lws_context *context);
	/* called during lws_destroy_context2 */
	int (*destroy_context2)(struct lws_context *context);
	/* init vhost listening wsi */
	int (*init_vhost_listen_wsi)(struct lws *wsi);
	/* init the event loop for a pt */
	int (*init_pt)(struct lws_context *context, void *_loop, int tsi);
	/* called at end of first phase of close_free_wsi()  */
	int (*wsi_logical_close)(struct lws *wsi);
	/* return nonzero if client connect not allowed  */
	int (*check_client_connect_ok)(struct lws *wsi);
	/* close handle manually  */
	void (*close_handle_manually)(struct lws *wsi);
	/* event loop accept processing  */
	void (*accept)(struct lws *wsi);
	/* control wsi active events  */
	void (*io)(struct lws *wsi, int flags);
	/* run the event loop for a pt */
	void (*run_pt)(struct lws_context *context, int tsi);
	/* called before pt is destroyed */
	void (*destroy_pt)(struct lws_context *context, int tsi);
	/* called just before wsi is freed  */
	void (*destroy_wsi)(struct lws *wsi);

	unsigned int periodic_events_available:1;
};

/* bring in event libs private declarations */

#if defined(LWS_WITH_POLL)
#include "event-libs/poll/private.h"
#endif

#if defined(LWS_WITH_LIBUV)
#include "event-libs/libuv/private.h"
#endif

#if defined(LWS_WITH_LIBEVENT)
#include "event-libs/libevent/private.h"
#endif

#if defined(LWS_WITH_LIBEV)
#include "event-libs/libev/private.h"
#endif

