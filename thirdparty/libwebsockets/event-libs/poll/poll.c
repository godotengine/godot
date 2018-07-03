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
 *  This is included from core/private.h if LWS_ROLE_WS
 */

#include <core/private.h>

struct lws_event_loop_ops event_loop_ops_poll = {
	/* name */			"poll",
	/* init_context */		NULL,
	/* destroy_context1 */		NULL,
	/* destroy_context2 */		NULL,
	/* init_vhost_listen_wsi */	NULL,
	/* init_pt */			NULL,
	/* wsi_logical_close */		NULL,
	/* check_client_connect_ok */	NULL,
	/* close_handle_manually */	NULL,
	/* accept */			NULL,
	/* io */			NULL,
	/* run */			NULL,
	/* destroy_pt */		NULL,
	/* destroy wsi */		NULL,

	/* periodic_events_available */	1,
};