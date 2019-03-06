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
rops_handle_POLLIN_raw_file(struct lws_context_per_thread *pt, struct lws *wsi,
			    struct lws_pollfd *pollfd)
{
	int n;

	if (pollfd->revents & LWS_POLLOUT) {
		n = lws_callback_as_writeable(wsi);
		if (lws_change_pollfd(wsi, LWS_POLLOUT, 0)) {
			lwsl_info("failed at set pollfd\n");
			return LWS_HPI_RET_WSI_ALREADY_DIED;
		}
		if (n)
			return LWS_HPI_RET_PLEASE_CLOSE_ME;
	}

	if (pollfd->revents & LWS_POLLIN) {
		if (user_callback_handle_rxflow(wsi->protocol->callback,
						wsi, LWS_CALLBACK_RAW_RX_FILE,
						wsi->user_space, NULL, 0)) {
			lwsl_debug("raw rx callback closed it\n");
			return LWS_HPI_RET_PLEASE_CLOSE_ME;
		}
	}

	if (pollfd->revents & LWS_POLLHUP)
		return LWS_HPI_RET_PLEASE_CLOSE_ME;

	return LWS_HPI_RET_HANDLED;
}

#if !defined(LWS_NO_SERVER)
static int
rops_adoption_bind_raw_file(struct lws *wsi, int type, const char *vh_prot_name)
{
	/* no socket or http: it can only be a raw file */
	if ((type & LWS_ADOPT_HTTP) || (type & LWS_ADOPT_SOCKET) ||
	    (type & _LWS_ADOPT_FINISH))
		return 0; /* no match */

	lws_role_transition(wsi, 0, LRS_ESTABLISHED, &role_ops_raw_file);

	if (!vh_prot_name)
		wsi->protocol = &wsi->vhost->protocols[
					wsi->vhost->default_protocol_index];

	return 1; /* bound */
}
#endif

struct lws_role_ops role_ops_raw_file = {
	/* role name */			"raw-file",
	/* alpn id */			NULL,
	/* check_upgrades */		NULL,
	/* init_context */		NULL,
	/* init_vhost */		NULL,
	/* destroy_vhost */		NULL,
	/* periodic_checks */		NULL,
	/* service_flag_pending */	NULL,
	/* handle_POLLIN */		rops_handle_POLLIN_raw_file,
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
	/* adoption_bind */		rops_adoption_bind_raw_file,
#else
					NULL,
#endif
	/* client_bind */		NULL,
	/* writeable cb clnt, srv */	{ LWS_CALLBACK_RAW_WRITEABLE_FILE, 0 },
	/* close cb clnt, srv */	{ LWS_CALLBACK_RAW_CLOSE_FILE, 0 },
	/* protocol_bind cb c, srv */	{ LWS_CALLBACK_RAW_FILE_BIND_PROTOCOL,
					  LWS_CALLBACK_RAW_FILE_BIND_PROTOCOL },
	/* protocol_unbind cb c, srv */	{ LWS_CALLBACK_RAW_FILE_DROP_PROTOCOL,
					  LWS_CALLBACK_RAW_FILE_DROP_PROTOCOL },
	/* file_handle */		1,
};
