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
rops_handle_POLLIN_listen(struct lws_context_per_thread *pt, struct lws *wsi,
			  struct lws_pollfd *pollfd)
{
	struct lws_context *context = wsi->context;
	lws_sockfd_type accept_fd = LWS_SOCK_INVALID;
	lws_sock_file_fd_type fd;
	int opts = LWS_ADOPT_SOCKET | LWS_ADOPT_ALLOW_SSL;
	struct sockaddr_storage cli_addr;
	socklen_t clilen;

	/* if our vhost is going down, ignore it */

	if (wsi->vhost->being_destroyed)
		return LWS_HPI_RET_HANDLED;

	/* pollin means a client has connected to us then
	 *
	 * pollout is a hack on esp32 for background accepts signalling
	 * they completed
	 */

	do {
		struct lws *cwsi;

		if (!(pollfd->revents & (LWS_POLLIN | LWS_POLLOUT)) ||
		    !(pollfd->events & LWS_POLLIN))
			break;

#if defined(LWS_WITH_TLS)
		/*
		 * can we really accept it, with regards to SSL limit?
		 * another vhost may also have had POLLIN on his
		 * listener this round and used it up already
		 */
		if (wsi->vhost->tls.use_ssl &&
		    context->simultaneous_ssl_restriction &&
		    context->simultaneous_ssl ==
				  context->simultaneous_ssl_restriction)
			/*
			 * no... ignore it, he won't come again until
			 * we are below the simultaneous_ssl_restriction
			 * limit and POLLIN is enabled on him again
			 */
			break;
#endif
		/* listen socket got an unencrypted connection... */

		clilen = sizeof(cli_addr);
		lws_latency_pre(context, wsi);

		/*
		 * We cannot identify the peer who is in the listen
		 * socket connect queue before we accept it; even if
		 * we could, not accepting it due to PEER_LIMITS would
		 * block the connect queue for other legit peers.
		 */

		accept_fd = accept((int)pollfd->fd,
				   (struct sockaddr *)&cli_addr, &clilen);
		lws_latency(context, wsi, "listener accept",
			    (int)accept_fd, accept_fd != LWS_SOCK_INVALID);
		if (accept_fd == LWS_SOCK_INVALID) {
			if (LWS_ERRNO == LWS_EAGAIN ||
			    LWS_ERRNO == LWS_EWOULDBLOCK) {
				break;
			}
			lwsl_err("accept: %s\n", strerror(LWS_ERRNO));
			return LWS_HPI_RET_HANDLED;
		}

		lws_plat_set_socket_options(wsi->vhost, accept_fd, 0);

#if defined(LWS_WITH_IPV6)
		lwsl_debug("accepted new conn port %u on fd=%d\n",
			((cli_addr.ss_family == AF_INET6) ?
			ntohs(((struct sockaddr_in6 *) &cli_addr)->sin6_port) :
			ntohs(((struct sockaddr_in *) &cli_addr)->sin_port)),
			accept_fd);
#else
		lwsl_debug("accepted new conn port %u on fd=%d\n",
			   ntohs(((struct sockaddr_in *) &cli_addr)->sin_port),
			   accept_fd);
#endif

		/*
		 * look at who we connected to and give user code a
		 * chance to reject based on client IP.  There's no
		 * protocol selected yet so we issue this to
		 * protocols[0]
		 */
		if ((wsi->vhost->protocols[0].callback)(wsi,
				LWS_CALLBACK_FILTER_NETWORK_CONNECTION,
				NULL,
				(void *)(lws_intptr_t)accept_fd, 0)) {
			lwsl_debug("Callback denied net connection\n");
			compatible_close(accept_fd);
			break;
		}

		if (!(wsi->vhost->options & LWS_SERVER_OPTION_ONLY_RAW))
			opts |= LWS_ADOPT_HTTP;
		else
			opts = LWS_ADOPT_SOCKET;

		fd.sockfd = accept_fd;
		cwsi = lws_adopt_descriptor_vhost(wsi->vhost, opts, fd,
						  NULL, NULL);
		if (!cwsi) {
			lwsl_err("%s: lws_adopt_descriptor_vhost failed\n",
					__func__);
			/* already closed cleanly as necessary */
			return LWS_HPI_RET_WSI_ALREADY_DIED;
		}

		if (lws_server_socket_service_ssl(cwsi, accept_fd)) {
			lws_close_free_wsi(cwsi, LWS_CLOSE_STATUS_NOSTATUS,
					   "listen svc fail");
			return LWS_HPI_RET_WSI_ALREADY_DIED;
		}

		lwsl_info("%s: new wsi %p: wsistate 0x%x, role_ops %s\n",
			    __func__, cwsi, cwsi->wsistate, cwsi->role_ops->name);

	} while (pt->fds_count < context->fd_limit_per_thread - 1 &&
		 wsi->position_in_fds_table != LWS_NO_FDS_POS &&
		 lws_poll_listen_fd(&pt->fds[wsi->position_in_fds_table]) > 0);

	return LWS_HPI_RET_HANDLED;
}

int rops_handle_POLLOUT_listen(struct lws *wsi)
{
	return LWS_HP_RET_USER_SERVICE;
}

struct lws_role_ops role_ops_listen = {
	/* role name */			"listen",
	/* alpn id */			NULL,
	/* check_upgrades */		NULL,
	/* init_context */		NULL,
	/* init_vhost */		NULL,
	/* destroy_vhost */		NULL,
	/* periodic_checks */		NULL,
	/* service_flag_pending */	NULL,
	/* handle_POLLIN */		rops_handle_POLLIN_listen,
	/* handle_POLLOUT */		rops_handle_POLLOUT_listen,
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
	/* adoption_bind */		NULL,
	/* client_bind */		NULL,
	/* writeable cb clnt, srv */	{ 0, 0 },
	/* close cb clnt, srv */	{ 0, 0 },
	/* protocol_bind_cb c,s */	{ 0, 0 },
	/* protocol_unbind_cb c,s */	{ 0, 0 },
	/* file_handle */		0,
};
