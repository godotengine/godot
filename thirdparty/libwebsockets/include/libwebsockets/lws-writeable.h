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
 *
 * included from libwebsockets.h
 */

/** \defgroup callback-when-writeable Callback when writeable
 *
 * ##Callback When Writeable
 *
 * lws can only write data on a connection when it is able to accept more
 * data without blocking.
 *
 * So a basic requirement is we should only use the lws_write() apis when the
 * connection we want to write on says that he can accept more data.
 *
 * When lws cannot complete your send at the time, it will buffer the data
 * and send it in the background, suppressing any further WRITEABLE callbacks
 * on that connection until it completes.  So it is important to write new
 * things in a new writeable callback.
 *
 * These apis reflect the various ways we can indicate we would like to be
 * called back when one or more connections is writeable.
 */
///@{

/**
 * lws_callback_on_writable() - Request a callback when this socket
 *					 becomes able to be written to without
 *					 blocking
 *
 * \param wsi:	Websocket connection instance to get callback for
 *
 * - Which:  only this wsi
 * - When:   when the individual connection becomes writeable
 * - What: LWS_CALLBACK_*_WRITEABLE
 */
LWS_VISIBLE LWS_EXTERN int
lws_callback_on_writable(struct lws *wsi);

/**
 * lws_callback_on_writable_all_protocol() - Request a callback for all
 *			connections using the given protocol when it
 *			becomes possible to write to each socket without
 *			blocking in turn.
 *
 * \param context:	lws_context
 * \param protocol:	Protocol whose connections will get callbacks
 *
 * - Which:  connections using this protocol on ANY VHOST
 * - When:   when the individual connection becomes writeable
 * - What: LWS_CALLBACK_*_WRITEABLE
 */
LWS_VISIBLE LWS_EXTERN int
lws_callback_on_writable_all_protocol(const struct lws_context *context,
				      const struct lws_protocols *protocol);

/**
 * lws_callback_on_writable_all_protocol_vhost() - Request a callback for
 *			all connections on same vhost using the given protocol
 *			when it becomes possible to write to each socket without
 *			blocking in turn.
 *
 * \param vhost:	Only consider connections on this lws_vhost
 * \param protocol:	Protocol whose connections will get callbacks
 *
 * - Which:  connections using this protocol on GIVEN VHOST ONLY
 * - When:   when the individual connection becomes writeable
 * - What: LWS_CALLBACK_*_WRITEABLE
 */
LWS_VISIBLE LWS_EXTERN int
lws_callback_on_writable_all_protocol_vhost(const struct lws_vhost *vhost,
				      const struct lws_protocols *protocol);

/**
 * lws_callback_all_protocol() - Callback all connections using
 *				the given protocol with the given reason
 *
 * \param context:	lws_context
 * \param protocol:	Protocol whose connections will get callbacks
 * \param reason:	Callback reason index
 *
 * - Which:  connections using this protocol on ALL VHOSTS
 * - When:   before returning
 * - What:   reason
 *
 * This isn't normally what you want... normally any update of connection-
 * specific information can wait until a network-related callback like rx,
 * writable, or close.
 */
LWS_VISIBLE LWS_EXTERN int
lws_callback_all_protocol(struct lws_context *context,
			  const struct lws_protocols *protocol, int reason);

/**
 * lws_callback_all_protocol_vhost() - Callback all connections using
 *			the given protocol with the given reason.  This is
 *			deprecated since v2.4: use lws_callback_all_protocol_vhost_args
 *
 * \param vh:		Vhost whose connections will get callbacks
 * \param protocol:	Which protocol to match.  NULL means all.
 * \param reason:	Callback reason index
 *
 * - Which:  connections using this protocol on GIVEN VHOST ONLY
 * - When:   now
 * - What:   reason
 */
LWS_VISIBLE LWS_EXTERN int
lws_callback_all_protocol_vhost(struct lws_vhost *vh,
				const struct lws_protocols *protocol,
				int reason)
LWS_WARN_DEPRECATED;

/**
 * lws_callback_all_protocol_vhost_args() - Callback all connections using
 *			the given protocol with the given reason and args
 *
 * \param vh:		Vhost whose connections will get callbacks
 * \param protocol:	Which protocol to match.  NULL means all.
 * \param reason:	Callback reason index
 * \param argp:		Callback "in" parameter
 * \param len:		Callback "len" parameter
 *
 * - Which:  connections using this protocol on GIVEN VHOST ONLY
 * - When:   now
 * - What:   reason
 */
LWS_VISIBLE int
lws_callback_all_protocol_vhost_args(struct lws_vhost *vh,
				     const struct lws_protocols *protocol,
				     int reason, void *argp, size_t len);

/**
 * lws_callback_vhost_protocols() - Callback all protocols enabled on a vhost
 *					with the given reason
 *
 * \param wsi:	wsi whose vhost will get callbacks
 * \param reason:	Callback reason index
 * \param in:		in argument to callback
 * \param len:	len argument to callback
 *
 * - Which:  connections using this protocol on same VHOST as wsi ONLY
 * - When:   now
 * - What:   reason
 *
 * This is deprecated since v2.5, use lws_callback_vhost_protocols_vhost()
 * which takes the pointer to the vhost directly without using or needing the
 * wsi.
 */
LWS_VISIBLE LWS_EXTERN int
lws_callback_vhost_protocols(struct lws *wsi, int reason, void *in, int len)
LWS_WARN_DEPRECATED;

/**
 * lws_callback_vhost_protocols_vhost() - Callback all protocols enabled on a vhost
 *					with the given reason
 *
 * \param vh:		vhost that will get callbacks
 * \param reason:	Callback reason index
 * \param in:		in argument to callback
 * \param len:		len argument to callback
 *
 * - Which:  connections using this protocol on same VHOST as wsi ONLY
 * - When:   now
 * - What:   reason
 */
LWS_VISIBLE LWS_EXTERN int
lws_callback_vhost_protocols_vhost(struct lws_vhost *vh, int reason, void *in,
				   size_t len);

LWS_VISIBLE LWS_EXTERN int
lws_callback_http_dummy(struct lws *wsi, enum lws_callback_reasons reason,
			void *user, void *in, size_t len);

/**
 * lws_get_socket_fd() - returns the socket file descriptor
 *
 * This is needed to use sendto() on UDP raw sockets
 *
 * \param wsi:	Websocket connection instance
 */
LWS_VISIBLE LWS_EXTERN lws_sockfd_type
lws_get_socket_fd(struct lws *wsi);

/**
 * lws_get_peer_write_allowance() - get the amount of data writeable to peer
 * 					if known
 *
 * \param wsi:	Websocket connection instance
 *
 * if the protocol does not have any guidance, returns -1.  Currently only
 * http2 connections get send window information from this API.  But your code
 * should use it so it can work properly with any protocol.
 *
 * If nonzero return is the amount of payload data the peer or intermediary has
 * reported it has buffer space for.  That has NO relationship with the amount
 * of buffer space your OS can accept on this connection for a write action.
 *
 * This number represents the maximum you could send to the peer or intermediary
 * on this connection right now without the protocol complaining.
 *
 * lws manages accounting for send window updates and payload writes
 * automatically, so this number reflects the situation at the peer or
 * intermediary dynamically.
 */
LWS_VISIBLE LWS_EXTERN lws_fileofs_t
lws_get_peer_write_allowance(struct lws *wsi);
///@}
