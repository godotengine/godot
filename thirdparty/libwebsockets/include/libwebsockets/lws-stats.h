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

/*
 * Stats are all uint64_t numbers that start at 0.
 * Index names here have the convention
 *
 *  _C_ counter
 *  _B_ byte count
 *  _MS_ millisecond count
 */

enum {
	LWSSTATS_C_CONNECTIONS, /**< count incoming connections */
	LWSSTATS_C_API_CLOSE, /**< count calls to close api */
	LWSSTATS_C_API_READ, /**< count calls to read from socket api */
	LWSSTATS_C_API_LWS_WRITE, /**< count calls to lws_write API */
	LWSSTATS_C_API_WRITE, /**< count calls to write API */
	LWSSTATS_C_WRITE_PARTIALS, /**< count of partial writes */
	LWSSTATS_C_WRITEABLE_CB_REQ, /**< count of writable callback requests */
	LWSSTATS_C_WRITEABLE_CB_EFF_REQ, /**< count of effective writable callback requests */
	LWSSTATS_C_WRITEABLE_CB, /**< count of writable callbacks */
	LWSSTATS_C_SSL_CONNECTIONS_FAILED, /**< count of failed SSL connections */
	LWSSTATS_C_SSL_CONNECTIONS_ACCEPTED, /**< count of accepted SSL connections */
	LWSSTATS_C_SSL_CONNECTIONS_ACCEPT_SPIN, /**< count of SSL_accept() attempts */
	LWSSTATS_C_SSL_CONNS_HAD_RX, /**< count of accepted SSL conns that have had some RX */
	LWSSTATS_C_TIMEOUTS, /**< count of timed-out connections */
	LWSSTATS_C_SERVICE_ENTRY, /**< count of entries to lws service loop */
	LWSSTATS_B_READ, /**< aggregate bytes read */
	LWSSTATS_B_WRITE, /**< aggregate bytes written */
	LWSSTATS_B_PARTIALS_ACCEPTED_PARTS, /**< aggreate of size of accepted write data from new partials */
	LWSSTATS_MS_SSL_CONNECTIONS_ACCEPTED_DELAY, /**< aggregate delay in accepting connection */
	LWSSTATS_MS_WRITABLE_DELAY, /**< aggregate delay between asking for writable and getting cb */
	LWSSTATS_MS_WORST_WRITABLE_DELAY, /**< single worst delay between asking for writable and getting cb */
	LWSSTATS_MS_SSL_RX_DELAY, /**< aggregate delay between ssl accept complete and first RX */
	LWSSTATS_C_PEER_LIMIT_AH_DENIED, /**< number of times we would have given an ah but for the peer limit */
	LWSSTATS_C_PEER_LIMIT_WSI_DENIED, /**< number of times we would have given a wsi but for the peer limit */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
	LWSSTATS_SIZE
};

#if defined(LWS_WITH_STATS)

LWS_VISIBLE LWS_EXTERN uint64_t
lws_stats_get(struct lws_context *context, int index);
LWS_VISIBLE LWS_EXTERN void
lws_stats_log_dump(struct lws_context *context);
#else
static LWS_INLINE uint64_t
lws_stats_get(struct lws_context *context, int index) { (void)context; (void)index;  return 0; }
static LWS_INLINE void
lws_stats_log_dump(struct lws_context *context) { (void)context; }
#endif
