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

/*! \defgroup timeout Connection timeouts

    APIs related to setting connection timeouts
*/
//@{

/*
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */
enum pending_timeout {
	NO_PENDING_TIMEOUT					=  0,
	PENDING_TIMEOUT_AWAITING_PROXY_RESPONSE			=  1,
	PENDING_TIMEOUT_AWAITING_CONNECT_RESPONSE		=  2,
	PENDING_TIMEOUT_ESTABLISH_WITH_SERVER			=  3,
	PENDING_TIMEOUT_AWAITING_SERVER_RESPONSE		=  4,
	PENDING_TIMEOUT_AWAITING_PING				=  5,
	PENDING_TIMEOUT_CLOSE_ACK				=  6,
	PENDING_TIMEOUT_UNUSED1					=  7,
	PENDING_TIMEOUT_SENT_CLIENT_HANDSHAKE			=  8,
	PENDING_TIMEOUT_SSL_ACCEPT				=  9,
	PENDING_TIMEOUT_HTTP_CONTENT				= 10,
	PENDING_TIMEOUT_AWAITING_CLIENT_HS_SEND			= 11,
	PENDING_FLUSH_STORED_SEND_BEFORE_CLOSE			= 12,
	PENDING_TIMEOUT_SHUTDOWN_FLUSH				= 13,
	PENDING_TIMEOUT_CGI					= 14,
	PENDING_TIMEOUT_HTTP_KEEPALIVE_IDLE			= 15,
	PENDING_TIMEOUT_WS_PONG_CHECK_SEND_PING			= 16,
	PENDING_TIMEOUT_WS_PONG_CHECK_GET_PONG			= 17,
	PENDING_TIMEOUT_CLIENT_ISSUE_PAYLOAD			= 18,
	PENDING_TIMEOUT_AWAITING_SOCKS_GREETING_REPLY	        = 19,
	PENDING_TIMEOUT_AWAITING_SOCKS_CONNECT_REPLY		= 20,
	PENDING_TIMEOUT_AWAITING_SOCKS_AUTH_REPLY		= 21,
	PENDING_TIMEOUT_KILLED_BY_SSL_INFO			= 22,
	PENDING_TIMEOUT_KILLED_BY_PARENT			= 23,
	PENDING_TIMEOUT_CLOSE_SEND				= 24,
	PENDING_TIMEOUT_HOLDING_AH				= 25,
	PENDING_TIMEOUT_UDP_IDLE				= 26,
	PENDING_TIMEOUT_CLIENT_CONN_IDLE			= 27,
	PENDING_TIMEOUT_LAGGING					= 28,
	PENDING_TIMEOUT_THREADPOOL				= 29,
	PENDING_TIMEOUT_THREADPOOL_TASK				= 30,
	PENDING_TIMEOUT_KILLED_BY_PROXY_CLIENT_CLOSE		= 31,

	/****** add new things just above ---^ ******/

	PENDING_TIMEOUT_USER_REASON_BASE			= 1000
};

/**
 * lws_time_in_microseconds() - Returns the unix time in microseconds
 */
LWS_VISIBLE LWS_EXTERN uint64_t
lws_time_in_microseconds(void);

#define LWS_TO_KILL_ASYNC -1
/**< If LWS_TO_KILL_ASYNC is given as the timeout sec in a lws_set_timeout()
 * call, then the connection is marked to be killed at the next timeout
 * check.  This is how you should force-close the wsi being serviced if
 * you are doing it outside the callback (where you should close by nonzero
 * return).
 */
#define LWS_TO_KILL_SYNC -2
/**< If LWS_TO_KILL_SYNC is given as the timeout sec in a lws_set_timeout()
 * call, then the connection is closed before returning (which may delete
 * the wsi).  This should only be used where the wsi being closed is not the
 * wsi currently being serviced.
 */
/**
 * lws_set_timeout() - marks the wsi as subject to a timeout
 *
 * You will not need this unless you are doing something special
 *
 * \param wsi:	Websocket connection instance
 * \param reason:	timeout reason
 * \param secs:	how many seconds.  You may set to LWS_TO_KILL_ASYNC to
 *		force the connection to timeout at the next opportunity, or
 *		LWS_TO_KILL_SYNC to close it synchronously if you know the
 *		wsi is not the one currently being serviced.
 */
LWS_VISIBLE LWS_EXTERN void
lws_set_timeout(struct lws *wsi, enum pending_timeout reason, int secs);

#define LWS_SET_TIMER_USEC_CANCEL ((lws_usec_t)-1ll)
#define LWS_USEC_PER_SEC (1000000ll)

/**
 * lws_set_timer_usecs() - schedules a callback on the wsi in the future
 *
 * \param wsi:	Websocket connection instance
 * \param usecs:  LWS_SET_TIMER_USEC_CANCEL removes any existing scheduled
 *		  callback, otherwise number of microseconds in the future
 *		  the callback will occur at.
 *
 * NOTE: event loop support for this:
 *
 *  default poll() loop:   yes
 *  libuv event loop:      yes
 *  libev:    not implemented (patch welcome)
 *  libevent: not implemented (patch welcome)
 *
 * After the deadline expires, the wsi will get a callback of type
 * LWS_CALLBACK_TIMER and the timer is exhausted.  The deadline may be
 * continuously deferred by further calls to lws_set_timer_usecs() with a later
 * deadline, or cancelled by lws_set_timer_usecs(wsi, -1).
 *
 * If the timer should repeat, lws_set_timer_usecs() must be called again from
 * LWS_CALLBACK_TIMER.
 *
 * Accuracy depends on the platform and the load on the event loop or system...
 * all that's guaranteed is the callback will come after the requested wait
 * period.
 */
LWS_VISIBLE LWS_EXTERN void
lws_set_timer_usecs(struct lws *wsi, lws_usec_t usecs);

/*
 * lws_timed_callback_vh_protocol() - calls back a protocol on a vhost after
 * 					the specified delay
 *
 * \param vh:	 the vhost to call back
 * \param protocol: the protocol to call back
 * \param reason: callback reason
 * \param secs:	how many seconds in the future to do the callback.  Set to
 *		-1 to cancel the timer callback.
 *
 * Callback the specified protocol with a fake wsi pointing to the specified
 * vhost and protocol, with the specified reason, at the specified time in the
 * future.
 *
 * Returns 0 if OK.
 *
 * In the multithreaded service case, the callback will occur in the same
 * service thread context as the call to this api that requested it.  If it is
 * called from a non-service thread, tsi 0 will handle it.
 */
LWS_VISIBLE LWS_EXTERN int
lws_timed_callback_vh_protocol(struct lws_vhost *vh,
			       const struct lws_protocols *prot,
			       int reason, int secs);
///@}
