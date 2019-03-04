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

/** \defgroup service Built-in service loop entry
 *
 * ##Built-in service loop entry
 *
 * If you're not using libev / libuv, these apis are needed to enter the poll()
 * wait in lws and service any connections with pending events.
 */
///@{

/**
 * lws_service() - Service any pending websocket activity
 * \param context:	Websocket context
 * \param timeout_ms:	Timeout for poll; 0 means return immediately if nothing needed
 *		service otherwise block and service immediately, returning
 *		after the timeout if nothing needed service.
 *
 *	This function deals with any pending websocket traffic, for three
 *	kinds of event.  It handles these events on both server and client
 *	types of connection the same.
 *
 *	1) Accept new connections to our context's server
 *
 *	2) Call the receive callback for incoming frame data received by
 *	    server or client connections.
 *
 *	You need to call this service function periodically to all the above
 *	functions to happen; if your application is single-threaded you can
 *	just call it in your main event loop.
 *
 *	Alternatively you can fork a new process that asynchronously handles
 *	calling this service in a loop.  In that case you are happy if this
 *	call blocks your thread until it needs to take care of something and
 *	would call it with a large nonzero timeout.  Your loop then takes no
 *	CPU while there is nothing happening.
 *
 *	If you are calling it in a single-threaded app, you don't want it to
 *	wait around blocking other things in your loop from happening, so you
 *	would call it with a timeout_ms of 0, so it returns immediately if
 *	nothing is pending, or as soon as it services whatever was pending.
 */
LWS_VISIBLE LWS_EXTERN int
lws_service(struct lws_context *context, int timeout_ms);

/**
 * lws_service_tsi() - Service any pending websocket activity
 *
 * \param context:	Websocket context
 * \param timeout_ms:	Timeout for poll; 0 means return immediately if nothing needed
 *		service otherwise block and service immediately, returning
 *		after the timeout if nothing needed service.
 * \param tsi:		Thread service index, starting at 0
 *
 * Same as lws_service(), but for a specific thread service index.  Only needed
 * if you are spawning multiple service threads.
 */
LWS_VISIBLE LWS_EXTERN int
lws_service_tsi(struct lws_context *context, int timeout_ms, int tsi);

/**
 * lws_cancel_service_pt() - Cancel servicing of pending socket activity
 *				on one thread
 * \param wsi:	Cancel service on the thread this wsi is serviced by
 *
 * Same as lws_cancel_service(), but targets a single service thread, the one
 * the wsi belongs to.  You probably want to use lws_cancel_service() instead.
 */
LWS_VISIBLE LWS_EXTERN void
lws_cancel_service_pt(struct lws *wsi);

/**
 * lws_cancel_service() - Cancel wait for new pending socket activity
 * \param context:	Websocket context
 *
 * This function creates an immediate "synchronous interrupt" to the lws poll()
 * wait or event loop.  As soon as possible in the serialzed service sequencing,
 * a LWS_CALLBACK_EVENT_WAIT_CANCELLED callback is sent to every protocol on
 * every vhost.
 *
 * lws_cancel_service() may be called from another thread while the context
 * exists, and its effect will be immediately serialized.
 */
LWS_VISIBLE LWS_EXTERN void
lws_cancel_service(struct lws_context *context);

/**
 * lws_service_fd() - Service polled socket with something waiting
 * \param context:	Websocket context
 * \param pollfd:	The pollfd entry describing the socket fd and which events
 *		happened, or NULL to tell lws to do only timeout servicing.
 *
 * This function takes a pollfd that has POLLIN or POLLOUT activity and
 * services it according to the state of the associated
 * struct lws.
 *
 * The one call deals with all "service" that might happen on a socket
 * including listen accepts, http files as well as websocket protocol.
 *
 * If a pollfd says it has something, you can just pass it to
 * lws_service_fd() whether it is a socket handled by lws or not.
 * If it sees it is a lws socket, the traffic will be handled and
 * pollfd->revents will be zeroed now.
 *
 * If the socket is foreign to lws, it leaves revents alone.  So you can
 * see if you should service yourself by checking the pollfd revents
 * after letting lws try to service it.
 *
 * You should also call this with pollfd = NULL to just allow the
 * once-per-second global timeout checks; if less than a second since the last
 * check it returns immediately then.
 */
LWS_VISIBLE LWS_EXTERN int
lws_service_fd(struct lws_context *context, struct lws_pollfd *pollfd);

/**
 * lws_service_fd_tsi() - Service polled socket in specific service thread
 * \param context:	Websocket context
 * \param pollfd:	The pollfd entry describing the socket fd and which events
 *		happened.
 * \param tsi: thread service index
 *
 * Same as lws_service_fd() but used with multiple service threads
 */
LWS_VISIBLE LWS_EXTERN int
lws_service_fd_tsi(struct lws_context *context, struct lws_pollfd *pollfd,
		   int tsi);

/**
 * lws_service_adjust_timeout() - Check for any connection needing forced service
 * \param context:	Websocket context
 * \param timeout_ms:	The original poll timeout value.  You can just set this
 *			to 1 if you don't really have a poll timeout.
 * \param tsi: thread service index
 *
 * Under some conditions connections may need service even though there is no
 * pending network action on them, this is "forced service".  For default
 * poll() and libuv / libev, the library takes care of calling this and
 * dealing with it for you.  But for external poll() integration, you need
 * access to the apis.
 *
 * If anybody needs "forced service", returned timeout is zero.  In that case,
 * you can call lws_service_tsi() with a timeout of -1 to only service
 * guys who need forced service.
 */
LWS_VISIBLE LWS_EXTERN int
lws_service_adjust_timeout(struct lws_context *context, int timeout_ms, int tsi);

/* Backwards compatibility */
#define lws_plat_service_tsi lws_service_tsi

LWS_VISIBLE LWS_EXTERN int
lws_handle_POLLOUT_event(struct lws *wsi, struct lws_pollfd *pollfd);

///@}

/*! \defgroup uv libuv helpers
 *
 * ##libuv helpers
 *
 * APIs specific to libuv event loop itegration
 */
///@{
#ifdef LWS_WITH_LIBUV
/*
 * Any direct libuv allocations in lws protocol handlers must participate in the
 * lws reference counting scheme.  Two apis are provided:
 *
 * - lws_libuv_static_refcount_add(handle, context) to mark the handle with
 *  a pointer to the context and increment the global uv object counter
 *
 * - lws_libuv_static_refcount_del() which should be used as the close callback
 *   for your own libuv objects declared in the protocol scope.
 *
 * Using the apis allows lws to detach itself from a libuv loop completely
 * cleanly and at the moment all of its libuv objects have completed close.
 */

LWS_VISIBLE LWS_EXTERN uv_loop_t *
lws_uv_getloop(struct lws_context *context, int tsi);

LWS_VISIBLE LWS_EXTERN void
lws_libuv_static_refcount_add(uv_handle_t *, struct lws_context *context);

LWS_VISIBLE LWS_EXTERN void
lws_libuv_static_refcount_del(uv_handle_t *);

#endif /* LWS_WITH_LIBUV */

#if defined(LWS_WITH_ESP32)
#define lws_libuv_static_refcount_add(_a, _b)
#define lws_libuv_static_refcount_del NULL
#endif
///@}
