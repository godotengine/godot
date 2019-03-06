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

/** \defgroup wsstatus Websocket status APIs
 * ##Websocket connection status APIs
 *
 * These provide information about ws connection or message status
 */
///@{
/**
 * lws_send_pipe_choked() - tests if socket is writable or not
 * \param wsi: lws connection
 *
 * Allows you to check if you can write more on the socket
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_send_pipe_choked(struct lws *wsi);

/**
 * lws_is_final_fragment() - tests if last part of ws message
 *
 * \param wsi: lws connection
 */
LWS_VISIBLE LWS_EXTERN int
lws_is_final_fragment(struct lws *wsi);

/**
 * lws_is_first_fragment() - tests if first part of ws message
 *
 * \param wsi: lws connection
 */
LWS_VISIBLE LWS_EXTERN int
lws_is_first_fragment(struct lws *wsi);

/**
 * lws_get_reserved_bits() - access reserved bits of ws frame
 * \param wsi: lws connection
 */
LWS_VISIBLE LWS_EXTERN unsigned char
lws_get_reserved_bits(struct lws *wsi);

/**
 * lws_partial_buffered() - find out if lws buffered the last write
 * \param wsi:	websocket connection to check
 *
 * Returns 1 if you cannot use lws_write because the last
 * write on this connection is still buffered, and can't be cleared without
 * returning to the service loop and waiting for the connection to be
 * writeable again.
 *
 * If you will try to do >1 lws_write call inside a single
 * WRITEABLE callback, you must check this after every write and bail if
 * set, ask for a new writeable callback and continue writing from there.
 *
 * This is never set at the start of a writeable callback, but any write
 * may set it.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_partial_buffered(struct lws *wsi);

/**
 * lws_frame_is_binary(): true if the current frame was sent in binary mode
 *
 * \param wsi: the connection we are inquiring about
 *
 * This is intended to be called from the LWS_CALLBACK_RECEIVE callback if
 * it's interested to see if the frame it's dealing with was sent in binary
 * mode.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_frame_is_binary(struct lws *wsi);
///@}
