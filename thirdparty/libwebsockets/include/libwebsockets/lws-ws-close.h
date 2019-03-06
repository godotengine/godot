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

/*! \defgroup wsclose Websocket Close
 *
 * ##Websocket close frame control
 *
 * When we close a ws connection, we can send a reason code and a short
 * UTF-8 description back with the close packet.
 */
///@{

/*
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */
/** enum lws_close_status - RFC6455 close status codes */
enum lws_close_status {
	LWS_CLOSE_STATUS_NOSTATUS				=    0,
	LWS_CLOSE_STATUS_NORMAL					= 1000,
	/**< 1000 indicates a normal closure, meaning that the purpose for
      which the connection was established has been fulfilled. */
	LWS_CLOSE_STATUS_GOINGAWAY				= 1001,
	/**< 1001 indicates that an endpoint is "going away", such as a server
      going down or a browser having navigated away from a page. */
	LWS_CLOSE_STATUS_PROTOCOL_ERR				= 1002,
	/**< 1002 indicates that an endpoint is terminating the connection due
      to a protocol error. */
	LWS_CLOSE_STATUS_UNACCEPTABLE_OPCODE			= 1003,
	/**< 1003 indicates that an endpoint is terminating the connection
      because it has received a type of data it cannot accept (e.g., an
      endpoint that understands only text data MAY send this if it
      receives a binary message). */
	LWS_CLOSE_STATUS_RESERVED				= 1004,
	/**< Reserved.  The specific meaning might be defined in the future. */
	LWS_CLOSE_STATUS_NO_STATUS				= 1005,
	/**< 1005 is a reserved value and MUST NOT be set as a status code in a
      Close control frame by an endpoint.  It is designated for use in
      applications expecting a status code to indicate that no status
      code was actually present. */
	LWS_CLOSE_STATUS_ABNORMAL_CLOSE				= 1006,
	/**< 1006 is a reserved value and MUST NOT be set as a status code in a
      Close control frame by an endpoint.  It is designated for use in
      applications expecting a status code to indicate that the
      connection was closed abnormally, e.g., without sending or
      receiving a Close control frame. */
	LWS_CLOSE_STATUS_INVALID_PAYLOAD			= 1007,
	/**< 1007 indicates that an endpoint is terminating the connection
      because it has received data within a message that was not
      consistent with the type of the message (e.g., non-UTF-8 [RFC3629]
      data within a text message). */
	LWS_CLOSE_STATUS_POLICY_VIOLATION			= 1008,
	/**< 1008 indicates that an endpoint is terminating the connection
      because it has received a message that violates its policy.  This
      is a generic status code that can be returned when there is no
      other more suitable status code (e.g., 1003 or 1009) or if there
      is a need to hide specific details about the policy. */
	LWS_CLOSE_STATUS_MESSAGE_TOO_LARGE			= 1009,
	/**< 1009 indicates that an endpoint is terminating the connection
      because it has received a message that is too big for it to
      process. */
	LWS_CLOSE_STATUS_EXTENSION_REQUIRED			= 1010,
	/**< 1010 indicates that an endpoint (client) is terminating the
      connection because it has expected the server to negotiate one or
      more extension, but the server didn't return them in the response
      message of the WebSocket handshake.  The list of extensions that
      are needed SHOULD appear in the /reason/ part of the Close frame.
      Note that this status code is not used by the server, because it
      can fail the WebSocket handshake instead */
	LWS_CLOSE_STATUS_UNEXPECTED_CONDITION			= 1011,
	/**< 1011 indicates that a server is terminating the connection because
      it encountered an unexpected condition that prevented it from
      fulfilling the request. */
	LWS_CLOSE_STATUS_TLS_FAILURE				= 1015,
	/**< 1015 is a reserved value and MUST NOT be set as a status code in a
      Close control frame by an endpoint.  It is designated for use in
      applications expecting a status code to indicate that the
      connection was closed due to a failure to perform a TLS handshake
      (e.g., the server certificate can't be verified). */

	LWS_CLOSE_STATUS_CLIENT_TRANSACTION_DONE		= 2000,

	/****** add new things just above ---^ ******/

	LWS_CLOSE_STATUS_NOSTATUS_CONTEXT_DESTROY		= 9999,
};

/**
 * lws_close_reason - Set reason and aux data to send with Close packet
 *		If you are going to return nonzero from the callback
 *		requesting the connection to close, you can optionally
 *		call this to set the reason the peer will be told if
 *		possible.
 *
 * \param wsi:	The websocket connection to set the close reason on
 * \param status:	A valid close status from websocket standard
 * \param buf:	NULL or buffer containing up to 124 bytes of auxiliary data
 * \param len:	Length of data in \param buf to send
 */
LWS_VISIBLE LWS_EXTERN void
lws_close_reason(struct lws *wsi, enum lws_close_status status,
		 unsigned char *buf, size_t len);

///@}
