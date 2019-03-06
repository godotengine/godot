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

/*! \defgroup usercb User Callback
 *
 * ##User protocol callback
 *
 * The protocol callback is the primary way lws interacts with
 * user code.  For one of a list of a few dozen reasons the callback gets
 * called at some event to be handled.
 *
 * All of the events can be ignored, returning 0 is taken as "OK" and returning
 * nonzero in most cases indicates that the connection should be closed.
 */
///@{

struct lws_ssl_info {
	int where;
	int ret;
};

enum lws_cert_update_state {
	LWS_CUS_IDLE,
	LWS_CUS_STARTING,
	LWS_CUS_SUCCESS,
	LWS_CUS_FAILED,

	LWS_CUS_CREATE_KEYS,
	LWS_CUS_REG,
	LWS_CUS_AUTH,
	LWS_CUS_CHALLENGE,
	LWS_CUS_CREATE_REQ,
	LWS_CUS_REQ,
	LWS_CUS_CONFIRM,
	LWS_CUS_ISSUE,
};

enum {
	LWS_TLS_REQ_ELEMENT_COUNTRY,
	LWS_TLS_REQ_ELEMENT_STATE,
	LWS_TLS_REQ_ELEMENT_LOCALITY,
	LWS_TLS_REQ_ELEMENT_ORGANIZATION,
	LWS_TLS_REQ_ELEMENT_COMMON_NAME,
	LWS_TLS_REQ_ELEMENT_EMAIL,

	LWS_TLS_REQ_ELEMENT_COUNT,

	LWS_TLS_SET_DIR_URL = LWS_TLS_REQ_ELEMENT_COUNT,
	LWS_TLS_SET_AUTH_PATH,
	LWS_TLS_SET_CERT_PATH,
	LWS_TLS_SET_KEY_PATH,

	LWS_TLS_TOTAL_COUNT
};

struct lws_acme_cert_aging_args {
	struct lws_vhost *vh;
	const char *element_overrides[LWS_TLS_TOTAL_COUNT]; /* NULL = use pvo */
};

/*
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */
/** enum lws_callback_reasons - reason you're getting a protocol callback */
enum lws_callback_reasons {

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to wsi and protocol binding lifecycle -----
	 */

	LWS_CALLBACK_PROTOCOL_INIT				= 27,
	/**< One-time call per protocol, per-vhost using it, so it can
	 * do initial setup / allocations etc */

	LWS_CALLBACK_PROTOCOL_DESTROY				= 28,
	/**< One-time call per protocol, per-vhost using it, indicating
	 * this protocol won't get used at all after this callback, the
	 * vhost is getting destroyed.  Take the opportunity to
	 * deallocate everything that was allocated by the protocol. */

	LWS_CALLBACK_WSI_CREATE					= 29,
	/**< outermost (earliest) wsi create notification to protocols[0] */

	LWS_CALLBACK_WSI_DESTROY				= 30,
	/**< outermost (latest) wsi destroy notification to protocols[0] */


	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to Server TLS -----
	 */

	LWS_CALLBACK_OPENSSL_LOAD_EXTRA_CLIENT_VERIFY_CERTS	= 21,
	/**< if configured for
	 * including OpenSSL support, this callback allows your user code
	 * to perform extra SSL_CTX_load_verify_locations() or similar
	 * calls to direct OpenSSL where to find certificates the client
	 * can use to confirm the remote server identity.  user is the
	 * OpenSSL SSL_CTX* */

	LWS_CALLBACK_OPENSSL_LOAD_EXTRA_SERVER_VERIFY_CERTS	= 22,
	/**< if configured for
	 * including OpenSSL support, this callback allows your user code
	 * to load extra certificates into the server which allow it to
	 * verify the validity of certificates returned by clients.  user
	 * is the server's OpenSSL SSL_CTX* and in is the lws_vhost */

	LWS_CALLBACK_OPENSSL_PERFORM_CLIENT_CERT_VERIFICATION	= 23,
	/**< if the libwebsockets vhost was created with the option
	 * LWS_SERVER_OPTION_REQUIRE_VALID_OPENSSL_CLIENT_CERT, then this
	 * callback is generated during OpenSSL verification of the cert
	 * sent from the client.  It is sent to protocol[0] callback as
	 * no protocol has been negotiated on the connection yet.
	 * Notice that the libwebsockets context and wsi are both NULL
	 * during this callback.  See
	 *  http://www.openssl.org/docs/ssl/SSL_CTX_set_verify.html
	 * to understand more detail about the OpenSSL callback that
	 * generates this libwebsockets callback and the meanings of the
	 * arguments passed.  In this callback, user is the x509_ctx,
	 * in is the ssl pointer and len is preverify_ok
	 * Notice that this callback maintains libwebsocket return
	 * conventions, return 0 to mean the cert is OK or 1 to fail it.
	 * This also means that if you don't handle this callback then
	 * the default callback action of returning 0 allows the client
	 * certificates. */

	LWS_CALLBACK_OPENSSL_CONTEXT_REQUIRES_PRIVATE_KEY	= 37,
	/**< if configured for including OpenSSL support but no private key
	 * file has been specified (ssl_private_key_filepath is NULL), this is
	 * called to allow the user to set the private key directly via
	 * libopenssl and perform further operations if required; this might be
	 * useful in situations where the private key is not directly accessible
	 * by the OS, for example if it is stored on a smartcard.
	 * user is the server's OpenSSL SSL_CTX* */

	LWS_CALLBACK_SSL_INFO					= 67,
	/**< SSL connections only.  An event you registered an
	 * interest in at the vhost has occurred on a connection
	 * using the vhost.  in is a pointer to a
	 * struct lws_ssl_info containing information about the
	 * event*/

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to Client TLS -----
	 */

	LWS_CALLBACK_OPENSSL_PERFORM_SERVER_CERT_VERIFICATION = 58,
	/**< Similar to LWS_CALLBACK_OPENSSL_PERFORM_CLIENT_CERT_VERIFICATION
	 * this callback is called during OpenSSL verification of the cert
	 * sent from the server to the client. It is sent to protocol[0]
	 * callback as no protocol has been negotiated on the connection yet.
	 * Notice that the wsi is set because lws_client_connect_via_info was
	 * successful.
	 *
	 * See http://www.openssl.org/docs/ssl/SSL_CTX_set_verify.html
	 * to understand more detail about the OpenSSL callback that
	 * generates this libwebsockets callback and the meanings of the
	 * arguments passed. In this callback, user is the x509_ctx,
	 * in is the ssl pointer and len is preverify_ok.
	 *
	 * THIS IS NOT RECOMMENDED BUT if a cert validation error shall be
	 * overruled and cert shall be accepted as ok,
	 * X509_STORE_CTX_set_error((X509_STORE_CTX*)user, X509_V_OK); must be
	 * called and return value must be 0 to mean the cert is OK;
	 * returning 1 will fail the cert in any case.
	 *
	 * This also means that if you don't handle this callback then
	 * the default callback action of returning 0 will not accept the
	 * certificate in case of a validation error decided by the SSL lib.
	 *
	 * This is expected and secure behaviour when validating certificates.
	 *
	 * Note: LCCSCF_ALLOW_SELFSIGNED and
	 * LCCSCF_SKIP_SERVER_CERT_HOSTNAME_CHECK still work without this
	 * callback being implemented.
	 */

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to HTTP Server  -----
	 */

	LWS_CALLBACK_SERVER_NEW_CLIENT_INSTANTIATED		= 19,
	/**< A new client has been accepted by the ws server.  This
	 * callback allows setting any relevant property to it. Because this
	 * happens immediately after the instantiation of a new client,
	 * there's no websocket protocol selected yet so this callback is
	 * issued only to protocol 0. Only wsi is defined, pointing to the
	 * new client, and the return value is ignored. */

	LWS_CALLBACK_HTTP					= 12,
	/**< an http request has come from a client that is not
	 * asking to upgrade the connection to a websocket
	 * one.  This is a chance to serve http content,
	 * for example, to send a script to the client
	 * which will then open the websockets connection.
	 * in points to the URI path requested and
	 * lws_serve_http_file() makes it very
	 * simple to send back a file to the client.
	 * Normally after sending the file you are done
	 * with the http connection, since the rest of the
	 * activity will come by websockets from the script
	 * that was delivered by http, so you will want to
	 * return 1; to close and free up the connection. */

	LWS_CALLBACK_HTTP_BODY					= 13,
	/**< the next len bytes data from the http
	 * request body HTTP connection is now available in in. */

	LWS_CALLBACK_HTTP_BODY_COMPLETION			= 14,
	/**< the expected amount of http request body has been delivered */

	LWS_CALLBACK_HTTP_FILE_COMPLETION			= 15,
	/**< a file requested to be sent down http link has completed. */

	LWS_CALLBACK_HTTP_WRITEABLE				= 16,
	/**< you can write more down the http protocol link now. */

	LWS_CALLBACK_CLOSED_HTTP				=  5,
	/**< when a HTTP (non-websocket) session ends */

	LWS_CALLBACK_FILTER_HTTP_CONNECTION			= 18,
	/**< called when the request has
	 * been received and parsed from the client, but the response is
	 * not sent yet.  Return non-zero to disallow the connection.
	 * user is a pointer to the connection user space allocation,
	 * in is the URI, eg, "/"
	 * In your handler you can use the public APIs
	 * lws_hdr_total_length() / lws_hdr_copy() to access all of the
	 * headers using the header enums lws_token_indexes from
	 * libwebsockets.h to check for and read the supported header
	 * presence and content before deciding to allow the http
	 * connection to proceed or to kill the connection. */

	LWS_CALLBACK_ADD_HEADERS				= 53,
	/**< This gives your user code a chance to add headers to a server
	 * transaction bound to your protocol.  `in` points to a
	 * `struct lws_process_html_args` describing a buffer and length
	 * you can add headers into using the normal lws apis.
	 *
	 * (see LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER to add headers to
	 * a client transaction)
	 *
	 * Only `args->p` and `args->len` are valid, and `args->p` should
	 * be moved on by the amount of bytes written, if any.  Eg
	 *
	 * 	case LWS_CALLBACK_ADD_HEADERS:
	 *
	 *          struct lws_process_html_args *args =
	 *          		(struct lws_process_html_args *)in;
	 *
	 *	    if (lws_add_http_header_by_name(wsi,
	 *			(unsigned char *)"set-cookie:",
	 *			(unsigned char *)cookie, cookie_len,
	 *			(unsigned char **)&args->p,
	 *			(unsigned char *)args->p + args->max_len))
	 *		return 1;
	 *
	 *          break;
	 */

	LWS_CALLBACK_CHECK_ACCESS_RIGHTS			= 51,
	/**< This gives the user code a chance to forbid an http access.
	 * `in` points to a `struct lws_process_html_args`, which
	 * describes the URL, and a bit mask describing the type of
	 * authentication required.  If the callback returns nonzero,
	 * the transaction ends with HTTP_STATUS_UNAUTHORIZED. */

	LWS_CALLBACK_PROCESS_HTML				= 52,
	/**< This gives your user code a chance to mangle outgoing
	 * HTML.  `in` points to a `struct lws_process_html_args`
	 * which describes the buffer containing outgoing HTML.
	 * The buffer may grow up to `.max_len` (currently +128
	 * bytes per buffer).
	 */

	LWS_CALLBACK_HTTP_BIND_PROTOCOL				= 49,
	/**< By default, all HTTP handling is done in protocols[0].
	 * However you can bind different protocols (by name) to
	 * different parts of the URL space using callback mounts.  This
	 * callback occurs in the new protocol when a wsi is bound
	 * to that protocol.  Any protocol allocation related to the
	 * http transaction processing should be created then.
	 * These specific callbacks are necessary because with HTTP/1.1,
	 * a single connection may perform at series of different
	 * transactions at different URLs, thus the lifetime of the
	 * protocol bind is just for one transaction, not connection. */

	LWS_CALLBACK_HTTP_DROP_PROTOCOL				= 50,
	/**< This is called when a transaction is unbound from a protocol.
	 * It indicates the connection completed its transaction and may
	 * do something different now.  Any protocol allocation related
	 * to the http transaction processing should be destroyed. */

	LWS_CALLBACK_HTTP_CONFIRM_UPGRADE			= 86,
	/**< This is your chance to reject an HTTP upgrade action.  The
	 * name of the protocol being upgraded to is in 'in', and the ah
	 * is still bound to the wsi, so you can look at the headers.
	 *
	 * The default of returning 0 (ie, also if not handled) means the
	 * upgrade may proceed.  Return <0 to just hang up the connection,
	 * or >0 if you have rejected the connection by returning http headers
	 * and response code yourself.
	 *
	 * There is no need for you to call transaction_completed() as the
	 * caller will take care of it when it sees you returned >0.
	 */

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to HTTP Client  -----
	 */

	LWS_CALLBACK_ESTABLISHED_CLIENT_HTTP			= 44,
	/**< The HTTP client connection has succeeded, and is now
	 * connected to the server */

	LWS_CALLBACK_CLOSED_CLIENT_HTTP				= 45,
	/**< The HTTP client connection is closing */

	LWS_CALLBACK_RECEIVE_CLIENT_HTTP_READ			= 48,
	/**< This is generated by lws_http_client_read() used to drain
	 * incoming data.  In the case the incoming data was chunked, it will
	 * be split into multiple smaller callbacks for each chunk block,
	 * removing the chunk headers. If not chunked, it will appear all in
	 * one callback. */

	LWS_CALLBACK_RECEIVE_CLIENT_HTTP			= 46,
	/**< This simply indicates data was received on the HTTP client
	 * connection.  It does NOT drain or provide the data.
	 * This exists to neatly allow a proxying type situation,
	 * where this incoming data will go out on another connection.
	 * If the outgoing connection stalls, we should stall processing
	 * the incoming data.  So a handler for this in that case should
	 * simply set a flag to indicate there is incoming data ready
	 * and ask for a writeable callback on the outgoing connection.
	 * In the writable callback he can check the flag and then get
	 * and drain the waiting incoming data using lws_http_client_read().
	 * This will use callbacks to LWS_CALLBACK_RECEIVE_CLIENT_HTTP_READ
	 * to get and drain the incoming data, where it should be sent
	 * back out on the outgoing connection. */
	LWS_CALLBACK_COMPLETED_CLIENT_HTTP			= 47,
	/**< The client transaction completed... at the moment this
	 * is the same as closing since transaction pipelining on
	 * client side is not yet supported.  */

	LWS_CALLBACK_CLIENT_HTTP_WRITEABLE			= 57,
	/**< when doing an HTTP type client connection, you can call
	 * lws_client_http_body_pending(wsi, 1) from
	 * LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER to get these callbacks
	 * sending the HTTP headers.
	 *
	 * From this callback, when you have sent everything, you should let
	 * lws know by calling lws_client_http_body_pending(wsi, 0)
	 */

	LWS_CALLBACK_CLIENT_HTTP_BIND_PROTOCOL			= 85,
	LWS_CALLBACK_CLIENT_HTTP_DROP_PROTOCOL			= 76,

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to Websocket Server -----
	 */

	LWS_CALLBACK_ESTABLISHED				=  0,
	/**< (VH) after the server completes a handshake with an incoming
	 * client.  If you built the library with ssl support, in is a
	 * pointer to the ssl struct associated with the connection or NULL.
	 *
	 * b0 of len is set if the connection was made using ws-over-h2
	 */

	LWS_CALLBACK_CLOSED					=  4,
	/**< when the websocket session ends */

	LWS_CALLBACK_SERVER_WRITEABLE				= 11,
	/**< See LWS_CALLBACK_CLIENT_WRITEABLE */

	LWS_CALLBACK_RECEIVE					=  6,
	/**< data has appeared for this server endpoint from a
	 * remote client, it can be found at *in and is
	 * len bytes long */

	LWS_CALLBACK_RECEIVE_PONG				=  7,
	/**< servers receive PONG packets with this callback reason */

	LWS_CALLBACK_WS_PEER_INITIATED_CLOSE			= 38,
	/**< The peer has sent an unsolicited Close WS packet.  in and
	 * len are the optional close code (first 2 bytes, network
	 * order) and the optional additional information which is not
	 * defined in the standard, and may be a string or non human-readable
	 * data.
	 * If you return 0 lws will echo the close and then close the
	 * connection.  If you return nonzero lws will just close the
	 * connection. */

	LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION			= 20,
	/**< called when the handshake has
	 * been received and parsed from the client, but the response is
	 * not sent yet.  Return non-zero to disallow the connection.
	 * user is a pointer to the connection user space allocation,
	 * in is the requested protocol name
	 * In your handler you can use the public APIs
	 * lws_hdr_total_length() / lws_hdr_copy() to access all of the
	 * headers using the header enums lws_token_indexes from
	 * libwebsockets.h to check for and read the supported header
	 * presence and content before deciding to allow the handshake
	 * to proceed or to kill the connection. */

	LWS_CALLBACK_CONFIRM_EXTENSION_OKAY			= 25,
	/**< When the server handshake code
	 * sees that it does support a requested extension, before
	 * accepting the extension by additing to the list sent back to
	 * the client it gives this callback just to check that it's okay
	 * to use that extension.  It calls back to the requested protocol
	 * and with in being the extension name, len is 0 and user is
	 * valid.  Note though at this time the ESTABLISHED callback hasn't
	 * happened yet so if you initialize user content there, user
	 * content during this callback might not be useful for anything. */

	LWS_CALLBACK_WS_SERVER_BIND_PROTOCOL			= 77,
	LWS_CALLBACK_WS_SERVER_DROP_PROTOCOL			= 78,

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to Websocket Client -----
	 */

	LWS_CALLBACK_CLIENT_CONNECTION_ERROR			=  1,
	/**< the request client connection has been unable to complete a
	 * handshake with the remote server.  If in is non-NULL, you can
	 * find an error string of length len where it points to
	 *
	 * Diagnostic strings that may be returned include
	 *
	 *     	"getaddrinfo (ipv6) failed"
	 *     	"unknown address family"
	 *     	"getaddrinfo (ipv4) failed"
	 *     	"set socket opts failed"
	 *     	"insert wsi failed"
	 *     	"lws_ssl_client_connect1 failed"
	 *     	"lws_ssl_client_connect2 failed"
	 *     	"Peer hung up"
	 *     	"read failed"
	 *     	"HS: URI missing"
	 *     	"HS: Redirect code but no Location"
	 *     	"HS: URI did not parse"
	 *     	"HS: Redirect failed"
	 *     	"HS: Server did not return 200"
	 *     	"HS: OOM"
	 *     	"HS: disallowed by client filter"
	 *     	"HS: disallowed at ESTABLISHED"
	 *     	"HS: ACCEPT missing"
	 *     	"HS: ws upgrade response not 101"
	 *     	"HS: UPGRADE missing"
	 *     	"HS: Upgrade to something other than websocket"
	 *     	"HS: CONNECTION missing"
	 *     	"HS: UPGRADE malformed"
	 *     	"HS: PROTOCOL malformed"
	 *     	"HS: Cannot match protocol"
	 *     	"HS: EXT: list too big"
	 *     	"HS: EXT: failed setting defaults"
	 *     	"HS: EXT: failed parsing defaults"
	 *     	"HS: EXT: failed parsing options"
	 *     	"HS: EXT: Rejects server options"
	 *     	"HS: EXT: unknown ext"
	 *     	"HS: Accept hash wrong"
	 *     	"HS: Rejected by filter cb"
	 *     	"HS: OOM"
	 *     	"HS: SO_SNDBUF failed"
	 *     	"HS: Rejected at CLIENT_ESTABLISHED"
	 */

	LWS_CALLBACK_CLIENT_FILTER_PRE_ESTABLISH		=  2,
	/**< this is the last chance for the client user code to examine the
	 * http headers and decide to reject the connection.  If the
	 * content in the headers is interesting to the
	 * client (url, etc) it needs to copy it out at
	 * this point since it will be destroyed before
	 * the CLIENT_ESTABLISHED call */

	LWS_CALLBACK_CLIENT_ESTABLISHED				=  3,
	/**< after your client connection completed the websocket upgrade
	 * handshake with the remote server */

	LWS_CALLBACK_CLIENT_CLOSED				= 75,
	/**< when a client websocket session ends */

	LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER		= 24,
	/**< this callback happens
	 * when a client handshake is being compiled.  user is NULL,
	 * in is a char **, it's pointing to a char * which holds the
	 * next location in the header buffer where you can add
	 * headers, and len is the remaining space in the header buffer,
	 * which is typically some hundreds of bytes.  So, to add a canned
	 * cookie, your handler code might look similar to:
	 *
	 *	char **p = (char **)in, *end = (*p) + len;
	 *
	 *	if (lws_add_http_header_by_token(wsi, WSI_TOKEN_HTTP_COOKIE,
	 *			(unsigned char)"a=b", 3, p, end))
	 *		return -1;
	 *
	 * See LWS_CALLBACK_ADD_HEADERS for adding headers to server
	 * transactions.
	 */

	LWS_CALLBACK_CLIENT_RECEIVE				=  8,
	/**< data has appeared from the server for the client connection, it
	 * can be found at *in and is len bytes long */

	LWS_CALLBACK_CLIENT_RECEIVE_PONG			=  9,
	/**< clients receive PONG packets with this callback reason */

	LWS_CALLBACK_CLIENT_WRITEABLE				= 10,
	/**<  If you call lws_callback_on_writable() on a connection, you will
	 * get one of these callbacks coming when the connection socket
	 * is able to accept another write packet without blocking.
	 * If it already was able to take another packet without blocking,
	 * you'll get this callback at the next call to the service loop
	 * function.  Notice that CLIENTs get LWS_CALLBACK_CLIENT_WRITEABLE
	 * and servers get LWS_CALLBACK_SERVER_WRITEABLE. */

	LWS_CALLBACK_CLIENT_CONFIRM_EXTENSION_SUPPORTED		= 26,
	/**< When a ws client
	 * connection is being prepared to start a handshake to a server,
	 * each supported extension is checked with protocols[0] callback
	 * with this reason, giving the user code a chance to suppress the
	 * claim to support that extension by returning non-zero.  If
	 * unhandled, by default 0 will be returned and the extension
	 * support included in the header to the server.  Notice this
	 * callback comes to protocols[0]. */

	LWS_CALLBACK_WS_EXT_DEFAULTS				= 39,
	/**< Gives client connections an opportunity to adjust negotiated
	 * extension defaults.  `user` is the extension name that was
	 * negotiated (eg, "permessage-deflate").  `in` points to a
	 * buffer and `len` is the buffer size.  The user callback can
	 * set the buffer to a string describing options the extension
	 * should parse.  Or just ignore for defaults. */


	LWS_CALLBACK_FILTER_NETWORK_CONNECTION			= 17,
	/**< called when a client connects to
	 * the server at network level; the connection is accepted but then
	 * passed to this callback to decide whether to hang up immediately
	 * or not, based on the client IP.  in contains the connection
	 * socket's descriptor. Since the client connection information is
	 * not available yet, wsi still pointing to the main server socket.
	 * Return non-zero to terminate the connection before sending or
	 * receiving anything. Because this happens immediately after the
	 * network connection from the client, there's no websocket protocol
	 * selected yet so this callback is issued only to protocol 0. */

	LWS_CALLBACK_WS_CLIENT_BIND_PROTOCOL			= 79,
	LWS_CALLBACK_WS_CLIENT_DROP_PROTOCOL			= 80,

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to external poll loop integration  -----
	 */

	LWS_CALLBACK_GET_THREAD_ID				= 31,
	/**< lws can accept callback when writable requests from other
	 * threads, if you implement this callback and return an opaque
	 * current thread ID integer. */

	/* external poll() management support */
	LWS_CALLBACK_ADD_POLL_FD				= 32,
	/**< lws normally deals with its poll() or other event loop
	 * internally, but in the case you are integrating with another
	 * server you will need to have lws sockets share a
	 * polling array with the other server.  This and the other
	 * POLL_FD related callbacks let you put your specialized
	 * poll array interface code in the callback for protocol 0, the
	 * first protocol you support, usually the HTTP protocol in the
	 * serving case.
	 * This callback happens when a socket needs to be
	 * added to the polling loop: in points to a struct
	 * lws_pollargs; the fd member of the struct is the file
	 * descriptor, and events contains the active events
	 *
	 * If you are using the internal lws polling / event loop
	 * you can just ignore these callbacks. */

	LWS_CALLBACK_DEL_POLL_FD				= 33,
	/**< This callback happens when a socket descriptor
	 * needs to be removed from an external polling array.  in is
	 * again the struct lws_pollargs containing the fd member
	 * to be removed.  If you are using the internal polling
	 * loop, you can just ignore it. */

	LWS_CALLBACK_CHANGE_MODE_POLL_FD			= 34,
	/**< This callback happens when lws wants to modify the events for
	 * a connection.
	 * in is the struct lws_pollargs with the fd to change.
	 * The new event mask is in events member and the old mask is in
	 * the prev_events member.
	 * If you are using the internal polling loop, you can just ignore
	 * it. */

	LWS_CALLBACK_LOCK_POLL					= 35,
	/**< These allow the external poll changes driven
	 * by lws to participate in an external thread locking
	 * scheme around the changes, so the whole thing is threadsafe.
	 * These are called around three activities in the library,
	 *	- inserting a new wsi in the wsi / fd table (len=1)
	 *	- deleting a wsi from the wsi / fd table (len=1)
	 *	- changing a wsi's POLLIN/OUT state (len=0)
	 * Locking and unlocking external synchronization objects when
	 * len == 1 allows external threads to be synchronized against
	 * wsi lifecycle changes if it acquires the same lock for the
	 * duration of wsi dereference from the other thread context. */

	LWS_CALLBACK_UNLOCK_POLL				= 36,
	/**< See LWS_CALLBACK_LOCK_POLL, ignore if using lws internal poll */

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to CGI serving -----
	 */

	LWS_CALLBACK_CGI					= 40,
	/**< CGI: CGI IO events on stdin / out / err are sent here on
	 * protocols[0].  The provided `lws_callback_http_dummy()`
	 * handles this and the callback should be directed there if
	 * you use CGI. */

	LWS_CALLBACK_CGI_TERMINATED				= 41,
	/**< CGI: The related CGI process ended, this is called before
	 * the wsi is closed.  Used to, eg, terminate chunking.
	 * The provided `lws_callback_http_dummy()`
	 * handles this and the callback should be directed there if
	 * you use CGI.  The child PID that terminated is in len. */

	LWS_CALLBACK_CGI_STDIN_DATA				= 42,
	/**< CGI: Data is, to be sent to the CGI process stdin, eg from
	 * a POST body.  The provided `lws_callback_http_dummy()`
	 * handles this and the callback should be directed there if
	 * you use CGI. */

	LWS_CALLBACK_CGI_STDIN_COMPLETED			= 43,
	/**< CGI: no more stdin is coming.  The provided
	 * `lws_callback_http_dummy()` handles this and the callback
	 * should be directed there if you use CGI. */

	LWS_CALLBACK_CGI_PROCESS_ATTACH				= 70,
	/**< CGI: Sent when the CGI process is spawned for the wsi.  The
	 * len parameter is the PID of the child process */

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to Generic Sessions -----
	 */

	LWS_CALLBACK_SESSION_INFO				= 54,
	/**< This is only generated by user code using generic sessions.
	 * It's used to get a `struct lws_session_info` filled in by
	 * generic sessions with information about the logged-in user.
	 * See the messageboard sample for an example of how to use. */

	LWS_CALLBACK_GS_EVENT					= 55,
	/**< Indicates an event happened to the Generic Sessions session.
	 * `in` contains a `struct lws_gs_event_args` describing the event. */

	LWS_CALLBACK_HTTP_PMO					= 56,
	/**< per-mount options for this connection, called before
	 * the normal LWS_CALLBACK_HTTP when the mount has per-mount
	 * options.
	 */

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to RAW sockets -----
	 */

	LWS_CALLBACK_RAW_RX					= 59,
	/**< RAW mode connection RX */

	LWS_CALLBACK_RAW_CLOSE					= 60,
	/**< RAW mode connection is closing */

	LWS_CALLBACK_RAW_WRITEABLE				= 61,
	/**< RAW mode connection may be written */

	LWS_CALLBACK_RAW_ADOPT					= 62,
	/**< RAW mode connection was adopted (equivalent to 'wsi created') */

	LWS_CALLBACK_RAW_SKT_BIND_PROTOCOL			= 81,
	LWS_CALLBACK_RAW_SKT_DROP_PROTOCOL			= 82,

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to RAW file handles -----
	 */

	LWS_CALLBACK_RAW_ADOPT_FILE				= 63,
	/**< RAW mode file was adopted (equivalent to 'wsi created') */

	LWS_CALLBACK_RAW_RX_FILE				= 64,
	/**< This is the indication the RAW mode file has something to read.
	 *   This doesn't actually do the read of the file and len is always
	 *   0... your code should do the read having been informed there is
	 *   something to read now. */

	LWS_CALLBACK_RAW_WRITEABLE_FILE				= 65,
	/**< RAW mode file is writeable */

	LWS_CALLBACK_RAW_CLOSE_FILE				= 66,
	/**< RAW mode wsi that adopted a file is closing */

	LWS_CALLBACK_RAW_FILE_BIND_PROTOCOL			= 83,
	LWS_CALLBACK_RAW_FILE_DROP_PROTOCOL			= 84,

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to generic wsi events -----
	 */

	LWS_CALLBACK_TIMER					= 73,
	/**< When the time elapsed after a call to
	 * lws_set_timer_usecs(wsi, usecs) is up, the wsi will get one of
	 * these callbacks.  The deadline can be continuously extended into the
	 * future by later calls to lws_set_timer_usecs() before the deadline
	 * expires, or cancelled by lws_set_timer_usecs(wsi, -1);
	 */

	LWS_CALLBACK_EVENT_WAIT_CANCELLED			= 71,
	/**< This is sent to every protocol of every vhost in response
	 * to lws_cancel_service() or lws_cancel_service_pt().  This
	 * callback is serialized in the lws event loop normally, even
	 * if the lws_cancel_service[_pt]() call was from a different
	 * thread. */

	LWS_CALLBACK_CHILD_CLOSING				= 69,
	/**< Sent to parent to notify them a child is closing / being
	 * destroyed.  in is the child wsi.
	 */

	/* ---------------------------------------------------------------------
	 * ----- Callbacks related to TLS certificate management -----
	 */

	LWS_CALLBACK_VHOST_CERT_AGING				= 72,
	/**< When a vhost TLS cert has its expiry checked, this callback
	 * is broadcast to every protocol of every vhost in case the
	 * protocol wants to take some action with this information.
	 * \p in is a pointer to a struct lws_acme_cert_aging_args,
	 * and \p len is the number of days left before it expires, as
	 * a (ssize_t).  In the struct lws_acme_cert_aging_args, vh
	 * points to the vhost the cert aging information applies to,
	 * and element_overrides[] is an optional way to update information
	 * from the pvos... NULL in an index means use the information from
	 * from the pvo for the cert renewal, non-NULL in the array index
	 * means use that pointer instead for the index. */

	LWS_CALLBACK_VHOST_CERT_UPDATE				= 74,
	/**< When a vhost TLS cert is being updated, progress is
	 * reported to the vhost in question here, including completion
	 * and failure.  in points to optional JSON, and len represents the
	 * connection state using enum lws_cert_update_state */


	/****** add new things just above ---^ ******/

	LWS_CALLBACK_USER = 1000,
	/**<  user code can use any including above without fear of clashes */
};



/**
 * typedef lws_callback_function() - User server actions
 * \param wsi:	Opaque websocket instance pointer
 * \param reason:	The reason for the call
 * \param user:	Pointer to per-session user data allocated by library
 * \param in:		Pointer used for some callback reasons
 * \param len:	Length set for some callback reasons
 *
 *	This callback is the way the user controls what is served.  All the
 *	protocol detail is hidden and handled by the library.
 *
 *	For each connection / session there is user data allocated that is
 *	pointed to by "user".  You set the size of this user data area when
 *	the library is initialized with lws_create_server.
 */
typedef int
lws_callback_function(struct lws *wsi, enum lws_callback_reasons reason,
		    void *user, void *in, size_t len);

#define LWS_CB_REASON_AUX_BF__CGI		1
#define LWS_CB_REASON_AUX_BF__PROXY		2
#define LWS_CB_REASON_AUX_BF__CGI_CHUNK_END	4
#define LWS_CB_REASON_AUX_BF__CGI_HEADERS	8
#define LWS_CB_REASON_AUX_BF__PROXY_TRANS_END	16
#define LWS_CB_REASON_AUX_BF__PROXY_HEADERS	32
///@}
