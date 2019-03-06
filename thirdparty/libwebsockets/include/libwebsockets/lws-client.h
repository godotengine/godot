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

/*! \defgroup client Client related functions
 * ##Client releated functions
 * \ingroup lwsapi
 *
 * */
///@{

/** enum lws_client_connect_ssl_connection_flags - flags that may be used
 * with struct lws_client_connect_info ssl_connection member to control if
 * and how SSL checks apply to the client connection being created
 */

enum lws_client_connect_ssl_connection_flags {
	LCCSCF_USE_SSL 				= (1 << 0),
	LCCSCF_ALLOW_SELFSIGNED			= (1 << 1),
	LCCSCF_SKIP_SERVER_CERT_HOSTNAME_CHECK	= (1 << 2),
	LCCSCF_ALLOW_EXPIRED			= (1 << 3),

	LCCSCF_PIPELINE				= (1 << 16),
		/**< Serialize / pipeline multiple client connections
		 * on a single connection where possible.
		 *
		 * HTTP/1.0: possible if Keep-Alive: yes sent by server
		 * HTTP/1.1: always possible... uses pipelining
		 * HTTP/2:   always possible... uses parallel streams
		 * */
};

/** struct lws_client_connect_info - parameters to connect with when using
 *				    lws_client_connect_via_info() */

struct lws_client_connect_info {
	struct lws_context *context;
	/**< lws context to create connection in */
	const char *address;
	/**< remote address to connect to */
	int port;
	/**< remote port to connect to */
	int ssl_connection;
	/**< 0, or a combination of LCCSCF_ flags */
	const char *path;
	/**< uri path */
	const char *host;
	/**< content of host header */
	const char *origin;
	/**< content of origin header */
	const char *protocol;
	/**< list of ws protocols we could accept */
	int ietf_version_or_minus_one;
	/**< deprecated: currently leave at 0 or -1 */
	void *userdata;
	/**< if non-NULL, use this as wsi user_data instead of malloc it */
	const void *client_exts;
	/**< UNUSED... provide in info.extensions at context creation time */
	const char *method;
	/**< if non-NULL, do this http method instead of ws[s] upgrade.
	 * use "GET" to be a simple http client connection.  "RAW" gets
	 * you a connected socket that lws itself will leave alone once
	 * connected. */
	struct lws *parent_wsi;
	/**< if another wsi is responsible for this connection, give it here.
	 * this is used to make sure if the parent closes so do any
	 * child connections first. */
	const char *uri_replace_from;
	/**< if non-NULL, when this string is found in URIs in
	 * text/html content-encoding, it's replaced with uri_replace_to */
	const char *uri_replace_to;
	/**< see uri_replace_from */
	struct lws_vhost *vhost;
	/**< vhost to bind to (used to determine related SSL_CTX) */
	struct lws **pwsi;
	/**< if not NULL, store the new wsi here early in the connection
	 * process.  Although we return the new wsi, the call to create the
	 * client connection does progress the connection somewhat and may
	 * meet an error that will result in the connection being scrubbed and
	 * NULL returned.  While the wsi exists though, he may process a
	 * callback like CLIENT_CONNECTION_ERROR with his wsi: this gives the
	 * user callback a way to identify which wsi it is that faced the error
	 * even before the new wsi is returned and even if ultimately no wsi
	 * is returned.
	 */
	const char *iface;
	/**< NULL to allow routing on any interface, or interface name or IP
	 * to bind the socket to */
	const char *local_protocol_name;
	/**< NULL: .protocol is used both to select the local protocol handler
	 *         to bind to and as the list of remote ws protocols we could
	 *         accept.
	 *   non-NULL: this protocol name is used to bind the connection to
	 *             the local protocol handler.  .protocol is used for the
	 *             list of remote ws protocols we could accept */
	const char *alpn;
	/**< NULL: allow lws default ALPN list, from vhost if present or from
	 *       list of roles built into lws
	 * non-NULL: require one from provided comma-separated list of alpn
	 *           tokens
	 */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility
	 *
	 * The below is to ensure later library versions with new
	 * members added above will see 0 (default) even if the app
	 * was not built against the newer headers.
	 */

	void *_unused[4]; /**< dummy */
};

/**
 * lws_client_connect_via_info() - Connect to another websocket server
 * \param ccinfo: pointer to lws_client_connect_info struct
 *
 *	This function creates a connection to a remote server using the
 *	information provided in ccinfo.
 */
LWS_VISIBLE LWS_EXTERN struct lws *
lws_client_connect_via_info(const struct lws_client_connect_info *ccinfo);

/**
 * lws_init_vhost_client_ssl() - also enable client SSL on an existing vhost
 *
 * \param info: client ssl related info
 * \param vhost: which vhost to initialize client ssl operations on
 *
 * You only need to call this if you plan on using SSL client connections on
 * the vhost.  For non-SSL client connections, it's not necessary to call this.
 *
 * The following members of info are used during the call
 *
 *	 - options must have LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT set,
 *	     otherwise the call does nothing
 *	 - provided_client_ssl_ctx must be NULL to get a generated client
 *	     ssl context, otherwise you can pass a prepared one in by setting it
 *	 - ssl_cipher_list may be NULL or set to the client valid cipher list
 *	 - ssl_ca_filepath may be NULL or client cert filepath
 *	 - ssl_cert_filepath may be NULL or client cert filepath
 *	 - ssl_private_key_filepath may be NULL or client cert private key
 *
 * You must create your vhost explicitly if you want to use this, so you have
 * a pointer to the vhost.  Create the context first with the option flag
 * LWS_SERVER_OPTION_EXPLICIT_VHOSTS and then call lws_create_vhost() with
 * the same info struct.
 */
LWS_VISIBLE LWS_EXTERN int
lws_init_vhost_client_ssl(const struct lws_context_creation_info *info,
			  struct lws_vhost *vhost);
/**
 * lws_http_client_read() - consume waiting received http client data
 *
 * \param wsi: client connection
 * \param buf: pointer to buffer pointer - fill with pointer to your buffer
 * \param len: pointer to chunk length - fill with max length of buffer
 *
 * This is called when the user code is notified client http data has arrived.
 * The user code may choose to delay calling it to consume the data, for example
 * waiting until an onward connection is writeable.
 *
 * For non-chunked connections, up to len bytes of buf are filled with the
 * received content.  len is set to the actual amount filled before return.
 *
 * For chunked connections, the linear buffer content contains the chunking
 * headers and it cannot be passed in one lump.  Instead, this function will
 * call back LWS_CALLBACK_RECEIVE_CLIENT_HTTP_READ with in pointing to the
 * chunk start and len set to the chunk length.  There will be as many calls
 * as there are chunks or partial chunks in the buffer.
 */
LWS_VISIBLE LWS_EXTERN int
lws_http_client_read(struct lws *wsi, char **buf, int *len);

/**
 * lws_http_client_http_response() - get last HTTP response code
 *
 * \param wsi: client connection
 *
 * Returns the last server response code, eg, 200 for client http connections.
 *
 * You should capture this during the LWS_CALLBACK_ESTABLISHED_CLIENT_HTTP
 * callback, because after that the memory reserved for storing the related
 * headers is freed and this value is lost.
 */
LWS_VISIBLE LWS_EXTERN unsigned int
lws_http_client_http_response(struct lws *wsi);

LWS_VISIBLE LWS_EXTERN void
lws_client_http_body_pending(struct lws *wsi, int something_left_to_send);

/**
 * lws_client_http_body_pending() - control if client connection neeeds to send body
 *
 * \param wsi: client connection
 * \param something_left_to_send: nonzero if need to send more body, 0 (default)
 * 				if nothing more to send
 *
 * If you will send payload data with your HTTP client connection, eg, for POST,
 * when you set the related http headers in
 * LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER callback you should also call
 * this API with something_left_to_send nonzero, and call
 * lws_callback_on_writable(wsi);
 *
 * After sending the headers, lws will call your callback with
 * LWS_CALLBACK_CLIENT_HTTP_WRITEABLE reason when writable.  You can send the
 * next part of the http body payload, calling lws_callback_on_writable(wsi);
 * if there is more to come, or lws_client_http_body_pending(wsi, 0); to
 * let lws know the last part is sent and the connection can move on.
 */

///@}
