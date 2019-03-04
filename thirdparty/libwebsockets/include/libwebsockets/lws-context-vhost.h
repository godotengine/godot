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

/*! \defgroup context-and-vhost context and vhost related functions
 * ##Context and Vhost releated functions
 * \ingroup lwsapi
 *
 *
 *  LWS requires that there is one context, in which you may define multiple
 *  vhosts.  Each vhost is a virtual host, with either its own listen port
 *  or sharing an existing one.  Each vhost has its own SSL context that can
 *  be set up individually or left disabled.
 *
 *  If you don't care about multiple "site" support, you can ignore it and
 *  lws will create a single default vhost at context creation time.
 */
///@{

/*
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */

/** enum lws_context_options - context and vhost options */
enum lws_context_options {
	LWS_SERVER_OPTION_REQUIRE_VALID_OPENSSL_CLIENT_CERT	= (1 << 1) |
								  (1 << 12),
	/**< (VH) Don't allow the connection unless the client has a
	 * client cert that we recognize; provides
	 * LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT */
	LWS_SERVER_OPTION_SKIP_SERVER_CANONICAL_NAME		= (1 << 2),
	/**< (CTX) Don't try to get the server's hostname */
	LWS_SERVER_OPTION_ALLOW_NON_SSL_ON_SSL_PORT		= (1 << 3) |
								  (1 << 12),
	/**< (VH) Allow non-SSL (plaintext) connections on the same
	 * port as SSL is listening... undermines the security of SSL;
	 * provides  LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT */
	LWS_SERVER_OPTION_LIBEV					= (1 << 4),
	/**< (CTX) Use libev event loop */
	LWS_SERVER_OPTION_DISABLE_IPV6				= (1 << 5),
	/**< (VH) Disable IPV6 support */
	LWS_SERVER_OPTION_DISABLE_OS_CA_CERTS			= (1 << 6),
	/**< (VH) Don't load OS CA certs, you will need to load your
	 * own CA cert(s) */
	LWS_SERVER_OPTION_PEER_CERT_NOT_REQUIRED		= (1 << 7),
	/**< (VH) Accept connections with no valid Cert (eg, selfsigned) */
	LWS_SERVER_OPTION_VALIDATE_UTF8				= (1 << 8),
	/**< (VH) Check UT-8 correctness */
	LWS_SERVER_OPTION_SSL_ECDH				= (1 << 9) |
								  (1 << 12),
	/**< (VH)  initialize ECDH ciphers */
	LWS_SERVER_OPTION_LIBUV					= (1 << 10),
	/**< (CTX)  Use libuv event loop */
	LWS_SERVER_OPTION_REDIRECT_HTTP_TO_HTTPS		= (1 << 11) |
								  (1 << 12),
	/**< (VH) Use http redirect to force http to https
	 * (deprecated: use mount redirection) */
	LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT			= (1 << 12),
	/**< (CTX) Initialize the SSL library at all */
	LWS_SERVER_OPTION_EXPLICIT_VHOSTS			= (1 << 13),
	/**< (CTX) Only create the context when calling context
	 * create api, implies user code will create its own vhosts */
	LWS_SERVER_OPTION_UNIX_SOCK				= (1 << 14),
	/**< (VH) Use Unix socket */
	LWS_SERVER_OPTION_STS					= (1 << 15),
	/**< (VH) Send Strict Transport Security header, making
	 * clients subsequently go to https even if user asked for http */
	LWS_SERVER_OPTION_IPV6_V6ONLY_MODIFY			= (1 << 16),
	/**< (VH) Enable LWS_SERVER_OPTION_IPV6_V6ONLY_VALUE to take effect */
	LWS_SERVER_OPTION_IPV6_V6ONLY_VALUE			= (1 << 17),
	/**< (VH) if set, only ipv6 allowed on the vhost */
	LWS_SERVER_OPTION_UV_NO_SIGSEGV_SIGFPE_SPIN		= (1 << 18),
	/**< (CTX) Libuv only: Do not spin on SIGSEGV / SIGFPE.  A segfault
	 * normally makes the lib spin so you can attach a debugger to it
	 * even if it happened without a debugger in place.  You can disable
	 * that by giving this option.
	 */
	LWS_SERVER_OPTION_JUST_USE_RAW_ORIGIN			= (1 << 19),
	/**< For backwards-compatibility reasons, by default
	 * lws prepends "http://" to the origin you give in the client
	 * connection info struct.  If you give this flag when you create
	 * the context, only the string you give in the client connect
	 * info for .origin (if any) will be used directly.
	 */
	LWS_SERVER_OPTION_FALLBACK_TO_RAW			= (1 << 20),
	/**< (VH) if invalid http is coming in the first line,  */
	LWS_SERVER_OPTION_LIBEVENT				= (1 << 21),
	/**< (CTX) Use libevent event loop */
	LWS_SERVER_OPTION_ONLY_RAW				= (1 << 22),
	/**< (VH) All connections to this vhost / port are RAW as soon as
	 * the connection is accepted, no HTTP is going to be coming.
	 */
	LWS_SERVER_OPTION_ALLOW_LISTEN_SHARE			= (1 << 23),
	/**< (VH) Set to allow multiple listen sockets on one interface +
	 * address + port.  The default is to strictly allow only one
	 * listen socket at a time.  This is automatically selected if you
	 * have multiple service threads.
	 */
	LWS_SERVER_OPTION_CREATE_VHOST_SSL_CTX			= (1 << 24),
	/**< (VH) Force setting up the vhost SSL_CTX, even though the user
	 * code doesn't explicitly provide a cert in the info struct.  It
	 * implies the user code is going to provide a cert at the
	 * LWS_CALLBACK_OPENSSL_LOAD_EXTRA_SERVER_VERIFY_CERTS callback, which
	 * provides the vhost SSL_CTX * in the user parameter.
	 */
	LWS_SERVER_OPTION_SKIP_PROTOCOL_INIT			= (1 << 25),
	/**< (VH) You probably don't want this.  It forces this vhost to not
	 * call LWS_CALLBACK_PROTOCOL_INIT on its protocols.  It's used in the
	 * special case of a temporary vhost bound to a single protocol.
	 */
	LWS_SERVER_OPTION_IGNORE_MISSING_CERT			= (1 << 26),
	/**< (VH) Don't fail if the vhost TLS cert or key are missing, just
	 * continue.  The vhost won't be able to serve anything, but if for
	 * example the ACME plugin was configured to fetch a cert, this lets
	 * you bootstrap your vhost from having no cert to start with.
	 */
	LWS_SERVER_OPTION_VHOST_UPG_STRICT_HOST_CHECK		= (1 << 27),
	/**< (VH) On this vhost, if the connection is being upgraded, insist
	 * that there's a Host: header and that the contents match the vhost
	 * name + port (443 / 80 are assumed if no :port given based on if the
	 * connection is using TLS).
	 *
	 * By default, without this flag, on upgrade lws just checks that the
	 * Host: header was given without checking the contents... this is to
	 * allow lax hostname mappings like localhost / 127.0.0.1, and CNAME
	 * mappings like www.mysite.com / mysite.com
	 */
	LWS_SERVER_OPTION_HTTP_HEADERS_SECURITY_BEST_PRACTICES_ENFORCE	= (1 << 28),
	/**< (VH) Send lws default HTTP headers recommended by Mozilla
	 * Observatory for security.  This is a helper option that sends canned
	 * headers on each http response enabling a VERY strict Content Security
	 * Policy.  The policy is so strict, for example it won't let the page
	 * run its own inline JS nor show images or take CSS from a different
	 * server.  In many cases your JS only comes from your server as do the
	 * image sources and CSS, so that is what you want... attackers hoping
	 * to inject JS into your DOM are completely out of luck since even if
	 * they succeed, it will be rejected for execution by the browser
	 * according to the strict CSP.  In other cases you have to deviate from
	 * the complete strictness, in which case don't use this flag: use the
	 * .headers member in the vhost init described in struct
	 * lws_context_creation_info instead to send the adapted headers
	 * yourself.
	 */

	/****** add new things just above ---^ ******/
};

#define lws_check_opt(c, f) (((c) & (f)) == (f))

struct lws_plat_file_ops;

/** struct lws_context_creation_info - parameters to create context and /or vhost with
 *
 * This is also used to create vhosts.... if LWS_SERVER_OPTION_EXPLICIT_VHOSTS
 * is not given, then for backwards compatibility one vhost is created at
 * context-creation time using the info from this struct.
 *
 * If LWS_SERVER_OPTION_EXPLICIT_VHOSTS is given, then no vhosts are created
 * at the same time as the context, they are expected to be created afterwards.
 */
struct lws_context_creation_info {
	int port;
	/**< VHOST: Port to listen on. Use CONTEXT_PORT_NO_LISTEN to suppress
	 * listening for a client. Use CONTEXT_PORT_NO_LISTEN_SERVER if you are
	 * writing a server but you are using \ref sock-adopt instead of the
	 * built-in listener.
	 *
	 * You can also set port to 0, in which case the kernel will pick
	 * a random port that is not already in use.  You can find out what
	 * port the vhost is listening on using lws_get_vhost_listen_port() */
	const char *iface;
	/**< VHOST: NULL to bind the listen socket to all interfaces, or the
	 * interface name, eg, "eth2"
	 * If options specifies LWS_SERVER_OPTION_UNIX_SOCK, this member is
	 * the pathname of a UNIX domain socket. you can use the UNIX domain
	 * sockets in abstract namespace, by prepending an at symbol to the
	 * socket name. */
	const struct lws_protocols *protocols;
	/**< VHOST: Array of structures listing supported protocols and a
	 * protocol-specific callback for each one.  The list is ended with an
	 * entry that has a NULL callback pointer. */
	const struct lws_extension *extensions;
	/**< VHOST: NULL or array of lws_extension structs listing the
	 * extensions this context supports. */
	const struct lws_token_limits *token_limits;
	/**< CONTEXT: NULL or struct lws_token_limits pointer which is
	 * initialized with a token length limit for each possible WSI_TOKEN_ */
	const char *ssl_private_key_password;
	/**< VHOST: NULL or the passphrase needed for the private key. (For
	 * backwards compatibility, this can also be used to pass the client
	 * cert passphrase when setting up a vhost client SSL context, but it is
	 * preferred to use .client_ssl_private_key_password for that.) */
	const char *ssl_cert_filepath;
	/**< VHOST: If libwebsockets was compiled to use ssl, and you want
	 * to listen using SSL, set to the filepath to fetch the
	 * server cert from, otherwise NULL for unencrypted.  (For backwards
	 * compatibility, this can also be used to pass the client certificate
	 * when setting up a vhost client SSL context, but it is preferred to
	 * use .client_ssl_cert_filepath for that.) */
	const char *ssl_private_key_filepath;
	/**<  VHOST: filepath to private key if wanting SSL mode;
	 * if this is set to NULL but ssl_cert_filepath is set, the
	 * OPENSSL_CONTEXT_REQUIRES_PRIVATE_KEY callback is called
	 * to allow setting of the private key directly via openSSL
	 * library calls.   (For backwards compatibility, this can also be used
	 * to pass the client cert private key filepath when setting up a
	 * vhost client SSL context, but it is preferred to use
	 * .client_ssl_private_key_filepath for that.) */
	const char *ssl_ca_filepath;
	/**< VHOST: CA certificate filepath or NULL.  (For backwards
	 * compatibility, this can also be used to pass the client CA
	 * filepath when setting up a vhost client SSL context,
	 * but it is preferred to use .client_ssl_ca_filepath for that.) */
	const char *ssl_cipher_list;
	/**< VHOST: List of valid ciphers to use ON TLS1.2 AND LOWER ONLY (eg,
	 * "RC4-MD5:RC4-SHA:AES128-SHA:AES256-SHA:HIGH:!DSS:!aNULL"
	 * or you can leave it as NULL to get "DEFAULT" (For backwards
	 * compatibility, this can also be used to pass the client cipher
	 * list when setting up a vhost client SSL context,
	 * but it is preferred to use .client_ssl_cipher_list for that.)
	 * SEE .tls1_3_plus_cipher_list and .client_tls_1_3_plus_cipher_list
	 * for the equivalent for tls1.3.
	 */
	const char *http_proxy_address;
	/**< VHOST: If non-NULL, attempts to proxy via the given address.
	 * If proxy auth is required, use format
	 * "username:password\@server:port" */
	unsigned int http_proxy_port;
	/**< VHOST: If http_proxy_address was non-NULL, uses this port */
	int gid;
	/**< CONTEXT: group id to change to after setting listen socket,
	 *   or -1. */
	int uid;
	/**< CONTEXT: user id to change to after setting listen socket,
	 *   or -1. */
	unsigned int options;
	/**< VHOST + CONTEXT: 0, or LWS_SERVER_OPTION_... bitfields */
	void *user;
	/**< VHOST + CONTEXT: optional user pointer that will be associated
	 * with the context when creating the context (and can be retrieved by
	 * lws_context_user(context), or with the vhost when creating the vhost
	 * (and can be retrieved by lws_vhost_user(vhost)).  You will need to
	 * use LWS_SERVER_OPTION_EXPLICIT_VHOSTS and create the vhost separately
	 * if you care about giving the context and vhost different user pointer
	 * values.
	 */
	int ka_time;
	/**< CONTEXT: 0 for no TCP keepalive, otherwise apply this keepalive
	 * timeout to all libwebsocket sockets, client or server */
	int ka_probes;
	/**< CONTEXT: if ka_time was nonzero, after the timeout expires how many
	 * times to try to get a response from the peer before giving up
	 * and killing the connection */
	int ka_interval;
	/**< CONTEXT: if ka_time was nonzero, how long to wait before each ka_probes
	 * attempt */
#if defined(LWS_WITH_TLS) && !defined(LWS_WITH_MBEDTLS)
	SSL_CTX *provided_client_ssl_ctx;
	/**< CONTEXT: If non-null, swap out libwebsockets ssl
	  * implementation for the one provided by provided_ssl_ctx.
	  * Libwebsockets no longer is responsible for freeing the context
	  * if this option is selected. */
#else /* maintain structure layout either way */
	void *provided_client_ssl_ctx; /**< dummy if ssl disabled */
#endif

	unsigned short max_http_header_data;
	/**< CONTEXT: The max amount of header payload that can be handled
	 * in an http request (unrecognized header payload is dropped) */
	unsigned short max_http_header_pool;
	/**< CONTEXT: The max number of connections with http headers that
	 * can be processed simultaneously (the corresponding memory is
	 * allocated and deallocated dynamically as needed).  If the pool is
	 * fully busy new incoming connections must wait for accept until one
	 * becomes free. 0 = allow as many ah as number of availble fds for
	 * the process */

	unsigned int count_threads;
	/**< CONTEXT: how many contexts to create in an array, 0 = 1 */
	unsigned int fd_limit_per_thread;
	/**< CONTEXT: nonzero means restrict each service thread to this
	 * many fds, 0 means the default which is divide the process fd
	 * limit by the number of threads. */
	unsigned int timeout_secs;
	/**< VHOST: various processes involving network roundtrips in the
	 * library are protected from hanging forever by timeouts.  If
	 * nonzero, this member lets you set the timeout used in seconds.
	 * Otherwise a default timeout is used. */
	const char *ecdh_curve;
	/**< VHOST: if NULL, defaults to initializing server with
	 *   "prime256v1" */
	const char *vhost_name;
	/**< VHOST: name of vhost, must match external DNS name used to
	 * access the site, like "warmcat.com" as it's used to match
	 * Host: header and / or SNI name for SSL. */
	const char * const *plugin_dirs;
	/**< CONTEXT: NULL, or NULL-terminated array of directories to
	 * scan for lws protocol plugins at context creation time */
	const struct lws_protocol_vhost_options *pvo;
	/**< VHOST: pointer to optional linked list of per-vhost
	 * options made accessible to protocols */
	int keepalive_timeout;
	/**< VHOST: (default = 0 = 5s) seconds to allow remote
	 * client to hold on to an idle HTTP/1.1 connection */
	const char *log_filepath;
	/**< VHOST: filepath to append logs to... this is opened before
	 *		any dropping of initial privileges */
	const struct lws_http_mount *mounts;
	/**< VHOST: optional linked list of mounts for this vhost */
	const char *server_string;
	/**< CONTEXT: string used in HTTP headers to identify server
 *		software, if NULL, "libwebsockets". */
	unsigned int pt_serv_buf_size;
	/**< CONTEXT: 0 = default of 4096.  This buffer is used by
	 * various service related features including file serving, it
	 * defines the max chunk of file that can be sent at once.
	 * At the risk of lws having to buffer failed large sends, it
	 * can be increased to, eg, 128KiB to improve throughput. */
	unsigned int max_http_header_data2;
	/**< CONTEXT: if max_http_header_data is 0 and this
	 * is nonzero, this will be used in place of the default.  It's
	 * like this for compatibility with the original short version,
	 * this is unsigned int length. */
	long ssl_options_set;
	/**< VHOST: Any bits set here will be set as server SSL options */
	long ssl_options_clear;
	/**< VHOST: Any bits set here will be cleared as server SSL options */
	unsigned short ws_ping_pong_interval;
	/**< CONTEXT: 0 for none, else interval in seconds between sending
	 * PINGs on idle websocket connections.  When the PING is sent,
	 * the PONG must come within the normal timeout_secs timeout period
	 * or the connection will be dropped.
	 * Any RX or TX traffic on the connection restarts the interval timer,
	 * so a connection which always sends or receives something at intervals
	 * less than the interval given here will never send PINGs / expect
	 * PONGs.  Conversely as soon as the ws connection is established, an
	 * idle connection will do the PING / PONG roundtrip as soon as
	 * ws_ping_pong_interval seconds has passed without traffic
	 */
	const struct lws_protocol_vhost_options *headers;
		/**< VHOST: pointer to optional linked list of per-vhost
		 * canned headers that are added to server responses */

	const struct lws_protocol_vhost_options *reject_service_keywords;
	/**< CONTEXT: Optional list of keywords and rejection codes + text.
	 *
	 * The keywords are checked for existing in the user agent string.
	 *
	 * Eg, "badrobot" "404 Not Found"
	 */
	void *external_baggage_free_on_destroy;
	/**< CONTEXT: NULL, or pointer to something externally malloc'd, that
	 * should be freed when the context is destroyed.  This allows you to
	 * automatically sync the freeing action to the context destruction
	 * action, so there is no need for an external free() if the context
	 * succeeded to create.
	 */

	const char *client_ssl_private_key_password;
	/**< VHOST: Client SSL context init: NULL or the passphrase needed
	 * for the private key */
	const char *client_ssl_cert_filepath;
	/**< VHOST: Client SSL context init:T he certificate the client
	 * should present to the peer on connection */
	const char *client_ssl_private_key_filepath;
	/**<  VHOST: Client SSL context init: filepath to client private key
	 * if this is set to NULL but client_ssl_cert_filepath is set, you
	 * can handle the LWS_CALLBACK_OPENSSL_LOAD_EXTRA_CLIENT_VERIFY_CERTS
	 * callback of protocols[0] to allow setting of the private key directly
	 * via openSSL library calls */
	const char *client_ssl_ca_filepath;
	/**< VHOST: Client SSL context init: CA certificate filepath or NULL */
	const void *client_ssl_ca_mem;
	/**< VHOST: Client SSL context init: CA certificate memory buffer or
	 * NULL... use this to load CA cert from memory instead of file */
	unsigned int client_ssl_ca_mem_len;
	/**< VHOST: Client SSL context init: length of client_ssl_ca_mem in
	 * bytes */

	const char *client_ssl_cipher_list;
	/**< VHOST: Client SSL context init: List of valid ciphers to use (eg,
	* "RC4-MD5:RC4-SHA:AES128-SHA:AES256-SHA:HIGH:!DSS:!aNULL"
	* or you can leave it as NULL to get "DEFAULT" */

	const struct lws_plat_file_ops *fops;
	/**< CONTEXT: NULL, or pointer to an array of fops structs, terminated
	 * by a sentinel with NULL .open.
	 *
	 * If NULL, lws provides just the platform file operations struct for
	 * backwards compatibility.
	 */
	int simultaneous_ssl_restriction;
	/**< CONTEXT: 0 (no limit) or limit of simultaneous SSL sessions
	 * possible.*/
	const char *socks_proxy_address;
	/**< VHOST: If non-NULL, attempts to proxy via the given address.
	 * If proxy auth is required, use format
	 * "username:password\@server:port" */
	unsigned int socks_proxy_port;
	/**< VHOST: If socks_proxy_address was non-NULL, uses this port */
#if defined(LWS_HAVE_SYS_CAPABILITY_H) && defined(LWS_HAVE_LIBCAP)
	cap_value_t caps[4];
	/**< CONTEXT: array holding Linux capabilities you want to
	 * continue to be available to the server after it transitions
	 * to a noprivileged user.  Usually none are needed but for, eg,
	 * .bind_iface, CAP_NET_RAW is required.  This gives you a way
	 * to still have the capability but drop root.
	 */
	char count_caps;
	/**< CONTEXT: count of Linux capabilities in .caps[].  0 means
	 * no capabilities will be inherited from root (the default) */
#endif
	int bind_iface;
	/**< VHOST: nonzero to strictly bind sockets to the interface name in
	 * .iface (eg, "eth2"), using SO_BIND_TO_DEVICE.
	 *
	 * Requires SO_BINDTODEVICE support from your OS and CAP_NET_RAW
	 * capability.
	 *
	 * Notice that common things like access network interface IP from
	 * your local machine use your lo / loopback interface and will be
	 * disallowed by this.
	 */
	int ssl_info_event_mask;
	/**< VHOST: mask of ssl events to be reported on LWS_CALLBACK_SSL_INFO
	 * callback for connections on this vhost.  The mask values are of
	 * the form SSL_CB_ALERT, defined in openssl/ssl.h.  The default of
	 * 0 means no info events will be reported.
	 */
	unsigned int timeout_secs_ah_idle;
	/**< VHOST: seconds to allow a client to hold an ah without using it.
	 * 0 defaults to 10s. */
	unsigned short ip_limit_ah;
	/**< CONTEXT: max number of ah a single IP may use simultaneously
	 *	      0 is no limit. This is a soft limit: if the limit is
	 *	      reached, connections from that IP will wait in the ah
	 *	      waiting list and not be able to acquire an ah until
	 *	      a connection belonging to the IP relinquishes one it
	 *	      already has.
	 */
	unsigned short ip_limit_wsi;
	/**< CONTEXT: max number of wsi a single IP may use simultaneously.
	 *	      0 is no limit.  This is a hard limit, connections from
	 *	      the same IP will simply be dropped once it acquires the
	 *	      amount of simultaneous wsi / accepted connections
	 *	      given here.
	 */
	uint32_t	http2_settings[7];
	/**< VHOST:  if http2_settings[0] is nonzero, the values given in
	 *	      http2_settings[1]..[6] are used instead of the lws
	 *	      platform default values.
	 *	      Just leave all at 0 if you don't care.
	 */
	const char *error_document_404;
	/**< VHOST: If non-NULL, when asked to serve a non-existent file,
	 *          lws attempts to server this url path instead.  Eg,
	 *          "/404.html" */
	const char *alpn;
	/**< CONTEXT: If non-NULL, default list of advertised alpn, comma-
	 *	      separated
	 *
	 *     VHOST: If non-NULL, per-vhost list of advertised alpn, comma-
	 *	      separated
	 */
	void **foreign_loops;
	/**< CONTEXT: This is ignored if the context is not being started with
	 *		an event loop, ie, .options has a flag like
	 *		LWS_SERVER_OPTION_LIBUV.
	 *
	 *		NULL indicates lws should start its own even loop for
	 *		each service thread, and deal with closing the loops
	 *		when the context is destroyed.
	 *
	 *		Non-NULL means it points to an array of external
	 *		("foreign") event loops that are to be used in turn for
	 *		each service thread.  In the default case of 1 service
	 *		thread, it can just point to one foreign event loop.
	 */
	void (*signal_cb)(void *event_lib_handle, int signum);
	/**< CONTEXT: NULL: default signal handling.  Otherwise this receives
	 *		the signal handler callback.  event_lib_handle is the
	 *		native event library signal handle, eg uv_signal_t *
	 *		for libuv.
	 */
	struct lws_context **pcontext;
	/**< CONTEXT: if non-NULL, at the end of context destroy processing,
	 * the pointer pointed to by pcontext is written with NULL.  You can
	 * use this to let foreign event loops know that lws context destruction
	 * is fully completed.
	 */
	void (*finalize)(struct lws_vhost *vh, void *arg);
	/**< VHOST: NULL, or pointer to function that will be called back
	 *	    when the vhost is just about to be freed.  The arg parameter
	 *	    will be set to whatever finalize_arg is below.
	 */
	void *finalize_arg;
	/**< VHOST: opaque pointer lws ignores but passes to the finalize
	 *	    callback.  If you don't care, leave it NULL.
	 */
	unsigned int max_http_header_pool2;
	/**< CONTEXT: if max_http_header_pool is 0 and this
	 * is nonzero, this will be used in place of the default.  It's
	 * like this for compatibility with the original short version:
	 * this is unsigned int length. */

	long ssl_client_options_set;
	/**< VHOST: Any bits set here will be set as CLIENT SSL options */
	long ssl_client_options_clear;
	/**< VHOST: Any bits set here will be cleared as CLIENT SSL options */

	const char *tls1_3_plus_cipher_list;
	/**< VHOST: List of valid ciphers to use for incoming server connections
	 * ON TLS1.3 AND ABOVE (eg, "TLS_CHACHA20_POLY1305_SHA256" on this vhost
	 * or you can leave it as NULL to get "DEFAULT".
	 * SEE .client_tls_1_3_plus_cipher_list to do the same on the vhost
	 * client SSL_CTX.
	 */
	const char *client_tls_1_3_plus_cipher_list;
	/**< VHOST: List of valid ciphers to use for outgoing client connections
	 * ON TLS1.3 AND ABOVE on this vhost (eg,
	 * "TLS_CHACHA20_POLY1305_SHA256") or you can leave it as NULL to get
	 * "DEFAULT".
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
 * lws_create_context() - Create the websocket handler
 * \param info:	pointer to struct with parameters
 *
 *	This function creates the listening socket (if serving) and takes care
 *	of all initialization in one step.
 *
 *	If option LWS_SERVER_OPTION_EXPLICIT_VHOSTS is given, no vhost is
 *	created; you're expected to create your own vhosts afterwards using
 *	lws_create_vhost().  Otherwise a vhost named "default" is also created
 *	using the information in the vhost-related members, for compatibility.
 *
 *	After initialization, it returns a struct lws_context * that
 *	represents this server.  After calling, user code needs to take care
 *	of calling lws_service() with the context pointer to get the
 *	server's sockets serviced.  This must be done in the same process
 *	context as the initialization call.
 *
 *	The protocol callback functions are called for a handful of events
 *	including http requests coming in, websocket connections becoming
 *	established, and data arriving; it's also called periodically to allow
 *	async transmission.
 *
 *	HTTP requests are sent always to the FIRST protocol in protocol, since
 *	at that time websocket protocol has not been negotiated.  Other
 *	protocols after the first one never see any HTTP callback activity.
 *
 *	The server created is a simple http server by default; part of the
 *	websocket standard is upgrading this http connection to a websocket one.
 *
 *	This allows the same server to provide files like scripts and favicon /
 *	images or whatever over http and dynamic data over websockets all in
 *	one place; they're all handled in the user callback.
 */
LWS_VISIBLE LWS_EXTERN struct lws_context *
lws_create_context(const struct lws_context_creation_info *info);


/**
 * lws_context_destroy() - Destroy the websocket context
 * \param context:	Websocket context
 *
 *	This function closes any active connections and then frees the
 *	context.  After calling this, any further use of the context is
 *	undefined.
 */
LWS_VISIBLE LWS_EXTERN void
lws_context_destroy(struct lws_context *context);

typedef int (*lws_reload_func)(void);

/**
 * lws_context_deprecate() - Deprecate the websocket context
 *
 * \param context:	Websocket context
 * \param cb: Callback notified when old context listen sockets are closed
 *
 *	This function is used on an existing context before superceding it
 *	with a new context.
 *
 *	It closes any listen sockets in the context, so new connections are
 *	not possible.
 *
 *	And it marks the context to be deleted when the number of active
 *	connections into it falls to zero.
 *
 *	This is aimed at allowing seamless configuration reloads.
 *
 *	The callback cb will be called after the listen sockets are actually
 *	closed and may be reopened.  In the callback the new context should be
 *	configured and created.  (With libuv, socket close happens async after
 *	more loop events).
 */
LWS_VISIBLE LWS_EXTERN void
lws_context_deprecate(struct lws_context *context, lws_reload_func cb);

LWS_VISIBLE LWS_EXTERN int
lws_context_is_deprecated(struct lws_context *context);

/**
 * lws_set_proxy() - Setups proxy to lws_context.
 * \param vhost:	pointer to struct lws_vhost you want set proxy for
 * \param proxy: pointer to c string containing proxy in format address:port
 *
 * Returns 0 if proxy string was parsed and proxy was setup.
 * Returns -1 if proxy is NULL or has incorrect format.
 *
 * This is only required if your OS does not provide the http_proxy
 * environment variable (eg, OSX)
 *
 *   IMPORTANT! You should call this function right after creation of the
 *   lws_context and before call to connect. If you call this
 *   function after connect behavior is undefined.
 *   This function will override proxy settings made on lws_context
 *   creation with genenv() call.
 */
LWS_VISIBLE LWS_EXTERN int
lws_set_proxy(struct lws_vhost *vhost, const char *proxy);

/**
 * lws_set_socks() - Setup socks to lws_context.
 * \param vhost:	pointer to struct lws_vhost you want set socks for
 * \param socks: pointer to c string containing socks in format address:port
 *
 * Returns 0 if socks string was parsed and socks was setup.
 * Returns -1 if socks is NULL or has incorrect format.
 *
 * This is only required if your OS does not provide the socks_proxy
 * environment variable (eg, OSX)
 *
 *   IMPORTANT! You should call this function right after creation of the
 *   lws_context and before call to connect. If you call this
 *   function after connect behavior is undefined.
 *   This function will override proxy settings made on lws_context
 *   creation with genenv() call.
 */
LWS_VISIBLE LWS_EXTERN int
lws_set_socks(struct lws_vhost *vhost, const char *socks);

struct lws_vhost;

/**
 * lws_create_vhost() - Create a vhost (virtual server context)
 * \param context:	pointer to result of lws_create_context()
 * \param info:		pointer to struct with parameters
 *
 * This function creates a virtual server (vhost) using the vhost-related
 * members of the info struct.  You can create many vhosts inside one context
 * if you created the context with the option LWS_SERVER_OPTION_EXPLICIT_VHOSTS
 */
LWS_VISIBLE LWS_EXTERN struct lws_vhost *
lws_create_vhost(struct lws_context *context,
		 const struct lws_context_creation_info *info);

/**
 * lws_vhost_destroy() - Destroy a vhost (virtual server context)
 *
 * \param vh:		pointer to result of lws_create_vhost()
 *
 * This function destroys a vhost.  Normally, if you just want to exit,
 * then lws_destroy_context() will take care of everything.  If you want
 * to destroy an individual vhost and all connections and allocations, you
 * can do it with this.
 *
 * If the vhost has a listen sockets shared by other vhosts, it will be given
 * to one of the vhosts sharing it rather than closed.
 *
 * The vhost close is staged according to the needs of the event loop, and if
 * there are multiple service threads.  At the point the vhost itself if
 * about to be freed, if you provided a finalize callback and optional arg at
 * vhost creation time, it will be called just before the vhost is freed.
 */
LWS_VISIBLE LWS_EXTERN void
lws_vhost_destroy(struct lws_vhost *vh);

/**
 * lwsws_get_config_globals() - Parse a JSON server config file
 * \param info:		pointer to struct with parameters
 * \param d:		filepath of the config file
 * \param config_strings: storage for the config strings extracted from JSON,
 * 			  the pointer is incremented as strings are stored
 * \param len:		pointer to the remaining length left in config_strings
 *			  the value is decremented as strings are stored
 *
 * This function prepares a n lws_context_creation_info struct with global
 * settings from a file d.
 *
 * Requires CMake option LWS_WITH_LEJP_CONF to have been enabled
 */
LWS_VISIBLE LWS_EXTERN int
lwsws_get_config_globals(struct lws_context_creation_info *info, const char *d,
			 char **config_strings, int *len);

/**
 * lwsws_get_config_vhosts() - Create vhosts from a JSON server config file
 * \param context:	pointer to result of lws_create_context()
 * \param info:		pointer to struct with parameters
 * \param d:		filepath of the config file
 * \param config_strings: storage for the config strings extracted from JSON,
 * 			  the pointer is incremented as strings are stored
 * \param len:		pointer to the remaining length left in config_strings
 *			  the value is decremented as strings are stored
 *
 * This function creates vhosts into a context according to the settings in
 *JSON files found in directory d.
 *
 * Requires CMake option LWS_WITH_LEJP_CONF to have been enabled
 */
LWS_VISIBLE LWS_EXTERN int
lwsws_get_config_vhosts(struct lws_context *context,
			struct lws_context_creation_info *info, const char *d,
			char **config_strings, int *len);

/** lws_vhost_get() - \deprecated deprecated: use lws_get_vhost() */
LWS_VISIBLE LWS_EXTERN struct lws_vhost *
lws_vhost_get(struct lws *wsi) LWS_WARN_DEPRECATED;

/**
 * lws_get_vhost() - return the vhost a wsi belongs to
 *
 * \param wsi: which connection
 */
LWS_VISIBLE LWS_EXTERN struct lws_vhost *
lws_get_vhost(struct lws *wsi);

/**
 * lws_get_vhost_name() - returns the name of a vhost
 *
 * \param vhost: which vhost
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_get_vhost_name(struct lws_vhost *vhost);

/**
 * lws_get_vhost_port() - returns the port a vhost listens on, or -1
 *
 * \param vhost: which vhost
 */
LWS_VISIBLE LWS_EXTERN int
lws_get_vhost_port(struct lws_vhost *vhost);

/**
 * lws_get_vhost_user() - returns the user pointer for the vhost
 *
 * \param vhost: which vhost
 */
LWS_VISIBLE LWS_EXTERN void *
lws_get_vhost_user(struct lws_vhost *vhost);

/**
 * lws_get_vhost_iface() - returns the binding for the vhost listen socket
 *
 * \param vhost: which vhost
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_get_vhost_iface(struct lws_vhost *vhost);

/**
 * lws_json_dump_vhost() - describe vhost state and stats in JSON
 *
 * \param vh: the vhost
 * \param buf: buffer to fill with JSON
 * \param len: max length of buf
 */
LWS_VISIBLE LWS_EXTERN int
lws_json_dump_vhost(const struct lws_vhost *vh, char *buf, int len);

/**
 * lws_json_dump_context() - describe context state and stats in JSON
 *
 * \param context: the context
 * \param buf: buffer to fill with JSON
 * \param len: max length of buf
 * \param hide_vhosts: nonzero to not provide per-vhost mount etc information
 *
 * Generates a JSON description of vhost state into buf
 */
LWS_VISIBLE LWS_EXTERN int
lws_json_dump_context(const struct lws_context *context, char *buf, int len,
		      int hide_vhosts);

/**
 * lws_vhost_user() - get the user data associated with the vhost
 * \param vhost: Websocket vhost
 *
 * This returns the optional user pointer that can be attached to
 * a vhost when it was created.  Lws never dereferences this pointer, it only
 * sets it when the vhost is created, and returns it using this api.
 */
LWS_VISIBLE LWS_EXTERN void *
lws_vhost_user(struct lws_vhost *vhost);

/**
 * lws_context_user() - get the user data associated with the context
 * \param context: Websocket context
 *
 * This returns the optional user allocation that can be attached to
 * the context the sockets live in at context_create time.  It's a way
 * to let all sockets serviced in the same context share data without
 * using globals statics in the user code.
 */
LWS_VISIBLE LWS_EXTERN void *
lws_context_user(struct lws_context *context);

/*! \defgroup vhost-mounts Vhost mounts and options
 * \ingroup context-and-vhost-creation
 *
 * ##Vhost mounts and options
 */
///@{
/** struct lws_protocol_vhost_options - linked list of per-vhost protocol
 * 					name=value options
 *
 * This provides a general way to attach a linked-list of name=value pairs,
 * which can also have an optional child link-list using the options member.
 */
struct lws_protocol_vhost_options {
	const struct lws_protocol_vhost_options *next; /**< linked list */
	const struct lws_protocol_vhost_options *options; /**< child linked-list of more options for this node */
	const char *name; /**< name of name=value pair */
	const char *value; /**< value of name=value pair */
};

/** enum lws_mount_protocols
 * This specifies the mount protocol for a mountpoint, whether it is to be
 * served from a filesystem, or it is a cgi etc.
 */
enum lws_mount_protocols {
	LWSMPRO_HTTP		= 0, /**< http reverse proxy */
	LWSMPRO_HTTPS		= 1, /**< https reverse proxy */
	LWSMPRO_FILE		= 2, /**< serve from filesystem directory */
	LWSMPRO_CGI		= 3, /**< pass to CGI to handle */
	LWSMPRO_REDIR_HTTP	= 4, /**< redirect to http:// url */
	LWSMPRO_REDIR_HTTPS	= 5, /**< redirect to https:// url */
	LWSMPRO_CALLBACK	= 6, /**< hand by named protocol's callback */
};

/** struct lws_http_mount
 *
 * arguments for mounting something in a vhost's url namespace
 */
struct lws_http_mount {
	const struct lws_http_mount *mount_next;
	/**< pointer to next struct lws_http_mount */
	const char *mountpoint;
	/**< mountpoint in http pathspace, eg, "/" */
	const char *origin;
	/**< path to be mounted, eg, "/var/www/warmcat.com" */
	const char *def;
	/**< default target, eg, "index.html" */
	const char *protocol;
	/**<"protocol-name" to handle mount */

	const struct lws_protocol_vhost_options *cgienv;
	/**< optional linked-list of cgi options.  These are created
	 * as environment variables for the cgi process
	 */
	const struct lws_protocol_vhost_options *extra_mimetypes;
	/**< optional linked-list of mimetype mappings */
	const struct lws_protocol_vhost_options *interpret;
	/**< optional linked-list of files to be interpreted */

	int cgi_timeout;
	/**< seconds cgi is allowed to live, if cgi://mount type */
	int cache_max_age;
	/**< max-age for reuse of client cache of files, seconds */
	unsigned int auth_mask;
	/**< bits set here must be set for authorized client session */

	unsigned int cache_reusable:1; /**< set if client cache may reuse this */
	unsigned int cache_revalidate:1; /**< set if client cache should revalidate on use */
	unsigned int cache_intermediaries:1; /**< set if intermediaries are allowed to cache */

	unsigned char origin_protocol; /**< one of enum lws_mount_protocols */
	unsigned char mountpoint_len; /**< length of mountpoint string */

	const char *basic_auth_login_file;
	/**<NULL, or filepath to use to check basic auth logins against */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility
	 *
	 * The below is to ensure later library versions with new
	 * members added above will see 0 (default) even if the app
	 * was not built against the newer headers.
	 */

	void *_unused[2]; /**< dummy */
};

///@}
///@}
