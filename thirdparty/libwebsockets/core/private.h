/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010 - 2018 Andy Green <andy@warmcat.com>
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

#include "lws_config.h"
#include "lws_config_private.h"

#if defined(LWS_WITH_CGI) && defined(LWS_HAVE_VFORK)
 #define  _GNU_SOURCE
#endif

#if defined(__COVERITY__) && !defined(LWS_COVERITY_WORKAROUND)
 #define LWS_COVERITY_WORKAROUND
 typedef float _Float32;
 typedef float _Float64;
 typedef float _Float128;
 typedef float _Float32x;
 typedef float _Float64x;
 typedef float _Float128x;
#endif

#ifdef LWS_HAVE_SYS_TYPES_H
 #include <sys/types.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <inttypes.h>

#if defined(LWS_WITH_ESP32)
 #define MSG_NOSIGNAL 0
 #define SOMAXCONN 3
#endif

#define STORE_IN_ROM
#include <assert.h>
#if LWS_MAX_SMP > 1
 #include <pthread.h>
#endif

#ifdef LWS_HAVE_SYS_STAT_H
 #include <sys/stat.h>
#endif

#if defined(WIN32) || defined(_WIN32)

 #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
 #endif

 #if (WINVER < 0x0501)
  #undef WINVER
  #undef _WIN32_WINNT
  #define WINVER 0x0501
  #define _WIN32_WINNT WINVER
 #endif

 #define LWS_NO_DAEMONIZE
 #define LWS_ERRNO WSAGetLastError()
 #define LWS_EAGAIN WSAEWOULDBLOCK
 #define LWS_EALREADY WSAEALREADY
 #define LWS_EINPROGRESS WSAEINPROGRESS
 #define LWS_EINTR WSAEINTR
 #define LWS_EISCONN WSAEISCONN
 #define LWS_EWOULDBLOCK WSAEWOULDBLOCK
 #define MSG_NOSIGNAL 0
 #define SHUT_RDWR SD_BOTH
 #define SOL_TCP IPPROTO_TCP
 #define SHUT_WR SD_SEND

 #define compatible_close(fd) closesocket(fd)
 #define lws_set_blocking_send(wsi) wsi->sock_send_blocking = 1
 #define LWS_SOCK_INVALID (INVALID_SOCKET)

 #include <winsock2.h>
 #include <ws2tcpip.h>
 #include <windows.h>
 #include <tchar.h>
 #ifdef LWS_HAVE_IN6ADDR_H
  #include <in6addr.h>
 #endif
 #include <mstcpip.h>
 #include <io.h>

 #if !defined(LWS_HAVE_ATOLL)
  #if defined(LWS_HAVE__ATOI64)
   #define atoll _atoi64
  #else
   #warning No atoll or _atoi64 available, using atoi
   #define atoll atoi
  #endif
 #endif

 #ifndef __func__
  #define __func__ __FUNCTION__
 #endif

 #ifdef LWS_HAVE__VSNPRINTF
  #define vsnprintf _vsnprintf
 #endif

 /* we don't have an implementation for this on windows... */
 int kill(int pid, int sig);
 int fork(void);
 #ifndef SIGINT
  #define SIGINT 2
 #endif

#else /* not windows --> */

 #include <fcntl.h>
 #include <strings.h>
 #include <unistd.h>
 #include <sys/types.h>

 #ifndef __cplusplus
  #include <errno.h>
 #endif
 #include <netdb.h>
 #include <signal.h>
 #include <sys/socket.h>

 #if defined(LWS_BUILTIN_GETIFADDRS)
  #include "./misc/getifaddrs.h"
 #else
  #if !defined(LWS_WITH_ESP32)
   #if defined(__HAIKU__)
    #define _BSD_SOURCE
   #endif
   #include <ifaddrs.h>
  #endif
 #endif
 #if defined (__ANDROID__)
  #include <syslog.h>
  #include <sys/resource.h>
 #elif defined (__sun) || defined(__HAIKU__) || defined(__QNX__)
  #include <syslog.h>
 #else
  #if !defined(LWS_WITH_ESP32)
   #include <sys/syslog.h>
  #endif
 #endif
 #include <netdb.h>
 #if !defined(LWS_WITH_ESP32)
  #include <sys/mman.h>
  #include <sys/un.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <poll.h>
 #endif
 #ifndef LWS_NO_FORK
  #ifdef LWS_HAVE_SYS_PRCTL_H
   #include <sys/prctl.h>
  #endif
 #endif

 #include <sys/time.h>

 #define LWS_ERRNO errno
 #define LWS_EAGAIN EAGAIN
 #define LWS_EALREADY EALREADY
 #define LWS_EINPROGRESS EINPROGRESS
 #define LWS_EINTR EINTR
 #define LWS_EISCONN EISCONN
 #define LWS_EWOULDBLOCK EWOULDBLOCK

 #define lws_set_blocking_send(wsi)

 #define LWS_SOCK_INVALID (-1)
#endif /* not windows */

#ifndef LWS_HAVE_BZERO
 #ifndef bzero
  #define bzero(b, len) (memset((b), '\0', (len)), (void) 0)
 #endif
#endif

#ifndef LWS_HAVE_STRERROR
 #define strerror(x) ""
#endif


#define lws_socket_is_valid(x) (x != LWS_SOCK_INVALID)

#include "libwebsockets.h"

#include "tls/private.h"

#if defined(WIN32) || defined(_WIN32)
 #include <gettimeofday.h>

 #ifndef BIG_ENDIAN
  #define BIG_ENDIAN    4321  /* to show byte order (taken from gcc) */
 #endif
 #ifndef LITTLE_ENDIAN
  #define LITTLE_ENDIAN 1234
 #endif
 #ifndef BYTE_ORDER
  #define BYTE_ORDER LITTLE_ENDIAN
 #endif

 #undef __P
 #ifndef __P
  #if __STDC__
   #define __P(protos) protos
  #else
   #define __P(protos) ()
  #endif
 #endif

#else /* not windows */
 static LWS_INLINE int compatible_close(int fd) { return close(fd); }

 #include <sys/stat.h>
 #include <sys/time.h>

 #if defined(__APPLE__)
  #include <machine/endian.h>
 #elif defined(__FreeBSD__)
  #include <sys/endian.h>
 #elif defined(__linux__)
  #include <endian.h>
 #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__QNX__)
	#include <gulliver.h>
	#if defined(__LITTLEENDIAN__)
		#define BYTE_ORDER __LITTLEENDIAN__
		#define LITTLE_ENDIAN __LITTLEENDIAN__
		#define BIG_ENDIAN 4321  /* to show byte order (taken from gcc); for suppres warning that BIG_ENDIAN is not defined. */
	#endif
	#if defined(__BIGENDIAN__)
		#define BYTE_ORDER __BIGENDIAN__
		#define LITTLE_ENDIAN 1234  /* to show byte order (taken from gcc); for suppres warning that LITTLE_ENDIAN is not defined. */
		#define BIG_ENDIAN __BIGENDIAN__
	#endif
#endif

#if defined(__sun) && defined(__GNUC__)

 #include <arpa/nameser_compat.h>

 #if !defined (BYTE_ORDER)
  #define BYTE_ORDER __BYTE_ORDER__
 #endif

 #if !defined(LITTLE_ENDIAN)
  #define LITTLE_ENDIAN __ORDER_LITTLE_ENDIAN__
 #endif

 #if !defined(BIG_ENDIAN)
  #define BIG_ENDIAN __ORDER_BIG_ENDIAN__
 #endif

#endif /* sun + GNUC */

#if !defined(BYTE_ORDER)
 #define BYTE_ORDER __BYTE_ORDER
#endif
#if !defined(LITTLE_ENDIAN)
 #define LITTLE_ENDIAN __LITTLE_ENDIAN
#endif
#if !defined(BIG_ENDIAN)
 #define BIG_ENDIAN __BIG_ENDIAN
#endif


/*
 * Mac OSX as well as iOS do not define the MSG_NOSIGNAL flag,
 * but happily have something equivalent in the SO_NOSIGPIPE flag.
 */
#ifdef __APPLE__
#define MSG_NOSIGNAL SO_NOSIGPIPE
#endif

/*
 * Solaris 11.X only supports POSIX 2001, MSG_NOSIGNAL appears in
 * POSIX 2008.
 */
#ifdef __sun
 #define MSG_NOSIGNAL 0
#endif

#ifdef _WIN32
 #ifndef FD_HASHTABLE_MODULUS
  #define FD_HASHTABLE_MODULUS 32
 #endif
#endif

#ifndef LWS_DEF_HEADER_LEN
#define LWS_DEF_HEADER_LEN 4096
#endif
#ifndef LWS_DEF_HEADER_POOL
#define LWS_DEF_HEADER_POOL 4
#endif
#ifndef LWS_MAX_PROTOCOLS
#define LWS_MAX_PROTOCOLS 5
#endif
#ifndef LWS_MAX_EXTENSIONS_ACTIVE
#define LWS_MAX_EXTENSIONS_ACTIVE 1
#endif
#ifndef LWS_MAX_EXT_OFFERS
#define LWS_MAX_EXT_OFFERS 8
#endif
#ifndef SPEC_LATEST_SUPPORTED
#define SPEC_LATEST_SUPPORTED 13
#endif
#ifndef AWAITING_TIMEOUT
#define AWAITING_TIMEOUT 20
#endif
#ifndef CIPHERS_LIST_STRING
#define CIPHERS_LIST_STRING "DEFAULT"
#endif
#ifndef LWS_SOMAXCONN
#define LWS_SOMAXCONN SOMAXCONN
#endif

#define MAX_WEBSOCKET_04_KEY_LEN 128

#ifndef SYSTEM_RANDOM_FILEPATH
#define SYSTEM_RANDOM_FILEPATH "/dev/urandom"
#endif

#define LWS_H2_RX_SCRATCH_SIZE 512

#if defined(WIN32) || defined(_WIN32)
	 // Visual studio older than 2015 and WIN_CE has only _stricmp
	#if (defined(_MSC_VER) && _MSC_VER < 1900) || defined(_WIN32_WCE)
	#define strcasecmp _stricmp
	#elif !defined(__MINGW32__)
	#define strcasecmp stricmp
	#endif
	#define getdtablesize() 30000
#endif

/*
 * All lws_tls...() functions must return this type, converting the
 * native backend result and doing the extra work to determine which one
 * as needed.
 *
 * Native TLS backend return codes are NOT ALLOWED outside the backend.
 *
 * Non-SSL mode also uses these types.
 */
enum lws_ssl_capable_status {
	LWS_SSL_CAPABLE_ERROR = -1,		 /* it failed */
	LWS_SSL_CAPABLE_DONE = 0,		 /* it succeeded */
	LWS_SSL_CAPABLE_MORE_SERVICE_READ = -2,	 /* retry WANT_READ */
	LWS_SSL_CAPABLE_MORE_SERVICE_WRITE = -3,  /* retry WANT_WRITE */
	LWS_SSL_CAPABLE_MORE_SERVICE = -4,	 /* general retry */
};

#if defined(__clang__)
#define lws_memory_barrier() __sync_synchronize()
#elif defined(__GNUC__)
#define lws_memory_barrier() __sync_synchronize()
#else
#define lws_memory_barrier()
#endif

/*
 *
 *  ------ roles ------
 *
 */

#include "roles/private.h"

/* null-terminated array of pointers to roles lws built with */
extern const struct lws_role_ops *available_roles[];

#define LWS_FOR_EVERY_AVAILABLE_ROLE_START(xx) { \
		const struct lws_role_ops **ppxx = available_roles; \
		while (*ppxx) { \
			const struct lws_role_ops *xx = *ppxx++;

#define LWS_FOR_EVERY_AVAILABLE_ROLE_END }}

/*
 *
 *  ------ event_loop ops ------
 *
 */

#include "event-libs/private.h"

/* enums of socks version */
enum socks_version {
	SOCKS_VERSION_4 = 4,
	SOCKS_VERSION_5 = 5
};

/* enums of subnegotiation version */
enum socks_subnegotiation_version {
	SOCKS_SUBNEGOTIATION_VERSION_1 = 1,
};

/* enums of socks commands */
enum socks_command {
	SOCKS_COMMAND_CONNECT = 1,
	SOCKS_COMMAND_BIND = 2,
	SOCKS_COMMAND_UDP_ASSOCIATE = 3
};

/* enums of socks address type */
enum socks_atyp {
	SOCKS_ATYP_IPV4 = 1,
	SOCKS_ATYP_DOMAINNAME = 3,
	SOCKS_ATYP_IPV6 = 4
};

/* enums of socks authentication methods */
enum socks_auth_method {
	SOCKS_AUTH_NO_AUTH = 0,
	SOCKS_AUTH_GSSAPI = 1,
	SOCKS_AUTH_USERNAME_PASSWORD = 2
};

/* enums of subnegotiation status */
enum socks_subnegotiation_status {
	SOCKS_SUBNEGOTIATION_STATUS_SUCCESS = 0,
};

/* enums of socks request reply */
enum socks_request_reply {
	SOCKS_REQUEST_REPLY_SUCCESS = 0,
	SOCKS_REQUEST_REPLY_FAILURE_GENERAL = 1,
	SOCKS_REQUEST_REPLY_CONNECTION_NOT_ALLOWED = 2,
	SOCKS_REQUEST_REPLY_NETWORK_UNREACHABLE = 3,
	SOCKS_REQUEST_REPLY_HOST_UNREACHABLE = 4,
	SOCKS_REQUEST_REPLY_CONNECTION_REFUSED = 5,
	SOCKS_REQUEST_REPLY_TTL_EXPIRED = 6,
	SOCKS_REQUEST_REPLY_COMMAND_NOT_SUPPORTED = 7,
	SOCKS_REQUEST_REPLY_ATYP_NOT_SUPPORTED = 8
};

/* enums used to generate socks messages */
enum socks_msg_type {
	/* greeting */
	SOCKS_MSG_GREETING,
	/* credential, user name and password */
	SOCKS_MSG_USERNAME_PASSWORD,
	/* connect command */
	SOCKS_MSG_CONNECT
};

enum {
	LWS_RXFLOW_ALLOW = (1 << 0),
	LWS_RXFLOW_PENDING_CHANGE = (1 << 1),
};

struct lws_ring {
	void *buf;
	void (*destroy_element)(void *element);
	uint32_t buflen;
	uint32_t element_len;
	uint32_t head;
	uint32_t oldest_tail;
};

struct lws_protocols;
struct lws;

struct lws_io_watcher {
#ifdef LWS_WITH_LIBEV
	struct lws_io_watcher_libev ev;
#endif
#ifdef LWS_WITH_LIBUV
	struct lws_io_watcher_libuv uv;
#endif
#ifdef LWS_WITH_LIBEVENT
	struct lws_io_watcher_libevent event;
#endif
	struct lws_context *context;

	uint8_t actual_events;
};

struct lws_signal_watcher {
#ifdef LWS_WITH_LIBEV
	struct lws_signal_watcher_libev ev;
#endif
#ifdef LWS_WITH_LIBUV
	struct lws_signal_watcher_libuv uv;
#endif
#ifdef LWS_WITH_LIBEVENT
	struct lws_signal_watcher_libevent event;
#endif
	struct lws_context *context;
};

#ifdef _WIN32
#define LWS_FD_HASH(fd) ((fd ^ (fd >> 8) ^ (fd >> 16)) % FD_HASHTABLE_MODULUS)
struct lws_fd_hashtable {
	struct lws **wsi;
	int length;
};
#endif

struct lws_foreign_thread_pollfd {
	struct lws_foreign_thread_pollfd *next;
	int fd_index;
	int _and;
	int _or;
};


#define LWS_HRTIMER_NOWAIT (0x7fffffffffffffffll)

/*
 * so we can have n connections being serviced simultaneously,
 * these things need to be isolated per-thread.
 */

struct lws_context_per_thread {
#if LWS_MAX_SMP > 1
	pthread_mutex_t lock;
	pthread_mutex_t lock_stats;
	pthread_t lock_owner;
	const char *last_lock_reason;
#endif

	struct lws_context *context;

	/*
	 * usable by anything in the service code, but only if the scope
	 * does not last longer than the service action (since next service
	 * of any socket can likewise use it and overwrite)
	 */
	unsigned char *serv_buf;

	struct lws_dll_lws dll_head_timeout;
	struct lws_dll_lws dll_head_hrtimer;
	struct lws_dll_lws dll_head_buflist; /* guys with pending rxflow */

#if defined(LWS_WITH_TLS)
	struct lws_pt_tls tls;
#endif

	struct lws_pollfd *fds;
	volatile struct lws_foreign_thread_pollfd * volatile foreign_pfd_list;
#ifdef _WIN32
	WSAEVENT *events;
#endif
	lws_sockfd_type dummy_pipe_fds[2];
	struct lws *pipe_wsi;

	/* --- role based members --- */

#if defined(LWS_ROLE_WS) && !defined(LWS_WITHOUT_EXTENSIONS)
	struct lws_pt_role_ws ws;
#endif
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	struct lws_pt_role_http http;
#endif

	/* --- event library based members --- */

#if defined(LWS_WITH_LIBEV)
	struct lws_pt_eventlibs_libev ev;
#endif
#if defined(LWS_WITH_LIBUV)
	struct lws_pt_eventlibs_libuv uv;
#endif
#if defined(LWS_WITH_LIBEVENT)
	struct lws_pt_eventlibs_libevent event;
#endif

#if defined(LWS_WITH_LIBEV) || defined(LWS_WITH_LIBUV) || defined(LWS_WITH_LIBEVENT)
	struct lws_signal_watcher w_sigint;
#endif

	/* --- */

	unsigned long count_conns;
	unsigned int fds_count;

	volatile unsigned char inside_poll;
	volatile unsigned char foreign_spinlock;

	unsigned char tid;

	unsigned char lock_depth;
	unsigned char inside_service:1;
	unsigned char event_loop_foreign:1;
	unsigned char event_loop_destroy_processing_done:1;
};

struct lws_conn_stats {
	unsigned long long rx, tx;
	unsigned long h1_conn, h1_trans, h2_trans, ws_upg, h2_alpn, h2_subs,
		      h2_upg, rejected;
};

void
lws_sum_stats(const struct lws_context *ctx, struct lws_conn_stats *cs);

struct lws_timed_vh_protocol {
	struct lws_timed_vh_protocol *next;
	const struct lws_protocols *protocol;
	time_t time;
	int reason;
};

/*
 * virtual host -related context information
 *   vhostwide SSL context
 *   vhostwide proxy
 *
 * hierarchy:
 *
 * context -> vhost -> wsi
 *
 * incoming connection non-SSL vhost binding:
 *
 *    listen socket -> wsi -> select vhost after first headers
 *
 * incoming connection SSL vhost binding:
 *
 *    SSL SNI -> wsi -> bind after SSL negotiation
 */


struct lws_vhost {
#if !defined(LWS_WITHOUT_CLIENT)
	char proxy_basic_auth_token[128];
#endif
#if LWS_MAX_SMP > 1
	pthread_mutex_t lock;
#endif

#if defined(LWS_ROLE_H2)
	struct lws_vhost_role_h2 h2;
#endif
#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	struct lws_vhost_role_http http;
#endif
#if defined(LWS_ROLE_WS) && !defined(LWS_WITHOUT_EXTENSIONS)
	struct lws_vhost_role_ws ws;
#endif

#if defined(LWS_WITH_SOCKS5)
	char socks_proxy_address[128];
	char socks_user[96];
	char socks_password[96];
#endif
#if defined(LWS_WITH_LIBEV)
	struct lws_io_watcher w_accept;
#endif
	struct lws_conn_stats conn_stats;
	struct lws_context *context;
	struct lws_vhost *vhost_next;

	struct lws *lserv_wsi;
	const char *name;
	const char *iface;

#if !defined(LWS_WITH_ESP32) && !defined(OPTEE_TA) && !defined(WIN32)
	int bind_iface;
#endif
	const struct lws_protocols *protocols;
	void **protocol_vh_privs;
	const struct lws_protocol_vhost_options *pvo;
	const struct lws_protocol_vhost_options *headers;
	struct lws **same_vh_protocol_list;
	struct lws_vhost *no_listener_vhost_list;
#if !defined(LWS_NO_CLIENT)
	struct lws_dll_lws dll_active_client_conns;
#endif

#if defined(LWS_WITH_TLS)
	struct lws_vhost_tls tls;
#endif

	struct lws_timed_vh_protocol *timed_vh_protocol_list;
	void *user;

	int listen_port;

#if defined(LWS_WITH_SOCKS5)
	unsigned int socks_proxy_port;
#endif
	unsigned int options;
	int count_protocols;
	int ka_time;
	int ka_probes;
	int ka_interval;
	int keepalive_timeout;
	int timeout_secs_ah_idle;

#ifdef LWS_WITH_ACCESS_LOG
	int log_fd;
#endif

	unsigned int created_vhost_protocols:1;
	unsigned int being_destroyed:1;

	unsigned char default_protocol_index;
	unsigned char raw_protocol_index;
};

struct lws_deferred_free
{
	struct lws_deferred_free *next;
	time_t deadline;
	void *payload;
};

typedef union {
#ifdef LWS_WITH_IPV6
	struct sockaddr_in6 sa6;
#endif
	struct sockaddr_in sa4;
} sockaddr46;


#if defined(LWS_WITH_PEER_LIMITS)
struct lws_peer {
	struct lws_peer *next;
	struct lws_peer *peer_wait_list;

	time_t time_created;
	time_t time_closed_all;

	uint8_t addr[32];
	uint32_t hash;
	uint32_t count_wsi;
	uint32_t total_wsi;

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	struct lws_peer_role_http http;
#endif

	uint8_t af;
};
#endif

/*
 * the rest is managed per-context, that includes
 *
 *  - processwide single fd -> wsi lookup
 *  - contextwide headers pool
 */

struct lws_context {
	time_t last_timeout_check_s;
	time_t last_ws_ping_pong_check_s;
	time_t time_up;
	time_t time_discontiguity;
	time_t time_fixup;
	const struct lws_plat_file_ops *fops;
	struct lws_plat_file_ops fops_platform;
	struct lws_context **pcontext_finalize;

	const struct lws_tls_ops *tls_ops;

#if defined(LWS_WITH_HTTP2)
	struct http2_settings set;
#endif
#if defined(LWS_WITH_ZIP_FOPS)
	struct lws_plat_file_ops fops_zip;
#endif
	struct lws_context_per_thread pt[LWS_MAX_SMP];
	struct lws_conn_stats conn_stats;
#if LWS_MAX_SMP > 1
	pthread_mutex_t lock;
	int lock_depth;
#endif
#ifdef _WIN32
/* different implementation between unix and windows */
	struct lws_fd_hashtable fd_hashtable[FD_HASHTABLE_MODULUS];
#else
	struct lws **lws_lookup;  /* fd to wsi */
#endif
	struct lws_vhost *vhost_list;
	struct lws_vhost *no_listener_vhost_list;
	struct lws_vhost *vhost_pending_destruction_list;
	struct lws_plugin *plugin_list;
	struct lws_deferred_free *deferred_free_list;
#if defined(LWS_WITH_PEER_LIMITS)
	struct lws_peer **pl_hash_table;
	struct lws_peer *peer_wait_list;
	time_t next_cull;
#endif

	void *external_baggage_free_on_destroy;
	const struct lws_token_limits *token_limits;
	void *user_space;
	const struct lws_protocol_vhost_options *reject_service_keywords;
	lws_reload_func deprecation_cb;
	void (*eventlib_signal_cb)(void *event_lib_handle, int signum);

#if defined(LWS_HAVE_SYS_CAPABILITY_H) && defined(LWS_HAVE_LIBCAP)
	cap_value_t caps[4];
	char count_caps;
#endif

#if defined(LWS_WITH_LIBEV)
	struct lws_context_eventlibs_libev ev;
#endif
#if defined(LWS_WITH_LIBUV)
	struct lws_context_eventlibs_libuv uv;
#endif
#if defined(LWS_WITH_LIBEVENT)
	struct lws_context_eventlibs_libevent event;
#endif
	struct lws_event_loop_ops *event_loop_ops;


#if defined(LWS_WITH_TLS)
	struct lws_context_tls tls;
#endif

	char canonical_hostname[128];
	const char *server_string;

#ifdef LWS_LATENCY
	unsigned long worst_latency;
	char worst_latency_info[256];
#endif

#if defined(LWS_WITH_STATS)
	uint64_t lws_stats[LWSSTATS_SIZE];
	uint64_t last_dump;
	int updated;
#endif
#if defined(LWS_WITH_ESP32)
	unsigned long time_last_state_dump;
	uint32_t last_free_heap;
#endif

	int max_fds;
	int count_event_loop_static_asset_handles;
	int started_with_parent;
	int uid, gid;

	int fd_random;

	int count_wsi_allocated;
	int count_cgi_spawned;
	unsigned int options;
	unsigned int fd_limit_per_thread;
	unsigned int timeout_secs;
	unsigned int pt_serv_buf_size;
	int max_http_header_data;
	int max_http_header_pool;
	int simultaneous_ssl_restriction;
	int simultaneous_ssl;
#if defined(LWS_WITH_PEER_LIMITS)
	uint32_t pl_hash_elements;	/* protected by context->lock */
	uint32_t count_peers;		/* protected by context->lock */
	unsigned short ip_limit_ah;
	unsigned short ip_limit_wsi;
#endif
	unsigned int deprecated:1;
	unsigned int being_destroyed:1;
	unsigned int being_destroyed1:1;
	unsigned int being_destroyed2:1;
	unsigned int requested_kill:1;
	unsigned int protocol_init_done:1;
	unsigned int doing_protocol_init:1;
	unsigned int done_protocol_destroy_cb:1;
	unsigned int finalize_destroy_after_internal_loops_stopped:1;
	/*
	 * set to the Thread ID that's doing the service loop just before entry
	 * to poll indicates service thread likely idling in poll()
	 * volatile because other threads may check it as part of processing
	 * for pollfd event change.
	 */
	volatile int service_tid;
	int service_tid_detected;

	short count_threads;
	short plugin_protocol_count;
	short plugin_extension_count;
	short server_string_len;
	unsigned short ws_ping_pong_interval;
	unsigned short deprecation_pending_listen_close_count;

	uint8_t max_fi;
};

int
lws_check_deferred_free(struct lws_context *context, int force);

#define lws_get_context_protocol(ctx, x) ctx->vhost_list->protocols[x]
#define lws_get_vh_protocol(vh, x) vh->protocols[x]

LWS_EXTERN void
__lws_close_free_wsi_final(struct lws *wsi);
LWS_EXTERN void
lws_libuv_closehandle(struct lws *wsi);
LWS_EXTERN int
lws_libuv_check_watcher_active(struct lws *wsi);

LWS_VISIBLE LWS_EXTERN int
lws_plat_plugins_init(struct lws_context * context, const char * const *d);

LWS_VISIBLE LWS_EXTERN int
lws_plat_plugins_destroy(struct lws_context * context);

LWS_EXTERN void
lws_restart_ws_ping_pong_timer(struct lws *wsi);

struct lws *
lws_adopt_socket_vhost(struct lws_vhost *vh, lws_sockfd_type accept_fd);

int
lws_jws_base64_enc(const char *in, size_t in_len, char *out, size_t out_max);

void
lws_vhost_destroy1(struct lws_vhost *vh);

enum {
	LWS_EV_READ = (1 << 0),
	LWS_EV_WRITE = (1 << 1),
	LWS_EV_START = (1 << 2),
	LWS_EV_STOP = (1 << 3),

	LWS_EV_PREPARE_DELETION = (1 << 31),
};


#if defined(LWS_WITH_ESP32)
LWS_EXTERN int
lws_find_string_in_file(const char *filename, const char *string, int stringlen);
#endif

#ifdef LWS_WITH_IPV6
#define LWS_IPV6_ENABLED(vh) \
	(!lws_check_opt(vh->context->options, LWS_SERVER_OPTION_DISABLE_IPV6) && \
	 !lws_check_opt(vh->options, LWS_SERVER_OPTION_DISABLE_IPV6))
#else
#define LWS_IPV6_ENABLED(context) (0)
#endif

#ifdef LWS_WITH_UNIX_SOCK
#define LWS_UNIX_SOCK_ENABLED(vhost) \
	(vhost->options & LWS_SERVER_OPTION_UNIX_SOCK)
#else
#define LWS_UNIX_SOCK_ENABLED(vhost) (0)
#endif

enum uri_path_states {
	URIPS_IDLE,
	URIPS_SEEN_SLASH,
	URIPS_SEEN_SLASH_DOT,
	URIPS_SEEN_SLASH_DOT_DOT,
};

enum uri_esc_states {
	URIES_IDLE,
	URIES_SEEN_PERCENT,
	URIES_SEEN_PERCENT_H1,
};


#ifndef LWS_NO_CLIENT
struct client_info_stash {
	char *address;
	char *path;
	char *host;
	char *origin;
	char *protocol;
	char *method;
	char *iface;
	char *alpn;
};
#endif


signed char char_to_hex(const char c);


struct lws_buflist {
	struct lws_buflist *next;

	size_t len;
	size_t pos;

	uint8_t buf[1]; /* true length of this is set by the oversize malloc */
};

#define lws_wsi_is_udp(___wsi) (!!___wsi->udp)

#define LWS_H2_FRAME_HEADER_LENGTH 9


struct lws {
	/* structs */

#if defined(LWS_ROLE_H1) || defined(LWS_ROLE_H2)
	struct _lws_http_mode_related http;
#endif
#if defined(LWS_ROLE_H2)
	struct _lws_h2_related h2;
#endif
#if defined(LWS_ROLE_WS)
	struct _lws_websocket_related *ws; /* allocated if we upgrade to ws */
#endif

	const struct lws_role_ops *role_ops;
	lws_wsi_state_t	wsistate;
	lws_wsi_state_t wsistate_pre_close;

	/* lifetime members */

#if defined(LWS_WITH_LIBEV) || defined(LWS_WITH_LIBUV) || defined(LWS_WITH_LIBEVENT)
	struct lws_io_watcher w_read;
#endif
#if defined(LWS_WITH_LIBEV) || defined(LWS_WITH_LIBEVENT)
	struct lws_io_watcher w_write;
#endif

	/* pointers */

	struct lws_context *context;
	struct lws_vhost *vhost;
	struct lws *parent; /* points to parent, if any */
	struct lws *child_list; /* points to first child */
	struct lws *sibling_list; /* subsequent children at same level */

	const struct lws_protocols *protocol;
	struct lws **same_vh_protocol_prev, *same_vh_protocol_next;

	struct lws_dll_lws dll_timeout;
	struct lws_dll_lws dll_hrtimer;
	struct lws_dll_lws dll_buflist; /* guys with pending rxflow */

#if defined(LWS_WITH_PEER_LIMITS)
	struct lws_peer *peer;
#endif

	struct lws_udp *udp;
#ifndef LWS_NO_CLIENT
	struct client_info_stash *stash;
	char *client_hostname_copy;
	struct lws_dll_lws dll_active_client_conns;
	struct lws_dll_lws dll_client_transaction_queue_head;
	struct lws_dll_lws dll_client_transaction_queue;
#endif
	void *user_space;
	void *opaque_parent_data;

	struct lws_buflist *buflist;

	/* truncated send handling */
	unsigned char *trunc_alloc; /* non-NULL means buffering in progress */

#if defined(LWS_WITH_TLS)
	struct lws_lws_tls tls;
#endif

	lws_sock_file_fd_type desc; /* .filefd / .sockfd */
#if defined(LWS_WITH_STATS)
	uint64_t active_writable_req_us;
#if defined(LWS_WITH_TLS)
	uint64_t accept_start_us;
#endif
#endif

	lws_usec_t pending_timer; /* hrtimer fires */
	time_t pending_timeout_set; /* second-resolution timeout start */

#ifdef LWS_LATENCY
	unsigned long action_start;
	unsigned long latency_start;
#endif

	/* ints */
#define LWS_NO_FDS_POS (-1)
	int position_in_fds_table;
	unsigned int trunc_alloc_len; /* size of malloc */
	unsigned int trunc_offset; /* where we are in terms of spilling */
	unsigned int trunc_len; /* how much is buffered */
#ifndef LWS_NO_CLIENT
	int chunk_remaining;
#endif
	unsigned int cache_secs;

	unsigned int hdr_parsing_completed:1;
	unsigned int http2_substream:1;
	unsigned int upgraded_to_http2:1;
	unsigned int h2_stream_carries_ws:1;
	unsigned int seen_nonpseudoheader:1;
	unsigned int listener:1;
	unsigned int user_space_externally_allocated:1;
	unsigned int socket_is_permanently_unusable:1;
	unsigned int rxflow_change_to:2;
	unsigned int conn_stat_done:1;
	unsigned int cache_reuse:1;
	unsigned int cache_revalidate:1;
	unsigned int cache_intermediaries:1;
	unsigned int favoured_pollin:1;
	unsigned int sending_chunked:1;
	unsigned int interpreting:1;
	unsigned int already_did_cce:1;
	unsigned int told_user_closed:1;
	unsigned int told_event_loop_closed:1;
	unsigned int waiting_to_send_close_frame:1;
	unsigned int close_needs_ack:1;
	unsigned int ipv6:1;
	unsigned int parent_carries_io:1;
	unsigned int parent_pending_cb_on_writable:1;
	unsigned int cgi_stdout_zero_length:1;
	unsigned int seen_zero_length_recv:1;
	unsigned int rxflow_will_be_applied:1;
	unsigned int event_pipe:1;
	unsigned int on_same_vh_list:1;
	unsigned int handling_404:1;
	unsigned int protocol_bind_balance:1;

	unsigned int could_have_pending:1; /* detect back-to-back writes */
	unsigned int outer_will_close:1;

#ifdef LWS_WITH_ACCESS_LOG
	unsigned int access_log_pending:1;
#endif
#ifndef LWS_NO_CLIENT
	unsigned int do_ws:1; /* whether we are doing http or ws flow */
	unsigned int chunked:1; /* if the clientside connection is chunked */
	unsigned int client_rx_avail:1;
	unsigned int client_http_body_pending:1;
	unsigned int transaction_from_pipeline_queue:1;
	unsigned int keepalive_active:1;
	unsigned int keepalive_rejected:1;
	unsigned int client_pipeline:1;
	unsigned int client_h2_alpn:1;
	unsigned int client_h2_substream:1;
#endif

#ifdef _WIN32
	unsigned int sock_send_blocking:1;
#endif

#ifndef LWS_NO_CLIENT
	unsigned short c_port;
#endif
	unsigned short pending_timeout_limit;

	/* chars */

	char lws_rx_parse_state; /* enum lws_rx_parse_state */
	char rx_frame_type; /* enum lws_write_protocol */
	char pending_timeout; /* enum pending_timeout */
	char tsi; /* thread service index we belong to */
	char protocol_interpret_idx;
	char redirects;
	uint8_t rxflow_bitmap;
#ifdef LWS_WITH_CGI
	char cgi_channel; /* which of stdin/out/err */
	char hdr_state;
#endif
#ifndef LWS_NO_CLIENT
	char chunk_parser; /* enum lws_chunk_parser */
#endif
#if defined(LWS_WITH_CGI) || !defined(LWS_NO_CLIENT)
	char reason_bf; /* internal writeable callback reason bitfield */
#endif
#if defined(LWS_WITH_STATS) && defined(LWS_WITH_TLS)
	char seen_rx;
#endif
	uint8_t ws_over_h2_count;
	/* volatile to make sure code is aware other thread can change */
	volatile char handling_pollout;
	volatile char leave_pollout_active;
};

#define lws_is_flowcontrolled(w) (!!(wsi->rxflow_bitmap))

void
lws_service_do_ripe_rxflow(struct lws_context_per_thread *pt);

LWS_EXTERN int log_level;

LWS_EXTERN int
lws_socket_bind(struct lws_vhost *vhost, lws_sockfd_type sockfd, int port,
		const char *iface);

#if defined(LWS_WITH_IPV6)
LWS_EXTERN unsigned long
lws_get_addr_scope(const char *ipaddr);
#endif

LWS_EXTERN void
lws_close_free_wsi(struct lws *wsi, enum lws_close_status, const char *caller);
LWS_EXTERN void
__lws_close_free_wsi(struct lws *wsi, enum lws_close_status, const char *caller);

LWS_EXTERN void
__lws_free_wsi(struct lws *wsi);

LWS_EXTERN int
__remove_wsi_socket_from_fds(struct lws *wsi);
LWS_EXTERN int
lws_rxflow_cache(struct lws *wsi, unsigned char *buf, int n, int len);

#ifndef LWS_LATENCY
static LWS_INLINE void
lws_latency(struct lws_context *context, struct lws *wsi, const char *action,
	    int ret, int completion) {
	do {
		(void)context; (void)wsi; (void)action; (void)ret;
		(void)completion;
	} while (0);
}
static LWS_INLINE void
lws_latency_pre(struct lws_context *context, struct lws *wsi) {
	do { (void)context; (void)wsi; } while (0);
}
#else
#define lws_latency_pre(_context, _wsi) lws_latency(_context, _wsi, NULL, 0, 0)
extern void
lws_latency(struct lws_context *context, struct lws *wsi, const char *action,
	    int ret, int completion);
#endif

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ws_client_rx_sm(struct lws *wsi, unsigned char c);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_parse(struct lws *wsi, unsigned char *buf, int *len);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_parse_urldecode(struct lws *wsi, uint8_t *_c);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_http_action(struct lws *wsi);

LWS_EXTERN int
lws_b64_selftest(void);

LWS_EXTERN int
lws_service_flag_pending(struct lws_context *context, int tsi);

LWS_EXTERN int
lws_timed_callback_remove(struct lws_vhost *vh, struct lws_timed_vh_protocol *p);

#if defined(_WIN32)
LWS_EXTERN struct lws *
wsi_from_fd(const struct lws_context *context, lws_sockfd_type fd);

LWS_EXTERN int
insert_wsi(struct lws_context *context, struct lws *wsi);

LWS_EXTERN int
delete_from_fd(struct lws_context *context, lws_sockfd_type fd);
#else
#define wsi_from_fd(A,B)  A->lws_lookup[B - lws_plat_socket_offset()]
#define insert_wsi(A,B)   assert(A->lws_lookup[B->desc.sockfd - lws_plat_socket_offset()] == 0); A->lws_lookup[B->desc.sockfd - lws_plat_socket_offset()]=B
#define delete_from_fd(A,B) A->lws_lookup[B - lws_plat_socket_offset()]=0
#endif

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
__insert_wsi_socket_into_fds(struct lws_context *context, struct lws *wsi);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_issue_raw(struct lws *wsi, unsigned char *buf, size_t len);

LWS_EXTERN void
lws_remove_from_timeout_list(struct lws *wsi);

LWS_EXTERN struct lws * LWS_WARN_UNUSED_RESULT
lws_client_connect_2(struct lws *wsi);

LWS_VISIBLE struct lws * LWS_WARN_UNUSED_RESULT
lws_client_reset(struct lws **wsi, int ssl, const char *address, int port,
		 const char *path, const char *host);

LWS_EXTERN struct lws * LWS_WARN_UNUSED_RESULT
lws_create_new_server_wsi(struct lws_vhost *vhost, int fixed_tsi);

LWS_EXTERN char * LWS_WARN_UNUSED_RESULT
lws_generate_client_handshake(struct lws *wsi, char *pkt);

LWS_EXTERN int
lws_handle_POLLOUT_event(struct lws *wsi, struct lws_pollfd *pollfd);

LWS_EXTERN struct lws *
lws_client_connect_via_info2(struct lws *wsi);



LWS_EXTERN void
lws_client_stash_destroy(struct lws *wsi);

/*
 * EXTENSIONS
 */

#if defined(LWS_WITHOUT_EXTENSIONS)
#define lws_any_extension_handled(_a, _b, _c, _d) (0)
#define lws_ext_cb_active(_a, _b, _c, _d) (0)
#define lws_ext_cb_all_exts(_a, _b, _c, _d, _e) (0)
#define lws_issue_raw_ext_access lws_issue_raw
#define lws_context_init_extensions(_a, _b)
#endif

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_client_interpret_server_handshake(struct lws *wsi);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ws_rx_sm(struct lws *wsi, char already_processed, unsigned char c);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_issue_raw_ext_access(struct lws *wsi, unsigned char *buf, size_t len);

LWS_EXTERN void
lws_role_transition(struct lws *wsi, enum lwsi_role role, enum lwsi_state state,
			struct lws_role_ops *ops);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
user_callback_handle_rxflow(lws_callback_function, struct lws *wsi,
			    enum lws_callback_reasons reason, void *user,
			    void *in, size_t len);

LWS_EXTERN int
lws_plat_socket_offset(void);

LWS_EXTERN int
lws_plat_set_socket_options(struct lws_vhost *vhost, lws_sockfd_type fd);

LWS_EXTERN int
lws_plat_check_connection_error(struct lws *wsi);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_header_table_attach(struct lws *wsi, int autoservice);

LWS_EXTERN int
lws_header_table_detach(struct lws *wsi, int autoservice);
LWS_EXTERN int
__lws_header_table_detach(struct lws *wsi, int autoservice);

LWS_EXTERN void
lws_header_table_reset(struct lws *wsi, int autoservice);

void
__lws_header_table_reset(struct lws *wsi, int autoservice);

LWS_EXTERN char * LWS_WARN_UNUSED_RESULT
lws_hdr_simple_ptr(struct lws *wsi, enum lws_token_indexes h);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_hdr_simple_create(struct lws *wsi, enum lws_token_indexes h, const char *s);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ensure_user_space(struct lws *wsi);

LWS_EXTERN int
lws_change_pollfd(struct lws *wsi, int _and, int _or);

#ifndef LWS_NO_SERVER
 int _lws_vhost_init_server(const struct lws_context_creation_info *info,
			      struct lws_vhost *vhost);
 LWS_EXTERN struct lws_vhost *
 lws_select_vhost(struct lws_context *context, int port, const char *servername);
 LWS_EXTERN int LWS_WARN_UNUSED_RESULT
 lws_parse_ws(struct lws *wsi, unsigned char **buf, size_t len);
 LWS_EXTERN void
 lws_server_get_canonical_hostname(struct lws_context *context,
				   const struct lws_context_creation_info *info);
#else
 #define _lws_vhost_init_server(_a, _b) (0)
 #define lws_parse_ws(_a, _b, _c) (0)
 #define lws_server_get_canonical_hostname(_a, _b)
#endif

#ifndef LWS_NO_DAEMONIZE
 LWS_EXTERN int get_daemonize_pid();
#else
 #define get_daemonize_pid() (0)
#endif

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
interface_to_sa(struct lws_vhost *vh, const char *ifname,
		struct sockaddr_in *addr, size_t addrlen);
LWS_EXTERN void lwsl_emit_stderr(int level, const char *line);

#if !defined(LWS_WITH_TLS)
 #define LWS_SSL_ENABLED(context) (0)
 #define lws_context_init_server_ssl(_a, _b) (0)
 #define lws_ssl_destroy(_a)
 #define lws_context_init_alpn(_a)
 #define lws_ssl_capable_read lws_ssl_capable_read_no_ssl
 #define lws_ssl_capable_write lws_ssl_capable_write_no_ssl
 #define lws_ssl_pending lws_ssl_pending_no_ssl
 #define lws_server_socket_service_ssl(_b, _c) (0)
 #define lws_ssl_close(_a) (0)
 #define lws_ssl_context_destroy(_a)
 #define lws_ssl_SSL_CTX_destroy(_a)
 #define lws_ssl_remove_wsi_from_buffered_list(_a)
 #define __lws_ssl_remove_wsi_from_buffered_list(_a)
 #define lws_context_init_ssl_library(_a)
 #define lws_tls_check_all_cert_lifetimes(_a)
 #define lws_tls_acme_sni_cert_destroy(_a)
#endif


#if LWS_MAX_SMP > 1

static LWS_INLINE void
lws_pt_mutex_init(struct lws_context_per_thread *pt)
{
	pthread_mutex_init(&pt->lock, NULL);
	pthread_mutex_init(&pt->lock_stats, NULL);
}

static LWS_INLINE void
lws_pt_mutex_destroy(struct lws_context_per_thread *pt)
{
	pthread_mutex_destroy(&pt->lock_stats);
	pthread_mutex_destroy(&pt->lock);
}

static LWS_INLINE void
lws_pt_lock(struct lws_context_per_thread *pt, const char *reason)
{
	if (pt->lock_owner == pthread_self()) {
		pt->lock_depth++;
		return;
	}
	pthread_mutex_lock(&pt->lock);
	pt->last_lock_reason = reason;
	pt->lock_owner = pthread_self();
	//lwsl_notice("tid %d: lock %s\n", pt->tid, reason);
}

static LWS_INLINE void
lws_pt_unlock(struct lws_context_per_thread *pt)
{
	if (pt->lock_depth) {
		pt->lock_depth--;
		return;
	}
	pt->last_lock_reason = "free";
	pt->lock_owner = 0;
	//lwsl_notice("tid %d: unlock %s\n", pt->tid, pt->last_lock_reason);
	pthread_mutex_unlock(&pt->lock);
}

static LWS_INLINE void
lws_pt_stats_lock(struct lws_context_per_thread *pt)
{
	pthread_mutex_lock(&pt->lock_stats);
}

static LWS_INLINE void
lws_pt_stats_unlock(struct lws_context_per_thread *pt)
{
	pthread_mutex_unlock(&pt->lock_stats);
}

static LWS_INLINE void
lws_context_lock(struct lws_context *context)
{
	pthread_mutex_lock(&context->lock);
}

static LWS_INLINE void
lws_context_unlock(struct lws_context *context)
{
	pthread_mutex_unlock(&context->lock);
}

static LWS_INLINE void
lws_vhost_lock(struct lws_vhost *vhost)
{
	pthread_mutex_lock(&vhost->lock);
}

static LWS_INLINE void
lws_vhost_unlock(struct lws_vhost *vhost)
{
	pthread_mutex_unlock(&vhost->lock);
}


#else
#define lws_pt_mutex_init(_a) (void)(_a)
#define lws_pt_mutex_destroy(_a) (void)(_a)
#define lws_pt_lock(_a, b) (void)(_a)
#define lws_pt_unlock(_a) (void)(_a)
#define lws_context_lock(_a) (void)(_a)
#define lws_context_unlock(_a) (void)(_a)
#define lws_vhost_lock(_a) (void)(_a)
#define lws_vhost_unlock(_a) (void)(_a)
#define lws_pt_stats_lock(_a) (void)(_a)
#define lws_pt_stats_unlock(_a) (void)(_a)
#endif

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ssl_capable_read_no_ssl(struct lws *wsi, unsigned char *buf, int len);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ssl_capable_write_no_ssl(struct lws *wsi, unsigned char *buf, int len);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ssl_pending_no_ssl(struct lws *wsi);

int
lws_tls_check_cert_lifetime(struct lws_vhost *vhost);

int lws_jws_selftest(void);


#ifndef LWS_NO_CLIENT
LWS_EXTERN int lws_client_socket_service(struct lws *wsi,
					 struct lws_pollfd *pollfd,
					 struct lws *wsi_conn);
LWS_EXTERN struct lws *
lws_client_wsi_effective(struct lws *wsi);
LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_http_transaction_completed_client(struct lws *wsi);
#if !defined(LWS_WITH_TLS)
	#define lws_context_init_client_ssl(_a, _b) (0)
#endif
LWS_EXTERN void
lws_decode_ssl_error(void);
#else
#define lws_context_init_client_ssl(_a, _b) (0)
#endif

LWS_EXTERN int
__lws_rx_flow_control(struct lws *wsi);

LWS_EXTERN int
_lws_change_pollfd(struct lws *wsi, int _and, int _or, struct lws_pollargs *pa);

#ifndef LWS_NO_SERVER
LWS_EXTERN int
lws_handshake_server(struct lws *wsi, unsigned char **buf, size_t len);
#else
#define lws_server_socket_service(_b, _c) (0)
#define lws_handshake_server(_a, _b, _c) (0)
#endif

#ifdef LWS_WITH_ACCESS_LOG
LWS_EXTERN int
lws_access_log(struct lws *wsi);
LWS_EXTERN void
lws_prepare_access_log_info(struct lws *wsi, char *uri_ptr, int meth);
#else
#define lws_access_log(_a)
#endif

LWS_EXTERN int
lws_cgi_kill_terminated(struct lws_context_per_thread *pt);

LWS_EXTERN void
lws_cgi_remove_and_kill(struct lws *wsi);

int
lws_protocol_init(struct lws_context *context);

int
lws_bind_protocol(struct lws *wsi, const struct lws_protocols *p);

const struct lws_http_mount *
lws_find_mount(struct lws *wsi, const char *uri_ptr, int uri_len);

/*
 * custom allocator
 */
LWS_EXTERN void *
lws_realloc(void *ptr, size_t size, const char *reason);

LWS_EXTERN void * LWS_WARN_UNUSED_RESULT
lws_zalloc(size_t size, const char *reason);

#ifdef LWS_PLAT_OPTEE
void *lws_malloc(size_t size, const char *reason);
void lws_free(void *p);
#define lws_free_set_NULL(P)    do { lws_free(P); (P) = NULL; } while(0)
#else
#define lws_malloc(S, R)	lws_realloc(NULL, S, R)
#define lws_free(P)	lws_realloc(P, 0, "lws_free")
#define lws_free_set_NULL(P)	do { lws_realloc(P, 0, "free"); (P) = NULL; } while(0)
#endif

char *
lws_strdup(const char *s);

int
lws_plat_pipe_create(struct lws *wsi);
int
lws_plat_pipe_signal(struct lws *wsi);
void
lws_plat_pipe_close(struct lws *wsi);
int
lws_create_event_pipes(struct lws_context *context);

int lws_open(const char *__file, int __oflag, ...);
void lws_plat_apply_FD_CLOEXEC(int n);

const struct lws_plat_file_ops *
lws_vfs_select_fops(const struct lws_plat_file_ops *fops, const char *vfs_path,
		    const char **vpath);

/* lws_plat_ */
LWS_EXTERN void
lws_plat_delete_socket_from_fds(struct lws_context *context,
				struct lws *wsi, int m);
LWS_EXTERN void
lws_plat_insert_socket_into_fds(struct lws_context *context,
				struct lws *wsi);
LWS_EXTERN void
lws_plat_service_periodic(struct lws_context *context);

LWS_EXTERN int
lws_plat_change_pollfd(struct lws_context *context, struct lws *wsi,
		       struct lws_pollfd *pfd);
LWS_EXTERN void
lws_add_wsi_to_draining_ext_list(struct lws *wsi);
LWS_EXTERN void
lws_remove_wsi_from_draining_ext_list(struct lws *wsi);
LWS_EXTERN int
lws_plat_context_early_init(void);
LWS_EXTERN void
lws_plat_context_early_destroy(struct lws_context *context);
LWS_EXTERN void
lws_plat_context_late_destroy(struct lws_context *context);
LWS_EXTERN int
lws_poll_listen_fd(struct lws_pollfd *fd);
LWS_EXTERN int
lws_plat_service(struct lws_context *context, int timeout_ms);
LWS_EXTERN LWS_VISIBLE int
_lws_plat_service_tsi(struct lws_context *context, int timeout_ms, int tsi);
LWS_EXTERN int
lws_plat_init(struct lws_context *context,
	      const struct lws_context_creation_info *info);
LWS_EXTERN void
lws_plat_drop_app_privileges(const struct lws_context_creation_info *info);
LWS_EXTERN unsigned long long
time_in_microseconds(void);
LWS_EXTERN const char * LWS_WARN_UNUSED_RESULT
lws_plat_inet_ntop(int af, const void *src, char *dst, int cnt);
LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_plat_inet_pton(int af, const char *src, void *dst);

LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_check_utf8(unsigned char *state, unsigned char *buf, size_t len);
LWS_EXTERN int alloc_file(struct lws_context *context, const char *filename, uint8_t **buf,
		                lws_filepos_t *amount);


LWS_EXTERN void
lws_same_vh_protocol_remove(struct lws *wsi);
LWS_EXTERN void
lws_same_vh_protocol_insert(struct lws *wsi, int n);

LWS_EXTERN int
lws_broadcast(struct lws_context *context, int reason, void *in, size_t len);

#if defined(LWS_WITH_STATS)
 void
 lws_stats_atomic_bump(struct lws_context * context,
		struct lws_context_per_thread *pt, int index, uint64_t bump);
 void
 lws_stats_atomic_max(struct lws_context * context,
		struct lws_context_per_thread *pt, int index, uint64_t val);
#else
 static LWS_INLINE uint64_t lws_stats_atomic_bump(struct lws_context * context,
		struct lws_context_per_thread *pt, int index, uint64_t bump) {
	(void)context; (void)pt; (void)index; (void)bump; return 0; }
 static LWS_INLINE uint64_t lws_stats_atomic_max(struct lws_context * context,
		struct lws_context_per_thread *pt, int index, uint64_t val) {
	(void)context; (void)pt; (void)index; (void)val; return 0; }
#endif

/* socks */
void socks_generate_msg(struct lws *wsi, enum socks_msg_type type,
			ssize_t *msg_len);

#if defined(LWS_WITH_PEER_LIMITS)
void
lws_peer_track_wsi_close(struct lws_context *context, struct lws_peer *peer);
int
lws_peer_confirm_ah_attach_ok(struct lws_context *context, struct lws_peer *peer);
void
lws_peer_track_ah_detach(struct lws_context *context, struct lws_peer *peer);
void
lws_peer_cull_peer_wait_list(struct lws_context *context);
struct lws_peer *
lws_get_or_create_peer(struct lws_vhost *vhost, lws_sockfd_type sockfd);
void
lws_peer_add_wsi(struct lws_context *context, struct lws_peer *peer,
		 struct lws *wsi);
void
lws_peer_dump_from_wsi(struct lws *wsi);
#endif

#ifdef LWS_WITH_HTTP_PROXY
hubbub_error
html_parser_cb(const hubbub_token *token, void *pw);
#endif


void
__lws_remove_from_timeout_list(struct lws *wsi);

lws_usec_t
__lws_hrtimer_service(struct lws_context_per_thread *pt);

void
__lws_set_timeout(struct lws *wsi, enum pending_timeout reason, int secs);
int
__lws_change_pollfd(struct lws *wsi, int _and, int _or);


int
lws_callback_as_writeable(struct lws *wsi);
int
lws_buflist_aware_read(struct lws_context_per_thread *pt, struct lws *wsi,
		       struct lws_tokens *ebuf);
int
lws_buflist_aware_consume(struct lws *wsi, struct lws_tokens *ebuf, int used,
			  int buffered);


char *
lws_generate_client_ws_handshake(struct lws *wsi, char *p);
int
lws_client_ws_upgrade(struct lws *wsi, const char **cce);
int
lws_create_client_ws_object(struct lws_client_connect_info *i, struct lws *wsi);
int
lws_alpn_comma_to_openssl(const char *comma, uint8_t *os, int len);
int
lws_role_call_alpn_negotiated(struct lws *wsi, const char *alpn);
int
lws_tls_server_conn_alpn(struct lws *wsi);

int
lws_ws_client_rx_sm_block(struct lws *wsi, unsigned char **buf, size_t len);
void
lws_destroy_event_pipe(struct lws *wsi);
void
lws_context_destroy2(struct lws_context *context);

#ifdef __cplusplus
};
#endif
