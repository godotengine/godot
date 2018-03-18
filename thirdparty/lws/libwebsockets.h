/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2016 Andy Green <andy@warmcat.com>
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

/** @file */

#ifndef LIBWEBSOCKET_H_3060898B846849FF9F88F5DB59B5950C
#define LIBWEBSOCKET_H_3060898B846849FF9F88F5DB59B5950C

#ifdef __cplusplus
#include <cstddef>
#include <cstdarg>
#
extern "C" {
#else
#include <stdarg.h>
#endif

#include "lws_config.h"

/*
 * CARE: everything using cmake defines needs to be below here
 */

#if defined(LWS_WITH_ESP8266)
struct sockaddr_in;
#define LWS_POSIX 0
#else
#define LWS_POSIX 1
#endif

#if defined(LWS_HAS_INTPTR_T)
#include <stdint.h>
#define lws_intptr_t intptr_t
#else
typedef unsigned long long lws_intptr_t;
#endif

#if defined(WIN32) || defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <stddef.h>
#include <basetsd.h>
#ifndef _WIN32_WCE
#include <fcntl.h>
#else
#define _O_RDONLY	0x0000
#define O_RDONLY	_O_RDONLY
#endif

// Visual studio older than 2015 and WIN_CE has only _stricmp
#if (defined(_MSC_VER) && _MSC_VER < 1900) || defined(_WIN32_WCE)
#define strcasecmp _stricmp
#elif !defined(__MINGW32__)
#define strcasecmp stricmp
#endif
#define getdtablesize() 30000

#define LWS_INLINE __inline
#define LWS_VISIBLE
#define LWS_WARN_UNUSED_RESULT
#define LWS_WARN_DEPRECATED
#define LWS_FORMAT(string_index)

#ifdef LWS_DLL
#ifdef LWS_INTERNAL
#define LWS_EXTERN extern __declspec(dllexport)
#else
#define LWS_EXTERN extern __declspec(dllimport)
#endif
#else
#define LWS_EXTERN
#endif

#define LWS_INVALID_FILE INVALID_HANDLE_VALUE
#define LWS_O_RDONLY _O_RDONLY
#define LWS_O_WRONLY _O_WRONLY
#define LWS_O_CREAT _O_CREAT
#define LWS_O_TRUNC _O_TRUNC

#if !defined(__MINGW32__) && (!defined(_MSC_VER) || _MSC_VER < 1900) /* Visual Studio 2015 already defines this in <stdio.h> */
#define lws_snprintf _snprintf
#endif

#ifndef __func__
#define __func__ __FUNCTION__
#endif

#if !defined(__MINGW32__) &&(!defined(_MSC_VER) || _MSC_VER < 1900) && !defined(snprintf)
#define snprintf(buf,len, format,...) _snprintf_s(buf, len,len, format, __VA_ARGS__)
#endif

#else /* NOT WIN32 */
#include <unistd.h>
#if defined(LWS_HAVE_SYS_CAPABILITY_H) && defined(LWS_HAVE_LIBCAP)
#include <sys/capability.h>
#endif

#if defined(__NetBSD__) || defined(__FreeBSD__)
#include <netinet/in.h>
#endif

#define LWS_INLINE inline
#define LWS_O_RDONLY O_RDONLY
#define LWS_O_WRONLY O_WRONLY
#define LWS_O_CREAT O_CREAT
#define LWS_O_TRUNC O_TRUNC

#if !defined(LWS_WITH_ESP8266) && !defined(OPTEE_TA) && !defined(LWS_WITH_ESP32)
#include <poll.h>
#include <netdb.h>
#define LWS_INVALID_FILE -1
#else
#define getdtablesize() (30)
#if defined(LWS_WITH_ESP32)
#define LWS_INVALID_FILE NULL
#else
#define LWS_INVALID_FILE NULL
#endif
#endif

#if defined(__GNUC__)

/* warn_unused_result attribute only supported by GCC 3.4 or later */
#if __GNUC__ >= 4 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#define LWS_WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#else
#define LWS_WARN_UNUSED_RESULT
#endif

#define LWS_VISIBLE __attribute__((visibility("default")))
#define LWS_WARN_DEPRECATED __attribute__ ((deprecated))
#define LWS_FORMAT(string_index) __attribute__ ((format(printf, string_index, string_index+1)))
#else
#define LWS_VISIBLE
#define LWS_WARN_UNUSED_RESULT
#define LWS_WARN_DEPRECATED
#define LWS_FORMAT(string_index)
#endif

#if defined(__ANDROID__)
#include <unistd.h>
#define getdtablesize() sysconf(_SC_OPEN_MAX)
#endif

#endif

#ifdef LWS_WITH_LIBEV
#include <ev.h>
#endif /* LWS_WITH_LIBEV */
#ifdef LWS_WITH_LIBUV
#include <uv.h>
#ifdef LWS_HAVE_UV_VERSION_H
#include <uv-version.h>
#endif
#endif /* LWS_WITH_LIBUV */
#ifdef LWS_WITH_LIBEVENT
#include <event2/event.h>
#endif /* LWS_WITH_LIBEVENT */

#ifndef LWS_EXTERN
#define LWS_EXTERN extern
#endif

#ifdef _WIN32
#define random rand
#else
#if !defined(OPTEE_TA)
#include <sys/time.h>
#include <unistd.h>
#endif
#endif

#ifdef LWS_OPENSSL_SUPPORT

#ifdef USE_WOLFSSL
#ifdef USE_OLD_CYASSL
#include <cyassl/openssl/ssl.h>
#include <cyassl/error-ssl.h>
#else
#include <wolfssl/openssl/ssl.h>
#include <wolfssl/error-ssl.h>
#endif /* not USE_OLD_CYASSL */
#else
#if defined(LWS_WITH_MBEDTLS)
#if defined(LWS_WITH_ESP32)
/* this filepath is passed to us but without quotes or <> */
#undef MBEDTLS_CONFIG_FILE
#define MBEDTLS_CONFIG_FILE <mbedtls/esp_config.h>
#endif
#include <mbedtls/ssl.h>
#endif
#include <openssl/ssl.h>
#if !defined(LWS_WITH_MBEDTLS)
#include <openssl/err.h>
#endif
#endif /* not USE_WOLFSSL */
#endif


#define CONTEXT_PORT_NO_LISTEN -1
#define CONTEXT_PORT_NO_LISTEN_SERVER -2

/** \defgroup log Logging
 *
 * ##Logging
 *
 * Lws provides flexible and filterable logging facilities, which can be
 * used inside lws and in user code.
 *
 * Log categories may be individually filtered bitwise, and directed to built-in
 * sinks for syslog-compatible logging, or a user-defined function.
 */
///@{

enum lws_log_levels {
	LLL_ERR = 1 << 0,
	LLL_WARN = 1 << 1,
	LLL_NOTICE = 1 << 2,
	LLL_INFO = 1 << 3,
	LLL_DEBUG = 1 << 4,
	LLL_PARSER = 1 << 5,
	LLL_HEADER = 1 << 6,
	LLL_EXT = 1 << 7,
	LLL_CLIENT = 1 << 8,
	LLL_LATENCY = 1 << 9,
	LLL_USER = 1 << 10,

	LLL_COUNT = 11 /* set to count of valid flags */
};

LWS_VISIBLE LWS_EXTERN void _lws_log(int filter, const char *format, ...) LWS_FORMAT(2);
LWS_VISIBLE LWS_EXTERN void _lws_logv(int filter, const char *format, va_list vl);
/**
 * lwsl_timestamp: generate logging timestamp string
 *
 * \param level:	logging level
 * \param p:		char * buffer to take timestamp
 * \param len:	length of p
 *
 * returns length written in p
 */
LWS_VISIBLE LWS_EXTERN int
lwsl_timestamp(int level, char *p, int len);

/* these guys are unconditionally included */

#define lwsl_err(...) _lws_log(LLL_ERR, __VA_ARGS__)
#define lwsl_user(...) _lws_log(LLL_USER, __VA_ARGS__)

#if !defined(LWS_WITH_NO_LOGS)
/* notice and warn are usually included by being compiled in */
#define lwsl_warn(...) _lws_log(LLL_WARN, __VA_ARGS__)
#define lwsl_notice(...) _lws_log(LLL_NOTICE, __VA_ARGS__)
#endif
/*
 *  weaker logging can be deselected by telling CMake to build in RELEASE mode
 *  that gets rid of the overhead of checking while keeping _warn and _err
 *  active
 */

#if defined(LWS_WITH_ESP8266)
#undef _DEBUG
#endif

#ifdef _DEBUG
#if defined(LWS_WITH_NO_LOGS)
/* notice, warn and log are always compiled in */
#define lwsl_warn(...) _lws_log(LLL_WARN, __VA_ARGS__)
#define lwsl_notice(...) _lws_log(LLL_NOTICE, __VA_ARGS__)
#endif
#define lwsl_info(...) _lws_log(LLL_INFO, __VA_ARGS__)
#define lwsl_debug(...) _lws_log(LLL_DEBUG, __VA_ARGS__)
#define lwsl_parser(...) _lws_log(LLL_PARSER, __VA_ARGS__)
#define lwsl_header(...)  _lws_log(LLL_HEADER, __VA_ARGS__)
#define lwsl_ext(...)  _lws_log(LLL_EXT, __VA_ARGS__)
#define lwsl_client(...) _lws_log(LLL_CLIENT, __VA_ARGS__)
#define lwsl_latency(...) _lws_log(LLL_LATENCY, __VA_ARGS__)

#else /* no debug */
#if defined(LWS_WITH_NO_LOGS)
#define lwsl_warn(...) do {} while(0)
#define lwsl_notice(...) do {} while(0)
#endif
#define lwsl_info(...) do {} while(0)
#define lwsl_debug(...) do {} while(0)
#define lwsl_parser(...) do {} while(0)
#define lwsl_header(...) do {} while(0)
#define lwsl_ext(...) do {} while(0)
#define lwsl_client(...) do {} while(0)
#define lwsl_latency(...) do {} while(0)

#endif

/**
 * lwsl_hexdump() - helper to hexdump a buffer
 *
 * \param level: one of LLL_ constants
 * \param buf: buffer start to dump
 * \param len: length of buffer to dump
 *
 * If \p level is visible, does a nice hexdump -C style dump of \p buf for
 * \p len bytes.  This can be extremely convenient while debugging.
 */
LWS_VISIBLE LWS_EXTERN void
lwsl_hexdump_level(int level, const void *vbuf, size_t len);

/**
 * lwsl_hexdump() - helper to hexdump a buffer (DEBUG builds only)
 *
 * \param buf: buffer start to dump
 * \param len: length of buffer to dump
 *
 * Calls through to lwsl_hexdump_level(LLL_DEBUG, ... for compatability.
 * It's better to use lwsl_hexdump_level(level, ... directly so you can control
 * the visibility.
 */
LWS_VISIBLE LWS_EXTERN void
lwsl_hexdump(const void *buf, size_t len);

/**
 * lws_is_be() - returns nonzero if the platform is Big Endian
 */
static LWS_INLINE int lws_is_be(void) {
	const int probe = ~0xff;

	return *(const char *)&probe;
}

/**
 * lws_set_log_level() - Set the logging bitfield
 * \param level:	OR together the LLL_ debug contexts you want output from
 * \param log_emit_function:	NULL to leave it as it is, or a user-supplied
 *			function to perform log string emission instead of
 *			the default stderr one.
 *
 *	log level defaults to "err", "warn" and "notice" contexts enabled and
 *	emission on stderr.  If stderr is a tty (according to isatty()) then
 *	the output is coloured according to the log level using ANSI escapes.
 */
LWS_VISIBLE LWS_EXTERN void
lws_set_log_level(int level,
		  void (*log_emit_function)(int level, const char *line));

/**
 * lwsl_emit_syslog() - helper log emit function writes to system log
 *
 * \param level: one of LLL_ log level indexes
 * \param line: log string
 *
 * You use this by passing the function pointer to lws_set_log_level(), to set
 * it as the log emit function, it is not called directly.
 */
LWS_VISIBLE LWS_EXTERN void
lwsl_emit_syslog(int level, const char *line);

/**
 * lwsl_visible() - returns true if the log level should be printed
 *
 * \param level: one of LLL_ log level indexes
 *
 * This is useful if you have to do work to generate the log content, you
 * can skip the work if the log level used to print it is not actually
 * enabled at runtime.
 */
LWS_VISIBLE LWS_EXTERN int
lwsl_visible(int level);

///@}


#include <stddef.h>

#ifndef lws_container_of
#define lws_container_of(P,T,M)	((T *)((char *)(P) - offsetof(T, M)))
#endif


struct lws;
#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))
#endif

/* api change list for user code to test against */

#define LWS_FEATURE_SERVE_HTTP_FILE_HAS_OTHER_HEADERS_ARG

/* the struct lws_protocols has the id field present */
#define LWS_FEATURE_PROTOCOLS_HAS_ID_FIELD

/* you can call lws_get_peer_write_allowance */
#define LWS_FEATURE_PROTOCOLS_HAS_PEER_WRITE_ALLOWANCE

/* extra parameter introduced in 917f43ab821 */
#define LWS_FEATURE_SERVE_HTTP_FILE_HAS_OTHER_HEADERS_LEN

/* File operations stuff exists */
#define LWS_FEATURE_FOPS


#if defined(_WIN32)
typedef SOCKET lws_sockfd_type;
typedef HANDLE lws_filefd_type;
#define lws_sockfd_valid(sfd) (!!sfd)
struct lws_pollfd {
	lws_sockfd_type fd; /**< file descriptor */
	SHORT events; /**< which events to respond to */
	SHORT revents; /**< which events happened */
};
#define LWS_POLLHUP (FD_CLOSE)
#define LWS_POLLIN (FD_READ | FD_ACCEPT)
#define LWS_POLLOUT (FD_WRITE)
#else


#if defined(LWS_WITH_ESP8266)

#include <user_interface.h>
#include <espconn.h>

typedef struct espconn * lws_sockfd_type;
typedef void * lws_filefd_type;
#define lws_sockfd_valid(sfd) (!!sfd)
struct pollfd {
	lws_sockfd_type fd; /**< fd related to */
	short events; /**< which POLL... events to respond to */
	short revents; /**< which POLL... events occurred */
};
#define POLLIN		0x0001
#define POLLPRI		0x0002
#define POLLOUT		0x0004
#define POLLERR		0x0008
#define POLLHUP		0x0010
#define POLLNVAL	0x0020

struct lws_vhost;

lws_sockfd_type esp8266_create_tcp_listen_socket(struct lws_vhost *vh);
void esp8266_tcp_stream_accept(lws_sockfd_type fd, struct lws *wsi);

#include <os_type.h>
#include <osapi.h>
#include "ets_sys.h"

int ets_snprintf(char *str, size_t size, const char *format, ...) LWS_FORMAT(3);
#define snprintf  ets_snprintf

typedef os_timer_t uv_timer_t;
typedef void uv_cb_t(uv_timer_t *);

void os_timer_disarm(void *);
void os_timer_setfn(os_timer_t *, os_timer_func_t *, void *);

void ets_timer_arm_new(os_timer_t *, int, int, int);

//void os_timer_arm(os_timer_t *, int, int);

#define UV_VERSION_MAJOR 1

#define lws_uv_getloop(a, b) (NULL)

static inline void uv_timer_init(void *l, uv_timer_t *t)
{
	(void)l;
	memset(t, 0, sizeof(*t));
	os_timer_disarm(t);
}

static inline void uv_timer_start(uv_timer_t *t, uv_cb_t *cb, int first, int rep)
{
	os_timer_setfn(t, (os_timer_func_t *)cb, t);
	/* ms, repeat */
	os_timer_arm(t, first, !!rep);
}

static inline void uv_timer_stop(uv_timer_t *t)
{
	os_timer_disarm(t);
}

#else
#if defined(LWS_WITH_ESP32)

typedef int lws_sockfd_type;
typedef int lws_filefd_type;
#define lws_sockfd_valid(sfd) (sfd >= 0)
struct pollfd {
	lws_sockfd_type fd; /**< fd related to */
	short events; /**< which POLL... events to respond to */
	short revents; /**< which POLL... events occurred */
};
#define POLLIN		0x0001
#define POLLPRI		0x0002
#define POLLOUT		0x0004
#define POLLERR		0x0008
#define POLLHUP		0x0010
#define POLLNVAL	0x0020

#include <freertos/FreeRTOS.h>
#include <freertos/event_groups.h>
#include <string.h>
#include "esp_wifi.h"
#include "esp_system.h"
#include "esp_event.h"
#include "esp_event_loop.h"
#include "nvs.h"
#include "driver/gpio.h"
#include "esp_spi_flash.h"
#include "freertos/timers.h"

#if !defined(CONFIG_FREERTOS_HZ)
#define CONFIG_FREERTOS_HZ 100
#endif

typedef TimerHandle_t uv_timer_t;
typedef void uv_cb_t(uv_timer_t *);
typedef void * uv_handle_t;

struct timer_mapping {
	uv_cb_t *cb;
	uv_timer_t *t;
};

#define UV_VERSION_MAJOR 1

#define lws_uv_getloop(a, b) (NULL)

static inline void uv_timer_init(void *l, uv_timer_t *t)
{
	(void)l;
	*t = NULL;
}

extern void esp32_uvtimer_cb(TimerHandle_t t);

static inline void uv_timer_start(uv_timer_t *t, uv_cb_t *cb, int first, int rep)
{
	struct timer_mapping *tm = (struct timer_mapping *)malloc(sizeof(*tm));

	if (!tm)
		return;

	tm->t = t;
	tm->cb = cb;

	*t = xTimerCreate("x", pdMS_TO_TICKS(first), !!rep, tm,
			  (TimerCallbackFunction_t)esp32_uvtimer_cb);
	xTimerStart(*t, 0);
}

static inline void uv_timer_stop(uv_timer_t *t)
{
	xTimerStop(*t, 0);
}

static inline void uv_close(uv_handle_t *h, void *v)
{
	free(pvTimerGetTimerID((uv_timer_t)h));
	xTimerDelete(*(uv_timer_t *)h, 0);
}

/* ESP32 helper declarations */

#include <mdns.h>
#include <esp_partition.h>

#define LWS_PLUGIN_STATIC
#define LWS_MAGIC_REBOOT_TYPE_ADS 0x50001ffc
#define LWS_MAGIC_REBOOT_TYPE_REQ_FACTORY 0xb00bcafe
#define LWS_MAGIC_REBOOT_TYPE_FORCED_FACTORY 0xfaceb00b
#define LWS_MAGIC_REBOOT_TYPE_FORCED_FACTORY_BUTTON 0xf0cedfac


/* user code provides these */

extern void
lws_esp32_identify_physical_device(void);

/* lws-plat-esp32 provides these */

typedef void (*lws_cb_scan_done)(uint16_t count, wifi_ap_record_t *recs, void *arg);

enum genled_state {
	LWSESP32_GENLED__INIT,
	LWSESP32_GENLED__LOST_NETWORK,
	LWSESP32_GENLED__NO_NETWORK,
	LWSESP32_GENLED__CONN_AP,
	LWSESP32_GENLED__GOT_IP,
	LWSESP32_GENLED__OK,
};

struct lws_group_member {
	struct lws_group_member *next;
	uint64_t last_seen;
	char model[16];
	char role[16];
	char host[32];
	char mac[20];
	int width, height;
	struct ip4_addr addr;
	struct ip6_addr addrv6;
	uint8_t	flags;
};

#define LWS_SYSTEM_GROUP_MEMBER_ADD		1
#define LWS_SYSTEM_GROUP_MEMBER_CHANGE		2
#define LWS_SYSTEM_GROUP_MEMBER_REMOVE		3

#define LWS_GROUP_FLAG_SELF 1

struct lws_esp32 {
	char sta_ip[16];
	char sta_mask[16];
	char sta_gw[16];
	char serial[16];
	char opts[16];
	char model[16];
	char group[16];
	char role[16];
	char ssid[4][16];
	char password[4][32];
	char active_ssid[32];
	char access_pw[16];
	char hostname[32];
	char mac[20];
	mdns_server_t *mdns;
       	char region;
       	char inet;
	char conn_ap;

	enum genled_state genled;
	uint64_t genled_t;

	lws_cb_scan_done scan_consumer;
	void *scan_consumer_arg;
	struct lws_group_member *first;
	int extant_group_members;
};

struct lws_esp32_image {
	uint32_t romfs;
	uint32_t romfs_len;
	uint32_t json;
	uint32_t json_len;
};

extern struct lws_esp32 lws_esp32;
struct lws_vhost;

extern esp_err_t
lws_esp32_event_passthru(void *ctx, system_event_t *event);
extern void
lws_esp32_wlan_config(void);
extern void
lws_esp32_wlan_start_ap(void);
extern void
lws_esp32_wlan_start_station(void);
struct lws_context_creation_info;
extern void
lws_esp32_set_creation_defaults(struct lws_context_creation_info *info);
extern struct lws_context *
lws_esp32_init(struct lws_context_creation_info *, struct lws_vhost **pvh);
extern int
lws_esp32_wlan_nvs_get(int retry);
extern esp_err_t
lws_nvs_set_str(nvs_handle handle, const char* key, const char* value);
extern void
lws_esp32_restart_guided(uint32_t type);
extern const esp_partition_t *
lws_esp_ota_get_boot_partition(void);
extern int
lws_esp32_get_image_info(const esp_partition_t *part, struct lws_esp32_image *i, char *json, int json_len);
extern int
lws_esp32_leds_network_indication(void);

extern uint32_t lws_esp32_get_reboot_type(void);
extern uint16_t lws_esp32_sine_interp(int n);

/* required in external code by esp32 plat (may just return if no leds) */
extern void lws_esp32_leds_timer_cb(TimerHandle_t th);
#else
typedef int lws_sockfd_type;
typedef int lws_filefd_type;
#define lws_sockfd_valid(sfd) (sfd >= 0)
#endif
#endif

#define lws_pollfd pollfd
#define LWS_POLLHUP (POLLHUP|POLLERR)
#define LWS_POLLIN (POLLIN)
#define LWS_POLLOUT (POLLOUT)
#endif


#if (defined(WIN32) || defined(_WIN32)) && !defined(__MINGW32__)
/* ... */
#define ssize_t SSIZE_T
#endif

#if defined(WIN32) && defined(LWS_HAVE__STAT32I64)
#include <sys/types.h>
#include <sys/stat.h>
#endif

#if defined(LWS_HAVE_STDINT_H)
#include <stdint.h>
#else
#if defined(WIN32) || defined(_WIN32)
/* !!! >:-[  */
typedef unsigned __int32 uint32_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int8 uint8_t;
#else
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
#endif
#endif

typedef unsigned long long lws_filepos_t;
typedef long long lws_fileofs_t;
typedef uint32_t lws_fop_flags_t;

/** struct lws_pollargs - argument structure for all external poll related calls
 * passed in via 'in' */
struct lws_pollargs {
	lws_sockfd_type fd;	/**< applicable socket descriptor */
	int events;		/**< the new event mask */
	int prev_events;	/**< the previous event mask */
};

struct lws_tokens;
struct lws_token_limits;

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

struct lws;
struct lws_context;
/* needed even with extensions disabled for create context */
struct lws_extension;

/*! \defgroup lwsmeta lws-meta
 *
 * ##lws-meta protocol
 *
 * The protocol wraps other muxed connections inside one tcp connection.
 *
 * Commands are assigned from 0x41 up (so they are valid unicode)
 */
///@{

enum lws_meta_commands {
	LWS_META_CMD_OPEN_SUBCHANNEL = 'A',
	/**< Client requests to open new subchannel
	 */
	LWS_META_CMD_OPEN_RESULT,
	/**< Result of client request to open new subchannel */
	LWS_META_CMD_CLOSE_NOTIFY,
	/**< Notification of subchannel closure */
	LWS_META_CMD_CLOSE_RQ,
	/**< client requests to close a subchannel */
	LWS_META_CMD_WRITE,
	/**< connection writes something to specific channel index */

	/****** add new things just above ---^ ******/
};

/* channel numbers are transported offset by 0x20 so they are valid unicode */

#define LWS_META_TRANSPORT_OFFSET 0x20

///@}

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

/*
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */
/** enum lws_callback_reasons - reason you're getting a protocol callback */
enum lws_callback_reasons {
	LWS_CALLBACK_ESTABLISHED				=  0,
	/**< (VH) after the server completes a handshake with an incoming
	 * client.  If you built the library with ssl support, in is a
	 * pointer to the ssl struct associated with the connection or NULL.*/
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
	/**< after your client connection completed
	 * a handshake with the remote server */
	LWS_CALLBACK_CLOSED					=  4,
	/**< when the websocket session ends */
	LWS_CALLBACK_CLOSED_HTTP				=  5,
	/**< when a HTTP (non-websocket) session ends */
	LWS_CALLBACK_RECEIVE					=  6,
	/**< data has appeared for this server endpoint from a
	 * remote client, it can be found at *in and is
	 * len bytes long */
	LWS_CALLBACK_RECEIVE_PONG				=  7,
	/**< servers receive PONG packets with this callback reason */
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
	LWS_CALLBACK_SERVER_WRITEABLE				= 11,
	/**< See LWS_CALLBACK_CLIENT_WRITEABLE */
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
	LWS_CALLBACK_SERVER_NEW_CLIENT_INSTANTIATED		= 19,
	/**< A new client just had
	 * been connected, accepted, and instantiated into the pool. This
	 * callback allows setting any relevant property to it. Because this
	 * happens immediately after the instantiation of a new client,
	 * there's no websocket protocol selected yet so this callback is
	 * issued only to protocol 0. Only wsi is defined, pointing to the
	 * new client, and the return value is ignored. */
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
	 * is the server's OpenSSL SSL_CTX* */
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
	LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER		= 24,
	/**< this callback happens
	 * when a client handshake is being compiled.  user is NULL,
	 * in is a char **, it's pointing to a char * which holds the
	 * next location in the header buffer where you can add
	 * headers, and len is the remaining space in the header buffer,
	 * which is typically some hundreds of bytes.  So, to add a canned
	 * cookie, your handler code might look similar to:
	 *
	 *	char **p = (char **)in;
	 *
	 *	if (len < 100)
	 *		return 1;
	 *
	 *	*p += sprintf(*p, "Cookie: a=b\x0d\x0a");
	 *
	 *	return 0;
	 *
	 * Notice if you add anything, you just have to take care about
	 * the CRLF on the line you added.  Obviously this callback is
	 * optional, if you don't handle it everything is fine.
	 *
	 * Notice the callback is coming to protocols[0] all the time,
	 * because there is no specific protocol negotiated yet. */
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
	LWS_CALLBACK_CLIENT_CONFIRM_EXTENSION_SUPPORTED		= 26,
	/**< When a client
	 * connection is being prepared to start a handshake to a server,
	 * each supported extension is checked with protocols[0] callback
	 * with this reason, giving the user code a chance to suppress the
	 * claim to support that extension by returning non-zero.  If
	 * unhandled, by default 0 will be returned and the extension
	 * support included in the header to the server.  Notice this
	 * callback comes to protocols[0]. */
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

	LWS_CALLBACK_OPENSSL_CONTEXT_REQUIRES_PRIVATE_KEY	= 37,
	/**< if configured for including OpenSSL support but no private key
	 * file has been specified (ssl_private_key_filepath is NULL), this is
	 * called to allow the user to set the private key directly via
	 * libopenssl and perform further operations if required; this might be
	 * useful in situations where the private key is not directly accessible
	 * by the OS, for example if it is stored on a smartcard.
	 * user is the server's OpenSSL SSL_CTX* */
	LWS_CALLBACK_WS_PEER_INITIATED_CLOSE			= 38,
	/**< The peer has sent an unsolicited Close WS packet.  in and
	 * len are the optional close code (first 2 bytes, network
	 * order) and the optional additional information which is not
	 * defined in the standard, and may be a string or non-human- readable data.
	 * If you return 0 lws will echo the close and then close the
	 * connection.  If you return nonzero lws will just close the
	 * connection. */

	LWS_CALLBACK_WS_EXT_DEFAULTS				= 39,
	/**< Gives client connections an opportunity to adjust negotiated
	 * extension defaults.  `user` is the extension name that was
	 * negotiated (eg, "permessage-deflate").  `in` points to a
	 * buffer and `len` is the buffer size.  The user callback can
	 * set the buffer to a string describing options the extension
	 * should parse.  Or just ignore for defaults. */

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
	LWS_CALLBACK_ESTABLISHED_CLIENT_HTTP			= 44,
	/**< The HTTP client connection has succeeded, and is now
	 * connected to the server */
	LWS_CALLBACK_CLOSED_CLIENT_HTTP				= 45,
	/**< The HTTP client connection is closing */
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
	LWS_CALLBACK_RECEIVE_CLIENT_HTTP_READ			= 48,
	/**< This is generated by lws_http_client_read() used to drain
	 * incoming data.  In the case the incoming data was chunked,
	 * it will be split into multiple smaller callbacks for each
	 * chunk block, removing the chunk headers. If not chunked,
	 * it will appear all in one callback. */
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
	 *  */
	LWS_CALLBACK_ADD_HEADERS				= 53,
	/**< This gives your user code a chance to add headers to a
	 * transaction bound to your protocol.  `in` points to a
	 * `struct lws_process_html_args` describing a buffer and length
	 * you can add headers into using the normal lws apis.
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
	LWS_CALLBACK_CLIENT_HTTP_WRITEABLE			= 57,
	/**< when doing an HTTP type client connection, you can call
	 * lws_client_http_body_pending(wsi, 1) from
	 * LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER to get these callbacks
	 * sending the HTTP headers.
	 *
	 * From this callback, when you have sent everything, you should let
	 * lws know by calling lws_client_http_body_pending(wsi, 0)
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
	LWS_CALLBACK_RAW_RX					= 59,
	/**< RAW mode connection RX */
	LWS_CALLBACK_RAW_CLOSE					= 60,
	/**< RAW mode connection is closing */
	LWS_CALLBACK_RAW_WRITEABLE				= 61,
	/**< RAW mode connection may be written */
	LWS_CALLBACK_RAW_ADOPT					= 62,
	/**< RAW mode connection was adopted (equivalent to 'wsi created') */
	LWS_CALLBACK_RAW_ADOPT_FILE				= 63,
	/**< RAW mode file was adopted (equivalent to 'wsi created') */
	LWS_CALLBACK_RAW_RX_FILE				= 64,
	/**< RAW mode file has something to read */
	LWS_CALLBACK_RAW_WRITEABLE_FILE				= 65,
	/**< RAW mode file is writeable */
	LWS_CALLBACK_RAW_CLOSE_FILE				= 66,
	/**< RAW mode wsi that adopted a file is closing */
	LWS_CALLBACK_SSL_INFO					= 67,
	/**< SSL connections only.  An event you registered an
	 * interest in at the vhost has occurred on a connection
	 * using the vhost.  in is a pointer to a
	 * struct lws_ssl_info containing information about the
	 * event*/
	LWS_CALLBACK_CHILD_WRITE_VIA_PARENT			= 68,
	/**< Child has been marked with parent_carries_io attribute, so
	 * lws_write directs the to this callback at the parent,
	 * in is a struct lws_write_passthru containing the args
	 * the lws_write() was called with.
	 */
	LWS_CALLBACK_CHILD_CLOSING				= 69,
	/**< Sent to parent to notify them a child is closing / being
	 * destroyed.  in is the child wsi.
	 */
	LWS_CALLBACK_CGI_PROCESS_ATTACH				= 70,
	/**< CGI: Sent when the CGI process is spawned for the wsi.  The
	 * len parameter is the PID of the child process */

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
///@}

/*! \defgroup generic hash
 * ## Generic Hash related functions
 *
 * Lws provides generic hash / digest accessors that abstract the ones
 * provided by whatever OpenSSL library you are linking against.
 *
 * It lets you use the same code if you build against mbedtls or OpenSSL
 * for example.
 */
///@{

#ifdef LWS_OPENSSL_SUPPORT

#if defined(LWS_WITH_MBEDTLS)
#include <mbedtls/sha1.h>
#include <mbedtls/sha256.h>
#include <mbedtls/sha512.h>
#endif

#define LWS_GENHASH_TYPE_SHA1		0
#define LWS_GENHASH_TYPE_SHA256		1
#define LWS_GENHASH_TYPE_SHA512		2

struct lws_genhash_ctx {
        uint8_t type;
#if defined(LWS_WITH_MBEDTLS)
        union {
		mbedtls_sha1_context sha1;
		mbedtls_sha256_context sha256;
		mbedtls_sha512_context sha512;
        } u;
#else
        const EVP_MD *evp_type;
        EVP_MD_CTX *mdctx;
#endif
};

/** lws_genhash_size() - get hash size in bytes
 *
 * \param type:	one of LWS_GENHASH_TYPE_...
 *
 * Returns number of bytes in this type of hash
 */
LWS_VISIBLE LWS_EXTERN size_t LWS_WARN_UNUSED_RESULT
lws_genhash_size(int type);

/** lws_genhash_init() - prepare your struct lws_genhash_ctx for use
 *
 * \param ctx: your struct lws_genhash_ctx
 * \param type:	one of LWS_GENHASH_TYPE_...
 *
 * Initializes the hash context for the type you requested
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_genhash_init(struct lws_genhash_ctx *ctx, int type);

/** lws_genhash_update() - digest len bytes of the buffer starting at in
 *
 * \param ctx: your struct lws_genhash_ctx
 * \param in: start of the bytes to digest
 * \param len: count of bytes to digest
 *
 * Updates the state of your hash context to reflect digesting len bytes from in
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_genhash_update(struct lws_genhash_ctx *ctx, const void *in, size_t len);

/** lws_genhash_destroy() - copy out the result digest and destroy the ctx
 *
 * \param ctx: your struct lws_genhash_ctx
 * \param result: NULL, or where to copy the result hash
 *
 * Finalizes the hash and copies out the digest.  Destroys any allocations such
 * that ctx can safely go out of scope after calling this.
 *
 * NULL result is supported so that you can destroy the ctx cleanly on error
 * conditions, where there is no valid result.
 */
LWS_VISIBLE LWS_EXTERN int
lws_genhash_destroy(struct lws_genhash_ctx *ctx, void *result);

#endif

///@}

/*! \defgroup extensions Extension related functions
 * ##Extension releated functions
 *
 *  Ws defines optional extensions, lws provides the ability to implement these
 *  in user code if so desired.
 *
 *  We provide one extensions permessage-deflate.
 */
///@{

/*
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */
enum lws_extension_callback_reasons {
	LWS_EXT_CB_SERVER_CONTEXT_CONSTRUCT		=  0,
	LWS_EXT_CB_CLIENT_CONTEXT_CONSTRUCT		=  1,
	LWS_EXT_CB_SERVER_CONTEXT_DESTRUCT		=  2,
	LWS_EXT_CB_CLIENT_CONTEXT_DESTRUCT		=  3,
	LWS_EXT_CB_CONSTRUCT				=  4,
	LWS_EXT_CB_CLIENT_CONSTRUCT			=  5,
	LWS_EXT_CB_CHECK_OK_TO_REALLY_CLOSE		=  6,
	LWS_EXT_CB_CHECK_OK_TO_PROPOSE_EXTENSION	=  7,
	LWS_EXT_CB_DESTROY				=  8,
	LWS_EXT_CB_DESTROY_ANY_WSI_CLOSING		=  9,
	LWS_EXT_CB_ANY_WSI_ESTABLISHED			= 10,
	LWS_EXT_CB_PACKET_RX_PREPARSE			= 11,
	LWS_EXT_CB_PACKET_TX_PRESEND			= 12,
	LWS_EXT_CB_PACKET_TX_DO_SEND			= 13,
	LWS_EXT_CB_HANDSHAKE_REPLY_TX			= 14,
	LWS_EXT_CB_FLUSH_PENDING_TX			= 15,
	LWS_EXT_CB_EXTENDED_PAYLOAD_RX			= 16,
	LWS_EXT_CB_CAN_PROXY_CLIENT_CONNECTION		= 17,
	LWS_EXT_CB_1HZ					= 18,
	LWS_EXT_CB_REQUEST_ON_WRITEABLE			= 19,
	LWS_EXT_CB_IS_WRITEABLE				= 20,
	LWS_EXT_CB_PAYLOAD_TX				= 21,
	LWS_EXT_CB_PAYLOAD_RX				= 22,
	LWS_EXT_CB_OPTION_DEFAULT			= 23,
	LWS_EXT_CB_OPTION_SET				= 24,
	LWS_EXT_CB_OPTION_CONFIRM			= 25,
	LWS_EXT_CB_NAMED_OPTION_SET			= 26,

	/****** add new things just above ---^ ******/
};

/** enum lws_ext_options_types */
enum lws_ext_options_types {
	EXTARG_NONE, /**< does not take an argument */
	EXTARG_DEC,  /**< requires a decimal argument */
	EXTARG_OPT_DEC /**< may have an optional decimal argument */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
};

/** struct lws_ext_options -	Option arguments to the extension.  These are
 *				used in the negotiation at ws upgrade time.
 *				The helper function lws_ext_parse_options()
 *				uses these to generate callbacks */
struct lws_ext_options {
	const char *name; /**< Option name, eg, "server_no_context_takeover" */
	enum lws_ext_options_types type; /**< What kind of args the option can take */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
};

/** struct lws_ext_option_arg */
struct lws_ext_option_arg {
	const char *option_name; /**< may be NULL, option_index used then */
	int option_index; /**< argument ordinal to use if option_name missing */
	const char *start; /**< value */
	int len; /**< length of value */
};

/**
 * typedef lws_extension_callback_function() - Hooks to allow extensions to operate
 * \param context:	Websockets context
 * \param ext:	This extension
 * \param wsi:	Opaque websocket instance pointer
 * \param reason:	The reason for the call
 * \param user:	Pointer to ptr to per-session user data allocated by library
 * \param in:		Pointer used for some callback reasons
 * \param len:	Length set for some callback reasons
 *
 *	Each extension that is active on a particular connection receives
 *	callbacks during the connection lifetime to allow the extension to
 *	operate on websocket data and manage itself.
 *
 *	Libwebsockets takes care of allocating and freeing "user" memory for
 *	each active extension on each connection.  That is what is pointed to
 *	by the user parameter.
 *
 *	LWS_EXT_CB_CONSTRUCT:  called when the server has decided to
 *		select this extension from the list provided by the client,
 *		just before the server will send back the handshake accepting
 *		the connection with this extension active.  This gives the
 *		extension a chance to initialize its connection context found
 *		in user.
 *
 *	LWS_EXT_CB_CLIENT_CONSTRUCT: same as LWS_EXT_CB_CONSTRUCT
 *		but called when client is instantiating this extension.  Some
 *		extensions will work the same on client and server side and then
 *		you can just merge handlers for both CONSTRUCTS.
 *
 *	LWS_EXT_CB_DESTROY:  called when the connection the extension was
 *		being used on is about to be closed and deallocated.  It's the
 *		last chance for the extension to deallocate anything it has
 *		allocated in the user data (pointed to by user) before the
 *		user data is deleted.  This same callback is used whether you
 *		are in client or server instantiation context.
 *
 *	LWS_EXT_CB_PACKET_RX_PREPARSE: when this extension was active on
 *		a connection, and a packet of data arrived at the connection,
 *		it is passed to this callback to give the extension a chance to
 *		change the data, eg, decompress it.  user is pointing to the
 *		extension's private connection context data, in is pointing
 *		to an lws_tokens struct, it consists of a char * pointer called
 *		token, and an int called token_len.  At entry, these are
 *		set to point to the received buffer and set to the content
 *		length.  If the extension will grow the content, it should use
 *		a new buffer allocated in its private user context data and
 *		set the pointed-to lws_tokens members to point to its buffer.
 *
 *	LWS_EXT_CB_PACKET_TX_PRESEND: this works the same way as
 *		LWS_EXT_CB_PACKET_RX_PREPARSE above, except it gives the
 *		extension a chance to change websocket data just before it will
 *		be sent out.  Using the same lws_token pointer scheme in in,
 *		the extension can change the buffer and the length to be
 *		transmitted how it likes.  Again if it wants to grow the
 *		buffer safely, it should copy the data into its own buffer and
 *		set the lws_tokens token pointer to it.
 *
 *	LWS_EXT_CB_ARGS_VALIDATE:
 */
typedef int
lws_extension_callback_function(struct lws_context *context,
			      const struct lws_extension *ext, struct lws *wsi,
			      enum lws_extension_callback_reasons reason,
			      void *user, void *in, size_t len);

/** struct lws_extension -	An extension we support */
struct lws_extension {
	const char *name; /**< Formal extension name, eg, "permessage-deflate" */
	lws_extension_callback_function *callback; /**< Service callback */
	const char *client_offer; /**< String containing exts and options client offers */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
};

/**
 * lws_set_extension_option(): set extension option if possible
 *
 * \param wsi:	websocket connection
 * \param ext_name:	name of ext, like "permessage-deflate"
 * \param opt_name:	name of option, like "rx_buf_size"
 * \param opt_val:	value to set option to
 */
LWS_VISIBLE LWS_EXTERN int
lws_set_extension_option(struct lws *wsi, const char *ext_name,
			 const char *opt_name, const char *opt_val);

#ifndef LWS_NO_EXTENSIONS
/* lws_get_internal_extensions() - DEPRECATED
 *
 * \Deprecated There is no longer a set internal extensions table.  The table is provided
 * by user code along with application-specific settings.  See the test
 * client and server for how to do.
 */
static LWS_INLINE LWS_WARN_DEPRECATED const struct lws_extension *
lws_get_internal_extensions(void) { return NULL; }

/**
 * lws_ext_parse_options() - deal with parsing negotiated extension options
 *
 * \param ext: related extension struct
 * \param wsi:	websocket connection
 * \param ext_user: per-connection extension private data
 * \param opts: list of supported options
 * \param o: option string to parse
 * \param len: length
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_ext_parse_options(const struct lws_extension *ext, struct lws *wsi,
		       void *ext_user, const struct lws_ext_options *opts,
		       const char *o, int len);
#endif

/** lws_extension_callback_pm_deflate() - extension for RFC7692
 *
 * \param context:	lws context
 * \param ext:	related lws_extension struct
 * \param wsi:	websocket connection
 * \param reason:	incoming callback reason
 * \param user:	per-connection extension private data
 * \param in:	pointer parameter
 * \param len:	length parameter
 *
 * Built-in callback implementing RFC7692 permessage-deflate
 */
LWS_EXTERN
int lws_extension_callback_pm_deflate(
	struct lws_context *context, const struct lws_extension *ext,
	struct lws *wsi, enum lws_extension_callback_reasons reason,
	void *user, void *in, size_t len);

/*
 * The internal exts are part of the public abi
 * If we add more extensions, publish the callback here  ------v
 */
///@}

/*! \defgroup Protocols-and-Plugins Protocols and Plugins
 * \ingroup lwsapi
 *
 * ##Protocol and protocol plugin -related apis
 *
 * Protocols bind ws protocol names to a custom callback specific to that
 * protocol implementaion.
 *
 * A list of protocols can be passed in at context creation time, but it is
 * also legal to leave that NULL and add the protocols and their callback code
 * using plugins.
 *
 * Plugins are much preferable compared to cut and pasting code into an
 * application each time, since they can be used standalone.
 */
///@{
/** struct lws_protocols -	List of protocols and handlers client or server
 *					supports. */

struct lws_protocols {
	const char *name;
	/**< Protocol name that must match the one given in the client
	 * Javascript new WebSocket(url, 'protocol') name. */
	lws_callback_function *callback;
	/**< The service callback used for this protocol.  It allows the
	 * service action for an entire protocol to be encapsulated in
	 * the protocol-specific callback */
	size_t per_session_data_size;
	/**< Each new connection using this protocol gets
	 * this much memory allocated on connection establishment and
	 * freed on connection takedown.  A pointer to this per-connection
	 * allocation is passed into the callback in the 'user' parameter */
	size_t rx_buffer_size;
	/**< lws allocates this much space for rx data and informs callback
	 * when something came.  Due to rx flow control, the callback may not
	 * be able to consume it all without having to return to the event
	 * loop.  That is supported in lws.
	 *
	 * If .tx_packet_size is 0, this also controls how much may be sent at once
	 * for backwards compatibility.
	 */
	unsigned int id;
	/**< ignored by lws, but useful to contain user information bound
	 * to the selected protocol.  For example if this protocol was
	 * called "myprotocol-v2", you might set id to 2, and the user
	 * code that acts differently according to the version can do so by
	 * switch (wsi->protocol->id), user code might use some bits as
	 * capability flags based on selected protocol version, etc. */
	void *user; /**< ignored by lws, but user code can pass a pointer
			here it can later access from the protocol callback */
	size_t tx_packet_size;
	/**< 0 indicates restrict send() size to .rx_buffer_size for backwards-
	 * compatibility.
	 * If greater than zero, a single send() is restricted to this amount
	 * and any remainder is buffered by lws and sent afterwards also in
	 * these size chunks.  Since that is expensive, it's preferable
	 * to restrict one fragment you are trying to send to match this
	 * size.
	 */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
};

struct lws_vhost;

/**
 * lws_vhost_name_to_protocol() - get vhost's protocol object from its name
 *
 * \param vh: vhost to search
 * \param name: protocol name
 *
 * Returns NULL or a pointer to the vhost's protocol of the requested name
 */
LWS_VISIBLE LWS_EXTERN const struct lws_protocols *
lws_vhost_name_to_protocol(struct lws_vhost *vh, const char *name);

/**
 * lws_get_protocol() - Returns a protocol pointer from a websocket
 *				  connection.
 * \param wsi:	pointer to struct websocket you want to know the protocol of
 *
 *
 *	Some apis can act on all live connections of a given protocol,
 *	this is how you can get a pointer to the active protocol if needed.
 */
LWS_VISIBLE LWS_EXTERN const struct lws_protocols *
lws_get_protocol(struct lws *wsi);

/** lws_protocol_get() -  deprecated: use lws_get_protocol */
LWS_VISIBLE LWS_EXTERN const struct lws_protocols *
lws_protocol_get(struct lws *wsi) LWS_WARN_DEPRECATED;

/**
 * lws_protocol_vh_priv_zalloc() - Allocate and zero down a protocol's per-vhost
 *				   storage
 * \param vhost:	vhost the instance is related to
 * \param prot:		protocol the instance is related to
 * \param size:		bytes to allocate
 *
 * Protocols often find it useful to allocate a per-vhost struct, this is a
 * helper to be called in the per-vhost init LWS_CALLBACK_PROTOCOL_INIT
 */
LWS_VISIBLE LWS_EXTERN void *
lws_protocol_vh_priv_zalloc(struct lws_vhost *vhost, const struct lws_protocols *prot,
			    int size);

/**
 * lws_protocol_vh_priv_get() - retreive a protocol's per-vhost storage
 *
 * \param vhost:	vhost the instance is related to
 * \param prot:		protocol the instance is related to
 *
 * Recover a pointer to the allocated per-vhost storage for the protocol created
 * by lws_protocol_vh_priv_zalloc() earlier
 */
LWS_VISIBLE LWS_EXTERN void *
lws_protocol_vh_priv_get(struct lws_vhost *vhost, const struct lws_protocols *prot);

/**
 * lws_adjust_protocol_psds - change a vhost protocol's per session data size
 *
 * \param wsi: a connection with the protocol to change
 * \param new_size: the new size of the per session data size for the protocol
 *
 * Returns user_space for the wsi, after allocating
 *
 * This should not be used except to initalize a vhost protocol's per session
 * data size one time, before any connections are accepted.
 *
 * Sometimes the protocol wraps another protocol and needs to discover and set
 * its per session data size at runtime.
 */
LWS_VISIBLE LWS_EXTERN void *
lws_adjust_protocol_psds(struct lws *wsi, size_t new_size);

/**
 * lws_finalize_startup() - drop initial process privileges
 *
 * \param context:	lws context
 *
 * This is called after the end of the vhost protocol initializations, but
 * you may choose to call it earlier
 */
LWS_VISIBLE LWS_EXTERN int
lws_finalize_startup(struct lws_context *context);

LWS_VISIBLE LWS_EXTERN int
lws_protocol_init(struct lws_context *context);

#ifdef LWS_WITH_PLUGINS

/* PLUGINS implies LIBUV */

#define LWS_PLUGIN_API_MAGIC 180

/** struct lws_plugin_capability - how a plugin introduces itself to lws */
struct lws_plugin_capability {
	unsigned int api_magic;	/**< caller fills this in, plugin fills rest */
	const struct lws_protocols *protocols; /**< array of supported protocols provided by plugin */
	int count_protocols; /**< how many protocols */
	const struct lws_extension *extensions; /**< array of extensions provided by plugin */
	int count_extensions; /**< how many extensions */
};

typedef int (*lws_plugin_init_func)(struct lws_context *,
				    struct lws_plugin_capability *);
typedef int (*lws_plugin_destroy_func)(struct lws_context *);

/** struct lws_plugin */
struct lws_plugin {
	struct lws_plugin *list; /**< linked list */
#if (UV_VERSION_MAJOR > 0)
	uv_lib_t lib; /**< shared library pointer */
#else
	void *l; /**< so we can compile on ancient libuv */
#endif
	char name[64]; /**< name of the plugin */
	struct lws_plugin_capability caps; /**< plugin capabilities */
};

#endif

///@}


/*! \defgroup generic-sessions plugin: generic-sessions
 * \ingroup Protocols-and-Plugins
 *
 * ##Plugin Generic-sessions related
 *
 * generic-sessions plugin provides a reusable, generic session and login /
 * register / forgot password framework including email verification.
 */
///@{

#define LWSGS_EMAIL_CONTENT_SIZE 16384
/**< Maximum size of email we might send */

/* SHA-1 binary and hexified versions */
/** typedef struct lwsgw_hash_bin */
typedef struct { unsigned char bin[20]; /**< binary representation of hash */} lwsgw_hash_bin;
/** typedef struct lwsgw_hash */
typedef struct { char id[41]; /**< ascii hex representation of hash */ } lwsgw_hash;

/** enum lwsgs_auth_bits */
enum lwsgs_auth_bits {
	LWSGS_AUTH_LOGGED_IN = 1, /**< user is logged in as somebody */
	LWSGS_AUTH_ADMIN = 2,	/**< logged in as the admin user */
	LWSGS_AUTH_VERIFIED = 4,  /**< user has verified his email */
	LWSGS_AUTH_FORGOT_FLOW = 8,	/**< he just completed "forgot password" flow */
};

/** struct lws_session_info - information about user session status */
struct lws_session_info {
	char username[32]; /**< username logged in as, or empty string */
	char email[100]; /**< email address associated with login, or empty string */
	char ip[72]; /**< ip address session was started from */
	unsigned int mask; /**< access rights mask associated with session
	 	 	    * see enum lwsgs_auth_bits */
	char session[42]; /**< session id string, usable as opaque uid when not logged in */
};

/** enum lws_gs_event */
enum lws_gs_event {
	LWSGSE_CREATED, /**< a new user was created */
	LWSGSE_DELETED  /**< an existing user was deleted */
};

/** struct lws_gs_event_args */
struct lws_gs_event_args {
	enum lws_gs_event event; /**< which event happened */
	const char *username; /**< which username the event happened to */
	const char *email; /**< the email address of that user */
};

///@}


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
	 * built-in listener */
	const char *iface;
	/**< VHOST: NULL to bind the listen socket to all interfaces, or the
	 * interface name, eg, "eth2"
	 * If options specifies LWS_SERVER_OPTION_UNIX_SOCK, this member is
	 * the pathname of a UNIX domain socket. you can use the UNIX domain
	 * sockets in abstract namespace, by prepending an at symbol to the
	 * socket name. */
	const struct lws_protocols *protocols;
	/**< VHOST: Array of structures listing supported protocols and a protocol-
	 * specific callback for each one.  The list is ended with an
	 * entry that has a NULL callback pointer. */
	const struct lws_extension *extensions;
	/**< VHOST: NULL or array of lws_extension structs listing the
	 * extensions this context supports. */
	const struct lws_token_limits *token_limits;
	/**< CONTEXT: NULL or struct lws_token_limits pointer which is initialized
	 * with a token length limit for each possible WSI_TOKEN_ */
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
	/**< VHOST: List of valid ciphers to use (eg,
	 * "RC4-MD5:RC4-SHA:AES128-SHA:AES256-SHA:HIGH:!DSS:!aNULL"
	 * or you can leave it as NULL to get "DEFAULT" (For backwards
	 * compatibility, this can also be used to pass the client cipher
	 * list when setting up a vhost client SSL context,
	 * but it is preferred to use .client_ssl_cipher_list for that.)*/
	const char *http_proxy_address;
	/**< VHOST: If non-NULL, attempts to proxy via the given address.
	 * If proxy auth is required, use format "username:password\@server:port" */
	unsigned int http_proxy_port;
	/**< VHOST: If http_proxy_address was non-NULL, uses this port */
	int gid;
	/**< CONTEXT: group id to change to after setting listen socket, or -1. */
	int uid;
	/**< CONTEXT: user id to change to after setting listen socket, or -1. */
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
#ifdef LWS_OPENSSL_SUPPORT
	SSL_CTX *provided_client_ssl_ctx;
	/**< CONTEXT: If non-null, swap out libwebsockets ssl
 *		implementation for the one provided by provided_ssl_ctx.
 *		Libwebsockets no longer is responsible for freeing the context
 *		if this option is selected. */
#else /* maintain structure layout either way */
	void *provided_client_ssl_ctx; /**< dummy if ssl disabled */
#endif

	short max_http_header_data;
	/**< CONTEXT: The max amount of header payload that can be handled
	 * in an http request (unrecognized header payload is dropped) */
	short max_http_header_pool;
	/**< CONTEXT: The max number of connections with http headers that
	 * can be processed simultaneously (the corresponding memory is
	 * allocated for the lifetime of the context).  If the pool is
	 * busy new incoming connections must wait for accept until one
	 * becomes free. */

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
	/**< VHOST: if NULL, defaults to initializing server with "prime256v1" */
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
	/**< VHOST: (default = 0 = 60s) seconds to allow remote
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
	/**< VHOST: Any bits set here will be set as SSL options */
	long ssl_options_clear;
	/**< VHOST: Any bits set here will be cleared as SSL options */
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
	/**< CONTEXT: 0 (no limit) or limit of simultaneous SSL sessions possible.*/
	const char *socks_proxy_address;
	/**< VHOST: If non-NULL, attempts to proxy via the given address.
	 * If proxy auth is required, use format "username:password\@server:port" */
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
	/**< CONTEXT: after context creation http2_settings[1] thru [6] have
	 *	      been set to the lws platform default values.
	 *   VHOST:   if http2_settings[0] is nonzero, the values given in
	 *	      http2_settings[1]..[6] are used instead of the lws
	 *	      platform default values.
	 *	      Just leave all at 0 if you don't care.
	 */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility
	 *
	 * The below is to ensure later library versions with new
	 * members added above will see 0 (default) even if the app
	 * was not built against the newer headers.
	 */

	void *_unused[8]; /**< dummy */
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
lws_create_context(struct lws_context_creation_info *info);

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

LWS_VISIBLE LWS_EXTERN void
lws_context_destroy2(struct lws_context *context);

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
 *	Otherwise if you attach the deprecated context to the replacement
 *	context when it has been created using lws_context_attach_deprecated()
 *	both any deprecated and the new context will service their connections.
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
		 struct lws_context_creation_info *info);

/**
 * lws_vhost_destroy() - Destroy a vhost (virtual server context)
 *
 * \param vh:	pointer to result of lws_create_vhost()
 *
 * This function destroys a vhost.  Normally, if you just want to exit,
 * then lws_destroy_context() will take care of everything.  If you want
 * to destroy an individual vhost and all connections and allocations, you
 * can do it with this.
 *
 * If the vhost has a listen sockets shared by other vhosts, it will be given
 * to one of the vhosts sharing it rather than closed.
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
	LCCSCF_ALLOW_EXPIRED			= (1 << 3)
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
	/**< nonzero for ssl */
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
	 * use "GET" to be a simple http client connection */
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
lws_client_connect_via_info(struct lws_client_connect_info * ccinfo);

/**
 * lws_client_connect() - Connect to another websocket server
 * 		\deprecated DEPRECATED use lws_client_connect_via_info
 * \param clients:	Websocket context
 * \param address:	Remote server address, eg, "myserver.com"
 * \param port:	Port to connect to on the remote server, eg, 80
 * \param ssl_connection:	0 = ws://, 1 = wss:// encrypted, 2 = wss:// allow self
 *			signed certs
 * \param path:	Websocket path on server
 * \param host:	Hostname on server
 * \param origin:	Socket origin name
 * \param protocol:	Comma-separated list of protocols being asked for from
 *		the server, or just one.  The server will pick the one it
 *		likes best.  If you don't want to specify a protocol, which is
 *		legal, use NULL here.
 * \param ietf_version_or_minus_one: -1 to ask to connect using the default, latest
 *		protocol supported, or the specific protocol ordinal
 *
 *	This function creates a connection to a remote server
 */
/* deprecated, use lws_client_connect_via_info() */
LWS_VISIBLE LWS_EXTERN struct lws * LWS_WARN_UNUSED_RESULT
lws_client_connect(struct lws_context *clients, const char *address,
		   int port, int ssl_connection, const char *path,
		   const char *host, const char *origin, const char *protocol,
		   int ietf_version_or_minus_one) LWS_WARN_DEPRECATED;
/* deprecated, use lws_client_connect_via_info() */
/**
 * lws_client_connect_extended() - Connect to another websocket server
 * 			\deprecated DEPRECATED use lws_client_connect_via_info
 * \param clients:	Websocket context
 * \param address:	Remote server address, eg, "myserver.com"
 * \param port:	Port to connect to on the remote server, eg, 80
 * \param ssl_connection:	0 = ws://, 1 = wss:// encrypted, 2 = wss:// allow self
 *			signed certs
 * \param path:	Websocket path on server
 * \param host:	Hostname on server
 * \param origin:	Socket origin name
 * \param protocol:	Comma-separated list of protocols being asked for from
 *		the server, or just one.  The server will pick the one it
 *		likes best.
 * \param ietf_version_or_minus_one: -1 to ask to connect using the default, latest
 *		protocol supported, or the specific protocol ordinal
 * \param userdata: Pre-allocated user data
 *
 *	This function creates a connection to a remote server
 */
LWS_VISIBLE LWS_EXTERN struct lws * LWS_WARN_UNUSED_RESULT
lws_client_connect_extended(struct lws_context *clients, const char *address,
			    int port, int ssl_connection, const char *path,
			    const char *host, const char *origin,
			    const char *protocol, int ietf_version_or_minus_one,
			    void *userdata) LWS_WARN_DEPRECATED;

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
 *	This function lets a call to lws_service() waiting for a timeout
 *	immediately return.
 *
 *	It works by creating a phony event and then swallowing it silently.
 *
 *	The reason it may be needed is when waiting in poll(), changes to
 *	the event masks are ignored by the OS until poll() is reentered.  This
 *	lets you halt the poll() wait and make the reentry happen immediately
 *	instead of having the wait out the rest of the poll timeout.
 */
LWS_VISIBLE LWS_EXTERN void
lws_cancel_service_pt(struct lws *wsi);

/**
 * lws_cancel_service() - Cancel wait for new pending socket activity
 * \param context:	Websocket context
 *
 *	This function let a call to lws_service() waiting for a timeout
 *	immediately return.
 *
 *	What it basically does is provide a fake event that will be swallowed,
 *	so the wait in poll() is ended.  That's useful because poll() doesn't
 *	attend to changes in POLLIN/OUT/ERR until it re-enters the wait.
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

/*! \defgroup http HTTP

    Modules related to handling HTTP
*/
//@{

/*! \defgroup httpft HTTP File transfer
 * \ingroup http

    APIs for sending local files in response to HTTP requests
*/
//@{

/**
 * lws_get_mimetype() - Determine mimetype to use from filename
 *
 * \param file:		filename
 * \param m:		NULL, or mount context
 *
 * This uses a canned list of known filetypes first, if no match and m is
 * non-NULL, then tries a list of per-mount file suffix to mimtype mappings.
 *
 * Returns either NULL or a pointer to the mimetype matching the file.
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_get_mimetype(const char *file, const struct lws_http_mount *m);

/**
 * lws_serve_http_file() - Send a file back to the client using http
 * \param wsi:		Websocket instance (available from user callback)
 * \param file:		The file to issue over http
 * \param content_type:	The http content type, eg, text/html
 * \param other_headers:	NULL or pointer to header string
 * \param other_headers_len:	length of the other headers if non-NULL
 *
 *	This function is intended to be called from the callback in response
 *	to http requests from the client.  It allows the callback to issue
 *	local files down the http link in a single step.
 *
 *	Returning <0 indicates error and the wsi should be closed.  Returning
 *	>0 indicates the file was completely sent and
 *	lws_http_transaction_completed() called on the wsi (and close if != 0)
 *	==0 indicates the file transfer is started and needs more service later,
 *	the wsi should be left alone.
 */
LWS_VISIBLE LWS_EXTERN int
lws_serve_http_file(struct lws *wsi, const char *file, const char *content_type,
		    const char *other_headers, int other_headers_len);

LWS_VISIBLE LWS_EXTERN int
lws_serve_http_file_fragment(struct lws *wsi);
//@}


enum http_status {
	HTTP_STATUS_CONTINUE					= 100,

	HTTP_STATUS_OK						= 200,
	HTTP_STATUS_NO_CONTENT					= 204,
	HTTP_STATUS_PARTIAL_CONTENT				= 206,

	HTTP_STATUS_MOVED_PERMANENTLY				= 301,
	HTTP_STATUS_FOUND					= 302,
	HTTP_STATUS_SEE_OTHER					= 303,
	HTTP_STATUS_NOT_MODIFIED				= 304,

	HTTP_STATUS_BAD_REQUEST					= 400,
	HTTP_STATUS_UNAUTHORIZED,
	HTTP_STATUS_PAYMENT_REQUIRED,
	HTTP_STATUS_FORBIDDEN,
	HTTP_STATUS_NOT_FOUND,
	HTTP_STATUS_METHOD_NOT_ALLOWED,
	HTTP_STATUS_NOT_ACCEPTABLE,
	HTTP_STATUS_PROXY_AUTH_REQUIRED,
	HTTP_STATUS_REQUEST_TIMEOUT,
	HTTP_STATUS_CONFLICT,
	HTTP_STATUS_GONE,
	HTTP_STATUS_LENGTH_REQUIRED,
	HTTP_STATUS_PRECONDITION_FAILED,
	HTTP_STATUS_REQ_ENTITY_TOO_LARGE,
	HTTP_STATUS_REQ_URI_TOO_LONG,
	HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE,
	HTTP_STATUS_REQ_RANGE_NOT_SATISFIABLE,
	HTTP_STATUS_EXPECTATION_FAILED,

	HTTP_STATUS_INTERNAL_SERVER_ERROR			= 500,
	HTTP_STATUS_NOT_IMPLEMENTED,
	HTTP_STATUS_BAD_GATEWAY,
	HTTP_STATUS_SERVICE_UNAVAILABLE,
	HTTP_STATUS_GATEWAY_TIMEOUT,
	HTTP_STATUS_HTTP_VERSION_NOT_SUPPORTED,
};
/*! \defgroup html-chunked-substitution HTML Chunked Substitution
 * \ingroup http
 *
 * ##HTML chunked Substitution
 *
 * APIs for receiving chunks of text, replacing a set of variable names via
 * a callback, and then prepending and appending HTML chunked encoding
 * headers.
 */
//@{

struct lws_process_html_args {
	char *p; /**< pointer to the buffer containing the data */
	int len; /**< length of the original data at p */
	int max_len; /**< maximum length we can grow the data to */
	int final; /**< set if this is the last chunk of the file */
};

typedef const char *(*lws_process_html_state_cb)(void *data, int index);

struct lws_process_html_state {
	char *start; /**< pointer to start of match */
	char swallow[16]; /**< matched character buffer */
	int pos; /**< position in match */
	void *data; /**< opaque pointer */
	const char * const *vars; /**< list of variable names */
	int count_vars; /**< count of variable names */

	lws_process_html_state_cb replace; /**< called on match to perform substitution */
};

/*! lws_chunked_html_process() - generic chunked substitution
 * \param args: buffer to process using chunked encoding
 * \param s: current processing state
 */
LWS_VISIBLE LWS_EXTERN int
lws_chunked_html_process(struct lws_process_html_args *args,
			 struct lws_process_html_state *s);
//@}

/** \defgroup HTTP-headers-read HTTP headers: read
 * \ingroup http
 *
 * ##HTTP header releated functions
 *
 *  In lws the client http headers are temporarily stored in a pool, only for the
 *  duration of the http part of the handshake.  It's because in most cases,
 *  the header content is ignored for the whole rest of the connection lifetime
 *  and would then just be taking up space needlessly.
 *
 *  During LWS_CALLBACK_HTTP when the URI path is delivered is the last time
 *  the http headers are still allocated, you can use these apis then to
 *  look at and copy out interesting header content (cookies, etc)
 *
 *  Notice that the header total length reported does not include a terminating
 *  '\0', however you must allocate for it when using the _copy apis.  So the
 *  length reported for a header containing "123" is 3, but you must provide
 *  a buffer of length 4 so that "123\0" may be copied into it, or the copy
 *  will fail with a nonzero return code.
 *
 *  In the special case of URL arguments, like ?x=1&y=2, the arguments are
 *  stored in a token named for the method, eg,  WSI_TOKEN_GET_URI if it
 *  was a GET or WSI_TOKEN_POST_URI if POST.  You can check the total
 *  length to confirm the method.
 *
 *  For URL arguments, each argument is stored urldecoded in a "fragment", so
 *  you can use the fragment-aware api lws_hdr_copy_fragment() to access each
 *  argument in turn: the fragments contain urldecoded strings like x=1 or y=2.
 *
 *  As a convenience, lws has an api that will find the fragment with a
 *  given name= part, lws_get_urlarg_by_name().
 */
///@{

/** struct lws_tokens
 * you need these to look at headers that have been parsed if using the
 * LWS_CALLBACK_FILTER_CONNECTION callback.  If a header from the enum
 * list below is absent, .token = NULL and token_len = 0.  Otherwise .token
 * points to .token_len chars containing that header content.
 */
struct lws_tokens {
	char *token; /**< pointer to start of the token */
	int token_len; /**< length of the token's value */
};

/* enum lws_token_indexes
 * these have to be kept in sync with lextable.h / minilex.c
 *
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */
enum lws_token_indexes {
	WSI_TOKEN_GET_URI					=  0,
	WSI_TOKEN_POST_URI					=  1,
	WSI_TOKEN_OPTIONS_URI					=  2,
	WSI_TOKEN_HOST						=  3,
	WSI_TOKEN_CONNECTION					=  4,
	WSI_TOKEN_UPGRADE					=  5,
	WSI_TOKEN_ORIGIN					=  6,
	WSI_TOKEN_DRAFT						=  7,
	WSI_TOKEN_CHALLENGE					=  8,
	WSI_TOKEN_EXTENSIONS					=  9,
	WSI_TOKEN_KEY1						= 10,
	WSI_TOKEN_KEY2						= 11,
	WSI_TOKEN_PROTOCOL					= 12,
	WSI_TOKEN_ACCEPT					= 13,
	WSI_TOKEN_NONCE						= 14,
	WSI_TOKEN_HTTP						= 15,
	WSI_TOKEN_HTTP2_SETTINGS				= 16,
	WSI_TOKEN_HTTP_ACCEPT					= 17,
	WSI_TOKEN_HTTP_AC_REQUEST_HEADERS			= 18,
	WSI_TOKEN_HTTP_IF_MODIFIED_SINCE			= 19,
	WSI_TOKEN_HTTP_IF_NONE_MATCH				= 20,
	WSI_TOKEN_HTTP_ACCEPT_ENCODING				= 21,
	WSI_TOKEN_HTTP_ACCEPT_LANGUAGE				= 22,
	WSI_TOKEN_HTTP_PRAGMA					= 23,
	WSI_TOKEN_HTTP_CACHE_CONTROL				= 24,
	WSI_TOKEN_HTTP_AUTHORIZATION				= 25,
	WSI_TOKEN_HTTP_COOKIE					= 26,
	WSI_TOKEN_HTTP_CONTENT_LENGTH				= 27,
	WSI_TOKEN_HTTP_CONTENT_TYPE				= 28,
	WSI_TOKEN_HTTP_DATE					= 29,
	WSI_TOKEN_HTTP_RANGE					= 30,
	WSI_TOKEN_HTTP_REFERER					= 31,
	WSI_TOKEN_KEY						= 32,
	WSI_TOKEN_VERSION					= 33,
	WSI_TOKEN_SWORIGIN					= 34,

	WSI_TOKEN_HTTP_COLON_AUTHORITY				= 35,
	WSI_TOKEN_HTTP_COLON_METHOD				= 36,
	WSI_TOKEN_HTTP_COLON_PATH				= 37,
	WSI_TOKEN_HTTP_COLON_SCHEME				= 38,
	WSI_TOKEN_HTTP_COLON_STATUS				= 39,

	WSI_TOKEN_HTTP_ACCEPT_CHARSET				= 40,
	WSI_TOKEN_HTTP_ACCEPT_RANGES				= 41,
	WSI_TOKEN_HTTP_ACCESS_CONTROL_ALLOW_ORIGIN		= 42,
	WSI_TOKEN_HTTP_AGE					= 43,
	WSI_TOKEN_HTTP_ALLOW					= 44,
	WSI_TOKEN_HTTP_CONTENT_DISPOSITION			= 45,
	WSI_TOKEN_HTTP_CONTENT_ENCODING				= 46,
	WSI_TOKEN_HTTP_CONTENT_LANGUAGE				= 47,
	WSI_TOKEN_HTTP_CONTENT_LOCATION				= 48,
	WSI_TOKEN_HTTP_CONTENT_RANGE				= 49,
	WSI_TOKEN_HTTP_ETAG					= 50,
	WSI_TOKEN_HTTP_EXPECT					= 51,
	WSI_TOKEN_HTTP_EXPIRES					= 52,
	WSI_TOKEN_HTTP_FROM					= 53,
	WSI_TOKEN_HTTP_IF_MATCH					= 54,
	WSI_TOKEN_HTTP_IF_RANGE					= 55,
	WSI_TOKEN_HTTP_IF_UNMODIFIED_SINCE			= 56,
	WSI_TOKEN_HTTP_LAST_MODIFIED				= 57,
	WSI_TOKEN_HTTP_LINK					= 58,
	WSI_TOKEN_HTTP_LOCATION					= 59,
	WSI_TOKEN_HTTP_MAX_FORWARDS				= 60,
	WSI_TOKEN_HTTP_PROXY_AUTHENTICATE			= 61,
	WSI_TOKEN_HTTP_PROXY_AUTHORIZATION			= 62,
	WSI_TOKEN_HTTP_REFRESH					= 63,
	WSI_TOKEN_HTTP_RETRY_AFTER				= 64,
	WSI_TOKEN_HTTP_SERVER					= 65,
	WSI_TOKEN_HTTP_SET_COOKIE				= 66,
	WSI_TOKEN_HTTP_STRICT_TRANSPORT_SECURITY		= 67,
	WSI_TOKEN_HTTP_TRANSFER_ENCODING			= 68,
	WSI_TOKEN_HTTP_USER_AGENT				= 69,
	WSI_TOKEN_HTTP_VARY					= 70,
	WSI_TOKEN_HTTP_VIA					= 71,
	WSI_TOKEN_HTTP_WWW_AUTHENTICATE				= 72,

	WSI_TOKEN_PATCH_URI					= 73,
	WSI_TOKEN_PUT_URI					= 74,
	WSI_TOKEN_DELETE_URI					= 75,

	WSI_TOKEN_HTTP_URI_ARGS					= 76,
	WSI_TOKEN_PROXY						= 77,
	WSI_TOKEN_HTTP_X_REAL_IP				= 78,
	WSI_TOKEN_HTTP1_0					= 79,
	WSI_TOKEN_X_FORWARDED_FOR				= 80,
	WSI_TOKEN_CONNECT					= 81,
	WSI_TOKEN_HEAD_URI					= 82,
	WSI_TOKEN_TE						= 83,
	/****** add new things just above ---^ ******/

	/* use token storage to stash these internally, not for
	 * user use */

	_WSI_TOKEN_CLIENT_SENT_PROTOCOLS,
	_WSI_TOKEN_CLIENT_PEER_ADDRESS,
	_WSI_TOKEN_CLIENT_URI,
	_WSI_TOKEN_CLIENT_HOST,
	_WSI_TOKEN_CLIENT_ORIGIN,
	_WSI_TOKEN_CLIENT_METHOD,
	_WSI_TOKEN_CLIENT_IFACE,

	/* always last real token index*/
	WSI_TOKEN_COUNT,

	/* parser state additions, no storage associated */
	WSI_TOKEN_NAME_PART,
	WSI_TOKEN_SKIPPING,
	WSI_TOKEN_SKIPPING_SAW_CR,
	WSI_PARSING_COMPLETE,
	WSI_INIT_TOKEN_MUXURL,
};

struct lws_token_limits {
	unsigned short token_limit[WSI_TOKEN_COUNT]; /**< max chars for this token */
};

/**
 * lws_token_to_string() - returns a textual representation of a hdr token index
 *
 * \param token: token index
 */
LWS_VISIBLE LWS_EXTERN const unsigned char *
lws_token_to_string(enum lws_token_indexes token);

/**
 * lws_hdr_total_length: report length of all fragments of a header totalled up
 *		The returned length does not include the space for a
 *		terminating '\0'
 *
 * \param wsi: websocket connection
 * \param h: which header index we are interested in
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_hdr_total_length(struct lws *wsi, enum lws_token_indexes h);

/**
 * lws_hdr_fragment_length: report length of a single fragment of a header
 *		The returned length does not include the space for a
 *		terminating '\0'
 *
 * \param wsi: websocket connection
 * \param h: which header index we are interested in
 * \param frag_idx: which fragment of h we want to get the length of
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_hdr_fragment_length(struct lws *wsi, enum lws_token_indexes h, int frag_idx);

/**
 * lws_hdr_copy() - copy a single fragment of the given header to a buffer
 *		The buffer length len must include space for an additional
 *		terminating '\0', or it will fail returning -1.
 *
 * \param wsi: websocket connection
 * \param dest: destination buffer
 * \param len: length of destination buffer
 * \param h: which header index we are interested in
 *
 * copies the whole, aggregated header, even if it was delivered in
 * several actual headers piece by piece
 */
LWS_VISIBLE LWS_EXTERN int
lws_hdr_copy(struct lws *wsi, char *dest, int len, enum lws_token_indexes h);

/**
 * lws_hdr_copy_fragment() - copy a single fragment of the given header to a buffer
 *		The buffer length len must include space for an additional
 *		terminating '\0', or it will fail returning -1.
 *		If the requested fragment index is not present, it fails
 *		returning -1.
 *
 * \param wsi: websocket connection
 * \param dest: destination buffer
 * \param len: length of destination buffer
 * \param h: which header index we are interested in
 * \param frag_idx: which fragment of h we want to copy
 *
 * Normally this is only useful
 * to parse URI arguments like ?x=1&y=2, token index WSI_TOKEN_HTTP_URI_ARGS
 * fragment 0 will contain "x=1" and fragment 1 "y=2"
 */
LWS_VISIBLE LWS_EXTERN int
lws_hdr_copy_fragment(struct lws *wsi, char *dest, int len,
		      enum lws_token_indexes h, int frag_idx);

/**
 * lws_get_urlarg_by_name() - return pointer to arg value if present
 * \param wsi: the connection to check
 * \param name: the arg name, like "token="
 * \param buf: the buffer to receive the urlarg (including the name= part)
 * \param len: the length of the buffer to receive the urlarg
 *
 *     Returns NULL if not found or a pointer inside buf to just after the
 *     name= part.
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_get_urlarg_by_name(struct lws *wsi, const char *name, char *buf, int len);
///@}

/*! \defgroup HTTP-headers-create HTTP headers: create
 *
 * ## HTTP headers: Create
 *
 * These apis allow you to create HTTP response headers in a way compatible with
 * both HTTP/1.x and HTTP/2.
 *
 * They each append to a buffer taking care about the buffer end, which is
 * passed in as a pointer.  When data is written to the buffer, the current
 * position p is updated accordingly.
 *
 * All of these apis are LWS_WARN_UNUSED_RESULT as they can run out of space
 * and fail with nonzero return.
 */
///@{

#define LWSAHH_CODE_MASK			((1 << 16) - 1)
#define LWSAHH_FLAG_NO_SERVER_NAME		(1 << 30)

/**
 * lws_add_http_header_status() - add the HTTP response status code
 *
 * \param wsi: the connection to check
 * \param code: an HTTP code like 200, 404 etc (see enum http_status)
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Adds the initial response code, so should be called first.
 *
 * Code may additionally take OR'd flags:
 *
 *    LWSAHH_FLAG_NO_SERVER_NAME:  don't apply server name header this time
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_add_http_header_status(struct lws *wsi,
			   unsigned int code, unsigned char **p,
			   unsigned char *end);
/**
 * lws_add_http_header_by_name() - append named header and value
 *
 * \param wsi: the connection to check
 * \param name: the hdr name, like "my-header"
 * \param value: the value after the = for this header
 * \param length: the length of the value
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Appends name: value to the headers
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_add_http_header_by_name(struct lws *wsi, const unsigned char *name,
			    const unsigned char *value, int length,
			    unsigned char **p, unsigned char *end);
/**
 * lws_add_http_header_by_token() - append given header and value
 *
 * \param wsi: the connection to check
 * \param token: the token index for the hdr
 * \param value: the value after the = for this header
 * \param length: the length of the value
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Appends name=value to the headers, but is able to take advantage of better
 * HTTP/2 coding mechanisms where possible.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_add_http_header_by_token(struct lws *wsi, enum lws_token_indexes token,
			     const unsigned char *value, int length,
			     unsigned char **p, unsigned char *end);
/**
 * lws_add_http_header_content_length() - append content-length helper
 *
 * \param wsi: the connection to check
 * \param content_length: the content length to use
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Appends content-length: content_length to the headers
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_add_http_header_content_length(struct lws *wsi,
				   lws_filepos_t content_length,
				   unsigned char **p, unsigned char *end);
/**
 * lws_finalize_http_header() - terminate header block
 *
 * \param wsi: the connection to check
 * \param p: pointer to current position in buffer pointer
 * \param end: pointer to end of buffer
 *
 * Indicates no more headers will be added
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_finalize_http_header(struct lws *wsi, unsigned char **p,
			 unsigned char *end);
///@}

/** \defgroup form-parsing  Form Parsing
 * \ingroup http
 * ##POSTed form parsing functions
 *
 * These lws_spa (stateful post arguments) apis let you parse and urldecode
 * POSTed form arguments, both using simple urlencoded and multipart transfer
 * encoding.
 *
 * It's capable of handling file uploads as well a named input parsing,
 * and the apis are the same for both form upload styles.
 *
 * You feed it a list of parameter names and it creates pointers to the
 * urldecoded arguments: file upload parameters pass the file data in chunks to
 * a user-supplied callback as they come.
 *
 * Since it's stateful, it handles the incoming data needing more than one
 * POST_BODY callback and has no limit on uploaded file size.
 */
///@{

/** enum lws_spa_fileupload_states */
enum lws_spa_fileupload_states {
	LWS_UFS_CONTENT,
	/**< a chunk of file content has arrived */
	LWS_UFS_FINAL_CONTENT,
	/**< the last chunk (possibly zero length) of file content has arrived */
	LWS_UFS_OPEN
	/**< a new file is starting to arrive */
};

/**
 * lws_spa_fileupload_cb() - callback to receive file upload data
 *
 * \param data: opt_data pointer set in lws_spa_create
 * \param name: name of the form field being uploaded
 * \param filename: original filename from client
 * \param buf: start of data to receive
 * \param len: length of data to receive
 * \param state: information about how this call relates to file
 *
 * Notice name and filename shouldn't be trusted, as they are passed from
 * HTTP provided by the client.
 */
typedef int (*lws_spa_fileupload_cb)(void *data, const char *name,
			const char *filename, char *buf, int len,
			enum lws_spa_fileupload_states state);

/** struct lws_spa - opaque urldecode parser capable of handling multipart
 *			and file uploads */
struct lws_spa;

/**
 * lws_spa_create() - create urldecode parser
 *
 * \param wsi: lws connection (used to find Content Type)
 * \param param_names: array of form parameter names, like "username"
 * \param count_params: count of param_names
 * \param max_storage: total amount of form parameter values we can store
 * \param opt_cb: NULL, or callback to receive file upload data.
 * \param opt_data: NULL, or user pointer provided to opt_cb.
 *
 * Creates a urldecode parser and initializes it.
 *
 * opt_cb can be NULL if you just want normal name=value parsing, however
 * if one or more entries in your form are bulk data (file transfer), you
 * can provide this callback and filter on the name callback parameter to
 * treat that urldecoded data separately.  The callback should return -1
 * in case of fatal error, and 0 if OK.
 */
LWS_VISIBLE LWS_EXTERN struct lws_spa *
lws_spa_create(struct lws *wsi, const char * const *param_names,
	       int count_params, int max_storage, lws_spa_fileupload_cb opt_cb,
	       void *opt_data);

/**
 * lws_spa_process() - parses a chunk of input data
 *
 * \param spa: the parser object previously created
 * \param in: incoming, urlencoded data
 * \param len: count of bytes valid at \param in
 */
LWS_VISIBLE LWS_EXTERN int
lws_spa_process(struct lws_spa *spa, const char *in, int len);

/**
 * lws_spa_finalize() - indicate incoming data completed
 *
 * \param spa: the parser object previously created
 */
LWS_VISIBLE LWS_EXTERN int
lws_spa_finalize(struct lws_spa *spa);

/**
 * lws_spa_get_length() - return length of parameter value
 *
 * \param spa: the parser object previously created
 * \param n: parameter ordinal to return length of value for
 */
LWS_VISIBLE LWS_EXTERN int
lws_spa_get_length(struct lws_spa *spa, int n);

/**
 * lws_spa_get_string() - return pointer to parameter value
 * \param spa: the parser object previously created
 * \param n: parameter ordinal to return pointer to value for
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_spa_get_string(struct lws_spa *spa, int n);

/**
 * lws_spa_destroy() - destroy parser object
 *
 * \param spa: the parser object previously created
 */
LWS_VISIBLE LWS_EXTERN int
lws_spa_destroy(struct lws_spa *spa);
///@}

/*! \defgroup urlendec Urlencode and Urldecode
 * \ingroup http
 *
 * ##HTML chunked Substitution
 *
 * APIs for receiving chunks of text, replacing a set of variable names via
 * a callback, and then prepending and appending HTML chunked encoding
 * headers.
 */
//@{

/**
 * lws_urlencode() - like strncpy but with urlencoding
 *
 * \param escaped: output buffer
 * \param string: input buffer ('/0' terminated)
 * \param len: output buffer max length
 *
 * Because urlencoding expands the output string, it's not
 * possible to do it in-place, ie, with escaped == string
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_urlencode(char *escaped, const char *string, int len);

/*
 * URLDECODE 1 / 2
 *
 * This simple urldecode only operates until the first '\0' and requires the
 * data to exist all at once
 */
/**
 * lws_urldecode() - like strncpy but with urldecoding
 *
 * \param string: output buffer
 * \param escaped: input buffer ('\0' terminated)
 * \param len: output buffer max length
 *
 * This is only useful for '\0' terminated strings
 *
 * Since urldecoding only shrinks the output string, it is possible to
 * do it in-place, ie, string == escaped
 *
 * Returns 0 if completed OK or nonzero for urldecode violation (non-hex chars
 * where hex required, etc)
 */
LWS_VISIBLE LWS_EXTERN int
lws_urldecode(char *string, const char *escaped, int len);
///@}
/**
 * lws_return_http_status() - Return simple http status
 * \param wsi:		Websocket instance (available from user callback)
 * \param code:		Status index, eg, 404
 * \param html_body:		User-readable HTML description < 1KB, or NULL
 *
 *	Helper to report HTTP errors back to the client cleanly and
 *	consistently
 */
LWS_VISIBLE LWS_EXTERN int
lws_return_http_status(struct lws *wsi, unsigned int code,
		       const char *html_body);

/**
 * lws_http_redirect() - write http redirect into buffer
 *
 * \param wsi:	websocket connection
 * \param code:	HTTP response code (eg, 301)
 * \param loc:	where to redirect to
 * \param len:	length of loc
 * \param p:	pointer current position in buffer (updated as we write)
 * \param end:	pointer to end of buffer
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_http_redirect(struct lws *wsi, int code, const unsigned char *loc, int len,
		  unsigned char **p, unsigned char *end);

/**
 * lws_http_transaction_completed() - wait for new http transaction or close
 * \param wsi:	websocket connection
 *
 *	Returns 1 if the HTTP connection must close now
 *	Returns 0 and resets connection to wait for new HTTP header /
 *	  transaction if possible
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_http_transaction_completed(struct lws *wsi);
///@}

/*! \defgroup pur Sanitize / purify SQL and JSON helpers
 *
 * ##Sanitize / purify SQL and JSON helpers
 *
 * APIs for escaping untrusted JSON and SQL safely before use
 */
//@{

/**
 * lws_sql_purify() - like strncpy but with escaping for sql quotes
 *
 * \param escaped: output buffer
 * \param string: input buffer ('/0' terminated)
 * \param len: output buffer max length
 *
 * Because escaping expands the output string, it's not
 * possible to do it in-place, ie, with escaped == string
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_sql_purify(char *escaped, const char *string, int len);

/**
 * lws_json_purify() - like strncpy but with escaping for json chars
 *
 * \param escaped: output buffer
 * \param string: input buffer ('/0' terminated)
 * \param len: output buffer max length
 *
 * Because escaping expands the output string, it's not
 * possible to do it in-place, ie, with escaped == string
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_json_purify(char *escaped, const char *string, int len);
///@}

/*! \defgroup ev libev helpers
 *
 * ##libev helpers
 *
 * APIs specific to libev event loop itegration
 */
///@{

#ifdef LWS_WITH_LIBEV
typedef void (lws_ev_signal_cb_t)(EV_P_ struct ev_signal *w, int revents);

LWS_VISIBLE LWS_EXTERN int
lws_ev_sigint_cfg(struct lws_context *context, int use_ev_sigint,
		  lws_ev_signal_cb_t *cb);

LWS_VISIBLE LWS_EXTERN int
lws_ev_initloop(struct lws_context *context, struct ev_loop *loop, int tsi);

LWS_VISIBLE LWS_EXTERN void
lws_ev_sigint_cb(struct ev_loop *loop, struct ev_signal *watcher, int revents);
#endif /* LWS_WITH_LIBEV */

///@}

/*! \defgroup uv libuv helpers
 *
 * ##libuv helpers
 *
 * APIs specific to libuv event loop itegration
 */
///@{
#ifdef LWS_WITH_LIBUV
LWS_VISIBLE LWS_EXTERN int
lws_uv_sigint_cfg(struct lws_context *context, int use_uv_sigint,
		  uv_signal_cb cb);

LWS_VISIBLE LWS_EXTERN void
lws_libuv_run(const struct lws_context *context, int tsi);

LWS_VISIBLE LWS_EXTERN void
lws_libuv_stop(struct lws_context *context);

LWS_VISIBLE LWS_EXTERN void
lws_libuv_stop_without_kill(const struct lws_context *context, int tsi);

LWS_VISIBLE LWS_EXTERN int
lws_uv_initloop(struct lws_context *context, uv_loop_t *loop, int tsi);

LWS_VISIBLE LWS_EXTERN uv_loop_t *
lws_uv_getloop(struct lws_context *context, int tsi);

LWS_VISIBLE LWS_EXTERN void
lws_uv_sigint_cb(uv_signal_t *watcher, int signum);

LWS_VISIBLE LWS_EXTERN void
lws_close_all_handles_in_loop(uv_loop_t *loop);
#endif /* LWS_WITH_LIBUV */
///@}

/*! \defgroup event libevent helpers
 *
 * ##libevent helpers
 *
 * APIs specific to libevent event loop itegration
 */
///@{

#ifdef LWS_WITH_LIBEVENT
typedef void (lws_event_signal_cb_t) (evutil_socket_t sock_fd, short revents,
		  void *ctx);

LWS_VISIBLE LWS_EXTERN int
lws_event_sigint_cfg(struct lws_context *context, int use_event_sigint,
		  lws_event_signal_cb_t cb);

LWS_VISIBLE LWS_EXTERN int
lws_event_initloop(struct lws_context *context, struct event_base *loop,
		  int tsi);

LWS_VISIBLE LWS_EXTERN void
lws_event_sigint_cb(evutil_socket_t sock_fd, short revents,
		  void *ctx);
#endif /* LWS_WITH_LIBEVENT */

///@}

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
	PENDING_TIMEOUT_AWAITING_EXTENSION_CONNECT_RESPONSE	=  7,
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

	/****** add new things just above ---^ ******/

	PENDING_TIMEOUT_USER_REASON_BASE			= 1000
};

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
///@}

/*! \defgroup sending-data Sending data

    APIs related to writing data on a connection
*/
//@{
#if !defined(LWS_SIZEOFPTR)
#define LWS_SIZEOFPTR (sizeof (void *))
#endif

#if defined(__x86_64__)
#define _LWS_PAD_SIZE 16	/* Intel recommended for best performance */
#else
#define _LWS_PAD_SIZE LWS_SIZEOFPTR   /* Size of a pointer on the target arch */
#endif
#define _LWS_PAD(n) (((n) % _LWS_PAD_SIZE) ? \
		((n) + (_LWS_PAD_SIZE - ((n) % _LWS_PAD_SIZE))) : (n))
/* last 2 is for lws-meta */
#define LWS_PRE _LWS_PAD(4 + 10 + 2)
/* used prior to 1.7 and retained for backward compatibility */
#define LWS_SEND_BUFFER_PRE_PADDING LWS_PRE
#define LWS_SEND_BUFFER_POST_PADDING 0

/*
 * NOTE: These public enums are part of the abi.  If you want to add one,
 * add it at where specified so existing users are unaffected.
 */
enum lws_write_protocol {
	LWS_WRITE_TEXT						= 0,
	/**< Send a ws TEXT message,the pointer must have LWS_PRE valid
	 * memory behind it.  The receiver expects only valid utf-8 in the
	 * payload */
	LWS_WRITE_BINARY					= 1,
	/**< Send a ws BINARY message, the pointer must have LWS_PRE valid
	 * memory behind it.  Any sequence of bytes is valid */
	LWS_WRITE_CONTINUATION					= 2,
	/**< Continue a previous ws message, the pointer must have LWS_PRE valid
	 * memory behind it */
	LWS_WRITE_HTTP						= 3,
	/**< Send HTTP content */

	/* LWS_WRITE_CLOSE is handled by lws_close_reason() */
	LWS_WRITE_PING						= 5,
	LWS_WRITE_PONG						= 6,

	/* Same as write_http but we know this write ends the transaction */
	LWS_WRITE_HTTP_FINAL					= 7,

	/* HTTP2 */

	LWS_WRITE_HTTP_HEADERS					= 8,
	/**< Send http headers (http2 encodes this payload and LWS_WRITE_HTTP
	 * payload differently, http 1.x links also handle this correctly. so
	 * to be compatible with both in the future,header response part should
	 * be sent using this regardless of http version expected)
	 */
	LWS_WRITE_HTTP_HEADERS_CONTINUATION			= 9,
	/**< Continuation of http/2 headers
	 */

	/****** add new things just above ---^ ******/

	/* flags */

	LWS_WRITE_NO_FIN = 0x40,
	/**< This part of the message is not the end of the message */

	LWS_WRITE_H2_STREAM_END = 0x80,
	/**< Flag indicates this packet should go out with STREAM_END if h2
	 * STREAM_END is allowed on DATA or HEADERS.
	 */

	LWS_WRITE_CLIENT_IGNORE_XOR_MASK = 0x80
	/**< client packet payload goes out on wire unmunged
	 * only useful for security tests since normal servers cannot
	 * decode the content if used */
};

/* used with LWS_CALLBACK_CHILD_WRITE_VIA_PARENT */

struct lws_write_passthru {
	struct lws *wsi;
	unsigned char *buf;
	size_t len;
	enum lws_write_protocol wp;
};


/**
 * lws_write() - Apply protocol then write data to client
 * \param wsi:	Websocket instance (available from user callback)
 * \param buf:	The data to send.  For data being sent on a websocket
 *		connection (ie, not default http), this buffer MUST have
 *		LWS_PRE bytes valid BEFORE the pointer.
 *		This is so the protocol header data can be added in-situ.
 * \param len:	Count of the data bytes in the payload starting from buf
 * \param protocol:	Use LWS_WRITE_HTTP to reply to an http connection, and one
 *		of LWS_WRITE_BINARY or LWS_WRITE_TEXT to send appropriate
 *		data on a websockets connection.  Remember to allow the extra
 *		bytes before and after buf if LWS_WRITE_BINARY or LWS_WRITE_TEXT
 *		are used.
 *
 *	This function provides the way to issue data back to the client
 *	for both http and websocket protocols.
 *
 * IMPORTANT NOTICE!
 *
 * When sending with websocket protocol
 *
 * LWS_WRITE_TEXT,
 * LWS_WRITE_BINARY,
 * LWS_WRITE_CONTINUATION,
 * LWS_WRITE_PING,
 * LWS_WRITE_PONG
 *
 * the send buffer has to have LWS_PRE bytes valid BEFORE
 * the buffer pointer you pass to lws_write().
 *
 * This allows us to add protocol info before and after the data, and send as
 * one packet on the network without payload copying, for maximum efficiency.
 *
 * So for example you need this kind of code to use lws_write with a
 * 128-byte payload
 *
 *   char buf[LWS_PRE + 128];
 *
 *   // fill your part of the buffer... for example here it's all zeros
 *   memset(&buf[LWS_PRE], 0, 128);
 *
 *   lws_write(wsi, &buf[LWS_PRE], 128, LWS_WRITE_TEXT);
 *
 * When sending HTTP, with
 *
 * LWS_WRITE_HTTP,
 * LWS_WRITE_HTTP_HEADERS
 * LWS_WRITE_HTTP_FINAL
 *
 * there is no protocol data prepended, and don't need to take care about the
 * LWS_PRE bytes valid before the buffer pointer.
 *
 * LWS_PRE is at least the frame nonce + 2 header + 8 length
 * LWS_SEND_BUFFER_POST_PADDING is deprecated, it's now 0 and can be left off.
 * The example apps no longer use it.
 *
 * Pad LWS_PRE to the CPU word size, so that word references
 * to the address immediately after the padding won't cause an unaligned access
 * error. Sometimes for performance reasons the recommended padding is even
 * larger than sizeof(void *).
 *
 *	In the case of sending using websocket protocol, be sure to allocate
 *	valid storage before and after buf as explained above.  This scheme
 *	allows maximum efficiency of sending data and protocol in a single
 *	packet while not burdening the user code with any protocol knowledge.
 *
 *	Return may be -1 for a fatal error needing connection close, or the
 *	number of bytes sent.
 *
 * Truncated Writes
 * ================
 *
 * The OS may not accept everything you asked to write on the connection.
 *
 * Posix defines POLLOUT indication from poll() to show that the connection
 * will accept more write data, but it doesn't specifiy how much.  It may just
 * accept one byte of whatever you wanted to send.
 *
 * LWS will buffer the remainder automatically, and send it out autonomously.
 *
 * During that time, WRITABLE callbacks will be suppressed.
 *
 * This is to handle corner cases where unexpectedly the OS refuses what we
 * usually expect it to accept.  You should try to send in chunks that are
 * almost always accepted in order to avoid the inefficiency of the buffering.
 */
LWS_VISIBLE LWS_EXTERN int
lws_write(struct lws *wsi, unsigned char *buf, size_t len,
	  enum lws_write_protocol protocol);

/* helper for case where buffer may be const */
#define lws_write_http(wsi, buf, len) \
	lws_write(wsi, (unsigned char *)(buf), len, LWS_WRITE_HTTP)
///@}

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
			  const struct lws_protocols *protocol, int reason)
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
			  const struct lws_protocols *protocol, int reason,
			  void *argp, size_t len);

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
 */
LWS_VISIBLE LWS_EXTERN int
lws_callback_vhost_protocols(struct lws *wsi, int reason, void *in, int len);

LWS_VISIBLE LWS_EXTERN int
lws_callback_http_dummy(struct lws *wsi, enum lws_callback_reasons reason,
		    void *user, void *in, size_t len);

/**
 * lws_get_socket_fd() - returns the socket file descriptor
 *
 * You will not need this unless you are doing something special
 *
 * \param wsi:	Websocket connection instance
 */
LWS_VISIBLE LWS_EXTERN int
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
LWS_VISIBLE LWS_EXTERN size_t
lws_get_peer_write_allowance(struct lws *wsi);
///@}

enum {
	/*
	 * Flags for enable and disable rxflow with reason bitmap and with
	 * backwards-compatible single bool
	 */
	LWS_RXFLOW_REASON_USER_BOOL		= (1 << 0),
	LWS_RXFLOW_REASON_HTTP_RXBUFFER		= (1 << 6),
	LWS_RXFLOW_REASON_H2_PPS_PENDING	= (1 << 7),

	LWS_RXFLOW_REASON_APPLIES		= (1 << 14),
	LWS_RXFLOW_REASON_APPLIES_ENABLE_BIT	= (1 << 13),
	LWS_RXFLOW_REASON_APPLIES_ENABLE	= LWS_RXFLOW_REASON_APPLIES |
						  LWS_RXFLOW_REASON_APPLIES_ENABLE_BIT,
	LWS_RXFLOW_REASON_APPLIES_DISABLE	= LWS_RXFLOW_REASON_APPLIES,
	LWS_RXFLOW_REASON_FLAG_PROCESS_NOW	= (1 << 12),

};

/**
 * lws_rx_flow_control() - Enable and disable socket servicing for
 *				received packets.
 *
 * If the output side of a server process becomes choked, this allows flow
 * control for the input side.
 *
 * \param wsi:	Websocket connection instance to get callback for
 * \param enable:	0 = disable read servicing for this connection, 1 = enable
 *
 * If you need more than one additive reason for rxflow control, you can give
 * iLWS_RXFLOW_REASON_APPLIES_ENABLE or _DISABLE together with one or more of
 * b5..b0 set to idicate which bits to enable or disable.  If any bits are
 * enabled, rx on the connection is suppressed.
 *
 * LWS_RXFLOW_REASON_FLAG_PROCESS_NOW  flag may also be given to force any change
 * in rxflowbstatus to benapplied immediately, this should be used when you are
 * changing a wsi flow control state from outside a callback on that wsi.
 */
LWS_VISIBLE LWS_EXTERN int
lws_rx_flow_control(struct lws *wsi, int enable);

/**
 * lws_rx_flow_allow_all_protocol() - Allow all connections with this protocol to receive
 *
 * When the user server code realizes it can accept more input, it can
 * call this to have the RX flow restriction removed from all connections using
 * the given protocol.
 * \param context:	lws_context
 * \param protocol:	all connections using this protocol will be allowed to receive
 */
LWS_VISIBLE LWS_EXTERN void
lws_rx_flow_allow_all_protocol(const struct lws_context *context,
			       const struct lws_protocols *protocol);

/**
 * lws_remaining_packet_payload() - Bytes to come before "overall"
 *					      rx packet is complete
 * \param wsi:		Websocket instance (available from user callback)
 *
 *	This function is intended to be called from the callback if the
 *  user code is interested in "complete packets" from the client.
 *  libwebsockets just passes through payload as it comes and issues a buffer
 *  additionally when it hits a built-in limit.  The LWS_CALLBACK_RECEIVE
 *  callback handler can use this API to find out if the buffer it has just
 *  been given is the last piece of a "complete packet" from the client --
 *  when that is the case lws_remaining_packet_payload() will return
 *  0.
 *
 *  Many protocols won't care becuse their packets are always small.
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_remaining_packet_payload(struct lws *wsi);


/** \defgroup sock-adopt Socket adoption helpers
 * ##Socket adoption helpers
 *
 * When integrating with an external app with its own event loop, these can
 * be used to accept connections from someone else's listening socket.
 *
 * When using lws own event loop, these are not needed.
 */
///@{

/**
 * lws_adopt_socket() - adopt foreign socket as if listen socket accepted it
 * for the default vhost of context.
 *
 * \param context: lws context
 * \param accept_fd: fd of already-accepted socket to adopt
 *
 * Either returns new wsi bound to accept_fd, or closes accept_fd and
 * returns NULL, having cleaned up any new wsi pieces.
 *
 * LWS adopts the socket in http serving mode, it's ready to accept an upgrade
 * to ws or just serve http.
 */
LWS_VISIBLE LWS_EXTERN struct lws *
lws_adopt_socket(struct lws_context *context, lws_sockfd_type accept_fd);
/**
 * lws_adopt_socket_vhost() - adopt foreign socket as if listen socket accepted it
 * for vhost
 *
 * \param vh: lws vhost
 * \param accept_fd: fd of already-accepted socket to adopt
 *
 * Either returns new wsi bound to accept_fd, or closes accept_fd and
 * returns NULL, having cleaned up any new wsi pieces.
 *
 * LWS adopts the socket in http serving mode, it's ready to accept an upgrade
 * to ws or just serve http.
 */
LWS_VISIBLE LWS_EXTERN struct lws *
lws_adopt_socket_vhost(struct lws_vhost *vh, lws_sockfd_type accept_fd);

typedef enum {
	LWS_ADOPT_RAW_FILE_DESC = 0,	/* convenience constant */
	LWS_ADOPT_HTTP = 1,		/* flag: absent implies RAW */
	LWS_ADOPT_SOCKET = 2,		/* flag: absent implies file descr */
	LWS_ADOPT_ALLOW_SSL = 4,	/* flag: if set requires LWS_ADOPT_SOCKET */
	LWS_ADOPT_WS_PARENTIO = 8,	/* flag: ws mode parent handles IO
					 *   if given must be only flag
					 *   wsi put directly into ws mode
					 */
} lws_adoption_type;

typedef union {
	lws_sockfd_type sockfd;
	lws_filefd_type filefd;
} lws_sock_file_fd_type;

/*
* lws_adopt_descriptor_vhost() - adopt foreign socket or file descriptor
* if socket descriptor, should already have been accepted from listen socket
*
* \param vhost: lws vhost
* \param type: OR-ed combinations of lws_adoption_type flags
* \param fd: union with either .sockfd or .filefd set
* \param vh_prot_name: NULL or vh protocol name to bind raw connection to
* \param parent: NULL or struct lws to attach new_wsi to as a child
*
* Either returns new wsi bound to accept_fd, or closes accept_fd and
* returns NULL, having cleaned up any new wsi pieces.
*
* If LWS_ADOPT_SOCKET is set, LWS adopts the socket in http serving mode, it's
* ready to accept an upgrade to ws or just serve http.
*
* parent may be NULL, if given it should be an existing wsi that will become the
* parent of the new wsi created by this call.
*/
LWS_VISIBLE LWS_EXTERN struct lws *
lws_adopt_descriptor_vhost(struct lws_vhost *vh, lws_adoption_type type,
			   lws_sock_file_fd_type fd, const char *vh_prot_name,
			   struct lws *parent);

/**
 * lws_adopt_socket_readbuf() - adopt foreign socket and first rx as if listen socket accepted it
 * for the default vhost of context.
 * \param context:	lws context
 * \param accept_fd:	fd of already-accepted socket to adopt
 * \param readbuf:	NULL or pointer to data that must be drained before reading from
 *		accept_fd
 * \param len:	The length of the data held at \param readbuf
 *
 * Either returns new wsi bound to accept_fd, or closes accept_fd and
 * returns NULL, having cleaned up any new wsi pieces.
 *
 * LWS adopts the socket in http serving mode, it's ready to accept an upgrade
 * to ws or just serve http.
 *
 * If your external code did not already read from the socket, you can use
 * lws_adopt_socket() instead.
 *
 * This api is guaranteed to use the data at \param readbuf first, before reading from
 * the socket.
 *
 * readbuf is limited to the size of the ah rx buf, currently 2048 bytes.
 */
LWS_VISIBLE LWS_EXTERN struct lws *
lws_adopt_socket_readbuf(struct lws_context *context, lws_sockfd_type accept_fd,
                         const char *readbuf, size_t len);
/**
 * lws_adopt_socket_vhost_readbuf() - adopt foreign socket and first rx as if listen socket
 * accepted it for vhost.
 * \param vhost:	lws vhost
 * \param accept_fd:	fd of already-accepted socket to adopt
 * \param readbuf:	NULL or pointer to data that must be drained before reading from
 *			accept_fd
 * \param len:		The length of the data held at \param readbuf
 *
 * Either returns new wsi bound to accept_fd, or closes accept_fd and
 * returns NULL, having cleaned up any new wsi pieces.
 *
 * LWS adopts the socket in http serving mode, it's ready to accept an upgrade
 * to ws or just serve http.
 *
 * If your external code did not already read from the socket, you can use
 * lws_adopt_socket() instead.
 *
 * This api is guaranteed to use the data at \param readbuf first, before reading from
 * the socket.
 *
 * readbuf is limited to the size of the ah rx buf, currently 2048 bytes.
 */
LWS_VISIBLE LWS_EXTERN struct lws *
lws_adopt_socket_vhost_readbuf(struct lws_vhost *vhost, lws_sockfd_type accept_fd,
                               const char *readbuf, size_t len);
///@}

/** \defgroup net Network related helper APIs
 * ##Network related helper APIs
 *
 * These wrap miscellaneous useful network-related functions
 */
///@{

/**
 * lws_canonical_hostname() - returns this host's hostname
 *
 * This is typically used by client code to fill in the host parameter
 * when making a client connection.  You can only call it after the context
 * has been created.
 *
 * \param context:	Websocket context
 */
LWS_VISIBLE LWS_EXTERN const char * LWS_WARN_UNUSED_RESULT
lws_canonical_hostname(struct lws_context *context);

/**
 * lws_get_peer_addresses() - Get client address information
 * \param wsi:	Local struct lws associated with
 * \param fd:		Connection socket descriptor
 * \param name:	Buffer to take client address name
 * \param name_len:	Length of client address name buffer
 * \param rip:	Buffer to take client address IP dotted quad
 * \param rip_len:	Length of client address IP buffer
 *
 *	This function fills in name and rip with the name and IP of
 *	the client connected with socket descriptor fd.  Names may be
 *	truncated if there is not enough room.  If either cannot be
 *	determined, they will be returned as valid zero-length strings.
 */
LWS_VISIBLE LWS_EXTERN void
lws_get_peer_addresses(struct lws *wsi, lws_sockfd_type fd, char *name,
		       int name_len, char *rip, int rip_len);

/**
 * lws_get_peer_simple() - Get client address information without RDNS
 *
 * \param wsi:	Local struct lws associated with
 * \param name:	Buffer to take client address name
 * \param namelen:	Length of client address name buffer
 *
 * This provides a 123.123.123.123 type IP address in name from the
 * peer that has connected to wsi
 */
LWS_VISIBLE LWS_EXTERN const char *
lws_get_peer_simple(struct lws *wsi, char *name, int namelen);
#if !defined(LWS_WITH_ESP8266) && !defined(LWS_WITH_ESP32)
/**
 * lws_interface_to_sa() - Convert interface name or IP to sockaddr struct
 *
 * \param ipv6:	Allow IPV6 addresses
 * \param ifname:	Interface name or IP
 * \param addr:	struct sockaddr_in * to be written
 * \param addrlen:	Length of addr
 *
 * This converts a textual network interface name to a sockaddr usable by
 * other network functions
 */
LWS_VISIBLE LWS_EXTERN int
lws_interface_to_sa(int ipv6, const char *ifname, struct sockaddr_in *addr,
		    size_t addrlen);
///@}
#endif

/** \defgroup misc Miscellaneous APIs
* ##Miscellaneous APIs
*
* Various APIs outside of other categories
*/
///@{

/**
 * lws_start_foreach_ll(): linkedlist iterator helper start
 *
 * \param type: type of iteration, eg, struct xyz *
 * \param it: iterator var name to create
 * \param start: start of list
 *
 * This helper creates an iterator and starts a while (it) {
 * loop.  The iterator runs through the linked list starting at start and
 * ends when it gets a NULL.
 * The while loop should be terminated using lws_start_foreach_ll().
 */
#define lws_start_foreach_ll(type, it, start)\
{ \
	type it = start; \
	while (it) {

/**
 * lws_end_foreach_ll(): linkedlist iterator helper end
 *
 * \param it: same iterator var name given when starting
 * \param nxt: member name in the iterator pointing to next list element
 *
 * This helper is the partner for lws_start_foreach_ll() that ends the
 * while loop.
 */

#define lws_end_foreach_ll(it, nxt) \
		it = it->nxt; \
	} \
}

/**
 * lws_start_foreach_llp(): linkedlist pointer iterator helper start
 *
 * \param type: type of iteration, eg, struct xyz **
 * \param it: iterator var name to create
 * \param start: start of list
 *
 * This helper creates an iterator and starts a while (it) {
 * loop.  The iterator runs through the linked list starting at the
 * address of start and ends when it gets a NULL.
 * The while loop should be terminated using lws_start_foreach_llp().
 *
 * This helper variant iterates using a pointer to the previous linked-list
 * element.  That allows you to easily delete list members by rewriting the
 * previous pointer to the element's next pointer.
 */
#define lws_start_foreach_llp(type, it, start)\
{ \
	type it = &(start); \
	while (*(it)) {

/**
 * lws_end_foreach_llp(): linkedlist pointer iterator helper end
 *
 * \param it: same iterator var name given when starting
 * \param nxt: member name in the iterator pointing to next list element
 *
 * This helper is the partner for lws_start_foreach_llp() that ends the
 * while loop.
 */

#define lws_end_foreach_llp(it, nxt) \
		it = &(*(it))->nxt; \
	} \
}

/**
 * lws_snprintf(): snprintf that truncates the returned length too
 *
 * \param str: destination buffer
 * \param size: bytes left in destination buffer
 * \param format: format string
 * \param ...: args for format
 *
 * This lets you correctly truncate buffers by concatenating lengths, if you
 * reach the limit the reported length doesn't exceed the limit.
 */
LWS_VISIBLE LWS_EXTERN int
lws_snprintf(char *str, size_t size, const char *format, ...) LWS_FORMAT(3);

/**
 * lws_get_random(): fill a buffer with platform random data
 *
 * \param context: the lws context
 * \param buf: buffer to fill
 * \param len: how much to fill
 *
 * This is intended to be called from the LWS_CALLBACK_RECEIVE callback if
 * it's interested to see if the frame it's dealing with was sent in binary
 * mode.
 */
LWS_VISIBLE LWS_EXTERN int
lws_get_random(struct lws_context *context, void *buf, int len);
/**
 * lws_daemonize(): make current process run in the background
 *
 * \param _lock_path: the filepath to write the lock file
 *
 * Spawn lws as a background process, taking care of various things
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_daemonize(const char *_lock_path);
/**
 * lws_get_library_version(): return string describing the version of lws
 *
 * On unix, also includes the git describe
 */
LWS_VISIBLE LWS_EXTERN const char * LWS_WARN_UNUSED_RESULT
lws_get_library_version(void);

/**
 * lws_wsi_user() - get the user data associated with the connection
 * \param wsi: lws connection
 *
 * Not normally needed since it's passed into the callback
 */
LWS_VISIBLE LWS_EXTERN void *
lws_wsi_user(struct lws *wsi);

/**
 * lws_wsi_set_user() - set the user data associated with the client connection
 * \param wsi: lws connection
 * \param user: user data
 *
 * By default lws allocates this and it's not legal to externally set it
 * yourself.  However client connections may have it set externally when the
 * connection is created... if so, this api can be used to modify it at
 * runtime additionally.
 */
LWS_VISIBLE LWS_EXTERN void
lws_set_wsi_user(struct lws *wsi, void *user);

/**
 * lws_parse_uri:	cut up prot:/ads:port/path into pieces
 *			Notice it does so by dropping '\0' into input string
 *			and the leading / on the path is consequently lost
 *
 * \param p:			incoming uri string.. will get written to
 * \param prot:		result pointer for protocol part (https://)
 * \param ads:		result pointer for address part
 * \param port:		result pointer for port part
 * \param path:		result pointer for path part
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_parse_uri(char *p, const char **prot, const char **ads, int *port,
	      const char **path);

/**
 * lws_now_secs(): return seconds since 1970-1-1
 */
LWS_VISIBLE LWS_EXTERN unsigned long
lws_now_secs(void);

/**
 * lws_get_context - Allow getting lws_context from a Websocket connection
 * instance
 *
 * With this function, users can access context in the callback function.
 * Otherwise users may have to declare context as a global variable.
 *
 * \param wsi:	Websocket connection instance
 */
LWS_VISIBLE LWS_EXTERN struct lws_context * LWS_WARN_UNUSED_RESULT
lws_get_context(const struct lws *wsi);

/**
 * lws_get_count_threads(): how many service threads the context uses
 *
 * \param context: the lws context
 *
 * By default this is always 1, if you asked for more than lws can handle it
 * will clip the number of threads.  So you can use this to find out how many
 * threads are actually in use.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_get_count_threads(struct lws_context *context);

/**
 * lws_get_parent() - get parent wsi or NULL
 * \param wsi: lws connection
 *
 * Specialized wsi like cgi stdin/out/err are associated to a parent wsi,
 * this allows you to get their parent.
 */
LWS_VISIBLE LWS_EXTERN struct lws * LWS_WARN_UNUSED_RESULT
lws_get_parent(const struct lws *wsi);

/**
 * lws_get_child() - get child wsi or NULL
 * \param wsi: lws connection
 *
 * Allows you to find a related wsi from the parent wsi.
 */
LWS_VISIBLE LWS_EXTERN struct lws * LWS_WARN_UNUSED_RESULT
lws_get_child(const struct lws *wsi);

/**
 * lws_parent_carries_io() - mark wsi as needing to send messages via parent
 *
 * \param wsi: child lws connection
 */

LWS_VISIBLE LWS_EXTERN void
lws_set_parent_carries_io(struct lws *wsi);

LWS_VISIBLE LWS_EXTERN void *
lws_get_opaque_parent_data(const struct lws *wsi);

LWS_VISIBLE LWS_EXTERN void
lws_set_opaque_parent_data(struct lws *wsi, void *data);

LWS_VISIBLE LWS_EXTERN int
lws_get_child_pending_on_writable(const struct lws *wsi);

LWS_VISIBLE LWS_EXTERN void
lws_clear_child_pending_on_writable(struct lws *wsi);

LWS_VISIBLE LWS_EXTERN int
lws_get_close_length(struct lws *wsi);

LWS_VISIBLE LWS_EXTERN unsigned char *
lws_get_close_payload(struct lws *wsi);

/**
 * lws_get_network_wsi() - Returns wsi that has the tcp connection for this wsi
 *
 * \param wsi: wsi you have
 *
 * Returns wsi that has the tcp connection (which may be the incoming wsi)
 *
 * HTTP/1 connections will always return the incoming wsi
 * HTTP/2 connections may return a different wsi that has the tcp connection
 */
LWS_VISIBLE LWS_EXTERN
struct lws *lws_get_network_wsi(struct lws *wsi);

/*
 * \deprecated DEPRECATED Note: this is not normally needed as a user api.
 * It's provided in case it is
 * useful when integrating with other app poll loop service code.
 */
LWS_VISIBLE LWS_EXTERN int
lws_read(struct lws *wsi, unsigned char *buf, lws_filepos_t len);

/**
 * lws_set_allocator() - custom allocator support
 *
 * \param realloc
 *
 * Allows you to replace the allocator (and deallocator) used by lws
 */
LWS_VISIBLE LWS_EXTERN void
lws_set_allocator(void *(*realloc)(void *ptr, size_t size, const char *reason));
///@}

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

/**
 * lws_is_ssl() - Find out if connection is using SSL
 * \param wsi:	websocket connection to check
 *
 *	Returns 0 if the connection is not using SSL, 1 if using SSL and
 *	using verified cert, and 2 if using SSL but the cert was not
 *	checked (appears for client wsi told to skip check on connection)
 */
LWS_VISIBLE LWS_EXTERN int
lws_is_ssl(struct lws *wsi);
/**
 * lws_is_cgi() - find out if this wsi is running a cgi process
 * \param wsi: lws connection
 */
LWS_VISIBLE LWS_EXTERN int
lws_is_cgi(struct lws *wsi);

#ifdef LWS_OPENSSL_SUPPORT
/**
 * lws_get_ssl() - Return wsi's SSL context structure
 * \param wsi:	websocket connection
 *
 * Returns pointer to the SSL library's context structure
 */
LWS_VISIBLE LWS_EXTERN SSL*
lws_get_ssl(struct lws *wsi);
#endif
///@}

/** \defgroup lws_ring LWS Ringbuffer APIs
 * ##lws_ring: generic ringbuffer struct
 *
 * Provides an abstract ringbuffer api supporting one head and one or an
 * unlimited number of tails.
 *
 * All of the members are opaque and manipulated by lws_ring_...() apis.
 *
 * The lws_ring and its buffer is allocated at runtime on the heap, using
 *
 *  - lws_ring_create()
 *  - lws_ring_destroy()
 *
 * It may contain any type, the size of the "element" stored in the ring
 * buffer and the number of elements is given at creation time.
 *
 * When you create the ringbuffer, you can optionally provide an element
 * destroy callback that frees any allocations inside the element.  This is then
 * automatically called for elements with no tail behind them, ie, elements
 * which don't have any pending consumer are auto-freed.
 *
 * Whole elements may be inserted into the ringbuffer and removed from it, using
 *
 *  - lws_ring_insert()
 *  - lws_ring_consume()
 *
 * You can find out how many whole elements are free or waiting using
 *
 *  - lws_ring_get_count_free_elements()
 *  - lws_ring_get_count_waiting_elements()
 *
 * In addition there are special purpose optional byte-centric apis
 *
 *  - lws_ring_next_linear_insert_range()
 *  - lws_ring_bump_head()
 *
 *  which let you, eg, read() directly into the ringbuffer without needing
 *  an intermediate bounce buffer.
 *
 *  The accessors understand that the ring wraps, and optimizes insertion and
 *  consumption into one or two memcpy()s depending on if the head or tail
 *  wraps.
 *
 *  lws_ring only supports a single head, but optionally multiple tails with
 *  an API to inform it when the "oldest" tail has moved on.  You can give
 *  NULL where-ever an api asks for a tail pointer, and it will use an internal
 *  single tail pointer for convenience.
 *
 *  The "oldest tail", which is the only tail if you give it NULL instead of
 *  some other tail, is used to track which elements in the ringbuffer are
 *  still unread by anyone.
 *
 *   - lws_ring_update_oldest_tail()
 */
///@{
struct lws_ring;

/**
 * lws_ring_create(): create a new ringbuffer
 *
 * \param element_len: the size in bytes of one element in the ringbuffer
 * \param count: the number of elements the ringbuffer can contain
 * \param destroy_element: NULL, or callback to be called for each element
 *			   that is removed from the ringbuffer due to the
 *			   oldest tail moving beyond it
 *
 * Creates the ringbuffer and allocates the storage.  Returns the new
 * lws_ring *, or NULL if the allocation failed.
 *
 * If non-NULL, destroy_element will get called back for every element that is
 * retired from the ringbuffer after the oldest tail has gone past it, and for
 * any element still left in the ringbuffer when it is destroyed.  It replaces
 * all other element destruction code in your user code.
 */
LWS_VISIBLE LWS_EXTERN struct lws_ring *
lws_ring_create(size_t element_len, size_t count,
		void (*destroy_element)(void *element));

/**
 * lws_ring_destroy():  destroy a previously created ringbuffer
 *
 * \param ring: the struct lws_ring to destroy
 *
 * Destroys the ringbuffer allocation and the struct lws_ring itself.
 */
LWS_VISIBLE LWS_EXTERN void
lws_ring_destroy(struct lws_ring *ring);

/**
 * lws_ring_get_count_free_elements():  return how many elements can fit
 *				      in the free space
 *
 * \param ring: the struct lws_ring to report on
 *
 * Returns how much room is left in the ringbuffer for whole element insertion.
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_ring_get_count_free_elements(struct lws_ring *ring);

/**
 * lws_ring_get_count_waiting_elements():  return how many elements can be consumed
 *
 * \param ring: the struct lws_ring to report on
 * \param tail: a pointer to the tail struct to use, or NULL for single tail
 *
 * Returns how many elements are waiting to be consumed from the perspective
 * of the tail pointer given.
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_ring_get_count_waiting_elements(struct lws_ring *ring, uint32_t *tail);

/**
 * lws_ring_insert():  attempt to insert up to max_count elements from src
 *
 * \param ring: the struct lws_ring to report on
 * \param src: the array of elements to be inserted
 * \param max_count: the number of available elements at src
 *
 * Attempts to insert as many of the elements at src as possible, up to the
 * maximum max_count.  Returns the number of elements actually inserted.
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_ring_insert(struct lws_ring *ring, const void *src, size_t max_count);

/**
 * lws_ring_consume():  attempt to copy out and remove up to max_count elements
 *		        to src
 *
 * \param ring: the struct lws_ring to report on
 * \param tail: a pointer to the tail struct to use, or NULL for single tail
 * \param dest: the array of elements to be inserted. or NULL for no copy
 * \param max_count: the number of available elements at src
 *
 * Attempts to copy out as many waiting elements as possible into dest, from
 * the perspective of the given tail, up to max_count.  If dest is NULL, the
 * copying out is not done but the elements are logically consumed as usual.
 * NULL dest is useful in combination with lws_ring_get_element(), where you
 * can use the element direct from the ringbuffer and then call this with NULL
 * dest to logically consume it.
 *
 * Increments the tail position according to how many elements could be
 * consumed.
 *
 * Returns the number of elements consumed.
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_ring_consume(struct lws_ring *ring, uint32_t *tail, void *dest,
		 size_t max_count);

/**
 * lws_ring_get_element():  get a pointer to the next waiting element for tail
 *
 * \param ring: the struct lws_ring to report on
 * \param tail: a pointer to the tail struct to use, or NULL for single tail
 *
 * Points to the next element that tail would consume, directly in the
 * ringbuffer.  This lets you write() or otherwise use the element without
 * having to copy it out somewhere first.
 *
 * After calling this, you must call lws_ring_consume(ring, &tail, NULL, 1)
 * which will logically consume the element you used up and increment your
 * tail (tail may also be NULL there if you use a single tail).
 *
 * Returns NULL if no waiting element, or a const void * pointing to it.
 */
LWS_VISIBLE LWS_EXTERN const void *
lws_ring_get_element(struct lws_ring *ring, uint32_t *tail);

/**
 * lws_ring_update_oldest_tail():  free up elements older than tail for reuse
 *
 * \param ring: the struct lws_ring to report on
 * \param tail: a pointer to the tail struct to use, or NULL for single tail
 *
 * If you are using multiple tails, you must use this API to inform the
 * lws_ring when none of the tails still need elements in the fifo any more,
 * by updating it when the "oldest" tail has moved on.
 */
LWS_VISIBLE LWS_EXTERN void
lws_ring_update_oldest_tail(struct lws_ring *ring, uint32_t tail);

/**
 * lws_ring_get_oldest_tail():  get current oldest available data index
 *
 * \param ring: the struct lws_ring to report on
 *
 * If you are initializing a new ringbuffer consumer, you can set its tail to
 * this to start it from the oldest ringbuffer entry still available.
 */
LWS_VISIBLE LWS_EXTERN uint32_t
lws_ring_get_oldest_tail(struct lws_ring *ring);

/**
 * lws_ring_next_linear_insert_range():  used to write directly into the ring
 *
 * \param ring: the struct lws_ring to report on
 * \param start: pointer to a void * set to the start of the next ringbuffer area
 * \param bytes: pointer to a size_t set to the max length you may use from *start
 *
 * This provides a low-level, bytewise access directly into the ringbuffer
 * allowing direct insertion of data without having to use a bounce buffer.
 *
 * The api reports the position and length of the next linear range that can
 * be written in the ringbuffer, ie, up to the point it would wrap, and sets
 * *start and *bytes accordingly.  You can then, eg, directly read() into
 * *start for up to *bytes, and use lws_ring_bump_head() to update the lws_ring
 * with what you have done.
 *
 * Returns nonzero if no insertion is currently possible.
 */
LWS_VISIBLE LWS_EXTERN int
lws_ring_next_linear_insert_range(struct lws_ring *ring, void **start,
				  size_t *bytes);

/**
 * lws_ring_bump_head():  used to write directly into the ring
 *
 * \param ring: the struct lws_ring to operate on
 * \param bytes: the number of bytes you inserted at the current head
 */
LWS_VISIBLE LWS_EXTERN void
lws_ring_bump_head(struct lws_ring *ring, size_t bytes);
///@}

/** \defgroup sha SHA and B64 helpers
 * ##SHA and B64 helpers
 *
 * These provide SHA-1 and B64 helper apis
 */
///@{
#ifdef LWS_SHA1_USE_OPENSSL_NAME
#define lws_SHA1 SHA1
#else
/**
 * lws_SHA1(): make a SHA-1 digest of a buffer
 *
 * \param d: incoming buffer
 * \param n: length of incoming buffer
 * \param md: buffer for message digest (must be >= 20 bytes)
 *
 * Reduces any size buffer into a 20-byte SHA-1 hash.
 */
LWS_VISIBLE LWS_EXTERN unsigned char *
lws_SHA1(const unsigned char *d, size_t n, unsigned char *md);
#endif
/**
 * lws_b64_encode_string(): encode a string into base 64
 *
 * \param in: incoming buffer
 * \param in_len: length of incoming buffer
 * \param out: result buffer
 * \param out_size: length of result buffer
 *
 * Encodes a string using b64
 */
LWS_VISIBLE LWS_EXTERN int
lws_b64_encode_string(const char *in, int in_len, char *out, int out_size);
/**
 * lws_b64_decode_string(): decode a string from base 64
 *
 * \param in: incoming buffer
 * \param out: result buffer
 * \param out_size: length of result buffer
 *
 * Decodes a string using b64
 */
LWS_VISIBLE LWS_EXTERN int
lws_b64_decode_string(const char *in, char *out, int out_size);
///@}


/*! \defgroup cgi cgi handling
 *
 * ##CGI handling
 *
 * These functions allow low-level control over stdin/out/err of the cgi.
 *
 * However for most cases, binding the cgi to http in and out, the default
 * lws implementation already does the right thing.
 */

enum lws_enum_stdinouterr {
	LWS_STDIN = 0,
	LWS_STDOUT = 1,
	LWS_STDERR = 2,
};

enum lws_cgi_hdr_state {
	LCHS_HEADER,
	LCHS_CR1,
	LCHS_LF1,
	LCHS_CR2,
	LCHS_LF2,
	LHCS_RESPONSE,
	LHCS_DUMP_HEADERS,
	LHCS_PAYLOAD,
	LCHS_SINGLE_0A,
};

struct lws_cgi_args {
	struct lws **stdwsi; /**< get fd with lws_get_socket_fd() */
	enum lws_enum_stdinouterr ch; /**< channel index */
	unsigned char *data; /**< for messages with payload */
	enum lws_cgi_hdr_state hdr_state; /**< track where we are in cgi headers */
	int len; /**< length */
};

#ifdef LWS_WITH_CGI
/**
 * lws_cgi: spawn network-connected cgi process
 *
 * \param wsi: connection to own the process
 * \param exec_array: array of "exec-name" "arg1" ... "argn" NULL
 * \param script_uri_path_len: how many chars on the left of the uri are the
 *        path to the cgi, or -1 to spawn without URL-related env vars
 * \param timeout_secs: seconds script should be allowed to run
 * \param mp_cgienv: pvo list with per-vhost cgi options to put in env
 */
LWS_VISIBLE LWS_EXTERN int
lws_cgi(struct lws *wsi, const char * const *exec_array,
	int script_uri_path_len, int timeout_secs,
	const struct lws_protocol_vhost_options *mp_cgienv);

/**
 * lws_cgi_write_split_stdout_headers: write cgi output accounting for header part
 *
 * \param wsi: connection to own the process
 */
LWS_VISIBLE LWS_EXTERN int
lws_cgi_write_split_stdout_headers(struct lws *wsi);

/**
 * lws_cgi_kill: terminate cgi process associated with wsi
 *
 * \param wsi: connection to own the process
 */
LWS_VISIBLE LWS_EXTERN int
lws_cgi_kill(struct lws *wsi);

/**
 * lws_cgi_get_stdwsi: get wsi for stdin, stdout, or stderr
 *
 * \param wsi: parent wsi that has cgi
 * \param ch: which of LWS_STDIN, LWS_STDOUT or LWS_STDERR
 */
LWS_VISIBLE LWS_EXTERN struct lws *
lws_cgi_get_stdwsi(struct lws *wsi, enum lws_enum_stdinouterr ch);

#endif
///@}


/*! \defgroup fops file operation wrapping
 *
 * ##File operation wrapping
 *
 * Use these helper functions if you want to access a file from the perspective
 * of a specific wsi, which is usually the case.  If you just want contextless
 * file access, use the fops callbacks directly with NULL wsi instead of these
 * helpers.
 *
 * If so, then it calls the platform handler or user overrides where present
 * (as defined in info->fops)
 *
 * The advantage from all this is user code can be portable for file operations
 * without having to deal with differences between platforms.
 */
//@{

/** struct lws_plat_file_ops - Platform-specific file operations
 *
 * These provide platform-agnostic ways to deal with filesystem access in the
 * library and in the user code.
 */

#if defined(LWS_WITH_ESP32)
/* sdk preprocessor defs? compiler issue? gets confused with member names */
#define LWS_FOP_OPEN		_open
#define LWS_FOP_CLOSE		_close
#define LWS_FOP_SEEK_CUR	_seek_cur
#define LWS_FOP_READ		_read
#define LWS_FOP_WRITE		_write
#else
#define LWS_FOP_OPEN		open
#define LWS_FOP_CLOSE		close
#define LWS_FOP_SEEK_CUR	seek_cur
#define LWS_FOP_READ		read
#define LWS_FOP_WRITE		write
#endif

#define LWS_FOP_FLAGS_MASK		   ((1 << 23) - 1)
#define LWS_FOP_FLAG_COMPR_ACCEPTABLE_GZIP (1 << 24)
#define LWS_FOP_FLAG_COMPR_IS_GZIP	   (1 << 25)
#define LWS_FOP_FLAG_MOD_TIME_VALID	   (1 << 26)
#define LWS_FOP_FLAG_VIRTUAL		   (1 << 27)

struct lws_plat_file_ops;

struct lws_fop_fd {
	lws_filefd_type			fd;
	/**< real file descriptor related to the file... */
	const struct lws_plat_file_ops	*fops;
	/**< fops that apply to this fop_fd */
	void				*filesystem_priv;
	/**< ignored by lws; owned by the fops handlers */
	lws_filepos_t			pos;
	/**< generic "position in file" */
	lws_filepos_t			len;
	/**< generic "length of file" */
	lws_fop_flags_t			flags;
	/**< copy of the returned flags */
	uint32_t			mod_time;
	/**< optional "modification time of file", only valid if .open()
	 * set the LWS_FOP_FLAG_MOD_TIME_VALID flag */
};
typedef struct lws_fop_fd *lws_fop_fd_t;

struct lws_fops_index {
	const char *sig;	/* NULL or vfs signature, eg, ".zip/" */
	uint8_t len;		/* length of above string */
};

struct lws_plat_file_ops {
	lws_fop_fd_t (*LWS_FOP_OPEN)(const struct lws_plat_file_ops *fops,
				     const char *filename, const char *vpath,
				     lws_fop_flags_t *flags);
	/**< Open file (always binary access if plat supports it)
	 * vpath may be NULL, or if the fops understands it, the point at which
	 * the filename's virtual part starts.
	 * *flags & LWS_FOP_FLAGS_MASK should be set to O_RDONLY or O_RDWR.
	 * If the file may be gzip-compressed,
	 * LWS_FOP_FLAG_COMPR_ACCEPTABLE_GZIP is set.  If it actually is
	 * gzip-compressed, then the open handler should OR
	 * LWS_FOP_FLAG_COMPR_IS_GZIP on to *flags before returning.
	 */
	int (*LWS_FOP_CLOSE)(lws_fop_fd_t *fop_fd);
	/**< close file AND set the pointer to NULL */
	lws_fileofs_t (*LWS_FOP_SEEK_CUR)(lws_fop_fd_t fop_fd,
					  lws_fileofs_t offset_from_cur_pos);
	/**< seek from current position */
	int (*LWS_FOP_READ)(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
			    uint8_t *buf, lws_filepos_t len);
	/**< Read from file, on exit *amount is set to amount actually read */
	int (*LWS_FOP_WRITE)(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
			     uint8_t *buf, lws_filepos_t len);
	/**< Write to file, on exit *amount is set to amount actually written */

	struct lws_fops_index fi[3];
	/**< vfs path signatures implying use of this fops */

	const struct lws_plat_file_ops *next;
	/**< NULL or next fops in list */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
};

/**
 * lws_get_fops() - get current file ops
 *
 * \param context: context
 */
LWS_VISIBLE LWS_EXTERN struct lws_plat_file_ops * LWS_WARN_UNUSED_RESULT
lws_get_fops(struct lws_context *context);
LWS_VISIBLE LWS_EXTERN void
lws_set_fops(struct lws_context *context, const struct lws_plat_file_ops *fops);
/**
 * lws_vfs_tell() - get current file position
 *
 * \param fop_fd: fop_fd we are asking about
 */
LWS_VISIBLE LWS_EXTERN lws_filepos_t LWS_WARN_UNUSED_RESULT
lws_vfs_tell(lws_fop_fd_t fop_fd);
/**
 * lws_vfs_get_length() - get current file total length in bytes
 *
 * \param fop_fd: fop_fd we are asking about
 */
LWS_VISIBLE LWS_EXTERN lws_filepos_t LWS_WARN_UNUSED_RESULT
lws_vfs_get_length(lws_fop_fd_t fop_fd);
/**
 * lws_vfs_get_mod_time() - get time file last modified
 *
 * \param fop_fd: fop_fd we are asking about
 */
LWS_VISIBLE LWS_EXTERN uint32_t LWS_WARN_UNUSED_RESULT
lws_vfs_get_mod_time(lws_fop_fd_t fop_fd);
/**
 * lws_vfs_file_seek_set() - seek relative to start of file
 *
 * \param fop_fd: fop_fd we are seeking in
 * \param offset: offset from start of file
 */
LWS_VISIBLE LWS_EXTERN lws_fileofs_t
lws_vfs_file_seek_set(lws_fop_fd_t fop_fd, lws_fileofs_t offset);
/**
 * lws_vfs_file_seek_end() - seek relative to end of file
 *
 * \param fop_fd: fop_fd we are seeking in
 * \param offset: offset from start of file
 */
LWS_VISIBLE LWS_EXTERN lws_fileofs_t
lws_vfs_file_seek_end(lws_fop_fd_t fop_fd, lws_fileofs_t offset);

extern struct lws_plat_file_ops fops_zip;

/**
 * lws_plat_file_open() - open vfs filepath
 *
 * \param fops: file ops struct that applies to this descriptor
 * \param vfs_path: filename to open
 * \param flags: pointer to open flags
 *
 * The vfs_path is scanned for known fops signatures, and the open directed
 * to any matching fops open.
 *
 * User code should use this api to perform vfs opens.
 *
 * returns semi-opaque handle
 */
LWS_VISIBLE LWS_EXTERN lws_fop_fd_t LWS_WARN_UNUSED_RESULT
lws_vfs_file_open(const struct lws_plat_file_ops *fops, const char *vfs_path,
		  lws_fop_flags_t *flags);

/**
 * lws_plat_file_close() - close file
 *
 * \param fop_fd: file handle to close
 */
static LWS_INLINE int
lws_vfs_file_close(lws_fop_fd_t *fop_fd)
{
	return (*fop_fd)->fops->LWS_FOP_CLOSE(fop_fd);
}

/**
 * lws_plat_file_seek_cur() - close file
 *
 *
 * \param fop_fd: file handle
 * \param offset: position to seek to
 */
static LWS_INLINE lws_fileofs_t
lws_vfs_file_seek_cur(lws_fop_fd_t fop_fd, lws_fileofs_t offset)
{
	return fop_fd->fops->LWS_FOP_SEEK_CUR(fop_fd, offset);
}
/**
 * lws_plat_file_read() - read from file
 *
 * \param fop_fd: file handle
 * \param amount: how much to read (rewritten by call)
 * \param buf: buffer to write to
 * \param len: max length
 */
static LWS_INLINE int LWS_WARN_UNUSED_RESULT
lws_vfs_file_read(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		   uint8_t *buf, lws_filepos_t len)
{
	return fop_fd->fops->LWS_FOP_READ(fop_fd, amount, buf, len);
}
/**
 * lws_plat_file_write() - write from file
 *
 * \param fop_fd: file handle
 * \param amount: how much to write (rewritten by call)
 * \param buf: buffer to read from
 * \param len: max length
 */
static LWS_INLINE int LWS_WARN_UNUSED_RESULT
lws_vfs_file_write(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		    uint8_t *buf, lws_filepos_t len)
{
	return fop_fd->fops->LWS_FOP_WRITE(fop_fd, amount, buf, len);
}

/* these are the platform file operations implementations... they can
 * be called directly and used in fops arrays
 */

LWS_VISIBLE LWS_EXTERN lws_fop_fd_t
_lws_plat_file_open(const struct lws_plat_file_ops *fops, const char *filename,
		    const char *vpath, lws_fop_flags_t *flags);
LWS_VISIBLE LWS_EXTERN int
_lws_plat_file_close(lws_fop_fd_t *fop_fd);
LWS_VISIBLE LWS_EXTERN lws_fileofs_t
_lws_plat_file_seek_cur(lws_fop_fd_t fop_fd, lws_fileofs_t offset);
LWS_VISIBLE LWS_EXTERN int
_lws_plat_file_read(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		    uint8_t *buf, lws_filepos_t len);
LWS_VISIBLE LWS_EXTERN int
_lws_plat_file_write(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		     uint8_t *buf, lws_filepos_t len);

LWS_VISIBLE LWS_EXTERN int
lws_alloc_vfs_file(struct lws_context *context, const char *filename,
		   uint8_t **buf, lws_filepos_t *amount);
//@}

/** \defgroup smtp SMTP related functions
 * ##SMTP related functions
 * \ingroup lwsapi
 *
 * These apis let you communicate with a local SMTP server to send email from
 * lws.  It handles all the SMTP sequencing and protocol actions.
 *
 * Your system should have postfix, sendmail or another MTA listening on port
 * 25 and able to send email using the "mail" commandline app.  Usually distro
 * MTAs are configured for this by default.
 *
 * It runs via its own libuv events if initialized (which requires giving it
 * a libuv loop to attach to).
 *
 * It operates using three callbacks, on_next() queries if there is a new email
 * to send, on_get_body() asks for the body of the email, and on_sent() is
 * called after the email is successfully sent.
 *
 * To use it
 *
 *  - create an lws_email struct
 *
 *  - initialize data, loop, the email_* strings, max_content_size and
 *    the callbacks
 *
 *  - call lws_email_init()
 *
 *  When you have at least one email to send, call lws_email_check() to
 *  schedule starting to send it.
 */
//@{
#ifdef LWS_WITH_SMTP

/** enum lwsgs_smtp_states - where we are in SMTP protocol sequence */
enum lwsgs_smtp_states {
	LGSSMTP_IDLE, /**< awaiting new email */
	LGSSMTP_CONNECTING, /**< opening tcp connection to MTA */
	LGSSMTP_CONNECTED, /**< tcp connection to MTA is connected */
	LGSSMTP_SENT_HELO, /**< sent the HELO */
	LGSSMTP_SENT_FROM, /**< sent FROM */
	LGSSMTP_SENT_TO, /**< sent TO */
	LGSSMTP_SENT_DATA, /**< sent DATA request */
	LGSSMTP_SENT_BODY, /**< sent the email body */
	LGSSMTP_SENT_QUIT, /**< sent the session quit */
};

/** struct lws_email - abstract context for performing SMTP operations */
struct lws_email {
	void *data;
	/**< opaque pointer set by user code and available to the callbacks */
	uv_loop_t *loop;
	/**< the libuv loop we will work on */

	char email_smtp_ip[32]; /**< Fill before init, eg, "127.0.0.1" */
	char email_helo[32];	/**< Fill before init, eg, "myserver.com" */
	char email_from[100];	/**< Fill before init or on_next */
	char email_to[100];	/**< Fill before init or on_next */

	unsigned int max_content_size;
	/**< largest possible email body size */

	/* Fill all the callbacks before init */

	int (*on_next)(struct lws_email *email);
	/**< (Fill in before calling lws_email_init)
	 * called when idle, 0 = another email to send, nonzero is idle.
	 * If you return 0, all of the email_* char arrays must be set
	 * to something useful. */
	int (*on_sent)(struct lws_email *email);
	/**< (Fill in before calling lws_email_init)
	 * called when transfer of the email to the SMTP server was
	 * successful, your callback would remove the current email
	 * from its queue */
	int (*on_get_body)(struct lws_email *email, char *buf, int len);
	/**< (Fill in before calling lws_email_init)
	 * called when the body part of the queued email is about to be
	 * sent to the SMTP server. */


	/* private things */
	uv_timer_t timeout_email; /**< private */
	enum lwsgs_smtp_states estate; /**< private */
	uv_connect_t email_connect_req; /**< private */
	uv_tcp_t email_client; /**< private */
	time_t email_connect_started; /**< private */
	char email_buf[256]; /**< private */
	char *content; /**< private */
};

/**
 * lws_email_init() - Initialize a struct lws_email
 *
 * \param email: struct lws_email to init
 * \param loop: libuv loop to use
 * \param max_content: max email content size
 *
 * Prepares a struct lws_email for use ending SMTP
 */
LWS_VISIBLE LWS_EXTERN int
lws_email_init(struct lws_email *email, uv_loop_t *loop, int max_content);

/**
 * lws_email_check() - Request check for new email
 *
 * \param email: struct lws_email context to check
 *
 * Schedules a check for new emails in 1s... call this when you have queued an
 * email for send.
 */
LWS_VISIBLE LWS_EXTERN void
lws_email_check(struct lws_email *email);
/**
 * lws_email_destroy() - stop using the struct lws_email
 *
 * \param email: the struct lws_email context
 *
 * Stop sending email using email and free allocations
 */
LWS_VISIBLE LWS_EXTERN void
lws_email_destroy(struct lws_email *email);

#endif
//@}

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
lws_stats_get(struct lws_context *context, int index) { return 0; }
static LWS_INLINE void
lws_stats_log_dump(struct lws_context *context) { }
#endif

#ifdef __cplusplus
}
#endif

#endif
