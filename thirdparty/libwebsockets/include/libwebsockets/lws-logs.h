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
	LLL_ERR		= 1 << 0,
	LLL_WARN	= 1 << 1,
	LLL_NOTICE	= 1 << 2,
	LLL_INFO	= 1 << 3,
	LLL_DEBUG	= 1 << 4,
	LLL_PARSER	= 1 << 5,
	LLL_HEADER	= 1 << 6,
	LLL_EXT		= 1 << 7,
	LLL_CLIENT	= 1 << 8,
	LLL_LATENCY	= 1 << 9,
	LLL_USER	= 1 << 10,
	LLL_THREAD	= 1 << 11,

	LLL_COUNT	= 12 /* set to count of valid flags */
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
#define lwsl_thread(...) _lws_log(LLL_THREAD, __VA_ARGS__)

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
#define lwsl_thread(...) do {} while(0)

#endif

#define lwsl_hexdump_err(...) lwsl_hexdump_level(LLL_ERR, __VA_ARGS__)
#define lwsl_hexdump_warn(...) lwsl_hexdump_level(LLL_WARN, __VA_ARGS__)
#define lwsl_hexdump_notice(...) lwsl_hexdump_level(LLL_NOTICE, __VA_ARGS__)
#define lwsl_hexdump_info(...) lwsl_hexdump_level(LLL_INFO, __VA_ARGS__)
#define lwsl_hexdump_debug(...) lwsl_hexdump_level(LLL_DEBUG, __VA_ARGS__)

/**
 * lwsl_hexdump_level() - helper to hexdump a buffer at a selected debug level
 *
 * \param level: one of LLL_ constants
 * \param vbuf: buffer start to dump
 * \param len: length of buffer to dump
 *
 * If \p level is visible, does a nice hexdump -C style dump of \p vbuf for
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
 * lwsl_emit_stderr() - helper log emit function writes to stderr
 *
 * \param level: one of LLL_ log level indexes
 * \param line: log string
 *
 * You use this by passing the function pointer to lws_set_log_level(), to set
 * it as the log emit function, it is not called directly.
 *
 * It prepends a system timestamp like [2018/11/13 07:41:57:3989]
 *
 * If stderr is a tty, then ansi colour codes are added.
 */
LWS_VISIBLE LWS_EXTERN void
lwsl_emit_stderr(int level, const char *line);

/**
 * lwsl_emit_stderr_notimestamp() - helper log emit function writes to stderr
 *
 * \param level: one of LLL_ log level indexes
 * \param line: log string
 *
 * You use this by passing the function pointer to lws_set_log_level(), to set
 * it as the log emit function, it is not called directly.
 *
 * If stderr is a tty, then ansi colour codes are added.
 */
LWS_VISIBLE LWS_EXTERN void
lwsl_emit_stderr_notimestamp(int level, const char *line);

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
