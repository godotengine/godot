/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_LOG_H
#define PIPEWIRE_LOG_H

#include <spa/support/log.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup pw_log Logging
 *
 * \brief Logging functions of PipeWire
 *
 * Logging is performed to stdout and stderr. Trace logging is performed
 * in a lockfree ringbuffer and written out from the main thread as to not
 * block the realtime threads.
 */

/**
 * \addtogroup pw_log
 * \{
 */
/** The global log level */
extern enum spa_log_level pw_log_level;

extern struct spa_log_topic *PW_LOG_TOPIC_DEFAULT;

/** Configure a logging module. This is usually done automatically
 * in pw_init() but you can install a custom logger before calling
 * pw_init(). */
void pw_log_set(struct spa_log *log);

/** Get the log interface */
struct spa_log *pw_log_get(void);

/** Configure the logging level */
void pw_log_set_level(enum spa_log_level level);

/** Log a message for a topic */
void
pw_log_logt(enum spa_log_level level,
	    const struct spa_log_topic *topic,
	    const char *file,
	    int line, const char *func,
	    const char *fmt, ...) SPA_PRINTF_FUNC(6, 7);

/** Log a message for a topic */
void
pw_log_logtv(enum spa_log_level level,
	     const struct spa_log_topic *topic,
	     const char *file,
	     int line, const char *func,
	     const char *fmt, va_list args) SPA_PRINTF_FUNC(6, 0);



/** Log a message for the default topic */
void
pw_log_log(enum spa_log_level level,
	   const char *file,
	   int line, const char *func,
	   const char *fmt, ...) SPA_PRINTF_FUNC(5, 6);

/** Log a message for the default topic */
void
pw_log_logv(enum spa_log_level level,
	    const char *file,
	    int line, const char *func,
	    const char *fmt, va_list args) SPA_PRINTF_FUNC(5, 0);

/**
 * Declare a static log topic named \a var. The usual usage is:
 * \code
 *  PW_LOG_TOPIC_STATIC(my_topic);
 *  #define PW_LOG_TOPIC_DEFAULT my_topic
 *
 *  void foo() {
 *      pw_log_debug("bar");
 *  }
 * \endcode
 */
#define PW_LOG_TOPIC_STATIC(var, topic) \
  static struct spa_log_topic var##__LINE__ = SPA_LOG_TOPIC(0, topic); \
  static struct spa_log_topic *var = &(var##__LINE__)

/**
 * Declare a static log topic named \a var.
 * See \ref PW_LOG_TOPIC_STATIC for an example usage.
 */
#define PW_LOG_TOPIC_EXTERN(var) \
  extern struct spa_log_topic *var

/**
 * Declare a static log topic named \a var.
 * See \ref PW_LOG_TOPIC_STATIC for an example usage.
 */
#define PW_LOG_TOPIC(var, topic) \
  struct spa_log_topic var##__LINE__ = SPA_LOG_TOPIC(0, topic); \
  struct spa_log_topic *var = &(var##__LINE__)

#define PW_LOG_TOPIC_INIT(var) \
   spa_log_topic_init(pw_log_get(), var);

/** Check if a loglevel is enabled */
#define pw_log_level_enabled(lev) (pw_log_level >= (lev))
#define pw_log_topic_enabled(lev,t) ((t) && (t)->has_custom_level ? (t)->level >= (lev) : pw_log_level_enabled((lev)))

#define pw_logtv(lev,topic,fmt,ap)						\
({										\
	if (SPA_UNLIKELY(pw_log_topic_enabled(lev,topic)))			\
		pw_log_logtv(lev,topic,__FILE__,__LINE__,__func__,fmt,ap);	\
})

#define pw_logt(lev,topic,...)							\
({										\
	if (SPA_UNLIKELY(pw_log_topic_enabled(lev,topic)))			\
		pw_log_logt(lev,topic,__FILE__,__LINE__,__func__,__VA_ARGS__);	\
})

#define pw_log(lev,...) pw_logt(lev,PW_LOG_TOPIC_DEFAULT,__VA_ARGS__)

#define pw_log_error(...)   pw_log(SPA_LOG_LEVEL_ERROR,__VA_ARGS__)
#define pw_log_warn(...)    pw_log(SPA_LOG_LEVEL_WARN,__VA_ARGS__)
#define pw_log_info(...)    pw_log(SPA_LOG_LEVEL_INFO,__VA_ARGS__)
#define pw_log_debug(...)   pw_log(SPA_LOG_LEVEL_DEBUG,__VA_ARGS__)
#define pw_log_trace(...)   pw_log(SPA_LOG_LEVEL_TRACE,__VA_ARGS__)

#define pw_logt_error(t,...)   pw_logt(SPA_LOG_LEVEL_ERROR,t,__VA_ARGS__)
#define pw_logt_warn(t,...)    pw_logt(SPA_LOG_LEVEL_WARN,t,__VA_ARGS__)
#define pw_logt_info(t,...)    pw_logt(SPA_LOG_LEVEL_INFO,t,__VA_ARGS__)
#define pw_logt_debug(t,...)   pw_logt(SPA_LOG_LEVEL_DEBUG,t,__VA_ARGS__)
#define pw_logt_trace(t,...)   pw_logt(SPA_LOG_LEVEL_TRACE,t,__VA_ARGS__)

#ifndef FASTPATH
#define pw_log_trace_fp(...)   pw_log(SPA_LOG_LEVEL_TRACE,__VA_ARGS__)
#else
#define pw_log_trace_fp(...)
#endif

/**
 * \}
 */

#ifdef __cplusplus
}
#endif
#endif /* PIPEWIRE_LOG_H */
