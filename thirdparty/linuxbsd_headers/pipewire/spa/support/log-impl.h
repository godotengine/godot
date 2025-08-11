/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_LOG_IMPL_H
#define SPA_LOG_IMPL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

#include <spa/utils/type.h>
#include <spa/support/log.h>

/**
 * \addtogroup spa_log
 * \{
 */

static inline SPA_PRINTF_FUNC(7, 0) void spa_log_impl_logtv(void *object SPA_UNUSED,
				     enum spa_log_level level,
				     const struct spa_log_topic *topic,
				     const char *file,
				     int line,
				     const char *func,
				     const char *fmt,
				     va_list args)
{
	static const char * const levels[] = { "-", "E", "W", "I", "D", "T" };

	const char *basename = strrchr(file, '/');
	char text[512], location[1024];
	char topicstr[32] = {0};

	if (basename)
		basename += 1; /* skip '/' */
	else
		basename = file; /* use whole string if no '/' is found */

	if (topic && topic->topic)
		snprintf(topicstr, sizeof(topicstr), " %s ", topic->topic);
	vsnprintf(text, sizeof(text), fmt, args);
	snprintf(location, sizeof(location), "[%s]%s[%s:%i %s()] %s\n",
		 levels[level],
		 topicstr,
		 basename, line, func, text);
	fputs(location, stderr);
}

static inline SPA_PRINTF_FUNC(7,8) void spa_log_impl_logt(void *object,
				    enum spa_log_level level,
				    const struct spa_log_topic *topic,
				    const char *file,
				    int line,
				    const char *func,
				    const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	spa_log_impl_logtv(object, level, topic, file, line, func, fmt, args);
	va_end(args);
}

static inline SPA_PRINTF_FUNC(6, 0) void spa_log_impl_logv(void *object,
				     enum spa_log_level level,
				     const char *file,
				     int line,
				     const char *func,
				     const char *fmt,
				     va_list args)
{

	spa_log_impl_logtv(object, level, NULL, file, line, func, fmt, args);
}

static inline SPA_PRINTF_FUNC(6,7) void spa_log_impl_log(void *object,
				    enum spa_log_level level,
				    const char *file,
				    int line,
				    const char *func,
				    const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	spa_log_impl_logv(object, level, file, line, func, fmt, args);
	va_end(args);
}

static inline void spa_log_impl_topic_init(void *object SPA_UNUSED, struct spa_log_topic *topic SPA_UNUSED)
{
	/* noop */
}

#define SPA_LOG_IMPL_DEFINE(name)		\
struct {					\
	struct spa_log log;			\
	struct spa_log_methods methods;		\
} name

#define SPA_LOG_IMPL_INIT(name)				\
	{ { { SPA_TYPE_INTERFACE_Log, SPA_VERSION_LOG,	\
	      SPA_CALLBACKS_INIT(&(name).methods, &(name)) },	\
	    SPA_LOG_LEVEL_INFO,	},			\
	  { SPA_VERSION_LOG_METHODS,			\
	    spa_log_impl_log,				\
	    spa_log_impl_logv,				\
	    spa_log_impl_logt,				\
	    spa_log_impl_logtv,				\
	  } }

#define SPA_LOG_IMPL(name)			\
        SPA_LOG_IMPL_DEFINE(name) = SPA_LOG_IMPL_INIT(name)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* SPA_LOG_IMPL_H */
