/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_LOG_H
#define SPA_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>

#include <spa/utils/type.h>
#include <spa/utils/defs.h>
#include <spa/utils/hook.h>

/** \defgroup spa_log Log
 * Logging interface
 */

/**
 * \addtogroup spa_log
 * \{
 */

/** The default log topic. Redefine this in your code to
 * allow for the spa_log_* macros to work correctly, e.g:
 *
 * \code{.c}
 * struct spa_log_topic *mylogger;
 * #undef SPA_LOG_TOPIC_DEFAULT
 * #define SPA_LOG_TOPIC_DEFAULT mylogger
 * \endcode
 */
#define SPA_LOG_TOPIC_DEFAULT NULL

enum spa_log_level {
	SPA_LOG_LEVEL_NONE = 0,
	SPA_LOG_LEVEL_ERROR,
	SPA_LOG_LEVEL_WARN,
	SPA_LOG_LEVEL_INFO,
	SPA_LOG_LEVEL_DEBUG,
	SPA_LOG_LEVEL_TRACE,
};

/**
 * The Log interface
 */
#define SPA_TYPE_INTERFACE_Log	SPA_TYPE_INFO_INTERFACE_BASE "Log"


struct spa_log {
	/** the version of this log. This can be used to expand this
	 * structure in the future */
#define SPA_VERSION_LOG		0
	struct spa_interface iface;
	/**
	 * Logging level, everything above this level is not logged
	 */
	enum spa_log_level level;
};

/**
 * \struct spa_log_topic
 *
 * Identifier for a topic. Topics are string-based filters that logically
 * group messages together. An implementation may decide to filter different
 * topics on different levels, for example the "protocol" topic may require
 * debug level TRACE while the "core" topic defaults to debug level INFO.
 *
 * spa_log_topics require a spa_log_methods version of 1 or higher.
 */
struct spa_log_topic {
#define SPA_VERSION_LOG_TOPIC	0
	/** the version of this topic. This can be used to expand this
	 * structure in the future */
	uint32_t version;
	/** The string identifier for the topic */
	const char *topic;
	/** Logging level set for this topic */
	enum spa_log_level level;
	/** False if this topic follows the \ref spa_log level */
	bool has_custom_level;
};

/**
 * Enumeration of log topics in a plugin
 *
 * \since 1.1.0
 */
struct spa_log_topic_enum {
#define SPA_VERSION_LOG_TOPIC_ENUM	0
	uint32_t version;
	/** Array of pointers to log topics */
	struct spa_log_topic * const * const topics;
	/** End of topics array */
	struct spa_log_topic * const * const topics_end;
};


struct spa_log_methods {
#define SPA_VERSION_LOG_METHODS		1
	uint32_t version;
	/**
	 * Log a message with the given log level.
	 *
	 * \note If compiled with this header, this function is only called
	 * for implementations of version 0. For versions 1 and above, see
	 * logt() instead.
	 *
	 * \param log a spa_log
	 * \param level a spa_log_level
	 * \param file the file name
	 * \param line the line number
	 * \param func the function name
	 * \param fmt printf style format
	 * \param ... format arguments
	 */
	void (*log) (void *object,
		     enum spa_log_level level,
		     const char *file,
		     int line,
		     const char *func,
		     const char *fmt, ...) SPA_PRINTF_FUNC(6, 7);

	/**
	 * Log a message with the given log level.
	 *
	 * \note If compiled with this header, this function is only called
	 * for implementations of version 0. For versions 1 and above, see
	 * logtv() instead.
	 *
	 * \param log a spa_log
	 * \param level a spa_log_level
	 * \param file the file name
	 * \param line the line number
	 * \param func the function name
	 * \param fmt printf style format
	 * \param args format arguments
	 */
	void (*logv) (void *object,
		      enum spa_log_level level,
		      const char *file,
		      int line,
		      const char *func,
		      const char *fmt,
		      va_list args) SPA_PRINTF_FUNC(6, 0);
	/**
	 * Log a message with the given log level for the given topic.
	 *
	 * \note Callers that do not use topic-based logging (version 0), the \a
	 * topic is NULL
	 *
	 * \param log a spa_log
	 * \param level a spa_log_level
	 * \param topic the topic for this message, may be NULL
	 * \param file the file name
	 * \param line the line number
	 * \param func the function name
	 * \param fmt printf style format
	 * \param ... format arguments
	 *
	 * \since 1
	 */
	void (*logt) (void *object,
		     enum spa_log_level level,
		     const struct spa_log_topic *topic,
		     const char *file,
		     int line,
		     const char *func,
		     const char *fmt, ...) SPA_PRINTF_FUNC(7, 8);

	/**
	 * Log a message with the given log level for the given topic.
	 *
	 * \note For callers that do not use topic-based logging (version 0),
	 * the \a topic is NULL
	 *
	 * \param log a spa_log
	 * \param level a spa_log_level
	 * \param topic the topic for this message, may be NULL
	 * \param file the file name
	 * \param line the line number
	 * \param func the function name
	 * \param fmt printf style format
	 * \param args format arguments
	 *
	 * \since 1
	 */
	void (*logtv) (void *object,
		      enum spa_log_level level,
		      const struct spa_log_topic *topic,
		      const char *file,
		      int line,
		      const char *func,
		      const char *fmt,
		      va_list args) SPA_PRINTF_FUNC(7, 0);

	/**
	 * Initializes a \ref spa_log_topic to the correct logging level.
	 *
	 * \deprecated
	 * Plugin host should obtain log topics from \ref SPA_LOG_TOPIC_ENUM_NAME
	 * and update them itself.
	 *
	 * \since 1
	 */
	void (*topic_init) (void *object, struct spa_log_topic *topic);
};


#define SPA_LOG_TOPIC(v, t) \
   (struct spa_log_topic){ .version = (v), .topic = (t)}

static inline void spa_log_topic_init(struct spa_log *log, struct spa_log_topic *topic)
{
	if (SPA_UNLIKELY(!log))
		return;

	spa_interface_call(&log->iface, struct spa_log_methods, topic_init, 1, topic);
}

static inline bool spa_log_level_topic_enabled(const struct spa_log *log,
					       const struct spa_log_topic *topic,
					       enum spa_log_level level)
{
	enum spa_log_level max_level;

	if (SPA_UNLIKELY(!log))
		return false;

	if (topic && topic->has_custom_level)
		max_level = topic->level;
	else
		max_level = log->level;

	return level <= max_level;
}

/* Transparently calls to version 0 log if v1 is not supported */
#define spa_log_logt(l,lev,topic,...)					\
({									\
	struct spa_log *_l = l;						\
	if (SPA_UNLIKELY(spa_log_level_topic_enabled(_l, topic, lev))) { \
		struct spa_interface *_if = &_l->iface;			\
		if (!spa_interface_call(_if,				\
				struct spa_log_methods, logt, 1,	\
				lev, topic,				\
				__VA_ARGS__))				\
		    spa_interface_call(_if,				\
				struct spa_log_methods, log, 0,		\
				lev, __VA_ARGS__);			\
	}								\
})

/* Transparently calls to version 0 logv if v1 is not supported */
#define spa_log_logtv(l,lev,topic,...)					\
({									\
	struct spa_log *_l = l;						\
	if (SPA_UNLIKELY(spa_log_level_topic_enabled(_l, topic, lev))) { \
		struct spa_interface *_if = &_l->iface;			\
		if (!spa_interface_call(_if,				\
				struct spa_log_methods, logtv, 1,	\
				lev, topic,				\
				__VA_ARGS__))				\
		    spa_interface_call(_if,				\
				struct spa_log_methods, logv, 0,	\
				lev, __VA_ARGS__);			\
	}								\
})

#define spa_logt_lev(l,lev,t,...)					\
	spa_log_logt(l,lev,t,__FILE__,__LINE__,__func__,__VA_ARGS__)

#define spa_log_lev(l,lev,...)					\
	spa_logt_lev(l,lev,SPA_LOG_TOPIC_DEFAULT,__VA_ARGS__)

#define spa_log_log(l,lev,...)					\
	spa_log_logt(l,lev,SPA_LOG_TOPIC_DEFAULT,__VA_ARGS__)

#define spa_log_logv(l,lev,...)					\
	spa_log_logtv(l,lev,SPA_LOG_TOPIC_DEFAULT,__VA_ARGS__)

#define spa_log_error(l,...)	spa_log_lev(l,SPA_LOG_LEVEL_ERROR,__VA_ARGS__)
#define spa_log_warn(l,...)	spa_log_lev(l,SPA_LOG_LEVEL_WARN,__VA_ARGS__)
#define spa_log_info(l,...)	spa_log_lev(l,SPA_LOG_LEVEL_INFO,__VA_ARGS__)
#define spa_log_debug(l,...)	spa_log_lev(l,SPA_LOG_LEVEL_DEBUG,__VA_ARGS__)
#define spa_log_trace(l,...)	spa_log_lev(l,SPA_LOG_LEVEL_TRACE,__VA_ARGS__)

#define spa_logt_error(l,t,...)	spa_logt_lev(l,SPA_LOG_LEVEL_ERROR,t,__VA_ARGS__)
#define spa_logt_warn(l,t,...)	spa_logt_lev(l,SPA_LOG_LEVEL_WARN,t,__VA_ARGS__)
#define spa_logt_info(l,t,...)	spa_logt_lev(l,SPA_LOG_LEVEL_INFO,t,__VA_ARGS__)
#define spa_logt_debug(l,t,...)	spa_logt_lev(l,SPA_LOG_LEVEL_DEBUG,t,__VA_ARGS__)
#define spa_logt_trace(l,t,...)	spa_logt_lev(l,SPA_LOG_LEVEL_TRACE,t,__VA_ARGS__)

#ifndef FASTPATH
#define spa_log_trace_fp(l,...)	spa_log_lev(l,SPA_LOG_LEVEL_TRACE,__VA_ARGS__)
#else
#define spa_log_trace_fp(l,...)
#endif


/**
 * Name of the symbol indicating a \ref spa_log_topic_enum enumerating
 * the static log topics in a plugin,
 *
 * \since 1.1.0
 */
#define SPA_LOG_TOPIC_ENUM_NAME "spa_log_topic_enum"

/**
 * Define the symbol for \ref SPA_LOG_TOPIC_ENUM_NAME
 *
 * \since 1.1.0
 */
#define SPA_LOG_TOPIC_ENUM_DEFINE(s, e) \
	SPA_EXPORT struct spa_log_topic_enum spa_log_topic_enum = (struct spa_log_topic_enum) { \
		.version = SPA_VERSION_LOG_TOPIC_ENUM, \
		.topics = (s), \
		.topics_end = (e), \
	}

/**
 * Magically register a statically defined \ref spa_log_topic into
 * the log topic enumeration for a plugin.
 *
 * \since 1.1.0
 */
#define SPA_LOG_TOPIC_REGISTER(v)		      \
	__attribute__((used)) __attribute__((retain)) \
	__attribute__((section("spa_log_topic"))) __attribute__((aligned(__alignof__(struct spa_log_topic *)))) \
	static struct spa_log_topic * const spa_log_topic_export_##v = &v

/**
 * Define and magically register a \ref spa_log_topic
 *
 * \since 1.1.0
 */
#define SPA_LOG_TOPIC_DEFINE(var,name) \
	struct spa_log_topic var = SPA_LOG_TOPIC(SPA_VERSION_LOG_TOPIC, name); \
	SPA_LOG_TOPIC_REGISTER(var)

/**
 * Define and magically register a \ref spa_log_topic with static scope
 *
 * \since 1.1.0
 */
#define SPA_LOG_TOPIC_DEFINE_STATIC(var,name) \
	static struct spa_log_topic var = SPA_LOG_TOPIC(SPA_VERSION_LOG_TOPIC, name); \
	SPA_LOG_TOPIC_REGISTER(var)

/**
 * Do \ref SPA_LOG_TOPIC_ENUM_DEFINE for the auto-registered
 * \ref spa_log_topic in the plugin.
 *
 * \since 1.1.0
 */
#define SPA_LOG_TOPIC_ENUM_DEFINE_REGISTERED \
	extern struct spa_log_topic * const __start_spa_log_topic[]; \
	extern struct spa_log_topic * const __stop_spa_log_topic[]; \
	SPA_LOG_TOPIC_ENUM_DEFINE(__start_spa_log_topic, __stop_spa_log_topic)

/** \fn spa_log_error */

/** keys can be given when initializing the logger handle */
#define SPA_KEY_LOG_LEVEL		"log.level"		/**< the default log level */
#define SPA_KEY_LOG_COLORS		"log.colors"		/**< enable colors in the logger, set to "force" to enable
								  *  colors even when not logging to a terminal */
#define SPA_KEY_LOG_FILE		"log.file"		/**< log to the specified file instead of
								  *  stderr. */
#define SPA_KEY_LOG_TIMESTAMP		"log.timestamp"		/**< log timestamps */
#define SPA_KEY_LOG_LINE		"log.line"		/**< log file and line numbers */
#define SPA_KEY_LOG_PATTERNS		"log.patterns"		/**< Spa:String:JSON array of [ {"pattern" : level}, ... ] */

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* SPA_LOG_H */
