/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2022 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_DEBUG_LOG_H
#define SPA_DEBUG_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdarg.h>

#include <spa/utils/defs.h>
#include <spa/support/log.h>
#include <spa/debug/context.h>
#include <spa/debug/dict.h>
#include <spa/debug/format.h>
#include <spa/debug/mem.h>
#include <spa/debug/pod.h>

/**
 * \addtogroup spa_debug
 * \{
 */

struct spa_debug_log_ctx {
	struct spa_debug_context ctx;
	struct spa_log *log;
	enum spa_log_level level;
	const struct spa_log_topic *topic;
	const char *file;
	int line;
	const char *func;
};

SPA_PRINTF_FUNC(2,3)
static inline void spa_debug_log_log(struct spa_debug_context *ctx, const char *fmt, ...)
{
	struct spa_debug_log_ctx *c = SPA_CONTAINER_OF(ctx, struct spa_debug_log_ctx, ctx);
	va_list args;
	va_start(args, fmt);
	spa_log_logtv(c->log, c->level, c->topic, c->file, c->line, c->func, fmt, args);
	va_end(args);
}

#define SPA_LOGF_DEBUG_INIT(_l,_lev,_t,_file,_line,_func)			\
	(struct spa_debug_log_ctx){ { spa_debug_log_log }, _l, _lev, _t,	\
		_file, _line, _func }

#define SPA_LOGT_DEBUG_INIT(_l,_lev,_t)						\
	SPA_LOGF_DEBUG_INIT(_l,_lev,_t,__FILE__,__LINE__,__func__)

#define SPA_LOG_DEBUG_INIT(l,lev) 	\
	SPA_LOGT_DEBUG_INIT(l,lev,SPA_LOG_TOPIC_DEFAULT)

#define spa_debug_log_pod(l,lev,indent,info,pod) 				\
({										\
	struct spa_debug_log_ctx c = SPA_LOG_DEBUG_INIT(l,lev);			\
	if (SPA_UNLIKELY(spa_log_level_topic_enabled(c.log, c.topic, c.level)))	\
		spa_debugc_pod(&c.ctx, indent, info, pod);			\
})

#define spa_debug_log_format(l,lev,indent,info,format) 				\
({										\
	struct spa_debug_log_ctx c = SPA_LOG_DEBUG_INIT(l,lev);			\
	if (SPA_UNLIKELY(spa_log_level_topic_enabled(c.log, c.topic, c.level)))	\
		spa_debugc_format(&c.ctx, indent, info, format);		\
})

#define spa_debug_log_mem(l,lev,indent,data,len)				\
({										\
	struct spa_debug_log_ctx c = SPA_LOG_DEBUG_INIT(l,lev);			\
	if (SPA_UNLIKELY(spa_log_level_topic_enabled(c.log, c.topic, c.level)))	\
		spa_debugc_mem(&c.ctx, indent, data, len);			\
})

#define spa_debug_log_dict(l,lev,indent,dict)					\
({										\
	struct spa_debug_log_ctx c = SPA_LOG_DEBUG_INIT(l,lev);			\
	if (SPA_UNLIKELY(spa_log_level_topic_enabled(c.log, c.topic, c.level)))	\
		spa_debugc_dict(&c.ctx, indent, dict);				\
})

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_DEBUG_LOG_H */
