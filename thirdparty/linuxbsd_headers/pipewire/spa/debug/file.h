/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2022 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_DEBUG_FILE_H
#define SPA_DEBUG_FILE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>

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

struct spa_debug_file_ctx {
	struct spa_debug_context ctx;
	FILE *f;
};

SPA_PRINTF_FUNC(2,3)
static inline void spa_debug_file_log(struct spa_debug_context *ctx, const char *fmt, ...)
{
	struct spa_debug_file_ctx *c = SPA_CONTAINER_OF(ctx, struct spa_debug_file_ctx, ctx);
	va_list args;
	va_start(args, fmt);
	vfprintf(c->f, fmt, args); fputc('\n', c->f);
	va_end(args);
}

#define SPA_DEBUG_FILE_INIT(_f)							\
	(struct spa_debug_file_ctx){ { spa_debug_file_log }, _f, }

#define spa_debug_file_error_location(f,loc,fmt,...)				\
({										\
	struct spa_debug_file_ctx c = SPA_DEBUG_FILE_INIT(f);			\
	if (fmt) spa_debugc(&c.ctx, fmt, __VA_ARGS__);				\
	spa_debugc_error_location(&c.ctx, loc);					\
})

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_DEBUG_FILE_H */
