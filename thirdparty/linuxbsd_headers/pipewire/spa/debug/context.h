/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_DEBUG_CONTEXT_H
#define SPA_DEBUG_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>

#include <spa/utils/defs.h>
/**
 * \addtogroup spa_debug
 * \{
 */

#ifndef spa_debugn
#define spa_debugn(_fmt,...)	printf((_fmt), ## __VA_ARGS__)
#endif
#ifndef spa_debug
#define spa_debug(_fmt,...)	spa_debugn(_fmt"\n", ## __VA_ARGS__)
#endif

struct spa_debug_context {
	void (*log) (struct spa_debug_context *ctx, const char *fmt, ...) SPA_PRINTF_FUNC(2, 3);
};

#define spa_debugc(_c,_fmt,...)	(_c)?((_c)->log((_c),_fmt, ## __VA_ARGS__)):(void)spa_debug(_fmt, ## __VA_ARGS__)

static inline void spa_debugc_error_location(struct spa_debug_context *c,
		struct spa_error_location *loc)
{
	int i, skip = loc->col > 80 ? loc->col - 40 : 0, lc = loc->col-skip-1;
	char buf[80];

	for (i = 0; (size_t)i < sizeof(buf)-1 && (size_t)(i + skip) < loc->len; i++) {
		char ch = loc->location[i + skip];
		if (ch == '\n' || ch == '\0')
			break;
		buf[i] = isspace(ch) ? ' ' : ch;
	}
	buf[i] = '\0';
	spa_debugc(c, "line:%6d | %s%s", loc->line, skip ? "..." : "", buf);
	for (i = 0; buf[i]; i++)
		buf[i] = i < lc ? '-' : i == lc ? '^' : ' ';
	spa_debugc(c, "column:%4d |-%s%s", loc->col, skip ? "---" : "", buf);
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_DEBUG_CONTEXT_H */
